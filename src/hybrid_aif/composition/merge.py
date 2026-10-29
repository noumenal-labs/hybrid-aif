"""``merge()`` and ``subsystem()``: one model from several, and its per-mind views.

:func:`merge` runs at declaration time and returns a plain ``(System, SystemParams)``, so every
face applies to a composite unchanged. It namespaces every node as ``"<sub>.<name>"`` (the whole
key grammar with it: ``"likelihood:arm.pose"``), concatenates state factors, modalities and control
nodes in ``subsystems`` order with every dependency index offset, and applies the bindings, each an
identification of latents across components: :class:`ActionControl`, :class:`ObservationLink`,
:class:`TargetEmission`, :class:`BoundaryPrior`. Control nodes no binding touches stay the
composite's own, chosen by its planner. An ``ActionControl`` names a control node of one component,
and the latent it is identified with decides the timing: another component's control node is read
at the same step (the two nodes become one), a state of another component at the previous step (a
one-tick relay). A composite may be merged again (``"outer.inner.x"``).

:func:`subsystem` goes the other way: one mind's bona-fide System, its own factors plus boundary
stubs for the neighbours it must model. The driver that interleaves the views is
:mod:`hybrid_aif.composition.driver`.

Design: docs/design/nesting-systems.md; docs/design/control-states.md C12.
"""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from hybrid_aif.controls import (
    FLOAT,
    LEVEL,
    NAMESPACE_SEPARATOR,
    control_name,
    declared_parents,
)
from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalDirichlet,
)
from hybrid_aif.factors import NonlinearGaussianTransition, Recognition
from hybrid_aif.planning.enumerate import enumerate_policies
from hybrid_aif.system import _KEY_KINDS, Subsystem, System, SystemParams, _is_pattern

SEPARATOR = NAMESPACE_SEPARATOR
#: The namespace separator (N3, superseding the per-variable-infogain doc's ``/``
#: example): prefix on the left via ``.``, slot-qualification via ``:`` unchanged.


# ------------------------------------------------------------------ the bindings


class ActionControl(NamedTuple):
    """Binding (a): a control node of one component identified with a latent of another.

    ``control`` names a control node of one component (``"world.u_location"``). ``driver``
    names either

    - a **control node** of another component (``"agent.u_location"``): read at the same step.
      The two nodes become one, the driver's: every factor that read ``control`` reads the
      driver's node at the step it is chosen, and ``control`` and its action node leave the
      composite. Level counts (or, for continuous nodes, widths) must agree; the driver's box
      governs a continuous pair;
    - a **state factor** of another component (``"agent.location"``): read a step later. The
      state becomes a ``B_deps`` parent of the one factor reading ``control``, in its place, the
      table's control axis re-read as that parent and a no-op axis of size 1 appended. The
      state's level count must equal the node's.
    """

    driver: str
    control: str


class ObservationLink(NamedTuple):
    """Binding (b): a modality of one component identified as reading another's factor(s).

    ``factors`` is the modality's complete new ``A_deps`` parent list, unchanged parents included.
    Arity and per-axis support must match the declared likelihood's parent axes.
    """

    modality: str
    factors: tuple[str, ...]


class TargetEmission(NamedTuple):
    """Binding (c): one component's terminal target conditioned on another's factor(s).

    Declares the ``H_deps`` entry of ``target``: its terminal target, already declared as a
    ``ConditionalCategorical`` in the component's ``terminal_targets``, is resolved against
    ``parents``' final predicted beliefs.
    """

    target: str
    parents: tuple[str, ...]


class BoundaryPrior(NamedTuple):
    """Terminate a control node: replace its decision with a declared prior.

    ``control`` names a discrete control node (``"world.u_location"``). ``prior`` is an ``int``
    level (a delta: slices every reading factor's control axis, pinning the dynamics) or a
    :class:`~hybrid_aif.distributions.Categorical` / probability vector over the node's levels
    (averages every reading factor's table over the axis; uniform means no one is steering).
    """

    control: str
    prior: "int | Categorical | Array"


Binding = ActionControl | ObservationLink | TargetEmission | BoundaryPrior


# ------------------------------------------------------------------- key helpers


def namespaced_key(prefix: str, key: str) -> str:
    """Qualify one key of the key grammar with a subsystem prefix, on its name segment.

    ``"pose" -> "arm.pose"``, ``"pose:b1" -> "arm.pose:b1"``, ``"obs:cue" -> "obs:arm.cue"``,
    ``"likelihood:pose:noise" -> "likelihood:arm.pose:noise"``. Wildcards must be resolved first
    (:func:`resolved_epistemic`).
    """
    parts = key.split(":")
    if "*" in parts:
        raise ValueError(
            f"cannot namespace the wildcard pattern {key!r}: patterns are resolved against "
            "the subsystem's own variable table first (merge resolves them for you)"
        )
    at = 1 if len(parts) > 1 and parts[0] in _KEY_KINDS else 0
    parts[at] = f"{prefix}{SEPARATOR}{parts[at]}"
    return ":".join(parts)


def resolved_epistemic(system: System, params: SystemParams) -> tuple[str, ...]:
    """A component's ``epistemic`` selector as an explicit tuple of literal keys.

    Patterns and ``True`` are expanded against the component's own variable table, so a component's
    ``("*",)`` is fixed at merge time and does not widen to variables the composite gained.
    ``True`` excludes initial-prior parameter variables as it does everywhere, and names each
    continuous factor whole rather than by its dimensions; a learnable family with no
    registered novelty estimator raises here. A continuous factor no channel with an
    information-gain estimator observes (and its dimensions) resolves to nothing under ``True``
    or a pattern, as in the component itself, rather than to a literal key the composite would
    refuse. :func:`merge` calls this only for a mixed merge: all-``True`` or all-off components
    keep the bool.
    """
    sel = system.epistemic
    if sel is False or sel == ():
        return ()
    variables = system._epistemic_variables(params)
    unobserved = set(system._eligible_state_blocks(params)[1])

    def observed(key: str) -> bool:
        var = variables[key]
        return not (var.kind in ("state", "dim") and var.index in unobserved)

    if sel is True:
        return tuple(
            key
            for key, var in variables.items()
            if var.kind != "dim"
            and not (var.kind == "param" and var.slot == "initial_state_prior")
            and observed(key)
        )
    out: list[str] = []
    for key in sel:
        if _is_pattern(key):
            out.extend(k for k in system._expand_epistemic(key, params, variables) if observed(k))
        else:
            system._check_epistemic_key(key, params, variables)
            out.append(key)
    return tuple(dict.fromkeys(out))


# ------------------------------------------------------------------ small pieces


def _lookup(name: str, table: Mapping[str, int], what: str) -> int:
    if name in table:
        return table[name]
    raise ValueError(
        f"binding names unknown {what} {name!r}; the composite's {what}s are "
        f"{sorted(table)} (names are subsystem-qualified with {SEPARATOR!r}, as in "
        f"'boat{SEPARATOR}x')"
    )


def _agree(field: str, systems: Mapping[str, System]):
    """One inference-config value shared by every component, or an error naming both values.

    Use ``dataclasses.replace`` on the returned System to choose a different composite value.
    """
    values = {name: getattr(system, field) for name, system in systems.items()}
    distinct = set(values.values())
    if len(distinct) > 1:
        raise ValueError(
            f"merged Systems disagree on {field}: {values}. A composite is one planning "
            f"problem with one {field}; declare the components alike, or set the composite's "
            f"value afterwards with dataclasses.replace(composite, {field}=...)"
        )
    return next(iter(distinct))


#: A factor's discrete cardinality, or ``None`` for a non-tabular one — the typing measure
#: for both binding kinds that pair a support with an axis. One definition, shared
#: with the ``H_deps`` parent-shape check.
_factor_levels = System._factor_levels


def _drop_control_axis(decl, name: str, keep: "int | None", weights: "Array | None"):
    """Retire a tabular transition's control axis, keeping a trailing no-op axis.

    With neither ``keep`` nor ``weights`` the control axis is re-read as a parent axis (the
    ``ActionControl`` rewrite; only the declared ``parent_ndims`` changes); ``keep`` slices one
    level; ``weights`` averages the axis in probability space. All three end with a size-1 trailing
    control axis. A nonlinear transition is refused: its map takes the control node as an input.
    """
    if isinstance(decl, NonlinearGaussianTransition):
        raise NotImplementedError(
            f"factor {name!r}: this binding would remove a control node its "
            "NonlinearGaussianTransition reads, and f's signature, one controls entry per "
            "control parent, is fixed code; bind a node f does not read, or identify the node "
            "with another control node of the same kind and size (ActionControl between two "
            "control nodes), which leaves f's inputs whole (design: nonlinear-transitions.md N5)"
        )
    if not isinstance(decl, ConditionalCategorical | ConditionalDirichlet):
        raise NotImplementedError(
            f"factor {name!r}: merging or terminating a control node a "
            f"{type(decl).__name__} transition reads is not built — a continuous/neural "
            "reader would have to receive the identified latent as a continuous parent "
            "vector, which only MLPConditional accepts and only for its own control parent "
            "(design: controls.py, amortization.md A9; nesting-systems.md build chunk 2)"
        )
    rebound = keep is None and weights is None
    if weights is not None and isinstance(decl, ConditionalDirichlet):
        raise NotImplementedError(
            f"factor {name!r}: terminating a *learnable* transition's control node with a "
            "distribution would average Dirichlet counts across control levels, which is not "
            "a conjugate operation. Pin a level instead (an int prior slices the counts), or "
            "freeze the slot and terminate the point form."
        )
    if isinstance(decl, ConditionalDirichlet):
        table = decl.concentration if rebound else jnp.take(decl.concentration, keep, axis=-1)
        anchor = decl.alpha0
        if anchor is not None and anchor.ndim != 0:  # a scalar (uniform) anchor is kept whole
            anchor = anchor if rebound else jnp.take(anchor, keep, axis=-1)
            anchor = anchor[..., None]
        return ConditionalDirichlet(
            concentration=table[..., None],
            parent_ndims=decl.parent_ndims + (1 if rebound else 0),
            alpha0=anchor,
            growth=decl.growth,
        )
    if weights is None:
        table = decl.log_table if rebound else jnp.take(decl.log_table, keep, axis=-1)
        return ConditionalCategorical(
            log_table=table[..., None], parent_ndims=decl.parent_ndims + (1 if rebound else 0)
        )
    averaged = jnp.einsum("...u,u->...", jnp.exp(decl.log_probs), weights)
    return ConditionalCategorical(
        log_table=jnp.log(averaged)[..., None], parent_ndims=decl.parent_ndims
    )


def _cut_axes(decl, name: str, n: int, cuts: Sequence[tuple]) -> object:
    """Terminate some of a tabular transition's ``n`` control axes.

    ``cuts`` holds ``(pos, keep, weights)`` per axis, ``pos`` the axis's place among the
    declared parents: ``keep`` slices one level, ``weights`` averages the axis in probability
    space. The other axes keep their order; a table left reading no control node carries a
    trailing no-op axis of size 1.
    """
    if not isinstance(decl, ConditionalCategorical | ConditionalDirichlet):
        return _drop_control_axis(decl, name, 0, None)  # raises with the family's sentence
    if n == 1 and len(cuts) == 1 and cuts[0][0] == decl.parent_ndims - 1:
        _, keep, weights = cuts[0]
        return _drop_control_axis(decl, name, keep, weights)
    averaging = any(weights is not None for _, _, weights in cuts)
    if averaging and isinstance(decl, ConditionalDirichlet):
        return _drop_control_axis(decl, name, None, jnp.ones(1))  # raises: not conjugate
    tables = (
        [decl.concentration, decl.alpha0]
        if isinstance(decl, ConditionalDirichlet)
        else [jnp.exp(decl.log_probs) if averaging else decl.log_table]
    )
    out = []
    for table in tables:
        if table is None or table.ndim == 0:  # a scalar (uniform) anchor is kept whole
            out.append(table)
            continue
        first = table.ndim - decl.parent_ndims
        for pos, keep, weights in sorted(cuts, key=lambda cut: cut[0], reverse=True):
            if weights is None:
                table = jnp.take(table, keep, axis=first + pos)
            else:
                table = jnp.einsum("...u,u->...", jnp.moveaxis(table, first + pos, -1), weights)
        if len(cuts) == n:
            table = table[..., None]
        out.append(table)
    parent_ndims = decl.parent_ndims - len(cuts) + (1 if len(cuts) == n else 0)
    if isinstance(decl, ConditionalDirichlet):
        return ConditionalDirichlet(
            concentration=out[0], parent_ndims=parent_ndims, alpha0=out[1], growth=decl.growth
        )
    table = jnp.log(out[0]) if averaging else out[0]
    return ConditionalCategorical(log_table=table, parent_ndims=parent_ndims)


def _prior_weights(prior, nu: int, what: str) -> Array:
    probs = prior.probs if isinstance(prior, Categorical) else jnp.asarray(prior, jnp.float32)
    if probs.shape != (nu,):
        raise ValueError(
            f"{what} has shape {probs.shape} but the node has {nu} level(s); declare "
            "one probability per level (or an int level for a delta)"
        )
    return probs


def _termination(prior, nu: int, what: str) -> tuple["int | None", "Array | None"]:
    """A terminating prior over a control node, as ``(keep, weights)``.

    An ``int`` is a level (slice the axis), anything else a distribution over levels (average it).
    Shared by :class:`BoundaryPrior` in :func:`merge` and the partner nodes of :func:`subsystem`.
    """
    scalar = not isinstance(prior, Categorical) and jnp.ndim(prior) == 0
    if scalar and not jnp.issubdtype(jnp.asarray(prior).dtype, jnp.integer):
        raise ValueError(
            f"{what} is the scalar {prior!r}: a scalar prior is a control *level* (an integer "
            "index — the delta reading), so pass an int, or a probability vector over the "
            "node's levels to marginalize over who is steering"
        )
    if not scalar:
        return None, _prior_weights(prior, nu, what)
    keep = int(prior)
    if not 0 <= keep < nu:
        raise ValueError(f"{what} pins level {keep}, which is not one of its {nu} control level(s)")
    return keep, None


# ------------------------------------------------------------------- the builder


class _Node:
    """One control node of the composite under construction (mutable, builder-local)."""

    def __init__(self, name, action, kind, size, box, kernel):
        self.name, self.action = name, action
        self.kind, self.size, self.box, self.kernel = kind, size, box, kernel
        self.live = True


def merge(
    subsystems: Mapping[str, System],
    params: Mapping[str, SystemParams],
    bindings: Sequence[Binding] = (),
) -> tuple[System, SystemParams]:
    """Merge components into one composite ``(System, SystemParams)``; see the module docstring.

    ``subsystems`` and ``params`` are keyed alike; the insertion order of ``subsystems`` fixes the
    composite's node order: every component's state factors in turn, then every component's
    control nodes in turn (each followed by its action node), hence the action keys' and policy
    set's layout. ``bindings`` are applied in the order given. The composite always carries
    explicit ``factor_names``/``modality_names``/``action_names``: a component's generated names
    are namespaced.

    - Inference config (``policy_len``, ``planner``, ``gamma``, ``num_iters``, ``decomposition``,
      ``mixture_schedule``, any non-``None`` ``mixture_k``) must agree.
    - ``policy_prior`` composes as a sum in log space over the composite policy set, ``log E(π) =
      Σ_sub log E_sub(π|_sub)``; a component whose control nodes were merged away raises (declare
      the composite's habit on the returned params). No component habit gives ``None``.
    - ``epistemic`` is resolved and unioned (:func:`resolved_epistemic`); ``dims`` moves with its
      factors, under the namespaced names (a key ``pose:b1`` becomes ``arm.pose:b1``).
    - Designations keep each component's preferences under namespaced keys; ``C_deps``
      namespaces its keys and parents as ``H_deps`` does, a parent following its node through a
      same-step identification or a relay.
    - The G side does not compose: a component declaring ``System.terms``, either ``scoring`` half,
      or a policy module in ``policy_prior`` raises. Declare it on the composite.
    - ``recognition`` namespaces both halves, the parent addresses included (``Recognition("stoch",
      ("deter", "recognition:embed"))`` → ``Recognition("mc.stoch", ("mc.deter",
      "recognition:mc.embed"))``).
    - A continuous control node's box (``params.control``) and an action node's kernel
      (``params.actuator``) move with their nodes, under the namespaced names; a node identified
      with a driver's node leaves its box and its action node, the driver's governing.
    - ``params.durations`` carries under the component's prefix; a round has one clock, so two
      components that both declare durations, or a binding that removes the node carrying them,
      raise (design: step-durations.md T1).
    - ``params.precision`` is the composite's one policy precision: components that declare it
      must all declare the same value, and a component declaring it beside one that does not
      raises (the others read their ``System.gamma``, which must agree already).
    - A target on a node a binding removed raises, naming what the node became.

    The composite is validated (``System._validate``) before it is returned.
    """
    if not subsystems:
        raise ValueError("merge needs at least one subsystem")
    if set(subsystems) != set(params):
        raise ValueError(
            f"subsystems and params must be keyed alike; got {sorted(subsystems)} and "
            f"{sorted(params)}"
        )
    not_systems = sorted(name for name, sub in subsystems.items() if not isinstance(sub, System))
    if not_systems:
        raise TypeError(
            f"merge() component(s) {not_systems} are not Systems. A component is a System — a "
            "thing with kernels of its own, which is what a merge composes (its factors "
            "become the composite's factors, planned over and learned). SystemParams, and in "
            "particular a *scoring* entry's parameters, has no kernel at the loop boundary: it "
            "is material consumed interior to computing P(a|s), never a partner in the loop "
            "(design: system.md, the three parameter groups and the criterion "
            "that separates them)."
        )
    bad_names = [name for name in subsystems if SEPARATOR in name or not name]
    if bad_names:
        raise ValueError(
            f"subsystem name(s) {bad_names} contain the namespace separator {SEPARATOR!r} "
            "(or are empty); a component name is one path segment"
        )
    for system in subsystems.values():
        system._refuse_sensed("merge")

    #  each component's params as the positional view the concatenation indexes
    params = {name: subsystems[name]._params_in(params[name], "merge") for name in subsystems}

    # -------------------------------------------------- concatenate the frames
    fac_offset: dict[str, int] = {}
    mod_offset: dict[str, int] = {}
    node_offset: dict[str, int] = {}
    factor_names: list[str] = []
    modality_names: list[str] = []
    A_deps: list[tuple[int, ...]] = []
    parents: list[list[tuple[str, int]]] = []  # per state factor, ("state", g) / ("control", k)
    D_deps: list[tuple[int, ...]] = []
    H_deps: list[tuple[int, ...]] = []
    nodes: list[_Node] = []
    likelihood: list[object] = []
    transition: list[object] = []
    initial_state_prior: list[object] = []
    obs_targets: dict[str, object] = {}
    state_targets: dict[str, object] = {}
    terminal_targets: dict[str, object] = {}
    recognition_decl: list[Recognition] = []
    recognition_params: dict[str, object] = {}
    epistemic_keys: list[str] = []
    dims: list[tuple[str, tuple[str, ...]]] = []
    C_deps: list[tuple[str, tuple[str, ...]]] = []
    durations: dict[str, object] = {}

    scored = sorted(
        name
        for name, system in subsystems.items()
        if system.terms or system.scoring or params[name].scoring
    )
    if scored:
        raise NotImplementedError(
            f"subsystem(s) {scored} declare objective rows (System.terms / System.scoring / "
            "SystemParams.scoring), whose composition is not built. The obstacle is the key "
            "grammar, not the mathematics: a row is addressed by row *name* and the name "
            "segment of a terms= key is library vocabulary that namespacing does "
            "not touch, so two components' critics would both answer to 'g_tail' in the "
            "composite with no way to say which. Declare the composite's rows on the returned "
            "pair instead — its factors and modalities are namespaced, so "
            "'occupancy:transition:mind.stoch' addresses exactly one of them (design: "
            "amortization.md, the A12 implementation note on the terms= key grammar, flags 4 "
            "and 5). A component's *world model* merges as it always did; it is the G side "
            "that must be declared once, on the composite."
        )
    for name, system in subsystems.items():
        p = params[name]
        mods, facs = (system.modality_names, system._facs)
        if (
            len(p.transition) != len(system._B)
            or len(p.initial_state_prior) != len(system._B)
            or len(p.likelihood) != len(system.A_deps)
        ):
            raise ValueError(
                f"subsystem {name!r}: params declare {len(p.transition)} transition(s), "
                f"{len(p.initial_state_prior)} initial prior(s) and {len(p.likelihood)} "
                f"likelihood(s) for {len(system._B)} factor(s) and "
                f"{len(system.A_deps)} modality/ies — the pair is mismatched"
            )
        table = system._table
        f_off, m_off, k_off = len(factor_names), len(modality_names), len(nodes)
        fac_offset[name], mod_offset[name], node_offset[name] = f_off, m_off, k_off
        factor_names.extend(f"{name}{SEPARATOR}{f}" for f in facs)
        modality_names.extend(f"{name}{SEPARATOR}{m}" for m in mods)
        A_deps.extend(tuple(g + f_off for g in deps) for deps in system.A_deps)
        parents.extend(
            [
                (kind, i + (f_off if kind == "state" else k_off))
                for kind, i in declared_parents(table, f)
            ]
            for f in range(table.num_states)
        )
        D_deps.extend(tuple(g + f_off for g in deps) for deps in system._effective_D_deps())
        H_deps.extend(tuple(g + f_off for g in deps) for deps in system._effective_H_deps())
        space = system._space(p)
        kernels = system._actuator(p, "merge")
        for node, clock in system._durations(p, "merge").items():
            durations[f"{name}{SEPARATOR}{node}"] = clock
        nodes.extend(
            _Node(
                f"{name}{SEPARATOR}{node}",
                f"{name}{SEPARATOR}{action}",
                kind,
                size,
                box,
                kernels.get(action),
            )
            for node, action, kind, size, box in zip(
                table.control_names,
                system.action_names,
                space.kinds,
                space.sizes,
                space.boxes,
                strict=True,
            )
        )
        likelihood.extend(p.likelihood)
        transition.extend(p.transition)
        initial_state_prior.extend(p.initial_state_prior)
        for src, dst in (
            (p.obs_targets, obs_targets),
            (p.state_targets, state_targets),
            (p.terminal_targets, terminal_targets),
            (p.recognition, recognition_params),
        ):
            dst.update({f"{name}{SEPARATOR}{k}": v for k, v in src.items()})
        recognition_decl.extend(
            Recognition(
                variable=f"{name}{SEPARATOR}{entry.variable}",
                parents=tuple(namespaced_key(name, parent) for parent in entry.parents),
            )
            for entry in (system.recognition or ())
        )
        epistemic_keys.extend(namespaced_key(name, key) for key in resolved_epistemic(system, p))
        dims.extend((f"{name}{SEPARATOR}{factor}", names) for factor, names in system.dims)
        C_deps.extend(
            (namespaced_key(name, key), tuple(namespaced_key(name, p) for p in parents))
            for key, parents in system.C_deps
        )

    factor_index = {name: i for i, name in enumerate(factor_names)}
    modality_index = {name: i for i, name in enumerate(modality_names)}
    node_index = {node.name: k for k, node in enumerate(nodes)}
    owner = {
        f: name
        for name, off in fac_offset.items()
        for f in range(off, off + subsystems[name]._table.num_states)
    }
    #  the nodes a nested composite's own bindings removed, with what each became
    nested_gone = {
        node: tail
        for name, system in subsystems.items()
        for node, tail in system._bound_away(prefix=f"{name}{SEPARATOR}")
    }
    relayed: dict[int, int] = {}  # node -> the state it was identified with
    relay_reader: dict[int, int] = {}  # node -> the factor that read it
    terminations: dict[int, tuple] = {}  # a terminated node's (keep, weights)
    same_step: dict[str, str] = {}

    def resolve(node: str) -> str:
        """The node ``node`` is, following same-step identifications to their end."""
        while node in same_step:
            node = same_step[node]
        return node

    def readers_of(k: int) -> list[tuple[int, int]]:
        """``(factor, position)`` for every factor that reads node ``k``."""
        return [
            (g, pos)
            for g, ps in enumerate(parents)
            for pos, (kind, i) in enumerate(ps)
            if kind == "control" and i == k
        ]

    def control_parents(g: int) -> list[tuple[int, int]]:
        return [(pos, i) for pos, (kind, i) in enumerate(parents[g]) if kind == "control"]

    def relay_refusal(state: str, node: str, readers: list[int], partners: list[str]):
        """A relay into a node more than one factor reads: one sentence for either order."""
        source = (
            f"made one with {sorted(partners)} by a same-step ActionControl"
            if partners
            else "shared within its component"
        )
        same_step_node = (
            repr(control_name(state)) if control_name(state) in node_index else "a control node"
        )
        return ValueError(
            f"ActionControl({state!r} -> {node!r}) merges a state with {node!r}, which "
            f"{[factor_names[g] for g in sorted(readers)]} read ({source}); a relay re-reads "
            f"one factor's control axis, so bind from {same_step_node} for a same-step read, or "
            "relay into a node one factor reads"
        )

    def driven_node(name: str, kind: str) -> int:
        """The live control node ``name`` a binding identifies or terminates."""
        live = [node.name for node in nodes if node.live]
        if name in node_index:
            k = node_index[name]
            if nodes[k].live:
                return k
            if k in relayed:
                to = [factor_names[relayed[k]]]
            elif k in terminations:
                to = "a boundary prior"
            else:
                to = repr(resolve(name))
            raise ValueError(
                f"control node {name!r} is already merged (with {to}); a node is identified "
                "with exactly one partner"
            )
        if name in nested_gone:
            raise ValueError(f"{kind} names {name!r}, {nested_gone[name]}")
        actions = {node.action: node.name for node in nodes}
        if name in actions:
            raise ValueError(
                f"{kind} names {name!r}, an action node, where a binding names a control "
                f"node: pass {actions[name]!r}, its control node"
            )
        if name in factor_index:
            if control_name(name) in node_index:
                raise ValueError(
                    f"{kind} names {name!r}, a state factor, where a binding names a control "
                    f"node: pass {control_name(name)!r}, the control node of {name!r}"
                )
            raise ValueError(
                f"{kind} names {name!r}, a state factor, where a binding names a control node; "
                f"pass one of the control nodes {live}"
            )
        raise ValueError(
            f"{kind} names unknown control node {name!r}; the composite's control nodes are "
            f"{live} (names are subsystem-qualified with {SEPARATOR!r}, as in "
            f"'boat{SEPARATOR}u_x')"
        )

    def kernels_agree(c: int, d: int | None, what: str) -> None:
        """Refuse a binding that removes node ``c`` when its action node's kernel would be
        dropped: with no driver node ``d`` any kernel, else one unequal to the driver's."""
        kc = nodes[c].kernel
        kd = None if d is None else nodes[d].kernel
        if kc is None and kd is None:
            return
        if kc is not None and kd is not None and kc.shape == kd.shape:
            if bool(jnp.array_equal(kc, kd)):
                return
        if d is None:
            raise ValueError(
                f"{what} removes control node {nodes[c].name!r}, and params.actuator holds a "
                f"kernel for its action node {nodes[c].action!r} that would be dropped: drop the "
                "kernel, or leave the node to be chosen"
            )
        raise ValueError(
            f"{what} makes control node {nodes[c].name!r} one with {nodes[d].name!r}, and the "
            f"kernels params.actuator holds for their action nodes {nodes[c].action!r} and "
            f"{nodes[d].action!r} differ: declare the same kernel on both, or on neither"
        )

    def cut(k: int, keep, weights) -> None:
        """Slice or average node ``k`` out of every table that reads it."""
        for g, pos in readers_of(k):
            transition[g] = _cut_axes(
                transition[g], factor_names[g], len(control_parents(g)), [(pos, keep, weights)]
            )
            del parents[g][pos]
        nodes[k].live = False

    def identify(d: int, c: int, binding: ActionControl) -> None:
        """Same-step: every reader of node ``c`` reads node ``d``, and ``c`` leaves. A driver a
        BoundaryPrior terminated terminates ``c``'s readers with the same prior."""
        drv, ctl = binding.driver, binding.control
        if d == c:
            raise ValueError(
                f"ActionControl({drv!r} -> {ctl!r}) pairs {nodes[c].name!r} with itself (the "
                "bindings make both names one node); a binding identifies two control nodes"
            )
        nd, nc = nodes[d], nodes[c]
        noun = {LEVEL: "discrete", FLOAT: "continuous"}
        if nd.kind != nc.kind:
            raise ValueError(
                f"node kind mismatch: driver node {drv!r} is {noun[nd.kind]} and {ctl!r} is "
                f"{noun[nc.kind]}; a same-step binding makes two nodes one, so bind nodes of one "
                "kind"
            )
        if nd.size != nc.size:
            unit = "level(s)" if nd.kind == LEVEL else "dimension(s)"
            raise ValueError(
                f"node arity mismatch: driver node {drv!r} has {nd.size} {unit} but {ctl!r} has "
                f"{nc.size}; a same-step binding makes two nodes one, so their "
                f"{'level counts' if nd.kind == LEVEL else 'widths'} must agree"
            )
        if d in relayed:  # the driver's node was merged with a state
            partners = [k for k in same_step if resolve(k) == nd.name]
            raise relay_refusal(
                factor_names[relayed[d]],
                nd.name,
                [relay_reader[d], *(g for g, _ in readers_of(c))],
                [*partners, ctl],
            )
        kernels_agree(c, None if d in terminations else d, f"ActionControl({drv!r} -> {ctl!r})")
        if d in terminations:
            cut(c, *terminations[d])
            same_step[ctl] = drv
            return
        both = [
            factor_names[g] for g, _ in readers_of(c) if any(i == d for _, i in control_parents(g))
        ]
        if both:
            raise ValueError(
                f"ActionControl({drv!r} -> {ctl!r}): {both} read both nodes, so making them one "
                f"would have each read one node twice; bind a node that no reader of {drv!r} "
                "reads"
            )
        for g, pos in readers_of(c):
            parents[g][pos] = ("control", d)
        nc.live = False
        same_step[ctl] = drv

    # ----------------------------------------------------------- the bindings
    for binding in bindings:
        if isinstance(binding, ActionControl):
            if binding.driver in node_index:
                d = node_index[resolve(binding.driver)]
                identify(d, driven_node(binding.control, "ActionControl"), binding)
                continue
            if binding.driver not in factor_index:
                if binding.driver in nested_gone:
                    raise ValueError(
                        f"ActionControl names {binding.driver!r}, {nested_gone[binding.driver]}"
                    )
                raise ValueError(
                    f"ActionControl driver {binding.driver!r} is neither a control node nor a "
                    "state factor of the composite: pass a control node (read at the same "
                    f"step), one of {sorted(node_index)}, or a state factor (read a step later), "
                    f"one of {list(factor_names)}"
                )
            d = factor_index[binding.driver]
            c = driven_node(binding.control, "ActionControl")
            levels = _factor_levels(params[owner[d]], d - fac_offset[owner[d]])
            if levels is None:
                hint = (
                    repr(control_name(binding.driver))
                    if control_name(binding.driver) in node_index
                    else "a control node"
                )
                raise NotImplementedError(
                    f"ActionControl driver {binding.driver!r} is a continuous state, and a relay "
                    f"reads a state's level: bind from {hint} for a same-step read"
                )
            if nodes[c].kind == FLOAT:
                raise ValueError(
                    f"node kind mismatch: driver {binding.driver!r} is a discrete state and "
                    f"{binding.control!r} is a continuous node; a relay reads a state's level, "
                    f"so drive {binding.control!r} from a continuous control node at the same "
                    "step"
                )
            if levels != nodes[c].size:
                raise ValueError(
                    f"node arity mismatch: driver {binding.driver!r} has {levels} level(s) "
                    f"but {binding.control!r} has {nodes[c].size}; a relay identifies a state's "
                    "levels with a control node's one-for-one"
                )
            readers = readers_of(c)
            if len(readers) != 1 or len(control_parents(readers[0][0])) != 1:
                partners = [k for k in same_step if resolve(k) == binding.control]
                raise relay_refusal(
                    binding.driver, binding.control, [g for g, _ in readers], partners
                )
            kernels_agree(c, None, f"ActionControl({binding.driver!r} -> {binding.control!r})")
            g, pos = readers[0]
            transition[g] = _drop_control_axis(transition[g], factor_names[g], None, None)
            parents[g][pos] = ("state", d)
            nodes[c].live = False
            relayed[c] = d
            relay_reader[c] = g
        elif isinstance(binding, BoundaryPrior):
            c = driven_node(binding.control, "BoundaryPrior")
            if nodes[c].kind == FLOAT:
                raise NotImplementedError(
                    f"BoundaryPrior for {binding.control!r}: a continuous control node's prior "
                    "would have to enter its readers as a continuous parent distribution, which "
                    "no transition family accepts; terminate a discrete node"
                )
            keep, weights = _termination(
                binding.prior, nodes[c].size, f"BoundaryPrior for {binding.control!r}"
            )
            kernels_agree(c, None, f"BoundaryPrior for {binding.control!r}")
            cut(c, keep, weights)
            terminations[c] = (keep, weights)
        elif isinstance(binding, ObservationLink):
            m = _lookup(binding.modality, modality_index, "modality")
            linked = tuple(_lookup(f, factor_index, "factor") for f in binding.factors)
            a = likelihood[m]
            if isinstance(a, ConditionalCategorical | ConditionalDirichlet):
                if a.parent_ndims != len(linked):
                    raise ValueError(
                        f"ObservationLink for {binding.modality!r} names {len(linked)} "
                        f"parent(s) but its likelihood table has {a.parent_ndims} parent "
                        "axis/es; a binding states the modality's complete new parent list, "
                        "in A_deps order"
                    )
                sizes = tuple(
                    _factor_levels(params[owner[g]], g - fac_offset[owner[g]]) for g in linked
                )
                if all(s is not None for s in sizes) and a.parent_shape != sizes:
                    raise ValueError(
                        f"support mismatch: modality {binding.modality!r} has parent axes "
                        f"{a.parent_shape} but {list(binding.factors)} have {sizes} level(s); "
                        "an observation binding pairs a modality's parent axes with the "
                        "observed factors' supports"
                    )
            A_deps[m] = linked
        elif isinstance(binding, TargetEmission):
            f = _lookup(binding.target, factor_index, "factor")
            emitters = tuple(_lookup(g, factor_index, "factor") for g in binding.parents)
            if H_deps[f]:
                raise ValueError(
                    f"factor {binding.target!r} already declares H_deps parent(s) "
                    f"{[factor_names[g] for g in H_deps[f]]} of its own; a target-emission "
                    "binding declares the parents of an as-yet-unconditioned terminal "
                    "target, it does not override a component's own declaration"
                )
            if not emitters:
                raise ValueError(
                    f"TargetEmission for {binding.target!r} names no parents; a conditional "
                    "terminal designation is conditional on something"
                )
            H_deps[f] = emitters
        else:
            raise TypeError(
                f"unknown binding {binding!r}; the kinds are ActionControl (control-node "
                "identification or relay), ObservationLink (a modality's parents), "
                "TargetEmission (H_deps, N1) and BoundaryPrior (a control node's fixed prior)"
            )

    same_step = {node: resolve(node) for node in same_step}
    for what, targets in (("state_targets", state_targets), ("terminal_targets", terminal_targets)):
        for key in targets:
            if key in same_step and nodes[node_index[same_step[key]]].live:
                raise ValueError(
                    f"merge: {what}[{key!r}] is a target on a control node a same-step "
                    f"ActionControl made one with {same_step[key]!r}, so the composite's node is "
                    f"{same_step[key]!r}: declare the target on {same_step[key]!r}"
                )
            if key in same_step:
                raise ValueError(
                    f"merge: {what}[{key!r}] is a target on a control node a same-step "
                    f"ActionControl made one with {same_step[key]!r}, which a BoundaryPrior "
                    "terminated, so the composite has no such node: drop the target"
                )
            k = node_index.get(key)
            if k in relayed or k in terminations:
                how = (
                    f"merged with the state {factor_names[relayed[k]]!r}"
                    if k in relayed
                    else "terminated with a BoundaryPrior"
                )
                raise ValueError(
                    f"merge: {what}[{key!r}] is a target on a control node {how}, so the "
                    "composite has no such node: drop the target"
                )

    #  a conditional target's parent follows its node to the driver a same-step ActionControl
    #  made it one with; a relayed or terminated node is gone
    for i, (key, c_parents) in enumerate(C_deps):
        renamed = []
        for p in c_parents:
            p = resolve(p)
            k = node_index.get(p)
            if k in terminations:
                raise ValueError(
                    f"merge: the target on {key!r} is conditional on the control node {p!r}, "
                    "which a BoundaryPrior terminated, so the composite has no such node: drop "
                    "the target or condition it on another node"
                )
            if k in relayed:
                raise ValueError(
                    f"merge: the target on {key!r} is conditional on the control node {p!r}, "
                    f"which a relay identified with the state {factor_names[relayed[k]]!r}; a "
                    "relay reads the state a step later while a per-step target reads the "
                    "step's own belief, so condition the target on the driver or drop the "
                    "relay"
                )
            renamed.append(p)
        C_deps[i] = (key, tuple(renamed))

    # every same-step identification, the components' own re-qualified, as (node, driver) names
    identified = [
        (namespaced_key(name, node), namespaced_key(name, driver))
        for name, system in subsystems.items()
        for _, sub in system.subsystems or ()
        for node, driver in sub.identified
    ]
    identified = list(dict.fromkeys([*identified, *sorted(same_step.items())]))

    # ---------------------------------------------------------- the metadata
    # One entry per component, plus a re-qualified, offset entry per nested component. A nested
    # entry inherits its own merge's record, with this merge's bindings added on top.
    live_names = [node.name for node in nodes if node.live]
    action_of = {node.name: node.action for node in nodes}
    for name, system in subsystems.items():
        for _, sub in system.subsystems or ():
            for node, action in sub.actions:
                action_of.setdefault(namespaced_key(name, node), namespaced_key(name, action))
    this_relayed = {nodes[k].name: factor_names[d] for k, d in relayed.items()}
    this_terminated = [nodes[k].name for k in terminations]
    entries: list[tuple[str, tuple[int, ...], tuple[int, ...], dict, list]] = []
    for name, system in subsystems.items():
        f_off, m_off = fac_offset[name], mod_offset[name]
        facs = tuple(range(f_off, f_off + len(system._B)))
        mods = tuple(range(m_off, m_off + len(system.A_deps)))
        entries.append((name, facs, mods, {}, []))
        for child_name, child in system.subsystems or ():
            entries.append(
                (
                    f"{name}{SEPARATOR}{child_name}",
                    tuple(f + f_off for f in child.factors),
                    tuple(m + m_off for m in child.modalities),
                    {
                        namespaced_key(name, node): namespaced_key(name, state)
                        for node, state in child.relayed
                    },
                    [namespaced_key(name, node) for node in child.terminated],
                )
            )
    metadata: list[tuple[str, Subsystem]] = []
    for name, facs, mods, inherited_relayed, inherited_terminated in entries:

        def mine(node: str, prefix: str = f"{name}{SEPARATOR}") -> bool:
            return node.startswith(prefix)

        relays = {**inherited_relayed, **{k: v for k, v in this_relayed.items() if mine(k)}}
        ended = list(
            dict.fromkeys([*inherited_terminated, *(k for k in this_terminated if mine(k))])
        )
        pairs = tuple(pair for pair in identified if mine(pair[0]))
        chosen = tuple(node for node in live_names if mine(node))
        relays = dict(sorted(relays.items(), key=lambda kv: _node_order(kv[0], factor_names)))
        ended = sorted(ended, key=lambda node: _node_order(node, factor_names))
        named = [*chosen, *relays, *ended, *(node for node, _ in pairs)]
        metadata.append(
            (
                name,
                Subsystem(
                    factors=facs,
                    modalities=mods,
                    controls=chosen,
                    actions=tuple((node, action_of[node]) for node in dict.fromkeys(named)),
                    relayed=tuple(relays.items()),
                    terminated=tuple(ended),
                    identified=pairs,
                    free_modalities=tuple(m for m in mods if all(g in facs for g in A_deps[m])),
                    bound_modalities=tuple(
                        (m, tuple(g for g in A_deps[m] if g not in facs))
                        for m in mods
                        if any(g not in facs for g in A_deps[m])
                    ),
                    target_emissions=tuple(
                        (f, H_deps[f])
                        for f in facs
                        if H_deps[f] and any(g not in facs for g in H_deps[f])
                    ),
                ),
            )
        )

    # ------------------------------------------------------ the composite System
    mixture_ks = {s.mixture_k for s in subsystems.values()} - {None}
    if len(mixture_ks) > 1:
        raise ValueError(
            f"merged Systems declare different mixture budgets mixture_k={sorted(mixture_ks)}; "
            "the budget is shared by the whole belief carry, so declare one value"
        )
    if all(s.epistemic is True for s in subsystems.values()):
        epistemic: bool | tuple[str, ...] = True
    elif all(s.epistemic is False or s.epistemic == () for s in subsystems.values()):
        epistemic = False
    else:
        epistemic = tuple(epistemic_keys)
    live = [k for k, node in enumerate(nodes) if node.live]
    if len(durations) > 1:
        raise ValueError(
            f"components declare durations on {sorted(durations)}: a round steps every factor "
            "once, so the composite has one clock; declare durations on one component's control "
            "node (design: step-durations.md T1)"
        )
    precisions = {name: p.precision for name, p in params.items() if p.precision is not None}
    if precisions and len(precisions) != len(params):
        raise ValueError(
            f"components {sorted(precisions)} declare params.precision and the others do not: "
            "the composite plans with one policy precision, so declare it on every component "
            "with the same value, or on none and declare the composite's on the returned params"
        )
    if len({float(v) for v in precisions.values()}) > 1:
        raise ValueError(
            f"components declare different params.precision values "
            f"{ {n: float(v) for n, v in precisions.items()} }: the composite plans with one "
            "policy precision, so declare one value"
        )
    removed = sorted(set(durations) - {nodes[k].name for k in live})
    if removed:
        raise ValueError(
            f"a binding removes control node {removed[0]!r}, which carries the durations; "
            "declare the composite's durations on a control node that stays "
            "(design: step-durations.md T1)"
        )
    n_states = len(factor_names)
    column = {k: n_states + j for j, k in enumerate(live)}
    B_deps = tuple(tuple(i if kind == "state" else column[i] for kind, i in ps) for ps in parents)
    pad = ((),) * len(live)
    composite = System(
        A_deps=tuple(A_deps),
        B_deps=(*B_deps, *pad),
        controls=tuple(column[k] for k in live),
        action_names=tuple(nodes[k].action for k in live),
        policy_len=_agree("policy_len", subsystems),
        planner=_agree("planner", subsystems),
        gamma=_agree("gamma", subsystems),
        num_iters=_agree("num_iters", subsystems),
        decomposition=_agree("decomposition", subsystems),
        planning_belief=_agree("planning_belief", subsystems),
        feedback=_agree("feedback", subsystems),
        epistemic=epistemic,
        mixture_k=next(iter(mixture_ks), None),
        mixture_schedule=_agree("mixture_schedule", subsystems),
        D_deps=None if all(not d for d in D_deps) else (*D_deps, *pad),
        H_deps=None if all(not h for h in H_deps) else (*H_deps, *pad),
        C_deps=tuple(C_deps),
        modality_names=tuple(modality_names),
        factor_names=(*factor_names, *(nodes[k].name for k in live)),
        dims=tuple(dims),
        recognition=tuple(recognition_decl) or None,
        subsystems=tuple(metadata),
    )
    boxes = {nodes[k].name: nodes[k].box for k in live if nodes[k].box is not None}
    composite_params = SystemParams(
        transition=tuple(transition),
        initial_state_prior=tuple(initial_state_prior),
        likelihood=tuple(likelihood),
        recognition=recognition_params,
        obs_targets=obs_targets,
        state_targets=state_targets,
        terminal_targets=terminal_targets,
        policy_prior=_compose_policy_prior(
            subsystems,
            params,
            composite,
            tuple(nodes[k].size if nodes[k].kind == LEVEL else 1 for k in live),
            node_offset,
            {k for k, node in enumerate(nodes) if not node.live},
            bool(boxes),
        ),
        control=boxes,
        actuator={nodes[k].action: nodes[k].kernel for k in live if nodes[k].kernel is not None},
        durations=durations,
        precision=next(iter(precisions.values()), None),
    )
    composite._validate(composite_params)
    return composite, composite._params_out(composite_params)


def _node_order(node: str, factor_names: Sequence[str]) -> tuple[int, str]:
    """Sort key for a record's node names: the state a sugar-named node ``u_<f>`` belongs to,
    by factor index, before any other node, by name."""
    space, sep, leaf = node.rpartition(SEPARATOR)
    state = f"{space}{sep}{leaf[2:]}" if leaf.startswith("u_") else None
    if state in factor_names:
        return (list(factor_names).index(state), "")
    return (len(factor_names), node)


def _compose_policy_prior(
    subsystems: Mapping[str, System],
    params: Mapping[str, SystemParams],
    composite: System,
    levels: tuple[int, ...],
    node_offset: Mapping[str, int],
    merged_away: set[int],
    continuous: bool = False,
) -> Array | None:
    """Sum the components' habit priors over the composite policy set.

    ``log E(π) = Σ_sub log E_sub(π restricted to sub's control nodes)``, as a static gather: each
    component's policy index is recomputed from the composite enumeration's digits.
    """
    declared = {name: p.policy_prior for name, p in params.items() if p.policy_prior is not None}
    if not declared:
        return None
    modules = sorted(
        name
        for name, prior in declared.items()
        if not hasattr(prior, "shape") and not isinstance(prior, Categorical)
    )
    if modules:
        raise NotImplementedError(
            f"subsystem(s) {modules} declare a *policy module* in policy_prior (A3's habit net), "
            "which does not compose as a sum over the composite's joint policy set: a module is "
            "a conditional prior P(u|s) over its own beliefs, so composing two of them is a "
            "question about the joint control-state distribution of two minds, not a log-space "
            "addition. Declare the composite's own habit on the returned params, or merge "
            "world models and keep each mind's actor in its own subsystem() view."
        )
    if continuous:
        raise NotImplementedError(
            f"subsystem(s) {sorted(declared)} declare policy_prior, but the composite's control "
            "space is continuous: a habit prior over an enumerated policy set has no meaning "
            "there (CEM's E is its proposal distribution — design: amortization.md A9). Drop the "
            "policy_prior leaves before merging."
        )
    policies = enumerate_policies(levels, composite.policy_len, (True,) * len(levels))
    total = jnp.zeros((policies.shape[0],), jnp.float32)
    #  each original node's composite column, for the nodes that stayed
    columns: dict[int, int] = {}
    j = 0
    for k in range(sum(len(s._table.control_names) for s in subsystems.values())):
        if k not in merged_away:
            columns[k] = j
            j += 1
    for name, policy_prior in declared.items():
        system = subsystems[name]
        off = node_offset[name]
        own = range(off, off + len(system._table.control_names))
        rebound = [k - off for k in own if k in merged_away]
        if rebound:
            names = [system._table.control_names[k] for k in rebound]
            raise ValueError(
                f"subsystem {name!r} declares policy_prior over its own policy set, but its "
                f"control node(s) {names} were merged in this call, so that policy set does "
                "not exist in the composite. Drop the component's policy_prior "
                "(dataclasses.replace(params, policy_prior=None)) and declare the composite's "
                "habit prior on the returned SystemParams."
            )
        own_levels = system._control_levels(params[name])
        expected = 1
        for _ in range(system.policy_len):
            for n in own_levels:
                expected *= n
        lp = (
            policy_prior.log_probs
            if isinstance(policy_prior, Categorical)
            else jnp.asarray(policy_prior)
        )
        if lp.shape != (expected,):
            raise ValueError(
                f"subsystem {name!r} has policy_prior of shape {lp.shape}, but its own policy "
                f"set has {expected} member(s) (levels={own_levels}, "
                f"policy_len={system.policy_len}); a habit prior is one weight per policy"
            )
        index = jnp.zeros((policies.shape[0],), jnp.int32)
        for t in range(system.policy_len):
            for k, n in zip(own, own_levels, strict=True):
                index = index * n + policies[:, t, columns[k]]
        total = total + lp[index]
    return total


# ------------------------------------------------------- the subjective-view constructor


def _subsystem_meta(composite: System, name: str) -> Subsystem:
    if composite.subsystems is None:
        raise ValueError(
            f"this System carries no subsystems metadata, so it has no component named "
            f"{name!r} to take a view of; subjective views are derived from the record "
            "merge() writes (nesting-systems.md req. 7 — the decomposition cannot be "
            "recovered from the wiring alone)"
        )
    table = dict(composite.subsystems)
    if name not in table:
        raise ValueError(
            f"unknown subsystem {name!r}; this composite's components are {sorted(table)}"
        )
    return table[name]


def subsystem_factors(composite: System, name: str) -> tuple[int, ...]:
    """Composite factor indices one mind's view carries, in composite order.

    Its own factors, plus every other component's factor that is a direct parent of something of
    ours (a ``B_deps``/``D_deps`` parent of our factor, an ``A_deps`` parent of our modality, an
    ``H_deps`` parent of our terminal designation, a ``C_deps`` state parent of our per-step
    designation), closed transitively over those stubs' own
    declared dynamics, since a stub is carried verbatim. Composite order keeps ``D_deps``
    ancestral.
    """
    meta = _subsystem_meta(composite, name)
    D_deps, H_deps = composite._effective_D_deps(), composite._effective_H_deps()
    table = composite._table
    needed = set(meta.factors)
    stack: list[int] = []
    for f in meta.factors:
        stack.extend(composite._B[f] + D_deps[f] + H_deps[f])
    for m in meta.modalities:
        stack.extend(composite.A_deps[m])
    mine = {*(table.state_names[f] for f in meta.factors)}
    mine |= {composite.modality_names[m] for m in meta.modalities}
    for key, parents in composite._effective_C_deps().items():
        if key in mine:
            stack.extend(f for kind, f in parents if kind == "state")
    while stack:
        g = stack.pop()
        if g in needed:
            continue
        needed.add(g)
        # a stub carries its declared dynamics verbatim, so its own parents come too
        stack.extend(composite._B[g] + D_deps[g])
    return tuple(sorted(needed))


def subsystem(
    composite: System,
    params: SystemParams,
    name: str,
    partner_priors: Mapping[str, "int | Categorical | Array"] | None = None,
) -> tuple[System, SystemParams]:
    """One mind's bona-fide ``(System, SystemParams)``, derived from the composite and its metadata.

    The view holds:

    - its own factors and modalities, under the composite's namespaced names, and the control
      nodes it still chooses (``Subsystem.controls``) with their action nodes;
    - boundary stubs (:func:`subsystem_factors`) as ordinary latent factors with their declared
      dynamics;
    - for every control node another component chooses that a factor of the view reads (a stub's
      own node, or a node a same-step ``ActionControl`` made one with ours), that factor's axis
      marginalised under a prior over the node: uniform by default, or given in
      ``partner_priors`` keyed by the node's name, read as in :class:`BoundaryPrior`; the node
      itself is absent, as is a node merged away;
    - nothing for an out-bound observation link.

    Parameters are the composite's arrays, shared by reference; beliefs are private (they live in
    the caller's carry). Other components' designations and ``epistemic`` keys naming absent
    variables are dropped; ``policy_prior`` is carried only when the view's policy set is the
    composite's, and raises otherwise.

    Design: docs/design/nesting-systems.md (architecture sketch; partner modeling).
    """
    composite._refuse_sensed("subsystem")
    params = composite._params_in(params, "subsystem")
    if composite._is_amortized(params):
        raise NotImplementedError(
            f"subsystem({name!r}) of an amortized composite: a per-mind view would have to slice "
            "the recognition declaration too — which q-side factors this mind owns, and what a "
            "conditioning set that reached across the old boundary becomes — and that is a "
            "genuine nesting question about *inference* structure, not a projection of the "
            "generative one (design: amortization.md A11, nesting-systems.md req. 7). merge() "
            "itself namespaces recognition and works; only this view is deferred."
        )
    meta = _subsystem_meta(composite, name)
    facs = subsystem_factors(composite, name)
    mods = meta.modalities
    own = set(meta.factors)
    index = {f: i for i, f in enumerate(facs)}
    mod_index = {m: i for i, m in enumerate(mods)}
    table = composite._table
    c_mods, c_facs, c_nodes = composite.modality_names, table.state_names, table.control_names
    chosen = set(meta.controls)
    ours = [k for k, node in enumerate(c_nodes) if node in chosen]
    #  another mind's node a factor of the view reads, by node name
    partner_nodes = {
        c_nodes[k]: k for f in facs for k, _ in table.control_parents[f] if k not in ours
    }

    priors = dict(partner_priors or {})
    unknown = sorted(set(priors) - set(partner_nodes))
    action_node = dict(zip(composite.action_names, c_nodes, strict=True))
    gone = dict(composite._bound_away())
    for key in unknown:
        if key in action_node:
            raise ValueError(
                f"partner_priors names {key!r}, the action node of control node "
                f"{action_node[key]!r}; key the prior by {action_node[key]!r}"
            )
        if key in gone:
            raise ValueError(f"partner_priors names {key!r}, {gone[key]}")
    if unknown:
        raise ValueError(
            f"partner_priors names {unknown}, which are not control nodes the view of {name!r} "
            "reads and another component chooses; a partner prior declares what this mind "
            f"expects of another component's control node, and this view reads "
            f"{sorted(partner_nodes)}"
        )
    levels = composite._control_levels(params)
    space = composite._space(params)

    def partner_prior(k: int) -> tuple[object, str]:
        """The declared prior over node ``k`` (uniform by default), and its label."""
        key = c_nodes[k]
        if key in priors:
            return priors[key], f"partner_priors[{key!r}]"
        return jnp.full(levels[k], 1.0 / levels[k], jnp.float32), f"partner_priors[{key!r}]"

    transition: list[object] = []
    view_parents: list[list[tuple[str, int]]] = []
    for f in facs:
        reads = table.control_parents[f]
        cuts = [(pos, k) for k, pos in reads if k not in ours]
        view_parents.append(
            [(kind, i) for kind, i in declared_parents(table, f) if kind == "state" or i in ours]
        )
        if not cuts:
            transition.append(params.transition[f])
            continue
        boxed = [c_nodes[k] for _, k in cuts if space.kinds[k] == FLOAT]
        if boxed:
            raise NotImplementedError(
                f"factor {c_facs[f]!r} of the view of {name!r} reads another component's "
                f"continuous control node {boxed}: a prior over a partner's *continuous* "
                "control node would have to enter the declaration as a continuous parent "
                "distribution, which no transition family accepts (design: controls.py, "
                "amortization.md A9)"
            )
        terminations = []
        for pos, k in cuts:
            prior, label = partner_prior(k)
            terminations.append((pos, *_termination(prior, levels[k], label)))
        transition.append(_cut_axes(params.transition[f], c_facs[f], len(reads), terminations))

    n_view = len(facs)
    column = {k: n_view + j for j, k in enumerate(ours)}
    pad = ((),) * len(ours)

    def remap(deps: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(index[g] for g in deps)

    D_all, H_all = composite._effective_D_deps(), composite._effective_H_deps()
    B_deps = tuple(
        tuple(index[i] if kind == "state" else column[i] for kind, i in ps) for ps in view_parents
    )
    D_deps = tuple(remap(D_all[f]) for f in facs)
    H_deps = tuple(remap(H_all[f]) if f in own else () for f in facs)
    A_deps = tuple(remap(composite.A_deps[m]) for m in mods)

    view_facs = tuple(c_facs[f] for f in facs)
    view_mods = tuple(c_mods[m] for m in mods)
    view_nodes = tuple(c_nodes[k] for k in ours)
    view_actions = tuple(composite.action_names[k] for k in ours)
    #  a target on a node goes with its view while the view still chooses it
    targeted = {c_facs[f] for f in own} | set(view_nodes)
    C_deps = []
    for key, c_parents in composite.C_deps:
        if key not in targeted and key not in view_mods:
            continue
        absent = [p for p in c_parents if p in c_nodes and p not in view_nodes]
        if absent:
            raise NotImplementedError(
                f"the view of {name!r} carries the target on {key!r}, which is conditional on "
                f"{absent}, another component's control node; a per-mind view resolves a target "
                "under its own choices and beliefs, and a partner's choice is neither. Condition "
                "the target on this mind's node or a state factor, or drop it"
            )
        C_deps.append((key, c_parents))
    view_params = SystemParams(
        transition=tuple(transition),
        initial_state_prior=tuple(params.initial_state_prior[f] for f in facs),
        likelihood=tuple(params.likelihood[m] for m in mods),
        obs_targets={k: v for k, v in params.obs_targets.items() if k in set(view_mods)},
        state_targets={k: v for k, v in params.state_targets.items() if k in targeted},
        terminal_targets={k: v for k, v in params.terminal_targets.items() if k in targeted},
        policy_prior=_view_policy_prior(composite, params, ours, name),
        control={k: v for k, v in params.control.items() if k in set(view_nodes)},
        actuator={k: v for k, v in params.actuator.items() if k in set(view_actions)},
        durations={k: v for k, v in params.durations.items() if k in set(view_nodes)},
        precision=params.precision,
    )
    view = System(
        A_deps=A_deps,
        B_deps=(*B_deps, *pad),
        controls=tuple(column[k] for k in ours),
        action_names=view_actions,
        policy_len=composite.policy_len,
        planner=composite.planner,
        gamma=composite.gamma,
        num_iters=composite.num_iters,
        decomposition=composite.decomposition,
        planning_belief=composite.planning_belief,
        feedback=composite.feedback,
        epistemic=False,
        mixture_k=composite.mixture_k,
        mixture_schedule=composite.mixture_schedule,
        D_deps=None if all(not d for d in D_deps) else (*D_deps, *pad),
        H_deps=None if all(not h for h in H_deps) else (*H_deps, *pad),
        C_deps=tuple(C_deps),
        modality_names=view_mods,
        factor_names=(*view_facs, *view_nodes),
        dims=tuple((factor, names) for factor, names in composite.dims if factor in view_facs),
        subsystems=_view_metadata(composite, name, index, mod_index, set(view_nodes), A_deps),
    )
    mine = {c_facs[f] for f in own} | set(view_mods)
    view = dataclasses.replace(view, epistemic=_view_epistemic(composite, view, view_params, mine))
    view._validate(view_params)
    return view, view._params_out(view_params)


def _view_policy_prior(
    composite: System,
    params: SystemParams,
    ours: list[int],
    name: str,
) -> "Array | None":
    """The composite's habit prior if the view's policy set is the composite's, else raise.

    The sets coincide when the view chooses every control node of the composite.
    Otherwise declare the mind's own ``policy_prior`` on the returned params.
    """
    if params.policy_prior is None:
        return None
    if not hasattr(params.policy_prior, "shape") and not isinstance(
        params.policy_prior, Categorical
    ):
        raise NotImplementedError(
            f"the composite declares a policy module in policy_prior and the view of {name!r} "
            "cannot inherit it: a module reads *beliefs*, and this mind's beliefs are not "
            "the composite's. Declare this mind's own actor on the returned SystemParams."
        )
    if list(ours) == list(range(len(composite._table.control_names))):
        return params.policy_prior
    raise NotImplementedError(
        f"the composite declares policy_prior over its joint policy set, but the view of {name!r} "
        "chooses only part of its control nodes, so that policy set does not exist here. Declare "
        "this mind's own habit prior on the returned SystemParams "
        "(dataclasses.replace(view_params, policy_prior=...)), or drop the composite's."
    )


def _view_metadata(
    composite: System,
    name: str,
    index: Mapping[int, int],
    mod_index: Mapping[int, int],
    chosen: set[str],
    A_deps: tuple[tuple[int, ...], ...],
) -> tuple[tuple[str, Subsystem], ...]:
    """This component's record (and its nested children's) re-expressed in the view's frame.

    A stub belongs to no entry, which is what marks the own/stub boundary.
    """
    out: list[tuple[str, Subsystem]] = []
    for other, sub in composite.subsystems or ():
        if other != name and not other.startswith(f"{name}{SEPARATOR}"):
            continue
        if not (set(sub.factors) <= set(index) and set(sub.modalities) <= set(mod_index)):
            continue
        facs = tuple(index[f] for f in sub.factors)
        mods = tuple(mod_index[m] for m in sub.modalities)
        out.append(
            (
                other,
                sub._replace(
                    factors=facs,
                    modalities=mods,
                    controls=tuple(n for n in sub.controls if n in chosen),
                    free_modalities=tuple(m for m in mods if all(g in facs for g in A_deps[m])),
                    bound_modalities=tuple(
                        (m, tuple(g for g in A_deps[m] if g not in facs))
                        for m in mods
                        if any(g not in facs for g in A_deps[m])
                    ),
                    target_emissions=tuple(
                        (index[f], tuple(index[g] for g in parents))
                        for f, parents in sub.target_emissions
                        if all(g in index for g in parents)
                    ),
                ),
            )
        )
    return tuple(out)


def _view_epistemic(
    composite: System, view: System, view_params: SystemParams, ours: set[str]
) -> "bool | tuple[str, ...]":
    """Restrict the composite's selector to this mind's own variables.

    ``True``/``False`` pass through (so ``True`` also mounts the stubs). An explicit key tuple is
    restricted to this mind's own keys; widen it on the returned view with
    ``dataclasses.replace(view, epistemic=(...))``.
    """
    if isinstance(composite.epistemic, bool):
        return composite.epistemic
    variables = view._epistemic_variables(view_params)
    return tuple(key for key in composite.epistemic if key in variables and _key_name(key) in ours)


def _key_name(key: str) -> str:
    """The name segment of one key (the inverse of :func:`namespaced_key`'s qualification)."""
    parts = key.split(":")
    return parts[1] if len(parts) > 1 and parts[0] in _KEY_KINDS else parts[0]
