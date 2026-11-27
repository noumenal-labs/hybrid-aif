"""``merge()`` and ``subsystem()``: one model from several, and its per-mind views.

:func:`merge` runs at declaration time and returns a plain ``(System, SystemParams)``. It
namespaces every node as ``"<sub>.<name>"`` (the whole key grammar with it:
``"likelihood:arm.pose"``), concatenates state factors, modalities, control nodes and emissions in
``subsystems`` order with every dependency index offset (the **splice**, the first phase of
:func:`merge` and of :func:`~hybrid_aif.composition.couple.couple`), and applies the bindings, of
three kinds: :class:`Link`, an arc of the graphical model of the whole; :class:`Identify`, two
parameter variables made one, or an out-port (an action node or an emission) joined to an
in-port (a modality or a control node); and :class:`Clamp`, a control node held at a constant.
Each leaf (a component that is not itself composed) keeps its own planner configuration, habit,
precision, durations, scoring and recognition on the result (``Subsystem.config``,
``SystemParams.parts``): a merge states a planner field where its leaves agree and holds
:data:`~hybrid_aif.system.UNSET` where they differ, and states no habit, precision, durations or
scoring of its own. It is planned as one mind through
:func:`~hybrid_aif.composition.joint.joint_planner`, which composes them, or run leaf by leaf
through :func:`subsystem`.

A :class:`Link` from a state factor to a modality's parents or to a control node's latent
(a relay) or to a terminal target identifies a latent outright, sharing that variable with
messages both ways; so does :class:`Identify`. A :class:`Link` from an action node to a control
node is the identification of the two ports: the control node stays in its own component, driven
(it carries the driver's realised value and has no action node of its own), and a
:func:`~hybrid_aif.composition.joint.joint_planner` of the node holding both collapses it into
the driver's. A decision is not a belief, so no belief of the driven component is identified
with the driver's, and the link is blanket-shaped at the driven component's boundary.
A :class:`Link` with a ``realise`` from a state factor to a modality is sensing inside one
particle: a modality of one component takes its datum from another component's belief, through a
kernel, realised. A component whose every binding across its boundary is of those two forms keeps
a blanket of its own inside the particle, and is nested (:func:`nested`, ``Subsystem.nested``).
Control nodes no binding touches stay the composite's own, chosen by its planner. A link into a
control node names it, and the latent it is linked from decides the timing: another component's
action node is read at the same step, a state of another component at the previous step (a
one-tick relay). A composite may be merged again (``"outer.inner.x"``).

:func:`subsystem` goes the other way: one mind's bona-fide System, its own factors plus boundary
stubs for the neighbours it must model. :func:`~hybrid_aif.composition.runner.run` steps the
views of a closed System.

Design: docs/design/nesting-systems.md; docs/design/control-states.md C12;
nesting-and-coupling.md N2, N3.
"""

import dataclasses
import functools
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.cells import Grid
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
    ConditionalGaussian,
    Gaussian,
    InverseGamma,
    InverseWishart,
)
from hybrid_aif.dtypes import as_float, tree_float_dtype
from hybrid_aif.factors import (
    LinearGaussianTransition,
    NonlinearGaussianTransition,
    Parameter,
    Recognition,
    TabularObservation,
    _noise_batch,
    map_noise_modes,
)
from hybrid_aif.objectives import hybrid as _hybrid
from hybrid_aif.objectives import registry
from hybrid_aif.reads import REALISATIONS, Emission
from hybrid_aif.system import (
    _KEY_KINDS,
    UNSET,
    PartParams,
    Subsystem,
    System,
    SystemParams,
    _is_pattern,
)

SEPARATOR = NAMESPACE_SEPARATOR
#: The namespace separator (nesting-systems.md N3, superseding the per-variable-infogain doc's
#: ``/`` example): prefix on the left via ``.``, slot-qualification via ``:`` unchanged.


# ------------------------------------------------------------------ the bindings


class Link(NamedTuple):
    """An arc of the graphical model of the whole: ``source`` to ``target``.

    ``source`` is a node, or a tuple of nodes where the link states the complete parent list of its
    target (a structural link). ``target`` is a modality, a control node, or ``"terminal:<factor>"``
    (a state factor's terminal target). What the link means follows from its two ends and
    ``realise``:

    - **from an action node to a control node** (``Link("agent.a_v", "world.u_v")``): the
      identification of the two ports, ``Identify("agent.a_v", "world.u_v")``. The control node
      is *driven*: it stays in its own component, carries the driver's one realised value
      (control-states.md C13), and has no action node of its own, so no planner chooses it. A
      :func:`~hybrid_aif.composition.joint.joint_planner` over a node holding both collapses it
      into the driver's: every state factor that read the driven node reads the driver's, and the
      driven node leaves the planning problem. Level counts (or, for continuous nodes, widths) must
      agree. A decision is not a belief, so no belief of the driven component is identified with
      the driver's, and the link is blanket-shaped at the driven component's boundary. The link
      takes no ``realise``, and a control node as the source is refused, naming its action node;
    - **from a state factor to a control node** (``Link("agent.location", "world.u_location")``):
      the relay. The state becomes a ``B_deps`` parent of the one state factor reading the node, in
      its place, read a step later, the table's control axis re-read as that parent and a no-op axis
      of size 1 appended. The state's level count must equal the node's. Structural: it identifies a
      latent outright;
    - **from state factors to a modality, with no ``realise``** (``Link(("a.x", "b.y"),
      "a.obs")``): structural. The sources are the modality's complete new parent list, unchanged
      parents included, in its likelihood's axis order, with messages both ways. Arity and
      per-axis support must match the declared likelihood's parent axes. One link per modality;
    - **from one state factor to a modality, with ``realise``** (``Link("rover.x",
      "mission.cell_obs", realise="mode", kernel=grid)``): a belief read, one way. It is the pair
      of an emission and an observation: the source's component gains the emission
      ``rover.cell_obs`` (named by the modality's local name under the source's component),
      ``Emission(("rover.x",), realise, kernel)`` with ``kernel="identity"`` when ``kernel`` is
      ``None`` or ``"identity"``, or a :class:`~hybrid_aif.cells.Grid`, the indicator of its cells
      over a continuous source, and the composite's ``observes`` maps the modality to it
      (``{"mission.cell_obs": "rover.cell_obs"}``). A second read of the same source with the same
      kernel and realisation under the same name reuses the emission (one out-port, many
      in-ports); a name that is already another variable of the composite is refused. The
      modality's datum each round is the emission realised from the source's posterior
      (``"sample"``, ``"mode"`` or ``"stats"``; see :class:`~hybrid_aif.reads.Emission`), which a
      belief read must give. :func:`subsystem` leaves the modality an ordinary observation of the
      reading component's view, and :func:`~hybrid_aif.composition.runner.run` supplies it each
      round from the emission the source component's step realised, as it stands at the reader's
      turn. The source and the modality lie in different components (a read inside one is an
      emission and ``System.observes``). A level count that is not the source's under
      the identity kernel, nor the grid's cell count under an indicator, a categorical modality
      reading a continuous source with no grid and ``"stats"`` of a Gaussian source are refused;
    - **from state factors to ``"terminal:<factor>"``** (``Link(("a.x", "b.y"), "terminal:a.z")``,
      the spelling ``terminal:agent.z`` under a prefix): the terminal target of the state factor,
      already a ``ConditionalCategorical`` in the component's ``terminal_targets``, resolved against
      the sources' final predicted beliefs (its ``H_deps``). Structural.

    A link from a **realised port** (an action node, or an emission) into a modality is a
    coupling's, and a ``merge`` refuses it, naming :func:`~hybrid_aif.composition.couple.couple`.
    A port is realised once, by its own kernel, so a link from it takes no ``realise``, and a
    ``kernel`` without ``realise`` is refused. Design: nesting-and-coupling.md N2, N3, N7, N8;
    engine-consolidation.md.
    """

    #: A node, or a tuple of nodes where the link states the complete parent list of its target.
    source: "str | tuple[str, ...]"
    #: A modality, a control node, or ``"terminal:<factor>"`` (a state factor's terminal target).
    target: str
    #: For a belief read only: how the source's posterior is realised, ``"sample"``, ``"mode"``
    #: or ``"stats"``; ``None`` for every other link.
    realise: str | None = None
    #: For a belief read only: ``None`` or ``"identity"`` (the identity kernel), or a
    #: :class:`~hybrid_aif.cells.Grid` (the indicator of its cells over a continuous source).
    kernel: "str | Grid | None" = None


class Identify(NamedTuple):
    """Two parameter variables identified as one, or an out-port identified with an in-port.

    **Ports.** ``a`` and ``b`` name an out-port (an action node or an emission) and an in-port (a
    modality, or a control node), in either order. The in-port observes the out-port's variable:
    the composite's ``System.observes`` gains ``(in-port, out-port)``. An in-port is identified at
    most once, an out-port feeds any number, and the two port types must be equal. A control node
    in-port is driven: it stays in its leaf, carries the out-port's realised value, and has no
    action node of its own (``action_names`` entry ``None``), so no planner chooses it; a
    :func:`~hybrid_aif.composition.joint.joint_planner` collapses it into the driver's node. Under
    :func:`~hybrid_aif.composition.couple.couple` the two ports lie in different components.

    **Parameter variables.**

    ``a`` and ``b`` name entries of the composite's ``parameters`` (``"rover.field"``,
    ``"mission.field"``: a component's own parameter name under its prefix). The composite keeps
    one variable under ``a``, with ``a``'s declaration (its prior and its counts), and every
    state factor, modality or emission that read ``b`` reads ``a``. The two must be the same family
    with the same table shape; anything else raises, naming both. ``b``'s counts are discarded: only
    ``a``'s declaration survives, so identify before ``b``'s member has learned anything worth
    keeping.

    An identification inside one particle: the variable has one home and the readers share its
    counts. It is bookkeeping, not a crossing; under
    :func:`~hybrid_aif.composition.couple.couple` a parameter is reported by
    ``stats``, the counts since the last report, and ``mode`` and ``sample`` of a
    parameter stay deferred. Design: nesting-and-coupling.md N2, N6.
    """

    #: A port, or the parameter variable that survives (with its declaration).
    a: str
    #: The other port, or the parameter variable whose readers are redirected to ``a``.
    b: str


class Clamp(NamedTuple):
    """A control node no planner chooses and no link drives, clamped at a constant.

    The average under a distribution is taken once, at merge time, with a fixed vector: the
    static case of a node consumed as a probability vector. The per-tick consumption is the
    planner's contraction (:func:`~hybrid_aif.controls.take_controls`). Design:
    nesting-and-coupling.md N8.
    """

    #: A discrete control node, by name (``"world.u_location"``).
    node: str
    #: An ``int`` level (a delta: slices every reading state factor's control axis, pinning the
    #: dynamics) or a :class:`~hybrid_aif.distributions.Categorical` or probability vector over
    #: the node's levels (averages every reading state factor's table over the axis; uniform
    #: means no one is steering).
    level: "int | Categorical | Array"


Binding = Link | Identify | Clamp


class _BeliefRead(NamedTuple):
    """A belief read a :class:`Link` declares: the reading modality, the source state factor, the
    kernel (``"identity"`` or a :class:`~hybrid_aif.cells.Grid`) and the realisation."""

    modality: str
    source: str
    kernel: "str | Grid"
    realise: str


#: What :func:`classify` calls a port that generates a value, and one that takes it.
_OUT_PORTS = ("an action node", "an emission")
_IN_PORTS = ("a modality", "a control node")

#: The first segment of the spelling of a factor's terminal target as a link's target.
TERMINAL = "terminal"


# ------------------------------------------------------------------- key helpers


#: The first segments whose second segment is a name to qualify: the slots, and ``parameter``.
_NAMED_KINDS = (*_KEY_KINDS, "parameter")


def namespaced_key(prefix: str, key: str) -> str:
    """Qualify one key of the key grammar with a subsystem prefix, on its name segment.

    ``"pose" -> "arm.pose"``, ``"pose:b1" -> "arm.pose:b1"``, ``"obs:cue" -> "obs:arm.cue"``,
    ``"likelihood:pose:noise" -> "likelihood:arm.pose:noise"``, ``"parameter:field" ->
    "parameter:arm.field"`` (a parameter variable's rate key). Wildcards must be resolved first
    (:func:`resolved_epistemic`).
    """
    parts = key.split(":")
    if "*" in parts:
        raise ValueError(
            f"cannot namespace the wildcard pattern {key!r}: patterns are resolved against "
            "the subsystem's own variable table first (merge resolves them for you)"
        )
    at = 1 if len(parts) > 1 and parts[0] in _NAMED_KINDS else 0
    parts[at] = f"{prefix}{SEPARATOR}{parts[at]}"
    return ":".join(parts)


def namespaced_term(prefix: str, key: str) -> str:
    """Qualify one ``System.terms`` selector with a subsystem prefix, on its variable segment.

    The row segment is library vocabulary and stays literal: ``"uncertainty:x" ->
    "uncertainty:arm.x"``, ``"uncertainty:pose:b1" -> "uncertainty:arm.pose:b1"``,
    ``"predicted_reward:obs:reward" -> "predicted_reward:obs:arm.reward"``,
    ``"occupancy:transition:stoch" -> "occupancy:transition:arm.stoch"``. A row that takes no
    target (a bare row name) is unchanged. A trailing dimension wildcard after the variable
    segment is kept (``"uncertainty:x:*" -> "uncertainty:arm.x:*"``); a wildcard in the variable
    segment itself raises: name the variables.
    """
    row, sep, rest = key.partition(":")
    if registry.get_row(row).target is None or not sep:
        return key
    parts = rest.split(":")
    at = 1 if len(parts) > 1 and parts[0] in _NAMED_KINDS else 0
    if "*" in parts[: at + 1]:
        raise ValueError(
            f"cannot namespace the wildcard selector terms={key!r}: a wildcard in the variable "
            "segment would range over variables the component did not have; name the "
            "variables, or declare the composite's terms on the returned System"
        )
    head = namespaced_key(prefix, ":".join(parts[: at + 1]))
    return ":".join([row, head, *parts[at + 1 :]])


def _namespaced_parent(prefix: str, parent: str) -> str:
    """A recognition parent address under a component prefix: a factor, ``obs:``,
    ``recognition:`` or ``external:`` operand, each renamed on its name."""
    kind, sep, name = parent.partition(":")
    if sep and kind == "external":
        return f"external:{prefix}{SEPARATOR}{name}"
    return namespaced_key(prefix, parent)


def resolved_epistemic(system: System, params: SystemParams) -> tuple[str, ...]:
    """A component's ``epistemic`` selector as an explicit tuple of literal keys.

    Patterns and ``True`` are expanded against the component's own variable table, so a component's
    ``("*",)`` is fixed at merge time and does not widen to variables the composite gained.
    ``True`` excludes initial-prior parameter variables as it does everywhere, and names each
    continuous state factor whole rather than by its dimensions; a learnable family with no
    registered novelty estimator raises here. A continuous state factor no channel with an
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


def resolved_plan_learning(system: System, params: SystemParams) -> tuple[str, ...]:
    """A component's ``plan_learning`` as an explicit tuple of reader keys
    (``transition:<factor>``, ``likelihood:<modality>``): ``True`` and a ``parameter:<name>``
    key expanded to the readers they name, so a composite's keys survive a binding that
    identifies two parameter variables (readers of one variable share its working table
    whatever they are keyed by). Design: parameters-within-the-plan.md D1."""
    names = {"transition": system._facs, "likelihood": system.modality_names}
    return tuple(
        f"{slot}:{names[slot][i]}"
        for _, readers in system._plan_learning(params)
        for slot, i in readers
    )


# ------------------------------------------------------------------ small pieces


def _lookup(name: str, table: Mapping[str, int], what: str) -> int:
    if name in table:
        return table[name]
    raise ValueError(
        f"binding names unknown {what} {name!r}; the composite's {what}s are "
        f"{sorted(table)} (names are subsystem-qualified with {SEPARATOR!r}, as in "
        f"'boat{SEPARATOR}x')"
    )


#: The per-leaf configuration a leaf records (``Subsystem.config``), in order; the first group is
#: the planning fields a composite states where its leaves agree and holds :data:`UNSET` where
#: they differ, the second two are shared by one belief carry and must agree.
_PLANNING_FIELDS = (
    "policy_len",
    "planner",
    "gamma",
    "num_iters",
    "decomposition",
    "planning_belief",
    "feedback",
)
_CARRY_FIELDS = ("mixture_schedule", "mixture_k")


def _unanimous(values: Mapping[str, object]):
    """The one value a configuration field takes across leaves (``values`` is keyed by leaf), or
    :data:`~hybrid_aif.system.UNSET` where they differ."""
    distinct = set(values.values())
    return next(iter(distinct)) if len(distinct) == 1 else UNSET


def _requalify_config(prefix: str, config: tuple) -> tuple:
    """A leaf's configuration under a further component prefix: its selectors qualified."""
    out = []
    for key, value in config:
        if key == "epistemic" and isinstance(value, tuple):
            value = tuple(namespaced_key(prefix, k) for k in value)
        elif key == "terms":
            value = tuple(namespaced_term(prefix, t) for t in value)
        out.append((key, value))
    return tuple(out)


def _requalify_part(prefix: str, part: PartParams) -> PartParams:
    """A leaf's planner parameters under a further component prefix: its durations' node names
    qualified."""
    return PartParams(
        part.policy_prior,
        part.precision,
        {f"{prefix}{SEPARATOR}{node}": v for node, v in part.durations.items()},
        part.scoring,
    )


def _leaves(name: str, system: System, p: SystemParams, epistemic: Sequence[str]) -> list[tuple]:
    """The leaves component ``name`` brings to a splice, as ``(leaf name, config, parts, owned
    recognition variables)``: itself when it is a plain System, its own leaves re-qualified when it
    is a splice."""
    inner = [(child, sub) for child, sub in system.subsystems or () if sub.config]
    if not inner:
        picked = (
            True
            if system.epistemic is True
            else False
            if system.epistemic is False or system.epistemic == ()
            else tuple(epistemic)
        )
        config = (
            *((f, getattr(system, f)) for f in (*_PLANNING_FIELDS, *_CARRY_FIELDS)),
            ("epistemic", picked),
            ("terms", tuple(namespaced_term(name, key) for key in system.terms)),
            ("scoring", system.scoring),
        )
        part = PartParams(
            p.policy_prior,
            p.precision,
            {f"{name}{SEPARATOR}{node}": v for node, v in system._durations(p, "merge").items()},
            p.scoring,
        )
        owned = tuple(f"{name}{SEPARATOR}{e.variable}" for e in system.recognition or ())
        return [(name, config, part, owned)]
    own = {f: getattr(system, f) for f in _PLANNING_FIELDS}
    clocks = system._durations(p, "merge")
    out = []
    for child, sub in inner:
        config = tuple(
            (key, own[key] if key in own and own[key] is not UNSET else value)
            for key, value in _requalify_config(name, sub.config)
        )
        held = p.parts.get(child, PartParams())
        habit = held.policy_prior
        if habit is None and p.policy_prior is not None:
            if len(inner) > 1:
                raise NotImplementedError(
                    f"component {name!r} declares a habit over the joint policy set of its "
                    f"{len(inner)} leaves, which cannot be split among them; couple the leaves "
                    "individually, or drop the habit"
                )
            habit = p.policy_prior
        precision = held.precision if held.precision is not None else p.precision
        mine = {f"{name}{SEPARATOR}{n}": v for n, v in clocks.items() if n.startswith(f"{child}.")}
        durations = {
            **mine,
            **{f"{name}{SEPARATOR}{n}": v for n, v in held.durations.items()},
        }
        out.append(
            (
                f"{name}{SEPARATOR}{child}",
                config,
                PartParams(habit, precision, durations, held.scoring),
                tuple(f"{name}{SEPARATOR}{v}" for v in sub.recognition),
            )
        )
    return out


#: A factor's discrete cardinality, or ``None`` for a non-tabular one — the typing measure
#: for both binding kinds that pair a support with an axis. One definition, shared
#: with the ``H_deps`` parent-shape check.
_factor_levels = System._factor_levels


class MixtureRefusal(NotImplementedError):
    """A distribution over a control node's levels refused where a factor reading the node has no
    form for it: a linear-Gaussian transition or a Gaussian target averaged over levels is a
    mixture, not a Gaussian. A level is accepted; ``run`` catches this type at a reset call, whose
    view it then takes at a level."""


def _drop_control_axis(decl, name: str, keep: "int | None", weights: "Array | None"):
    """Retire a tabular transition's control axis, keeping a trailing no-op axis.

    With neither ``keep`` nor ``weights`` the control axis is re-read as a parent axis (the
    ``Link`` rewrite; only the declared ``parent_ndims`` changes); ``keep`` slices one
    level; ``weights`` averages the axis in probability space. All three end with a size-1 trailing
    control axis. A nonlinear transition is refused: its map takes the control node as an input.
    A :class:`LinearGaussianTransition` keeps its tensors under the identification rewrite (the
    innovations' control axis keeps its place and size and is read as a parent axis).
    """
    if isinstance(decl, NonlinearGaussianTransition):
        raise NotImplementedError(
            f"factor {name!r}: this binding would remove a control node its "
            "NonlinearGaussianTransition reads, and f's signature, one controls entry per "
            "control parent, is fixed code; bind a node f does not read, or identify the node "
            "with another control node of the same kind and size (a Link between two "
            "control nodes), which leaves f's inputs whole (design: nonlinear-transitions.md N5)"
        )
    if isinstance(decl, LinearGaussianTransition):
        if keep is None and weights is None:
            return decl  # the axis keeps its place and size; only its reading changes
        raise MixtureRefusal(_mixture_sentence(name))
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


def _mixture_sentence(name: str) -> str:
    return (
        f"factor {name!r}: terminating a control node a LinearGaussianTransition reads with a "
        "distribution would average Gaussians across control levels, which is a mixture and "
        "not a Gaussian. Pin a level instead (an int prior slices the node's axis), or "
        "identify the node with another control node."
    )


def _cut_gaussian(decl: LinearGaussianTransition, name: str, cuts: Sequence[tuple], floats):
    """Pin discrete control axes of a linear-Gaussian transition.

    ``cuts`` holds ``(pos, keep, weights)`` with ``pos`` the declared-parent position; ``floats``
    the declared positions of the factor's continuous control parents, which carry no mode axis.
    Each pinned axis is sliced out of the innovations' ``loc`` (and ``chol`` unless shared) and of
    a per-mode ``F`` and ``G``; a factor left with no mode axis keeps a trailing size-1 one.
    """
    if any(weights is not None for _, _, weights in cuts):
        raise MixtureRefusal(_mixture_sentence(name))
    inn = decl.innovations
    pn = inn.parent_ndims
    axes = sorted((pos - sum(1 for c in floats if c < pos), keep) for pos, keep, _ in cuts)

    def slice_modes(arr, core: int):
        first = arr.ndim - core - pn
        for axis, keep in reversed(axes):
            arr = jnp.take(arr, keep, axis=first + axis)
        if pn == len(axes):
            arr = jnp.expand_dims(arr, arr.ndim - core)
        return arr

    loc = slice_modes(inn.loc, 1)
    chol = inn.chol if inn.shared_chol else slice_modes(inn.chol, 2)
    F = slice_modes(decl.F, 2) if decl.F.ndim > 2 else decl.F
    G = decl.G if decl.G is None or decl.G.ndim <= 2 else slice_modes(decl.G, 2)
    parent_ndims = pn - len(axes) + (1 if pn == len(axes) else 0)
    return LinearGaussianTransition(
        F=F,
        innovations=ConditionalGaussian(loc=loc, chol=chol, parent_ndims=parent_ndims),
        G=G,
        noise_prior=map_noise_modes(decl.noise_prior, lambda arr, core: slice_modes(arr, core)),
    )


def _cut_axes(decl, name: str, n: int, cuts: Sequence[tuple], floats: Sequence[int] = ()) -> object:
    """Terminate some of a tabular transition's ``n`` control axes.

    ``cuts`` holds ``(pos, keep, weights)`` per axis, ``pos`` the axis's place among the
    declared parents: ``keep`` slices one level, ``weights`` averages the axis in probability
    space. The other axes keep their order; a table left reading no control node carries a
    trailing no-op axis of size 1. ``floats`` lists the declared positions of the factor's
    continuous control parents, which a linear-Gaussian transition's mode axes skip.
    """
    if isinstance(decl, LinearGaussianTransition):
        return _cut_gaussian(decl, name, cuts, floats)
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
    probs = prior.probs if isinstance(prior, Categorical) else as_float(prior)
    if probs.shape != (nu,):
        raise ValueError(
            f"{what} has shape {probs.shape} but the node has {nu} level(s); declare "
            "one probability per level (or an int level for a delta)"
        )
    return probs


def _termination(prior, nu: int, what: str) -> tuple["int | None", "Array | None"]:
    """A terminating prior over a control node, as ``(keep, weights)``.

    An ``int`` is a level (slice the axis), anything else a distribution over levels (average
    it); a traced integer scalar is a level too, kept as the traced index of the gathers (the
    level of a view sliced inside ``run``'s scanned round, design batched-run.md R4). Shared by
    :class:`Clamp` in :func:`merge` and the partner nodes of :func:`subsystem`.
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
    if isinstance(prior, jax.core.Tracer):
        #  a level traced inside ``run``'s scanned round: the gathers take it as an index, and its
        #  range is its producer's contract (an action node is realised in range)
        return jnp.asarray(prior, jnp.int32), None
    keep = int(prior)
    if not 0 <= keep < nu:
        raise ValueError(f"{what} pins level {keep}, which is not one of its {nu} control level(s)")
    return keep, None


def _contract_target(target, key: str, cuts: Sequence[tuple[int, int | None, Array | None]]):
    """A conditional per-step target with some parent axes terminated.

    ``cuts`` holds ``(pos, keep, weights)`` per terminated parent, ``pos`` its place in the
    target's ``C_deps`` order. A level slices the axis; a distribution mixes a categorical
    target's normalized preferences in probability space (a mixture of Gaussian targets is not
    a Gaussian, so a Gaussian target takes a level). A target left with no parent is the
    unconditional ``Categorical`` or ``Gaussian``.
    """
    n = len(cuts)
    ndims = target.parent_ndims - n
    if isinstance(target, ConditionalGaussian):
        if any(weights is not None for _, _, weights in cuts):
            raise MixtureRefusal(
                f"the target on {key!r} is a Gaussian conditional on another component's control "
                "node, and a prior spread over that node's levels makes the target a mixture of "
                "Gaussians, which is not a Gaussian; pass an int level in inputs for "
                "that node's variable"
            )
        loc, chol = target.loc, target.chol
        shared = target.shared_chol
        for pos, keep, _ in sorted(cuts, reverse=True):
            loc = jnp.take(loc, keep, axis=pos)
            if not shared:
                chol = jnp.take(chol, keep, axis=pos)
        if ndims == 0:
            return Gaussian(loc=loc, chol=chol)
        if target.diagonal and ndims == 1 and loc.shape[-2] == loc.shape[-1]:
            # (L, d) with L == d would read as a shared full factor: stack the full factors
            chol = chol[..., :, None] * jnp.eye(loc.shape[-1], dtype=chol.dtype)
        return ConditionalGaussian(loc=loc, chol=chol, parent_ndims=ndims)
    table = target.log_probs if any(w is not None for _, _, w in cuts) else target.log_table
    for pos, keep, weights in sorted(cuts, key=lambda cut: cut[0], reverse=True):
        axis = 1 + pos  # the child axis leads
        if weights is None:
            table = jnp.take(table, keep, axis=axis)
        else:
            #  a mixture of the slices, in probability space
            table = jax.scipy.special.logsumexp(
                jnp.moveaxis(table, axis, -1) + jnp.log(weights), axis=-1
            )
    if ndims == 0:
        return Categorical(logits=table)
    return ConditionalCategorical(log_table=table, parent_ndims=ndims)


# ------------------------------------------------------------------- the builder


class _Node:
    """One control node of the composite under construction (mutable, builder-local)."""

    def __init__(self, name, action, kind, size, box, kernel):
        self.name, self.action = name, action
        self.kind, self.size, self.box, self.kernel = kind, size, box, kernel
        self.live = True


class _Spliced:
    """The splice of components: every declaration of every component concatenated under its
    prefix, with each leaf's own planner configuration and parameters kept beside it."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def _splice(subsystems: Mapping[str, System], params: Mapping[str, SystemParams]) -> "_Spliced":
    """The first phase of :func:`merge` and :func:`~hybrid_aif.composition.couple.couple`:
    namespaces every component's nodes as ``"<component>.<name>"``, concatenates state factors,
    modalities, control nodes, emissions, tables, parameter variables and references, recognition,
    ``observes`` and readings in component order with every dependency index offset, and records
    each leaf's own planner configuration (``configs``), planner parameters (``parts``) and
    recognition variables. A component that is itself a splice contributes its leaves, each
    re-qualified. Nothing is bound and nothing is refused for a disagreement of leaves; the
    bindings and the composite's own planner fields come after.
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
    #  each component's params as the positional view the concatenation indexes
    params = {name: subsystems[name]._params_in(params[name], "merge") for name in subsystems}

    # -------------------------------------------------- concatenate the frames
    fac_offset: dict[str, int] = {}
    mod_offset: dict[str, int] = {}
    state_names: list[str] = []
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
    external_inputs: list[tuple[str, int]] = []
    epistemic_keys: list[str] = []
    plan_learning_keys: list[str] = []
    dims: list[tuple[str, tuple[str, ...]]] = []
    cells: list[tuple[str, object]] = []
    readings: list[tuple[str, str]] = []
    reading_counts: dict[str, object] = {}
    C_deps: list[tuple[str, tuple[str, ...]]] = []
    parameters: dict[str, object] = {}  # the named parameter variables, by composite name
    reads: dict[tuple[str, int], str] = {}  # (slot, composite index) -> the variable it reads
    noise_reads: dict[tuple[str, int], str] = {}  # the same for a reader of a noise variable
    own_reads: list[tuple[str, System, tuple[str, str]]] = []  # a component's observes entries
    emissions: list[tuple[str, Emission]] = []  # composite names, parents namespaced
    emission_kernels: dict[str, object] = {}  # composite name -> kernel (or its reference)
    emission_reads: dict[str, str] = {}  # composite name -> the parameter variable it reads
    black_box: list[tuple[str, tuple[str, int]]] = []  # a black box's ports, composite names

    terms: list[str] = []
    configs: dict[str, tuple] = {}  # leaf name (composite) -> its own planner configuration
    parts: dict[str, PartParams] = {}  # leaf name -> its own planner parameters
    leaf_recognition: dict[str, tuple[str, ...]] = {}  # leaf name -> its recognition variables
    leaf_order: list[str] = []  # every leaf, in order
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
        if set(p.emission) != {n for n, e in system.emissions if e.kernel is None}:
            raise ValueError(
                f"subsystem {name!r}: params hold emission kernels for {sorted(p.emission)} and "
                f"the System declares the emissions {list(system._emission_names)} (the ones "
                "with kernel=None take a kernel in the params): the pair is mismatched"
            )
        table = system._table
        f_off, m_off, k_off = len(state_names), len(modality_names), len(nodes)
        fac_offset[name], mod_offset[name] = f_off, m_off
        state_names.extend(f"{name}{SEPARATOR}{f}" for f in facs)
        modality_names.extend(f"{name}{SEPARATOR}{m}" for m in mods)
        #  a state factor shifts by the component's offset; an action node a modality observes
        #  (index at or above the component's node count) is held as -(node + 1) until the
        #  bindings fix which control nodes stay
        n_nodes = len(system.B_deps)
        A_deps.extend(
            tuple(g + f_off if g < n_nodes else -(k_off + g - n_nodes) - 1 for g in deps)
            for deps in system.A_deps
        )
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
        nodes.extend(
            _Node(
                f"{name}{SEPARATOR}{node}",
                None if action is None else f"{name}{SEPARATOR}{action}",
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
        for emission, emit in system.emissions:
            composite_name = f"{name}{SEPARATOR}{emission}"
            emissions.append(
                (
                    composite_name,
                    Emission(
                        tuple(f"{name}{SEPARATOR}{q}" for q in emit.parents),
                        emit.realise,
                        emit.kernel,
                    ),
                )
            )
            if emit.kernel is not None:
                continue
            kernel = p.emission[emission]
            if isinstance(kernel, Parameter):
                emission_reads[composite_name] = f"{name}{SEPARATOR}{kernel.name}"
            else:
                emission_kernels[composite_name] = kernel
        black_box.extend((f"{name}{SEPARATOR}{port}", t) for port, t in system.black_box)
        likelihood.extend(p.likelihood)
        transition.extend(p.transition)
        initial_state_prior.extend(p.initial_state_prior)
        parameters.update({f"{name}{SEPARATOR}{h}": decl for h, decl in p.parameters.items()})
        for slot, i, home in p.references:
            reads[(slot, i + (f_off if slot == "transition" else m_off))] = (
                f"{name}{SEPARATOR}{home}"
            )
        for slot, i, home in p.noise_references:
            noise_reads[(slot, i + (f_off if slot == "transition" else m_off))] = (
                f"{name}{SEPARATOR}{home}"
            )
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
                parents=tuple(_namespaced_parent(name, parent) for parent in entry.parents),
            )
            for entry in (system.recognition or ())
        )
        external_inputs.extend(
            (f"{name}{SEPARATOR}{ext}", width) for ext, width in system.external_inputs
        )
        epistemic_keys_of_component = [
            namespaced_key(name, key) for key in resolved_epistemic(system, p)
        ]
        epistemic_keys.extend(epistemic_keys_of_component)
        plan_learning_keys.extend(
            namespaced_key(name, key) for key in resolved_plan_learning(system, p)
        )
        terms.extend(namespaced_term(name, key) for key in system.terms)
        dims.extend((f"{name}{SEPARATOR}{factor}", names) for factor, names in system.dims)
        cells.extend((f"{name}{SEPARATOR}{factor}", grid) for factor, grid in system.cells)
        for slot, variable in system.observes:
            own_reads.append((name, system, (slot, variable)))
        readings.extend(
            (f"{name}{SEPARATOR}{table}", f"{name}{SEPARATOR}{node}")
            for table, node in system.readings
        )
        reading_counts.update({f"{name}{SEPARATOR}{k}": v for k, v in p.readings.items()})
        C_deps.extend(
            (namespaced_key(name, key), tuple(namespaced_key(name, p) for p in parents))
            for key, parents in system.C_deps
        )
        for leaf, config, part, owned in _leaves(name, system, p, epistemic_keys_of_component):
            configs[leaf] = config
            if part.policy_prior is not None or part.precision is not None:
                parts[leaf] = part
            elif part.durations or part.scoring:
                parts[leaf] = part
            leaf_recognition[leaf] = owned
            leaf_order.append(leaf)

    return _Spliced(
        subsystems=subsystems,
        params=params,
        fac_offset=fac_offset,
        mod_offset=mod_offset,
        state_names=state_names,
        modality_names=modality_names,
        A_deps=A_deps,
        parents=parents,
        D_deps=D_deps,
        H_deps=H_deps,
        nodes=nodes,
        likelihood=likelihood,
        transition=transition,
        initial_state_prior=initial_state_prior,
        obs_targets=obs_targets,
        state_targets=state_targets,
        terminal_targets=terminal_targets,
        recognition_decl=recognition_decl,
        recognition_params=recognition_params,
        external_inputs=external_inputs,
        epistemic_keys=epistemic_keys,
        plan_learning_keys=plan_learning_keys,
        dims=dims,
        cells=cells,
        readings=readings,
        reading_counts=reading_counts,
        C_deps=C_deps,
        parameters=parameters,
        reads=reads,
        noise_reads=noise_reads,
        own_reads=own_reads,
        emissions=emissions,
        emission_kernels=emission_kernels,
        emission_reads=emission_reads,
        black_box=black_box,
        terms=terms,
        configs=configs,
        parts=parts,
        leaf_recognition=leaf_recognition,
        leaf_order=leaf_order,
    )


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
    explicit ``state_names``/``control_names``/``modality_names``/``action_names``: a
    component's generated names are namespaced.

    - Planner fields (``policy_len``, ``planner``, ``gamma``, ``num_iters``, ``decomposition``,
      ``planning_belief``, ``feedback``) are the leaves' value where every leaf has one and
      :data:`~hybrid_aif.system.UNSET` where they differ; ``mixture_schedule`` and a non-``None``
      ``mixture_k`` are shared by one belief carry and must agree. Each leaf's own are
      ``Subsystem.config``.
    - A leaf's ``policy_prior``, ``precision``, ``durations`` and scoring parameters are
      ``SystemParams.parts[<leaf>]`` (present when it holds any); the composite states none of
      them, and :func:`~hybrid_aif.composition.joint.joint_planner` composes them.
    - ``epistemic`` is resolved and unioned (:func:`resolved_epistemic`); ``dims`` moves with its
      state factors, under the namespaced names (a key ``pose:b1`` becomes ``arm.pose:b1``).
    - A modality observing an action node keeps it: the node's index at or above the component's
      node count becomes the composite's, ``len(B_deps) + position`` among the control nodes
      that stay (one a relay or ``Clamp`` removed, or an action node a driven node lost, raises).
    - Designations keep each component's preferences under namespaced keys; ``C_deps``
      namespaces its keys and parents as ``H_deps`` does.
    - ``System.terms`` selectors carry under the component prefix, on their variable segment
      (:func:`namespaced_term`: ``uncertainty:x`` becomes ``uncertainty:arm.x``); the union is the
      composite's, less a term a leaf's scoring owns (that leaf's ``config`` carries it).
    - ``recognition`` namespaces both halves, the parent addresses included (``Recognition("stoch",
      ("deter", "recognition:embed"))`` → ``Recognition("mc.stoch", ("mc.deter",
      "recognition:mc.embed"))``; ``obs:`` and ``external:`` operands keep their prefix and are
      renamed on their name). ``external_inputs`` carries under the component prefix, so two
      components' inputs of one name stay distinct.
    - A continuous control node's box (``params.control``) and an action node's kernel
      (``params.actuator``) move with their nodes, under the namespaced names; a driven node
      refuses a kernel on its own action node.
    - ``readings`` (a table's pooled reading, ``System.readings`` and ``params.readings``) carries
      under the component's prefix; a node a relay or ``Clamp`` removed raises. A view carries the
      entries of its own tables and raises where the node is another component's.
    - ``observes`` (a slot's variable: a modality that observes an emission or an action node of
      its component, which is how a component reads its own belief or its previous action)
      carries under the component's prefix, slot and variable alike; a :class:`Link` with a
      ``realise``
      adds an emission to the source's component and a modality's entry, whose variable is in
      another component. A component whose every binding across its boundary is such a belief
      read or a link from an action node into a control node is nested (:func:`nested`). An
      action node an entry names must stay one its component chooses. Design:
      nesting-and-coupling.md N2, N3.
    - ``emissions`` (a component's outputs drawn from its state factors, with their kernels in
      ``params.emission`` or declared static) carry under the component's prefix; a parent a relay
      or a ``Clamp`` removed raises.
    - A target on a node a binding removed raises, naming what the node became.

    The composite is validated (``System._validate``) before it is returned.
    """
    return _assemble(subsystems, params, bindings)


def _assemble(
    subsystems: Mapping[str, System],
    params: Mapping[str, SystemParams],
    bindings: Sequence[Binding] = (),
    *,
    coupled: bool = False,
) -> tuple[System, SystemParams]:
    """The splice of components followed by bindings: the body of :func:`merge` and of
    :func:`~hybrid_aif.composition.couple.couple`. With ``coupled`` the bindings are port
    identifications and clamps only, every top-level component is a leaf of the coupling
    (``boundary="coupled"``); neither states a planning problem of its own.
    """
    sp = _splice(subsystems, params)
    params = sp.params
    fac_offset, mod_offset = sp.fac_offset, sp.mod_offset
    state_names, modality_names, A_deps, parents = (
        sp.state_names,
        sp.modality_names,
        sp.A_deps,
        sp.parents,
    )
    D_deps, H_deps, nodes = sp.D_deps, sp.H_deps, sp.nodes
    likelihood, transition, initial_state_prior = (
        sp.likelihood,
        sp.transition,
        sp.initial_state_prior,
    )
    obs_targets, state_targets, terminal_targets = (
        sp.obs_targets,
        sp.state_targets,
        sp.terminal_targets,
    )
    recognition_decl, recognition_params, external_inputs = (
        sp.recognition_decl,
        sp.recognition_params,
        sp.external_inputs,
    )
    epistemic_keys, dims, cells, readings, reading_counts = (
        sp.epistemic_keys,
        sp.dims,
        sp.cells,
        sp.readings,
        sp.reading_counts,
    )
    C_deps, parameters, reads, own_reads = (
        sp.C_deps,
        sp.parameters,
        sp.reads,
        sp.own_reads,
    )
    emissions, emission_kernels, emission_reads, terms = (
        sp.emissions,
        sp.emission_kernels,
        sp.emission_reads,
        sp.terms,
    )
    factor_index = {name: i for i, name in enumerate(state_names)}
    modality_index = {name: i for i, name in enumerate(modality_names)}
    node_index = {node.name: k for k, node in enumerate(nodes)}
    owner = {
        f: name
        for name, off in fac_offset.items()
        for f in range(off, off + subsystems[name]._table.num_states)
    }
    #  the nodes an inner composite's own bindings removed, with what each became
    nested_gone = {
        node: tail
        for name, system in subsystems.items()
        for node, tail in system._bound_away(prefix=f"{name}{SEPARATOR}")
    }
    relayed: dict[int, int] = {}  # node -> the state it was identified with
    relay_reader: dict[int, int] = {}  # node -> the factor that read it
    terminations: dict[int, tuple] = {}  # a terminated node's (keep, weights)
    identified_pairs: list[tuple[str, str]] = []  # (driven node, driver's node), as written
    inert: dict[str, str] = {}  # a driven node -> the action node it had before it was driven
    alias: dict[str, str] = {}  # that inert action node -> the variable that drives its node

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
        """A relay into a node that not exactly one factor reads: one sentence for either order."""
        source = (
            f"made one with {sorted(partners)} by a same-step Link"
            if partners
            else "shared within its component"
        )
        same_step_node = (
            repr(control_name(state)) if control_name(state) in node_index else "a control node"
        )
        if not readers:
            return ValueError(
                f"Link({state!r} -> {node!r}) merges a state with {node!r}, which no "
                "factor reads (only a target, a modality or durations names it); a relay "
                f"re-reads one factor's control axis, so bind from {same_step_node} for a "
                "same-step read, or relay into a node one factor reads"
            )
        return ValueError(
            f"Link({state!r} -> {node!r}) merges a state with {node!r}, which "
            f"{[state_names[g] for g in sorted(readers)]} read ({source}); a relay re-reads "
            f"one factor's control axis, so bind from {same_step_node} for a same-step read, or "
            "relay into a node one factor reads"
        )

    def driven_node(name: str, kind: str) -> int:
        """The live control node ``name`` a binding identifies or terminates."""
        live = [node.name for node in nodes if node.live]
        if name in node_index:
            k = node_index[name]
            if k in driven_by:
                raise ValueError(
                    f"{kind} names the control node {name!r}, which {driven_by[k]!r} already "
                    "drives (its realised value is the node's level); a node is driven once"
                )
            if nodes[k].live:
                return k
            if k in relayed:
                to = [state_names[relayed[k]]]
            elif k in terminations:
                to = "a Clamp"
            else:
                to = "a driver's node"
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

    def kernel_not_dropped(c: int, what: str) -> None:
        """Refuse a binding that takes the action node of node ``c`` away (the node is removed, or
        driven and so without one) while ``params.actuator`` holds a kernel for it."""
        if nodes[c].kernel is not None:
            raise ValueError(
                f"{what} takes away the action node {nodes[c].action!r} of control node "
                f"{nodes[c].name!r} (the node is removed or driven), and params.actuator holds a "
                "kernel for it that would be dropped: drop the kernel, or leave the node to be "
                "chosen"
            )

    def table_is_owned(g: int, what: str) -> None:
        """Refuse a rewrite of the table of a factor that reads a parameter variable: the table
        is the variable's, shared by every reader, and one reader's axes cannot change alone."""
        if ("transition", g) in reads:
            raise NotImplementedError(
                f"{what} rewrites the control axis of factor {state_names[g]!r}'s transition "
                f"table, which is the parameter variable {reads[('transition', g)]!r} that other "
                "readers share; declare the factor's own table (not a Parameter reference) "
                "where a binding changes its axes"
            )
        home = sp.noise_reads.get(("transition", g))
        if home is not None and _noise_batch(parameters[home]):
            raise NotImplementedError(
                f"{what} rewrites the control axis of factor {state_names[g]!r}, whose noise is "
                f"the per-mode noise variable {home!r} that other readers share; declare the "
                "factor's own noise_prior where a binding changes its axes"
            )

    def cut(k: int, keep, weights) -> None:
        """Slice or average node ``k`` out of every table that reads it."""
        for g, pos in readers_of(k):
            table_is_owned(g, f"a binding on control node {nodes[k].name!r}")
            floats = [p for p, i in control_parents(g) if nodes[i].kind == FLOAT]
            transition[g] = _cut_axes(
                transition[g],
                state_names[g],
                len(control_parents(g)),
                [(pos, keep, weights)],
                floats,
            )
            del parents[g][pos]
        nodes[k].live = False

    #  what each binding that crosses a component's boundary is shaped as, per component record
    #  (a top-level component and the components of an inner composite): "blanket" for a
    #  belief read and a link from an action node into a control node, "shared" for a binding that
    #  identifies a latent outright
    entry_names = [
        *subsystems,
        *(
            f"{name}{SEPARATOR}{child}"
            for name, s in subsystems.items()
            for child, _ in s.subsystems or ()
        ),
    ]
    crossing: dict[str, set[str]] = {entry: set() for entry in entry_names}
    for name, system in subsystems.items():
        for child, sub in system.subsystems or ():
            if sub.boundary != "none":
                crossing[f"{name}{SEPARATOR}{child}"].add(sub.boundary)

    def chain(node: str) -> set[str]:
        """The component records that contain the namespaced name ``node``."""
        return {e for e in entry_names if node.startswith(f"{e}{SEPARATOR}")}

    def cross(kind: str, *ends: str) -> None:
        """Record a binding of shape ``kind`` among the namespaced names ``ends``, on every
        component record that holds some of them and not all."""
        chains = [chain(end) for end in ends]
        for held in chains:
            for entry in held - set.intersection(*chains):
                crossing[entry].add(kind)

    sibling: list[_BeliefRead] = []
    belief_emissions: list[tuple[str, Emission]] = []  # the emissions a belief read declares
    belief_observes: list[tuple[str, str]] = []  # and the modalities that observe them
    own_read_modalities = {f"{name}{SEPARATOR}{slot}" for name, _, (slot, _) in own_reads}
    emission_names = {e for e, _ in emissions}
    action_to_node = {node.action: node.name for node in nodes if node.action is not None}
    node_to_action = {node.name: node.action for node in nodes if node.action is not None}
    port_observes: list[tuple[str, str]] = []  # the slots a port Identify feeds, with variables
    identified_outs: dict[str, str] = {}  # an out-port an Identify named -> the binding's label
    driven_by: dict[int, str] = {  # a driven node -> the variable that drives it
        node_index[f"{name}{SEPARATOR}{slot}"]: f"{name}{SEPARATOR}{variable}"
        for name, _, (slot, variable) in own_reads
        if f"{name}{SEPARATOR}{slot}" in node_index
    }

    def classify(name: str) -> str:
        """What the namespaced ``name`` is in the composite under construction."""
        if name in factor_index:
            return "a state factor"
        if name in action_to_node:
            return "an action node"
        if name in node_index:
            return "a control node"
        if name in emission_names:
            return "an emission"
        if name in modality_index:
            return "a modality"
        if name in nested_gone:
            return "gone"
        return "unknown"

    def no_inert_end(label: str, names) -> None:
        """Refuse a binding that names the action node of a driven control node: the node and its
        driver's action node are one variable, named by the driver's action node."""
        for end in names:
            if isinstance(end, str) and end in alias:
                node = next(n for n, a in inert.items() if a == end)
                raise ValueError(
                    f"{label} names {end!r}, the action node of control node {node!r}, which "
                    f"{alias[end]!r} drives, so {end!r} has no value of its own: it would carry "
                    f"{alias[end]!r}'s; name {alias[end]!r} itself (the reader then steps after "
                    "its driver, a join on it rather than a chain)"
                )

    def bind_port(variable: str, slot: str, label: str) -> None:
        """Identify the out-port ``variable`` with the in-port ``slot``: the slot observes the
        variable. A control node slot is driven (it keeps its place in its leaf and loses its
        action node); the pair is recorded for the leaf's record."""
        if slot in {s for s, _ in port_observes} or slot in own_read_modalities:
            raise ValueError(
                f"{label}: the in-port {slot!r} already observes a variable (its "
                "component's System.observes, a belief read or another Identify); an "
                "in-port is identified at most once"
            )
        if slot in {b.modality for b in sibling}:
            raise ValueError(
                f"{label}: the in-port {slot!r} already observes a variable (a belief "
                "read); an in-port is identified at most once"
            )
        if slot in node_index:
            c = driven_node(slot, "Identify")
            if variable in action_to_node:
                cross("blanket", variable, slot)
            kernel_not_dropped(c, label)
            driver = action_to_node.get(variable)
            if driver is not None and node_index[driver] == c:
                raise ValueError(
                    f"{label}: control node {slot!r} cannot be driven by its own action node"
                )
            d = None if driver is None else node_index[driver]
            if d is not None and d in relayed:
                raise relay_refusal(
                    state_names[relayed[d]],
                    nodes[d].name,
                    [relay_reader[d], *(g for g, _ in readers_of(c))],
                    [*(nodes[k].name for k, v in driven_by.items() if v == nodes[d].action), slot],
                )
            sensing = [
                m for m, deps in enumerate(A_deps) if any(g == -c - 1 for g in deps if g < 0)
            ]
            if sensing and d is None:
                raise NotImplementedError(
                    f"{label}: modality {modality_names[sensing[0]]!r} observes the action node "
                    f"of {slot!r}, which a driven node does not have (nothing chooses it)"
                )
            for m in sensing:  # the node's realised value is its driver's: observe that
                A_deps[m] = tuple(-d - 1 if g == -c - 1 else g for g in A_deps[m])
            old = nodes[c].action  # None for an in-port of a black box, which has no action node
            if old is not None and old in identified_outs:
                raise ValueError(
                    f"{identified_outs[old]} names {old!r}, the action node of control node "
                    f"{slot!r}, and {label} drives {slot!r} by {variable!r}, so {old!r} has no "
                    f"value of its own: it would carry {variable!r}'s; name {variable!r} "
                    f"itself, or drive {slot!r} from elsewhere"
                )
            if old is not None:
                inert[slot] = old
                alias[old] = variable
                del action_to_node[old]
            node_to_action.pop(slot, None)
            nodes[c].action = None
            if d is not None:
                identified_pairs.append((slot, driver))
            for k, v in driven_by.items():  # a node this node's action node drove follows it
                if old is not None and v == old:
                    driven_by[k] = variable
            for i, (pair_slot, pair_variable) in enumerate(port_observes):
                if old is not None and pair_variable == old:
                    port_observes[i] = (pair_slot, variable)
            for i, (pair_node, pair_driver) in enumerate(identified_pairs):
                if pair_driver == slot and d is not None:
                    identified_pairs[i] = (pair_node, driver)
            if d is not None and d in terminations:
                #  a driver a Clamp terminated terminates the node it drives, with the same prior
                cut(c, *terminations[d])
                return
            driven_by[c] = variable
        else:
            cross("blanket", variable, slot)
        port_observes.append((slot, variable))

    # ----------------------------------------------------------- the bindings
    for binding in bindings:
        if coupled and not isinstance(binding, Identify | Clamp):
            raise ValueError(
                f"couple(): {binding!r} is not an Identify or a Clamp, the bindings of a "
                "coupling: a Link is an arc of one model's graph, a structural binding that "
                "merge() makes (a belief read is an emission and an observing slot, written "
                "with Identify of the two ports)"
            )
        if isinstance(binding, Link):
            source, target = binding.source, binding.target
            label = f"Link({source!r} -> {target!r})"
            if not isinstance(target, str):
                raise TypeError(
                    f"{label}: the target is one name, a modality, a control node or "
                    f"'{TERMINAL}:<factor>', got {target!r}"
                )
            ends = (source,) if isinstance(source, str) else tuple(source)
            if classify(target) != "a modality":  # a port into a modality is refused below
                no_inert_end(label, (*ends, target))
            ends = tuple(alias.get(end, end) if isinstance(end, str) else end for end in ends)
            source = ends[0] if isinstance(source, str) else ends
            if not ends or not all(isinstance(end, str) for end in ends):
                raise ValueError(
                    f"{label}: the source is a node's name, or a tuple of names where the link "
                    "states a modality's or a terminal target's complete parent list"
                )
            if binding.kernel is not None and binding.realise is None:
                raise ValueError(
                    f"{label}: a kernel without realise: a kernel pushes a belief to the value a "
                    "realisation then draws or takes, so a belief read gives both; a structural "
                    "link, which names a modality's parents, has no kernel"
                )
            if binding.realise is not None and binding.realise not in REALISATIONS:
                raise ValueError(
                    f"{label}: realise must be one of {list(REALISATIONS)}, got {binding.realise!r}"
                )
            kinds = [classify(end) for end in ends]
            for end, kind in zip(ends, kinds, strict=True):
                if kind == "gone":
                    raise ValueError(f"{label}: the source {end!r}, {nested_gone[end]}")
                if kind == "a control node":
                    raise ValueError(
                        f"{label}: the source {end!r} is a control node, which is no port: a link "
                        f"carries the realised value of its action node, so write "
                        f"{node_to_action[end]!r}"
                    )
            ported = [
                (end, kind)
                for end, kind in zip(ends, kinds, strict=True)
                if kind in ("an action node", "an emission")
            ]
            if ported and binding.realise is not None:
                raise ValueError(
                    f"{label}: {ported[0][0]!r} is {ported[0][1]}, a realised port: it is "
                    "realised once, by its own kernel, and every link from it carries that one "
                    "value, so a link from it takes no realise"
                )
            if target.startswith(f"{TERMINAL}:"):
                factor = target[len(TERMINAL) + 1 :]
                if binding.realise is not None:
                    raise ValueError(
                        f"{label}: a link to a terminal target is structural (the sources are the "
                        "parents of its H_deps entry) and takes no realise"
                    )
                if ported:
                    raise ValueError(
                        f"{label}: a terminal target is conditioned on state factors, and "
                        f"{ported[0][0]!r} is {ported[0][1]}"
                    )
                cross("shared", factor, *ends)
                f = _lookup(factor, factor_index, "factor")
                emitters = tuple(_lookup(g, factor_index, "factor") for g in ends)
                if H_deps[f]:
                    raise ValueError(
                        f"factor {factor!r} already declares H_deps parent(s) "
                        f"{[state_names[g] for g in H_deps[f]]} of its own; a terminal link "
                        "declares the parents of an as-yet-unconditioned terminal target, it "
                        "does not override a component's own declaration"
                    )
                H_deps[f] = emitters
                continue
            what = classify(target)
            if what == "a modality":
                if ported:
                    raise ValueError(
                        f"{label}: a link from {ported[0][1]} {ported[0][0]!r} into the modality "
                        f"{target!r} is a coupling's wire, which couple() carries between "
                        "components run as separate particles; a merge makes one particle, so "
                        "link state factors to the modality (structurally, or with realise= for "
                        "a belief read)"
                    )
                if binding.realise is None:
                    cross("shared", target, *ends)
                    m = _lookup(target, modality_index, "modality")
                    linked = tuple(_lookup(f, factor_index, "factor") for f in ends)
                    a = likelihood[m]
                    if isinstance(a, ConditionalCategorical | ConditionalDirichlet):
                        if a.parent_ndims != len(linked):
                            raise ValueError(
                                f"{label} names {len(linked)} parent(s) but the likelihood table "
                                f"of {target!r} has {a.parent_ndims} parent axis/es; a link "
                                "states the modality's complete new parent list, in A_deps order"
                            )
                        sizes = tuple(
                            _factor_levels(params[owner[g]], g - fac_offset[owner[g]])
                            for g in linked
                        )
                        if all(s is not None for s in sizes) and a.parent_shape != sizes:
                            raise ValueError(
                                f"support mismatch: modality {target!r} has parent axes "
                                f"{a.parent_shape} but {list(ends)} have {sizes} level(s); "
                                "a structural link pairs a modality's parent axes with the "
                                "linked factors' supports"
                            )
                    A_deps[m] = linked
                    continue
                if not isinstance(source, str):
                    raise ValueError(
                        f"{label}: a belief read has one source, a state factor of another "
                        "component; a tuple of sources states a modality's parents (drop realise)"
                    )
                if source not in factor_index:
                    raise ValueError(
                        f"{label}: the source is {classify(source)}; a belief read takes a state "
                        f"factor of another component, one of {list(state_names)}"
                    )
                if target.split(SEPARATOR)[0] == source.split(SEPARATOR)[0]:
                    raise ValueError(
                        f"{label}: the modality and the source are both in component "
                        f"{target.split(SEPARATOR)[0]!r}; a read inside one component is declared "
                        "on that System, as an emission and System.observes (a belief read "
                        "crosses a component boundary)"
                        + (
                            "; for components of an inner composite, declare the link in the "
                            "inner merge"
                            if target.count(SEPARATOR) > 1
                            else ""
                        )
                    )
                kernel = binding.kernel
                if not (kernel is None or kernel == "identity" or isinstance(kernel, Grid)):
                    raise ValueError(
                        f"{label}: kernel must be 'identity', None or a Grid, got {kernel!r}"
                    )
                belief = _BeliefRead(target, source, kernel or "identity", binding.realise)
                if target in {*own_read_modalities, *(b.modality for b in sibling)}:
                    raise ValueError(
                        f"{label}: the modality already observes a variable (its component's "
                        "System.observes or another belief read); a modality's datum has one "
                        "source"
                    )
                tail = target.rpartition(SEPARATOR)[2]
                emitted = f"{source.rpartition(SEPARATOR)[0]}{SEPARATOR}{tail}"
                declared = Emission((source,), binding.realise, kernel or "identity")
                made = dict(belief_emissions)
                taken = (
                    emission_names
                    | set(factor_index)
                    | set(modality_index)
                    | set(action_to_node)
                    | set(node_index)
                )
                if emitted in made and made[emitted] == declared:
                    pass  # one out-port, many in-ports: the readers share the emission
                elif emitted in taken:
                    raise ValueError(
                        f"{label}: the belief read is an emission of the source's component named "
                        f"{emitted!r} (the modality's name under the source's component), and "
                        "the composite already has a variable of that name with another source, "
                        "kernel or realisation; name the reading modality otherwise, or declare "
                        "the emission on the source's System under a name of its own"
                    )
                else:
                    emission_names.add(emitted)
                    belief_emissions.append((emitted, declared))
                belief_observes.append((target, emitted))
                cross("blanket", target, source)
                sibling.append(belief)
                continue
            if what == "a control node":
                if len(ends) != 1:
                    raise ValueError(
                        f"{label}: a control node is driven by one source, not a tuple of "
                        f"{len(ends)}"
                    )
                (driver,) = ends
                (kind,) = kinds
                if kind == "an emission":
                    raise ValueError(
                        f"{label}: a link from the emission {driver!r} into the control node "
                        f"{target!r} is a coupling's wire, which couple() carries between "
                        "components run as separate particles; a merge makes one particle, so "
                        "link an action node, or a state factor (a relay)"
                    )
                if kind == "an action node":
                    nd, nc = (
                        nodes[node_index[action_to_node[driver]]],
                        nodes[driven_node(target, "Link")],
                    )
                    noun = {LEVEL: "discrete", FLOAT: "continuous"}
                    if nd.kind != nc.kind:
                        raise ValueError(
                            f"node kind mismatch: driver node {driver!r} is {noun[nd.kind]} and "
                            f"{target!r} is {noun[nc.kind]}; a same-step binding makes two nodes "
                            "one, so bind nodes of one kind"
                        )
                    if nd.size != nc.size:
                        unit = "level(s)" if nd.kind == LEVEL else "dimension(s)"
                        raise ValueError(
                            f"node arity mismatch: driver node {driver!r} has {nd.size} {unit} "
                            f"but {target!r} has {nc.size}; a same-step binding makes two nodes "
                            f"one, so their {'level counts' if nd.kind == LEVEL else 'widths'} "
                            "must agree"
                        )
                    bind_port(driver, target, label)
                    identified_outs.setdefault(driver, label)
                    continue
                if kind != "a state factor":
                    raise ValueError(
                        f"{label}: the source is {kind}; a link into a control node comes from "
                        "an action node (read at the same step), one of "
                        f"{sorted(action_to_node)}, or a state factor (read a step later), one "
                        f"of {list(state_names)}"
                    )
                if binding.realise is not None:
                    raise ValueError(
                        f"{label}: a relay from a state factor into a control node is "
                        "structural (the state becomes the node's reader's parent) and takes no "
                        "realise"
                    )
                cross("shared", driver, target)
                d = factor_index[driver]
                c = driven_node(target, "Link")
                levels = _factor_levels(params[owner[d]], d - fac_offset[owner[d]])
                if levels is None:
                    hint = (
                        repr(node_to_action[control_name(driver)])
                        if control_name(driver) in node_index
                        else "an action node"
                    )
                    raise NotImplementedError(
                        f"{label}: the source is a continuous state, and a relay reads a state's "
                        f"level: link from {hint} for a same-step read"
                    )
                if nodes[c].kind == FLOAT:
                    raise ValueError(
                        f"node kind mismatch: driver {driver!r} is a discrete state and "
                        f"{target!r} is a continuous node; a relay reads a state's level, so "
                        f"drive {target!r} from the action node of a continuous control node at "
                        "the same step"
                    )
                if levels != nodes[c].size:
                    raise ValueError(
                        f"node arity mismatch: driver {driver!r} has {levels} level(s) but "
                        f"{target!r} has {nodes[c].size}; a relay identifies a state's levels "
                        "with a control node's one-for-one"
                    )
                readers = readers_of(c)
                partnered = [k for k, v in driven_by.items() if v == nodes[c].action]
                if partnered or len(readers) != 1 or len(control_parents(readers[0][0])) != 1:
                    raise relay_refusal(
                        driver,
                        target,
                        [g for g, _ in readers] + [g for k in partnered for g, _ in readers_of(k)],
                        [nodes[k].name for k in partnered],
                    )
                kernel_not_dropped(c, label)
                g, pos = readers[0]
                table_is_owned(g, label)
                transition[g] = _drop_control_axis(transition[g], state_names[g], None, None)
                parents[g][pos] = ("state", d)
                nodes[c].live = False
                relayed[c] = d
                relay_reader[c] = g
                continue
            if what in ("gone", "a state factor", "an action node"):
                driven_node(target, "Link")  # names what the target is and the control node to give
            raise ValueError(
                f"{label}: the target {target!r} is {what if what != 'unknown' else 'unknown'}; "
                "a link's target is a modality (a structural link or a belief read), a control "
                f"node (a driver's action node, or a relayed state) or '{TERMINAL}:<factor>'; the "
                f"modalities are {sorted(modality_index)} and the control nodes "
                f"{sorted(node_index)}"
            )
        elif isinstance(binding, Clamp):
            if binding.node in dict(sp.black_box):
                raise ValueError(
                    f"Clamp for {binding.node!r}: it is an in-port of a black box, which a Clamp "
                    "would remove from the tables that read it, and a black box's step reads it "
                    "itself; drive it from a component that emits the constant (an Identify of "
                    "that out-port with it)"
                )
            c = driven_node(binding.node, "Clamp")
            if nodes[c].kind == FLOAT:
                raise NotImplementedError(
                    f"Clamp for {binding.node!r}: a continuous control node's prior would have to "
                    "enter its readers as a continuous parent distribution, which no transition "
                    "family accepts; clamp a discrete node"
                )
            keep, weights = _termination(
                binding.level, nodes[c].size, f"Clamp for {binding.node!r}"
            )
            kernel_not_dropped(c, f"Clamp for {binding.node!r}")
            cut(c, keep, weights)
            terminations[c] = (keep, weights)
            for k in [k for k, v in driven_by.items() if v == nodes[c].action]:
                #  the nodes this node's action node drives end with the same prior
                cut(k, keep, weights)
                del driven_by[k]
                port_observes[:] = [(a, v) for a, v in port_observes if a != nodes[k].name]
        elif isinstance(binding, Identify):
            no_inert_end(f"Identify({binding.a!r}, {binding.b!r})", (binding.a, binding.b))
            ends = (alias.get(binding.a, binding.a), alias.get(binding.b, binding.b))
            if not any(end in parameters for end in ends) and all(
                classify(end) != "unknown" or end in parameters for end in ends
            ):
                what_a, what_b = classify(ends[0]), classify(ends[1])
                outs = [e for e, w in zip(ends, (what_a, what_b), strict=True) if w in _OUT_PORTS]
                ins = [e for e, w in zip(ends, (what_a, what_b), strict=True) if w in _IN_PORTS]
                label = f"Identify({binding.a!r}, {binding.b!r})"
                if len(outs) != 1 or len(ins) != 1:
                    raise ValueError(
                        f"{label}: an identification of ports joins one out-port (an action node "
                        "or an emission) to one in-port (a modality or a control node); "
                        f"{binding.a!r} is {what_a} and {binding.b!r} is {what_b}"
                        + (
                            ". Two latents are identified in one model with a Link, two parameter "
                            "variables with Identify of their names"
                            if "a state factor" in (what_a, what_b)
                            else ""
                        )
                    )
                if (
                    coupled
                    and outs[0].split(SEPARATOR)[0] == ins[0].split(SEPARATOR)[0]
                    and outs[0] in dict(sp.black_box)
                ):
                    raise ValueError(
                        f"couple(): {label} joins two ports of the black box "
                        f"{outs[0].split(SEPARATOR)[0]!r}; a coupling identifies ports of "
                        "different components, and what a black box reads of its own outputs it "
                        "keeps in its step's carry"
                    )
                if coupled and outs[0].split(SEPARATOR)[0] == ins[0].split(SEPARATOR)[0]:
                    raise ValueError(
                        f"couple(): {label} joins two ports of component "
                        f"{outs[0].split(SEPARATOR)[0]!r}; a coupling identifies ports of "
                        "different components, and a read inside one component is its own "
                        "System.observes"
                    )
                bind_port(outs[0], ins[0], label)
                identified_outs.setdefault(outs[0], label)
                continue
            cross("shared", binding.a, binding.b)
            for end in (binding.a, binding.b):
                if end not in parameters and classify(end) not in (*_OUT_PORTS, *_IN_PORTS):
                    owner = end.split(SEPARATOR)[0]
                    if owner not in subsystems:
                        where = f"names no component; the components are {sorted(subsystems)}"
                    else:
                        prefix = f"{owner}{SEPARATOR}"
                        outs = sorted(
                            n for n in (*action_to_node, *emission_names) if n.startswith(prefix)
                        )
                        ins = sorted(
                            n for n in (*modality_index, *node_index) if n.startswith(prefix)
                        )
                        where = (
                            f"is no port of component {owner!r}; its out-ports are {outs} and "
                            f"its in-ports {ins}"
                        )
                    if parameters:
                        raise ValueError(
                            f"Identify names {end!r}, which is not a parameter variable of the "
                            f"composite; they are {sorted(parameters)} (a component's own name "
                            f"under its prefix), nor a port to identify with one: it {where}"
                        )
                    raise ValueError(f"Identify names {end!r}, which {where}")
            if coupled:
                raise ValueError(
                    f"couple(): Identify({binding.a!r}, {binding.b!r}) joins two parameter "
                    "variables, which a merge does (a coupling identifies ports)"
                )
            if binding.a == binding.b:
                raise ValueError(f"Identify names {binding.a!r} on both sides")
            _check_same_variable(binding, parameters[binding.a], parameters[binding.b])
            for address, home in reads.items():
                if home == binding.b:
                    reads[address] = binding.a
            for address, home in sp.noise_reads.items():
                if home == binding.b:
                    sp.noise_reads[address] = binding.a
            for emission, home in emission_reads.items():
                if home == binding.b:
                    emission_reads[emission] = binding.a
            del parameters[binding.b]
        else:
            raise TypeError(
                f"unknown binding {binding!r}; the kinds are Link (an arc: a driver's action "
                "node into a control node, a relayed state, a modality's parents, a belief read, "
                "a terminal target), Identify (two parameter variables made one) and Clamp (a "
                "control node held at a constant)"
            )

    #  an emission's control parent follows its node through a same-step identification; a node a
    #  relay or a Clamp removed leaves the emission nothing to read
    composite_emissions: list[tuple[str, Emission]] = []
    for emission, emit in emissions:
        resolved = []
        for q in emit.parents:
            if q in nested_gone:
                raise ValueError(f"merge: emission {emission!r} reads {q!r}, {nested_gone[q]}")
            if q in node_index:
                final = q
                k = node_index[final]
                if not nodes[k].live:
                    how = (
                        f"merged with the state {state_names[relayed[k]]!r} by a relay"
                        if k in relayed
                        else "terminated by a Clamp"
                    )
                    raise ValueError(
                        f"merge: emission {emission!r} reads the control node {final!r}, which a "
                        f"binding removed ({how}), so the composite has no such node to read; "
                        "read the driver's node instead, or drop the binding"
                    )
                q = final
            resolved.append(q)
        if len(set(resolved)) != len(resolved):
            raise ValueError(
                f"merge: emission {emission!r} reads one control node twice after its nodes were "
                "identified; its parents must stay distinct"
            )
        composite_emissions.append((emission, Emission(tuple(resolved), emit.realise, emit.kernel)))
    composite_emissions.extend(belief_emissions)

    #  every component's own observes entries under its prefix (an action variable follows its
    #  node through a same-step identification, to a node the component still chooses), then the
    #  belief reads' entries
    observes_out: list[tuple[str, str]] = []
    for name, system, (slot, variable) in own_reads:
        prefix = f"{name}{SEPARATOR}"
        target = prefix + variable
        if variable in (system.action_names or ()):
            node = next(
                (n for n in nodes if n.action == target or inert.get(n.name) == target), None
            )
            final = node
            via = None if node is None else driven_by.get(node_index[node.name])
            if node is not None and node.action is None and via in action_to_node:
                final = nodes[
                    node_index[action_to_node[via]]
                ]  # a driven node's value is its driver's
            if final is None or not final.live or not final.name.startswith(prefix):
                gone = (
                    "a binding removed"
                    if final is None or not final.live
                    else f"{final.name!r}, another component's"
                )
                raise ValueError(
                    f"merge: {prefix + slot!r} observes the action node {target!r} in "
                    f"System.observes, whose control node {node.name if node else '?'!r} is "
                    f"{gone} node; an observation of an action node stays inside the "
                    "component that chooses it"
                )
            target = final.action
        observes_out.append((prefix + slot, target))
    observes_out.extend(belief_observes)
    followed = {n.action: n for n in nodes if n.action is not None}
    for slot, variable in port_observes:
        if variable in followed:
            final = followed[variable]
            if not final.live:
                raise ValueError(
                    f"merge: {slot!r} observes the action node {variable!r}, whose control node a "
                    "binding removed (relayed into a state or terminated with a Clamp), so no "
                    "action is realised there"
                )
            variable = final.action
        observes_out.append((slot, variable))
    #  the modalities' entries in modality order, then the control nodes' in node order
    slot_order = {
        **{m: i for i, m in enumerate(modality_names)},
        **{node.name: len(modality_names) + k for k, node in enumerate(nodes)},
    }
    observes_out.sort(key=lambda entry: slot_order[entry[0]])

    #  an observed action node, held as -(node + 1), takes its column among the live nodes: the
    #  composite's action node k sits at len(B_deps) + k
    live_nodes = [k for k, node in enumerate(nodes) if node.live]
    pooled_over = []
    for table, node in readings:
        final = node
        k = node_index.get(final)
        if k is None or not nodes[k].live:
            raise ValueError(
                f"merge: {table!r} pools its reading over the control node {node!r}, which a "
                "binding removed (relayed into a state or terminated with a Clamp), so "
                "its count has no level to follow; identify the node with a driver's instead, "
                "or drop the binding"
            )
        pooled_over.append((table, final))
    n_states = len(state_names)
    n_columns = n_states + len(live_nodes)
    for m, deps in enumerate(A_deps):
        if all(g >= 0 for g in deps):
            continue
        placed = []
        for g in deps:
            if g >= 0:
                placed.append(g)
                continue
            k = node_index.get(nodes[-g - 1].name)
            if k is None or not nodes[k].live:
                raise NotImplementedError(
                    f"merge: modality {modality_names[m]!r} observes the action node "
                    f"{nodes[-g - 1].action!r}, whose control node a binding removed (relayed "
                    "into a state or terminated with a Clamp), so no action is realised "
                    "there: observe the driver's node instead, or drop the binding"
                )
            placed.append(n_columns + live_nodes.index(k))
        if len(set(placed)) != len(placed):
            raise ValueError(
                f"merge: modality {modality_names[m]!r} observes one action node twice after "
                "its control nodes were identified; its parents must stay distinct"
            )
        A_deps[m] = tuple(placed)

    for what, targets in (("state_targets", state_targets), ("terminal_targets", terminal_targets)):
        for key in targets:
            k = node_index.get(key)
            if k in relayed or k in terminations:
                how = (
                    f"merged with the state {state_names[relayed[k]]!r}"
                    if k in relayed
                    else "terminated with a Clamp"
                )
                raise ValueError(
                    f"merge: {what}[{key!r}] is a target on a control node {how}, so the "
                    "composite has no such node: drop the target"
                )

    #  a conditional target's parent must be a node the composite has; a relayed or terminated
    #  node is gone
    for i, (key, c_parents) in enumerate(C_deps):
        renamed = []
        for p in c_parents:
            k = node_index.get(p)
            if k in terminations:
                raise ValueError(
                    f"merge: the target on {key!r} is conditional on the control node {p!r}, "
                    "which a Clamp terminated, so the composite has no such node: drop "
                    "the target or condition it on another node"
                )
            if k in relayed:
                raise ValueError(
                    f"merge: the target on {key!r} is conditional on the control node {p!r}, "
                    f"which a relay identified with the state {state_names[relayed[k]]!r}; a "
                    "relay reads the state a step later while a per-step target reads the "
                    "step's own belief, so condition the target on the driver or drop the "
                    "relay"
                )
            renamed.append(p)
        C_deps[i] = (key, tuple(renamed))

    # every identification of an action node with a control node, the components' own
    # re-qualified, as (node, driver) names
    identified = [
        (namespaced_key(name, node), namespaced_key(name, driver))
        for name, system in subsystems.items()
        for _, sub in system.subsystems or ()
        for node, driver in sub.identified
    ]
    identified = list(dict.fromkeys([*identified, *sorted(identified_pairs)]))

    # ---------------------------------------------------------- the metadata
    # One entry per component, plus a re-qualified, offset entry per component of an inner
    # composite. An inner entry inherits its own merge's record, with this merge's bindings added
    # on top.
    live_names = [node.name for node in nodes if node.live]
    driven_names = {node.name for node in nodes if node.action is None}
    action_of = {node.name: node.action for node in nodes}
    action_of.update(inert)
    for name, system in subsystems.items():
        for _, sub in system.subsystems or ():
            for node, action in sub.actions:
                action_of.setdefault(namespaced_key(name, node), namespaced_key(name, action))
    boxed = dict(sp.black_box)
    this_relayed = {nodes[k].name: state_names[d] for k, d in relayed.items()}
    this_terminated = [nodes[k].name for k in terminations]
    entries: list[tuple[str, tuple[int, ...], tuple[int, ...], dict, list, list]] = []
    for name, system in subsystems.items():
        f_off, m_off = fac_offset[name], mod_offset[name]
        facs = tuple(range(f_off, f_off + len(system._B)))
        mods = tuple(range(m_off, m_off + len(system.A_deps)))
        entries.append((name, facs, mods, {}, [], []))
        for child_name, child in system.subsystems or ():
            entries.append(
                (
                    f"{name}{SEPARATOR}{child_name}",
                    tuple(f + f_off for f in child.states),
                    tuple(m + m_off for m in child.modalities),
                    {
                        namespaced_key(name, node): namespaced_key(name, state)
                        for node, state in child.relayed
                    },
                    [namespaced_key(name, node) for node in child.terminated],
                    [
                        (f"{name}{SEPARATOR}{o}", f"{name}{SEPARATOR}{src}", kernel, realise)
                        for o, src, kernel, realise in child.reads
                    ],
                )
            )
    metadata: list[tuple[str, Subsystem]] = []
    for name, facs, mods, inherited_relayed, inherited_terminated, inherited_reads in entries:

        def mine(node: str, prefix: str = f"{name}{SEPARATOR}") -> bool:
            return node.startswith(prefix)

        relays = {**inherited_relayed, **{k: v for k, v in this_relayed.items() if mine(k)}}
        ended = list(
            dict.fromkeys([*inherited_terminated, *(k for k in this_terminated if mine(k))])
        )
        pairs = tuple(pair for pair in identified if mine(pair[0]))
        chosen = tuple(node for node in live_names if mine(node) and node not in driven_names)
        read_here = tuple(
            [
                *inherited_reads,
                *(
                    (b.modality, b.source, b.kernel, b.realise)
                    for b in sibling
                    if modality_index[b.modality] in mods
                ),
            ]
        )
        boundary = (
            "coupled"
            if coupled and name in subsystems
            else "shared"
            if "shared" in crossing[name]
            else "blanket"
            if crossing[name]
            else "none"
        )
        relays = dict(sorted(relays.items(), key=lambda kv: _node_order(kv[0], state_names)))
        ended = sorted(ended, key=lambda node: _node_order(node, state_names))
        #  an in-port of a black box has no action node, so no pair of its own
        named = [*chosen, *relays, *ended, *(node for node, _ in pairs if node not in boxed)]
        metadata.append(
            (
                name,
                Subsystem(
                    states=facs,
                    modalities=mods,
                    controls=chosen,
                    actions=tuple((node, action_of[node]) for node in dict.fromkeys(named)),
                    relayed=tuple(relays.items()),
                    terminated=tuple(ended),
                    identified=pairs,
                    free_modalities=tuple(
                        m for m in mods if all(g in facs for g in _state_deps(A_deps[m], n_states))
                    ),
                    bound_modalities=tuple(
                        (m, tuple(g for g in _state_deps(A_deps[m], n_states) if g not in facs))
                        for m in mods
                        if any(g not in facs for g in _state_deps(A_deps[m], n_states))
                    ),
                    target_emissions=tuple(
                        (f, H_deps[f])
                        for f in facs
                        if H_deps[f] and any(g not in facs for g in H_deps[f])
                    ),
                    reads=read_here,
                    boundary=boundary,
                    emissions=tuple(
                        e for e, _ in composite_emissions if e.startswith(f"{name}{SEPARATOR}")
                    ),
                    config=sp.configs.get(name, ()),
                    recognition=sp.leaf_recognition.get(name, ()),
                ),
            )
        )

    # ------------------------------------------------------ the composite System
    #  a black box's configuration is inert (it plans nothing and carries no belief), so the
    #  composite's planner fields and its belief carry are its other leaves'
    boxed_ports = [port for port, _ in sp.black_box]
    leaf_config = {
        leaf: dict(config)
        for leaf, config in sp.configs.items()
        if not any(port.startswith(f"{leaf}{SEPARATOR}") for port in boxed_ports)
    } or {leaf: dict(config) for leaf, config in sp.configs.items()}
    #  a term a leaf's scoring owns answers to that leaf's critic alone: the leaf's own terms
    #  carry it, the coupling's none
    owned_terms = {
        *(name for c in leaf_config.values() for name, _ in c["scoring"]),
        *(name for part in sp.parts.values() for name in part.scoring),
    }
    mixture_ks = {c["mixture_k"] for c in leaf_config.values()} - {None}
    if len(mixture_ks) > 1:
        raise ValueError(
            f"merged Systems declare different mixture budgets mixture_k={sorted(mixture_ks)}; "
            "the budget is shared by the whole belief carry, so declare one value"
        )
    #  a component of black boxes alone selects nothing (it has no variable to gain information
    #  about), at whatever depth the boxes sit
    selecting = [s for s in subsystems.values() if not s._only_ports] or [*subsystems.values()]
    if all(s.epistemic is True for s in selecting):
        epistemic: bool | tuple[str, ...] = True
    elif all(s.epistemic is False or s.epistemic == () for s in selecting):
        epistemic = False
    else:
        epistemic = tuple(epistemic_keys)
    #  the tables a plan learns within it compose as epistemic's keys do: a union of each
    #  component's readers (parameters-within-the-plan.md D1)
    if all(s.plan_learning is True for s in selecting):
        plan_learning: bool | tuple[str, ...] = True
    else:
        plan_learning = tuple(dict.fromkeys(sp.plan_learning_keys))
    live = [k for k, node in enumerate(nodes) if node.live]
    planning = {
        field: _unanimous({leaf: c[field] for leaf, c in leaf_config.items()})
        for field in _PLANNING_FIELDS
    }
    schedules = {leaf: c["mixture_schedule"] for leaf, c in leaf_config.items()}
    if len(set(schedules.values())) > 1:
        raise ValueError(
            f"merged Systems disagree on mixture_schedule: {schedules}. The schedule is shared "
            "by the whole belief carry, so declare one value"
        )
    mixture_schedule = next(iter(schedules.values()))
    column = {k: n_states + j for j, k in enumerate(live)}
    B_deps = tuple(tuple(i if kind == "state" else column[i] for kind, i in ps) for ps in parents)
    pad = ((),) * len(live)
    composite = System(
        A_deps=tuple(A_deps),
        B_deps=(*B_deps, *pad),
        controls=tuple(column[k] for k in live),
        action_names=tuple(nodes[k].action for k in live),
        **planning,
        epistemic=epistemic,
        plan_learning=plan_learning,
        terms=tuple(t for t in dict.fromkeys(terms) if t.partition(":")[0] not in owned_terms),
        mixture_k=next(iter(mixture_ks), None),
        mixture_schedule=mixture_schedule,
        D_deps=None if all(not d for d in D_deps) else (*D_deps, *pad),
        H_deps=None if all(not h for h in H_deps) else (*H_deps, *pad),
        C_deps=tuple(C_deps),
        modality_names=tuple(modality_names),
        state_names=tuple(state_names),
        control_names=tuple(nodes[k].name for k in live),
        dims=tuple(dims),
        cells=tuple(cells),
        readings=tuple(pooled_over),
        observes=tuple(observes_out),
        emissions=tuple(composite_emissions),
        recognition=tuple(recognition_decl) or None,
        external_inputs=tuple(external_inputs),
        subsystems=tuple(metadata),
        black_box=tuple(sp.black_box),
    )
    boxes = {nodes[k].name: nodes[k].box for k in live if nodes[k].box is not None}
    for (slot, i), home in reads.items():  # every reader's position holds its variable's table
        (transition if slot == "transition" else likelihood)[i] = parameters[home]
    composite_params = SystemParams(
        readings=reading_counts,
        parameters=parameters,
        references=tuple(sorted((slot, i, home) for (slot, i), home in reads.items())),
        noise_references=tuple(sorted((s, i, home) for (s, i), home in sp.noise_reads.items())),
        transition=tuple(transition),
        initial_state_prior=tuple(initial_state_prior),
        likelihood=tuple(likelihood),
        recognition=recognition_params,
        obs_targets=obs_targets,
        state_targets=state_targets,
        terminal_targets=terminal_targets,
        control=boxes,
        emission={
            emission: Parameter(emission_reads[emission])
            if emission in emission_reads
            else emission_kernels[emission]
            for emission, _ in composite_emissions
            if emission in emission_reads or emission in emission_kernels
        },
        actuator={nodes[k].action: nodes[k].kernel for k in live if nodes[k].kernel is not None},
        parts=sp.parts,
    )
    for binding in sibling:
        _check_sibling_shapes(binding, composite, composite_params)
    composite._validate(composite_params)
    return composite, composite._params_out(composite_params)


def _check_sibling_shapes(binding: _BeliefRead, composite: System, params: SystemParams) -> None:
    """Refuse a belief read whose datum does not have the shape its modality takes."""
    label = f"Link({binding.source!r} -> {binding.modality!r}, realise={binding.realise!r})"
    f = composite._facs.index(binding.source)
    decl = params.likelihood[composite.modality_names.index(binding.modality)]
    categorical = composite._families(params)[f] == _hybrid.CATEGORICAL
    if isinstance(binding.kernel, Grid):
        if categorical:
            raise ValueError(
                f"{label}: an indicator kernel is the cell a continuous source falls in, and "
                f"{binding.source!r} is categorical; read it through the identity kernel"
            )
        levels, what = binding.kernel.n_cells, "the grid's cells"
    elif categorical:
        levels, what = _factor_levels(params, f), "the source's levels"
    else:
        if binding.realise == "stats":
            raise ValueError(
                f"{label}: realise='stats' of a Gaussian source is deferred (the sufficient "
                "statistics of a Gaussian belief; design: nesting-and-coupling.md, Deferred); "
                "declare 'sample' or 'mode', or read the source through an indicator over a Grid"
            )
        if isinstance(decl, TabularObservation):
            raise ValueError(
                f"{label}: {binding.modality!r} is a categorical modality and {binding.source!r} "
                "is continuous, so the read needs an indicator kernel, kernel=Grid(...), to give "
                "the modality cells to observe"
            )
        return  # a linear-Gaussian modality reads the mean; the composite's validation checks it
    if not isinstance(decl, TabularObservation) or decl.num_levels != levels:
        have = (
            f"{decl.num_levels} levels"
            if isinstance(decl, TabularObservation)
            else type(decl).__name__
        )
        raise ValueError(
            f"{label}: the datum takes {levels} values ({what}) and {binding.modality!r} is a "
            f"modality of {have}; a categorical modality's level count is {what}"
        )


def _table_of(decl):
    """A parameter variable's defining array: a table's, or a noise prior's scale."""
    if isinstance(decl, InverseWishart):
        return decl.Psi
    if isinstance(decl, InverseGamma):
        return decl.beta
    return decl.concentration if isinstance(decl, ConditionalDirichlet) else decl.log_table


def _check_same_variable(binding: Identify, left, right) -> None:
    """Refuse two parameter variables that are not one declaration's worth: a family or table
    shape that differs."""
    if type(left) is not type(right):
        raise ValueError(
            f"{binding}: {binding.a!r} is a {type(left).__name__} and {binding.b!r} a "
            f"{type(right).__name__}; two parameter variables are identified only when they are "
            "the same family"
        )
    if _table_of(left).shape != _table_of(right).shape or getattr(
        left, "parent_ndims", None
    ) != getattr(right, "parent_ndims", None):
        raise ValueError(
            f"{binding}: {binding.a!r} has table shape {tuple(_table_of(left).shape)} and "
            f"{binding.b!r} has {tuple(_table_of(right).shape)}; two parameter variables are "
            "identified only when their tables have the same shape"
        )


def write_back(
    composite: System,
    params: SystemParams,
    name: str,
    before: SystemParams,
    after: SystemParams,
) -> SystemParams:
    """``params`` with the parameter variables the view of ``name`` learned set to ``after``'s.

    ``before`` is the view's params as :func:`subsystem` returned them and ``after`` the same
    view's params after learning. Each variable ``before`` holds must equal the composite's
    current entry (same type, shape and values, compared concretely) and is then replaced by
    ``after``'s, so the composite holds exactly what the view learned, decay included. A stale
    ``before`` raises: take the view again from the composite after any write-back. Evidence
    that several views gather at once is accumulated by learning on the composite, not by writing
    views back. An ``after`` missing a variable ``before`` holds, a ``before`` of another shape
    or type, and a concentration in ``after`` that is not positive all raise. Nothing else is
    written: a view's own tables are its own to carry. Bookkeeping of one home between the views of
    one particle; nothing writes another particle's parameters. Design: nesting-and-coupling.md N4,
    N6.
    """
    return _write_back(composite, params, name, before, after)


def _write_back(composite, params, name, before, after, flag=None) -> SystemParams:
    """:func:`write_back`, also over traced params (inside ``run``'s scanned round): the check
    that ``before`` is current is made on concrete values only (``run`` takes every view from the
    home it writes back to), and a traced concentration that is not positive is handed to
    ``flag`` (``flag(bad, variable, sentence)``, which records a fault), the variable then keeping
    its current value. Design: batched-run.md R6."""
    params = composite._params_in(params, "write_back")
    unknown = sorted(set(before.parameters) - set(params.parameters))
    if unknown:
        raise ValueError(
            f"write_back: the view of {name!r} holds parameter variable(s) {unknown}, which the "
            f"composite does not declare; it declares {sorted(params.parameters)}"
        )
    homes = dict(params.parameters)
    for var, start in before.parameters.items():
        current = homes[var]
        if var not in after.parameters:
            raise ValueError(
                f"write_back: the view of {name!r} held parameter variable {var!r} before "
                f"learning and holds none after; pass the same view's params"
            )
        end = after.parameters[var]
        if type(start) is not type(current) or _table_of(start).shape != _table_of(current).shape:
            raise ValueError(
                f"write_back: the view of {name!r} was taken with parameter variable {var!r} as a "
                f"{type(start).__name__} of shape {tuple(_table_of(start).shape)}, and the "
                f"composite now holds a {type(current).__name__} of shape "
                f"{tuple(_table_of(current).shape)}; take the view again from the composite"
            )
        stale = not isinstance(_table_of(start), jax.core.Tracer) and not bool(
            jnp.array_equal(_table_of(start), _table_of(current))
        )
        if stale:
            raise ValueError(
                f"write_back: the view of {name!r} was taken before the composite's parameter "
                f"variable {var!r} last changed, so writing it back would erase that change. "
                "Take the view again from the composite after any write-back; evidence from "
                "several views at once is accumulated by learning on the composite, not by "
                "writing views back"
            )
        if type(end) is not type(start) or _table_of(end).shape != _table_of(start).shape:
            raise ValueError(
                f"write_back: the view of {name!r} holds parameter variable {var!r} after "
                "learning in another form than before"
            )
        if isinstance(end, ConditionalDirichlet):
            positive = jnp.all(end.concentration > 0.0)
            sentence = (
                f"write_back: parameter variable {var!r} has a concentration that is not "
                f"positive after the view of {name!r} learned (a decay or rate that overshoots "
                "the anchor); it is not a Dirichlet and is not written"
            )
            if not isinstance(positive, jax.core.Tracer):
                if not bool(positive):
                    raise ValueError(sentence)
            elif flag is None:
                raise ValueError(sentence)
            else:
                flag(~positive, var, sentence)
                end = jax.tree_util.tree_map(functools.partial(jnp.where, positive), end, current)
        homes[var] = end
    out = dataclasses.replace(params, parameters=homes)
    for slot, i, home in params.references:  # every reader's position holds its variable's table
        container = list(getattr(out, slot))
        container[i] = homes[home]
        out = dataclasses.replace(out, **{slot: tuple(container)})
    return composite._params_out(out)  # a noise reader's field is re-read from its variable


def _state_deps(deps: tuple[int, ...], n_states: int) -> tuple[int, ...]:
    """The state-factor parents in ``deps``: an index at or above ``n_states`` is a control or
    action node, which no subsystem's factor set contains."""
    return tuple(g for g in deps if g < n_states)


def _node_order(node: str, state_names: Sequence[str]) -> tuple[int, str]:
    """Sort key for a record's node names: the state a sugar-named node ``u_<f>`` belongs to,
    by factor index, before any other node, by name."""
    space, sep, leaf = node.rpartition(SEPARATOR)
    state = f"{space}{sep}{leaf[2:]}" if leaf.startswith("u_") else None
    if state in state_names:
        return (list(state_names).index(state), "")
    return (len(state_names), node)


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


def nested(composite: System, name: str) -> bool:
    """Whether component ``name`` is nested in the merge: it keeps a blanket of its own inside the
    particle, every binding across its boundary being a belief read (a :class:`Link` with a
    ``realise``) or a link from an action node into a control node, and at least one binding
    crossing it. A relay, a structural link into a modality or a terminal target, or an
    :class:`Identify` across its boundary identifies a latent outright and makes it not nested.

    Design: nesting-and-coupling.md N2.
    """
    return _subsystem_meta(composite, name).nested


def subsystem_states(composite: System, name: str) -> tuple[int, ...]:
    """Composite state factor indices one mind's view carries, in composite order.

    Its own state factors, plus every other component's state factor that is a direct parent of
    something of ours (a ``B_deps``/``D_deps`` parent of our state factor, an ``A_deps`` parent of
    our modality, an ``H_deps`` parent of our terminal designation, a ``C_deps`` state parent of our
    per-step designation), closed transitively over those stubs' own
    declared dynamics, since a stub is carried verbatim. Composite order keeps ``D_deps``
    ancestral.
    """
    meta = _subsystem_meta(composite, name)
    D_deps, H_deps = composite._effective_D_deps(), composite._effective_H_deps()
    table = composite._table
    needed = set(meta.states)
    stack: list[int] = []
    for f in meta.states:
        stack.extend(composite._B[f] + D_deps[f] + H_deps[f])
    for m in meta.modalities:
        stack.extend(_state_deps(composite.A_deps[m], table.num_states))
    emitted = dict(composite.emissions)
    for e in meta.emissions:
        stack.extend(
            table.state_names.index(p) for p in emitted[e].parents if p in table.state_names
        )
    mine = {*(table.state_names[f] for f in meta.states)}
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


def _driven_nodes(composite: System, name: str) -> list[int]:
    """The control nodes of component ``name`` that a variable drives: those with no action
    node, by node index (``System.observes`` names the driving variable of each)."""
    prefix = f"{name}{SEPARATOR}"
    return [
        k
        for k, (node, action) in enumerate(
            zip(composite._table.control_names, composite.action_names, strict=True)
        )
        if action is None and node.startswith(prefix)
    ]


def driven_nodes(composite: System, name: str | None = None) -> dict[str, str]:
    """The control nodes of component ``name`` (every component's when ``None``) that a variable
    drives, each with its variable: an action node or an emission another slot generates
    (``System.observes``). A driven node has no action node of its own and no planner chooses it;
    :func:`subsystem` takes its level as an input keyed by the variable, and
    :func:`~hybrid_aif.composition.joint.joint_planner` collapses it into its driver's node where
    the driver lies in the same node."""
    prefix = "" if name is None else f"{name}{SEPARATOR}"
    observed = dict(composite.observes)
    return {
        node: observed[node]
        for node, action in zip(composite._table.control_names, composite.action_names, strict=True)
        if action is None and node.startswith(prefix) and node in observed
    }


def _input_variable(composite: System, k: int) -> str:
    """The variable a control node's level is: its driver's action node (or emission) where
    ``System.observes`` names one, else the node's own name."""
    node = composite._table.control_names[k]
    return dict(composite.observes).get(node, node)


def _key_of_input(composite: System, k: int) -> str:
    """The key ``inputs`` takes for control node ``k``: its action node, or the variable that
    drives it when it has none."""
    action = composite.action_names[k]
    return _input_variable(composite, k) if action is None else action


def _view_nodes(
    composite: System, meta: Subsystem, facs: tuple[int, ...], name: str, sliced: set[int]
) -> tuple[list[int], tuple[str, ...], dict[str, int]]:
    """A view's control nodes by composite index, by name, and the nodes the view reads but does
    not hold (by name, to composite index): a node another component chooses that a factor reads
    or a carried target is conditional on, and a driven node of the component itself that
    ``sliced`` names, which the view takes at a level."""
    table = composite._table
    c_mods, c_facs, c_nodes = composite.modality_names, table.state_names, table.control_names
    chosen = set(meta.controls)
    mods = meta.modalities
    own = set(meta.states)
    #  a node a modality of the view observes joins the view as one of its own nodes, whoever
    #  chose it in the composite: the observation reads the node's realised action, so the view
    #  carries the node (a partner's, clamped per round by the caller's policy set)
    n_cols = len(composite.B_deps)
    sensed = {g - n_cols for m in mods for g in composite.A_deps[m] if g >= n_cols}
    driven = set(_driven_nodes(composite, name))
    ours = [
        k
        for k, node in enumerate(c_nodes)
        if node in chosen or k in sensed or (k in driven and k not in sliced)
    ]
    #  another component's node a factor of the view reads, by node name
    partner_nodes = {
        c_nodes[k]: k for f in facs for k, _ in table.control_parents[f] if k not in ours
    }
    partner_nodes.update({c_nodes[k]: k for k in sliced})
    #  a partner's node a carried target is conditional on is terminated like a factor's axis
    view_nodes = tuple(c_nodes[k] for k in ours)
    targeted = {c_facs[f] for f in own} | set(view_nodes)
    target_nodes = {
        p: c_nodes.index(p)
        for key, c_parents in composite.C_deps
        if key in targeted or key in {c_mods[m] for m in mods}
        for p in c_parents
        if p in c_nodes and p not in view_nodes
    }
    return ours, view_nodes, {**partner_nodes, **target_nodes}


def _inputs_of(composite: System, name: str) -> dict[str, list[int]]:
    """The variables the view of component ``name`` can be given a level of (``inputs=`` of
    :func:`subsystem`), each with the control nodes it is the level of: a control node another
    component chooses that the view reads, keyed by its action node, and a driven node of the
    component, keyed by the variable that drives it."""
    meta = _subsystem_meta(composite, name)
    facs = subsystem_states(composite, name)
    driven = set(_driven_nodes(composite, name))
    _, _, partners = _view_nodes(composite, meta, facs, name, driven)
    out: dict[str, list[int]] = {}
    for k in partners.values():
        out.setdefault(_key_of_input(composite, k), []).append(k)
    for k in driven:
        held = out.setdefault(_input_variable(composite, k), [])
        if k not in held:
            held.append(k)
    return out


def retake(
    composite: System,
    view: System,
    view_params: SystemParams,
    params: SystemParams,
) -> SystemParams:
    """``view_params`` with every parameter variable it holds taken again from ``params``, the
    composite's current home, and every reading position refilled: what :func:`subsystem` would
    return now for the same view, without rebuilding it. A view never slices a table that reads a
    variable (:func:`subsystem` refuses the pair), so only these entries depend on the home.
    Design: nesting-and-coupling.md N4."""
    homes = composite._params_in(params, "retake").parameters
    held = view._params_in(view_params, "retake")
    unknown = sorted(set(held.parameters) - set(homes))
    if unknown:
        raise ValueError(
            f"retake: the view holds parameter variable(s) {unknown}, which the composite does "
            f"not declare; it declares {sorted(homes)}"
        )
    out = dataclasses.replace(held, parameters={v: homes[v] for v in held.parameters})
    for slot, i, home in held.references:
        container = list(getattr(out, slot))
        container[i] = homes[home]
        out = dataclasses.replace(out, **{slot: tuple(container)})
    return view._params_out(out)


def subsystem(
    composite: System,
    params: SystemParams,
    name: str,
    inputs: Mapping[str, "int | Categorical | Array | None"] | None = None,
) -> tuple[System, SystemParams]:
    """One mind's bona-fide ``(System, SystemParams)``, derived from the composite and its metadata.

    The view holds:

    - its own state factors and modalities, under the composite's namespaced names, and the control
      nodes it still chooses (``Subsystem.controls``) with their action nodes;
    - boundary stubs (:func:`subsystem_states`) as ordinary latent state factors with their
      declared dynamics;
    - for every control node another component chooses that a state factor of the view reads (a
      stub's own node, or a node a same-step ``Link`` made one with ours), that state factor's axis
      marginalised under a prior over the node: uniform by default, or given in ``inputs`` keyed
      by the node's action node, read as in :class:`Clamp`; the node itself is absent, as is a
      node merged away;
    - for every control node of the component that a variable drives (``System.observes`` names
      a control node: ``Identify("mission.a_option", "rover.u_opt")``), the node as a control node
      with no action node (an input, driven from outside) and every state factor reading it; given
      an ``inputs`` entry for its driving variable, the node is sliced out at that level (an
      ``int``), averaged under a distribution (a :class:`~hybrid_aif.distributions.Categorical` or a
      probability vector), or averaged uniformly (``None``), as a partner's node is. A driven
      continuous node is never sliced;
    - for every carried per-step target conditional on such a node (``C_deps``), that parent axis
      contracted under the node's entry in ``inputs`` (required): a level slices it,
      a distribution mixes a categorical target's preferences in probability space, as a soft
      step does, and a Gaussian target takes a level;
    - nothing for an out-bound observation link;
    - the recognition entries and external inputs of the component's own, under the composite's
      names, with their modules shared by reference.

    Parameters are the composite's arrays, shared by reference; beliefs are private (they live in
    the caller's carry). Other components' designations and ``epistemic`` keys naming absent
    variables are dropped. A leaf takes its planner configuration (``Subsystem.config``) and its
    habit, precision, durations and scoring (``SystemParams.parts``) from its own record; the
    habit is carried only when the view holds all of the leaf's control nodes, and raises
    otherwise. A leaf that shares no latent with another gets its own System unsliced; recognition
    and scoring come with it. A view that carries stubs of an amortized composite is refused
    (a stub would cut a recognition conditioning set).

    A view is one component's side of one particle: the neighbours it models are stubs, not other
    minds' beliefs. A modality that observes an emission or an action node of this component
    (``System.observes``) keeps its entry; one that observes another component's (a belief
    :class:`Link`) is left an ordinary observation, which
    :func:`~hybrid_aif.composition.runner.run` supplies each round from the emission the source
    component's step realised.

    Design: docs/design/nesting-systems.md (architecture sketch; partner modeling);
    nesting-and-coupling.md N1, N2.
    """
    params = composite._params_in(params, "subsystem")
    meta = _subsystem_meta(composite, name)
    facs = subsystem_states(composite, name)
    mods = meta.modalities
    own = set(meta.states)
    if composite._is_amortized(params) and any(f not in own for f in facs):
        raise NotImplementedError(
            f"subsystem({name!r}) of an amortized composite with stubs: the view would carry "
            "factors of other components, and a stub would have to slice the recognition "
            "declaration too (which q-side factors this mind owns, and what a conditioning set "
            "that reached across the old boundary becomes), a composition question about "
            "inference structure, not a projection of the generative one (design: "
            "amortization.md A11, nesting-systems.md req. 7). A leaf that shares no latent "
            "with another has no stubs and gets its own recognition."
        )
    index = {f: i for i, f in enumerate(facs)}
    mod_index = {m: i for i, m in enumerate(mods)}
    table = composite._table
    c_mods, c_facs, c_nodes = composite.modality_names, table.state_names, table.control_names
    given = dict(inputs or {})
    valid = _inputs_of(composite, name)
    unknown = sorted(set(given) - set(valid))
    gone = dict(composite._bound_away())
    for key in unknown:
        if key in gone:
            raise ValueError(f"inputs names {key!r}, {gone[key]}")
    if unknown:
        raise ValueError(
            f"inputs names {unknown}, which are no variable the view of {name!r} takes a level "
            "of: an input is the action node of a control node another component chooses that "
            "this view reads, or the variable that drives a control node of the component; the "
            f"view takes {sorted(valid)}"
        )
    driven = set(_driven_nodes(composite, name))
    sliced = {k for variable in given for k in valid[variable] if k in driven}
    ported = dict(composite.black_box)
    opaque = sorted(c_nodes[k] for k in sliced if c_nodes[k] in ported)
    if opaque:
        raise ValueError(
            f"inputs names the variable driving {opaque}, in-port(s) of a black box: a sliced "
            "view takes a level of a node out of the tables that read it, and nothing inside a "
            "black box is stated; take its view unsliced (a custom step without the sliced mark)"
        )
    priors = {c_nodes[k]: given[variable] for variable in given for k in valid[variable]}
    ours, view_nodes, partner_nodes = _view_nodes(composite, meta, facs, name, sliced)
    n_cols = len(composite.B_deps)
    targeted = {c_facs[f] for f in own} | set(view_nodes)
    levels = composite._control_levels(params)
    space = composite._space(params)
    continuous = [c_nodes[k] for k in sliced if space.kinds[k] == FLOAT]
    if continuous:
        raise NotImplementedError(
            f"inputs names the variable of the continuous driven node(s) {continuous}: a prior "
            "over a continuous node would have to enter its readers as a continuous parent "
            "distribution, which no transition family accepts, so a driven continuous node is "
            "never sliced; leave it out of inputs and the view keeps it as an input"
        )

    def partner_prior(k: int, decl: object) -> tuple[object, str]:
        """The declared prior over node ``k`` (uniform by default or given ``None``), and its
        label; a uniform prior is in the float dtype of ``decl``, the declaration it averages."""
        key = c_nodes[k]
        label = f"inputs[{_key_of_input(composite, k)!r}]"
        if priors.get(key) is not None:
            return priors[key], label
        uniform = jnp.full(levels[k], 1.0 / levels[k], tree_float_dtype(decl))
        return uniform, label

    transition: list[object] = []
    view_parents: list[list[tuple[str, int]]] = []
    held = {(slot, i): home for slot, i, home in params.references}
    for f in facs:
        reads = table.control_parents[f]
        cuts = [(pos, k) for k, pos in reads if k not in ours]
        view_parents.append(
            [(kind, i) for kind, i in declared_parents(table, f) if kind == "state" or i in ours]
        )
        if not cuts:
            transition.append(params.transition[f])
            continue
        if ("transition", f) in held:
            raise NotImplementedError(
                f"factor {c_facs[f]!r} of the view of {name!r} reads the parameter variable "
                f"{held[('transition', f)]!r} and a control node another component chooses, so "
                "the view would average the shared table over that axis; declare the factor's "
                "own table where a view marginalises a partner's node"
            )
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
            prior, label = partner_prior(k, params.transition[f])
            terminations.append((pos, *_termination(prior, levels[k], label)))
        floats = [pos for k, pos in reads if space.kinds[k] == FLOAT]
        transition.append(
            _cut_axes(params.transition[f], c_facs[f], len(reads), terminations, floats)
        )

    #  the component's own emissions come with the view as its outputs; one that reads a control
    #  node another component chooses has nothing to read there (the view marginalises it away)
    view_emissions = []
    for e in meta.emissions:
        emit = dict(composite.emissions)[e]
        partner = [p for p in emit.parents if p in c_nodes and c_nodes.index(p) not in ours]
        if partner:
            raise NotImplementedError(
                f"the view of {name!r} carries the emission {e!r}, which reads the control node(s) "
                f"{partner} another component chooses: the view marginalises a partner's node away "
                "(a prior over its levels), so there is no realised level to draw the emission "
                "from; read a node this component chooses, or take the emission from the "
                "composite"
            )
        view_emissions.append((e, emit))

    n_view = len(facs)
    column = {k: n_view + j for j, k in enumerate(ours)}
    pad = ((),) * len(ours)

    def remap(deps: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(index[g] for g in deps)

    def remap_observed(deps: tuple[int, ...]) -> tuple[int, ...]:
        """A modality's parents: a state factor by its place in the view, an observed action
        node at ``len(view B_deps) + its place among the view's nodes``."""
        return tuple(
            index[g] if g < n_cols else n_view + len(ours) + ours.index(g - n_cols) for g in deps
        )

    D_all, H_all = composite._effective_D_deps(), composite._effective_H_deps()
    B_deps = tuple(
        tuple(index[i] if kind == "state" else column[i] for kind, i in ps) for ps in view_parents
    )
    D_deps = tuple(remap(D_all[f]) for f in facs)
    H_deps = tuple(remap(H_all[f]) if f in own else () for f in facs)
    A_deps = tuple(remap_observed(composite.A_deps[m]) for m in mods)

    view_facs = tuple(c_facs[f] for f in facs)
    view_mods = tuple(c_mods[m] for m in mods)
    view_actions = tuple(composite.action_names[k] for k in ours)
    own_tables = {c_facs[f] for f in own} | set(c_mods[m] for m in mods)
    view_readings = []
    for table, node in composite.readings:
        if table not in own_tables:
            continue  # a stub's reading is not this view's to score
        if node not in view_nodes:
            raise NotImplementedError(
                f"{table!r} of the view of {name!r} pools its reading over {node!r}, a control "
                "node another component chooses, so the view would score a count it cannot "
                "follow; declare the table's reading over a node this component chooses"
            )
        view_readings.append((table, node))
    #  a target on a node goes with its view while the view still chooses it
    contracted: dict[str, object] = {}
    C_deps = []
    for key, c_parents in composite.C_deps:
        if key not in targeted and key not in view_mods:
            continue
        absent = [p for p in c_parents if p in c_nodes and p not in view_nodes]
        if not absent:
            C_deps.append((key, c_parents))
            continue
        cuts = []
        for pos, p in enumerate(c_parents):
            if p not in absent:
                continue
            k = c_nodes.index(p)
            if p not in priors:
                variable = _key_of_input(composite, k)
                raise ValueError(
                    f"the view of {name!r} carries the target on {key!r}, which is conditional on "
                    f"{p!r}, another component's control node, and inputs has no entry "
                    f"for it; pass inputs={{{variable!r}: <level or probabilities>}} to "
                    "terminate that axis of the target"
                )
            prior, label = partner_prior(
                k, (params.obs_targets if key in view_mods else params.state_targets)[key]
            )
            cuts.append((pos, *_termination(prior, levels[k], label)))
        scope = params.obs_targets if key in view_mods else params.state_targets
        contracted[key] = _contract_target(scope[key], key, cuts)
        kept = tuple(p for p in c_parents if p not in absent)
        if kept:
            C_deps.append((key, kept))
    #  an observation of a variable this component generates (its own emission, or an action
    #  node it chooses) stays on the view; an observation of a sibling's emission or action node
    #  leaves the modality an ordinary observation, which the driver supplies
    view_observes = []
    held_emissions = {e for e, _ in view_emissions}
    for slot, variable in composite.observes:
        if slot not in set(view_mods):
            continue
        if variable in held_emissions or variable in set(view_actions):
            view_observes.append((slot, variable))
    prefix = f"{name}{SEPARATOR}"
    recognition = tuple(e for e in composite.recognition or () if e.variable.startswith(prefix))
    external = tuple((n, w) for n, w in composite.external_inputs if n.startswith(prefix))
    view_reads = [
        (slot, j, held[(slot, i)])
        for slot, items in (("transition", facs), ("likelihood", mods))
        for j, i in enumerate(items)
        if (slot, i) in held
    ]
    noise_held = {(slot, i): home for slot, i, home in params.noise_references}
    view_noise = [
        (slot, j, noise_held[(slot, i)])
        for slot, items in (("transition", facs), ("likelihood", mods))
        for j, i in enumerate(items)
        if (slot, i) in noise_held
    ]
    emission_homes = {
        params.emission[e].name
        for e, _ in view_emissions
        if e in params.emission and isinstance(params.emission[e], Parameter)
    }
    leaf = dict(meta.config) if meta.config else None
    part = params.parts.get(name) if leaf is not None else None
    #  the composite's own fields are authoritative where it states them; a coupling states none of
    #  its leaves' habit, precision or durations and an `UNSET` planner field, and the leaf's
    #  record speaks there
    if params.policy_prior is not None or part is None:
        habit = _view_policy_prior(composite, params, ours, name)
    else:
        habit = _leaf_habit(composite, params, part, ours, name)
    precision = params.precision if params.precision is not None or part is None else part.precision
    held = params.durations if params.durations or part is None else part.durations
    view_durations = {k: v for k, v in held.items() if k in set(view_nodes)}
    scoring_params: dict[str, object] = {} if part is None else part.scoring
    if part is None and leaf is not None:
        #  a leaf of a joint-planned composite: its estimators' parameters are the composite's
        #  composed scoring, under the rows the leaf's record owns
        mine = {n for n, _ in leaf["scoring"]} | {t.partition(":")[0] for t in leaf["terms"]}
        scoring_params = {k: v for k, v in params.scoring.items() if k in mine}

    def configured(field: str):
        """A field of the view: the composite's, the leaf's own where the composite's is UNSET."""
        value = getattr(composite, field)
        return leaf[field] if leaf is not None and value is UNSET else value

    view_params = SystemParams(
        parameters={
            home: params.parameters[home]
            for home in dict.fromkeys(
                [
                    *(h for _, _, h in view_reads),
                    *(h for _, _, h in view_noise),
                    *sorted(emission_homes),
                ]
            )
        },
        emission={e: params.emission[e] for e, _ in view_emissions if e in params.emission},
        references=tuple(sorted(view_reads)),
        noise_references=tuple(sorted(view_noise)),
        recognition={e.variable: params.recognition[e.variable] for e in recognition},
        transition=tuple(transition),
        initial_state_prior=tuple(params.initial_state_prior[f] for f in facs),
        likelihood=tuple(params.likelihood[m] for m in mods),
        obs_targets={
            k: contracted.get(k, v) for k, v in params.obs_targets.items() if k in set(view_mods)
        },
        state_targets={
            k: contracted.get(k, v) for k, v in params.state_targets.items() if k in targeted
        },
        terminal_targets={k: v for k, v in params.terminal_targets.items() if k in targeted},
        policy_prior=habit,
        scoring=scoring_params,
        control={k: v for k, v in params.control.items() if k in set(view_nodes)},
        actuator={k: v for k, v in params.actuator.items() if k in set(view_actions)},
        durations=view_durations,
        readings={t: params.readings[t] for t, _ in view_readings},
        precision=precision,
    )
    view = System(
        A_deps=A_deps,
        B_deps=(*B_deps, *pad),
        controls=tuple(column[k] for k in ours),
        action_names=view_actions,
        policy_len=configured("policy_len"),
        planner=configured("planner"),
        gamma=configured("gamma"),
        num_iters=configured("num_iters"),
        decomposition=configured("decomposition"),
        planning_belief=configured("planning_belief"),
        feedback=configured("feedback"),
        epistemic=False,
        mixture_k=configured("mixture_k"),
        mixture_schedule=configured("mixture_schedule"),
        scoring=leaf["scoring"] if leaf is not None else (),
        D_deps=None if all(not d for d in D_deps) else (*D_deps, *pad),
        H_deps=None if all(not h for h in H_deps) else (*H_deps, *pad),
        C_deps=tuple(C_deps),
        recognition=recognition or None,
        external_inputs=external,
        modality_names=view_mods,
        state_names=tuple(view_facs),
        control_names=tuple(view_nodes),
        dims=tuple((factor, names) for factor, names in composite.dims if factor in view_facs),
        cells=tuple((factor, grid) for factor, grid in composite.cells if factor in view_facs),
        readings=tuple(view_readings),
        observes=tuple(view_observes),
        emissions=tuple(view_emissions),
        black_box=tuple(
            (port, t)
            for port, t in composite.black_box
            if port in set(view_nodes) or port in {e for e, _ in view_emissions}
        ),
        subsystems=_view_metadata(
            composite,
            name,
            index,
            mod_index,
            set(view_nodes),
            A_deps,
            {e for e, _ in view_emissions},
        ),
    )
    mine = {c_facs[f] for f in own} | set(view_mods)
    epistemic = _view_epistemic(composite, view, view_params, mine)
    terms = _view_terms(composite, mine | set(view_nodes))
    if leaf is not None and meta.boundary == "coupled":
        epistemic = leaf["epistemic"]
    owned = (
        {name for name, _ in leaf["scoring"]} | set(scoring_params) if leaf is not None else set()
    )
    if owned:
        #  a term the leaf's scoring owns is the leaf's own, in the leaf's order
        kept = set(terms)
        terms = tuple(
            dict.fromkeys(
                [t for t in leaf["terms"] if t in kept or t.partition(":")[0] in owned] + [*terms]
            )
        )
    view = dataclasses.replace(
        view,
        epistemic=epistemic,
        terms=terms,
        plan_learning=_view_plan_learning(composite, view, view_params),
    )
    view._validate(view_params)
    return view, view._params_out(view_params)


def _leaf_habit(
    composite: System, params: SystemParams, part: PartParams, ours: list[int], name: str
) -> "Array | Categorical | None":
    """The leaf's own habit prior for its view: carried when the view holds all of the leaf's
    control nodes, over the leaf's own policy set, and raises otherwise (a habit is one weight per
    policy of its own set, which a view that slices or drops a node does not have)."""
    habit = part.policy_prior
    if habit is None:
        return None
    if not hasattr(habit, "shape") and not isinstance(habit, Categorical):
        raise NotImplementedError(
            f"the leaf {name!r} declares a policy module in policy_prior, which its view cannot "
            "carry: a module reads *beliefs*, and the view's beliefs are not the leaf's own "
            "carry. Declare the view's own actor on the returned SystemParams."
        )
    c_nodes = composite._table.control_names
    prefix = f"{name}{SEPARATOR}"
    own = [k for k, node in enumerate(c_nodes) if node.startswith(prefix)]
    if list(ours) != own:
        raise NotImplementedError(
            f"the leaf {name!r} declares policy_prior over its own policy set, and its view holds "
            f"only part of its control nodes ({[c_nodes[k] for k in ours]} of "
            f"{[c_nodes[k] for k in own]}), so that policy set does not exist in the view. "
            "Declare the view's own habit prior on the returned SystemParams "
            "(dataclasses.replace(view_params, policy_prior=...)), or drop the leaf's."
        )
    levels = composite._control_levels(params)
    lp = habit.log_probs if isinstance(habit, Categorical) else jnp.asarray(habit)
    expected = 1
    for k in ours:
        expected *= levels[k]
    expected **= dict(_subsystem_meta(composite, name).config)["policy_len"]
    if lp.shape != (expected,):
        raise ValueError(
            f"the leaf {name!r} has policy_prior of shape {tuple(lp.shape)}, but its view's "
            f"policy set has {expected} member(s); a binding that removed one of its control "
            "nodes changed the set, so drop the habit prior"
        )
    return habit


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
    emitted: set[str],
) -> tuple[tuple[str, Subsystem], ...]:
    """This component's record (and its inner composite's components') re-expressed in the view's
    frame.

    A stub belongs to no entry, which is what marks the own/stub boundary.
    """
    out: list[tuple[str, Subsystem]] = []
    for other, sub in composite.subsystems or ():
        if other != name and not other.startswith(f"{name}{SEPARATOR}"):
            continue
        if not (set(sub.states) <= set(index) and set(sub.modalities) <= set(mod_index)):
            continue
        facs = tuple(index[f] for f in sub.states)
        mods = tuple(mod_index[m] for m in sub.modalities)
        out.append(
            (
                other,
                sub._replace(
                    states=facs,
                    modalities=mods,
                    controls=tuple(n for n in sub.controls if n in chosen),
                    emissions=tuple(e for e in sub.emissions if e in emitted),
                    free_modalities=tuple(
                        m
                        for m in mods
                        if all(g in facs for g in _state_deps(A_deps[m], len(index)))
                    ),
                    bound_modalities=tuple(
                        (m, tuple(g for g in _state_deps(A_deps[m], len(index)) if g not in facs))
                        for m in mods
                        if any(g not in facs for g in _state_deps(A_deps[m], len(index)))
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


def _view_plan_learning(
    composite: System, view: System, view_params: SystemParams
) -> "bool | tuple[str, ...]":
    """Restrict the composite's ``plan_learning`` to the tables this mind reads: a key whose
    reader (or parameter variable) the view lacks is dropped; ``True`` and ``()`` pass through.
    Design: parameters-within-the-plan.md D1."""
    if isinstance(composite.plan_learning, bool):
        return composite.plan_learning
    held = {"transition": set(view._facs), "likelihood": set(view.modality_names)}
    held["parameter"] = set(view_params.parameters)
    return tuple(
        key
        for key in composite.plan_learning
        if key.partition(":")[2] in held.get(key.partition(":")[0], set())
    )


def _view_terms(composite: System, held: set[str]) -> tuple[str, ...]:
    """The composite's ``terms`` selectors whose variable segment the view holds.

    A selector on a row that takes no target carries over unchanged.
    """
    out = []
    for key in composite.terms:
        row, sep, rest = key.partition(":")
        if not sep or registry.get_row(row).target is None or _key_name(rest) in held:
            out.append(key)
    return tuple(out)


def _key_name(key: str) -> str:
    """The name segment of one key (the inverse of :func:`namespaced_key`'s qualification)."""
    parts = key.split(":")
    return parts[1] if len(parts) > 1 and parts[0] in _KEY_KINDS else parts[0]
