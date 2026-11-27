"""``joint_planner()``: one node of a merged or coupled System planned as one mind.

:func:`~hybrid_aif.composition.merge.merge` and :func:`~hybrid_aif.composition.couple.couple`
state no planning problem of their own: each leaf keeps its planner configuration
(``Subsystem.config``) and its habit, precision, durations and scoring parameters
(``SystemParams.parts``). :func:`joint_planner` builds the one planning problem of a node, the root
(``node=None``) or a nested component, as a plain ``(System, SystemParams)`` whose faces plan every
control node of the node together:

- **control identifications collapse.** A control node driven by an action node inside the node
  (``Identify(driver action node, control node)``) leaves the node's System: every factor that
  read it reads the driver's control node, so one mind plans one decision. Driven nodes whose
  driver lies outside the node stay driven inputs. An emission identified with a modality stays
  an emission and an observing slot: the joint model reads its own belief.
- **the configuration** of each field is the argument of :func:`joint_planner`, else the composite's
  where it states one, else the value the node's leaves agree on; leaves that differ are refused,
  naming the field and each leaf's value.
- **the habit** is the sum, in log space, of the leaves' habits over the joint policy set; **the
  precision** is the one value the leaves declare; **the durations** are the leaves' one clock;
  **the scoring** is the one leaf's that declares it. What cannot be joined is refused with a
  sentence.
- **recognition** is carried per leaf under the leaf's names.

A node that is itself a leaf is its own planner and is refused. A leaf the caller steps as a
process (``sample()``) is planned here as a self-model: a refusal needs the caller's ``steps``,
which :func:`joint_planner` does not see.

Design: docs/design/engine-consolidation.md; docs/composing-systems.md.
"""

import dataclasses
from collections.abc import Mapping, Sequence

import jax.numpy as jnp
from jax import Array

from hybrid_aif.composition.merge import (
    _CARRY_FIELDS,
    _PLANNING_FIELDS,
    SEPARATOR,
    subsystem,
)
from hybrid_aif.distributions import Categorical
from hybrid_aif.dtypes import as_float
from hybrid_aif.planning.enumerate import enumerate_policies
from hybrid_aif.reads import Emission
from hybrid_aif.system import UNSET, PartParams, Subsystem, System, SystemParams

_OVERRIDES = (*_PLANNING_FIELDS, *_CARRY_FIELDS, "epistemic", "terms")


def joint_planner(
    system: System,
    params: SystemParams,
    node: str | None = None,
    *,
    inputs: Mapping[str, "int | Categorical | Array | None"] | None = None,
    **config,
) -> tuple[System, SystemParams]:
    """The planning problem of ``node`` of a merged or coupled System, as one mind.

    ``node`` is the name of a component that is not a leaf (a merged component, or a component of
    a coupling that is itself composed); ``None`` is the whole System. The result is a
    ``(System, SystemParams)`` of the node's slots and variables whose faces plan every control
    node of the node jointly; the module docstring describes what it holds.

    ``config`` overrides the configuration: any of ``policy_len``, ``planner``, ``gamma``,
    ``num_iters``, ``decomposition``, ``planning_belief``, ``feedback``, ``mixture_schedule``,
    ``mixture_k``, ``epistemic`` and ``terms``. A field not given is the composite's value where it
    states one (the whole System's own; a nested node is the composite being planned, and a field
    stated on an enclosing composite does not reach into it), else the value every leaf of the
    node declares; leaves that differ are refused. ``UNSET`` is refused as an argument.
    The composite's own habit, precision, durations and scoring, where it states them, are
    authoritative; otherwise they are composed from the leaves' records.

    ``inputs`` as in :func:`~hybrid_aif.composition.merge.subsystem`: for a nested node, the
    level at which a control node driven from outside the node is held (a driven node whose
    driver is inside the node is collapsed and takes no input).

    Refused with a sentence: a System that is not composed, a node that is a leaf, leaves that
    differ on a configuration field, scoring declared by two leaves, durations on two nodes (a
    round has one clock) or on a node a collapse removes, precision declared by some leaves and
    not others or with different values, a habit that does not join (a policy module, a
    continuous node, a leaf whose own policy set the joint view does not have), an actuator
    kernel on the action node of a collapsed pair, and a collapse that makes a state factor read one
    node twice.
    """
    unset = sorted(k for k, v in config.items() if v is UNSET)
    if unset:
        raise ValueError(
            f"joint_planner: {unset} is UNSET, which states no value; give a value or leave it out "
            "so the composite's, or the leaves' agreed, value is used"
        )
    unknown = sorted(set(config) - set(_OVERRIDES))
    if unknown:
        raise TypeError(
            f"joint_planner got unexpected configuration {unknown}; it overrides {list(_OVERRIDES)}"
        )
    system._params_in(params, "joint_planner")  # the keyed form is required, as of every face
    leaves = _leaves_of(system, node)
    scope = f"{node}{SEPARATOR}" if node is not None else ""
    fields = _configuration(system, node, leaves, config)
    parts = params.parts
    composite = _own_boundaries(dataclasses.replace(system, **fields))
    held = dataclasses.replace(params, parts={})

    collapsed = _collapsed(composite, scope)
    if collapsed:
        composite, held = _collapse(composite, held, collapsed)
    if node is not None:
        composite, held = subsystem(composite, held, node, inputs=inputs)
    elif inputs:
        raise ValueError(
            "joint_planner: inputs holds the levels of control nodes driven from outside a "
            "node, and the whole System has no outside; name a nested node"
        )
    selectors = {k: config[k] for k in ("epistemic", "terms") if k in config}
    composite = dataclasses.replace(composite, **selectors)
    composite, held = _compose(system, params, parts, composite, held, leaves, collapsed)
    positional = composite._p(held)
    composite._validate(positional)
    return composite, composite._params_out(positional)


# ----------------------------------------------------------------------------- the node


def _leaves_of(system: System, node: str | None) -> list[tuple[str, Subsystem]]:
    """The leaf records under ``node``, in splice order; a System that is not composed, an
    unknown node and a leaf are refused."""
    if system.subsystems is None:
        raise ValueError(
            "joint_planner: this System has no components (it is not a merge or a coupling), so "
            "there is no node to plan as one mind; it is its own planner"
        )
    table = dict(system.subsystems)
    if node is not None:
        if node not in table:
            raise ValueError(
                f"joint_planner: unknown node {node!r}; the components are {sorted(table)}"
            )
        if table[node].config:
            raise ValueError(
                f"joint_planner: {node!r} is a leaf, which is its own planner: use the leaf "
                "itself, or subsystem(system, params, leaf) for its view"
            )
    prefix = "" if node is None else f"{node}{SEPARATOR}"
    leaves = [(n, sub) for n, sub in system.subsystems if sub.config and n.startswith(prefix)]
    if not leaves:
        raise ValueError(f"joint_planner: {node!r} holds no leaf to plan")
    boxes = [n for n, _ in leaves if n in system._black_boxes]
    if boxes:
        raise ValueError(
            f"joint_planner: {node or 'the System'!r} holds the black box {boxes[0]!r}, an "
            "external process declared by its ports alone, and one mind's model needs every "
            "leaf's inside; plan a node that leaves the black box out (the components it is "
            "coupled with), or declare the process as a System of factors and kernels"
        )
    return leaves


def _configuration(
    system: System,
    node: str | None,
    leaves: Sequence[tuple[str, Subsystem]],
    config: Mapping[str, object],
) -> dict[str, object]:
    """Each configuration field of the joint planner: the argument, else the composite's where it
    states one (for the whole System), else the one value the node's leaves declare."""
    out: dict[str, object] = {}
    for field in (*_PLANNING_FIELDS, *_CARRY_FIELDS):
        if field in config:
            out[field] = config[field]
            continue
        #  the composite is the node being planned: a field stated on an enclosing composite does
        #  not reach into a nested node
        stated = getattr(system, field) if node is None else UNSET
        if stated is not UNSET:
            out[field] = stated
            continue
        values = {name: dict(sub.config)[field] for name, sub in leaves}
        if len(set(values.values())) > 1:
            raise ValueError(
                f"joint_planner: the leaves of {node or 'the System'!r} disagree on {field}: "
                f"{values}. One mind plans with one {field}: pass {field}=... to joint_planner, "
                "or declare the leaves alike"
            )
        out[field] = next(iter(values.values()))
    return out


# ------------------------------------------------------------------------ the collapse


def _collapsed(system: System, scope: str) -> dict[str, str]:
    """The driven control nodes under ``scope`` whose driver is an action node under ``scope``,
    each with the control node of that action node: ``{driven node: driver's node}``."""
    table = system._table
    node_of = {
        a: n for n, a in zip(table.control_names, system.action_names, strict=True) if a is not None
    }
    observed = dict(system.observes)
    out = {}
    for node, action in zip(table.control_names, system.action_names, strict=True):
        variable = observed.get(node)
        if action is not None or variable is None or variable not in node_of:
            continue
        if node.startswith(scope) and variable.startswith(scope):
            out[node] = node_of[variable]
    return out


def _identity(kernel) -> bool:
    """Whether an actuator kernel is the identity: the realised value is the decision."""
    kernel = jnp.asarray(kernel)
    return (
        kernel.ndim == 2
        and kernel.shape[0] == kernel.shape[1]
        and bool(jnp.array_equal(kernel, jnp.eye(kernel.shape[0], dtype=kernel.dtype)))
    )


def _collapse(
    system: System, params: SystemParams, collapsed: Mapping[str, str]
) -> tuple[System, SystemParams]:
    """The System with each driven node in ``collapsed`` removed: every reader of it reads the
    driver's control node, and every name that referred to it follows."""
    table = system._table
    n_states = table.num_states
    names = list(table.control_names)
    n = len(names)
    if tuple(system.controls) != tuple(range(n_states, n_states + n)):
        raise ValueError(
            "joint_planner: this System's control nodes are not the trailing nodes of its "
            "factor table, so it is not the result of a merge or a coupling"
        )
    kernelled = sorted(
        node
        for node, driver in collapsed.items()
        for a in [system.action_names[names.index(driver)]]
        if params.actuator.get(a) is not None and not _identity(params.actuator[a])
    )
    if kernelled:
        raise ValueError(
            f"joint_planner: control node(s) {kernelled} are driven by an action node with a "
            "params.actuator kernel that is not the identity: one mind's intention and the value "
            "it plans on would be two, and a collapsed pair has one decision; drop the kernel, or "
            "plan the leaves separately"
        )
    for slot, targets in (
        ("state_targets", params.state_targets),
        ("terminal_targets", params.terminal_targets),
    ):
        for key in targets:
            if key in collapsed:
                raise ValueError(
                    f"joint_planner: {slot}[{key!r}] is a target on a control node driven by "
                    f"{collapsed[key]!r}'s action node, which the joint planner makes one node "
                    f"with {collapsed[key]!r}: declare the target on {collapsed[key]!r}"
                )
    removed = [node for node in params.durations if node in collapsed]
    if removed:
        raise ValueError(
            f"joint_planner: a binding removes control node {removed[0]!r}, which carries the "
            "durations; declare the durations on a control node that stays "
            "(design: step-durations.md T1)"
        )
    live = [k for k, node in enumerate(names) if node not in collapsed]
    place = {k: j for j, k in enumerate(live)}
    column = {n_states + k: n_states + place[k] for k in live}
    for node, driver in collapsed.items():
        column[n_states + names.index(node)] = column[n_states + names.index(driver)]
    n_live = len(live)

    B_deps = []
    for f in range(n_states):
        deps = tuple(column[g] if g >= n_states else g for g in system.B_deps[f])
        controls = [g for g in deps if g >= n_states]
        if len(set(controls)) != len(controls):
            both = [names[g - n_states] for g in system.B_deps[f] if g >= n_states]
            raise ValueError(
                f"joint_planner: factor {table.state_names[f]!r} reads both {both}, which the "
                "collapse makes one node, so it would read one node twice; bind a node that no "
                "reader of the driver reads"
            )
        B_deps.append(deps)
    pad = ((),) * n_live

    def repad(deps):
        return None if deps is None else (*deps[:n_states], *pad)

    n_columns = n_states + n
    A_deps = tuple(
        tuple(g if g < n_states else n_states + n_live + place[g - n_columns] for g in deps)
        for deps in system.A_deps
    )
    C_deps = tuple(
        (key, tuple(collapsed.get(p, p) for p in parents)) for key, parents in system.C_deps
    )
    readings = tuple(
        (table_name, collapsed.get(node, node)) for table_name, node in system.readings
    )
    emissions = []
    for name, emit in system.emissions:
        parents = tuple(collapsed.get(p, p) for p in emit.parents)
        if len(set(parents)) != len(parents):
            raise ValueError(
                f"joint_planner: emission {name!r} reads one control node twice after the "
                "collapse; its parents must stay distinct"
            )
        emissions.append((name, Emission(parents, emit.realise, emit.kernel)))
    observes = tuple(
        (slot, variable) for slot, variable in system.observes if slot not in collapsed
    )
    out = dataclasses.replace(
        system,
        B_deps=(*B_deps, *pad),
        A_deps=A_deps,
        controls=tuple(n_states + j for j in range(n_live)),
        action_names=tuple(system.action_names[k] for k in live),
        D_deps=repad(system.D_deps),
        H_deps=repad(system.H_deps),
        C_deps=C_deps,
        control_names=tuple(names[k] for k in live),
        readings=readings,
        observes=observes,
        emissions=tuple(emissions),
    )
    return out, dataclasses.replace(
        params, control={k: v for k, v in params.control.items() if k not in collapsed}
    )


def _own_boundaries(system: System) -> System:
    """A top-level component of a coupling is joined to the others by identified ports alone; in
    one mind it keeps a blanket of its own where a variable crosses its boundary, and none where
    nothing does."""
    if not any(sub.boundary == "coupled" for _, sub in system.subsystems or ()):
        return system
    observes = system.observes
    out = []
    for name, sub in system.subsystems:
        if sub.boundary == "coupled":
            prefix = f"{name}{SEPARATOR}"
            crossing = any(
                slot.startswith(prefix) != variable.startswith(prefix)
                for slot, variable in observes
            )
            sub = sub._replace(boundary="blanket" if crossing else "none")
        out.append((name, sub))
    return dataclasses.replace(system, subsystems=tuple(out))


# ----------------------------------------------------------------------- the composition


def _compose(
    original: System,
    original_params: SystemParams,
    parts: Mapping[str, PartParams],
    system: System,
    params: SystemParams,
    leaves: Sequence[tuple[str, Subsystem]],
    collapsed: Mapping[str, str],
) -> tuple[System, SystemParams]:
    """The habit, precision, durations and scoring of the joint planner, composed from the
    leaves' records wherever the composite (its view) states none of its own."""
    names = [name for name, _ in leaves]
    here = {name: parts[name] for name in names if name in parts}
    if not (system.scoring or params.scoring):
        system, params = _scoring(system, params, here, leaves)
    if params.policy_prior is None:
        params = dataclasses.replace(
            params, policy_prior=_habit(original, original_params, system, params, here, leaves)
        )
    if params.precision is None:
        params = dataclasses.replace(params, precision=_precision(here, len(names)))
    if not params.durations:
        params = dataclasses.replace(params, durations=_durations(system, here, collapsed))
    return system, params


def _habit(
    original: System,
    original_params: SystemParams,
    system: System,
    params: SystemParams,
    here: Mapping[str, PartParams],
    leaves: Sequence[tuple[str, Subsystem]],
) -> "Array | None":
    """The leaves' habits summed in log space over the joint policy set:
    ``log E(pi) = sum_leaf log E_leaf(pi restricted to the leaf's control nodes)``."""
    declared = {n: p.policy_prior for n, p in here.items() if p.policy_prior is not None}
    if not declared:
        return None
    modules = sorted(
        n
        for n, prior in declared.items()
        if not hasattr(prior, "shape") and not isinstance(prior, Categorical)
    )
    if modules:
        raise NotImplementedError(
            f"leaf(s) {modules} declare a *policy module* in policy_prior (a habit net), which "
            "does not compose as a sum over the joint policy set: a module is a conditional "
            "prior P(u|s) over its own beliefs, so composing two of them is a question about the "
            "joint control-state distribution of two minds, not a log-space addition. Declare "
            "the joint habit on the returned params, or plan each leaf on its own view."
        )
    if params.control:
        raise NotImplementedError(
            f"leaf(s) {sorted(declared)} declare policy_prior, but the joint control space is "
            "continuous: a habit prior over an enumerated policy set has no meaning there (CEM's "
            "E is its proposal distribution; design: amortization.md A9). Drop the leaves' "
            "policy_prior."
        )
    joint = list(system._table.control_names)
    chosen = {n for n, a in zip(joint, system.action_names, strict=True) if a is not None}
    levels = system._control_levels(params)
    policies = enumerate_policies(levels, system.policy_len, (True,) * len(levels))
    total = None
    names = list(original._table.control_names)
    own_levels = original._control_levels(original_params)
    config = dict(leaves)
    for leaf, habit in declared.items():
        mine = dict(config[leaf].config)
        own = [k for k, node in enumerate(names) if node.startswith(f"{leaf}{SEPARATOR}")]
        lost = [node for node, _ in config[leaf].actions if node not in chosen]
        if lost:
            raise ValueError(
                f"leaf {leaf!r} declares policy_prior over its own policy set, but its control "
                f"node(s) {lost} are not chosen by the joint planner (a binding removed them, a "
                "collapse made them the driver's, or they are driven from outside), so that "
                "policy set does not exist here. Drop the leaf's policy_prior and declare the "
                "joint habit on the returned params."
            )
        if mine["policy_len"] != system.policy_len:
            raise ValueError(
                f"leaf {leaf!r} declares policy_prior over its own policy_len "
                f"{mine['policy_len']}, and the joint planner's is {system.policy_len}: a habit "
                "is one weight per policy of its own set; declare the joint habit on the "
                "returned params"
            )
        expected = 1
        for _ in range(mine["policy_len"]):
            for k in own:
                expected *= own_levels[k]
        lp = as_float(habit.log_probs if isinstance(habit, Categorical) else habit)
        if lp.shape != (expected,):
            raise ValueError(
                f"leaf {leaf!r} has policy_prior of shape {lp.shape}, but its own policy set has "
                f"{expected} member(s) (levels={[own_levels[k] for k in own]}, "
                f"policy_len={mine['policy_len']}); a habit prior is one weight per policy"
            )
        index = jnp.zeros((policies.shape[0],), jnp.int32)
        for t in range(mine["policy_len"]):
            for k in own:
                index = index * own_levels[k] + policies[:, t, joint.index(names[k])]
        if total is None:
            total = jnp.zeros((policies.shape[0],), lp.dtype)
        total = total + lp[index]
    return total


def _precision(here: Mapping[str, PartParams], count: int) -> "Array | None":
    """The one policy precision the leaves declare: all of them with one value, or none."""
    declared = {n: p.precision for n, p in here.items() if p.precision is not None}
    if declared and len(declared) != count:
        raise ValueError(
            f"leaves {sorted(declared)} declare params.precision and the others do not: the joint "
            "planner plans with one policy precision, so declare it on every leaf with the same "
            "value, or on none and declare the joint planner's on the returned params"
        )
    if len({float(v) for v in declared.values()}) > 1:
        raise ValueError(
            f"leaves declare different params.precision values "
            f"{ {n: float(v) for n, v in declared.items()} }: the joint planner plans with one "
            "policy precision, so declare one value"
        )
    return next(iter(declared.values()), None)


def _durations(
    system: System, here: Mapping[str, PartParams], collapsed: Mapping[str, str]
) -> dict[str, Array]:
    """The leaves' durations, which are one clock: at most one node, and one the joint planner
    keeps."""
    clocks = {node: v for p in here.values() for node, v in p.durations.items()}
    if len(clocks) > 1:
        raise ValueError(
            f"leaves declare durations on {sorted(clocks)}: a round steps every factor once, so "
            "the joint planner has one clock; declare durations on one leaf's control node "
            "(design: step-durations.md T1)"
        )
    kept = {n for n, a in zip(system._table.control_names, system.action_names, strict=True) if a}
    for node in clocks:
        if node in collapsed or node not in kept:
            raise ValueError(
                f"a binding removes control node {node!r}, which carries the durations; declare "
                "the durations on a control node the joint planner chooses "
                "(design: step-durations.md T1)"
            )
    return clocks


def _scoring(
    system: System,
    params: SystemParams,
    here: Mapping[str, PartParams],
    leaves: Sequence[tuple[str, Subsystem]],
) -> tuple[System, SystemParams]:
    """The scoring of the one leaf that declares it, its terms back among the planner's."""
    scored = [
        (name, sub)
        for name, sub in leaves
        if dict(sub.config)["scoring"] or (name in here and here[name].scoring)
    ]
    if len(scored) > 1:
        raise NotImplementedError(
            f"leaves {[n for n, _ in scored]} declare scoring structure or parameters "
            "(System.scoring / SystemParams.scoring), whose composition into one planning "
            "problem is not built: an estimator that owns a model or parameters answers to its "
            "name alone, so two leaves' critics would both answer to it with no way to say which. "
            "Plan each leaf on its own view, or declare the joint planner's scoring on the "
            "returned pair (design: amortization.md, the A12 implementation note on the terms= "
            "key grammar, flags 4 and 5)"
        )
    if not scored:
        return system, params
    ((name, sub),) = scored
    config = dict(sub.config)
    part = here.get(name, PartParams())
    owned = {n for n, _ in config["scoring"]} | set(part.scoring)
    terms = tuple(
        dict.fromkeys(
            [*system.terms, *(t for t in config["terms"] if t.partition(":")[0] in owned)]
        )
    )
    return (
        dataclasses.replace(system, scoring=config["scoring"], terms=terms),
        dataclasses.replace(params, scoring=part.scoring),
    )
