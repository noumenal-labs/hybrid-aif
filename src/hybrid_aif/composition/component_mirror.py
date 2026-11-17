"""The mirror of a merged or coupled System, one leaf per component.

:func:`~hybrid_aif.system.mirror.mirror` takes a System whole: one world that steps every state
factor in one process. :func:`mirror_components` takes a System built from components
(:func:`~hybrid_aif.composition.merge.merge`, :func:`~hybrid_aif.composition.couple.couple`) and
returns the **merge** of one mirror per component, each a leaf named as the component is, so the
names of the System (``a.hand``, ``arena.lock_b``) are the names of the world and a component is
stepped on its own by :func:`~hybrid_aif.composition.runner.run`. A merge, so a parameter
variable that several components read stays one variable.

For a component ``c``:

- its own state factors, transitions, priors, control nodes (with their action nodes) and
  modalities are mirrored as :func:`~hybrid_aif.system.mirror.mirror` mirrors a System: every
  modality of ``c`` becomes an emission of ``c`` of the same name, the likelihood its kernel. The
  mirror carries no durations: a world is stepped every round;
- a **foreign state parent** (a state factor of another component) of one of ``c``'s transitions
  becomes a **driven control node** of ``c``, named ``u_<owner>_<factor>``. In the transition's
  table the parent's axis moves behind the other parent axes, and the node is read where the
  parent was. The owner of the factor gains the emission ``<factor>_read``, ``Emission((factor,),
  "mode", "identity")``: the mode of a point is the point, so the value is exact and draws
  nothing. The merge identifies that emission with the reader's control node. An emission-driven
  control node is read at the round's boundary under ``"simultaneous"`` and as it stands under
  ``"sequential"`` (:mod:`~hybrid_aif.composition.runner`), which is how the components of the
  System read each other's states;
- a **modality whose parents are all state factors of one other component** becomes an emission
  of that component, named ``<reader>_<modality>`` (:func:`world_emissions` gives the names): an
  emission is the state showing itself, and the state is the owner's, so it is drawn from the
  owner's state as the owner's step leaves it, as the System observes it;
- a modality with parents in **two or more components** stays an emission of its own component,
  its foreign parents fed through the driven nodes above. **Known limitation**: a foreign factor
  gets one driven node, and a transition needs the factor at the boundary where a modality needs
  it after the owner has stepped, so such a modality reads the boundary value under
  ``"simultaneous"`` (and the value as it stands under ``"sequential"`` and ``"async"``, which is
  this round's when the owner steps before the reader in the round's order): one round older than
  the System observes it.
  Per-use driven nodes are on the roadmap;
- a control node a component's action node drives inside the System (a same-step identification)
  is driven in the world by the same action node, so the world's component follows its driver when
  sampled alone; ``drive_internal=False`` leaves it a control node with an action node of its own,
  for a coupling (:func:`~hybrid_aif.composition.couple.with_mirror`) that drives the model's and
  the world's nodes by one realised value.

Refused, each with the roadmap item it waits for: a foreign state parent of a neural, nonlinear or
deterministic transition (the nodes' maps are code, not tables); a foreign discrete parent of a
linear-Gaussian transition (a continuous factor reads its discrete parents within the step, which
the one driven node cannot serve); a foreign continuous state parent (a continuous node needs a
bounded box, and cross-component continuous parents are not built); a foreign ``D_deps`` parent; a
habit (``policy_prior``) on a component (a habit is over the leaf's own policy set, which the
leaf's mirror changes); a modality that reads another component's action node; and a transition
that reads a parameter variable and needs its axes moved.

Design: docs/design/engine-consolidation.md; docs/design/emissions-and-mirror.md.
"""

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array

from hybrid_aif.composition.merge import SEPARATOR, Identify, driven_nodes, merge
from hybrid_aif.controls import declared_parents
from hybrid_aif.distributions import (
    ConditionalCategorical,
    ConditionalDirichlet,
    ConditionalGaussian,
)
from hybrid_aif.factors import (
    LinearGaussianTransition,
    Parameter,
    TabularPrior,
    map_noise_modes,
)
from hybrid_aif.reads import Emission
from hybrid_aif.system import keyed
from hybrid_aif.system.core import System, SystemParams
from hybrid_aif.system.mirror import _action_name, _prepared

__all__ = ["mirror_components"]


def _local(name: str, component: str) -> str:
    """``name`` without the namespace of ``component``."""
    return name.removeprefix(f"{component}{SEPARATOR}")


def _unique(name: str, taken: set[str]) -> str:
    while name in taken:
        name += "_"
    taken.add(name)
    return name


def _moved(table: Array, parent_ndims: int, order: Sequence[int]) -> Array:
    """``table`` with its last ``parent_ndims`` axes permuted: new parent axis ``i`` is old axis
    ``order[i]``; the trailing size-1 axes a table keeps beyond its declared parents are dropped."""
    first = table.ndim - parent_ndims
    table = jnp.transpose(
        table, (*range(first), *(first + o for o in order), *range(first + len(order), table.ndim))
    )
    return table.reshape(table.shape[: first + len(order)])


def _move_tabular(decl, order: Sequence[int]):
    """A tabular transition whose parent axes are permuted by ``order`` (one entry per declared
    parent); a no-op trailing axis of size 1 beyond them is dropped, since the factor now reads a
    control node."""
    n = len(order)
    if isinstance(decl, ConditionalDirichlet):
        anchor = decl.alpha0
        if anchor is not None and anchor.ndim != 0:
            anchor = _moved(anchor, decl.parent_ndims, order)
        return ConditionalDirichlet(
            concentration=_moved(decl.concentration, decl.parent_ndims, order),
            parent_ndims=n,
            alpha0=anchor,
            growth=decl.growth,
        )
    return ConditionalCategorical(
        log_table=_moved(decl.log_table, decl.parent_ndims, order), parent_ndims=n
    )


def _move_gaussian(decl: LinearGaussianTransition, order: Sequence[int]):
    """A linear-Gaussian transition with its mode axes (the discrete parents) permuted."""
    inn = decl.innovations
    pn = inn.parent_ndims

    def moved(arr, core: int):
        first = arr.ndim - core - pn
        return jnp.transpose(
            arr, (*range(first), *(first + o for o in order), *range(first + pn, arr.ndim))
        )

    loc = moved(inn.loc, 1)
    chol = inn.chol if inn.shared_chol else moved(inn.chol, 2)
    F = moved(decl.F, 2) if decl.F.ndim > 2 else decl.F
    G = decl.G if decl.G is None or decl.G.ndim <= 2 else moved(decl.G, 2)
    return LinearGaussianTransition(
        F=F,
        innovations=ConditionalGaussian(loc=loc, chol=chol, parent_ndims=pn),
        G=G,
        noise_prior=map_noise_modes(decl.noise_prior, moved),
    )


def _leaves(system: System, where: str):
    """The components of ``system`` that are not composed of others, as ``(name, record)``."""
    names = [name for name, _ in system.subsystems]
    leaves = [
        (name, record)
        for name, record in system.subsystems
        if not any(other.startswith(f"{name}{SEPARATOR}") for other in names)
    ]
    nested = [name for name, _ in leaves if SEPARATOR in name]
    if nested:
        raise NotImplementedError(
            f"{where}: the components {nested} are leaves of a composed component, and a world's "
            "leaf is named by one path segment; mirror the System whole, or merge the components "
            "at one level"
        )
    return leaves


def _relocations(system: System, leaves, taken: dict[str, set[str]]):
    """The modalities whose parents are all state factors of one other component, as
    ``{modality index: (reader, owner, emission name inside the owner)}``. Such a modality is
    the state showing itself, and the state is the owner's, so it is an emission of the owner.
    ``taken`` holds each component's names, and gains the emission names."""
    n_states = system._table.num_states
    owner_factor = {g: name for name, rec in leaves for g in rec.states}
    out = {}
    for name, rec in leaves:
        for m in rec.modalities:
            deps = system.A_deps[m]
            owners = {owner_factor.get(g) for g in deps if g < n_states}
            if deps and all(g < n_states for g in deps) and len(owners) == 1 and name not in owners:
                (owner,) = owners
                local = _unique(f"{name}_{_local(system.modality_names[m], name)}", taken[owner])
                out[m] = (name, owner, local)
    return out


def _taken(system: System, leaves) -> dict[str, set[str]]:
    return {
        name: {
            *(_local(system.node_names[g], name) for g in rec.states),
            *(_local(u, name) for u, _ in rec.actions),
            *(_local(system.modality_names[m], name) for m in rec.modalities),
        }
        for name, rec in leaves
    }


def world_emissions(system: System) -> dict[str, str]:
    """The emission of the per-component mirror of ``system`` that stands for each modality, by
    the modality's name: the modality's own name, or the name inside the component that owns the
    states it reads where :func:`mirror_components` relocates it."""
    leaves = _leaves(system, "with_mirror(per_component=True)")
    taken = _taken(system, leaves)
    moved = _relocations(system, leaves, taken)
    return {
        system.modality_names[m]: (
            f"{moved[m][1]}{SEPARATOR}{moved[m][2]}" if m in moved else system.modality_names[m]
        )
        for _, rec in leaves
        for m in rec.modalities
    }


def mirror_components(
    system: System,
    params: SystemParams,
    *,
    lower: bool = False,
    param_key: Array | None = None,
    face: str = "mirror",
    drive_internal: bool = True,
) -> tuple[System, SystemParams]:
    """The merge of one mirror per component of ``system``: ``(System, SystemParams)``.

    See the module docstring. ``lower`` and ``param_key`` are those of
    :func:`~hybrid_aif.system.mirror.mirror`; ``face`` names the face a refusal speaks for.
    ``drive_internal=True`` drives a control node that a component's action node drives inside
    ``system`` by that action node in the world as well.
    """
    keyed_params, params, lowered = _prepared(system, params, lower, param_key, face)
    where = f"{face}(per_component=True)"
    if system.subsystems is None:
        raise ValueError(
            f"{where}: this System has no components (it is not a merge or a coupling), so there "
            "is no component to mirror; mirror it whole (per_component=False)"
        )
    leaves = _leaves(system, where)
    habits = [n for n, part in dict(keyed_params.parts or {}).items() if part.policy_prior]
    if habits or keyed_params.policy_prior is not None:
        raise NotImplementedError(
            f"{where}: a habit (policy_prior) over a component's policy set "
            f"({habits or 'the System'}) is not converted to the first-step habit of a leaf's "
            "free control nodes; drop the habit, or mirror the System whole (the per-component "
            "habit is on the roadmap)"
        )
    mirror = _Components(system, lowered, leaves, where, lower)
    specs = {name: mirror.leaf(name, rec) for name, rec in leaves}
    for name, spec in specs.items():
        mirror.relocate_into(name, spec)
        spec.emissions.extend(mirror.reads_of(name))
    leaf_systems: dict[str, System] = {}
    leaf_params: dict[str, SystemParams] = {}
    for name, rec in leaves:
        spec = specs[name]
        leaf = System(
            A_deps=(),
            modality_names=(),
            mixture_k=system.mixture_k,
            mixture_schedule=system.mixture_schedule,
            policy_len=1,
            epistemic=False,
            **{**spec.system, "emissions": tuple(spec.emissions)},
        )
        kernels = dict(spec.kernels)
        for local, home in spec.kernel_homes.items():
            kernels[local] = Parameter(spec.homes[home])
        mirrored = SystemParams(
            transition=tuple(spec.transition),
            initial_state_prior=tuple(lowered.initial_state_prior[g] for g in rec.states),
            likelihood=(),
            emission=kernels,
            control={
                _local(u, name): box
                for u, box in lowered.control.items()
                if mirror.node_owner.get(u) == name
            },
            parameters={spec.homes[h]: lowered.parameters[h] for h in spec.homes},
            references=tuple(("transition", i, spec.homes[h]) for i, h in spec.references),
        )
        leaf_params[name] = keyed.params_keyed(leaf._names(), mirrored)
        leaf_systems[name] = leaf
    bindings = [
        Identify(f"{owner}.{emission}", f"{reader}.{node}")
        for owner, emission, node, reader in mirror.feeds
    ]
    if drive_internal:
        owner_action = {a for _, rec in leaves for _, a in rec.actions}
        bindings += [
            Identify(variable, node)
            for node, variable in driven_nodes(system).items()
            if variable in owner_action
        ]
    for home, readers in mirror.readers.items():
        merged = [f"{n}.{specs[n].homes[home]}" for n in dict.fromkeys(readers)]
        canonical = home if home in merged else merged[0]
        bindings += [Identify(canonical, other) for other in merged if other != canonical]
    return merge(leaf_systems, leaf_params, bindings)


class _Leaf:
    """What one component's mirror is built from: the System's keywords and the tables."""

    def __init__(self):
        self.system: dict = {}
        self.emissions: list[tuple[str, Emission]] = []
        self.kernels: dict[str, object] = {}
        self.kernel_homes: dict[str, str] = {}  # emission -> the parameter variable it reads
        self.transition: list = []
        self.references: list[tuple[int, str]] = []  # (local factor, variable)
        self.homes: dict[str, str] = {}  # parameter variable -> its name inside the leaf


class _Components:
    """The per-component mirror of one System: the leaves' declarations, the identity emissions
    the owners gain, and the feeds that identify each with the control node that reads it."""

    def __init__(self, system: System, lowered: SystemParams, leaves, where: str, lower: bool):
        self.system, self.lowered, self.where = system, lowered, where
        self.table = system._table
        self.owner_factor = {g: name for name, rec in leaves for g in rec.states}
        self.node_owner = {u: name for name, rec in leaves for u, _ in rec.actions}
        stray = sorted(set(self.table.control_names) - set(self.node_owner))
        if stray:
            raise ValueError(
                f"{where}: control node(s) {stray} belong to no component record of this System; "
                "build the System with merge() or couple() so that every node has one, or mirror "
                "it whole (per_component=False)"
            )
        self.tabular = [isinstance(p, TabularPrior) for p in lowered.initial_state_prior]
        self.read_by = {} if lower else {(s, i): h for s, i, h in lowered.references}
        self.reads: dict[str, dict[int, str]] = {name: {} for name, _ in leaves}
        self.taken = _taken(system, leaves)
        self.moved = _relocations(system, leaves, self.taken)
        self.feeds: list[tuple[str, str, str, str]] = []  # (owner, emission, node, reader)
        self.readers: dict[str, list[str]] = {}  # parameter variable -> its reading leaves
        self.leaves: dict[str, _Leaf] = {}

    def reads_of(self, owner: str) -> list[tuple[str, Emission]]:
        """The identity emissions ``owner`` gains, one per factor another component reads."""
        return [
            (emission, Emission((_local(self.table.state_names[g], owner),), "mode", "identity"))
            for g, emission in self.reads[owner].items()
        ]

    def _read_emission(self, owner: str, g: int) -> str:
        if g not in self.reads[owner]:
            local = _local(self.table.state_names[g], owner)
            self.reads[owner][g] = _unique(f"{local}_read", self.taken[owner])
        return self.reads[owner][g]

    def relocate_into(self, owner: str, spec: _Leaf) -> None:
        """Add to ``owner``'s leaf the emissions of the modalities relocated to it."""
        system = self.system
        for m, (_, to, local) in self.moved.items():
            if to != owner:
                continue
            parents = tuple(_local(self.table.state_names[g], owner) for g in system.A_deps[m])
            spec.emissions.append((local, Emission(parents)))
            spec.kernels[local] = self.lowered.likelihood[m]
            if ("likelihood", m) in self.read_by:
                home = self.read_by[("likelihood", m)]
                self._home(spec, owner, home)
                spec.kernel_homes[local] = home

    def _home(self, leaf: _Leaf, name: str, home: str) -> None:
        leaf.homes.setdefault(home, _local(home, name))
        self.readers.setdefault(home, []).append(name)

    def leaf(self, name: str, rec) -> _Leaf:
        system, table, lowered, where = self.system, self.table, self.lowered, self.where
        n_states, n_nodes = table.num_states, len(system.B_deps)
        spec = self.leaves[name] = _Leaf()
        own = list(rec.states)
        own_nodes = [k for k, u in enumerate(table.control_names) if self.node_owner[u] == name]
        local_state = [_local(table.state_names[g], name) for g in own]
        local_nodes = [_local(table.control_names[k], name) for k in own_nodes]
        taken = self.taken[name]
        state_at = {g: i for i, g in enumerate(own)}
        node_at = {k: len(own) + j for j, k in enumerate(own_nodes)}
        foreign: dict[int, str] = {}

        def foreign_node(h: int, what: str) -> str:
            owner = self.owner_factor.get(h)
            if owner is None:
                raise ValueError(
                    f"{where}: {what} reads the state factor {table.state_names[h]!r}, which "
                    "belongs to no component record of this System; build the System with "
                    "merge() or couple() so that every factor has one, or mirror it whole "
                    "(per_component=False)"
                )
            if not self.tabular[h]:
                raise NotImplementedError(
                    f"{where}: {what} reads the continuous state factor "
                    f"{table.state_names[h]!r} of component {owner!r}; a foreign continuous "
                    "parent would be a continuous control node, which needs a bounded box, and "
                    "cross-component continuous parents are not built (on the roadmap); mirror "
                    "the System whole (per_component=False)"
                )
            if h not in foreign:
                foreign[h] = _unique(f"u_{owner}_{_local(table.state_names[h], owner)}", taken)
                self.feeds.append((owner, self._read_emission(owner, h), foreign[h], name))
            return foreign[h]

        steps: list[tuple[int, list, list, object]] = []
        for i, g in enumerate(own):
            decl = lowered.transition[g]
            parents = declared_parents(table, g)
            moved = [p for p in parents if p[0] == "state" and p[1] not in state_at]
            kept = [p for p in parents if p not in moved]
            what = f"the transition of {table.state_names[g]!r}"
            for kind, k in kept:
                if kind == "control" and self.node_owner[table.control_names[k]] != name:
                    raise ValueError(
                        f"{where}: {what} reads the control node {table.control_names[k]!r} of "
                        f"component {self.node_owner[table.control_names[k]]!r}; a transition "
                        "reads its own component's control nodes, and a node shared by "
                        "components (the collapse of joint_planner) belongs to one, so mirror "
                        "the merge as it was before the collapse"
                    )
            if moved and isinstance(decl, LinearGaussianTransition):
                raise NotImplementedError(
                    f"{where}: {what} is linear-Gaussian and reads the discrete state factor(s) "
                    f"{[table.state_names[h] for _, h in moved]} of another component; a "
                    "continuous factor reads its discrete parents within the step, as the owner "
                    "realises them this round, which the one driven control node a foreign "
                    "factor gets cannot serve (it gives the boundary value; per-use driven nodes "
                    "are on the roadmap); mirror the System whole (per_component=False)"
                )
            if moved and not isinstance(decl, ConditionalCategorical | ConditionalDirichlet):
                raise NotImplementedError(
                    f"{where}: {what} is a {type(decl).__name__} and reads the state factor(s) "
                    f"{[table.state_names[h] for _, h in moved]} of another component; a "
                    "neural, nonlinear or deterministic transition takes its parents through a "
                    "map that is code, so its foreign parent cannot become a control node (on "
                    "the roadmap); mirror the System whole (per_component=False)"
                )
            if moved and ("transition", g) in self.read_by:
                raise NotImplementedError(
                    f"{where}: {what} reads the parameter variable "
                    f"{self.read_by[('transition', g)]!r} and reads a state factor of another "
                    "component, whose axis would move and change the layout every reader of the "
                    "variable shares; declare the factor's own table (on the roadmap)"
                )
            for _, h in moved:
                foreign_node(h, what)
            steps.append((i, kept, moved, decl))
        for m in rec.modalities:
            for g in system.A_deps[m]:
                if m not in self.moved and g < n_states and g not in state_at:
                    foreign_node(g, f"the modality {system.modality_names[m]!r}")
        foreign_nodes = list(foreign.values())
        foreign_at = {h: len(own) + len(own_nodes) + j for j, h in enumerate(foreign)}

        B_deps = []
        for g, (_, kept, moved, decl) in zip(own, steps, strict=True):
            parents = declared_parents(table, g)
            new_order = kept + moved
            if moved and new_order != parents:
                at = {p: j for j, p in enumerate(parents)}
                decl = _move_tabular(decl, [at[p] for p in new_order])
            spec.transition.append(decl)
            B_deps.append(
                tuple(state_at[k] if kind == "state" else node_at[k] for kind, k in kept)
                + tuple(foreign_at[h] for _, h in moved)
            )
            if ("transition", g) in self.read_by:
                home = self.read_by[("transition", g)]
                self._home(spec, name, home)
                spec.references.append((state_at[g], home))

        if system.D_deps is not None:
            d_all = system._effective_D_deps()
            if any(h not in state_at for g in own for h in d_all[g]):
                raise NotImplementedError(
                    f"{where}: a state factor of component {name!r} has a prior conditioned on a "
                    "state factor of another component (a foreign D_deps parent); a leaf's prior "
                    "cannot read a control node (on the roadmap); mirror the System whole "
                    "(per_component=False)"
                )
            mapped = tuple(tuple(state_at[h] for h in d_all[g]) for g in own)
            d_deps = (
                (*mapped, *((),) * (len(own_nodes) + len(foreign_nodes))) if any(mapped) else None
            )
        else:
            d_deps = None

        for m in rec.modalities:
            if m in self.moved:
                continue
            parents = []
            for g in system.A_deps[m]:
                if g < n_states:
                    parents.append(local_state[state_at[g]] if g in state_at else foreign[g])
                    continue
                node = table.control_names[g - n_nodes]
                if self.node_owner[node] != name:
                    raise NotImplementedError(
                        f"{where}: the modality {system.modality_names[m]!r} reads the action "
                        f"node of {node!r}, a control node of component "
                        f"{self.node_owner[node]!r}; a leaf's emission reads its own control "
                        "nodes (on the roadmap); mirror the System whole (per_component=False)"
                    )
                parents.append(_local(node, name))
            local = _local(system.modality_names[m], name)
            spec.emissions.append((local, Emission(tuple(parents))))
            spec.kernels[local] = lowered.likelihood[m]
            if ("likelihood", m) in self.read_by:
                home = self.read_by[("likelihood", m)]
                self._home(spec, name, home)
                spec.kernel_homes[local] = home

        taken_actions: set[str] = set()
        actions = []
        for k in own_nodes:
            action = system.action_names[k]
            actions.append(
                _local(action, name)
                if action is not None
                else _action_name(_local(table.control_names[k], name), taken_actions)
            )
        taken_actions.update(actions)
        actions += [_action_name(node, taken_actions) for node in foreign_nodes]
        own_names = {table.state_names[g] for g in own}
        spec.system = dict(
            B_deps=(*B_deps, *(() for _ in range(len(own_nodes) + len(foreign_nodes)))),
            D_deps=d_deps,
            controls=tuple(range(len(own), len(own) + len(own_nodes) + len(foreign_nodes))),
            state_names=tuple(local_state),
            control_names=(*local_nodes, *foreign_nodes),
            action_names=tuple(actions),
            emissions=spec.emissions,
            dims=tuple((_local(f, name), d) for f, d in system.dims if f in own_names),
            cells=tuple((_local(f, name), grid) for f, grid in system.cells if f in own_names),
        )
        return spec
