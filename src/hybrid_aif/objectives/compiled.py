"""Fused prob-space planning path: precompiled tables and closures over raw prob arrays.

The compiled equivalent of :mod:`hybrid_aif.objectives.discrete` (the reference).
:func:`compile_tables` hoists every loop-invariant table once per planning call; a
learnable declaration contributes its posterior-mean table and its Dirichlet
novelty weights. Beliefs cross the scan as tuples of probability vectors, and each
step computes the predicted-observation marginal once per modality. The prob-space
carry follows pymdp's ``control.compute_neg_efe_policy``, which never round-trips through
logits. Results match the reference to float tolerance, not bitwise. Tables change with
every ``learn``, so a ``PlannerContext`` is per-call scratch and must not be cached across ticks.
A table the System learns within the plan (``System.plan_learning``) is the exception to the
hoisting: :func:`make_learning` carries its working counts through the scan and re-reads its mean,
entropy and novelty at each step (design: parameters-within-the-plan.md A3).
"""

import dataclasses
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import xlogy

from hybrid_aif.controls import (
    ControlDeps,
    ControlStep,
    ControlTable,
    SensedNodes,
    action_node_beliefs,
    control_read,
    has_control_axes,
    take_controls,
)
from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalDirichlet,
    Distribution,
    Learnable,
)
from hybrid_aif.interval import Interval
from hybrid_aif.objectives import hybrid as _hybrid
from hybrid_aif.objectives import registry
from hybrid_aif.objectives.base import INFOGAIN, TermTree
from hybrid_aif.objectives.discrete import (
    TargetParents,
    _block_names,
    resolve_conditional_targets,
    resolve_step_targets,
    selected_conditional_entropy,
    target_parent_entries,
    terminal_cross_entropy_risk,
)
from hybrid_aif.objectives.discrete import contract_probs as _contract
from hybrid_aif.objectives.pooling import Pooled, PooledEntry, count_at, scale_pooled
from hybrid_aif.objectives.presence import Presence, mask_absent, presence_at


class CompiledTables(eqx.Module):
    """Prob-space view of the planning tensors, built once per planning call.

    A plain pytree, legal to close over under jit.
    """

    A_probs: tuple[Array, ...]  # prob table per modality (softmax or posterior mean), child-first
    A_H: tuple[Array, ...]  # conditional entropy of A_probs per parent config, per modality
    B_probs: tuple[Array, ...]  # prob table per factor, control axis intact
    wA: tuple[Array | None, ...]  # Dirichlet novelty weights per learnable modality
    wB: tuple[Array | None, ...]  # Dirichlet novelty weights per learnable transition
    obs_target_logs: dict[str, Array]  # normalized log_probs of C designations; a conditional
    #   one's table is child-first, (levels, *parent levels), resolved per step (C_deps)
    state_targets: dict[str, Distribution]  # standing risk-over-states designations, a
    #   ConditionalCategorical entry resolved per step (C_deps)
    terminal_targets: dict[str, Distribution | ConditionalCategorical]  # h designations,
    #   scored at horizon end; a ConditionalCategorical entry is resolved against the final
    #   beliefs at scoring time (H_deps, N1)


def _prob_table(decl: ConditionalCategorical | ConditionalDirichlet) -> Array:
    return (
        decl.mean()
        if isinstance(decl, Learnable)
        else jax.nn.softmax(decl.log_table, axis=-(decl.parent_ndims + 1))
    )


def compile_tables(
    A: tuple[ConditionalCategorical | ConditionalDirichlet, ...],
    B: tuple[ConditionalCategorical | ConditionalDirichlet, ...],
    obs_targets: dict[str, Categorical],
    state_targets: dict[str, Distribution],
    terminal_targets: dict[str, Distribution | ConditionalCategorical],
) -> CompiledTables:
    """Hoist every loop-invariant softmax/entropy/novelty table out of the scan."""
    A_probs = tuple(_prob_table(a) for a in A)
    A_H = tuple(
        -xlogy(p, p).sum(axis=-(a.parent_ndims + 1)) for p, a in zip(A_probs, A, strict=True)
    )
    return CompiledTables(
        A_probs=A_probs,
        A_H=A_H,
        B_probs=tuple(_prob_table(b) for b in B),
        wA=tuple(a.novelty() if isinstance(a, Learnable) else None for a in A),
        wB=tuple(b.novelty() if isinstance(b, Learnable) else None for b in B),
        obs_target_logs={k: t.log_probs for k, t in obs_targets.items()},
        state_targets=dict(state_targets),
        terminal_targets=dict(terminal_targets),
    )


def make_transition(
    tables: CompiledTables,
    B_deps: tuple[tuple[int, ...], ...],
    control_deps: ControlDeps | None = None,
):
    """``expected_states`` over raw prob vectors: slice B by action, contract parents.

    ``control_deps`` says which controls slice each factor's table: the System's
    :class:`~hybrid_aif.controls.ControlTable` (``action`` a control step, one level per
    control node), or a
    per-factor declaration (``action`` one level per factor). A
    :class:`~hybrid_aif.controls.ControlStep` action mixes each table over its belief entries.
    The closure's ``tables_t`` replaces ``tables`` for one step (:func:`make_learning`).
    """

    def transition(
        qs: tuple[Array, ...], action: Array, tables_t: CompiledTables | None = None
    ) -> tuple[Array, ...]:
        if not isinstance(action, ControlStep):
            action = jnp.asarray(action, jnp.int32)
        B_probs = (tables if tables_t is None else tables_t).B_probs
        return tuple(
            _contract(
                take_controls(B_probs[f], control_read(action, f, control_deps))
                if has_control_axes(control_deps, f, B_probs[f].ndim - 1)
                else B_probs[f],
                tuple(qs[i] for i in deps),
                keep_child=True,
            )
            for f, deps in enumerate(B_deps)
        )

    return transition


def make_step_terms(
    tables: CompiledTables,
    A_deps: tuple[tuple[int, ...], ...],
    B_deps: tuple[tuple[int, ...], ...],
    num_factors: int,
    *,
    decomposition: str,
    state_infogain_modalities: tuple[int, ...],
    param_infogain_modalities: tuple[int, ...],
    param_infogain_factors: tuple[int, ...],
    modality_names: tuple[str, ...],
    state_names: tuple[str, ...],
    mounts: tuple[registry.Mount, ...] = (),
    channel_selection: tuple[tuple[bool, ...] | None, ...] = (),
    control_deps: ControlDeps | None = None,
    branching: bool = False,
    c_deps: dict[str, TargetParents] | None = None,
    sensed: SensedNodes | None = None,
    present: Presence | None = None,
    pooled: Pooled | None = None,
):
    """Fused per-step term evaluator with the System's designation/selector semantics.

    Block selection is resolved at build time, so the scan computes only leaves
    that appear in the tree. The ``*_modalities``/``*_factors`` arguments are
    resolved declaration indices (from ``System._term_targets``), not name
    selectors. With ``param_infogain_factors`` nonempty, or a mounted row that
    reads the previous step, the closure needs ``qs_prev`` as its fourth argument.
    Epistemic leaves come from :mod:`hybrid_aif.objectives.registry`; ``mounts``
    are ``System.terms``' resolved rows, each landing under its own term. The channels
    in ``state_infogain_modalities`` carry an ``infogain`` leaf under ``utility_infogain``
    and an ``ambiguity`` leaf under ``risk_ambiguity``, where an undesignated channel's
    risk is ``−H[q(o)]``, its implicit flat preference (design: per-variable-infogain.md
    V11). ``channel_selection`` holds one parent mask per modality (``None`` = all); a
    mounted channel with a partial mask reads ``E[H[o | S]]`` in place of the hoisted
    conditional entropy, in its info gain and its ambiguity alike (design:
    per-variable-infogain.md V9). ``control_deps`` is ``System.B_control_deps``. ``branching``
    marks a closure built for a branching planner, where a state target's ``risk`` leaf is its
    cross-entropy ``−E_q log C(s)`` (design: per-variable-infogain.md V11 decision 6). ``c_deps``
    names the parents of each conditional target (``System.C_deps`` resolved to columns and
    engine positions): the target is resolved at each step under the step's control entries and
    beliefs (:func:`~hybrid_aif.objectives.discrete.resolve_step_targets`) and then scored as an
    unconditional one. ``sensed`` names the action nodes some modality observes: ``A_deps`` then
    holds their engine positions after the state factors, and each step's belief tuple is
    extended by ``P(a | u_t)`` of each (:func:`~hybrid_aif.controls.action_node_beliefs`) for the
    channels' parent lookups: an action node's belief at a step is a parent only, no leaf is
    keyed by it and nothing about it is carried, and a channel over action nodes alone has no
    information-gain leaf (its ambiguity, risk and utility leaves are scored as any channel's).
    ``present`` (:class:`~hybrid_aif.objectives.presence.Presence`) zeroes the leaves of a
    modality absent at the step, keyed by its name and by ``likelihood:<name>``. ``pooled``
    (:class:`~hybrid_aif.objectives.pooling.Pooled`) multiplies a declared table's novelty leaf by
    the count of the step's level of its control node. The closure's ``tables_t`` keyword
    replaces ``tables`` for one step (the working posterior of a plan that learns tables within
    it, :func:`make_learning`); every reader of a table reads it there.
    """
    mods = _block_names("modality", len(A_deps), modality_names)
    c_deps = dict(c_deps or {})
    obs_c_deps = {k: v for k, v in c_deps.items() if k in tables.obs_target_logs}
    state_c_deps = {k: v for k, v in c_deps.items() if k in tables.state_targets}
    facs = _block_names("factor", num_factors, state_names)

    obs_idx = tuple(m for m, name in enumerate(mods) if name in tables.obs_target_logs)
    ig_decomposition = decomposition == "utility_infogain"
    sig_idx = state_infogain_modalities if ig_decomposition else ()
    amb_idx = () if ig_decomposition else tuple(state_infogain_modalities)
    flat_idx = () if ig_decomposition else tuple(m for m in range(len(mods)) if m not in obs_idx)
    state_idx = tuple(f for f in range(num_factors) if facs[f] in tables.state_targets)
    need_qobs = tuple(sorted({*obs_idx, *flat_idx, *sig_idx, *param_infogain_modalities}))
    need_h = tuple(sorted({*sig_idx, *amb_idx}))
    # partially selected channels, recomputed per step; the rest keep the hoisted table
    partial = {
        m: channel_selection[m]
        for m in state_infogain_modalities
        if channel_selection and channel_selection[m] is not None and m in need_h
    }

    statics = registry.Statics(
        index=(),
        modality_names=mods,
        state_names=facs,
        A_deps=A_deps,
        B_deps=B_deps,
        B_control_deps=control_deps,
        tables=tables,
    )
    mounted = (
        (registry.STATE, registry.CATEGORICAL, sig_idx),
        (registry.param_kind("likelihood"), registry.DIRICHLET, param_infogain_modalities),
        (registry.param_kind("transition"), registry.DIRICHLET, param_infogain_factors),
    )
    infogain_steps = tuple(
        registry.prepare(kind, family, statics._replace(index=idx), compiled=True)
        for kind, family, idx in mounted
        if idx
    )
    row_steps = registry.prepare_step_rows(mounts, statics, compiled=True)

    def step_terms(
        qs: tuple[Array, ...],
        action: Array,
        t: Array,
        qs_prev: tuple[Array, ...] | None = None,
        tables_t: CompiledTables | None = None,
    ) -> TermTree:
        held = tables if tables_t is None else tables_t
        present_t = None if present is None else presence_at(present, action, t, None)
        if sensed is not None:
            qs = (*qs, *action_node_beliefs(action, sensed))
        obs_logs = dict(tables.obs_target_logs)
        for name, deps in obs_c_deps.items():
            obs_logs[name] = take_controls(
                obs_logs[name], target_parent_entries(deps, action, qs), log=True
            )
        state_targets = (
            resolve_step_targets(tables.state_targets, state_c_deps, action, qs)
            if state_c_deps
            else tables.state_targets
        )
        parents = {m: tuple(qs[i] for i in A_deps[m]) for m in {*need_qobs, *need_h}}
        q_obs = {m: _contract(held.A_probs[m], parents[m], keep_child=True) for m in need_qobs}
        h_cond = {
            m: selected_conditional_entropy(held.A_probs[m], parents[m], partial[m])
            if m in partial
            else _contract(held.A_H[m], parents[m], keep_child=False)
            for m in need_h
        }
        tree: TermTree = {}
        if ig_decomposition:
            if obs_idx:
                tree["utility"] = {
                    mods[m]: Interval.exact(jnp.sum(q_obs[m] * obs_logs[mods[m]], axis=-1))
                    for m in obs_idx
                }
        else:
            if obs_idx:
                tree["risk"] = {
                    mods[m]: Interval.exact(
                        jnp.sum(
                            q_obs[m] * (jnp.log(q_obs[m]) - obs_logs[mods[m]]),
                            axis=-1,
                        )
                    )
                    for m in obs_idx
                }
            if flat_idx:
                tree.setdefault("risk", {}).update(
                    {
                        mods[m]: Interval.exact(jnp.sum(xlogy(q_obs[m], q_obs[m]), axis=-1))
                        for m in flat_idx
                    }
                )
            if amb_idx:
                tree["ambiguity"] = {mods[m]: Interval.exact(h_cond[m]) for m in amb_idx}
        if infogain_steps or row_steps:
            ctx = registry.StepContext(
                qs=qs,
                action=action,
                qs_prev=qs_prev,
                parents=parents,
                q_obs=q_obs,
                h_cond=h_cond,
                tables=tables_t,
            )
            if infogain_steps:
                leaves: dict[str, Interval] = {}
                for step in infogain_steps:
                    leaves.update(step(ctx))
                tree[INFOGAIN] = leaves
            for term, step in row_steps:
                tree.setdefault(term, {}).update(step(ctx))
        if state_idx:
            leaves = tree.setdefault("risk", {})
            for f in state_idx:
                target = state_targets[facs[f]]
                if branching:  # the cross-entropy, linear in the belief (V11 decision 6)
                    ce = jnp.where(qs[f] > 0, qs[f] * target.log_probs, 0.0)
                    leaves[facs[f]] = Interval.exact(-jnp.sum(ce, axis=-1))
                else:
                    q_f = Categorical(logits=jnp.log(qs[f]))
                    leaves[facs[f]] = q_f.kl_to(target)
        tree = scale_pooled(tree, pooled, action, None)
        if present_t is not None:
            tree = mask_absent(tree, present_t, mods)
        return tree

    return step_terms


def make_terminal_terms(
    tables: CompiledTables,
    num_factors: int,
    state_names: tuple[str, ...],
    h_deps: tuple[tuple[int, ...], ...] | None = None,
    branching: bool = False,
):
    """Terminal-risk evaluator over raw prob vectors, or None without H designations.

    A conditional terminal target (``h_deps``) is resolved against the final
    beliefs inside the closure. ``branching`` marks a closure built for a branching planner,
    where each terminal target is scored by its cross-entropy
    (:func:`~hybrid_aif.objectives.discrete.terminal_cross_entropy_risk`; design:
    per-variable-infogain.md V11 decision 8).
    """
    if not tables.terminal_targets:
        return None
    facs = _block_names("factor", num_factors, state_names)
    idx = tuple(f for f in range(num_factors) if facs[f] in tables.terminal_targets)
    deps = h_deps if h_deps is not None else ((),) * num_factors
    conditional = any(deps[f] for f in idx)

    def terminal_terms(qs_final: tuple[Array, ...]) -> TermTree:
        if branching:
            beliefs = tuple(Categorical(logits=jnp.log(q)) for q in qs_final)
            return terminal_cross_entropy_risk(beliefs, tables.terminal_targets, deps, facs)
        targets = (
            resolve_conditional_targets(tables.terminal_targets, deps, qs_final, facs)
            if conditional
            else tables.terminal_targets
        )
        return {
            "terminal_risk": {
                facs[f]: Categorical(logits=jnp.log(qs_final[f])).kl_to(targets[facs[f]])
                for f in idx
            }
        }

    return terminal_terms


# ------------------------------------------------- parameters learned within the plan


class FusedLearning(NamedTuple):
    """The tables a plan on the fused path learns within it (design:
    parameters-within-the-plan.md D1, A3): the working ``tables``
    (:class:`~hybrid_aif.objectives.hybrid.WorkingTable`, one per key of the carry's counts, each
    with its readers), the declarations ``A`` and ``B`` (a learned reader's a
    ``ConditionalDirichlet``), their parents ``A_deps`` and ``B_deps`` (engine positions), the
    System's ``control`` table, and ``pooled``, the :class:`~hybrid_aif.objectives.pooling.
    PooledEntry` of each reader whose readings are pooled (keyed by ``(slot, index)``)."""

    tables: tuple
    A: tuple
    B: tuple
    A_deps: tuple[tuple[int, ...], ...]
    B_deps: tuple[tuple[int, ...], ...]
    control: ControlTable
    pooled: dict[tuple[str, int], PooledEntry]

    def decl(self, slot: str, i: int):
        return (self.A if slot == "likelihood" else self.B)[i]

    def seed(self) -> dict[str, Array]:
        """Each table's counts at the plan's start: the posterior's concentration."""
        return {t.key: self.decl(*t.readers[0]).concentration for t in self.tables}

    def scale(self, slot: str, i: int, action: object) -> Array:
        """How many readings the step's one reading of reader ``(slot, i)`` stands for: the
        pooled count of the step's level (1 where the reader is not pooled)."""
        entry = self.pooled.get((slot, i))
        dtype = self.decl(slot, i).concentration.dtype
        if entry is None:
            return jnp.ones((), dtype)
        return jnp.asarray(count_at(entry, action, None), dtype)

    def reads(self) -> dict[str, Array]:
        """The per-reader record a step leaves for a branch, zeroed (the carry's structure)."""
        out = {}
        for t in self.tables:
            for slot, i in t.readers:
                decl = self.decl(slot, i)
                shape = decl.concentration.shape if slot == "transition" else ()
                out[f"{slot}:{i}"] = jnp.zeros(shape, decl.concentration.dtype)
        return out


def pooled_readers(
    pooled: Pooled | None, modality_names: tuple[str, ...], state_names: tuple[str, ...]
) -> dict[tuple[str, int], PooledEntry]:
    """``pooled``'s entries keyed by reader, ``("likelihood", m)`` or ``("transition", f)``."""
    if pooled is None:
        return {}
    out = {}
    for entry in pooled.entries:
        slot, name = entry.key.split(":", 1)
        names = modality_names if slot == "likelihood" else state_names
        if name in names:
            out[(slot, names.index(name))] = entry
    return out


def working_tables(
    tables: CompiledTables, learning: FusedLearning, counts: dict[str, Array]
) -> CompiledTables:
    """``tables`` with each learned reader's entries read from the working ``counts``: its mean
    table, its conditional entropy (a likelihood) and its novelty weights. Every other entry is
    the hoisted one, untouched."""
    A_probs, A_H, wA = list(tables.A_probs), list(tables.A_H), list(tables.wA)
    B_probs, wB = list(tables.B_probs), list(tables.wB)
    for t in learning.tables:
        slot, i = t.readers[0]
        decl = learning.decl(slot, i)
        held = eqx.tree_at(lambda d: d.concentration, decl, counts[t.key])
        probs, novelty = held.mean(), held.novelty()
        entropy = -xlogy(probs, probs).sum(axis=-(decl.parent_ndims + 1))
        for slot, i in t.readers:
            if slot == "likelihood":
                A_probs[i], A_H[i], wA[i] = probs, entropy, novelty
            else:
                B_probs[i], wB[i] = probs, novelty
    return dataclasses.replace(
        tables,
        A_probs=tuple(A_probs),
        A_H=tuple(A_H),
        wA=tuple(wA),
        B_probs=tuple(B_probs),
        wB=tuple(wB),
    )


def _times(child: Array, parents: Array) -> Array:
    """``child ⊗ parents``, child-first; leading axes of ``child`` (``(..., L)``) batch."""
    lead = child.shape[:-1]
    return child.reshape(*child.shape, *([1] * parents.ndim)) * parents.reshape(
        (1,) * (len(lead) + 1) + parents.shape
    )


def make_learning(
    tables: CompiledTables,
    learning: FusedLearning,
    transition_fn,
    step_terms_fn,
    terminal_terms_fn,
    init_beliefs: tuple[Array, ...],
):
    """The fused closures over a :class:`~hybrid_aif.system.core.WorkingCarry` (design:
    parameters-within-the-plan.md D1, A3).

    ``transition(carry, action)`` reads each learned table through the mean of the working
    counts and absorbs, at rate one, the step's expected count statistic, the model's joint over
    child and parents (:func:`~hybrid_aif.objectives.hybrid.expected_joint`): a transition's
    ``B(s'_f | p) Π q(p)`` over its parents as the tabular reader reads them (the pre-step parent
    vectors, a control axis as the taken level's one-hot or a belief-valued step's probability
    vector), a likelihood's ``A(o | s) Π q(s)`` over the predicted parents; a reader whose
    readings are pooled absorbs its statistic times the step's count. The carry's ``reads``
    keeps each transition reader's joint (times the count) and each likelihood reader's count,
    for a branch. ``step_terms(carry,
    action, t, qs_prev=None)`` scores the step over the tables the counts before the absorption
    define (``working_tables``), so the novelty leaf, the occupancy row and every likelihood
    read take the working posterior; nothing else is recomputed. ``transition_fn`` and
    ``step_terms_fn`` are :func:`make_transition`'s and :func:`make_step_terms`'s closures over
    ``tables``. Returns the four closures and the seeded carry.
    """
    WorkingCarry = _hybrid.WorkingCarry

    B_deps, A_deps = learning.B_deps, learning.A_deps

    def transition(carry, action):
        if not isinstance(action, ControlStep):
            action = jnp.asarray(action, jnp.int32)
        work = working_tables(tables, learning, carry.counts)
        qs = carry.beliefs
        new = transition_fn(qs, action, work)
        after, reads = dict(carry.counts), {}
        for t in learning.tables:
            for slot, i in t.readers:
                decl, n = learning.decl(slot, i), learning.scale(slot, i, action)
                if slot == "transition":
                    parents = _hybrid._table_parents(
                        decl, i, tuple(qs[g] for g in B_deps[i]), action, learning.control, None
                    )
                    joint = _hybrid.expected_joint(carry.counts[t.key], decl.parent_ndims, parents)
                    reads[f"{slot}:{i}"] = (n * joint).astype(n.dtype)
                    stats = reads[f"{slot}:{i}"]
                else:
                    parents = tuple(new[g] for g in A_deps[i])
                    reads[f"{slot}:{i}"] = n
                    stats = n * _hybrid.expected_joint(
                        carry.counts[t.key], decl.parent_ndims, parents
                    )
                after[t.key] = after[t.key] + stats.astype(after[t.key].dtype)
        return WorkingCarry(new, after, carry.counts, reads)

    def step_terms(carry, action, t, qs_prev=None):
        work = working_tables(tables, learning, carry.before)
        prev = None if qs_prev is None else qs_prev.beliefs
        return step_terms_fn(carry.beliefs, action, t, prev, tables_t=work)

    terminal = (
        None if terminal_terms_fn is None else (lambda carry: terminal_terms_fn(carry.beliefs))
    )
    counts = learning.seed()
    return (
        transition,
        step_terms,
        terminal,
        WorkingCarry(tuple(init_beliefs), counts, counts, learning.reads()),
    )


def make_learning_branch(
    tables: CompiledTables,
    learning: FusedLearning,
    likelihood: tuple,
    A_deps: tuple[tuple[int, ...], ...],
    modalities: tuple[int, ...],
):
    """The ``branch_on`` closure over a :class:`~hybrid_aif.system.core.WorkingCarry` (design:
    parameters-within-the-plan.md A3): each outcome is predicted through the working posterior
    the step read (its counts before the absorption), and each branch's counts absorb that
    branch's outcome in place of the step's expected statistic: a learned likelihood of a
    branched modality the exact conjugate update ``onehot(o) ⊗ q(s_P | o)`` (the joint
    posterior over its parents), every other learned table its expected joint conditioned on the
    branch (a transition's step joint reweighted along the child by ``q(s'_f | o) / q(s'_f)``,
    a likelihood's ``A(o' | s) Π q(s | o)`` over the branch's parents). ``likelihood`` holds
    each modality's point table."""
    from hybrid_aif.planning.branching import make_branch_fn

    WorkingCarry = _hybrid.WorkingCarry

    learned = {i: t for t in learning.tables for slot, i in t.readers if slot == "likelihood"}

    def branch(carry):
        work = working_tables(tables, learning, carry.before)
        tables_t = list(likelihood)
        for m in modalities:
            if m in learned:
                decl = learning.A[m]
                tables_t[m] = ConditionalCategorical(
                    log_table=jnp.log(work.A_probs[m]), parent_ndims=decl.parent_ndims
                )
        p, after, evidence = make_branch_fn(tables_t, A_deps, modalities, evidence=True)(
            carry.beliefs
        )
        exact = dict(zip(modalities, evidence, strict=True))
        n = p.shape[0]

        def spread(tree):
            return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n, *x.shape)), tree)

        counts = {}
        for t in learning.tables:
            total = jnp.broadcast_to(carry.before[t.key], (n, *carry.before[t.key].shape))
            for slot, i in t.readers:
                read = carry.reads[f"{slot}:{i}"]
                if slot == "transition":
                    prior = carry.beliefs[i]
                    ratio = jnp.where(prior > 0, after[i] / jnp.where(prior > 0, prior, 1.0), 0.0)
                    stats = _times(ratio, jnp.ones(read.shape[1:], read.dtype)) * read
                elif i in exact:
                    stats = read * exact[i]
                else:
                    decl = learning.A[i]
                    stats = read * jax.vmap(
                        lambda *ps, i=i, decl=decl: _hybrid.expected_joint(
                            carry.before[learned[i].key], decl.parent_ndims, ps
                        )
                    )(*(after[g] for g in A_deps[i]))
                total = total + stats.astype(total.dtype)
            counts[t.key] = total
        return p, WorkingCarry(after, counts, spread(carry.before), spread(carry.reads))

    return branch
