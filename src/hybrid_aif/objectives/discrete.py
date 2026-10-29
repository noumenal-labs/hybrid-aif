"""Exact EFE terms for the discrete engine, assembled into per-block term trees.

Semantics follow pymdp's control.py (pymdp/control.py): expected state/obs, utility, info gain
and the exact ``spm_wnorm`` novelty; pymdp's functions are the parity oracle
(tests/test_objectives.py, tests/test_transition_novelty.py). Leaves are
:meth:`Interval.exact` in each term's natural sign; functions are unbatched over policies
(batch via ``vmap``).
pymdp's control.py itself follows SPM's ``spm_MDP_VB_X`` and its successors (``toolbox/DEM``).
``targets`` arguments are dicts keyed by block name, and undesignated blocks are
omitted; an undesignated observation's implicit flat preference is :func:`flat_risk`.
With normalized targets ``risk + ambiguity = −(utility + state info
gain)`` exactly; against unnormalized log-preferences they differ by the
policy-independent ``Σ_m logsumexp(C_m)``. A learnable declaration is scored
through its posterior-mean table, so ``I(o; θ | s)`` appears only as the explicit
novelty leaf. Epistemic leaves land under ``infogain``: state leaves keyed by
channel, parameter leaves by :func:`param_key`.

Design: docs/design/parameter-learning.md D2
"""

import string

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import xlogy

from hybrid_aif.controls import (
    ControlDeps,
    ControlLayout,
    ControlTable,
    control_read,
    has_control_axes,
    is_belief,
    node_entry,
    num_control_axes,
    take_controls,
)
from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalDirichlet,
    ConditionalGaussian,
    Gaussian,
    Learnable,
    MixtureGaussian,
)
from hybrid_aif.distributions.base import Distribution
from hybrid_aif.interval import Interval
from hybrid_aif.objectives.base import INFOGAIN, TermTree, register_term

register_term("terminal_risk", 1.0)

_AXES = string.ascii_lowercase


def param_key(slot: str, name: str) -> str:
    """The slot-qualified block key of a parameter variable, e.g. ``"likelihood:vision"``.

    The same spelling ``epistemic`` and ``lr_by_slot`` select it with.
    """
    return f"{slot}:{name}"


def _point(decl):
    """A declaration's point (posterior-mean) form; point declarations pass through."""
    return decl.point() if isinstance(decl, Learnable) else decl


def _block_names(prefix: str, n: int, block_names: tuple[str, ...] | None) -> tuple[str, ...]:
    if block_names is None:
        return tuple(f"{prefix}_{i}" for i in range(n))
    if len(block_names) != n:
        raise ValueError(f"got {len(block_names)} block names for {n} blocks")
    return block_names


def contract_probs(table: Array, parents: tuple[Array, ...], keep_child: bool) -> Array:
    """Contract a child-first (or parent-only) table against one prob vector per parent axis."""
    subs = _AXES[: table.ndim]
    off = 1 if keep_child else 0
    spec = ",".join([subs, *subs[off:]]) + "->" + (subs[0] if keep_child else "")
    return jnp.einsum(spec, table, *parents)


def _contract(table: Array, beliefs: tuple[Categorical, ...], keep_child: bool) -> Array:
    """Contract a child-first (or parent-only) table against one belief per parent axis."""
    return contract_probs(table, tuple(b.probs for b in beliefs), keep_child)


def expected_states(
    qs: tuple[Categorical, ...],
    B: tuple[ConditionalCategorical, ...],
    action: Array,
    B_deps: tuple[tuple[int, ...], ...],
    control_deps: ControlDeps | None = None,
) -> tuple[Categorical, ...]:
    """One-step push of factor beliefs through control-conditioned transition CPDs.

    ``B[f]`` has parent axes ``(*B_deps[f] factors, control)`` (pymdp's
    ``(ns_f, *dep_sizes, nu_f)`` layout) and ``action[f]`` indexes the trailing
    control axis; with ``control_deps`` (``System.B_control_deps``) the trailing axes are
    one per listed control, each indexed by that control node's level. A learnable transition
    predicts through its mean table.
    """
    action = jnp.asarray(action, jnp.int32)
    out = []
    for f, (B_f, deps) in enumerate(zip(B, B_deps, strict=True)):
        B_f = _point(B_f)  # a learnable transition predicts through its mean table (D2)
        sliced = (
            take_controls(B_f.log_table, control_read(action, f, control_deps))
            if has_control_axes(control_deps, f, B_f.parent_ndims)
            else B_f.log_table
        )
        if not deps:
            #  a memoryless factor: the control axes were its only parent axes, so the
            #  slice is next tick's distribution
            out.append(Categorical(logits=jax.nn.log_softmax(sliced, axis=-1)))
            continue
        conditional = ConditionalCategorical(
            log_table=sliced,
            parent_ndims=len(deps)
            if isinstance(control_deps, ControlTable)
            else B_f.parent_ndims - num_control_axes(control_deps, f),
        )
        out.append(conditional.average(tuple(qs[i] for i in deps)))
    return tuple(out)


def expected_obs(
    qs: tuple[Categorical, ...],
    A: tuple[ConditionalCategorical, ...],
    A_deps: tuple[tuple[int, ...], ...],
) -> tuple[Categorical, ...]:
    """Predicted observation marginals per modality (pymdp's ``control.compute_expected_obs``).

    A learnable modality contributes its posterior-mean table.
    """
    return tuple(
        _point(A_m).average(tuple(qs[i] for i in deps)) for A_m, deps in zip(A, A_deps, strict=True)
    )


def utility(
    q_obs: tuple[Categorical, ...],
    targets: dict[str, Categorical],
    block_names: tuple[str, ...] | None = None,
) -> TermTree:
    """E_q(o)[ln C] per designated observation block (pymdp's ``control.compute_expected_utility``).

    ``targets`` maps block names to normalized Categorical preference targets;
    blocks without a designation are omitted from the tree.
    """
    names = _block_names("modality", len(q_obs), block_names)
    leaves = {
        name: Interval.exact(jnp.sum(q.probs * targets[name].log_probs, axis=-1))
        for name, q in zip(names, q_obs, strict=True)
        if name in targets
    }
    return {"utility": leaves}


def risk(
    q_blocks: tuple[Distribution, ...],
    targets: dict[str, Distribution],
    block_names: tuple[str, ...] | None = None,
) -> TermTree:
    """KL(q(block) ‖ target) per designated block, via each family's ``kl_to``.

    Family-agnostic: observation, state and continuous blocks alike; mixture
    bounds propagate as Intervals. ``KL(q‖target)`` is the risk term of SPM's ``spm_forwards``
    (``spm_MDP_VB_XX.m``, ``spm_MDP_VB_XXX.m``).
    """
    names = _block_names("block", len(q_blocks), block_names)
    leaves = {
        name: q.kl_to(targets[name])
        for name, q in zip(names, q_blocks, strict=True)
        if name in targets
    }
    return {"risk": leaves}


def cross_entropy_risk(
    q_blocks: tuple[Distribution, ...],
    targets: dict[str, Distribution],
    block_names: tuple[str, ...] | None = None,
) -> TermTree:
    """``−E_q log C(block)`` per designated block, keyed under ``risk``: a state target's score
    inside a branching planner. Linear in the belief, so averaged over a branch's outcomes it
    charges nothing for what the branch learns, where the KL would charge the information gained
    at every later step. Exact on a categorical or Gaussian belief against a Gaussian target, a
    Kolchinsky–Tracey pair where a mixture is involved. Design: per-variable-infogain.md V11
    decision 6."""
    names = _block_names("block", len(q_blocks), block_names)
    return {
        "risk": {
            name: _cross_entropy(q, targets[name])
            for name, q in zip(names, q_blocks, strict=True)
            if name in targets
        }
    }


def _cross_entropy(q: Distribution, target: Distribution) -> Interval:
    if isinstance(q, Categorical):
        p = q.probs
        return Interval.exact(-jnp.sum(jnp.where(p > 0, p * target.log_probs, 0.0), axis=-1))
    if isinstance(q, MixtureGaussian):
        return q.cross_entropy_to(target)
    if isinstance(q, Gaussian):
        if isinstance(target, Gaussian):
            return q.kl_to(target) + q.entropy()
        return MixtureGaussian.from_gaussian(q).cross_entropy_to(target)
    return q.kl_to(target) + q.entropy()


def flat_risk(
    q_blocks: tuple[Distribution, ...],
    block_names: tuple[str, ...] | None = None,
) -> TermTree:
    """``−H[q(o)]`` per observation block: the risk of an implicit flat preference, the
    preference's normaliser dropped (it is the same for every policy, and a flat preference over
    a continuous channel has none). A mixture predictive's Kolchinsky–Tracey pair propagates as
    the Interval. SPM's ``spm_MDP_VB_XXX`` charges the same ``−H[q(o)]`` on a modality with no
    preference (adding the policy-independent log of its outcome count, dropped here).
    Design: docs/design/per-variable-infogain.md V11 decision 2."""
    names = _block_names("block", len(q_blocks), block_names)
    leaves = {
        name: Interval.exact(jnp.sum(xlogy(q.probs, q.probs), axis=-1))
        if isinstance(q, Categorical)
        else -q.entropy()
        for name, q in zip(names, q_blocks, strict=True)
    }
    return {"risk": leaves}


def terminal_risk(
    final_beliefs: tuple[Distribution, ...],
    targets: dict[str, Distribution],
    block_names: tuple[str, ...] | None = None,
) -> TermTree:
    """Terminal-scope risk: the same KL machinery scored once against H designations.

    Terminal targets are the ``H`` of Friston et al. 2023 (*Active inference and intentional
    behaviour*, arXiv:2312.07547) and of SPM's ``spm_MDP_VB_XXX``. That routine scores
    ``KL(q(s)‖H)`` at every step of its search and adds ``spm_induction``'s constraints; this
    scores it once, at the horizon.
    """
    return {"terminal_risk": risk(final_beliefs, targets, block_names)["risk"]}


def terminal_cross_entropy_risk(
    final_beliefs: tuple[Distribution, ...],
    targets: dict[str, Distribution | ConditionalCategorical],
    h_deps: tuple[tuple[int, ...], ...],
    block_names: tuple[str, ...],
) -> TermTree:
    """Terminal-scope ``−E_q log H(block)`` per designated block, keyed ``terminal_risk``: a
    terminal target's score inside a branching planner, linear in the final belief as
    :func:`cross_entropy_risk` is. A conditional target (nonempty ``h_deps``) is scored by its
    negative expected log-probability under the mean-field final belief, ``−Σ q(s) Σ_pa Π q(pa)
    log h(s | pa)``, linear in each factor's belief, where the resolved target ``Σ_pa h(s | pa)
    q(pa)`` would charge for knowing the parents. Design: per-variable-infogain.md V11
    decision 8."""
    leaves = {}
    for f, name in enumerate(block_names):
        if name not in targets:
            continue
        q, target = final_beliefs[f], targets[name]
        if h_deps[f]:
            log_table = jax.nn.log_softmax(target.log_table, axis=-(target.parent_ndims + 1))
            joint = _target_probs(final_beliefs[h_deps[f][0]])
            for g in h_deps[f][1:]:
                joint = joint[..., None] * _target_probs(final_beliefs[g])
            # a parent configuration of zero belief adds nothing, a zero of h included
            weighted = jnp.where(joint > 0, joint * log_table, 0.0)
            expected = jnp.sum(weighted, axis=tuple(range(1, log_table.ndim)))
            p = _target_probs(q)
            leaves[name] = Interval.exact(-jnp.sum(jnp.where(p > 0, p * expected, 0.0), axis=-1))
        else:
            leaves[name] = _cross_entropy(q, target)
    return {"terminal_risk": leaves}


def _target_probs(q) -> Array:
    """A parent belief (a ``Categorical`` or a raw prob array) as a probability vector."""
    return q.probs if isinstance(q, Categorical) else jnp.asarray(q)


def resolve_conditional_targets(
    targets: dict[str, Distribution | ConditionalCategorical],
    h_deps: tuple[tuple[int, ...], ...],
    beliefs: tuple,
    block_names: tuple[str, ...],
) -> dict[str, Distribution]:
    """Mean-field resolution of conditional designations (``h_deps``).

    A block with nonempty ``h_deps[f]`` gets ``Σ_parents h(· | parents) q(parents)``
    evaluated at ``beliefs`` (distribution objects or raw prob vectors);
    unconditional entries pass through. Which beliefs to resolve against is the
    caller's choice (terminal risk uses the final predicted beliefs).
    """
    out: dict[str, Distribution] = dict(targets)
    for f, deps in enumerate(h_deps):
        name = block_names[f]
        if not deps or name not in out:
            continue
        cond = out[name]
        table = jax.nn.softmax(cond.log_table, axis=-(cond.parent_ndims + 1))
        parents = tuple(_target_probs(beliefs[g]) for g in deps)
        out[name] = Categorical(logits=jnp.log(contract_probs(table, parents, keep_child=True)))
    return out


#: A conditional per-step target's parents, as the planning closures read them: ``("control",
#: k)`` for control node column ``k``, ``("state", f)`` for state factor ``f`` (engine position).
TargetParents = tuple[tuple[str, int], ...]


def target_parent_entries(
    parents: "TargetParents", action: object, beliefs: tuple, layout: ControlLayout | None = None
) -> tuple:
    """What each parent of a conditional per-step target reads at one step, in ``parents``
    order: a control parent's entry of ``action`` (a level, or a probability vector), a state
    parent's belief as a probability vector. Each is an index :func:`take_controls` reads an
    axis of the target's table at."""
    return tuple(
        node_entry(action, i, layout) if kind == "control" else _target_probs(beliefs[i])
        for kind, i in parents
    )


def resolve_step_targets(
    targets: dict[str, Distribution],
    c_deps: "dict[str, TargetParents]",
    action: object,
    beliefs: tuple,
    layout: ControlLayout | None = None,
) -> dict[str, Distribution]:
    """The per-step targets at one step, the conditional ones (``c_deps``) resolved under their
    parents: ``Σ_parents C(· | parents) q(parents)``, where a control parent's ``q`` is the
    step's entry (a level, its one-hot, so the target is the level's own distribution; a
    probability vector of a soft :class:`~hybrid_aif.controls.ControlStep`, the mixture it
    weights) and a state parent's is the step's belief, the mean-field rule of
    :func:`resolve_conditional_targets`. A :class:`ConditionalCategorical` resolves to a
    :class:`Categorical`; a :class:`ConditionalGaussian` selects the level's ``(loc, chol)``
    (its parents are control nodes, read as levels: a probability vector over them would make it
    a mixture, and raises). Unconditional entries pass through. Cf. SPM's state-dependent
    preferences in ``spm_MDP_VB_XXX`` (``id.C``), which average ``log C`` under the parents'
    beliefs; this resolves the mixture ``Σ C q`` before the log, and also admits control
    parents. ``action`` is the step's control step and ``layout`` the flat layout of a mixed
    space (``None`` when the step is a level array or a ``ControlStep``).
    Design: docs/design/step-durations.md T5.
    """
    out: dict[str, Distribution] = dict(targets)
    for name, parents in c_deps.items():
        if name not in out:
            continue
        target = out[name]
        indices = target_parent_entries(parents, action, beliefs, layout)
        if isinstance(target, ConditionalGaussian):
            soft = [i for (kind, i), idx in zip(parents, indices, strict=True) if is_belief(idx)]
            if soft:
                raise ValueError(
                    f"the Gaussian target on {name!r} is conditional on control node column(s) "
                    f"{soft}, whose entries at this step are probability vectors; a Gaussian "
                    "target under a soft control step is a mixture per step, which is not "
                    "built: clamp the node's levels"
                )
            loc, chol = target.loc, target.chol
            for idx in reversed(indices):
                loc = jnp.take(loc, jnp.asarray(idx, jnp.int32), axis=-2)
                if not target.shared_chol:
                    chol = jnp.take(chol, jnp.asarray(idx, jnp.int32), axis=-3)
            out[name] = Gaussian(loc=loc, chol=chol)
        else:
            out[name] = Categorical(logits=take_controls(target.log_probs, indices, log=True))
    return out


def _mean_table_conditional_entropy(
    A_m: ConditionalCategorical | ConditionalDirichlet, parent_beliefs: tuple[Categorical, ...]
) -> Array:
    probs = jnp.exp(_point(A_m).log_probs)
    h = -xlogy(probs, probs).sum(axis=-(A_m.parent_ndims + 1))
    return _contract(h, parent_beliefs, keep_child=False)


def transition_occupancy_leaf(
    B_f: ConditionalCategorical | ConditionalDirichlet,
    parent_beliefs: tuple[Categorical | Array, ...],
    action_f: Array | tuple[Array, ...] | None,
) -> Array:
    """One factor's ``E_q(s) H[B(·|s,a)]`` for one step (the ``occupancy`` leaf).

    The clamped control's slice of the mean table, its child entropy per parent
    configuration, contracted against the previous step's parent beliefs
    (``Categorical`` or raw prob vectors). ``action_f`` is one level per trailing
    control axis (a scalar for one), ``None`` when the declaration has no control axis.
    """
    log_table = _point(B_f).log_probs
    if action_f is not None:
        log_table = take_controls(log_table, action_f, log=True)
    probs = jnp.exp(log_table)
    h = -xlogy(probs, probs).sum(axis=-(len(parent_beliefs) + 1))
    return contract_probs(h, tuple(_target_probs(q) for q in parent_beliefs), keep_child=False)


def selected_conditional_entropy(
    probs: Array, parents: tuple[Array, ...], selected: tuple[bool, ...]
) -> Array:
    """``E_q[H[o | S]]`` for a child-first table, ``S`` the parent axes ``selected`` keeps.

    ``probs`` is ``p(o | parents)`` with one parent axis per entry of ``parents`` (probability
    vectors). The unselected axes are contracted against their beliefs, giving ``p(o | S)``
    exactly under the mean-field belief; its child entropy is then averaged over ``q(S)``.
    Raises :class:`ValueError` for a mask selecting no parent.
    """
    _check_mask(selected, len(parents))
    subs = _AXES[: probs.ndim]
    axes = tuple(zip(subs[1:], parents, selected, strict=True))
    kept = "".join(ax for ax, _, k in axes if k)
    dropped = tuple((ax, q) for ax, q, k in axes if not k)
    spec = ",".join([subs, *(ax for ax, _ in dropped)]) + "->" + subs[0] + kept
    p_cond = jnp.einsum(spec, probs, *(q for _, q in dropped))
    h = -xlogy(p_cond, p_cond).sum(axis=0)
    return contract_probs(h, tuple(q for _, q, k in axes if k), keep_child=False)


def _full(selected: tuple[bool, ...] | None) -> bool:
    """Whether a channel's selection mask covers its whole dependency set (``None`` = all)."""
    return selected is None or all(selected)


def _check_mask(selected: tuple[bool, ...], num_deps: int) -> None:
    """A partial mask must have one entry per dependency and select at least one of them."""
    if len(selected) != num_deps:
        raise ValueError(
            f"selection mask {selected} has {len(selected)} entries for a channel with "
            f"{num_deps} dependencies"
        )
    if not any(selected):
        raise ValueError(
            f"selection mask {selected} selects none of the channel's parents: an empty "
            "selection has no information to score (a System never mounts such a channel)"
        )


def _channel_conditional_entropy(
    A_m: ConditionalCategorical | ConditionalDirichlet,
    parent_beliefs: tuple[Categorical, ...],
    selected: tuple[bool, ...] | None,
) -> Array:
    if _full(selected):
        return _mean_table_conditional_entropy(A_m, parent_beliefs)
    probs = jnp.exp(_point(A_m).log_probs)
    return selected_conditional_entropy(probs, tuple(b.probs for b in parent_beliefs), selected)


def ambiguity(
    qs: tuple[Categorical, ...],
    A: tuple[ConditionalCategorical | ConditionalDirichlet, ...],
    A_deps: tuple[tuple[int, ...], ...],
    block_names: tuple[str, ...] | None = None,
    selected: tuple[tuple[bool, ...] | None, ...] | None = None,
) -> TermTree:
    """E_q(s)[H[A(·|s)]] per modality: conditional entropy contracted over the CPD table.

    A learnable modality contributes its mean-table entropy. ``selected[i]`` masks modality
    ``i``'s parents (``A_deps`` order; ``None`` = all); a partial mask scores ``E_q[H[o | S]]``
    with the unselected parents marginalised, the ambiguity that pairs with
    :func:`state_info_gain` under the same mask. The per-configuration entropy table contracted
    with ``q(s)`` is SPM's ambiguity term (``H`` in ``spm_MDP_VB_XX.m``, ``spm_hnorm`` in
    ``spm_MDP_VB_XXX.m``); with :func:`risk` it is the risk-plus-ambiguity form of G those
    routines evaluate (Da Costa et al. 2020).
    """
    names = _block_names("modality", len(A), block_names)
    masks = selected if selected is not None else (None,) * len(A)
    leaves = {
        name: Interval.exact(_channel_conditional_entropy(A_m, tuple(qs[i] for i in deps), sel))
        for name, A_m, deps, sel in zip(names, A, A_deps, masks, strict=True)
    }
    return {"ambiguity": leaves}


def state_info_gain(
    qs: tuple[Categorical, ...],
    A: tuple[ConditionalCategorical | ConditionalDirichlet, ...],
    A_deps: tuple[tuple[int, ...], ...],
    block_names: tuple[str, ...] | None = None,
    selected: tuple[tuple[bool, ...] | None, ...] | None = None,
) -> TermTree:
    """``I(o; S) = H[q(o)] − E_q[H[o | S]]`` per modality (pymdp's ``control.compute_info_gain``).

    The ``predictive`` state estimator: one ``infogain`` leaf per observing channel, scored
    through the mean table. By default ``S`` is the channel's whole dependency set. ``selected[i]``
    masks modality ``i``'s parents in ``A_deps`` order, and the unselected ones are marginalised
    under their beliefs (exact). SPM's ``spm_MDP_G`` scores the joint outcome across modalities;
    the per-modality sum is pymdp's, and it bounds the joint from above.
    Design: docs/design/per-variable-infogain.md V9
    """
    names = _block_names("modality", len(A), block_names)
    masks = selected if selected is not None else (None,) * len(A)
    leaves = {}
    for name, A_m, deps, sel in zip(names, A, A_deps, masks, strict=True):
        parent_beliefs = tuple(qs[i] for i in deps)
        p_o = _point(A_m).average(parent_beliefs).probs
        h_qo = -xlogy(p_o, p_o).sum(axis=-1)
        leaves[name] = Interval.exact(h_qo - _channel_conditional_entropy(A_m, parent_beliefs, sel))
    return {INFOGAIN: leaves}


def param_info_gain(
    A: tuple[ConditionalDirichlet, ...],
    qs: tuple[Categorical, ...],
    q_obs: tuple[Categorical, ...],
    A_deps: tuple[tuple[int, ...], ...],
    block_names: tuple[str, ...] | None = None,
) -> TermTree:
    """Expected Dirichlet information gain about the likelihood counts, per modality.

    ``E_q(o) E_q(s)[wA(o|s)]`` with the declarations' exact ``spm_wnorm``
    ``novelty()`` weights (pymdp's ``calc_negative_pA_info_gain``, positive sign): the average is
    pymdp's product-of-marginals form, not the exact expectation under the joint; the weights are
    exact. pymdp's form is SPM's likelihood-novelty term in ``spm_MDP_VB_X``/``spm_MDP_VB_XXX``
    (``toolbox/DEM``), the same average over ``q(o)`` and ``q(s)``.
    ``A`` holds the learnable declarations; ``block_names`` are modality names,
    and leaves are keyed ``"likelihood:<modality>"``.
    """
    names = _block_names("modality", len(A), block_names)
    leaves = {}
    for name, A_m, q_o, deps in zip(names, A, q_obs, A_deps, strict=True):
        fd = _contract(A_m.novelty(), tuple(qs[i] for i in deps), keep_child=True)
        leaves[param_key("likelihood", name)] = Interval.exact(jnp.sum(q_o.probs * fd, axis=-1))
    return {INFOGAIN: leaves}


def transition_novelty_leaf(
    B_f: ConditionalDirichlet,
    q_next_f: Categorical | Array,
    q_prev_parents: tuple[Categorical | Array, ...],
    action_f: Array | tuple[Array, ...] | None,
) -> Array:
    """One factor's transition parameter info gain for one step.

    ``E_q(s_t) E_q(s_{t-1})[wB(s_t | s_{t-1}, u)]`` with the clamped control's slice
    of the ``novelty()`` weights (``action_f`` one level per trailing control axis, a
    scalar for one; ``None`` when there is no control axis); pymdp's
    ``calc_negative_pB_info_gain``, positive sign. Beliefs may be ``Categorical`` or raw
    prob vectors. After SPM's transition-novelty term in ``spm_MDP_VB_X`` (``wB``) and
    ``spm_MDP_VB_XXX`` (``I``).
    """
    w = B_f.novelty()
    if action_f is not None:
        w = take_controls(w, action_f)
    probs = tuple(b.probs if isinstance(b, Categorical) else b for b in (q_next_f, *q_prev_parents))
    subs = _AXES[: w.ndim]
    fd = jnp.einsum(",".join([subs, *subs[1:]]) + "->" + subs[0], w, *probs[1:])
    return jnp.sum(probs[0] * fd, axis=-1)


def transition_param_info_gain(
    B: tuple[ConditionalDirichlet, ...],
    qs_next: tuple[Categorical, ...],
    qs_prev: tuple[Categorical, ...],
    action: Array,
    factor_idx: tuple[int, ...],
    B_deps: tuple[tuple[int, ...], ...],
    block_names: tuple[str, ...] | None = None,
    control_deps: ControlDeps | None = None,
) -> TermTree:
    """Expected Dirichlet information gain about the transition counts.

    ``B``/``B_deps``/``block_names`` are aligned over the learnable transitions
    scored; ``factor_idx`` gives each one's index into the belief tuples and
    ``action``; ``control_deps`` is the System-wide ``B_control_deps``. Only the clamped
    control's slice is scored. Leaves are keyed ``"transition:<factor>"``.
    Design: docs/design/parameter-learning.md D6
    """
    names = _block_names("factor", len(B), block_names)
    leaves = {}
    for name, B_f, f, deps in zip(names, B, factor_idx, B_deps, strict=True):
        has_control = B_f.parent_ndims == len(deps) + num_control_axes(control_deps, f)
        a_f = control_read(action, f, control_deps) if has_control else None
        prev = tuple(qs_prev[g] for g in deps)
        leaves[param_key("transition", name)] = Interval.exact(
            transition_novelty_leaf(B_f, qs_next[f], prev, a_f)
        )
    return {INFOGAIN: leaves}
