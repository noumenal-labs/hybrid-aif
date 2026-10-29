"""Explicit-enumeration planner: exhaustive policy search over a discrete action grid.

Policies are the full Cartesian product of control levels over the horizon,
built in pure Python and memoised on the grid's static structure (mirroring
pymdp's ``construct_policies``). The planner scores the grid as a tree of shared
action prefixes (:mod:`hybrid_aif.planning.tree`); :func:`evaluate_policies`
scores an arbitrary policy set by one scan per policy. Both push beliefs
through an injected ``transition_fn`` and score each step with an injected
``step_terms`` evaluator returning a :data:`TermTree`; the planner never
imports concrete objective terms. Composition follows the design doc:
``q_pi ∝ softmax(-gamma * G + log E)``.
"""

import functools
import itertools
from typing import Literal

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax import Array

from hybrid_aif.controls import ControlStep
from hybrid_aif.distributions import Categorical
from hybrid_aif.interval import Interval
from hybrid_aif.objectives.base import TermTree, total
from hybrid_aif.planning.base import (
    Beliefs,
    PlannerContext,
    StepTermsFn,
    TerminalTermsFn,
    TransitionFn,
    UseBound,
    refuse_mixed,
    register_planner,
)


def enumerate_policies(
    num_controls: tuple[int, ...],
    policy_len: int,
    controlled: tuple[bool, ...] | None = None,
) -> Array:
    """Exhaustive policy matrix, int32 ``(num_policies, policy_len, len(num_controls))``.

    One column per entry of ``num_controls`` (a System passes one per control
    node). Ordering matches pymdp's ``construct_policies``: time-major, the last
    column at the last timestep varying fastest. A column ``controlled`` marks
    ``False`` contributes only level 0. Memoised on its arguments.
    """
    num_controls = tuple(int(n) for n in num_controls)
    if controlled is None:
        controlled = tuple(n > 1 for n in num_controls)
    grid = _enumerate_policies(num_controls, int(policy_len), tuple(bool(c) for c in controlled))
    return jnp.asarray(grid)


@functools.cache
def _enumerate_policies(
    num_controls: tuple[int, ...], policy_len: int, controlled: tuple[bool, ...]
) -> np.ndarray:
    levels = [n if c else 1 for n, c in zip(num_controls, controlled, strict=True)]
    flat = list(itertools.product(*([range(n) for n in levels] * policy_len)))
    return np.asarray(flat, dtype=np.int32).reshape(len(flat), policy_len, len(num_controls))


def _merge_trees(base: TermTree, extra: TermTree) -> TermTree:
    out = {term: dict(blocks) for term, blocks in base.items()}
    for term, blocks in extra.items():
        slot = out.setdefault(term, {})
        for block, iv in blocks.items():
            slot[block] = slot[block] + iv if block in slot else iv
    return out


def evaluate_policies(
    policies: Array,
    init_qs: Beliefs,
    transition_fn: TransitionFn,
    step_terms: StepTermsFn,
    terminal_terms: TerminalTermsFn | None = None,
) -> TermTree:
    """Score every policy: scan beliefs over the horizon, accumulate step TermTrees.

    Structure after pymdp's ``control.compute_neg_efe_policy`` and ``update_posterior_policies``
    (pymdp/control.py): one ``lax.scan`` per policy over the horizon, vmapped over the policy
    matrix.

    Per step the beliefs are pushed through ``transition_fn(qs, action_t)`` and
    the post-transition beliefs scored by ``step_terms(qs_next, action_t, t)``;
    leaves are summed over time and ``terminal_terms(qs_final)`` is merged in once.
    Every leaf carries a leading per-policy axis. Action-type-agnostic: a float
    ``(P, H, action_dim)`` set of continuous sequences scores the same way, and so does a
    pytree of ``(P, H, ...)`` leaves (a :class:`~hybrid_aif.controls.ControlStep`).
    """
    policy_len = jtu.tree_leaves(policies)[0].shape[1]

    def eval_one(policy: Array) -> TermTree:
        def body(qs: Beliefs, xs: tuple[Array, Array]) -> tuple[Beliefs, TermTree]:
            action, t = xs
            qs_next = transition_fn(qs, action)
            return qs_next, step_terms(qs_next, action, t)

        ts = jnp.arange(policy_len, dtype=jnp.int32)
        qs_final, stacked = jax.lax.scan(body, init_qs, (policy, ts))
        tree = jtu.tree_map(lambda x: jnp.sum(x, axis=0), stacked)
        if terminal_terms is not None:
            tree = _merge_trees(tree, terminal_terms(qs_final))
        return tree

    return jax.vmap(eval_one)(policies)


def legal_logits(
    legal: "tuple | list | Array | None",
    num_controls: tuple[int, ...],
    policy_len: int,
    controlled: tuple[bool, ...],
    *,
    policies: Array | None = None,
    names: tuple[str, ...] | None = None,
    face: str | None = None,
) -> Array | None:
    """Expand a ``legal=`` mask to ``log 1[legal(pi)]`` over the enumerated policy set.

    ``legal`` is either a tuple with one entry per control node (policy column) — ``None``
    (all legal), a bool ``(num_controls[k],)`` masking the first step, or a bool
    ``(policy_len, num_controls[k])`` masking per step — or a bool ``(P,)`` array over the
    policies. Returns ``0.0``/``-inf`` per policy, or ``None`` for ``legal=None``.
    ``policies`` is the set a planner returned (``None`` enumerates the grid). ``names``
    labels the entries in the sentences (the System's keyed ``legal=`` in node order), and
    ``face`` prefixes them. A concrete mask leaving no legal policy (or no legal level for some
    entry and step) raises. Design: docs/design/action-legality.md L1
    """
    if legal is None:
        return None
    lead = "" if face is None else f"{face}: "
    if policies is None:
        policies = enumerate_policies(num_controls, policy_len, controlled)  # (P, H, K)
    n_pol, horizon = policies.shape[:2]
    if isinstance(legal, tuple):
        if len(legal) != len(num_controls):
            raise ValueError(
                f"{lead}legal= has {len(legal)} entries for {len(num_controls)} control node(s); "
                "pass one entry per node (None for a node with no constraint), or one bool per "
                "policy"
            )
        ok = jnp.ones((n_pol,), bool)
        pol_np = _concrete(policies)
        ok_np = None if pol_np is None else np.ones((n_pol,), bool)
        for k, entry in enumerate(legal):
            if entry is None:
                continue
            label = f"legal[{k}]" if names is None else f"legal[{names[k]!r}]"
            noun = f"control node {k}" if names is None else f"control node {names[k]!r}"
            m_np = _concrete(entry)
            if m_np is None:
                ok_np = None
            m = jnp.asarray(entry, bool)
            n = int(num_controls[k])
            if m.shape == (n,):
                _refuse_all_illegal(m_np, lead + label, noun, per_step=False)
                ok = ok & m[policies[:, 0, k]]
                if ok_np is not None:
                    ok_np &= m_np[pol_np[:, 0, k]]
            elif m.shape == (horizon, n):
                _refuse_all_illegal(m_np, lead + label, noun, per_step=True)
                steps = jnp.arange(horizon, dtype=jnp.int32)[None, :]
                ok = ok & jnp.all(m[steps, policies[:, :, k]], axis=-1)
                if ok_np is not None:
                    ok_np &= np.all(m_np[np.arange(horizon)[None, :], pol_np[:, :, k]], axis=-1)
            else:
                raise ValueError(
                    f"{lead}{label} has shape {tuple(m.shape)}; {noun} takes ({n},) to mask this "
                    f"tick's levels, or ({horizon}, {n}) to mask per step of the plan"
                )
    else:
        ok_np = _concrete(legal)
        ok = jnp.asarray(legal, bool)
        if ok.shape != (n_pol,):
            raise ValueError(
                f"{lead}legal= as a bare array is one bool per enumerated policy, shape "
                f"({n_pol},); got {tuple(ok.shape)}. A mask over a control node's levels is "
                "keyed by the node"
            )
    if ok_np is not None and not ok_np.any():
        raise ValueError(
            f"{lead}legal= leaves no legal policy: every enumerated policy is masked, so there is "
            "nothing to select. At least one policy must be legal"
        )
    return jnp.where(ok, 0.0, -jnp.inf)


def legal_steps(
    legal: tuple,
    num_controls: tuple[int, ...],
    policy_len: int,
    actions: Array,
    *,
    names: tuple[str, ...] | None = None,
    face: str | None = None,
) -> Array:
    """A per-node ``legal=`` tuple as a bool mask over the first-step ``actions`` (int32
    ``(A, K)``): ``[t, a]`` is true when every node's level in action ``a`` is legal at step
    ``t``. The mask is ``(policy_len, A)`` when some entry masks per step, and ``(1, A)`` (the
    first step alone) when none does.

    Each entry is ``None`` (every level legal), a bool ``(num_controls[k],)`` or
    ``(1, num_controls[k])`` masking the first step alone, or a bool ``(policy_len,
    num_controls[k])`` masking each step. ``names`` and ``face`` label the sentences as in
    :func:`legal_logits`. A concrete entry leaving no legal level at some step raises. Design:
    docs/design/sophisticated-memory.md (SM3)
    """
    lead = "" if face is None else f"{face}: "
    if len(legal) != len(num_controls):
        raise ValueError(
            f"{lead}legal= has {len(legal)} entries for {len(num_controls)} control node(s); "
            "pass one entry per node (None for a node with no constraint), or one bool per "
            "policy"
        )
    n_act = actions.shape[0]
    shapes = [jnp.shape(e) for e in legal if e is not None]
    rows = policy_len if any(len(sh) == 2 and sh[0] == policy_len for sh in shapes) else 1
    ok = jnp.ones((rows, n_act), bool)
    for k, entry in enumerate(legal):
        if entry is None:
            continue
        label = f"legal[{k}]" if names is None else f"legal[{names[k]!r}]"
        noun = f"control node {k}" if names is None else f"control node {names[k]!r}"
        m_np = _concrete(entry)
        m = jnp.asarray(entry, bool)
        n = int(num_controls[k])
        if m.shape == (n,) or m.shape == (1, n):
            _refuse_all_illegal(m_np, lead + label, noun, per_step=False)
            first = m.reshape(n)[actions[:, k]]
            ok = ok.at[0].set(ok[0] & first)
        elif m.shape == (policy_len, n):  # (rows is then policy_len)
            _refuse_all_illegal(m_np, lead + label, noun, per_step=True)
            ok = ok & m[:, actions[:, k]]
        else:
            raise ValueError(
                f"{lead}{label} has shape {tuple(m.shape)}; {noun} takes ({n},) to mask this "
                f"tick's levels, or ({policy_len}, {n}) to mask per step of the plan"
            )
    return ok


def _concrete(x: object) -> np.ndarray | None:
    """``x`` as a numpy array when its value is known at trace time (a Python value, a numpy
    array or a concrete JAX array, closed over or not), ``None`` when it is traced."""
    if isinstance(x, jax.core.Tracer):
        return None
    return np.asarray(x)


def _refuse_all_illegal(m: np.ndarray | None, label: str, noun: str, *, per_step: bool) -> None:
    """Raise when a concrete mask leaves no legal level at some step; a traced mask is not
    checked."""
    if m is None:
        return
    m = np.asarray(m, bool)
    rows = m if per_step else m[None, :]
    dead = np.flatnonzero(~rows.any(axis=-1))
    if dead.size:
        raise ValueError(
            f"{label} masks every level of {noun} at step(s) {dead.tolist()}; at least one "
            "level must stay legal at every masked step"
        )


def policy_posterior(
    G: Interval,
    gamma: float,
    policy_prior: Array | Categorical | None = None,
    use_bound: Literal["midpoint", "lower", "upper"] = "midpoint",
    legal: Array | None = None,
) -> Categorical:
    """Policy posterior ``q_pi ∝ softmax(-gamma * G + log E + log 1[legal])``.

    Each policy is scored by a point of the interval ``G``: ``"midpoint"``
    (default), ``"lower"`` (optimistic) or ``"upper"`` (pessimistic).
    ``policy_prior`` is E over the policy set, a log-space array or a Categorical.
    ``legal`` is :func:`legal_logits`' output, added after ``log E`` and never
    touching ``G``. The System faces apply the mask themselves, after the planner
    returns. The composition is pymdp's ``control.update_posterior_policies`` (pymdp/control.py),
    in G's natural sign; pymdp's values are the parity oracle (tests/test_planning.py).
    pymdp's composition is SPM's prior over policies in ``spm_MDP_VB_X``
    (``spm_softmax(qE + w·Q)``). The precision γ is fixed here, where SPM can also update it under
    a Gamma prior; E is not learned.
    Design: docs/design/action-legality.md L2
    """
    g = {"midpoint": G.midpoint, "lower": G.lower, "upper": G.upper}[use_bound]
    logits = -gamma * g
    if policy_prior is not None:
        logits = logits + _prior_logits(policy_prior)
    if legal is not None:
        logits = logits + legal  # the hard constraint, after E and never inside it
    return Categorical(logits=logits)


def _prior_logits(
    policy_prior: "Array | Categorical | object", planner: str = "enumeration"
) -> Array:
    """E as one log-weight per enumerated policy; a policy module raises, naming ``planner``.

    A module E acts through the ``"habit"`` planner. Design:
    docs/design/amortization-debt-ledger.md entry 12
    """
    if isinstance(policy_prior, Categorical):
        return policy_prior.log_probs
    if not hasattr(policy_prior, "shape") and callable(policy_prior):
        raise NotImplementedError(
            f"policy_prior holds a policy module, and the {planner} planner would need "
            "E(pi) = prod_t pi(a_t | s_t) — the module evaluated at every step of every "
            "enumerated policy, inside the scoring scan (design: amortization.md A3). "
            f'The {planner} planner does not accept one: plan with the "habit" planner, which '
            "acts from the module directly, or declare a vector E over the policy set."
        )
    return jnp.asarray(policy_prior)


def evaluate_policies_total(
    policies: Array,
    init_qs: Beliefs,
    transition_fn: TransitionFn,
    step_terms: StepTermsFn,
    terminal_terms: TerminalTermsFn | None = None,
) -> Interval:
    """Score every policy straight to its G Interval, accumulated in the scan carry.

    The metrics-free sibling of :func:`evaluate_policies`, which carries ``(qs, G)`` as pymdp's
    ``control.compute_neg_efe_policy`` scan does (pymdp/control.py): each step's tree is
    folded with :func:`~hybrid_aif.objectives.base.total` at once, so no leaves
    are stacked; equal up to float reassociation. An empty step tree (a
    terminal-only System) skips the per-step fold. The accumulator is seeded with
    the step tree's static ``derived`` flag (:func:`_step_warrant`), keeping the
    scan carry's structure fixed. Like :func:`evaluate_policies` it takes a pytree of ``(P, H,
    ...)`` leaves (a :class:`~hybrid_aif.controls.ControlStep`) as the policy set.
    """

    warrant = _step_warrant(init_qs, transition_fn, step_terms, policies)

    def eval_one(policy: Array) -> Interval:
        def body(carry: tuple[Beliefs, Interval], xs: tuple[Array, Array]):
            qs, g = carry
            action, t = xs
            qs_next = transition_fn(qs, action)
            tree = step_terms(qs_next, action, t)
            return (qs_next, (g + total(tree)) if tree else g), None

        ts = jnp.arange(jtu.tree_leaves(policies)[0].shape[1], dtype=jnp.int32)
        init = (init_qs, Interval(jnp.zeros(()), jnp.zeros(()), warrant))
        (qs_final, g), _ = jax.lax.scan(body, init, (policy, ts))
        if terminal_terms is not None:
            g = g + total(terminal_terms(qs_final))
        return g

    return jax.vmap(eval_one)(policies)


def _step_warrant(
    init_qs: Beliefs, transition_fn: TransitionFn, step_terms: StepTermsFn, policies: Array
) -> bool:
    """Whether one step's folded G is a *derived* quantity, read statically.

    Uses ``jax.eval_shape``, so no computation is emitted. An empty step tree is
    vacuously derived.
    """

    def one_step(qs, action, t):
        return step_terms(transition_fn(qs, action), action, t)

    spec = jax.eval_shape(
        one_step, init_qs, jtu.tree_map(lambda x: x[0, 0], policies), jnp.int32(0)
    )
    return all(iv.derived for blocks in spec.values() for iv in blocks.values())


@register_planner("enumeration")
def enumeration_planner(
    ctx: PlannerContext, use_bound: UseBound = "midpoint", metrics: bool = True
) -> tuple[Categorical, TermTree | None, Array]:
    """The :class:`~hybrid_aif.planning.base.Planner` face of this module.

    Registered as ``"enumeration"`` (the System default). Scores the grid as a
    tree of shared prefixes (:mod:`hybrid_aif.planning.tree`) and composes
    :func:`policy_posterior`.
    """
    from hybrid_aif.planning.tree import evaluate_policy_tree, evaluate_policy_tree_total

    refuse_mixed(ctx, "enumeration")
    if ctx.continuous_controls is not None:
        raise ValueError(
            "the enumeration planner cannot serve a continuous action space: there is no "
            "finite policy grid to enumerate (design: amortization.md A9). Plan with the "
            '"gradient" or "cem" planner — System(planner="cem"), or '
            "infer_policies(planner=make_cem_planner(...)) with a plan_key."
        )
    args = (
        ctx.num_controls,
        ctx.controlled,
        ctx.policy_len,
        ctx.init_beliefs,
        ctx.transition_fn,
        ctx.step_terms_fn,
        ctx.terminal_terms_fn,
    )
    if metrics:
        tree, policies = evaluate_policy_tree(*args)
        G = total(tree)
    else:
        tree = None
        G, policies = evaluate_policy_tree_total(*args)
    q_pi = policy_posterior(G, ctx.gamma, ctx.policy_prior, use_bound=use_bound)
    return q_pi, tree, policies


def action_marginals(
    q_pi: Categorical,
    policies: Array,
    num_controls: tuple[int, ...],
) -> tuple[Categorical, ...]:
    """First-step marginal posterior over each policy column's levels (pymdp
    ``get_marginals``); ``num_controls`` gives each column's level count."""
    probs = q_pi.probs
    first = policies[:, 0, :]
    out = []
    for f, n in enumerate(num_controls):
        mask = (first[:, f] == jnp.arange(n, dtype=jnp.int32)[:, None]).astype(probs.dtype)
        out.append(Categorical(logits=jnp.log(jnp.einsum("...p,ap->...a", probs, mask))))
    return tuple(out)


def sample_action(
    key: Array,
    q_pi_or_marginals: Categorical | tuple[Categorical, ...],
    policies: Array | None = None,
    num_controls: tuple[int, ...] | None = None,
    mode: Literal["marginal", "policy"] = "marginal",
    selection: Literal["stochastic", "deterministic"] = "stochastic",
    alpha: float | None = None,
) -> Array:
    """Select an action from a policy posterior: a control level per control node, or a vector.

    After pymdp's ``control.sample_action`` and ``control.sample_policy`` (pymdp/control.py): the
    marginal and policy modes, deterministic and stochastic selection, and the ``alpha``
    precision on log-probabilities follow pymdp's ``Agent.sample_action``; the ``alpha``-sharpened
    action marginal is SPM's action selection in ``spm_MDP_VB_X``.

    ``mode="marginal"`` acts from the first-step action marginals (given as a
    tuple, or computed from ``q_pi``, which then requires ``policies`` and
    ``num_controls``); ``mode="policy"`` from a policy drawn from ``q_pi``.
    ``selection="deterministic"`` takes the argmax (ties: lowest index; ``key``
    unused). ``alpha`` sharpens marginal logits before stochastic selection. Over
    continuous actions only ``mode="policy"`` is defined; it returns the sequence's
    first step, dtype preserved, and ``"marginal"`` raises. A mixed policy set, a
    :class:`~hybrid_aif.controls.ControlStep` with leading axes ``(P, policy_len)``, takes
    ``mode="policy"`` and returns the drawn policy's first step as a ``ControlStep`` with one
    entry per control node.
    """
    stepped = isinstance(policies, ControlStep)
    continuous = (
        policies is not None
        and not stepped
        and not jnp.issubdtype(jnp.asarray(policies).dtype, jnp.integer)
    )

    def _pick(dist: Categorical, k: Array) -> Array:
        if selection == "deterministic":
            return jnp.argmax(dist.log_probs, axis=-1).astype(jnp.int32)
        logits = dist.log_probs if alpha is None else alpha * dist.log_probs
        return Categorical(logits=logits).sample(k)

    if stepped and mode != "policy":
        raise ValueError(
            "a mixed policy set (a ControlStep) pairs each discrete policy with its own "
            'continuous controls, so an action is drawn from one policy: select mode="policy"'
        )
    if mode == "policy":
        if not isinstance(q_pi_or_marginals, Categorical) or policies is None:
            raise ValueError('mode="policy" requires a q_pi Categorical and policies')
        index = _pick(q_pi_or_marginals, key)
        if stepped:
            return ControlStep(entries=tuple(entry[index, 0] for entry in policies.entries))
        chosen = policies[index, 0]
        return chosen if continuous else chosen.astype(jnp.int32)
    if mode == "marginal":
        if continuous:
            raise ValueError(
                'mode="marginal" over a continuous action space: action marginals are a '
                "distribution over control *levels*, of which a continuous control has none. "
                'Select mode="policy" (act from a sequence drawn from q_pi), or take the '
                "planner's own recommendation (planning.cem.mean_action for CEM's "
                "receding-horizon mean)."
            )
        marginals = q_pi_or_marginals
        if isinstance(marginals, Categorical):
            if policies is None or num_controls is None:
                raise ValueError("computing marginals from q_pi requires policies and num_controls")
            marginals = action_marginals(marginals, policies, num_controls)
        if not marginals:  # nothing to choose: the empty action
            return jnp.zeros((0,), jnp.int32)
        keys = jax.random.split(key, len(marginals))
        return jnp.stack([_pick(m, k) for m, k in zip(marginals, keys, strict=True)]).astype(
            jnp.int32
        )
    raise ValueError(f"unknown mode {mode!r}")
