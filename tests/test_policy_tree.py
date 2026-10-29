"""Prefix-sharing evaluation of the policy tree (design: system.md addendum, E1–E6).

The enumeration planner scores the policy grid as a tree of shared action
prefixes (``planning.tree``). Every claim here is pinned against the scan
evaluators it replaced, which stay in ``planning.enumerate`` as the reference:
leaf-wise term-tree parity and G parity on every fixture family, the leaf order
against ``enumerate_policies``, the planner's posterior end to end (with a
legality mask and a habit prior), and the count of transitions actually
evaluated — ``b·(b^H − 1)/(b − 1)`` against the scan's ``H·b^H``.

Tolerance is rtol = atol = 1e-6: float reassociation is the Planner protocol's
allowance, and the two evaluators associate the same sums differently.
"""

import dataclasses
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest

from hybrid_aif.interval import Interval
from hybrid_aif.objectives import total
from hybrid_aif.planning.enumerate import (
    _enumerate_policies,
    enumerate_policies,
    evaluate_policies,
    evaluate_policies_total,
    policy_posterior,
)
from hybrid_aif.planning.tree import (
    evaluate_policy_tree,
    evaluate_policy_tree_total,
    expand_level,
)
from tests.test_compiled_parity import synthetic_system
from tests.test_hybrid_lg_system import GOAL
from tests.test_hybrid_lg_system import make_params as lg_params
from tests.test_hybrid_lg_system import make_system as lg_system
from tests.test_hybrid_switching_system import initial_beliefs as switching_beliefs
from tests.test_hybrid_switching_system import make_params as switching_params
from tests.test_hybrid_switching_system import make_system as switching_system
from tests.test_system import _initial_beliefs, tmaze_system

TOL = 1e-6


def old_enumeration_planner(ctx, use_bound="midpoint", metrics=True):
    """The enumeration planner before the tree fold: one scan per policy (the reference)."""
    policies = enumerate_policies(ctx.num_controls, ctx.policy_len, ctx.controlled)
    args = (policies, ctx.init_beliefs, ctx.transition_fn, ctx.step_terms_fn, ctx.terminal_terms_fn)
    if metrics:
        tree = evaluate_policies(*args)
        G = total(tree)
    else:
        tree = None
        G = evaluate_policies_total(*args)
    return policy_posterior(G, ctx.gamma, ctx.policy_prior, use_bound=use_bound), tree, policies


def _assert_interval(got: Interval, ref: Interval, name: str) -> float:
    assert got.derived == ref.derived, name
    assert got.lower.shape == ref.lower.shape == ref.upper.shape == got.upper.shape, name
    np.testing.assert_allclose(np.asarray(got.lower), np.asarray(ref.lower), rtol=TOL, atol=TOL)
    np.testing.assert_allclose(np.asarray(got.upper), np.asarray(ref.upper), rtol=TOL, atol=TOL)
    return max(
        float(np.max(np.abs(np.asarray(got.lower) - np.asarray(ref.lower)))),
        float(np.max(np.abs(np.asarray(got.upper) - np.asarray(ref.upper)))),
    )


def assert_tree_parity(ctx) -> float:
    """Both tree evaluators against both scan evaluators; returns the largest deviation."""
    policies = enumerate_policies(ctx.num_controls, ctx.policy_len, ctx.controlled)
    closures = (ctx.init_beliefs, ctx.transition_fn, ctx.step_terms_fn, ctx.terminal_terms_fn)
    ref_tree = evaluate_policies(policies, *closures)
    ref_g = evaluate_policies_total(policies, *closures)
    tree, pols = evaluate_policy_tree(ctx.num_controls, ctx.controlled, ctx.policy_len, *closures)
    g, pols_g = evaluate_policy_tree_total(
        ctx.num_controls, ctx.controlled, ctx.policy_len, *closures
    )
    assert np.array_equal(np.asarray(pols), np.asarray(policies))
    assert np.array_equal(np.asarray(pols_g), np.asarray(policies))
    assert {t: set(b) for t, b in tree.items()} == {t: set(b) for t, b in ref_tree.items()}
    worst = 0.0
    for term, blocks in ref_tree.items():
        for block, iv in blocks.items():
            worst = max(worst, _assert_interval(tree[term][block], iv, f"{term}/{block}"))
    worst = max(worst, _assert_interval(total(tree), total(ref_tree), "total"))
    worst = max(worst, _assert_interval(g, ref_g, "G"))
    return worst


# ----------------------------------------------------------------------- fixtures


@pytest.mark.parametrize("epistemic", [True, False])
@pytest.mark.parametrize("policy_len", [1, 2, 3, 4])
def test_tmaze_parity(policy_len, epistemic):
    system, params = tmaze_system(policy_len=policy_len, epistemic=epistemic)
    beliefs = _initial_beliefs(system, params)
    assert_tree_parity(system.planner_context(params, beliefs))


def test_switching_mixture_parity():
    """Hybrid carry: a Categorical and a MixtureGaussian pytree through the tree."""
    system, params = switching_system(), switching_params()
    beliefs = switching_beliefs()
    if not isinstance(beliefs, Mapping):
        beliefs = system.from_positional("beliefs", beliefs)
    ctx = system.planner_context(params, beliefs)
    assert_tree_parity(ctx)


def test_lg_terminal_parity():
    """A Gaussian carry with terminal terms merged at the leaves."""
    system = lg_system(policy_len=3)
    params = dataclasses.replace(lg_params(), terminal_targets={"state": GOAL})
    ctx = system.planner_context(params, dict(params.initial_state_prior))
    assert ctx.terminal_terms_fn is not None
    assert_tree_parity(ctx)


def test_carry_prev_parity():
    """A learnable transition's novelty reads the previous beliefs: the (qs, qs_prev) carry."""
    system, params, beliefs = synthetic_system("utility_infogain", ("f1", "transition:f1"))
    ctx = system.planner_context(params, system.from_positional("beliefs", beliefs))
    qs, qs_prev = ctx.init_beliefs  # the pair _carry_prev built
    assert isinstance(qs, tuple) and len(qs) == len(qs_prev) == 2
    assert ctx.terminal_terms_fn is not None
    assert_tree_parity(ctx)


# ---------------------------------------------------------------- leaf order (E2)


def _synthetic_closures(num_controls, key):
    """Hand-made closures over prob-vector beliefs whose step terms depend on
    the action, the step index and the beliefs, one of them a spread interval."""
    ns = tuple(3 + f for f in range(len(num_controls)))
    keys = iter(jax.random.split(key, 16))
    tables = tuple(
        jax.random.dirichlet(next(keys), jnp.ones(n), (n, u)).transpose(2, 0, 1)  # (n_out, n_in, u)
        for n, u in zip(ns, num_controls, strict=True)
    )
    targets = tuple(jax.random.dirichlet(next(keys), jnp.ones(n)) for n in ns)
    costs = tuple(jax.random.normal(next(keys), (u,)) for u in num_controls)
    init = tuple(jax.random.dirichlet(next(keys), jnp.ones(n)) for n in ns)

    def transition_fn(qs, action):
        return tuple(tables[f][:, :, action[f]] @ qs[f] for f in range(len(qs)))

    def step_terms(qs, action, t):
        risk = {
            f"f{f}": Interval.exact(jnp.sum(qs[f] * (jnp.log(qs[f]) - jnp.log(targets[f]))))
            for f in range(len(qs))
        }
        cost = sum(costs[f][action[f]] for f in range(len(qs))) * (t + 1).astype(jnp.float32)
        return {"risk": risk, "ambiguity": {"act": Interval(cost - 0.1, cost + 0.1)}}

    def terminal_terms(qs):
        return {"terminal_risk": {"f0": Interval.exact(-jnp.max(qs[0]))}}

    return init, transition_fn, step_terms, terminal_terms


def test_leaf_order_is_the_grid_order_with_uncontrolled_factors():
    num_controls, controlled, H = (3, 1, 2), (True, False, True), 3
    b = 3 * 2  # uncontrolled factors are excluded from the branching
    policies = enumerate_policies(num_controls, H, controlled)
    assert policies.shape == (b**H, H, 3)
    # the ancestor of policy p at depth t is node p // b^(H-1-t) of the (t+1)-step grid
    for t in range(H):
        prefixes = enumerate_policies(num_controls, t + 1, controlled)
        anc = np.arange(b**H) // b ** (H - 1 - t)
        assert np.array_equal(np.asarray(prefixes)[anc], np.asarray(policies)[:, : t + 1])

    init, transition_fn, step_terms, terminal_terms = _synthetic_closures(
        num_controls, jax.random.PRNGKey(3)
    )
    closures = (init, transition_fn, step_terms, terminal_terms)
    tree, pols = evaluate_policy_tree(num_controls, controlled, H, *closures)
    g, pols_g = evaluate_policy_tree_total(num_controls, controlled, H, *closures)
    assert np.array_equal(np.asarray(pols), np.asarray(policies))
    assert np.array_equal(np.asarray(pols_g), np.asarray(policies))
    ref_tree = evaluate_policies(policies, *closures)
    ref_g = evaluate_policies_total(policies, *closures)
    for term, blocks in ref_tree.items():
        for block, iv in blocks.items():
            _assert_interval(tree[term][block], iv, f"{term}/{block}")
    _assert_interval(total(tree), total(ref_tree), "total")
    _assert_interval(g, ref_g, "G")
    # a policy whose leaves are not its own path would show here: policies 0 and b^H-1
    # share no prefix beyond the root, so their step-cost leaves differ
    cost = tree["ambiguity"]["act"].lower
    assert not np.isclose(float(cost[0]), float(cost[-1]))


def test_expand_level_child_index_and_shapes():
    num_controls, controlled = (3, 1, 2), (True, False, True)
    init, transition_fn, step_terms, _ = _synthetic_closures(num_controls, jax.random.PRNGKey(5))
    actions = enumerate_policies(num_controls, 1, controlled)[:, 0, :]
    b = actions.shape[0]
    nodes = jtu.tree_map(lambda x: jnp.stack([x, x[::-1] / x[::-1].sum()]), init)  # two nodes
    children, tree = expand_level(nodes, actions, transition_fn, step_terms, 1)
    assert all(leaf.shape[0] == 2 * b for leaf in jtu.tree_leaves(children))
    assert all(leaf.shape[0] == 2 * b for leaf in jtu.tree_leaves(tree))
    for i in range(2):
        for a in range(b):
            node = jtu.tree_map(lambda x, i=i: x[i], nodes)
            ref = transition_fn(node, actions[a])
            got = jtu.tree_map(lambda x, i=i, a=a: x[i * b + a], children)
            for r, c in zip(ref, got, strict=True):
                np.testing.assert_allclose(np.asarray(c), np.asarray(r), rtol=TOL, atol=TOL)
            ref_tree = step_terms(ref, actions[a], jnp.int32(1))
            np.testing.assert_allclose(
                float(tree["ambiguity"]["act"].lower[i * b + a]),
                float(ref_tree["ambiguity"]["act"].lower),
                rtol=TOL,
                atol=TOL,
            )


# ---------------------------------------------------------------- the planner (E3)


def _assert_probs(got, ref):
    np.testing.assert_allclose(np.asarray(got.probs), np.asarray(ref.probs), rtol=TOL, atol=TOL)


@pytest.mark.parametrize("metrics", [True, False])
def test_infer_policies_matches_the_scan_planner(metrics):
    system, params = tmaze_system(policy_len=3)
    beliefs = _initial_beliefs(system, params)
    q_pi, tree = system.infer_policies(params, beliefs, metrics=metrics)
    ref_q_pi, ref_tree = system.infer_policies(
        params, beliefs, planner=old_enumeration_planner, metrics=metrics
    )
    _assert_probs(q_pi, ref_q_pi)
    assert (tree is None) == (ref_tree is None) == (not metrics)
    if metrics:
        for term, blocks in ref_tree.items():
            for block, iv in blocks.items():
                _assert_interval(tree[term][block], iv, f"{term}/{block}")

    legal = {"u_location": jnp.asarray([True, True, False, True, False])}
    q_pi, _ = system.infer_policies(params, beliefs, metrics=metrics, legal=legal)
    ref_q_pi, _ = system.infer_policies(
        params, beliefs, planner=old_enumeration_planner, metrics=metrics, legal=legal
    )
    _assert_probs(q_pi, ref_q_pi)
    assert float(q_pi.probs[np.asarray(system.policies(params))[:, 0, 0] == 2].sum()) == 0.0

    prior = jax.nn.log_softmax(jax.random.normal(jax.random.PRNGKey(0), (125,)))
    params = dataclasses.replace(params, policy_prior=prior)
    q_pi, _ = system.infer_policies(params, beliefs, metrics=metrics)
    ref_q_pi, _ = system.infer_policies(
        params, beliefs, planner=old_enumeration_planner, metrics=metrics
    )
    _assert_probs(q_pi, ref_q_pi)


def test_infer_policies_jits_on_the_tree():
    system, params = tmaze_system(policy_len=3)
    beliefs = _initial_beliefs(system, params)
    f = jax.jit(lambda p, b: system.infer_policies(p, b)[0].probs)
    np.testing.assert_allclose(
        np.asarray(f(params, beliefs)),
        np.asarray(system.infer_policies(params, beliefs)[0].probs),
        rtol=TOL,
        atol=TOL,
    )


# ------------------------------------------------------------------- sharing (E5)


def _counting_transition(transition_fn, sizes: list):
    """Record, at execution, how many beliefs each transition call produced.

    The callback sees the batched leading axes (``vmap_method="broadcast_all"``),
    so its records sum to the number of belief transitions actually evaluated.
    """

    def transition(qs, action):
        out = transition_fn(qs, action)
        lead, *rest = out

        def record(x):
            sizes.append(int(np.prod(x.shape[:-1])))
            return x

        lead = jax.pure_callback(
            record, jax.ShapeDtypeStruct(lead.shape, lead.dtype), lead, vmap_method="broadcast_all"
        )
        return (lead, *rest)

    return transition


@pytest.mark.parametrize("policy_len", [2, 3])
def test_tree_evaluates_each_prefix_once(policy_len):
    """``b·(b^H − 1)/(b − 1)`` transitions for the tree against ``H·b^H`` for the scans."""
    system, params = tmaze_system(policy_len=policy_len)
    ctx = system.planner_context(params, _initial_beliefs(system, params))
    b = 5
    sizes: list[int] = []
    counting = _counting_transition(ctx.transition_fn, sizes)
    policies = enumerate_policies(ctx.num_controls, ctx.policy_len, ctx.controlled)

    evaluate_policies(policies, ctx.init_beliefs, counting, ctx.step_terms_fn)
    assert sum(sizes) == policy_len * b**policy_len
    sizes.clear()
    evaluate_policy_tree(
        ctx.num_controls,
        ctx.controlled,
        ctx.policy_len,
        ctx.init_beliefs,
        counting,
        ctx.step_terms_fn,
    )
    assert sizes == [b ** (t + 1) for t in range(policy_len)]
    assert sum(sizes) == b * (b**policy_len - 1) // (b - 1)
    sizes.clear()
    evaluate_policy_tree_total(
        ctx.num_controls,
        ctx.controlled,
        ctx.policy_len,
        ctx.init_beliefs,
        counting,
        ctx.step_terms_fn,
    )
    assert sum(sizes) == b * (b**policy_len - 1) // (b - 1)


def test_enumerate_policies_is_memoised_and_safe_under_jit():
    _enumerate_policies.cache_clear()
    a = enumerate_policies((7, 1), 3, (True, False))
    assert _enumerate_policies.cache_info().misses == 1
    b = enumerate_policies([7, 1], 3, [True, False])  # list spellings hit the same entry
    c = enumerate_policies((7, 1), 3)  # default controlled: num_controls > 1
    assert _enumerate_policies.cache_info().hits == 2
    assert np.array_equal(np.asarray(a), np.asarray(b)) and np.array_equal(
        np.asarray(a), np.asarray(c)
    )
    # a first call under jit must not cache a tracer: the same grid is then used eagerly
    grid = jax.jit(lambda: enumerate_policies((7, 1), 2, (True, False)))()
    assert np.array_equal(
        np.asarray(enumerate_policies((7, 1), 2, (True, False))), np.asarray(grid)
    )
