"""The sophisticated planner's memory and legality below the root
(``docs/design/sophisticated-memory.md``): chunked subtrees give the whole-level result bit for
bit, the batch is chosen from a memory budget, and a per-step ``legal=`` mask is read at every
depth.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalGaussian,
    Gaussian,
)
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianTransition
from hybrid_aif.objectives.base import total
from hybrid_aif.planning.enumerate import enumerate_policies
from hybrid_aif.planning.sophisticated import (
    BranchingTreeMemory,
    branching_tree_memory,
    chunk_plan,
    make_sophisticated_planner,
    sophisticated_planner,
)
from hybrid_aif.planning.tree import (
    BRANCHING_BELIEF_FACTOR,
    BRANCHING_BYTES_PER_NODE,
    belief_bytes,
    branching_tree_bytes,
    chunk_choice,
    evaluate_branching_tree,
    expand_level,
)
from hybrid_aif.system import System, SystemParams
from tests.test_sophisticated import _replanned
from tests.test_system import _initial_beliefs, tmaze_system

ONE = ("cue",)
TWO = ("cue", "reward")
PAIRS = [(1, 1), (1, 3), (1, 7), (2, 5), (2, 1000), (3, 4), (5, 2)]


def _planned(horizon: int, branch_on=ONE):
    system, params = tmaze_system(policy_len=horizon)
    system = _replanned(system, planner="sophisticated", branch_on=branch_on)
    return system, params, _initial_beliefs(system, params)


def _tree_call(system, params, beliefs, **options):
    ctx = system.planner_context(params, beliefs)
    actions = enumerate_policies(ctx.num_controls, 1, ctx.controlled)[:, 0, :]
    return evaluate_branching_tree(
        actions,
        ctx.policy_len,
        ctx.gamma,
        ctx.init_beliefs,
        ctx.transition_fn,
        ctx.step_terms_fn,
        ctx.terminal_terms_fn,
        ctx.branch_fn,
        **options,
    )


def _same(a, b) -> bool:
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


# Chunking changes the batch shape of every contraction below the split, and XLA's CPU kernels
# accumulate a batched contraction in an order that can depend on the batch size, so chunked and
# whole-level results agree up to float rounding (a few ulps; the order of eager against jit),
# and are often identical.
RELATIVE_ROUNDING = 1e-6


def _equal_to_rounding(cases: list[tuple[str, object, object]]) -> None:
    """Each case is ``(label, chunked, whole)``: equal exactly, or within float rounding."""
    for label, got, want in cases:
        np.testing.assert_allclose(
            np.asarray(got, np.float64),
            np.asarray(want, np.float64),
            rtol=RELATIVE_ROUNDING,
            atol=0.0,
            err_msg=label,
        )


def _random_model(horizon: int, seed: int = 0, n_actions: int = 3):
    """A model with random tables, unequal outcome probabilities and a different ``G`` for every
    action, branching on both modalities: a ties-free check of which node a result belongs to."""
    rng = np.random.default_rng(seed)

    def table(*shape):
        x = rng.gamma(0.7, size=shape) + 1e-3
        return x / x.sum(0, keepdims=True)

    def cc(t, n):
        return ConditionalCategorical(log_table=jnp.log(jnp.asarray(t)), parent_ndims=n)

    system = System(
        A_deps=((0, 1), (0,)),
        B_deps=((0,), (1,)),
        controlled=(True, False),
        policy_len=horizon,
        gamma=4.0,
        epistemic=True,
        planner="sophisticated",
        branch_on=("o1", "o2"),
        modality_names=("o1", "o2"),
        factor_names=("s", "c"),
    )
    params = SystemParams(
        likelihood={"o1": cc(table(3, 4, 2), 2), "o2": cc(table(2, 4), 1)},
        transition={"s": cc(table(4, 4, n_actions), 2), "c": cc(table(2, 2, 1), 2)},
        initial_state_prior={
            "s": Categorical.from_probs(jnp.asarray(table(4))),
            "c": Categorical.from_probs(jnp.asarray(table(2))),
        },
        obs_targets={
            "o1": Categorical(logits=jnp.asarray(rng.normal(size=3) * 2)),
            "o2": Categorical(logits=jnp.asarray(rng.normal(size=2) * 2)),
        },
        state_targets={},
        terminal_targets={},
    )
    return system, params, system.initial_priors(params)


def _gaussian_model(dim: int, horizon: int):
    """A categorical context and a ``dim``-dimensional Gaussian state; branching on the context's
    cue. Its beliefs are large, so the memory per child is mostly the belief's."""
    system = System(
        A_deps=((0,), (1,)),
        B_deps=((), ()),
        controlled=(True, True),
        policy_len=horizon,
        modality_names=("cue", "gps"),
        factor_names=("ctx", "x"),
        epistemic=True,
        planner="sophisticated",
        branch_on=("cue",),
    )
    innovations = ConditionalGaussian(
        loc=jnp.stack([u * 0.1 * jnp.ones(dim) for u in (-1.0, 0.0, 1.0)]),
        chol=jnp.stack([0.05 * jnp.eye(dim)] * 3),
        parent_ndims=1,
    )
    params = SystemParams(
        likelihood={
            "cue": ConditionalCategorical(
                log_table=jnp.log(jnp.array([[0.85, 0.15], [0.15, 0.85]])), parent_ndims=1
            ),
            "gps": LinearGaussianObservation(
                H=jnp.eye(dim), c=jnp.zeros(dim), noise=0.1 * jnp.eye(dim)
            ),
        },
        transition={
            "ctx": ConditionalCategorical(
                log_table=jnp.log(jnp.array([[0.9, 0.1], [0.1, 0.9]])), parent_ndims=1
            ),
            "x": LinearGaussianTransition(F=0.95 * jnp.eye(dim), innovations=innovations),
        },
        initial_state_prior={
            "ctx": Categorical(logits=jnp.zeros(2)),
            "x": Gaussian(loc=jnp.zeros(dim), chol=0.2 * jnp.eye(dim)),
        },
        obs_targets={},
        state_targets={"x": Gaussian(loc=jnp.ones(dim), chol=0.3 * jnp.eye(dim))},
        terminal_targets={},
    )
    return system, params, system.initial_priors(params)


@pytest.mark.parametrize("branch_on", [ONE, TWO], ids=["one-modality", "two-modalities"])
@pytest.mark.parametrize("horizon", [1, 2, 3, 4])
def test_chunked_evaluation_is_the_whole_level(horizon, branch_on):
    system, params, beliefs = _planned(horizon, branch_on)
    whole = _tree_call(system, params, beliefs)
    whole_jit = jax.jit(lambda p, b: _tree_call(system, p, b))(params, beliefs)
    cases = []
    for depth, batch in PAIRS:
        eager = _tree_call(system, params, beliefs, split_depth=depth, batch_size=batch)
        cases.append((f"eager d={depth} k={batch}", eager, whole))

        def chunked(p, b, depth=depth, batch=batch):
            return _tree_call(system, p, b, split_depth=depth, batch_size=batch)

        cases.append((f"jit d={depth} k={batch}", jax.jit(chunked)(params, beliefs), whole_jit))
    _equal_to_rounding(cases)


@pytest.mark.parametrize("horizon", [2, 3, 4])
def test_chunked_evaluation_of_a_random_asymmetric_model_is_the_whole_level(horizon):
    system, params, beliefs = _random_model(horizon, seed=horizon)
    whole = _tree_call(system, params, beliefs)
    assert len(np.unique(np.asarray(whole))) == whole.shape[0]  # no ties to hide a permutation
    cases = []
    for depth, batch in [(1, 1), (1, 5), (2, 7), (horizon - 1, 3)]:
        chunked = _tree_call(system, params, beliefs, split_depth=depth, batch_size=batch)
        cases.append((f"d={depth} k={batch}", chunked, whole))
    _equal_to_rounding(cases)
    ones = jnp.ones((horizon, 3), bool)
    masked = _tree_call(system, params, beliefs, legal=ones.at[1, 0].set(False))
    chunked = _tree_call(
        system, params, beliefs, legal=ones.at[1, 0].set(False), split_depth=1, batch_size=5
    )
    _equal_to_rounding([("masked", chunked, masked)])


def test_chunked_float64_evaluation_agrees_to_a_few_ulps():
    with jax.enable_x64(True):

        def wide(x):
            return x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x

        system, params, beliefs = _random_model(3, seed=2)
        params, beliefs = jax.tree_util.tree_map(wide, (params, beliefs))
        whole = _tree_call(system, params, beliefs)
        assert whole.dtype == jnp.float64
        for depth, batch in [(1, 1), (1, 5), (2, 7)]:
            chunked = _tree_call(system, params, beliefs, split_depth=depth, batch_size=batch)
            np.testing.assert_allclose(chunked, whole, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("branch_on", [ONE, TWO], ids=["one-modality", "two-modalities"])
def test_chunked_q_pi_and_the_selected_action_are_the_whole_levels(branch_on):
    system, params, beliefs = _planned(4, branch_on)
    q_whole, _ = system.infer_policies(params, beliefs)
    cases = []
    for depth, batch in [(1, 2), (2, 7), (3, 3)]:
        planner = make_sophisticated_planner(split_depth=depth, batch_size=batch)
        q_chunk, _ = system.infer_policies(params, beliefs, planner=planner)
        assert int(jnp.argmax(q_chunk.probs)) == int(jnp.argmax(q_whole.probs))
        cases.append((f"logits d={depth} k={batch}", q_chunk.logits, q_whole.logits))
    _equal_to_rounding(cases)


def _temp_bytes(fn, *args) -> int:
    return jax.jit(fn).lower(*args).compile().memory_analysis().temp_size_in_bytes


def test_a_tree_over_the_budget_runs_under_it():
    system, params, beliefs = _planned(4, TWO)
    ctx = system.planner_context(params, beliefs)
    whole = _temp_bytes(lambda p, b: _tree_call(system, p, b), params, beliefs)
    budget = whole // 8
    depth, batch = chunk_plan(ctx, budget, None, None)
    assert batch is not None and batch >= 1
    planner = make_sophisticated_planner(memory_budget=budget)

    def planned(p, b):
        return system.infer_policies(p, b, planner=planner)[0].logits

    chunked = _temp_bytes(planned, params, beliefs)
    assert chunked < budget < whole
    q_whole, _ = system.infer_policies(params, beliefs)
    _equal_to_rounding([("budgeted", jax.jit(planned)(params, beliefs), q_whole.logits)])


@pytest.mark.parametrize("dim", [16, 32])
def test_gaussian_beliefs_run_under_the_budget_and_the_estimate_errs_high(dim):
    system, params, beliefs = _gaussian_model(dim, 4)
    ctx = system.planner_context(params, beliefs)
    outcomes = jax.eval_shape(ctx.branch_fn, ctx.init_beliefs)[0].shape[0]

    def logits(planner):
        return lambda p, b: system.infer_policies(p, b, planner=planner)[0].logits

    whole = _temp_bytes(logits(None), params, beliefs)
    estimate = branching_tree_bytes(6, outcomes, 4, belief_bytes(ctx.init_beliefs))
    assert estimate > whole
    for fraction in (2, 4, 8):
        budget = whole // fraction
        planner = make_sophisticated_planner(memory_budget=budget)
        assert _temp_bytes(logits(planner), params, beliefs) < budget
    chunked = jax.jit(logits(make_sophisticated_planner(memory_budget=whole // 4)))
    _equal_to_rounding([("gaussian", chunked(params, beliefs), logits(None)(params, beliefs))])


def test_a_batch_that_does_not_divide_the_nodes_is_not_chosen():
    """45 nodes at depth 1 (5 actions, 9 outcomes): a batch of 23 would leave a remainder of 22
    evaluated beside its two batches, and hold more than the budget asked."""
    belief = 28
    per_child = BRANCHING_BELIEF_FACTOR * belief + BRANCHING_BYTES_PER_NODE
    subtree = 45**2 * 5 * per_child
    budget = 45 * per_child + 23 * subtree
    depth, batch = chunk_choice(5, 9, 4, belief, budget)
    assert (depth, batch) == (1, 15)
    system, params, beliefs = _planned(4, TWO)
    ctx = system.planner_context(params, beliefs)
    whole = _temp_bytes(lambda p, b: _tree_call(system, p, b), params, beliefs)
    for fraction in (2, 3, 5):
        _, k = chunk_plan(ctx, whole // fraction, None, None)
        assert k is not None and 45 % k == 0


def test_the_split_deepens_until_one_subtree_fits():
    belief, A, m, H = 28, 5, 3, 5
    per_child = BRANCHING_BELIEF_FACTOR * belief + BRANCHING_BYTES_PER_NODE
    width = A * m

    def held(d):
        return width**d * per_child + width ** (H - 1 - d) * A * per_child

    assert held(2) < held(1) and held(2) < held(3) and held(4) > held(2)
    depth, batch = chunk_choice(A, m, H, belief, held(2))
    assert (depth, batch) == (2, 1)  # depth 1 holds one subtree of more than the budget
    depth, batch = chunk_choice(A, m, H, belief, held(1))
    assert depth == 1
    assert chunk_choice(A, m, H, belief, held(2), split_depth=1) == (1, 1)  # a given depth stays
    #  nothing fits: batch 1 where the estimate is least, the budget a best effort
    assert chunk_choice(A, m, H, belief, 1) == (min(range(1, H), key=held), 1)
    assert chunk_choice(A, m, H, belief, 1, split_depth=3) == (3, 1)


def test_the_deepened_split_runs_under_the_budget():
    system, params, beliefs = _planned(5, ONE)
    ctx = system.planner_context(params, beliefs)
    whole = _temp_bytes(lambda p, b: _tree_call(system, p, b), params, beliefs)
    budget = whole // 12
    depth, batch = chunk_plan(ctx, budget, None, None)
    assert depth >= 2 and batch is not None
    planner = make_sophisticated_planner(memory_budget=budget)

    def planned(p, b):
        return system.infer_policies(p, b, planner=planner)[0].logits

    assert _temp_bytes(planned, params, beliefs) < budget
    q_whole, _ = system.infer_policies(params, beliefs)
    _equal_to_rounding([("deepened", jax.jit(planned)(params, beliefs), q_whole.logits)])


def test_without_a_budget_or_a_batch_the_program_is_the_whole_level():
    system, params, beliefs = _planned(3, TWO)

    def default(p, b):
        return system.infer_policies(p, b)[0].logits

    def plain(p, b):
        return system.infer_policies(p, b, planner=_PlainTree())[0].logits

    class _PlainTree:
        needs_key, branching, legal_depth = False, True, True

        def __call__(self, ctx, use_bound="midpoint", metrics=True):
            actions = enumerate_policies(ctx.num_controls, 1, ctx.controlled)[:, 0, :]
            G = evaluate_branching_tree(
                actions,
                ctx.policy_len,
                ctx.gamma,
                ctx.init_beliefs,
                ctx.transition_fn,
                ctx.step_terms_fn,
                ctx.terminal_terms_fn,
                ctx.branch_fn,
                use_bound,
            )
            from hybrid_aif.distributions import Categorical

            return Categorical(logits=-ctx.gamma * G), {}, actions[:, None, :]

    def made(p, b):
        return system.infer_policies(p, b, planner=make_sophisticated_planner())[0].logits

    text = str(jax.make_jaxpr(default)(params, beliefs))
    assert text == str(jax.make_jaxpr(plain)(params, beliefs))
    assert text == str(jax.make_jaxpr(made)(params, beliefs))
    assert "scan" not in text
    forced = str(
        jax.make_jaxpr(
            lambda p, b: (
                system.infer_policies(p, b, planner=make_sophisticated_planner(batch_size=2))[
                    0
                ].logits
            )
        )(params, beliefs)
    )
    assert "scan" in forced


def test_the_batch_chosen_is_the_largest_divisor_that_fits():
    A, m, H, belief = 5, 3, 4, 40
    per_child = BRANCHING_BELIEF_FACTOR * belief + BRANCHING_BYTES_PER_NODE
    whole = branching_tree_bytes(A, m, H, belief)
    assert whole == (A * m) ** (H - 1) * A * per_child
    assert chunk_choice(A, m, H, belief, whole) == (1, None)
    for budget in (whole - 1, whole // 3, whole // 50):
        for depth in (1, 2, 3):
            got_depth, got = chunk_choice(A, m, H, belief, budget, depth)
            nodes = (A * m) ** depth
            fits = [
                k
                for k in range(1, nodes + 1)
                if nodes % k == 0 and branching_tree_bytes(A, m, H, belief, depth, k) <= budget
            ]
            assert got_depth == depth and got == (max(fits) if fits else 1), (budget, depth)
    assert chunk_choice(A, m, 1, belief, 1) == (1, None)  # one step: nothing to chunk
    assert chunk_choice(A, m, H, belief, 1, split_depth=H) == (H, None)  # no node at that depth


def test_a_given_batch_overrides_the_choice_and_a_budget_is_read_when_given():
    system, params, beliefs = _planned(4, TWO)
    ctx = system.planner_context(params, beliefs)
    assert chunk_plan(ctx, None, None, None) == (1, None)  # no option: the whole level
    assert chunk_plan(ctx, None, 2, None) == (2, None)
    assert chunk_plan(ctx, None, None, 7) == (1, 7)  # a batch alone splits at depth 1
    assert chunk_plan(ctx, 1, None, 7) == (1, 7)
    assert chunk_plan(ctx, 1, 2, 7) == (2, 7)
    assert chunk_plan(ctx, 10**15, None, None) == (1, None)
    spec = belief_bytes(ctx.init_beliefs)
    outcomes = jax.eval_shape(ctx.branch_fn, ctx.init_beliefs)[0].shape[0]
    want = chunk_choice(5, outcomes, 4, spec, 3_000_000, 2)
    assert chunk_plan(ctx, 3_000_000, 2, None) == want
    assert chunk_plan(ctx, 3_000_000, None, 5)[1] == 5  # a given batch overrides the budget's


def test_branching_tree_memory_reports_the_estimate_and_the_budgets_choice():
    system, params, beliefs = _planned(4, TWO)
    ctx = system.planner_context(params, beliefs)
    spec = belief_bytes(ctx.init_beliefs)
    whole = branching_tree_bytes(5, 9, 4, spec)
    assert branching_tree_memory(system, params, beliefs) == (whole, None, None, None)
    assert branching_tree_memory(system, params, beliefs, whole) == (whole, None, None, whole)
    for budget in (whole // 3, whole // 40, 1):
        depth, batch = chunk_choice(5, 9, 4, spec, budget)
        got = branching_tree_memory(system, params, beliefs, budget)
        assert got == (whole, depth, batch, branching_tree_bytes(5, 9, 4, spec, depth, batch))
        assert got.split_depth == chunk_plan(ctx, budget, None, None)[0]
        assert got.batch_size == chunk_plan(ctx, budget, None, None)[1]
    assert isinstance(got, BranchingTreeMemory)


def test_branching_tree_memory_evaluates_nothing(monkeypatch):
    """No step term runs, and a tree whose whole level could not be held (horizon 8 over nine
    outcomes) is sized at once."""
    calls = []
    real = System.planner_context

    def counting(self, *args, **kwargs):
        ctx = real(self, *args, **kwargs)
        step = ctx.step_terms_fn

        def counted(*a, **k):
            calls.append(1)
            return step(*a, **k)

        return ctx._replace(step_terms_fn=counted)

    monkeypatch.setattr(System, "planner_context", counting)
    system, params, beliefs = _planned(8, TWO)
    got = branching_tree_memory(system, params, beliefs, 10**9)
    assert calls == []
    assert got.whole_level_bytes > 10**14
    assert got.batch_size is not None and got.estimated_bytes is not None
    # the same hook does count a step term when the tree is evaluated
    small, small_params, small_beliefs = _planned(2, ONE)
    _tree_call(small, small_params, small_beliefs)
    assert calls


def test_the_options_are_validated():
    with pytest.raises(ValueError, match="split_depth must be at least 1"):
        make_sophisticated_planner(split_depth=0)
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        make_sophisticated_planner(batch_size=0)


# ---------------------------------------------------------------------------------- legality


def _step_mask(horizon: int, rows: dict[int, list[int]]):
    """A bool ``(horizon, 5)`` mask over the T-maze's locations, those in ``rows[t]`` illegal."""
    mask = np.ones((horizon, 5), bool)
    for t, illegal in rows.items():
        mask[t, illegal] = False
    return jnp.asarray(mask)


def test_with_every_action_legal_the_result_is_the_unmasked_bit_for_bit():
    system, params, beliefs = _planned(3, TWO)
    free = _tree_call(system, params, beliefs)
    ones = jnp.ones((3, 5), bool)
    assert _same(_tree_call(system, params, beliefs, legal=ones), free)
    jitted = jax.jit(lambda p, b, m: _tree_call(system, p, b, legal=m))
    assert _same(
        jitted(params, beliefs, ones),
        jax.jit(lambda p, b: _tree_call(system, p, b))(params, beliefs),
    )
    q_root, _ = system.infer_policies(params, beliefs, legal={"u_location": ones[0]})
    q_ones, _ = system.infer_policies(params, beliefs, legal={"u_location": ones})
    assert _same(q_ones.logits, q_root.logits)
    q_free, _ = system.infer_policies(params, beliefs)
    assert _same(q_ones.probs, q_free.probs)


def test_a_mask_at_the_first_step_alone_is_the_root_mask_of_today():
    system, params, beliefs = _planned(3)
    first = jnp.asarray([True, False, True, True, True])
    q_first, _ = system.infer_policies(params, beliefs, legal={"u_location": first})
    q_row, _ = system.infer_policies(params, beliefs, legal={"u_location": first[None]})
    padded = jnp.ones((3, 5), bool).at[0].set(first)
    q_pad, _ = system.infer_policies(params, beliefs, legal={"u_location": padded})
    assert float(q_first.probs[1]) == 0.0
    assert _same(q_first.logits, q_row.logits) and _same(q_first.logits, q_pad.logits)
    assert float(q_first.probs.sum()) == pytest.approx(1.0)


def test_an_action_illegal_at_a_depth_has_no_weight_in_that_nodes_value():
    """Horizon 2, by hand: ``G(a) = g1(a) + Σ_o p(o) V(a, o)`` with ``V`` the softmax value over
    the legal second actions only."""
    system, params, beliefs = _planned(2)
    ctx = system.planner_context(params, beliefs)
    acts = enumerate_policies(ctx.num_controls, 1, ctx.controlled)[:, 0, :]
    illegal = [1, 3]
    mask = _step_mask(2, {1: illegal})
    got = _tree_call(system, params, beliefs, legal=mask)

    qs = jax.tree_util.tree_map(lambda x: x[None], ctx.init_beliefs)
    kids, tree1 = expand_level(qs, acts, ctx.transition_fn, ctx.step_terms_fn, 0)
    g1 = np.asarray(total(tree1).midpoint)
    p, after = jax.vmap(ctx.branch_fn)(kids)
    legal_ids = [a for a in range(5) if a not in illegal]
    want = []
    for a in range(5):
        value = 0.0
        for o in range(p.shape[1]):
            node = jax.tree_util.tree_map(lambda x, a=a, o=o: x[a, o][None], after)
            gkids, tree2 = expand_level(node, acts, ctx.transition_fn, ctx.step_terms_fn, 1)
            g2 = total(tree2).midpoint
            if ctx.terminal_terms_fn is not None:
                g2 = g2 + total(jax.vmap(ctx.terminal_terms_fn)(gkids)).midpoint
            g2 = np.asarray(g2, np.float64)[legal_ids]
            w = np.exp(-ctx.gamma * (g2 - g2.min()))
            value += float(p[a, o]) * float((w / w.sum()) @ g2)
        want.append(g1[a] + value)
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-5)
    free = _tree_call(system, params, beliefs)
    assert not np.allclose(np.asarray(got), np.asarray(free), rtol=1e-6)


def test_what_an_illegal_action_would_score_does_not_reach_the_value():
    system, params, beliefs = _planned(3)
    ctx = system.planner_context(params, beliefs)
    acts = enumerate_policies(ctx.num_controls, 1, ctx.controlled)[:, 0, :]
    mask = _step_mask(3, {1: [2], 2: [0, 4]})

    def with_penalty(penalty):
        def step_terms(qs, action, t):
            tree = ctx.step_terms_fn(qs, action, t)
            from hybrid_aif.interval import Interval

            bad = ((t == 1) & (action[0] == 2)) | ((t == 2) & ((action[0] == 0) | (action[0] == 4)))
            extra = jnp.where(bad, penalty, 0.0)
            tree = {k: dict(v) for k, v in tree.items()}
            tree["utility"]["reward"] = tree["utility"]["reward"] + Interval.exact(extra)
            return tree

        return evaluate_branching_tree(
            acts,
            3,
            ctx.gamma,
            ctx.init_beliefs,
            ctx.transition_fn,
            step_terms,
            ctx.terminal_terms_fn,
            ctx.branch_fn,
            legal=mask,
        )

    base = with_penalty(0.0)
    assert _same(with_penalty(5.0), base)
    assert _same(with_penalty(jnp.nan), base)
    assert _same(with_penalty(jnp.inf), base)


def test_a_mask_that_leaves_no_legal_action_at_a_depth_is_refused():
    system, params, beliefs = _planned(3)
    dead = _step_mask(3, {1: [0, 1, 2, 3, 4]})
    with pytest.raises(ValueError, match=r"masks every level of control node 'u_location'"):
        system.infer_policies(params, beliefs, legal={"u_location": dead})
    with pytest.raises(ValueError, match=r"takes \(5,\) to mask this tick's levels, or \(3, 5\)"):
        system.infer_policies(params, beliefs, legal={"u_location": jnp.ones((2, 5), bool)})


def test_per_step_legality_composes_with_chunking():
    system, params, beliefs = _planned(4, TWO)
    mask = _step_mask(4, {1: [2], 2: [0, 3], 3: [4]})
    whole = _tree_call(system, params, beliefs, legal=mask)
    cases = []
    for depth, batch in [(1, 3), (2, 5), (3, 2)]:
        chunked = _tree_call(
            system, params, beliefs, legal=mask, split_depth=depth, batch_size=batch
        )
        cases.append((f"d={depth} k={batch}", chunked, whole))
    q_whole, _ = system.infer_policies(params, beliefs, legal={"u_location": mask})
    planner = make_sophisticated_planner(batch_size=4)
    q_chunk, _ = system.infer_policies(params, beliefs, planner=planner, legal={"u_location": mask})
    cases.append(("q_pi", q_chunk.logits, q_whole.logits))
    _equal_to_rounding(cases)


def test_a_per_step_mask_reaches_a_registered_sophisticated_planner_by_name():
    system, params, beliefs = _planned(3)
    assert system.planner == "sophisticated"
    assert sophisticated_planner.legal_depth is True
    mask = _step_mask(3, {1: [2]})
    q_mask, _ = system.infer_policies(params, beliefs, legal={"u_location": mask})
    q_free, _ = system.infer_policies(params, beliefs)
    assert not _same(q_mask.logits, q_free.logits)


def test_no_retrace_across_values_of_the_beliefs_and_of_the_legal_mask():
    system, params, beliefs = _planned(3)
    for planner in (None, make_sophisticated_planner(batch_size=3)):
        traces = []

        @jax.jit
        def plan(p, b, mask, planner=planner, traces=traces):
            traces.append(1)
            return system.infer_policies(p, b, planner=planner, legal={"u_location": mask})[
                0
            ].logits

        first = plan(params, beliefs, _step_mask(3, {1: [2]}))
        other = plan(params, beliefs, _step_mask(3, {1: [1], 2: [0, 4]}))
        nudged = jax.tree_util.tree_map(lambda x: x * 0.9 + 0.1, beliefs)
        moved = plan(params, nudged, jnp.ones((3, 5), bool))
        assert len(traces) == 1
        assert not _same(first, other) and not _same(moved, first)


def test_a_first_step_mask_adds_no_operations_to_the_tree():
    system, params, beliefs = _planned(3)

    def selects(legal):
        def plan(p, b, mask):
            return system.infer_policies(p, b, legal=legal(mask))[0].logits

        return str(jax.make_jaxpr(plan)(params, beliefs, jnp.ones((3, 5), bool))).count("select_n")

    free = str(
        jax.make_jaxpr(lambda p, b: system.infer_policies(p, b)[0].logits)(params, beliefs)
    ).count("select_n")
    root = selects(lambda m: {"u_location": m[0]})
    padded_row = selects(lambda m: {"u_location": m[:1]})
    stepped = selects(lambda m: {"u_location": m})
    assert root == padded_row and root - free <= 2  # the root's own indicator
    assert stepped > root  # a mask per step adds its selects at the depths below
