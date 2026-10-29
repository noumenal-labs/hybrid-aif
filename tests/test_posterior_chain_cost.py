"""The posterior chain over a Gaussian factor read through noise maps: three exact savings.

The model has the shape of a pose-and-gap plan: a four-dimensional pose factor (``x, y, yaw, b``)
with three dimensions selected, a gap factor, two channels on the pose through a
:class:`~hybrid_aif.factors.GridNoiseMap`, a gap channel, a fourth pose channel declared absent for
the whole plan, a continuous two-dimensional control and the posterior planning belief.

- A channel absent at every step (a concrete all-``False`` column of ``present``, no per-level
  entry) takes no part in the fused readings or the chain, and its leaves are zeros.
- One read of each noise map per step serves the epistemic chain and the information-gain
  leaves, and conditioning on the expected reading leaves the mean as it was.
- The conditional covariance of the fused sub-block route comes from the trailing block of the
  belief's Cholesky factor, with no factorisation for a leading selection.

Each is checked against the route without it (:func:`switch_off`): the whole term trees, the
gradient of ``G`` with respect to the controls, ``G`` and the planned controls agree. The lowered
cost of the gradient is bounded in ``test_plan_evaluation_cost.py``.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.controls import Box
from hybrid_aif.distributions import ConditionalGaussian, Gaussian
from hybrid_aif.factors import (
    GridNoiseMap,
    LinearGaussianObservation,
    LinearGaussianTransition,
)
from hybrid_aif.inference.linear_gaussian import condition
from hybrid_aif.objectives import continuous
from hybrid_aif.objectives import hybrid as _hybrid
from hybrid_aif.objectives.base import total
from hybrid_aif.objectives.hybrid import BeliefPair, Conditioner, where_tree
from hybrid_aif.planning import gradient_plan
from hybrid_aif.planning.gradient import evaluate_policies
from hybrid_aif.system import System, SystemParams

HORIZON = 6
SELECTION = ("pose:x", "pose:y", "pose:yaw", "gap")
CHANNELS = ("pose_fix", "nearest", "gps", "pile")

# ------------------------------------------------------------------------ previous routes


def _previous_condition_expected(q, cond, flag, R_chol=None):
    """Conditioning on the expected reading by the full update, which recomputes the mean from a
    zero innovation and reads the noise at the mean again."""
    R_chol = cond.R_chol_at(q.loc)
    y = (cond.H @ q.loc[..., None])[..., 0] + cond.c
    posterior, _ = condition(q, cond.H, cond.c, R_chol, y)
    return posterior if flag is None else where_tree(flag, posterior, q)


def _previous_pair_beliefs(qs, families, conditioners=()):
    """Pairs that carry no noise reads."""
    return tuple(
        BeliefPair(q, q, q, jnp.asarray(0, jnp.int32)) if fam == _hybrid.GAUSSIAN else q
        for q, fam in zip(qs, families, strict=True)
    )


def _previous_sub_block_readings(q, models, dims, present=None, noises=None):
    """The fused sub-block readings with the conditional covariance from ``_given``."""
    head, covs, sizes = continuous._stacked_noise(q, models, noises)
    noise = jax.scipy.linalg.block_diag(*covs)
    HL = head @ q.chol
    S_y = HL @ jnp.swapaxes(HL, -1, -2) + noise
    S_c = head @ continuous._given(q.cov, dims) @ jnp.swapaxes(head, -1, -2) + noise
    halves_y, diag_y = continuous._block_half_logs(S_y, sizes, present is None)
    halves_c, diag_c = continuous._block_half_logs(S_c, sizes, present is None)
    tc_y = continuous._correlation_of(S_y, halves_y, diag_y, sizes, present)
    tc_c = continuous._correlation_of(S_c, halves_c, diag_c, sizes, present)
    return continuous.SubBlockReadings(
        gains=tuple(
            continuous.Interval.exact(h - c) for h, c in zip(halves_y, halves_c, strict=True)
        ),
        interaction=tc_y - tc_c,
        total_correlation=tc_y,
        conditional_correlation=tc_c,
        ambiguity=tuple(
            0.5 * n * (1.0 + continuous._LOG_2PI) + c for n, c in zip(sizes, halves_c, strict=True)
        ),
    )


def switch_off(which, monkeypatch=None):
    """Replace the changes named in ``which`` (``"absent"``, ``"shared_read"``, ``"trailing"``)
    by the routes they replaced."""
    patch = (
        monkeypatch.setattr
        if monkeypatch is not None
        else lambda obj, name, value: setattr(obj, name, value)
    )
    if "absent" in which:
        patch(_hybrid, "static_absent", lambda present: frozenset())
    if "shared_read" in which:
        patch(_hybrid, "pair_beliefs", _previous_pair_beliefs)
        patch(_hybrid, "condition_expected", _previous_condition_expected)
    if "trailing" in which:
        patch(_hybrid, "sub_block_readings", _previous_sub_block_readings)


# ------------------------------------------------------------------------------ the model


def _grid(values, low, high, axes):
    return GridNoiseMap(jnp.asarray(values), low, high, axes=axes)


def _noise_maps():
    xy = np.zeros((5, 5, 2, 2))
    xy[:] = np.diag([0.5, 0.4])
    xy[3, 1] = np.diag([0.08, 0.1])
    xy[1, 3] = np.diag([0.3, 0.05])
    pile = np.full((5, 5, 3, 1, 1), 0.4)
    pile[2, 2, 1] = 0.05
    pile[4, 0, 2] = 0.9
    return (
        _grid(xy, (-2.0, -2.0), (2.0, 2.0), (0, 1)),
        _grid(pile, (-2.0, -2.0, -1.0), (2.0, 2.0, 1.0), (0, 1, 2)),
    )


def e5_model(
    decomposition="utility_infogain",
    planning_belief="posterior",
    present=None,
    epistemic=SELECTION,
    fix_absent=True,
):
    """The pose-and-gap plan: ``(ctx, init, system, params)``. ``present`` is the ``present=`` of
    the planner context; the default declares the channel ``pose_fix`` absent for the plan."""
    if present is None:
        present = {"pose_fix": False, "gps": jnp.asarray([False, True] * 3)} if fix_absent else {}
    gps_map, pile_map = _noise_maps()
    facs = ("pose", "gap", "u_u")
    A_deps = ((0,), (1,), (0,), (0,))
    system = System(
        A_deps=A_deps,
        B_deps=((2,), (2,), ()),
        controls=(2,),
        action_names=("u",),
        factor_names=facs,
        dims={"pose": ("x", "y", "yaw", "b")},
        modality_names=CHANNELS,
        policy_len=HORIZON,
        planner="gradient",
        gamma=1.0,
        planning_belief=planning_belief,
        decomposition=decomposition,
        epistemic=epistemic,
    )
    F = jnp.eye(4) + 0.05 * jnp.asarray(
        [[0, 0, 0, 1], [0, 0, 1, 0], [0.5, 0, 0, 0.2], [0, 0, 0, 0]]
    )
    G = jnp.asarray([[0.3, 0.0], [0.0, 0.3], [0.0, 0.2], [0.0, 0.0]])
    pose_chol = jnp.asarray(
        [[0.4, 0, 0, 0], [0.1, 0.35, 0, 0], [-0.05, 0.05, 0.3, 0], [0.1, -0.1, 0.1, 0.5]]
    )
    h_pose = {
        "pose_fix": jnp.eye(4),
        "gps": jnp.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        "pile": jnp.asarray([[0.6, 0.3, 0.0, 0.8]]),
    }
    likelihood = {
        "pose_fix": LinearGaussianObservation(
            H=h_pose["pose_fix"], c=jnp.zeros(4), noise=0.2 * jnp.eye(4)
        ),
        "nearest": LinearGaussianObservation(
            H=jnp.ones((1, 1)), c=jnp.zeros(1), noise=0.3 * jnp.eye(1)
        ),
        "gps": LinearGaussianObservation(H=h_pose["gps"], c=jnp.zeros(2), noise=gps_map),
        "pile": LinearGaussianObservation(H=h_pose["pile"], c=jnp.zeros(1), noise=pile_map),
    }
    params = SystemParams(
        transition={
            "pose": LinearGaussianTransition(
                F=F,
                G=G,
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 4)),
                    chol=jnp.diag(jnp.asarray([0.05, 0.05, 0.03, 0.02]))[None],
                    parent_ndims=1,
                ),
            ),
            "gap": LinearGaussianTransition(
                F=jnp.eye(1),
                G=jnp.asarray([[0.1, -0.1]]),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 1)), chol=0.05 * jnp.ones((1, 1, 1)), parent_ndims=1
                ),
            ),
        },
        initial_state_prior={
            "pose": Gaussian(loc=jnp.asarray([0.2, -0.3, 0.1, 0.0]), chol=pose_chol),
            "gap": Gaussian(loc=jnp.asarray([0.5]), chol=0.4 * jnp.ones((1, 1))),
        },
        likelihood=likelihood,
        obs_targets={"nearest": Gaussian(loc=jnp.asarray([0.2]), chol=0.3 * jnp.ones((1, 1)))},
        terminal_targets={"pose": Gaussian(loc=jnp.zeros(4), chol=jnp.eye(4))},
        state_targets={
            "u_u": Gaussian(loc=jnp.zeros(2), chol=jnp.eye(2)),
            "pose": Gaussian(loc=jnp.asarray([0.5, 0.5, 0.0, 0.0]), chol=jnp.eye(4)),
        },
        control={"u_u": Box(-jnp.ones(2), jnp.ones(2))},
    )
    beliefs = system.initial_priors(params)
    ctx = system.planner_context(params, beliefs, present=present)
    init = 0.3 * jnp.sin(jnp.arange(HORIZON * 2).astype(float)).reshape(HORIZON, 2)
    return ctx, init, system, params


def _score_tree(ctx, init):
    return evaluate_policies(
        init[None], ctx.init_beliefs, ctx.transition_fn, ctx.step_terms_fn, ctx.terminal_terms_fn
    )


def _g(ctx, init):
    return total(_score_tree(ctx, init)).midpoint[0]


def _flat(tree):
    return {
        (term, key): (np.asarray(iv.lower), np.asarray(iv.upper))
        for term, leaves in tree.items()
        for key, iv in leaves.items()
    }


def _assert_trees(got, want, label, atol=0.0):
    a, b = _flat(got), _flat(want)
    assert a.keys() == b.keys(), label
    for key in b:
        for side in (0, 1):
            if atol == 0.0:
                np.testing.assert_array_equal(a[key][side], b[key][side], err_msg=f"{label} {key}")
            else:
                np.testing.assert_allclose(
                    a[key][side], b[key][side], rtol=atol, atol=atol, err_msg=f"{label} {key}"
                )


# Whole term trees and gradients differ from the previous routes only where a change replaces a
# factorisation by another of the same matrix (a masked seven-by-seven Cholesky factor by the
# live three-by-three, a solve by a trailing block); float32 agreement is pinned at this bound.
TREE_ATOL = 1e-5

PRESENT_CASES = {
    "absent": None,
    "per-step all-false": {
        "pose_fix": jnp.zeros(HORIZON, bool),
        "gps": jnp.asarray([False, True] * 3),
    },
    "partly present": {"pose_fix": jnp.asarray([True, False] * 3)},
}


@pytest.mark.parametrize("decomposition", ["utility_infogain", "risk_ambiguity"])
@pytest.mark.parametrize("belief", ["posterior", "prior"])
@pytest.mark.parametrize("case", list(PRESENT_CASES))
def test_the_term_trees_and_the_gradient_equal_the_previous_routes(
    decomposition, belief, case, monkeypatch
):
    present = PRESENT_CASES[case]
    ctx, init, *_ = e5_model(decomposition, belief, present)
    tree, grad = _score_tree(ctx, init), jax.grad(lambda p: _g(ctx, p))(init)
    switch_off({"absent", "shared_read", "trailing"}, monkeypatch)
    old, *_ = e5_model(decomposition, belief, present)
    want, want_grad = _score_tree(old, init), jax.grad(lambda p: _g(old, p))(init)
    _assert_trees(tree, want, f"{decomposition}/{belief}/{case}", TREE_ATOL)
    np.testing.assert_allclose(grad, want_grad, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("which", ["absent", "shared_read", "trailing"])
@pytest.mark.parametrize("decomposition", ["utility_infogain", "risk_ambiguity"])
def test_each_change_alone_equals_the_route_without_it(which, decomposition, monkeypatch):
    ctx, init, *_ = e5_model(decomposition)
    tree = _score_tree(ctx, init)
    switch_off({which}, monkeypatch)
    old, *_ = e5_model(decomposition)
    _assert_trees(tree, _score_tree(old, init), f"{which}/{decomposition}", TREE_ATOL)


def test_the_shared_read_changes_nothing_bitwise(monkeypatch):
    """One read of each noise map per step is the same value as two."""
    for decomposition in ("utility_infogain", "risk_ambiguity"):
        ctx, init, *_ = e5_model(decomposition)
        tree = _score_tree(ctx, init)
        switch_off({"shared_read"}, monkeypatch)
        old, *_ = e5_model(decomposition)
        _assert_trees(tree, _score_tree(old, init), f"shared_read/{decomposition}")
        monkeypatch.undo()


def test_the_planned_controls_and_g_are_the_previous_routes(monkeypatch):
    """``G`` of the clamped plan and the 100-step gradient plan, against the previous routes."""
    ctx, init, *_ = e5_model()
    g = float(_g(ctx, init))
    plan = gradient_plan(ctx, init=init, steps=100, metrics=False).plan
    switch_off({"absent", "shared_read", "trailing"}, monkeypatch)
    old, *_ = e5_model()
    assert g == pytest.approx(float(_g(old, init)), rel=1e-6, abs=1e-5)
    want = gradient_plan(old, init=init, steps=100, metrics=False).plan
    np.testing.assert_allclose(plan, want, atol=2e-4)


# ------------------------------------------------------------------- the presence classes


def _kept(ctx_present, monkeypatch=None):
    """The channels ``static_absent`` reports for a ``present=`` of the planner context."""
    _, _, system, params = e5_model(present=ctx_present)
    return _hybrid.static_absent(system._present_in(ctx_present, HORIZON, "test", params))


def test_a_concrete_all_false_column_is_statically_absent():
    assert _kept({"pose_fix": False}) == {0}
    assert _kept({"pose_fix": jnp.zeros(HORIZON, bool)}) == {0}
    assert _kept({"pose_fix": False, "pile": False}) == {0, 3}


def test_a_partial_or_traced_or_per_level_availability_is_not_dropped():
    assert _kept({"pose_fix": jnp.asarray([False] * 5 + [True])}) == frozenset()
    assert _kept(None) == frozenset()

    from hybrid_aif.objectives.presence import Presence

    @jax.jit
    def traced(mask):
        return jnp.asarray(len(_hybrid.static_absent(Presence(mask)))), mask

    assert int(traced(jnp.zeros((HORIZON, 4), bool))[0]) == 0
    from hybrid_aif.objectives.presence import LevelEntry

    never = Presence(jnp.zeros((HORIZON, 4), bool), (LevelEntry(0, 0, jnp.asarray([False, True])),))
    assert _hybrid.static_absent(never) == {1, 2, 3}


def test_a_traced_presence_keeps_the_channel_in_the_computation():
    """Under ``jit`` a traced availability is dynamic: the channel stays in the chain, and the
    leaves are those of the concrete plan."""
    ctx, init, *_ = e5_model()
    seen = {}

    @jax.jit
    def traced(mask_fix, p):
        ctx_t, *_ = e5_model(present={"pose_fix": mask_fix, "gps": jnp.asarray([False, True] * 3)})
        seen["columns"] = set(ctx_t.init_beliefs[0].noises)
        return _score_tree(ctx_t, p)

    got = traced(jnp.zeros(HORIZON, bool), init)
    assert seen["columns"] == {0, 2, 3}
    _assert_trees(got, _score_tree(ctx, init), "traced", TREE_ATOL)


def test_a_python_false_is_static_under_jit():
    """A context built while tracing, with the absent channel declared as a Python ``False``,
    still drops it."""
    ctx, init, *_ = e5_model()
    seen = {}

    @jax.jit
    def built_in_jit(p):
        ctx_t, *_ = e5_model(present={"pose_fix": False, "gps": jnp.asarray([False, True] * 3)})
        seen["columns"] = set(ctx_t.init_beliefs[0].noises)
        return _score_tree(ctx_t, p)

    got = built_in_jit(init)
    assert seen["columns"] == {2, 3}
    _assert_trees(got, _score_tree(ctx, init), "built in jit", TREE_ATOL)


def test_the_dropped_channel_keeps_its_leaves_as_zeros():
    ctx, init, *_ = e5_model()
    tree = _score_tree(ctx, init)
    dropped = [(t, k) for t, leaves in tree.items() for k in leaves if k in ("pose_fix",)]
    assert dropped, "the channel's leaves stay in the tree"
    for term, key in dropped:
        np.testing.assert_array_equal(tree[term][key].lower, 0.0)
        np.testing.assert_array_equal(tree[term][key].upper, 0.0)


def test_a_dropped_channel_is_not_in_the_chain_or_the_readings(monkeypatch):
    """The fused groups and the chain see the live channels only."""
    seen = []
    real = continuous.sub_block_readings

    def spy(q, models, dims, present=None, noises=None):
        seen.append(tuple(m.H.shape for m in models))
        return real(q, models, dims, present, noises)

    monkeypatch.setattr(_hybrid, "sub_block_readings", spy)
    ctx, init, *_ = e5_model()
    jax.make_jaxpr(lambda p: _g(ctx, p))(init)
    assert seen and all((4, 4) not in shapes for shapes in seen)
    pair = ctx.init_beliefs[0]
    assert set(pair.noises) == {2, 3}


# --------------------------------------------------------------- one read of a map per step


def test_the_chain_and_the_leaves_read_the_maps_at_one_mean():
    """The noise factors a pair carries are the maps at its epistemic mean, which is the mean the
    chain conditioned at and the leaves read."""
    ctx, init, system, params = e5_model()
    carry = ctx.init_beliefs
    gps, pile = params.likelihood["gps"], params.likelihood["pile"]
    for t in range(HORIZON):
        carry = ctx.transition_fn(carry, init[t])
        pair = carry[0]
        np.testing.assert_array_equal(pair.noises[2], gps.R_chol_at(pair.epistemic.loc))
        np.testing.assert_array_equal(pair.noises[3], pile.R_chol_at(pair.epistemic.loc))
        np.testing.assert_array_equal(pair.conditioned.loc, pair.epistemic.loc)
        np.testing.assert_array_equal(pair.predictive.loc, pair.epistemic.loc)


def _maps_read(monkeypatch, decomposition, belief):
    """Reads of the two noise maps per plan step, evaluating one plan eagerly."""
    ctx, init, *_ = e5_model(decomposition, belief)
    counts = {"n": 0}
    real = GridNoiseMap.__call__

    def counted(self, loc):
        counts["n"] += 1
        return real(self, loc)

    monkeypatch.setattr(GridNoiseMap, "__call__", counted)
    with jax.disable_jit():
        _score_tree(ctx, init)
    return counts["n"] / HORIZON


def test_each_map_is_read_once_per_step_under_utility_infogain(monkeypatch):
    assert _maps_read(monkeypatch, "utility_infogain", "posterior") == 2.0


def test_the_leaves_read_the_maps_themselves_without_a_chain(monkeypatch):
    assert _maps_read(monkeypatch, "utility_infogain", "prior") == 2.0


def test_conditioning_on_the_expected_reading_leaves_the_mean_bitwise():
    ctx, init, system, params = e5_model()
    q = ctx.init_beliefs[0].conditioned
    for name in ("gps", "pile", "pose_fix"):
        model = params.likelihood[name]
        cond = Conditioner(0, 0, model.H, model.c, model.R_chol_at)
        for flag in (None, jnp.asarray(True), jnp.asarray(False)):
            got = _hybrid.condition_expected(q, cond, flag)
            np.testing.assert_array_equal(got.loc, q.loc)
            R = model.R_chol_at(q.loc)
            shared = _hybrid.condition_expected(q, cond, flag, R)
            np.testing.assert_array_equal(shared.chol, got.chol)
        old = _previous_condition_expected(q, cond, None)
        np.testing.assert_array_equal(old.loc, q.loc)
        np.testing.assert_allclose(
            _hybrid.condition_expected(q, cond, None).chol, old.chol, rtol=0, atol=0
        )


# ------------------------------------------------------------- the trailing block of the factor


def _random_belief(seed, d=4, n=16):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    A = jax.random.normal(k1, (n, d, d)) * 0.5
    cov = A @ jnp.swapaxes(A, -1, -2) + 0.1 * jnp.eye(d)
    return Gaussian(loc=jax.random.normal(k2, (n, d)), chol=jnp.linalg.cholesky(cov))


@pytest.mark.parametrize("dims", [(0, 1, 2), (0,), (0, 1), (1, 3), (3,), (2, 1), (0, 2, 3)])
def test_the_trailing_factor_is_the_conditional_covariance(dims):
    """``L_TT L_TTᵀ`` is ``Σ_TT − Σ_TS Σ_SS⁻¹ Σ_ST`` for leading and non-leading selections."""
    q = _random_belief(0)
    tail, L = continuous._trailing_factor(q, dims)
    got = L @ jnp.swapaxes(L, -1, -2)
    want = jax.vmap(lambda c: continuous._given(c, dims))(q.cov)[..., tail[:, None], tail[None, :]]
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def _channel(seed, rows, d=4):
    H = jax.random.normal(jax.random.PRNGKey(seed), (rows, d))
    return LinearGaussianObservation(H=H, c=jnp.zeros(rows), noise=0.3 * jnp.eye(rows))


@pytest.mark.parametrize("dims", [(0, 1, 2), (0, 1), (1, 3), (3,), (2, 1)])
def test_the_fused_sub_block_leaves_equal_the_general_covariance(dims):
    q = _random_belief(1)
    models = (_channel(1, 2), _channel(2, 1), _channel(3, 2))
    present = jnp.asarray([True, False, True])
    for mask in (None, present):
        got = jax.vmap(lambda g, m=mask: continuous.sub_block_readings(g, models, dims, m))(q)
        want = jax.vmap(lambda g, m=mask: _previous_sub_block_readings(g, models, dims, m))(q)
        for a, b in zip(got.gains, want.gains, strict=True):
            np.testing.assert_allclose(a.lower, b.lower, rtol=1e-4, atol=1e-5)
        for a, b in zip(got.ambiguity, want.ambiguity, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
        for name in ("interaction", "total_correlation", "conditional_correlation"):
            np.testing.assert_allclose(
                getattr(got, name), getattr(want, name), rtol=1e-4, atol=1e-5, err_msg=name
            )


def test_a_non_leading_selection_matches_the_general_route_on_the_model():
    """A selection of ``(y, b)`` out of ``(x, y, yaw, b)`` on the plan model: the whole term trees
    equal those of the general conditional covariance."""
    selection = ("pose:y", "pose:b", "gap")
    ctx, init, *_ = e5_model(epistemic=selection)
    for decomposition in ("utility_infogain", "risk_ambiguity"):
        ctx, init, *_ = e5_model(decomposition, epistemic=selection)
        tree = _score_tree(ctx, init)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_hybrid, "sub_block_readings", _previous_sub_block_readings)
            old, *_ = e5_model(decomposition, epistemic=selection)
            _assert_trees(tree, _score_tree(old, init), decomposition, TREE_ATOL)


def test_no_factorisation_is_taken_for_a_leading_selection():
    """The conditional covariance of a leading selection adds no Cholesky to the stacked reading
    of a state of side four: only the unrolled factorisations of the readings remain."""
    q = _random_belief(2)
    models = (_channel(1, 2), _channel(2, 1))

    def count(dims):
        text = str(
            jax.make_jaxpr(lambda g: continuous.sub_block_readings(g, models, dims))(
                jax.tree.map(lambda x: x[0], q)
            )
        )
        return len(re.findall(r"\bcholesky\b", text))

    assert count((0, 1)) == 0
    assert count((1, 3)) == 0  # the permuted covariance has side four: unrolled
