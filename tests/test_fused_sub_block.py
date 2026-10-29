"""The fused evaluation of the epistemic leaves over a dimension selection.

A factor of four dimensions ``(x, y, yaw, b)`` with ``(x, y, yaw)`` selected is read by three
channels: ``xy`` reads the two-dimensional block ``(x, y)``, ``full`` reads all four dimensions
(the unselected ``b`` too), ``alt`` reads a second factor. A ``GridNoiseMap`` gives two channels
a noise that varies over the state. The leaves of the fused route
(:func:`~hybrid_aif.objectives.continuous.sub_block_readings`, called from the step terms) equal
the general route's (``state_info_gain``, ``observation_interaction``, ``lg_ambiguity``,
``observation_correlation``, ``conditional_correlation``) under both decompositions, on the prior
and the posterior chain, with per-step presence; and the whole term trees of the model are the
same under either route.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import ConditionalGaussian, Gaussian
from hybrid_aif.factors import (
    GridNoiseMap,
    LinearGaussianObservation,
    LinearGaussianTransition,
)
from hybrid_aif.objectives import continuous
from hybrid_aif.objectives import hybrid as _hybrid
from hybrid_aif.system import System, SystemParams

NAMES = ("x", "y", "yaw", "b")
SELECTION = (0, 1, 2)
CHANNELS = ("xy", "full", "alt")
STEPS = 3


def _map(sd_far, sd_near):
    values = np.zeros((4, 4, 2, 2), np.float32)
    values[:] = np.diag([sd_far, sd_far])
    values[3, 1] = np.diag([sd_near, sd_near])
    return GridNoiseMap(jnp.asarray(values), (-1.0, -1.0), (1.0, 1.0), axes=(0, 1))


def _models():
    h_xy = jnp.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    h_full = jnp.asarray([[0.7, 0.2, 0.0, 1.0], [0.0, 0.5, 1.0, -0.6]])
    return {
        "xy": LinearGaussianObservation(H=h_xy, c=jnp.zeros(2), noise=_map(0.5, 0.05)),
        "full": LinearGaussianObservation(H=h_full, c=jnp.zeros(2), noise=_map(0.4, 0.1)),
        "alt": LinearGaussianObservation(
            H=jnp.ones((1, 1)), c=jnp.zeros(1), noise=0.3 * jnp.eye(1)
        ),
    }


def _model(**overrides):
    system = System(
        A_deps=((0,), (0,), (1,)),
        B_deps=((), ()),
        controlled=(True, False),
        policy_len=STEPS,
        modality_names=CHANNELS,
        factor_names=("pose", "z"),
        dims={"pose": NAMES},
        epistemic=("pose:x", "pose:y", "pose:yaw", "z"),
        **overrides,
    )
    F = jnp.eye(4) + 0.1 * jnp.asarray(
        [[0, 0, 0, 1], [0, 0, 1, 0], [0.5, 0, 0, 0.2], [0, 0, 0, 0]], jnp.float32
    )
    pose_noise = ConditionalGaussian(
        loc=jnp.zeros((2, 4)),
        chol=jnp.broadcast_to(jnp.diag(jnp.asarray([0.1, 0.1, 0.05, 0.02])), (2, 4, 4)),
        parent_ndims=1,
    )
    params = SystemParams(
        likelihood=_models(),
        transition={
            "pose": LinearGaussianTransition(F=F, innovations=pose_noise),
            "z": LinearGaussianTransition(
                F=jnp.eye(1),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 1)), chol=0.1 * jnp.ones((1, 1, 1)), parent_ndims=1
                ),
            ),
        },
        initial_state_prior={
            "pose": _prior(),
            "z": Gaussian(loc=jnp.zeros(1), chol=jnp.ones((1, 1))),
        },
    )
    return system, params


def _prior():
    chol = jnp.asarray(
        [
            [0.5, 0, 0, 0],
            [0.2, 0.4, 0, 0],
            [-0.1, 0.1, 0.3, 0],
            [0.3, -0.2, 0.2, 0.8],
        ],
        jnp.float32,
    )
    return Gaussian(loc=jnp.asarray([0.1, -0.2, 0.0, 0.3]), chol=chol)


def _random_pose(key, n):
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (n, 4, 4)) * 0.4
    cov = A @ jnp.swapaxes(A, -1, -2) + 0.05 * jnp.eye(4)
    return Gaussian(loc=0.5 * jax.random.normal(k2, (n, 4)), chol=jnp.linalg.cholesky(cov))


def _tree_leaves(tree):
    return {(term, key): leaf for term, blocks in tree.items() for key, leaf in blocks.items()}


def _close(got, want, label, rtol=1e-5, atol=1e-5):
    assert set(got) == set(want), label
    for key in want:
        for side in ("lower", "upper"):
            np.testing.assert_allclose(
                getattr(got[key], side),
                getattr(want[key], side),
                rtol=rtol,
                atol=atol,
                err_msg=f"{label} {key} {side}",
            )


def _forced_general(monkeypatch):
    """The general route throughout: no group takes a fused evaluation, whole factor or
    sub-block, under either decomposition."""
    monkeypatch.setattr(_hybrid, "_shares_factorisation", lambda chans, masks: False)
    monkeypatch.setattr(_hybrid, "_fuses_correlation", lambda chans: False)

    def refused(*args, **kwargs):
        raise AssertionError("a fused evaluation ran under the forced general route")

    monkeypatch.setattr(_hybrid, "linear_gaussian_readings", refused)
    monkeypatch.setattr(_hybrid, "sub_block_readings", refused)


def _run(system, params, present=None):
    """The carries after each transition and the step trees scored on them."""
    ctx = system.planner_context(params, system.initial_priors(params), present=present)
    carry, trees = ctx.init_beliefs, []
    for t in range(system.policy_len):
        action = jnp.asarray([0], jnp.int32)
        carry = ctx.transition_fn(carry, action)
        trees.append(ctx.step_terms_fn(carry, action, jnp.asarray(t, jnp.int32)))
    return trees


PRESENT = {"full": jnp.asarray([True, False, True]), "xy": jnp.asarray([True, True, False])}


def _step_tree(ctx, pose, z, t):
    """The step terms at step ``t`` on a batch of beliefs of the two factors."""
    action = jnp.zeros(1, dtype=jnp.int32)
    return jax.vmap(lambda a, b: ctx.step_terms_fn((a, b), action, jnp.asarray(t)))(pose, z)


def _present_mask(present, t):
    if present is None:
        return None
    return jnp.asarray([bool(present[k][t]) for k in ("xy", "full")])


def _absent_zeroed(leaves, present, t):
    """The leaves of the channels ``xy`` and ``full`` with an absent channel's at zero, as the
    step terms mask it."""
    if present is None:
        return leaves
    return {
        k: jax.tree.map(lambda x, keep=bool(present[k][t]): x if keep else 0.0 * x, leaves[k])
        for k in leaves
    }


def _batch(seed, n=128):
    pose = _random_pose(jax.random.PRNGKey(seed), n)
    return pose, Gaussian(loc=jnp.zeros((n, 1)), chol=jnp.ones((n, 1, 1)))


@pytest.mark.parametrize("present", [None, PRESENT], ids=["always", "per-step presence"])
def test_the_fused_leaves_equal_the_general_route_utility_infogain(present):
    """Per-channel ``I(y_m; x_S)`` and the interaction ``TC(Y) − TC(Y | x_S)`` on random beliefs."""
    system, params = _model(planning_belief="prior")
    ctx = system.planner_context(params, system.initial_priors(params), present=present)
    pose, z = _batch(0)
    models = _models()
    pair = (models["xy"], models["full"])
    expected_gains = jax.vmap(
        lambda q: continuous.state_info_gain(
            (q, q), pair, ("xy", "full"), dims=(SELECTION, SELECTION)
        )["infogain"]
    )(pose)
    for t in range(STEPS):
        tree = _step_tree(ctx, pose, z, t)
        got = {k: tree["infogain"][k] for k in ("xy", "full")}
        _close(
            got, _absent_zeroed(expected_gains, present, t), f"infogain t={t}", rtol=1e-5, atol=1e-6
        )
        mask = _present_mask(present, t)
        want = jax.vmap(
            lambda q, m=mask: continuous.observation_interaction(q, pair, SELECTION, m)
        )(pose)
        for side in ("lower", "upper"):
            np.testing.assert_allclose(
                getattr(tree["observation_interaction"]["pose"], side), want, atol=1e-4
            )


@pytest.mark.parametrize("present", [None, PRESENT], ids=["always", "per-step presence"])
def test_the_fused_leaves_equal_the_general_route_risk_ambiguity(present):
    """Each channel's ``E H[y_m | x_S]``, ``TC(Y)`` and ``−TC(Y | x_S)`` on random beliefs."""
    system, params = _model(planning_belief="prior", decomposition="risk_ambiguity")
    ctx = system.planner_context(params, system.initial_priors(params), present=present)
    pose, z = _batch(1)
    models = _models()
    pair = (models["xy"], models["full"])
    expected = jax.vmap(
        lambda q: continuous.lg_ambiguity(
            pair, ((), ()), ("xy", "full"), q_blocks=(q, q), dims=(SELECTION, SELECTION)
        )["ambiguity"]
    )(pose)
    for t in range(STEPS):
        tree = _step_tree(ctx, pose, z, t)
        got = {k: tree["ambiguity"][k] for k in ("xy", "full")}
        _close(got, _absent_zeroed(expected, present, t), f"ambiguity t={t}", rtol=1e-5, atol=1e-5)
        mask = _present_mask(present, t)
        tc = jax.vmap(lambda q, m=mask: continuous.observation_correlation(q, pair, m))(pose)
        cc = jax.vmap(lambda q, m=mask: continuous.conditional_correlation(q, pair, SELECTION, m))(
            pose
        )
        np.testing.assert_allclose(tree["observation_correlation"]["pose"].lower, tc, atol=1e-4)
        np.testing.assert_allclose(tree["observation_interaction"]["pose"].lower, -cc, atol=1e-4)


@pytest.mark.parametrize("planning_belief", ["prior", "posterior"])
@pytest.mark.parametrize("decomposition", ["utility_infogain", "risk_ambiguity"])
@pytest.mark.parametrize("present", [None, PRESENT], ids=["always", "per-step presence"])
def test_the_term_trees_are_unchanged_by_the_route(
    monkeypatch, planning_belief, decomposition, present
):
    """The whole term trees of the model, step by step along the chain, hold the same keys and
    the same values under the fused route as under the general one."""
    system, params = _model(planning_belief=planning_belief, decomposition=decomposition)
    fused = _run(system, params, present)
    _forced_general(monkeypatch)
    general = _run(system, params, present)
    for step, (a, b) in enumerate(zip(fused, general, strict=True)):
        assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
        got, want = _tree_leaves(a), _tree_leaves(b)
        assert set(got) == set(want)
        for key in want:
            for side in ("lower", "upper"):
                np.testing.assert_allclose(
                    getattr(got[key], side),
                    getattr(want[key], side),
                    rtol=1e-5,
                    atol=1e-4,
                    err_msg=f"{step} {key} {side}",
                )


def test_the_fused_route_is_taken_over_a_selection(monkeypatch):
    """The general route's per-channel leaf is not called when the fused one covers the
    factor's channels."""
    calls = []
    original = continuous.sub_block_obs_entropy

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(continuous, "sub_block_obs_entropy", spy)
    system, params = _model(planning_belief="prior")
    _run(system, params)
    assert not calls
    _forced_general(monkeypatch)
    _run(system, params)
    assert calls
