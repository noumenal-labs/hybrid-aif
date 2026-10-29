"""Plan evaluation on a hybrid model with several linear-Gaussian channels on one
Gaussian-carried factor: the fused evaluation of the epistemic leaves agrees with the general
route, and the lowered cost of ``infer_policies`` stays below a pinned bound.

The model: factors ``ctx`` (categorical, controlled), ``pos`` (2-d Gaussian, controlled by a
3-level thrust node) and ``vel`` (1-d Gaussian); channels ``cue`` (categorical), ``gps`` (2-d)
and ``alt`` (1-d) on ``pos``, ``speed`` (1-d) on ``vel``; ``epistemic=True``, ``policy_len=4``,
6 joint control levels per step, so 1296 policies.
"""

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalGaussian,
    Gaussian,
)
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianTransition
from hybrid_aif.objectives import continuous
from hybrid_aif.system import System, SystemParams

DT = 0.1
THRUSTS = (-1.0, 0.0, 1.0)
GPS = LinearGaussianObservation(H=jnp.eye(2), c=jnp.zeros(2), noise=0.1 * jnp.eye(2))
ALT = LinearGaussianObservation(
    H=jnp.array([[1.0, 0.0]]), c=jnp.zeros(1), noise=jnp.array([[0.05]])
)
SPEED = LinearGaussianObservation(H=jnp.ones((1, 1)), c=jnp.zeros(1), noise=0.2 * jnp.eye(1))

# Lowered cost (XLA flops of the jitted infer_policies call below, CPU) of the plan evaluation
# on 1296 policies, so a later regression is visible:
#   0.14.0 (one entropy leaf per factor)                      110k
#   0.15.0 (per-channel leaves + interaction, composed)       451k
#   fused (one factorisation per factor per step)             322k
# The bound is 1.3x the fused value. The count is XLA's for the installed jax version, so a
# jax upgrade can move it either way with no change to this library.
FLOPS_BOUND = 418_000

# The same model with one innovation noise shared by the three thrust levels (the shared
# ``chol`` of a ``ConditionalGaussian``, the mean per level): the covariance recursion is then
# the same for every policy and is evaluated once per step. The same noise written as equal
# slices per level lowers to 320k flops. The bound is 1.3x the measured 90k.
SHARED_FLOPS_BOUND = 117_000


def _model(shared_noise=False, **overrides):
    system = System(
        A_deps=((0,), (1,), (1,), (2,)),
        B_deps=((), (), ()),
        controlled=(True, True, False),
        policy_len=4,
        modality_names=("cue", "gps", "alt", "speed"),
        factor_names=("ctx", "pos", "vel"),
        epistemic=True,
        **overrides,
    )
    if shared_noise:
        # one noise for every thrust level, the mean per level
        innov_pos = ConditionalGaussian(
            loc=jnp.stack([jnp.array([0.0, DT * u]) for u in THRUSTS]),
            chol=jnp.diag(jnp.array([1e-3, 7.5e-3])),
            parent_ndims=1,
        )
    else:
        innov_pos = ConditionalGaussian(
            loc=jnp.stack([jnp.array([0.0, DT * u]) for u in THRUSTS]),
            chol=jnp.stack([jnp.diag(jnp.array([1e-3, 5e-3 + 5e-3 * abs(u)])) for u in THRUSTS]),
            parent_ndims=1,
        )
    innov_vel = ConditionalGaussian(
        loc=jnp.zeros((1, 1)), chol=0.05 * jnp.ones((1, 1, 1)), parent_ndims=1
    )
    params = SystemParams(
        likelihood={
            "cue": ConditionalCategorical(
                log_table=jnp.log(jnp.array([[0.85, 0.15], [0.15, 0.85]])), parent_ndims=1
            ),
            "gps": GPS,
            "alt": ALT,
            "speed": SPEED,
        },
        transition={
            "ctx": ConditionalCategorical(
                log_table=jnp.log(jnp.array([[0.9, 0.1], [0.1, 0.9]])), parent_ndims=1
            ),
            "pos": LinearGaussianTransition(
                F=jnp.array([[1.0, DT], [0.0, 1.0]]), innovations=innov_pos
            ),
            "vel": LinearGaussianTransition(F=0.9 * jnp.eye(1), innovations=innov_vel),
        },
        initial_state_prior={
            "ctx": Categorical(logits=jnp.zeros(2)),
            "pos": Gaussian(loc=jnp.zeros(2), chol=0.2 * jnp.eye(2)),
            "vel": Gaussian(loc=jnp.zeros(1), chol=0.3 * jnp.eye(1)),
        },
        obs_targets={},
        state_targets={
            "pos": Gaussian(loc=jnp.array([1.0, 0.0]), chol=jnp.diag(jnp.array([0.08, 0.25])))
        },
        terminal_targets={},
    )
    beliefs = {
        "ctx": Categorical(logits=jnp.array([0.2, -0.2])),
        "pos": Gaussian(loc=jnp.array([0.3, -0.1]), chol=jnp.diag(jnp.array([0.2, 0.3]))),
        "vel": Gaussian(loc=jnp.array([0.1]), chol=0.25 * jnp.eye(1)),
    }
    return system, params, beliefs


def _random_gaussians(key, n, d, scale):
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (n, d, d)) * scale
    cov = A @ jnp.swapaxes(A, -1, -2) + 0.05 * jnp.eye(d)
    return Gaussian(loc=jax.random.normal(k2, (n, d)), chol=jnp.linalg.cholesky(cov))


def _mixture_gain(q, model):
    """``I(y; x)`` as the mixture route composes it: the predictive's entropy less the
    conditional entropy."""
    predictive = continuous.predictive_obs(q, model).entropy().lower
    return predictive - continuous.conditional_obs_entropy(model)


def _slogdet_interaction(q, models, present=None):
    """The total correlation of the channels' readings by determinants of the stacked
    covariance."""
    H, R, sizes = continuous._stacked_readings(q, models, present)
    S = H @ q.cov @ jnp.swapaxes(H, -1, -2) + R
    edges = np.cumsum((0, *sizes))
    parts = sum(
        jnp.linalg.slogdet(S[..., a:b, a:b])[1] for a, b in zip(edges[:-1], edges[1:], strict=True)
    )
    return 0.5 * (parts - jnp.linalg.slogdet(S)[1])


def test_fused_leaves_equal_the_general_route():
    """Per-channel gains and the interaction, as the step terms return them, equal the
    library's own evaluation (``state_info_gain``, ``observation_interaction``) on the same
    beliefs."""
    system, params, beliefs = _model()
    context = system.planner_context(params, beliefs)
    n = 256
    pos = _random_gaussians(jax.random.PRNGKey(1), n, 2, 0.5)
    vel = _random_gaussians(jax.random.PRNGKey(2), n, 1, 0.5)
    ctx_belief = jax.tree.map(lambda x: jnp.broadcast_to(x, (n, *x.shape)), context.init_beliefs[0])
    action = jnp.zeros(2, dtype=jnp.int32)
    tree = jax.vmap(lambda a, b, c: context.step_terms_fn((a, b, c), action, jnp.asarray(0)))(
        ctx_belief, pos, vel
    )
    for name, model, q in (("gps", GPS, pos), ("alt", ALT, pos), ("speed", SPEED, vel)):
        expected = _mixture_gain(q, model)
        got = tree["infogain"][name]
        np.testing.assert_allclose(got.lower, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(got.upper, expected, rtol=1e-5, atol=1e-6)
    expected = _slogdet_interaction(pos, (GPS, ALT))
    got = tree["observation_interaction"]["pos"]
    np.testing.assert_allclose(got.lower, expected, atol=1e-4)
    np.testing.assert_allclose(got.upper, expected, atol=1e-4)
    assert set(tree["infogain"]) == {"alt", "cue", "gps", "speed"}
    assert set(tree["observation_interaction"]) == {"pos"}


def test_readings_drop_an_absent_channel_from_the_total_correlation_only():
    """An absent channel leaves the total correlation as a zero head with unit noise does, and
    its gain as declared."""
    pos = _random_gaussians(jax.random.PRNGKey(5), 64, 2, 0.5)
    models = (GPS, ALT, ALT)
    for present in ([True, False, True], [False, True, True], [True, True, True]):
        mask = jnp.asarray(present)
        gains, tc = continuous.linear_gaussian_readings(pos, models, mask)
        np.testing.assert_allclose(tc, _slogdet_interaction(pos, models, mask), atol=1e-4)
        for gain, model in zip(gains, models, strict=True):
            np.testing.assert_allclose(gain.lower, _mixture_gain(pos, model), rtol=1e-5, atol=1e-6)


def test_a_gaussian_predictive_entropy_is_the_mixture_route_s_value():
    """``predictive_entropy`` of a Gaussian belief through a non-switching channel is the
    entropy of the one-component predictive."""
    pos = _random_gaussians(jax.random.PRNGKey(3), 64, 2, 0.5)
    for model in (GPS, ALT):
        direct = continuous.predictive_entropy(pos, model)
        mixture = continuous.predictive_obs(pos, model).entropy()
        np.testing.assert_allclose(direct.lower, mixture.lower, rtol=1e-5, atol=1e-5)


def test_total_correlation_is_the_slogdet_value():
    """The Cholesky evaluation of the total correlation equals the determinant form."""
    pos = _random_gaussians(jax.random.PRNGKey(4), 64, 2, 0.5)
    got = continuous.observation_correlation(pos, (GPS, ALT))
    H = jnp.concatenate([GPS.H, ALT.H])
    R = jax.scipy.linalg.block_diag(GPS.R_chol @ GPS.R_chol.T, ALT.R_chol @ ALT.R_chol.T)
    S = H @ pos.cov @ jnp.swapaxes(H, -1, -2) + R
    parts = jnp.linalg.slogdet(S[..., :2, :2])[1] + jnp.linalg.slogdet(S[..., 2:, 2:])[1]
    np.testing.assert_allclose(got, 0.5 * (parts - jnp.linalg.slogdet(S)[1]), atol=1e-4)


def test_plan_evaluation_cost_is_bounded():
    """The lowered cost of ``infer_policies`` on 1296 policies stays under the fused value
    times 1.3, well under the composed 0.15.0 evaluation."""
    system, params, beliefs = _model()

    def logits(p, b):
        return system.infer_policies(p, b)[0].logits

    compiled = jax.jit(logits).lower(params, beliefs).compile()
    cost = compiled.cost_analysis()
    cost = cost[0] if isinstance(cost, list) else cost
    assert jax.eval_shape(logits, params, beliefs).shape == (1296,)
    assert cost["flops"] <= FLOPS_BOUND, cost["flops"]


def test_plan_evaluation_cost_with_a_shared_innovation_noise_is_bounded():
    """With the noise declared once for every level, the lowered flops of ``infer_policies``
    fall well under the per-level figure, to the shared bound."""
    system, params, beliefs = _model(shared_noise=True)

    def logits(p, b):
        return system.infer_policies(p, b)[0].logits

    cost = jax.jit(logits).lower(params, beliefs).compile().cost_analysis()
    cost = cost[0] if isinstance(cost, list) else cost
    assert cost["flops"] <= SHARED_FLOPS_BOUND, cost["flops"]
    assert SHARED_FLOPS_BOUND < FLOPS_BOUND / 3


# Lines of the compiled HLO of the jitted gradient of ``G`` with respect to the controls, on the
# pose-and-gap plan of ``test_posterior_chain_cost.e5_model`` (two noise maps on a four-dimensional
# factor with three dimensions selected, an always-absent channel, posterior planning belief, 6
# steps), which tracks the run time of the gradient plan where flops do not:
#   before the three changes (an absent channel in the chain, two reads of each map per step,
#   a Cholesky factor of Σ_SS per step)                                   20083 lines, 41.7k flops
#   after                                                                  13102 lines, 13.7k flops
# The bound is 1.3x the figure after. It catches the loss of the shared map read (17422 lines)
# and of all three changes (20083), not the loss of the absent-channel drop (15212) or of the
# trailing block (13444) alone. The count is XLA's for the installed jax version.
POSTERIOR_CHAIN_HLO_LINES_BOUND = 17_100


def test_the_gradient_of_g_on_the_posterior_chain_with_noise_maps_is_bounded():
    from tests.test_posterior_chain_cost import _g, e5_model

    ctx, init, *_ = e5_model()
    text = jax.jit(jax.grad(lambda p: _g(ctx, p))).lower(init).compile().as_text()
    assert len(text.splitlines()) <= POSTERIOR_CHAIN_HLO_LINES_BOUND, len(text.splitlines())
