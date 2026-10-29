"""LG observation-noise learning (design: parameter-learning.md D8, step 3b).

Acceptance 4 (the R half): a two-cell (two-mode) pose model with
known-distinct R_j recovers both noise levels through the face, with the
responsibility weighting doing the credit assignment while the mode is
uncertain — and the ``HΣHᵀ`` correction verified as a *bias* test, not a
comment: an accumulator without it recovers noise biased low by exactly the
posterior-uncertainty term.

Unit surface: the expected-precision substitution (``point()`` is
``diag(sqrt(β/α))`` / ``chol(Ψ/ν)``, deliberately not the mean covariance),
conjugate absorb math for both families, sampling consistency, anchors
required for decay, floors validated, and the deferred-novelty raises.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import Gaussian, InverseGamma, InverseWishart
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianTransition
from tests.test_switching_observation import (
    MODE_PRIOR,
    X_PRIOR,
    make_switch_params,
    make_switch_system,
)

# ------------------------------------------------------------------- unit: InverseGamma


def test_inverse_gamma_point_is_expected_precision_not_mean():
    ig = InverseGamma(alpha=jnp.asarray([4.0]), beta=jnp.asarray([2.0]))
    # E[1/sigma^2] = alpha/beta = 2  =>  effective sigma^2 = 1/2
    np.testing.assert_allclose(np.asarray(ig.point()), [[np.sqrt(0.5)]], rtol=1e-6)
    # mean covariance beta/(alpha-1) = 2/3 is a different number, exposed separately
    np.testing.assert_allclose(np.asarray(ig.mean()), [2.0 / 3.0], rtol=1e-6)


def test_inverse_gamma_absorb_conjugate_math():
    ig = InverseGamma(alpha=jnp.asarray([2.0]), beta=jnp.asarray([1.0]))
    out = ig.absorb(ig.stats(jnp.asarray([0.5]), jnp.asarray(1.0)))
    np.testing.assert_allclose(np.asarray(out.alpha), [2.5])  # + count/2
    np.testing.assert_allclose(np.asarray(out.beta), [1.25])  # + scatter/2


def test_inverse_gamma_decay_requires_anchor():
    ig = InverseGamma(alpha=jnp.asarray([2.0]), beta=jnp.asarray([1.0]))
    with pytest.raises(ValueError, match="requires an explicit alpha0 anchor"):
        ig.absorb(ig.stats(jnp.asarray([0.5]), jnp.asarray(1.0)), decay=0.9)
    anchored = InverseGamma(
        alpha=jnp.asarray([2.0]),
        beta=jnp.asarray([1.0]),
        alpha0=jnp.asarray([2.0]),
        beta0=jnp.asarray([1.0]),
    )
    out = anchored.absorb(anchored.stats(jnp.zeros(1), jnp.asarray(0.0)), decay=0.5)
    np.testing.assert_allclose(np.asarray(out.alpha), [2.0])  # relaxes toward itself


def test_inverse_gamma_sample_concentrates_on_point():
    strength = 4000.0
    ig = InverseGamma.from_R_chol(jnp.asarray([[0.7]]), strength=strength)
    draws = jax.vmap(ig.sample)(jax.random.split(jax.random.PRNGKey(0), 512))
    np.testing.assert_allclose(np.mean(np.asarray(draws) ** 2), 0.49, rtol=0.05)


def test_inverse_gamma_validation_floors():
    with pytest.raises(ValueError, match="strictly positive"):
        InverseGamma(alpha=jnp.asarray([0.0]), beta=jnp.asarray([1.0]))
    with pytest.raises(ValueError, match="strictly positive"):
        InverseGamma(alpha=jnp.asarray([1.0]), beta=jnp.asarray([-1.0]))


def test_deferred_members_raise():
    ig = InverseGamma(alpha=jnp.asarray([2.0]), beta=jnp.asarray([1.0]))
    with pytest.raises(NotImplementedError, match="novelty is deferred"):
        ig.novelty()
    with pytest.raises(NotImplementedError, match="expected precision"):
        ig.expected_log_params()


# ----------------------------------------------------------------- unit: InverseWishart


def test_inverse_wishart_point_and_absorb():
    psi = jnp.asarray([[2.0, 0.4], [0.4, 1.0]])
    iw = InverseWishart(Psi=psi, nu=jnp.asarray(4.0))
    chol = np.asarray(iw.point())
    np.testing.assert_allclose(chol @ chol.T, np.asarray(psi) / 4.0, rtol=1e-6)
    scatter = jnp.asarray([[0.5, 0.1], [0.1, 0.3]])
    out = iw.absorb(iw.stats(scatter, jnp.asarray(1.0)))
    np.testing.assert_allclose(np.asarray(out.nu), 5.0)
    np.testing.assert_allclose(np.asarray(out.Psi), np.asarray(psi + scatter), rtol=1e-6)


def test_inverse_wishart_nu_floor():
    with pytest.raises(ValueError, match="propriety floor"):
        InverseWishart(Psi=jnp.eye(3), nu=jnp.asarray(1.5))


def test_inverse_wishart_sample_concentrates():
    d = 2
    true_r = jnp.asarray([[0.5, 0.1], [0.1, 0.4]])
    nu = jnp.asarray(2000.0)
    iw = InverseWishart(Psi=true_r * nu, nu=nu)
    draws = jax.vmap(iw.sample)(jax.random.split(jax.random.PRNGKey(1), 256))
    covs = np.asarray(draws) @ np.swapaxes(np.asarray(draws), -1, -2)
    np.testing.assert_allclose(covs.mean(axis=0), np.asarray(true_r), rtol=0.08)
    assert covs.shape[-1] == d


# ------------------------------------------------------- the declaration slot (factors)


def test_r_chol_property_lowers_by_expected_precision():
    ig = InverseGamma.from_R_chol(jnp.asarray([[0.7]]), strength=10.0)
    a = LinearGaussianObservation(H=jnp.asarray([[1.0]]), c=jnp.zeros(1), noise=ig)
    assert a.learnable_noise
    np.testing.assert_allclose(np.asarray(a.R_chol), np.asarray(ig.point()), rtol=1e-6)
    point = LinearGaussianObservation(
        H=jnp.asarray([[1.0]]), c=jnp.zeros(1), noise=jnp.asarray([[0.7]])
    )
    assert not point.learnable_noise
    np.testing.assert_allclose(np.asarray(point.R_chol), [[0.7]])


# ------------------------------------------------------------- learning through the face

TRUE_R = (0.3, 0.9)  # the two cells' noise scales, deliberately distinct


def _learnable_switch_params(strength=2.0, r0=0.6):
    """The switching fixture with per-mode noise replaced by a shared-ignorance
    InverseGamma prior over both modes (both start at r0, must separate)."""
    ig = InverseGamma.from_R_chol(jnp.full((2, 1, 1), r0), strength=strength)
    base = make_switch_params()
    return dataclasses.replace(
        base,
        likelihood={"y": dataclasses.replace(base.likelihood["y"], noise=ig)},
    )


def test_switching_noise_learning_requires_detail():
    system, params = make_switch_system(), _learnable_switch_params()
    y = {"y": jnp.asarray([0.8])}
    beliefs, _ = system.infer_states(params, y, priors={"mode": MODE_PRIOR, "x": X_PRIOR})
    with pytest.raises(ValueError, match="detail=True"):
        system.learn(params, y, beliefs)


def _coverage_like_system_and_params():
    """The customer's regime in miniature: a *cell* modality observes the mode
    near-directly (the robot knows which cell it is in), and the pose modality
    ``y`` carries per-cell noise. Two modalities, so the responsibilities that
    weight the noise statistic track the true cell."""
    from hybrid_aif.distributions import ConditionalCategorical, ConditionalGaussian
    from hybrid_aif.system import System

    system = System(
        A_deps=((0,), (1, 0)),  # cell observes mode; y observes x, switched by mode
        B_deps=((), ()),
        controlled=(False, False),
        policy_len=1,
        mixture_k=4,
        mixture_schedule="runnalls",
        modality_names=("cell", "y"),
        factor_names=("mode", "x"),
    )
    ident = jnp.log(jnp.asarray([[0.95, 0.05], [0.05, 0.95]]))
    cell_obs = ConditionalCategorical(log_table=ident, parent_ndims=1)
    stick = ConditionalCategorical(log_table=jnp.log(jnp.asarray([[0.5], [0.5]])), parent_ndims=1)
    x_innov = ConditionalGaussian(
        loc=jnp.zeros((1, 1)), chol=jnp.asarray([[[0.05]]]), parent_ndims=1
    )

    def params_with(noise):
        return dataclasses.replace(
            make_switch_params(),
            likelihood={
                "cell": cell_obs,
                "y": LinearGaussianObservation(H=jnp.asarray([[1.0]]), c=jnp.zeros(1), noise=noise),
            },
            transition={
                "mode": stick,
                "x": LinearGaussianTransition(F=jnp.asarray([[1.0]]), innovations=x_innov),
            },
        )

    return system, params_with


def test_switching_noise_recovery_two_cells():
    """Acceptance 4 (R): both per-cell noise levels recovered through the face,
    credit assigned by the pre-reduction responsibilities (mode inferred from a
    noisy cell observation, never given).

    Uses mildly anchored decay (D7): it washes the early biased-estimate
    phase out of the conjugate average. Measured on this model (seed 7, 8000
    ticks, sustained-15%-band criterion): known-mode-and-pose floor ~115
    ticks; batch EM on 2000 samples ~5 sweeps; online decay=0.999 ~1430
    ticks; pure conjugate ~2620; heavier decay (0.996/0.99) is WORSE — the
    shorter memory raises steady-state variance and the estimate keeps
    leaving the band. Tracking speed and steady-state variance trade against
    each other through one knob, and 0.999 is this model's sweet spot. The
    pose prior is kept comparable to the sensor noise — the customer's
    regime (repeated fixes keep pose uncertainty small); with a broad pose
    prior the EM contraction itself becomes the bottleneck for every
    variant."""
    from hybrid_aif.distributions import MixtureGaussian

    system, params_with = _coverage_like_system_and_params()
    x_prior = MixtureGaussian(
        log_weights=jnp.zeros(1), means=jnp.asarray([[0.0]]), chols=jnp.asarray([[[0.35]]])
    )
    true = dataclasses.replace(
        params_with(jnp.asarray([[[TRUE_R[0]]], [[TRUE_R[1]]]])),
        initial_state_prior={"mode": MODE_PRIOR, "x": x_prior},
    )
    params = dataclasses.replace(
        params_with(InverseGamma.from_R_chol(jnp.full((2, 1, 1), 0.6), strength=2.0, anchor=True)),
        initial_state_prior={"mode": MODE_PRIOR, "x": x_prior},
    )
    env, env_params = system.as_env(true)
    n = 6000
    obs_batch = jax.vmap(lambda k: env.reset(k, env_params)[0])(
        jax.random.split(jax.random.PRNGKey(7), n)
    )
    priors = {"mode": MODE_PRIOR, "x": x_prior}

    @jax.jit
    def run(p0):
        def tick(p, obs_t):
            beliefs, det = system.infer_states(p, obs_t, priors=priors, detail=True)
            return (
                system.learn(
                    p,
                    obs_t,
                    beliefs,
                    detail=det,
                    decay_by_slot={"likelihood:y": 0.999},
                ),
                None,
            )

        return jax.lax.scan(tick, p0, obs_batch)[0]

    learned = run(params).likelihood["y"].noise
    sigma = np.sqrt(np.asarray(learned.beta / learned.alpha))[:, 0]
    np.testing.assert_allclose(sigma, np.asarray(TRUE_R), rtol=0.15)
    assert sigma[0] < 0.5 * sigma[1]  # the contrast, sharper than the tolerance


def test_scatter_carries_posterior_uncertainty():
    """The HΣHᵀ correction as a bias test (at the conjugate fixed point, where
    the identity ``E[(y−Hμ)²] + HΣHᵀ = R`` holds exactly): a residual-only
    accumulator recovers ``R·R/(P+R)`` — biased low by the posterior-gain
    factor — while the face's statistic is unbiased."""
    key = jax.random.PRNGKey(3)
    true_var = 0.25
    prior_var = 1.0
    prior = Gaussian(loc=jnp.zeros(1), chol=jnp.asarray([[np.sqrt(prior_var)]]))
    h, c = jnp.asarray([[1.0]]), jnp.zeros(1)
    r_chol = jnp.asarray([[np.sqrt(true_var)]])
    from hybrid_aif.inference.linear_gaussian import condition

    n = 4000
    keys = jax.random.split(key, n)

    def one(k):
        kx, ky = jax.random.split(k)
        x = jax.random.normal(kx, (1,)) * np.sqrt(prior_var)
        y = x + np.sqrt(true_var) * jax.random.normal(ky, (1,))
        post, _ = condition(prior, h, c, r_chol, y)
        e = y - post.loc
        return e[0] ** 2, e[0] ** 2 + post.chol[0, 0] ** 2

    naive, face = jax.vmap(one)(keys)
    est_naive, est_face = float(jnp.mean(naive)), float(jnp.mean(face))
    biased_expectation = true_var * true_var / (prior_var + true_var)  # R²/(P+R)
    np.testing.assert_allclose(est_naive, biased_expectation, rtol=0.1)
    assert est_naive < true_var * 0.5  # visibly biased low
    np.testing.assert_allclose(est_face, true_var, rtol=0.1)


def test_static_zero_rate_skips_noise_update_bitwise():
    system, params = make_switch_system(), _learnable_switch_params()
    y = {"y": jnp.asarray([0.8])}
    beliefs, det = system.infer_states(
        params, y, priors={"mode": MODE_PRIOR, "x": X_PRIOR}, detail=True
    )
    out = system.learn(params, y, beliefs, detail=det, lr=0.0)
    assert out.likelihood["y"].noise is params.likelihood["y"].noise


def test_novelty_selector_on_lg_noise_raises_deferral():
    from hybrid_aif.system import System

    kwargs = dict(
        A_deps=((1, 0),),
        B_deps=((), ()),
        controlled=(False, False),
        policy_len=1,
        mixture_k=4,
        modality_names=("y",),
        factor_names=("mode", "x"),
    )
    params = _learnable_switch_params()
    with pytest.raises(NotImplementedError, match="deferred"):
        System(**kwargs, epistemic=("likelihood:y",))._term_targets(params)
    with pytest.raises(NotImplementedError, match="must not silently ignore"):
        System(**kwargs, epistemic=True)._term_targets(params)


def test_as_env_lowers_learnable_noise_point_and_sample():
    system, params = make_switch_system(), _learnable_switch_params()
    _, env_params = system.as_env(params)
    lowered = env_params.likelihood["y"].noise
    assert isinstance(lowered, jnp.ndarray)
    np.testing.assert_allclose(
        np.asarray(lowered), np.asarray(params.likelihood["y"].noise.point()), rtol=1e-6
    )
    _, env_params_drawn = system.as_env(params, param_key=jax.random.PRNGKey(0))
    drawn = env_params_drawn.likelihood["y"].noise
    assert isinstance(drawn, jnp.ndarray)
    assert not np.allclose(np.asarray(drawn), np.asarray(lowered))


def test_non_switching_mixture_belief_noise_stats():
    """A learnable noise on a plain (non-switching) LG modality over a
    mixture-carried factor learns from the mixture-weighted scatter."""
    ig = InverseGamma.from_R_chol(jnp.asarray([[0.6]]), strength=2.0)
    a = LinearGaussianObservation(H=jnp.asarray([[1.0]]), c=jnp.zeros(1), noise=ig)
    from tests.test_mixture_filtering import make_params, make_system

    system = make_system()
    params = dataclasses.replace(make_params(), likelihood={"y": a})
    y = {"y": jnp.asarray([0.8])}
    beliefs, _ = system.infer_states(
        params,
        y,
        priors={
            "mode": __import__(
                "hybrid_aif.distributions", fromlist=["Categorical"]
            ).Categorical.from_probs(jnp.asarray([0.6, 0.4])),
            "x": __import__("tests.test_mixture_filtering", fromlist=["X_PRIOR"]).X_PRIOR,
        },
    )
    out = system.learn(params, y, beliefs)
    assert float(out.likelihood["y"].noise.alpha[0]) == pytest.approx(float(ig.alpha[0]) + 0.5)


def test_switching_noise_learning_with_a_head_per_parent_config():
    """A head batched over several switching parents learns the noise it would
    learn shared, when every config's head is the same (it once failed to
    broadcast (A, B, d, x) to (A*B, d, x))."""
    from hybrid_aif.distributions import Categorical, ConditionalCategorical, ConditionalGaussian
    from hybrid_aif.system import System

    system = System(
        A_deps=((2, 0, 1),),  # y observes x, switched by a (2) and b (3)
        B_deps=((), (), ()),
        controlled=(False, False, False),
        policy_len=1,
        mixture_k=6,
        mixture_schedule="runnalls",
        modality_names=("y",),
        factor_names=("a", "b", "x"),
    )
    ig = InverseGamma.from_R_chol(jnp.full((2, 3, 1, 1), 0.6), strength=2.0)
    stick2 = ConditionalCategorical(log_table=jnp.log(jnp.full((2, 1), 0.5)), parent_ndims=1)
    stick3 = ConditionalCategorical(log_table=jnp.log(jnp.full((3, 1), 1 / 3)), parent_ndims=1)
    x_innov = ConditionalGaussian(
        loc=jnp.zeros((1, 1)), chol=jnp.asarray([[[0.05]]]), parent_ndims=1
    )

    def params_with(H):
        return dataclasses.replace(
            make_switch_params(),
            likelihood={"y": LinearGaussianObservation(H=H, c=jnp.zeros(1), noise=ig)},
            transition={
                "a": stick2,
                "b": stick3,
                "x": LinearGaussianTransition(F=jnp.asarray([[1.0]]), innovations=x_innov),
            },
            initial_state_prior={
                "a": Categorical.from_probs(jnp.asarray([0.6, 0.4])),
                "b": Categorical.from_probs(jnp.asarray([0.2, 0.3, 0.5])),
                "x": X_PRIOR,
            },
        )

    y = {"y": jnp.asarray([0.8])}
    out = []
    for H in (jnp.asarray([[1.0]]), jnp.ones((2, 3, 1, 1))):
        params = params_with(H)
        beliefs, detail = system.infer_states(params, y, detail=True)
        learned = system.learn(params, y, beliefs, detail=detail)
        out.append(np.asarray(learned.likelihood["y"].noise.point()))
    np.testing.assert_allclose(out[1], out[0], rtol=1e-5)


def test_switching_noise_stats_lay_the_heads_out_as_inference_does():
    """Distinct heads per parent configuration, two ways of saying the same model:
    parents (a: 2, b: 3) with ``H[i, j]``, and one parent of 6 whose level ``3i + j``
    carries ``H[i, j]`` and whose prior is the product of the two. Inference lays
    the modes out row-major; the noise learned per configuration must agree,
    which pins the statistic to the same order (a column-major flattening would
    learn the wrong configuration's noise from the right residual)."""
    from hybrid_aif.distributions import Categorical, ConditionalCategorical, ConditionalGaussian
    from hybrid_aif.system import System

    x_innov = ConditionalGaussian(
        loc=jnp.zeros((1, 1)), chol=jnp.asarray([[[0.05]]]), parent_ndims=1
    )
    over_parents = jnp.arange(1.0, 7.0).reshape(2, 3, 1, 1) / 3.0  # a distinct head per config
    p_a, p_b = jnp.asarray([0.6, 0.4]), jnp.asarray([0.2, 0.3, 0.5])

    def stick(n):
        return ConditionalCategorical(log_table=jnp.log(jnp.full((n, 1), 1.0 / n)), parent_ndims=1)

    two = System(
        A_deps=((2, 0, 1),),
        B_deps=((), (), ()),
        controlled=(False, False, False),
        policy_len=1,
        mixture_k=6,
        mixture_schedule="runnalls",
        modality_names=("y",),
        factor_names=("a", "b", "x"),
    )
    one = System(
        A_deps=((1, 0),),
        B_deps=((), ()),
        controlled=(False, False),
        policy_len=1,
        mixture_k=6,
        mixture_schedule="runnalls",
        modality_names=("y",),
        factor_names=("ab", "x"),
    )
    lg_x = LinearGaussianTransition(F=jnp.asarray([[1.0]]), innovations=x_innov)
    params_two = dataclasses.replace(
        make_switch_params(),
        likelihood={
            "y": LinearGaussianObservation(
                H=over_parents,
                c=jnp.zeros(1),
                noise=InverseGamma.from_R_chol(jnp.full((2, 3, 1, 1), 0.6), strength=2.0),
            ),
        },
        transition={"a": stick(2), "b": stick(3), "x": lg_x},
        initial_state_prior={
            "a": Categorical.from_probs(p_a),
            "b": Categorical.from_probs(p_b),
            "x": X_PRIOR,
        },
    )
    params_one = dataclasses.replace(
        make_switch_params(),
        likelihood={
            "y": LinearGaussianObservation(
                H=over_parents.reshape(6, 1, 1),
                c=jnp.zeros(1),
                noise=InverseGamma.from_R_chol(jnp.full((6, 1, 1), 0.6), strength=2.0),
            ),
        },
        transition={"ab": stick(6), "x": lg_x},
        initial_state_prior={
            "ab": Categorical.from_probs(jnp.outer(p_a, p_b).reshape(-1)),
            "x": X_PRIOR,
        },
    )
    y = {"y": jnp.asarray([0.8])}
    out = []
    for system, params in ((two, params_two), (one, params_one)):
        beliefs, detail = system.infer_states(params, y, detail=True)
        learned = system.learn(params, y, beliefs, detail=detail)
        out.append(np.asarray(learned.likelihood["y"].noise.point()).reshape(-1))
    assert np.ptp(out[1]) > 1e-3, "the configs must learn different noise for the order to matter"
    np.testing.assert_allclose(out[0], out[1], rtol=1e-5)
