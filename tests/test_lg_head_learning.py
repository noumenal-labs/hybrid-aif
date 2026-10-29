"""LG head learning under MNIW (design: parameter-learning.md D8, step 3c).

Acceptance 4 (the H half): a System with a
:class:`LinearGaussianObservationMNIW` likelihood recovers the true sensor
map, offset, and noise jointly, through the face, from its own forward
model's data. Plus: MatrixNormal / MNIW conjugate math against hand
calculations, joint sampling, ``as_env`` lowering for both learnable-head
forms, the independent-head learning raise (deferred to MNIW, loudly), and
the anti-silent-drop check that an MNIW modality actually conditions
inference (the union-filter hazard).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import (
    ConditionalGaussian,
    Gaussian,
    InverseGamma,
    Learnable,
    MatrixNormal,
)
from hybrid_aif.factors import (
    LinearGaussianObservation,
    LinearGaussianObservationMNIW,
    LinearGaussianTransition,
)
from hybrid_aif.system import System, SystemParams

# ------------------------------------------------------------------ unit: MatrixNormal


def test_matrix_normal_point_and_regression_absorb():
    mn = MatrixNormal.from_head(jnp.asarray([[1.0, 0.0]]), strength=2.0)
    np.testing.assert_allclose(np.asarray(mn.point()), [[1.0, 0.0]])
    # one whitened observation y=2 at x̃=[1, 1]: sxx = x̃x̃ᵀ, syx = y·x̃ᵀ
    x = np.array([1.0, 1.0])
    sxx = jnp.asarray(np.outer(x, x))
    syx = jnp.asarray(2.0 * x[None, :])
    out = mn.absorb(mn.stats(syx, sxx, jnp.asarray(1.0)))
    # hand: Vinv' = 2I + x̃x̃ᵀ; M' = (M·2I + syx) Vinv'⁻¹
    vinv_new = 2.0 * np.eye(2) + np.outer(x, x)
    m_new = (np.array([[2.0, 0.0]]) + np.asarray(syx)) @ np.linalg.inv(vinv_new)
    np.testing.assert_allclose(np.asarray(out.V_inv), vinv_new, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(out.M), m_new, rtol=1e-6)


def test_matrix_normal_sample_concentrates_on_mean():
    mn = MatrixNormal.from_head(jnp.asarray([[1.5, 0.3]]), strength=4000.0)
    draws = jax.vmap(lambda k: mn.sample(k, row_chol=jnp.asarray([[0.5]])))(
        jax.random.split(jax.random.PRNGKey(0), 256)
    )
    np.testing.assert_allclose(np.asarray(draws).mean(0), [[1.5, 0.3]], atol=0.02)


def test_matrix_normal_decay_requires_anchor():
    mn = MatrixNormal.from_head(jnp.asarray([[1.0, 0.0]]), strength=2.0)
    with pytest.raises(ValueError, match="requires an explicit V_inv0 anchor"):
        mn.absorb(mn.stats(jnp.zeros((1, 2)), jnp.eye(2), jnp.asarray(1.0)), decay=0.9)


# ------------------------------------------------------------------------- unit: MNIW


def test_mniw_properties_reproduce_construction():
    m = LinearGaussianObservationMNIW.from_head_noise(
        jnp.asarray([[1.5, 0.3]]), jnp.asarray([[0.4]]), strength=2.0
    )
    np.testing.assert_allclose(np.asarray(m.H), [[1.5]])
    np.testing.assert_allclose(np.asarray(m.c), [0.3])
    np.testing.assert_allclose(np.asarray(m.R_chol), [[0.4]], rtol=1e-6)
    assert m.learnable_head and m.learnable_noise
    assert isinstance(m, Learnable)


def test_mniw_absorb_hand_math():
    m = LinearGaussianObservationMNIW(
        M=jnp.zeros((1, 2)),
        V_inv=jnp.eye(2),
        Psi=jnp.asarray([[2.0]]),
        nu=jnp.asarray(3.0),
    )
    x = np.array([1.0, 1.0])
    y = 2.0
    out = m.absorb(
        m.stats(
            jnp.asarray([[y * y]]),
            jnp.asarray([[y, y]]),
            jnp.asarray(np.outer(x, x)),
            jnp.asarray(1.0),
        )
    )
    vinv_new = np.eye(2) + np.outer(x, x)
    m_new = np.array([[y, y]]) @ np.linalg.inv(vinv_new)
    # Psi' = Phi' − M' Vinv' M'ᵀ with Phi' = Psi + M Vinv Mᵀ + syy
    psi_new = 2.0 + y * y - (m_new @ vinv_new @ m_new.T)[0, 0]
    np.testing.assert_allclose(np.asarray(out.V_inv), vinv_new, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(out.M), m_new, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(out.Psi), [[psi_new]], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(out.nu), 4.0)


def test_mniw_point_and_sample_lower_to_point_declarations():
    m = LinearGaussianObservationMNIW.from_head_noise(
        jnp.asarray([[1.5, 0.3]]), jnp.asarray([[0.4]]), strength=3000.0
    )
    p = m.point()
    assert isinstance(p, LinearGaussianObservation) and not p.learnable_noise
    s = m.sample(jax.random.PRNGKey(1))
    assert isinstance(s, LinearGaussianObservation)
    np.testing.assert_allclose(np.asarray(s.H), [[1.5]], atol=0.1)


# --------------------------------------------------------------- through the System


def _mniw_system_and_params(likelihood, anchor_sensor=False):
    """One latent x; optionally a second, calibrated sensor. Without it,
    (H, R) are jointly unidentifiable from the marginal y ~ N(c, H²P + R) —
    a property of the model, not the update — so the recovery test uses the
    anchored variant (the coverage stack's calibrated odometry/marker fixes
    play this role)."""
    mods = ("y", "anchor") if anchor_sensor else ("y",)
    system = System(
        A_deps=((0,),) * len(mods),
        B_deps=((),),
        controlled=(False,),
        policy_len=1,
        modality_names=mods,
        factor_names=("x",),
    )
    anchor = LinearGaussianObservation(
        H=jnp.asarray([[1.0]]), c=jnp.zeros(1), noise=jnp.asarray([[0.1]])
    )
    params = SystemParams(
        likelihood={"y": likelihood, "anchor": anchor} if anchor_sensor else {"y": likelihood},
        transition={
            "x": LinearGaussianTransition(
                F=jnp.asarray([[1.0]]),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 1)), chol=jnp.asarray([[[0.05]]]), parent_ndims=1
                ),
            ),
        },
        initial_state_prior={"x": Gaussian(loc=jnp.zeros(1), chol=jnp.asarray([[1.0]]))},
    )
    return system, params


TRUE_H, TRUE_C, TRUE_SIGMA = 1.5, 0.3, 0.4


def test_mniw_modality_conditions_inference():
    """Anti-silent-drop: an MNIW likelihood must move the posterior, not pass
    the prior through (the positive-filter hazard from the step-1 audit)."""
    m = LinearGaussianObservationMNIW.from_head_noise(
        jnp.asarray([[TRUE_H, TRUE_C]]), jnp.asarray([[TRUE_SIGMA]]), strength=2.0
    )
    system, params = _mniw_system_and_params(m)
    obs = system.from_positional("observations", (jnp.asarray([1.2]),))
    beliefs, det = system.infer_states(params, obs, detail=True)
    (modality,) = obs
    assert abs(float(beliefs["x"].loc[0])) > 1e-3  # prior mean was 0
    assert det.log_evidence[modality] is not None


def test_mniw_joint_recovery_of_head_and_noise():
    """Acceptance 4 (H): the sensor map, offset, and noise recovered jointly
    through the face from the System's own forward samples."""
    true = LinearGaussianObservation(
        H=jnp.asarray([[TRUE_H]]), c=jnp.asarray([TRUE_C]), noise=jnp.asarray([[TRUE_SIGMA]])
    )
    system, true_params = _mniw_system_and_params(true, anchor_sensor=True)
    start = LinearGaussianObservationMNIW.from_head_noise(
        jnp.asarray([[1.0, 0.0]]), jnp.asarray([[0.8]]), strength=2.0
    )
    _, params = _mniw_system_and_params(start, anchor_sensor=True)
    env, env_params = system.as_env(true_params)
    obs_batch = jax.vmap(lambda k: env.reset(k, env_params)[0])(
        jax.random.split(jax.random.PRNGKey(11), 4000)
    )

    @jax.jit
    def run(p0):
        def tick(p, obs_t):
            beliefs, det = system.infer_states(p, obs_t, detail=True)
            return system.learn(p, obs_t, beliefs, detail=det), None

        return jax.lax.scan(tick, p0, obs_batch)[0]

    learned = run(params).likelihood["y"]
    sigma = float(learned.R_chol[0, 0])
    np.testing.assert_allclose(float(learned.H[0, 0]), TRUE_H, rtol=0.1)
    np.testing.assert_allclose(float(learned.c[0]), TRUE_C, atol=0.1)
    np.testing.assert_allclose(sigma, TRUE_SIGMA, rtol=0.2)


def test_independent_head_learning_raises_pointing_at_mniw():
    a = LinearGaussianObservation(
        head=MatrixNormal.from_head(jnp.asarray([[1.0, 0.0]]), strength=2.0),
        noise=jnp.asarray([[0.5]]),
    )
    system, params = _mniw_system_and_params(a)
    obs = system.from_positional("observations", (jnp.asarray([0.5]),))
    beliefs, _ = system.infer_states(params, obs)
    with pytest.raises(NotImplementedError, match="LinearGaussianObservationMNIW"):
        system.learn(params, obs, beliefs)
    # frozen slot is the documented workaround
    out = system.learn(params, obs, beliefs, lr_by_slot={"likelihood:y": 0.0})
    assert out.likelihood["y"].head is params.likelihood["y"].head


def test_as_env_lowers_learnable_head_point_and_sample():
    a = LinearGaussianObservation(
        head=MatrixNormal.from_head(jnp.asarray([[1.5, 0.3]]), strength=2.0),
        noise=InverseGamma.from_R_chol(jnp.asarray([[0.5]]), strength=2.0),
    )
    system, params = _mniw_system_and_params(a)
    _, env_params = system.as_env(params)
    low = env_params.likelihood["y"]
    assert isinstance(low.head, jnp.ndarray) and isinstance(low.noise, jnp.ndarray)
    np.testing.assert_allclose(np.asarray(low.H), [[1.5]], rtol=1e-6)
    _, drawn = system.as_env(params, param_key=jax.random.PRNGKey(2))
    assert not np.allclose(np.asarray(drawn.likelihood["y"].head), np.asarray(low.head))

    m = LinearGaussianObservationMNIW.from_head_noise(
        jnp.asarray([[1.5, 0.3]]), jnp.asarray([[0.4]]), strength=2.0
    )
    _, params_m = _mniw_system_and_params(m)
    _, envp = system.as_env(params_m)
    assert isinstance(envp.likelihood["y"], LinearGaussianObservation)
    _, envp_drawn = system.as_env(params_m, param_key=jax.random.PRNGKey(3))
    assert isinstance(envp_drawn.likelihood["y"], LinearGaussianObservation)


def test_mniw_novelty_selector_resolves():
    m = LinearGaussianObservationMNIW.from_head_noise(
        jnp.asarray([[1.0, 0.0]]), jnp.asarray([[0.5]]), strength=2.0
    )
    system, params = _mniw_system_and_params(m)
    sel = dataclasses.replace(system, epistemic=("likelihood:y",))
    assert sel._term_targets(params) == ((), (0,), ())
