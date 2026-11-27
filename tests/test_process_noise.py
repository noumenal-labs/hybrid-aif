"""Learnable process noise: a Gaussian transition's ``noise_prior``, learned from the pairwise
residual scatter, and a noise held once as a named parameter variable.

Pinned by (the design record's "What to pin"): the statistic against a hand-rolled lag-one
smoother, exactly (1); recovery of a known ``Q``, and the marginal-only statistic recovering a
larger one (2); the control slice and a ``G u`` input in the residual (3); an affine ``f`` equal
to the linear case bitwise at first order (4); a replay on a sixteen-dimensional model of the
customer's shape moving a scaled ``Q`` toward the generating one (5); the description's round trip
and schema, the description length and the evidence (6); the refusals (7); the noise home, read
by two transitions, by a likelihood and a transition, recorded once, and refused under aliasing
(9). Every existing test and the notebook gate are pin 8. Also the lowering (every consumer
reads the prior's point) and the rate keys.

Design: docs/design/process-noise-learning.md D1-D4; parameter-learning.md D8 (the observation
side this mirrors); shared-parameters.md (the 2026-11-05 amendment).
"""

import dataclasses
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import (
    Box,
    ConditionalGaussian,
    Gaussian,
    LinearGaussianObservation,
    LinearGaussianTransition,
    MixtureGaussian,
    NonlinearGaussianTransition,
    Parameter,
    System,
    SystemParams,
)
from hybrid_aif.description import build, describe, schema
from hybrid_aif.description.mdl import description_length
from hybrid_aif.distributions import InverseGamma, InverseWishart
from hybrid_aif.factors import transition_point
from hybrid_aif.learning.evidence import log_evidence
from hybrid_aif.replay import replay

F = np.array([[0.9, 0.1], [0.0, 0.8]])
B = np.array([0.05, -0.02])
Q_TRUE = np.array([[0.3, 0.1], [0.1, 0.2]])
R = 0.05
H = np.eye(2)


def _f32(x):
    return jnp.asarray(x, jnp.float32)


def _iw(scale: float = 1.0, nu: float = 6.0, batch: tuple = ()) -> InverseWishart:
    """An inverse-Wishart whose point is ``scale · I``."""
    psi = np.broadcast_to(scale * nu * np.eye(2), (*batch, 2, 2))
    return InverseWishart(Psi=_f32(psi), nu=_f32(np.full(batch, nu)))


def _innovations(modes: int = 1, loc=None) -> ConditionalGaussian:
    loc = np.broadcast_to(B, (modes, 2)) if loc is None else np.asarray(loc)
    # the declared factor is ignored under a prior: only its shape is a declaration
    return ConditionalGaussian(
        loc=_f32(loc), chol=_f32(np.broadcast_to(np.eye(2), (modes, 2, 2))), parent_ndims=1
    )


def _transition(prior=None, **kwargs) -> LinearGaussianTransition:
    return LinearGaussianTransition(
        F=_f32(F), innovations=_innovations(), noise_prior=prior, **kwargs
    )


def _observation(noise=None) -> LinearGaussianObservation:
    noise = _f32(np.sqrt(R) * np.eye(2)) if noise is None else noise
    return LinearGaussianObservation(H=_f32(H), c=_f32(np.zeros(2)), noise=noise)


def _system(**kwargs) -> System:
    defaults = dict(
        A_deps=((0,),),
        B_deps=((),),
        state_names=("z",),
        modality_names=("y",),
        policy_len=1,
        epistemic=False,
    )
    return System(**{**defaults, **kwargs})


def _params(prior=None, transition=None, **kwargs) -> SystemParams:
    return SystemParams(
        transition={"z": _transition(prior) if transition is None else transition},
        initial_state_prior={"z": Gaussian(loc=_f32([0.2, -0.1]), chol=_f32(0.5 * np.eye(2)))},
        likelihood={"y": _observation()},
        **kwargs,
    )


Y0, Y1 = _f32([0.1, 0.2]), _f32([0.4, -0.1])


def _two_ticks(system, params, action=None, y0=Y0, y1=Y1):
    """The filtered posteriors at two ticks: ``(b0, b1)``."""
    b0, _ = system.infer_states(params, {"y": y0})
    pred = system.update_empirical_prior(params, b0, {} if action is None else action)
    b1, _ = system.infer_states(params, {"y": y1}, priors=pred)
    return b0, b1


def _increment(before: InverseWishart, after: InverseWishart):
    return np.asarray(after.Psi) - np.asarray(before.Psi), np.asarray(after.nu - before.nu)


def _joint_smoother(mu, cov, F, b, Q, H, R, y):
    """The lag-one smoother by joint conditioning, in float64: the Gaussian over ``(x, x')`` from
    ``N(mu, cov)`` and the step, conditioned on ``y = H x' + v``; returns the expected residual
    scatter ``E[(x' − F x − b)(·)ᵀ]``. Not the RTS recursion the library uses."""
    mu, cov, F, b, Q, H, R, y = (np.asarray(a, np.float64) for a in (mu, cov, F, b, Q, H, R, y))
    d = mu.shape[0]
    mean = np.concatenate([mu, F @ mu + b])
    joint = np.block([[cov, cov @ F.T], [F @ cov, F @ cov @ F.T + Q]])
    Hz = np.concatenate([np.zeros_like(H), H], axis=1)
    S = Hz @ joint @ Hz.T + R
    K = joint @ Hz.T @ np.linalg.inv(S)
    post_mean = mean + K @ (y - Hz @ mean)
    post = joint - K @ Hz @ joint
    A = np.concatenate([-F, np.eye(d)], axis=1)
    r = A @ post_mean - b
    return A @ post @ A.T + np.outer(r, r)


def _rts(mu, cov, mu1, cov1, F, b, Q):
    """The scatter by the RTS one-step smoother written out in float64 numpy."""
    mu, cov, mu1, cov1, F, b, Q = (np.asarray(a, np.float64) for a in (mu, cov, mu1, cov1, F, b, Q))
    pred = F @ cov @ F.T + Q
    J = cov @ F.T @ np.linalg.inv(pred)
    mu_s = mu + J @ (mu1 - F @ mu - b)
    cov_s = cov + J @ (cov1 - pred) @ J.T
    C = cov1 @ J.T
    r = mu1 - F @ mu_s - b
    return np.outer(r, r) + cov1 - F @ C.T - C @ F.T + F @ cov_s @ F.T


# ------------------------------------------------------------------- D1: the declaration


def test_the_prior_is_a_field_and_learnable_noise_says_so():
    assert not _transition().learnable_noise
    assert _transition(_iw()).learnable_noise
    assert not _transition(Parameter("Q")).learnable_noise  # resolved by the faces
    with pytest.raises(TypeError, match="InverseGamma or InverseWishart"):
        _transition(_f32(np.eye(2)))
    with pytest.raises(ValueError, match="2-dimensional"):
        _transition(InverseWishart(Psi=_f32(np.eye(3)), nu=_f32(5.0)))
    with pytest.raises(ValueError, match="leading axes"):
        _transition(_iw(batch=(3,)))


def test_every_consumer_reads_the_priors_point_and_ignores_the_declared_factor():
    system = _system()
    prior = _iw(scale=0.4)
    learnable = _params(prior)
    point = transition_point(learnable.transition["z"])
    assert point.noise_prior is None
    np.testing.assert_allclose(np.asarray(point.innovations.chol), np.asarray(prior.point()))
    fixed = _params(transition=point)
    b0, _ = system.infer_states(learnable, {"y": Y0})
    got = system.update_empirical_prior(learnable, b0, {})["z"]
    want = system.update_empirical_prior(fixed, b0, {})["z"]
    np.testing.assert_array_equal(np.asarray(got.cov), np.asarray(want.cov))
    # the declared (identity) factor would give a different prediction
    declared = _params(transition=_transition())
    other = system.update_empirical_prior(declared, b0, {})["z"]
    assert not np.allclose(np.asarray(other.cov), np.asarray(got.cov))


def test_an_environment_samples_at_the_point_or_from_the_posterior():
    system = _system()
    params = _params(_iw(scale=0.4))
    env, env_params = system.as_env(params)
    lowered = env_params.transition["z"]
    assert lowered.noise_prior is None
    np.testing.assert_allclose(
        np.asarray(lowered.innovations.chol), np.asarray(_iw(scale=0.4).point()), rtol=1e-6
    )
    _, drawn = system.as_env(params, param_key=jax.random.PRNGKey(3))
    chol = np.asarray(drawn.transition["z"].innovations.chol)
    assert chol.shape == (2, 2) and not np.allclose(chol, lowered.innovations.chol)


def test_the_rate_key_freezes_or_rates_the_noise():
    system = _system()
    params = _params(_iw())
    b0, b1 = _two_ticks(system, params)
    frozen = system.learn(
        params, {"y": Y1}, b1, prev_beliefs=b0, lr_by_slot={"transition:z:noise": 0.0}
    )
    np.testing.assert_array_equal(
        np.asarray(frozen.transition["z"].noise_prior.Psi), np.asarray(_iw().Psi)
    )
    half = system.learn(
        params, {"y": Y1}, b1, prev_beliefs=b0, lr_by_slot={"transition:z:noise": 0.5}
    )
    full = system.learn(params, {"y": Y1}, b1, prev_beliefs=b0)
    d_half, n_half = _increment(_iw(), half.transition["z"].noise_prior)
    d_full, n_full = _increment(_iw(), full.transition["z"].noise_prior)
    np.testing.assert_allclose(d_half, 0.5 * d_full, rtol=1e-5)
    assert float(n_half) == pytest.approx(0.5) and float(n_full) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="prev_beliefs"):
        system.learn(params, {"y": Y1}, b1)


# ------------------------------------------------------------- 1. the exact statistic


def test_the_statistic_is_the_lag_one_smoothers_scatter_exactly():
    system = _system()
    prior = _iw(scale=0.4)
    params = _params(prior)
    b0, b1 = _two_ticks(system, params)
    learned = system.learn(params, {"y": Y1}, b1, prev_beliefs=b0)
    scatter, count = _increment(prior, learned.transition["z"].noise_prior)
    want = _joint_smoother(
        b0["z"].loc,
        b0["z"].cov,
        F,
        B,
        np.asarray(prior.point()) @ np.asarray(prior.point()).T,
        H,
        R * np.eye(2),
        Y1,
    )
    np.testing.assert_allclose(scatter, want, rtol=1e-4, atol=1e-6)
    assert float(count) == 1.0


def test_an_inverse_gamma_takes_the_diagonal():
    system = _system()
    ig = InverseGamma.from_R_chol(_f32(np.sqrt(0.4) * np.eye(2)), strength=4.0)
    iw = _iw(scale=0.4)
    np.testing.assert_allclose(np.asarray(ig.point()), np.asarray(iw.point()), rtol=1e-6)
    b0, b1 = _two_ticks(system, _params(iw))
    full = system.learn(_params(iw), {"y": Y1}, b1, prev_beliefs=b0).transition["z"].noise_prior
    diag = system.learn(_params(ig), {"y": Y1}, b1, prev_beliefs=b0).transition["z"].noise_prior
    scatter, _ = _increment(iw, full)
    np.testing.assert_allclose(
        2.0 * (np.asarray(diag.beta) - np.asarray(ig.beta)), np.diag(scatter), rtol=1e-5
    )
    np.testing.assert_allclose(np.asarray(diag.alpha) - np.asarray(ig.alpha), 0.5, rtol=1e-6)


# ---------------------------------------------------------------------- 2. recovery


def _trace(T: int, seed: int = 0, F=F, b=B, Q=Q_TRUE, d=2, r=R):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(Q)
    x, ys = np.zeros(d), []
    for _ in range(T):
        x = F @ x + b + L @ rng.standard_normal(d)
        ys.append(x + np.sqrt(r) * rng.standard_normal(d))
    return _f32(np.array(ys))


def test_a_known_q_is_recovered_and_the_marginal_statistic_overstates_it():
    system = _system()
    nu0 = 4.0
    prior = InverseWishart(Psi=_f32(nu0 * np.eye(2)), nu=_f32(nu0))  # broad, centred at I
    ys = _trace(800)
    out = replay(
        system,
        _params(prior),
        {"y": ys},
        {},
        learn=True,
        lr=0.0,
        lr_by_slot={"transition:z:noise": 1.0},
        metrics=False,
    )
    post = out.params.transition["z"].noise_prior
    learned = np.asarray(post.Psi) / (float(post.nu) - 3.0)
    np.testing.assert_allclose(learned, Q_TRUE, atol=0.06)
    # the same posteriors through the two marginals alone, the cross-covariance dropped
    mu, cov = np.asarray(out.beliefs["z"].loc), np.asarray(out.beliefs["z"].cov)
    marginal = nu0 * np.eye(2)
    for t in range(1, ys.shape[0]):
        r = mu[t] - F @ mu[t - 1] - B
        marginal = marginal + np.outer(r, r) + cov[t] + F @ cov[t - 1] @ F.T
    marginal = marginal / (float(post.nu) - 3.0)
    assert np.trace(marginal) > np.trace(learned) + 0.1
    assert np.linalg.norm(marginal - Q_TRUE) > 2.0 * np.linalg.norm(learned - Q_TRUE)


# --------------------------------------------------------- 3. the control slice and G u


def _controlled():
    system = _system(controlled=(True,))
    locs = np.array([[0.05, -0.02], [-0.3, 0.2]])
    Fs = np.stack([F, 0.5 * F])
    prior = _iw(scale=0.4, batch=(2,))
    transition = LinearGaussianTransition(
        F=_f32(Fs), innovations=_innovations(2, locs), noise_prior=prior
    )
    return system, _params(transition=transition), prior, Fs, locs


def test_the_taken_levels_mode_alone_learns():
    system, params, prior, Fs, locs = _controlled()
    action = {"a_z": jnp.asarray(1)}
    b0, b1 = _two_ticks(system, params, action)
    learned = system.learn(params, {"y": Y1}, b1, prev_beliefs=b0, action=action)
    scatter, count = _increment(prior, learned.transition["z"].noise_prior)
    np.testing.assert_array_equal(scatter[0], 0.0)
    np.testing.assert_allclose(count, [0.0, 1.0])
    Q = np.asarray(prior.point()[1]) @ np.asarray(prior.point()[1]).T
    want = _rts(b0["z"].loc, b0["z"].cov, b1["z"].loc, b1["z"].cov, Fs[1], locs[1], Q)
    np.testing.assert_allclose(scatter[1], want, rtol=1e-4, atol=1e-6)
    with pytest.raises(ValueError, match="needs action"):
        system.learn(params, {"y": Y1}, b1, prev_beliefs=b0)


def test_a_shared_prior_takes_the_taken_modes_scatter():
    system, params, _, Fs, locs = _controlled()
    shared = _iw(scale=0.4)
    params = _params(transition=dataclasses.replace(params.transition["z"], noise_prior=shared))
    action = {"a_z": jnp.asarray(0)}
    b0, b1 = _two_ticks(system, params, action)
    learned = system.learn(params, {"y": Y1}, b1, prev_beliefs=b0, action=action)
    scatter, count = _increment(shared, learned.transition["z"].noise_prior)
    Q = np.asarray(shared.point()) @ np.asarray(shared.point()).T
    want = _rts(b0["z"].loc, b0["z"].cov, b1["z"].loc, b1["z"].cov, Fs[0], locs[0], Q)
    np.testing.assert_allclose(scatter, want, rtol=1e-4, atol=1e-6)
    assert float(count) == 1.0


def test_a_continuous_inputs_g_u_enters_the_residual():
    system = _system(
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("u",),
        control_names=("u_z",),
    )
    G = np.array([[0.5], [1.0]])
    prior = _iw(scale=0.4)
    transition = LinearGaussianTransition(
        F=_f32(F), innovations=_innovations(), G=_f32(G), noise_prior=prior
    )
    params = _params(transition=transition, control={"u_z": Box(-jnp.ones(1), jnp.ones(1))})
    action = {"u": _f32([0.7])}
    b0, b1 = _two_ticks(system, params, action)
    learned = system.learn(params, {"y": Y1}, b1, prev_beliefs=b0, action=action)
    scatter, _ = _increment(prior, learned.transition["z"].noise_prior)
    Q = np.asarray(prior.point()) @ np.asarray(prior.point()).T
    args = (b0["z"].loc, b0["z"].cov, b1["z"].loc, b1["z"].cov, F)
    want = _rts(*args, B + G @ np.array([0.7]), Q)
    np.testing.assert_allclose(scatter, want, rtol=1e-4, atol=1e-6)
    without = _rts(*args, B, Q)
    assert not np.allclose(scatter, without, atol=1e-3)


# ----------------------------------------------------------------- 4. an affine f


class _Linear(eqx.Module):
    M: jax.Array

    def __call__(self, x, controls):
        return self.M @ x


def test_an_affine_f_learns_as_the_linear_transition_bitwise():
    system = _system()
    prior = _iw(scale=0.4)
    linear = _params(prior)
    nonlinear = _params(
        transition=NonlinearGaussianTransition(
            f=_Linear(_f32(F)), innovations=_innovations(), noise_prior=prior
        )
    )
    b0, b1 = _two_ticks(system, linear)
    a = system.learn(linear, {"y": Y1}, b1, prev_beliefs=b0).transition["z"].noise_prior
    b = system.learn(nonlinear, {"y": Y1}, b1, prev_beliefs=b0).transition["z"].noise_prior
    np.testing.assert_array_equal(np.asarray(a.Psi), np.asarray(b.Psi))
    np.testing.assert_array_equal(np.asarray(a.nu), np.asarray(b.nu))


class _Scaled(eqx.Module):
    M: jax.Array
    s: float = eqx.field(static=True)

    def __call__(self, x, controls):
        return self.M @ x, jnp.full_like(x, self.s)


def test_a_constant_scale_standardises_the_residual():
    system = _system()
    prior = _iw(scale=0.4)
    s = 2.0
    scaled = NonlinearGaussianTransition(
        f=_Scaled(_f32(F), s),
        innovations=_innovations(),
        noise_scale="per_coordinate",
        noise_prior=prior,
    )
    params = _params(transition=scaled)
    b0, b1 = _two_ticks(system, params)
    learned = system.learn(params, {"y": Y1}, b1, prev_beliefs=b0).transition["z"].noise_prior
    scatter, _ = _increment(prior, learned)
    Q = s**2 * np.asarray(prior.point()) @ np.asarray(prior.point()).T
    want = _rts(b0["z"].loc, b0["z"].cov, b1["z"].loc, b1["z"].cov, F, B, Q) / s**2
    np.testing.assert_allclose(scatter, want, rtol=1e-4, atol=1e-6)


# ------------------------------------------------- 5. the customer's shape, by replay


def test_a_replay_on_a_sixteen_dimensional_model_moves_a_scaled_q_toward_the_truth():
    d = 16
    rng = np.random.default_rng(7)
    Fd = 0.8 * np.eye(d) + 0.02 * rng.standard_normal((d, d))
    q_true = 0.05 * np.ones(d)
    system = _system()
    start = InverseGamma(alpha=_f32(np.full(d, 5.0)), beta=_f32(np.full(d, 5.0 * 4.0 * 0.05)))
    params = SystemParams(
        transition={
            "z": LinearGaussianTransition(
                F=_f32(Fd),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, d)), chol=jnp.eye(d)[None], parent_ndims=1
                ),
                noise_prior=start,
            )
        },
        initial_state_prior={"z": Gaussian(loc=jnp.zeros(d), chol=0.3 * jnp.eye(d))},
        likelihood={
            "y": LinearGaussianObservation(
                H=jnp.eye(d), c=jnp.zeros(d), noise=_f32(np.sqrt(0.02) * np.eye(d))
            )
        },
    )
    ys = _trace(300, seed=1, F=Fd, b=np.zeros(d), Q=np.diag(q_true), d=d, r=0.02)
    out = replay(
        system,
        params,
        {"y": ys},
        {},
        learn=True,
        lr=0.0,
        lr_by_slot={"transition:z:noise": 1.0},
        metrics=False,
    )
    post = out.params.transition["z"].noise_prior
    before = np.asarray(start.beta / start.alpha)
    after = np.asarray(post.beta / post.alpha)  # the expected-precision point the filter reads
    assert np.linalg.norm(after - q_true) < 0.25 * np.linalg.norm(before - q_true)
    np.testing.assert_allclose(np.asarray(post.alpha), 5.0 + 0.5 * 299, rtol=1e-6)


# ------------------------------------------- 6. description, length and evidence


def test_the_description_writes_the_prior_and_round_trips():
    system = _system()
    params = _params(_iw(scale=0.4))
    doc, arrays = describe(system, params)
    chol = doc["factors"]["z"]["transition"]["parameters"]["innovations"]["chol"]
    assert doc["factors"][chol]["prior"]["family"] == "InverseWishart"
    assert doc["variables"][chol]["shape"] == [2, 2] and chol not in doc["clamps"]
    import jsonschema

    jsonschema.Draft202012Validator(schema()).validate(doc)
    rebuilt_system, rebuilt = build(doc, arrays=arrays)
    decl = rebuilt.transition["z"]
    assert isinstance(decl.noise_prior, InverseWishart)
    np.testing.assert_allclose(np.asarray(decl.noise_prior.Psi), np.asarray(_iw(0.4).Psi))
    np.testing.assert_allclose(  # the rebuilt innovations hold the prior's point
        np.asarray(decl.innovations.chol), np.asarray(_iw(0.4).point()), rtol=1e-6
    )
    again, _ = describe(rebuilt_system, rebuilt)
    assert json.dumps(again, sort_keys=True) == json.dumps(doc, sort_keys=True)


def test_a_nonlinear_transitions_prior_round_trips():
    from hybrid_aif.description import register_transition

    register_transition("tests.process_noise_linear", _Linear)
    system = _system()
    params = _params(
        transition=NonlinearGaussianTransition(
            f=_Linear(_f32(F)), innovations=_innovations(), noise_prior=_iw(scale=0.4)
        )
    )
    doc, arrays = describe(system, params)
    _, rebuilt = build(doc, arrays=arrays)
    assert isinstance(rebuilt.transition["z"].noise_prior, InverseWishart)


def test_the_noise_prior_is_priced_as_the_emissions():
    system = _system()
    prior = _iw(scale=0.4)
    n = 50.0
    learnable = description_length(system, n, params=_params(prior))
    point = transition_point(_transition(prior))
    point = description_length(system, n, params=_params(transition=point))
    observed = dataclasses.replace(_params(), likelihood={"y": _observation(prior)})
    as_noise = description_length(system, n, params=observed)
    link = learnable.per_address["transition:z"]
    assert link.prior == pytest.approx(as_noise.per_address["likelihood:y"].prior)
    assert link.prior is not None and point.per_address["transition:z"].prior is None
    assert link.family > point.per_address["transition:z"].family  # the room of the prior


def test_the_evidence_is_the_noise_priors_closed_form():
    system = _system()
    prior = _params(_iw(scale=0.4))
    b0, b1 = _two_ticks(system, prior)
    posterior = system.learn(prior, {"y": Y1}, b1, prev_beliefs=b0)
    got = system.log_evidence(prior, posterior)["transition:z"]
    want = log_evidence(prior.transition["z"].noise_prior, posterior.transition["z"].noise_prior)
    assert got.exact and float(got.count) == pytest.approx(1.0)
    assert float(got.nats) == pytest.approx(float(jnp.sum(want)), rel=1e-6)


# --------------------------------------------------------------------- 7. refusals


def test_a_mixture_carried_factor_is_refused():
    system = _system(mixture_k=2, mixture_schedule="mode_aligned")
    params = dataclasses.replace(
        _params(_iw()),
        initial_state_prior={
            "z": MixtureGaussian(
                log_weights=jnp.log(_f32([0.5, 0.5])),
                means=_f32([[0.0, 0.0], [1.0, 1.0]]),
                chols=_f32(np.broadcast_to(0.5 * np.eye(2), (2, 2, 2))),
            )
        },
    )
    b0, b1 = _two_ticks(system, params)
    with pytest.raises(NotImplementedError, match="mixture-carried"):
        system.learn(params, {"y": Y1}, b1, prev_beliefs=b0)
    frozen = system.learn(
        params, {"y": Y1}, b1, prev_beliefs=b0, lr_by_slot={"transition:z:noise": 0.0}
    )
    assert isinstance(frozen.transition["z"].noise_prior, InverseWishart)


def test_the_unscented_rule_is_refused():
    system = _system()
    params = _params(
        transition=NonlinearGaussianTransition(
            f=_Linear(_f32(F)),
            innovations=_innovations(),
            propagation="unscented",
            noise_prior=_iw(),
        )
    )
    b0, b1 = _two_ticks(system, params)
    with pytest.raises(NotImplementedError, match="unscented"):
        system.learn(params, {"y": Y1}, b1, prev_beliefs=b0)


@pytest.mark.parametrize("epistemic", [("transition:z:noise",), ("transition:z",), True])
def test_novelty_on_the_process_noise_is_refused(epistemic):
    params = _params(_iw())
    beliefs, _ = _system().infer_states(params, {"y": Y0})
    with pytest.raises(NotImplementedError, match="transition-side twin"):
        _system(epistemic=epistemic).infer_policies(params, beliefs)


def test_plan_learning_refuses_a_noise_prior_as_no_table():
    params = _params(_iw())
    beliefs, _ = _system().infer_states(params, {"y": Y0})
    with pytest.raises(ValueError, match="not a learnable table"):
        _system(plan_learning=("transition:z",)).infer_policies(params, beliefs)
    homed = dataclasses.replace(_params(Parameter("Q")), parameters={"Q": _iw()})
    with pytest.raises(ValueError, match="not a learnable table"):
        _system(plan_learning=("parameter:Q",)).infer_policies(homed, beliefs)


def test_composing_a_learned_noise_is_refused():
    from hybrid_aif.durations import compose

    with pytest.raises(ValueError, match="noise_prior"):
        compose(_transition(_iw()), 3)


# -------------------------------------------------------------------- 9. the noise home


def _two_factors(readers=("a", "b")):
    """Two independent copies of the model, factors ``a`` and ``b``, each observed by its own
    modality; ``readers`` read the noise variable ``Q``, the others hold their own prior."""
    system = System(
        A_deps=((0,), (1,)),
        B_deps=((), ()),
        state_names=("a", "b"),
        modality_names=("ya", "yb"),
        policy_len=1,
        epistemic=False,
    )
    params = SystemParams(
        transition={f: _transition(Parameter("Q") if f in readers else _iw(0.4)) for f in "ab"},
        initial_state_prior={
            f: Gaussian(loc=_f32([0.2, -0.1]), chol=_f32(0.5 * np.eye(2))) for f in ("a", "b")
        },
        likelihood={"ya": _observation(), "yb": _observation()},
        parameters={"Q": _iw(scale=0.4)} if readers else {},
    )
    return system, params


OBS_AB = ({"ya": Y0, "yb": _f32([-0.2, 0.3])}, {"ya": Y1, "yb": _f32([0.0, 0.5])})


def _ab_ticks(system, params):
    b0, _ = system.infer_states(params, OBS_AB[0])
    pred = system.update_empirical_prior(params, b0, {})
    b1, _ = system.infer_states(params, OBS_AB[1], priors=pred)
    return b0, b1


def test_two_transitions_reading_one_noise_learn_one_posterior_from_the_sum():
    prior = _iw(scale=0.4)
    shared_system, shared = _two_factors()
    own_system, own = _two_factors(readers=())
    b0, b1 = _ab_ticks(shared_system, shared)
    learned = shared_system.learn(shared, OBS_AB[1], b1, prev_beliefs=b0)
    separate = own_system.learn(own, OBS_AB[1], b1, prev_beliefs=b0)
    total, count = _increment(prior, learned.parameters["Q"])
    sa, _ = _increment(prior, separate.transition["a"].noise_prior)
    sb, _ = _increment(prior, separate.transition["b"].noise_prior)
    np.testing.assert_allclose(total, sa + sb, rtol=1e-5, atol=1e-7)
    assert float(count) == 2.0
    for f in ("a", "b"):  # the faces return the references, never copies
        assert learned.transition[f].noise_prior == Parameter("Q")


def test_the_homes_rate_key_is_the_variables_and_a_reader_key_only_excludes():
    system, params = _two_factors()
    b0, b1 = _ab_ticks(system, params)
    frozen = system.learn(params, OBS_AB[1], b1, prev_beliefs=b0, lr_by_slot={"parameter:Q": 0.0})
    np.testing.assert_array_equal(np.asarray(frozen.parameters["Q"].Psi), np.asarray(_iw(0.4).Psi))
    with pytest.raises(ValueError, match="parameter:Q"):
        system.learn(params, OBS_AB[1], b1, prev_beliefs=b0, lr_by_slot={"transition:a:noise": 1.0})
    only_b = system.learn(
        params, OBS_AB[1], b1, prev_beliefs=b0, lr_by_slot={"transition:a:noise": 0.0}
    )
    _, count = _increment(_iw(0.4), only_b.parameters["Q"])
    assert float(count) == 1.0


def test_a_likelihood_and_a_transition_share_one_noise_variable():
    system = _system()
    prior = _iw(scale=0.4)
    shared = SystemParams(
        transition={"z": _transition(Parameter("N"))},
        initial_state_prior={"z": Gaussian(loc=_f32([0.2, -0.1]), chol=_f32(0.5 * np.eye(2)))},
        likelihood={"y": _observation(Parameter("N"))},
        parameters={"N": prior},
    )
    b0, b1 = _two_ticks(system, shared)
    # the filter reads the one point for both noises
    positional = system.to_positional("params", shared)
    np.testing.assert_allclose(
        np.asarray(positional.likelihood[0].R_chol), np.asarray(prior.point())
    )
    learned = system.learn(shared, {"y": Y1}, b1, prev_beliefs=b0)
    total, count = _increment(prior, learned.parameters["N"])
    inline = SystemParams(
        transition={"z": _transition(prior)},
        initial_state_prior=shared.initial_state_prior,
        likelihood={"y": _observation(_iw(scale=0.4))},
    )
    each = system.learn(inline, {"y": Y1}, b1, prev_beliefs=b0)
    s_q, _ = _increment(prior, each.transition["z"].noise_prior)
    s_r, _ = _increment(prior, each.likelihood["y"].noise)
    np.testing.assert_allclose(total, s_q + s_r, rtol=1e-5, atol=1e-7)
    assert float(count) == 2.0
    assert learned.likelihood["y"].noise == Parameter("N")


def test_the_description_records_the_variable_once_with_its_readers():
    system = _system()
    params = SystemParams(
        transition={"z": _transition(Parameter("N"))},
        initial_state_prior={"z": Gaussian(loc=_f32([0.2, -0.1]), chol=_f32(0.5 * np.eye(2)))},
        likelihood={"y": _observation(Parameter("N"))},
        parameters={"N": _iw(scale=0.4)},
    )
    doc, arrays = describe(system, params)
    assert doc["parameters"]["N"]["family"] == "InverseWishart"
    record = doc["factors"]["z"]["transition"]["parameters"]
    assert record["innovations"]["chol"] == {"parameter": "N"}
    assert doc["factors"]["y"]["likelihood"]["parameters"]["noise"] == {"parameter": "N"}
    assert sum(1 for v in doc["variables"] if v.endswith("_Psi")) == 1
    import jsonschema

    jsonschema.Draft202012Validator(schema()).validate(doc)
    rebuilt_system, rebuilt = build(doc, arrays=arrays)
    assert rebuilt.transition["z"].noise_prior == Parameter("N")
    assert rebuilt.likelihood["y"].noise == Parameter("N")
    np.testing.assert_allclose(np.asarray(rebuilt.parameters["N"].Psi), np.asarray(_iw(0.4).Psi))
    again, _ = describe(rebuilt_system, rebuilt)
    assert json.dumps(again, sort_keys=True) == json.dumps(doc, sort_keys=True)
    # priced once, at its readers' summed resolution
    length = description_length(system, 10.0, params=params)
    assert length.per_address["parameter:N"].prior is not None
    assert length.per_address["transition:z"].prior is None
    assert length.per_address["parameter:N"].n == pytest.approx(20.0)


def test_one_prior_object_under_two_readers_is_refused():
    prior = _iw()
    with pytest.raises(ValueError, match="one noise prior object"):
        SystemParams(
            transition={"a": _transition(prior), "b": _transition(prior)},
            initial_state_prior={},
            likelihood={},
        )
    with pytest.raises(ValueError, match="one noise prior object"):
        SystemParams(
            transition={"z": _transition(prior)},
            initial_state_prior={},
            likelihood={"y": _observation(prior)},
        )
    with pytest.raises(ValueError, match="one"):
        SystemParams(
            transition={"z": _transition(prior)},
            initial_state_prior={},
            likelihood={},
            parameters={"Q": prior},
        )


def test_a_reference_must_name_a_noise_variable_of_the_readers_size():
    system = _system()
    missing = _params(Parameter("Q"))
    with pytest.raises(ValueError, match="declare the variable there"):
        system.infer_states(missing, {"y": Y0})
    table = jnp.log(jnp.full((2, 2), 0.5))
    from hybrid_aif.distributions import ConditionalCategorical

    wrong = dataclasses.replace(
        missing, parameters={"Q": ConditionalCategorical(log_table=table, parent_ndims=1)}
    )
    with pytest.raises(ValueError, match="InverseGamma or an InverseWishart"):
        system.infer_states(wrong, {"y": Y0})
    small = dataclasses.replace(
        missing, parameters={"Q": InverseWishart(Psi=_f32(np.eye(3)), nu=_f32(5.0))}
    )
    with pytest.raises(ValueError, match="dimensional"):
        system.infer_states(small, {"y": Y0})
    whole = dataclasses.replace(
        _params(), transition={"z": Parameter("Q")}, parameters={"Q": _iw()}
    )
    with pytest.raises(ValueError, match="one field"):
        system.infer_states(whole, {"y": Y0})


def test_the_carry_takes_a_learned_noise_across_a_rebuild():
    system = _system()
    params = _params(_iw(scale=0.4))
    b0, b1 = _two_ticks(system, params)
    learned = system.learn(params, {"y": Y1}, b1, prev_beliefs=b0)
    carried = system.carry(_params(_iw(scale=0.4)), learned)
    np.testing.assert_array_equal(
        np.asarray(carried.transition["z"].noise_prior.Psi),
        np.asarray(learned.transition["z"].noise_prior.Psi),
    )


# ------------------------------------------------------------------- composition


def _component():
    params = SystemParams(
        transition={"z": _transition(Parameter("Q"))},
        initial_state_prior={"z": Gaussian(loc=_f32([0.2, -0.1]), chol=_f32(0.5 * np.eye(2)))},
        likelihood={"y": _observation()},
        parameters={"Q": _iw(scale=0.4)},
    )
    return _system(), params


def test_a_merge_identifies_two_components_noise_variables_as_one():
    from hybrid_aif.composition import Identify, merge, subsystem, write_back

    (a, pa), (b, pb) = _component(), _component()
    composite, params = merge({"a": a, "b": b}, {"a": pa, "b": pb}, [Identify("a.Q", "b.Q")])
    assert list(params.parameters) == ["a.Q"]
    assert params.transition["a.z"].noise_prior == Parameter("a.Q")
    assert params.transition["b.z"].noise_prior == Parameter("a.Q")
    obs0 = {"a.y": OBS_AB[0]["ya"], "b.y": OBS_AB[0]["yb"]}
    obs1 = {"a.y": OBS_AB[1]["ya"], "b.y": OBS_AB[1]["yb"]}
    b0, _ = composite.infer_states(params, obs0)
    b1, _ = composite.infer_states(
        params, obs1, priors=composite.update_empirical_prior(params, b0, {})
    )
    learned = composite.learn(params, obs1, b1, prev_beliefs=b0)
    _, count = _increment(_iw(0.4), learned.parameters["a.Q"])
    assert float(count) == 2.0
    view, view_params = subsystem(composite, params, "b")
    assert list(view_params.parameters) == ["a.Q"]
    assert view_params.transition["b.z"].noise_prior == Parameter("a.Q")
    vb0, _ = view.infer_states(view_params, {"b.y": obs0["b.y"]})
    vb1, _ = view.infer_states(
        view_params, {"b.y": obs1["b.y"]}, priors=view.update_empirical_prior(view_params, vb0, {})
    )
    after = view.learn(view_params, {"b.y": obs1["b.y"]}, vb1, prev_beliefs=vb0)
    written = write_back(composite, params, "b", view_params, after)
    np.testing.assert_array_equal(
        np.asarray(written.parameters["a.Q"].Psi), np.asarray(after.parameters["a.Q"].Psi)
    )


def test_a_composites_noise_variable_describes_and_rebuilds():
    from hybrid_aif.composition import Identify, merge

    (a, pa), (b, pb) = _component(), _component()
    composite, params = merge({"a": a, "b": b}, {"a": pa, "b": pb}, [Identify("a.Q", "b.Q")])
    doc, arrays = describe(composite, params)
    assert list(doc["parameters"]) == ["a.Q"]
    rebuilt_system, rebuilt = build(doc, arrays=arrays)
    again, _ = describe(rebuilt_system, rebuilt)
    assert json.dumps(again, sort_keys=True) == json.dumps(doc, sort_keys=True)


def test_the_noise_learns_under_jit_with_a_traced_rate():
    system, params = _two_factors()
    b0, b1 = _ab_ticks(system, params)

    @eqx.filter_jit
    def step(p, r):
        return system.learn(p, OBS_AB[1], b1, prev_beliefs=b0, lr_by_slot={"parameter:Q": r})

    _, count = _increment(_iw(0.4), step(params, jnp.asarray(0.5)).parameters["Q"])
    assert float(count) == pytest.approx(1.0)
    one = _system()
    p = _params(_iw())
    c0, c1 = _two_ticks(one, p)
    learned = eqx.filter_jit(lambda p, r: one.learn(p, {"y": Y1}, c1, prev_beliefs=c0, lr=r))(
        p, jnp.asarray(0.5)
    )
    assert float(learned.transition["z"].noise_prior.nu) == pytest.approx(6.5)
