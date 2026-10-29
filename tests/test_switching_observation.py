"""Workstream C phase 1.5: switching observation models (E1-E4, in-repo half).

The symmetric completion of phase 1: the mode conditions the *observation*
(``y = H_j x + c_j + eps_j``) rather than the transition. Covered here:

- exactness of the measurement-grow conditioning kernel
  (:func:`condition_mixture_switching`) vs a brute-force joint enumeration over
  ``(component, mode)`` pairs, at tight tolerance;
- the padding-invariance battery extended to the new kernel (dead components
  change nothing, incl. under grad);
- per-mode broadcast: ``R_j`` only (shared ``H``/``c``) and full
  ``H_j``/``c_j``/``R_j``;
- mode-posterior parity vs an independent numpy reference on a hand-computable
  case, through the ``infer_states`` face;
- filtering fidelity at a modest budget vs the no-reduction exact fallback (the
  mode posterior and x-mean are exact at any k — the message is read pre-reduce
  and Runnalls preserves the mixture mean/cov);
- error paths for every new validation;
- the concentrated-mode planning collapse: as ``q(mode)`` concentrates the
  hybrid channel ``infogain`` leaf collapses to the closed form with the surviving
  mode's ``R_j`` — verified directly and through ``score_policies``.

The world: a hidden categorical mode gates a noisy 1-D position sensor; mode 0
is a sharp fix (small R), mode 1 a loose fix (large R). Coverage's "cell-
dependent localizability" in miniature.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalGaussian,
    Gaussian,
    MixtureGaussian,
)
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianTransition
from hybrid_aif.inference.mixture_filter import (
    condition_mixture_switching,
    switching_mode_message,
)
from hybrid_aif.objectives import total
from hybrid_aif.objectives.continuous import state_info_gain
from hybrid_aif.system import System, SystemParams

# --------------------------------------------------------------------------- fixtures

H_SHARED = jnp.asarray([[1.0]])
C_SHARED = jnp.zeros(1)
# per-mode fix noise: mode 0 sharp, mode 1 loose (the localizability contrast).
R_PER_MODE = jnp.asarray([[[0.3]], [[0.9]]])  # (M=2, d_obs=1, d_obs=1)
# per-mode sensor geometry (for the full H_j/c_j/R_j coverage): mode-dependent gain+bias.
H_PER_MODE = jnp.asarray([[[1.0]], [[0.7]]])  # (M, d_obs, d)
C_PER_MODE = jnp.asarray([[0.0], [0.4]])  # (M, d_obs)

X_PRIOR = MixtureGaussian(
    log_weights=jnp.log(jnp.asarray([0.5, 0.3, 0.2])),
    means=jnp.asarray([[-1.0], [0.5], [1.5]]),
    chols=jnp.asarray([[[0.4]], [[0.3]], [[0.5]]]),
)
MODE_PRIOR = Categorical.from_probs(jnp.asarray([0.6, 0.4]))


def _normal_pdf(y, mean, var):
    return np.exp(-0.5 * (y - mean) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def brute_force(x_prior, mode_probs, Hs, cs, Rs, y_val):
    """Textbook joint (component, mode) enumeration — no library kernels.

    ``Hs``/``cs``/``Rs`` are per-mode arrays ``(M, ...)``. Returns the flat
    ``K*M`` posterior (row-major slot ``k*M + j``) and per-mode evidence
    ``p(y|mode=j) = Σ_k w_k N(y; H_j mu_k + c_j, H_j^2 var_k + R_j)``.
    """
    w = np.asarray(jax.nn.softmax(x_prior.log_weights))
    mu = np.asarray(x_prior.means)[:, 0]
    var = np.asarray(x_prior.chols)[:, 0, 0] ** 2
    q = np.asarray(mode_probs)
    h = np.asarray(Hs)[:, 0, 0]
    cc = np.asarray(cs)[:, 0]
    r = np.asarray(Rs)[:, 0, 0] ** 2
    K, M = len(w), len(q)
    pw = np.zeros((K, M))
    pmu = np.zeros((K, M))
    pvar = np.zeros((K, M))
    p_y_j = np.zeros(M)
    for k in range(K):
        for j in range(M):
            S = h[j] * var[k] * h[j] + r[j]
            n = _normal_pdf(y_val, h[j] * mu[k] + cc[j], S)
            pw[k, j] = w[k] * q[j] * n
            gain = var[k] * h[j] / S
            pmu[k, j] = mu[k] + gain * (y_val - (h[j] * mu[k] + cc[j]))
            pvar[k, j] = (1.0 - gain * h[j]) * var[k]
            p_y_j[j] += w[k] * n
    flat_w = pw.reshape(-1)
    return flat_w / flat_w.sum(), pmu.reshape(-1), pvar.reshape(-1), p_y_j


def _per_mode(arr, m, event_ndims):
    event = arr.shape[arr.ndim - event_ndims :]
    return jnp.broadcast_to(arr, (m,) + event) if arr.ndim == event_ndims else arr


# ------------------------------------------------------------- E2: exactness vs brute force


@pytest.mark.parametrize(
    "H, c",
    [(H_SHARED, C_SHARED), (H_PER_MODE, C_PER_MODE)],
    ids=["R_j-only", "full-H_j-c_j-R_j"],
)
def test_grow_conditioning_exact_vs_brute_force(H, c):
    y = jnp.asarray([0.7])
    grown, mode_log_ev = condition_mixture_switching(
        X_PRIOR, H, c, R_PER_MODE, y, MODE_PRIOR.log_probs
    )
    Hs, cs, Rs = _per_mode(H, 2, 2), _per_mode(c, 2, 1), _per_mode(R_PER_MODE, 2, 2)
    flat_w, pmu, pvar, p_y_j = brute_force(X_PRIOR, MODE_PRIOR.probs, Hs, cs, Rs, 0.7)
    assert grown.num_components == X_PRIOR.num_components * MODE_PRIOR.num_levels
    np.testing.assert_allclose(np.asarray(jax.nn.softmax(grown.log_weights)), flat_w, atol=1e-6)
    np.testing.assert_allclose(np.asarray(grown.means)[:, 0], pmu, atol=1e-6)
    np.testing.assert_allclose(np.asarray(grown.chols)[:, 0, 0] ** 2, pvar, atol=1e-6)
    # per-mode evidence: exp(mode_log_evidence) == p(y|mode=j) (within-hypothesis normalized)
    np.testing.assert_allclose(np.asarray(jnp.exp(mode_log_ev)), p_y_j, atol=1e-6)


def test_grow_conditioning_jits():
    y = jnp.asarray([0.7])
    f = jax.jit(condition_mixture_switching)
    grown, mode_log_ev = f(X_PRIOR, H_SHARED, C_SHARED, R_PER_MODE, y, MODE_PRIOR.log_probs)
    g2, ev2 = condition_mixture_switching(
        X_PRIOR, H_SHARED, C_SHARED, R_PER_MODE, y, MODE_PRIOR.log_probs
    )
    np.testing.assert_allclose(np.asarray(grown.means), np.asarray(g2.means), atol=1e-6)
    np.testing.assert_allclose(np.asarray(mode_log_ev), np.asarray(ev2), atol=1e-6)


# ------------------------------------------------------------------- mode-posterior parity


def test_switching_mode_message_parity():
    y = jnp.asarray([1.2])
    _, mode_log_ev = condition_mixture_switching(
        X_PRIOR, H_SHARED, C_SHARED, R_PER_MODE, y, MODE_PRIOR.log_probs
    )
    (mode_post,) = switching_mode_message((MODE_PRIOR,), mode_log_ev)
    _, _, _, p_y_j = brute_force(
        X_PRIOR,
        MODE_PRIOR.probs,
        _per_mode(H_SHARED, 2, 2),
        _per_mode(C_SHARED, 2, 1),
        R_PER_MODE,
        1.2,
    )
    ref = np.asarray(MODE_PRIOR.probs) * p_y_j
    ref = ref / ref.sum()
    np.testing.assert_allclose(np.asarray(mode_post.probs), ref, atol=1e-6)


# --------------------------------------------------------------------- padding invariance


def _padded(m, pad):
    """Append ``pad`` dead components (−inf weight, component-0 moments)."""
    neg = jnp.full((pad,), -jnp.inf)
    return MixtureGaussian(
        log_weights=jnp.concatenate([m.log_weights, neg]),
        means=jnp.concatenate([m.means, jnp.repeat(m.means[:1], pad, axis=0)]),
        chols=jnp.concatenate([m.chols, jnp.repeat(m.chols[:1], pad, axis=0)]),
    )


def test_grow_conditioning_padding_invariant():
    y = jnp.asarray([0.7])
    grown, ev = condition_mixture_switching(
        X_PRIOR, H_SHARED, C_SHARED, R_PER_MODE, y, MODE_PRIOR.log_probs
    )
    grown_p, ev_p = condition_mixture_switching(
        _padded(X_PRIOR, 2), H_SHARED, C_SHARED, R_PER_MODE, y, MODE_PRIOR.log_probs
    )
    # mode evidence is invariant to padding
    np.testing.assert_allclose(np.asarray(ev_p), np.asarray(ev), atol=1e-6)
    # the live (finite-weight) grown components carry the same posterior mass
    live = np.isfinite(np.asarray(grown_p.log_weights))
    w_ref = np.sort(np.asarray(jax.nn.softmax(grown.log_weights)))
    w_pad = np.sort(np.asarray(jax.nn.softmax(grown_p.log_weights))[live])
    np.testing.assert_allclose(w_pad, w_ref, atol=1e-6)


@pytest.mark.parametrize("field", ["log_weights", "means", "chols"])
def test_gradients_through_switching_kernel_finite(field):
    padded = _padded(X_PRIOR, 2)
    y = jnp.asarray([0.7])

    def evidence(x):
        m = dataclasses.replace(padded, **{field: x})
        _, ev = condition_mixture_switching(
            m, H_SHARED, C_SHARED, R_PER_MODE, y, MODE_PRIOR.log_probs
        )
        (q,) = switching_mode_message((MODE_PRIOR,), ev)
        return q.probs[1]

    g = jax.grad(evidence)(getattr(padded, field))
    assert bool(jnp.all(jnp.isfinite(g))), f"non-finite grad d/d{field}"


# ----------------------------------------------------------- infer_states + fidelity vs exact


def make_switch_system(**overrides):
    """Observation-side switching: y observes x(mixture) gated by mode(categorical)."""
    defaults = dict(
        A_deps=((1, 0),),  # y observes x (factor 1), switching parent mode (factor 0)
        B_deps=((), ()),  # mode uncontrolled+sticky (own trivial control); x parentless
        controlled=(False, False),
        policy_len=1,
        mixture_k=6,
        mixture_schedule="runnalls",  # no mode-owned slots needed (Guidance)
        modality_names=("y",),
        factor_names=("mode", "x"),
    )
    return System(**{**defaults, **overrides})


# uncontrolled parentless x: trivial (1,) innovation parent axis, near-static dynamics.
_X_INNOV = ConditionalGaussian(loc=jnp.zeros((1, 1)), chol=jnp.asarray([[[0.05]]]), parent_ndims=1)
# sticky uncontrolled mode: single trivial control column.
_MODE_STICK = ConditionalCategorical(log_table=jnp.log(jnp.asarray([[0.5], [0.5]])), parent_ndims=1)


def make_switch_params(H=H_SHARED, c=C_SHARED, R=R_PER_MODE, **overrides):
    defaults = dict(
        likelihood={"y": LinearGaussianObservation(H=H, c=c, noise=R)},
        transition={
            "mode": _MODE_STICK,
            "x": LinearGaussianTransition(F=jnp.asarray([[1.0]]), innovations=_X_INNOV),
        },
        initial_state_prior={"mode": MODE_PRIOR, "x": X_PRIOR},
        obs_targets={},
        state_targets={},
        terminal_targets={},
    )
    return SystemParams(**{**defaults, **overrides})


def test_as_env_observe_selects_per_mode_noise():
    """as_env samples the sensor with the sampled mode selecting R_j (E1 env side)."""
    system, params = make_switch_system(), make_switch_params()
    env, env_params = system.as_env(params)
    process_params = system.to_positional("params", env_params)  # the process reads tuples
    x0 = jnp.asarray([0.5])
    ys = {}
    for mode in (0, 1):
        obs, _ = env.reset(
            jax.random.PRNGKey(1), env_params, state={"mode": jnp.asarray(mode, jnp.int32), "x": x0}
        )
        keys = jax.random.split(jax.random.PRNGKey(mode), 4000)
        samples = jax.vmap(
            lambda k, mo=mode: env.inner._observe(k, (jnp.asarray(mo), x0), process_params)[0]
        )(keys)
        ys[mode] = np.asarray(samples)[:, 0]
    # sharp mode 0 (R chol 0.3) vs loose mode 1 (R chol 0.9); shared H=1, c=0 => mean ~ x0.
    np.testing.assert_allclose(ys[0].std(), 0.3, atol=0.03)
    np.testing.assert_allclose(ys[1].std(), 0.9, atol=0.06)
    np.testing.assert_allclose(ys[0].mean(), 0.5, atol=0.03)


def test_as_env_observe_selects_per_mode_geometry():
    """Full per-mode H_j/c_j: mode 1 uses gain 0.7, bias 0.4 => mean 0.7*x + 0.4."""
    system = make_switch_system()
    params = make_switch_params(H=H_PER_MODE, c=C_PER_MODE, R=R_PER_MODE)
    env, env_params = system.as_env(params)
    x0 = jnp.asarray([1.0])
    keys = jax.random.split(jax.random.PRNGKey(3), 4000)
    process_params = system.to_positional("params", env_params)  # the process reads tuples
    samples = jax.vmap(lambda k: env.inner._observe(k, (jnp.asarray(1), x0), process_params)[0])(
        keys
    )
    ys = np.asarray(samples)[:, 0]
    np.testing.assert_allclose(ys.mean(), 0.7 * 1.0 + 0.4, atol=0.05)
    np.testing.assert_allclose(ys.std(), 0.9, atol=0.06)


def test_infer_states_switching_mode_posterior_matches_reference():
    system, params = make_switch_system(), make_switch_params()
    y = jnp.asarray([1.2])
    post, _ = system.infer_states(params, {"y": y}, priors={"mode": MODE_PRIOR, "x": X_PRIOR})
    mode_post = post["mode"]
    _, _, _, p_y_j = brute_force(
        X_PRIOR,
        MODE_PRIOR.probs,
        _per_mode(H_SHARED, 2, 2),
        _per_mode(C_SHARED, 2, 1),
        R_PER_MODE,
        1.2,
    )
    ref = np.asarray(MODE_PRIOR.probs) * p_y_j
    ref = ref / ref.sum()
    assert isinstance(mode_post, Categorical)
    np.testing.assert_allclose(np.asarray(mode_post.probs), ref, atol=1e-5)
    # the loose-fix reading (y=1.2, right basin) shifts mass toward mode 1's slots overall,
    # but the sharp mode 0 fits its right-most component too; assert the posterior is proper.
    np.testing.assert_allclose(float(mode_post.probs.sum()), 1.0, atol=1e-6)


def test_infer_states_switching_default_schedule_ok():
    """Observation-side switching works under the default 'runnalls' schedule (no
    mode-owned slots needed), unlike the transition-side phase-1 path."""
    system = make_switch_system()
    assert system.mixture_schedule == "runnalls"
    post, _ = system.infer_states(make_switch_params(), {"y": jnp.asarray([1.2])})
    mode_post = post["mode"]
    assert isinstance(mode_post, Categorical)


def test_filtering_fidelity_mode_and_mean_exact_at_reduced_k():
    """The mode posterior and x-mean are exact at any budget (message read pre-reduce;
    Runnalls preserves the mixture mean), so a reduced k matches the no-reduction fallback."""
    y = jnp.asarray([1.2])
    params = make_switch_params()
    exact = make_switch_system(mixture_k=6)  # K*M = 3*2, pad_mixture keeps all -> no reduction
    reduced = make_switch_system(mixture_k=3)  # genuine 6 -> 3 reduction
    priors = {"mode": MODE_PRIOR, "x": X_PRIOR}
    post_e, _ = exact.infer_states(params, {"y": y}, priors=priors)
    post_r, _ = reduced.infer_states(params, {"y": y}, priors=priors)
    m_e, x_e = post_e["mode"], post_e["x"]
    m_r, x_r = post_r["mode"], post_r["x"]
    np.testing.assert_allclose(np.asarray(m_r.probs), np.asarray(m_e.probs), atol=1e-6)
    np.testing.assert_allclose(float(x_r.mean()[0]), float(x_e.mean()[0]), atol=1e-5)
    # x-mean at the exact budget matches the brute-force joint posterior mean
    flat_w, pmu, _, _ = brute_force(
        X_PRIOR,
        MODE_PRIOR.probs,
        _per_mode(H_SHARED, 2, 2),
        _per_mode(C_SHARED, 2, 1),
        R_PER_MODE,
        1.2,
    )
    np.testing.assert_allclose(float(x_e.mean()[0]), float((flat_w * pmu).sum()), atol=1e-5)


def test_infer_states_switching_full_per_mode_H_c():
    """Full per-mode H_j/c_j/R_j threads through infer_states (mode-dependent geometry)."""
    system = make_switch_system(mixture_k=6)
    params = make_switch_params(H=H_PER_MODE, c=C_PER_MODE, R=R_PER_MODE)
    post, _ = system.infer_states(
        params, {"y": jnp.asarray([0.7])}, priors={"mode": MODE_PRIOR, "x": X_PRIOR}
    )
    mode_post = post["mode"]
    _, _, _, p_y_j = brute_force(X_PRIOR, MODE_PRIOR.probs, H_PER_MODE, C_PER_MODE, R_PER_MODE, 0.7)
    ref = np.asarray(MODE_PRIOR.probs) * p_y_j
    ref = ref / ref.sum()
    np.testing.assert_allclose(np.asarray(mode_post.probs), ref, atol=1e-5)


# --------------------------------------------------------------------------- error paths


def test_error_non_categorical_switching_parent():
    """A switching parent that is a continuous factor is rejected."""
    system = System(
        A_deps=((0, 1),),  # observe x0, name continuous x1 as a 'parent'
        B_deps=((), ()),
        controlled=(False, False),
        policy_len=1,
        mixture_k=4,
        modality_names=("y",),
        factor_names=("x0", "x1"),
    )
    innov = ConditionalGaussian(loc=jnp.zeros((1, 1)), chol=jnp.asarray([[[0.3]]]), parent_ndims=1)
    params = SystemParams(
        likelihood={"y": LinearGaussianObservation(H=H_SHARED, c=C_SHARED, noise=R_PER_MODE)},
        transition={
            "x0": LinearGaussianTransition(F=jnp.asarray([[1.0]]), innovations=innov),
            "x1": LinearGaussianTransition(F=jnp.asarray([[1.0]]), innovations=innov),
        },
        initial_state_prior={
            "x0": X_PRIOR,
            "x1": Gaussian(loc=jnp.zeros(1), chol=jnp.asarray([[0.5]])),
        },
    )
    with pytest.raises(ValueError, match="must be categorical"):
        system.infer_states(
            params,
            {"y": jnp.asarray([0.7])},
            priors={"x0": X_PRIOR, "x1": params.initial_state_prior["x1"]},
        )


def test_error_switching_observation_on_gaussian_factor():
    """A switching modality on a Gaussian-family factor must be declared mixture-family."""
    system = make_switch_system()
    params = make_switch_params(
        initial_state_prior={
            "mode": MODE_PRIOR,
            "x": Gaussian(loc=jnp.zeros(1), chol=jnp.asarray([[0.5]])),
        }
    )
    with pytest.raises(ValueError, match="carried as a mixture"):
        system.infer_states(params, {"y": jnp.asarray([0.7])})


def test_error_per_mode_leading_shape_mismatch():
    system = make_switch_system()
    bad_R = jnp.broadcast_to(R_PER_MODE[:1], (3, 1, 1))  # 3 modes, but parent has 2 levels
    params = make_switch_params(R=bad_R)
    with pytest.raises(ValueError, match="per-mode.*axes"):
        system.infer_states(params, {"y": jnp.asarray([0.7])})


def test_error_parents_named_but_no_per_mode_array():
    system = make_switch_system()
    params = make_switch_params(R=jnp.asarray([[0.3]]))  # all of H/c/R shared
    with pytest.raises(ValueError, match="does not depend on the mode"):
        system.infer_states(params, {"y": jnp.asarray([0.7])})


def test_error_shared_event_shape_mismatch():
    system = make_switch_system()
    params = make_switch_params(H=jnp.asarray([[1.0, 0.0]]))  # H (1,2), factor dim is 1
    with pytest.raises(ValueError, match="trailing shape"):
        system.infer_states(params, {"y": jnp.asarray([0.7])})


# ----------------------------------------------------- D4 switching info gain + collapse


def test_switching_info_gain_conditional_entropy_is_mode_average():
    """H[y|x] = Σ_j q(j) ½log|2πe R_j|; with a genuine mixture the leaf is a positive
    KT bound pair sandwiching a quadrature H[q(y)]."""
    model = LinearGaussianObservation(H=H_SHARED, c=C_SHARED, noise=R_PER_MODE)
    leaf = state_info_gain((X_PRIOR,), (model,), ("y",), mode_beliefs=((MODE_PRIOR,),))["infogain"][
        "y"
    ]
    # conditional entropy reference (exact)
    q = np.asarray(MODE_PRIOR.probs)
    r = np.asarray(R_PER_MODE)[:, 0, 0]
    cond_h = float((q * (0.5 * (1.0 + np.log(2.0 * np.pi)) + np.log(r))).sum())
    # predictive H[q(y)] by dense quadrature over the (K*M) obs mixture
    w = np.asarray(jax.nn.softmax(X_PRIOR.log_weights))
    mu = np.asarray(X_PRIOR.means)[:, 0]
    var = np.asarray(X_PRIOR.chols)[:, 0, 0] ** 2
    ws, ms, vs = [], [], []
    for k in range(len(w)):
        for j in range(2):
            ws.append(w[k] * q[j])
            ms.append(mu[k])
            vs.append(var[k] + r[j] ** 2)
    ws, ms, vs = np.asarray(ws), np.asarray(ms), np.asarray(vs)
    grid = np.linspace(ms.min() - 8 * np.sqrt(vs.max()), ms.max() + 8 * np.sqrt(vs.max()), 40001)
    dens = np.sum(ws[:, None] * _normal_pdf(grid[None, :], ms[:, None], vs[:, None]), axis=0)
    h_qy = np.trapezoid(np.where(dens > 0, -dens * np.log(dens), 0.0), grid)
    ref = h_qy - cond_h
    assert float(leaf.lower) <= float(leaf.upper)
    assert float(leaf.lower) - 1e-4 <= ref <= float(leaf.upper) + 1e-4
    assert float(leaf.lower) > 0.0


def test_switching_info_gain_collapses_to_closed_form_when_mode_concentrated():
    """Concentrated q(mode) + a single-Gaussian factor: the predictive is (nearly) a
    single Gaussian, so the leaf collapses to ½log((P + R_{j*})/R_{j*}) with the
    surviving mode's noise — the coverage 'refix' closed form, no mixtures needed."""
    g = Gaussian(loc=jnp.asarray([0.3]), chol=jnp.asarray([[0.7]]))
    model = LinearGaussianObservation(H=H_SHARED, c=C_SHARED, noise=R_PER_MODE)
    concentrated = Categorical(logits=jnp.asarray([12.0, 0.0]))  # ~mode 0
    leaf = state_info_gain((g,), (model,), ("y",), mode_beliefs=((concentrated,),))["infogain"]["y"]
    p = 0.7**2
    r0 = float(R_PER_MODE[0, 0, 0]) ** 2  # surviving mode's noise
    closed = 0.5 * (np.log(p + r0) - np.log(r0))
    np.testing.assert_allclose(float(leaf.midpoint), closed, atol=2e-3)
    assert float(leaf.width) < 5e-3  # near-degenerate mixture -> near-zero-width interval


# ------------------------------------------------------- planning: collapse through faces


def make_planning_system(**overrides):
    """A controllable mode gating a switching sensor over a 1-comp mixture x, for planning.

    Action 0 drives the mode near-deterministically to 0, action 1 to 1 — so a
    policy concentrates q(mode) along the rollout and the info-gain term collapses.
    """
    defaults = dict(
        A_deps=((1, 0),),
        B_deps=((), ()),
        controlled=(True, False),
        policy_len=1,
        gamma=8.0,
        mixture_k=2,
        mixture_schedule="runnalls",
        modality_names=("y",),
        factor_names=("mode", "x"),
    )
    return System(**{**defaults, **overrides})


# action->mode table (child-first (modes, actions)): near-deterministic per action.
_MODE_CTRL = ConditionalCategorical(
    log_table=jnp.log(jnp.asarray([[0.9999, 0.0001], [0.0001, 0.9999]])), parent_ndims=1
)
_X1 = MixtureGaussian(
    log_weights=jnp.zeros(1), means=jnp.asarray([[0.3]]), chols=jnp.asarray([[[0.7]]])
)


def make_planning_params():
    return SystemParams(
        likelihood={"y": LinearGaussianObservation(H=H_SHARED, c=C_SHARED, noise=R_PER_MODE)},
        transition={
            "mode": _MODE_CTRL,
            "x": LinearGaussianTransition(F=jnp.asarray([[1.0]]), innovations=_X_INNOV),
        },
        initial_state_prior={"mode": Categorical(logits=jnp.zeros(2)), "x": _X1},
        state_targets={"x": _X1},
    )


def test_planning_collapse_state_info_gain_through_score_policies():
    system, params = make_planning_system(), make_planning_params()
    beliefs = {"mode": Categorical(logits=jnp.zeros(2)), "x": _X1}
    pols = jnp.asarray([[0], [1]], jnp.int32)[..., None]
    tree = system.score_policies(params, beliefs, pols)
    assert set(tree["infogain"]) == {"y"}
    leaf = tree["infogain"]["y"]
    assert leaf.lower.shape == (2,)
    total(tree)  # folds into G without error (registered term)
    # policy driving mode->0 collapses to mode 0's closed form; mode->1 to mode 1's.
    p = 0.7**2 + 0.05**2  # x prior var + one innovation step (F=1)
    for a, mode in ((0, 0), (1, 1)):
        r = float(R_PER_MODE[mode, 0, 0]) ** 2
        closed = 0.5 * (np.log(p + r) - np.log(r))
        np.testing.assert_allclose(float(leaf.midpoint[a]), closed, atol=5e-3)
    # the two policies genuinely differ (sharp vs loose fix), so the term is policy-dependent
    assert abs(float(leaf.midpoint[0]) - float(leaf.midpoint[1])) > 0.1


def test_planning_switching_info_gain_toggles_off():
    system = make_planning_system(epistemic=False)
    params = make_planning_params()
    beliefs = {"mode": Categorical(logits=jnp.zeros(2)), "x": _X1}
    pols = jnp.asarray([[0], [1]], jnp.int32)[..., None]
    tree = system.score_policies(params, beliefs, pols)
    assert "infogain" not in tree
