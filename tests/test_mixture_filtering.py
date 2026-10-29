"""Workstream C phase 1, Agent 2: infer_states over mixtures (D3) + hybrid info gain (D4).

Covers the design doc's required checks:

- D3 parity vs *exact filtering by enumeration* (no reduction): both the factor
  posterior (Gaussian-sum condition) and the mode posterior (per-bank evidence
  message), at 1e-5;
- the mode-evidence message on a hand-computable 2-mode case;
- D4 state info gain vs a quadrature reference, bound ordering (lower <= upper),
  and the Koudahl flatness ablation (info gain constant across policies when the
  observation stream is action-independent);
- error paths for both narrowed ``NotImplementedError`` shapes;
- padding invariance of the new paths (dead components change nothing, incl.
  under grad).

The generative model is a 1-D switching LG world (the flagship class): M=2
mode-conditioned dynamics, a noisy position-only LG observation of the mixture
factor. ``mixture_schedule="mode_aligned"`` gives mode-owned banks (contiguous
``k/M`` slots per mode), which is what filtering the discrete mode from
continuous evidence requires.
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
    condition_mixture,
    condition_mixture_components,
    mode_evidence_message,
    observation_loglik,
)
from hybrid_aif.objectives import total
from hybrid_aif.objectives.continuous import state_info_gain
from hybrid_aif.system import System, SystemParams

# --------------------------------------------------------------------------- fixtures

H = jnp.asarray([[1.0]])
C = jnp.zeros(1)
R_CHOL = jnp.asarray([[0.5]])

# Mode-aligned prior over x: K=4, banks slots [0,1]=mode 0, [2,3]=mode 1.
# Bank masses (0.6, 0.4) are deliberately unequal and consistent with MODE_PRIOR.
X_PRIOR = MixtureGaussian(
    log_weights=jnp.log(jnp.asarray([0.36, 0.24, 0.28, 0.12])),
    means=jnp.asarray([[-1.0], [-0.5], [1.0], [1.5]]),
    chols=jnp.asarray([[[0.4]], [[0.3]], [[0.5]], [[0.35]]]),
)
MODE_PRIOR = Categorical.from_probs(jnp.asarray([0.6, 0.4]))
SPB = 2  # slots per bank


def make_system(**overrides):
    defaults = dict(
        A_deps=((1,),),  # modality y observes factor x
        B_deps=((), (0,)),  # x rides the mode
        controlled=(True, False),
        policy_len=1,
        mixture_k=4,
        mixture_schedule="mode_aligned",
        modality_names=("y",),
        factor_names=("mode", "x"),
    )
    return System(**{**defaults, **overrides})


# Action->mode table (child-first (modes, actions)); x's switching transition.
ACTION_MODE_LOGW = jnp.log(jnp.asarray([[0.50, 0.92, 0.08], [0.50, 0.08, 0.92]]))
INNOVATIONS = ConditionalGaussian(
    loc=jnp.asarray([[-1.2], [1.2]]),
    chol=jnp.asarray([[[0.3]], [[0.45]]]),
    parent_ndims=1,
)


def make_params(**overrides):
    defaults = dict(
        likelihood={"y": LinearGaussianObservation(H=H, c=C, noise=R_CHOL)},
        transition={
            "mode": ConditionalCategorical(log_table=ACTION_MODE_LOGW, parent_ndims=1),
            "x": LinearGaussianTransition(F=jnp.asarray([[0.9]]), innovations=INNOVATIONS),
        },
        initial_state_prior={"mode": Categorical(logits=jnp.zeros(2)), "x": X_PRIOR},
        obs_targets={},
        state_targets={},
        terminal_targets={},
    )
    return SystemParams(**{**defaults, **overrides})


# ---------------------------------------------- independent numpy enumeration reference


def _normal_pdf(y, mean, var):
    return np.exp(-0.5 * (y - mean) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def exact_filter_reference(x_prior, mode_prior_probs, y_val):
    """Textbook Gaussian-sum filter, by enumeration — no library kernels.

    Returns (post_weights, post_means, post_vars, mode_posterior) for a scalar
    LG observation ``y = x + eps``, ``eps ~ N(0, R)``, mode-owned banks of SPB.
    """
    w = np.asarray(jax.nn.softmax(x_prior.log_weights))
    mu = np.asarray(x_prior.means)[:, 0]
    var = np.asarray(x_prior.chols)[:, 0, 0] ** 2
    r = float(R_CHOL[0, 0]) ** 2
    pred_var = var + r
    n = _normal_pdf(y_val, mu, pred_var)  # per-component predictive density
    # factor posterior: Kalman update per component, weights ∝ w_k N_k
    gain = var / pred_var
    post_mu = mu + gain * (y_val - mu)
    post_var = (1.0 - gain) * var
    post_w = w * n
    post_w = post_w / post_w.sum()
    # mode posterior: q(mode=j) ∝ q_prior(mode=j) · Σ_{k in bank j} (w_k/W_j) N_k
    m = len(mode_prior_probs)
    w_banked = w.reshape(m, SPB)
    n_banked = n.reshape(m, SPB)
    within = w_banked / w_banked.sum(axis=1, keepdims=True)
    p_y_given_mode = (within * n_banked).sum(axis=1)
    mode_post = np.asarray(mode_prior_probs) * p_y_given_mode
    mode_post = mode_post / mode_post.sum()
    return post_w, post_mu, post_var, mode_post


# --------------------------------------------------------------------- D3: factor + mode


def test_infer_states_factor_posterior_matches_enumeration():
    system, params = make_system(), make_params()
    y = jnp.asarray([0.8])
    priors = {"mode": MODE_PRIOR, "x": X_PRIOR}
    x_post = system.infer_states(params, {"y": y}, priors=priors)[0]["x"]
    post_w, post_mu, post_var, _ = exact_filter_reference(X_PRIOR, MODE_PRIOR.probs, 0.8)
    np.testing.assert_allclose(np.asarray(jax.nn.softmax(x_post.log_weights)), post_w, atol=1e-5)
    np.testing.assert_allclose(np.asarray(x_post.means)[:, 0], post_mu, atol=1e-5)
    np.testing.assert_allclose(np.asarray(x_post.chols)[:, 0, 0] ** 2, post_var, atol=1e-5)


def test_infer_states_mode_posterior_matches_enumeration():
    system, params = make_system(), make_params()
    y = jnp.asarray([0.8])
    priors = {"mode": MODE_PRIOR, "x": X_PRIOR}
    mode_post = system.infer_states(params, {"y": y}, priors=priors)[0]["mode"]
    _, _, _, mode_ref = exact_filter_reference(X_PRIOR, MODE_PRIOR.probs, 0.8)
    assert isinstance(mode_post, Categorical)
    np.testing.assert_allclose(np.asarray(mode_post.probs), mode_ref, atol=1e-5)


def test_mode_posterior_shifts_toward_evidence():
    """A strongly positive observation should move the mode belief toward mode 1
    (the right-basin bank), starting from a mode-0-favoring prior."""
    system, params = make_system(), make_params()
    post, _ = system.infer_states(
        params, {"y": jnp.asarray([1.4])}, priors={"mode": MODE_PRIOR, "x": X_PRIOR}
    )
    mode_post = post["mode"]
    assert float(mode_post.probs[1]) > float(MODE_PRIOR.probs[1])
    assert float(mode_post.probs[1]) > 0.5


def test_mode_evidence_message_hand_computed_two_mode():
    """A hand-computable 2-mode case, one live slot per bank (banks = pure Gaussians)."""
    # bank 0: single Gaussian N(0, 1) at slot 0; bank 1: single Gaussian N(2, 1) at slot 1.
    lw = jnp.log(jnp.asarray([0.5, 0.5]))  # within-bank trivial (spb=1)
    means = jnp.asarray([[0.0], [2.0]])
    chols = jnp.asarray([[[1.0]], [[1.0]]])
    mix = MixtureGaussian(log_weights=lw, means=means, chols=chols)
    mode_prior = Categorical.from_probs(jnp.asarray([0.7, 0.3]))
    y = jnp.asarray([1.0])
    ll = observation_loglik(mix, H, C, R_CHOL, y)
    (mode_post,) = mode_evidence_message((mode_prior,), mix.log_weights, ll)
    # by hand: p(y|mode=j) = N(1; mu_j, sigma_j^2 + R). R = 0.25.
    r = 0.25
    p0 = _normal_pdf(1.0, 0.0, 1.0 + r)
    p1 = _normal_pdf(1.0, 2.0, 1.0 + r)
    post = np.asarray([0.7 * p0, 0.3 * p1])
    post = post / post.sum()
    np.testing.assert_allclose(np.asarray(mode_post.probs), post, atol=1e-6)


def test_infer_states_priors_none_uses_D():
    """priors=None resolves the declared D (unconditional mode prior + mixture D)."""
    system = make_system()
    params = make_params(initial_state_prior={"mode": MODE_PRIOR, "x": X_PRIOR})
    post, _ = system.infer_states(params, {"y": jnp.asarray([0.8])})
    post2, _ = system.infer_states(
        params, {"y": jnp.asarray([0.8])}, priors={"mode": MODE_PRIOR, "x": X_PRIOR}
    )
    a, b = post["mode"], post["x"]
    a2, b2 = post2["mode"], post2["x"]
    np.testing.assert_allclose(np.asarray(a.probs), np.asarray(a2.probs), atol=1e-6)
    np.testing.assert_allclose(np.asarray(b.means), np.asarray(b2.means), atol=1e-6)


def test_infer_states_jits():
    system, params = make_system(), make_params()
    priors = {"mode": MODE_PRIOR, "x": X_PRIOR}
    f = jax.jit(lambda p, y: system.infer_states(p, {"y": y}, priors=priors))
    mode_post = f(params, jnp.asarray([0.8]))[0]["mode"]
    mode_ref = system.infer_states(params, {"y": jnp.asarray([0.8])}, priors=priors)[0]["mode"]
    np.testing.assert_allclose(np.asarray(mode_post.probs), np.asarray(mode_ref.probs), atol=1e-6)


def test_condition_mixture_reference_log_evidence():
    """The reference kernel's log_evidence is the integrated evidence logsumexp."""
    y = jnp.asarray([0.8])
    post, log_ev = condition_mixture(X_PRIOR, H, C, R_CHOL, y)
    ll = observation_loglik(X_PRIOR, H, C, R_CHOL, y)
    lw = jax.nn.log_softmax(X_PRIOR.log_weights)
    ref = jax.scipy.special.logsumexp(lw + ll)
    np.testing.assert_allclose(float(log_ev), float(ref), atol=1e-6)
    assert post.num_components == X_PRIOR.num_components  # K unchanged


@pytest.mark.parametrize("prior", [X_PRIOR, "banked_padded"])
def test_condition_mixture_components_is_condition_mixture_plus_observation_loglik(prior):
    """One Kalman pass returns what the two reference kernels return, bitwise."""
    m = _banked_padded() if prior == "banked_padded" else prior
    y = jnp.asarray([0.8])
    post, log_ev, ll = condition_mixture_components(m, H, C, R_CHOL, y)
    post_ref, log_ev_ref = condition_mixture(m, H, C, R_CHOL, y)
    ll_ref = observation_loglik(m, H, C, R_CHOL, y)
    for a, b in zip(
        jax.tree_util.tree_leaves(post), jax.tree_util.tree_leaves(post_ref), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    np.testing.assert_array_equal(np.asarray(log_ev), np.asarray(log_ev_ref))
    np.testing.assert_array_equal(np.asarray(ll), np.asarray(ll_ref))


def test_infer_states_mode_aligned_path_is_the_two_kernel_reference():
    """``infer_states`` on the mode-aligned fixture reproduces the reference
    computation — ``condition_mixture`` for the factor posterior and evidence,
    ``observation_loglik`` + ``mode_evidence_message`` for the mode message."""
    system, params = make_system(), make_params()
    y = jnp.asarray([0.8])
    post, detail = system.infer_states(
        params, {"y": y}, priors={"mode": MODE_PRIOR, "x": X_PRIOR}, detail=True
    )
    mode_post, x_post = post["mode"], post["x"]
    post_ref, log_ev_ref = condition_mixture(X_PRIOR, H, C, R_CHOL, y)
    ll_ref = observation_loglik(X_PRIOR, H, C, R_CHOL, y)
    (mode_ref,) = mode_evidence_message((MODE_PRIOR,), X_PRIOR.log_weights, ll_ref)
    for a, b in zip(
        jax.tree_util.tree_leaves(x_post), jax.tree_util.tree_leaves(post_ref), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # the evidence is read under the mode belief, which here is the mixture's own bank mass:
    # the same number by another summation order
    np.testing.assert_allclose(
        np.asarray(detail.log_evidence["y"]), np.asarray(log_ev_ref), rtol=1e-6
    )
    np.testing.assert_array_equal(np.asarray(detail.mode_log_evidence["y"]), np.asarray(ll_ref))
    np.testing.assert_allclose(np.asarray(mode_post.probs), np.asarray(mode_ref.probs), atol=1e-12)


# ------------------------------------------------------------------- D3: mode-aligned only


def test_runnalls_schedule_raises_pointing_to_mode_aligned():
    system = make_system(mixture_schedule="runnalls")
    params = make_params()
    with pytest.raises(NotImplementedError, match="mode_aligned"):
        system.infer_states(
            params, {"y": jnp.asarray([0.8])}, priors={"mode": MODE_PRIOR, "x": X_PRIOR}
        )


def test_default_schedule_is_runnalls_and_still_raises():
    """Backward compatible: the default schedule (no override) refuses mixture evidence."""
    default = System(
        A_deps=((1,),),
        B_deps=((), (0,)),
        controlled=(True, False),
        policy_len=1,
        mixture_k=4,
        modality_names=("y",),
        factor_names=("mode", "x"),
    )
    assert default.mixture_schedule == "runnalls"
    with pytest.raises(NotImplementedError, match="mode_aligned"):
        default.infer_states(
            make_params(), {"y": jnp.asarray([0.8])}, priors={"mode": MODE_PRIOR, "x": X_PRIOR}
        )


def test_non_lg_modality_on_continuous_factor_raises_mnlr():
    """A categorical (non-LG) modality observing the mixture factor: MNLR / phase 2.

    ``infer_states`` raises this workstream-specific error ahead of the generic
    family-purity ``ValueError`` that ``planner_context`` (via ``_validate``)
    still gives, so the diagnostic names the successor (discrete-given-continuous
    likelihoods) rather than a blanket "family-pure only".
    """
    system = make_system(A_deps=((1,),), modality_names=("cat_y",))
    params = make_params(
        likelihood={"cat_y": ConditionalCategorical(log_table=jnp.zeros((2, 2)), parent_ndims=1)},
    )
    with pytest.raises(NotImplementedError, match="MNLR"):
        system.infer_states(
            params,
            {"cat_y": jnp.asarray(0, jnp.int32)},
            priors={"mode": MODE_PRIOR, "x": X_PRIOR},
        )


# ----------------------------------------------------------------------- D3: no mode parent


def test_mixture_factor_without_mode_parent_conditions_but_no_message():
    """A mixture-prior factor with no discrete parent: exact factor posterior, no
    mode message (there is no mode to update). Single factor, single LG modality."""
    system = System(
        A_deps=((0,),),
        B_deps=((),),
        controlled=(False,),
        policy_len=1,
        mixture_k=4,
        mixture_schedule="mode_aligned",
        modality_names=("y",),
        factor_names=("x",),
    )
    # uncontrolled parentless LG transition: trivial (1,) innovation parent axis.
    innov = ConditionalGaussian(loc=jnp.zeros((1, 1)), chol=jnp.asarray([[[0.3]]]), parent_ndims=1)
    params = SystemParams(
        likelihood={"y": LinearGaussianObservation(H=H, c=C, noise=R_CHOL)},
        transition={"x": LinearGaussianTransition(F=jnp.asarray([[0.9]]), innovations=innov)},
        initial_state_prior={"x": X_PRIOR},
        obs_targets={},
        state_targets={},
        terminal_targets={},
    )
    x_post = system.infer_states(params, {"y": jnp.asarray([0.8])}, priors={"x": X_PRIOR})[0]["x"]
    # factor-posterior reference: two mode-0 slots + two mode-1 slots as one flat
    # bank (mode split is irrelevant with no discrete parent to update).
    post_w, post_mu, _, _ = exact_filter_reference(X_PRIOR, [0.6, 0.4], 0.8)
    np.testing.assert_allclose(np.asarray(jax.nn.softmax(x_post.log_weights)), post_w, atol=1e-5)
    np.testing.assert_allclose(np.asarray(x_post.means)[:, 0], post_mu, atol=1e-5)


# ------------------------------------------------------------------------ D4: state info gain


def _mixture_entropy_quadrature(mix):
    """H[q(y)] for a scalar mixture, by dense trapezoid quadrature (the oracle)."""
    w = np.asarray(jax.nn.softmax(mix.log_weights))
    mu = np.asarray(mix.means)[:, 0]
    var = np.asarray(mix.chols)[:, 0, 0] ** 2
    lo = float(mu.min() - 8.0 * np.sqrt(var.max()))
    hi = float(mu.max() + 8.0 * np.sqrt(var.max()))
    grid = np.linspace(lo, hi, 40001)
    dens = np.sum(w[:, None] * _normal_pdf(grid[None, :], mu[:, None], var[:, None]), axis=0)
    integrand = np.where(dens > 0, -dens * np.log(dens), 0.0)
    return np.trapezoid(integrand, grid)


def _predictive_obs_scalar(mix):
    """The predictive obs mixture N(mu_j, sigma_j^2 + R) as a MixtureGaussian."""
    var = np.asarray(mix.chols)[:, 0, 0] ** 2 + float(R_CHOL[0, 0]) ** 2
    return MixtureGaussian(
        log_weights=mix.log_weights,
        means=mix.means,
        chols=jnp.sqrt(jnp.asarray(var))[:, None, None],
    )


def test_state_info_gain_within_quadrature_bounds():
    model = LinearGaussianObservation(H=H, c=C, noise=R_CHOL)
    leaf = state_info_gain((X_PRIOR,), (model,), ("y",))["infogain"]["y"]
    # reference: H[q(y)] (quadrature) - H[y|x] (exact constant)
    cond_h = 0.5 * (1.0 + np.log(2.0 * np.pi)) + np.log(float(R_CHOL[0, 0]))
    h_qy = _mixture_entropy_quadrature(_predictive_obs_scalar(X_PRIOR))
    ref = h_qy - cond_h
    assert float(leaf.lower) <= float(leaf.upper)  # bound ordering
    assert float(leaf.lower) - 1e-4 <= ref <= float(leaf.upper) + 1e-4  # KT sandwiches the oracle
    assert float(leaf.lower) > 0.0  # a genuine (bimodal) predictive: positive info gain


def test_state_info_gain_single_gaussian_is_exact_and_matches_closed_form():
    """K=1 (Gaussian factor): zero-width interval, equal to 0.5 log(1 + H P H^T / R)."""
    g = Gaussian(loc=jnp.asarray([0.3]), chol=jnp.asarray([[0.7]]))
    model = LinearGaussianObservation(H=H, c=C, noise=R_CHOL)
    leaf = state_info_gain((g,), (model,), ("y",))["infogain"]["y"]
    p = 0.7**2
    r = float(R_CHOL[0, 0]) ** 2
    closed = 0.5 * (np.log(p + r) - np.log(r))
    np.testing.assert_allclose(float(leaf.lower), float(leaf.upper), atol=1e-6)  # exact
    np.testing.assert_allclose(float(leaf.midpoint), closed, atol=1e-6)


def _three_policies(controlled):
    del controlled  # one column per control node: the mode's
    return jnp.asarray([[0, 0], [1, 1], [2, 2]], jnp.int32)[..., None]


def test_state_info_gain_flows_through_score_policies_and_toggles():
    system = make_system(policy_len=2)
    params = make_params(state_targets={"x": X_PRIOR})
    beliefs = {"mode": Categorical(logits=jnp.zeros(2)), "x": X_PRIOR}
    pols = _three_policies(system.controlled)
    tree = system.score_policies(params, beliefs, pols)
    assert set(tree["infogain"]) == {"y"}
    assert tree["infogain"]["y"].lower.shape == (3,)
    # x is observed by the eligible LG channel y, so it takes the channel estimator
    # and no leaf of its own (one state estimator per variable)
    assert set(tree) == {"risk", "infogain"}
    total(tree)  # registered term folds into G without error
    # block-grain toggle off removes the leaf entirely
    tree_off = make_system(policy_len=2, epistemic=False).score_policies(params, beliefs, pols)
    assert "infogain" not in tree_off
    # named-block toggle keeps only the named modality
    tree_named = make_system(policy_len=2, epistemic=("x",)).score_policies(params, beliefs, pols)
    assert set(tree_named["infogain"]) == {"y"}  # the variable selects, the channel keys


def test_state_info_gain_flat_when_observation_stream_action_independent():
    """Koudahl flatness: when actions do not change the mode belief, q(x) — and so
    the predictive observation mixture — is identical across policies, so the info
    gain leaf is constant (its per-policy spread is ~0). Ablating it changes nothing."""
    flat_table = jnp.log(jnp.asarray([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]))  # action-independent mode
    system = make_system(policy_len=2)
    params = make_params(
        transition={
            "mode": ConditionalCategorical(log_table=flat_table, parent_ndims=1),
            "x": make_params().transition["x"],
        },
        state_targets={"x": X_PRIOR},
    )
    beliefs = {"mode": Categorical(logits=jnp.zeros(2)), "x": X_PRIOR}
    pols = _three_policies(system.controlled)
    tree = system.score_policies(params, beliefs, pols)
    sig = tree["infogain"]["y"]
    assert float(jnp.std(sig.lower)) < 1e-6 and float(jnp.std(sig.upper)) < 1e-6


# --------------------------------------------------------------- padding invariance (new paths)


def _banked_padded():
    """The mode-aligned prior with one dead slot appended *inside each bank* (spb 2->3)."""
    # bank 0: slots 0,1 live + slot 2 dead; bank 1: slots 3,4 live + slot 5 dead.
    neg = -jnp.inf
    lw = jnp.asarray([jnp.log(0.36), jnp.log(0.24), neg, jnp.log(0.28), jnp.log(0.12), neg])
    means = jnp.asarray([[-1.0], [-0.5], [-1.0], [1.0], [1.5], [1.0]])
    chols = jnp.asarray([[[0.4]], [[0.3]], [[0.4]], [[0.5]], [[0.35]], [[0.5]]])
    return MixtureGaussian(log_weights=lw, means=means, chols=chols)


def test_mode_message_padding_invariant():
    y = jnp.asarray([0.8])
    ll_ref = observation_loglik(X_PRIOR, H, C, R_CHOL, y)
    (ref,) = mode_evidence_message((MODE_PRIOR,), X_PRIOR.log_weights, ll_ref)
    padded = _banked_padded()
    ll_pad = observation_loglik(padded, H, C, R_CHOL, y)
    (pad,) = mode_evidence_message((MODE_PRIOR,), padded.log_weights, ll_pad)
    np.testing.assert_allclose(np.asarray(pad.probs), np.asarray(ref.probs), atol=1e-6)


def test_state_info_gain_padding_invariant():
    model = LinearGaussianObservation(H=H, c=C, noise=R_CHOL)
    ref = state_info_gain((X_PRIOR,), (model,), ("y",))["infogain"]["y"]
    pad = state_info_gain((_banked_padded(),), (model,), ("y",))["infogain"]["y"]
    np.testing.assert_allclose(float(pad.lower), float(ref.lower), atol=1e-5)
    np.testing.assert_allclose(float(pad.upper), float(ref.upper), atol=1e-5)


@pytest.mark.parametrize("field", ["log_weights", "means", "chols"])
def test_gradients_through_new_paths_are_finite(field):
    model = LinearGaussianObservation(H=H, c=C, noise=R_CHOL)
    padded = _banked_padded()
    y = jnp.asarray([0.8])

    def info_gain(x):
        m = dataclasses.replace(padded, **{field: x})
        return state_info_gain((m,), (model,), ("y",))["infogain"]["y"].lower

    def mode_msg(x):
        m = dataclasses.replace(padded, **{field: x})
        ll = observation_loglik(m, H, C, R_CHOL, y)
        (q,) = mode_evidence_message((MODE_PRIOR,), m.log_weights, ll)
        return q.probs[1]

    for fn in (info_gain, mode_msg):
        g = jax.grad(fn)(getattr(padded, field))
        assert bool(jnp.all(jnp.isfinite(g))), f"non-finite grad through {fn.__name__} d/d{field}"


def test_per_mode_F_bank_is_a_documented_contract():
    """A leading-M ``F`` bank — ``F.shape == (*parent_shape, d, d)`` — conditions
    the linear map by the same discrete parents as the innovations (the SLDS
    pattern the class docstring names). Pinned here at the request of the first
    external consumer (the downstream mission model: committed modes contract pose
    toward a site, F=(1-alpha)I; ESCALATE/IDLE hold, F=I), whose friction log
    noted the bank was accepted and behaved correctly but had no test showing
    it. Semantics pinned: mode j's one-step prediction uses F[j], b[j] — a
    concentrated mode belief reproduces the single-map prediction with that
    mode's row, exactly."""
    F_bank = jnp.asarray([[[0.9]], [[1.0]]])  # mode 0 contracts, mode 1 holds
    # a DELTA action->mode table so the post-transition mode belief (D2's
    # weights) is concentrated and the bank's row is isolated exactly
    delta = 1.0 - 1e-12
    mode_delta = jnp.log(jnp.asarray([[delta, delta, 1 - delta], [1 - delta, 1 - delta, delta]]))
    per_mode = make_params(
        transition={
            "mode": ConditionalCategorical(log_table=mode_delta, parent_ndims=1),
            "x": LinearGaussianTransition(F=F_bank, innovations=INNOVATIONS),
        }
    )
    for j, f_jj in enumerate((0.9, 1.0)):
        single = make_params(
            transition={
                "mode": ConditionalCategorical(log_table=mode_delta, parent_ndims=1),
                "x": LinearGaussianTransition(F=jnp.asarray([[f_jj]]), innovations=INNOVATIONS),
            }
        )
        mode_j = Categorical(
            logits=jnp.log(jnp.asarray([1.0 - 1e-12, 1e-12] if j == 0 else [1e-12, 1.0 - 1e-12]))
        )
        x0 = Gaussian(loc=jnp.asarray([0.7]), chol=jnp.asarray([[0.2]]))
        system = make_system()
        beliefs = {"mode": mode_j, "x": x0}
        pred_bank = system.predict_beliefs(per_mode, beliefs, jnp.asarray([[1 + j]]))
        pred_single = system.predict_beliefs(single, beliefs, jnp.asarray([[1 + j]]))
        got = np.asarray(pred_bank[-1]["x"].mean())
        want = np.asarray(pred_single[-1]["x"].mean())
        np.testing.assert_allclose(got, want, atol=1e-6)
