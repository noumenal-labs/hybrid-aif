"""``obs_targets`` on linear-Gaussian modalities: preferences over continuous observations.

The v0.2 fence ("obs_targets on LG modality(ies) ... not supported") comes out
here. A C designation on an LG modality is a Gaussian (or ``MixtureGaussian``)
preference density over the modality's *observation* space —
covariance-as-tolerance, the same reading ``state_targets`` has — scored against
the predicted observation ``q(o)``, the pushforward of the observed factor's
belief through ``(H, c, R)``:

    ``q(o) = Σ_k w_k N(H μ_k + c, H Σ_k Hᵀ + R Rᵀ)``      (plain LG)
    ``q(o) = Σ_{k,j} w_k q(j) N(H_j μ_k + c_j, H_j Σ_k H_jᵀ + R_j R_jᵀ)``  (switching)

and the two leaves mirror the discrete ones exactly:

- ``utility_infogain`` → ``utility[modality] = E_q(o)[log C]``, computed as
  ``−CE(q(o), C)``. Per component and Gaussian ``C = N(ν, Λ)``:
  ``E_{N(m,S)}[log C] = −½[d log 2π + log|Λ| + tr(Λ⁻¹S) + (m−ν)ᵀΛ⁻¹(m−ν)]`` —
  exact however many components ``q(o)`` has (linearity), a genuine
  Kolchinsky–Tracey Interval when ``C`` is itself a mixture;
- ``risk_ambiguity`` → ``risk[modality] = KL(q(o) ‖ C)`` through the existing
  ``kl_to`` dispatch, plus ``ambiguity[modality] = E_q(x)[H[y|x]]``
  (``= Σ_j q(j) ½ log|2πe R_j R_jᵀ|``, exact and constant in ``x``).

The acceptance's heart is that those still decompose into the same G:
``KL(q(o)‖C) + E[H[y|x]] = −E_q(o)[log C] − H[q(o)] + E[H[y|x]]`` and the
channel's info-gain leaf is ``H[q(o)] − E[H[y|x]]``.

The behavioral case is friction log finding 6: "prefer observing my pose at the
committed task's location" — a switching modality with per-mode offsets and a
zero-centred target — and the two measured failure modes of the
``state_targets`` workaround it replaces (holding-still caging, centroid blur).
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
from hybrid_aif.objectives import total
from hybrid_aif.objectives.continuous import (
    conditional_obs_entropy,
    lg_ambiguity,
    lg_utility,
    predictive_obs,
    risk,
)
from hybrid_aif.system import System, SystemParams

# --------------------------------------------------------------- hand-computable pieces

_LOG_2PI = float(np.log(2.0 * np.pi))

QX = Gaussian(loc=jnp.asarray([0.4, -0.2]), chol=jnp.asarray([[0.3, 0.0], [0.1, 0.25]]))
QX_MIX = MixtureGaussian(
    log_weights=jnp.log(jnp.asarray([0.7, 0.3])),
    means=jnp.asarray([[0.4, -0.2], [-0.5, 0.6]]),
    chols=jnp.stack([jnp.asarray([[0.3, 0.0], [0.1, 0.25]]), 0.5 * jnp.eye(2)]),
)
H2 = jnp.asarray([[1.0, 0.0], [0.5, 1.0]])
C2 = jnp.asarray([0.2, -0.1])
R2 = jnp.asarray([[0.15, 0.0], [0.05, 0.2]])
MODEL = LinearGaussianObservation(H=H2, c=C2, noise=R2)
TARGET = Gaussian(loc=jnp.zeros(2), chol=jnp.diag(jnp.asarray([0.5, 0.4])))
TARGET_MIX = MixtureGaussian(
    log_weights=jnp.log(jnp.asarray([0.6, 0.4])),
    means=jnp.asarray([[0.6, 0.0], [-0.6, 0.3]]),
    chols=jnp.stack([0.3 * jnp.eye(2), jnp.diag(jnp.asarray([0.25, 0.4]))]),
)

# switching geometry: mode 1 has a different gain, a big offset and looser noise
H_MODES = jnp.stack([H2, 0.8 * H2])
C_MODES = jnp.stack([C2, jnp.asarray([-1.0, 0.5])])
R_MODES = jnp.stack([R2, 2.0 * R2])
SWITCH_MODEL = LinearGaussianObservation(H=H_MODES, c=C_MODES, noise=R_MODES)
MODE = Categorical(logits=jnp.log(jnp.asarray([0.7, 0.3])))


def _cov(chol):
    chol = np.asarray(chol, np.float64)
    return chol @ chol.T


def _expected_log_gaussian(m, S, target_loc, target_cov):
    """``E_{N(m,S)}[log N(x; ν, Λ)]``, the closed form the utility leaf must hit."""
    m, S = np.asarray(m, np.float64), np.asarray(S, np.float64)
    nu, lam = np.asarray(target_loc, np.float64), np.asarray(target_cov, np.float64)
    prec = np.linalg.inv(lam)
    dev = m - nu
    return -0.5 * (
        len(m) * _LOG_2PI + np.log(np.linalg.det(lam)) + np.trace(prec @ S) + dev @ prec @ dev
    )


def _gaussian_entropy(S):
    S = np.asarray(S, np.float64)
    return 0.5 * (len(S) * (1.0 + _LOG_2PI) + np.log(np.linalg.det(S)))


def _pushforward(H, c, R_chol, loc, cov):
    """One component's predictive moments ``(H μ + c, H Σ Hᵀ + R Rᵀ)``."""
    H, c = np.asarray(H, np.float64), np.asarray(c, np.float64)
    loc, cov = np.asarray(loc, np.float64), np.asarray(cov, np.float64)
    return H @ loc + c, H @ cov @ H.T + _cov(R_chol)


def _leaf(tree, term, name="look"):
    return tree[term][name]


# ------------------------------------------------------ 1. closed forms, plain LG modality


def test_gaussian_target_on_a_plain_lg_modality_is_the_closed_form():
    """A Gaussian belief through a shared ``(H, c, R)`` against a Gaussian target:
    utility, risk and the ambiguity leaf all hand-computed, all exact."""
    q_obs = predictive_obs(QX, MODEL)
    assert q_obs.num_components == 1  # a Gaussian belief is the K=1 case
    m, S = _pushforward(H2, C2, R2, QX.loc, QX.cov)
    np.testing.assert_allclose(np.asarray(q_obs.means[0]), m, atol=1e-6)
    np.testing.assert_allclose(np.asarray(q_obs.components.cov[0]), S, atol=1e-6)

    want_utility = _expected_log_gaussian(m, S, TARGET.loc, _cov(TARGET.chol))
    leaf = _leaf(lg_utility((q_obs,), {"look": TARGET}, ("look",)), "utility")
    np.testing.assert_allclose(float(leaf.lower), want_utility, atol=1e-5)
    assert float(leaf.width) == 0.0  # exact: Gaussian q(o), Gaussian C

    # risk is the same quantity minus the predictive entropy (kl_to's closed form)
    want_risk = -want_utility - _gaussian_entropy(S)
    risk_leaf = _leaf(risk((q_obs,), {"look": TARGET}, ("look",)), "risk")
    np.testing.assert_allclose(float(risk_leaf.lower), want_risk, atol=1e-5)
    assert float(risk_leaf.width) == 0.0

    # ambiguity: H[y|x] = ½ log|2πe R Rᵀ|, constant in x
    amb = _leaf(lg_ambiguity((MODEL,), ((),), ("look",)), "ambiguity")
    np.testing.assert_allclose(float(amb.lower), _gaussian_entropy(_cov(R2)), atol=1e-6)
    assert float(amb.width) == 0.0


def test_a_mixture_belief_with_a_gaussian_target_is_exact_by_linearity():
    """``E_q(o)[log C]`` over a *mixture* predictive is the weighted sum of the
    per-component closed forms — exact, no bounds, because the expectation is
    linear and only the target's log-sum would need bounding."""
    q_obs = predictive_obs(QX_MIX, MODEL)
    assert q_obs.num_components == 2
    weights = np.asarray(jax.nn.softmax(QX_MIX.log_weights), np.float64)
    want = 0.0
    for k in range(2):
        m, S = _pushforward(H2, C2, R2, QX_MIX.means[k], QX_MIX.components.cov[k])
        want += weights[k] * _expected_log_gaussian(m, S, TARGET.loc, _cov(TARGET.chol))
    leaf = _leaf(lg_utility((q_obs,), {"look": TARGET}, ("look",)), "utility")
    np.testing.assert_allclose(float(leaf.lower), want, atol=1e-5)
    assert float(leaf.width) == 0.0


def test_undesignated_modalities_are_omitted_leaves_not_zeros():
    """The term-tree toggle convention: no designation, no leaf."""
    q_obs = predictive_obs(QX, MODEL)
    tree = lg_utility((q_obs, q_obs), {"look": TARGET}, ("look", "other"))
    assert set(tree["utility"]) == {"look"}


# ------------------------------------------------------------- 2. mixture target: bounds


def test_mixture_target_interval_brackets_a_monte_carlo_estimate():
    """A mixture C has no closed-form ``E_q[log C]``, so the leaf is a genuine
    Kolchinsky–Tracey Interval — and it must bracket the sampled truth."""
    q_obs = predictive_obs(QX_MIX, MODEL)
    leaf = _leaf(lg_utility((q_obs,), {"look": TARGET_MIX}, ("look",)), "utility")
    assert float(leaf.width) > 0.01, "a mixture target must not report exactness"

    samples = q_obs.sample(jax.random.PRNGKey(0), (400_000,))
    mc = float(jnp.mean(TARGET_MIX.log_prob(samples)))
    assert float(leaf.lower) <= mc <= float(leaf.upper), (
        f"MC {mc} outside [{float(leaf.lower)}, {float(leaf.upper)}]"
    )

    # and the risk leaf brackets the same quantity's KL reading
    risk_leaf = _leaf(risk((q_obs,), {"look": TARGET_MIX}, ("look",)), "risk")
    h = q_obs.entropy()
    assert float(risk_leaf.lower) <= -mc - float(h.midpoint) <= float(risk_leaf.upper)


# -------------------------------------------- 2b. the certainty tax (friction log 11-13)

#: a 1-D channel with near-noiseless sensing, so the predictive covariance can be
#: driven well *below* the target's tolerance — where the two forms disagree
TAX_MODEL = LinearGaussianObservation(
    H=jnp.asarray([[1.0]]), c=jnp.zeros(1), noise=jnp.asarray([[0.02]])
)
TAX_TARGET = Gaussian(loc=jnp.zeros(1), chol=jnp.asarray([[0.5]]))
TAX_BIAS = 0.12  # fixed mean offset: only the belief's *width* varies below


def _tax_utility_and_risk(sigma):
    q = Gaussian(loc=jnp.asarray([TAX_BIAS]), chol=jnp.asarray([[sigma]]))
    q_obs = predictive_obs(q, TAX_MODEL)
    utility = _leaf(lg_utility((q_obs,), {"look": TAX_TARGET}, ("look",)), "utility")
    risk_leaf = _leaf(risk((q_obs,), {"look": TAX_TARGET}, ("look",)), "risk")
    return float(utility.lower), float(risk_leaf.lower)


def test_the_utility_form_carries_no_certainty_tax():
    """The customer's hard requirement (friction log 11-13). ``E_q(o)[log C]`` has
    **no** ``−H(q)`` term, so at fixed bias it is

        ``const − (bias² + Var_q) / 2σ_C²``   (per dimension, Gaussian target)

    — strictly *improving* as the belief tightens, all the way past the target's
    own tolerance. A tighter-than-target belief is never penalized for its
    tightness alone. Pinned as a sign, because this is what moved a fielded
    escalate crossover by four attempts when the risk form supplied it instead.
    """
    sigmas = [0.6, 0.45, 0.3, 0.2, 0.1, 0.05, 0.03]
    utilities = [_tax_utility_and_risk(s)[0] for s in sigmas]
    for wide, tight in zip(utilities[:-1], utilities[1:], strict=True):
        assert tight > wide, f"utility worsened as the belief tightened: {utilities}"

    # and it really is the −Var_q/2σ_C² slope: differences match the closed form
    sigma_c2 = float(TAX_TARGET.chol[0, 0]) ** 2
    r2 = float(TAX_MODEL.R_chol[0, 0]) ** 2
    for s, u in zip(sigmas, utilities, strict=True):
        want = _expected_log_gaussian([TAX_BIAS], [[s**2 + r2]], [0.0], [[sigma_c2]])
        np.testing.assert_allclose(u, want, atol=1e-5)


def test_the_risk_form_does_carry_the_certainty_tax_and_can_invert_the_ranking():
    """The other side of the same identity, stated so nobody reaches for ``risk``
    à la carte: ``KL(q(o)‖C) = −E_q(o)[log C] − H[q(o)]``, and that ``−H`` turns
    around once the belief is tighter than the target — so risk alone *prefers* the
    blurrier of two equally-centred beliefs. Coherent only inside the full
    ``risk_ambiguity`` fold, where the channel's info-gain/ambiguity leaves restore
    what the ``−H`` removed."""
    tight, blurry = 0.05, float(TAX_TARGET.chol[0, 0])
    u_tight, r_tight = _tax_utility_and_risk(tight)
    u_blurry, r_blurry = _tax_utility_and_risk(blurry)
    assert u_tight > u_blurry  # the utility form prefers the sharper belief
    assert r_tight > r_blurry  # risk alone prefers the blurrier one: the tax
    # the wedge between the two rankings is exactly the predictive entropy
    q_obs = predictive_obs(
        Gaussian(loc=jnp.asarray([TAX_BIAS]), chol=jnp.asarray([[tight]])), TAX_MODEL
    )
    np.testing.assert_allclose(r_tight, -u_tight - float(q_obs.entropy().lower), atol=1e-5)


# ----------------------------------------------------------- 3. the switching modality


def _manual_switching_predictive():
    """Every ``(component k, mode j)`` piece, in the component-major layout."""
    weights = np.asarray(jax.nn.softmax(QX_MIX.log_weights), np.float64)
    modes = np.asarray(MODE.probs, np.float64)
    out = []
    for k in range(QX_MIX.num_components):
        for j in range(2):
            m, S = _pushforward(
                H_MODES[j], C_MODES[j], R_MODES[j], QX_MIX.means[k], QX_MIX.components.cov[k]
            )
            out.append((weights[k] * modes[j], m, S))
    return out


def test_switching_modality_predictive_matches_a_manual_per_mode_computation():
    """The ``(K·M)`` predictive: weights, means and covariances slot by slot."""
    q_obs = predictive_obs(QX_MIX, SWITCH_MODEL, (MODE,))
    manual = _manual_switching_predictive()
    assert q_obs.num_components == len(manual) == 4
    got_w = np.asarray(jax.nn.softmax(q_obs.log_weights), np.float64)
    for slot, (w, m, S) in enumerate(manual):
        np.testing.assert_allclose(got_w[slot], w, atol=1e-6)
        np.testing.assert_allclose(np.asarray(q_obs.means[slot]), m, atol=1e-6)
        np.testing.assert_allclose(np.asarray(q_obs.components.cov[slot]), S, atol=1e-5)


def test_switching_modality_utility_and_ambiguity_match_the_manual_sums():
    """Per-mode offsets are what the friction-6 preference rides on, so pin the
    leaves against the explicit mode sums."""
    q_obs = predictive_obs(QX_MIX, SWITCH_MODEL, (MODE,))
    want = sum(
        w * _expected_log_gaussian(m, S, TARGET.loc, _cov(TARGET.chol))
        for w, m, S in _manual_switching_predictive()
    )
    leaf = _leaf(lg_utility((q_obs,), {"look": TARGET}, ("look",)), "utility")
    np.testing.assert_allclose(float(leaf.lower), want, atol=1e-5)
    assert float(leaf.width) == 0.0

    modes = np.asarray(MODE.probs, np.float64)
    want_amb = sum(modes[j] * _gaussian_entropy(_cov(R_MODES[j])) for j in range(2))
    amb = _leaf(lg_ambiguity((SWITCH_MODEL,), ((MODE,),), ("look",)), "ambiguity")
    np.testing.assert_allclose(float(amb.lower), want_amb, atol=1e-6)
    np.testing.assert_allclose(
        float(conditional_obs_entropy(SWITCH_MODEL, (MODE,))), want_amb, atol=1e-6
    )


def test_leafwise_decomposition_identity_holds_for_every_configuration():
    """``KL(q(o)‖C) = −E_q(o)[log C] − H[q(o)]`` at both interval endpoints, over
    the whole cross of {plain, switching} × {Gaussian, mixture} target.

    The one documented wedge is ``kl_to``'s clamp at zero: when *both* the CE and
    the entropy are bound pairs their crosswise difference can dip below zero
    (``mixture/mix-target`` here does, by 0.43 nats), and the KL leaf reports 0
    instead of a value it knows to be impossible. Applying the same clamp to the
    utility-side reconstruction is the identity's exact statement.
    """
    cases = {
        "plain/gauss": (predictive_obs(QX, MODEL), TARGET),
        "plain/mix-target": (predictive_obs(QX, MODEL), TARGET_MIX),
        "mixture/gauss": (predictive_obs(QX_MIX, MODEL), TARGET),
        "mixture/mix-target": (predictive_obs(QX_MIX, MODEL), TARGET_MIX),
        "switching/gauss": (predictive_obs(QX_MIX, SWITCH_MODEL, (MODE,)), TARGET),
        "switching/mix-target": (predictive_obs(QX_MIX, SWITCH_MODEL, (MODE,)), TARGET_MIX),
    }
    for label, (q_obs, target) in cases.items():
        utility = _leaf(lg_utility((q_obs,), {"look": target}, ("look",)), "utility")
        risk_leaf = _leaf(risk((q_obs,), {"look": target}, ("look",)), "risk")
        h = q_obs.entropy()
        assert float(h.lower) <= float(h.upper), label
        assert float(risk_leaf.lower) <= float(risk_leaf.upper), label
        # bounds combine crosswise: KL_lo = CE_lo − H_hi, KL_hi = CE_hi − H_lo
        np.testing.assert_allclose(
            float(risk_leaf.lower),
            max(0.0, -float(utility.upper) - float(h.upper)),
            atol=1e-5,
            err_msg=f"{label} lower",
        )
        np.testing.assert_allclose(
            float(risk_leaf.upper),
            max(0.0, -float(utility.lower) - float(h.lower)),
            atol=1e-5,
            err_msg=f"{label} upper",
        )
    # the clamp really is reachable, and only with bounds on both sides
    both_bounded = risk((predictive_obs(QX_MIX, MODEL),), {"look": TARGET_MIX}, ("look",))["risk"][
        "look"
    ]
    assert float(both_bounded.lower) == 0.0


# --------------------------------------------------- 4. the friction-6 mission fixture

ALPHA = 0.5  # per-step fraction of the gap to the committed site that transit closes
SITES = (0.0, 2.0)  # two task locations on a 1-D corridor
TOL = 0.3  # "pose is at the committed site" tolerance
COMMIT, MOVE = 0, 0  # control-index names: commit level, pose control
FREEZE = 1

#: the commitment factor: the control sets the level (near-deterministically)
COMMIT_B = ConditionalCategorical(
    log_table=jnp.log(jnp.asarray([[0.98, 0.02], [0.02, 0.98]])), parent_ndims=1
)
#: pose innovations conditioned on (committed site, pose control): transit closes
#: ALPHA of the gap to the committed site, FREEZE holds position
POSE_B = LinearGaussianTransition(
    F=jnp.asarray([[[[1.0 - ALPHA]], [[1.0]]]] * 2),
    innovations=ConditionalGaussian(
        loc=jnp.asarray([[[ALPHA * SITES[0]], [0.0]], [[ALPHA * SITES[1]], [0.0]]]),
        chol=jnp.broadcast_to(jnp.asarray([[0.05]]), (2, 2, 1, 1)),
        parent_ndims=2,
    ),
)
#: the switching modality: pose measured in the committed site's frame (per-mode
#: offset −site), so "at the committed site" is literally "y ≈ 0"
LOOK = LinearGaussianObservation(
    H=jnp.asarray([[1.0]]),
    c=jnp.asarray([[-SITES[0]], [-SITES[1]]]),
    noise=jnp.asarray([[0.1]]),
)
AT_SITE = Gaussian(loc=jnp.zeros(1), chol=jnp.asarray([[TOL]]))


def mission_system(**overrides):
    defaults = dict(
        A_deps=((1, 0),),  # look observes pose, switching on commit
        B_deps=((), (0,)),
        controlled=(True, True),
        policy_len=2,
        mixture_k=4,
        modality_names=("look",),
        factor_names=("commit", "pose"),
        epistemic=False,
    )
    return System(**{**defaults, **overrides})


def mission_params(pose0=1.4, target=AT_SITE, **overrides):
    defaults = dict(
        likelihood={"look": LOOK},
        transition={"commit": COMMIT_B, "pose": POSE_B},
        initial_state_prior={
            "commit": Categorical(logits=jnp.zeros(2)),
            "pose": MixtureGaussian(
                log_weights=jnp.zeros(1),
                means=jnp.asarray([[pose0]]),
                chols=jnp.asarray([[[0.15]]]),
            ),
        },
        obs_targets={"look": target},
    )
    return SystemParams(**{**defaults, **overrides})


#: (commit level, pose control) held for both steps — the four standing plans
def _plan(commit, pose_ctl):
    return [[commit, pose_ctl], [commit, pose_ctl]]


PLANS = jnp.asarray([_plan(0, MOVE), _plan(0, FREEZE), _plan(1, MOVE), _plan(1, FREEZE)], jnp.int32)
A_MOVE, A_FREEZE, B_MOVE, B_FREEZE = range(4)


def _mission_g(pose0=1.4, target=AT_SITE, **overrides):
    system = mission_system(**overrides)
    params = mission_params(pose0, target)
    tree = system.score_policies(params, dict(params.initial_state_prior), PLANS)
    return system, params, tree, total(tree)


# ------------------------------------------------ 5. the term tree's shape and coherence


def test_lg_leaves_join_the_existing_terms_keyed_by_modality():
    """D4/V3: the new leaves are ``utility``/``risk``/``ambiguity`` entries under the
    modality's own name — no new term names, so ``total``'s signs already cover them."""
    _, _, ig_tree, _ = _mission_g(epistemic=True)
    assert set(ig_tree) == {"utility", "infogain"}
    assert set(ig_tree["utility"]) == {"look"} and set(ig_tree["infogain"]) == {"look"}
    _, _, ra_tree, _ = _mission_g(epistemic=True, decomposition="risk_ambiguity")
    assert set(ra_tree) == {"risk", "ambiguity"}
    assert set(ra_tree["risk"]) == {"look"} and set(ra_tree["ambiguity"]) == {"look"}


#: a two-bump "either edge of the site will do" preference — the mixture-target case
NEAR_EITHER_EDGE = MixtureGaussian(
    log_weights=jnp.log(jnp.asarray([0.5, 0.5])),
    means=jnp.asarray([[-0.35], [0.35]]),
    chols=jnp.asarray([[[0.2]], [[0.2]]]),
)


@pytest.mark.parametrize("target", [AT_SITE, NEAR_EITHER_EDGE])
def test_the_two_decompositions_agree_on_g_at_both_endpoints(target):
    """The acceptance's heart, the way ``test_infogain_tree`` pins it for discrete:
    with the channel's epistemic leaf mounted, ``utility + infogain`` and
    ``risk + ambiguity`` fold to the same G — *both* interval endpoints, since the
    CE and entropy pieces the two paths use are literally the same ones."""
    totals = {}
    for decomposition in ("utility_infogain", "risk_ambiguity"):
        _, _, tree, g = _mission_g(target=target, epistemic=True, decomposition=decomposition)
        assert np.all(np.asarray(g.lower) <= np.asarray(g.upper) + 1e-6)
        totals[decomposition] = g
    lo, hi = totals["utility_infogain"], totals["risk_ambiguity"]
    np.testing.assert_allclose(np.asarray(lo.lower), np.asarray(hi.lower), atol=1e-5)
    np.testing.assert_allclose(np.asarray(lo.upper), np.asarray(hi.upper), atol=1e-5)


def test_a_mixture_target_widens_g_while_a_gaussian_target_leaves_it_narrow():
    """Bound semantics stay honest: the only width a Gaussian target contributes is
    the predictive's own KT entropy gap; a mixture target adds the CE gap on top."""
    _, _, _, g_gauss = _mission_g(target=AT_SITE, epistemic=True)
    _, _, _, g_mix = _mission_g(target=NEAR_EITHER_EDGE, epistemic=True)
    assert np.all(np.asarray(g_mix.width) > np.asarray(g_gauss.width))
    assert np.all(np.asarray(g_gauss.width) >= 0.0)


def test_each_decomposition_ablates_its_own_epistemic_leaf():
    """V11 decision 1: ``epistemic=False`` ablates ``infogain`` under ``utility_infogain`` and
    ``ambiguity`` under ``risk_ambiguity``, and risk keeps its ``−H[q(o)]``, so the two forms
    differ by exactly the channel's predictive entropy ``H[q(look)] = I(look; S) + E H[look |
    S]`` (read off the mounted leaves); with the leaf mounted they agree (the test above)."""
    _, _, ig_tree, g_ig = _mission_g(epistemic=False)
    _, _, ra_tree, g_ra = _mission_g(epistemic=False, decomposition="risk_ambiguity")
    assert set(ig_tree) == {"utility"} and set(ra_tree) == {"risk"}
    _, _, ig_on, _ = _mission_g(epistemic=True)
    _, _, ra_on, _ = _mission_g(epistemic=True, decomposition="risk_ambiguity")
    entropy = ig_on["infogain"]["look"].midpoint + ra_on["ambiguity"]["look"].midpoint
    np.testing.assert_allclose(
        np.asarray(g_ig.midpoint) - np.asarray(g_ra.midpoint), np.asarray(entropy), atol=1e-4
    )


# ---------------------------------------------------- 6. friction 6, the behavioral case


def test_prefers_the_commitment_whose_mode_offset_cancels_the_observation():
    """The mission model's natural preference, expressed directly: pose starts at
    1.4 on a corridor with sites at 0 and 2, the modality measures pose in the
    *committed* site's frame, and a zero-centred target therefore picks the
    commitment whose offset cancels — site B, the near one."""
    system, params, tree, g = _mission_g(pose0=1.4)
    assert system._is_hybrid(params)
    best = int(np.argmin(np.asarray(g.lower)))
    assert best == B_MOVE, f"argmin was {best}, expected B_MOVE"
    # the preference is ordered by how well each commitment's frame cancels the pose
    assert float(g.lower[B_MOVE]) < float(g.lower[A_MOVE])
    assert float(g.lower[A_MOVE]) < float(g.lower[A_FREEZE])


def test_no_holding_still_caging_transit_beats_freezing():
    """Workaround failure mode 1 (the pose term caging the robot): under the obs
    target, committing *and transiting* strictly beats freezing in place, for either
    commitment — because the preference is over the observation, so the only way to
    drive ``y`` to zero is to close the gap."""
    _, _, _, g = _mission_g(pose0=1.4)
    assert float(g.lower[B_MOVE]) < float(g.lower[B_FREEZE])
    assert float(g.lower[A_MOVE]) < float(g.lower[A_FREEZE])
    assert min(float(g.lower[B_MOVE]), float(g.lower[A_MOVE])) < min(
        float(g.lower[B_FREEZE]), float(g.lower[A_FREEZE])
    )


def test_no_centroid_blur_the_preferred_mode_is_the_argmax_not_the_centroid():
    """Workaround failure mode 2 (components blurring into one bump at the task
    centroid, so central moves beat near moves). Two readings:

    - the winning plan's *predicted pose* heads to site B, not to the centroid;
    - started exactly at the centroid (pose 1.0, equidistant), the two site
      commitments **tie** and both beat freezing there — the target never rewards
      the centroid itself, which is precisely what a blurred state-target mixture
      does.
    """
    system, params, _, g = _mission_g(pose0=1.4)
    traj = system.predict_beliefs(params, dict(params.initial_state_prior), PLANS[B_MOVE])
    poses = [float(step["pose"].mean()[0]) for step in traj]
    assert poses[0] > 1.4 and poses[-1] > poses[0]  # moving toward site B
    centroid = 0.5 * (SITES[0] + SITES[1])
    assert abs(poses[-1] - SITES[1]) < abs(poses[-1] - centroid)

    _, _, _, g_mid = _mission_g(pose0=centroid)
    np.testing.assert_allclose(float(g_mid.lower[A_MOVE]), float(g_mid.lower[B_MOVE]), atol=1e-5)
    assert float(g_mid.lower[B_MOVE]) < float(g_mid.lower[B_FREEZE])
    del g


def test_lg_obs_targets_live_on_the_hybrid_path_only():
    """The compiled prob-space path is untouched by this work: an LG declaration
    forces the hybrid dispatch, so ``compile_tables`` never sees a continuous C
    designation and the fused closures keep their raw-probability carry."""
    system, params, _, _ = _mission_g()
    assert system._is_hybrid(params)
    ctx = system.planner_context(params, dict(params.initial_state_prior))
    assert not any(isinstance(b, jnp.ndarray) for b in ctx.init_beliefs)


def test_the_designation_survives_a_jit_of_the_whole_planning_face():
    system = mission_system(epistemic=True)
    params = mission_params()
    beliefs = dict(params.initial_state_prior)
    f = jax.jit(lambda p, b: total(system.score_policies(p, b, PLANS)).lower)
    np.testing.assert_allclose(
        np.asarray(f(params, beliefs)),
        np.asarray(total(system.score_policies(params, beliefs, PLANS)).lower),
        atol=1e-6,
    )


# ------------------------------------------------------------------- 7. loud validation


def test_a_categorical_target_on_an_lg_modality_names_the_right_families():
    system, params = mission_system(), mission_params()
    bad = dataclasses.replace(params, obs_targets={"look": Categorical(logits=jnp.zeros(2))})
    with pytest.raises(ValueError, match="Gaussian or MixtureGaussian"):
        system.planner_context(bad, dict(bad.initial_state_prior))


def test_a_target_on_the_wrong_space_is_rejected_naming_state_targets():
    """A 1-D observation with a 2-D target is the "I meant the factor" mistake; the
    error says so and names the factor."""
    system, params = mission_system(), mission_params()
    bad = dataclasses.replace(
        params, obs_targets={"look": Gaussian(loc=jnp.zeros(2), chol=jnp.eye(2))}
    )
    with pytest.raises(ValueError, match=r"target dim 2 .*observation dim 1"):
        system.planner_context(bad, dict(bad.initial_state_prior))
    with pytest.raises(ValueError, match="state_targets"):
        system.planner_context(bad, dict(bad.initial_state_prior))
