"""A selection buys information about what it selects (per-variable-infogain.md V9).

A channel's state info-gain leaf is ``I(o; S ∩ D)``: its information about the selected
members ``S`` of its dependency set ``D``, the unselected parents marginalised under the current
belief. Its ambiguity under ``risk_ambiguity`` is ``E H[o | S ∩ D]``, so the decomposition
identity holds under any selection. ``epistemic=True`` (and any selection covering ``D``) keeps
the joint leaf, bit for bit.

Covered here, per family: the chain rule (selected share + remainder = joint leaf; exact on the
discrete path, within the intervals on the switching one), the risk/ambiguity identity under a
partial selection, Monte Carlo oracles for shared and per-mode ``H``/``c``, identical modes
(mode share 0), the consumer brief's uninformative extra mode, bitwise invariance at full
selection, planning through the faces, and the ``sigma_points`` utility.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    Gaussian,
    MixtureGaussian,
)
from hybrid_aif.distributions.gaussian import sigma_points
from hybrid_aif.factors import LinearGaussianObservation
from hybrid_aif.objectives import discrete, total
from hybrid_aif.objectives.continuous import (
    conditional_obs_entropy,
    lg_ambiguity,
    lg_utility,
    predictive_obs,
    risk,
    selected_obs_entropy,
    state_info_gain,
)
from hybrid_aif.system import System, SystemParams
from tests.test_lg_obs_targets import PLANS, mission_params, mission_system
from tests.test_system import CUE, _initial_beliefs, tmaze_system

_LOG_2PI = float(np.log(2.0 * np.pi))


def _leaf(tree, term="infogain", name="modality_0"):
    return tree[term][name]


def _f(x) -> float:
    return float(np.asarray(x))


# ------------------------------------------------------------------ Monte Carlo oracle


def _gauss_logpdf(y, mean, cov):
    """``log N(y; mean, cov)`` for samples ``y (n, d)``; ``mean`` broadcasts against ``y``."""
    chol = np.linalg.cholesky(cov)
    z = np.linalg.solve(chol, (y - mean).T).T
    return -0.5 * (y.shape[-1] * _LOG_2PI + (z**2).sum(-1)) - np.log(np.diag(chol)).sum()


def _logsumexp(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def mc_info(q_w, q_mu, q_cov, mode_probs, H, c, R, selected, n=200_000, seed=0):
    """Monte Carlo ``I(y; S)`` for a switching LG channel, the brief's estimator generalised.

    ``q(x) = Σ_k w_k N(μ_k, Σ_k)``; ``mode_probs`` one probability vector per mode parent;
    ``H``/``c``/``R`` (covariance) on the full mode grid ``(*sizes, ...)``; ``selected`` the mask
    over ``(x, *modes)``. ``I = E[log p(y | S) − log p(y)]`` over joint samples; returns the
    estimate and its standard error.
    """
    rng = np.random.default_rng(seed)
    q_w = q_w / q_w.sum()
    mode_probs = tuple(p / p.sum() for p in mode_probs)
    sizes = tuple(len(p) for p in mode_probs)
    configs = np.array(list(itertools.product(*[range(s) for s in sizes])))
    p_cfg = np.prod([np.asarray(mode_probs[i])[configs[:, i]] for i in range(len(sizes))], 0)
    K, d = q_mu.shape
    k = rng.choice(K, size=n, p=q_w)
    x = q_mu[k] + np.einsum("nij,nj->ni", np.linalg.cholesky(q_cov)[k], rng.standard_normal((n, d)))
    j = rng.choice(len(configs), size=n, p=p_cfg)
    Hg = np.stack([H[tuple(cf)] for cf in configs])
    cg = np.stack([c[tuple(cf)] for cf in configs])
    Rg = np.stack([R[tuple(cf)] for cf in configs])
    noise = np.einsum("nij,nj->ni", np.linalg.cholesky(Rg)[j], rng.standard_normal((n, len(cg[0]))))
    y = np.einsum("nij,nj->ni", Hg[j], x) + cg[j] + noise

    def predictive(ys, members, weights):
        """log of the mixture over q(x)'s components and ``members``, with config weights."""
        return _logsumexp(
            np.stack(
                [
                    np.log(q_w[kk] * w)
                    + _gauss_logpdf(
                        ys, Hg[m] @ q_mu[kk] + cg[m], Hg[m] @ q_cov[kk] @ Hg[m].T + Rg[m]
                    )
                    for kk in range(K)
                    for m, w in zip(members, weights, strict=True)
                ],
                -1,
            ),
            -1,
        )

    log_py = predictive(y, range(len(configs)), p_cfg)
    sel = [i for i, s in enumerate(selected[1:]) if s]
    key = configs[:, sel]  # each configuration's selected sub-configuration
    log_pys = np.empty(n)
    for sub in np.unique(key, axis=0):
        members = np.flatnonzero((key == sub).all(1))
        q_u = p_cfg[members] / p_cfg[members].sum()
        idx = np.isin(j, members)
        if selected[0]:
            terms = [
                np.log(qu) + _gauss_logpdf(y[idx], x[idx] @ Hg[m].T + cg[m], Rg[m])
                for m, qu in zip(members, q_u, strict=True)
            ]
            log_pys[idx] = _logsumexp(np.stack(terms, -1), -1)
        else:
            log_pys[idx] = predictive(y[idx], members, q_u)
    vals = log_pys - log_py
    return float(vals.mean()), float(vals.std() / np.sqrt(n))


# ------------------------------------------------------------------- sigma points


def test_sigma_points_are_exact_for_quadratics_and_cubics():
    """The unscented rule integrates every polynomial of degree ≤ 3 exactly, for any κ > −d."""
    rng = np.random.default_rng(3)
    d = 3
    mu = rng.normal(size=d)
    L = np.tril(rng.normal(size=(d, d))) + 2.0 * np.eye(d)
    S = L @ L.T
    A = rng.normal(size=(d, d))
    b = rng.normal(size=d)
    for kappa in (0.0, 1.0, 2.5):
        pts, wm = sigma_points(jnp.asarray(mu), jnp.asarray(L), kappa=kappa)
        pts, wm = np.asarray(pts, np.float64), np.asarray(wm, np.float64)
        assert pts.shape == (2 * d + 1, d) and wm.shape == (2 * d + 1,)
        np.testing.assert_allclose(wm.sum(), 1.0, atol=1e-6)
        quad = np.einsum("ni,ij,nj->n", pts, A, pts) + pts @ b + 0.7
        np.testing.assert_allclose(
            wm @ quad, np.trace(A @ S) + mu @ A @ mu + b @ mu + 0.7, rtol=1e-5
        )
        # a pure cubic in the deviation integrates to zero (odd central moment)
        dev = pts - mu
        np.testing.assert_allclose(wm @ (dev[:, 0] ** 2 * dev[:, 1]), 0.0, atol=1e-4)
        np.testing.assert_allclose(wm @ pts, mu, atol=1e-5)
        cov = np.einsum("n,ni,nj->ij", wm, dev, dev)
        np.testing.assert_allclose(cov, S, rtol=1e-5, atol=1e-5)


def test_sigma_points_batch_over_leading_axes():
    loc = jnp.zeros((4, 2))
    chol = jnp.broadcast_to(jnp.eye(2), (4, 2, 2))
    pts, wm = sigma_points(loc, chol)
    assert pts.shape == (4, 5, 2)
    assert np.all(np.asarray(wm) > 0)  # the default κ = 1: every weight positive


# --------------------------------------------------------------- the discrete path

N_S = (3, 2, 4)


def _discrete_channel(seed=0):
    rng = np.random.default_rng(seed)
    table = rng.gamma(0.6, size=(5, *N_S)) + 1e-3
    table /= table.sum(0, keepdims=True)
    A = ConditionalCategorical(log_table=jnp.log(jnp.asarray(table, jnp.float32)), parent_ndims=3)
    qs = tuple(Categorical(logits=jnp.asarray(rng.normal(size=n), jnp.float32)) for n in N_S)
    return A, qs, table


def _np_mi(joint, a_axes, b_axes):
    """``I(A; B | C)`` from a joint probability array; C = every other axis."""
    axes = set(range(joint.ndim))
    c_axes = axes - set(a_axes) - set(b_axes)

    def h(keep):
        drop = tuple(sorted(axes - set(keep)))
        p = joint.sum(axis=drop) if drop else joint
        p = p[p > 0]
        return -(p * np.log(p)).sum()

    return h(set(a_axes) | c_axes) + h(set(b_axes) | c_axes) - h(axes) - h(c_axes)


@pytest.mark.parametrize("mask", [(True, False, False), (False, True, True), (True, False, True)])
def test_discrete_selected_share_plus_remainder_is_the_joint_leaf(mask):
    """Chain rule, exact: ``I(o; D) = I(o; S) + I(o; D∖S | S)``, both pieces checked against a
    brute-force joint ``p(o, s) = A(o|s) Π q(s_f)``."""
    A, qs, table = _discrete_channel()
    deps = ((0, 1, 2),)
    joint_leaf = _leaf(discrete.state_info_gain(qs, (A,), deps))
    share = _leaf(discrete.state_info_gain(qs, (A,), deps, selected=(mask,)))
    q = [np.asarray(b.probs, np.float64) for b in qs]
    joint = table * np.einsum("i,j,k->ijk", *q)[None]
    sel = [1 + i for i, k in enumerate(mask) if k]
    uns = [1 + i for i, k in enumerate(mask) if not k]
    want_share = _np_mi(joint.sum(axis=tuple(uns)), [0], list(range(1, 1 + len(sel))))
    remainder = _np_mi(joint, [0], uns)
    np.testing.assert_allclose(_f(share.lower), want_share, atol=2e-5)
    np.testing.assert_allclose(_f(share.lower) + remainder, _f(joint_leaf.lower), atol=2e-5)
    assert _f(share.width) == 0.0
    # a mask covering the whole set is the joint leaf itself
    full = _leaf(discrete.state_info_gain(qs, (A,), deps, selected=((True,) * 3,)))
    assert _f(full.lower) == _f(joint_leaf.lower)


def test_discrete_identity_under_a_partial_selection_leafwise():
    """``risk + ambiguity = −(utility + infogain)`` with both sides relative to the same mask."""
    A, qs, _ = _discrete_channel(seed=1)
    deps = ((0, 1, 2),)
    mask = ((False, True, False),)
    target = {"modality_0": Categorical(logits=jnp.asarray([0.3, -1.0, 0.2, 0.0, 0.5]))}
    q_obs = discrete.expected_obs(qs, (A,), deps)
    u = _leaf(discrete.utility(q_obs, target), "utility")
    r = _leaf(discrete.risk(q_obs, target, ("modality_0",)), "risk")
    ig = _leaf(discrete.state_info_gain(qs, (A,), deps, selected=mask))
    amb = _leaf(discrete.ambiguity(qs, (A,), deps, selected=mask), "ambiguity")
    np.testing.assert_allclose(_f(r.lower + amb.lower), -_f(u.lower + ig.lower), atol=1e-5)
    # and the ambiguity really is the selected one, not the full table's
    amb_full = _leaf(discrete.ambiguity(qs, (A,), deps), "ambiguity")
    assert _f(amb.lower) > _f(amb_full.lower)


def _two_channel_discrete(decomposition, epistemic):
    """Two factors, two channels that both observe both: every mounted channel is partial
    under ``("f1",)``, so the decompositions agree on G (no unmounted wedge)."""
    keys = iter(jax.random.split(jax.random.PRNGKey(5), 12))
    ns, no = (3, 2), (4, 3)

    def table(key, shape):
        x = jax.random.uniform(key, shape, minval=0.05, maxval=1.0) ** 3
        return jnp.log(x / x.sum(0, keepdims=True))

    A = tuple(
        ConditionalCategorical(log_table=table(next(keys), (n, *ns)), parent_ndims=2) for n in no
    )
    B = (
        ConditionalCategorical(log_table=table(next(keys), (3, 3, 2)), parent_ndims=2),
        ConditionalCategorical(log_table=table(next(keys), (2, 2, 1)), parent_ndims=2),
    )
    system = System(
        A_deps=((0, 1), (0, 1)),
        B_deps=((0,), (1,)),
        controlled=(True, False),
        policy_len=2,
        decomposition=decomposition,
        epistemic=epistemic,
        modality_names=("m0", "m1"),
        factor_names=("f0", "f1"),
    )
    params = system.from_positional(
        "params",
        SystemParams(
            likelihood=A,
            transition=B,
            initial_state_prior=tuple(Categorical(logits=jnp.zeros(n)) for n in ns),
            obs_targets={
                "m0": Categorical(logits=jax.random.normal(next(keys), (4,))),
                "m1": Categorical(logits=jax.random.normal(next(keys), (3,))),
            },
        ),
    )
    beliefs = tuple(Categorical(logits=jax.random.normal(next(keys), (n,))) for n in ns)
    return system, params, system.from_positional("beliefs", beliefs)


@pytest.mark.parametrize("epistemic", [("f1",), ("f0",)])
def test_discrete_decompositions_agree_on_g_under_a_partial_selection(epistemic):
    """System level, compiled path: G from ``utility + infogain`` equals G from
    ``risk + ambiguity`` when each channel is scored relative to the selection."""
    g = {}
    for decomposition in ("utility_infogain", "risk_ambiguity"):
        system, params, beliefs = _two_channel_discrete(decomposition, epistemic)
        tree = system.score_policies(params, beliefs, system.policies(params))
        g[decomposition] = total(tree)
    np.testing.assert_allclose(
        np.asarray(g["utility_infogain"].lower), np.asarray(g["risk_ambiguity"].lower), atol=1e-5
    )
    # and the partial selection really changed the leaf
    system, params, beliefs = _two_channel_discrete("utility_infogain", True)
    full = system.score_policies(params, beliefs, system.policies(params))["infogain"]["m0"]
    system, params, beliefs = _two_channel_discrete("utility_infogain", epistemic)
    part = system.score_policies(params, beliefs, system.policies(params))["infogain"]["m0"]
    assert np.all(np.asarray(part.lower) < np.asarray(full.lower))


def test_discrete_planning_under_a_partial_selection_matches_across_paths():
    """infer_policies (compiled, metrics on and off) under a partial selection: finite,
    and the fused G-fold agrees with the tree path."""
    system, params, beliefs = _two_channel_discrete("utility_infogain", ("f1",))
    q1, tree = system.infer_policies(params, beliefs)
    q2, none = system.infer_policies(params, beliefs, metrics=False)
    assert none is None and set(tree["infogain"]) == {"m0", "m1"}
    assert np.all(np.isfinite(np.asarray(q1.logits)))
    np.testing.assert_allclose(np.asarray(q1.logits), np.asarray(q2.logits), atol=1e-5)


def test_selecting_the_observation_variable_buys_the_joint():
    """``obs:<modality>`` selects what the observation reveals: the channel's joint leaf,
    whatever its parents' selection."""
    _, params = tmaze_system()
    trees = {}
    for sel in (True, ("obs:cue",), ("location", "obs:cue")):
        system = tmaze_system(epistemic=sel)[0]
        beliefs = _initial_beliefs(system, params)
        trees[sel] = system.infer_policies(params, beliefs)[1]["infogain"]["cue"]
    for sel in (("obs:cue",), ("location", "obs:cue")):
        np.testing.assert_array_equal(np.asarray(trees[sel].lower), np.asarray(trees[True].lower))


def test_tmaze_location_selection_drops_the_cue_drive():
    """The T-maze's cue channel observes (location, context): selecting ``location`` alone
    scores its location information only, which the agent's own move already fixes."""
    _, params = tmaze_system()
    system = tmaze_system(epistemic=("location",))[0]
    beliefs = _initial_beliefs(system, params)
    q_pi, tree = system.infer_policies(params, beliefs)
    assert int(system.policies(params)[jnp.argmax(q_pi.probs)][0, 0]) != CUE
    full = tmaze_system(epistemic=True)[0].infer_policies(params, beliefs)[1]
    assert np.all(
        np.asarray(tree["infogain"]["cue"].upper)
        <= np.asarray(full["infogain"]["cue"].lower) + 1e-6
    )


# ------------------------------------------------------------- the switching LG path

#: the consumer brief's toy (its §3): a 2-D block, sd 30 mm; answer noise 22 mm (free) and
#: an uninformative held mode at 3 %
D, SD, R_FREE, W_HELD = 2, 0.030, 0.022, 0.03
Q_BLOCK = Gaussian(loc=jnp.zeros(D), chol=SD * jnp.eye(D))
GRIP = Categorical(logits=jnp.log(jnp.asarray([1 - W_HELD, W_HELD])))
#: the brief's §3 Monte Carlo truth for I(y; block) at held noise 0.15 / 0.3 / 10 m
BRIEF_TRUTH = {0.15: 0.993, 0.3: 1.007, 10.0: 1.018}
BRIEF_NO_EXTRA = 1.050  # the same with the held mode identical to the free one


def _brief_model(r_held, per_mode_head=False):
    noise = jnp.stack([R_FREE * jnp.eye(D), r_held * jnp.eye(D)])
    if per_mode_head:
        return LinearGaussianObservation(
            H=jnp.broadcast_to(jnp.eye(D), (2, D, D)), c=jnp.zeros((2, D)), noise=noise
        )
    return LinearGaussianObservation(H=jnp.eye(D), c=jnp.zeros(D), noise=noise)


def _ig(q, model, modes, mask=None):
    return _leaf(state_info_gain((q,), (model,), mode_beliefs=(modes,), selected=(mask,)))


@pytest.mark.parametrize("r_held", sorted(BRIEF_TRUTH))
@pytest.mark.parametrize("per_mode_head", [False, True])
def test_brief_uninformative_extra_mode_does_not_raise_the_block_share(r_held, per_mode_head):
    """The brief's case: the block share's midpoint stays at or below the no-extra-mode value
    and within 0.03 nats of the brief's Monte Carlo truth; the joint leaf rises above it. The
    brief's own per-mode-shaped declaration (equal values) gives the shared-head leaf."""
    model = _brief_model(r_held, per_mode_head)
    share = _ig(Q_BLOCK, model, (GRIP,), (True, False))
    joint = _ig(Q_BLOCK, model, (GRIP,))
    no_extra = _ig(Q_BLOCK, _brief_model(R_FREE), (GRIP,), (True, False))
    np.testing.assert_allclose(_f(no_extra.midpoint), BRIEF_NO_EXTRA, atol=2e-3)
    assert _f(share.lower) <= BRIEF_TRUTH[r_held] <= _f(share.upper)
    assert _f(joint.midpoint) > _f(no_extra.midpoint)
    assert _f(share.midpoint) <= _f(no_extra.midpoint)
    assert abs(_f(share.midpoint) - BRIEF_TRUTH[r_held]) < 0.03
    shared = _ig(Q_BLOCK, _brief_model(r_held, per_mode_head=False), (GRIP,), (True, False))
    np.testing.assert_allclose(_f(share.lower), _f(shared.lower), atol=1e-5)
    np.testing.assert_allclose(_f(share.upper), _f(shared.upper), atol=1e-5)
    # the mode remainder I(y; mode | x) lies in [0, H(q(mode))]
    h_mode = -float(np.sum(np.asarray(GRIP.probs) * np.log(np.asarray(GRIP.probs))))
    rem = selected_obs_entropy(Q_BLOCK, model, (GRIP,), (True, False)) - conditional_obs_entropy(
        model, (GRIP,)
    )
    assert -1e-6 <= _f(rem.lower) <= _f(rem.upper) <= h_mode + 1e-6


#: the reviewer's counterexample: two equiprobable modes H = ±1, c = 0, R = 0.01, q(x) =
#: N(0, σ²), the factor selected. ``σ -> (Monte Carlo I(y; x) at 200k samples, the
#: midpoint error measured at build time)``; the bracket's midpoint is pinned at error + 0.03.
SIGN_FLIP = {0.3: (0.653, 0.152), 1.0: (1.677, 0.284), 3.0: (2.731, 0.324)}


#: the block-share midpoints the brief's §3 declaration must reproduce (held 0.15 / 0.3 / 10 m)
BRIEF_SHARE_MIDPOINTS = {0.15: 1.010, 0.3: 1.011, 10.0: 1.018}


def test_the_brief_script_declaration_verbatim():
    """The brief's §3 script builds its channel per-mode shaped with equal values
    (``H = broadcast_to(eye, (2, d, d))``, ``c = zeros((2, d))``); its block share is the
    shared-head leaf, every midpoint below the no-extra-mode 1.051."""
    d, s, r, w_held = 2, 0.030, 0.022, 0.03
    q = Gaussian(loc=jnp.zeros(d), chol=s * jnp.eye(d))
    grip = Categorical(logits=jnp.log(jnp.array([1 - w_held, w_held])))
    for R, want in BRIEF_SHARE_MIDPOINTS.items():
        H = jnp.broadcast_to(jnp.eye(d), (2, d, d))
        c = jnp.zeros((2, d))
        m = LinearGaussianObservation(H=H, c=c, noise=jnp.stack([r * jnp.eye(d), R * jnp.eye(d)]))
        ig = state_info_gain((q,), (m,), mode_beliefs=((grip,),), selected=((True, False),))
        leaf = ig["infogain"]["modality_0"]
        np.testing.assert_allclose(_f(leaf.midpoint), want, atol=2e-3)
        assert _f(leaf.midpoint) < BRIEF_NO_EXTRA


def test_a_near_equal_per_mode_head_takes_the_bracket():
    """Equality is exact: a head perturbed by 1e-6 across the unselected mode is treated as
    varying, so it reports the guaranteed bracket (here [H[y|x,mode], ... + H(q(mode))])."""
    noise = jnp.stack([R_FREE * jnp.eye(D), 0.3 * jnp.eye(D)])
    H = jnp.stack([jnp.eye(D), jnp.eye(D) + 1e-6])
    model = LinearGaussianObservation(H=H, c=jnp.zeros((2, D)), noise=noise)
    h = selected_obs_entropy(Q_BLOCK, model, (GRIP,), (True, False))
    floor = _f(conditional_obs_entropy(model, (GRIP,)))
    np.testing.assert_allclose(_f(h.lower), floor, atol=1e-6)
    exact = selected_obs_entropy(Q_BLOCK, _brief_model(0.3, True), (GRIP,), (True, False))
    assert _f(exact.lower) > floor + 1e-3  # the equal head's KT pair sits above the floor


@pytest.mark.parametrize("sd", sorted(SIGN_FLIP))
def test_a_per_mode_head_takes_the_guaranteed_bracket(sd):
    """``H[y | x]`` depends on ``x`` when the unselected mode flips the gain: the leaf is the
    guaranteed bracket, which holds the truth."""
    model = LinearGaussianObservation(
        H=jnp.asarray([[[1.0]], [[-1.0]]]),
        c=jnp.zeros((2, 1)),
        noise=jnp.full((2, 1, 1), 0.1),
    )
    mode = Categorical(logits=jnp.zeros(2))
    q = Gaussian(loc=jnp.zeros(1), chol=jnp.asarray([[sd]]))
    share = _ig(q, model, (mode,), (True, False))
    truth, err = SIGN_FLIP[sd]
    assert _f(share.lower) <= truth <= _f(share.upper)
    assert abs(_f(share.midpoint) - truth) < err + 0.03
    # the conditional entropy is exactly the bracket [H[y|x,mode], H[y|x,mode] + ln 2]
    h = selected_obs_entropy(q, model, (mode,), (True, False))
    floor = _f(conditional_obs_entropy(model, (mode,)))
    np.testing.assert_allclose(_f(h.lower), floor, atol=1e-6)
    assert _f(h.upper) <= floor + np.log(2.0) + 1e-6


def test_identical_modes_give_a_zero_mode_share():
    """Modes with identical ``(H, c, R)``: the block share equals the non-switching leaf and
    the modes-only share is zero at its midpoint."""
    H = jnp.asarray([[1.0, 0.3], [0.0, 0.8]])
    c = jnp.asarray([0.1, -0.2])
    R = jnp.asarray([[0.2, 0.0], [0.05, 0.15]])
    q = MixtureGaussian(
        log_weights=jnp.log(jnp.asarray([0.6, 0.4])),
        means=jnp.asarray([[0.0, 0.0], [1.0, -0.5]]),
        chols=jnp.stack([0.4 * jnp.eye(2), 0.3 * jnp.eye(2)]),
    )
    plain = _leaf(state_info_gain((q,), (LinearGaussianObservation(H=H, c=c, noise=R),)))
    mode = Categorical(logits=jnp.log(jnp.asarray([0.3, 0.7])))
    for per_mode in (False, True):
        model = LinearGaussianObservation(
            H=jnp.stack([H, H]) if per_mode else H,
            c=jnp.stack([c, c]) if per_mode else c,
            noise=jnp.stack([R, R]),
        )
        block = _ig(q, model, (mode,), (True, False))
        np.testing.assert_allclose(_f(block.lower), _f(plain.lower), atol=1e-5)
        np.testing.assert_allclose(_f(block.upper), _f(plain.upper), atol=1e-5)
        # the same KT pair on both sides of the difference: the midpoint is exactly zero, the
        # width the subtraction's (V9 decision 5)
        modes_only = _ig(q, model, (mode,), (False, True))
        np.testing.assert_allclose(_f(modes_only.midpoint), 0.0, atol=1e-5)
        assert _f(modes_only.lower) <= 0.0 <= _f(modes_only.upper)


#: a per-mode geometry: mode 1 has a different gain, an offset and looser noise
Q_MIX = MixtureGaussian(
    log_weights=jnp.log(jnp.asarray([0.7, 0.3])),
    means=jnp.asarray([[0.4, -0.2], [-0.5, 0.6]]),
    chols=jnp.stack([jnp.asarray([[0.3, 0.0], [0.1, 0.25]]), 0.5 * jnp.eye(2)]),
)
H0 = np.asarray([[1.0, 0.0], [0.5, 1.0]])
C0 = np.asarray([0.2, -0.1])
R0 = np.asarray([[0.15, 0.0], [0.05, 0.2]])
#: two mode parents (2 × 3): per-mode gain on the first, per-mode offset and noise on both
SIZES = (2, 3)
H_GRID = np.stack([H0, 0.6 * H0])[:, None].repeat(3, 1)
C_GRID = np.asarray([[[0.0, 0.0], [0.6, -0.3], [-0.4, 0.5]]] * 2) + np.asarray(
    [[[0.0, 0.0]], [[0.3, 0.3]]]
)
R_GRID = np.stack([[R0 * s for s in (1.0, 2.0, 0.6)], [R0 * s for s in (1.5, 0.8, 3.0)]])
MODES2 = (
    Categorical(logits=jnp.log(jnp.asarray([0.65, 0.35]))),
    Categorical(logits=jnp.log(jnp.asarray([0.5, 0.3, 0.2]))),
)
SHARED_HEAD = LinearGaussianObservation(
    H=jnp.asarray(H0), c=jnp.asarray(C0), noise=jnp.asarray(R_GRID, jnp.float32)
)
PER_MODE = LinearGaussianObservation(
    H=jnp.asarray(H_GRID, jnp.float32),
    c=jnp.asarray(C_GRID, jnp.float32),
    noise=jnp.asarray(R_GRID, jnp.float32),
)


def _grids(model):
    H = np.broadcast_to(np.asarray(model.H, np.float64), SIZES + (2, 2))
    c = np.broadcast_to(np.asarray(model.c, np.float64), SIZES + (2,))
    Rc = np.broadcast_to(np.asarray(model.R_chol, np.float64), SIZES + (2, 2))
    return H, c, Rc @ np.swapaxes(Rc, -1, -2)


def _mc(model, mask, seed=0, n=120_000):
    H, c, R = _grids(model)
    return mc_info(
        np.asarray(jax.nn.softmax(Q_MIX.log_weights), np.float64),
        np.asarray(Q_MIX.means, np.float64),
        np.asarray(Q_MIX.components.cov, np.float64),
        tuple(np.asarray(b.probs, np.float64) for b in MODES2),
        H,
        c,
        R,
        mask,
        n=n,
        seed=seed,
    )


MASKS = [
    (True, False, False),  # the factor alone
    (True, True, False),  # the factor and one mode (the mixed selection)
    (False, True, False),  # one mode alone
    (False, True, True),  # the modes alone
]

#: the midpoint error vs Monte Carlo, pinned (nats): measured at build time (300k samples)
#: plus 0.02 or more. On the shared head and the modes-only masks most of it is the predictive
#: entropy's KT pair, shared with the joint leaf (whose own error is +0.04 shared, +0.13
#: per-mode); with the factor selected on a per-mode head the leaf is the guaranteed bracket
#: and its midpoint sits well above the truth.
MIDPOINT_PIN = {
    ("shared_head", MASKS[0]): 0.04,  # measured +0.018
    ("shared_head", MASKS[1]): 0.05,  # +0.026
    ("shared_head", MASKS[2]): 0.02,  # +0.001
    ("shared_head", MASKS[3]): 0.05,  # +0.027
    ("per_mode", MASKS[0]): 0.58,  # +0.551, the bracket
    ("per_mode", MASKS[1]): 0.52,  # +0.496, the bracket
    ("per_mode", MASKS[2]): 0.06,  # +0.033
    ("per_mode", MASKS[3]): 0.13,  # +0.110
}


@pytest.mark.parametrize("mask", MASKS)
@pytest.mark.parametrize("model_name", ["shared_head", "per_mode"])
def test_switching_share_against_monte_carlo(model_name, mask):
    """The Monte Carlo ``I(y; S)`` lies inside the share's interval (within 4 standard errors),
    and the midpoint error stays within its pin."""
    model = SHARED_HEAD if model_name == "shared_head" else PER_MODE
    share = _ig(Q_MIX, model, MODES2, mask)
    truth, se = _mc(model, mask)
    assert _f(share.lower) - 4 * se <= truth <= _f(share.upper) + 4 * se, (share, truth)
    assert abs(_f(share.midpoint) - truth) < MIDPOINT_PIN[model_name, mask]


@pytest.mark.parametrize("mask", MASKS)
@pytest.mark.parametrize("model_name", ["shared_head", "per_mode"])
def test_switching_share_plus_remainder_contains_the_joint(model_name, mask):
    """Chain rule within the intervals: share ``I(y; S)`` plus the remainder
    ``I(y; D∖S | S) = E H[y|S] − E H[y|D]`` (computed here from the two conditional entropies)
    encloses the joint leaf."""
    model = SHARED_HEAD if model_name == "shared_head" else PER_MODE
    share = _ig(Q_MIX, model, MODES2, mask)
    joint = _ig(Q_MIX, model, MODES2)
    remainder = selected_obs_entropy(Q_MIX, model, MODES2, mask) - conditional_obs_entropy(
        model, MODES2
    )
    total_iv = share + remainder
    assert _f(total_iv.lower) <= _f(joint.lower) + 1e-5
    assert _f(joint.upper) <= _f(total_iv.upper) + 1e-5
    assert _f(remainder.lower) >= -1e-5  # conditioning on more never adds entropy


def test_switching_modes_then_factor_order_against_an_exact_remainder():
    """The other chain-rule order, with an exact remainder: for a Gaussian ``q(x)``,
    ``I(y; x | mode) = Σ_j q_j ½ log(|H_j Σ H_jᵀ + R_j| / |R_j|)``. The modes-only share plus
    that remainder overlaps the joint leaf."""
    q = Gaussian(loc=jnp.asarray([0.3, -0.1]), chol=jnp.asarray([[0.4, 0.0], [0.1, 0.3]]))
    S = np.asarray(q.cov, np.float64)
    H, _, R = _grids(PER_MODE)
    p = np.einsum("i,j->ij", *(np.asarray(b.probs, np.float64) for b in MODES2))
    rem = sum(
        p[i, j]
        * 0.5
        * (np.linalg.slogdet(H[i, j] @ S @ H[i, j].T + R[i, j])[1] - np.linalg.slogdet(R[i, j])[1])
        for i in range(2)
        for j in range(3)
    )
    share = _ig(q, PER_MODE, MODES2, (False, True, True))
    joint = _ig(q, PER_MODE, MODES2)
    assert _f(share.lower) + rem <= _f(joint.upper) + 1e-5
    assert _f(joint.lower) <= _f(share.upper) + rem + 1e-5


def test_selected_obs_entropy_brackets_hold():
    """The factor-selected conditional entropy sits in its guaranteed bracket:
    ``E H[y|x,mode] ≤ · ≤ E H[y|x,mode] + H(q(unselected modes))``."""
    for model in (SHARED_HEAD, PER_MODE):
        h = selected_obs_entropy(Q_MIX, model, MODES2, (True, False, False))
        floor = _f(conditional_obs_entropy(model, MODES2))
        p = np.einsum("i,j->ij", *(np.asarray(b.probs, np.float64) for b in MODES2)).ravel()
        ceiling = floor - float(np.sum(p * np.log(p)))
        assert floor - 1e-5 <= _f(h.lower) <= _f(h.upper) <= ceiling + 1e-5


def test_switching_identity_under_a_partial_selection_leafwise():
    """``risk + ambiguity = −(utility + infogain)`` at both endpoints with the same mask."""
    target = Gaussian(loc=jnp.zeros(2), chol=jnp.diag(jnp.asarray([0.5, 0.4])))
    for model in (SHARED_HEAD, PER_MODE):
        q_obs = predictive_obs(Q_MIX, model, MODES2)
        for mask in MASKS:
            u = _leaf(lg_utility((q_obs,), {"look": target}, ("look",)), "utility", "look")
            r = _leaf(risk((q_obs,), {"look": target}, ("look",)), "risk", "look")
            ig = _leaf(
                state_info_gain((Q_MIX,), (model,), ("look",), (MODES2,), (mask,)), name="look"
            )
            amb = _leaf(
                lg_ambiguity((model,), (MODES2,), ("look",), (Q_MIX,), (mask,)), "ambiguity", "look"
            )
            lhs, rhs = r + amb, -(u + ig)
            np.testing.assert_allclose(_f(lhs.lower), _f(rhs.lower), atol=1e-5)
            np.testing.assert_allclose(_f(lhs.upper), _f(rhs.upper), atol=1e-5)


def _mission_beliefs(params):
    """The mission model's declared initial priors, keyed by factor."""
    return dict(params.initial_state_prior)


@pytest.mark.parametrize("epistemic", [("pose",), ("commit",)])
def test_switching_decompositions_agree_on_g_under_a_partial_selection(epistemic):
    """System level, hybrid path (the friction-6 mission: one switching channel on
    ``(pose, commit)``, per-mode offsets): G agrees across decompositions at both endpoints."""
    totals = {}
    for decomposition in ("utility_infogain", "risk_ambiguity"):
        system = mission_system(epistemic=epistemic, decomposition=decomposition)
        params = mission_params()
        tree = system.score_policies(params, _mission_beliefs(params), PLANS)
        totals[decomposition] = total(tree)
    a, b = totals["utility_infogain"], totals["risk_ambiguity"]
    np.testing.assert_allclose(np.asarray(a.lower), np.asarray(b.lower), atol=1e-4)
    np.testing.assert_allclose(np.asarray(a.upper), np.asarray(b.upper), atol=1e-4)


def test_switching_planning_under_a_partial_selection():
    """score_policies and infer_policies (metrics on and off) under a partial selection on a
    switching model: finite, channel-keyed, and different from the joint leaf."""
    params = mission_params()
    beliefs = _mission_beliefs(params)
    joint = mission_system(epistemic=True).score_policies(params, beliefs, PLANS)
    for sel in (("pose",), ("commit",)):
        system = mission_system(epistemic=sel)
        tree = system.score_policies(params, beliefs, PLANS)
        leaf = tree["infogain"]["look"]
        assert np.all(np.isfinite(np.asarray(leaf.lower))) and np.all(
            np.isfinite(np.asarray(leaf.upper))
        )
        assert np.all(np.asarray(leaf.lower) <= np.asarray(leaf.upper) + 1e-6)
        assert not np.allclose(np.asarray(leaf.lower), np.asarray(joint["infogain"]["look"].lower))
        q1, _ = system.infer_policies(params, beliefs)
        q2, none = system.infer_policies(params, beliefs, metrics=False)
        assert none is None
        np.testing.assert_allclose(np.asarray(q1.logits), np.asarray(q2.logits), atol=1e-4)


# --------------------------------------------------------- full selection is bitwise


def _assert_trees_equal(a, b):
    assert {t: set(v) for t, v in a.items()} == {t: set(v) for t, v in b.items()}
    for term, blocks in a.items():
        for name, iv in blocks.items():
            np.testing.assert_array_equal(np.asarray(iv.lower), np.asarray(b[term][name].lower))
            np.testing.assert_array_equal(np.asarray(iv.upper), np.asarray(b[term][name].upper))


@pytest.mark.parametrize("decomposition", ["utility_infogain", "risk_ambiguity"])
def test_a_selection_covering_every_parent_is_bitwise_the_joint(decomposition):
    """``epistemic=True`` and an explicit tuple covering every parent take the same code path:
    identical trees, bit for bit, on the switching and the discrete path."""
    params = mission_params()
    beliefs = _mission_beliefs(params)
    a = mission_system(epistemic=True, decomposition=decomposition).score_policies(
        params, beliefs, PLANS
    )
    b = mission_system(epistemic=("pose", "commit"), decomposition=decomposition).score_policies(
        params, beliefs, PLANS
    )
    _assert_trees_equal(a, b)
    s1, p1, bl = _two_channel_discrete(decomposition, True)
    s2, _, _ = _two_channel_discrete(decomposition, ("f0", "f1"))
    _assert_trees_equal(
        s1.score_policies(p1, bl, s1.policies(p1)), s2.score_policies(p1, bl, s2.policies(p1))
    )


def test_leaf_functions_with_a_full_mask_are_bitwise_the_default():
    q_obs_model = PER_MODE
    a = _ig(Q_MIX, q_obs_model, MODES2)
    b = _ig(Q_MIX, q_obs_model, MODES2, (True, True, True))
    assert _f(a.lower) == _f(b.lower) and _f(a.upper) == _f(b.upper)
    A, qs, _ = _discrete_channel()
    deps = ((0, 1, 2),)
    x = _leaf(discrete.ambiguity(qs, (A,), deps), "ambiguity")
    y = _leaf(discrete.ambiguity(qs, (A,), deps, selected=((True,) * 3,)), "ambiguity")
    assert _f(x.lower) == _f(y.lower)


def test_channel_selections_resolve_masks():
    """The selector's per-channel masks: ``None`` for full, unmounted, or ``obs:``-selected
    channels; a mask in ``A_deps`` order otherwise."""
    params = mission_params()
    assert mission_system(epistemic=True)._channel_selections(params) == (None,)
    assert mission_system(epistemic=("pose",))._channel_selections(params) == ((True, False),)
    assert mission_system(epistemic=("commit",))._channel_selections(params) == ((False, True),)
    assert mission_system(epistemic=("*",))._channel_selections(params) == (None,)
    assert mission_system(epistemic=("pose", "obs:look"))._channel_selections(params) == (None,)


# ------------------------------------------------- review hardening: bracket, spread, paths


def test_the_max_entropy_ceiling_binds_on_a_shared_head():
    """1-D, shared head, noise variances 1 and 1.5 at 50/50, factor selected: the KT upper
    bound of the noise mixture exceeds ``½ log 2πe (½·1 + ½·1.5)``, so the guaranteed bracket's
    ceiling is what the leaf reports."""
    model = LinearGaussianObservation(
        H=jnp.eye(1), c=jnp.zeros(1), noise=jnp.sqrt(jnp.asarray([[[1.0]], [[1.5]]]))
    )
    mode = Categorical(logits=jnp.zeros(2))
    q = Gaussian(loc=jnp.zeros(1), chol=jnp.eye(1))
    h = selected_obs_entropy(q, model, (mode,), (True, False))
    ceiling = 0.5 * np.log(2 * np.pi * np.e * 1.25)
    np.testing.assert_allclose(_f(h.upper), ceiling, atol=1e-5)  # 1.5305; unclipped KT: 1.5409
    floor = 0.5 * (0.5 * np.log(2 * np.pi * np.e) + 0.5 * np.log(2 * np.pi * np.e * 1.5))
    assert floor - 1e-6 <= _f(h.lower) <= _f(h.upper)


def test_expected_mode_spread_matches_a_direct_computation():
    """``E_q(x)[Cov_{q(u)}(H_u x + c_u)]`` on a mixture ``q(x)``, per-mode ``H``/``c``, and a
    mixed selection (the first mode parent selected, the second marginalised): against the
    covariance over ``u`` computed at Monte Carlo draws of ``x`` and averaged."""
    from hybrid_aif.objectives.continuous import _expected_mode_spread

    q_u = np.asarray(MODES2[1].probs, np.float64)
    # gain and offset both vary over the marginalised mode, and differ per selected mode
    gains = np.stack([[H0 * g for g in (1.0, 0.4, -0.7)], [H0.T * g for g in (0.5, 1.2, 0.0)]])
    offsets = C_GRID
    head = np.concatenate([gains, offsets[..., None]], axis=-1)  # (P=2, U=3, d_obs, d+1)
    got = np.asarray(
        _expected_mode_spread(jnp.asarray(head, jnp.float32), jnp.asarray(q_u, jnp.float32), Q_MIX)
    )
    rng = np.random.default_rng(0)
    w = np.asarray(jax.nn.softmax(Q_MIX.log_weights), np.float64)
    w /= w.sum()
    n = 200_000
    k = rng.choice(2, size=n, p=w)
    chols = np.asarray(Q_MIX.chols, np.float64)
    x = np.asarray(Q_MIX.means, np.float64)[k] + np.einsum(
        "nij,nj->ni", chols[k], rng.standard_normal((n, 2))
    )
    for p in range(2):
        means = np.einsum("uod,nd->nuo", gains[p], x) + offsets[p][None]  # (n, U, d_obs)
        centred = means - np.einsum("u,nuo->no", q_u, means)[:, None]
        want = np.einsum("u,nuo,nup->op", q_u, centred, centred) / n
        np.testing.assert_allclose(got[p], want, rtol=2e-2, atol=2e-3)


def _count_selected_calls(monkeypatch):
    """Wrap every selection-relative entry point with a call counter."""
    from hybrid_aif.objectives import compiled, continuous

    calls = {"n": 0}

    def counting(fn):
        def wrapped(*args, **kwargs):
            calls["n"] += 1
            return fn(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(
        continuous, "selected_obs_entropy", counting(continuous.selected_obs_entropy)
    )
    monkeypatch.setattr(
        discrete, "selected_conditional_entropy", counting(discrete.selected_conditional_entropy)
    )
    monkeypatch.setattr(
        compiled, "selected_conditional_entropy", counting(compiled.selected_conditional_entropy)
    )
    return calls


@pytest.mark.parametrize("decomposition", ["utility_infogain", "risk_ambiguity"])
def test_the_default_selection_never_reaches_the_selected_estimators(monkeypatch, decomposition):
    """Structural: under ``epistemic=True`` planning on the hybrid (switching) and compiled
    (tabular) paths calls no selection-relative estimator; a partial selection does."""
    calls = _count_selected_calls(monkeypatch)
    # the mission model plus a tabular channel on (commit, pose-free) so the hybrid path also
    # scores a discrete leaf: "flag" reads the commitment through a noisy table
    flag = ConditionalCategorical(
        log_table=jnp.log(jnp.asarray([[0.8, 0.3], [0.2, 0.7]])), parent_ndims=1
    )
    params = mission_params(likelihood={"look": mission_params().likelihood["look"], "flag": flag})
    beliefs = _mission_beliefs(params)

    def plan_all(epistemic_lg, epistemic_tab):
        with jax.disable_jit():
            mission_system(
                epistemic=epistemic_lg,
                decomposition=decomposition,
                A_deps=((1, 0), (0,)),
                modality_names=("look", "flag"),
            ).infer_policies(params, beliefs)
            system, p, b = _two_channel_discrete(decomposition, epistemic_tab)
            system.infer_policies(p, b)

    plan_all(True, True)
    assert calls["n"] == 0
    plan_all(("pose",), ("f1",))  # positive control: the counters are on the live names
    assert calls["n"] > 0


def test_lg_ambiguity_with_a_partial_mask_needs_the_factor_belief():
    with pytest.raises(ValueError, match="q_blocks"):
        lg_ambiguity((PER_MODE,), (MODES2,), ("look",), selected=((True, False, False),))


def test_an_empty_mask_raises_on_both_families():
    with pytest.raises(ValueError, match="selects none"):
        selected_obs_entropy(Q_MIX, PER_MODE, MODES2, (False, False, False))
    A, qs, _ = _discrete_channel()
    with pytest.raises(ValueError, match="selects none"):
        discrete.state_info_gain(qs, (A,), ((0, 1, 2),), selected=((False,) * 3,))
