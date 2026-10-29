"""Each decomposition ablates its own epistemic leaf (design: per-variable-infogain.md V11).

The ``epistemic`` selector mounts ``infogain`` under ``utility_infogain`` and ``ambiguity`` under
``risk_ambiguity``; risk is never touched by it, and a channel without a designation carries an
implicit flat preference, its risk ``−H[q(o)]``. So under full selection the two forms give the
same G at both interval ends, per channel under any selection, and under ablation they differ by
the ablated channels' predictive entropy: ``Σ H[q(o)]`` where they share no factor, ``H[q(Y)] −
H[q(Y_S)]`` on a shared Gaussian one. On a Gaussian-carried factor the information gain is per
channel, ``I(y_m; x)``; ``observation_interaction`` holds the interaction information of the
selected readings about the selected state (subtracted from their sum), and under
``risk_ambiguity`` ``observation_correlation`` holds the readings' total correlation, part of the
joint risk, with ``observation_interaction`` the joint ambiguity's ``−TC(Y | x_S)`` on a
sub-block. A Gaussian emission over categorical parents follows the same rule. The chain-less
entropy leaf is now the opt-in ``uncertainty`` row, the opposite sign.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import xlogy
from scipy.stats import multivariate_normal

from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalGaussian,
    Gaussian,
    MixtureGaussian,
)
from hybrid_aif.factors import (
    LinearGaussianObservation,
    LinearGaussianTransition,
    gaussian_emission,
    lg_point,
)
from hybrid_aif.objectives import total
from hybrid_aif.objectives.discrete import expected_obs
from hybrid_aif.reference.envs.tmaze import CUE
from hybrid_aif.system import System, SystemParams
from tests.test_switching_observation import H_SHARED, R_PER_MODE
from tests.test_system import _initial_beliefs, tmaze_system

FORMS = ("utility_infogain", "risk_ambiguity")
ATOL = 2e-5


def _forms(system):
    return {form: dataclasses.replace(system, decomposition=form) for form in FORMS}


def _assert_same_g(a, b, atol=ATOL):
    ga, gb = total(a), total(b)
    np.testing.assert_allclose(np.asarray(ga.lower), np.asarray(gb.lower), atol=atol)
    np.testing.assert_allclose(np.asarray(ga.upper), np.asarray(gb.upper), atol=atol)


def _leaf(tree, term, key):
    """A leaf's (lower, upper), or zeros when the tree does not carry it."""
    if term in tree and key in tree[term]:
        iv = tree[term][key]
        return np.asarray(iv.lower), np.asarray(iv.upper)
    return 0.0, 0.0


def _channel_identity(ig_tree, ra_tree, name):
    """``risk + ambiguity`` and ``−(utility + infogain)`` of one channel, both interval ends."""
    r, a = _leaf(ra_tree, "risk", name), _leaf(ra_tree, "ambiguity", name)
    u, i = _leaf(ig_tree, "utility", name), _leaf(ig_tree, "infogain", name)
    lhs = (np.add(r[0], a[0]), np.add(r[1], a[1]))
    rhs = (-np.add(u[1], i[1]), -np.add(u[0], i[0]))
    return lhs, rhs


# --------------------------------------------------------------------------- the T-maze


def _tmaze(epistemic=True):
    system, params = tmaze_system(epistemic=epistemic)
    return system, params, _initial_beliefs(system, params)


def _tmaze_trees(epistemic):
    system, params, beliefs = _tmaze(epistemic)
    policies = system.policies(params)
    return {
        form: s.score_policies(params, beliefs, policies) for form, s in _forms(system).items()
    }, (system, params, beliefs, policies)


def _obs_entropy(system, params, beliefs, policies, m):
    """``Σ_t H[q(o_m)]`` per policy, by hand: predicted beliefs, then the channel's marginal."""
    positional = system.to_positional("params", params)
    out = []
    for policy in np.asarray(policies):
        h = 0.0
        for qs in system.predict_beliefs(params, beliefs, jnp.asarray(policy)):
            q = system.to_positional("beliefs", qs)
            p = expected_obs(q, positional.likelihood, system.A_deps)[m].probs
            h += float(-jnp.sum(xlogy(p, p)))
        out.append(h)
    return np.asarray(out)


def test_tmaze_full_selection_gives_one_g_at_both_ends():
    trees, _ = _tmaze_trees(True)
    _assert_same_g(trees["utility_infogain"], trees["risk_ambiguity"])
    # the undesignated cue and location channels carry their flat-preference risk
    assert set(trees["risk_ambiguity"]["risk"]) == {"location_obs", "reward", "cue"}
    assert set(trees["risk_ambiguity"]["ambiguity"]) == {"location_obs", "reward", "cue"}


def test_tmaze_cue_policy_is_the_same_under_both_forms():
    """The addendum's item 4, fixed: the cue visit is rewarded alike in both forms."""
    system, params, beliefs = _tmaze()
    firsts = {}
    for form, s in _forms(system).items():
        q_pi, tree = s.infer_policies(params, beliefs)
        firsts[form] = int(s.policies(params)[jnp.argmax(q_pi.probs)][0, 0])
    assert firsts == {"utility_infogain": CUE, "risk_ambiguity": CUE}


def test_tmaze_per_channel_identity_under_a_partial_selection():
    """``("context",)`` mounts reward and cue with the location parent marginalised (V9);
    on each the identity holds, and the unmounted location channel's forms differ by its
    ``H[q(o)]``."""
    trees, (system, params, beliefs, policies) = _tmaze_trees(("context",))
    ig, ra = trees["utility_infogain"], trees["risk_ambiguity"]
    assert set(ig["infogain"]) == set(ra["ambiguity"]) == {"reward", "cue"}
    for name in ("reward", "cue"):
        lhs, rhs = _channel_identity(ig, ra, name)
        for a, b in zip(lhs, rhs, strict=True):
            np.testing.assert_allclose(a, b, atol=ATOL)
    lhs, rhs = _channel_identity(ig, ra, "location_obs")
    h = _obs_entropy(system, params, beliefs, policies, 0)
    np.testing.assert_allclose(np.asarray(rhs[0]) - np.asarray(lhs[0]), h, atol=1e-4)


@pytest.mark.parametrize(
    ("epistemic", "ablated"),
    [(False, (0, 1, 2)), (("obs:cue",), (0, 1)), (("obs:reward", "obs:location_obs"), (2,))],
)
def test_tmaze_ablation_differs_by_the_ablated_channels_entropy(epistemic, ablated):
    trees, (system, params, beliefs, policies) = _tmaze_trees(epistemic)
    g_ig = np.asarray(total(trees["utility_infogain"]).midpoint)
    g_ra = np.asarray(total(trees["risk_ambiguity"]).midpoint)
    h = sum(_obs_entropy(system, params, beliefs, policies, m) for m in ablated)
    np.testing.assert_allclose(g_ig - g_ra, h, atol=1e-4)


def test_epistemic_false_leaves_risk_untouched():
    trees_on, _ = _tmaze_trees(True)
    trees_off, _ = _tmaze_trees(False)
    on, off = trees_on["risk_ambiguity"], trees_off["risk_ambiguity"]
    assert "ambiguity" not in off
    assert set(on["risk"]) == set(off["risk"])
    for key in on["risk"]:
        assert np.array_equal(np.asarray(on["risk"][key].lower), np.asarray(off["risk"][key].lower))
    # and on the Gaussian path
    system, params = _two_sensors()
    beliefs = system.initial_priors(params)
    ra = dataclasses.replace(system, decomposition="risk_ambiguity")
    policies = system.policies(params)
    on = ra.score_policies(params, beliefs, policies)
    off = dataclasses.replace(ra, epistemic=False).score_policies(params, beliefs, policies)
    # the readings' total correlation is part of the joint risk, so it stays
    assert set(off) == {"risk", "observation_correlation"} and set(on["risk"]) == set(off["risk"])
    for term in ("risk", "observation_correlation"):
        for key in on[term]:
            np.testing.assert_array_equal(
                np.asarray(on[term][key].lower), np.asarray(off[term][key].lower)
            )


# ------------------------------------------------------------ a switching LG channel


def _switching():
    system = System(
        A_deps=((1, 0),),
        B_deps=((), ()),
        controlled=(True, False),
        policy_len=2,
        mixture_k=2,
        modality_names=("y",),
        factor_names=("mode", "x"),
    )
    x0 = MixtureGaussian(
        log_weights=jnp.log(jnp.asarray([0.4, 0.6])),
        means=jnp.asarray([[-0.8], [0.9]]),
        chols=jnp.asarray([[[0.5]], [[0.7]]]),
    )
    params = SystemParams(
        likelihood={"y": LinearGaussianObservation(H=H_SHARED, c=jnp.zeros(1), noise=R_PER_MODE)},
        transition={
            "mode": ConditionalCategorical(
                log_table=jnp.log(jnp.asarray([[0.8, 0.3], [0.2, 0.7]])), parent_ndims=1
            ),
            "x": LinearGaussianTransition(
                F=jnp.asarray([[0.9]]),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 1)), chol=jnp.asarray([[[0.2]]]), parent_ndims=1
                ),
            ),
        },
        initial_state_prior={"mode": Categorical(logits=jnp.zeros(2)), "x": x0},
        state_targets={"x": Gaussian(loc=jnp.asarray([1.0]), chol=jnp.asarray([[0.5]]))},
    )
    return system, params


def test_switching_full_selection_gives_one_g_at_both_ends():
    system, params = _switching()
    beliefs = system.initial_priors(params)
    policies = system.policies(params)
    trees = {
        form: s.score_policies(params, beliefs, policies) for form, s in _forms(system).items()
    }
    width = np.asarray(trees["risk_ambiguity"]["risk"]["y"].width)
    assert np.all(width > 0.0)  # a Kolchinsky–Tracey pair, not a point
    _assert_same_g(trees["utility_infogain"], trees["risk_ambiguity"])
    lhs, rhs = _channel_identity(trees["utility_infogain"], trees["risk_ambiguity"], "y")
    for a, b in zip(lhs, rhs, strict=True):
        np.testing.assert_allclose(a, b, atol=ATOL)


# ------------------------------------------- a Gaussian-carried factor, two overlapping sensors

H_GPS = np.eye(2)
H_RANGE = np.asarray([[1.0, 0.5]])
H_TILT = np.asarray([[0.0, 1.0]])
R_GPS, R_RANGE, R_TILT = 0.4, 0.3, 0.5


def _two_sensors(**overrides):
    """A 2-D position under three discrete pushes, read by a GPS, a range sensor and a tilt
    sensor, all overlapping; the tilt reading has a preference, the others none."""
    defaults = dict(
        A_deps=((0,), (0,), (0,)),
        B_deps=((),),
        controlled=(True,),
        policy_len=2,
        modality_names=("gps", "range", "tilt"),
        factor_names=("pos",),
    )
    system = System(**{**defaults, **overrides})
    params = SystemParams(
        likelihood={
            "gps": LinearGaussianObservation(
                H=jnp.asarray(H_GPS, jnp.float32), c=jnp.zeros(2), noise=R_GPS * jnp.eye(2)
            ),
            "range": LinearGaussianObservation(
                H=jnp.asarray(H_RANGE, jnp.float32), c=jnp.zeros(1), noise=R_RANGE * jnp.eye(1)
            ),
            "tilt": LinearGaussianObservation(
                H=jnp.asarray(H_TILT, jnp.float32), c=jnp.zeros(1), noise=R_TILT * jnp.eye(1)
            ),
        },
        transition={
            "pos": LinearGaussianTransition(
                F=jnp.eye(2),
                innovations=ConditionalGaussian(
                    loc=jnp.asarray([[0.3, 0.0], [0.0, 0.3], [-0.2, -0.2]]),
                    chol=jnp.stack(
                        [jnp.diag(jnp.asarray(d)) for d in ((0.2, 0.1), (0.1, 0.3), (0.25, 0.25))]
                    ),
                    parent_ndims=1,
                ),
            )
        },
        initial_state_prior={
            "pos": Gaussian(
                loc=jnp.zeros(2), chol=jnp.linalg.cholesky(jnp.asarray([[0.5, 0.2], [0.2, 0.4]]))
            )
        },
        obs_targets={"tilt": Gaussian(loc=jnp.asarray([0.5]), chol=jnp.asarray([[0.4]]))},
        state_targets={"pos": Gaussian(loc=jnp.asarray([0.4, 0.4]), chol=0.6 * jnp.eye(2))},
    )
    return system, params


SENSORS = (("gps", H_GPS, R_GPS), ("range", H_RANGE, R_RANGE), ("tilt", H_TILT, R_TILT))


def _cov(q: Gaussian) -> np.ndarray:
    chol = np.asarray(q.chol, np.float64)
    return chol @ chol.T


def _tc(cov, sensors) -> float:
    """``Σ_m H[y_m] − H[Y]`` of the stacked readings, by hand."""
    H = np.vstack([h for _, h, _ in sensors])
    R = np.diag(np.concatenate([np.full(h.shape[0], r**2) for _, h, r in sensors]))
    S = H @ cov @ H.T + R
    parts, start = 0.0, 0
    for _, h, _ in sensors:
        stop = start + h.shape[0]
        parts += np.linalg.slogdet(S[start:stop, start:stop])[1]
        start = stop
    return 0.5 * (parts - np.linalg.slogdet(S)[1])


def _given(cov, dims) -> np.ndarray:
    idx = np.asarray(dims)
    return cov - cov[:, idx] @ np.linalg.solve(cov[np.ix_(idx, idx)], cov[idx, :])


def _half_logdet(cov, dims=None) -> float:
    if dims is not None:
        cov = cov[np.ix_(np.asarray(dims), np.asarray(dims))]
    return 0.5 * np.linalg.slogdet(cov)[1]


def _run(system, params, plan=(0, 2)):
    ctx = system.planner_context(params, system.initial_priors(params))
    carry, carries, trees = ctx.init_beliefs, [], []
    for t, u in enumerate(plan):
        action = jnp.asarray([u], jnp.int32)
        carry = ctx.transition_fn(carry, action)
        carries.append(carry[0])
        trees.append(ctx.step_terms_fn(carry, action, jnp.asarray(t, jnp.int32)))
    return carries, trees


def _joint_entropy(cov, sensors) -> float:
    """``H[q(Y)]`` of the stacked readings, by hand."""
    H = np.vstack([h for _, h, _ in sensors])
    R = np.diag(np.concatenate([np.full(h.shape[0], r**2) for _, h, r in sensors]))
    S = H @ cov @ H.T + R
    return 0.5 * (S.shape[0] * (1.0 + np.log(2.0 * np.pi)) + np.linalg.slogdet(S)[1])


def _g(trees) -> float:
    return sum(float(total(tree).midpoint) for tree in trees)


@pytest.mark.parametrize("epistemic", [True, ("pos:dim0",)])
def test_two_sensors_full_selection_gives_one_g_at_both_ends(epistemic):
    """Every channel selected, on the whole factor or a sub-block: ``utility_infogain`` carries
    the interaction ``TC(Y) − TC(Y | x_S)``, ``risk_ambiguity`` the correlation ``TC(Y)`` and,
    on a sub-block, the ambiguity's ``−TC(Y | x_S)``; the two give one G."""
    system, params = _two_sensors(epistemic=epistemic)
    beliefs = system.initial_priors(params)
    policies = system.policies(params)
    trees = {
        form: s.score_policies(params, beliefs, policies) for form, s in _forms(system).items()
    }
    ig, ra = trees["utility_infogain"], trees["risk_ambiguity"]
    assert set(ig["infogain"]) == {"gps", "range", "tilt"}  # channel-keyed, as everywhere
    assert set(ig["observation_interaction"]) == set(ra["observation_correlation"]) == {"pos"}
    assert "observation_correlation" not in ig
    correlation = np.asarray(ra["observation_correlation"]["pos"].lower)
    interaction = np.asarray(ig["observation_interaction"]["pos"].lower)
    if epistemic is True:
        assert "observation_interaction" not in ra  # TC(Y | x) = 0
        np.testing.assert_allclose(interaction, correlation, atol=1e-6)
    else:
        conditional = -np.asarray(ra["observation_interaction"]["pos"].lower)
        assert np.all(conditional > 1e-3)  # given x_0 the readings still share y's spread
        np.testing.assert_allclose(interaction, correlation - conditional, atol=1e-5)
    _assert_same_g(ig, ra)
    for name in ("gps", "range", "tilt"):
        lhs, rhs = _channel_identity(ig, ra, name)
        for a, b in zip(lhs, rhs, strict=True):
            np.testing.assert_allclose(a, b, atol=ATOL)


@pytest.mark.parametrize("dims", [None, (0,)])
def test_channel_leaves_less_interaction_are_the_old_block_drop_under_the_chain(dims):
    """Under ``"posterior"``: ``Σ_m I(y_m; x_S) − observation_interaction`` equals the old block
    leaf ``H[q_epi(x_S)] − H[q_cond(x_S)]``, and the interaction is ``TC(Y) − TC(Y | x_S)`` by
    hand on the epistemic belief."""
    epistemic = ("pos",) if dims is None else ("pos:dim0",)
    system, params = _two_sensors(planning_belief="posterior", epistemic=epistemic)
    carries, trees = _run(system, params)
    for pair, tree in zip(carries, trees, strict=True):
        before, after = _cov(pair.epistemic), _cov(pair.conditioned)
        drop = _half_logdet(before, dims) - _half_logdet(after, dims)
        interaction = float(tree["observation_interaction"]["pos"].lower)
        info = sum(float(iv.lower) for iv in tree["infogain"].values())
        assert info - interaction == pytest.approx(drop, abs=1e-5)
        want = _tc(before, SENSORS)
        if dims is not None:
            want -= _tc(_given(before, dims), SENSORS)
        assert interaction == pytest.approx(want, abs=1e-5)
        assert interaction > 1e-3  # the sensors overlap


def test_interaction_can_be_negative_on_a_sub_block():
    """Two readings of ``x0 + x1`` and ``x0 − x1`` on an isotropic belief are uncorrelated, but
    given ``x0`` they are correlated: about ``x0`` they carry information together that neither
    carries alone (synergy), so the interaction is negative and the joint information exceeds
    the channels' sum."""
    from hybrid_aif.objectives.continuous import observation_interaction, state_info_gain

    q = Gaussian(loc=jnp.zeros(2), chol=jnp.eye(2))
    sensors = (("a", np.asarray([[1.0, 1.0]]), 0.3), ("b", np.asarray([[1.0, -1.0]]), 0.3))
    models = tuple(
        LinearGaussianObservation(H=jnp.asarray(h), c=jnp.zeros(1), noise=r * jnp.eye(1))
        for _, h, r in sensors
    )
    interaction = float(observation_interaction(q, models, dims=(0,)))
    cov = np.eye(2)
    assert interaction == pytest.approx(
        _tc(cov, sensors) - _tc(_given(cov, (0,)), sensors), abs=1e-5
    )
    assert interaction < -0.1
    leaves = state_info_gain((q, q), models, ("a", "b"), dims=((0,), (0,)))["infogain"]
    joint = _half_logdet(cov, (0,)) - _half_logdet(
        _kalman(cov, np.vstack([h for _, h, _ in sensors]), 0.09 * np.eye(2)), (0,)
    )
    assert sum(float(iv.lower) for iv in leaves.values()) - interaction == pytest.approx(
        joint, abs=1e-5
    )


def _kalman(cov, H, R) -> np.ndarray:
    S = H @ cov @ H.T + R
    return cov - cov @ H.T @ np.linalg.solve(S, H @ cov)


def test_the_terms_follow_the_selection_and_read_the_predictive_belief():
    """Under ``"prior"`` the interaction is the total correlation of the selected channels'
    readings; a single selected channel mounts none. Under ``risk_ambiguity`` the correlation is
    every channel's, whatever the selection, on the predictive belief; on a sub-block the
    ambiguity's ``−TC(Y | x_S)`` is the selected channels', on the predictive belief too."""
    system, params = _two_sensors(epistemic=("obs:gps", "obs:range"))
    carries, trees = _run(system, params)
    for q, tree in zip(carries, trees, strict=True):
        assert set(tree["infogain"]) == {"gps", "range"}
        want = _tc(_cov(q), SENSORS[:2])
        assert float(tree["observation_interaction"]["pos"].lower) == pytest.approx(want, abs=1e-5)
    _, single = _run(*_two_sensors(epistemic=("obs:gps",)))
    assert all("observation_interaction" not in tree for tree in single)
    for epistemic in (True, ("obs:gps",), False):
        system, params = _two_sensors(
            planning_belief="posterior", decomposition="risk_ambiguity", epistemic=epistemic
        )
        carries, trees = _run(system, params)
        for pair, tree in zip(carries, trees, strict=True):
            want = _tc(_cov(pair.predictive), SENSORS)
            got = float(tree["observation_correlation"]["pos"].lower)
            assert got == pytest.approx(want, abs=1e-5)
            assert "observation_interaction" not in tree
    system, params = _two_sensors(
        planning_belief="posterior",
        decomposition="risk_ambiguity",
        epistemic=("pos:dim1",),
    )
    carries, trees = _run(system, params)
    for pair, tree in zip(carries, trees, strict=True):
        want = -_tc(_given(_cov(pair.predictive), (1,)), SENSORS)
        got = float(tree["observation_interaction"]["pos"].lower)
        assert got == pytest.approx(want, abs=1e-5)


@pytest.mark.parametrize(
    ("epistemic", "selected"),
    [(False, ()), (("obs:gps",), (0,)), (("obs:gps", "obs:tilt"), (0, 2))],
)
def test_two_sensors_ablation_differs_by_the_ablated_readings_joint_entropy(epistemic, selected):
    """On a shared Gaussian factor the forms differ by ``H[q(Y)] − H[q(Y_S)]`` per step, the
    entropy of the ablated channels' readings given the selected ones' (with nothing selected,
    the ablated readings' joint entropy), not the sum of their marginal entropies."""
    system, params = _two_sensors(epistemic=epistemic)
    carries, ig = _run(system, params)
    _, ra = _run(dataclasses.replace(system, decomposition="risk_ambiguity"), params)
    want = sum(
        _joint_entropy(_cov(q), SENSORS)
        - (_joint_entropy(_cov(q), [SENSORS[m] for m in selected]) if selected else 0.0)
        for q in carries
    )
    assert _g(ig) - _g(ra) == pytest.approx(want, abs=1e-4)
    marginals = sum(
        _joint_entropy(_cov(q), [SENSORS[m]])
        for q in carries
        for m in range(3)
        if m not in selected
    )
    assert abs(want - marginals) > 1e-2  # the readings overlap


def test_an_absent_channel_drops_out_of_the_interaction():
    system, params = _two_sensors(planning_belief="posterior")
    ctx = system.planner_context(
        params, system.initial_priors(params), present={"range": jnp.asarray([True, False])}
    )
    carry = ctx.init_beliefs
    for t, u in enumerate((0, 2)):
        action = jnp.asarray([u], jnp.int32)
        carry = ctx.transition_fn(carry, action)
        tree = ctx.step_terms_fn(carry, action, jnp.asarray(t, jnp.int32))
        sensors = SENSORS if t == 0 else (SENSORS[0], SENSORS[2])
        want = _tc(_cov(carry[0].epistemic), sensors)
        assert float(tree["observation_interaction"]["pos"].lower) == pytest.approx(want, abs=1e-5)


def test_obs_key_beside_a_sub_block_of_its_factor_is_refused():
    system, params = _two_sensors(epistemic=("pos:dim0", "obs:gps"))
    with pytest.raises(ValueError, match="'obs:gps' selects everything"):
        _run(system, params)
    _run(*_two_sensors(epistemic=("pos", "obs:gps")))  # the whole factor beside it is fine


# ------------------------------------------------------------------ the uncertainty row


def _hill(**overrides):
    from tests.test_hybrid_lg_system import make_params, make_system

    return make_system(policy_len=1, **overrides), make_params()


def _uncertainty_cases():
    """``(system, params)`` with the row on one factor, and that factor's name: Gaussian and
    mixture-carried, observed and unobserved."""
    from tests.test_hybrid_switching_system import make_params as switching_params
    from tests.test_hybrid_switching_system import make_system as switching_system
    from tests.test_switching_observation import make_planning_params, make_planning_system

    observed = _hill(terms=("uncertainty:state",))
    blind_system, blind_params = _hill(terms=("uncertainty:state",), A_deps=(), modality_names=())
    blind_params = dataclasses.replace(blind_params, likelihood={})
    return {
        "gaussian, observed": (*observed, "state"),
        "gaussian, unobserved": (blind_system, blind_params, "state"),
        "mixture, observed": (
            make_planning_system(terms=("uncertainty:x",)),
            make_planning_params(),
            "x",
        ),
        "mixture, unobserved": (switching_system(), switching_params(), "x"),
    }


@pytest.mark.parametrize(
    "case",
    ["gaussian, observed", "gaussian, unobserved", "mixture, observed", "mixture, unobserved"],
)
@pytest.mark.parametrize("planning_belief", ["prior", "posterior"])
def test_uncertainty_is_the_old_chain_less_leaf_with_the_opposite_sign(case, planning_belief):
    """The old leaf was ``−H[q(x)]`` of the predictive belief under ``infogain``; the row is
    ``H[q(x)]``, a cost, read on the predictive belief under either planning belief."""
    system, params, factor = _uncertainty_cases()[case]
    system = dataclasses.replace(system, planning_belief=planning_belief)
    beliefs = system.initial_priors(params)
    policies = system.policies(params)[:3]
    tree = system.score_policies(params, beliefs, policies)
    assert "infogain" not in tree or factor not in tree["infogain"]
    leaf = tree["uncertainty"][factor]
    for i, policy in enumerate(np.asarray(policies)):
        steps = system.predict_beliefs(params, beliefs, jnp.asarray(policy))
        old = [-q[factor].entropy() for q in steps]  # the old chain-less infogain leaf
        lower = -sum(float(iv.upper) for iv in old)
        upper = -sum(float(iv.lower) for iv in old)
        assert float(leaf.lower[i]) == pytest.approx(lower, abs=1e-5)
        assert float(leaf.upper[i]) == pytest.approx(upper, abs=1e-5)


def test_a_literal_key_on_an_unobserved_factor_raises_and_a_wildcard_passes():
    system, params, _ = _uncertainty_cases()["gaussian, unobserved"]
    beliefs = system.initial_priors(params)
    for key in ("state", "state:dim1"):
        with pytest.raises(ValueError, match=r'terms=\("uncertainty:state",\)'):
            dataclasses.replace(system, epistemic=(key,)).infer_policies(params, beliefs)
    for epistemic in (True, ("*",), ("state:*",)):
        _, tree = dataclasses.replace(system, epistemic=epistemic).infer_policies(params, beliefs)
        assert "infogain" not in tree and set(tree["uncertainty"]) == {"state"}


def test_the_row_takes_a_sub_block_and_round_trips_through_a_description():
    from hybrid_aif.description import build, describe

    system, params = _two_sensors(terms=("uncertainty:pos:dim1",))
    carries, trees = _run(system, params)
    for q, tree in zip(carries, trees, strict=True):
        want = _half_logdet(_cov(q), (1,)) + 0.5 * (1.0 + np.log(2.0 * np.pi))
        assert float(tree["uncertainty"]["pos"].lower) == pytest.approx(want, abs=1e-5)
    document, arrays = describe(system, params)
    assert {"term": "uncertainty", "on": "pos:dim1"} in document["objective"]["terms"]
    rebuilt, _ = build(document, arrays=arrays)
    assert rebuilt.terms == system.terms
    with pytest.raises(ValueError, match="unknown dimension"):
        _two_sensors(terms=("uncertainty:pos:nope",))[0]._row_mounts(params)


# ----------------------------------------------- a Gaussian emission over categorical parents


def _growing():
    """The notebook's growing mixture after forty points of its stream, several levels live."""
    path = pathlib.Path(__file__).resolve().parents[1] / "examples" / "growing_mixture.py"
    spec = importlib.util.spec_from_file_location("growing_mixture", path)
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    points, _ = g.stream()
    system, prior = g.growing_model()
    learned, _, live = g.run_stream(system, prior, points[:40])
    assert live[-1] >= 3
    return system, learned


def test_a_growing_mixture_s_emission_follows_the_selection():
    """Its information gain ``I(y; z)`` (a KT pair on the predictive mixture) under
    ``utility_infogain``, its ambiguity and flat-preference risk under ``risk_ambiguity``: one G
    under full selection, and the leaf brackets a Monte Carlo estimate."""
    system, params = _growing()
    beliefs = system.initial_priors(params)
    policies = system.policies(params)
    selected = dataclasses.replace(system, epistemic=("z",))
    trees = {
        form: s.score_policies(params, beliefs, policies) for form, s in _forms(selected).items()
    }
    ig, ra = trees["utility_infogain"], trees["risk_ambiguity"]
    assert set(ig) == {"infogain"} and set(ra) == {"risk", "ambiguity"}
    _assert_same_g(ig, ra)
    lhs, rhs = _channel_identity(ig, ra, "point")
    for a, b in zip(lhs, rhs, strict=True):
        np.testing.assert_allclose(a, b, atol=ATOL)
    # by Monte Carlo on the predicted weights and the point lowering of the emission
    (q,) = system.predict_beliefs(params, beliefs, policies[0])
    w = np.asarray(q["z"].probs, np.float64)
    point = lg_point(params.likelihood["point"])
    c, chol = np.asarray(point.c, np.float64), np.asarray(point.R_chol, np.float64)
    rng = np.random.default_rng(0)
    z = rng.choice(len(w), p=w / w.sum(), size=40_000)
    y = c[z] + np.einsum("nij,nj->ni", chol[z], rng.standard_normal((len(z), 2)))
    live = np.flatnonzero(w > 0)
    log_dens = [
        np.log(w[k]) + multivariate_normal(c[k], chol[k] @ chol[k].T).logpdf(y) for k in live
    ]
    h_y = -np.mean(np.logaddexp.reduce(np.stack(log_dens), axis=0))
    h_cond = sum(
        w[k] * 0.5 * np.linalg.slogdet(2 * np.pi * np.e * chol[k] @ chol[k].T)[1] for k in live
    )
    leaf = ig["infogain"]["point"]
    assert float(leaf.lower[0]) - 0.03 <= h_y - h_cond <= float(leaf.upper[0]) + 0.03
    assert float(leaf.lower[0]) > 0.5
    # unselected: no epistemic leaf, the flat-preference risk alone
    off = dataclasses.replace(system, epistemic=False, decomposition="risk_ambiguity")
    tree = off.score_policies(params, beliefs, policies)
    assert set(tree) == {"risk"}
    np.testing.assert_array_equal(
        np.asarray(tree["risk"]["point"].lower), np.asarray(ra["risk"]["point"].lower)
    )


def test_an_emission_s_information_about_some_of_its_parents():
    """Over two parents whose second only shifts every offset, the reading given the second is
    a shifted copy of the one-parent mixture, so ``E H[y | z2]`` is that mixture's entropy, and
    given the first it is the shift's own mixture."""
    from hybrid_aif.objectives.continuous import emission_entropies

    loc = jnp.asarray([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
    q1 = Categorical(logits=jnp.log(jnp.asarray([0.5, 0.3, 0.2])))
    q2 = Categorical(logits=jnp.log(jnp.asarray([0.6, 0.4])))
    one = gaussian_emission(loc, 0.5 * jnp.eye(2))
    two = gaussian_emission(jnp.stack([loc, loc + 1.0], axis=1), 0.5 * jnp.eye(2))
    single, noise = emission_entropies(one, (q1,))
    _, given_second = emission_entropies(two, (q1, q2), (False, True))
    np.testing.assert_allclose(float(given_second.lower), float(single.lower), atol=1e-5)
    np.testing.assert_allclose(float(given_second.upper), float(single.upper), atol=1e-5)
    predictive, given_both = emission_entropies(two, (q1, q2))
    _, given_first = emission_entropies(two, (q1, q2), (True, False))
    assert float(given_both.lower) == pytest.approx(float(noise.lower), abs=1e-5)
    assert float(given_both.upper) < float(given_first.lower)
    assert float(given_first.upper) < float(predictive.lower)


# ------------------------------------------------------------- what resolves to nothing


def test_a_merged_component_s_unobserved_factor_resolves_to_nothing():
    """A component with no channel names its continuous factor under ``True`` or ``("*",)``
    nowhere in the composite's literal selection, which would refuse it."""
    from hybrid_aif.composition.merge import merge

    blind, blind_params = _hill(A_deps=(), modality_names=())
    blind_params = dataclasses.replace(blind_params, likelihood={})
    seen, seen_params = _hill()
    for blind_sel, seen_sel, want in (
        (True, False, ()),
        (("*",), False, ()),
        (True, ("state",), ("b.state",)),
    ):
        composite, params = merge(
            {
                "a": dataclasses.replace(blind, epistemic=blind_sel),
                "b": dataclasses.replace(seen, epistemic=seen_sel),
            },
            {"a": blind_params, "b": seen_params},
        )
        assert composite.epistemic == want
        composite.infer_policies(params, composite.initial_priors(params))


def test_the_refusal_names_a_channel_without_an_estimator():
    from hybrid_aif.distributions import MLPConditional

    net = MLPConditional.from_shapes((2,), 1, key=jax.random.PRNGKey(0), width=8, depth=1)
    system = System(
        A_deps=((0,),),
        B_deps=((),),
        controlled=(True,),
        policy_len=1,
        modality_names=("y",),
        factor_names=("x",),
        epistemic=("x",),
    )
    params = SystemParams(
        likelihood={"y": net},
        transition={
            "x": LinearGaussianTransition(
                F=jnp.eye(2),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((2, 2)), chol=jnp.stack([0.1 * jnp.eye(2)] * 2), parent_ndims=1
                ),
            )
        },
        initial_state_prior={"x": Gaussian(loc=jnp.zeros(2), chol=jnp.eye(2))},
    )
    with pytest.raises(
        ValueError,
        match=r"no channel with an information-gain estimator observes \('y' observes it but "
        r"has no information-gain estimator\)",
    ):
        system.infer_policies(params, system.initial_priors(params))
