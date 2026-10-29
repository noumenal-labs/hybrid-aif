"""The sampled rollout beside the expected one (ledger 6.5, ruled 2026-09-20/21).

Every prediction face — ``update_empirical_prior``, ``predict_beliefs``, the amortized
``predict`` — takes an optional key. Keyless it carries the expected belief, bitwise as
before; keyed it carries one draw of the same family. The scoring seam carries the key
to rows that roll a model (the CRSPP's preferred rollout), and ``imagine`` has a static
switch for it that defaults to the expected rollout.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import Categorical, Gaussian, MixtureGaussian
from hybrid_aif.inference import amortized as _am
from hybrid_aif.inference.sample import collapse, collapse_beliefs
from hybrid_aif.objectives.amortized import features
from hybrid_aif.objectives.base import total
from hybrid_aif.train.imagine import (
    SCORING_KEY_TAG,
    _code_and_carry,
    _sampled_step,
    imagine,
    rollout_returns,
)
from tests.test_amortized_objective import agent as amortized_agent
from tests.test_hybrid_lg_system import make_params as lg_params
from tests.test_hybrid_lg_system import make_system as lg_system
from tests.test_hybrid_switching_system import initial_beliefs as switching_beliefs
from tests.test_hybrid_switching_system import make_params as switching_params
from tests.test_hybrid_switching_system import make_system as switching_system
from tests.test_system import tmaze_system

KEY = jax.random.PRNGKey(0)


def _leaves_equal(a, b) -> None:
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


# ----------------------------------------------------------------- the collapse, per family


def test_a_categorical_collapses_to_a_one_hot_of_its_own_family():
    q = Categorical(logits=jnp.log(jnp.array([0.2, 0.5, 0.3])))
    out = collapse(q, KEY)
    assert isinstance(out, Categorical)
    probs = np.asarray(out.probs)
    assert set(np.round(probs, 6)) <= {0.0, 1.0} and probs.sum() == 1.0
    draws = np.asarray([collapse(q, k).probs for k in jax.random.split(KEY, 4000)]).mean(0)
    np.testing.assert_allclose(draws, [0.2, 0.5, 0.3], atol=0.03)
    #  the collapsed categorical is still readable: its zeros are floored, not -inf
    assert np.isfinite(float(out.entropy().midpoint)) and np.isfinite(float(out.kl_to(q).midpoint))


def test_a_gaussian_collapses_to_a_point_mass():
    q = Gaussian(loc=jnp.array([1.0, -2.0]), chol=0.5 * jnp.eye(2))
    out = collapse(q, KEY)
    assert isinstance(out, Gaussian)
    np.testing.assert_array_equal(np.asarray(out.chol), 0.0)
    pts = np.asarray([collapse(q, k).loc for k in jax.random.split(KEY, 4000)])
    np.testing.assert_allclose(pts.mean(0), [1.0, -2.0], atol=0.05)
    np.testing.assert_allclose(pts.std(0), [0.5, 0.5], atol=0.05)


def test_a_mixture_collapses_to_one_component_then_one_point():
    q = MixtureGaussian(
        log_weights=jnp.log(jnp.array([0.7, 0.3])),
        means=jnp.array([[0.0, 0.0], [10.0, 10.0]]),
        chols=jnp.stack([0.1 * jnp.eye(2), 0.1 * jnp.eye(2)]),
    )
    out = collapse(q, KEY)
    assert isinstance(out, MixtureGaussian) and out.means.shape == q.means.shape
    w = np.asarray(jnp.exp(out.log_weights))
    assert set(np.round(w, 6)) == {0.0, 1.0}
    np.testing.assert_array_equal(np.asarray(out.chols), 0.0)
    near_second = [float(collapse(q, k).means[0, 0]) > 5.0 for k in jax.random.split(KEY, 2000)]
    assert abs(np.mean(near_second) - 0.3) < 0.04


def test_a_probability_vector_collapses_to_a_one_hot_vector():
    out = collapse(jnp.array([0.1, 0.9]), KEY)
    assert out.shape == (2,) and set(np.round(np.asarray(out), 6)) <= {0.0, 1.0}


def test_an_unknown_belief_is_refused_by_name():
    with pytest.raises(TypeError, match="cannot collapse a belief of type str"):
        collapse("not a belief", KEY)


def test_collapse_beliefs_splits_the_key_per_factor():
    qs = (Categorical(logits=jnp.zeros(3)), Categorical(logits=jnp.zeros(3)))
    a, b = collapse_beliefs(qs, KEY)
    same = np.array_equal(np.asarray(a.probs), np.asarray(b.probs))
    #  two factors under one key would draw the same level every time; split, they need not
    draws = [collapse_beliefs(qs, k) for k in jax.random.split(KEY, 50)]
    assert not all(np.array_equal(np.asarray(x.probs), np.asarray(y.probs)) for x, y in draws)
    del same


# ----------------------------------------------------------------------------- the faces


def test_keyless_faces_are_unchanged_and_keyed_ones_collapse_on_the_tabular_path():
    system, params = tmaze_system()
    beliefs = system.initial_priors(params)
    action = system.from_positional("actions", jnp.zeros(len(system.names.controls), jnp.int32))
    expected = system.update_empirical_prior(params, beliefs, action)
    again = system.update_empirical_prior(params, beliefs, action, key=None)
    _leaves_equal(expected, again)
    sampled = system.update_empirical_prior(params, beliefs, action, key=KEY)
    for q in (sampled[k] for k in system.names.states):  # a control node's belief is not drawn
        probs = np.asarray(q.probs if isinstance(q, Categorical) else q)
        assert set(np.round(probs, 6)) <= {0.0, 1.0}
    #  the sampled belief is a legal carry for the next tick
    system.update_empirical_prior(params, sampled, action)


def test_the_lg_path_collapses_to_a_point_and_predicts_on_from_it():
    system, params = lg_system(), lg_params()
    q = {"state": Gaussian(loc=jnp.array([0.3, -0.1]), chol=0.15 * jnp.eye(2))}
    action = system.from_positional("actions", jnp.array([1], jnp.int32))
    _leaves_equal(
        system.update_empirical_prior(params, q, action),
        system.update_empirical_prior(params, q, action, key=None),
    )
    point = system.update_empirical_prior(params, q, action, key=KEY)["state"]
    np.testing.assert_array_equal(np.asarray(point.chol), 0.0)
    nxt = system.update_empirical_prior(params, {"state": point}, action)["state"]
    assert float(jnp.linalg.det(nxt.cov)) > 0.0  # the process noise comes back


def test_the_switching_path_collapses_the_mode_and_the_state():
    system, params = switching_system(), switching_params()
    action = system.from_positional("actions", jnp.array([1], jnp.int32))
    pushed = system.update_empirical_prior(params, switching_beliefs(), action, key=KEY)
    mode, x = pushed["mode"], pushed["x"]
    assert set(np.round(np.asarray(mode.probs), 6)) <= {0.0, 1.0}
    assert isinstance(x, MixtureGaussian)
    assert set(np.round(np.asarray(jnp.exp(x.log_weights)), 6)) <= {0.0, 1.0}
    system.update_empirical_prior(params, {"mode": mode, "x": x}, action)  # a legal carry


def test_predict_beliefs_splits_the_key_once_per_step():
    system, params = tmaze_system(policy_len=3)
    beliefs = system.initial_priors(params)
    plan = jnp.zeros((3, len(system.names.controls)), jnp.int32)
    _leaves_equal(
        system.predict_beliefs(params, beliefs, plan),
        system.predict_beliefs(params, beliefs, plan, key=None),
    )
    keys = jax.random.split(KEY, 3)
    stepwise, b = [], beliefs
    for t in range(3):
        b = system.update_empirical_prior(
            params, b, system.from_positional("actions", plan[t]), key=keys[t]
        )
        stepwise.append(b)
    _leaves_equal(system.predict_beliefs(params, beliefs, plan, key=KEY), stepwise)


# --------------------------------------------------------------------- the amortized path


def test_keyed_predict_is_the_trainers_sampled_step_bitwise():
    system, params, _ = amortized_agent()
    structure = system._amortized_structure(params)
    code, carry = _code_and_carry(structure, system)
    beliefs = system.initial_priors(params)
    action = jnp.array([0.3], jnp.float32)
    positional = system.to_positional("beliefs", beliefs)
    new, logits, probs = _sampled_step(params, system, carry, code, positional, action, KEY)
    keyed_action = system.from_positional("actions", action, params)
    via_face = system.to_positional(
        "beliefs", system.update_empirical_prior(params, beliefs, keyed_action, key=KEY)
    )
    _leaves_equal(new, via_face)
    expected = system.to_positional(
        "beliefs", system.update_empirical_prior(params, beliefs, keyed_action)
    )
    np.testing.assert_allclose(np.asarray(expected[code]), np.asarray(probs), rtol=1e-6)
    np.testing.assert_array_equal(
        np.asarray(via_face[code]).sum(-1), np.ones(via_face[code].shape[:-1])
    )
    assert set(np.round(np.asarray(via_face[code]), 5).ravel()) <= {0.0, 1.0}
    del logits


def test_the_predict_seam_is_keyless_by_default_and_bitwise_as_before():
    system, params, _ = amortized_agent()
    structure = system._amortized_structure(params)
    beliefs = system.initial_priors(params)
    args = (
        system.to_positional("params", params).transition,  # the internal tier reads tuples
        system._table.B_deps,
        structure.families,
        system.controlled,
        (1, 1),  # the levels per factor: the action is continuous, the no-op level
        system._control_layout(system.to_positional("params", params)),
        system.to_positional("beliefs", beliefs),
        jnp.array([0.0], jnp.float32),
    )
    _leaves_equal(_am.predict(*args), _am.predict(*args, key=None))


# ------------------------------------------------------------------------ the scoring seam


def test_the_cosine_row_samples_its_preferred_rollout_under_a_key():
    system, params, (crspp, cparams, *_) = amortized_agent()
    beliefs = system.initial_priors(params)
    plan = jnp.zeros((system.policy_len, 1), jnp.float32)
    expected = system.score_policies(params, beliefs, plan)
    again = system.score_policies(params, beliefs, plan, key=None)
    _leaves_equal(expected, again)
    sampled = system.score_policies(params, beliefs, plan, key=KEY)
    row = "cosine_sim_to_pref_rollout"
    e = float(total({row: expected[row]}).midpoint)
    s = float(total({row: sampled[row]}).midpoint)
    assert e != s
    #  the preferred side is one-hot under the key: its stoch-part norm is sqrt(num_vars)
    #  the row's own action-free push, through the CRSPP's public face ({}: no control)
    pushed = crspp.update_empirical_prior(cparams, crspp.initial_priors(cparams), {}, key=KEY)
    keyed = crspp.to_positional("beliefs", pushed)
    fam = list(crspp._amortized_structure(cparams).families)
    code = fam.index(_am.CODE)
    np.testing.assert_allclose(
        float(jnp.linalg.norm(features(keyed)[-keyed[code].size :])),
        float(jnp.sqrt(keyed[code].shape[0])),
        rtol=1e-5,
    )
    #  and the other rows are untouched by the key
    for name in expected:
        if name != row:
            _leaves_equal(expected[name], sampled[name])


def test_imagine_defaults_to_the_expected_rollout_and_switches_statically():
    system, params, _ = amortized_agent()
    n = 6
    starts = {
        k: jnp.broadcast_to(b, (n, *b.shape)) for k, b in system.initial_priors(params).items()
    }
    ls = system.init_learn_state(params)
    _, _, off = imagine(system, params, starts, ls, key=KEY, horizon=2)
    _, _, off2 = imagine(system, params, starts, ls, key=KEY, horizon=2, sample_scoring=False)
    _, _, on = imagine(system, params, starts, ls, key=KEY, horizon=2, sample_scoring=True)
    row = "cosine_sim_to_pref_rollout"
    assert float(off.per_term[row]) == float(off2.per_term[row])
    assert float(on.per_term[row]) != float(off.per_term[row])
    #  the switch only touches the scoring rows' key: the actor's draws, and so the
    #  imagined trajectories and every other row, are bitwise the same
    assert float(on.policy_entropy) == float(off.policy_entropy)
    assert float(on.per_term["occupancy"]) == float(off.per_term["occupancy"])


def test_the_run_config_carries_the_switch_and_old_runs_resume_as_expected(tmp_path):
    import dataclasses
    import json

    from hybrid_aif.reference.raif import run as run_mc

    cfg, _ = run_mc.parse(["--out", str(tmp_path / "a")])
    assert cfg.sample_scoring is False
    cfg, _ = run_mc.parse(["--out", str(tmp_path / "a"), "--sample-scoring"])
    assert cfg.sample_scoring is True
    #  a run.json written before the knob existed resumes as the expected rollout
    old = dataclasses.asdict(run_mc.parse(["--out", str(tmp_path / "old")])[0])
    del old["sample_scoring"]
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "run.json").write_text(json.dumps(old))
    resumed, _ = run_mc.parse(["--out", str(tmp_path / "old"), "--resume", "--sample-scoring"])
    assert resumed.sample_scoring is False  # the run's own configuration wins


# ------------------------------------------------- what the adversarial review found


def test_the_scoring_rows_key_collides_with_none_of_the_rollouts_draws():
    """Under partitionable threefry ``fold_in(k, d) == split(k, n)[d]``, so a small tag
    would hand the rows the actor's own step key (the first cut used 1). The tag must be
    reached by no split the scan performs, and the rows' per-step keys must differ from
    every action and code key of every imagined step."""
    horizon = 8
    for seed in range(5):
        k = jax.random.PRNGKey(seed)
        row_keys = jax.random.split(jax.random.fold_in(k, SCORING_KEY_TAG), horizon)
        step_keys = jax.random.split(k, horizon)
        scan_keys = jnp.concatenate([jax.random.split(sk) for sk in step_keys] + [step_keys])
        assert not bool((row_keys[:, None, :] == scan_keys[None, :, :]).all(-1).any())
        assert not bool(
            (jax.random.fold_in(k, SCORING_KEY_TAG) == jax.random.split(k, 256)).all(-1).any()
        )


def test_a_plan_key_does_not_imply_a_sampled_scoring():
    """``infer_policies``/``act`` under a search planner draw from ``plan_key``; the
    scoring rows must not read that key. Only ``score_policies(key=)`` and
    ``imagine(sample_scoring=True)`` select the sampled rollout."""
    from unittest import mock

    from hybrid_aif.system import System

    system, params, _ = amortized_agent()
    beliefs = system.initial_priors(params)
    seen = []
    original = System._amortized_scoring

    def spy(self, params_, aux_beliefs=None, key=None, horizon=None, face="this face"):
        seen.append(key)
        return original(self, params_, aux_beliefs, key=key, horizon=horizon, face=face)

    with mock.patch.object(System, "_amortized_scoring", spy):
        system.planner_context(params, beliefs, key=KEY)  # a planner's key
        assert seen == [None]
        plan = jnp.zeros((system.policy_len, 1), jnp.float32)
        system.score_policies(params, beliefs, plan, key=KEY)  # the scoring key
        assert len(seen) == 2 and seen[1] is not None


def test_two_model_rolling_rows_would_not_share_a_stream():
    from hybrid_aif.objectives import registry

    system, params, _ = amortized_agent()
    mounts = system._row_mounts(params)
    (m,) = [m for m in mounts if m.name == "cosine_sim_to_pref_rollout"]
    base = registry.Statics(
        index=(), key=KEY, scoring=params.scoring, declared=dict(system.scoring)
    )
    k1 = registry.mount_statics(m, base).key
    assert k1 is not None and not bool((k1 == KEY).all())  # folded per row, never verbatim
    assert registry.mount_statics(m, base._replace(key=None)).key is None


def test_rollout_returns_reads_the_rollout_the_switch_selects():
    system, params, _ = amortized_agent()
    n = 4
    starts = {
        k: jnp.broadcast_to(b, (n, *b.shape)) for k, b in system.initial_priors(params).items()
    }
    off = rollout_returns(system, params, starts, key=KEY, horizon=2)
    same = rollout_returns(system, params, starts, key=KEY, horizon=2, sample_scoring=False)
    on = rollout_returns(system, params, starts, key=KEY, horizon=2, sample_scoring=True)
    _leaves_equal(off, same)
    assert not np.array_equal(np.asarray(on.value), np.asarray(off.value))
    np.testing.assert_array_equal(np.asarray(on.critic), np.asarray(off.critic))  # same states


# ------------------------------------------------------------ the recognition face


def test_keyed_recognition_is_the_collapsed_posterior_on_the_tabular_path():
    """``infer_states(key=)`` is ``collapse_beliefs(infer_states(), key)``, bitwise; the
    keyless call is unchanged; the collapsed posterior is a legal carry."""
    system, params = tmaze_system()
    priors = system.initial_priors(params)
    obs = system.from_positional(
        "observations", tuple(jnp.zeros((), jnp.int32) for _ in system.names.modalities)
    )
    expected, _ = system.infer_states(params, obs, priors)
    again, _ = system.infer_states(params, obs, priors, key=None)
    _leaves_equal(expected, again)
    keyed, _ = system.infer_states(params, obs, priors, key=KEY)
    _leaves_equal(
        system.to_positional("beliefs", keyed),
        collapse_beliefs(system.to_positional("beliefs", expected), KEY),
    )
    for q in (keyed[k] for k in system.names.states):  # a control node's belief is not drawn
        assert set(np.round(np.asarray(q.probs), 6)) <= {0.0, 1.0}
    no_op = jnp.zeros(len(system.names.controls), jnp.int32)
    system.update_empirical_prior(params, keyed, system.from_positional("actions", no_op))


def test_keyed_recognition_is_a_point_mass_on_the_lg_path():
    system, params = lg_system(), lg_params()
    q = {"state": Gaussian(loc=jnp.array([0.3, -0.1]), chol=0.15 * jnp.eye(2))}
    obs = {"position": jnp.asarray([0.07])}
    expected = system.infer_states(params, obs, q)[0]["state"]
    point = system.infer_states(params, obs, q, key=KEY)[0]["state"]
    np.testing.assert_array_equal(np.asarray(point.chol), 0.0)
    (ref,) = collapse_beliefs((expected,), KEY)
    _leaves_equal((point,), (ref,))
    action = system.from_positional("actions", jnp.array([1], jnp.int32))
    nxt = system.update_empirical_prior(params, {"state": point}, action)["state"]
    assert float(jnp.linalg.det(nxt.cov)) > 0.0


def test_keyed_recognition_collapses_the_codes_and_keeps_the_carries_on_the_amortized_path():
    system, params, _ = amortized_agent()
    structure = system._amortized_structure(params)
    code, carry = _code_and_carry(structure, system)
    priors = system.initial_priors(params)
    #  the reference world's modalities: the observation, then reward and cont, which no
    #  recognition module reads (the runner passes the same ``(obs, None, None)``)
    obs = system.from_positional(
        "observations", (jnp.zeros((params.likelihood["state"].out_dim,), jnp.float32), None, None)
    )
    expected, _ = system.infer_states(params, obs, priors)
    keyed_beliefs, _ = system.infer_states(params, obs, priors, key=KEY)
    expected = system.to_positional("beliefs", expected)
    keyed = system.to_positional("beliefs", keyed_beliefs)
    _leaves_equal((keyed[carry],), (expected[carry],))
    d = np.asarray(keyed[code])
    assert set(np.round(d, 5).ravel()) <= {0.0, 1.0} and np.all(d.sum(-1) == 1)
    #  the split is per factor, carries included: the code's draw is collapse under ITS key
    keys = jax.random.split(KEY, len(priors))
    _leaves_equal((keyed[code],), (collapse(jnp.asarray(expected[code]), keys[code]),))
    action = system.from_positional("actions", jnp.array([0.3], jnp.float32), params)
    system.update_empirical_prior(params, keyed_beliefs, action)
