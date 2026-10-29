"""Replay, chunk two (``docs/design/replay.md`` R4/R5/R6 as built): the amortized
path, ``externals``, the actor's density as the recorded action's likelihood, and
the ``LearnState`` carry under ``learn=True`` on a gradient-slot model.
"""

from __future__ import annotations

import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.description import describe
from hybrid_aif.objectives.amortized import features
from hybrid_aif.reference.raif.agent import raif_mountain_car
from hybrid_aif.reference.raif.run import SMALL
from hybrid_aif.replay import replay, replay_document
from hybrid_aif.system import LearnState

KEY = jax.random.PRNGKey(0)
_EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"


def _example(name: str):
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _close(a, b, rtol=1e-5, atol=1e-6) -> None:
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    for x, y in zip(la, lb, strict=True):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=rtol, atol=atol)


@pytest.fixture(scope="module")
def agent():
    return raif_mountain_car(KEY, **SMALL)


def _trace(system, T=5, seed=3):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    obs = jax.random.normal(k1, (T, 2)) * 0.3
    actions = jnp.clip(jax.random.normal(k2, (T, 1)) * 0.5, -1.0, 1.0)
    #  reward and continuation are predicted, never read
    return (
        system.from_positional("observations", (obs, None, None)),
        {system.names.actions[0]: actions},  # the one continuous node's (T, 1) column
    )


def _length(actions) -> int:
    return int(jax.tree_util.tree_leaves(actions)[0].shape[0])


def _keyed_mask(system, mask):
    """A ``(T, M)`` flag array as the keyed mask, one ``(T,)`` entry per modality."""
    return system.from_positional("observations", tuple(jnp.moveaxis(mask, -1, 0)))


def _eager(system, params, observations, actions, key):
    T = _length(actions)
    priors = system.initial_priors(params)
    keys = jax.random.split(key, T)
    rows, priors_rows, chosen_rows = [], [], []
    for t in range(T):
        obs = {k: None if o is None else o[t] for k, o in observations.items()}
        beliefs, _ = system.infer_states(params, obs, priors)
        chosen, _, _ = system.act(
            KEY, params, beliefs, selection="deterministic", mode="policy", plan_key=keys[t]
        )
        rows.append(beliefs)
        priors_rows.append(priors)
        chosen_rows.append(chosen)
        priors = system.update_empirical_prior(
            params, beliefs, {k: a[t] for k, a in actions.items()}
        )
    stack = lambda xs: jax.tree_util.tree_map(lambda *a: jnp.stack(a), *xs)  # noqa: E731
    return stack(rows), stack(priors_rows), stack(chosen_rows)


# ------------------------------------------------------------- R5: the amortized path


def test_the_amortized_path_replays_through_the_per_tick_faces(agent):
    system, params = agent.system, agent.params
    observations, actions = _trace(system)
    out = replay(system, params, observations, actions, key=KEY, metrics=False)
    beliefs, priors, chosen = _eager(system, params, observations, actions, KEY)
    _close(out.beliefs, beliefs)
    _close(out.priors, priors)
    _close(out.chosen, chosen)
    for k, a in actions.items():
        assert out.chosen[k].shape == a.shape and out.chosen[k].dtype == a.dtype
    assert out.unsupported == {} and out.log_evidence is None and out.trees is None
    assert out.params is None and out.learn_state is None


def test_the_recorded_action_gets_the_actors_own_density(agent):
    system, params = agent.system, agent.params
    observations, actions = _trace(system)
    out = replay(system, params, observations, actions, key=KEY, metrics=False)
    actor = params.policy_prior
    recorded = system.to_positional("actions", actions, params)
    for t in range(int(recorded.shape[0])):
        b_t = jax.tree_util.tree_map(lambda x, t=t: x[t], out.beliefs)
        expect = actor.log_prob(recorded[t], (features(system.to_positional("beliefs", b_t)),))
        np.testing.assert_allclose(float(out.action_log_prob[t]), float(expect), rtol=1e-5)
    #  the habit's own draw is more likely under the actor than a random action is
    own = replay(system, params, observations, out.chosen, key=KEY, metrics=False)
    assert float(own.action_log_prob.sum()) > float(out.action_log_prob.sum())


def test_a_policy_module_needs_the_planner_key(agent):
    system, params = agent.system, agent.params
    observations, actions = _trace(system)
    with pytest.raises(ValueError, match="no plan_key was given"):
        replay(system, params, observations, actions, metrics=False)


def test_learn_is_the_trainers_business_on_the_amortized_path(agent):
    system, params = agent.system, agent.params
    observations, actions = _trace(system)
    with pytest.raises(NotImplementedError, match="fit_world"):
        replay(system, params, observations, actions, key=KEY, metrics=False, learn=True)


# ---------------------------------------------------------- R6: masks, amortized


def test_a_fully_masked_tick_is_predict_only_on_the_amortized_path(agent):
    system, params = agent.system, agent.params
    observations, actions = _trace(system)
    T = _length(actions)
    #  the state observation dropped at tick 2
    mask = _keyed_mask(system, jnp.ones((T, 3), bool).at[2, 0].set(False))
    out = replay(system, params, observations, actions, key=KEY, metrics=False, present=mask)
    tick = lambda tree: jax.tree_util.tree_map(lambda x: x[2], tree)  # noqa: E731
    _close(tick(out.beliefs), tick(out.priors))
    plain = replay(system, params, observations, actions, key=KEY, metrics=False)
    _close(
        jax.tree_util.tree_map(lambda x: x[:2], out.beliefs),
        jax.tree_util.tree_map(lambda x: x[:2], plain.beliefs),
    )
    #  masking the predicted-only channels is a no-op
    noop = _keyed_mask(system, jnp.ones((T, 3), bool).at[:, 1:].set(False))
    same = replay(system, params, observations, actions, key=KEY, metrics=False, present=noop)
    _close(same.beliefs, plain.beliefs)
    exercised = {k: bool(v) for k, v in same.exercised.items()}
    assert exercised == system.from_positional("observations", (True, False, False))


def test_externals_are_checked_against_the_trace_length(agent):
    """The CRSPP is the one declared model with an external input, and it is a
    preference model without a policy posterior, so slicing is exercised through the
    reference agent's replay with an unread external and pinned by the shape sentence."""
    system, params = agent.system, agent.params
    observations, actions = _trace(system)
    T = _length(actions)
    out = replay(
        system,
        params,
        observations,
        actions,
        key=KEY,
        metrics=False,
        externals={"unused": jnp.zeros((T, 3))},
    )
    assert system.to_positional("beliefs", out.beliefs)[0].shape[0] == T
    with pytest.raises(ValueError, match=r"externals\['unused'\] has leading axis 3"):
        replay(
            system,
            params,
            observations,
            actions,
            key=KEY,
            metrics=False,
            externals={"unused": jnp.zeros((3, 3))},
        )


# ------------------------------------------------ R4: gradient slots under learn=True


def test_learn_carries_a_learn_state_through_a_gradient_slot():
    mc = _example("mountain_car")
    system, params = mc.make_agent(KEY, horizon=2, width=8, depth=1)
    T = 4
    obs = system.from_positional("observations", (jax.random.normal(KEY, (T, 2)) * 0.1,))
    actions = system.from_positional(
        "actions", jnp.clip(jax.random.normal(jax.random.PRNGKey(1), (T, 1)), -1.0, 1.0), params
    )
    out = replay(system, params, obs, actions, key=KEY, metrics=False, learn=True, lr=1e-2)
    assert isinstance(out.learn_state, LearnState)
    before = jax.tree_util.tree_leaves(params.transition["state"])
    after = jax.tree_util.tree_leaves(out.params.transition["state"])
    assert any(
        not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(before, after, strict=True)
    )
    fixed = replay(system, params, obs, actions, key=KEY, metrics=False)
    assert fixed.params is None and fixed.learn_state is None


# ------------------------------------------------------------------- R8: document


def test_replay_document_on_the_amortized_agent(agent):
    system, params = agent.system, agent.params
    observations, actions = _trace(system)
    document, arrays = describe(system, params, arrays="file")
    via_doc = replay_document(
        document, observations, actions, arrays=arrays, key=KEY, metrics=False
    )
    direct = replay(system, params, observations, actions, key=KEY, metrics=False)
    _close(via_doc.beliefs, direct.beliefs)


# ---------------------------------------------- what the adversarial review found


def test_a_split_mask_is_refused_concretely_by_tick(agent):
    """Inside the scan the mask is a tracer and cannot be refused; replay holds the concrete
    mask and refuses a split before tracing. The reference agent's modules read one
    modality, so the split is staged on a copy whose embed module reads two."""
    import dataclasses

    from hybrid_aif.factors import Recognition
    from hybrid_aif.inference.amortized import read_modalities

    system, params = agent.system, agent.params
    plan = system._amortized_structure(params).plan
    assert read_modalities(plan) == (0,)
    #  a mask that drops the read modality on one tick is a dropped frame, not a split
    observations, actions = _trace(system)
    T = _length(actions)
    fine = _keyed_mask(system, jnp.ones((T, 3), bool).at[1, 0].set(False))
    replay(system, params, observations, actions, key=KEY, metrics=False, present=fine)
    #  stage two read modalities: reward observed by the embed module as well
    two = dataclasses.replace(
        system,
        recognition=tuple(
            Recognition(r.variable, r.parents + ("obs:reward",)) if "embed" in r.variable else r
            for r in system.recognition
        ),
    )
    #  an embed module that reads the state and the reward (widths 2 and 1)
    from hybrid_aif.distributions.neural import MLPPredictor

    embed = MLPPredictor.from_shapes((2, 1), params.recognition["embed"].out_dim, key=KEY, width=8)
    params = dataclasses.replace(params, recognition={**params.recognition, "embed": embed})
    assert read_modalities(two._amortized_structure(params).plan) == (0, 1)
    split = _keyed_mask(two, jnp.ones((T, 3), bool).at[2, 1].set(False))
    state = system.to_positional("observations", observations)[0]
    staged = two.from_positional("observations", (state, jnp.zeros((T, 1)), None))
    with pytest.raises(
        ValueError, match=r"at tick\(s\) \[2\].*present together or absent together"
    ):
        replay(two, params, staged, actions, key=KEY, present=split)


def test_stack_traces_carries_unread_modalities_as_none(agent):
    from hybrid_aif.replay import stack_traces

    system, params = agent.system, agent.params
    long, a_long = _trace(system, T=5, seed=1)
    short, a_short = _trace(system, T=3, seed=2)
    obs, actions, mask, valid = stack_traces([(long, a_long, None), (short, a_short, None)])
    first, *rest = system.modality_names
    assert all(obs[m] is None for m in rest) and obs[first].shape == (2, 5, 2)
    assert valid.tolist() == [[True] * 5, [True] * 3 + [False] * 2]
    batched = jax.vmap(
        lambda o, a, m, v: replay(
            system,
            params,
            {first: o, **{n: None for n in rest}},
            a,
            present=m,
            valid=v,
            key=KEY,
            metrics=False,
        )
    )(obs[first], actions, mask, valid)
    single = replay(system, params, short, a_short, key=KEY, metrics=False)
    _close(jax.tree_util.tree_map(lambda x: x[1, :3], batched.beliefs), single.beliefs)
    with pytest.raises(ValueError, match="None for different modalities"):
        stack_traces(
            [(long, a_long, None), ({**short, rest[0]: short[first][:, :1]}, a_short, None)]
        )


def test_none_observations_are_refused_off_the_amortized_path():
    from tests.test_system import tmaze_system

    system, params = tmaze_system()
    T = 3
    observations = {
        "location_obs": jnp.zeros(T, jnp.int32),
        "reward": None,
        "cue": jnp.zeros(T, jnp.int32),
    }
    with pytest.raises(ValueError, match=r"observations \['reward'\] are None.*use present="):
        replay(system, params, observations, {"a_location": jnp.zeros(T, jnp.int32)})


def test_externals_accept_array_likes(agent):
    system, params = agent.system, agent.params
    observations, actions = _trace(system)
    T = _length(actions)
    out = replay(
        system,
        params,
        observations,
        actions,
        key=KEY,
        metrics=False,
        externals={"unused": [[0.0] * 3] * T},
    )
    assert system.to_positional("beliefs", out.beliefs)[0].shape[0] == T


def test_a_partial_present_mapping_still_keeps_read_modalities_together(agent):
    """A modality left out of present= is present, so a static mapping naming only one of the
    modalities a module reads, as absent, splits them and is refused (infer_states and
    replay alike)."""
    import dataclasses

    from hybrid_aif.distributions.neural import MLPPredictor
    from hybrid_aif.factors import Recognition
    from hybrid_aif.inference.amortized import read_modalities

    system, params = agent.system, agent.params
    two = dataclasses.replace(
        system,
        recognition=tuple(
            Recognition(r.variable, r.parents + ("obs:reward",)) if "embed" in r.variable else r
            for r in system.recognition
        ),
    )
    #  an embed module that reads the state and the reward (widths 2 and 1)
    old = params.recognition["embed"]
    embed = MLPPredictor.from_shapes((2, 1), old.out_dim, key=KEY, width=8)
    params = dataclasses.replace(params, recognition={**params.recognition, "embed": embed})
    assert read_modalities(two._amortized_structure(params).plan) == (0, 1)
    observations, actions = _trace(system)
    T = _length(actions)
    state = system.to_positional("observations", observations)[0]
    staged = two.from_positional("observations", (state, jnp.zeros((T, 1)), None))
    tick = {k: None if v is None else v[0] for k, v in staged.items()}
    with pytest.raises(ValueError, match="present together or absent together"):
        two.infer_states(params, tick, present={"reward": False})
    with pytest.raises(
        ValueError, match=r"at tick\(s\) \[2\].*present together or absent together"
    ):
        replay(
            two,
            params,
            staged,
            actions,
            key=KEY,
            present={"reward": jnp.ones(T, bool).at[2].set(False)},
        )
