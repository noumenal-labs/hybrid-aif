"""A generative process steps the factors it is told to and keeps the rest.

``GenerativeEnv.step(..., factors=...)`` names the state factors to step; a kept factor keeps
its value, the observation is of the state so made, and on the hybrid path a factor that reads
this step's values reads a kept factor as kept. The driver's sequential and async orders step
each component's factors at that component's turn with it.
"""

import jax
import jax.numpy as jnp
import pytest

from hybrid_aif.envs import GenerativeEnv
from tests.test_action_nodes import pusher
from tests.test_merge import agent_and_world, merged

KEY = jax.random.PRNGKey(3)


def test_a_masked_step_moves_the_named_factors_and_keeps_the_others():
    composite, cp = merged(policy_len=1)
    env, ep = composite.as_env(cp)
    _, state = env.reset(KEY, ep)
    action = {a: jnp.asarray(1, jnp.int32) for a in env.names.actions}
    key = jax.random.PRNGKey(7)
    _, full = env.step(key, state, action, ep)
    _, part = env.step(key, state, action, ep, factors=("agent.location",))
    #  the named factor moved as in the full step (each factor draws under its own key)
    assert bool(jnp.array_equal(part["agent.location"], full["agent.location"]))
    for f in env.names.factors:
        if f != "agent.location":
            assert bool(jnp.array_equal(part[f], state[f])), f


def test_the_observation_is_of_the_state_the_masked_step_made():
    system, params = pusher(noise=0.0)
    env, ep = system.as_env(params)
    _, state = env.reset(KEY, ep)
    push = {"push": jnp.asarray(0, jnp.int32)}
    _, nxt = env.step(KEY, state, push, ep, factors=())
    assert bool(jnp.array_equal(nxt["x"], state["x"]))
    #  ``see`` is 0.8 on the diagonal: over many keys the kept state is what it mostly reports
    seen = [
        int(env.step(k, state, push, ep, factors=())[0]["see"]) for k in jax.random.split(KEY, 200)
    ]
    assert seen.count(int(state["x"])) > 120


def test_unknown_factor_names_and_outside_environments_are_refused():
    system, params = pusher()
    env, ep = system.as_env(params)
    _, state = env.reset(KEY, ep)
    with pytest.raises(ValueError, match="not state factors of this process"):
        env.step(KEY, state, {"push": jnp.asarray(0, jnp.int32)}, ep, factors=("y",))
    with pytest.raises(TypeError, match="as a sequence"):
        env.step(KEY, state, {"push": jnp.asarray(0, jnp.int32)}, ep, factors="x")
    (agent, ap), _ = agent_and_world(policy_len=1)
    inner, ip = agent.as_env(ap)
    outside = GenerativeEnv.wrap(inner.inner, agent, ap)
    _, ostate = outside.reset(KEY, ip)
    per_factor = {f: jnp.asarray(0, jnp.int32) for f in outside.names.factors}
    with pytest.raises(ValueError, match="has no parts to name"):
        outside.step(KEY, ostate, per_factor, ip, factors=("location",))
