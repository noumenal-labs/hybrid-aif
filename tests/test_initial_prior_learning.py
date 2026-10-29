"""Learnable initial priors across episodes (D5, acceptance 11; step 4).

Initial-state evidence arrives once per *episode*, at tick 0 — not per tick
— so the update lives in the ordinary ``learn`` gated by rate, never in an
episode-end face (which would hold ``q(s_T)`` when the update needs
``q(s_0)``). Recovery, not parity: pymdp does not currently include an ``update_D``.

The pattern under test is the canonical loop from the ``learn`` docstring:
every tick calls ``learn`` identically, and the initial-prior slot's rate is
1 at tick 0 and 0 afterwards. (The traced-gate variant of tick gating is
covered in ``test_learn``; here the inner loop is unrolled, so the static
0.0 exercises the trace-time skip path on the off ticks.)
"""

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_aif.distributions import Categorical, ConditionalCategorical, Dirichlet
from hybrid_aif.system import System, SystemParams

TRUE_D = jnp.asarray([0.8, 0.2])
EPISODE_LEN = 3


def _system():
    return System(
        A_deps=((0,),),
        B_deps=((0,),),  # explicit self-dependency at t-1, the tabular convention
        controlled=(False,),
        policy_len=1,
        modality_names=("sense",),
        factor_names=("s",),
    )


def _params(initial_prior):
    ident = jnp.log(jnp.asarray([[0.9, 0.1], [0.1, 0.9]]))
    sticky = jnp.log(jnp.asarray([[[0.95], [0.05]], [[0.05], [0.95]]]).reshape(2, 2, 1))
    return SystemParams(
        likelihood={"sense": ConditionalCategorical(log_table=ident, parent_ndims=1)},
        transition={"s": ConditionalCategorical(log_table=sticky, parent_ndims=2)},
        initial_state_prior={"s": initial_prior},
    )


def test_initial_prior_recovery_across_episodes():
    system = _system()
    true_params = _params(Categorical.from_probs(TRUE_D))
    agent = _params(Dirichlet(concentration=jnp.ones(2)))
    env, env_params = system.as_env(true_params)

    def episode_obs(key):
        obs, state = env.reset(key, env_params)
        seq = [obs["sense"]]
        for _ in range(EPISODE_LEN - 1):
            key, k = jax.random.split(key)
            obs, state = env.step(k, state, {}, env_params)
            seq.append(obs["sense"])
        return jnp.stack(seq)

    n_episodes = 600
    obs = jax.vmap(episode_obs)(jax.random.split(jax.random.PRNGKey(0), n_episodes))

    @jax.jit
    def run(p0):
        def one_episode(p, obs_e):
            priors = None
            for t in range(EPISODE_LEN):  # unrolled; t static
                beliefs, _ = system.infer_states(p, {"sense": obs_e[t]}, priors=priors)
                p = system.learn(
                    p,
                    {"sense": obs_e[t]},
                    beliefs,
                    lr=0.0,  # every other slot frozen (static skip)
                    lr_by_slot={"initial_state_prior:s": 1.0 if t == 0 else 0.0},
                )
                priors = system.update_empirical_prior(p, beliefs, {})
            return p, None

        return jax.lax.scan(one_episode, p0, obs)[0]

    learned = run(agent).initial_state_prior["s"]
    assert isinstance(learned, Dirichlet)
    # ~1 pseudo-count per episode, on top of the (1, 1) prior
    total = float(learned.concentration.sum())
    np.testing.assert_allclose(total, 2.0 + n_episodes, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(learned.point().probs), np.asarray(TRUE_D), atol=0.05)


def test_off_tick_learn_is_identity_on_the_prior():
    system = _system()
    agent = _params(Dirichlet(concentration=jnp.asarray([3.0, 1.0])))
    beliefs, _ = system.infer_states(agent, {"sense": jnp.asarray(0, jnp.int32)})
    out = system.learn(
        agent,
        {"sense": jnp.asarray(0, jnp.int32)},
        beliefs,
        lr=0.0,
        lr_by_slot={"initial_state_prior:s": 0.0},
    )
    assert (
        out.initial_state_prior["s"].concentration is agent.initial_state_prior["s"].concentration
    )
