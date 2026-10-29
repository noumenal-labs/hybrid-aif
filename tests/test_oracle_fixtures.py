"""The checked-in oracle seed fixtures stay loadable, match their provenance, and
**reproduce under our environment**.

The eight ``oracle_mc_seed<N>.npz`` files under ``reference/raif/fixtures`` are the
expert episodes the oracle's own training was seeded with (addendum §7's
record; A12 decision 6). Two guards live here.

The **freshness** guard: the loader's format contract holds, the counts match the
provenance JSON, and the seed policy's quality matches the oracle report's expert
line (mean return ≈ −76), so 2d's "seed stats reported beside acceptance numbers"
is a parity check by construction.

The **env-parity** guard, which ``reference/raif/fixtures/__init__.py`` has promised
since it was written ("the fixture doubles as a bitwise env-parity check") and
which did not exist until 2026-09-07: step our ``MountainCar`` with their recorded
action column and reproduce their recorded observations and rewards. It is the one
test that fails loudly if the row convention is ever changed in only one of the two
places it lives — our collection loop and their export — because it pins the
*joint* claim that our physics is theirs **and** that our columns are indexed as
theirs are.
"""

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.reference.envs.mountain_car import MountainCar
from hybrid_aif.reference.raif.fixtures import load_fixture
from hybrid_aif.train.buffers import seed_stats

FIXTURES = pathlib.Path(__file__).parent.parent / "src/hybrid_aif/reference/raif/fixtures"

#: Float32 round-off between their numpy physics and our JAX one, measured over all
#: 192 fixture episodes (acting ledger B3: ``8.9e-07`` on seed 0's twelve, ``1.3e-06``
#: over all four seeds). Rewards are integers and reproduce **exactly**.
OBS_TOL = 2e-6


@pytest.mark.parametrize("seed", range(8))
def test_oracle_fixture_loads_and_matches_provenance(seed):
    episodes, record = load_fixture(FIXTURES / f"oracle_mc_seed{seed}.npz")
    assert record["seed"] == seed
    assert record["commit_sha"].startswith("9b1864e8")
    assert record["n_episodes"] == len(episodes) == 48
    assert record["n_steps"] == sum(int(np.asarray(e.obs).shape[0]) for e in episodes)
    stats = seed_stats(episodes, success="is_success")
    assert stats.success_rate == 1.0
    assert -78.0 < stats.mean_return < -74.0  # the oracle report's expert line: −76.1 mean
    for ep in episodes:
        csi = np.asarray(ep.extras["can_self_imitate"])
        assert csi[-1] == 0.0  # the goal step's action is the reset filler, never expert-marked
        assert (csi[:-1] == 1.0).all()  # every seed's expert episodes all succeeded


@pytest.mark.parametrize("seed", range(8))
def test_our_env_reproduces_their_recorded_observations_and_rewards(seed):
    """The env-parity check the fixture module promises: their actions, our physics.

    Their recorded episode is a run of driver transitions, so stepping the env from
    ``obs[t]`` with ``action[t]`` must land on ``obs[t+1]`` and earn ``reward[t+1]``.
    That indexing is the whole content of the test: it is the convention their
    ``driver.py:63`` writes and, since 2026-09-07, the one
    :func:`hybrid_aif.reference.raif.run.make_collect_fn` collects, so the two halves of
    the buffer are now one convention rather than two (previously 100% of M⁺ at step 0
    was in theirs while every collected episode was in ours — acting ledger D4).

    Both the wrong alignment and the wrong physics are asserted against, because a
    test that only checked observations could pass with the reward column a row out:
    under **our** old convention (``reward[t]`` the reward of ``action[t]``) the
    reward error is exactly **2**, and that is asserted as the *negative* control.
    Rewards are integer sums of ``{0, −1, −2}``, so their reproduction is exact.

    This also exercises, on real data, the one composition the acting audit checked by
    hand: ``ActionRepeat(2)``'s ``break`` on ``is_last or is_terminal``
    (``wrappers.py:59-69``) against our statically unrolled masked pair, whose three
    reward cases (0 if the first raw step reaches the flag, −1 if the second, −2
    otherwise) a successful episode's tail exercises directly.
    """
    env = MountainCar()
    step = jax.jit(lambda x, a: env.agent_step(jax.random.PRNGKey(0), (x,), a))
    episodes, _ = load_fixture(FIXTURES / f"oracle_mc_seed{seed}.npz")
    worst_obs = worst_reward = worst_ours = 0.0
    for ep in episodes:
        obs, action, reward = (np.asarray(getattr(ep, n)) for n in ("obs", "action", "reward"))
        assert reward[0] == 0.0  # their reset row: a real observation at zero reward
        x = jnp.asarray(obs[0])
        for t in range(obs.shape[0] - 1):  # the last row's action is the driver's zero
            _, (x,), r, _ = step(x, jnp.asarray(action[t]))
            worst_obs = max(worst_obs, float(np.abs(np.asarray(x) - obs[t + 1]).max()))
            worst_reward = max(worst_reward, abs(float(r) - reward[t + 1]))
            worst_ours = max(worst_ours, abs(float(r) - reward[t]))
        #  their is_success is a predicate on the row's OWN observation, so it is
        #  reproducible from the obs column alone and lands on the last row
        np.testing.assert_array_equal(
            obs[:, 0] >= env.goal_position, np.asarray(ep.extras["is_success"])
        )
    assert worst_obs < OBS_TOL, worst_obs
    assert worst_reward == 0.0, worst_reward
    #  the negative control: one row out and the reward column is wrong by a full
    #  agent step. This is the assertion that fails if the convention is half-changed
    assert worst_ours == 2.0, worst_ours
