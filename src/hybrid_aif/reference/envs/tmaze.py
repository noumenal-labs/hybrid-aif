"""TMaze as a positional (pymdp-style) process, tensor-identical to ``pymdp.envs.TMaze``.

Transcribed from pymdp's ``TMaze`` (pymdp/envs/tmaze.py: ``_generate_A_separate``,
``_valid_connections``, ``generate_B``, ``generate_D``; separate-cue, adjacent variant) as a
pure constructor; tests/test_envs.py checks tensor parity against it.

Classic variant: five locations (centre 0, left 1, right 2, cue 3, middle 4),
adjacent connectivity, separate cue modality. Factors: location (5, controlled)
x context/reward-condition (2, fixed). Modalities: location (identity), reward
(none/reward/punishment), cue (none/left/right). Construction is traceable in
the scalar probability arguments, so batches of param sets can be built under
``vmap`` as well as by stacking.
"""

import jax.numpy as jnp
from jax import Array

from hybrid_aif.envs._process import DiscreteProcess
from hybrid_aif.envs.generative import GenerativeEnvParams

CENTRE, LEFT, RIGHT, CUE, MIDDLE = 0, 1, 2, 3, 4
NUM_LOCATIONS = 5
_CONNECTIONS = frozenset(
    [
        (CENTRE, CUE),
        (CUE, CENTRE),
        (CENTRE, MIDDLE),
        (MIDDLE, CENTRE),
        (MIDDLE, LEFT),
        (LEFT, MIDDLE),
        (MIDDLE, RIGHT),
        (RIGHT, MIDDLE),
    ]
)


def tmaze(
    reward_probability: float | Array = 1.0,
    punishment_probability: float | Array = 1.0,
    cue_validity: float | Array = 0.95,
    reward_condition: int | None = None,
) -> tuple[DiscreteProcess, GenerativeEnvParams]:
    """Env + params for the classic TMaze; ``reward_condition=None`` means 50/50."""
    rp = jnp.asarray(reward_probability, jnp.float32)
    pp = jnp.asarray(punishment_probability, jnp.float32)
    cv = jnp.asarray(cue_validity, jnp.float32)

    A_loc = jnp.eye(NUM_LOCATIONS, dtype=jnp.float32)

    A_reward = jnp.zeros((3, NUM_LOCATIONS, 2), jnp.float32)
    for loc in (CENTRE, CUE, MIDDLE):
        A_reward = A_reward.at[0, loc, :].set(1.0)
    for cond, (arm, other) in enumerate(((LEFT, RIGHT), (RIGHT, LEFT))):
        A_reward = A_reward.at[1, arm, cond].set(rp)
        A_reward = A_reward.at[0, arm, cond].set(1.0 - rp)
        A_reward = A_reward.at[2, other, cond].set(pp)
        A_reward = A_reward.at[0, other, cond].set(1.0 - pp)

    A_cue = jnp.zeros((3, NUM_LOCATIONS, 2), jnp.float32)
    for loc in (CENTRE, LEFT, RIGHT, MIDDLE):
        A_cue = A_cue.at[0, loc, :].set(1.0)
    for cond in range(2):
        A_cue = A_cue.at[cond + 1, CUE, cond].set(cv)
        A_cue = A_cue.at[2 - cond, CUE, cond].set(1.0 - cv)

    B_loc = jnp.zeros((NUM_LOCATIONS, NUM_LOCATIONS, NUM_LOCATIONS), jnp.float32)
    for frm in range(NUM_LOCATIONS):
        for act in range(NUM_LOCATIONS):
            to = act if (frm, act) in _CONNECTIONS else frm
            B_loc = B_loc.at[to, frm, act].set(1.0)
    B_ctx = jnp.eye(2, dtype=jnp.float32).reshape(2, 2, 1)

    D_loc = jnp.zeros(NUM_LOCATIONS, jnp.float32).at[CENTRE].set(1.0)
    if reward_condition is None:
        D_ctx = jnp.full(2, 0.5, jnp.float32)
    else:
        D_ctx = jnp.zeros(2, jnp.float32).at[reward_condition].set(1.0)

    env = DiscreteProcess(A_deps=((0,), (0, 1), (0, 1)), B_deps=((0,), (1,)))
    params = GenerativeEnvParams(
        likelihood=(A_loc, A_reward, A_cue),
        transition=(B_loc, B_ctx),
        initial_state_prior=(D_loc, D_ctx),
    )
    return env, params
