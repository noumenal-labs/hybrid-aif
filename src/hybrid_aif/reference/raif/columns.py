"""R-AIF's episode columns, as task columns of the engine's :class:`Episode`.

``is_success`` is the goal predicate on the row's observation, ``has_succeeded`` its running
OR, ``can_self_imitate`` the expert flag that :mod:`.revise` overwrites with the imitation
mask, and ``pref_label`` is ρ. In Mountain Car the goal ends the episode, so
:func:`raif_episode` writes ``cont = 1 − is_success``.

Design: ``docs/design/amortization-p0-addendum.md`` §4.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

from hybrid_aif.dtypes import cast
from hybrid_aif.train.buffers import ColumnSpec, Episode, episode

IS_SUCCESS = "is_success"
HAS_SUCCEEDED = "has_succeeded"
CAN_SELF_IMITATE = "can_self_imitate"
PREF_LABEL = "pref_label"

#: The precision the R-AIF reference runs in, in every mode: float32. Every network of the agent
#: draws its weights in this dtype, so it computes in it and casts its inputs to it; every other
#: parameter, each episode and buffer, and what the reference takes from the environment are
#: made in it or cast to it, and a training batch is cast to the agent's dtype, so under JAX's
#: 64-bit mode the reference still computes in float32. Their mixed float16 precision is not
#: followed (``docs/design/amortization.md`` divergence 9).
DTYPE = jnp.float32

#: The task columns and how a buffer pads them: the two flags hold their last
#: value (a padded step is still "after the goal"), the weight and the label are
#: zero where nothing was done or earned.
EXTRAS: dict[str, ColumnSpec] = {
    IS_SUCCESS: ColumnSpec((), "bool", "hold"),
    HAS_SUCCEEDED: ColumnSpec((), "bool", "hold"),
    CAN_SELF_IMITATE: ColumnSpec((), "float32", "zero"),
    PREF_LABEL: ColumnSpec((), "float32", "zero"),
}

#: The **recorded** columns — the on-disk fixture format and the collect stream's
#: arrays; ``pref_label`` is derived and never stored.
RECORDED: tuple[str, ...] = (
    "obs",
    "action",
    "reward",
    IS_SUCCESS,
    HAS_SUCCEEDED,
    CAN_SELF_IMITATE,
    "is_first",
    "is_last",
)


def raif_episode(
    *,
    obs: Array,
    action: Array,
    reward: Array,
    is_success: Array,
    has_succeeded: Array,
    can_self_imitate: Array,
    is_first: Array,
    is_last: Array,
    pref_label: Array | None = None,
) -> Episode:
    """An :class:`Episode` from their recorded columns: ``cont = 1 − is_success``,
    the four task columns in ``extras`` (ρ zero until :func:`.revise.label`). Every float column
    is cast to :data:`DTYPE`."""
    success = jnp.asarray(is_success, jnp.bool_)
    t = int(success.shape[0])
    ep = episode(
        obs,
        action,
        reward,
        is_first=is_first,
        is_last=is_last,
        cont=1.0 - success.astype(DTYPE),
        **{
            IS_SUCCESS: success,
            HAS_SUCCEEDED: jnp.asarray(has_succeeded, jnp.bool_),
            CAN_SELF_IMITATE: jnp.asarray(can_self_imitate, DTYPE),
            PREF_LABEL: jnp.zeros((t,), DTYPE)
            if pref_label is None
            else jnp.asarray(pref_label, DTYPE),
        },
    )
    return cast(ep, DTYPE)


def recorded_columns(ep: Episode) -> dict[str, np.ndarray]:
    """The recorded columns of ``ep`` as numpy arrays, in :data:`RECORDED` order."""
    out = {}
    for name in RECORDED:
        out[name] = np.asarray(ep.extras[name] if name in ep.extras else getattr(ep, name))
    return out


def succeeded(ep: Episode) -> bool:
    """Whether the episode reached the goal: its final ``has_succeeded``."""
    return bool(ep.extras[HAS_SUCCEEDED][-1])
