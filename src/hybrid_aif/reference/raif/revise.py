"""Self-revision: the per-episode backward pass that produces ρ and the imitation mask.

A port of their ``callbacks.py::lookback_on_episode`` with their ``configs.yaml:109``
hyperparameters. It runs once per finished episode, before the episode is written to the
buffers (:func:`label`). Pure, unbatched and jit-able: one episode in, two ``(T,)`` arrays
out.

Design: ``docs/design/amortization-p0-addendum.md`` §4.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.reference.raif.columns import (
    CAN_SELF_IMITATE,
    HAS_SUCCEEDED,
    IS_SUCCESS,
    PREF_LABEL,
)
from hybrid_aif.train.buffers import Episode

#: Their ``configs.yaml:109`` triple: the backward discount, the quantization
#: floor below which the carry is zeroed, and the failure branch's decay.
ALPHA = 0.4
EPSILON = 0.001
BETA = 0.3


class PrefLabels(NamedTuple):
    """What ``lookback_on_episode`` writes back into the episode. Both ``(T,)``.

    ``pref_label`` is ρ ∈ [−1, 1]. ``can_self_imitate`` is the propagated imitation mask, a
    float weight in [0, 1] that replaces the recorded flag and gates the actor's BC term.
    """

    pref_label: Array
    can_self_imitate: Array


def pref_labels(
    ep: Episode,
    *,
    alpha: float = ALPHA,
    epsilon: float = EPSILON,
    beta: float = BETA,
) -> PrefLabels:
    """``lookback_on_episode``, ported: ``(pref_label, can_self_imitate)``.

    ``alpha``, ``epsilon`` and ``beta`` are their ``disc``, ``thr`` and ``fail``. The steps:

    - the expert flags are gated by episode success, ``can_self_imitate *= has_succeeded[-1]``;
    - a backward carry seeded from the last step's gated flag credits step ``t−1`` from
      ``is_success[t]`` geometrically at ``alpha``, zeroed below ``epsilon``; ``mask[T−1]`` is
      the seed and is not thresholded;
    - the mask is the clipped sum of the gated flags and the carry, shifted one step forward
      into ``positive_mask`` (slot 0 from ``has_succeeded[0]``);
    - in a failed episode each step gets ``−beta**(T−1−t)``, so the last step is at −1; the
      expert exemption on this branch is a no-op after the success gate and is kept as their
      code has it;
    - ``pref_label = clip(positive_mask + neg, −1, 1)`` with ``pref_label[0] = 0``.

    Raises ``ValueError`` if ``alpha`` is outside [0, 1), ``epsilon`` is negative or ``beta``
    is outside (0, 1).
    """
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha (their disc) must be in [0, 1), got {alpha}")
    if epsilon < 0.0:
        raise ValueError(f"epsilon (their thr) must be non-negative, got {epsilon}")
    if not 0.0 < beta < 1.0:
        # strictly positive: their failure_decays construction divides by `fail`
        raise ValueError(f"beta (their fail) must be in (0, 1), got {beta}")
    t = ep.length
    succeeded = ep.extras[HAS_SUCCEEDED][-1].astype(jnp.float32)
    gated = ep.extras[CAN_SELF_IMITATE].astype(jnp.float32) * succeeded
    success = ep.extras[IS_SUCCESS].astype(jnp.float32)

    def backward(carry: Array, s: Array) -> tuple[Array, Array]:
        # their `carry = max(carry * disc, is_success[t]); carry *= (carry >= thr)`
        c = jnp.maximum(carry * alpha, s)
        c = c * (c >= epsilon).astype(jnp.float32)
        return c, c

    if t > 1:
        # the scan reads is_success[1:] in reverse; the step at index j credits mask[j]
        _, credited = jax.lax.scan(backward, jnp.maximum(gated[-1], 0.0), success[1:], reverse=True)
        mask = jnp.concatenate([credited, jnp.maximum(gated[-1:], 0.0)])
    else:
        mask = jnp.maximum(gated[-1:], 0.0)
    final = jnp.clip(gated + mask, 0.0, 1.0)
    positive = jnp.concatenate([ep.extras[HAS_SUCCEEDED][:1].astype(jnp.float32), final[:-1]])
    decays = jnp.cumprod(jnp.full((t,), beta, jnp.float32))[::-1] / beta
    decays = decays * (1.0 - succeeded)
    neg = -decays * (1.0 - gated)
    rho = jnp.clip((positive + neg).at[0].set(0.0), -1.0, 1.0)
    return PrefLabels(pref_label=rho, can_self_imitate=final)


def label(ep: Episode, **kwargs) -> Episode:
    """``ep`` with both columns ``lookback_on_episode`` writes: ρ and the imitation mask.

    Call it on a finished episode, before :func:`hybrid_aif.train.buffers.add`; on a stored
    window the last step is not the episode's last step. Afterwards ``can_self_imitate`` is the
    float mask, not the recorded expert flag.
    """
    out = pref_labels(ep, **kwargs)
    return ep._replace(
        extras={**ep.extras, PREF_LABEL: out.pref_label, CAN_SELF_IMITATE: out.can_self_imitate}
    )


def positive_gate(rho: Array) -> Array:
    """``max(0, sgn ρ)``: the CRSPP KL's hard mask to positive states (paper Eq. 5)."""
    return jnp.maximum(0.0, jnp.sign(rho))
