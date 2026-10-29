"""Collapsing a belief onto one draw: the sampled rollout's carry, per family.

The sampled rollout carries one draw from each predicted belief, as a belief of the same family
(a one-hot categorical, a zero-covariance Gaussian, a mixture with one component's point), so
every consumer reads it as any other belief. The amortized code family collapses inside
:func:`hybrid_aif.inference.amortized.predict` instead.

A collapsed belief is a point mass: fine as a carry, but a collapsed Gaussian or mixture has no
finite entropy or KL and cannot be conditioned (no Cholesky factor). A collapsed categorical
keeps finite entropy and KL, its zeros floored at :data:`~hybrid_aif.inference.discrete.LOG_ZERO`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.distributions import Categorical, Gaussian, MixtureGaussian
from hybrid_aif.inference.discrete import LOG_ZERO


def collapse(belief: object, key: Array) -> object:
    """One draw from ``belief``, returned as a belief of the same family.

    A raw probability vector collapses to a one-hot vector, a :class:`Categorical` to a one-hot
    :class:`Categorical`, a :class:`Gaussian` to a point mass, a :class:`MixtureGaussian` to one
    component drawn by weight, then its point. Anything else is refused by name.
    """
    if isinstance(belief, Categorical):
        draw = jax.random.categorical(key, belief.logits, axis=-1)
        onehot = jax.nn.one_hot(draw, belief.num_levels, dtype=belief.logits.dtype)
        #  log(0) floored at LOG_ZERO: the probabilities are still exactly one-hot in
        #  float32, and entropy/KL stay finite instead of 0 * -inf
        return Categorical(logits=jnp.maximum(jnp.log(onehot), LOG_ZERO))
    if isinstance(belief, Gaussian):
        return Gaussian(loc=belief.sample(key), chol=jnp.zeros_like(belief.chol))
    if isinstance(belief, MixtureGaussian):
        k_comp, k_point = jax.random.split(key)
        comp = jax.random.categorical(k_comp, belief.log_weights, axis=-1)
        onehot = jax.nn.one_hot(comp, belief.num_components, dtype=belief.means.dtype)
        mean = jnp.sum(onehot[..., None] * belief.means, axis=-2)
        chol = jnp.sum(onehot[..., None, None] * belief.chols, axis=-3)
        point = Gaussian(loc=mean, chol=chol).sample(k_point)
        means = jnp.broadcast_to(point[..., None, :], belief.means.shape)
        return MixtureGaussian(
            log_weights=jnp.log(onehot), means=means, chols=jnp.zeros_like(belief.chols)
        )
    if hasattr(belief, "shape"):  # a bare probability vector
        probs = jnp.asarray(belief)
        draw = jax.random.categorical(key, jnp.log(probs), axis=-1)
        return jax.nn.one_hot(draw, probs.shape[-1], dtype=probs.dtype)
    raise TypeError(
        f"cannot collapse a belief of type {type(belief).__name__} onto a draw: the sampled "
        "rollout is defined for Categorical, Gaussian and MixtureGaussian beliefs, probability "
        "vectors, and amortized codes (inference.amortized.predict)"
    )


def collapse_beliefs(beliefs: tuple[object, ...], key: Array) -> tuple[object, ...]:
    """:func:`collapse` each factor's belief under its own split of ``key``."""
    keys = jax.random.split(key, len(beliefs))
    return tuple(collapse(b, k) for b, k in zip(beliefs, keys, strict=True))
