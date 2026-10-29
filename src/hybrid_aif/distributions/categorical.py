"""Categorical distribution over ``num_levels`` outcomes.

Stored parameterization: ``logits`` (unnormalized log-probabilities), the natural
parameters up to an additive gauge constant — numerically safest for softmax/log
arithmetic and what conjugate updates against Dirichlet counts want to touch.
Event shape: ``()`` (samples are integer indices); the ``logits`` trailing axis
is the level (parameter) axis.
"""

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.distributions.base import ExponentialFamily
from hybrid_aif.interval import Interval


class Categorical(ExponentialFamily):
    logits: Array  # (..., num_levels)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.logits.shape[:-1]

    @property
    def num_levels(self) -> int:
        return self.logits.shape[-1]

    @property
    def log_probs(self) -> Array:
        return jax.nn.log_softmax(self.logits, axis=-1)

    @property
    def probs(self) -> Array:
        return jax.nn.softmax(self.logits, axis=-1)

    @classmethod
    def from_probs(cls, probs: Array) -> "Categorical":
        return cls(logits=jnp.log(probs))

    def log_prob(self, x: Array) -> Array:
        return jnp.take_along_axis(self.log_probs, x[..., None].astype(jnp.int32), axis=-1)[..., 0]

    def sample(self, key: Array, sample_shape: tuple[int, ...] = ()) -> Array:
        z = jax.random.categorical(key, self.logits, shape=sample_shape + self.batch_shape)
        return z.astype(jnp.int32)

    def mean(self) -> Array:
        """Moment view: the probability vector (one-hot expectation)."""
        return self.probs

    def entropy(self) -> Interval:
        lp = self.log_probs
        return Interval.exact(-jnp.sum(jnp.exp(lp) * lp, axis=-1))

    def kl_to(self, other: "Categorical") -> Interval:
        p, lp, lq = self.probs, self.log_probs, other.log_probs
        return Interval.exact(jnp.sum(p * (lp - lq), axis=-1))

    def natural(self) -> tuple[Array, ...]:
        """Logits with the last state pinned to zero (gauge choice)."""
        return (self.logits - self.logits[..., -1:],)

    @classmethod
    def from_natural(cls, logits: Array) -> "Categorical":
        return cls(logits=logits)
