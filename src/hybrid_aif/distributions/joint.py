"""Joint categorical over several discrete axes: one simplex, jointly normalized.

``JointCategorical`` normalizes over all ``event_ndims`` trailing axes jointly; ``event_ndims``
is a static field (the event/batch split is not shape-derivable). It is isomorphic to a
Categorical on the flattened event axes, which supplies ``log_prob``/``entropy``/``kl_to``.

Design: docs/design/conditional-families.md (joint families)
"""

import math

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from hybrid_aif.distributions.base import Distribution
from hybrid_aif.distributions.categorical import Categorical
from hybrid_aif.distributions.conditional import ConditionalCategorical
from hybrid_aif.interval import Interval


class JointCategorical(Distribution):
    """Joint distribution over ``event_ndims`` discrete axes.

    ``logits`` are unnormalized log-probabilities over the trailing
    ``event_ndims`` axes jointly (a single additive gauge for the whole
    table). Samples and ``log_prob`` arguments are stacked ``int32`` indices
    with a trailing axis of length ``event_ndims``, one index per event axis.
    """

    logits: Array  # (..., *event_shape)
    event_ndims: int = eqx.field(static=True)

    def __check_init__(self) -> None:
        if self.event_ndims < 1:
            raise ValueError(f"need event_ndims >= 1, got {self.event_ndims}")
        if self.logits.ndim < self.event_ndims:
            raise ValueError(f"logits rank {self.logits.ndim} < event_ndims {self.event_ndims}")

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.logits.shape[: self.logits.ndim - self.event_ndims]

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self.logits.shape[self.logits.ndim - self.event_ndims :]

    def _flat(self) -> Categorical:
        """The isomorphic Categorical on the flattened (row-major) event axes."""
        n = math.prod(self.event_shape)
        return Categorical(logits=self.logits.reshape(self.batch_shape + (n,)))

    @property
    def log_probs(self) -> Array:
        """Jointly normalized log-probability table, same shape as ``logits``."""
        return self._flat().log_probs.reshape(self.logits.shape)

    @property
    def probs(self) -> Array:
        return self._flat().probs.reshape(self.logits.shape)

    @classmethod
    def from_probs(cls, table: Array, event_ndims: int) -> "JointCategorical":
        return cls(logits=jnp.log(table), event_ndims=event_ndims)

    def _flat_index(self, x: Array) -> Array:
        idx = jnp.asarray(x)
        strides = [math.prod(self.event_shape[i + 1 :]) for i in range(self.event_ndims)]
        return jnp.sum(idx * jnp.asarray(strides, jnp.int32), axis=-1)

    def log_prob(self, x: Array) -> Array:
        """Log mass at stacked indices ``x`` (trailing axis of length ``event_ndims``)."""
        return self._flat().log_prob(self._flat_index(x))

    def sample(self, key: Array, sample_shape: tuple[int, ...] = ()) -> Array:
        """Stacked ``int32`` indices, shape ``sample_shape + batch_shape + (event_ndims,)``."""
        flat = self._flat().sample(key, sample_shape)
        return jnp.stack(jnp.unravel_index(flat, self.event_shape), axis=-1).astype(jnp.int32)

    def mean(self) -> Array:
        """Moment view: the joint probability table (one-hot expectation)."""
        return self.probs

    def entropy(self) -> Interval:
        return self._flat().entropy()

    def kl_to(self, other: "JointCategorical") -> Interval:
        if other.event_shape != self.event_shape:
            raise ValueError(f"event shapes differ: {self.event_shape} vs {other.event_shape}")
        return self._flat().kl_to(other._flat())

    def to_conditional(self, child_axis: int) -> ConditionalCategorical:
        """P(child | rest): normalize along one event axis per configuration of the others.

        ``child_axis`` indexes the event block (negatives allowed). The child axis moves to the
        front of the structure block and the other event axes become parent axes in order;
        normalization happens on read, so the conversion is a pure axis move.
        """
        n = self.event_ndims
        if n < 2:
            raise ValueError("to_conditional needs event_ndims >= 2")
        if not -n <= child_axis < n:
            raise ValueError(f"child_axis {child_axis} out of range for event_ndims {n}")
        offset = self.logits.ndim - n
        moved = jnp.moveaxis(self.logits, offset + child_axis % n, offset)
        return ConditionalCategorical(log_table=moved, parent_ndims=n - 1)

    def marginalize(self, axes: int | tuple[int, ...]) -> "JointCategorical | Categorical":
        """Sum OUT the given event axes (event-relative indices; negatives allowed).

        Returns the joint over the remaining event axes, or a
        :class:`Categorical` when exactly one remains.
        """
        if isinstance(axes, int):
            axes = (axes,)
        n = self.event_ndims
        norm = sorted({a % n for a in axes})
        if len(norm) != len(axes):
            raise ValueError(f"duplicate axes in {axes}")
        if len(norm) >= n:
            raise ValueError("marginalize must keep at least one event axis")
        offset = self.logits.ndim - n
        summed = self.probs.sum(axis=tuple(offset + a for a in norm))
        remaining = n - len(norm)
        if remaining == 1:
            return Categorical(logits=jnp.log(summed))
        return JointCategorical(logits=jnp.log(summed), event_ndims=remaining)
