"""Interval-typed scalars: uncertainty measures that carry their own error bars.

Quantities with no closed form for some families (mixture entropies and KLs) are an
``Interval``, a (lower, upper) pair of arrays; exact quantities are zero-width
(``Interval.exact``). Two orthogonal axes: width (point vs spread) and warrant
(:attr:`Interval.derived`, whether a derivation guarantees the enclosure).

Design: docs/design/amortization.md A4
"""

import equinox as eqx
import jax.numpy as jnp
from jax import Array


class Interval(eqx.Module):
    """A closed interval [lower, upper], broadcastable like its member arrays.

    ``derived`` (default ``True``) asserts the enclosure is guaranteed given a correct
    implementation; ``False`` marks a number a net produced, which nothing bounds. It is a static
    field, so it cannot be traced and must agree across a ``lax.scan`` carry. A result is derived
    iff every input is (an AND through every combinator below). Rows declare it once at
    registration (:mod:`hybrid_aif.objectives.registry`).
    """

    lower: Array
    upper: Array
    derived: bool = eqx.field(static=True, default=True)

    @classmethod
    def exact(cls, value: Array, derived: bool = True) -> "Interval":
        """A zero-width interval: the estimate is a point.

        ``derived=False`` is the amortized leaf, a point estimate with no warrant (a critic's value,
        an ensemble's Monte-Carlo info gain).
        """
        value = jnp.asarray(value)
        return cls(lower=value, upper=value, derived=derived)

    @property
    def midpoint(self) -> Array:
        return 0.5 * (self.lower + self.upper)

    @property
    def width(self) -> Array:
        return self.upper - self.lower

    def __add__(self, other: "Interval | Array | float") -> "Interval":
        if isinstance(other, Interval):
            return Interval(
                self.lower + other.lower, self.upper + other.upper, self.derived and other.derived
            )
        return Interval(self.lower + other, self.upper + other, self.derived)

    __radd__ = __add__

    def __neg__(self) -> "Interval":
        return Interval(-self.upper, -self.lower, self.derived)

    def __sub__(self, other: "Interval | Array | float") -> "Interval":
        return self + (-other if isinstance(other, Interval) else -jnp.asarray(other))

    def clip_lower(self, x: "Array | float") -> "Interval":
        """Clamp both bounds below at ``x`` (elementwise); keeps lower <= upper."""
        return Interval(jnp.maximum(self.lower, x), jnp.maximum(self.upper, x), self.derived)

    def __mul__(self, scalar: float) -> "Interval":
        """Multiply by a python scalar; a negative scalar flips the bounds."""
        lo, hi = self.lower * scalar, self.upper * scalar
        if scalar >= 0:
            return Interval(lo, hi, self.derived)
        return Interval(hi, lo, self.derived)

    __rmul__ = __mul__
