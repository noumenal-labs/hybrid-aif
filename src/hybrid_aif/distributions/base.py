"""The distribution contract: the interface every belief, message payload and preference obeys.

This package is the pure-math bottom layer: no wiring, no block ids, no update rules.

1. **Pytree, value-semantic.** A Distribution is an ``equinox.Module`` whose leaves are
   ``jax.Array`` parameters, with no callables or shared mutable state, so it crosses any jit,
   vmap, scan or serialization boundary unchanged.
2. **Structure from shapes.** Static structure (event dim, number of states, mixture size K) is
   derived from leaf shapes; static fields only where it is not shape-derivable.
3. **Batch semantics.** Event/parameter structure occupies a fixed number of trailing axes,
   documented per class; ``batch_shape`` is everything to the left, with no ``to_event()``-style
   reinterpretation. Elementwise operations broadcast over batch axes; discrete selection
   (``reduce``, ``condition_on``) and heterogeneous-list inputs (``fpi``) batch via ``vmap``.
   Leaves carry mutually consistent, materialized batch axes (``broadcast_to`` at construction).
4. **Dual parameter views.** Each family stores its numerically safest parameterization;
   exponential-family members also provide ``natural()``/``from_natural()``, a round trip up to
   gauge and float error. Conjugate updates add in natural coordinates; rollouts read moments.
5. **Interval-typed uncertainty.** ``entropy()`` and ``kl_to()`` always return an
   :class:`~hybrid_aif.interval.Interval`: zero-width where a closed form exists, a genuine
   bound pair otherwise (mixtures).
6. **Sampling.** ``sample(key, sample_shape)`` takes a JAX PRNG key and a static sample shape;
   discrete draws are ``int32``. It is the default PRNG backend's kernel; distributions never
   change per backend.
7. **Dtype.** float32 unless a class documents otherwise.
"""

import abc

import equinox as eqx
from jax import Array

from hybrid_aif.interval import Interval


class Distribution(eqx.Module):
    """Abstract base for all distribution/belief objects. See module docstring."""

    @property
    @abc.abstractmethod
    def batch_shape(self) -> tuple[int, ...]:
        """Leading (batch) axes: leaf shape minus the family's fixed trailing structure."""

    @abc.abstractmethod
    def log_prob(self, x: Array) -> Array:
        """Log density/mass at ``x``; broadcasts over batch axes."""

    @abc.abstractmethod
    def sample(self, key: Array, sample_shape: tuple[int, ...] = ()) -> Array:
        """Draw samples with shape ``sample_shape + batch_shape + event_shape``."""

    @abc.abstractmethod
    def mean(self) -> Array:
        """Mean (moment view)."""

    @abc.abstractmethod
    def entropy(self) -> Interval:
        """Differential/discrete entropy as an Interval (zero-width when exact)."""

    @abc.abstractmethod
    def kl_to(self, other: "Distribution") -> Interval:
        """KL(self ‖ other) for ``other`` of the same family, as an Interval."""


class ExponentialFamily(Distribution):
    """Mixin contract for exponential-family members: dual parameter views."""

    @abc.abstractmethod
    def natural(self) -> tuple[Array, ...]:
        """Natural parameters as a tuple of arrays (gauge documented per class)."""

    @classmethod
    @abc.abstractmethod
    def from_natural(cls, *natural_params: Array) -> "ExponentialFamily":
        """Inverse of :meth:`natural`, up to gauge and float error."""
