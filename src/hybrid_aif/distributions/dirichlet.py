"""Dirichlet distribution over probability vectors of ``num_levels`` outcomes.

Stored parameterization: ``concentration`` (positive counts). Event shape ``(num_levels,)``;
natural view ``eta = concentration - 1``. As a declaration (a learnable initial-state prior) it
satisfies :class:`~hybrid_aif.distributions.learnable.Learnable`.

Design: docs/design/parameter-learning.md D5
"""

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import digamma, gammaln

from hybrid_aif.distributions.base import ExponentialFamily
from hybrid_aif.distributions.categorical import Categorical
from hybrid_aif.interval import Interval

#: pymdp's MINVAL clip constant (shared by the exact ``spm_wnorm`` port below).
_MINVAL = float(jnp.finfo(jnp.float64).eps)


def _wnorm(concentration: Array, child_axis: int) -> Array:
    """Exact Dirichlet novelty weights: pymdp's ``maths._exact_wnorm`` (pymdp/maths.py), after
    SPM's ``spm_wnorm`` (the subfunction in ``toolbox/DEM/spm_MDP_VB_XXX.m``; Da Costa et al. 2020
    eq. D.15), sign-flipped to better-when-larger. The ``(α > 0)`` mask is the one SPM's
    ``spm_MDP_VB_X`` applies to ``wA``/``wB``, carried by pymdp's
    ``control.calc_negative_pA_info_gain``. SPM also floors the weights at zero and skips tables
    whose counts all reach 256; this clips at eps instead."""
    safe = jnp.clip(concentration, _MINVAL)
    safe_sum = jnp.clip(safe.sum(axis=child_axis, keepdims=True), _MINVAL)
    w = (
        jnp.log(safe_sum)
        - jnp.log(safe)
        + 1.0 / safe
        - 1.0 / safe_sum
        + digamma(safe)
        - digamma(safe_sum)
    )
    return w * (concentration > 0.0)


def _absorbed(concentration: Array, alpha0: Array | None, stats: Array, lr, decay) -> Array:
    """Shared decayed-count kernel: ``decay·α + (1−decay)·anchor + lr·stats``.

    A static ``decay == 1.0`` takes the pure-conjugate branch ``α + lr·stats``, bitwise identical to
    the no-forgetting update, and never reads the anchor. Any other decay, traced included, needs
    ``alpha0``: the declaration's own concentration (forget toward the prior) or a scalar
    (forget toward a uniform table) are the two usual choices; a declaration with none raises.
    Anchors are validated strictly positive, but a very small anchor with ``decay < 1`` still
    drives ``digamma(α)`` toward ``−∞`` in every expected-log consumer.

    Design: docs/design/parameter-learning.md D7
    """
    if isinstance(decay, int | float) and float(decay) == 1.0:
        return concentration + lr * stats
    if alpha0 is None:
        raise ValueError(
            "a decay other than a static 1.0 on a Dirichlet declaration needs an anchor to "
            "relax toward, alpha0: its own concentration (the prior; one more table of "
            "parameters) or a scalar 1.0 (a uniform table; free), as in "
            "Dirichlet(concentration, alpha0=jnp.asarray(1.0)); ConditionalDirichlet.from_table "
            "takes anchor='prior' or anchor='uniform' (D7)"
        )
    return decay * concentration + (1.0 - decay) * alpha0 + lr * stats


def anchor_for(concentration: Array, anchor: str | bool | Array | None) -> Array | None:
    """The ``alpha0`` a constructor stores for ``anchor``: ``"prior"`` or ``True`` is the
    concentration itself, ``"uniform"`` a scalar ``1.0``, ``False`` or ``None`` nothing, and an
    array is kept as given, in the concentration's dtype."""
    if anchor is None or anchor is False:
        return None
    if anchor is True:
        return concentration
    if isinstance(anchor, str):
        if anchor == "prior":
            return concentration
        if anchor == "uniform":
            return jnp.asarray(1.0, concentration.dtype)
        raise ValueError(
            f"anchor must be 'prior', 'uniform', a bool, an array or None, got {anchor!r}"
        )
    return jnp.asarray(anchor, concentration.dtype)


def _concrete(value: Array):
    """``value`` as a NumPy array when it is concrete, ``None`` under a trace. The check goes
    through NumPy because an array operation inside a trace is staged even on a constant."""
    if isinstance(value, jax.core.Tracer):
        return None
    try:
        return np.asarray(value)
    except jax.errors.TracerArrayConversionError:
        return None


def _check_anchor(alpha0: Array | None, concentration: Array) -> None:
    """Construction-time anchor validation (D7): a scalar or the concentration's shape; values
    when concrete."""
    if alpha0 is None:
        return
    if alpha0.ndim != 0 and alpha0.shape != concentration.shape:
        raise ValueError(
            f"alpha0 (decay anchor) has shape {alpha0.shape}, expected a scalar or the "
            f"concentration shape {concentration.shape}"
        )
    value = _concrete(alpha0)
    if value is not None and not bool(np.all(np.isfinite(value) & (value > 0.0))):
        raise ValueError("alpha0 (decay anchor) must be finite and strictly positive")


def _check_growth(growth: Array | None) -> None:
    """Construction-time check of the growth propensity: a scalar, positive when concrete."""
    if growth is None:
        return
    if growth.ndim != 0:
        raise ValueError(f"growth (the growth propensity α) is a scalar, got shape {growth.shape}")
    value = _concrete(growth)
    if value is not None and not bool(np.isfinite(value) & (value > 0.0)):
        raise ValueError("growth (the growth propensity α) must be finite and strictly positive")


class Dirichlet(ExponentialFamily):
    """``growth`` is the growth propensity ``α`` of a growable factor (structure-learning.md
    S3): the pseudo-count the prior gives a level not yet live. A level with zero concentration
    is absent from the family whatever ``growth`` says; ``growth`` is what lets such a level be
    woken, and without it the level stays absent."""

    concentration: Array  # (..., num_levels)
    alpha0: Array | None = None  # decay anchor (D7): the prior, a scalar (uniform), or absent
    growth: Array | None = None  # the growth propensity α (S3): a scalar, or absent

    def __check_init__(self) -> None:
        _check_anchor(self.alpha0, self.concentration)
        _check_growth(self.growth)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.concentration.shape[:-1]

    @property
    def num_levels(self) -> int:
        return self.concentration.shape[-1]

    def _total(self) -> Array:
        return jnp.sum(self.concentration, axis=-1)

    def _log_beta(self) -> Array:
        return jnp.sum(gammaln(self.concentration), axis=-1) - gammaln(self._total())

    def log_prob(self, x: Array) -> Array:
        return jnp.sum((self.concentration - 1.0) * jnp.log(x), axis=-1) - self._log_beta()

    def sample(self, key: Array, sample_shape: tuple[int, ...] = ()) -> Array:
        return jax.random.dirichlet(key, self.concentration, shape=sample_shape + self.batch_shape)

    def mean(self) -> Array:
        return self.concentration / self._total()[..., None]

    def mode(self) -> Array:
        """Density mode; defined only when all concentrations exceed 1."""
        return (self.concentration - 1.0) / (self._total() - self.num_levels)[..., None]

    def expected_log_params(self) -> Array:
        """E[log p] per level (digamma differences), averaging over the parameters.

        Unnormalized as a log-prior, so an FPI prior takes ``log(mean())`` via :meth:`point`
        instead.
        """
        return digamma(self.concentration) - digamma(self._total())[..., None]

    def point(self) -> Categorical:
        """Posterior-mean probability vector as a :class:`Categorical` (the slot's
        point form).
        """
        return Categorical(logits=jnp.log(self.mean()))

    def stats(self, child_obs: Array) -> Array:
        """One tick's sufficient statistic: onehot of an index, or a probability
        vector (e.g. ``q(s_0)`` for an initial-prior count update) passed through."""
        obs = jnp.asarray(child_obs)
        if jnp.issubdtype(obs.dtype, jnp.integer):
            obs = jax.nn.one_hot(obs, self.num_levels, dtype=self.concentration.dtype)
        return obs

    def absorb(self, stats: Array, lr: float | Array = 1.0, decay: float | Array = 1.0):
        """Absorb one tick's statistic: ``decay·α + (1−decay)·anchor + lr·stats``.

        A static ``decay=1.0`` is the exact conjugate update, bitwise; any other decay needs
        ``alpha0`` (see ``_absorbed``).
        """
        return Dirichlet(
            concentration=_absorbed(self.concentration, self.alpha0, stats, lr, decay),
            alpha0=self.alpha0,
            growth=self.growth,
        )

    def novelty(self) -> Array:
        """Parameter-info-gain weights over the levels (exact ``spm_wnorm``,
        better-when-larger, zero counts masked)."""
        return _wnorm(self.concentration, -1)

    def entropy(self) -> Interval:
        a, a0 = self.concentration, self._total()
        h = (
            self._log_beta()
            + (a0 - self.num_levels) * digamma(a0)
            - jnp.sum((a - 1.0) * digamma(a), axis=-1)
        )
        return Interval.exact(h)

    def kl_to(self, other: "Dirichlet") -> Interval:
        a, b = self.concentration, other.concentration
        a0 = self._total()
        kl = (
            other._log_beta()
            - self._log_beta()
            + jnp.sum((a - b) * (digamma(a) - digamma(a0)[..., None]), axis=-1)
        )
        return Interval.exact(kl)

    def natural(self) -> tuple[Array, ...]:
        return (self.concentration - 1.0,)

    @classmethod
    def from_natural(cls, eta: Array) -> "Dirichlet":
        return cls(concentration=eta + 1.0)
