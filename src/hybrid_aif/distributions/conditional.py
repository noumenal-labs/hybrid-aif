"""Conditional families: Markov kernels P(child | parents) with discrete parents.

A conditional holds one child distribution per parent configuration; parent axes are model
structure, not batch axes, and normalization is over the child axis only.

Tabular layout: the trailing structure block is ``(num_levels, *parent_shape)``, child axis
first (the pymdp/SPM layout; SPM's ``spm_dot``/``spm_cross`` convention), which never leaks
because all use goes through methods.
``parent_ndims`` is a static field. Batch axes sit to the left; ``condition_on`` takes unbatched
parent indices and batches via ``vmap``. :class:`ConditionalGaussian` is the non-tabular
sibling, whose parent axes lead the Gaussian's own trailing axes.

Design: docs/design/conditional-families.md
"""

import abc
import string
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import digamma

from hybrid_aif.distributions.base import Distribution
from hybrid_aif.distributions.categorical import Categorical
from hybrid_aif.distributions.dirichlet import (
    Dirichlet,
    _absorbed,
    _check_anchor,
    _check_growth,
    _wnorm,
    anchor_for,
)
from hybrid_aif.distributions.gaussian import Gaussian

if TYPE_CHECKING:
    from hybrid_aif.distributions.joint import JointCategorical

_AXES = string.ascii_lowercase


def _contract_spec(parent_ndims: int) -> str:
    """Einsum spec contracting a child-first table with one belief per parent axis."""
    subs = _AXES[: parent_ndims + 1]
    parents = ",".join("..." + s for s in subs[1:])
    return f"...{subs},{parents}->...{subs[0]}"


def _count_stats(
    num_levels: int, child_obs: Array, parent_probs: tuple[Array, ...], dtype
) -> Array:
    """The conjugate count statistic ``onehot(child_obs) ⊗ q(parent_1) ⊗ ...``
    on a child-first table. The child observation is an integer index or a
    one-hot/probability vector over the child axis.

    The conjugate count update ``α + lr · onehot(o) ⊗ q(parents)`` is pymdp's
    ``learning.update_obs_likelihood_dirichlet_m`` / ``update_state_transition_dirichlet_f``
    (pymdp/learning.py), written as one einsum; pymdp is the parity oracle
    (tests/test_discrete_parity.py, tests/test_learn.py); after SPM's count update in
    ``spm_MDP_VB_X`` (``a + η·spm_cross(o, x)``). SPM also masks the statistic by ``a > 0`` so
    structural zeros stay zero, which pymdp and this kernel do not."""
    obs = jnp.asarray(child_obs)
    if jnp.issubdtype(obs.dtype, jnp.integer):
        obs = jax.nn.one_hot(obs, num_levels, dtype=dtype)
    subs = _AXES[: len(parent_probs) + 1]
    spec = ",".join("..." + s for s in subs) + "->..." + subs
    return jnp.einsum(spec, obs, *parent_probs)


def _count_update(
    concentration: Array, child_obs: Array, parent_probs: tuple[Array, ...], lr: float = 1.0
) -> Array:
    """Array kernel for the conjugate count update on a child-first table:
    ``concentration + lr * _count_stats(...)``, after SPM's count update in ``spm_MDP_VB_X``
    (``a + η·spm_cross(o, x)``); SPM also masks the statistic by ``a > 0`` so structural zeros
    stay zero, which pymdp and this kernel do not."""
    n = len(parent_probs)
    return concentration + lr * _count_stats(
        concentration.shape[-(n + 1)], child_obs, parent_probs, concentration.dtype
    )


class Conditional(eqx.Module):
    """Abstract base for conditional families. See module docstring."""

    @property
    @abc.abstractmethod
    def batch_shape(self) -> tuple[int, ...]:
        """Leading (batch) axes: leaf shape minus parent and family structure."""

    @property
    @abc.abstractmethod
    def parent_shape(self) -> tuple[int, ...]:
        """Sizes of the parent axes (one per discrete parent)."""

    @abc.abstractmethod
    def condition_on(self, parent_indices: tuple[Array | int, ...]) -> Distribution:
        """The child distribution at one parent configuration (unbatched indices)."""


class _TabularConditional(Conditional):
    """Shared shape logic for tabular conditionals (child axis first)."""

    @property
    @abc.abstractmethod
    def _table(self) -> Array:
        """The child-first parameter table ``(..., num_levels, *parent_shape)``."""

    @property
    def _child_axis(self) -> int:
        return -(self.parent_ndims + 1)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._table.shape[: self._table.ndim - self.parent_ndims - 1]

    @property
    def parent_shape(self) -> tuple[int, ...]:
        return self._table.shape[self._table.ndim - self.parent_ndims :]

    @property
    def num_levels(self) -> int:
        return self._table.shape[self._child_axis]

    def _check(self) -> None:
        if self.parent_ndims < 1:
            raise ValueError(f"need parent_ndims >= 1, got {self.parent_ndims}")
        if self._table.ndim < self.parent_ndims + 1:
            raise ValueError(
                f"table rank {self._table.ndim} < parent_ndims + 1 = {self.parent_ndims + 1}"
            )

    def _check_indices(self, parent_indices: tuple[Array | int, ...]) -> tuple[Array, ...]:
        if len(parent_indices) != self.parent_ndims:
            raise ValueError(f"need {self.parent_ndims} parent indices, got {len(parent_indices)}")
        return tuple(jnp.asarray(i, jnp.int32) for i in parent_indices)


class ConditionalCategorical(_TabularConditional):
    """Tabular CPD P(child | parents): one simplex per parent configuration.

    ``log_table`` holds unnormalized log-probabilities with structure block
    ``(num_levels, *parent_shape)``; methods normalize over the child axis, so the stored table
    carries a per-config additive gauge, as ``Categorical.logits`` does.
    """

    log_table: Array  # (..., num_levels, *parent_shape)
    parent_ndims: int = eqx.field(static=True)

    def __check_init__(self) -> None:
        self._check()

    @property
    def _table(self) -> Array:
        return self.log_table

    @property
    def log_probs(self) -> Array:
        """Normalized log P(child | parents), same layout as ``log_table``."""
        return jax.nn.log_softmax(self.log_table, axis=self._child_axis)

    @classmethod
    def from_probs(cls, table: Array, parent_ndims: int) -> "ConditionalCategorical":
        return cls(log_table=jnp.log(table), parent_ndims=parent_ndims)

    @classmethod
    def from_parent_leading(cls, log_table: Array, parent_ndims: int) -> "ConditionalCategorical":
        """Build from the parent-leading layout ``(*parent_shape, num_levels)``.

        Moves the child axis to the front of the structure block; batch axes stay leading and only
        the trailing ``parent_ndims + 1`` axes are re-laid-out. For ``W`` of shape ``(num_actions=3,
        num_modes=2)`` with ``W[u, j] = log P(mode j | action u)``, ``from_parent_leading(W, 1)``
        equals ``ConditionalCategorical(log_table=W.T, parent_ndims=1)``.
        """
        moved = jnp.moveaxis(log_table, -1, -(parent_ndims + 1))
        return cls(log_table=moved, parent_ndims=parent_ndims)

    def expected_log(self, parent_beliefs: tuple[Categorical, ...]) -> Array:
        """E_q(parents)[log P(child | parents)] — the FPI contraction; (..., num_levels)."""
        probs = [b.probs for b in parent_beliefs]
        return jnp.einsum(_contract_spec(self.parent_ndims), self.log_probs, *probs)

    def average(self, parent_beliefs: tuple[Categorical, ...]) -> Categorical:
        """Bayesian model average Σ_parents P(child|parents) q(parents) as a Categorical."""
        probs = [b.probs for b in parent_beliefs]
        table = jax.nn.softmax(self.log_table, axis=self._child_axis)
        avg = jnp.einsum(_contract_spec(self.parent_ndims), table, *probs)
        return Categorical(logits=jnp.log(avg))

    def condition_on(self, parent_indices: tuple[Array | int, ...]) -> Categorical:
        idx = self._check_indices(parent_indices)
        return Categorical(logits=self.log_table[(Ellipsis, *idx)])

    def is_normalized(self, atol: float = 1e-5) -> Array:
        """Whether the *stored* table exponentiates to a simplex per parent config.

        Methods renormalize regardless; this validates externally supplied
        tables. Returns a boolean array over batch axes.
        """
        sums = jnp.exp(self.log_table).sum(axis=self._child_axis)
        return jnp.all(jnp.abs(sums - 1.0) <= atol, axis=tuple(range(-self.parent_ndims, 0)))

    def join(self, parent_marginal: "Categorical | JointCategorical") -> "JointCategorical":
        """Chain-rule product P(child|parents)·P(parents) as a JointCategorical.

        The joint's event axes are ``(num_levels, *parent_shape)`` — child
        axis first, parent axes in table order. ``parent_marginal`` is a
        ``Categorical`` (single parent) or a ``JointCategorical`` whose event
        shape equals ``parent_shape``.
        """
        from hybrid_aif.distributions.joint import JointCategorical

        lp = parent_marginal.log_probs
        marginal_shape = lp.shape[lp.ndim - self.parent_ndims :]
        if marginal_shape != self.parent_shape:
            raise ValueError(f"parent marginal shape {marginal_shape} != {self.parent_shape}")
        joint = self.log_probs + jnp.expand_dims(lp, axis=self._child_axis)
        return JointCategorical(logits=joint, event_ndims=self.parent_ndims + 1)


class ConditionalDirichlet(_TabularConditional):
    """Independent Dirichlet priors over a CPD, one per parent configuration.

    Same child-first layout as :class:`ConditionalCategorical`: a table of per-config Dirichlets on
    the child simplex, not a joint Dirichlet on the product simplex. As a declaration it is the
    learnable form of a tabular CPD slot (a
    :class:`~hybrid_aif.distributions.learnable.Learnable`), placed in ``SystemParams.likelihood`` /
    ``transition`` / ``initial_state_prior`` in place of a point table. Swapping forms changes the
    pytree structure; to stop learning without losing the counts, pass a rate of ``0.0`` to
    ``System.learn``. ``alpha0`` is the decay anchor a ``decay < 1`` update relaxes toward: the
    declaration's own concentration (:meth:`from_table` ``anchor="prior"``), a scalar (a uniform
    table, ``anchor="uniform"``), or absent, in which case decaying the slot raises. ``growth``
    is the growth propensity ``α`` of a growable factor whose weights this table holds (the
    transition column of a memoryless factor; structure-learning.md S3): the pseudo-count the
    prior gives a level not yet live. A level whose concentration is zero is absent from the
    family whatever ``growth`` says; ``growth`` is what lets such a level be woken, and without
    it the level stays absent.

    Design: docs/design/parameter-learning.md D3, D7; docs/design/structure-learning.md S3
    """

    concentration: Array  # (..., num_levels, *parent_shape), positive counts
    parent_ndims: int = eqx.field(static=True)
    alpha0: Array | None = None  # decay anchor (D7): the prior, a scalar (uniform), or absent
    growth: Array | None = None  # the growth propensity α (S3): a scalar, or absent

    def __check_init__(self) -> None:
        self._check()
        _check_anchor(self.alpha0, self.concentration)
        _check_growth(self.growth)

    @property
    def _table(self) -> Array:
        return self.concentration

    def _totals(self) -> Array:
        return self.concentration.sum(axis=self._child_axis, keepdims=True)

    @classmethod
    def from_table(
        cls,
        table: Array,
        parent_ndims: int,
        strength: float = 1.0,
        anchor: str | bool | Array | None = False,
        growth: float | Array | None = None,
    ) -> "ConditionalDirichlet":
        """Concentration ``strength · table`` from a child-normalized probability table.

        ``strength`` is the pseudo-count total per parent configuration (default ``1.0``).
        ``anchor`` is what a ``decay < 1`` update relaxes toward: ``"prior"`` (or ``True``) keeps
        the initial concentration as ``alpha0``, a second table of parameters; ``"uniform"`` keeps
        a scalar ``1.0``, which costs nothing; an array is kept as given; ``False`` keeps none, and
        the slot then cannot decay. ``growth`` is the growth propensity of a growable factor's
        weights (a table with zero rows for its dead levels).
        """
        conc = strength * jnp.asarray(table)
        return cls(
            concentration=conc,
            parent_ndims=parent_ndims,
            alpha0=anchor_for(conc, anchor),
            growth=None if growth is None else jnp.asarray(growth, conc.dtype),
        )

    def expected_log_params(self) -> Array:
        """E[log p] per level and parent config (digamma table) — what FPI consumes
        when averaging over likelihood parameters; same layout as ``concentration``."""
        return digamma(self.concentration) - digamma(self._totals())

    def mean(self) -> Array:
        """Posterior-mean CPD table (normalized over the child axis)."""
        return self.concentration / self._totals()

    def point(self) -> ConditionalCategorical:
        """Posterior-mean CPD as a :class:`ConditionalCategorical` (the slot's
        point form, consumed by ``as_env`` and every probability-kind term)."""
        return ConditionalCategorical(
            log_table=jnp.log(self.mean()), parent_ndims=self.parent_ndims
        )

    def expected_entropy(self) -> Array:
        """Expected child entropy under the parameters, per parent config.

        ``ψ(α₀+1) − Σ_k (α_k/α₀)·ψ(α_k+1)`` with ``α₀`` the config's total: the irreducible part of
        ambiguity. A diagnostic; the term tree does not consume it.

        Design: docs/design/parameter-learning.md D2
        """
        a = self.concentration
        a0 = self._totals()
        return digamma(a0.squeeze(self._child_axis) + 1.0) - jnp.sum(
            (a / a0) * digamma(a + 1.0), axis=self._child_axis
        )

    def stats(self, child_obs: Array, parent_beliefs: tuple[Categorical | Array, ...]) -> Array:
        """One tick's count statistic ``onehot(child_obs) ⊗ q(parent_1) ⊗ ...``.

        ``child_obs`` is an integer index or a one-hot/probability vector over child levels; each
        parent belief is a :class:`Categorical` or a raw probability vector.
        """
        probs = tuple(
            b.probs if isinstance(b, Categorical) else jnp.asarray(b) for b in parent_beliefs
        )
        if len(probs) != self.parent_ndims:
            raise ValueError(f"need {self.parent_ndims} parent beliefs, got {len(probs)}")
        return _count_stats(self.num_levels, child_obs, probs, self.concentration.dtype)

    def absorb(
        self, stats: Array, lr: float | Array = 1.0, decay: float | Array = 1.0
    ) -> "ConditionalDirichlet":
        """Absorb one tick's statistic: ``decay·α + (1−decay)·anchor + lr·stats``.

        A static ``decay == 1.0`` is the pure-conjugate update ``α + lr·stats``, bitwise. Any other
        decay needs ``alpha0``; see ``dirichlet._absorbed`` for small anchors.
        """
        return ConditionalDirichlet(
            concentration=_absorbed(self.concentration, self.alpha0, stats, lr, decay),
            parent_ndims=self.parent_ndims,
            alpha0=self.alpha0,
            growth=self.growth,
        )

    def update(
        self, child_obs: Array, parent_beliefs: tuple[Categorical, ...], lr: float = 1.0
    ) -> "ConditionalDirichlet":
        """Conjugate count update ``concentration + lr·(onehot(child_obs) ⊗ q(parents))``.

        :meth:`stats` then :meth:`absorb`, with no decay.
        """
        return self.absorb(self.stats(child_obs, parent_beliefs), lr=lr)

    def novelty(self) -> Array:
        """Parameter-info-gain weights over the child axis, per parent config
        (exact ``spm_wnorm``, better-when-larger, zero counts masked)."""
        return _wnorm(self.concentration, self._child_axis)

    def sample(self, key: Array) -> Array:
        """Draw a CPD table from the posterior: one independent Dirichlet draw per
        parent configuration, returned in the child-first ``concentration`` layout."""
        conc = jnp.moveaxis(self.concentration, self._child_axis, -1)
        draw = jax.random.dirichlet(key, conc)
        return jnp.moveaxis(draw, -1, self._child_axis)

    def condition_on(self, parent_indices: tuple[Array | int, ...]) -> Dirichlet:
        idx = self._check_indices(parent_indices)
        return Dirichlet(concentration=self.concentration[(Ellipsis, *idx)], growth=self.growth)


class ConditionalGaussian(Conditional):
    """Parent-indexed stack of Gaussians: P(x | parents) with discrete parents.

    Parent axes lead the Gaussian's trailing parameter axes (``loc`` ``(..., *parent_shape, dim)``,
    ``chol`` ``(..., *parent_shape, dim, dim)``). With ``parent_shape == (M,)`` it is the
    mode-indexed innovation family ``N(b_j, Q_j)`` that
    :func:`hybrid_aif.inference.mixture_rollout.propagate` consumes.

    ``chol`` may instead omit the parent axes, ``(..., dim, dim)`` beside a ``loc`` that has them
    (:attr:`shared_chol`): one covariance factor shared by every parent configuration, the mean
    alone depending on the parents. The parent axes are read from ``loc``; a ``chol`` carrying
    some of them is refused.
    """

    loc: Array  # (..., *parent_shape, dim)
    chol: Array  # (..., *parent_shape, dim, dim) or, shared, (..., dim, dim); lower-triangular
    parent_ndims: int = eqx.field(static=True)

    def __check_init__(self) -> None:
        if self.parent_ndims < 1:
            raise ValueError(f"need parent_ndims >= 1, got {self.parent_ndims}")
        if self.loc.ndim < self.parent_ndims + 1:
            raise ValueError(
                f"loc rank {self.loc.ndim} < parent_ndims + 1 = {self.parent_ndims + 1}"
            )
        loc_shape, chol_shape = tuple(self.loc.shape), tuple(self.chol.shape)
        d = loc_shape[-1]
        batch = loc_shape[: len(loc_shape) - self.parent_ndims - 1]
        if chol_shape not in (loc_shape + (d,), batch + (d, d)):
            raise ValueError(
                f"chol has shape {chol_shape} beside a loc of shape {loc_shape} with "
                f"parent_ndims={self.parent_ndims}: chol is {loc_shape + (d,)}, one factor per "
                f"parent configuration, or {batch + (d, d)}, one factor shared by all of them"
                + (
                    " (a chol with only some of the parent axes is not a declaration)"
                    if chol_shape[-2:] == (d, d)
                    else ""
                )
            )

    @property
    def shared_chol(self) -> bool:
        """Whether ``chol`` omits the parent axes: one covariance factor for every parent
        configuration."""
        return self.chol.ndim == self.loc.ndim - self.parent_ndims + 1

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.loc.shape[: self.loc.ndim - self.parent_ndims - 1]

    @property
    def parent_shape(self) -> tuple[int, ...]:
        return self.loc.shape[self.loc.ndim - self.parent_ndims - 1 : -1]

    @property
    def dim(self) -> int:
        return self.loc.shape[-1]

    def stacked_chol(self) -> Array:
        """``chol`` with the parent axes, ``(..., *parent_shape, dim, dim)``: a shared factor
        broadcast over them."""
        if not self.shared_chol:
            return self.chol
        d = self.dim
        shape = (*self.batch_shape, *self.parent_shape, d, d)
        lead = self.chol.reshape((*self.batch_shape, *(1,) * self.parent_ndims, d, d))
        return jnp.broadcast_to(lead, shape)

    def gaussians(self) -> Gaussian:
        """The whole family as one Gaussian with parent axes as trailing batch axes."""
        return Gaussian(loc=self.loc, chol=self.stacked_chol())

    def condition_on(self, parent_indices: tuple[Array | int, ...]) -> Gaussian:
        if len(parent_indices) != self.parent_ndims:
            raise ValueError(f"need {self.parent_ndims} parent indices, got {len(parent_indices)}")
        idx = tuple(jnp.asarray(i, jnp.int32) for i in parent_indices)
        loc = self.loc[(Ellipsis, *idx, slice(None))]
        if self.shared_chol:
            return Gaussian(loc=loc, chol=self.chol)
        return Gaussian(loc=loc, chol=self.chol[(Ellipsis, *idx, slice(None), slice(None))])
