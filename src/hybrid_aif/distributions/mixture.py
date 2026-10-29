"""Gaussian mixture with static component count K.

K is shape-derived and fixed; dead components are ``-inf`` log-weights, never a shape change,
so the object stays jit/vmap/scan-legal. ``entropy()`` returns the Kolchinsky & Tracey (2017)
pairwise-distance bound pair (Bhattacharyya lower, KL upper; gap ≤ H(weights) ≤ log K).
``moment_match()`` is the exact single-Gaussian collapse (GPB1); ``reduce(k)`` is Runnalls
(2007) greedy pairwise reduction to a static budget.

Design: docs/de-risk-report.md §2
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp

from hybrid_aif.distributions.base import Distribution
from hybrid_aif.distributions.gaussian import Gaussian
from hybrid_aif.interval import Interval


class MixtureGaussian(Distribution):
    """Gaussian mixture belief.

    ``log_weights`` are defined up to an additive gauge: every method is gauge-invariant in them and
    may return any gauge (``reduce`` returns normalized weights).
    """

    log_weights: Array  # (..., K)
    means: Array  # (..., K, dim)
    chols: Array  # (..., K, dim, dim), lower-triangular

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.log_weights.shape[:-1]

    @property
    def num_components(self) -> int:
        return self.log_weights.shape[-1]

    @property
    def dim(self) -> int:
        return self.means.shape[-1]

    @property
    def components(self) -> Gaussian:
        """The K components as a Gaussian with a trailing batch axis of size K."""
        return Gaussian(loc=self.means, chol=self.chols)

    @classmethod
    def from_gaussian(cls, g: Gaussian) -> "MixtureGaussian":
        """Promote a Gaussian to the equivalent single-component (K=1) mixture."""
        return cls(
            log_weights=jnp.zeros(g.batch_shape + (1,), g.loc.dtype),
            means=g.loc[..., None, :],
            chols=g.chol[..., None, :, :],
        )

    def log_prob(self, x: Array) -> Array:
        lp = self.components.log_prob(x[..., None, :])  # (..., K)
        return logsumexp(jax.nn.log_softmax(self.log_weights, axis=-1) + lp, axis=-1)

    def sample(self, key: Array, sample_shape: tuple[int, ...] = ()) -> Array:
        key_z, key_x = jax.random.split(key)
        z = jax.random.categorical(
            key_z, self.log_weights, shape=sample_shape + self.batch_shape
        ).astype(jnp.int32)
        eps = jax.random.normal(key_x, sample_shape + self.batch_shape + (self.dim,))
        # Gather the chosen component per draw straight from the K axis (the
        # unit sample axes broadcast inside the gather; nothing is materialised).
        lead = (None,) * len(sample_shape)
        means = jnp.take_along_axis(self.means[lead], z[..., None, None], axis=-2)[..., 0, :]
        chols = jnp.take_along_axis(self.chols[lead], z[..., None, None, None], axis=-3)
        return means + (chols[..., 0, :, :] @ eps[..., None])[..., 0]

    def mean(self) -> Array:
        w = jax.nn.softmax(self.log_weights, axis=-1)
        return jnp.sum(w[..., None] * self.means, axis=-2)

    def cov(self) -> Array:
        """Exact mixture covariance: E[Σ_z] + Cov[μ_z]."""
        w = jax.nn.softmax(self.log_weights, axis=-1)
        covs = self.chols @ jnp.swapaxes(self.chols, -1, -2)
        dev = self.means - self.mean()[..., None, :]
        spread = dev[..., :, None] * dev[..., None, :]
        return jnp.sum(w[..., None, None] * (covs + spread), axis=-3)

    def moment_match(self) -> Gaussian:
        """Exact single-Gaussian collapse (preserves mean and covariance)."""
        return Gaussian.from_cov(loc=self.mean(), cov=self.cov())

    def entropy(self) -> Interval:
        """Kolchinsky–Tracey pairwise bounds: Ĥ_D = Σᵢwᵢ H(pᵢ) − Σᵢwᵢ log Σⱼwⱼ e^{−D(pᵢ‖pⱼ)}.

        D = Bhattacharyya → lower bound; D = KL → upper bound.
        """
        w = jax.nn.log_softmax(self.log_weights, axis=-1)
        comp_entropy = jnp.sum(jnp.exp(w) * self.components.entropy().lower, axis=-1)

        def bound(pairwise_d: Array) -> Array:
            inner = logsumexp(w[..., None, :] - pairwise_d, axis=-1)  # (..., K)
            return comp_entropy - jnp.sum(jnp.exp(w) * inner, axis=-1)

        return Interval(
            lower=bound(self._pairwise_bhattacharyya()),
            upper=bound(self._pairwise_kl()),
        )

    def cross_entropy_to(self, other: "MixtureGaussian | Gaussian") -> Interval:
        """CE(f, g) = −E_f[log g] bounds (Kolchinsky–Tracey §4), f = ``self``.

        - upper (variational): CE ≤ Σ_i w_i H(f_i) − Σ_i w_i log Σ_j v_j e^{−KL(f_i‖g_j)};
        - lower (Jensen): E_{f_i}[−log g] ≥ −log Σ_j v_j Z_ij, with
          Z_ij = ∫ f_i g_j = N(μ_i; ν_j, Σ_i + Λ_j).

        A plain :class:`Gaussian` ``other`` is promoted via :meth:`from_gaussian`. When g has one
        component the upper bound equals the exact cross-entropy and the interval collapses to a
        point, which makes a Gaussian preference's utility over a mixture observation exact.
        """
        if isinstance(other, Gaussian):
            other = MixtureGaussian.from_gaussian(other)
        lw_f = jax.nn.log_softmax(self.log_weights, axis=-1)
        lw_g = jax.nn.log_softmax(other.log_weights, axis=-1)
        w_f = jnp.exp(lw_f)
        gi = Gaussian(loc=self.means[..., :, None, :], chol=self.chols[..., :, None, :, :])
        gj = Gaussian(loc=other.means[..., None, :, :], chol=other.chols[..., None, :, :, :])
        pair_kl = gi.kl_to(gj).lower  # (..., K_f, K_g)
        # log Z_ij: Gaussian density of mu_i - nu_j under N(0, Sigma_i + Lambda_j).
        cov_sum = gi.cov + gj.cov
        chol_sum = jnp.linalg.cholesky(cov_sum)
        dev = self.means[..., :, None, :] - other.means[..., None, :, :]
        z = solve_triangular(chol_sum, dev[..., None], lower=True)[..., 0]
        half_log_det = jnp.sum(jnp.log(jnp.diagonal(chol_sum, axis1=-2, axis2=-1)), axis=-1)
        log_z = -0.5 * (self.dim * jnp.log(2.0 * jnp.pi) + jnp.sum(z**2, axis=-1)) - half_log_det
        comp_h = jnp.sum(w_f * self.components.entropy().lower, axis=-1)
        ce_lower = -jnp.sum(w_f * logsumexp(lw_g[..., None, :] + log_z, axis=-1), axis=-1)
        ce_upper = comp_h - jnp.sum(w_f * logsumexp(lw_g[..., None, :] - pair_kl, axis=-1), axis=-1)
        if other.num_components == 1:  # static: upper bound is the exact CE (docstring)
            ce_lower = ce_upper
        return Interval(lower=ce_lower, upper=ce_upper)

    def kl_to(self, other: "MixtureGaussian | Gaussian") -> Interval:
        """Mixture-mixture KL bounds via KL(f‖g) = CE(f,g) − H(f) (Kolchinsky–Tracey §4).

        Lower = CE_lo − H_hi, clamped at 0; upper = CE_hi − H_lo. With a single-component ``other``
        the width is f's entropy-bound width, zero when f is a single Gaussian.
        """
        ce = self.cross_entropy_to(other)
        h_f = self.entropy()
        return Interval(lower=ce.lower - h_f.upper, upper=ce.upper - h_f.lower).clip_lower(0.0)

    def reduce(self, k: int) -> "MixtureGaussian":
        """Runnalls (2007) greedy reduction to exactly ``k`` components.

        Exactly K−k merge iterations (a fixed trip count). Each merges the valid pair of least cost
        B(i,j) = ½[(w_i+w_j)·logdet Σ_ij − w_i·logdet Σ_i − w_j·logdet Σ_j] (Σ_ij the moment-matched
        merge) into slot min(i,j) and retires slot max(i,j). Dead (``-inf``) components are merged
        first; a surplus of dead components survives as dead output slots. The output is compacted
        to the top-k slots by log-weight (descending, ties by slot index) and renormalized.
        Gradients are defined a.e. but discontinuous at merge-decision boundaries.
        """
        n = self.num_components
        if not 1 <= k <= n:
            raise ValueError(f"reduce: need 1 <= k <= {n}, got {k}")
        if k == n:
            return self
        if self.batch_shape:
            b = math.prod(self.batch_shape)
            lw = self.log_weights.reshape((b, n))
            means = self.means.reshape((b, n, self.dim))
            chols = self.chols.reshape((b, n, self.dim, self.dim))
            out_lw, out_means, out_chols = jax.vmap(partial(_runnalls_reduce, k=k))(
                lw, means, chols
            )
            return MixtureGaussian(
                log_weights=out_lw.reshape(self.batch_shape + (k,)),
                means=out_means.reshape(self.batch_shape + (k, self.dim)),
                chols=out_chols.reshape(self.batch_shape + (k, self.dim, self.dim)),
            )
        out_lw, out_means, out_chols = _runnalls_reduce(
            self.log_weights, self.means, self.chols, k=k
        )
        return MixtureGaussian(log_weights=out_lw, means=out_means, chols=out_chols)

    def reduce_mode_aligned(self, k: int, num_modes: int) -> "MixtureGaussian":
        """Mode-aligned grouped Runnalls reduction to ``k`` components (IMM/GPB2).

        On a mixture in the component-major grow layout of
        :func:`~hybrid_aif.inference.mixture_rollout.propagate` (slot ``s`` produced by mode ``s %
        num_modes``), Runnalls runs within each producing-mode group, reducing it to ``k //
        num_modes`` components. The output is contiguous blocks of ``k // num_modes`` slots per
        mode, so the mass of block ``g`` is the posterior mass of mode ``g``. ``k`` and the
        component count must both be multiples of ``num_modes``.

        Design: docs/design/mixture-filtering.md D2
        """
        n = self.num_components
        if num_modes < 1:
            raise ValueError(f"reduce_mode_aligned: num_modes must be >= 1, got {num_modes}")
        if k % num_modes != 0:
            raise ValueError(
                f"reduce_mode_aligned: k={k} must be a multiple of num_modes={num_modes} so "
                f"each mode owns k//num_modes slots; set mixture_k to a multiple of {num_modes}"
            )
        if n % num_modes != 0:
            raise ValueError(
                f"reduce_mode_aligned: component count {n} must be a multiple of "
                f"num_modes={num_modes} (the component-major grow layout)"
            )
        if not num_modes <= k <= n:
            raise ValueError(f"reduce_mode_aligned: need num_modes <= k <= {n}, got k={k}")
        if self.batch_shape:
            b = math.prod(self.batch_shape)
            lw = self.log_weights.reshape((b, n))
            means = self.means.reshape((b, n, self.dim))
            chols = self.chols.reshape((b, n, self.dim, self.dim))
            out_lw, out_means, out_chols = jax.vmap(
                partial(_mode_aligned_reduce, k=k, num_modes=num_modes)
            )(lw, means, chols)
            return MixtureGaussian(
                log_weights=out_lw.reshape(self.batch_shape + (k,)),
                means=out_means.reshape(self.batch_shape + (k, self.dim)),
                chols=out_chols.reshape(self.batch_shape + (k, self.dim, self.dim)),
            )
        out_lw, out_means, out_chols = _mode_aligned_reduce(
            self.log_weights, self.means, self.chols, k=k, num_modes=num_modes
        )
        return MixtureGaussian(log_weights=out_lw, means=out_means, chols=out_chols)

    def _pairwise_kl(self) -> Array:
        """KL(component_i ‖ component_j), shape (..., K, K)."""
        gi = Gaussian(loc=self.means[..., :, None, :], chol=self.chols[..., :, None, :, :])
        gj = Gaussian(loc=self.means[..., None, :, :], chol=self.chols[..., None, :, :, :])
        return gi.kl_to(gj).lower

    def _pairwise_bhattacharyya(self) -> Array:
        """Bhattacharyya distance between components, shape (..., K, K)."""
        covs = self.chols @ jnp.swapaxes(self.chols, -1, -2)
        cov_bar = 0.5 * (covs[..., :, None, :, :] + covs[..., None, :, :, :])
        chol_bar = jnp.linalg.cholesky(cov_bar)
        dev = self.means[..., :, None, :] - self.means[..., None, :, :]
        z = solve_triangular(chol_bar, dev[..., None], lower=True)[..., 0]
        maha = jnp.sum(z**2, axis=-1)
        half_log_det = jnp.sum(jnp.log(jnp.diagonal(chol_bar, axis1=-2, axis2=-1)), axis=-1)
        comp_half_log_det = jnp.sum(
            jnp.log(jnp.diagonal(self.chols, axis1=-2, axis2=-1)), axis=-1
        )  # (..., K)
        cross = 0.5 * (comp_half_log_det[..., :, None] + comp_half_log_det[..., None, :])
        return 0.125 * maha + half_log_det - cross


def _log_det(covs: Array) -> Array:
    chol = jnp.linalg.cholesky(covs)
    return 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)


def _merge_moments(
    lw_i: Array, lw_j: Array, mu_i: Array, mu_j: Array, cov_i: Array, cov_j: Array
) -> tuple[Array, Array, Array]:
    """Moment-matched pair merge; ratios guarded so dead-dead pairs stay NaN-free."""
    lw_sum = jnp.logaddexp(lw_i, lw_j)
    finite = jnp.isfinite(lw_sum)
    r_i = jnp.where(finite, jnp.exp(lw_i - jnp.where(finite, lw_sum, 0.0)), 0.5)
    r_j = 1.0 - r_i
    mu = r_i[..., None] * mu_i + r_j[..., None] * mu_j
    dev = mu_i - mu_j
    spread = dev[..., :, None] * dev[..., None, :]
    cov = r_i[..., None, None] * cov_i + r_j[..., None, None] * cov_j
    cov = cov + (r_i * r_j)[..., None, None] * spread
    return lw_sum, mu, cov


def _pair_cost(lw_i, lw_j, mu_i, mu_j, cov_i, cov_j, ld_i, ld_j) -> Array:
    """Runnalls' merge cost for one pair, given each side's cached log-determinant."""
    lw_sum, _, cov_m = _merge_moments(lw_i, lw_j, mu_i, mu_j, cov_i, cov_j)
    return 0.5 * (jnp.exp(lw_sum) * _log_det(cov_m) - jnp.exp(lw_i) * ld_i - jnp.exp(lw_j) * ld_j)


def _runnalls_merge(log_w: Array, means: Array, chols: Array, k: int) -> tuple[Array, Array, Array]:
    """Runnalls merge core: n components -> the k live slots, dead slots dropped.

    Returns the k live slots' log-weights in the input gauge (unnormalized) and their covariances
    (not Cholesky factors); merge decisions are gauge- and scale-invariant. Robust to a fully dead
    input. The cost matrix and per-slot log-determinants are updated incrementally (only the merged
    slot's row and column change), so decisions match a full rebuild except on exact ties.
    """
    n = log_w.shape[0]
    covs = chols @ jnp.swapaxes(chols, -1, -2)
    upper = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    peak = jnp.max(log_w)  # -inf only if every slot is dead
    offset = jnp.where(jnp.isfinite(peak), peak, 0.0)
    log_w = log_w - offset  # bound exp() without changing merge decisions or gauge
    ld = _log_det(covs)
    cost = _pair_cost(
        log_w[:, None],
        log_w[None, :],
        means[:, None, :],
        means[None, :, :],
        covs[:, None, :, :],
        covs[None, :, :, :],
        ld[:, None],
        ld[None, :],
    )

    def body(_, state):
        log_w, means, covs, ld, alive, cost = state
        dead_pair = jnp.isneginf(log_w)[:, None] | jnp.isneginf(log_w)[None, :]
        valid = upper & alive[:, None] & alive[None, :]
        # Runnalls costs are >= 0, so -1 forces dead slots to be consumed first.
        masked = jnp.where(valid, jnp.where(dead_pair, -1.0, cost), jnp.inf)
        idx = jnp.argmin(masked.reshape(-1))
        i, j = idx // n, idx % n  # upper triangle only, so i < j
        lw_ij, mu_ij, cov_ij = _merge_moments(
            log_w[i], log_w[j], means[i], means[j], covs[i], covs[j]
        )
        log_w = log_w.at[i].set(lw_ij).at[j].set(-jnp.inf)
        means = means.at[i].set(mu_ij)
        covs = covs.at[i].set(cov_ij)
        ld = ld.at[i].set(_log_det(cov_ij))
        alive = alive.at[j].set(False)
        # only pairs involving the merged slot changed: recompute its row and column
        row = _pair_cost(log_w[i], log_w, means[i], means, covs[i], covs, ld[i], ld)
        cost = cost.at[i, :].set(row).at[:, i].set(row)
        return log_w, means, covs, ld, alive, cost

    alive = jnp.ones(n, dtype=bool)
    log_w, means, covs, ld, alive, cost = lax.fori_loop(
        0, n - k, body, (log_w, means, covs, ld, alive, cost)
    )
    top_lw, top_idx = lax.top_k(log_w, k)
    return top_lw + offset, means[top_idx], covs[top_idx]


def _runnalls_reduce(
    log_w: Array, means: Array, chols: Array, k: int
) -> tuple[Array, Array, Array]:
    top_lw, means, covs = _runnalls_merge(log_w, means, chols, k)
    return jax.nn.log_softmax(top_lw), means, jnp.linalg.cholesky(covs)


def _mode_aligned_reduce(
    log_w: Array, means: Array, chols: Array, k: int, num_modes: int
) -> tuple[Array, Array, Array]:
    """Grouped Runnalls: reduce each producing-mode group independently.

    The input is :func:`propagate`'s component-major grow output (slot ``s`` from mode
    ``s % num_modes``); each group is reduced to ``k // num_modes`` components with its total mass
    preserved, then the output is renormalized globally. Output layout: contiguous blocks of
    ``k // num_modes`` slots per mode.
    """
    n = log_w.shape[0]
    per_out = k // num_modes
    per_in = n // num_modes
    # component-major layout: slot s belongs to mode s % num_modes -> (num_modes, per_in, ...)
    g_lw = log_w.reshape(per_in, num_modes).T
    g_means = jnp.swapaxes(means.reshape(per_in, num_modes, -1), 0, 1)
    g_chols = jnp.swapaxes(chols.reshape(per_in, num_modes, *chols.shape[-2:]), 0, 1)
    mass = logsumexp(g_lw, axis=-1)  # group total log-mass (may be -inf if fully dead)
    offset = jnp.where(jnp.isfinite(mass), mass, 0.0)
    red_lw, red_means, red_covs = jax.vmap(partial(_runnalls_merge, k=per_out))(
        g_lw - offset[:, None], g_means, g_chols
    )
    cat_lw = (red_lw + offset[:, None]).reshape(-1)  # restore absolute group mass
    out_means = red_means.reshape(k, -1)
    out_covs = red_covs.reshape(k, *chols.shape[-2:])
    return jax.nn.log_softmax(cat_lw), out_means, jnp.linalg.cholesky(out_covs)
