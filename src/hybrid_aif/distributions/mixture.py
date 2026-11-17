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

import equinox as eqx
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

    #: The components' log weights, ``(..., K)``, up to an additive gauge.
    log_weights: Array
    #: The components' means, ``(..., K, dim)``.
    means: Array
    #: The components' lower-triangular Cholesky factors, ``(..., K, dim, dim)``.
    chols: Array

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

    def reduce(self, k: int, aux=None):
        """Runnalls (2007) greedy reduction to exactly ``k`` components.

        Exactly K−k merge iterations (a fixed trip count). Each merges the valid pair of least cost
        B(i,j) = ½[(w_i+w_j)·logdet Σ_ij − w_i·logdet Σ_i − w_j·logdet Σ_j] (Σ_ij the moment-matched
        merge) into slot min(i,j) and retires slot max(i,j). Dead (``-inf``) components are merged
        first; a surplus of dead components survives as dead output slots. The output is compacted
        to the top-k slots by log-weight (descending, ties by slot index) and renormalized.
        Gradients are defined a.e. but discontinuous at merge-decision boundaries.

        ``aux`` is a per-component side statistic (a pytree of arrays, each with the component
        axis where ``log_weights`` has it, ``(..., K, ...)``): it rides through the reduction by
        the assignment that merges the moments, a merged slot holding its members'
        weight-average, and the call returns ``(mixture, aux)``. It never steers a merge, so the
        moments are those of the call without it (design: parameters-within-the-plan.md D2).
        """
        n = self.num_components
        if not 1 <= k <= n:
            raise ValueError(f"reduce: need 1 <= k <= {n}, got {k}")
        if k == n:
            return self if aux is None else (self, aux)
        nb = len(self.batch_shape)
        if self.batch_shape:
            b = math.prod(self.batch_shape)
            lw = self.log_weights.reshape((b, n))
            means = self.means.reshape((b, n, self.dim))
            chols = self.chols.reshape((b, n, self.dim, self.dim))
            if aux is None:
                out_lw, out_means, out_chols = jax.vmap(partial(_runnalls_reduce, k=k))(
                    lw, means, chols
                )
            else:
                flat = jax.tree.map(lambda a: a.reshape((b, n) + a.shape[nb + 1 :]), aux)
                out_lw, out_means, out_chols, out_aux = jax.vmap(
                    lambda lw, mu, ch, ax: _runnalls_reduce(lw, mu, ch, k=k, aux=ax)
                )(lw, means, chols, flat)
            reduced = MixtureGaussian(
                log_weights=out_lw.reshape(self.batch_shape + (k,)),
                means=out_means.reshape(self.batch_shape + (k, self.dim)),
                chols=out_chols.reshape(self.batch_shape + (k, self.dim, self.dim)),
            )
            if aux is None:
                return reduced
            return reduced, jax.tree.map(
                lambda a: a.reshape(self.batch_shape + (k,) + a.shape[2:]), out_aux
            )
        if aux is None:
            out_lw, out_means, out_chols = _runnalls_reduce(
                self.log_weights, self.means, self.chols, k=k
            )
            return MixtureGaussian(log_weights=out_lw, means=out_means, chols=out_chols)
        out_lw, out_means, out_chols, out_aux = _runnalls_reduce(
            self.log_weights, self.means, self.chols, k=k, aux=aux
        )
        return MixtureGaussian(log_weights=out_lw, means=out_means, chols=out_chols), out_aux

    def reduce_mode_aligned(self, k: int, num_modes: int, aux=None):
        """Mode-aligned grouped Runnalls reduction to ``k`` components (IMM/GPB2).

        On a mixture in the component-major grow layout of
        :func:`~hybrid_aif.inference.mixture_rollout.propagate` (slot ``s`` produced by mode ``s %
        num_modes``), Runnalls runs within each producing-mode group, reducing it to ``k //
        num_modes`` components. The output is contiguous blocks of ``k // num_modes`` slots per
        mode, so the mass of block ``g`` is the posterior mass of mode ``g``. ``k`` and the
        component count must both be multiples of ``num_modes``. ``aux`` is a per-component side
        statistic merged by the same assignment, as :meth:`reduce` takes it; the call then
        returns ``(mixture, aux)``.

        Design: docs/design/mixture-filtering.md D2; parameters-within-the-plan.md D2
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
        nb = len(self.batch_shape)
        if self.batch_shape:
            b = math.prod(self.batch_shape)
            lw = self.log_weights.reshape((b, n))
            means = self.means.reshape((b, n, self.dim))
            chols = self.chols.reshape((b, n, self.dim, self.dim))
            if aux is None:
                out_lw, out_means, out_chols = jax.vmap(
                    partial(_mode_aligned_reduce, k=k, num_modes=num_modes)
                )(lw, means, chols)
            else:
                flat = jax.tree.map(lambda a: a.reshape((b, n) + a.shape[nb + 1 :]), aux)
                out_lw, out_means, out_chols, out_aux = jax.vmap(
                    lambda lw, mu, ch, ax: _mode_aligned_reduce(
                        lw, mu, ch, k=k, num_modes=num_modes, aux=ax
                    )
                )(lw, means, chols, flat)
            reduced = MixtureGaussian(
                log_weights=out_lw.reshape(self.batch_shape + (k,)),
                means=out_means.reshape(self.batch_shape + (k, self.dim)),
                chols=out_chols.reshape(self.batch_shape + (k, self.dim, self.dim)),
            )
            if aux is None:
                return reduced
            return reduced, jax.tree.map(
                lambda a: a.reshape(self.batch_shape + (k,) + a.shape[2:]), out_aux
            )
        if aux is None:
            out_lw, out_means, out_chols = _mode_aligned_reduce(
                self.log_weights, self.means, self.chols, k=k, num_modes=num_modes
            )
            return MixtureGaussian(log_weights=out_lw, means=out_means, chols=out_chols)
        out_lw, out_means, out_chols, out_aux = _mode_aligned_reduce(
            self.log_weights, self.means, self.chols, k=k, num_modes=num_modes, aux=aux
        )
        return MixtureGaussian(log_weights=out_lw, means=out_means, chols=out_chols), out_aux

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


def _merge_ratio(lw_i: Array, lw_j: Array) -> Array:
    """The weight share ``w_i / (w_i + w_j)`` :func:`_merge_moments` merges by (one half for a
    dead-dead pair)."""
    lw_sum = jnp.logaddexp(lw_i, lw_j)
    finite = jnp.isfinite(lw_sum)
    return jnp.where(finite, jnp.exp(lw_i - jnp.where(finite, lw_sum, 0.0)), 0.5)


def _runnalls_merge(log_w: Array, means: Array, chols: Array, k: int, aux=None):
    """Runnalls merge core: n components -> the k live slots, dead slots dropped.

    Returns the k live slots' log-weights in the input gauge (unnormalized) and their covariances
    (not Cholesky factors); merge decisions are gauge- and scale-invariant. Robust to a fully dead
    input. The cost matrix and per-slot log-determinants are updated incrementally (only the merged
    slot's row and column change), so decisions match a full rebuild except on exact ties.

    ``aux`` (a pytree of ``(n, ...)`` arrays, or ``None``) is a per-component side statistic
    merged by the same assignment, a merged slot holding its members' weight-average; it is
    returned as a fourth value, the k slots' entries, and never steers a merge.
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
        log_w, means, covs, ld, alive, cost, aux = state
        dead_pair = jnp.isneginf(log_w)[:, None] | jnp.isneginf(log_w)[None, :]
        valid = upper & alive[:, None] & alive[None, :]
        # Runnalls costs are >= 0, so -1 forces dead slots to be consumed first.
        masked = jnp.where(valid, jnp.where(dead_pair, -1.0, cost), jnp.inf)
        idx = jnp.argmin(masked.reshape(-1))
        i, j = idx // n, idx % n  # upper triangle only, so i < j
        if aux is not None:
            r_i = _merge_ratio(log_w[i], log_w[j])
            aux = jax.tree.map(lambda a: a.at[i].set(r_i * a[i] + (1.0 - r_i) * a[j]), aux)
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
        return log_w, means, covs, ld, alive, cost, aux

    alive = jnp.ones(n, dtype=bool)
    log_w, means, covs, ld, alive, cost, aux = lax.fori_loop(
        0, n - k, body, (log_w, means, covs, ld, alive, cost, aux)
    )
    top_lw, top_idx = lax.top_k(log_w, k)
    if aux is None:
        return top_lw + offset, means[top_idx], covs[top_idx]
    return top_lw + offset, means[top_idx], covs[top_idx], jax.tree.map(lambda a: a[top_idx], aux)


def _runnalls_reduce(log_w: Array, means: Array, chols: Array, k: int, aux=None):
    merged = _runnalls_merge(log_w, means, chols, k, aux)
    top_lw, means, covs = merged[:3]
    out = (jax.nn.log_softmax(top_lw), means, jnp.linalg.cholesky(covs))
    return out if aux is None else (*out, merged[3])


def _mode_aligned_reduce(
    log_w: Array, means: Array, chols: Array, k: int, num_modes: int, aux=None
):
    """Grouped Runnalls: reduce each producing-mode group independently.

    The input is :func:`propagate`'s component-major grow output (slot ``s`` from mode
    ``s % num_modes``); each group is reduced to ``k // num_modes`` components with its total mass
    preserved, then the output is renormalized globally. Output layout: contiguous blocks of
    ``k // num_modes`` slots per mode. ``aux`` (``(n, ...)`` arrays) rides through each group's
    merge (:func:`_runnalls_merge`) and is returned fourth, in the output layout.
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
    if aux is None:
        red_lw, red_means, red_covs = jax.vmap(partial(_runnalls_merge, k=per_out))(
            g_lw - offset[:, None], g_means, g_chols
        )
    else:
        g_aux = jax.tree.map(
            lambda a: jnp.swapaxes(a.reshape((per_in, num_modes) + a.shape[1:]), 0, 1), aux
        )
        red_lw, red_means, red_covs, red_aux = jax.vmap(
            lambda lw, mu, ch, ax: _runnalls_merge(lw, mu, ch, per_out, ax)
        )(g_lw - offset[:, None], g_means, g_chols, g_aux)
    cat_lw = (red_lw + offset[:, None]).reshape(-1)  # restore absolute group mass
    out_means = red_means.reshape(k, -1)
    out_covs = red_covs.reshape(k, *chols.shape[-2:])
    out = (jax.nn.log_softmax(cat_lw), out_means, jnp.linalg.cholesky(out_covs))
    if aux is None:
        return out
    return (*out, jax.tree.map(lambda a: a.reshape((k,) + a.shape[2:]), red_aux))


_LOG_2PI = math.log(2.0 * math.pi)


def _spd_factors(m: Array) -> tuple[Array, Array]:
    """``(M^{-1}, log det M)`` of a batch of small symmetric positive-definite matrices."""
    chol = jnp.linalg.cholesky(m)
    eye = jnp.broadcast_to(jnp.eye(m.shape[-1], dtype=m.dtype), m.shape)
    inv = jax.scipy.linalg.cho_solve((chol, True), eye)
    return inv, 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)


class LowRankMixture(eqx.Module):
    """Gaussian mixture whose component covariances are diagonal plus low rank,
    ``S_i = diag(D_i) + U_i U_iᵀ``, with ``U_i`` of ``r`` columns in a ``p``-dimensional space.

    ``entropy()`` is the Kolchinsky-Tracey pair of :meth:`MixtureGaussian.entropy`, computed
    without forming a ``p`` by ``p`` array: the matrix determinant lemma gives every log
    determinant and the Woodbury identity every trace and quadratic form, from ``r`` by ``r`` and
    ``2r`` by ``2r`` factorisations. The cost is ``O(N² p r²)`` for ``N`` components against
    ``O(N² p³)`` dense.
    """

    #: The components' log weights, ``(..., N)``.
    log_weights: Array
    #: The components' means, ``(..., N, p)``.
    means: Array
    #: The positive diagonal ``D_i`` of each covariance, ``(..., N, p)``.
    diag: Array
    #: The low-rank factors ``U_i``, ``(..., N, p, r)``.
    factors: Array

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.log_weights.shape[:-1]

    @property
    def num_components(self) -> int:
        return self.log_weights.shape[-1]

    @property
    def dim(self) -> int:
        return self.means.shape[-1]

    def _prepare(self) -> tuple[Array, Array, Array]:
        """``(D^{-1}, M^{-1}, log det S)`` per component, ``M = I + Uᵀ D^{-1} U``."""
        dinv = 1.0 / self.diag
        m = _capacitance(self.factors, dinv)
        m_inv, log_det_m = _spd_factors(m)
        return dinv, m_inv, jnp.sum(jnp.log(self.diag), axis=-1) + log_det_m

    def component_entropies(self) -> Array:
        """``H`` of each component, ``½ log|2πe S_i|``, shape ``(..., N)``."""
        _, _, log_det = self._prepare()
        return 0.5 * self.dim * (1.0 + _LOG_2PI) + 0.5 * log_det

    def pairwise_kl(self) -> Array:
        """``KL(component_i ‖ component_j)``, shape ``(..., N, N)``, an exact zero on the
        diagonal.

        ``tr(S_j⁻¹ S_i) - p`` is formed from the differences ``D_i - D_j`` and ``Δ = U_i - U_j``
        as ``tr(S_j⁻¹ (D_i - D_j)) + 2 tr(X_jᵀ Δ) + tr(Δᵀ S_j⁻¹ Δ)``, ``X_j = S_j⁻¹ U_j``,
        which is small for near-coincident components, where subtracting the two traces
        ``tr(S_j⁻¹ S_i)`` and ``p`` loses all precision in single precision."""
        dinv, m_inv, log_det = self._prepare()
        u, d = self.factors, self.diag
        x = jnp.einsum("...jp,...jpa,...jab->...jpb", dinv, u, m_inv)  # S_j^{-1} U_j
        s_diag = dinv - jnp.einsum("...jpa,...jpa->...jp", x, dinv[..., None] * u)
        d_diff = d[..., :, None, :] - d[..., None, :, :]  # (i, j): D_i - D_j
        delta = u[..., :, None, :, :] - u[..., None, :, :, :]  # (i, j): U_i - U_j
        dmu = self.means[..., None, :, :] - self.means[..., :, None, :]  # (i, j): mu_j - mu_i

        def quad(v: Array) -> Array:  # tr(vᵀ S_j^{-1} v), v (..., i, j, p, k)
            proj = jnp.einsum("...jpa,...jp,...ijpk->...ijak", u, dinv, v)
            return jnp.einsum("...jp,...ijpk->...ij", dinv, v**2) - jnp.einsum(
                "...ijak,...jab,...ijbk->...ij", proj, m_inv, proj
            )

        trace_gap = (
            jnp.einsum("...jp,...ijp->...ij", s_diag, d_diff)
            + 2.0 * jnp.einsum("...jpa,...ijpa->...ij", x, delta)
            + quad(delta)
        )
        kl = 0.5 * (
            trace_gap + quad(dmu[..., None]) + log_det[..., None, :] - log_det[..., :, None]
        )
        return jnp.where(jnp.eye(self.num_components, dtype=bool), 0.0, kl)

    def pairwise_bhattacharyya(self) -> Array:
        """Bhattacharyya distance between components, shape ``(..., N, N)``, an exact zero on the
        diagonal."""
        _, _, log_det = self._prepare()
        d_bar = 0.5 * (self.diag[..., :, None, :] + self.diag[..., None, :, :])  # (..., N, N, p)
        e = 1.0 / d_bar
        u_i = jnp.broadcast_to(
            self.factors[..., :, None, :, :], d_bar.shape + (self.factors.shape[-1],)
        )
        u_j = jnp.broadcast_to(self.factors[..., None, :, :, :], u_i.shape)
        w = jnp.concatenate([u_i, u_j], axis=-1) / jnp.sqrt(2.0)  # (..., N, N, p, 2r)
        m_inv, log_det_m = _spd_factors(_capacitance(w, e))
        log_det_bar = jnp.sum(jnp.log(d_bar), axis=-1) + log_det_m
        dmu = self.means[..., :, None, :] - self.means[..., None, :, :]
        proj = jnp.einsum("...ijpa,...ijp->...ija", w, e * dmu)
        maha = jnp.sum(e * dmu**2, axis=-1) - jnp.einsum(
            "...ija,...ijab,...ijb->...ij", proj, m_inv, proj
        )
        dist = (
            0.125 * maha
            + 0.5 * log_det_bar
            - 0.25 * (log_det[..., :, None] + log_det[..., None, :])
        )
        return jnp.where(jnp.eye(self.num_components, dtype=bool), 0.0, dist)

    def entropy(self) -> Interval:
        """Kolchinsky-Tracey pairwise bounds, as :meth:`MixtureGaussian.entropy`."""
        w = jax.nn.log_softmax(self.log_weights, axis=-1)
        comp_entropy = jnp.sum(jnp.exp(w) * self.component_entropies(), axis=-1)

        def bound(pairwise_d: Array) -> Array:
            inner = logsumexp(w[..., None, :] - pairwise_d, axis=-1)
            return comp_entropy - jnp.sum(jnp.exp(w) * inner, axis=-1)

        return Interval(lower=bound(self.pairwise_bhattacharyya()), upper=bound(self.pairwise_kl()))


def _capacitance(u: Array, dinv: Array) -> Array:
    """``I + Uᵀ D^{-1} U`` over the leading axes of ``u`` ``(..., p, r)`` and ``dinv``
    ``(..., p)``."""
    r = u.shape[-1]
    return jnp.eye(r, dtype=u.dtype) + jnp.einsum("...pa,...p,...pb->...ab", u, dinv, u)
