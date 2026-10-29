"""Diagonal Gaussian restricted to a box and renormalised: the bounded family of a control node.

Stored parameterization: ``loc`` and ``scale`` (the untruncated Gaussian's mean and standard
deviation, per dimension) and the box ``low``/``high``. Event shape ``(dim,)``; every closed form
factorises over dimensions, since a diagonal Gaussian on a box is a product of one-dimensional
truncated normals. All four leaves share one shape, broadcast at construction.

With ``α = (low − loc)/scale``, ``β = (high − loc)/scale`` and the normaliser
``Z = Φ(β) − Φ(α)``, per dimension:

- density ``φ((x − loc)/scale) / (scale·Z)`` inside the box, zero outside;
- mean ``loc + scale·(φ(α) − φ(β))/Z``;
- variance ``scale²·(1 + (αφ(α) − βφ(β))/Z − ((φ(α) − φ(β))/Z)²)``;
- entropy ``log(√(2πe)·scale·Z) + (αφ(α) − βφ(β))/(2Z)``.

An infinite bound is read 40 standard deviations beyond the other bound (the mass past it is
zero in float32), so a half-bounded or unbounded member has finite moments and gradients.

``log Z`` is computed in log space from ``log_ndtr``, a box in the upper tail reflected onto the
lower one, so a box many standard deviations from ``loc`` keeps a finite normaliser and finite
moments. Sampling is reparameterised through the inverse CDF, inverted in log space.

Design: docs/design/control-states.md C4 (Bounded Gaussian).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import log_ndtr, ndtri

from hybrid_aif.distributions.base import Distribution
from hybrid_aif.interval import Interval

_LOG_2PI = float(jnp.log(2.0 * jnp.pi))
_LOG_2 = float(jnp.log(2.0))
#: Newton steps of the log-space inverse CDF (quadratic convergence from the asymptotic start).
_NEWTON_STEPS = 8
#: Beyond this many standard deviations a box's moments come from the tail forms (see
#: :func:`_tail_moments`); the continued fraction converges to float precision there.
_TAIL = 3.0
_CF_TERMS = 24
#: A box is *narrow* when its half-width ``h`` (in standard deviations) and centre ``c`` satisfy
#: ``h·(1 + |c|) < _NARROW``: its moments then come from the expansion about the midpoint
#: (:func:`_narrow_series`), summed far enough that the neglected terms are below float32
#: precision there.
_NARROW = 1.0
#: Below this half-width (in standard deviations) a draw comes from the exponential tilt
#: ``exp(−c t)`` about the lower bound: the dropped ``t²/2`` is under float32 precision there.
_TILT_HALF_WIDTH = 1e-3
_SERIES_TERMS = 12
#: An infinite bound is read this many standard deviations out (beyond the other bound): the
#: mass past it is zero in float32, and every quantity stays finite and differentiable.
_INFINITE = 40.0


def _log1mexp(x: Array) -> Array:
    """``log(1 − exp(x))`` for ``x < 0``, accurate at both ends."""
    near = jnp.log(-jnp.expm1(jnp.minimum(x, -1e-30)))
    far = jnp.log1p(-jnp.exp(jnp.minimum(x, -_LOG_2)))
    return jnp.where(x > -_LOG_2, near, far)


def log_normaliser(alpha: Array, beta: Array) -> Array:
    """``log(Φ(β) − Φ(α))`` for ``α < β``, elementwise, in log space.

    A box below zero reads the lower tail directly; a box above zero is reflected onto it
    (``Φ(β) − Φ(α) = Φ(−α) − Φ(−β)``); a box straddling zero holds at least the mass between
    its nearer bound and zero and is taken directly. Each branch is evaluated at safe stand-in
    values where it is not selected, so gradients stay finite.
    """
    low_side, high_side = beta <= 0.0, alpha >= 0.0
    mid = ~(low_side | high_side)
    a_l, b_l = jnp.where(low_side, alpha, -2.0), jnp.where(low_side, beta, -1.0)
    a_h, b_h = jnp.where(high_side, alpha, 1.0), jnp.where(high_side, beta, 2.0)
    a_m, b_m = jnp.where(mid, alpha, -1.0), jnp.where(mid, beta, 1.0)
    lower = log_ndtr(b_l) + _log1mexp(log_ndtr(a_l) - log_ndtr(b_l))
    upper = log_ndtr(-a_h) + _log1mexp(log_ndtr(-b_h) - log_ndtr(-a_h))
    middle = jnp.log(jax.scipy.special.ndtr(b_m) - jax.scipy.special.ndtr(a_m))
    return jnp.where(low_side, lower, jnp.where(high_side, upper, middle))


def _log_phi(z: Array) -> Array:
    return -0.5 * z**2 - 0.5 * _LOG_2PI


def _narrow_series(c: Array, q: Array) -> tuple[Array, Array, Array]:
    """Normalised moments of a standard normal on ``[c − h, c + h]`` about its midpoint, as
    series in ``q = h²``: with ``t = z − c`` the weight is ``exp(−c t − t²/2)``, whose Taylor
    coefficients are ``(−1)ⁿ He_n(c)/n!`` (``He`` the probabilists' Hermite polynomials), so
    ``∫ w = 2h·M₀``, ``∫ t w = 2h³·M₁``, ``∫ t² w = 2h³·M₂`` with
    ``M₀ = Σ_k He_{2k} q^k/((2k)!(2k+1))``, ``M₁ = −Σ_k He_{2k+1} q^k/((2k+1)!(2k+3))``,
    ``M₂ = Σ_k He_{2k} q^k/((2k)!(2k+3))``, summed to :data:`_SERIES_TERMS` terms.
    Returns ``(M₀, M₁/M₀, M₂/M₀)``."""
    he = [jnp.ones_like(c), c]
    for n in range(1, 2 * _SERIES_TERMS):
        he.append(c * he[n] - n * he[n - 1])
    m0 = m1 = m2 = jnp.zeros_like(c)
    power, factorial = jnp.ones_like(q), 1.0
    for k in range(_SERIES_TERMS):
        even = factorial  # (2k)!
        odd = factorial * (2 * k + 1)  # (2k+1)!
        m0 = m0 + he[2 * k] * power / (even * (2 * k + 1))
        m1 = m1 - he[2 * k + 1] * power / (odd * (2 * k + 3))
        m2 = m2 + he[2 * k] * power / (even * (2 * k + 3))
        power, factorial = power * q, odd * (2 * k + 2)
    return m0, m1 / m0, m2 / m0


def _mills_fraction(x: Array) -> tuple[Array, Array]:
    """The first two tails ``c₁, c₂`` of the Mills-ratio continued fraction at ``x > 0``:
    ``c_k = k/(x + c_{k+1})``, so that ``(1 − Φ(x))/φ(x) = 1/(x + c₁)`` and
    ``1 − x·c₁ = c₁·c₂`` without cancellation."""
    c = jnp.zeros_like(x)
    c2 = c
    for k in range(_CF_TERMS, 0, -1):
        c = k / (x + c)
        if k == 2:
            c2 = c
    return c, c2


def _tail_moments(alpha: Array, beta: Array) -> tuple[Array, Array, Array]:
    """Mean, variance and entropy of a standard normal on ``[α, β]`` with ``α ≥`` :data:`_TAIL`,
    free of the cancellation the general forms suffer there.

    With ``w = z − α`` on ``[0, L]``, ``A = R(α)``, ``B = ρ·R(β)`` (``R`` the Mills ratio,
    ``ρ = φ(β)/φ(α)``) and ``V_x = c₁(c₂ − c₁)`` the one-sided variance at ``x``:
    ``E[w] = (c₁(α)A − B(c₁(β) + L))/(A − B)``,
    ``Var = ((A − B)(A·V_α − B·V_β) − A·B·(L + c₁(β) − c₁(α))²)/(A − B)²``, and, since
    ``Z = φ(α)(A − B)``, the entropy ``log(A − B) + α E[w] + E[w²]/2``.
    An infinite ``β`` is the one-sided case (``B = 0``).
    """
    c1a, c2a = _mills_fraction(alpha)
    finite = jnp.isfinite(beta)
    b = jnp.where(finite, beta, alpha + 1.0)
    c1b, c2b = _mills_fraction(b)
    width = b - alpha
    a_mills, b_mills = 1.0 / (alpha + c1a), 1.0 / (b + c1b)
    rho = jnp.where(finite, jnp.exp(-width * (alpha + b) / 2.0), 0.0)
    big, small = a_mills, rho * b_mills
    va, vb = c1a * (c2a - c1a), c1b * (c2b - c1b)
    norm = big - small
    offset = (c1a * big - small * (c1b + width)) / norm
    var = (norm * (big * va - small * vb) - big * small * (width + c1b - c1a) ** 2) / norm**2
    entropy = jnp.log(norm) + alpha * offset + 0.5 * (var + offset**2)
    return alpha + offset, var, entropy


def _ndtri_log(log_p: Array) -> Array:
    """``Φ⁻¹(exp(log_p))`` for ``log_p ≤ log ½``, accurate where ``exp(log_p)`` underflows:
    Newton's method on ``log Φ(z) = log_p`` from ``ndtri`` (or the tail asymptote
    ``−√(−2 log_p)`` when ``exp(log_p)`` is not representable)."""
    start = jnp.where(
        log_p > -60.0,
        ndtri(jnp.exp(jnp.maximum(log_p, -60.0))),
        -jnp.sqrt(-2.0 * jnp.minimum(log_p, -60.0)),
    )

    def newton(z, _):
        lz = log_ndtr(z)
        slope = jnp.exp(_log_phi(z) - lz)  # d log Φ / dz
        return z - (lz - log_p) / slope, None

    z, _ = jax.lax.scan(newton, start, None, length=_NEWTON_STEPS)
    return z


class TruncatedGaussian(Distribution):
    """A diagonal Gaussian ``N(loc, diag(scale²))`` restricted to ``[low, high]`` (see module).

    Every leaf is ``(..., dim)``; ``low < high`` elementwise.
    """

    loc: Array
    scale: Array
    low: Array
    high: Array

    def __init__(self, loc: Array, scale: Array, low: Array, high: Array):
        arrays = [jnp.asarray(v, jnp.float32) for v in (loc, scale, low, high)]
        shape = jnp.broadcast_shapes(*(a.shape for a in arrays))
        if not shape:
            raise ValueError(
                "TruncatedGaussian takes (..., dim) arrays: give loc, scale, low and high a "
                "trailing dimension axis"
            )
        self.loc, self.scale, self.low, self.high = (jnp.broadcast_to(a, shape) for a in arrays)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.loc.shape[:-1]

    @property
    def dim(self) -> int:
        return self.loc.shape[-1]

    # -------------------------------------------------------------- per-dimension pieces

    def _alpha_beta(self) -> tuple[Array, Array]:
        """``(α, β)`` per dimension, an infinite bound read :data:`_INFINITE` standard
        deviations beyond the other (both substituted before dividing, so gradients stay
        finite)."""
        fl, fh = jnp.isfinite(self.low), jnp.isfinite(self.high)
        low = jnp.where(fl, self.low, self.loc)
        high = jnp.where(fh, self.high, self.loc)
        alpha = (low - self.loc) / self.scale
        beta = (high - self.loc) / self.scale
        alpha = jnp.where(fl, alpha, jnp.minimum(-_INFINITE, jnp.where(fh, beta, 0.0) - _INFINITE))
        beta = jnp.where(fh, beta, jnp.maximum(_INFINITE, alpha + _INFINITE))
        return alpha, beta

    def _narrow(self) -> tuple[Array, Array, Array, Array, Array]:
        """``(narrow, c, q, M₀, M₁/M₀, M₂/M₀)`` pieces of the midpoint expansion; the series
        is evaluated at stand-in values where the box is not narrow."""
        alpha, beta = self._alpha_beta()
        c, h = 0.5 * (alpha + beta), 0.5 * (beta - alpha)
        narrow = h * (1.0 + jnp.abs(c)) < _NARROW
        c, h = jnp.where(narrow, c, 0.0), jnp.where(narrow, h, 0.01)
        m0, r1, r2 = _narrow_series(c, h**2)
        return narrow, c, h, m0, r1, r2

    def _standard(self) -> tuple[Array, Array, Array]:
        """``(α, β, log Z)`` per dimension; a narrow box's ``log Z`` is
        ``log φ(c) + log 2h + log M₀`` (:func:`_narrow_series`)."""
        alpha, beta = self._alpha_beta()
        narrow, c, h, m0, _, _ = self._narrow()
        wide = log_normaliser(jnp.where(narrow, -1.0, alpha), jnp.where(narrow, 1.0, beta))
        log_z = jnp.where(narrow, _log_phi(c) + jnp.log(2.0 * h) + jnp.log(m0), wide)
        return alpha, beta, log_z

    def _ratios(self) -> tuple[Array, Array, Array, Array]:
        """``(φ(α)/Z, φ(β)/Z, αφ(α)/Z, βφ(β)/Z)`` per dimension."""
        alpha, beta, log_z = self._standard()
        ra = jnp.exp(_log_phi(alpha) - log_z)
        rb = jnp.exp(_log_phi(beta) - log_z)
        return ra, rb, alpha * ra, beta * rb

    def log_normalisers(self) -> Array:
        """``log Z`` per dimension, ``(..., dim)``."""
        return self._standard()[2]

    def _standard_moments(self) -> tuple[Array, Array]:
        """Mean and variance of the standardised variable, per dimension: the general forms
        near ``loc``, the tail forms (:func:`_tail_moments`, a lower-tail box reflected) beyond
        :data:`_TAIL` standard deviations. Narrow boxes are handled by :meth:`mean` and
        :meth:`variance` in the box's own units."""
        alpha, beta = self._alpha_beta()
        ra, rb, a_ra, b_rb = self._ratios()
        mean, var = ra - rb, 1.0 + a_ra - b_rb - (ra - rb) ** 2
        upper, lower = alpha >= _TAIL, beta <= -_TAIL
        up_mean, up_var, _ = _tail_moments(
            jnp.where(upper, alpha, _TAIL), jnp.where(upper, beta, _TAIL + 1.0)
        )
        lo_mean, lo_var, _ = _tail_moments(
            jnp.where(lower, -beta, _TAIL), jnp.where(lower, -alpha, _TAIL + 1.0)
        )
        mean = jnp.where(upper, up_mean, jnp.where(lower, -lo_mean, mean))
        var = jnp.where(upper, up_var, jnp.where(lower, lo_var, var))
        return mean, var

    def _half_width(self) -> Array:
        """Half the box's width in its own units (a stand-in where it is unbounded)."""
        finite = jnp.isfinite(self.low) & jnp.isfinite(self.high)
        return jnp.where(finite, 0.5 * (jnp.where(finite, self.high - self.low, 0.0)), 1.0)

    def mean(self) -> Array:
        """The mean; a narrow box's is ``mid + half·h·M₁/M₀`` about its midpoint. Clamped into
        the box as a guard."""
        mean_std, _ = self._standard_moments()
        narrow, _, h, _, r1, _ = self._narrow()
        half = self._half_width()
        mid = jnp.where(narrow, self.low + half, 0.0)
        mean = jnp.where(narrow, mid + half * h * r1, self.loc + self.scale * mean_std)
        return jnp.clip(mean, self.low, self.high)

    def variance(self) -> Array:
        """Per-dimension variance, ``(..., dim)`` (the covariance is diagonal); a narrow box's
        is ``half²·(M₂/M₀ − h²(M₁/M₀)²)``, ``w²/12`` in the uniform limit. Clamped into
        ``(0, w²/4]`` as a guard."""
        _, var_std = self._standard_moments()
        narrow, _, h, _, r1, r2 = self._narrow()
        half = self._half_width()
        var = jnp.where(narrow, half**2 * (r2 - h**2 * r1**2), self.scale**2 * var_std)
        bounded = jnp.isfinite(self.low) & jnp.isfinite(self.high)
        cap = jnp.where(bounded, half**2, jnp.inf)
        return jnp.clip(var, jnp.finfo(jnp.float32).tiny, cap)

    @property
    def cov(self) -> Array:
        """The diagonal covariance as a matrix, ``(..., dim, dim)``."""
        var = self.variance()
        return var[..., :, None] * jnp.eye(self.dim, dtype=var.dtype)

    # ------------------------------------------------------------------ the contract

    def log_prob(self, x: Array) -> Array:
        """Log density at ``x``, summed over dimensions; ``−inf`` outside the box."""
        x = jnp.asarray(x, jnp.float32)
        z = (x - self.loc) / self.scale
        per_dim = _log_phi(z) - jnp.log(self.scale) - self.log_normalisers()
        inside = jnp.all((x >= self.low) & (x <= self.high), axis=-1)
        return jnp.where(inside, jnp.sum(per_dim, axis=-1), -jnp.inf)

    def sample(self, key: Array, sample_shape: tuple[int, ...] = ()) -> Array:
        """Reparameterised draws through the inverse CDF: ``u ~ U(0, 1)`` per dimension and
        ``x = loc + scale·Φ⁻¹(Φ(α) + u·Z)``, with ``Φ(α) + u·Z`` formed in log space and a box
        in the upper tail drawn from its reflection. Draws are clipped into the box against
        rounding. A box narrower than :data:`_TILT_HALF_WIDTH` standard deviations, where the
        inverse CDF runs out of float32 resolution and ``t²/2`` is below it, is drawn from its
        exponential tilt about the lower bound."""
        u = jax.random.uniform(key, sample_shape + self.loc.shape, jnp.float32)
        u = jnp.clip(u, 1e-7, 1.0 - 1e-7)
        narrow, c, h, _, _, _ = self._narrow()
        narrow = narrow & (h < _TILT_HALF_WIDTH)
        tilted = jnp.abs(c) > 1e-6
        a = jnp.where(tilted, c, 1.0)
        s_std = jnp.where(tilted, -jnp.log1p(u * jnp.expm1(-2.0 * a * h)) / a, 2.0 * h * u)
        thin = self.low + self.scale * s_std
        alpha, beta, log_z = self._standard()
        flip = alpha > 0.0
        lo = jnp.where(flip, -beta, alpha)
        log_p = jnp.logaddexp(log_ndtr(lo), jnp.log(u) + log_z)
        #  below Φ = ½ invert directly; above, invert the complement (less rounding)
        upper = log_p > -_LOG_2
        log_q = _log1mexp(jnp.where(upper, log_p, -1.0))
        z = jnp.where(upper, -_ndtri_log(log_q), _ndtri_log(jnp.where(upper, -1.0, log_p)))
        z = jnp.where(flip, -z, z)
        draw = jnp.where(narrow, jnp.where(narrow, thin, 0.0), self.loc + self.scale * z)
        return jnp.clip(draw, self.low, self.high)

    def entropy(self) -> Interval:
        """The differential entropy, summed over dimensions; a box beyond :data:`_TAIL`
        standard deviations takes the tail form (:func:`_tail_moments`)."""
        alpha, beta = self._alpha_beta()
        _, _, a_ra, b_rb = self._ratios()
        standard = 0.5 * (1.0 + _LOG_2PI) + self.log_normalisers() + 0.5 * (a_ra - b_rb)
        narrow, c, h, m0, r1, r2 = self._narrow()
        #  narrow: log 2h + log M₀ + c·E[t] + E[t²]/2, with t = z − c
        thin = jnp.log(2.0 * h) + jnp.log(m0) + c * h**2 * r1 + 0.5 * h**2 * r2
        upper, lower = alpha >= _TAIL, beta <= -_TAIL
        _, _, up = _tail_moments(
            jnp.where(upper, alpha, _TAIL), jnp.where(upper, beta, _TAIL + 1.0)
        )
        _, _, lo = _tail_moments(
            jnp.where(lower, -beta, _TAIL), jnp.where(lower, -alpha, _TAIL + 1.0)
        )
        standard = jnp.where(upper, up, jnp.where(lower, lo, standard))
        standard = jnp.where(narrow, thin, standard)
        return Interval.exact(jnp.sum(standard + jnp.log(self.scale), axis=-1))

    def expected_square(self, centre: Array) -> Array:
        """``E[(x − centre)²]`` per dimension, from the closed mean and variance."""
        return self.variance() + (self.mean() - centre) ** 2

    def kl_to(self, other: "TruncatedGaussian") -> Interval:
        """``KL(self ‖ other)`` for two members on the same box, in closed form:
        ``−H[self] − E_self[log other]``, the cross term read off the closed first two moments.

        Members on different boxes raise (eagerly, or under a trace by ``equinox.error_if``):
        outside a shared box the KL is infinite or undefined.
        """
        if not isinstance(other, TruncatedGaussian):
            raise TypeError(
                f"TruncatedGaussian.kl_to takes another TruncatedGaussian on the same box, got "
                f"{type(other).__name__}"
            )
        self = _check_same_box(self, other)  # noqa: PLW0642
        per_dim = (
            -0.5 * _LOG_2PI
            - jnp.log(other.scale)
            - other.log_normalisers()
            - 0.5 * self.expected_square(other.loc) / other.scale**2
        )
        return Interval.exact(-self.entropy().lower - jnp.sum(per_dim, axis=-1))


def _check_same_box(p: TruncatedGaussian, q: TruncatedGaussian) -> TruncatedGaussian:
    """``p``, checked to share ``q``'s box: concrete bounds raise here, traced ones when the
    computation runs (``equinox.error_if``)."""
    if p.dim != q.dim:
        raise ValueError(
            f"KL between truncated Gaussians of dimension {p.dim} and {q.dim}: pass two members "
            "on the same box"
        )
    sentence = (
        "KL between truncated Gaussians on different boxes is not finite: pass two members on "
        "the same box"
    )
    same = jnp.all(p.low == q.low) & jnp.all(p.high == q.high)
    if not isinstance(same, jax.core.Tracer):
        if not bool(same):
            raise ValueError(sentence)
        return p
    return TruncatedGaussian(p.loc, p.scale, eqx.error_if(p.low, ~same, sentence), p.high)
