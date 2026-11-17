"""Linear-Gaussian conditioning and prediction kernels over :class:`Gaussian` objects.

Cholesky-based throughout, with no explicit inverse: covariances are sums of Gram factors (SPD
by construction) and re-factored, or square-root factors from a QR decomposition. Everything
broadcasts over leading batch axes.

A reading ``y = H x + c + eps`` updates a belief over ``x`` (dimension ``d``) in one of two forms,
chosen by the noise's type. A noise given as a Cholesky factor ``R_chol`` (any ``(d_obs, d_obs)``
lower-triangular array) takes the update in observation space: the innovation covariance
``H Σ Hᵀ + R`` is formed and factored, ``O(d_obs³)``. A noise given as a :class:`DiagonalNoise`
(independent readings) takes it in state space, the square-root information form: no
``d_obs`` by ``d_obs`` array is formed, and the cost is ``O(d_obs d²)``. The two are the same
mathematics. In floating point the state-space form is the more accurate where the readings are
precise against the belief: it decomposes ``[diag(r)⁻¹ H L; I]``, while the observation-space form
factors ``H Σ Hᵀ + R``, whose condition number is that matrix's squared. Design:
docs/design/recognition-beside-analytic.md L10.
"""

import math
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import cho_solve, solve_triangular

from hybrid_aif.distributions.gaussian import Gaussian

_LOG_2PI = math.log(2.0 * math.pi)


def _tr(a: Array) -> Array:
    return jnp.swapaxes(a, -1, -2)


class DiagonalNoise(NamedTuple):
    """An observation noise of independent readings, ``N(0, diag(scale²))``, in the form the
    state-space update reads.

    ``scale`` ``(..., d_obs)`` holds the readings' standard deviations, which must be strictly
    positive (a zero or negative value is not checked and gives an infinite or wrong log
    determinant). ``low_rank`` ``(..., d_obs, k)``, when given, adds a correlated part of rank at
    most ``k``: the noise covariance is then ``diag(scale²) + low_rank low_rankᵀ`` (the residual
    covariance of an unscented reading, :func:`~hybrid_aif.inference.linearise.linear_model`).
    The update treats that part as ``k`` further standard-normal latent coordinates read through
    ``low_rank``, so it stays in state space at dimension ``d + k``. The update checks the shapes
    (statically, at no cost under ``jit``): ``scale``'s last axis is the reading's length, its
    batch axes are no more than the head's (a Cholesky factor passed as ``scale`` is refused), and
    ``low_rank``'s ``d_obs`` axis matches ``scale``'s.
    """

    scale: Array
    low_rank: Array | None = None

    def cov(self) -> Array:
        """The dense ``(..., d_obs, d_obs)`` covariance."""
        d_obs = self.scale.shape[-1]
        cov = self.scale[..., :, None] ** 2 * jnp.eye(d_obs, dtype=self.scale.dtype)
        if self.low_rank is not None:
            cov = cov + self.low_rank @ _tr(self.low_rank)
        return cov

    def chol(self) -> Array:
        """The dense lower Cholesky factor of :meth:`cov`: ``diag(scale)`` without ``low_rank``."""
        if self.low_rank is None:
            d_obs = self.scale.shape[-1]
            return self.scale[..., :, None] * jnp.eye(d_obs, dtype=self.scale.dtype)
        return jnp.linalg.cholesky(self.cov())


def predict(g: Gaussian, F: Array, b: Array, Q_chol: Array) -> Gaussian:
    """Push ``g`` through ``x' = F x + b + eps``, ``eps ~ N(0, Q_chol Q_chol^T)``.

    ``F`` (..., dim, dim), ``b`` (..., dim), ``Q_chol`` (..., dim, r): any factor with ``Q =
    Q_chol Q_cholᵀ``, read only through that product; batch axes of ``g`` and the system matrices
    broadcast against each other.
    """
    loc = (F @ g.loc[..., None])[..., 0] + b
    FL = F @ g.chol
    cov = FL @ _tr(FL) + Q_chol @ _tr(Q_chol)
    chol = jnp.linalg.cholesky(cov)
    batch = jnp.broadcast_shapes(loc.shape[:-1], chol.shape[:-2])
    dim = loc.shape[-1]
    return Gaussian(
        loc=jnp.broadcast_to(loc, batch + (dim,)),
        chol=jnp.broadcast_to(chol, batch + (dim, dim)),
    )


def _gain_and_factor(g: Gaussian, H: Array, R_chol: Array) -> tuple[Array, Array, Array, Array]:
    """The innovation Cholesky factor, the gain ``K = Σ Hᵀ S⁻¹``, the posterior covariance's
    Cholesky factor (Joseph form from Gram factors) and ``H L``, for a reading ``y = H x + c +
    eps`` with ``eps ~ N(0, R_chol R_cholᵀ)``; none depends on ``y`` or ``c``."""
    HL = H @ g.chol
    S = HL @ _tr(HL) + R_chol @ _tr(R_chol)
    S_chol = jnp.linalg.cholesky(S)
    # Gain K = Sigma H^T S^{-1}, via S X = (Sigma H^T)^T then K = X^T.
    PHt = g.chol @ _tr(HL)
    gain = _tr(cho_solve((S_chol, True), _tr(PHt)))
    AL = g.chol - gain @ HL  # (I - K H) L
    KR = gain @ R_chol
    chol = jnp.linalg.cholesky(AL @ _tr(AL) + KR @ _tr(KR))
    return S_chol, gain, chol, HL


def _condition_dense(
    g: Gaussian, H: Array, c: Array, R_chol: Array, y: Array
) -> tuple[Gaussian, Array]:
    """:func:`condition` in observation space, for a noise given as its Cholesky factor."""
    dim_y = y.shape[-1]
    v = y - ((H @ g.loc[..., None])[..., 0] + c)
    S_chol, gain, chol, _ = _gain_and_factor(g, H, R_chol)
    loc = g.loc + (gain @ v[..., None])[..., 0]
    z = solve_triangular(S_chol, v[..., None], lower=True)[..., 0]
    half_log_det = jnp.sum(jnp.log(jnp.diagonal(S_chol, axis1=-2, axis2=-1)), axis=-1)
    log_marginal = -0.5 * (dim_y * _LOG_2PI + jnp.sum(z**2, axis=-1)) - half_log_det
    batch = jnp.broadcast_shapes(loc.shape[:-1], chol.shape[:-2])
    dim = loc.shape[-1]
    posterior = Gaussian(
        loc=jnp.broadcast_to(loc, batch + (dim,)),
        chol=jnp.broadcast_to(chol, batch + (dim, dim)),
    )
    return posterior, jnp.broadcast_to(log_marginal, batch)


def _check_noise(noise: DiagonalNoise, H: Array, d_obs: int | None = None) -> None:
    """Refuse a :class:`DiagonalNoise` whose shapes do not fit the head ``H`` ``(..., d_obs,
    d)`` (and the reading's length ``d_obs`` where it is known): static shapes only."""
    scale = noise.scale
    rows = H.shape[-2]
    if scale.shape[-1] != rows or (d_obs is not None and d_obs != rows):
        raise ValueError(
            f"DiagonalNoise: the scale has {scale.shape[-1]} readings on its last axis, the head "
            f"{rows} rows"
            + ("" if d_obs is None else f" and the reading {d_obs} values")
            + "; they must be equal"
        )
    if scale.ndim > H.ndim - 1:
        raise ValueError(
            f"DiagonalNoise: the scale has shape {scale.shape}, more batch axes than the head "
            f"{H.shape}; the scale is the readings' standard deviations, (..., d_obs), not a "
            "Cholesky factor (pass a Cholesky factor as the noise itself for the dense update)"
        )
    if noise.low_rank is not None and noise.low_rank.shape[-2] != scale.shape[-1]:
        raise ValueError(
            f"DiagonalNoise: low_rank has shape {noise.low_rank.shape}, whose d_obs axis (the "
            f"second last) must equal the scale's {scale.shape[-1]} readings"
        )


class _Information(NamedTuple):
    """The state-space quantities of one diagonal-noise update (:func:`_information`)."""

    B: Array  # (..., d_obs, n): the whitened head in the prior's square-root coordinates
    Q: Array  # (..., d_obs, n): the leading rows of the orthonormal factor, [B; I] = [Q; Q'] T
    T: Array  # (..., n, n): upper triangular, Tᵀ T = I + Bᵀ B
    chol: Array  # (..., d, d): the posterior covariance's lower Cholesky factor


def _information(g: Gaussian, H: Array, noise: DiagonalNoise, batch: tuple[int, ...]):
    """The square-root information quantities of a reading ``y = H x + c + eps``, ``eps ~
    N(0, noise.cov())``, on the belief ``g = N(m, L Lᵀ)``, broadcast to ``batch``.

    In the prior's square-root coordinates ``x = m + L xi``, ``xi ~ N(0, I)``, the whitened
    reading is ``w = B xi + e`` with ``B = diag(scale)⁻¹ H L`` and ``e ~ N(0, I)``. A ``low_rank``
    ``V`` appends ``k`` latent coordinates ``zeta ~ N(0, I)`` read through it, ``B = diag(scale)⁻¹
    [H L, V]``, and the update below runs at dimension ``n = d + k``. The QR decomposition of the
    stacked ``[B; I]`` (``(d_obs + n, n)``) gives ``T`` with ``Tᵀ T = I + Bᵀ B``, the posterior
    precision of ``xi``, without squaring ``B``'s condition number. The posterior covariance of
    ``x`` is ``[L, 0] T⁻¹ T⁻ᵀ [L, 0]ᵀ``; its lower Cholesky factor is the transposed triangular
    factor of a second QR, of ``T⁻ᵀ [L, 0]ᵀ`` (``(n, d)``), with its diagonal made positive.
    """
    L = g.chol
    d = L.shape[-1]
    d_obs = noise.scale.shape[-1]
    cols = H @ L
    if noise.low_rank is not None:
        V = noise.low_rank
        shape = jnp.broadcast_shapes(cols.shape[:-2], V.shape[:-2])
        cols = jnp.concatenate(
            [
                jnp.broadcast_to(cols, shape + cols.shape[-2:]),
                jnp.broadcast_to(V, shape + V.shape[-2:]),
            ],
            axis=-1,
        )
    n = cols.shape[-1]
    B = jnp.broadcast_to(cols / noise.scale[..., :, None], batch + (d_obs, n))
    eye = jnp.broadcast_to(jnp.eye(n, dtype=B.dtype), batch + (n, n))
    Q, T = jnp.linalg.qr(jnp.concatenate([B, eye], axis=-2))
    L_pad = jnp.broadcast_to(L, batch + (d, d))
    if n > d:
        L_pad = jnp.concatenate([L_pad, jnp.zeros(batch + (d, n - d), L.dtype)], axis=-1)
    root_t = solve_triangular(T, _tr(L_pad), trans=1, lower=False)  # T⁻ᵀ [L, 0]ᵀ, (n, d)
    _, R2 = jnp.linalg.qr(root_t)
    sign = jnp.where(jnp.diagonal(R2, axis1=-2, axis2=-1) < 0, -1.0, 1.0).astype(R2.dtype)
    chol = _tr(R2) * sign[..., None, :]
    return _Information(B=B, Q=Q[..., :d_obs, :], T=T, chol=chol)


def _condition_diagonal(
    g: Gaussian, H: Array, c: Array, noise: DiagonalNoise, y: Array
) -> tuple[Gaussian, Array]:
    """:func:`condition` in state space, for a :class:`DiagonalNoise`.

    With ``w = diag(scale)⁻¹ (y - H m - c)`` the whitened innovation and ``B``, ``Q``, ``T`` as in
    :func:`_information`, the posterior mean of ``xi`` is ``s = (I + Bᵀ B)⁻¹ Bᵀ w``, the solution
    of the least squares ``min ‖w - B xi‖² + ‖xi‖²``, taken from the same QR decomposition as
    ``s = T⁻¹ Qᵀ w`` (not through ``Bᵀ w``, whose rounding would leak into the directions the
    reading does not inform when ``B`` is large), and the mean of ``x`` is ``m + [L, 0] s``. The
    log evidence is ``log N(y; H m + c, S)`` with
    ``S = H L Lᵀ Hᵀ + noise.cov()``, by the matrix determinant lemma ``log|S| = 2 Σ log scale +
    2 Σ log|T_ii|`` and the Woodbury identity ``(y - H m - c)ᵀ S⁻¹ (y - H m - c) = ‖w‖² - wᵀ B (I +
    Bᵀ B)⁻¹ Bᵀ w = ‖w - B s‖² + ‖s‖²`` (the minimum of the regularised least squares, a sum of
    two non-negative terms, so the subtraction is never taken).
    """
    d, d_obs = g.loc.shape[-1], y.shape[-1]
    _check_noise(noise, H, d_obs)
    v = y - ((H @ g.loc[..., None])[..., 0] + c)
    w = v / noise.scale
    batch = jnp.broadcast_shapes(
        g.loc.shape[:-1],
        g.chol.shape[:-2],
        H.shape[:-2],
        noise.scale.shape[:-1],
        w.shape[:-1],
        () if noise.low_rank is None else noise.low_rank.shape[:-2],
    )
    info = _information(g, H, noise, batch)
    w = jnp.broadcast_to(w, batch + (d_obs,))
    z = _tr(info.Q) @ w[..., None]
    s = solve_triangular(info.T, z, lower=False)[..., 0]
    loc = g.loc + (g.chol @ s[..., :d, None])[..., 0]
    resid = w - (info.B @ s[..., None])[..., 0]
    quad = jnp.sum(resid**2, axis=-1) + jnp.sum(s**2, axis=-1)
    half_log_det = jnp.sum(jnp.log(noise.scale), axis=-1) + jnp.sum(
        jnp.log(jnp.abs(jnp.diagonal(info.T, axis1=-2, axis2=-1))), axis=-1
    )
    log_marginal = -0.5 * (d_obs * _LOG_2PI + quad) - half_log_det
    posterior = Gaussian(loc=jnp.broadcast_to(loc, batch + (d,)), chol=info.chol)
    return posterior, jnp.broadcast_to(log_marginal, batch)


def condition(
    g: Gaussian, H: Array, c: Array, noise: Array | DiagonalNoise, y: Array
) -> tuple[Gaussian, Array]:
    """Bayesian update of ``g`` on ``y = H x + c + eps``, ``eps ~ N(0, R)``.

    Returns ``(posterior, log_marginal)``, ``log_marginal = log p(y)``. ``noise`` is the noise:
    a lower-triangular Cholesky factor ``R_chol`` with ``R = R_chol R_cholᵀ``, or a
    :class:`DiagonalNoise`.

    - A Cholesky factor takes the update in observation space: ``log p(y)`` from the innovation
      covariance's Cholesky factor, and the covariance from the Joseph form in Gram factors, so the
      posterior stays SPD for near-singular gains. ``O(d_obs³ + d_obs² d)``.
    - A :class:`DiagonalNoise` takes it in state space (:func:`_condition_diagonal`): the
      square-root information form, with the log evidence from the matrix determinant lemma and
      the Woodbury identity. No ``d_obs`` by ``d_obs`` array is formed; ``O(d_obs n²)`` with
      ``n = d``, or ``d`` plus the number of columns of the noise's ``low_rank`` when it has one.
    """
    if isinstance(noise, DiagonalNoise):
        return _condition_diagonal(g, H, c, noise, y)
    return _condition_dense(g, H, c, noise, y)


def condition_on_expected(g: Gaussian, H: Array, noise: Array | DiagonalNoise) -> Gaussian:
    """``g`` conditioned on the reading ``H x + c + eps`` it is expected to give at its own mean:
    the innovation is zero, so the mean is ``g.loc`` itself and the covariance takes the same
    update as :func:`condition`, from ``noise`` (the noise at that mean: a lower-triangular
    Cholesky factor, or a :class:`DiagonalNoise` for the state-space form)."""
    if isinstance(noise, DiagonalNoise):
        _check_noise(noise, H)
        batch = jnp.broadcast_shapes(
            g.loc.shape[:-1],
            g.chol.shape[:-2],
            H.shape[:-2],
            noise.scale.shape[:-1],
            () if noise.low_rank is None else noise.low_rank.shape[:-2],
        )
        chol = _information(g, H, noise, batch).chol
    else:
        _, _, chol, _ = _gain_and_factor(g, H, noise)
    batch = jnp.broadcast_shapes(g.loc.shape[:-1], chol.shape[:-2])
    dim = g.loc.shape[-1]
    return Gaussian(
        loc=jnp.broadcast_to(g.loc, batch + (dim,)),
        chol=jnp.broadcast_to(chol, batch + (dim, dim)),
    )
