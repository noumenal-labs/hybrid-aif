"""EFE terms for continuous (Gaussian/mixture) blocks, as Interval term trees.

``risk`` is the family-agnostic KL builder shared with the discrete module;
``ambiguity`` is the expected entropy of a parent-indexed Gaussian observation
model. :func:`state_info_gain` is the channel-keyed ``infogain`` builder for LG
channels and :func:`emission_info_gain` for Gaussian emissions over categorical
parents; :func:`observation_interaction` and :func:`observation_correlation` are what a
per-channel sum miscounts when several channels read one Gaussian factor, in the
information gain and in the joint risk; :func:`uncertainty` is a block's entropy, the
opt-in ``uncertainty`` row.
:func:`lg_utility` and :func:`lg_ambiguity` score a preference over a continuous
observation; with them ``risk + ambiguity = −(utility + state info gain)`` holds per
channel at both interval endpoints, up to ``kl_to``'s clamp at 0, under any selection
of a channel's parents (:func:`selected_obs_entropy`). Taken à la carte, prefer the
utility form: risk alone carries a ``−H[q(o)]`` certainty tax. Unbatched over
policies; batch via ``vmap``.

Design: docs/design/objective-terms.md "The three axes"; docs/design/per-variable-infogain.md V11
"""

import math
import string
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.linalg import cho_solve
from jax.scipy.special import xlogy

from hybrid_aif.distributions import (
    Categorical,
    ConditionalGaussian,
    Distribution,
    Gaussian,
    MixtureGaussian,
)
from hybrid_aif.factors import (
    LinearGaussianObservation,
    LinearGaussianObservationMNIW,
    lg_point,
)
from hybrid_aif.interval import Interval
from hybrid_aif.objectives.base import INFOGAIN, TermTree
from hybrid_aif.objectives.discrete import _block_names, _check_mask, _full, risk, terminal_risk

__all__ = [
    "ambiguity",
    "conditional_correlation",
    "conditional_obs_entropy",
    "emission_ambiguity",
    "emission_entropies",
    "emission_info_gain",
    "lg_ambiguity",
    "lg_utility",
    "linear_gaussian_readings",
    "marginal",
    "mniw_draws",
    "observation_correlation",
    "observation_interaction",
    "param_info_gain",
    "predictive_entropy",
    "predictive_obs",
    "risk",
    "selected_obs_entropy",
    "state_info_gain",
    "sub_block_obs_entropy",
    "sub_block_readings",
    "SubBlockReadings",
    "terminal_risk",
    "uncertainty",
]

_AXES = string.ascii_lowercase
_LOG_2PI = jnp.log(2.0 * jnp.pi)


def uncertainty(
    q_blocks: tuple[Distribution, ...],
    block_names: tuple[str, ...] | None = None,
) -> TermTree:
    """``H[q(block)]`` per continuous block, the ``uncertainty`` row's leaf (a cost); a
    mixture's Kolchinsky–Tracey pair propagates as the Interval. Design:
    per-variable-infogain.md V11 decision 4."""
    names = _block_names("block", len(q_blocks), block_names)
    return {"uncertainty": {name: q.entropy() for name, q in zip(names, q_blocks, strict=True)}}


#: Largest matrix side whose Cholesky factor is unrolled into elementwise operations.
_UNROLLED_CHOLESKY = 4


def _unrolled_cholesky(M: Array) -> list[list[Array | None]]:
    """The entries ``L[i][j]`` (``i ≥ j``; ``None`` above the diagonal) of the Cholesky factor of
    a symmetric positive-definite ``M``, by elementwise operations on its ``(...,)`` batch."""
    d = M.shape[-1]
    L: list[list[Array | None]] = [[None] * d for _ in range(d)]
    for j in range(d):
        L[j][j] = jnp.sqrt(M[..., j, j] - sum((L[j][k] ** 2 for k in range(j)), 0.0))
        for i in range(j + 1, d):
            dot = sum((L[i][k] * L[j][k] for k in range(j)), 0.0)
            L[i][j] = (M[..., i, j] - dot) / L[j][j]
    return L


def _cholesky(M: Array) -> Array:
    """The lower Cholesky factor of a symmetric positive-definite ``M``: unrolled elementwise
    operations for a matrix of side :data:`_UNROLLED_CHOLESKY` or less, ``jnp.linalg.cholesky``
    for a larger one."""
    d = M.shape[-1]
    if d > _UNROLLED_CHOLESKY:
        return jnp.linalg.cholesky(M)
    L = _unrolled_cholesky(M)
    zero = jnp.zeros_like(M[..., 0, 0])
    return jnp.stack(
        [jnp.stack([L[i][j] if j <= i else zero for j in range(d)], axis=-1) for i in range(d)],
        axis=-2,
    )


def _log_diagonal(M: Array) -> Array:
    """The log of the diagonal of the Cholesky factor of a symmetric positive-definite ``M``,
    ``(..., d)``; its sum is ``½ log|M|``, and its first ``k`` entries are those of the factor
    of ``M``'s leading ``k × k`` block. A matrix of side four or less is factored by unrolled
    elementwise operations, a larger one by ``jnp.linalg.cholesky``."""
    d = M.shape[-1]
    if d > _UNROLLED_CHOLESKY:
        return jnp.log(jnp.diagonal(jnp.linalg.cholesky(M), axis1=-2, axis2=-1))
    L = _unrolled_cholesky(M)
    return jnp.log(jnp.stack([L[j][j] for j in range(d)], axis=-1))


def _half_log_det(M: Array) -> Array:
    """``½ log|M|`` of a symmetric positive-definite ``M``, from its Cholesky factor."""
    return jnp.sum(_log_diagonal(M), axis=-1)


def _total_correlation(H: Array, R: Array, cov: Array, sizes: tuple[int, ...]) -> Array:
    """``Σ_m H[y_m] − H[Y]`` of ``Y = H x + e``, ``Cov(x) = cov``, ``Cov(e) = R``, with the
    channels' row counts ``sizes``: ``½ (Σ_m log|S_mm| − log|S|)``, ``S = H cov Hᵀ + R``, each
    log-determinant from a Cholesky factor."""
    S = H @ cov @ jnp.swapaxes(H, -1, -2) + R
    edges = np.cumsum((0,) + sizes)
    parts = sum(_half_log_det(S[..., a:b, a:b]) for a, b in zip(edges[:-1], edges[1:], strict=True))
    return parts - _half_log_det(S)


def _block_half_logs(
    S: Array, sizes: tuple[int, ...], joint: bool = True
) -> tuple[list[Array], Array | None]:
    """``½ log|S_mm|`` of each diagonal block of a symmetric positive-definite ``S`` with the
    row counts ``sizes``, and, with ``joint``, the log diagonal of ``S``'s Cholesky factor (one
    factorisation: the leading rows of the factor are the factor of the first block). Without
    ``joint`` every block is factored on its own and the second value is ``None``."""
    edges = np.cumsum((0,) + tuple(sizes))
    if not joint:
        return [
            _half_log_det(S[..., a:b, a:b]) for a, b in zip(edges[:-1], edges[1:], strict=True)
        ], None
    log_diag = _log_diagonal(S)
    halves = [jnp.sum(log_diag[..., : sizes[0]], axis=-1)]
    halves += [_half_log_det(S[..., a:b, a:b]) for a, b in zip(edges[1:-1], edges[2:], strict=True)]
    return halves, log_diag


def _stacked_noise(
    q: Gaussian,
    models: tuple[LinearGaussianObservation, ...],
    chols: tuple[Array | None, ...] | None = None,
) -> tuple[Array, list[Array], tuple[int, ...]]:
    """The stacked head, each channel's noise covariance and the channels' row counts. A
    channel's noise is read from its model at ``q``'s mean (a noise map), unless ``chols`` gives
    its Cholesky factor there already (``chols[i]`` not ``None``)."""
    noises = []
    for i, model in enumerate(models):
        R_chol = model.R_chol_at(q.loc) if chols is None or chols[i] is None else chols[i]
        noises.append(R_chol @ jnp.swapaxes(R_chol, -1, -2))
    sizes = tuple(model.H.shape[-2] for model in models)
    return jnp.concatenate([model.H for model in models], axis=-2), noises, sizes


def _correlation_of(
    S: Array,
    halves: list[Array],
    log_diag: Array | None,
    sizes: tuple[int, ...],
    present: Array | None,
) -> Array:
    """``TC(Y) = Σ_m ½ log|S_mm| − ½ log|S|`` of the readings with covariance ``S``; an absent
    channel (``present[i]`` false) is an independent unit-noise reading, adding nothing."""
    if len(sizes) == 1:
        return jnp.zeros_like(halves[0])
    if present is None:
        return sum(halves) - jnp.sum(log_diag, axis=-1)
    rows = jnp.concatenate([jnp.broadcast_to(present[i], (n,)) for i, n in enumerate(sizes)])
    kept = jnp.where(rows[:, None] & rows[None, :], S, jnp.eye(S.shape[-1], dtype=S.dtype))
    parts = sum(jnp.where(present[i], h, 0.0) for i, h in enumerate(halves))
    return parts - _half_log_det(kept)


def linear_gaussian_readings(
    q: Gaussian,
    models: tuple[LinearGaussianObservation, ...],
    present: Array | None = None,
    noises: tuple[Array | None, ...] | None = None,
) -> tuple[tuple[Interval, ...], Array]:
    """The per-channel information gains and the total correlation of several non-switching LG
    channels read on one Gaussian belief ``q``, from one factorisation of the stacked predictive
    covariance ``S = H Σ Hᵀ + blockdiag(R_m)`` (a noise map at ``q``'s mean).

    Returns ``(gains, tc)``: ``gains[m] = I(y_m; x) = ½ log|S_mm| − ½ log|R_m|``, exact, from the
    Cholesky factor of the diagonal block ``S_mm``; and ``tc = Σ_m I_m − I(Y; x)`` with ``I(Y;
    x) = ½ log|S| − ½ log|blockdiag(R)|``, from one Cholesky factor of ``S``. For jointly Gaussian
    readings of independent noises ``Σ_m I_m − I(Y; x) = Σ_m H[y_m] − H[Y]``, the total
    correlation ``TC(Y)`` (the :func:`observation_interaction` of the whole state, the
    :func:`observation_correlation`), as the ``log|R_m|`` terms cancel. ``present`` (one bool per
    channel, ``None``: all) drops an absent channel from ``tc`` only, as :func:`_stacked_readings`
    does (zero head, unit noise); the gains are those of the channel as declared. ``tc`` is 0 for
    one channel. ``noises`` (``None``: every channel's noise is read at ``q``'s mean) holds the
    Cholesky factor of a channel's noise at that mean where it has been read already, ``None``
    for a channel still to read.
    """
    head, covs, sizes = _stacked_noise(q, models, noises)
    HL = head @ q.chol
    S = HL @ jnp.swapaxes(HL, -1, -2) + jax.scipy.linalg.block_diag(*covs)
    halves, log_diag = _block_half_logs(S, sizes, present is None)
    gains = tuple(Interval.exact(h - _half_log_det(R)) for h, R in zip(halves, covs, strict=True))
    return gains, _correlation_of(S, halves, log_diag, sizes, present)


class SubBlockReadings(NamedTuple):
    """What :func:`sub_block_readings` returns, each exact: ``gains[m] = I(y_m; x_S)``;
    ``interaction = TC(Y) − TC(Y | x_S)``; ``total_correlation = TC(Y)``;
    ``conditional_correlation = TC(Y | x_S)``; ``ambiguity[m] = E H[y_m | x_S]``."""

    gains: tuple[Interval, ...]
    interaction: Array
    total_correlation: Array
    conditional_correlation: Array
    ambiguity: tuple[Array, ...]


def sub_block_readings(
    q: Gaussian,
    models: tuple[LinearGaussianObservation, ...],
    dims: tuple[int, ...],
    present: Array | None = None,
    noises: tuple[Array | None, ...] | None = None,
) -> SubBlockReadings:
    """The leaves of several non-switching LG channels read on one Gaussian belief ``q``, scored
    against the dimensions ``dims`` of the state (``x_S``, the others marginalised), from two
    factorisations: the stacked predictive covariance ``Σ_Y = H Σ Hᵀ + blockdiag(R_m)`` and the
    stacked conditional one ``Σ_{Y|S} = H Σ_given Hᵀ + blockdiag(R_m)``, with ``Σ_given = Σ −
    Σ_{:,S} Σ_SS⁻¹ Σ_{S,:}`` (a noise map at ``q``'s mean).

    With ``h_m = ½ log|Σ_Y,mm|`` and ``c_m = ½ log|Σ_{Y|S},mm|`` the diagonal blocks, ``gains[m] =
    h_m − c_m`` (``I(y_m; x_S) = H[y_m] − E H[y_m | x_S]``), ``ambiguity[m] = ½ d_m (1 + log 2π)
    + c_m``, and the joint gain is ``I(Y; x_S) = ½ log|Σ_Y| − ½ log|Σ_{Y|S}|``. The
    interaction is ``Σ_m I(y_m; x_S) − I(Y; x_S) = TC(Y) − TC(Y | x_S)``, positive for readings
    sharing information about ``x_S`` and negative for readings informative about it only
    together; the ``log|R_m|`` terms cancel. ``present`` (one bool per channel, ``None``: all)
    drops an absent channel from the correlations only (zero head, unit noise), as
    :func:`_stacked_readings` does; gains and ambiguities are those of the channel as declared.
    The correlations are 0 for one channel. ``noises`` is as in :func:`linear_gaussian_readings`.

    ``Σ_given`` is not formed: with the selected dimensions leading (``Σ = L Lᵀ``, ``L`` lower
    triangular), ``Σ_given = blockdiag(0, L_TT L_TTᵀ)`` with ``L_TT`` the trailing block of ``L``
    (the Schur complement of ``Σ_SS``), so ``Σ_{Y|S} = (H_T L_TT)(H_T L_TT)ᵀ + R`` with ``H_T`` the
    head's trailing columns, and no factorisation is taken. Selected dimensions that do not lead
    are brought to the front by a permutation of the state, and ``L`` is the Cholesky factor of
    the permuted covariance (:func:`_trailing_factor`).
    """
    head, covs, sizes = _stacked_noise(q, models, noises)
    noise = jax.scipy.linalg.block_diag(*covs)
    HL = head @ q.chol
    S_y = HL @ jnp.swapaxes(HL, -1, -2) + noise
    tail, L_TT = _trailing_factor(q, dims)
    HT = head[..., :, tail] @ L_TT
    S_c = HT @ jnp.swapaxes(HT, -1, -2) + noise
    halves_y, diag_y = _block_half_logs(S_y, sizes, present is None)
    halves_c, diag_c = _block_half_logs(S_c, sizes, present is None)
    tc_y = _correlation_of(S_y, halves_y, diag_y, sizes, present)
    tc_c = _correlation_of(S_c, halves_c, diag_c, sizes, present)
    return SubBlockReadings(
        gains=tuple(Interval.exact(h - c) for h, c in zip(halves_y, halves_c, strict=True)),
        interaction=tc_y - tc_c,
        total_correlation=tc_y,
        conditional_correlation=tc_c,
        ambiguity=tuple(
            0.5 * n * (1.0 + _LOG_2PI) + c for n, c in zip(sizes, halves_c, strict=True)
        ),
    )


def _stacked_readings(
    q: Gaussian, models: tuple[LinearGaussianObservation, ...], present: Array | None
) -> tuple[Array, Array, tuple[int, ...]]:
    """The stacked head ``H``, the block-diagonal noise ``R`` and the row counts of several
    non-switching LG channels read on one Gaussian belief (a noise map at ``q``'s mean). An
    absent channel (``present[i]`` false) reads independent unit noise through a zero head,
    which adds nothing to any total correlation."""
    heads, noises, sizes = [], [], []
    for i, model in enumerate(models):
        H = model.H
        R_chol = model.R_chol_at(q.loc)
        R = R_chol @ jnp.swapaxes(R_chol, -1, -2)
        if present is not None:
            H = jnp.where(present[i], H, 0.0)
            R = jnp.where(present[i], R, jnp.eye(R.shape[-1], dtype=R.dtype))
        heads.append(H)
        noises.append(R)
        sizes.append(H.shape[-2])
    return jnp.concatenate(heads, axis=-2), jax.scipy.linalg.block_diag(*noises), tuple(sizes)


def _trailing_factor(q: Gaussian, dims: tuple[int, ...]) -> tuple[np.ndarray, Array]:
    """``(tail, L_TT)`` for the dimensions ``dims`` of ``q``'s state: the state's other
    dimensions ``tail`` (ascending) and a factor ``L_TT`` with ``L_TT L_TTᵀ`` the covariance of
    ``x_tail`` given ``x_dims``. For ``dims`` the leading dimensions it is the trailing block of
    ``q.chol``; otherwise it is that of the Cholesky factor of the covariance with ``dims``
    permuted to the front (one factorisation of the state's own side)."""
    d = q.chol.shape[-1]
    head = np.asarray(sorted(dims), dtype=int)
    tail = np.setdiff1d(np.arange(d), head)
    k = head.size
    if np.array_equal(head, np.arange(k)):
        return tail, q.chol[..., k:, k:]
    order = np.concatenate([head, tail])
    cov = q.cov[..., order, :][..., :, order]
    return tail, _cholesky(cov)[..., k:, k:]


def _given(cov: Array, dims: tuple[int, ...]) -> Array:
    """The covariance of ``x`` given its dimensions ``dims``: ``Σ − Σ_{:,S} Σ_SS⁻¹ Σ_{S,:}``."""
    idx = np.asarray(dims)
    cross = cov[..., :, idx]
    block = jnp.linalg.cholesky(cross[..., idx, :])
    return cov - cross @ cho_solve((block, True), jnp.swapaxes(cross, -1, -2))


def observation_correlation(
    q: Gaussian,
    models: tuple[LinearGaussianObservation, ...],
    present: Array | None = None,
) -> Array:
    """The total correlation ``TC(Y) = Σ_m H[q(y_m)] − H[q(Y)]`` of one step's predicted readings
    of several non-switching LG channels on one Gaussian belief ``q``, exact (``Cov(Y) = H Σ Hᵀ +
    blockdiag(R_m)``, the noises independent given the state). It is the part of the joint risk
    a per-channel sum leaves out: ``KL(q(Y) ‖ ∏_m C_m) = Σ_m KL(q(y_m) ‖ C_m) + TC(Y)``.
    ``present`` (one bool per channel, ``None``: all) drops an absent channel. Design:
    per-variable-infogain.md V11 decision 3.
    """
    H, R, sizes = _stacked_readings(q, models, present)
    return _total_correlation(H, R, q.cov, sizes)


def conditional_correlation(
    q: Gaussian,
    models: tuple[LinearGaussianObservation, ...],
    dims: tuple[int, ...],
    present: Array | None = None,
) -> Array:
    """``TC(Y | x_S)``: the total correlation of the readings given the dimensions ``dims`` of
    the state, the others marginalised, the same for every value of ``x_S``. It is the part of
    the joint ambiguity a per-channel sum leaves out: ``E H[Y | x_S] = Σ_m E H[y_m | x_S] −
    TC(Y | x_S)``; given the whole state the readings are independent and it is 0."""
    H, R, sizes = _stacked_readings(q, models, present)
    return _total_correlation(H, R, _given(q.cov, dims), sizes)


def observation_interaction(
    q: Gaussian,
    models: tuple[LinearGaussianObservation, ...],
    dims: tuple[int, ...] | None = None,
    present: Array | None = None,
) -> Array:
    """The interaction information ``TC(Y) − TC(Y | x_S)`` of several non-switching LG channels'
    readings about the dimensions ``dims`` of a Gaussian state (``None``: the whole state, where
    ``TC(Y | x) = 0`` and it is the total correlation ``TC(Y)``): what a per-channel sum of
    information gains miscounts, ``I(Y; x_S) = Σ_m I(y_m; x_S) − (TC(Y) − TC(Y | x_S))``.

    Positive where the readings share information about ``x_S`` (redundancy, counted twice by the
    sum); negative where they carry information about ``x_S`` only together (synergy, which the
    sum misses), possible only on a sub-block. Exact. Design: per-variable-infogain.md V11
    decision 3.
    """
    H, R, sizes = _stacked_readings(q, models, present)
    out = _total_correlation(H, R, q.cov, sizes)
    if dims is None:
        return out
    return out - _total_correlation(H, R, _given(q.cov, dims), sizes)


def marginal(q: Gaussian | MixtureGaussian, dims: tuple[int, ...]) -> Gaussian | MixtureGaussian:
    """The marginal of ``q`` over the dimensions ``dims`` (positions, static): ``N(μ_S, Σ_SS)``,
    re-factored from the covariance; a mixture's is the mixture of its components' marginals,
    weights unchanged. Design: per-variable-infogain.md V10."""
    idx = np.asarray(dims)
    if isinstance(q, MixtureGaussian):
        comps = marginal(q.components, dims)
        return MixtureGaussian(log_weights=q.log_weights, means=comps.loc, chols=comps.chol)
    cov = q.cov[..., idx, :][..., :, idx]
    return Gaussian(loc=q.loc[..., idx], chol=jnp.linalg.cholesky(cov))


def sub_block_obs_entropy(
    q: Gaussian, model: LinearGaussianObservation, dims: tuple[int, ...]
) -> Array:
    """``E_q[H[y | x_S]]`` for a non-switching LG channel on a Gaussian belief, ``S`` the
    dimensions ``dims`` of the observed factor and the others marginalised.

    Given ``x_S`` the rest of the state is Gaussian with the covariance ``Σ − Σ_{:,S} Σ_SS⁻¹
    Σ_{S,:}``, whatever ``x_S`` is, so ``y | x_S`` is Gaussian with covariance ``H Σ_{x|x_S} Hᵀ
    + R`` and the entropy is ``½ log|2πe (H Σ_{x|x_S} Hᵀ + R)|``, exact. A noise map is read at
    the belief's mean. Design: per-variable-infogain.md V10.
    """
    idx = np.asarray(dims)
    cov = q.cov
    cross = cov[..., :, idx]  # Σ_{:,S}
    block = jnp.linalg.cholesky(cross[..., idx, :])  # Σ_SS
    explained = cross @ cho_solve((block, True), jnp.swapaxes(cross, -1, -2))
    H = model.H
    R_chol = model.R_chol_at(q.loc)
    S = H @ (cov - explained) @ jnp.swapaxes(H, -1, -2) + R_chol @ jnp.swapaxes(R_chol, -1, -2)
    return _observation_entropy(jnp.linalg.cholesky(S))


def _contract_parents(values: Array, parent_beliefs: tuple[Categorical, ...]) -> Array:
    subs = _AXES[: values.ndim]
    spec = ",".join([subs, *subs]) + "->"
    return jnp.einsum(spec, values, *[b.probs for b in parent_beliefs])


def ambiguity(
    q_parents: tuple[Categorical, ...],
    model: ConditionalGaussian,
    block_names: tuple[str, ...] | None = None,
) -> TermTree:
    """E_q(parents)[H[N(·|parents)]] for a parent-indexed Gaussian observation model.

    One belief per parent axis (mean-field); the per-configuration Gaussian
    entropies are closed-form, so the leaf is exact.
    """
    (name,) = _block_names("modality", 1, block_names)
    h = model.gaussians().entropy()
    leaf = Interval(
        lower=_contract_parents(h.lower, q_parents),
        upper=_contract_parents(h.upper, q_parents),
    )
    return {"ambiguity": {name: leaf}}


def emission_entropies(
    model: LinearGaussianObservation | LinearGaussianObservationMNIW,
    parent_beliefs: tuple[Categorical, ...],
    selected: tuple[bool, ...] | None = None,
) -> tuple[Interval, Interval]:
    """``(H[q(y)], E_q H[y | S])`` of a Gaussian emission over categorical parents alone,
    ``y = c_j + noise_j``, ``S`` the selected parents (``A_deps`` order; ``None`` or all
    ``True``: every parent).

    The predictive ``q(y) = Σ_j q(j) N(c_j, R_j)`` is a mixture over the parents' joint
    configurations (mean-field), its entropy a Kolchinsky–Tracey pair. Given every parent the
    entropy is the configurations' noise entropies under ``q``, exact; given some, each selected
    configuration's conditional predictive is a mixture over the others, a KT pair per
    configuration (:func:`selected_obs_entropy`'s "factor unselected" case, the emission having
    no state behind it). A matrix-normal inverse-Wishart emission is read through its point lowering
    (posterior-mean offset, expected-precision noise), as its ambiguity is. Design:
    per-variable-infogain.md V11.
    """
    point = lg_point(model)
    parents = tuple(parent_beliefs)
    predictive = _emission_conditional_entropy(point, parents, (False,) * len(parents))
    if _full(selected):
        shape = tuple(b.probs.shape[-1] for b in parents)
        h = jnp.broadcast_to(_observation_entropy(point.R_chol), shape)
        return predictive, Interval.exact(_contract_parents(h, parents))
    _check_mask(selected, len(parents))
    return predictive, _emission_conditional_entropy(point, parents, tuple(selected))


def _emission_conditional_entropy(
    point: LinearGaussianObservation, parents: tuple[Categorical, ...], keep: tuple[bool, ...]
) -> Interval:
    """``Σ_s q(s) H[q(y | s)]`` of a point emission over the kept parents' configurations ``s``,
    each conditional predictive a KT pair over the others (no kept parent: ``H[q(y)]``)."""
    sizes = tuple(b.probs.shape[-1] for b in parents)
    stateless = MixtureGaussian(
        log_weights=jnp.zeros((1,), point.c.dtype),
        means=jnp.zeros((1, 0), point.c.dtype),
        chols=jnp.zeros((1, 0, 0), point.c.dtype),
    )
    q_p = _group_probs(tuple(b for b, k in zip(parents, keep, strict=True) if k))
    unsel = tuple(b for b, k in zip(parents, keep, strict=True) if not k)
    log_q_u = _joint_mode_log_probs(unsel) if unsel else jnp.zeros((1,), point.c.dtype)
    head = _split_modes(point.head, sizes, keep, 2)
    R_chol = _split_modes(point.R_chol, sizes, keep, 2)
    h = _conditional_predictive_entropy(stateless, head, R_chol, log_q_u)
    return Interval(lower=jnp.sum(q_p * h.lower), upper=jnp.sum(q_p * h.upper))


def emission_ambiguity(
    models: tuple[LinearGaussianObservation, ...],
    parent_beliefs: tuple[tuple[Categorical, ...], ...],
    block_names: tuple[str, ...] | None = None,
    selected: tuple[tuple[bool, ...] | None, ...] | None = None,
) -> TermTree:
    """``E_q(parents)[H[y | S]]`` per Gaussian emission over categorical parents alone, ``S``
    the selected parents (``selected[i]``, ``None``: all): with every parent, the
    per-configuration Gaussian entropies of its noise under the parents' beliefs (mean-field),
    closed-form and exact; with some, :func:`emission_entropies`'s pair. Design:
    docs/design/structure-learning.md S2, S7; per-variable-infogain.md V11."""
    names = _block_names("modality", len(models), block_names)
    masks = selected if selected is not None else (None,) * len(models)
    return {
        "ambiguity": {
            name: emission_entropies(model, parents, mask)[1]
            for name, model, parents, mask in zip(names, models, parent_beliefs, masks, strict=True)
        }
    }


def emission_info_gain(
    models: tuple[LinearGaussianObservation, ...],
    parent_beliefs: tuple[tuple[Categorical, ...], ...],
    block_names: tuple[str, ...] | None = None,
    selected: tuple[tuple[bool, ...] | None, ...] | None = None,
) -> TermTree:
    """``I(y; S) = H[q(y)] − E_q H[y | S]`` per Gaussian emission over categorical parents
    alone (:func:`emission_entropies`), an Interval from the predictive mixture's KT pair.
    Design: per-variable-infogain.md V11."""
    names = _block_names("modality", len(models), block_names)
    masks = selected if selected is not None else (None,) * len(models)
    leaves = {}
    for name, model, parents, mask in zip(names, models, parent_beliefs, masks, strict=True):
        predictive, cond = emission_entropies(model, parents, mask)
        leaves[name] = predictive - cond
    return {INFOGAIN: leaves}


def _predictive_obs(
    q: Gaussian | MixtureGaussian, H: Array, c: Array, R_chol: Array
) -> MixtureGaussian:
    """Predictive observation density of ``q`` through ``y = H x + c + eps``.

    Push each component ``N(mu_j, Sigma_j)`` of ``q`` (a Gaussian is the K=1
    case) forward: ``N(H mu_j + c, H Sigma_j H^T + R)``, weights unchanged. The
    K-component mixture whose entropy is the marginal predictive ``H[q(y)]``.
    """
    m = q if isinstance(q, MixtureGaussian) else MixtureGaussian.from_gaussian(q)
    comps = m.components
    means = (H @ comps.loc[..., None])[..., 0] + c
    HL = H @ comps.chol
    R = R_chol @ jnp.swapaxes(R_chol, -1, -2)
    S = HL @ jnp.swapaxes(HL, -1, -2) + R
    return MixtureGaussian(log_weights=m.log_weights, means=means, chols=jnp.linalg.cholesky(S))


def _observation_entropy(R_chol: Array) -> Array:
    """H[N(., R R^T)] — the conditional entropy H[y|x], constant in x (additive noise).

    Broadcasts over any leading (e.g. per-mode) axes of ``R_chol``.
    """
    d_obs = R_chol.shape[-1]
    half_log_det = jnp.sum(jnp.log(jnp.diagonal(R_chol, axis1=-2, axis2=-1)), axis=-1)
    return 0.5 * d_obs * (1.0 + _LOG_2PI) + half_log_det


def _flatten_modes(arr: Array, m: int, event_ndims: int) -> Array:
    """Bring a shared or per-mode observation array to a leading mode axis ``(M, *event)``.

    A shared array is broadcast across the ``M`` modes; a per-mode array's leading
    parent axes are flattened row-major, as in
    :func:`hybrid_aif.inference.mixture_filter._per_mode`.
    """
    event = arr.shape[arr.ndim - event_ndims :]
    if arr.ndim == event_ndims:
        return jnp.broadcast_to(arr, (m,) + event)
    return arr.reshape((m,) + event)


def _predictive_obs_switching(
    q: Gaussian | MixtureGaussian, H: Array, c: Array, R_chol: Array, mode_logq: Array
) -> MixtureGaussian:
    """Predictive observation density of a switching LG modality: the (K·M) mixture.

    Each ``(component k, mode j)`` gives ``N(H_j mu_k + c_j, H_j Sigma_k H_j^T +
    R_j)`` with weight ``w_k · q(mode=j)``; ``mode_logq`` is the joint mode
    log-belief ``(..., M)``. Component-major layout (slot ``k*M + j``).
    """
    m = q if isinstance(q, MixtureGaussian) else MixtureGaussian.from_gaussian(q)
    mm = mode_logq.shape[-1]
    Hs = _flatten_modes(H, mm, 2)  # (M, d_obs, d)
    cs = _flatten_modes(c, mm, 1)  # (M, d_obs)
    Rs = _flatten_modes(R_chol, mm, 2)  # (M, d_obs, d_obs)
    comps = m.components
    mu = comps.loc[..., :, None, :]  # (..., K, 1, d)
    means = (Hs @ mu[..., None])[..., 0] + cs  # (..., K, M, d_obs)
    HL = Hs @ comps.chol[..., :, None, :, :]  # (..., K, M, d_obs, d)
    R = Rs @ jnp.swapaxes(Rs, -1, -2)
    S = HL @ jnp.swapaxes(HL, -1, -2) + R  # (..., K, M, d_obs, d_obs)
    log_w = m.log_weights[..., :, None] + mode_logq[..., None, :]  # (..., K, M)
    d_obs = means.shape[-1]
    batch = log_w.shape[:-2]
    n = m.num_components * mm
    return MixtureGaussian(
        log_weights=log_w.reshape(batch + (n,)),
        means=means.reshape(batch + (n, d_obs)),
        chols=jnp.linalg.cholesky(S).reshape(batch + (n, d_obs, d_obs)),
    )


def predictive_obs(
    q: Gaussian | MixtureGaussian,
    model: LinearGaussianObservation,
    mode_beliefs: tuple[Categorical, ...] = (),
) -> MixtureGaussian:
    """``q(o)``: the belief pushed forward through a (possibly switching) LG head.

    Returns a ``K·M``-component mixture: one component per belief component, times
    one per joint mode configuration (flattened row-major) of ``mode_beliefs``, the
    modality's discrete parent beliefs in declaration order (empty when it does not
    switch). A :class:`Gaussian` belief is the ``K=1`` case.
    """
    if mode_beliefs:
        return _predictive_obs_switching(
            q, model.H, model.c, model.R_chol, _joint_mode_log_probs(mode_beliefs)
        )
    return _predictive_obs(q, model.H, model.c, model.R_chol_at(_component_means(q)[0]))


def predictive_entropy(
    q: Gaussian | MixtureGaussian,
    model: LinearGaussianObservation,
    mode_beliefs: tuple[Categorical, ...] = (),
) -> Interval:
    """``H[q(y)]`` of an LG channel's predictive (:func:`predictive_obs`): exact on a Gaussian
    belief through a non-switching channel, whose predictive is the one Gaussian ``N(H μ + c,
    H Σ Hᵀ + R)``, evaluated as ``½ log|2πe (H Σ Hᵀ + R)|`` from a Cholesky factor without
    building the mixture; a Kolchinsky–Tracey pair on the mixture otherwise."""
    if isinstance(q, Gaussian) and not mode_beliefs:
        HL = model.H @ q.chol
        R_chol = model.R_chol_at(q.loc)
        S = HL @ jnp.swapaxes(HL, -1, -2) + R_chol @ jnp.swapaxes(R_chol, -1, -2)
        return Interval.exact(0.5 * S.shape[-1] * (1.0 + _LOG_2PI) + _half_log_det(S))
    return predictive_obs(q, model, mode_beliefs).entropy()


def _component_means(q: Gaussian | MixtureGaussian) -> tuple[Array, Array]:
    """``(means, log_weights)`` of a belief's components: ``(K, d)`` and ``(K,)``, a Gaussian
    the ``K = 1`` case."""
    if isinstance(q, MixtureGaussian):
        return q.components.loc, q.log_weights
    return q.loc[..., None, :], jnp.zeros((*q.loc.shape[:-1], 1), q.loc.dtype)


def conditional_obs_entropy(
    model: LinearGaussianObservation,
    mode_beliefs: tuple[Categorical, ...] = (),
    q: Gaussian | MixtureGaussian | None = None,
) -> Array:
    """``E_q[H[y | x, mode]]`` for an LG channel: exact, and constant in ``x`` for a point or
    conjugate noise.

    ``½ log|2πe R Rᵀ|`` for a non-switching channel; for a switching one, its ``q(mode)``
    average over the per-mode noises, which is the entropy left once *both* the factor and the
    mode are known. The entropy given the factor alone, with the mode unknown, is
    :func:`selected_obs_entropy`'s business. A noise map (``model.noise_map``) is read at the
    belief ``q``'s component means and averaged under its weights, the extended filter's
    convention, so ``q`` is required then.
    """
    if not mode_beliefs:
        if not model.noise_map:
            return _observation_entropy(model.R_chol)
        if q is None:
            raise ValueError("a noise map's conditional entropy needs the belief it is read at")
        means, log_w = _component_means(q)
        return jnp.sum(jax.nn.softmax(log_w) * _observation_entropy(model.R_chol_at(means)), -1)
    mode_logq = _joint_mode_log_probs(mode_beliefs)
    mm = mode_logq.shape[-1]
    Rs = _flatten_modes(model.R_chol, mm, 2)
    return jnp.sum(jax.nn.softmax(mode_logq, axis=-1) * _observation_entropy(Rs), axis=-1)


def _mode_grid(arr: Array, sizes: tuple[int, ...], event_ndims: int) -> Array:
    """A shared or per-mode observation array on the full mode grid ``(*sizes, *event)``."""
    flat = _flatten_modes(arr, math.prod(sizes), event_ndims)
    return flat.reshape(sizes + flat.shape[1:])


def _split_modes(
    arr: Array, sizes: tuple[int, ...], keep: tuple[bool, ...], event_ndims: int
) -> Array:
    """Mode-grid array → ``(P, U, *event)``: kept parent axes first, then the others, each
    group flattened row-major."""
    grid = _mode_grid(arr, sizes, event_ndims)
    kept = [i for i, k in enumerate(keep) if k]
    rest = [i for i, k in enumerate(keep) if not k]
    n_ev = grid.ndim - len(sizes)
    grid = jnp.transpose(grid, kept + rest + list(range(len(sizes), len(sizes) + n_ev)))
    p = math.prod(sizes[i] for i in kept)
    u = math.prod(sizes[i] for i in rest)
    return grid.reshape((p, u) + grid.shape[len(sizes) :])


def _group_probs(beliefs: tuple[Categorical, ...]) -> Array:
    """The mean-field joint over ``beliefs``, flattened row-major (``(1,)`` when empty)."""
    if not beliefs:
        return jnp.ones((1,))
    return jax.nn.softmax(_joint_mode_log_probs(beliefs), axis=-1)


def _expected_mode_spread(head: Array, q_u: Array, q: MixtureGaussian) -> Array:
    """``E_q(x)[Cov_{q(u)}(H_u x + c_u)]`` per selected configuration, shape ``(P, d_obs, d_obs)``.

    ``head`` is the augmented ``[H_u | c_u]`` split as ``(P, U, d_obs, d+1)``. With ``x̃ = (x, 1)``
    and ``Ḡ = Σ_u q_u G_u`` the spread is ``Σ_u q_u (G_u − Ḡ) E[x̃ x̃ᵀ] (G_u − Ḡ)ᵀ``, and
    ``E[x̃ x̃ᵀ]`` is the mixture's second moment over the components.
    """
    w = jax.nn.softmax(q.log_weights, axis=-1)
    comps = q.components
    mean = jnp.sum(w[:, None] * comps.loc, axis=0)
    second = jnp.sum(
        w[:, None, None] * (comps.cov + comps.loc[:, :, None] * comps.loc[:, None, :]), axis=0
    )
    d = mean.shape[-1]
    aug = jnp.zeros((d + 1, d + 1), second.dtype)
    aug = aug.at[:d, :d].set(second).at[:d, d].set(mean).at[d, :d].set(mean).at[d, d].set(1.0)
    dev = head - jnp.sum(q_u[None, :, None, None] * head, axis=1, keepdims=True)
    spread = dev @ aug @ jnp.swapaxes(dev, -1, -2)  # (P, U, d_obs, d_obs)
    return jnp.sum(q_u[None, :, None, None] * spread, axis=1)


def _noise_mixture_entropy(
    q: MixtureGaussian, head: Array, R_chol: Array, log_q_u: Array, shared_head: bool
) -> Interval:
    """``E_q(x) H[Σ_u q_u N(H_u x + c_u, R_u)]`` per selected configuration, an Interval ``(P,)``.

    ``head``/``R_chol`` are split ``(P, U, ...)``. Both cases are bounded by the guaranteed
    bracket ``[H[y|x, mode], min(H[y|x, mode] + H(q(u)), ½ log|2πe E_x Cov(y|x)|)]``. Where the
    head is equal across the unselected modes (statically shared, or per-mode with exactly equal
    values, compared at run time per selected configuration) every member is centred at the same
    point, so the mixture is the zero-mean noise mixture, constant in ``x``, and its KT pair is
    intersected with the bracket. Otherwise the entropy depends on ``x`` and the bracket itself
    is the interval; a head that differs by any amount, however small, takes the bracket.
    """
    q_u = jax.nn.softmax(log_q_u, axis=-1)
    floor = jnp.sum(q_u * _observation_entropy(R_chol), axis=-1)  # H[y | x, mode], (P,)
    noise_cov = R_chol @ jnp.swapaxes(R_chol, -1, -2)
    cov = jnp.sum(q_u[None, :, None, None] * noise_cov, axis=1)
    if not shared_head:
        cov = cov + _expected_mode_spread(head, q_u, q)
    max_ent = _observation_entropy(jnp.linalg.cholesky(cov))
    ceiling = jnp.minimum(floor - jnp.sum(xlogy(q_u, q_u)), max_ent)
    p, u, d_obs = R_chol.shape[0], R_chol.shape[1], R_chol.shape[-1]
    kt = MixtureGaussian(
        log_weights=jnp.broadcast_to(log_q_u, (p, u)),
        means=jnp.zeros((p, u, d_obs), R_chol.dtype),
        chols=R_chol,
    ).entropy()
    lower = jnp.clip(kt.lower, floor, ceiling)
    upper = jnp.clip(kt.upper, floor, ceiling)
    if shared_head:
        return Interval(lower=lower, upper=upper)
    equal = jnp.all(head == head[:, :1], axis=(1, 2, 3))  # (P,): one head across u
    return Interval(lower=jnp.where(equal, lower, floor), upper=jnp.where(equal, upper, ceiling))


def selected_obs_entropy(
    q: Gaussian | MixtureGaussian,
    model: LinearGaussianObservation,
    mode_beliefs: tuple[Categorical, ...] = (),
    selected: tuple[bool, ...] | None = None,
) -> Interval:
    """``E_q[H[y | S]]`` for an LG channel, ``S`` the selected members of its dependency set.

    ``selected`` masks the dependency set in ``A_deps`` order: the observed factor, then the
    mode parents of ``mode_beliefs``; ``None`` or an all-``True`` mask is the whole set, the
    exact :func:`conditional_obs_entropy`. Unselected parents are marginalised under their
    current belief:

    - factor selected, the head ``[H | c]`` shared across modes: the zero-mean noise mixture's
      Kolchinsky–Tracey pair, constant in ``x``;
    - factor selected, a per-mode head: the same pair where the head's values are exactly equal
      across the unselected modes; elsewhere the guaranteed bracket ``[E H[y | x, mode],
      min(E H[y | x, mode] + H(q(unselected modes)), ½ log|2πe E_x Cov(y | x)|)]`` itself;
    - factor unselected: ``Σ_s q(s) H[q(y | s)]`` over the selected mode configurations, each
      conditional predictive a mixture over the factor's components and the unselected modes.

    Design: docs/design/per-variable-infogain.md V9
    """
    if _full(selected):
        return Interval.exact(conditional_obs_entropy(model, mode_beliefs))
    _check_mask(selected, len(mode_beliefs) + 1)
    m = q if isinstance(q, MixtureGaussian) else MixtureGaussian.from_gaussian(q)
    sizes = tuple(b.probs.shape[-1] for b in mode_beliefs)
    keep = tuple(selected[1:])
    q_p = _group_probs(tuple(b for b, k in zip(mode_beliefs, keep, strict=True) if k))
    unsel = tuple(b for b, k in zip(mode_beliefs, keep, strict=True) if not k)
    log_q_u = _joint_mode_log_probs(unsel) if unsel else jnp.zeros((1,), m.log_weights.dtype)
    model = lg_point(model)
    head = _split_modes(model.head, sizes, keep, 2)
    R_chol = _split_modes(model.R_chol, sizes, keep, 2)
    if selected[0]:
        h = _noise_mixture_entropy(m, head, R_chol, log_q_u, shared_head=model.head.ndim == 2)
    else:
        h = _conditional_predictive_entropy(m, head, R_chol, log_q_u)
    return Interval(lower=jnp.sum(q_p * h.lower), upper=jnp.sum(q_p * h.upper))


def _conditional_predictive_entropy(
    q: MixtureGaussian, head: Array, R_chol: Array, log_q_u: Array
) -> Interval:
    """``H[q(y | s)]`` per selected mode configuration ``s``: a KT pair over the ``K·U`` mixture
    of the factor's components and the unselected modes, shape ``(P,)``."""
    p, u = R_chol.shape[0], R_chol.shape[1]
    H, c = head[..., :-1], head[..., -1]  # (P, U, d_obs, d), (P, U, d_obs)
    comps = q.components
    means = jnp.einsum("puod,kd->pkuo", H, comps.loc) + c[:, None]  # (P, K, U, d_obs)
    HL = jnp.einsum("puod,kde->pkuoe", H, comps.chol)
    R = R_chol @ jnp.swapaxes(R_chol, -1, -2)
    S = HL @ jnp.swapaxes(HL, -1, -2) + R[:, None]
    k = q.num_components
    log_w = q.log_weights[:, None] + log_q_u[None, :]  # (K, U)
    d_obs = means.shape[-1]
    return MixtureGaussian(
        log_weights=jnp.broadcast_to(log_w.reshape(-1), (p, k * u)),
        means=means.reshape(p, k * u, d_obs),
        chols=jnp.linalg.cholesky(S).reshape(p, k * u, d_obs, d_obs),
    ).entropy()


def lg_utility(
    q_obs: tuple[MixtureGaussian, ...],
    targets: dict[str, Distribution],
    block_names: tuple[str, ...] | None = None,
) -> TermTree:
    """``E_q(o)[log C(o)]`` per designated continuous observation block.

    ``q_obs`` are the predicted observation densities (:func:`predictive_obs`);
    ``targets`` maps modality names to Gaussian or mixture preference densities.
    Undesignated modalities are omitted. The leaf is exact for a Gaussian target
    and a Kolchinsky–Tracey Interval for a mixture target.
    """
    names = _block_names("modality", len(q_obs), block_names)
    leaves = {
        name: -q.cross_entropy_to(targets[name])
        for name, q in zip(names, q_obs, strict=True)
        if name in targets
    }
    return {"utility": leaves}


def lg_ambiguity(
    models: tuple[LinearGaussianObservation, ...],
    mode_beliefs: tuple[tuple[Categorical, ...], ...],
    block_names: tuple[str, ...] | None = None,
    q_blocks: tuple[Gaussian | MixtureGaussian, ...] | None = None,
    selected: tuple[tuple[bool, ...] | None, ...] | None = None,
    dims: tuple[tuple[int, ...] | None, ...] | None = None,
) -> TermTree:
    """``E_q[H[y | S]]`` per LG modality: the channel's ambiguity leaf.

    One leaf per modality given, designated or not. ``mode_beliefs[i]`` is modality ``i``'s
    discrete parent beliefs (empty for a non-switching modality). ``selected[i]`` masks its
    dependency set as in :func:`selected_obs_entropy`; with the whole set (the default) the leaf
    is the exact ``E[H[y | x, mode]]``. A partial mask needs ``q_blocks[i]``, the observed
    factor's belief, and pairs with the same mask in :func:`state_info_gain`. ``dims[i]``
    (``None``: the whole factor) selects dimensions of a non-switching modality's Gaussian
    observed factor, and the leaf is :func:`sub_block_obs_entropy` at ``q_blocks[i]``.
    """
    names = _block_names("modality", len(models), block_names)
    masks = selected if selected is not None else (None,) * len(models)
    leaves = {}
    for i, (name, model, parents) in enumerate(zip(names, models, mode_beliefs, strict=True)):
        if dims is not None and dims[i] is not None:
            leaves[name] = Interval.exact(sub_block_obs_entropy(q_blocks[i], model, dims[i]))
        elif _full(masks[i]):
            q = None if q_blocks is None else q_blocks[i]
            leaves[name] = Interval.exact(conditional_obs_entropy(model, parents, q=q))
        elif q_blocks is None:
            raise ValueError(
                f"lg_ambiguity: modality {name!r} has the partial selection {masks[i]}, whose "
                "conditional entropy depends on the observed factor's belief; pass q_blocks= "
                "(one belief per modality)"
            )
        else:
            leaves[name] = selected_obs_entropy(q_blocks[i], model, tuple(parents), masks[i])
    return {"ambiguity": leaves}


def state_info_gain(
    q_blocks: tuple[Gaussian | MixtureGaussian, ...],
    models: tuple[LinearGaussianObservation, ...],
    block_names: tuple[str, ...] | None = None,
    mode_beliefs: tuple[tuple[Categorical, ...], ...] | None = None,
    selected: tuple[tuple[bool, ...] | None, ...] | None = None,
    dims: tuple[tuple[int, ...] | None, ...] | None = None,
) -> TermTree:
    """``I(y; S) = H[q(y)] − E_q[H[y | S]]`` per LG modality, ``S`` the selected parents.

    Channel-keyed ``infogain`` leaves. The dependency set is the observed factor ``x`` plus, on a
    switching modality, its mode parents ``mode_beliefs[i]``. By default ``S`` is the whole set
    and the leaf is the joint ``I(y; x, mode)`` (``I(y; x)`` without modes), with the exact
    :func:`conditional_obs_entropy`. ``selected[i]`` masks the set in ``A_deps`` order, and the
    unselected parents are marginalised (:func:`selected_obs_entropy`). ``dims[i]`` (``None``:
    the whole factor) selects dimensions ``x_S`` of a non-switching modality's Gaussian factor,
    and the leaf is ``I(y; x_S)`` with :func:`sub_block_obs_entropy`. ``H[q(y)]`` is a
    Kolchinsky–Tracey bound pair over the ``K·M`` predictive mixture of :func:`predictive_obs`, so
    every leaf is an Interval. Design: docs/design/per-variable-infogain.md V9, V10, V11
    """
    names = _block_names("modality", len(q_blocks), block_names)
    masks = selected if selected is not None else (None,) * len(q_blocks)
    leaves = {}
    for i, (name, q, model) in enumerate(zip(names, q_blocks, models, strict=True)):
        parents = () if mode_beliefs is None else tuple(mode_beliefs[i])
        model = lg_point(model)  # one noise lowering shared by both terms
        if dims is not None and dims[i] is not None:
            cond_h = Interval.exact(sub_block_obs_entropy(q, model, dims[i]))
        elif _full(masks[i]):
            cond_h = Interval.exact(conditional_obs_entropy(model, parents, q=q))
        else:
            cond_h = selected_obs_entropy(q, model, parents, masks[i])
        leaves[name] = predictive_entropy(q, model, parents) - cond_h
    return {INFOGAIN: leaves}


def _joint_mode_log_probs(parents: tuple[Categorical, ...]) -> Array:
    """Joint mode log-belief over the discrete parents, flattened row-major ``(..., M)``."""
    joint = parents[0].log_probs
    for b in parents[1:]:
        joint = joint[..., :, None] + b.log_probs[..., None, :]
        joint = joint.reshape(joint.shape[:-2] + (-1,))
    return joint


#: Draws of a matrix-normal inverse-Wishart posterior :func:`param_info_gain` averages over.
MNIW_DRAWS = 256


def mniw_draws(
    model: LinearGaussianObservationMNIW, n: int = MNIW_DRAWS
) -> tuple[Array, Array, Array, Array]:
    """What :func:`param_info_gain` reads off the posterior, hoisted out of the planning scan:
    ``n`` joint draws of the head and the noise covariance, ``(n, d, p)`` and ``(n, d, d)``,
    from fixed keys, so the leaf is deterministic and every policy is scored on the same draws;
    the noise mean ``Ψ / (ν − d − 1)``; and ``E[log det Σ]``, exact.

    The noise mean exists only for ``ν > d + 1``: a concrete ``ν`` below it raises here, a
    traced one when the computation runs.
    """
    d = model.obs_dim
    floor = f"the novelty of a matrix-normal inverse-Wishart emission needs nu > d + 1 = {d + 1}"
    if not isinstance(model.nu, jax.core.Tracer) and not np.all(np.asarray(model.nu) > d + 1):
        raise ValueError(f"{floor} (the noise mean must exist); got nu = {np.asarray(model.nu)}")
    nu = eqx.error_if(model.nu, ~jnp.all(model.nu > d + 1), floor)
    draws = jax.vmap(model.sample)(jax.random.split(jax.random.PRNGKey(0), n))
    noise = draws.noise @ jnp.swapaxes(draws.noise, -1, -2)
    i = jnp.arange(d, dtype=model.Psi.dtype)
    log_det_noise = (
        jnp.linalg.slogdet(model.Psi)[1]
        - d * jnp.log(2.0)
        - jnp.sum(jax.scipy.special.digamma((nu[..., None] - i) / 2.0), axis=-1)
    )
    return draws.head, noise, model.Psi / (nu - d - 1.0)[..., None, None], log_det_noise


def _student_t_entropy(cov: Array, df: Array) -> Array:
    """Entropy of the multivariate Student-t with ``df`` degrees of freedom (``df > 2``) and
    covariance ``cov``."""
    d = cov.shape[-1]
    half_df, half_sum = df / 2.0, (df + d) / 2.0
    return (
        0.5 * jnp.linalg.slogdet(cov * ((df - 2.0) / df)[..., None, None])[1]
        + 0.5 * d * jnp.log(df * jnp.pi)
        + jax.scipy.special.gammaln(half_df)
        - jax.scipy.special.gammaln(half_sum)
        + half_sum * (jax.scipy.special.digamma(half_sum) - jax.scipy.special.digamma(half_df))
    )


def param_info_gain(
    q: Gaussian | MixtureGaussian,
    model: LinearGaussianObservationMNIW,
    draws: tuple[Array, Array, Array, Array],
) -> Array:
    """``I(θ; y)``, the information one reading ``y = W x̃ + e`` carries about a matrix-normal
    inverse-Wishart emission's ``θ = (W, Σ)``, with the state ``x`` marginalised under ``q``
    (a mixture through its mean and covariance) and ``draws`` from :func:`mniw_draws`.

    ``H[y] − E_θ H[y | θ]``. The reading at a fixed ``θ`` is exactly Gaussian,
    ``N(W μ̃, W_x Σ_x W_xᵀ + Σ)``, so the second term is ``E[log det Σ]``, exact, plus the
    draws' average of what the state's spread adds to it. ``H[y]`` has no closed form; ``y``'s
    covariance does, ``A + B`` with ``A = (1 + μ̃ᵀVμ̃ + tr(V_xx Σ_x)) E[Σ]`` the noise's
    part and ``B = M_x Σ_x M_xᵀ`` the state's spread through the mean head. At a known state
    ``y`` is the Student-t predictive with ``ν − d + 1`` degrees of freedom; the state's
    spread adds a Gaussian part. So ``H[y]`` is the Student-t entropy at ``y``'s covariance,
    moved toward the Gaussian entropy there (a ceiling) by ``B``'s share, ``tr(B (A + B)⁻¹) /
    d``. At a known state the leaf is exact, ``(d/2) log(1 + x̃ᵀVx̃)`` plus the noise's share;
    elsewhere it is an estimate, clipped at zero, which the draws' noise could cross. Design:
    docs/design/parameter-learning.md D10
    """
    if isinstance(q, MixtureGaussian):
        q = q.moment_match()
    heads, noise, mean_noise, log_det_noise = draws
    d, k = model.obs_dim, q.dim
    mu = jnp.concatenate([q.loc, jnp.ones((*q.loc.shape[:-1], 1), q.loc.dtype)], axis=-1)
    v = jnp.linalg.inv(model.V_inv)
    sx = q.cov
    m_x = model.M[..., :, :k]
    noise_part = (1.0 + mu @ v @ mu + jnp.trace(v[..., :k, :k] @ sx)) * mean_noise
    state_part = m_x @ sx @ jnp.swapaxes(m_x, -1, -2)
    cov_y = noise_part + state_part
    share = jnp.trace(jnp.linalg.solve(cov_y, state_part)) / d
    gaussian = 0.5 * (d * (1.0 + _LOG_2PI) + jnp.linalg.slogdet(cov_y)[1])
    student = _student_t_entropy(cov_y, model.nu - d + 1.0)
    w_x = heads[..., :, :k]
    spread = w_x @ sx @ jnp.swapaxes(w_x, -1, -2)
    added = jnp.linalg.slogdet(spread + noise)[1] - jnp.linalg.slogdet(noise)[1]
    conditional = 0.5 * (d * (1.0 + _LOG_2PI) + log_det_noise + jnp.mean(added, axis=0))
    return jnp.maximum(student + share * (gaussian - student) - conditional, 0.0)
