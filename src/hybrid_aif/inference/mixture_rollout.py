"""Grow-then-reduce mixture propagation through mode-conditioned linear dynamics.

Each step pushes every component through each of M linear-Gaussian transitions (exact grow to
K·M components), then reduces to a static budget: Runnalls (``rollout_reduced``) or
single-Gaussian moment matching (``rollout_gpb1``). Horizon, component counts and shapes are
static.

Design: docs/de-risk-report.md §2
"""

import jax.numpy as jnp
from jax import Array, lax

from hybrid_aif.distributions.conditional import ConditionalGaussian
from hybrid_aif.distributions.gaussian import Gaussian
from hybrid_aif.distributions.mixture import MixtureGaussian
from hybrid_aif.inference.linear_gaussian import predict


def propagate(
    m: MixtureGaussian,
    Fs: Array,
    bs: Array | ConditionalGaussian,
    Q_chols: Array | None = None,
    transition_log_weights: Array | None = None,
) -> MixtureGaussian:
    """Exact grow: K components through M transitions -> K*M components.

    ``Fs`` (M, dim, dim), ``bs`` (M, dim), ``Q_chols`` (M, dim, r), any factors of the process
    noise; ``transition_log_weights`` (M,) shared or (K, M) per-component, added to the
    component log-weights as given. ``bs`` may
    instead be a :class:`ConditionalGaussian` with ``parent_shape (M,)`` supplying offsets and noise
    factors, with ``Q_chols`` then ``None``. Output is component-major: slot k*M + j is component k
    through mode j. Batch axes of ``m`` broadcast through.
    """
    if isinstance(bs, ConditionalGaussian):
        if Q_chols is not None:
            raise ValueError("pass Q_chols=None when bs is a ConditionalGaussian")
        bs, Q_chols = bs.loc, bs.stacked_chol()
    if Q_chols is None or transition_log_weights is None:
        raise ValueError("Q_chols and transition_log_weights are required with array inputs")
    K, M, dim = m.num_components, Fs.shape[-3], m.dim
    grown = predict(
        Gaussian(loc=m.means[..., :, None, :], chol=m.chols[..., :, None, :, :]),
        Fs,
        bs,
        Q_chols,
    )  # batch (..., K, M)
    log_w = m.log_weights[..., :, None] + transition_log_weights
    log_w = jnp.broadcast_to(log_w, grown.loc.shape[:-1])
    batch = log_w.shape[:-2]
    return MixtureGaussian(
        log_weights=log_w.reshape(batch + (K * M,)),
        means=grown.loc.reshape(batch + (K * M, dim)),
        chols=grown.chol.reshape(batch + (K * M, dim, dim)),
    )


def _to_budget(m: MixtureGaussian, k: int) -> MixtureGaussian:
    """Bring a mixture to exactly k slots: reduce if over, pad dead slots if under."""
    if m.num_components > k:
        return m.reduce(k)
    if m.num_components < k:
        pad = k - m.num_components
        dead_lw = jnp.full(m.log_weights.shape[:-1] + (pad,), -jnp.inf, m.log_weights.dtype)
        pad_means = jnp.repeat(m.means[..., :1, :], pad, axis=-2)
        pad_chols = jnp.repeat(m.chols[..., :1, :, :], pad, axis=-3)
        return MixtureGaussian(
            log_weights=jnp.concatenate([m.log_weights, dead_lw], axis=-1),
            means=jnp.concatenate([m.means, pad_means], axis=-2),
            chols=jnp.concatenate([m.chols, pad_chols], axis=-3),
        )
    return m


def _as_weight_schedule(tlw: Array, horizon: int, k: int) -> Array:
    """Lift ``transition_log_weights`` to a per-step schedule ``(horizon, ...)``.

    Static tables, ``(M,)`` or ``(k, M)``, are broadcast across the horizon; arrays with a leading
    horizon axis, ``(horizon, M)`` or ``(horizon, k, M)``, pass through. Detection is static, by
    ``ndim`` and leading length: a 2-D array of leading length ``k`` is always the per-component
    table, so when ``horizon == k`` pass a per-step shared schedule in the explicit 3-D form.
    """
    if tlw.ndim == 1 or (tlw.ndim == 2 and tlw.shape[0] == k):
        return jnp.broadcast_to(tlw, (horizon,) + tlw.shape)
    if (tlw.ndim == 2 or tlw.ndim == 3) and tlw.shape[0] == horizon:
        return tlw
    raise ValueError(
        "transition_log_weights must be (M,), (k, M), (horizon, M), or (horizon, k, M); "
        f"got shape {tlw.shape} with horizon={horizon}, k={k}"
    )


def rollout_reduced(
    m: MixtureGaussian,
    Fs: Array,
    bs: Array,
    Q_chols: Array,
    transition_log_weights: Array,
    horizon: int,
    k: int,
) -> tuple[MixtureGaussian, MixtureGaussian]:
    """Scan ``horizon`` steps of propagate -> reduce(k) with static shapes.

    ``transition_log_weights`` is a static table, ``(M,)`` or ``(K, M)``, or a per-step schedule,
    ``(horizon, M)`` or ``(horizon, K, M)`` (see :func:`_as_weight_schedule`); a per-component table
    needs K == k. The carry is first brought to exactly k components. Returns
    ``(final, trajectory)``, the trajectory stacked on a leading ``horizon`` axis.
    """
    m = _to_budget(m, k)
    schedule = _as_weight_schedule(transition_log_weights, horizon, k)

    def step(carry: MixtureGaussian, lw_t: Array) -> tuple[MixtureGaussian, MixtureGaussian]:
        nxt = propagate(carry, Fs, bs, Q_chols, lw_t).reduce(k)
        return nxt, nxt

    return lax.scan(step, m, schedule)


def rollout_gpb1(
    m: MixtureGaussian | Gaussian,
    Fs: Array,
    bs: Array,
    Q_chols: Array,
    transition_log_weights: Array,
    horizon: int,
) -> tuple[Gaussian, Gaussian]:
    """GPB1 variant: propagate -> exact single-Gaussian moment_match each step."""
    g = m.moment_match() if isinstance(m, MixtureGaussian) else m

    def step(carry: Gaussian, _) -> tuple[Gaussian, Gaussian]:
        as_mixture = MixtureGaussian.from_gaussian(carry)
        nxt = propagate(as_mixture, Fs, bs, Q_chols, transition_log_weights).moment_match()
        return nxt, nxt

    return lax.scan(step, g, None, length=horizon)
