"""Gaussian-sum conditioning: exact Bayesian update of a mixture on LG evidence.

Conditioning a :class:`MixtureGaussian` on ``y = H x + c + eps`` is exact and keeps ``K``: each
component gets the exact Kalman ``condition`` and is re-weighted by its marginal likelihood
(the Gaussian-sum filter measurement step, Alspach & Sorenson 1972); dead components stay dead.
Under mode-aligned scheduling, :func:`mode_evidence_message` lifts each mode's slot group's
evidence to the discrete parent(s); the switching functions condition on a mode-dependent
observation model.

Design: docs/design/mixture-filtering.md D1, D3, Phase 1.5 addendum
"""

import jax.numpy as jnp
from jax import Array
from jax.nn import log_softmax
from jax.scipy.special import logsumexp

from hybrid_aif.distributions.categorical import Categorical
from hybrid_aif.distributions.gaussian import Gaussian
from hybrid_aif.distributions.mixture import MixtureGaussian
from hybrid_aif.inference.linear_gaussian import condition


def condition_mixture_components(
    m: MixtureGaussian, H: Array, c: Array, R_chol: Array, y: Array
) -> tuple[MixtureGaussian, Array, Array]:
    """:func:`condition_mixture` plus the per-component predictive log-densities.

    Returns ``(posterior, log_evidence, loglik)``: the first two as :func:`condition_mixture`, and
    ``loglik`` the ``(..., K)`` per-component ``log N(y; H mu_j + c, H Sigma_j H^T + R)`` (what
    :func:`observation_loglik` returns), from one Kalman ``condition``.
    """
    # Per-component exact Kalman condition (y broadcast across the K axis).
    post, log_marginal = condition(m.components, H, c, R_chol, y[..., None, :])
    prior_lw = log_softmax(m.log_weights, axis=-1)  # normalize gauge; dead stay -inf
    unnorm_post_lw = prior_lw + log_marginal  # -inf + finite = -inf for dead slots
    log_evidence = logsumexp(unnorm_post_lw, axis=-1)
    posterior = MixtureGaussian(
        log_weights=unnorm_post_lw - log_evidence[..., None],
        means=post.loc,
        chols=post.chol,
    )
    return posterior, log_evidence, log_marginal


def condition_mixture(
    m: MixtureGaussian, H: Array, c: Array, R_chol: Array, y: Array
) -> tuple[MixtureGaussian, Array]:
    """Exact Bayesian update of ``m`` on ``y = H x + c + eps``, ``eps ~ N(0, RR^T)``.

    Returns ``(posterior, log_evidence)``: the ``K``-component mixture of per-component Kalman
    posteriors with log-weights ``log w_j + log N(y; H mu_j + c, H Sigma_j H^T + R)``, renormalized,
    and the exact ``log p(y) = log sum_j w_j N(y; H mu_j + c, S_j)``. Prior weights are normalized
    first, so ``log_evidence`` is gauge-free; dead (``-inf``) components contribute nothing, so both
    outputs are invariant to zero-weight padding. Batch axes broadcast per the distribution
    contract.
    """
    posterior, log_evidence, _ = condition_mixture_components(m, H, c, R_chol, y)
    return posterior, log_evidence


def observation_loglik(m: MixtureGaussian, H: Array, c: Array, R_chol: Array, y: Array) -> Array:
    """Per-component predictive log-density ``log N(y; H mu_j + c, H Sigma_j H^T + R)``.

    Shape ``(..., K)``. Dead components carry a finite padding value; the mode message masks them
    through their ``-inf`` prior log-weight.
    """
    _, ll = condition(m.components, H, c, R_chol, y[..., None, :])
    return ll


def mode_evidence_message(
    mode_beliefs: tuple[Categorical, ...],
    prior_log_weights: Array,
    obs_loglik: Array,
) -> tuple[Categorical, ...]:
    """Update discrete mode belief(s) from mode-aligned per-bank integrated evidence.

    ``mode_beliefs`` is one prior :class:`Categorical` per discrete parent, in declaration order,
    with cardinalities multiplying to ``M``. ``prior_log_weights`` and ``obs_loglik`` are
    ``(..., K)``, laid out as mode-owned contiguous banks (slot ``= config * (K // M) + local``,
    configurations row-major over the parents). The message is
    ``q(mode=cfg) ∝ q_prior(mode=cfg) · p(y | mode=cfg)``, with ``p(y | mode=cfg)`` the bank's
    within-group-normalized evidence, so the prior mode mass enters once. Returns one Categorical
    per parent: exact for a single parent, mean-field marginals otherwise.
    """
    parent_shape = tuple(b.probs.shape[-1] for b in mode_beliefs)
    M = 1
    for s in parent_shape:
        M *= s
    return _mode_marginals(mode_beliefs, mode_bank_evidence(prior_log_weights, obs_loglik, M))


def mode_bank_evidence(prior_log_weights: Array, obs_loglik: Array, M: int) -> Array:
    """``log p(y | mode=cfg)`` per mode configuration, ``(..., M)``: each mode-owned bank's
    within-bank-normalised evidence, so the prior mode mass enters nowhere here."""
    K = prior_log_weights.shape[-1]
    if K % M != 0:
        raise ValueError(
            f"mode_evidence_message: K={K} is not a multiple of the mode count M={M}; "
            "mode-aligned banks require mixture_k to be a multiple of M"
        )
    spb = K // M
    lw = prior_log_weights.reshape(prior_log_weights.shape[:-1] + (M, spb))
    ll = obs_loglik.reshape(obs_loglik.shape[:-1] + (M, spb))
    within = log_softmax(lw, axis=-1)  # p(x | mode=cfg) weights, per bank
    return logsumexp(within + ll, axis=-1)


def _mode_marginals(
    mode_beliefs: tuple[Categorical, ...], log_ev: Array
) -> tuple[Categorical, ...]:
    """Combine a joint mode prior with per-configuration log-evidence into marginals.

    ``log_ev`` is ``log p(y | mode=cfg)``, shape ``(..., M)``, row-major over the parents. Returns
    ``q(mode=cfg) ∝ q_prior(mode=cfg) · p(y | mode=cfg)`` as one Categorical per parent (exact for
    one parent, mean-field marginals otherwise).
    """
    parent_shape = tuple(b.probs.shape[-1] for b in mode_beliefs)
    joint = mode_beliefs[0].log_probs
    for b in mode_beliefs[1:]:
        joint = joint[..., :, None] + b.log_probs[..., None, :]
        joint = joint.reshape(joint.shape[:-2] + (-1,))
    post = log_softmax(joint + log_ev, axis=-1)  # joint mode posterior, (..., M)
    if len(parent_shape) == 1:
        return (Categorical(logits=post),)
    resh = post.reshape(post.shape[:-1] + parent_shape)
    out = []
    n = len(parent_shape)
    for a in range(n):
        others = tuple(-n + i for i in range(n) if i != a)
        out.append(Categorical(logits=logsumexp(resh, axis=others)))
    return tuple(out)


def _per_mode(arr: Array, m: int, event_ndims: int) -> Array:
    """Bring a shared or per-mode observation array to a leading mode axis ``(M, *event)``.

    A shared array (rank ``event_ndims``) is broadcast across the ``M`` modes; a per-mode array's
    leading parent axes are flattened row-major to ``M``, matching :func:`_mode_marginals`.
    """
    event = arr.shape[arr.ndim - event_ndims :]
    if arr.ndim == event_ndims:
        return jnp.broadcast_to(arr, (m,) + event)
    return arr.reshape((m,) + event)


def condition_mixture_switching(
    m: MixtureGaussian,
    H: Array,
    c: Array,
    R_chol: Array,
    y: Array,
    mode_log_probs: Array,
) -> tuple[MixtureGaussian, Array]:
    """Grow-condition ``m`` on a switching LG observation with the mode uncertain.

    The model is ``y = H_j x + c_j + eps``, ``eps ~ N(0, R_j R_j^T)``, with mode prior
    ``mode_log_probs`` ``(..., M)`` (joint over the modality's categorical parents, row-major);
    ``H``/``c``/``R_chol`` are shared or per-mode (:func:`_per_mode`). Each ``(component k, mode
    j)`` pair gets the exact Kalman condition under ``(H_j, c_j, R_j)`` with weight ``w_k · q(j) ·
    N(y; H_j mu_k + c_j, S_kj)``. Returns

    - the ``K·M``-component posterior, component-major (slot ``k*M + j``, as
      :func:`~hybrid_aif.inference.mixture_rollout.propagate`), log-weights normalized; reduce it
      afterwards under any schedule, ``runnalls`` included;
    - ``mode_log_evidence`` ``(..., M)``: ``log p(y | mode=j)``, normalized within each hypothesis
      (the mode prior is not folded in, so :func:`switching_mode_message` counts it once).

    Dead components stay dead; batch axes of ``m`` broadcast through.

    Design: docs/design/mixture-filtering.md, Phase 1.5 addendum
    """
    mm = mode_log_probs.shape[-1]
    d = m.dim
    Hs = _per_mode(H, mm, 2)  # (M, d_obs, d)
    cs = _per_mode(c, mm, 1)  # (M, d_obs)
    Rs = _per_mode(R_chol, mm, 2)  # (M, d_obs, d_obs)
    comps = Gaussian(loc=m.means[..., :, None, :], chol=m.chols[..., :, None, :, :])  # (..., K, 1)
    post, log_marginal = condition(comps, Hs, cs, Rs, y[..., None, None, :])  # batch (..., K, M)
    prior_lw = log_softmax(m.log_weights, axis=-1)  # (..., K); dead stay -inf
    mode_log_evidence = logsumexp(prior_lw[..., :, None] + log_marginal, axis=-2)  # (..., M)
    grown_lw = prior_lw[..., :, None] + mode_log_probs[..., None, :] + log_marginal  # (..., K, M)
    grown_lw = grown_lw - logsumexp(grown_lw, axis=(-2, -1))[..., None, None]
    batch = grown_lw.shape[:-2]
    grown = MixtureGaussian(
        log_weights=grown_lw.reshape(batch + (m.num_components * mm,)),
        means=post.loc.reshape(batch + (m.num_components * mm, d)),
        chols=post.chol.reshape(batch + (m.num_components * mm, d, d)),
    )
    return grown, mode_log_evidence


def switching_mode_message(
    mode_beliefs: tuple[Categorical, ...], mode_log_evidence: Array
) -> tuple[Categorical, ...]:
    """Discrete mode posterior from a switching observation's per-mode log-evidence.

    ``q(mode=cfg) ∝ q_prior(mode=cfg) · p(y | mode=cfg)``, with ``mode_log_evidence`` from
    :func:`condition_mixture_switching`; one updated :class:`Categorical` per parent.
    """
    return _mode_marginals(mode_beliefs, mode_log_evidence)
