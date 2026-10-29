"""Mean-field fixed-point iteration for factorized discrete models.

Each modality's likelihood is a raw log-likelihood tensor ``(num_obs_m, *[num_levels_f for f in
A_deps[m]])``, a :class:`ConditionalCategorical` (its normalized log table) or a
:class:`ConditionalDirichlet` (its ``expected_log_params`` table); priors are one
:class:`Categorical` per factor; observations are integer indices or one-hot arrays, one per
modality. Collections are canonically tuples; any sequence is accepted. The iteration is
``Q(s_f) ∝ exp(E_{q(-f)}[log P(o|s)] + log prior_f)`` for a fixed number of sweeps (static
shapes, jit-safe, vmappable). Adapted from pymdp's ``algos.run_factorized_fpi`` (pymdp/algos.py);
pymdp is the parity oracle in tests/test_discrete_parity.py.

Design: docs/design/zero-support-evidence.md (tempering and the support flag)
"""

import math
import string

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.distributions import Categorical, ConditionalCategorical, ConditionalDirichlet

_AXES = string.ascii_lowercase

Likelihood = Array | ConditionalCategorical | ConditionalDirichlet

#: The tempering clip (``zero-support-evidence.md`` Z4): every log term the FPI reads is
#: clipped here, so ``-inf`` is a legal table entry and an all-impossible row normalises
#: instead of producing NaN. ``exp`` of it is exactly 0.0 in float32; it is not -1e30 because
#: a finite term added to -1e30 is lost to the ulp, and every impossible level would tie.
LOG_ZERO = -1.0e4

#: The default support floor of :func:`unsupported` (Z1/Z7), ``log(1e-12)``: below any
#: probability a composer meant to be possible, above any tempered zero.
SUPPORT_FLOOR = math.log(1e-12)


def unsupported(
    log_priors: list[Array],
    messages: list[list[Array]],
    support_floor: float = SUPPORT_FLOOR,
) -> tuple[Array, ...]:
    """Per factor: is no level acceptable to every modality and the prior?

    ``messages[f]`` are the converged expected log-likelihood messages each present modality sends
    factor ``f``; the prior is one more term. A level is acceptable to a term above
    ``support_floor``; the factor is unsupported when the maximum over levels of the minimum over
    terms is below it. A masked-out modality sends the zero message, which no floor rejects.

    Design: docs/design/zero-support-evidence.md Z1
    """
    out = []
    for lp, msgs in zip(log_priors, messages, strict=True):
        worst = jnp.min(jnp.stack([lp, *msgs]), axis=0) if msgs else lp
        out.append(jnp.max(worst, axis=-1) < support_floor)
    return tuple(out)


def _as_log_likelihood(ll: Likelihood) -> Array:
    """Lower a likelihood input to the child-first log-likelihood tensor, tempered.

    The clip at :data:`LOG_ZERO` happens before normalisation, so an all-``-inf`` parent column
    does not normalise to ``NaN``. Every reader of a tabular likelihood lowers through here.
    """
    if isinstance(ll, ConditionalDirichlet):
        return jnp.maximum(ll.expected_log_params(), LOG_ZERO)
    if isinstance(ll, ConditionalCategorical):
        raw = ll.log_table
        normalised = jax.nn.log_softmax(jnp.maximum(raw, LOG_ZERO), axis=ll._child_axis)
        #  an entry impossible before normalisation stays impossible after it (an all -inf
        #  column must not normalise to a uniform emission)
        return jnp.where(raw <= LOG_ZERO, LOG_ZERO, normalised)
    return jnp.maximum(jnp.asarray(ll), LOG_ZERO)


def _observed_log_likelihood(log_likelihood: Array, obs: Array) -> Array:
    """Contract the modality axis with an observation (int index or one-hot)."""
    if jnp.issubdtype(obs.dtype, jnp.integer):
        return jnp.take(log_likelihood, obs, axis=0)
    return jnp.tensordot(obs, log_likelihood, axes=([0], [0]))


def _marginalize_keep(ll: Array, qs: list[Array], keep: int) -> Array:
    """E_q[ll] over all local axes except ``keep``; static einsum from dep structure."""
    in_sub = _AXES[: ll.ndim]
    other = [(qs[j], _AXES[j]) for j in range(ll.ndim) if j != keep]
    spec = ",".join([in_sub] + [s for _, s in other]) + "->" + _AXES[keep]
    return jnp.einsum(spec, ll, *[q for q, _ in other])


def mean_field_free_energy(
    log_likelihoods: tuple[Likelihood, ...],
    A_deps: tuple[tuple[int, ...], ...],
    priors: tuple[Categorical, ...],
    posteriors: tuple[Categorical, ...],
    observations: tuple[Array, ...],
    masks: tuple[bool | Array, ...] | None = None,
) -> Array:
    """The variational free energy of mean-field ``posteriors`` for one tick's observations:
    ``Σ_f KL(q_f ‖ prior_f) − Σ_m E_{q(parents_m)}[log A_m[o_m | parents_m]]``, a scalar.

    ``−F`` is a lower bound on the tick's log evidence ``log p(o)`` under the priors, exact for
    one factor whose posterior the sweep found; the terms are tempered as :func:`fpi` tempers
    them, and a masked-out modality contributes nothing.
    """
    total = jnp.zeros(())
    for prior, q in zip(priors, posteriors, strict=True):
        log_prior = jnp.maximum(prior.log_probs, LOG_ZERO)
        probs = q.probs
        total = total + jnp.sum(jax.scipy.special.xlogy(probs, probs) - probs * log_prior)
    for m, (ll, obs, deps) in enumerate(zip(log_likelihoods, observations, A_deps, strict=True)):
        present = True if masks is None else masks[m]
        if present is False:
            continue
        observed = _observed_log_likelihood(_as_log_likelihood(ll), jnp.asarray(obs))
        expected = observed
        for f in reversed(deps):  # contract the parent axes, last first
            expected = jnp.tensordot(expected, posteriors[f].probs, axes=([expected.ndim - 1], [0]))
        if present is not True:
            expected = jnp.where(jnp.asarray(present), expected, jnp.zeros_like(expected))
        total = total - expected
    return total


def fpi(
    log_likelihoods: tuple[Likelihood, ...],
    A_deps: tuple[tuple[int, ...], ...],
    priors: tuple[Categorical, ...],
    observations: tuple[Array, ...],
    num_iters: int = 16,
    masks: tuple[bool | Array, ...] | None = None,
) -> tuple[Categorical, ...]:
    """Mean-field posteriors over state factors given one observation per modality.

    :func:`fpi_support` is the same sweep, also returning the per-factor ``unsupported`` flags.
    Every log term is clipped at :data:`LOG_ZERO` on the way in, so ``-inf`` entries are legal and
    an observation no state can explain gives a finite posterior rather than ``NaN``. The returned
    logits are unnormalised (messages plus log prior).

    ``masks``, one entry per modality, marks which observations are present; a masked-out modality
    contributes the zero message. A Python ``False`` skips the modality statically and never reads
    its observation (which may be ``None``); a traced flag selects at run time, so padding of the
    right shape must be there.

    Design: docs/design/zero-support-evidence.md Z4
    """
    return fpi_support(log_likelihoods, A_deps, priors, observations, num_iters, masks)[0]


def fpi_support(
    log_likelihoods: tuple[Likelihood, ...],
    A_deps: tuple[tuple[int, ...], ...],
    priors: tuple[Categorical, ...],
    observations: tuple[Array, ...],
    num_iters: int = 16,
    masks: tuple[bool | Array, ...] | None = None,
    support_floor: float = SUPPORT_FLOOR,
    support_exempt: tuple[int, ...] = (),
) -> tuple[tuple[Categorical, ...], tuple[Array, ...]]:
    """:func:`fpi`, returning ``(beliefs, unsupported)`` — see :func:`unsupported`.

    ``support_exempt`` names modalities whose messages the support check skips: a density's
    scores (a Gaussian emission's) are not probabilities, and a density has every level in its
    support.
    """
    obs_lls = []
    for m, (ll, obs) in enumerate(zip(log_likelihoods, observations, strict=True)):
        tensor = _as_log_likelihood(ll)  # tempered on the way in: -inf is legal
        present = True if masks is None else masks[m]
        if present is False:  # statically absent: the observation is never read
            obs_lls.append(jnp.zeros(tensor.shape[1:], tensor.dtype))
            continue
        observed = _observed_log_likelihood(tensor, jnp.asarray(obs))
        if present is not True:
            observed = jnp.where(jnp.asarray(present), observed, jnp.zeros_like(observed))
        obs_lls.append(observed)
    log_priors = [jnp.maximum(p.log_probs, LOG_ZERO) for p in priors]

    def messages(log_q: list[Array], skip: tuple[int, ...] = ()) -> list[list[Array]]:
        q = [jax.nn.softmax(lq, axis=-1) for lq in log_q]
        per_factor: list[list[Array]] = [[] for _ in log_priors]
        for m, deps in enumerate(A_deps):
            if m in skip:
                continue
            local_qs = [q[f] for f in deps]
            for i, f in enumerate(deps):
                per_factor[f].append(_marginalize_keep(obs_lls[m], local_qs, i))
        return per_factor

    def sweep(log_q: list[Array], _: None) -> tuple[list[Array], None]:
        per_factor = messages(log_q)
        return [
            sum(msgs, jnp.zeros_like(lp)) + lp
            for msgs, lp in zip(per_factor, log_priors, strict=True)
        ], None

    init = [jnp.zeros_like(lp) for lp in log_priors]
    log_q, _ = jax.lax.scan(sweep, init, None, length=num_iters)
    flags = unsupported(log_priors, messages(log_q, tuple(support_exempt)), support_floor)
    return tuple(Categorical(logits=lq) for lq in log_q), flags
