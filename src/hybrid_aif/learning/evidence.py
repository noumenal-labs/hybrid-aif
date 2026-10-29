"""Closed-form evidence per conjugate family: what a posterior's statistics were worth.

:func:`log_evidence` is the log marginal likelihood of the statistics a posterior absorbed,
relative to the prior it absorbed them from: the ratio of the family's normaliser at the two
parameter settings, which is the whole content of conjugacy. It is exact for the families the
engine learns in closed form (a Dirichlet over a table or an initial prior, an inverse-Gamma
or inverse-Wishart noise under a point head, the matrix-normal inverse-Wishart over a head
and its noise together), and it is the quantity a merge of two levels
(``docs/design/structure-learning.md`` S4) and a description length
(``docs/design/description-length.md`` D5, D6) both read. :func:`absorbed_count` is the mass a
posterior took in, the resolution a family is priced at (D1). A Dirichlet with dead levels
(zero concentration, a growable factor's) is scored over its live levels, a woken level's
prior pseudo-count being the growth propensity (structure-learning.md S3).

A posterior learned under a learning rate other than one holds tempered statistics; the
number is then the evidence of the statistics it holds. Decay anchored at the prior only
reweights those statistics, and the number is their evidence likewise. Decay toward any other
anchor lets the prior's own mass go: the absorbed count can be negative and the ratio of
normalisers is no marginal likelihood at all, so a slot whose count is not positive is
reported unscored. Learning never stores the prior beside the posterior, so both are passed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gammaln, multigammaln

from hybrid_aif.distributions import ConditionalDirichlet, Dirichlet, InverseGamma, InverseWishart
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianObservationMNIW

_LOG_2PI = math.log(2.0 * math.pi)
_LOG_2 = math.log(2.0)


class Evidence(NamedTuple):
    """One slot's evidence: ``nats`` (``None`` when the slot is unscored), whether the number is
    the exact marginal likelihood, and the mass the slot absorbed (``None`` when it holds no
    posterior)."""

    nats: Array | None
    exact: bool
    count: Array | None


def _log_det(spd: Array) -> Array:
    chol = jnp.linalg.cholesky(spd)
    return 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)


def _log_z_inverse_wishart(psi: Array, nu: Array) -> Array:
    """``log Γ_d(ν/2) + (νd/2) log 2 − (ν/2) log|Ψ|``: the inverse-Wishart's log normaliser."""
    d = psi.shape[-1]
    return multigammaln(nu / 2.0, d) + 0.5 * nu * d * _LOG_2 - 0.5 * nu * _log_det(psi)


def _same(prior, posterior) -> None:
    if type(prior) is not type(posterior):
        raise TypeError(
            f"log_evidence needs a prior and a posterior of one family, got "
            f"{type(prior).__name__} and {type(posterior).__name__}"
        )


def _live_dirichlet(prior_conc: Array, post_conc: Array, growth: Array | None, axis: int):
    """The Dirichlet-multinomial ratio over the posterior's live levels: a level with zero
    posterior concentration is dead and is no level of the family, and a level the prior holds
    dead that the posterior holds live was woken, its prior pseudo-count the growth
    propensity. Without dead levels this is the plain ratio of log-Beta functions, the evidence
    of Bayesian model reduction (Friston et al., arXiv:1805.07092) that SPM's
    ``spm_MDP_log_evidence`` applies to Dirichlet counts."""
    live = post_conc > 0.0
    if growth is not None:
        prior_conc = jnp.where((prior_conc == 0.0) & live, growth, prior_conc)
    prior_conc = jnp.broadcast_to(prior_conc, post_conc.shape)

    def log_beta(c: Array) -> Array:
        terms = jnp.where(live, gammaln(jnp.where(live, c, 1.0)), 0.0)
        return jnp.sum(terms, axis=axis) - gammaln(jnp.sum(jnp.where(live, c, 0.0), axis=axis))

    return log_beta(post_conc) - log_beta(prior_conc)


def _dirichlet(prior: Dirichlet, posterior: Dirichlet) -> Array:
    return _live_dirichlet(prior.concentration, posterior.concentration, posterior.growth, -1)


def _conditional_dirichlet(prior: ConditionalDirichlet, posterior: ConditionalDirichlet) -> Array:
    return _live_dirichlet(
        prior.concentration, posterior.concentration, posterior.growth, prior._child_axis
    )


def _live_count(prior_conc: Array, post_conc: Array, growth: Array | None, axis: int) -> Array:
    live = post_conc > 0.0
    if growth is not None:
        prior_conc = jnp.where((prior_conc == 0.0) & live, growth, prior_conc)
    prior_conc = jnp.broadcast_to(prior_conc, post_conc.shape)
    return jnp.sum(jnp.where(live, post_conc - prior_conc, 0.0), axis=axis)


def _inverse_gamma(prior: InverseGamma, posterior: InverseGamma) -> Array:
    """Per dimension ``log Γ(α') − log Γ(α) + α log β − α' log β' − (α' − α) log 2π``,
    summed over the dimensions: the evidence for ``2(α' − α)`` zero-mean draws."""
    a, b, a1, b1 = prior.alpha, prior.beta, posterior.alpha, posterior.beta
    per_dim = gammaln(a1) - gammaln(a) + a * jnp.log(b) - a1 * jnp.log(b1) - (a1 - a) * _LOG_2PI
    return jnp.sum(per_dim, axis=-1)


def _inverse_wishart(prior: InverseWishart, posterior: InverseWishart) -> Array:
    """``log Z(Ψ', ν') − log Z(Ψ, ν) − (ν' − ν) d/2 · log 2π``: the evidence for ``ν' − ν``
    zero-mean draws in ``d`` dimensions."""
    d = prior.obs_dim
    return (
        _log_z_inverse_wishart(posterior.Psi, posterior.nu)
        - _log_z_inverse_wishart(prior.Psi, prior.nu)
        - 0.5 * (posterior.nu - prior.nu) * d * _LOG_2PI
    )


def _mniw(prior: LinearGaussianObservationMNIW, posterior: LinearGaussianObservationMNIW) -> Array:
    """The inverse-Wishart ratio with the head integrated out: ``(d/2)(log|V⁻¹| − log|V'⁻¹|)``
    is the column-covariance ratio, the evidence for ``ν' − ν`` regression pairs."""
    d = prior.obs_dim
    return (
        _log_z_inverse_wishart(posterior.Psi, posterior.nu)
        - _log_z_inverse_wishart(prior.Psi, prior.nu)
        + 0.5 * d * (_log_det(prior.V_inv) - _log_det(posterior.V_inv))
        - 0.5 * (posterior.nu - prior.nu) * d * _LOG_2PI
    )


def _lg_noise(prior: LinearGaussianObservation, posterior: LinearGaussianObservation):
    if not prior.learnable_noise or prior.learnable_head:
        return None
    _same(prior.noise, posterior.noise)
    if isinstance(prior.noise, InverseGamma):
        return _inverse_gamma(prior.noise, posterior.noise)
    return _inverse_wishart(prior.noise, posterior.noise)


def log_evidence(prior, posterior) -> Array | None:
    """The log marginal likelihood of the statistics ``posterior`` absorbed from ``prior``, one
    number per independent family (a table's parent combinations, an LG declaration's modes):
    the array has the declaration's parent shape, and a slot's evidence is its sum.

    ``None`` for a declaration with no closed form: a point table or head, a neural
    conditional, an LG head with its own matrix-normal posterior.
    """
    _same(prior, posterior)
    if isinstance(prior, ConditionalDirichlet):
        return _conditional_dirichlet(prior, posterior)
    if isinstance(prior, Dirichlet):
        return _dirichlet(prior, posterior)
    if isinstance(prior, InverseGamma):
        return _inverse_gamma(prior, posterior)
    if isinstance(prior, InverseWishart):
        return _inverse_wishart(prior, posterior)
    if isinstance(prior, LinearGaussianObservationMNIW):
        return _mniw(prior, posterior)
    if isinstance(prior, LinearGaussianObservation):
        return _lg_noise(prior, posterior)
    return None


def absorbed_count(prior, posterior) -> Array | None:
    """The mass ``posterior`` absorbed beyond ``prior``, per independent family and with the same
    shape as :func:`log_evidence`: the concentration a Dirichlet gained, the degrees of
    freedom an inverse-Wishart or a matrix-normal inverse-Wishart gained, twice the shape an
    inverse-Gamma gained (read off its first dimension; learning gives every dimension the
    same mass). ``None`` where :func:`log_evidence` is."""
    _same(prior, posterior)
    if isinstance(prior, ConditionalDirichlet):
        return _live_count(
            prior.concentration, posterior.concentration, posterior.growth, prior._child_axis
        )
    if isinstance(prior, Dirichlet):
        return _live_count(prior.concentration, posterior.concentration, posterior.growth, -1)
    if isinstance(prior, InverseGamma):
        return 2.0 * (posterior.alpha - prior.alpha)[..., 0]
    if isinstance(prior, InverseWishart | LinearGaussianObservationMNIW):
        return posterior.nu - prior.nu
    if isinstance(prior, LinearGaussianObservation):
        if not prior.learnable_noise or prior.learnable_head:
            return None
        return absorbed_count(prior.noise, posterior.noise)
    return None


def evidence_by_slot(system, prior, posterior) -> dict[str, Evidence]:
    """Every slot's :class:`Evidence`, keyed as ``learn`` keys its rates
    (``"likelihood:<modality>"``, ``"transition:<factor>"``,
    ``"initial_state_prior:<factor>"``), summed over the slot's parent combinations; the two
    params are name-keyed as every face takes them.

    A slot with no closed form is ``Evidence(None, False, None)``; a conjugate slot that absorbed
    no mass, or let mass go under a decay not anchored at its prior, is
    ``Evidence(None, True, count)``, unscored rather than zero, so a trace that never informed
    it is not read as having confirmed it. Eager.
    """
    out: dict[str, Evidence] = {}
    names = {
        "likelihood": system.modality_names,
        "transition": system._facs,
        "initial_state_prior": system._facs,
    }
    for slot, labels in names.items():
        before, after = getattr(prior, slot), getattr(posterior, slot)
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise TypeError(
                f"{slot}: params are keyed by name; a positional container needs "
                'System.from_positional("params", params)'
            )
        for label in labels:
            key = f"{slot}:{label}"
            if label not in before or label not in after:
                raise ValueError(f"{key}: missing from the prior or the posterior")
            p, q = before[label], after[label]
            if type(p) is not type(q):
                raise TypeError(
                    f"{key}: the prior holds {type(p).__name__} and the posterior "
                    f"{type(q).__name__}; a posterior is its prior's family"
                )
            nats = log_evidence(p, q)
            if nats is None:
                out[key] = Evidence(None, False, None)
                continue
            count = jnp.sum(absorbed_count(p, q))
            total = jnp.sum(nats)
            out[key] = Evidence(total if float(count) > 0.0 else None, True, count)
    return out
