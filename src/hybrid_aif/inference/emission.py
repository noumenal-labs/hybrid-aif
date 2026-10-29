"""Continuous observations over categorical parents: their per-level scores.

A linear-Gaussian modality whose every parent is categorical is ``y = c_j + noise_j``, the
parents' joint configuration ``j`` selecting the offset and the noise: a Gaussian mixture whose
component label is the parent, with no state to filter. Its message to the parents is the exact
categorical message ``q(z) ∝ prior(z) · exp ℓ_j(y)``, composed with the tabular modalities'
messages in the mean-field sweep. ``ℓ_j(y)`` is the predictive log-density of ``y`` under level
``j``: a Gaussian under a point declaration, and under a matrix-normal inverse-Wishart posterior
its Student-t posterior predictive, the exact conjugate answer rather than the point head's
density or the expected log-density under the posterior.

Design: docs/design/structure-learning.md S2
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve_triangular

from hybrid_aif.factors import LGObservation, LinearGaussianObservationMNIW, lg_point

_LOG_2PI = math.log(2.0 * math.pi)


def is_emission(decl: object) -> bool:
    """Whether an observation declaration is a Gaussian emission over categorical parents
    alone: a linear-Gaussian head with an empty regressor block (the offset column only)."""
    return isinstance(decl, LGObservation) and decl.H.shape[-1] == 0


def emission_scores(decl: LGObservation, y: Array) -> Array:
    """``ℓ_j(y)`` for every parent configuration ``j``: ``(*parent_shape,)``.

    A point declaration scores the Gaussian ``N(y; c_j, R_j Rᵀ_j)``; a matrix-normal
    inverse-Wishart declaration scores its Student-t posterior predictive at the regressor
    ``[1]`` (``LinearGaussianObservationMNIW.predictive_log_prob``). A shared offset or noise
    broadcasts over the per-level one.
    """
    y = jnp.asarray(y)
    if isinstance(decl, LinearGaussianObservationMNIW):
        ones = jnp.ones((*decl.M.shape[:-2], 1), decl.M.dtype)
        return decl.predictive_log_prob(y, ones)
    point = lg_point(decl)
    c, chol = point.c, point.R_chol
    batch = jnp.broadcast_shapes(c.shape[:-1], chol.shape[:-2])
    c = jnp.broadcast_to(c, (*batch, c.shape[-1]))
    chol = jnp.broadcast_to(chol, (*batch, *chol.shape[-2:]))
    dev = y - c
    z = solve_triangular(chol, dev[..., None], lower=True)[..., 0]
    half_log_det = jnp.sum(jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)
    return -0.5 * (y.shape[-1] * _LOG_2PI + jnp.sum(z**2, axis=-1)) - half_log_det


def emission_log_likelihood(decl: LGObservation, y: Array, parent_shape: tuple[int, ...]) -> Array:
    """The emission's scores as the one-row log-likelihood tensor the mean-field sweep reads,
    ``(1, *parent_shape)``, its observation the index ``0``."""
    scores = jnp.broadcast_to(emission_scores(decl, y), parent_shape)
    return scores[None]
