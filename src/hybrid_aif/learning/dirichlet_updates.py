"""Conjugate Dirichlet count updates for discrete observation likelihoods.

The public API is :meth:`~hybrid_aif.distributions.conditional.ConditionalDirichlet.update`
(one modality at a time; the array kernel lives beside it in the distributions
layer). This module keeps the multi-modality convenience wrapper over raw
concentration arrays, one per modality, shaped ``(num_obs_m, *[num_levels_f
for f in A_deps[m]])`` — the same child-first layout as
:mod:`hybrid_aif.inference.discrete`'s likelihood tensors. Collections are
canonically tuples; any sequence is accepted at runtime. The multi-modality wrapper is after
pymdp's ``learning.update_obs_likelihood_dirichlet`` (pymdp/learning.py).
"""

import jax.numpy as jnp
from jax import Array

from hybrid_aif.distributions import Categorical, ConditionalDirichlet


def update_likelihood_dirichlet(
    prior: tuple[Array, ...],
    observations: tuple[Array, ...],
    posteriors: tuple[Categorical, ...],
    A_deps: tuple[tuple[int, ...], ...],
    lr: float = 1.0,
) -> tuple[Array, ...]:
    """One conjugate step per modality via :meth:`ConditionalDirichlet.update`."""
    updated = []
    for conc_m, obs_m, deps_m in zip(prior, observations, A_deps, strict=True):
        cd = ConditionalDirichlet(concentration=jnp.asarray(conc_m), parent_ndims=len(deps_m))
        beliefs = tuple(posteriors[f] for f in deps_m)
        updated.append(cd.update(jnp.asarray(obs_m), beliefs, lr=lr).concentration)
    return tuple(updated)
