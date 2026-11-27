"""Realising an emission: the draw, the mode and the statistics of a kernel.

An emission's kernel (``SystemParams.emission``) is a likelihood of any family, or the static
identity or indicator kernel its declaration names (:func:`identity`, :func:`indicator`), and its
parents are values or beliefs. Over **point** parents (a world's state) a kernel conditions on
them exactly. Over **belief** parents (an agent's posterior) the draw is a draw of the parents
followed by the kernel's draw, an exact predictive draw, and the mode and the statistics are
those of the predictive distribution: a table contracted against the parents' probabilities, a
linear-Gaussian kernel's mixture over its discrete parents moment-matched to a mean and a
covariance (``H mu + c`` and ``H S H^T + R`` for one mode). A neural kernel is realised by a draw
only. :meth:`hybrid_aif.system.System.emit` is the face; the functions here are its parts.
"""

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.cells import Grid, cell_index, cell_mass
from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    Gaussian,
    MixtureGaussian,
    MLPConditional,
)
from hybrid_aif.inference.sample import collapse

#: The tag a key is folded with before the parents of an emission are drawn from a belief, so the
#: kernel's draw (which uses the key itself) is that of the same emission over point parents.
PARENT_TAG = 0x656D6974

BELIEFS = (Categorical, Gaussian, MixtureGaussian)


def is_belief(value: object) -> bool:
    """Whether ``value`` is a belief (a distribution) rather than a point value."""
    return isinstance(value, BELIEFS)


def point_of(key: Array, value: object) -> Array:
    """``value`` as a point: a belief is drawn from, a point value is returned as it is."""
    if isinstance(value, Categorical):
        return value.sample(key).astype(jnp.int32)
    if isinstance(value, Gaussian | MixtureGaussian):
        return value.sample(key)
    return jnp.asarray(value)


def _weights(value: object, levels: int) -> Array:
    """A discrete parent's probabilities over ``levels``: a belief's, or a level's one-hot."""
    if isinstance(value, Categorical):
        return value.probs
    return jax.nn.one_hot(jnp.asarray(value, jnp.int32), levels)


def _moments(value: object) -> tuple[Array, Array]:
    """A continuous parent's mean and covariance: a belief's, or a point's with none."""
    if isinstance(value, Gaussian):
        return value.mean(), value.cov
    if isinstance(value, MixtureGaussian):
        return value.mean(), value.cov()
    point = jnp.asarray(value)
    return point, jnp.zeros((point.shape[-1], point.shape[-1]), point.dtype)


def tabular(decl: ConditionalCategorical, parents: tuple, realise: str, name: str) -> Array:
    """The probabilities of a table's predictive over ``parents`` (``realise='stats'``), or their
    argmax (``'mode'``)."""
    probs = jnp.exp(decl.log_probs)
    for value in reversed(parents):
        if jnp.ndim(probs) < 2:
            break
        probs = probs @ _weights(value, probs.shape[-1])
    return probs if realise == "stats" else jnp.argmax(probs).astype(jnp.int32)


def linear_gaussian(decl, parents: tuple, realise: str, name: str):
    """The mean (``'mode'``) or the ``(mean, covariance)`` (``'stats'``) of a linear-Gaussian
    kernel's predictive: its mixture over the discrete parents, moment-matched."""
    if decl.noise_map:
        raise NotImplementedError(
            f"emission {name!r}: its noise is a map over the observed state, which has no "
            f"{realise!r} realisation; realise it by 'sample' or declare a plain noise"
        )
    H, c, R = decl.H, decl.c, decl.R_chol
    zero_width = H.shape[-1] == 0
    modes = parents if zero_width else parents[1:]
    d_obs = c.shape[-1]
    lead = jnp.broadcast_shapes(H.shape[:-2], c.shape[:-1], R.shape[:-2])
    if zero_width:
        mu, S = jnp.zeros((0,), c.dtype), jnp.zeros((0, 0), c.dtype)
    else:
        mu, S = _moments(parents[0])
    if lead:
        if len(modes) != len(lead):
            raise ValueError(
                f"emission {name!r}: its kernel is per-mode over {len(lead)} discrete parent(s) "
                f"and the emission names {len(modes)} after the observed state"
            )
        w = jnp.ones((), c.dtype)  # the mixture weights start in the kernel's dtype
        for value, levels in zip(modes, lead, strict=True):
            w = w[..., None] * _weights(value, levels)
        Hs = jnp.broadcast_to(H, (*lead, d_obs, H.shape[-1])).reshape(-1, d_obs, H.shape[-1])
        cs = jnp.broadcast_to(c, (*lead, d_obs)).reshape(-1, d_obs)
        Rs = jnp.broadcast_to(R, (*lead, d_obs, d_obs)).reshape(-1, d_obs, d_obs)
        w = w.reshape(-1)
    else:
        Hs, cs, Rs, w = H[None], c[None], R[None], jnp.ones((1,), c.dtype)
    m = jnp.einsum("jod,d->jo", Hs, mu) + cs
    mean = jnp.einsum("j,jo->o", w, m)
    if realise == "mode":
        return mean
    within = jnp.einsum("jod,de,jpe->jop", Hs, S, Hs) + jnp.einsum("jok,jpk->jop", Rs, Rs)
    spread = m - mean
    cov = jnp.einsum("j,jop->op", w, within + jnp.einsum("jo,jp->jop", spread, spread))
    return mean, cov


def _draw(probs: Array, key: Array | None, name: str) -> Array:
    """A level drawn from the probability vector ``probs``."""
    if key is None:
        raise ValueError(
            f"emission {name!r} is realised by 'sample', which draws and needs a key; pass one, "
            "or declare realise='mode' or 'stats'"
        )
    return jax.random.categorical(key, jnp.log(jnp.maximum(probs, 1e-30))).astype(jnp.int32)


def identity(
    value: object, realise: str, key: Array | None, name: str, levels: int | None = None
) -> Array:
    """The identity emission of one state factor, realised from a belief or a point.

    A categorical factor gives a level (``sample``: a draw from the belief, ``mode``: its
    argmax; a point is its own level) or, under ``stats``, its probability vector (``levels``
    gives the width of a point's one-hot). A continuous factor gives its mean under ``mode`` (a
    point is its own value) and a draw's value under ``sample``; its ``stats`` are not defined.
    """
    if isinstance(value, Categorical):
        if realise == "sample":
            return _draw(value.probs, key, name)
        if realise == "mode":
            return jnp.argmax(value.probs).astype(jnp.int32)
        return value.probs
    if isinstance(value, Gaussian | MixtureGaussian):
        if realise == "mode":
            return value.mean()
        if realise == "sample":
            if key is None:
                _draw(jnp.ones(1), None, name)  # raises: a draw needs a key
            return collapse(value, key).mean()
    elif realise != "stats":  # a point
        return jnp.asarray(value)
    elif jnp.issubdtype(jnp.asarray(value).dtype, jnp.integer) and levels is not None:
        return jax.nn.one_hot(jnp.asarray(value, jnp.int32), levels)
    raise ValueError(
        f"emission {name!r}: realise {realise!r} of a continuous factor is not defined (the "
        "sufficient statistics of a Gaussian belief are deferred); declare 'sample' or 'mode', "
        "or read the factor through a Grid kernel"
    )


def indicator(value: object, grid: Grid, realise: str, key: Array | None, name: str) -> Array:
    """The indicator emission of one continuous state factor over ``grid``, from a belief or a
    point: a point's cell index (``stats``: its one-hot), a Gaussian or mixture belief's cell
    index by ``sample`` (a draw from the cell mass) or ``mode`` (the largest cell), and its cell
    mass under ``stats``."""
    if isinstance(value, Gaussian | MixtureGaussian):
        probs = cell_mass(value, grid)
        if realise == "sample":
            return _draw(probs, key, name)
        if realise == "mode":
            return jnp.argmax(probs).astype(jnp.int32)
        return probs
    cell = cell_index(jnp.asarray(value), grid)
    return jax.nn.one_hot(cell, grid.n_cells) if realise == "stats" else cell


def predictive(decl, parents: tuple, realise: str, name: str):
    """The mode or the statistics of an emission's kernel over ``parents`` (values or beliefs)."""
    if isinstance(decl, MLPConditional):
        raise NotImplementedError(
            f"emission {name!r}: a neural kernel has no closed-form {realise!r} realisation; "
            "realise it by 'sample'"
        )
    if isinstance(decl, ConditionalCategorical):
        return tabular(decl, parents, realise, name)
    return linear_gaussian(decl, parents, realise, name)
