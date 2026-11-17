"""Branching the planning carry on what a categorical observation will say.

:func:`make_branch_fn` builds, for the System's ``branch_on`` modalities, the closure
sophisticated planning (:mod:`hybrid_aif.planning.sophisticated`) calls: at a predicted belief,
the outcome distribution and the belief each outcome leaves. Only a categorical modality over
categorical parents can be branched; the belief after an outcome is exact over its parents,
``q(s_P | o) ∝ A[o | s_P] Π_{i∈P} q(s_i)``, marginalised back to the mean-field carry.
Continuous factors pass through. Several modalities branch in sequence, in declaration order,
so the outcome set is their product.
"""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.distributions import Categorical, ConditionalCategorical


def _probs(q) -> Array:
    return q.probs if isinstance(q, Categorical) else jnp.asarray(q)


def _like(template, probs: Array):
    """``probs`` in the carry's own representation for that factor."""
    if isinstance(template, Categorical):
        return Categorical(logits=jnp.log(jnp.clip(probs, 1e-30)))
    return probs


def _branch_one(
    carry: tuple, A: ConditionalCategorical, parents: tuple[int, ...]
) -> tuple[Array, tuple, Array]:
    """One modality: ``(p(o), carry after o, count statistic of o)`` with a leading outcome
    axis. The statistic of outcome ``o`` is ``onehot(o) ⊗ q(s_P | o)``, the joint posterior over
    the parents, ``(n_o, n_o, *parent sizes)``: the exact conjugate update of the table's counts
    by the outcome."""
    table = jnp.exp(A.log_table)  # (n_o, *parent sizes), child-first
    qs = [_probs(carry[f]) for f in parents]
    joint = table
    for i, q in enumerate(qs):
        shape = [1] * table.ndim
        shape[1 + i] = q.shape[-1]
        joint = joint * q.reshape(shape)
    axes = tuple(range(1, table.ndim))
    p_o = jnp.sum(joint, axis=axes)  # (n_o,)
    safe = jnp.where(p_o > 0, p_o, 1.0)
    posts = {}
    for i, f in enumerate(parents):
        others = tuple(a for a in axes if a != 1 + i)
        marginal = jnp.sum(joint, axis=others) / safe[:, None]  # (n_o, n_f)
        # An outcome the belief gives no chance keeps the prior (weight 0 anyway).
        posts[f] = jnp.where((p_o > 0)[:, None], marginal, _probs(carry[f])[None, :])
    n_o = p_o.shape[0]
    posterior = joint / safe.reshape((n_o,) + (1,) * (table.ndim - 1))  # (n_o, *parents)
    stats = (
        jnp.eye(n_o, dtype=posterior.dtype).reshape((n_o, n_o) + (1,) * (table.ndim - 1))
        * posterior[:, None]
    )
    out = tuple(
        jax.vmap(lambda p, t=carry[f]: _like(t, p))(posts[f])
        if f in posts
        else jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n_o, *x.shape)), carry[f])
        for f in range(len(carry))
    )
    return p_o, out, stats


def make_branch_fn(
    likelihood: Sequence,
    A_deps: Sequence[tuple[int, ...]],
    modalities: Sequence[int],
    evidence: bool = False,
) -> Callable[[tuple], tuple]:
    """``branch(carry) -> (p, carry_after)`` over the joint outcomes of ``modalities``.

    ``modalities`` index into ``likelihood``; ``p`` has shape ``(n,)`` over their product in
    row-major order, and the carry after each outcome has a leading ``(n,)`` axis. The System has
    checked each is categorical over categorical parents. With ``evidence`` the closure returns
    ``(p, carry_after, stats)``, ``stats`` one count statistic per modality in ``modalities``
    order, ``(n, n_o, *parent sizes)``: ``onehot(o) ⊗ q(s_P | o)`` of the branch's outcome of
    that modality, the joint posterior over its parents at the point the branch read it (design:
    parameters-within-the-plan.md A3).
    """
    specs = [(likelihood[m], tuple(A_deps[m])) for m in modalities]

    def branch(carry: tuple) -> tuple:
        p = jnp.ones((1,))
        stacked = jax.tree_util.tree_map(lambda x: x[None], carry)
        stats: list[Array] = []
        for A, parents in specs:
            p_o, after, ev = jax.vmap(lambda c, A=A, parents=parents: _branch_one(c, A, parents))(
                stacked
            )  # (n, n_o), (n, n_o, ...), (n, n_o, n_o, ...)
            n_o = p_o.shape[1]
            p = (p[:, None] * p_o).reshape(-1)
            stacked = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), after)
            stats = [
                jnp.broadcast_to(s[:, None], (s.shape[0], n_o, *s.shape[1:])).reshape(
                    -1, *s.shape[1:]
                )
                for s in stats
            ]
            stats.append(ev.reshape(-1, *ev.shape[2:]))
        if evidence:
            return p, stacked, tuple(stats)
        return p, stacked

    return branch
