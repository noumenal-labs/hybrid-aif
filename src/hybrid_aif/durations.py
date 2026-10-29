"""Helpers for a model whose rounds last more than one tick: whole-step kernels and targets.

A round steps every factor once and may last a different time per level of one control node
(``SystemParams.durations``); the engine composes and integrates nothing, so the kernel of a
step is written for the whole step. Three pure functions write the mechanical cases:

* :func:`compose` the kernel of ``n`` repeats of a per-tick linear-Gaussian kernel;
* :func:`discretise` the exact zero-order-hold discretisation of a continuous-time linear
  system over a duration, one per level if the levels last different times;
* :func:`accumulated_target` a per-tick target as the conditional target of a step, per level,
  its precision (Gaussian) or log-table (categorical) scaled by the level's duration.

Each returns an ordinary declaration (a ``LinearGaussianTransition``, a ``ConditionalGaussian``,
a ``ConditionalCategorical``). Design: docs/design/step-durations.md T3, T4, T5.
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import expm

from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalGaussian,
    Gaussian,
)
from hybrid_aif.factors import LinearGaussianTransition


def _compose_mode(F: Array, b: Array, chol: Array, G: Array | None, n: int):
    """``n`` repeats of ``x' = F x + G u + b + eps``, ``eps ~ N(0, chol cholᵀ)``, for one mode:
    ``(F^n, sum F^i b, sum F^i Q F^iᵀ, sum F^i G)`` over ``i < n``."""
    Q = chol @ chol.T
    eye = jnp.eye(F.shape[-1], dtype=F.dtype)

    def body(carry, _):
        Fp, bn, Qn, Gn = carry
        bn = bn + Fp @ b
        Qn = Qn + Fp @ Q @ Fp.T
        Gn = Gn if G is None else Gn + Fp @ G
        return (F @ Fp, bn, Qn, Gn), None

    init = (eye, jnp.zeros_like(b), jnp.zeros_like(Q), None if G is None else jnp.zeros_like(G))
    (Fn, bn, Qn, Gn), _ = jax.lax.scan(body, init, None, length=n)
    return Fn, bn, Qn, Gn


def compose(kernel: LinearGaussianTransition, n: int) -> LinearGaussianTransition:
    """The kernel of ``n`` repeats of a per-tick linear-Gaussian kernel, per mode.

    With ``x' = F x + G u + b + eps`` and ``eps ~ N(0, Q)`` per tick (a control input held over
    the step), ``n`` ticks give ``F_n = F^n``, ``b_n = sum_{i<n} F^i b``, ``Q_n = sum_{i<n} F^i Q
    F^iᵀ`` and ``G_n = sum_{i<n} F^i G``. A per-mode ``F`` or ``G`` (leading axes
    ``innovations.parent_shape``) is composed mode by mode; a shared one stays shared, and an
    innovation noise shared across the modes (:attr:`ConditionalGaussian.shared_chol`) under a
    shared ``F`` stays shared (under a per-mode ``F`` the composed noise is per mode). ``n`` is a
    static positive int. The returned innovation is a
    :class:`~hybrid_aif.distributions.ConditionalGaussian` carrying the Cholesky factor of
    ``Q_n`` (a kernel with a singular ``Q`` per tick needs ``n`` ticks to be non-singular in the
    directions it reaches, as the Cholesky factor requires).
    Design: docs/design/step-durations.md T3.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"compose: n must be a positive int, got {n!r}")
    inn = kernel.innovations
    shape = inn.parent_shape
    d = inn.dim
    count = 1
    for s in shape:
        count *= s
    shared_F = kernel.F.ndim == 2
    G = kernel.G
    shared_G = G is not None and G.ndim == 2

    def flat(x, shared, trailing):
        if shared:
            return jnp.broadcast_to(x, (count, *x.shape))
        return x.reshape(count, *trailing)

    F = flat(kernel.F, shared_F, (d, d))
    b = inn.loc.reshape(count, d)
    chol = inn.stacked_chol().reshape(count, d, d)
    Gf = None if G is None else flat(G, shared_G, G.shape[-2:])
    if Gf is None:
        Fn, bn, Qn, _ = jax.vmap(lambda f, c, ch: _compose_mode(f, c, ch, None, n))(F, b, chol)
        Gn = None
    else:
        Fn, bn, Qn, Gn = jax.vmap(lambda f, c, ch, g: _compose_mode(f, c, ch, g, n))(F, b, chol, Gf)
    Fn = Fn[0] if shared_F else Fn.reshape(*shape, d, d)
    if Gn is not None:
        Gn = Gn[0] if shared_G else Gn.reshape(*shape, *Gn.shape[-2:])
    # one noise composed under one map is one noise: the shared form is kept
    chol_n = jnp.linalg.cholesky(Qn)
    chol_n = chol_n[0] if inn.shared_chol and shared_F else chol_n.reshape(*shape, d, d)
    innovations = ConditionalGaussian(
        loc=bn.reshape(*shape, d), chol=chol_n, parent_ndims=inn.parent_ndims
    )
    return LinearGaussianTransition(F=Fn, innovations=innovations, G=Gn)


def _discretise_one(A: Array, B: Array | None, Q: Array, dt: Array):
    """Van Loan's exact zero-order-hold discretisation over ``dt``: ``(F, Q_d, G)``."""
    d = A.shape[-1]
    zeros = jnp.zeros_like(A)
    noise = expm(jnp.block([[-A, Q], [zeros, A.T]]) * dt)
    F = noise[d:, d:].T
    Qd = F @ noise[:d, d:]
    Qd = 0.5 * (Qd + Qd.T)
    if B is None:
        return F, Qd, None
    du = B.shape[-1]
    top = jnp.concatenate([A, B], axis=1)
    bottom = jnp.zeros((du, d + du), A.dtype)
    G = expm(jnp.concatenate([top, bottom], axis=0) * dt)[:d, d:]
    return F, Qd, G


def discretise(
    A: Array, B: Array | None, Q: Array, dt: Array | float, modes: int | None = None
) -> LinearGaussianTransition:
    """The exact zero-order-hold discretisation of ``dx = A x + B u dt + dw``, ``dw ~ N(0, Q dt)``.

    Over a duration ``dt``: ``F = expm(A dt)``, ``Q_d = int_0^dt expm(A s) Q expm(A s)ᵀ ds`` and
    ``G = int_0^dt expm(A s) ds B`` (a control held over the step), by Van Loan's matrix
    exponential. ``A`` and ``Q`` are ``(d, d)``, ``B`` is ``(d, du)`` or ``None`` (the kernel then
    has no ``G``). ``dt`` is one positive number, shared (a kernel whose innovations have the
    trivial parent shape ``(1,)``, or ``(modes,)`` copies when ``modes`` is given), or a
    ``(levels,)`` array, one duration per level of a control node: the kernel is then per-mode,
    ``F``, ``G`` and the innovation's parent axis the level, which is how an option's kernel is
    written (``modes``, if given, must equal ``levels``). Design: docs/design/step-durations.md T3.
    """
    A = jnp.asarray(A)
    Q = jnp.asarray(Q, A.dtype)
    Bm = None if B is None else jnp.asarray(B, A.dtype)
    dts = jnp.asarray(dt, A.dtype)
    if dts.ndim > 1:
        raise ValueError(f"discretise: dt is one number or a (levels,) array, got {dts.shape}")
    if dts.ndim == 1 and modes is not None and modes != dts.shape[0]:
        raise ValueError(
            f"discretise: dt has {dts.shape[0]} level(s) but modes={modes}; give one or the other"
        )
    shared = dts.ndim == 0
    if shared:
        dts = jnp.broadcast_to(dts, (1 if modes is None else modes,))
    d = A.shape[-1]
    F, Qd, G = jax.vmap(lambda t: _discretise_one(A, Bm, Q, t))(dts)
    innovations = ConditionalGaussian(
        loc=jnp.zeros((dts.shape[0], d), A.dtype),
        chol=jnp.linalg.cholesky(Qd),
        parent_ndims=1,
    )
    if shared and modes is None:
        F, G = F[0], None if G is None else G[0]
    return LinearGaussianTransition(F=F, innovations=innovations, G=G)


def accumulated_target(
    target: Gaussian | Categorical, durations: Array
) -> ConditionalGaussian | ConditionalCategorical:
    """A per-tick target as the conditional target of a step, one distribution per level.

    ``durations`` is a positive ``(levels,)`` array. A per-tick ``Gaussian`` ``(loc, chol)`` with
    precision ``L`` becomes a :class:`~hybrid_aif.distributions.ConditionalGaussian`
    (``parent_ndims=1``) with the same ``loc`` and precision ``D_k L`` at level ``k``
    (``chol_k = chol / sqrt(D_k)``). A per-tick ``Categorical`` becomes a
    :class:`~hybrid_aif.distributions.ConditionalCategorical` whose table has the child axis first
    and the level axis last, column ``k`` being ``log_softmax(D_k log_softmax(logits))``.

    Each level is a normalised density with its own normaliser, scored once: the cross-entropy
    against level ``k`` is ``D_k`` times the per-tick one up to a constant of that level, and
    that constant is not a cost the model sets. A fixed cost per option is the control node's own
    target, never this helper's. Design: docs/design/step-durations.md T4, T5.
    """
    dur = jnp.asarray(durations)
    if dur.ndim != 1:
        raise ValueError(f"accumulated_target: durations is a (levels,) array, got {dur.shape}")
    if isinstance(target, Gaussian):
        if target.batch_shape != ():
            raise ValueError(
                "accumulated_target: a Gaussian target is one distribution (batch shape ()), "
                f"got batch shape {target.batch_shape}"
            )
        scale = 1.0 / jnp.sqrt(dur)
        return ConditionalGaussian(
            loc=jnp.broadcast_to(target.loc, (dur.shape[0], target.dim)),
            chol=target.chol[None] * scale[:, None, None],
            parent_ndims=1,
        )
    if isinstance(target, Categorical):
        if target.logits.ndim != 1:
            raise ValueError(
                "accumulated_target: a Categorical target is one distribution over its levels, "
                f"got logits of shape {target.logits.shape}"
            )
        log_c = jax.nn.log_softmax(target.logits)
        table = jax.nn.log_softmax(dur[None, :] * log_c[:, None], axis=0)
        return ConditionalCategorical(log_table=table, parent_ndims=1)
    raise TypeError(
        f"accumulated_target takes a Gaussian or a Categorical target, got {type(target).__name__}"
    )
