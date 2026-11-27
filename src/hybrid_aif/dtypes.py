"""Float precision: the dtype a value takes on the way in, and casting a tree of arrays.

The package computes in whatever float dtype it is handed. A float array a caller passes keeps its
dtype (at least float32); float arrays of different dtypes promote to the wider wherever they
meet; a Python number or an integer array takes the dtype of the float arrays beside it, or alone
the default float dtype (float32, or float64 under JAX's 64-bit mode, ``jax_enable_x64``); an
array the package creates takes the dtype of the values it is combined with. :func:`cast` moves a
model, or any part of one, to a chosen float dtype. A neural network computes in the dtype of its
weights, casting its inputs to it (:mod:`hybrid_aif.distributions.neural`).
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax import Array


def as_floats(*xs: object) -> tuple[Array, ...]:
    """``xs`` as float arrays of one dtype: the widest among the float arrays in ``xs`` (at
    least float32), or the default float dtype (float32, or float64 under JAX's 64-bit mode)
    when none of them is a float array. A Python number or an integer array never widens a float
    array beside it."""
    xs = [x if isinstance(x, (bool, int, float)) else jnp.asarray(x) for x in xs]
    dtype = jnp.result_type(*xs)
    if not jnp.issubdtype(dtype, jnp.floating):
        dtype = jnp.result_type(float)
    dtype = jnp.promote_types(dtype, jnp.float32)
    return tuple(jax.lax.convert_element_type(jnp.asarray(x), dtype) for x in xs)


def as_float(x: object) -> Array:
    """``x`` as a float array (:func:`as_floats` of one value)."""
    return as_floats(x)[0]


def tree_float_dtype(tree: object) -> np.dtype:
    """The widest float dtype among the arrays of ``tree`` (at least float32), or the default
    float dtype (float32, or float64 under JAX's 64-bit mode) when it holds no float array: the
    dtype an array the package makes beside ``tree`` takes."""
    dtypes = [
        x.dtype
        for x in jax.tree_util.tree_leaves(tree)
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
    ]
    if not dtypes:
        return np.dtype(jnp.result_type(float))
    return np.result_type(np.float32, *dtypes)


def table_dtype(params) -> np.dtype:
    """The float dtype a belief the package makes for ``params`` takes (a control node's uniform
    prior): the widest float dtype among the transitions and likelihoods of ``params`` (at least
    float32), or the default float dtype when they hold no float array."""
    return tree_float_dtype((params.transition, params.likelihood))


def widened(tree: object, dtype: object) -> object:
    """``tree`` with each float array narrower than ``dtype`` in ``dtype``; every other leaf, and
    a float array as wide or wider, as it is."""

    def leaf(x):
        if _is_float_array(x) and jnp.promote_types(x.dtype, dtype) != x.dtype:
            return jnp.asarray(x, dtype)
        return x

    return jtu.tree_map(leaf, tree)


def float_dtype(dtype: object, what: str = "dtype") -> np.dtype:
    """``dtype`` as a NumPy dtype, checked to be one the package computes in: float32, or
    float64 with JAX's 64-bit mode on. ``what`` names the argument in the refusal."""
    try:
        given = np.dtype(dtype)
    except TypeError as exc:
        raise ValueError(f"{what}={dtype!r} is not a dtype: pass float32 or float64") from exc
    if given not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"{what}={given.name} is not a float dtype: pass float32 or float64")
    if given == np.dtype(np.float64) and not jax.config.jax_enable_x64:
        raise ValueError(
            f"{what}=float64 needs JAX's 64-bit mode, which is off, so the arrays would stay "
            'float32: call jax.config.update("jax_enable_x64", True) at startup, before any '
            "array is made"
        )
    return given


def _is_float_array(x: object) -> bool:
    return isinstance(x, (jax.Array, np.ndarray)) and jnp.issubdtype(x.dtype, jnp.floating)


def cast(tree: object, dtype: object, where: Callable | None = None) -> object:
    """``tree`` with every float array in ``dtype`` (float32 or float64).

    ``tree`` is any tree of arrays: a ``SystemParams``, a mapping of beliefs, one distribution,
    a network, a ``Box``, an episode. Integer and boolean arrays, Python numbers and static
    fields are returned as they are, and so is the tree's structure. float64 needs JAX's 64-bit
    mode and is refused without it.

    ``where`` restricts the cast to part of the tree: a function from the tree to the node, or
    the tuple of nodes, to cast (``where=lambda p: p.recognition["encoder"]``), as
    :func:`equinox.tree_at` takes it. The rest of the tree is returned unchanged. Parts in
    different dtypes make a mixed model, which computes in the wider dtype wherever the parts
    meet and takes longer to set up than a model in one dtype: ``run`` traces its first round
    before the loop to find the dtype of each value it carries, and a hand-written loop compiles
    the world's ``step`` twice.
    """
    dtype = float_dtype(dtype)
    if where is not None:
        return eqx.tree_at(where, tree, replace_fn=lambda node: cast(node, dtype))
    return jtu.tree_map(lambda x: jnp.asarray(x, dtype) if _is_float_array(x) else x, tree)


def _is_float(x: object) -> bool:
    return hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)


def _widen(tree, like):
    """``tree`` with each float leaf in the wider of its dtype and that of its leaf in ``like``
    (a tree of the same structure)."""

    def leaf(x, ref):
        if not (_is_float(x) and _is_float(ref)):
            return x
        return jnp.asarray(x, jnp.promote_types(x.dtype, ref.dtype))

    return jtu.tree_map(leaf, tree, like)


def settled_scan(body, init, xs=None, length: int | None = None):
    """:func:`jax.lax.scan` over a carry that starts in the float dtypes ``body`` returns for it.

    With JAX's 64-bit mode on (``jax_enable_x64``), a body handed 32-bit beliefs that reads a
    64-bit parameter returns some leaves of the carry wider than it was given, which
    ``lax.scan`` refuses. Each float leaf of ``init`` is therefore widened to the dtype one
    step returns for it, until a step returns no leaf wider than it was handed; a leaf the
    step returns narrower is widened on the way out. The values are unchanged. With 64-bit
    mode off every float has one dtype, and this is ``lax.scan``.
    """
    if not jax.config.jax_enable_x64:
        return jax.lax.scan(body, init, xs, length=length)
    steps = length if xs is None else jtu.tree_leaves(xs)[0].shape[0]
    if steps == 0:
        return jax.lax.scan(body, init, xs, length=length)
    x0 = None if xs is None else jtu.tree_map(lambda x: x[0], xs)
    structure = jtu.tree_structure(init)
    for _ in range(structure.num_leaves + 1):
        spec = jax.eval_shape(lambda carry: body(carry, x0)[0], init)
        if jtu.tree_structure(spec) != structure:
            return jax.lax.scan(body, init, xs, length=length)  # scan names the difference
        wider = _widen(init, spec)
        before = [getattr(x, "dtype", None) for x in jtu.tree_leaves(init)]
        if [getattr(x, "dtype", None) for x in jtu.tree_leaves(wider)] == before:
            break
        init = wider

    def settled(carry, x):
        out, y = body(carry, x)
        return _widen(out, init), y

    return jax.lax.scan(settled, init, xs, length=length)
