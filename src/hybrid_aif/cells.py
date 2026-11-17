"""A categorical field over a continuous state: the grid that discretises a continuous factor
into cells, and the read from a position belief to a belief over the cell it falls in.

``System(cells={"x": Grid(...)})`` declares the grid; a tabular factor may then name ``x`` in
its ``B_deps``, its table's axis for that parent having one entry per cell, row-major over the
grid's shape. Design: docs/design/terrain-field.md.
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import ndtr

from hybrid_aif.distributions import Categorical, Gaussian, MixtureGaussian


def _tuple(value, n: int, what: str, cast) -> tuple:
    """``value`` (a number, a sequence or an array, numpy or JAX) as ``n`` Python numbers: one
    number stands for every axis."""
    items = np.atleast_1d(np.asarray(value)).tolist()
    if len(items) == 1:
        items = items * n
    if len(items) != n:
        raise ValueError(
            f"Grid: {what} has {len(items)} entries for {n} axis/axes; give one per entry of "
            "shape, or a single number for every axis"
        )
    return tuple(cast(v) for v in items)


@dataclasses.dataclass(frozen=True)
class Grid:
    """A rectangular grid over some coordinates of a continuous state factor's state.

    The cells are flattened in C order over ``shape``, so a table's parent axis for the state
    factor has :attr:`n_cells` entries. The first and last cell on each axis absorb the tail
    beyond the extent.
    """

    #: The number of cells along each spanned axis.
    shape: tuple[int, ...]
    #: The lower end of the extent along each axis (one number each, or one number for every axis).
    low: tuple[float, ...]
    #: The upper end of the extent along each axis (one number each, or one number for every axis).
    high: tuple[float, ...]
    #: The state coordinates spanned, in order (``None``: ``0, 1, ...``).
    axes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        shape = tuple(int(n) for n in np.atleast_1d(np.asarray(self.shape)).tolist())
        k = len(shape)
        axes = (
            tuple(range(k))
            if self.axes is None
            else tuple(int(a) for a in np.atleast_1d(np.asarray(self.axes)).tolist())
        )
        low = _tuple(self.low, k, "low", float)
        high = _tuple(self.high, k, "high", float)
        if len(axes) != k:
            raise ValueError(
                f"Grid: axes={axes} names {len(axes)} coordinate(s) for shape={shape}; give one "
                "axis per entry of shape"
            )
        if k == 0 or any(n < 1 for n in shape):
            raise ValueError(f"Grid: shape={shape} needs at least one cell along every axis")
        if len(set(axes)) != k or any(a < 0 for a in axes):
            raise ValueError(f"Grid: axes={axes} must be distinct non-negative coordinates")
        if any(not h > lo for lo, h in zip(low, high, strict=True)):
            raise ValueError(f"Grid: every axis needs low < high, got low={low}, high={high}")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @property
    def n_cells(self) -> int:
        """The number of cells: the product of ``shape``."""
        return math.prod(self.shape)


def _axis_masses(mean: Array, std: Array, n: int, low: float, high: float) -> Array:
    """The mass of ``Normal(mean, std)`` in each of ``n`` equal cells over ``[low, high]``, the
    first and last absorbing the tails: ``(..., n)``, summing to one."""
    edges = jnp.linspace(low, high, n + 1)[1:-1]
    cdf = ndtr((edges - mean[..., None]) / std[..., None])
    zeros = jnp.zeros((*cdf.shape[:-1], 1), cdf.dtype)  # an axis of one cell has no edges
    cdf = jnp.concatenate([zeros, cdf, zeros + 1.0], axis=-1)
    return cdf[..., 1:] - cdf[..., :-1]


def _gaussian_mass(loc: Array, chol: Array, grid: Grid) -> Array:
    """:func:`cell_mass` of one Gaussian: ``(..., cells)``."""
    var = jnp.sum(chol**2, axis=-1)  # the marginal variances: the squared row norms of the factor
    mass = jnp.ones((*loc.shape[:-1], 1), loc.dtype)
    for a, n, low, high in zip(grid.axes, grid.shape, grid.low, grid.high, strict=True):
        floor = 1e-6 * (high - low)
        std = jnp.sqrt(jnp.maximum(var[..., a], floor**2))
        axis = _axis_masses(loc[..., a], std, n, low, high)
        mass = (mass[..., :, None] * axis[..., None, :]).reshape(*mass.shape[:-1], -1)
    return mass


def cell_mass(belief: Gaussian | MixtureGaussian, grid: Grid) -> Array:
    """The probability vector over the cells of ``grid`` that ``belief`` puts its mass in.

    A Gaussian's mass in a rectangular cell is the product of its marginal masses along the
    grid's axes, each a difference of the normal distribution function at the cell edges (the
    correlation between axes is dropped); the first and last cell on each axis absorb the tails.
    A mixture's is the weight-sum of its components' masses. Returns ``(*batch, cells)``, cells
    row-major over ``grid.shape``, summing to one, and smooth in the means.
    """
    if isinstance(belief, MixtureGaussian):
        per = _gaussian_mass(belief.means, belief.chols, grid)  # (..., K, cells)
        w = jax.nn.softmax(belief.log_weights, axis=-1)
        return jnp.sum(w[..., None] * per, axis=-2)
    if isinstance(belief, Gaussian):
        return _gaussian_mass(belief.loc, belief.chol, grid)
    raise TypeError(
        f"cell_mass reads a Gaussian or MixtureGaussian belief, got {type(belief).__name__}"
    )


def component_cell_mass(belief: MixtureGaussian, grid: Grid) -> Array:
    """Each component's :func:`cell_mass` of a mixture: ``(*batch, K, cells)``, the weights left
    out (the per-component read of a closed pair, design: parameters-within-the-plan.md D2)."""
    return _gaussian_mass(belief.means, belief.chols, grid)


def cell_index(point: Array, grid: Grid) -> Array:
    """The cell of ``grid`` that ``point`` (a state of shape ``(..., dim)``) falls in, as an
    ``int32`` index into the cells in :class:`Grid`'s order (``(...)``): the first and last cell
    on each axis absorb the tail beyond the extent, as :func:`cell_mass` does."""
    index = jnp.zeros(point.shape[:-1], jnp.int32)
    for a, n, low, high in zip(grid.axes, grid.shape, grid.low, grid.high, strict=True):
        edges = jnp.linspace(low, high, n + 1)[1:-1]
        along = jnp.sum(point[..., a, None] > edges, axis=-1).astype(jnp.int32)
        index = index * n + along
    return index


def with_cell_parents(beliefs: tuple, grids: dict) -> tuple:
    """``beliefs`` with each factor in ``grids`` (factor index to :class:`Grid`) replaced by a
    :class:`~hybrid_aif.distributions.Categorical` over its cells, so a tabular reader sees a
    cell parent as it sees any categorical one."""
    if not grids:
        return beliefs
    return tuple(
        Categorical.from_probs(jnp.maximum(cell_mass(q, grids[i]), 1e-30)) if i in grids else q
        for i, q in enumerate(beliefs)
    )
