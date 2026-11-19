"""The survey rover's model: a terrain field it learns by driving, and a vehicle whose battery it
must keep up.

Three factors in declaration order: ``terrain`` (three classes), which reads the parameter
variable ``field`` (one Dirichlet entry per cell of a grid over the position, learned by counting
the terrain seen, under the rate key ``parameter:field``); ``x`` (the vehicle's state: position,
heading as a unit vector, forward speed and the battery's energy level, moved by the world's own
``VehicleStep`` through a ``NonlinearGaussianTransition``, the same on every terrain: the battery
drains with driving and on slopes and charges in the sun, mostly while parked, and near empty the
drive fails, so the model predicts the stall the world applies; the model does not know the
field's wall clip); and the velocity command ``v``. The modalities: a satellite position fix whose
noise follows a map over the field, a gauge reading of the energy level (linear-Gaussian), and the
camera, in one of two forms: the image itself, through a frozen neural decoder over position,
heading and terrain (``camera=True``), or the terrain class a frozen recogniser reads off the
image, through the recogniser's measured confusion table (``look``), which informs the terrain and
not the position. The field's walls are a preference over the position fix (``wall_target``):
flat inside the field and falling off past its edges, so a plan that would leave the field is
scored down. The energy preference (``energy_preference``) is a Gaussian over the gauge's reading,
centred on a full battery.

    system, params, grid, centres = rover(decoder)

With ``tiered`` the model has a fourth node, the option control ``u_opt`` (survey sector 1 to 4, or
charge at one of the ``N_CHARGE`` charging places) that the scheduler supplies by a ``Link`` from
its option action node: the rover plans no decision for it, and its factors read what is supplied as
they read any control node. A target over the vehicle's state is conditional on the option: under a
survey option a broad Gaussian at the sector's centre, under a charge option a narrow one at that
option's charging place with the speed at rest. The charging places (``charge_sites``) are one per
sunny region (``sunny_regions``: a 4-connected patch of at least ``MIN_REGION_CELLS`` grid cells
whose mean solar yield is at least ``SUNNY_YIELD``), each the region's cell of highest yield, in
order of that yield; the first (``charge_site``) is the cell of highest yield on the field.

The objective has no uncertainty term on the position: the fix's information gain is the rover's
localisation drive, and the field's novelty its mapping drive.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import survey_world as wd

from hybrid_aif import (
    Categorical,
    ConditionalCategorical,
    ConditionalDirichlet,
    ConditionalGaussian,
    Gaussian,
    Grid,
    LinearGaussianObservation,
    Parameter,
)
from hybrid_aif.controls import Box
from hybrid_aif.distributions import MixtureGaussian
from hybrid_aif.factors import GridNoiseMap, NonlinearGaussianTransition
from hybrid_aif.system import System, SystemParams

N_TERRAIN = 3
N_BANDS = wd.N_BANDS  # the energy bands the slow tier reads, band 0 the empty band
N_SURVEY = wd.ZONES  # survey options: one per sector
CHARGE = N_SURVEY  # the first charge option's level (the charge levels: ``CHARGE_LEVELS``)
SUNNY_YIELD = 0.5  # a grid cell is sufficiently sunny to charge on at this mean yield or more
MIN_REGION_CELLS = 4  # a sunny region has at least this many cells
STATE_DIM = 6  # position, heading pair, speed, energy level
#  position sd of the survey target: half the width of a sector (1.5 on the field's quadrants)
SECTOR_SD = (wd.ZONE_EDGES[0][1] - wd.ZONE_EDGES[0][0]) / 2
SITE_SD = 0.25  # position sd of the charge target: within a map cell of the charging place
REST_SD = 0.05  # speed sd of the charge target: parked
FREE_SD = 100.0  # the sd of a coordinate a target leaves free (no effective pull)
GRID_SHAPE = 16  # field cells per axis: a cell of 0.375
NOISE_NODES = 25  # fix-noise map nodes per axis, 0.25 apart
POLICY_LEN = 12  # ticks the rover plans ahead: 9.6 map cells at the top speed of 0.3
#  the initial belief's sd: position, heading pair, speed, energy level (a full battery is
#  expected, but loosely: the first gauge reading sets the level)
START_SD = (0.3, 0.3, 0.05, 0.05, 0.02, 1.0)
PROPAGATION = "first_order"  # how a belief is pushed through the vehicle's motion
WALL_SPACING = 0.4  # the wall preference's component grid, field units
WALL_SD = 0.2  # each component's sd: the preference falls off over this distance past a wall
FIELD_STRENGTH = 0.1  # pseudo-counts per cell of the uniform field prior
MIXTURE_K = 1  # one component: the motion does not switch, the camera conditions one Gaussian
# the posterior chain does not reach a mixture-carried factor, so "posterior" would change nothing
PLANNING_BELIEF = "prior"
#  the rover's preference over its energy level: a Gaussian over the gauge's reading, (centre,
#  sd), centred on a full battery
ENERGY_PREFERENCE = (wd.ENERGY_MAX, 1.0)
TERRAIN_NAMES = list(wd.TERRAIN_NAMES)
#  the bands by their levels: the empty band, the bands between by their bounds, the full band
ENERGY_NAMES = (
    ["empty"]
    + [f"{b * wd.BAND_WIDTH:g}-{(b + 1) * wd.BAND_WIDTH:g}" for b in range(1, N_BANDS - 1)]
    + ["full"]
)


def field_grid(shape=None):
    """The grid over the position and the centre of each cell, ``(cells, 2)``, row-major."""
    shape = shape or GRID_SHAPE
    (x0, x1), (y0, y1) = wd.FIELD
    grid = Grid(shape=(shape, shape), low=(x0, y0), high=(x1, y1), axes=(0, 1))
    xs = x0 + (np.arange(shape) + 0.5) * (x1 - x0) / shape
    ys = y0 + (np.arange(shape) + 0.5) * (y1 - y0) / shape
    centres = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    return grid, centres.astype(np.float32)


def fix_noise_map():
    """A GridNoiseMap of the fix's noise over the field: values ``(G, G, 2, 2)``."""
    xs = np.linspace(*wd.FIELD[0], NOISE_NODES)
    ys = np.linspace(*wd.FIELD[1], NOISE_NODES)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    pos = jnp.stack([jnp.asarray(px), jnp.asarray(py)], axis=-1).reshape(-1, 2)
    sd = np.asarray(jax.vmap(wd.sd_fix)(pos)).reshape(NOISE_NODES, NOISE_NODES)
    values = np.zeros((NOISE_NODES, NOISE_NODES, 2, 2), np.float32)
    values[..., 0, 0] = sd
    values[..., 1, 1] = sd
    low = (wd.FIELD[0][0], wd.FIELD[1][0])
    high = (wd.FIELD[0][1], wd.FIELD[1][1])
    return GridNoiseMap(jnp.asarray(values), low, high, axes=(0, 1))


def wall_target():
    """The wall preference over the position fix: an equal-weight mixture of Gaussians of sd
    ``WALL_SD`` on a grid of spacing ``WALL_SPACING`` filling the field (the first centre half a
    spacing inside each wall). Its log density is flat to a few hundredths of a nat more than
    ``2 WALL_SD`` inside the walls and falls off past them like a Gaussian tail: about ``log 2``
    lower at a wall and about 3.8 nats lower ``2 WALL_SD`` beyond one. A mixture preference is
    scored as an interval, from the closed-form mixture cross-entropy bounds."""
    (x0, x1), (y0, y1) = wd.FIELD
    nx = int(round((x1 - x0) / WALL_SPACING))
    ny = int(round((y1 - y0) / WALL_SPACING))
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    centres = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    k = centres.shape[0]
    return MixtureGaussian(
        log_weights=jnp.full(k, -np.log(k), jnp.float32),
        means=jnp.asarray(centres, jnp.float32),
        chols=jnp.broadcast_to(WALL_SD * jnp.eye(2), (k, 2, 2)),
    )


def gauge_table():
    """The band meter's confusion ``(reading, band)``, the world's own table
    (``survey_world.gauge_table``)."""
    return wd.gauge_table()


def with_level_input(decoder):
    """The camera's decoder over the six-coordinate state: a decoder trained on ``(x, y, cos
    theta, sin theta, s)`` and the terrain gains an input for the energy level with zero weight in
    its first layer, so its picture and noise are those of the trained decoder at every level;
    ``None`` and a decoder that already reads six coordinates come back as given."""
    if decoder is None or decoder.parent_dims[0] == STATE_DIM:
        return decoder
    first = decoder.layers[0]
    n = decoder.parent_dims[0]
    weight = jnp.concatenate(
        [
            first.weight[:, :n],
            jnp.zeros((first.weight.shape[0], 1), first.weight.dtype),
            first.weight[:, n:],
        ],
        axis=1,
    )
    wide = eqx.nn.Linear(first.in_features + 1, first.out_features, key=jax.random.PRNGKey(0))
    wide = eqx.tree_at(lambda m: (m.weight, m.bias), wide, (weight, first.bias))
    return dataclasses.replace(
        decoder,
        layers=(wide, *decoder.layers[1:]),
        parent_dims=(n + 1, *decoder.parent_dims[1:]),
    )


def cell_yields(shape=None):
    """The mean solar yield of each grid cell (``SUN`` over the world's tabulated points in the
    cell), ``(cells,)``, row-major as ``field_grid``."""
    shape = shape or GRID_SHAPE
    (x0, x1), (y0, y1) = wd.FIELD
    ix = np.clip(((wd._GX - x0) / (x1 - x0) * shape).astype(int), 0, shape - 1)
    iy = np.clip(((wd._GY - y0) / (y1 - y0) * shape).astype(int), 0, shape - 1)
    cell = (ix * shape + iy).ravel()
    total = np.bincount(cell, wd.SUN.ravel(), shape * shape)
    count = np.bincount(cell, None, shape * shape)
    return total / count


def sunny_regions(shape=None):
    """The sunny regions of the field: the connected components (4-neighbour adjacency on the
    grid) of the cells whose mean solar yield (``cell_yields``) is at least ``SUNNY_YIELD``, those
    of at least ``MIN_REGION_CELLS`` cells, each an array of its cells' indices (row-major as
    ``field_grid``), in order of the yield of the region's sunniest cell, descending."""
    shape = shape or GRID_SHAPE
    yields = cell_yields(shape)
    sunny = (yields >= SUNNY_YIELD).reshape(shape, shape)
    label = np.full((shape, shape), -1)
    regions = []
    for start in zip(*np.nonzero(sunny), strict=True):
        if label[start] >= 0:
            continue
        label[start] = len(regions)
        stack, cells = [start], []
        while stack:
            i, j = stack.pop()
            cells.append(i * shape + j)
            for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if 0 <= a < shape and 0 <= b < shape and sunny[a, b] and label[a, b] < 0:
                    label[a, b] = len(regions)
                    stack.append((a, b))
        regions.append(np.asarray(sorted(cells)))
    regions = [r for r in regions if len(r) >= MIN_REGION_CELLS]
    return sorted(regions, key=lambda r: -float(yields[r].max()))


def charge_site_cells(shape=None):
    """The cell of each charging place, ``(N_CHARGE,)``: each sunny region's cell of highest
    yield (``sunny_regions``), in the regions' order."""
    yields = cell_yields(shape)
    return np.asarray([int(r[np.argmax(yields[r])]) for r in sunny_regions(shape)])


def charge_sites(shape=None):
    """The charging places, ``(N_CHARGE, 2)``: the centres of ``charge_site_cells``."""
    _, centres = field_grid(shape)
    return centres[charge_site_cells(shape)]


def charge_site(shape=None):
    """The first charging place: the centre of the grid cell whose mean solar yield
    (``cell_yields``) is highest, ``(2,)`` (the first of ``charge_sites``)."""
    return charge_sites(shape)[0]


N_CHARGE = len(charge_site_cells())  # charge options: one per charging place (sunny region)
N_OPTIONS = N_SURVEY + N_CHARGE  # the survey options, then the charge options
CHARGE_LEVELS = range(CHARGE, N_OPTIONS)  # the charge options' levels


def is_charge(option):
    """Whether an option level is a charge option (``option >= CHARGE``); works on arrays."""
    return option >= CHARGE


def sector_centres():
    """The centre of each survey sector, ``(N_SURVEY, 2)``, in zone order (``zone_of``)."""
    xe, ye = wd.ZONE_EDGES
    xs = [(a + b) / 2 for a, b in zip(xe[:-1], xe[1:], strict=True)]
    ys = [(a + b) / 2 for a, b in zip(ye[:-1], ye[1:], strict=True)]
    nx = len(xs)
    return np.asarray([(xs[z % nx], ys[z // nx]) for z in range(N_SURVEY)], np.float32)


def option_position_target():
    """The target over the vehicle's state conditional on the option, ``(options, 6)`` locations
    and ``(options, 6, 6)`` factors: under a survey option a Gaussian at the sector's centre
    (position sd ``SECTOR_SD``, heading pair and speed sd 2), the energy level left free (sd
    ``FREE_SD``: the energy preference is over the gauge). Under a charge option, a Gaussian at
    its charging place (``charge_sites``, position sd ``SITE_SD``) with the speed at rest (sd
    ``REST_SD``) and the level free. A soft form, no position target under charge (the rover to
    charge where it stands, its energy preference to find the light), was measured and lost: the
    rover parked where the option found it, mostly in shade, and both rovers of the two-tier pair
    stalled for hundreds of ticks (``design/terrain-field.md``, "The charger, rethought")."""
    n = N_OPTIONS
    loc = np.zeros((n, STATE_DIM), np.float32)
    loc[:N_SURVEY, :2] = sector_centres()
    sites = charge_sites()
    chol = np.zeros((n, STATE_DIM, STATE_DIM), np.float32)
    for k in range(n):
        if k < N_SURVEY:
            sd = [SECTOR_SD, SECTOR_SD, 2.0, 2.0, 2.0, FREE_SD]
        else:
            loc[k, :2] = sites[k - N_SURVEY]
            sd = [SITE_SD, SITE_SD, 2.0, 2.0, REST_SD, FREE_SD]
        chol[k] = np.diag(sd)
    return ConditionalGaussian(loc=jnp.asarray(loc), chol=jnp.asarray(chol), parent_ndims=1)


def energy_target(preference=ENERGY_PREFERENCE):
    """The energy preference over the gauge's reading: a Gaussian ``(centre, sd)``."""
    centre, sd = preference
    return Gaussian(loc=jnp.asarray([centre], jnp.float32), chol=jnp.asarray([[sd]], jnp.float32))


def rover(
    decoder,
    grid_shape=GRID_SHAPE,
    energy_preference=ENERGY_PREFERENCE,
    field_strength=FIELD_STRENGTH,
    epistemic=("transition:terrain", "obs:fix"),
    policy_len=POLICY_LEN,
    camera=True,
    planning_belief=PLANNING_BELIEF,
    tiered=False,
    wall=True,
    look=None,
    propagation=PROPAGATION,
):
    """``(System, SystemParams, Grid, centres)`` for the survey rover. ``decoder`` is the
    camera's likelihood (over five state coordinates or six; ``with_level_input``);
    ``epistemic`` names the information-gain terms the planner scores (the field's novelty and the
    fix by default; ``False`` for none); ``energy_preference`` is the Gaussian ``(centre, sd)``
    over the gauge's reading (``energy_target``), ``None`` for none; ``camera=False`` leaves the
    image out of the model; ``planning_belief`` is the System's setting of that name; ``tiered``
    adds the option control node ``u_opt`` that the scheduler supplies by a ``Link`` from its
    option action node (the rover plans no decision for it, and its factors read what is supplied
    as they read any control node), with the target over the vehicle's state conditional on it;
    ``wall=False`` leaves the wall preference (``wall_target``) out; ``look`` (a confusion table
    ``[reading, terrain]``) adds the modality ``look``, the recogniser's reading of the terrain,
    with that table as its likelihood."""
    grid, centres = field_grid(grid_shape)
    n_cells = grid.n_cells
    # factors: terrain 0, x 1, v 2 (float control), u_opt 3 (the option, tiered only)
    modalities = ["fix"] + (["img"] if camera else []) + (["look"] if look is not None else [])
    modalities += ["gauge"]
    A_deps = [(1,)] + ([(1, 0)] if camera else []) + ([(0,)] if look is not None else [])
    A_deps += [(1,)]
    system = System(
        A_deps=tuple(A_deps),
        B_deps=((1,), (2,), ()) + (((),) if tiered else ()),
        controls=(2, 3) if tiered else (2,),
        action_names=("a_v", "a_opt") if tiered else ("a_v",),
        state_names=("terrain", "x"),
        control_names=("v", "u_opt") if tiered else ("v",),
        modality_names=tuple(modalities),
        cells={"x": grid},
        C_deps={"x": ("u_opt",)} if tiered else (),
        policy_len=policy_len,
        mixture_k=MIXTURE_K,
        mixture_schedule="mode_aligned",
        planning_belief=planning_belief,
        epistemic=tuple(epistemic) if epistemic else False,
        terms=(),  # no uncertainty term: the fix's information gain is the localisation drive
    )
    start = jnp.asarray([*wd.START, 0.0, 1.0, 0.0, wd.ENERGY_MAX], jnp.float32)
    x0 = MixtureGaussian(
        log_weights=jnp.log(jnp.ones(MIXTURE_K) / MIXTURE_K),
        means=jnp.broadcast_to(start, (MIXTURE_K, STATE_DIM)),
        chols=jnp.broadcast_to(jnp.diag(jnp.asarray(START_SD)), (MIXTURE_K, STATE_DIM, STATE_DIM)),
    )
    uniform = jnp.full((N_TERRAIN, n_cells), 1.0 / N_TERRAIN)
    eye = jnp.eye(STATE_DIM)
    likelihood = {
        "fix": LinearGaussianObservation(H=eye[:2], c=jnp.zeros(2), noise=fix_noise_map()),
        "gauge": LinearGaussianObservation(
            H=eye[5:6], c=jnp.zeros(1), noise=jnp.asarray([[wd.GAUGE_SD]], jnp.float32)
        ),
    }
    if camera:
        likelihood["img"] = with_level_input(decoder)
    if look is not None:
        table = np.clip(np.asarray(look, np.float64), 1e-4, None)  # no reading impossible
        table = table / table.sum(axis=0, keepdims=True)
        likelihood["look"] = ConditionalCategorical(
            log_table=jnp.log(jnp.asarray(table, jnp.float32)), parent_ndims=1
        )
    params = SystemParams(
        parameters={"field": ConditionalDirichlet.from_table(uniform, 1, strength=field_strength)},
        likelihood=likelihood,
        transition={
            "terrain": Parameter("field"),
            "x": NonlinearGaussianTransition(
                f=wd.VehicleStep(),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, STATE_DIM), jnp.float32),
                    chol=jnp.diag(jnp.asarray(wd.STATE_SD, jnp.float32))[None],
                    parent_ndims=1,
                ),
                propagation=propagation,
            ),
        },
        initial_state_prior={
            "terrain": Categorical(logits=jnp.zeros(N_TERRAIN)),
            "x": x0,
        },
        control={"v": Box(jnp.asarray(wd.COMMAND_LOW), jnp.asarray(wd.COMMAND_HIGH))},
        state_targets={"x": option_position_target()} if tiered else {},
        obs_targets={
            **({"fix": wall_target()} if wall else {}),
            **({"gauge": energy_target(energy_preference)} if energy_preference else {}),
        },
    )
    return system, params, grid, centres
