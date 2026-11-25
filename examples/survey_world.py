"""The true world of the survey rover showcase, a black box coupled to the rover's mind.

A single rover drives over a walled field of smooth hills (the position is clipped to the field
and a rover that hits a wall stops). It holds the rover's state (position, heading, forward speed
and the battery's energy level), the terrain class under it (firm, rough or ice, a function of
position), and the readings the rover receives: a satellite position fix whose noise grows where
the terrain closes the sky, a camera image, a gauge reading of the energy level and the energy
band a coarse meter shows (the slow tier reads the band; the option of a slow tier never reaches
the world). The field's height map, sky openness, sunlight, solar yield, slope and fix noise are
tabulated at import. The field is 6 by 6, with four hills and three terrain patches.

The rover is a wheeled vehicle with a heading of its own and a solar battery (``VehicleStep``):
its state is the position, the heading as a unit vector, the forward speed and the energy level;
the command is a target speed, as a share of the top speed ``TOP_SPEED`` (the rover drives forward
only, and turns in place), and a turn rate (at most ``TURN_MAX`` a tick). The speed closes the
share ``1 - KEEP`` of its gap to the target each tick, so a held command reaches its speed within a
few ticks, and the top speed stays under one map cell per tick. Driving drains the battery (more
at speed and uphill); sunlight charges it, mainly while the rover is parked, and far more on
sun-facing ground than in shade. As the level nears empty the drive fails: the speed and turn the
command can reach fall smoothly to zero (the stall), so an empty rover stands where it is and
charges only as fast as the light there allows. The camera is mounted forward and looks along the
heading; it renders the scene from the field's geometry: the ground along each pixel's ray,
ray-marched over the height map, lit by the sun, textured in world coordinates, under a sky; see
``render``.

With ``vehicles=2`` (``world_system``, ``world_stepper``) the field holds two rovers, the first at
``START`` and the second at ``SECOND_START``, each with its own state, command and readings under
its index (the in-port ``v_1``, the out-ports ``fix_1``, ``img_1``, ``gauge_1``, ``band_1``, and the
state ``x_1`` and ``terrain_1``; ``v_2`` and so on for the second). The two share the field and
nothing else: the world has no rover-to-rover physics, so they never collide, never shade or
occlude one another, and neither's camera ever shows the other. Each vehicle moves and is read
exactly as a world of one vehicle would move and read it under the key ``fold_in(key, k)`` (``k``
its index), so a world of two is two worlds of one stepped side by side.

    python examples/survey_world.py
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates

from hybrid_aif.composition.black_box import black_box
from hybrid_aif.controls import FLOAT, LEVEL
from hybrid_aif.description import register_transition

FIELD = ((-1.0, 5.0), (-2.0, 4.0))  # the field, 6 by 6
START = (-0.2, -0.6)
#  the second rover's start in a world of two: START mirrored through the field's centre (2, 1), so
#  the two begin in opposite quadrants (sector 1 and sector 4), as far apart as the field allows
#  along its diagonal
SECOND_START = (4.2, 2.6)
STARTS = (START, SECOND_START)
DT = 1.0
GRID_N = 121  # the tabulated maps' points per axis, 0.05 apart

TERRAIN_NAMES = ("firm", "rough", "ice")
TOP_SPEED = 0.3  # the forward speed under a full drive command, under one 0.375 map cell a tick
KEEP = 0.5  # the share of the forward speed kept per tick; the rest moves to the target speed
GAIN = (1.0 - KEEP) * TOP_SPEED  # forward speed added per tick per unit of the drive command
COMMAND_LOW = (0.0, -1.0)  # the command (drive, turn): drive from stop to full, turn either way
COMMAND_HIGH = (1.0, 1.0)
TURN_MAX = np.deg2rad(20.0)  # the heading turns at most this far per tick (in place too)
#  process sd per state coordinate: position, heading pair, speed, energy level
STATE_SD = np.asarray([0.005, 0.005, 0.01, 0.01, 0.004, 0.004])
STATE_NAMES = ("x", "y", "cos heading", "sin heading", "forward speed", "energy level")
#  the field's hills ``(centre x, centre y, sd, height)`` (a negative height is a basin) and its
#  terrain patches ``((centre), radius)``
HILLS = (
    (2.0, 1.0, 1.1, 1.0),
    (4.2, 3.2, 0.9, 0.7),
    (-0.3, -1.2, 0.8, 0.5),
    (1.0, 3.0, 0.9, -0.5),
)
ICE_PATCHES = (((1.4, 2.1), 0.9),)  # in the low basin
ROUGH_PATCHES = (((3.6, 0.2), 0.8), ((-0.2, 2.6), 0.6))
ICE_PATCH = ICE_PATCHES[0]

SUN_AZIMUTH = np.deg2rad(200.0)
SUN_ELEVATION = np.deg2rad(35.0)
N_AZIMUTH = 16
HORIZON_RANGE = 4.0
HORIZON_STEP = 0.1
AMBIENT = 0.1

ENERGY_MAX = 4.0  # a full battery
N_BANDS = 8  # the energy bands a coarse meter shows, each BAND_WIDTH wide; band 0 is the empty band
BAND_WIDTH = ENERGY_MAX / N_BANDS
#  the stall: the share of the command the drive can deliver, a logistic in the energy level,
#  one half at STALL_LEVEL, near one above STALL_LEVEL + 4 STALL_WIDTH and near zero below
#  STALL_LEVEL - 4 STALL_WIDTH
STALL_LEVEL = 0.4
STALL_WIDTH = 0.08
#  the energy per tick: a drain at rest, a drain per unit of speed that grows with the climb (the
#  slope along the heading, through a smooth ramp: about zero going down or across, about linear
#  going up; no regeneration downhill), and a solar charge in proportion to the yield of the
#  ground the rover stands on, nearly all of it while parked (at speed only the share
#  MOVING_CHARGE)
BASE_DRAIN = 0.01  # energy per tick at rest
MOVE_DRAIN = 0.045  # energy per tick per unit of speed on flat ground
SLOPE_DRAIN = 1.5 * MOVE_DRAIN  # extra per tick per unit of climb and of speed
#  the climb's ramp, softplus(k c) / k for a slope c along the heading: within log(2) / k (0.035)
#  of zero on level ground, going down or across, and within it of c going up
CLIMB_SHARPNESS = 20.0
CHARGE_RATE = 0.07  # energy per tick parked on ground of full solar yield
MOVING_CHARGE = 0.1  # the share of the parked charge collected at speed
PARK_SPEED = 0.05  # the charge falls from parked to moving over about this speed
#  the solar yield of the ground: the sunlight there over the field's brightest, to this power,
#  so that sun-facing slopes yield several times what level ground does and shaded ground almost
#  nothing
SUN_POWER = 4.0
CLAMP_SHARPNESS = 20.0  # the level is held in [0, ENERGY_MAX] by softplus walls this sharp
#  the rover's typical forward speed, the median over the closed-loop runs of both settings (see
#  the build record); the scheduler's tables are built at it
CRUISE_SPEED = 0.17
GAUGE_SD = 0.05  # the gauge's noise on the energy level
GAUGE_ACC = 0.88  # the band meter is right with at least this probability, else one band off
#  the survey sectors: the field's quadrants, split at x 2 and y 1, the edges per axis
ZONE_SPLIT = (2.0, 1.0)
ZONE_EDGES = (
    (FIELD[0][0], ZONE_SPLIT[0], FIELD[0][1]),
    (FIELD[1][0], ZONE_SPLIT[1], FIELD[1][1]),
)
ZONES = (len(ZONE_EDGES[0]) - 1) * (len(ZONE_EDGES[1]) - 1)

IMG_SIDE = 32
IMG_PIXELS = IMG_SIDE * IMG_SIDE
PIXEL_SD = 0.04
FOCAL = (IMG_SIDE / 2) / np.tan(np.deg2rad(35.0))  # pixels: a 70 degree field of view
CAMERA_HEIGHT = 0.12  # the lens above the ground under the rover
CAMERA_TILT = np.deg2rad(8.0)  # the lens points this far below the rover's forward axis
NEAR, REACH, MARCH = 0.15, 5.0, 96  # the ray march: nearest and farthest distance, samples
SUBCOLUMNS = 3  # rays per pixel across its width (the pixel's footprint, a midpoint rule)
WALL_EDGE = 0.05  # the ground's albedo ramps to WALL_ALBEDO over this width across a wall
HAZE = 8.0  # distance over which the ground fades toward HAZE_VALUE
HAZE_VALUE = 0.5  # the tone far ground fades toward: a grey well below the sky
WALL_ALBEDO = 0.15  # the ground beyond the field's walls is dark
EXPOSURE = 1.4  # pixel value per unit of albedo times sunlight
#  mean ground albedo per terrain class: firm mid-dark, rough darker, ice bright (it may approach
#  the sky's tone)
ALBEDO = np.asarray([0.40, 0.26, 0.72])
#  texture strength per class: the albedo is scaled by 1 + TEXTURE_AMP * tanh(wave), the wave of
#  unit sd, so the texture stays positive and bounded (at most 1 + TEXTURE_AMP) and its contrast is
#  relative: rough the strongest, ice the weakest
TEXTURE_AMP = np.asarray([0.5, 0.7, 0.12])
TEXTURE_WAVES = 6  # plane waves summed per class
TEXTURE_WAVELENGTH = (0.9, 1.8)  # field units: about two and a half to five map cells
#  the sky: brightest at the top, somewhat less at the horizon, and brighter toward the sun's
#  azimuth; the same whether the rover stands in sun or shade
SKY_ZENITH, SKY_HORIZON, SUN_GLOW = 0.9, 0.76, 0.08


def _hill_sum(bump):
    """The sum over ``HILLS`` of ``bump(cx, cy, s, height)``, a basin's subtracted (``bump`` of
    its depth), in the order the hills are listed."""
    total = None
    for cx, cy, s, a in HILLS:
        if total is None:
            total = bump(cx, cy, s, a)
        elif a >= 0:
            total = total + bump(cx, cy, s, a)
        else:
            total = total - bump(cx, cy, s, -a)
    return total


def _height_fn(x, y):
    return _hill_sum(lambda cx, cy, s, a: a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s**2)))


XS = np.linspace(FIELD[0][0], FIELD[0][1], GRID_N)
YS = np.linspace(FIELD[1][0], FIELD[1][1], GRID_N)
_GX, _GY = np.meshgrid(XS, YS, indexing="ij")  # arrays are indexed [ix, iy]
_H0 = _height_fn(_GX, _GY)
HEIGHT = _H0 - _H0.min()
_DX = XS[1] - XS[0]
_DY = YS[1] - YS[0]


def _bilinear_np(table, x, y):
    """Bilinear lookup of a grid table at points (numpy), clamped to the field."""
    fx = np.clip((x - XS[0]) / _DX, 0, GRID_N - 1)
    fy = np.clip((y - YS[0]) / _DY, 0, GRID_N - 1)
    ix = np.minimum(fx.astype(int), GRID_N - 2)
    iy = np.minimum(fy.astype(int), GRID_N - 2)
    tx, ty = fx - ix, fy - iy
    return (
        table[ix, iy] * (1 - tx) * (1 - ty)
        + table[ix + 1, iy] * tx * (1 - ty)
        + table[ix, iy + 1] * (1 - tx) * ty
        + table[ix + 1, iy + 1] * tx * ty
    )


def _openness():
    """Mean over azimuths of cos(horizon elevation)^2, looking out to HORIZON_RANGE."""
    dists = np.arange(HORIZON_STEP, HORIZON_RANGE + 1e-9, HORIZON_STEP)
    total = np.zeros_like(HEIGHT)
    for az in np.linspace(0, 2 * np.pi, N_AZIMUTH, endpoint=False):
        best = np.zeros_like(HEIGHT)
        for d in dists:
            rx = _GX + d * np.cos(az)
            ry = _GY + d * np.sin(az)
            inside = (
                (rx >= FIELD[0][0])
                & (rx <= FIELD[0][1])
                & (ry >= FIELD[1][0])
                & (ry <= FIELD[1][1])
            )
            elev = np.arctan2(_bilinear_np(HEIGHT, rx, ry) - HEIGHT, d)
            best = np.where(inside, np.maximum(best, elev), best)
        total += np.cos(best) ** 2
    return total / N_AZIMUTH


OPEN = _openness()
_HX, _HY = np.gradient(HEIGHT, _DX, _DY)
SLOPE = np.hypot(_HX, _HY)
_NORM = np.sqrt(_HX**2 + _HY**2 + 1.0)
_SUN = np.asarray(
    [
        np.cos(SUN_ELEVATION) * np.cos(SUN_AZIMUTH),
        np.cos(SUN_ELEVATION) * np.sin(SUN_AZIMUTH),
        np.sin(SUN_ELEVATION),
    ]
)
_LAMBERT = np.maximum(0.0, (-_HX * _SUN[0] - _HY * _SUN[1] + _SUN[2]) / _NORM)
LIGHT = AMBIENT + (1 - AMBIENT) * _LAMBERT * OPEN**1.5
SD_FIX = np.clip(0.04 + 1.5 * (1 - OPEN), 0.04, 0.6)
SUN = (LIGHT / LIGHT.max()) ** SUN_POWER


def _terrain_grid():
    t = np.zeros(HEIGHT.shape, np.int32)
    for (cx, cy), r in ROUGH_PATCHES:
        t[np.hypot(_GX - cx, _GY - cy) <= r] = 1
    for (cx, cy), r in ICE_PATCHES:
        t[np.hypot(_GX - cx, _GY - cy) <= r] = 2
    return t


TERRAIN_GRID = _terrain_grid()


def _lookup(table):
    """A jittable bilinear lookup of ``table`` at a position ``xy`` (clamped to the field)."""
    t = jnp.asarray(table, jnp.float32)

    def f(xy):
        fx = jnp.clip((xy[0] - XS[0]) / _DX, 0, GRID_N - 1)
        fy = jnp.clip((xy[1] - YS[0]) / _DY, 0, GRID_N - 1)
        return map_coordinates(t, [fx, fy], order=1, mode="nearest")

    return f


height = _lookup(HEIGHT)
openness = _lookup(OPEN)
light = _lookup(LIGHT)
slope = _lookup(SLOPE)
slope_x = _lookup(_HX)  # the height's rate of change along +x
slope_y = _lookup(_HY)  # and along +y
sun = _lookup(SUN)
sd_fix = _lookup(SD_FIX)


_H_MIN = float(_H0.min())


def smooth_height(points):
    """The ground's height at world points ``(..., 2)``: the height map's own analytic form
    (``HEIGHT`` is it tabulated), smooth in the point, held beyond the walls at the value the
    form takes there."""
    x, y = points[..., 0], points[..., 1]

    def bump(cx, cy, s, a):
        return a * jnp.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s**2))

    return _hill_sum(bump) - _H_MIN


def maps():
    """The tabulated field maps, each indexed ``[ix, iy]``:
    ``(xs, ys, height, openness, light, slope, sd_fix, terrain, sun)``."""
    return XS, YS, HEIGHT, OPEN, LIGHT, SLOPE, SD_FIX, TERRAIN_GRID, SUN


def terrain_of(xy):
    """The terrain class (0 firm, 1 rough, 2 ice) at a position."""
    t = jnp.asarray(0, jnp.int32)
    for centre, radius in ROUGH_PATCHES:
        t = jnp.where(jnp.linalg.norm(xy - jnp.asarray(centre)) <= radius, 1, t)
    for centre, radius in ICE_PATCHES:
        t = jnp.where(jnp.linalg.norm(xy - jnp.asarray(centre)) <= radius, 2, t)
    return t.astype(jnp.int32)


def drive_share(level):
    """The share of the command the drive can deliver at an energy level: a logistic, one half at
    ``STALL_LEVEL``, near one a few ``STALL_WIDTH`` above it and near zero below (the stall)."""
    return jax.nn.sigmoid((level - STALL_LEVEL) / STALL_WIDTH)


def _soft_clamp(level):
    """The level held in ``[0, ENERGY_MAX]`` by softplus walls of sharpness ``CLAMP_SHARPNESS``:
    smooth, the identity well inside the range."""
    k = CLAMP_SHARPNESS
    low = jax.nn.softplus(k * level) / k
    return ENERGY_MAX - jax.nn.softplus(k * (ENERGY_MAX - low)) / k


def climb(xy, heading):
    """The climb at a position along a heading (a pair ``(cos theta, sin theta)``, renormalised):
    the slope along the heading (the height's directional derivative, from the tabulated
    gradient) through the ramp ``softplus(CLIMB_SHARPNESS c) / CLIMB_SHARPNESS``, smooth, about
    zero going down or across and about the slope going up."""
    h = jnp.asarray(heading)
    norm = jnp.sqrt(h[0] ** 2 + h[1] ** 2) + 1e-6
    along = (slope_x(xy) * h[0] + slope_y(xy) * h[1]) / norm
    return jax.nn.softplus(CLIMB_SHARPNESS * along) / CLIMB_SHARPNESS


def energy_rates(xy, heading, speed):
    """``(drain, charge)`` of energy per tick at a position, heading (a pair ``(cos theta, sin
    theta)``) and forward speed: the drain at rest plus a drain per unit of speed that grows with
    the climb along the heading (``climb``: uphill only, descending costs what level ground
    costs); the charge is the solar yield there (``sun``), all of it parked and the share
    ``MOVING_CHARGE`` at speed, falling smoothly over ``PARK_SPEED``."""
    s = jnp.sqrt(speed * speed + 1e-6)
    drain = BASE_DRAIN + (MOVE_DRAIN + SLOPE_DRAIN * climb(xy, heading)) * s
    parked = jnp.exp(-((speed / PARK_SPEED) ** 2))
    charge = CHARGE_RATE * sun(xy) * (MOVING_CHARGE + (1.0 - MOVING_CHARGE) * parked)
    return drain, charge


def next_level(xy, heading, speed, level):
    """The energy level after one tick at a position, heading, speed and level: the level plus the
    charge less the drain (``energy_rates``), held in ``[0, ENERGY_MAX]`` (``_soft_clamp``)."""
    drain, charge = energy_rates(xy, heading, speed)
    return _soft_clamp(level + DT * (charge - drain))


class VehicleStep(eqx.Module):
    """The rover's motion and battery: the mean of the next state ``(x, y, cos theta, sin theta,
    s, e)`` (position, the heading as a unit vector, the forward speed, the energy level) from the
    state and the command ``(drive, turn)``, drive in ``[0, 1]`` (the target speed as a share of
    ``TOP_SPEED``) and turn in ``[-1, 1]``::

        g = drive_share(e)                     (the stall: near one, falling to zero near empty)
        x += DT * s * cos theta,   y += DT * s * sin theta
        theta += DT * TURN_MAX * g * turn      (a rotation of the unit vector; in place too)
        s = KEEP * s + (1 - KEEP) * TOP_SPEED * g * drive
        e = next_level((x, y), theta, s, e)    (drain and solar charge, before the tick's move)

    The speed moves toward the target, so a constant command holds a constant speed (full speed
    ahead is ``(1, 0)``, a stop is ``(0, 0)``) and the rover never reverses. Near empty the drive
    delivers less and less of the command, so the speed decays toward zero whatever is commanded
    and the rover stands, charging as fast as the light where it stands allows.

    The heading is carried as ``(cos theta, sin theta)``, renormalised before it is used, so
    there is no wrap-around and the rotation is smooth in the state and the command. Called as a
    ``NonlinearGaussianTransition``'s mean map ``f(x, controls)``; the world's ``move`` calls it
    too, so the model and the world share one declaration of the motion and the battery. The same
    on every terrain."""

    def __call__(self, x, controls):
        a = jnp.clip(controls[0], jnp.asarray(COMMAND_LOW), jnp.asarray(COMMAND_HIGH))
        g = drive_share(x[5])
        norm = jnp.sqrt(x[2] ** 2 + x[3] ** 2) + 1e-6
        c, s = x[2] / norm, x[3] / norm
        w = DT * TURN_MAX * g * a[1]
        return jnp.stack(
            [
                x[0] + DT * x[4] * c,
                x[1] + DT * x[4] * s,
                c * jnp.cos(w) - s * jnp.sin(w),
                s * jnp.cos(w) + c * jnp.sin(w),
                KEEP * x[4] + GAIN * g * a[0],
                next_level(x[:2], x[2:4], x[4], x[5]),
            ]
        )


#  the motion's class is named for the model's description (``describe`` writes code by name);
#  replace=True keeps reloads / %autoreload bound to the freshly defined VehicleStep class
register_transition("survey.vehicle", VehicleStep, replace=True)


def top_speed():
    """The steady forward speed under a full drive command, ``GAIN / (1 - KEEP)``."""
    return float(GAIN / (1.0 - KEEP))


def heading_of(x):
    """The heading (radians from +x) of a state ``(x, y, cos theta, sin theta, s)``."""
    return jnp.arctan2(x[3], x[2])


def velocity_of(x):
    """The velocity ``s * (cos theta, sin theta)`` of a state."""
    norm = jnp.sqrt(x[2] ** 2 + x[3] ** 2) + 1e-6
    return x[4] * jnp.stack([x[2], x[3]]) / norm


def zone_of(xy):
    """The survey sector of a position, numbered along x first from the south-west, a row of
    sectors at a time (``ZONE_EDGES``): 0 low x and low y, 1 high x and low y, 2 low x and high
    y, 3 high x and high y."""
    xe, ye = ZONE_EDGES
    ix = sum((xy[0] >= e).astype(jnp.int32) for e in xe[1:-1])
    iy = sum((xy[1] >= e).astype(jnp.int32) for e in ye[1:-1])
    return (ix + (len(xe) - 1) * iy).astype(jnp.int32)


def band_of(energy):
    """The energy band of a level: ``N_BANDS`` bands of ``BAND_WIDTH`` from 0 (the empty band,
    where the drive stalls) to ``N_BANDS - 1`` (full)."""
    return jnp.clip(jnp.floor(energy / BAND_WIDTH), 0, N_BANDS - 1).astype(jnp.int32)


def stalled(energy):
    """Whether the drive delivers under half the command at a level (``drive_share``)."""
    return energy < STALL_LEVEL


def _waves():
    """Per class, ``TEXTURE_WAVES`` plane waves ``(kx, ky, phase)``, ``(3, TEXTURE_WAVES, 3)``,
    from a fixed seed, with wavelengths drawn from ``TEXTURE_WAVELENGTH``."""
    rng = np.random.default_rng(7)
    out = np.zeros((len(TERRAIN_NAMES), TEXTURE_WAVES, 3))
    for c in range(len(TERRAIN_NAMES)):
        k = 2 * np.pi / rng.uniform(*TEXTURE_WAVELENGTH, TEXTURE_WAVES)
        angle = rng.uniform(0, 2 * np.pi, TEXTURE_WAVES)
        phase = rng.uniform(0, 2 * np.pi, TEXTURE_WAVES)
        out[c] = np.stack([k * np.cos(angle), k * np.sin(angle), phase], 1)
    return out.astype(np.float32)


WAVES = _waves()


def texture(points, terrain):
    """The ground albedo of a terrain class at world points ``(..., 2)``: the class's mean albedo
    scaled by ``1 + TEXTURE_AMP * tanh(wave)``, the wave a sum of plane waves over world
    coordinates scaled to unit sd. A function of the world point alone, so the ground keeps its
    pattern as the rover moves and turns."""
    w = jnp.asarray(WAVES)[terrain]  # (TEXTURE_WAVES, 3)
    phase = points[..., 0, None] * w[:, 0] + points[..., 1, None] * w[:, 1] + w[:, 2]
    wave = jnp.sum(jnp.cos(phase), axis=-1) / np.sqrt(TEXTURE_WAVES / 2)
    amp = jnp.asarray(TEXTURE_AMP, jnp.float32)[terrain]
    return jnp.asarray(ALBEDO, jnp.float32)[terrain] * (1.0 + amp * jnp.tanh(wave))


def _outside(points):
    """How far world points ``(..., 2)`` lie beyond the field's walls, as a share from 0 (more than
    ``WALL_EDGE / 2`` inside) to 1 (more than ``WALL_EDGE / 2`` beyond): a linear ramp across the
    wall, so the ground's albedo has no step."""
    gap = jnp.maximum(
        jnp.maximum(FIELD[0][0] - points[..., 0], points[..., 0] - FIELD[0][1]),
        jnp.maximum(FIELD[1][0] - points[..., 1], points[..., 1] - FIELD[1][1]),
    )
    return jnp.clip(gap / WALL_EDGE + 0.5, 0.0, 1.0)


_COLS = np.arange(IMG_SIDE) + 0.5 - IMG_SIDE / 2  # pixel offsets from the centre, left to right
_ALPHA = np.arctan(_COLS / FOCAL)  # each column's angle to the right of the heading
_SUB = (np.arange(SUBCOLUMNS) + 0.5) / SUBCOLUMNS - 0.5  # sub-column offsets within a pixel
_SUBCOLS = (_COLS[:, None] + _SUB[None, :]).reshape(-1)  # every ray's column offset
_SUBALPHA = np.arctan(_SUBCOLS / FOCAL)
_ROWS = np.arange(IMG_SIDE) + 0.5  # pixel row centres, top to bottom
_DISTANCES = NEAR * (REACH / NEAR) ** (np.arange(MARCH) / (MARCH - 1))


def _pose(x, heading):
    """The camera's pitch and roll at a position ``x[:2]`` facing ``heading``: the body pitches
    with the slope of the ground along the heading and rolls with the slope across it, and the
    lens points ``CAMERA_TILT`` below the body; and the eye's height."""
    pos = x[:2]
    gx, gy = jax.grad(lambda p: smooth_height(p))(pos)
    pitch = jnp.arctan(gx * jnp.cos(heading) + gy * jnp.sin(heading)) - CAMERA_TILT
    roll = jnp.arctan(-gx * jnp.sin(heading) + gy * jnp.cos(heading))  # ground rising left
    return pitch, roll, smooth_height(pos) + CAMERA_HEIGHT


def _march(x, heading, offsets, alphas):
    """Per ray (a column offset ``offsets`` at angle ``alphas`` to the right of the heading): the
    ground points at the march distances, the row each projects to, the topmost row reached so
    far (nearer ground hides farther ground), and the azimuth."""
    pos = x[:2]
    pitch, roll, eye = _pose(x, heading)
    az = heading - jnp.asarray(alphas, jnp.float32)
    dirs = jnp.stack([jnp.cos(az), jnp.sin(az)], axis=-1)
    dist = jnp.asarray(_DISTANCES, jnp.float32)
    pts = pos[None, None, :] + dist[None, :, None] * dirs[:, None, :]  # (C, K, 2)
    fwd = dist[None, :] * jnp.cos(jnp.asarray(alphas, jnp.float32))[:, None]
    elev = jnp.arctan((smooth_height(pts) - eye) / fwd)
    cols = jnp.asarray(offsets, jnp.float32)[:, None]
    row = IMG_SIDE / 2 - FOCAL * jnp.tan(elev - pitch) - cols * jnp.tan(roll)
    return pts, jax.lax.cummin(row, axis=1), az, pitch


def ground_view(x, heading):
    """The geometry of the forward view at each pixel's centre, at a position ``x[:2]`` facing
    ``heading`` (radians from +x): per pixel (row-major ``(IMG_SIDE, IMG_SIDE)``) the world point
    the ray through the pixel's centre meets on the ground, the distance to it along the ground
    and whether that ray meets no ground (sky); plus each column's azimuth and the camera's
    pitch. The march is ``render``'s; the picture itself integrates over each pixel."""
    pts, top, az, pitch = _march(x, heading, _COLS, _ALPHA)
    r = jnp.asarray(_ROWS, jnp.float32)
    k = jnp.sum(top[:, None, :] > r[None, :, None], axis=-1)  # (C, R)
    sky = k >= MARCH
    k_hi = jnp.clip(k, 0, MARCH - 1)
    k_lo = jnp.clip(k - 1, 0, MARCH - 1)
    top_hi = jnp.take_along_axis(top, k_hi, axis=1)
    top_lo = jnp.take_along_axis(top, k_lo, axis=1)
    frac = jnp.where(k > 0, jnp.clip((top_lo - r) / jnp.maximum(top_lo - top_hi, 1e-6), 0, 1), 1.0)
    dist = jnp.asarray(_DISTANCES, jnp.float32)
    d = dist[k_lo] + frac * (dist[k_hi] - dist[k_lo])
    dirs = jnp.stack([jnp.cos(az), jnp.sin(az)], axis=-1)
    points = x[None, None, :2] + d[..., None] * dirs[:, None, :]
    return (
        jnp.swapaxes(points, 0, 1),
        jnp.swapaxes(d, 0, 1),
        jnp.swapaxes(sky, 0, 1),
        az,
        pitch,
    )


def _sky_value(rows, az, pitch):
    """The sky's brightness at image rows ``rows`` (any shape broadcast with ``az``): brightest
    at the top, somewhat less at the horizon, and brighter toward the sun's azimuth (a smooth
    function of the angle between the view and the sun); it does not depend on the sunlight where
    the rover stands."""
    elev = jnp.arctan((IMG_SIDE / 2 - rows) / FOCAL) + pitch
    up = jnp.clip(elev / (np.pi / 4), 0.0, 1.0)
    return (
        SKY_HORIZON
        + (SKY_ZENITH - SKY_HORIZON) * up
        + SUN_GLOW * (0.5 + 0.5 * jnp.cos(az - SUN_AZIMUTH)) ** 2
    )


def render(x, terrain, heading):
    """The noiseless camera image (IMG_PIXELS values in [0, 1], row-major, row 0 at the top) at a
    position ``x[:2]``, a terrain class and a heading (radians from +x).

    Each pixel's value is the light it collects over its footprint, a smooth function of the
    position and the heading:

    - across the pixel's width, the mean of ``SUBCOLUMNS`` rays (a midpoint rule);
    - down each ray, the exact integral over the pixel's row interval of the ray's profile: each
      ray marches ``MARCH`` distances from ``NEAR`` to ``REACH`` (spaced geometrically) over the
      analytic height field (``smooth_height``) and projects each sample to a row; the running
      topmost row hides farther ground behind nearer, and between two visible samples the
      ground's value varies linearly in the row. Rows below the nearest sample show the nearest
      sample's ground; rows above the topmost show the sky. A pixel straddling the skyline or an
      occluding ridge takes each part in proportion to the rows it covers, so the picture has no
      threshold.

    A ground sample shows the class's ``texture`` at its world point, blended toward the dark
    ``WALL_ALBEDO`` across a wall (``_outside``), times the sunlight there (``light``) and
    ``EXPOSURE``, fading over ``HAZE`` toward the haze's grey ``HAZE_VALUE``. The whole visible
    ground carries the texture of the class under the rover. The sky is brightest at the top and
    brighter toward the sun's azimuth (``_sky_value``)."""
    pts, top, az, pitch = _march(x, heading, _SUBCOLS, _SUBALPHA)  # (C', K, 2), (C', K)
    outside = _outside(pts)
    albedo = (1 - outside) * texture(pts, terrain) + outside * WALL_ALBEDO
    lit = light(jnp.moveaxis(pts, -1, 0))
    fade = jnp.exp(-jnp.asarray(_DISTANCES, jnp.float32) / HAZE)[None, :]
    value = EXPOSURE * albedo * lit * fade + HAZE_VALUE * (1 - fade)  # (C', K)
    lo_row = jnp.arange(IMG_SIDE, dtype=jnp.float32)[None, :, None]  # each pixel's [r, r + 1]
    hi_row = lo_row + 1.0
    # the visible segments between consecutive samples, from row top[k] up to row top[k + 1]
    t0, t1 = top[:, None, :-1], top[:, None, 1:]  # (C', 1, K - 1), t1 <= t0
    v0, v1 = value[:, None, :-1], value[:, None, 1:]
    a = jnp.maximum(t1, lo_row)
    b = jnp.minimum(t0, hi_row)
    seg = jnp.maximum(b - a, 0.0)
    mid = 0.5 * (a + b)
    span = jnp.maximum(t0 - t1, 1e-9)
    share = jnp.clip((t0 - mid) / span, 0.0, 1.0)
    ground = jnp.sum(seg * (v0 + share * (v1 - v0)), axis=-1)  # (C', R)
    lo_row, hi_row = lo_row[..., 0], hi_row[..., 0]
    near = jnp.maximum(hi_row - jnp.maximum(top[:, :1], lo_row), 0.0)  # below the nearest
    ground = ground + near * value[:, :1]
    far = top[:, -1:]
    sky_cover = jnp.maximum(jnp.minimum(far, hi_row) - lo_row, 0.0)  # above the topmost
    sky_mid = 0.5 * (lo_row + jnp.minimum(far, hi_row))
    sky = sky_cover * _sky_value(sky_mid, az[:, None], pitch)
    img = (ground + sky).reshape(IMG_SIDE, SUBCOLUMNS, IMG_SIDE).mean(axis=1)  # (C, R)
    return jnp.clip(img.T, 0.0, 1.0).reshape(-1)


def to_uint8(image):
    """An image as the 0..255 integers the viewer shows (values clipped to [0, 1])."""
    return np.asarray(np.round(np.clip(np.asarray(image), 0.0, 1.0) * 255), np.uint8)


def gauge_table():
    """The band meter's confusion ``(reading, band)``: the true band with probability
    ``GAUGE_ACC``, the rest split equally between the two adjacent bands; at the bottom and the top
    band the share of the missing neighbour reads the true band. No reading two or more bands
    away."""
    table = np.zeros((N_BANDS, N_BANDS))
    for band in range(N_BANDS):
        for b in (band - 1, band + 1):
            table[min(max(b, 0), N_BANDS - 1), band] += (1 - GAUGE_ACC) / 2
        table[band, band] += GAUGE_ACC
    return table


def level_of(state):
    """The energy level of a world state (the last coordinate of its ``x``)."""
    return state["x"][5]


def initial_state(given=None):
    """The world's state at the start: the rover at ``START``, at rest, facing +y, with a full
    battery (``x`` is ``(x, y, cos theta, sin theta, s, e)``), on the terrain there; ``given`` (a
    partial state keyed by name) overrides entries: ``x`` (six coordinates, or five with the
    level kept) and ``energy`` (the level alone)."""
    x = jnp.asarray([*START, 0.0, 1.0, 0.0, ENERGY_MAX], jnp.float32)
    given = dict(given or {})
    if "x" in given:
        v = jnp.asarray(given.pop("x"), jnp.float32)
        x = x.at[: v.shape[0]].set(v)
    if "energy" in given:
        x = x.at[5].set(jnp.asarray(given.pop("energy"), jnp.float32))
    full = {"terrain": terrain_of(x[:2]), "x": x}
    full.update({k: jnp.asarray(v, full[k].dtype) for k, v in given.items()})
    return full


def observe(key, state):
    """The readings at a state: the gauge (the energy level, noise ``GAUGE_SD``), the band meter
    (the level's band, right with ``GAUGE_ACC``), the position fix (noise ``sd_fix`` at the
    position) and the camera image: the render along the rover's heading (the camera is mounted
    forward), plus pixel noise."""
    k_gauge, k_band, k_fix, k_img = jax.random.split(key, 4)
    pos = state["x"][:2]
    level = level_of(state)
    band_p = jnp.asarray(gauge_table(), jnp.float32)[:, band_of(level)]  # readings given the band
    image = render(state["x"], state["terrain"], heading_of(state["x"]))
    return {
        "gauge": level[None] + GAUGE_SD * jax.random.normal(k_gauge, (1,)),
        "band": jax.random.categorical(k_band, jnp.log(band_p)).astype(jnp.int32),
        "fix": pos + sd_fix(pos) * jax.random.normal(k_fix, (2,)),
        "img": image + PIXEL_SD * jax.random.normal(k_img, (IMG_PIXELS,)),
    }


def move(key, state, command):
    """One tick of the true dynamics from ``state`` under the ``command`` (drive, turn): the
    vehicle's step (``VehicleStep``, the model's own mean map, with its battery and its stall)
    plus noise of sd ``STATE_SD``, the heading renormalised to a unit vector, the level held in
    ``[0, ENERGY_MAX]``, and where the world differs from the model: the field has walls (the
    position is clipped to them and a rover that hits one stops)."""
    a = jnp.clip(command, jnp.asarray(COMMAND_LOW), jnp.asarray(COMMAND_HIGH))
    noise = jnp.asarray(STATE_SD, jnp.float32) * jax.random.normal(key, (6,))
    x = VehicleStep()(state["x"], (a,)) + noise
    norm = jnp.sqrt(x[2] ** 2 + x[3] ** 2)
    lo = jnp.asarray([FIELD[0][0], FIELD[1][0]], jnp.float32)
    hi = jnp.asarray([FIELD[0][1], FIELD[1][1]], jnp.float32)
    hit = jnp.any((x[:2] < lo) | (x[:2] > hi))  # the field has walls: clip, and stop
    x = jnp.concatenate(
        [
            jnp.clip(x[:2], lo, hi),
            x[2:4] / norm,
            jnp.where(hit, 0.0, x[4:5]),
            jnp.clip(x[5:], 0.0, ENERGY_MAX),
        ]
    )
    return {"terrain": terrain_of(x[:2]), "x": x}


def vehicle_name(name, vehicle=None):
    """A port or state name of one vehicle: ``name`` itself in a world of one (``vehicle`` None),
    ``name_k`` for vehicle ``k`` of a world of several."""
    return name if vehicle is None else f"{name}_{vehicle}"


def vehicle_state(state, vehicle=None):
    """One vehicle's state (``terrain`` and ``x``) out of a world's state: the state itself in a
    world of one (``vehicle`` None), the entries under vehicle ``k``'s names otherwise."""
    if vehicle is None:
        return state
    return {k: state[vehicle_name(k, vehicle)] for k in ("terrain", "x")}


def _vehicles(vehicles):
    """The vehicle indices of a world of ``vehicles``: ``(None,)`` for one (its bare names),
    ``(1, ..., vehicles)`` otherwise."""
    return (None,) if vehicles == 1 else tuple(range(1, vehicles + 1))


def world_system(look=False, vehicles=1):
    """``(System, SystemParams)``: the world's declaration, for ``couple``: a black box, its
    ports and their types alone.

    The in-port ``v`` is the velocity command, a float vector of width 2 that the rover's action
    node drives once the two are coupled; the out-ports are the readings, ``fix`` (a float vector
    of width 2, the position fix), ``img`` (a float vector of ``IMG_PIXELS``, the camera image),
    ``gauge`` (a float vector of width 1, the energy level's reading) and ``band``
    (``N_BANDS`` levels, the band meter's reading), and with ``look`` the terrain class a
    recogniser reads off the image (three levels; ``world_stepper``). What lies between them is the
    hand-written step; the world states no factor and no table, so its description length is
    reported unscored. With ``vehicles`` of two or more, every port once per vehicle under its
    index (``v_1``, ``fix_1``, ...; ``vehicle_name``)."""
    ins, outs = {}, {}
    for k in _vehicles(vehicles):
        ins[vehicle_name("v", k)] = (FLOAT, 2)
        outs[vehicle_name("fix", k)] = (FLOAT, 2)
        outs[vehicle_name("img", k)] = (FLOAT, IMG_PIXELS)
        outs[vehicle_name("gauge", k)] = (FLOAT, 1)
        outs[vehicle_name("band", k)] = (LEVEL, N_BANDS)
        if look:
            outs[vehicle_name("look", k)] = (LEVEL, len(TERRAIN_NAMES))
    return black_box(ins=ins, outs=outs)


def _overrides(given, vehicle):
    """The overrides of ``initial_state`` for one vehicle out of a world's carry: in a world of one
    the carry itself; in a world of several, vehicle ``k`` starts at ``STARTS[k - 1]`` and takes
    the carry's entries under its own names (``x_1``, ``energy_1``) and its unindexed entries
    (``energy`` for every vehicle), its own winning."""
    if vehicle is None:
        return given
    given = dict(given or {})
    own = {k: v for k, v in given.items() if not k.rpartition("_")[2].isdigit()}
    own = {"x": jnp.asarray(STARTS[vehicle - 1], jnp.float32), **own}
    suffix = f"_{vehicle}"
    own.update({k[: -len(suffix)]: v for k, v in given.items() if k.endswith(suffix)})
    return own


def world_stepper(recognise=None, vehicles=1):
    """The world's step for ``run``: the carry is the true state (keyed by name), ``obs`` holds
    the velocity command under the name of ``v`` in ``system``, and the readings are returned
    under the names of the emissions. The first call (``obs is None``) starts the world, from
    ``initial_state`` with the carry as its overrides; every other call moves it one tick
    (``move``) and reads it (``observe``). With ``recognise`` (an image to a terrain class) the
    camera's pipeline also reports that class, the ``look`` out-port. It never learns, so
    ``params`` come back as given. With ``vehicles`` of two or more, each vehicle is stepped as a
    world of one would step it under the key ``fold_in(key, k)``, from ``STARTS[k - 1]``, its
    state and readings under its index (``vehicle_name``); nothing passes between vehicles."""

    def one(key, obs, carry, vehicle):
        if obs is None:
            state = initial_state(_overrides(carry, vehicle))
            readings = observe(key, state)
        else:
            k_move, k_obs = jax.random.split(key)
            state = move(k_move, vehicle_state(carry, vehicle), obs)
            readings = observe(k_obs, state)
        if recognise is not None:
            readings["look"] = jnp.asarray(recognise(readings["img"]), jnp.int32)
        return readings, state

    def step(system, key, obs, carry, params):
        names = system.names
        name = {full.rpartition(".")[2]: full for full in (*names.emissions, *names.controls)}
        if vehicles == 1:
            readings, state = one(key, None if obs is None else obs[name["v"]], carry, None)
            return {name[k]: value for k, value in readings.items() if k in name}, state, params
        out, states = {}, {}
        for k in _vehicles(vehicles):
            command = None if obs is None else obs[name[vehicle_name("v", k)]]
            readings, state = one(jax.random.fold_in(key, k), command, carry, k)
            for r, value in readings.items():
                if vehicle_name(r, k) in name:
                    out[name[vehicle_name(r, k)]] = value
            states.update({vehicle_name(s, k): value for s, value in state.items()})
        return out, states, params

    return step


world_step = world_stepper()


def world_summary():
    """Print the ranges of the tabulated maps and the terrain areas."""
    for name, arr in (
        ("height", HEIGHT),
        ("openness", OPEN),
        ("light", LIGHT),
        ("sun", SUN),
        ("slope", SLOPE),
        ("sd_fix", SD_FIX),
    ):
        print(f"{name:9s} min {arr.min():.3f}  mean {arr.mean():.3f}  max {arr.max():.3f}")
    for k, n in enumerate(TERRAIN_NAMES):
        print(f"{n:9s} {np.mean(TERRAIN_GRID == k):.3f} of the field")


def net_rate(speed, slope, solar_yield):
    """The declared net energy per tick (charge less drain) at a steady ``speed`` with the climb
    ``slope`` along the heading on ground of solar yield ``solar_yield``: the arithmetic of
    ``energy_rates`` for a held state, in floats."""
    s_eff = np.sqrt(speed * speed + 1e-6)
    ramp = np.logaddexp(0.0, CLIMB_SHARPNESS * slope) / CLIMB_SHARPNESS
    drain = BASE_DRAIN + (MOVE_DRAIN + SLOPE_DRAIN * ramp) * s_eff
    parked = np.exp(-((speed / PARK_SPEED) ** 2))
    return CHARGE_RATE * solar_yield * (MOVING_CHARGE + (1 - MOVING_CHARGE) * parked) - drain


def calibration(speed=CRUISE_SPEED):
    """The energy calibration table: the net energy per tick driving at ``speed`` on flat ground,
    up a steep slope and down it at the median solar yield, and parked on ground of high, median
    and low yield (``net_rate``)."""
    med = float(np.median(SUN))
    net = net_rate

    lines = [
        f"top forward speed {top_speed():.3f}; speed {speed:.3f}",
        "solar yield percentiles 10/50/90/99: "
        + " / ".join(f"{np.percentile(SUN, q):.3f}" for q in (10, 50, 90, 99)),
    ]
    steep = float(np.percentile(SLOPE, 98))
    for name, value in (
        ("drive, flat, median yield", net(speed, 0.0, med)),
        (f"drive up slope {steep:.2f} (98th pct), median yield", net(speed, steep, med)),
        (f"drive down slope {steep:.2f} (98th pct), median yield", net(speed, -steep, med)),
        ("drive at the top speed, flat, median yield", net(top_speed(), 0.0, med)),
    ):
        lines.append(f"{name}: net {value:+.4f}/tick, {24 * value:+.3f} per 24 ticks")
    for q in (99, 90, 50, 10):
        y = float(np.percentile(SUN, q))
        value = net(0.0, 0.0, y)
        lines.append(f"parked, yield {y:.3f} ({q}th pct): net {value:+.4f}/tick")
    return "\n".join(lines)


if __name__ == "__main__":
    world_summary()
    print(calibration())
