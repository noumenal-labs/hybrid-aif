"""Closed-loop runs of the survey rover: a rover that maps a terrain field by driving over it
while keeping its energy up.

The rover (``survey_model``) infers its state, plans a velocity command by gradient descent from
several starts (the plan it acted on in the round before, shifted one tick on, and spread points
of the control box), acts, and learns the terrain field by counting each
terrain it sees into the cells it was in. The true world (``survey_world``) is a black box, declared
by its ports alone and coupled to the rover with ``couple``: the rover's velocity action node drives
the world's in-port ``v``, and the world's out-ports (the fix, the image and the gauge) feed the
rover's modalities. The rover's battery is a coordinate of its vehicle state: driving drains it, the
sun charges it where the rover parks, and near empty the drive stalls, in the world and in the
rover's model alike.

The two settings are the scheduler over the rover (``two-level``) and the rover alone
(``one-level``, the fast tier with its own novelty term and no scheduler). For ``two-level`` the
rover and a survey scheduler (``survey_scheduler``) are first merged into one agent: the
scheduler's option action node drives the rover's option control node (an ``Identify``), the
field is one parameter variable (an ``Identify``), and the scheduler's cell modality observes an
emission of the rover, its position belief through the cell indicator realised by ``mode`` (the
``Link`` sugar). The agent is coupled to the world as one component, and one ``run`` call over the
closed System steps three leaves: the scheduler by ``plan()`` (the enumerated planner, its metrics
kept in the carry's trace), the rover by ``plan()`` too (the gradient planner warm-started from its
last plan, learning the field, its trace kept; ``rover_plan``), and the world by its hand-written
step. The page's per-tick records are read from the run's ``Trajectory`` afterwards
(``rover_record``, ``decision_record``), and the timing of the rover's parts by a separate harness
(``timing``). The scheduler's params declare a duration for every option, so it decides once per
option, every ``SLOW_TICKS`` rounds, and the rover holds the option between decisions. What the
rover learns is written back to the shared field each round, so the scheduler's novelty reads the
counts gathered so far; the scheduler learns its energy table alone, from its band readings, once
per decision (``survey_scheduler.learn_rates``). The order is sequential: the scheduler steps first,
the rover as soon as the option it reads is realised, the world last on the command just realised;
the scheduler's cell reading is the rover's emission of the round before, absent in round 0. The two
ablations of the page: ``--camera none`` (the rover's model without the image) and ``--no-fix`` (the
fix arrives and is never read); ``--camera look`` has the rover observe the recogniser's reading of
the terrain instead of the image. The runs are written to one bundle for the project page, and a
REPORT.md summarises them.

    python examples/survey_run.py --out DIR [--settings id,id] [--ticks 200] [--seed 0]
    python examples/survey_run.py --out DIR --assemble     # merge the part files into bundle
    python examples/survey_trim.py DIR/bundle.json OUT_DIR   # the page's bundle.js
"""

import argparse
import functools
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import survey_camera as cam
import survey_model as sm
import survey_scheduler as ss
import survey_world as wd

from hybrid_aif import Parameter, __version__
from hybrid_aif.composition import Identify, Link, couple, joint_planner, merge, plan, run
from hybrid_aif.description import describe
from hybrid_aif.description.mdl import description_length
from hybrid_aif.objectives import total
from hybrid_aif.objectives.base import term_sign
from hybrid_aif.planning.gradient import make_gradient_planner

PLANNER_STEPS = 40
PLANNER_STARTS = 4  # starts of the plan's descent (``rover_plan``); the best final plan is the
#  policy
TIMING_REPS = 20  # calls per part of the timing harness (``timing``)
LIVE_WEIGHT = 1e-3
FIELD_LR = 1.0  # weight of one tick's terrain counts in the field's Dirichlet (one observation)
FIELD_STRENGTH = 0.1  # pseudo-counts per cell of the field's uniform prior
COVERED_MAX = 0.6  # a cell counts as mapped when its field's top probability exceeds this
SEEN_COUNTS = 1.0  # a cell counts as seen once it has absorbed this many counts
MDL_RESOLUTION = 100.0
MARKS = (50, 100, 200)  # the ticks REPORT.md gives coverage at
MAP_SIDE = 61  # the world maps' samples per axis, published
EPI = ("transition:terrain", "obs:fix")
#  how the rover sees through its camera (``camera_mode``): "image", the forward camera's picture
#  through the decoder (the showcase); "look", the recogniser's reading (the comparison
#  experiment A); "none", no camera (the camera ablation)
CAMERA = "image"
NAMES = {"terrain": sm.TERRAIN_NAMES, "energy": sm.ENERGY_NAMES}


def interval_pair(interval, i):
    return [float(interval.lower[i]), float(interval.upper[i])]


def page_name(leaf, outer):
    """A term tree's leaf name as the page knows it, the agent's own: ``outer``, the prefix the
    coupling gives the agent's names (``"agent."`` or ``"rover."``), dropped after any
    ``<slot>:`` head (``transition:agent.rover.terrain`` is ``transition:rover.terrain``)."""
    head, colon, name = leaf.rpartition(":")
    return f"{head}{colon}{name.removeprefix(outer)}"


def term_record(tree, i, outer, offsets=None):
    """Plan ``i``'s leaves for the page, by term and page name, each an interval with its sign
    applied, less the constant ``offsets[term][name]`` where one is given (``wall_offset``)."""
    offsets = offsets or {}
    out = {}
    for term, leaves in tree.items():
        out[term] = {}
        for leaf, iv in leaves.items():
            name = page_name(leaf, outer)
            lo, hi = interval_pair(term_sign(term) * iv, i)
            c = offsets.get(term, {}).get(name, 0.0)
            out[term][name] = [lo - c, hi - c]
    return out


def wall_offset(policy_len):
    """The wall preference's value over a plan that stays inside the field: ``policy_len`` steps
    of ``-log C`` at the field's centre, where the preference's log density is flat. A constant
    across plans (the mixture's normaliser), so the page's records show the leaf less it."""
    target = sm.wall_target()
    centre = jnp.asarray([[np.mean(wd.FIELD[0]), np.mean(wd.FIELD[1])]], jnp.float32)
    return float(-policy_len * target.log_prob(centre)[0])


def position_risk_offset(tree, name, outer):
    """The constant the position risk's leaf (``risk`` on ``name``) is recorded less: the smallest
    lower bound of that leaf over the tick's plans, so the page shows each plan's risk above the
    best plan's and is not dominated by the part every plan shares (the divergence of a tight
    belief from the broad sector target); zero when there is no such leaf."""
    for leaf, iv in tree.get("risk", {}).items():
        if page_name(leaf, outer) == name:
            return float(jnp.min((term_sign("risk") * iv).lower))
    return 0.0


def approximate_record(tree, outer):
    return {
        term: {page_name(leaf, outer): bool(iv.approximate) for leaf, iv in leaves.items()}
        for term, leaves in tree.items()
    }


def position_belief(beliefs, pre=""):
    x = beliefs[pre + "x"]
    weights = np.exp(np.asarray(x.log_weights))
    weights = weights / weights.sum()
    comps = []
    for k in range(x.num_components):
        if weights[k] < LIVE_WEIGHT:
            continue
        chol = np.asarray(x.chols[k])
        cov = (chol @ chol.T)[:2, :2]
        comps.append(
            dict(
                w=float(weights[k]),
                mu=[float(m) for m in np.asarray(x.means[k])[:2]],
                cov=[[float(c) for c in row] for row in cov],
            )
        )
    mean = np.asarray(x.mean())
    cov = np.asarray(x.cov())
    # the heading's mean and sd from the moment-matched belief over (cos theta, sin theta),
    # linearised through atan2; the speed's from its own coordinate
    c, s = mean[2], mean[3]
    jac = np.asarray([-s, c]) / max(c * c + s * s, 1e-9)
    heading_sd = float(np.sqrt(max(jac @ cov[2:4, 2:4] @ jac, 0.0)))
    heading = float(np.arctan2(s, c))
    speed = float(mean[4])
    level, level_sd = float(mean[5]), float(np.sqrt(max(cov[5, 5], 1e-12)))
    return dict(
        components=comps,
        terrain=[float(p) for p in np.asarray(beliefs[pre + "terrain"].probs)],
        energy=band_probabilities(level, level_sd),
        energy_level=level,
        energy_sd=level_sd,
        velocity=[speed * float(np.cos(heading)), speed * float(np.sin(heading))],
        heading=heading,
        heading_sd=heading_sd,
        speed=speed,
        speed_sd=float(np.sqrt(max(cov[4, 4], 0.0))),
        mean=[float(mean[0]), float(mean[1])],
    )


def band_probabilities(level, sd):
    """The probability of each energy band (``survey_world.BAND_WIDTH`` each, the top band open
    above) under a Gaussian belief over the level, its mass below the first boundary counted in the
    empty band."""
    edges = np.arange(1, sm.N_BANDS, dtype=float) * wd.BAND_WIDTH
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)((edges - level) / (sd * np.sqrt(2.0))))
    cuts = np.concatenate([[0.0], cdf, [1.0]])
    return [float(p) for p in np.diff(cuts)]


def image_rows(flat):
    """A flat pixel vector (row-major, as the renderer lays it out) as uint8 rows of the picture:
    row 0 is the top of the picture, columns run left to right."""
    grid = np.asarray(flat).reshape(wd.IMG_SIDE, wd.IMG_SIDE)
    return [[int(v) for v in row] for row in wd.to_uint8(grid)]


def field_record(field, strength):
    """The field's posterior mean per cell as uint8 probabilities ``(cells, 3)``, the counts it
    has absorbed per cell beyond its prior, the fraction of cells whose top probability exceeds
    ``COVERED_MAX`` and the fraction that has absorbed ``SEEN_COUNTS`` counts; ``field`` is the
    parameter variable (a ``ConditionalDirichlet``)."""
    conc = np.asarray(field.concentration, np.float64)
    total_conc = conc.sum(axis=0)
    mean = conc / total_conc
    rows = np.asarray(np.round(mean.T * 255), np.uint8)
    return (
        [[int(v) for v in row] for row in rows],
        [float(c) for c in total_conc - strength],
        float(np.mean(mean.max(axis=0) > COVERED_MAX)),
        float(np.mean(total_conc - strength >= SEEN_COUNTS)),
    )


def rover_plan(setting, steps, starts, pre):
    """The rover's step for ``run``: the library's ``plan()`` with the gradient planner
    (``steps`` descent steps from ``starts`` starts, the best final plan acted on), the field
    learned each round at the rate ``field_lr`` (from the second round on), the round's
    :class:`~hybrid_aif.composition.Trace` kept (``trace``) and the plan the rover acted on,
    shifted one tick on, as the next round's first start (``warm_start``; zeros in the first
    round); the other starts are the library's spread points of the control box. The view's
    emissions (the cell the scheduler reads, two-tier only) are realised from the posterior.
    Under the fix ablation (the setting's ``fix`` false) the fix reaches the step and is handed
    on as absent (``None``), so it is never read. The step runs inside ``run``'s compiled scan of
    rounds, so it reads no clock; a run's wall time is taken around the ``run`` call."""
    step = plan(
        make_gradient_planner(steps=steps, starts=starts),
        learn={"lr": setting["field_lr"]},
        trace=True,
        warm_start=True,
        selection="deterministic",
    )
    if not setting.get("fix", True):
        step = unread(step, pre + "fix")
    return step


def unread(step, slot):
    """``step`` with the reading of the modality ``slot`` handed on as absent (``None``) every
    round: the fix ablation."""

    @functools.wraps(step)
    def wrapper(system, key, obs, carry, params):
        if obs is not None:
            obs = {**obs, slot: None}
        return step(system, key, obs, carry, params)

    return wrapper


def rover_record(carry, setting, pre, decoder, policy_len):
    """The page's record of one round of the rover, read from the carry its ``plan()`` step
    left (``MindCarry``): the posterior (``previous``), and from the round's ``Trace`` every
    start's plan with its term tree, ``G`` and ``q_pi``, the inference detail (the image's log
    evidence, the free energy) and the field the round planned with. ``decoder`` renders the
    picture it predicts at the belief. The wall preference's leaf and the position risk's leaf
    are recorded less constants (``wall_offset``, ``position_risk_offset``), as is ``G``."""
    outer = pre.split(".", 1)[0] + "."  # the coupling's prefix of the agent's names
    beliefs, trace = carry.previous, carry.trace
    q_pi, tree, v_plans = trace.q_pi, trace.tree, np.asarray(trace.plans)
    mean = np.asarray(beliefs[pre + "x"].mean())
    terrain = int(jnp.argmax(beliefs[pre + "terrain"].probs))
    camera = recon = None
    if decoder is not None:  # the decoder's picture and noise at the belief, for the page
        recon, recon_scale = decoder.diag_moments((jnp.asarray(mean), jax.nn.one_hot(terrain, 3)))
        camera = dict(
            log_evidence=float(trace.detail.log_evidence[pre + "img"]),
            scale_mean=float(jnp.mean(recon_scale)),
        )
    g = total(tree)
    wall = page_name(pre + "fix", outer)
    walled = any(page_name(leaf, outer) == wall for leaf in tree.get("utility", {}))
    offset = wall_offset(policy_len) if walled else 0.0
    x_risk = position_risk_offset(tree, page_name(pre + "x", outer), outer)
    offsets = {"utility": {wall: offset}, "risk": {page_name(pre + "x", outer): x_risk}}
    records = [
        dict(
            id=i,
            start=[float(mean[0]), float(mean[1])],
            v_plan=[[float(c) for c in step] for step in v_plans[i]],
            terms=term_record(tree, i, outer, offsets),
            G=[v - offset - x_risk for v in interval_pair(g, i)],
            q_pi=float(q_pi.probs[i]),
        )
        for i in range(v_plans.shape[0])
    ]
    table, counts, coverage, seen = field_record(
        trace.parameters[pre + "field"], setting["field_strength"]
    )
    return dict(
        #  the constants the page's records are shown less
        offsets=dict(wall=offset, position_risk=x_risk),
        belief=position_belief(beliefs, pre),
        policies=records,
        chosen=int(jnp.argmax(q_pi.probs)),
        approximate=approximate_record(tree, outer),
        camera=camera,
        free_energy=float(trace.detail.free_energy),
        recon=None if recon is None else image_rows(recon),
        map=table,
        counts=counts,
        coverage=coverage,
        seen=seen,
    )


def tick_rows(rec, rover, ticks, record, seconds, extra=None, recognise=None):
    """The per-tick records of a run, read from its ``Trajectory``: the world's true state at the
    start of the tick (its carry), the readings the rover was handed and the command it chose
    (the record of the stepped node ``rover``), the rover's record of the tick (``record`` of its
    carry after the tick, ``rover_record``) and the tick's wall time (``seconds``, the mean tick of
    the compiled scan of rounds, compile excluded: ``run_seconds``). The camera
    image is the one the rover was handed, or, when its model has no image, the one the world
    emitted for that round; ``recognise`` (an image to the recogniser's class probabilities) adds
    the recogniser's readout of it."""
    world, mind = rec["world"], rec[rover]
    rows = []
    for t in range(ticks):
        state = world.carries[t]
        energy = float(wd.level_of(state))
        if "img" in mind.received:
            image = mind.received["img"][t]
        else:  # the world's emission the round before (its start for round 0)
            image = world.start["img"] if t == 0 else world.emitted["img"][t - 1]
        row = dict(
            t=t,
            v=[float(c) for c in mind.intended["a_v"][t]],
            true_x=[
                *(float(c) for c in state["x"][:2]),
                *(float(c) for c in wd.velocity_of(state["x"])),
            ],
            true_heading=float(wd.heading_of(state["x"])),
            true_speed=float(state["x"][4]),
            true_terrain=int(state["terrain"]),
            true_energy=dict(band=int(wd.band_of(jnp.asarray(energy))), level=energy),
            stalled=bool(wd.stalled(energy)),
            drive_share=float(wd.drive_share(jnp.asarray(energy))),
            fix=[float(c) for c in mind.received["fix"][t]],
            gauge=float(mind.received["gauge"][t][0]),
            img=image_rows(image),
            seconds=seconds,
            **record(mind.carries[t + 1]),
        )
        if recognise is not None:
            row["recogniser"] = [float(p) for p in recognise(jnp.asarray(image))]
        if "look" in mind.received:
            row["look"] = int(mind.received["look"][t])
        if extra is not None:
            row.update(extra(t))
        rows.append(row)
    return rows


def camera_mode(setting):
    """How a setting's rover sees through its camera: ``"image"`` (the image, through the
    decoder), ``"look"`` (the recogniser's reading of the terrain) or ``"none"``; the setting's
    ``camera`` entry (``True`` and ``False`` read as image and none)."""
    mode = setting.get("camera", CAMERA)
    return {True: "image", False: "none"}.get(mode, mode)


def sees_image(mode):
    """Whether the rover's model inverts the image (through the decoder) under a camera mode."""
    return mode == "image"


def world_bindings(agent, mode="image"):
    """The identifications that couple the world to the rover inside the component ``agent``
    (``"agent.rover"`` or ``"rover"``): the rover's command drives the world's ``v``, and the
    world's emissions feed the rover's modalities (the image or the recogniser's reading, by the
    camera ``mode``)."""
    return [
        Identify(f"{agent}.a_v", "world.v"),
        Identify("world.fix", f"{agent}.fix"),
        *([Identify("world.img", f"{agent}.img")] if sees_image(mode) else []),
        *([Identify("world.look", f"{agent}.look")] if mode == "look" else []),
        Identify("world.gauge", f"{agent}.gauge"),
    ]


def sensors(mode, encoder):
    """``(world step, readout)`` for a camera ``mode``: the world's step (reporting the
    recogniser's class for ``"look"``) and the recogniser's class probabilities of an image for
    the page (``None`` without a recogniser)."""
    if encoder is None:
        return wd.world_stepper(), None

    def readout(image):
        return cam.terrain_probs(encoder, image)

    if mode != "look":
        return wd.world_stepper(), readout
    return wd.world_stepper(lambda image: jnp.argmax(readout(image))), readout


def epistemic_terms(setting):
    """A setting's epistemic terms for the rover: its ``epistemic`` list, less the fix's
    information gain when the setting observes no fix (``fix`` false: the fix ablation, in which
    the world's fix reaches the rover's step and is marked absent every round, so the camera is the
    only reading of position; the fix modality stays in the model as the carrier of the wall
    preference, a predicted reading the plans are scored on, and is never observed)."""
    terms = tuple(setting["epistemic"])
    return terms if setting.get("fix", True) else tuple(t for t in terms if t != "obs:fix")


def energy_preference(setting):
    """A setting's energy preference over the rover's gauge, ``(centre, sd)``, or ``None``."""
    pref = setting.get("energy_preference")
    return tuple(pref) if pref else None


def rover_of(setting, decoder, confusion=None, tiered=False):
    """The rover's declaration for a setting, ``survey_model.rover``'s ``(System, SystemParams,
    Grid, centres)``: its camera (``camera_mode``), epistemic terms, field prior, grid, planning
    belief, energy preference and horizon; ``tiered`` adds the option control node."""
    mode = camera_mode(setting)
    return sm.rover(
        decoder,
        camera=sees_image(mode),
        look=confusion if mode == "look" else None,
        epistemic=epistemic_terms(setting),
        field_strength=setting["field_strength"],
        grid_shape=setting["grid"],
        planning_belief=setting["planning_belief"],
        energy_preference=energy_preference(setting),
        policy_len=setting.get("policy_len", sm.POLICY_LEN),
        tiered=tiered,
    )


def timing(rec, node, setting, decoder, confusion, steps, starts, reps=TIMING_REPS):
    """The median wall time of the rover's two costly parts, measured apart from the run (a
    whole run is timed around the ``run`` call): the belief update (``infer_states``,
    every reading, the camera's the dominant cost) and the plan (``infer_policies`` with the
    rover's gradient planner, ``steps`` descent steps from ``starts`` starts, the first the plan
    it acted on shifted one tick on), each compiled and called ``reps`` times after two warm-up
    calls. They are timed on the rover's own declaration (``rover_of``, without the option node
    the scheduler drives) at the run's last round of ``node``: its readings, the prior it filtered
    from and the plan it acted on."""
    mind = rec[node]
    pre = node + "."
    carry_before, carry = mind.carries[-2], mind.carries[-1]
    system, params, _, _ = rover_of(setting, decoder, confusion)
    names = system.names
    last = len(mind.received[names.modalities[0]]) - 2
    obs = {m: mind.received[m][last] for m in names.modalities if m in mind.received}
    prior = None
    if carry_before.beliefs is not None:
        prior = {f: carry_before.beliefs[pre + f] for f in names.states}
    posterior = {f: carry.previous[pre + f] for f in names.states}
    shifted = None
    if carry_before.plan is not None:
        shifted = jnp.concatenate([carry_before.plan[1:], carry_before.plan[-1:]])
    planner = make_gradient_planner(steps=steps, starts=starts, init=shifted)
    compiled = system.compiled()

    def median(call):
        for _ in range(2):
            jax.block_until_ready(call())
        times = []
        for _ in range(reps):
            t0 = time.time()
            jax.block_until_ready(call())
            times.append(time.time() - t0)
        return float(np.median(times))

    return dict(
        infer=median(lambda: compiled.infer_states(params, obs, priors=prior)),
        plan=median(lambda: compiled.infer_policies(params, posterior, planner=planner)),
        reps=reps,
    )


def run_setting(
    setting,
    ticks,
    decoder,
    steps=PLANNER_STEPS,
    starts=PLANNER_STARTS,
    encoder=None,
    confusion=None,
):
    """One run of a setting: the two-tier agent (``run_tiered``, ``tiers`` 2), or the rover alone
    coupled to the world (the one-level setting). The
    camera's form is the setting's (``camera_mode``): the image through ``decoder``, or the
    reading of the recogniser ``encoder`` through its ``confusion`` table, or none."""
    if setting.get("tiers") == 2:
        return run_tiered(setting, ticks, decoder, steps, starts, encoder, confusion)
    mode = camera_mode(setting)
    rs, rp, _, _ = rover_of(setting, decoder, confusion)
    system, params = couple(
        {"rover": (rs, rp), "world": wd.world_system(look=mode == "look")},
        world_bindings("rover", mode),
    )
    world_step, readout = sensors(mode, encoder)
    start = setting.get("start_energy")
    rec = run(
        system,
        params,
        ticks,
        jax.random.PRNGKey(setting["seed"]),
        steps={"rover": rover_plan(setting, steps, starts, "rover."), "world": world_step},
        tick_order="sequential",
        carries=None if start is None else {"world": {"energy": start}},
        record_carries=True,
    )
    seconds = run_seconds(rec, ticks)
    decoder = params.likelihood.get("rover.img")

    def record(carry):
        return rover_record(carry, setting, "rover.", decoder, rs.policy_len)

    rows = tick_rows(rec, "rover", ticks, record, seconds, recognise=readout)
    return dict(
        ticks=rows,
        planner=dict(steps=steps, starts=starts),
        timing=timing(rec, "rover", setting, decoder, confusion, steps, starts),
    )


SLOW_TICKS = ss.SURVEY_TICKS  # fast ticks per slow step


def settings_grid_l2():
    """The settings of the two-tier comparison: the scheduler over the rover (two levels), and
    the rover alone (one level), the fast tier with its own novelty term and no scheduler, free to
    go where it likes. Both share the rover's epistemic terms, its energy preference and its
    field. The scheduler looks ``horizon`` options ahead, learns its energy table from its band
    readings (``survey_scheduler.LEARN``), scores a terminal target whose penalty on the empty band
    is ``terminal_penalty`` (``survey_scheduler.terminal_penalty`` of the horizon), and pools its
    map reading over an option (``readings``)."""

    def setting(sid, label, tiers, **extra):
        out = dict(
            id=sid,
            label=label,
            tiers=tiers,
            epistemic=list(EPI),
            field_lr=FIELD_LR,
            field_strength=FIELD_STRENGTH,
            energy_preference=list(sm.ENERGY_PREFERENCE),
            horizon=ss.HORIZON,
            terminal_penalty=ss.terminal_penalty(ss.HORIZON, FIELD_STRENGTH),
            planning_belief=sm.PLANNING_BELIEF,
            grid=sm.GRID_SHAPE,
            slow_ticks=SLOW_TICKS,
            readings=True,
            seed=0,
        )
        out.update(extra)
        return out

    return [
        setting("two-level", "Scheduler over the rover", 2),
        setting("one-level", "Rover alone", 1),
    ]


def agent_system(decoder, setting, confusion=None):
    """The two-tier agent, the merge of the scheduler ``mission`` and the rover:
    ``((System, SystemParams), grid)``. The option's action node drives the rover's option control
    node, the field is one parameter variable, and the scheduler's cell modality observes the
    rover's position belief through the cell indicator, realised by ``mode`` (the ``Link`` sugar:
    an emission ``rover.cell_obs`` of the rover, observed by ``mission.cell_obs``). The rover's
    camera is the setting's (``camera_mode``)."""
    rs, rp, grid, centres = rover_of(setting, decoder, confusion, tiered=True)
    ms, mp = ss.scheduler(
        grid,
        centres,
        terminal=setting["terminal_penalty"],
        field_strength=setting["field_strength"],
        #  the rover's horizon, so that the joint planner's leaves agree; the scheduler's step
        #  plans with its own (``replace`` in ``run_tiered``)
        policy_len=rs.policy_len,
        energy_counts=setting.get("energy_counts"),
        durations=True,  # the scheduler decides once per option, its duration read off these
        readings=setting.get("readings", False),
    )
    agent = merge(
        {"mission": ms, "rover": rs},
        {"mission": mp, "rover": rp},
        [
            Identify("mission.option", "rover.u_opt"),
            Identify("rover.field", "mission.field"),
            Link("rover.x", "mission.cell_obs", realise="mode", kernel=grid),
        ],
    )
    return agent, grid


def survey_system(decoder, setting, confusion=None):
    """The closed System of a two-tier run: the agent (``agent_system``) coupled to the world, the
    gauge reading the rover and the band meter the scheduler. Returns ``(System, SystemParams)``."""
    agent, _ = agent_system(decoder, setting, confusion)
    mode = camera_mode(setting)
    return couple(
        {"agent": agent, "world": wd.world_system(look=mode == "look")},
        [
            *world_bindings("agent.rover", mode),
            Identify("world.band", "agent.mission.gauge"),
        ],
    )


def run_seconds(rec, ticks):
    """The mean wall time of a tick of the run ``rec``, compile excluded: the execution of
    ``run``'s compiled scan of rounds over the rounds it ran (``Trajectory.timing``, the program
    compiled ahead of time and timed apart); a run of one round, which is not scanned, takes
    that round's wall time. The rover's costly parts are timed apart from the run (``timing``)."""
    timing = rec.timing
    if timing["scanned"]:
        return timing["scan"] / timing["scanned"]
    return timing["compile"] + timing["scan"] if ticks else 0.0


def decision_record(carry, option, reading, seconds, pre="agent.mission.", n_options=sm.N_OPTIONS):
    """The record of one decision of the scheduler for the page, read from the carry its
    ``plan(trace=True)`` step left: the option sequences it scored (``trace.policies``) with
    their term trees, ``G`` and ``q_pi``, the best ``G`` and the probability of each first option,
    the posterior over the energy band and the visited cell (``carry.previous``), and the energy
    table the decision planned with (``energy_record``). ``reading`` is the cell modality's value
    that round, ``None`` where it was absent; ``n_options`` the option control node's levels;
    ``seconds`` the run's mean tick time, compile excluded (``run_seconds``)."""
    q_pi, tree, sequences = carry.trace.q_pi, carry.trace.tree, np.asarray(carry.trace.policies)
    g = total(tree)
    records = [
        dict(
            options=[int(a) for a in sequences[i].ravel()],
            terms=term_record(tree, i, pre.split(".", 1)[0] + "."),
            G=interval_pair(g, i),
            q_pi=float(q_pi.probs[i]),
        )
        for i in range(sequences.shape[0])
    ]
    midpoint = [float(np.mean(r["G"])) for r in records]
    first = [r["options"][0] for r in records]
    efe = [min(m for m, f in zip(midpoint, first, strict=True) if f == o) for o in range(n_options)]
    q_option = [
        float(sum(r["q_pi"] for r in records if r["options"][0] == o)) for o in range(n_options)
    ]
    chosen = max((i for i, f in enumerate(first) if f == option), key=lambda i: records[i]["q_pi"])
    beliefs = carry.previous
    return dict(
        decision=True,
        option=option,
        cell=reading,
        efe_by_option=efe,
        q_by_option=q_option,
        policies=records,
        chosen=chosen,
        belief=dict(
            energy=[float(p) for p in np.asarray(beliefs[pre + "energy"].probs)],
            visit=[float(p) for p in np.asarray(beliefs[pre + "visit"].probs)],
        ),
        energy_table=energy_record(carry.trace.parameters[pre + "energy_table"]),
        seconds=seconds,
    )


def energy_record(table):
    """The scheduler's energy table as the page shows it, from the parameter variable (a
    ``ConditionalDirichlet`` ``[next, band, option]``): the expected change of band per band and
    option (``change``, ``[band][option]``), the next band's probabilities (``probs``,
    ``[band][option][next]``) and the counts each case has gathered beyond its prior (``counts``,
    ``[band][option]``)."""
    conc = np.asarray(table.concentration, np.float64)
    prior = ss.energy_prior()
    return dict(
        change=np.round(ss.expected_change(conc), 4).tolist(),
        probs=np.round(np.transpose(conc / conc.sum(axis=0, keepdims=True), (1, 2, 0)), 4).tolist(),
        counts=np.round(conc.sum(axis=0) - prior.sum(axis=0), 4).tolist(),
    )


def run_tiered(
    setting,
    ticks,
    decoder,
    steps=PLANNER_STEPS,
    starts=PLANNER_STARTS,
    encoder=None,
    confusion=None,
):
    """A run of the two-tier agent coupled to the world: one ``run`` call over the closed System
    steps the scheduler (``plan()``, once per option, its trace recorded), the rover (every round)
    and the world (every round)."""
    system, params = survey_system(decoder, setting, confusion)
    world_step, readout = sensors(camera_mode(setting), encoder)
    every = setting["slow_ticks"]
    start = setting.get("start_energy")
    mission = plan(
        trace=True,
        selection="deterministic",
        replace={"policy_len": setting["horizon"]},
        learn=ss.learn_rates("agent.mission."),
    )
    rover = rover_plan(setting, steps, starts, "agent.rover.")
    rec = run(
        system,
        params,
        ticks,
        jax.random.PRNGKey(setting["seed"]),
        steps={"agent.mission": mission, "agent.rover": rover, "world": world_step},
        clocks={"agent.mission": every / ss.SURVEY_TICKS},
        tick_order="sequential",
        carries=None if start is None else {"world": {"energy": start}},
        record_carries=True,
    )
    seconds = run_seconds(rec, ticks)
    scheduler = rec["agent.mission"]
    options = scheduler.intended["option"]
    decided = [t for t in range(ticks) if bool(scheduler.decided[t])]
    decisions = {}
    for t in decided:
        reading = int(scheduler.received["cell_obs"][t])
        present = bool(scheduler.present["cell_obs"][t])
        option = int(options[t])
        carry = scheduler.carries[t + 1]
        decisions[t] = decision_record(carry, option, reading if present else None, seconds)

    def extra(t):
        return dict(option=int(options[t]), slow=decisions.get(t, dict(decision=False)))

    pre = "agent.rover."
    decoder = params.likelihood.get(pre + "img")
    horizon = setting.get("policy_len", sm.POLICY_LEN)

    def record(carry):
        return rover_record(carry, setting, pre, decoder, horizon)

    rows = tick_rows(rec, "agent.rover", ticks, record, seconds, extra, recognise=readout)
    learned = rec.params.parameters["agent.mission.energy_table"]
    return dict(
        ticks=rows,
        planner=dict(steps=steps, starts=starts),
        timing=timing(rec, "agent.rover", setting, decoder, confusion, steps, starts),
        #  the scheduler's energy table at the run's end (the parameter home after the last
        #  round): its record, and its counts, which a later mission's setting may start from
        #  (``energy_counts``)
        energy_table=energy_record(learned),
        energy_counts=np.asarray(learned.concentration, np.float64).tolist(),
    )


def downsampled_maps():
    xs, ys, height, openness, light, slope, sd_fix, terrain, sun = wd.maps()
    step = (len(xs) - 1) // (MAP_SIDE - 1)

    def thin(a, digits=3):
        return np.round(np.asarray(a)[::step, ::step], digits).tolist()

    return dict(
        xs=[round(float(v), 4) for v in xs[::step]],
        ys=[round(float(v), 4) for v in ys[::step]],
        layout="each map is indexed [ix][iy], x first",
        height=thin(height),
        light=thin(light),
        openness=thin(openness),
        sd_fix=thin(sd_fix),
        slope=thin(slope),
        sun=thin(sun),
        terrain=np.asarray(terrain)[::step, ::step].astype(int).tolist(),
    )


def dynamics_record():
    """The vehicle's motion and battery, the rule a plan of (drive, turn) commands integrates
    under, the same on every terrain."""
    return dict(
        kind="vehicle",
        dt=float(wd.DT),
        gain=float(wd.GAIN),
        keep=float(wd.KEEP),
        turn_max_degrees=float(np.degrees(wd.TURN_MAX)),
        command_low=[float(v) for v in wd.COMMAND_LOW],
        command_high=[float(v) for v in wd.COMMAND_HIGH],
        state_sd=[float(v) for v in wd.STATE_SD],
        state=list(wd.STATE_NAMES),
        top_speed=wd.top_speed(),
        cruise_speed=float(wd.CRUISE_SPEED),
        terrain_names=list(wd.TERRAIN_NAMES),
        stall=dict(level=float(wd.STALL_LEVEL), width=float(wd.STALL_WIDTH)),
        energy=dict(
            max=float(wd.ENERGY_MAX),
            base_drain=float(wd.BASE_DRAIN),
            move_drain=float(wd.MOVE_DRAIN),
            slope_drain=float(wd.SLOPE_DRAIN),
            climb_sharpness=float(wd.CLIMB_SHARPNESS),
            charge_rate=float(wd.CHARGE_RATE),
            moving_charge=float(wd.MOVING_CHARGE),
            park_speed=float(wd.PARK_SPEED),
            sun_power=float(wd.SUN_POWER),
            gauge_sd=float(wd.GAUGE_SD),
            bands=wd.N_BANDS,
            band_width=float(wd.BAND_WIDTH),
            band_meter=(
                "the band meter shows the true band with probability gauge_acc, else one of the "
                "adjacent bands (the rest split equally between the two; at the bottom and the "
                "top band the missing neighbour's share reads the true band)"
            ),
            gauge_acc=float(wd.GAUGE_ACC),
            rule=(
                "per tick, before the move: level += charge - drain, drain = base_drain + "
                "(move_drain + slope_drain * climb) * speed, climb = softplus(climb_sharpness * "
                "c) / climb_sharpness, c the slope along the heading (the height's directional "
                "derivative: uphill positive; about zero going down or across, no "
                "regeneration), charge = charge_rate * sun * "
                "(moving_charge + (1 - moving_charge) * exp(-(speed / park_speed)^2)), sun the "
                "map world.sun (the sunlight over its maximum, to sun_power); held in [0, max]"
            ),
        ),
        rule=(
            "per tick, the command (drive, turn), drive in [0, 1] (the target speed as a share of "
            "the top speed) and turn in [-1, 1], scaled by the drive share g = 1 / (1 + "
            "exp(-(level - stall.level) / stall.width)), near one above the stall and near zero "
            "near empty: position += dt * speed * (cos heading, sin heading) (the heading and "
            "speed before the tick), heading += dt * turn_max * g * turn (in place too), speed = "
            "keep * speed + gain * g * drive (the speed moves toward top_speed * g * drive; the "
            "rover drives forward only), the level by the energy rule, with noise of sd "
            "state_sd; then the position is clipped to the field and a rover that hits a wall "
            "stops (the model does not know the walls; it knows the stall). The top speed is "
            "gain / (1 - keep)"
        ),
    )


def plan_path(tick, policy):
    """The positions a rover plan passes through, as the page draws it: from the plan's start,
    the believed heading, speed and energy level, rolled forward by the vehicle's rule without
    noise (``survey_world.VehicleStep``, ``dynamics_record``), one position per plan step after
    the start."""
    b = tick["belief"]
    x = np.asarray(
        [
            *policy["start"],
            np.cos(b["heading"]),
            np.sin(b["heading"]),
            b["speed"],
            b.get("energy_level", wd.ENERGY_MAX),
        ],
        np.float32,
    )
    return np.asarray(_roll(jnp.asarray(x), jnp.asarray(policy["v_plan"], jnp.float32)))


@jax.jit
def _roll(x, plan):
    """The positions the vehicle's noiseless rule passes through from ``x`` under ``plan``."""

    def tick(state, a):
        state = wd.VehicleStep()(state, (a,))
        return state, state[:2]

    return jax.lax.scan(tick, x, plan)[1]


def excursion(points):
    """How far points ``(..., 2)`` lie beyond the field's walls (zero inside)."""
    (x0, x1), (y0, y1) = wd.FIELD
    p = np.asarray(points, float)
    gaps = np.stack([x0 - p[..., 0], p[..., 0] - x1, y0 - p[..., 1], p[..., 1] - y1], -1)
    return np.maximum(gaps.max(-1), 0.0)


def plan_excursions(rows, chosen_only=False):
    """Every plan's largest excursion beyond the field over a run (the chosen plan's alone with
    ``chosen_only``)."""
    return np.asarray(
        [
            float(excursion(plan_path(t, p)).max())
            for t in rows
            for i, p in enumerate(t["policies"])
            if not chosen_only or i == t["chosen"]
        ]
    )


LOST_ERROR = 0.375  # a position belief whose mean is a map cell or more from the truth ...
LOST_TICKS = 10  # ... for more than this many ticks running is lost


def position_calibration(rows, lost_error=LOST_ERROR, lost_ticks=LOST_TICKS):
    """How the rover's position belief tracks the truth over a run: the largest and median error
    of its mean, the share of ticks whose true position lies inside the belief's 2-sigma ellipse
    (of the mixture's moment-matched Gaussian; ``1 - exp(-2)``, about 0.865, for a calibrated
    belief), the share of ticks whose true heading lies within 2 sd of the believed heading (about
    0.954 for a calibrated belief), and the spells of more than ``lost_ticks`` ticks with an error
    of ``lost_error`` or more (the run is lost if there is one)."""
    errors, inside, heading_in = [], [], []
    for r in rows:
        if "heading_sd" in r["belief"] and "true_heading" in r:
            gap = (r["true_heading"] - r["belief"]["heading"] + np.pi) % (2 * np.pi) - np.pi
            heading_in.append(abs(gap) <= 2 * r["belief"]["heading_sd"])
        comps = r["belief"]["components"]
        w = np.asarray([c["w"] for c in comps])
        w = w / w.sum()
        mu = np.asarray([c["mu"] for c in comps])
        mean = w @ mu
        cov = sum(
            wk * (np.asarray(c["cov"]) + np.outer(m - mean, m - mean))
            for wk, c, m in zip(w, comps, mu, strict=True)
        )
        d = np.asarray(r["true_x"][:2]) - mean
        errors.append(float(np.hypot(*d)))
        inside.append(float(d @ np.linalg.solve(cov, d)) <= 4.0)
    spells, run = [], 0
    for e in errors + [0.0]:
        if e >= lost_error:
            run += 1
        else:
            if run > lost_ticks:
                spells.append(run)
            run = 0
    return dict(
        max_error=max(errors),
        median_error=float(np.median(errors)),
        ellipse_coverage=float(np.mean(inside)),
        heading_coverage=float(np.mean(heading_in)) if heading_in else None,
        lost_spells=spells,
        lost=bool(spells),
    )


def motion_stats(rows, at_speed=0.1):
    """The true motion: the per-tick displacement, the speed, and the change of the rover's
    heading per tick (degrees; bounded by ``TURN_MAX`` plus noise), the median over the ticks
    where the speed is at least ``at_speed``."""
    xs = np.asarray([r["true_x"] for r in rows], float)
    step = np.hypot(*np.diff(xs[:, :2], axis=0).T)
    speed = np.hypot(xs[:, 2], xs[:, 3])
    angle = np.asarray([r["true_heading"] for r in rows], float)
    turn = np.degrees(np.abs((np.diff(angle) + np.pi) % (2 * np.pi) - np.pi))
    moving = (speed[:-1] >= at_speed) & (speed[1:] >= at_speed)
    return dict(
        step_median=float(np.median(step)),
        step_max=float(step.max()),
        speed_median=float(np.median(speed)),
        speed_max=float(speed.max()),
        turn_median=float(np.median(turn[moving])) if moving.any() else None,
        turn_max=float(turn.max()),
        moving_ticks=int(moving.sum()),
    )


def meta_record(ticks, camera_report, grid, camera="image"):
    return dict(
        library_version=__version__,
        ticks=ticks,
        grid=dict(shape=list(grid.shape), low=list(grid.low), high=list(grid.high)),
        field=[list(wd.FIELD[0]), list(wd.FIELD[1])],
        start=list(wd.START),
        world=downsampled_maps(),
        dynamics=dynamics_record(),
        names=NAMES,
        image=dict(
            side=wd.IMG_SIDE,
            layout=(
                "rows of uint8 0..255, the camera: row 0 is the top of the picture, "
                "columns run left to right"
            ),
            view=(
                "the scene ahead of the rover along its heading (the camera is mounted forward), "
                "a 70 degree field of view: the ground ray-marched over the height map out to 5 "
                "units, lit by the sun and textured in world coordinates with the class under "
                "the rover, the dark ground beyond the field's walls, fading into haze under a "
                "sky; the ground scrolls as the rover moves and swings as it turns"
            ),
            camera_mode=camera,
            heading_rule=(
                "the rover's heading (true_heading per tick, radians from +x) is part of its "
                "state: it turns only by the turn command, at most turn_max_degrees a tick, in "
                "place too, and holds otherwise; the model believes it too (belief.heading and "
                "belief.heading_sd)"
            ),
            field_of_view_degrees=70.0,
            tempering=dict(
                alpha=(camera_report.get("tempering") or {}).get("alpha"),
                note=(
                    "the image's evidence is tempered for the decoder's own error: the declared "
                    "noise sd is the calibrated sd over sqrt(alpha), alpha = 2 / (2 + E[m^2]), m^2 "
                    "the shift the decoder's error alone causes in a single image's position "
                    "estimate in units of its covariance, measured on held-out states; a stand-in "
                    "for a structured decoder-error term"
                ),
            ),
            camera_report=camera_report,
        ),
        map_layout=(
            "map: one row per grid cell (row-major over grid.shape, x first), three uint8 "
            "probabilities (firm, rough, ice) of the field's posterior mean, scaled to 0..255; "
            "counts: the observations absorbed per cell"
        ),
        sign_convention=(
            "every leaf is added to G with its registered sign already applied, so the leaves of "
            "a policy sum to its G and an epistemic leaf appears negative"
        ),
        wall_preference=dict(
            offset=wall_offset(sm.POLICY_LEN),
            note=(
                "the rover's utility leaf on the fix (the wall preference) and each rover plan's "
                "G are recorded less offset, the preference's value over a plan that stays "
                "inside the field (a constant across plans: the mixture's normaliser), so the "
                "leaf's lower bound is near zero inside the field and the leaf grows as a plan "
                "nears or crosses a wall"
            ),
        ),
        position_risk=dict(
            note=(
                "the rover's risk leaf on its position (the sector target, two-level only) and "
                "each rover plan's G are also recorded less a constant per tick: the smallest "
                "value of that leaf over the tick's plans (the tick's offsets.position_risk), so "
                "the leaf shows each plan's risk "
                "above the best one's; the constant is the same for every plan of a tick, so the "
                "ranking is untouched"
            ),
        ),
    )


def first_tick(rows, inside):
    """The first tick whose true position satisfies ``inside(x, y)``, or None."""
    return next((r["t"] for r in rows if inside(*r["true_x"][:2])), None)


def run_summary(run_, marks=MARKS):
    """The figures REPORT.md gives per setting; coverage and ``seen`` at the ticks ``marks``."""
    rows = run_["ticks"]
    n = len(rows)
    err = [
        float(np.hypot(*(np.asarray(r["belief"]["mean"]) - np.asarray(r["true_x"][:2]))))
        for r in rows
    ]
    bands = np.bincount([r["true_energy"]["band"] for r in rows], minlength=sm.N_BANDS)
    terrain = np.bincount([r["true_terrain"] for r in rows], minlength=3)
    secs = [r["seconds"] for r in rows[1:]] or [rows[0]["seconds"]]
    coverage = {f"t{k}": rows[min(k, n) - 1]["coverage"] for k in marks}
    (fx, fy), fr = wd.ROUGH_PATCHES[0]
    (ix, iy), ir = wd.ICE_PATCH
    return dict(
        coverage=coverage,
        seen={f"t{k}": rows[min(k, n) - 1]["seen"] for k in marks},
        first_ice=first_tick(rows, lambda x, y: np.hypot(x - ix, y - iy) <= ir),
        first_far_rough=first_tick(rows, lambda x, y: np.hypot(x - fx, y - fy) <= fr),
        first_in_sector=[
            first_tick(rows, lambda x, y, k=k: int(wd.zone_of(np.asarray([x, y]))) == k)
            for k in range(sm.N_SURVEY)
        ],
        position_error=float(np.mean(err)),
        motion=motion_stats(rows),
        calibration=position_calibration(rows),
        plan_excursion_max=float(plan_excursions(rows).max()),
        plan_excursion_chosen_max=float(plan_excursions(rows, chosen_only=True).max()),
        energy_floor=float(min(r["true_energy"]["level"] for r in rows)),
        ticks_per_band=[int(b) for b in bands],
        ticks_on_ice=int(terrain[2]),
        ticks_on_rough=int(terrain[1]),
        final_band=int(rows[-1]["true_energy"]["band"]),
        mean_seconds=float(np.mean(secs)),
        first_velocities=[[round(c, 3) for c in r["v"]] for r in rows[:10]],
    )


def report_text(settings, runs, marks=MARKS, mdl=None):
    lines = ["# Survey rover runs", ""]
    summaries = {s["id"]: run_summary(runs[s["id"]], marks) for s in settings}
    if len(settings) > 1 and all("tiers" in s for s in settings):
        head = "".join(f" tick {k} |" for k in marks)
        lines += [
            "Coverage over time (cells whose field's top probability exceeds "
            f"{COVERED_MAX}), side by side:",
            "",
            f"| setting |{head}",
            "|---|" + "---|" * len(marks),
        ]
        for s in settings:
            cells = "".join(f" {summaries[s['id']]['coverage'][f't{k}']:.3f} |" for k in marks)
            lines.append(f"| {s['id']} |{cells}")
        lines.append("")
    if any("tiers" in s for s in settings):
        lines += [
            "Two-tier runs are the merged agent coupled to the world; the scheduler's cell reading "
            "is the rover's posterior of the round before and is absent in round 0.",
            "",
        ]
    for s in settings:
        r = summaries[s["id"]]
        cov = ", ".join(f"{k[1:]}: {v:.3f}" for k, v in r["coverage"].items())
        seen = ", ".join(f"{k[1:]}: {v:.3f}" for k, v in r["seen"].items())
        bands = ", ".join(
            f"{n} {c}" for n, c in zip(sm.ENERGY_NAMES, r["ticks_per_band"], strict=True)
        )
        lines += [
            f"## {s['id']}: {s['label']}",
            "",
            f"- seen (cells with at least {SEEN_COUNTS} counts) at tick {seen}",
            f"- coverage (cells whose field's top probability exceeds {COVERED_MAX}) at tick {cov}",
            f"- mean position error: {r['position_error']:.3f}; largest "
            f"{r['calibration']['max_error']:.3f}; true position inside the belief's 2-sigma "
            f"ellipse {r['calibration']['ellipse_coverage']:.3f} of ticks (0.865 nominal); lost "
            f"spells (error {LOST_ERROR} or more for over {LOST_TICKS} ticks): "
            f"{r['calibration']['lost_spells'] or 'none'}",
            f"- motion: per-tick displacement median {r['motion']['step_median']:.3f}, max "
            f"{r['motion']['step_max']:.3f}; top speed {r['motion']['speed_max']:.3f}; change of "
            f"direction per tick at speed 0.1 or more, median {r['motion']['turn_median']} and "
            f"max {r['motion']['turn_max']} degrees over {r['motion']['moving_ticks']} ticks",
            f"- plans beyond the field's walls: the largest excursion of any plan "
            f"{r['plan_excursion_max']:.3f}, of a chosen plan {r['plan_excursion_chosen_max']:.3f}",
            f"- energy floor (lowest true level): {r['energy_floor']:.3f}",
            f"- ticks per true energy band: {bands}",
            f"- ticks on ice: {r['ticks_on_ice']}, on rough ground: {r['ticks_on_rough']}",
            f"- first tick on ice: {r['first_ice']}, "
            f"in the far rough patch: {r['first_far_rough']}",
            f"- final energy band: {sm.ENERGY_NAMES[r['final_band']]}",
            f"- mean seconds per tick (the compiled rounds, compile excluded): "
            f"{r['mean_seconds']:.2f}",
            f"- first ten chosen velocities: {r['first_velocities']}",
        ]
        slow = [t["slow"] for t in runs[s["id"]]["ticks"] if t.get("slow", {}).get("decision")]
        if slow:
            names = ss.OPTION_NAMES
            decisions = [
                (t["t"], names[t["slow"]["option"]])
                for t in runs[s["id"]]["ticks"]
                if t.get("slow", {}).get("decision")
            ]
            charges = [t for t, o in decisions if o == "charge"]
            first_charge = charges[0] if charges else None
            options = [d["option"] for d in slow]
            surveyed = options.index(0) if 0 in options else None
            after = (
                slow[surveyed + 1] if surveyed is not None and surveyed + 1 < len(slow) else None
            )
            lines += [
                f"- first charge tick: {first_charge}",
                f"- first tick in each sector (1 to 4): {r['first_in_sector']}",
                f"- EFE by option at the first decision: "
                f"{np.round(slow[0]['efe_by_option'], 2).tolist()}",
                f"- EFE by option at the decision after sector 1 is surveyed: "
                f"{np.round(after['efe_by_option'], 2).tolist() if after else 'no such decision'}",
                f"- options chosen (tick: option): {', '.join(f'{t}: {o}' for t, o in decisions)}",
                f"- charge chosen at ticks: {charges if charges else 'never'}",
                f"- mean seconds per tick at a slow decision (compile excluded): "
                f"{np.mean([d['seconds'] for d in slow]):.2f}",
            ]
        lines.append("")
    if mdl is not None:
        parts = {c["name"]: c["nats"] for c in mdl["components"]}
        lines += [
            "## Description length",
            "",
            f"- the composite: {mdl['total']:.0f} nats ({mdl['total_exact']:.0f} exact)",
            f"- the scheduler's known visit table alone (transition:mission.visit): "
            f"{parts.get('transition:mission.visit')} nats",
            f"- the scheduler's learned energy table (parameter:mission.energy_table): "
            f"{parts.get('parameter:mission.energy_table')} nats",
            f"- the cell reading (likelihood:mission.cell_obs): "
            f"{parts.get('likelihood:mission.cell_obs')} nats",
            f"- the shared field (parameter:rover.field): "
            f"{parts.get('parameter:rover.field')} nats",
            "",
        ]
    return "\n".join(lines)


def graph_record(system, params):
    """The model's graph for the page: tiers (a composite's members, by name prefix; one tier
    for a single System) of factors, modalities and controls, the transition, likelihood and
    target edges, and the parameter variables with the factors that read them."""
    names = system.names
    all_nodes = (*system.node_names, *system.action_names)
    tiers = {}

    def tier(n):
        name = n.split(".")[0] if "." in n else "rover"
        return tiers.setdefault(name, dict(name=name, factors=[], modalities=[], controls=[]))

    cells = dict(system.cells)
    for f in names.states:
        d = params.initial_state_prior[f]
        if hasattr(d, "logits"):
            rec = dict(name=f, family="categorical", levels=int(d.logits.shape[-1]))
        else:
            rec = dict(name=f, family="gaussian_mixture", dim=int(d.means.shape[-1]))
        if f in cells:
            rec["cells"] = dict(shape=list(cells[f].shape), axes=list(cells[f].axes))
        tier(f)["factors"].append(rec)
    families = {
        "LinearGaussianObservation": "linear_gaussian",
        "MLPConditional": "neural",
        "ConditionalCategorical": "categorical",
    }
    for m, parents in zip(names.modalities, system.A_deps, strict=True):
        lik = type(params.likelihood[m]).__name__
        if lik not in families:
            raise ValueError(f"no graph family for the likelihood of {m!r}: {lik}")
        tier(m)["modalities"].append(
            dict(name=m, family=families[lik], parents=[all_nodes[p] for p in parents])
        )
    levels = system._control_levels(params)
    for node, n_levels in zip(names.controls, levels, strict=True):
        if node in params.control:
            box = params.control[node]
            rec = dict(
                name=node,
                kind="float",
                box=[[float(b) for b in box.low], [float(b) for b in box.high]],
                durations=None,
            )
        else:
            rec = dict(name=node, kind="level", levels=int(n_levels), durations=None)
        tier(node)["controls"].append(rec)
    edges = []
    for child, parents in zip(system.node_names, system.B_deps, strict=True):
        if child in names.controls:
            continue
        for p in parents:
            if all_nodes[p] != child:
                edges.append({"from": all_nodes[p], "to": child, "kind": "transition"})
    for m, parents in zip(names.modalities, system.A_deps, strict=True):
        for p in parents:
            edges.append({"from": all_nodes[p], "to": m, "kind": "likelihood"})
    for key, parents in system.C_deps:
        for p in parents:
            edges.append({"from": p, "to": key, "kind": "target"})
    variables = []
    for home in params.parameters:
        readers = [
            f
            for group in (params.transition, params.likelihood)
            for f, decl in group.items()
            if isinstance(decl, Parameter) and decl.name == home
        ]
        variables.append(dict(name=home, readers=readers))
        edges += [{"from": home, "to": f, "kind": "parameter"} for f in readers]
    return dict(tiers=list(tiers.values()), edges=edges, parameters=variables)


def _entry_size(params, kind, name):
    table = {
        "likelihood": params.likelihood,
        "transition": params.transition,
        "initial_state_prior": params.initial_state_prior,
        "parameter": params.parameters,
    }.get(kind)
    if table is None or name not in table:
        return None
    leaves = jax.tree_util.tree_leaves(eqx.filter(table[name], eqx.is_array))
    return int(sum(int(np.prod(leaf.shape)) for leaf in leaves))


def mdl_record(system, params):
    """The description-length table: nats per address of the model's description."""
    document, arrays = describe(system, params)
    dl = description_length(document, MDL_RESOLUTION, arrays=arrays)
    unscored = {a for a, link in dl.unscored if link == "family"}
    rows = []
    for address, links in dl.per_address.items():
        kind, _, name = address.partition(":")
        nats = None if address in unscored else float(links.structure + links.parameters)
        rows.append(
            dict(
                name=address,
                kind=kind,
                nats=nats,
                exact=bool(links.exact),
                entries=_entry_size(params, kind, name),
            )
        )
    scored = [r for r in rows if r["nats"] is not None]
    return dict(
        total=float(sum(r["nats"] for r in scored)),
        total_exact=float(sum(r["nats"] for r in scored if r["exact"])),
        library_total=float(dl.total),
        components=rows,
    )


def assemble(out_dir, camera_report, decoder):
    """Merge the part files into the bundle and write the report. The bundle's model records
    (``graph``, ``mdl``, ``description``) are of the model the page shows: the agent's joint
    planner (the scheduler and the rover planned as one model), not the coupled System that
    ``run`` steps, which adds the world, a black box declared by its ports alone (its description
    length unscored)."""
    runs, settings = {}, {}
    for path in sorted(out_dir.glob("part_*.json")):
        part = json.loads(path.read_text())
        runs.update(part["runs"])
        settings.update(part["settings"])
    if not runs:
        raise SystemExit(f"no part files under {out_dir}")
    ticks = {len(r["ticks"]) for r in runs.values()}
    if len(ticks) != 1:
        raise SystemExit(f"the runs have different lengths: {sorted(ticks)}")
    order = [s["id"] for s in settings_grid_l2() if s["id"] in runs]
    settings = [settings[sid] for sid in order]
    runs = {sid: runs[sid] for sid in order}
    grid, _ = sm.field_grid(settings[0].get("grid", sm.GRID_SHAPE))
    #  the page's graph, description length and document are of the agent's joint planner (each
    #  tier's option read as one decision), not of the coupled System that run steps: the page
    #  shows the agent's model, and the world is a black box, its ports alone
    confusion = camera_report.get("recogniser_confusion")
    model = joint_planner(*agent_system(decoder, settings[0], confusion)[0])
    n = ticks.pop()
    meta = meta_record(n, camera_report, grid, camera_mode(settings[0]))
    seeds = {s.get("seed", 0) for s in settings}
    meta["seed"] = seeds.pop() if len(seeds) == 1 else None  # the run key the settings share
    tiered = next((s for s in settings if s.get("tiers") == 2), settings[0])
    meta["tiers"] = dict(
        slow_ticks=settings[0]["slow_ticks"],
        options=ss.OPTION_NAMES,
        zones=ss.SECTOR_NAMES,
        zone_split=list(wd.ZONE_SPLIT),
        sector_centres=[[float(c) for c in p] for p in sm.sector_centres()],
        charge_site=[float(c) for c in sm.charge_site()],
        survey_readings=ss.SURVEY_READINGS,
        charge_readings=ss.CHARGE_READINGS,
        horizon=tiered.get("horizon", ss.HORIZON),
        terminal_penalty=tiered.get("terminal_penalty"),
        energy_model=(
            "the scheduler's energy transition is the parameter variable energy_table, one "
            "Dirichlet per band and option over the next band, learned at each decision from its "
            "band readings (the band posterior against the decision before's, under the option "
            "taken then); its prior: three pseudo-counts per case, the band probabilities of "
            "the level shift the rover's declared energy rule gives over one option (a survey: "
            "cruise speed on flat ground at the median solar yield; charge: parked on the "
            "charging place's yield), the level uniform within its band; from the empty band, "
            "under every option, half each on the same and one higher; 0.001 on every entry; "
            "no information-gain term on it"
        ),
        terminal=(
            "the scheduler's energy term is a terminal target, scored once at the plan's end: "
            "equal over every band but the empty one, the empty band lower by "
            "terminal_penalty, twice the most novelty a plan of the horizon could earn at the "
            "start of a mission (the horizon, times the most readings an option pools, times the "
            "field's novelty weight under its prior)"
        ),
        cell_reading=(
            "the scheduler's cell reading is right with probability cell_acc, else another cell "
            "in proportion to a Gaussian of sd cell_error_sd (one cell side) in the distance "
            "between the cells' centres"
        ),
        cell_acc=float(ss.CELL_ACC),
        cell_error_sd=float(ss.CELL_ERROR_SD),
        measured=dict(
            cruise_speed=float(ss.CRUISE),
            survey_spread=float(ss.SURVEY_SPREAD),
            charge_speed=float(ss.CHARGE_SPEED),
            survey_readings=ss.SURVEY_READINGS,
            charge_readings=ss.CHARGE_READINGS,
            note=(
                "the constants of the scheduler's visit table and pooled readings measured on "
                "earlier runs of the two-tier agent; its energy table holds none"
            ),
        ),
        one_level=(
            "the one-level setting is the rover alone: the fast tier with its own objective "
            "(the field's novelty, the fix's information gain, the energy preference and the "
            "wall preference), no scheduler, no option and no sector or charging target, free "
            "to go where it likes; its ticks have no option and no slow record"
        ),
    )
    bundle = dict(meta=meta, graph=graph_record(*model), settings=settings, runs=runs)
    try:
        bundle["mdl"] = mdl_record(*model)
        bundle["description"] = describe(*model)[0]
    except Exception as exc:  # the description refuses some models; say so and carry on
        print(f"mdl skipped: {type(exc).__name__}: {exc}")
    text = json.dumps(bundle)
    (out_dir / "bundle.json").write_text(text)
    marks = tuple(range(100, n + 1, 100)) or (n,)
    (out_dir / "REPORT.md").write_text(report_text(settings, runs, marks, bundle.get("mdl")))
    print(f"bundle: {len(runs)} runs, {len(text) / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--out", required=True, help="output directory (also holds the camera cache)")
    ap.add_argument("--settings", default=None, help="comma-separated setting ids")
    ap.add_argument("--assemble", action="store_true", help="merge the part files and stop")
    ap.add_argument("--steps", type=int, default=PLANNER_STEPS, help="descent steps per plan")
    ap.add_argument("--starts", type=int, default=PLANNER_STARTS, help="warm starts per plan")
    ap.add_argument(
        "--level", type=int, default=2, choices=(2,), help="the two-tier settings (the one grid)"
    )
    ap.add_argument("--seed", type=int, default=None, help="the world's key (default: 0)")
    ap.add_argument(
        "--camera", default=CAMERA, choices=("image", "look", "none"), help="camera_mode"
    )
    ap.add_argument(
        "--fix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="observe the position fix (--no-fix: the fix ablation, vision-only localisation)",
    )
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    decoder, encoder, camera_report = cam.load_or_train(out_dir)
    confusion = camera_report.get("recogniser_confusion")
    if args.assemble:
        assemble(out_dir, camera_report, decoder)
        return
    settings = settings_grid_l2()
    if args.settings:
        keep = set(args.settings.split(","))
        settings = [s for s in settings if s["id"] in keep]
    for s in settings:
        if args.seed is not None:
            s["seed"] = args.seed
        t0 = time.time()
        s["camera"] = args.camera
        if not args.fix:
            s["fix"] = False
        result = run_setting(
            s, args.ticks, decoder, args.steps, args.starts, encoder=encoder, confusion=confusion
        )
        elapsed = time.time() - t0
        result["seconds"] = elapsed
        part = out_dir / f"part_{s['id']}{args.tag}.json"
        part.write_text(json.dumps(dict(runs={s["id"]: result}, settings={s["id"]: s})))
        print(
            f"{s['id']}: {elapsed:.1f}s ({elapsed / args.ticks:.2f}s/tick; the compiled "
            f"rounds {result['ticks'][0]['seconds']:.2f}s/tick) final coverage "
            f"{result['ticks'][-1]['coverage']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
