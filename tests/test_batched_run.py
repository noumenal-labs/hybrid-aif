"""``run(..., batch=B)``: replicas of a closed System stepped by one program, and groups.

Design: docs/design/batched-run.md R9-R13. The pins, in order: a replica of a batch is the
unbatched run under ``fold_in(key, b)`` array for array (simultaneous and sequential bitwise;
async to float32 rounding, its order a vmapped switch), and identical keys give identical
replicas (R10, R13 pin 2); the record's replica axis, ``record=``, ``chunk=`` and passes of
``replicas_per_pass`` (R11), which change no number; per-replica seeds and params; a group steps
copies by one vmapped call and equals the ungrouped simultaneous run (R12, pin 3), on two leaves
sharing a Dirichlet variable and on two survey rovers sharing their field; replicas holding
different options and a non-positive Dirichlet sum recorded as a fault per replica (pin 5); the
async ceiling (R9) and the refusals, each a sentence naming the fix; the corridor's coupled
minds under async with ``batch=4`` (pin 7's async case).
"""

import dataclasses
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.composition import Identify, couple, merge, plan, run, sample
from hybrid_aif.envs.generative import broadcast_params
from hybrid_aif.system.mirror import mirror
from tests.test_action_nodes import pusher
from tests.test_clocks import robots, steps_of
from tests.test_run import COUNTS, KEY, field_step, shared_field, two_node_leaf, wire

ORDERS = ("simultaneous", "sequential", "async")


def record_of(traj, carries=True):
    """The arrays of a record, in one flat list: every field stacked over rounds, the final
    carry and the faults."""
    fields = [traj.received, traj.present, traj.emitted, traj.intended, traj.decided]
    if carries:
        fields += [traj.carries, traj.carry]
    return jax.tree_util.tree_leaves_with_path((fields, traj.faults))


def assert_replica(unbatched, batched, b, exact=True, carries=True):
    """Replica ``b`` of ``batched`` equals ``unbatched``, leaf for leaf: bitwise, or to float32
    rounding (``exact=False``) for floats, integers and flags bitwise."""
    got = record_of(batched, carries)
    want = record_of(unbatched, carries)
    assert [p for p, _ in got] == [p for p, _ in want]
    for (path, x), (_, y) in zip(want, got, strict=True):
        x, y = np.asarray(x), np.asarray(y)[b]
        name = jax.tree_util.keystr(path)
        assert x.shape == y.shape and x.dtype == y.dtype, name
        if exact or x.dtype.kind != "f":
            np.testing.assert_array_equal(x, y, err_msg=name)
        else:
            np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-6, err_msg=name)


def pushers():
    return {"A": plan(selection="stochastic"), "B": plan(selection="stochastic"), "P": sample()}


# -------------------------------------------------------------------- replicas (R10, pin 2)


@pytest.mark.parametrize("order", ORDERS)
def test_a_replica_is_the_unbatched_run_under_its_folded_key(order):
    """Replica ``b`` of ``batch=3`` equals the run under ``fold_in(key, b)``: every record,
    every carry, the final carry and the faults; bitwise under simultaneous and sequential, to
    float32 rounding under async (each replica's order is a switch over the derived orders,
    which vmap lowers to a select, and XLA fuses the batched program differently)."""
    system, params = wire()
    batched = run(system, params, 6, KEY, pushers(), tick_order=order, batch=3, record_carries=True)
    for b in range(3):
        alone = run(
            system,
            params,
            6,
            jax.random.fold_in(KEY, b),
            pushers(),
            tick_order=order,
            record_carries=True,
        )
        assert_replica(alone, batched, b, exact=order != "async")
    assert batched.params is params  # nothing wrote back: the home handed in


def test_identical_keys_give_identical_replicas_and_folded_keys_differ():
    """``key`` may be a stack of one key per replica: three copies of one key give three
    identical trajectories, where one key folded per replica gives three different ones."""
    system, params = wire()
    stacked = jnp.stack([KEY] * 3)
    same = run(system, params, 8, stacked, pushers(), batch=3, record_carries=True)
    for path, x in record_of(same):
        x = np.asarray(x)
        for b in (1, 2):
            np.testing.assert_array_equal(x[b], x[0], err_msg=jax.tree_util.keystr(path))
    folded = run(system, params, 8, KEY, pushers(), batch=3)
    moves = np.asarray(folded["A"].emitted["push"])
    assert not (np.array_equal(moves[0], moves[1]) and np.array_equal(moves[0], moves[2]))


def test_the_record_gains_a_leading_replica_axis_and_batch_one_keeps_it():
    """Under ``batch=B`` every array of the record leads with ``B`` (``decided``, ``present``
    and the faults included; the carries as a tuple over rounds of carries whose arrays do);
    ``batch=None`` keeps the unbatched form and ``batch=1`` has an axis of length one."""
    system, params = wire()
    alone = run(system, params, 5, KEY, pushers(), record_carries=True)
    one = run(system, params, 5, KEY, pushers(), batch=1, record_carries=True)
    four = run(system, params, 5, KEY, pushers(), batch=4, record_carries=True)
    for (path, x), (_, y), (_, z) in zip(
        record_of(alone), record_of(one), record_of(four), strict=True
    ):
        name = jax.tree_util.keystr(path)
        assert jnp.shape(y) == (1, *jnp.shape(x)), name
        assert jnp.shape(z) == (4, *jnp.shape(x)), name
    assert four["A"].decided.shape == (4, 5) and four["A"].present["see"].shape == (4, 6)
    assert len(four["A"].carries) == 6
    assert four.timing["passes"] == 1 and four.timing["replicas_per_pass"] == 4


def test_passes_and_chunks_change_no_number():
    """``replicas_per_pass`` splits the replicas into passes of one program (the last pass
    filled with copies, then dropped) and ``chunk`` the scan into compiled chunks re-run by a
    host loop (a last chunk padded past the run's end); neither changes a number, batched or
    not."""
    system, params = robots([4.0, 2.0, 4.0])
    whole = run(system, params, 13, KEY, steps_of(plan(selection="stochastic")), batch=5)
    for kwargs in (
        dict(replicas_per_pass=2),
        dict(chunk=5),
        dict(chunk=4, replicas_per_pass=3),
        dict(chunk=50),
    ):
        split = run(
            system, params, 13, KEY, steps_of(plan(selection="stochastic")), batch=5, **kwargs
        )
        for (path, x), (_, y) in zip(record_of(whole, False), record_of(split, False), strict=True):
            np.testing.assert_array_equal(x, y, err_msg=f"{kwargs} {jax.tree_util.keystr(path)}")
        per_pass = kwargs.get("replicas_per_pass", 5)
        assert split.timing["passes"] == -(-5 // per_pass)
        chunks = split.timing["chunks"]
        assert sum(c["rounds"] for c in chunks if c["pass"] == 0) == split.timing["scanned"] == 12
    alone = run(system, params, 13, KEY, steps_of(plan(selection="stochastic")))
    chunked = run(system, params, 13, KEY, steps_of(plan(selection="stochastic")), chunk=5)
    for (path, x), (_, y) in zip(record_of(alone, False), record_of(chunked, False), strict=True):
        np.testing.assert_array_equal(x, y, err_msg=jax.tree_util.keystr(path))
    assert [c["rounds"] for c in chunked.timing["chunks"]] == [5, 5, 2]


def test_record_names_the_fields_stacked_over_rounds():
    """``record=`` keeps the fields it names (``decided`` and the faults always); the others are
    ``None`` at every node of the tree."""
    system, params = wire()
    traj = run(system, params, 4, KEY, pushers(), batch=2, record=("emitted",))
    assert traj["A"].emitted["push"].shape == (2, 4)
    assert traj["A"].received is None and traj["A"].present is None
    assert traj["A"].intended is None and traj.received is None and traj["A"].carries is None
    assert traj["A"].decided.shape == (2, 4)
    traj = run(system, params, 4, KEY, pushers(), record=("intended",), record_carries=True)
    assert traj["A"].emitted is None and len(traj["A"].carries) == 5
    with pytest.raises(ValueError, match=r"record names \['beliefs'\].*decided and faults"):
        run(system, params, 4, KEY, pushers(), record=("beliefs",))


def test_seeds_and_params_per_replica_are_named_in_per_replica():
    """``per_replica`` names the inputs that carry a leading replica axis: a node's seed in
    ``carries`` (here the process's point state, one start per replica) and ``"params"``
    (``broadcast_params`` then a leaf replaced); the rest is shared. Each replica is the
    unbatched run with its own seed and params."""
    system, params = wire()
    starts = {"P.x": jnp.asarray([0, 1, 2], jnp.int32)}
    batched = run(
        system, params, 5, KEY, pushers(), batch=3, carries={"P": starts}, per_replica=["P"]
    )
    for b in range(3):
        alone = run(
            system,
            params,
            5,
            jax.random.fold_in(KEY, b),
            pushers(),
            carries={"P": {"P.x": starts["P.x"][b]}},
        )
        assert_replica(alone, batched, b, carries=False)
    noise = jnp.stack([params.transition["P.x"].log_table * s for s in (1.0, 0.5, 2.0)])
    swapped = eqx.tree_at(
        lambda p: p.transition["P.x"].log_table, broadcast_params(params, 3), noise
    )
    batched = run(system, swapped, 5, KEY, pushers(), batch=3, per_replica=["params"])
    for b in range(3):
        mine = eqx.tree_at(lambda p: p.transition["P.x"].log_table, params, noise[b])
        alone = run(system, mine, 5, jax.random.fold_in(KEY, b), pushers())
        assert_replica(alone, batched, b, carries=False)


def test_the_batch_arguments_are_refused_naming_the_fix():
    system, params = wire()
    with pytest.raises(ValueError, match=r"batch=0 is not a positive integer"):
        run(system, params, 2, KEY, pushers(), batch=0)
    with pytest.raises(ValueError, match=r"chunk=-1 is not a positive integer"):
        run(system, params, 2, KEY, pushers(), chunk=-1)
    with pytest.raises(ValueError, match=r"replicas_per_pass= .* batch is None; pass batch=B"):
        run(system, params, 2, KEY, pushers(), replicas_per_pass=2)
    with pytest.raises(ValueError, match=r"no leading axis of batch=2 entries.*broadcast_params"):
        run(system, params, 2, KEY, pushers(), batch=2, per_replica=["params"])
    with pytest.raises(ValueError, match=r"per_replica names \['Q'\]"):
        run(system, params, 2, KEY, pushers(), batch=3, per_replica=["Q"])
    with pytest.raises(ValueError, match=r"a stack of 3 keys"):
        run(system, params, 2, jnp.stack([KEY] * 2), pushers(), batch=3)
    clocked, cparams = robots([4.0, 2.0, 4.0])
    bad = broadcast_params(cparams, 2)
    bad = eqx.tree_at(
        lambda p: p.parts["a"].durations["a.u_move"],
        bad,
        jnp.asarray([[4.0, 2.0, 4.0], [1.0, 1.0, 1.0]]),
    )
    with pytest.raises(ValueError, match=r"params.durations differs across the replicas"):
        run(clocked, bad, 2, KEY, steps_of(plan()), batch=2, per_replica=["params"])


def test_a_step_calling_host_code_at_its_reset_is_refused_under_batch_naming_it():
    """Under ``batch=`` every call is traced, the reset included: a step that reads its key on
    the host is refused naming it (unbatched, its reset runs on concrete values)."""
    system, params = wire()
    inner = plan()

    def host(system, key, obs, carry, params):
        if obs is None:
            np.asarray(key)
        return inner(system, key, obs, carry, params)

    host.sliced = True
    steps = {**pushers(), "A": host}
    run(system, params, 3, KEY, steps)
    with pytest.raises(ValueError, match=r"the step of 'A' calls host-side code .*under batch="):
        run(system, params, 3, KEY, steps, batch=2)


# --------------------------------------------------------------------- holds and faults (pin 5)


@pytest.mark.parametrize("order", ["simultaneous", "async"])
def test_replicas_hold_different_options_each_as_its_unbatched_run(order):
    """Robot ``a`` declares durations of 4, 2 and 4 rounds by level and chooses stochastically:
    the replicas decide on different rounds (its decide-or-hold a ``cond`` on a per-replica
    count, a select under vmap) and each is its unbatched run, decisions and holds bitwise."""
    system, params = robots([4.0, 2.0, 4.0])
    batched = run(
        system, params, 20, KEY, steps_of(plan(selection="stochastic")), tick_order=order, batch=4
    )
    decided = np.asarray(batched["a"].decided)
    assert decided.shape == (4, 20) and decided[:, 0].all()
    assert len({tuple(row) for row in decided.tolist()}) > 1  # the replicas hold differently
    for b in range(4):
        alone = run(
            system,
            params,
            20,
            jax.random.fold_in(KEY, b),
            steps_of(plan(selection="stochastic")),
            tick_order=order,
        )
        assert_replica(alone, batched, b, carries=False)


def test_a_non_positive_dirichlet_sum_is_a_fault_of_its_replica_and_the_run_goes_on():
    """Two learners decay the shared concentration by 0.3 each a round, from 1.0, 2.0 and 0.5 by
    replica (``per_replica=["params"]``). A sum that is not positive is recorded in the faults of
    its replica and round, the variable keeping its round-start value, and every round runs;
    under ``batch=`` round 0 is traced too, so replica 2's fault there is recorded, not refused
    (unbatched, round 0 refuses it)."""
    system, params = shared_field(COUNTS)
    steps = {"A": field_step("A", -0.3), "B": field_step("B", -0.3)}
    home = broadcast_params(params, 3)
    scale = jnp.asarray([1.0, 2.0, 0.5])[:, None, None, None]
    home = eqx.tree_at(
        lambda p: p.parameters["A.field"].concentration,
        home,
        home.parameters["A.field"].concentration * scale,
    )
    traj = run(system, home, 4, KEY, steps, batch=3, per_replica=["params"])
    np.testing.assert_array_equal(
        traj.faults["A.field"],
        [[False, True, True, True], [False, False, False, True], [True, True, True, True]],
    )
    np.testing.assert_allclose(
        traj.params.parameters["A.field"].concentration[:, 0, 0, 0], [0.4, 0.2, 0.5], atol=1e-6
    )
    assert traj["A"].emitted["a1"].shape == (3, 4)
    third = home.parameters["A.field"].concentration[2]
    mine = eqx.tree_at(lambda p: p.parameters["A.field"].concentration, params, third)
    with pytest.raises(ValueError, match="not positive after the views of"):
        run(system, mine, 4, KEY, steps)


# ----------------------------------------------------------------------- groups (R12, pin 3)


@pytest.mark.parametrize("batch", [None, 2])
def test_a_group_of_learners_equals_the_ungrouped_simultaneous_run(batch):
    """Two leaves sharing the Dirichlet ``field``, both learning it with ``plan(learn=True)``:
    stepped as one group by one vmapped call (their write-backs summed at the boundary) they
    equal the two stepped apart under simultaneous, record for record (unstacked to their own
    names), carries and the learned home included."""
    system, params = shared_field(COUNTS)
    step = plan(learn=True)
    apart = run(system, params, 5, KEY, {"A": step, "B": step}, record_carries=True, batch=batch)
    group = run(system, params, 5, KEY, {("A", "B"): step}, record_carries=True, batch=batch)
    for (path, x), (_, y) in zip(record_of(apart), record_of(group), strict=True):
        np.testing.assert_array_equal(x, y, err_msg=jax.tree_util.keystr(path))
    for x, y in zip(
        jax.tree_util.tree_leaves(apart.params),
        jax.tree_util.tree_leaves(group.params),
        strict=True,
    ):
        np.testing.assert_array_equal(x, y)
    field = np.asarray(group.params.parameters["A.field"].concentration)
    assert field.sum() > np.asarray(params.parameters["A.field"].concentration).sum()


def test_a_group_of_planners_reading_each_other_equals_the_ungrouped_run():
    """The two pushers read each other's action node: grouped under simultaneous (both read the
    boundary) they equal the ungrouped run, the process stepped apart."""
    system, params = wire()
    step = plan(selection="stochastic")
    apart = run(system, params, 6, KEY, {"A": step, "B": step, "P": sample()})
    group = run(system, params, 6, KEY, {("A", "B"): step, "P": sample()})
    for (path, x), (_, y) in zip(record_of(apart, False), record_of(group, False), strict=True):
        np.testing.assert_array_equal(x, y, err_msg=jax.tree_util.keystr(path))


def test_groups_are_refused_naming_the_fix():
    """Refused: a key that is not a tuple of distinct names, a node named twice, views that
    differ in static structure, a group mixing driven and undriven or clocked and unclocked
    nodes, and a group that is not contiguous in the sequential (or an async) order."""
    system, params = wire()
    step = plan()
    with pytest.raises(TypeError, match=r"a tuple of two or more distinct node names"):
        run(system, params, 2, KEY, {("A",): step, "B": step, "P": sample()})
    with pytest.raises(ValueError, match=r"steps names 'A' twice"):
        run(system, params, 2, KEY, {("A", "B"): step, "A": step, "P": sample()})
    with pytest.raises(ValueError, match=r"mixes nodes a node's action drives"):
        run(system, params, 2, KEY, {("A", "P"): step, "B": step})
    for order in ("sequential", "async"):
        with pytest.raises(ValueError, match=r"is not contiguous in .*\['P'\] steps between"):
            run(system, params, 2, KEY, {("A", "B"): step, "P": sample()}, tick_order=order)
    leaf, leaf_params = two_node_leaf()
    longer = dataclasses.replace(leaf, policy_len=2)
    unlike, unlike_params = merge({"A": leaf, "B": longer}, {"A": leaf_params, "B": leaf_params})
    with pytest.raises(ValueError, match=r"differ in static structure beyond their names"):
        run(unlike, unlike_params, 2, KEY, {("A", "B"): plan()})
    clocked, cparams = robots([4.0, 2.0, 4.0])
    with pytest.raises(ValueError, match=r"mixes clocked nodes \['a'\] with unclocked ones"):
        run(clocked, cparams, 2, KEY, {("a", "b"): plan(), "world": sample()})


def survey_example(name):
    examples = str(Path(__file__).resolve().parents[1] / "examples")
    sys.path.insert(0, examples)
    try:
        return __import__(name)
    finally:
        sys.path.remove(examples)


def two_rovers():
    """Two copies of the one-level survey rover (the small untrained camera, a 4 by 4 field, a
    short plan) merged with their field identified into one Dirichlet variable, each coupled to
    its own world: ``(system, params, rover step, world step)``."""
    from tests.test_survey_run import STARTS, STEPS, case_setting, small_camera

    sr = survey_example("survey_run")
    wd = survey_example("survey_world")
    setting, _ = case_setting(sr, "one-level")
    camera = small_camera(wd.IMG_PIXELS)
    rs, rp, _, _ = sr.rover_of(setting, camera)
    agent = merge({"r1": rs, "r2": rs}, {"r1": rp, "r2": rp}, [Identify("r1.field", "r2.field")])
    bindings = []
    for r, w in (("r1", "w1"), ("r2", "w2")):
        bindings += [
            Identify(f"agent.{r}.a_v", f"{w}.v"),
            Identify(f"{w}.fix", f"agent.{r}.fix"),
            Identify(f"{w}.img", f"agent.{r}.img"),
            Identify(f"{w}.gauge", f"agent.{r}.gauge"),
        ]
    system, params = couple(
        {"agent": agent, "w1": wd.world_system(), "w2": wd.world_system()}, bindings
    )
    return system, params, sr.rover_plan(setting, STEPS, STARTS, "agent.r1."), wd.world_stepper()


def test_two_survey_rovers_sharing_their_field_grouped_equal_the_ungrouped_run():
    """The survey rover twice, its field one variable both learn: the rovers grouped (one
    vmapped call of the gradient planner over both) equal the rovers stepped apart under
    simultaneous, to float32 rounding (the batched program's arithmetic), and the field learned
    from both rovers' counts is the same."""
    system, params, rover, world = two_rovers()
    worlds = {"w1": world, "w2": world}
    apart = run(system, params, 3, KEY, {"agent.r1": rover, "agent.r2": rover, **worlds})
    group = run(system, params, 3, KEY, {("agent.r1", "agent.r2"): rover, **worlds})
    for (path, x), (_, y) in zip(record_of(apart, False), record_of(group, False), strict=True):
        name = jax.tree_util.keystr(path)
        if np.asarray(x).dtype.kind == "f":
            np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-5, err_msg=name)
        else:
            np.testing.assert_array_equal(x, y, err_msg=name)
    np.testing.assert_allclose(
        apart.params.parameters["agent.r1.field"].concentration,
        group.params.parameters["agent.r1.field"].concentration,
        rtol=1e-5,
    )


# ------------------------------------------------------------------------- async (R9, pin 7)


def test_async_under_batch_is_refused_above_four_sources_naming_the_count_and_the_ceiling():
    """Five independent pushers each driving its own process: five sources, up to 120 orders,
    above the ceiling of four (24), refused before any step is called."""
    parts, bindings = {}, []
    for i in range(5):
        parts[f"m{i}"] = pusher()
        parts[f"w{i}"] = mirror(*pusher(noise=0.0), lower=True)
        bindings += [Identify(f"m{i}.push", f"w{i}.u_push"), Identify(f"w{i}.see", f"m{i}.see")]
    system, params = couple(parts, bindings)
    steps = {**{f"m{i}": plan() for i in range(5)}, **{f"w{i}": sample() for i in range(5)}}
    with pytest.raises(
        ValueError, match=r"5 sources, up to 120 orders, above the ceiling of 4 sources \(24"
    ):
        run(system, params, 2, KEY, steps, tick_order="async", batch=2)


def test_the_corridor_minds_under_async_with_batch_four_are_their_unbatched_runs():
    """``examples/compose_minds.py``'s coupled episode (two planning robots and the corridor)
    under async, ``batch=4``: each replica is its unbatched run under ``fold_in(key, b)``, to
    float32 rounding."""
    cm = survey_example("compose_minds")
    system, params = cm.coupled()
    steps = cm.coupled_steps("stochastic")
    batched = run(system, params, 8, KEY, steps, tick_order="async", batch=4)
    for b in range(4):
        alone = run(system, params, 8, jax.random.fold_in(KEY, b), steps, tick_order="async")
        assert_replica(alone, batched, b, exact=False, carries=False)
