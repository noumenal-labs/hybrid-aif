"""``run`` over a closed System: steps, rounds, timing, the first call, the record.

The pieces, in order: what ``run`` refuses (an open System, steps that are not an antichain
covering every leaf, a step that returns the wrong out-ports, an order that cannot be kept);
the per-round body (two phases under ``"simultaneous"``, a driven node stepping after its driver,
a read taken as it stands at its reader's turn, the views a ``plan()`` step is handed against the
inputs a custom step gets); the first call of each kind of step; the ``Trajectory`` tree;
``plan()``'s learning, trace and joint election; ``with_mirror``.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Categorical, ConditionalCategorical, ConditionalDirichlet, Parameter
from hybrid_aif.composition import (
    Identify,
    MindCarry,
    Trace,
    Trajectory,
    couple,
    joint_planner,
    merge,
    plan,
    run,
    runner,
    sample,
    subsystem,
    with_mirror,
)
from hybrid_aif.system import System, SystemParams
from hybrid_aif.system.mirror import mirror
from tests.host_log import logged, noted, unflagged
from tests.test_action_nodes import KERNEL, pusher
from tests.test_merge import agent_and_world
from tests.test_merge_gaussian_control import float_composite
from tests.test_run_nested import _shared

KEY = jax.random.PRNGKey(0)


def pair(kernel=None):
    """A pusher and its mirror, coupled: leaves ``model`` and ``world``."""
    system, params = pusher(kernel=kernel)
    return with_mirror(system, params)


def idle(system, key, obs, carry, params):
    """The step of a leaf that chooses nothing and emits nothing."""
    return {}, carry, params


def wire():
    """Two pushers on a direct wire (each reads the other's action node), A listed first, and a
    process driven by A: the three-leaf closed System."""
    return couple(
        {"A": pusher(), "B": pusher(), "P": mirror(*pusher(noise=0.0), lower=True)},
        [Identify("A.push", "P.u_push"), Identify("A.push", "B.see"), Identify("B.push", "A.see")],
    )


def level(value):
    def step(system, key, obs, carry, params):
        if obs is None:
            return {}, carry, params
        return {n: jnp.asarray(value, jnp.int32) for n in system.names.actions if n}, carry, params

    return step


def recorder(log, name, inner=None):
    """``inner`` (default: a constant level 1) that logs ``(name, obs, system, params)``."""
    inner = inner or level(1)

    def step(system, key, obs, carry, params):
        jax.debug.callback(
            lambda o, p: log.append((name, unflagged(o), system, p)), obs, params, ordered=True
        )
        return inner(system, key, obs, carry, params)

    return step


# ------------------------------------------------------------------------ the one engine


def test_run_is_the_closed_system_form_and_the_other_forms_are_gone_from_the_public_modules():
    """One engine: ``run`` is the closed-System form with its own signature, ``couple`` takes
    components and bindings only, and the names of the composite and wired forms exist in no
    public module."""
    import importlib
    import inspect

    import hybrid_aif
    from hybrid_aif import composition
    from hybrid_aif.composition import couple as couple_module
    from hybrid_aif.composition import merge as merge_module

    assert composition.run is runner.run
    assert list(inspect.signature(run).parameters) == [
        "system",
        "params",
        "ticks",
        "key",
        "steps",
        "clocks",
        "tick_order",
        "carries",
        "record_carries",
        "batch",
        "record",
        "chunk",
        "replicas_per_pass",
        "per_replica",
    ]
    assert list(inspect.signature(couple).parameters) == ["components", "bindings", "fixed"]
    assert "emitted" not in MindCarry._fields
    gone = (
        "particle",
        "ParticleCarry",
        "hold",
        "HoldCarry",
        "agent_step",
        "ancestral_step",
        "MindRecord",
        "ProcessRecord",
        "CompositeTrajectory",
        "MindStep",
        "Wiring",
        "Mind",
        "Process",
    )
    for module in (hybrid_aif, composition, runner, couple_module, merge_module):
        for name in gone:
            assert not hasattr(module, name), (module.__name__, name)
            assert name not in getattr(module, "__all__", ()), (module.__name__, name)
    for name in ("_run_composite", "_run_wired", "_partner_inputs"):
        assert not hasattr(runner, name) and not hasattr(merge_module, name), name
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("hybrid_aif.composition.driver")


# ----------------------------------------------------------------------------- what it refuses


def test_an_open_system_is_refused_naming_the_modality_and_the_fix():
    system, params = couple({"A": pusher(), "B": pusher()}, [])
    with pytest.raises(
        ValueError, match=r"modality 'A\.see' is observed by no identification.*open"
    ):
        run(system, params, 1, KEY, {"A": idle, "B": idle})


def test_a_system_with_no_components_is_refused():
    system, params = pusher()
    with pytest.raises(ValueError, match="no components"):
        run(system, params, 1, KEY, {"x": idle})


def test_steps_must_cover_every_leaf_and_name_only_components():
    system, params = wire()
    with pytest.raises(ValueError, match="no step for the leaf 'B'"):
        run(system, params, 1, KEY, {"A": idle, "P": sample()})
    with pytest.raises(ValueError, match=r"names \['Z'\].*leaves are \['A', 'B', 'P'\]"):
        run(system, params, 1, KEY, {"A": idle, "B": idle, "P": sample(), "Z": idle})
    with pytest.raises(TypeError, match="steps\\['A'\\] is a int"):
        run(system, params, 1, KEY, {"A": 3, "B": idle, "P": sample()})
    with pytest.raises(TypeError, match="mapping"):
        run(system, params, 1, KEY, {})


def merged_pushers():
    parts = {n: pusher() for n in "ab"}
    composite, cp = merge(
        {n: s for n, (s, _) in parts.items()}, {n: p for n, (_, p) in parts.items()}, []
    )
    return with_mirror(composite, cp)


def test_a_node_and_a_part_beneath_it_are_not_an_antichain():
    system, params = merged_pushers()
    with pytest.raises(ValueError, match="antichain.*'model' as one node or its parts"):
        run(system, params, 1, KEY, {"model": plan(), "model.a": plan(), "world": sample()})


def test_a_joint_node_with_a_sampled_leaf_beneath_it_is_refused_naming_the_others():
    system, params = merged_pushers()
    steps = {"model": plan(), "model.a": sample(), "world": sample()}
    with pytest.raises(
        ValueError, match="take the process's tables as part of its model.*step the other leaves"
    ):
        run(system, params, 1, KEY, steps)


def test_clocks_and_carries_must_name_stepped_nodes_and_clocks_a_node_with_durations():
    system, params = pair()
    steps = {"model": plan(), "world": sample()}
    with pytest.raises(
        ValueError, match=r"clocks names \['agent'\].*stepped nodes are \['model', 'world'\]"
    ):
        run(system, params, 1, KEY, steps, clocks={"agent": 1.0})
    with pytest.raises(ValueError, match=r"carries names \['nobody'\]"):
        run(system, params, 1, KEY, steps, carries={"nobody": None})
    with pytest.raises(ValueError, match=r"clocks names \['model'\], whose params declare no dur"):
        run(system, params, 1, KEY, steps, clocks={"model": 2.0})


def test_the_tick_order_and_the_tick_count_are_checked():
    system, params = pair()
    with pytest.raises(ValueError, match="tick_order must be"):
        run(system, params, 1, KEY, {"model": plan(), "world": sample()}, tick_order="whenever")
    with pytest.raises(ValueError, match="ticks=-1"):
        run(system, params, -1, KEY, {"model": plan(), "world": sample()})


def test_a_step_that_omits_an_emission_is_refused_on_the_first_round_naming_it():
    system, params = pair()
    steps = {"model": level(1), "world": idle}
    with pytest.raises(ValueError, match="world.see.*must realise every round.*view.emit"):
        run(system, params, 2, KEY, steps)


def test_a_step_that_omits_an_action_node_is_refused_naming_it():
    system, params = pair()
    with pytest.raises(ValueError, match="no value for action node 'model.push'"):
        run(system, params, 1, KEY, {"model": idle, "world": sample()})


def test_a_step_returning_an_unknown_or_malformed_out_port_is_refused():
    system, params = pair()

    def step(value, key="model.push"):
        return lambda system, k, obs, carry, p: (
            ({} if obs is None else {key: value}),
            carry,
            p,
        )

    def run_with(s):
        return run(system, params, 1, KEY, {"model": s, "world": sample()})

    with pytest.raises(ValueError, match=r"returned \['nope'\].*out-ports are \['model.push'\]"):
        run_with(step(jnp.asarray(1), "nope"))
    with pytest.raises(ValueError, match="a control node is keyed by its action node"):
        run_with(step(jnp.asarray(1), "model.u_push"))
    with pytest.raises(ValueError, match="one integer level, shape \\(\\)"):
        run_with(step(jnp.asarray([1, 1])))
    with pytest.raises(ValueError, match="one integer level"):
        run_with(step(jnp.asarray(1.0)))
    with pytest.raises(ValueError, match=r"= 7; the action node has 3 levels"):
        run_with(step(jnp.asarray(7)))
    with pytest.raises(TypeError, match="returned a list"):
        run_with(lambda system, k, obs, carry, p: ([], carry, p))


def test_a_step_returning_params_that_are_not_params_is_refused():
    system, params = pair()
    bad = lambda system, k, obs, carry, p: (  # noqa: E731
        ({} if obs is None else {"model.push": jnp.asarray(0)}),
        carry,
        3,
    )
    with pytest.raises(ValueError, match="returned int as its params"):
        run(system, params, 1, KEY, {"model": bad, "world": sample()})


def test_sequential_steps_a_driven_node_after_its_driver_whatever_its_place_in_the_listing():
    """The listing orders the sources; a driven node steps right after its driver, so a world
    listed before the model that drives it steps after it, reading the action as realised this
    round, and the model reads the world as the round before left it."""
    (m, mp), (w, wp) = pair_parts()
    system, params = couple(
        {"world": (w, wp), "model": (m, mp)},
        [Identify("model.push", "world.u_push"), Identify("world.see", "model.see")],
    )
    calls: list = []

    def tagged(name, step):
        def run_step(system, key, obs, carry, p):
            if obs is not None:
                noted(calls, name)
            return step(system, key, obs, carry, p)

        return run_step

    steps = {"model": tagged("model", level(1)), "world": tagged("world", sample())}
    traj = run(system, params, 3, KEY, steps, tick_order="sequential")
    assert calls == ["model", "world"] * 3
    np.testing.assert_array_equal(traj["world"].received["u_push"][:3], [1, 1, 1])
    # the model, a source, reads the world's emission as the previous round left it
    np.testing.assert_array_equal(
        traj["model"].received["see"][1:3], traj["world"].emitted["see"][:2]
    )
    traj = run(system, params, 2, KEY, steps)
    assert traj["world"].emitted["see"].shape == (2,)


def pair_parts():
    model = pusher()
    return model, mirror(*pusher(), lower=True)


def two_node_leaf():
    """A leaf with two control nodes, each moving one of two states, and no modality."""
    system = System(
        A_deps=(),
        B_deps=((0, 2), (1, 3), (), ()),
        controls=(2, 3),
        action_names=("a1", "a2"),
        state_names=("s1", "s2"),
        control_names=("u1", "u2"),
        modality_names=(),
        policy_len=1,
        epistemic=False,
    )
    roll = ConditionalCategorical(
        log_table=jnp.log(0.9 * jnp.eye(3)[:, :, None] * jnp.ones((1, 1, 3)) + 0.0333),
        parent_ndims=2,
    )
    params = SystemParams(
        likelihood={},
        transition={"s1": roll, "s2": roll},
        initial_state_prior={
            "s1": Categorical(logits=jnp.zeros(3)),
            "s2": Categorical(logits=jnp.zeros(3)),
        },
    )
    return system, params


def test_a_cycle_of_action_to_control_identifications_is_refused():
    system, params = couple(
        {"A": two_node_leaf(), "B": two_node_leaf()},
        [Identify("A.a1", "B.u2"), Identify("B.a1", "A.u2")],
    )
    with pytest.raises(ValueError, match=r"between \['A', 'B'\] form a cycle"):
        run(system, params, 1, KEY, {"A": idle, "B": idle})


def test_a_node_driving_its_own_control_node_is_refused_by_the_coupling_not_by_run():
    with pytest.raises(ValueError, match="joins two ports of component 'A'"):
        couple({"A": two_node_leaf(), "B": two_node_leaf()}, [Identify("A.a1", "A.u2")])


def test_a_driven_continuous_node_cannot_be_planned_and_a_custom_step_gets_it():
    composite, cp = float_composite()
    system, params = with_mirror(composite, cp)
    with pytest.raises(NotImplementedError, match="driven by a continuous variable.*custom step"):
        run(system, params, 1, KEY, {"model.c": plan(), "world": plan()})

    seen = []

    def push(system, key, obs, carry, p):
        if obs is None:
            return {}, carry, p
        return {"model.c.push": jnp.full((2,), 0.5, jnp.float32)}, carry, p

    def world(system, key, obs, carry, p):
        out, carry, p = sample()(system, key, obs, carry, p)
        if obs is not None:
            logged(seen, obs["world.c.v"])
        return out, carry, p

    traj = run(system, params, 3, KEY, {"model.c": push, "world": world})
    assert traj["model.c"].emitted["push"].shape == (3, 2)
    np.testing.assert_allclose(np.asarray(seen), 0.5)


# -------------------------------------------------------------------------------- the rounds


def test_the_simultaneous_order_steps_undriven_nodes_first_and_driven_ones_on_what_was_realised():
    """Phase one: A and B, from the round's boundary; phase two: the process, driven by A,
    on A's action as realised this round (the boundary holds the round before's)."""
    system, params = wire()
    log: list = []
    steps = {
        "A": recorder(log, "A", level(2)),
        "B": recorder(log, "B", level(1)),
        "P": recorder(log, "P", sample()),
    }
    traj = run(system, params, 3, KEY, steps)
    rounds = [log[i : i + 3] for i in range(3)]  # the reset calls come first
    assert [e[0] for e in log[:3]] == ["A", "B", "P"] and all(e[1] is None for e in log[:3])
    rounds = [log[3 + 3 * t : 6 + 3 * t] for t in range(3)]
    for t, entries in enumerate(rounds):
        assert [e[0] for e in entries] == ["A", "B", "P"]
        process = entries[2][1]
        assert int(process["P.u_push"]) == 2  # A's action of this round, realised before P
        first = entries[0][1]
        assert (first["A.see"] is None) if t == 0 else int(first["A.see"]) == 1
    assert traj["P"].received["u_push"].tolist() == [2, 2, 2, 2]


def test_a_driven_leaf_listed_first_steps_after_its_driver_under_every_order_that_allows_it():
    (m, mp), (w, wp) = pair_parts()
    system, params = couple(
        {"world": (w, wp), "model": (m, mp)},
        [Identify("model.push", "world.u_push"), Identify("world.see", "model.see")],
    )
    for order in ("simultaneous", "async"):
        calls: list = []

        def tagged(name, inner, calls=calls):
            def step(system, key, obs, carry, p):
                if obs is not None:
                    noted(calls, name)
                return inner(system, key, obs, carry, p)

            return step

        steps = {"model": tagged("model", level(2)), "world": tagged("world", sample())}
        traj = run(system, params, 3, KEY, steps, tick_order=order)
        assert calls == ["model", "world"] * 3
        assert traj["world"].received["u_push"].tolist() == [2, 2, 2, 2]


def test_a_read_is_the_value_standing_at_the_readers_turn_never_one_that_does_not_exist():
    """B reads A's action node: listed after A it sees this round's under ``"sequential"``, the
    boundary's under ``"simultaneous"``; A reads B's, listed before B it sees the round before's."""
    system, params = wire()
    for order, b_sees, a_sees in (
        ("sequential", [2, 2, 2], [None, 1, 1]),
        ("simultaneous", [None, 2, 2], [None, 1, 1]),
    ):
        log: list = []
        steps = {
            "A": recorder(log, "A", level(2)),
            "B": recorder(log, "B", level(1)),
            "P": sample(),
        }
        run(system, params, 3, KEY, steps, tick_order=order)
        seen = {"A": [], "B": []}
        for name, obs, *_ in log:
            if obs is not None:
                seen[name].append(obs[f"{name}.see"])
        norm = lambda xs: [None if x is None else int(x) for x in xs]  # noqa: E731
        assert norm(seen["B"]) == b_sees, order
        assert norm(seen["A"]) == a_sees, order


def test_a_read_of_an_emission_is_this_rounds_when_the_source_is_listed_first():
    """The world has no driver here, so it steps in component order under ``"sequential"``: a
    reader listed after it reads the emission just realised, a reader listed before it the one
    standing from the round before (the reset's, in round 0)."""
    world = mirror(*pusher(), lower=True)
    for first, after in ((("world", "model"), True), (("model", "world"), False)):
        components = {"world": world, "model": pusher()}
        system, params = couple(
            {n: components[n] for n in first}, [Identify("world.see", "model.see")]
        )
        steps = {"model": level(0), "world": sample()}
        traj = run(system, params, 5, KEY, steps, tick_order="sequential")
        seen, emitted = traj["model"].received["see"], traj["world"].emitted["see"]
        if after:
            np.testing.assert_array_equal(seen[:5], emitted)
        else:
            np.testing.assert_array_equal(seen[1:5], emitted[:4])
            np.testing.assert_array_equal(seen[0], traj["world"].start["see"])
        assert traj["model"].present["see"].all()
        # simultaneous: every read is of the round's boundary, whatever the listing
        sim = run(system, params, 5, KEY, steps)
        np.testing.assert_array_equal(
            sim["model"].received["see"][1:5], sim["world"].emitted["see"][:4]
        )


def test_the_async_order_is_reproducible_from_the_run_key_and_every_leaf_steps_once_a_round():
    system, params = wire()
    runs = []
    for _ in range(2):
        log: list = []
        steps = {"A": recorder(log, "A"), "B": recorder(log, "B"), "P": sample()}
        run(system, params, 20, KEY, steps, tick_order="async")
        runs.append([e[0] for e in log if e[1] is not None])
    assert runs[0] == runs[1]
    for t in range(20):
        assert sorted(runs[0][2 * t : 2 * t + 2]) == ["A", "B"]
    assert len({tuple(runs[0][2 * t : 2 * t + 2]) for t in range(20)}) == 2


def test_a_chain_of_driven_nodes_steps_in_driver_order_whatever_the_listing():
    """A drives B's first control node and B's second drives C's: whatever the order the
    components were coupled in, every order steps A, then B, then C."""
    system, params = couple(
        {"C": two_node_leaf(), "B": two_node_leaf(), "A": two_node_leaf()},
        [Identify("A.a1", "B.u1"), Identify("B.a2", "C.u1")],
    )
    for order in ("simultaneous", "sequential", "async"):
        calls: list = []

        def tagged(name, calls=calls):
            def step(system, key, obs, carry, p):
                if obs is not None:
                    noted(calls, name)
                return level(0)(system, key, obs, carry, p)

            return step

        steps = {n: tagged(n) for n in "ABC"}
        run(system, params, 2, KEY, steps, tick_order=order)
        assert calls == ["A", "B", "C"] * 2, order


def test_sequential_derives_the_order_of_a_join_and_of_nodes_ready_together():
    """Sources step in listing order; a node with two drivers waits for the last of them; nodes
    that become ready together step in listing order."""
    calls: list = []

    def tagged(name):
        def step(system, key, obs, carry, p):
            if obs is not None:
                noted(calls, name)
            return level(0)(system, key, obs, carry, p)

        return step

    leaf = two_node_leaf
    join, params = couple(
        {"D": leaf(), "B": leaf(), "A": leaf()},
        [Identify("A.a1", "D.u1"), Identify("B.a1", "D.u2")],
    )
    run(join, params, 1, KEY, {n: tagged(n) for n in "DBA"}, tick_order="sequential")
    assert calls == ["B", "A", "D"]
    calls.clear()
    fan, params = couple(
        {"E2": leaf(), "E1": leaf(), "A": leaf()},
        [Identify("A.a1", "E1.u1"), Identify("A.a2", "E2.u1")],
    )
    run(fan, params, 1, KEY, {n: tagged(n) for n in ("E2", "E1", "A")}, tick_order="sequential")
    assert calls == ["A", "E2", "E1"]


def test_the_action_node_of_a_driven_node_cannot_be_identified_onward_naming_the_driver():
    """``B.u1`` driven by ``A.a1`` leaves ``B.a1`` with no value of its own, so identifying it
    with ``C.u1`` would make ``C`` read ``A.a1``: a join on ``A`` written as a chain. Refused in
    either order of the bindings, naming the driver's action node; the join written out is
    accepted."""
    leaf = two_node_leaf
    parts = {"C": leaf(), "B": leaf(), "A": leaf()}
    with pytest.raises(ValueError, match=r"'B.a1'.*'B.u1'.*'A.a1' drives.*name 'A.a1'"):
        couple(parts, [Identify("A.a1", "B.u1"), Identify("B.a1", "C.u1")])
    with pytest.raises(ValueError, match=r"'B.a1'.*'B.u1' by 'A.a1'.*name 'A.a1'"):
        couple(parts, [Identify("B.a1", "C.u1"), Identify("A.a1", "B.u1")])
    with pytest.raises(ValueError, match=r"'B.a1'.*'A.a1' drives"):
        merge(
            {n: s for n, (s, _) in parts.items()},
            {n: p for n, (_, p) in parts.items()},
            [Identify("A.a1", "B.u1"), Identify("B.a1", "C.u1")],
        )
    join, _ = couple(parts, [Identify("A.a1", "B.u1"), Identify("A.a1", "C.u1")])
    assert dict(join.observes) == {"B.u1": "A.a1", "C.u1": "A.a1"}


def shared_field(declaration, own_counts=False):
    """Two leaves ``A`` and ``B`` shaped as :func:`two_node_leaf`, each reading the parameter
    variable ``field`` as ``s1``'s transition, identified into one (``A.field``); with
    ``own_counts`` each ``s2`` transition is a Dirichlet of its own, a slot learned apart from it.
    Every leaf chooses its own nodes and reads no modality, so the merge is closed."""
    parts = {}
    for name in "AB":
        system, params = two_node_leaf()
        transition = {**params.transition, "s1": Parameter("field")}
        if own_counts:
            transition["s2"] = ConditionalDirichlet(
                concentration=jnp.ones((3, 3, 3)), parent_ndims=2
            )
        params = dataclasses.replace(
            params, transition=transition, parameters={"field": declaration}
        )
        parts[name] = (system, params)
    return merge(
        {n: s for n, (s, _) in parts.items()},
        {n: p for n, (_, p) in parts.items()},
        [Identify("A.field", "B.field")],
    )


def field_step(name, by):
    """A custom step choosing level 0 on both nodes and returning its view's ``field`` moved by
    ``by`` (its concentration for a Dirichlet, its log-table for a table), or as given for
    ``None``."""

    def step(system, key, obs, carry, params):
        if obs is None:
            return {}, carry, params
        out = {f"{name}.a1": jnp.asarray(0, jnp.int32), f"{name}.a2": jnp.asarray(0, jnp.int32)}
        if by is None:
            return out, carry, params
        held = system._params_in(params, "test")
        field = held.parameters["A.field"]
        moved = jax.tree_util.tree_map(lambda x: x + by, field)
        held = dataclasses.replace(held, parameters={**held.parameters, "A.field": moved})
        for slot, i, home in held.references:
            container = list(getattr(held, slot))
            container[i] = held.parameters[home]
            held = dataclasses.replace(held, **{slot: tuple(container)})
        return out, carry, system._params_out(held)

    return step


COUNTS = ConditionalDirichlet(concentration=jnp.ones((3, 3, 3)), parent_ndims=2)
TABLE = ConditionalCategorical(log_table=jnp.log(jnp.full((3, 3, 3), 1 / 3)), parent_ndims=2)


def test_two_decays_of_one_dirichlet_summed_at_the_boundary_must_stay_positive():
    """Each learner decays the shared concentration from 1.0 by 0.6, which alone stays positive;
    summed at the boundary of a ``"simultaneous"`` round they reach -0.2, which ``write_back``
    refuses, as ``"sequential"`` refuses the second write-back on its own. Every round is traced
    (round 0 included, design batched-run.md Phase C), so the refusal is a recorded fault of
    the round under the variable, the variable keeping its round-start value (simultaneous) or
    the first write-back (sequential), and the run goes on."""
    system, params = shared_field(COUNTS)
    steps = {"A": field_step("A", -0.6), "B": field_step("B", -0.6)}
    together = run(system, params, 2, KEY, steps)
    assert together.faults["A.field"].tolist() == [True, True]
    np.testing.assert_array_equal(
        together.params.parameters["A.field"].concentration, COUNTS.concentration
    )
    one_by_one = run(system, params, 2, KEY, steps, tick_order="sequential")
    assert one_by_one.faults["A.field"].tolist() == [True, True]
    np.testing.assert_allclose(
        one_by_one.params.parameters["A.field"].concentration, COUNTS.concentration - 0.6
    )


def test_custom_steps_sharing_a_variable_that_is_not_counts_are_checked_at_the_boundary():
    """A custom step is not known to learn before it returns, so ``"simultaneous"`` runs two
    custom steps that share a table variable and learn nothing, and records a fault at the
    boundary of each round in which both wrote it back, under the variable."""
    system, params = shared_field(TABLE)
    run(system, params, 2, KEY, {"A": field_step("A", None), "B": field_step("B", None)})
    traj = run(system, params, 2, KEY, {"A": field_step("A", 0.5), "B": field_step("B", None)})
    np.testing.assert_allclose(
        traj.params.parameters["A.field"].log_table, TABLE.log_table + 1.0, rtol=1e-6
    )
    steps = {"A": field_step("A", 0.5), "B": field_step("B", 0.5)}
    # every round is traced (round 0 included), so the refusal is a recorded fault of each round
    # in which both wrote it back, and the variable keeps its round-start value
    both = run(system, params, 2, KEY, steps)
    assert both.faults["A.field"].tolist() == [True, True]
    np.testing.assert_array_equal(both.params.parameters["A.field"].log_table, TABLE.log_table)
    run(system, params, 2, KEY, steps, tick_order="sequential")


def test_plan_learners_of_one_variable_that_is_not_counts_are_refused_up_front_by_their_slots():
    """Two ``plan(learn=True)`` both reach the shared table, refused before the first round; a
    ``plan(learn=[...])`` naming only its own counts beside one of them does not reach it."""
    system, params = shared_field(TABLE, own_counts=True)
    with pytest.raises(
        ValueError, match=r"'A.field' \(ConditionalCategorical\) is learned by both"
    ):
        run(system, params, 1, KEY, {"A": plan(learn=True), "B": plan(learn=True)})
    run(system, params, 2, KEY, {"A": plan(learn=True), "B": plan(learn=["transition:B.s2"])})


def test_a_sampler_draws_the_action_node_of_a_control_node_nothing_drives():
    system, params = couple(
        {"model": pusher(), "world": mirror(*pusher(), lower=True)},
        [Identify("world.see", "model.see")],
    )
    traj = run(system, params, 6, KEY, {"model": level(0), "world": sample()})
    drawn = traj["world"].emitted["push"]
    assert drawn.shape == (6,) and set(np.asarray(drawn).tolist()) <= {0, 1, 2}
    assert len(set(np.asarray(drawn).tolist())) > 1  # a draw each round, not a constant
    np.testing.assert_array_equal(traj["world"].intended["push"], drawn)


def test_a_view_leaves_a_modality_that_observes_another_leafs_action_node_an_observation():
    system, params = wire()
    view, _ = subsystem(system, params, "A")
    assert view.observes == () and "A.see" in view.modality_names
    assert dict(system.observes)["A.see"] == "B.push"


# ----------------------------------------------------------------- views and what a step gets


class Recording(runner._Plan):
    """A ``plan()`` step that logs the view and its params with each observation."""

    def __init__(self, log, **kwargs):
        super().__init__(None, None, False, None, kwargs)
        self.log = log

    def __call__(self, system, key, obs, carry, params):
        jax.debug.callback(
            lambda o, p: self.log.append((unflagged(o), system, p)), obs, params, ordered=True
        )
        return super().__call__(system, key, obs, carry, params)


def driven_world(kernel=None):
    """Pusher ``A`` (with an actuator kernel if given) drives the control node of ``B``, and a
    mirror ``W`` of ``B`` feeds both pushers' ``see``."""
    b = pusher(noise=0.0)
    return couple(
        {"A": pusher(kernel=kernel), "B": b, "W": mirror(*b, lower=True)},
        [
            Identify("A.push", "B.u_push"),
            Identify("W.see", "B.see"),
            Identify("W.see", "A.see"),
            Identify("A.push", "W.u_push"),
        ],
    )


def same_params(a, b) -> bool:
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    return ta == tb and all(bool(jnp.array_equal(x, y)) for x, y in zip(la, lb, strict=True))


@pytest.mark.parametrize("order", ["simultaneous", "sequential", "async"])
def test_a_plan_step_gets_the_view_sliced_at_the_realised_level_and_a_custom_step_the_input(order):
    """The driver's kernel moves its draw off the intended level: the view's params are those of
    ``subsystem`` sliced at the level realised that round, not the intended one."""
    system, params = driven_world(KERNEL)
    log: list = []
    traj = run(
        system,
        params,
        24,
        KEY,
        {"A": level(2), "B": Recording(log), "W": sample()},
        tick_order=order,
    )
    realised = traj["A"].emitted["push"].tolist()
    assert len(set(realised)) > 1 and realised != [2] * 24
    rounds = [(obs, view, vp) for obs, view, vp in log if obs is not None]
    assert len(rounds) == 24
    for (obs, view, vp), lvl in zip(rounds, realised, strict=True):
        sliced, sp = subsystem(system, params, "B", inputs={"A.push": lvl})
        assert view == sliced and view.names.controls == () and "B.u_push" not in obs
        assert same_params(vp, sp)
    wrong = subsystem(system, params, "B", inputs={"A.push": (realised[0] + 1) % 3})[1]
    assert not same_params(rounds[0][2], wrong)
    # a custom step gets the unsliced view, and the driven node's value (as realised) in obs
    seen: list = []

    def custom(view, key, obs, carry, p):
        jax.debug.callback(lambda o: seen.append((o, view)), obs, ordered=True)
        return {}, carry, p

    custom_traj = run(
        system, params, 24, KEY, {"A": level(2), "B": custom, "W": sample()}, tick_order=order
    )
    got = [int(obs["B.u_push"]) for obs, view in seen if obs is not None]
    assert got == custom_traj["A"].emitted["push"].tolist()
    assert all(view.names.controls == ("B.u_push",) for obs, view in seen if obs is not None)


def test_a_driven_view_before_the_drivers_first_realisation_averages_the_input_uniformly():
    """The reset call of a ``plan()`` step is handed the view averaged over the driver's node:
    the driver has realised nothing yet."""
    system, params = driven_world()
    log: list = []
    run(system, params, 1, KEY, {"A": level(1), "B": Recording(log), "W": sample()})
    (reset_obs, _, reset_params), (obs, _, round_params) = log
    assert reset_obs is None and obs is not None
    assert same_params(reset_params, subsystem(system, params, "B", inputs={"A.push": None})[1])
    assert same_params(round_params, subsystem(system, params, "B", inputs={"A.push": 1})[1])
    assert not same_params(reset_params, round_params)


def test_a_wrapped_plan_keeps_its_slicing_through_functools_wraps_and_is_refused_without():
    system, params = driven_world()
    inner = plan()

    def marked(view, key, obs, carry, p):
        return inner(view, key, obs, carry, p)

    marked = functools.wraps(inner)(marked)
    traj = run(system, params, 2, KEY, {"A": level(1), "B": marked, "W": sample()})
    assert traj["B"].carry is not None and traj["B"].received["u_push"].tolist() == [1, 1, 1]

    def bare(view, key, obs, carry, p):
        return inner(view, key, obs, carry, p)

    with pytest.raises(
        ValueError, match="plan\\(\\): the view has driven control node.*wraps plan"
    ):
        run(system, params, 2, KEY, {"A": level(1), "B": bare, "W": sample()})
    bare.sliced = True
    run(system, params, 1, KEY, {"A": level(1), "B": bare, "W": sample()})


# ---------------------------------------------------------------------------- the first call


def test_every_step_is_called_first_with_no_observation_in_node_order_with_the_reset_keys():
    system, params = wire()
    log: list = []
    seen_keys: list = []
    original = runner._reset_keys

    def recording(key, count):
        keys = original(key, count)
        seen_keys.extend(keys)
        return keys

    steps = {"A": recorder(log, "A"), "B": recorder(log, "B"), "P": recorder(log, "P", sample())}
    from unittest import mock

    with mock.patch.object(runner, "_reset_keys", recording):
        run(system, params, 1, KEY, steps)
    assert [(n, o is None) for n, o, *_ in log[:3]] == [("A", True), ("B", True), ("P", True)]
    key, k_reset = jax.random.split(KEY)
    np.testing.assert_array_equal(seen_keys[0], k_reset)
    np.testing.assert_array_equal(seen_keys[2], jax.random.fold_in(k_reset, 2))


def test_the_first_call_of_each_kind_of_step():
    system, params = pair()
    traj = run(system, params, 2, KEY, {"model": plan(), "world": sample()}, record_carries=True)
    model, world = traj["model"], traj["world"]
    # plan() returns nothing and leaves the declared initial priors as the first prior: its
    # carry after the reset holds them, in the form every round's carry has (design
    # batched-run.md Phase C), and says no round has run
    assert model.start == {}
    first = model.carries[0]
    assert isinstance(first, MindCarry) and not bool(first.started)
    initial = system.initial_priors(params)
    for name, belief in first.beliefs.items():
        np.testing.assert_array_equal(belief.probs, initial[name].probs)
    # sample() returns the emissions of the initial state, which enter round 0, and the state
    assert set(world.start) == {"see"} and set(world.carries[0]) == {"world.x"}
    assert model.present["see"].tolist() == [True, True, True]
    # a custom step returns nothing on its first call, and whatever it returns on it is refused
    # when it names no out-port of the node
    custom = run(system, params, 1, KEY, {"model": level(0), "world": sample()})
    assert custom["model"].start == {}
    bad = lambda system, key, obs, carry, p: ({"nope": jnp.asarray(0)}, carry, p)  # noqa: E731
    with pytest.raises(ValueError, match=r"returned \['nope'\]"):
        run(system, params, 1, KEY, {"model": bad, "world": sample()})


def test_a_seeded_carry_of_a_sampler_is_a_partial_point_state_its_reset_takes():
    system, params = pair()
    steps = {"model": plan(), "world": sample()}
    traj = run(system, params, 1, KEY, steps, carries={"world": {"world.x": jnp.asarray(2)}})
    assert int(traj["world"].start["see"]) in range(3)
    seeded = run(system, params, 0, KEY, steps, carries={"world": {"world.x": jnp.asarray(2)}})
    assert int(seeded["world"].carry["world.x"]) == 2
    other = run(system, params, 0, jax.random.PRNGKey(5), steps)
    assert other["world"].carry["world.x"].shape == ()


# ------------------------------------------------------------------------------ the record


def test_the_trajectory_tree_dotted_lookup_and_layout():
    system, params = merged_pushers()
    ticks = 4
    steps = {"model.a": plan(), "model.b": level(1), "world": sample()}
    traj = run(system, params, ticks, KEY, steps, record_carries=True)
    assert isinstance(traj, Trajectory) and traj.params is not None
    assert set(traj.parts) == {"model", "world"} and set(traj["model"].parts) == {"a", "b"}
    assert traj["model.a"] is traj["model"].parts["a"] and traj["model.a"].params is None
    # keys are relative to a node; a node that is not stepped holds the prefixed union
    assert set(traj["model.a"].emitted) == {"push"}
    assert set(traj["model"].emitted) == {"a.push", "b.push"}
    assert set(traj.emitted) == {"model.a.push", "model.b.push", "world.a.see", "world.b.see"}
    assert set(traj["world"].emitted) == {"a.see", "b.see"}
    # received and present have ticks + 1 entries, emitted and intended ticks, carries ticks + 1
    assert traj["model.a"].received["see"].shape == (ticks + 1,)
    assert traj["model.a"].present["see"].shape == (ticks + 1,)
    assert (
        traj["model.a"].emitted["push"].shape == traj["model.a"].intended["push"].shape == (ticks,)
    )
    assert (
        len(traj["model.a"].carries) == ticks + 1
        and traj["model.a"].decided.tolist() == [True] * ticks
    )
    assert set(traj["model"].carries) == {"a", "b"} and set(traj["model"].carry) == {"a", "b"}
    assert traj["world"].start["a.see"].shape == ()
    # a name that is no node is refused, naming the nodes
    with pytest.raises(KeyError, match=r"'model.c'.*nodes here are \['a', 'b'\]"):
        traj["model.c"]
    # indexing by position is the tuple's
    assert traj[0] is traj.params
    # without record_carries the carries are not kept
    quiet = run(system, params, 2, KEY, steps)
    assert quiet["model.a"].carries is None and quiet["model.a"].carry is not None
    assert quiet.carries is None


def test_a_node_planned_jointly_is_one_leaf_of_the_record():
    system, params = merged_pushers()
    traj = run(system, params, 2, KEY, {"model": plan(), "world": sample()})
    assert traj["model"].parts == {} and set(traj["model"].emitted) == {"a.push", "b.push"}


def test_absent_values_are_zeros_and_flagged_and_the_last_row_is_what_the_slots_end_holding():
    system, params = wire()
    steps = {"A": level(2), "B": level(1), "P": sample()}
    traj = run(system, params, 3, KEY, steps)
    assert traj["B"].present["see"].tolist() == [False, True, True, True]
    assert traj["B"].received["see"].tolist() == [0, 2, 2, 2]
    assert traj["A"].received["see"].tolist() == [0, 1, 1, 1]


def test_a_zero_tick_run_records_only_the_start():
    system, params = wire()
    traj = run(system, params, 0, KEY, {"A": level(0), "B": level(0), "P": sample()})
    assert traj["P"].emitted["see"].shape[0] == 0 and traj["A"].received["see"].shape[0] == 1


# ----------------------------------------------------------------- plan(): learn, trace, joint


def test_plan_trace_keeps_the_policy_posterior_and_the_enumerated_policies_in_the_carry():
    (agent, ap), _ = agent_and_world(policy_len=1)
    system, params = with_mirror(agent, ap)
    traj = run(system, params, 2, KEY, {"model": plan(trace=True), "world": sample()})
    trace = traj["model"].carry.trace
    assert isinstance(trace, Trace)
    assert trace.q_pi.logits.shape[0] == trace.policies.shape[0] == 5
    assert trace.policies.shape[1:] == (1, 1) and trace.tree
    untraced = run(system, params, 2, KEY, {"model": plan(), "world": sample()})
    assert untraced["model"].carry.trace is None
    np.testing.assert_array_equal(
        traj["model"].emitted["a_location"], untraced["model"].emitted["a_location"]
    )


def test_plan_replace_sets_view_fields_and_act_keywords_pass_through():
    system, params = pair()
    steps = {"model": plan(replace={"policy_len": 2}, selection="deterministic"), "world": sample()}
    traj = run(system, params, 3, KEY, steps)
    again = run(system, params, 3, KEY, steps)
    np.testing.assert_array_equal(traj["model"].emitted["push"], again["model"].emitted["push"])


def test_plan_learn_forms_and_a_learner_writes_its_variable_back():
    from hybrid_aif.composition.runner import _learn_arguments

    assert _learn_arguments(None) is None and _learn_arguments(True) == ()
    assert dict(_learn_arguments({"lr": 0.5})) == {"lr": 0.5}
    assert dict(_learn_arguments(["likelihood:see"]))["lr"] == 0.0
    with pytest.raises(TypeError, match="learn is None, True"):
        _learn_arguments(3.0)


def test_a_one_mind_node_plans_as_the_hand_loop_of_the_joint_view_does():
    """``steps={"model": plan()}`` over a merge steps ``joint_planner(system, params, "model")``:
    its first action is the joint view's first ``plan()`` call on the world's reset emission."""
    system, params = merged_pushers()
    steps = {"model": plan(selection="deterministic"), "world": sample()}
    traj = run(system, params, 1, KEY, steps)
    view, vparams = joint_planner(system, params, "model")
    key, k_reset = jax.random.split(KEY)
    key, k_tick = jax.random.split(key)
    keys = runner._round_keys(k_tick, 2)
    obs = {f"model.{n}.see": traj["world"].start[f"{n}.see"] for n in "ab"}
    step = plan(selection="deterministic")
    _, carry, _ = step(view, k_reset, None, None, vparams)
    out, _, _ = step(view, keys[0], obs, carry, vparams)
    for n in "ab":
        assert int(out[f"model.{n}.push"]) == int(traj["model"].emitted[f"{n}.push"][0])


def test_plan_learn_writes_the_learned_variable_back_and_keeps_its_learning_carry():
    composite, cp, _, _ = _shared()
    system, params = with_mirror(composite, cp)
    name = "model.rover.field"
    steps = {"model.rover": plan(learn=True), "model.mission": plan(), "world": sample()}
    traj = run(system, params, 4, KEY, steps)
    learned = traj.params.parameters[name].concentration
    assert not np.array_equal(learned, params.parameters[name].concentration)
    half = run(system, params, 4, KEY, {**steps, "model.rover": plan(learn={"lr": 0.5})})
    gain = learned - params.parameters[name].concentration
    np.testing.assert_allclose(
        half.params.parameters[name].concentration - params.parameters[name].concentration,
        gain / 2,
        rtol=1e-5,
    )
    assert traj["model.rover"].carry.learn_state is not None
    frozen = run(system, params, 4, KEY, {**steps, "model.rover": plan()})
    assert frozen.params is params


def test_a_simultaneous_kernel_run_is_pinned_under_the_round_keys():
    """The zoo pair against its own mirror, a kernel on the agent's node: the agent's node is
    realised with the last key of the round (phase one), the mirror steps on it; pinned."""
    from tests.test_couple import KERNEL5
    from tests.test_merge import OBSERVATION_BINDINGS

    (agent, ap), (world, wp) = agent_and_world(policy_len=1)
    composite, cp = merge(
        {"agent": agent, "world": world}, {"agent": ap, "world": wp}, OBSERVATION_BINDINGS
    )
    cp = dataclasses.replace(cp, actuator={"agent.a_location": KERNEL5})
    system, params = with_mirror(composite, cp)
    from tests.test_run_parity import ancestral, as_recorded

    steps = {"model.agent": plan(), "model.world": ancestral(), "world": sample()}
    traj = run(system, as_recorded(params), 12, KEY, steps, record_carries=True)
    assert traj["model.agent"].emitted["a_location"].tolist() == [
        4,
        0,
        3,
        4,
        3,
        0,
        0,
        1,
        4,
        0,
        2,
        2,
    ]
    assert traj["model.agent"].intended["a_location"].tolist() == [
        4,
        3,
        3,
        4,
        2,
        0,
        2,
        3,
        4,
        0,
        2,
        4,
    ]
    assert traj["model.world"].emitted["a_location"].tolist() == [
        1,
        2,
        2,
        0,
        1,
        4,
        2,
        4,
        1,
        4,
        0,
        4,
    ]
    states = [int(c["world.world.location"]) for c in traj["world"].carries]
    assert states == [0, 0, 0, 0, 0, 0, 4, 2, 4, 1, 4, 0, 4]


# --------------------------------------------------------------------------- with_mirror


def test_with_mirror_couples_a_model_to_its_mirror_one_modality_and_action_node_each():
    system, params = pusher()
    coupled, cparams = with_mirror(system, params)
    assert [n for n, _ in coupled.subsystems] == ["model", "world"]
    assert dict(coupled.observes) == {"model.see": "world.see", "world.u_push": "model.push"}
    assert coupled.action_names == ("model.push", None)
    ports = coupled.ports(cparams)
    assert set(ports.outs) == {"model.push", "world.see"} and not any(
        m in ports.ins for m in coupled.modality_names
    )


def test_with_mirror_names_a_pair_of_components_and_refuses_what_it_cannot_do():
    system, params = pusher()
    coupled, _ = with_mirror(system, params, names=("agent", "env"))
    assert [n for n, _ in coupled.subsystems] == ["agent", "env"]
    with pytest.raises(ValueError, match="two distinct component names"):
        with_mirror(system, params, names=("a", "a"))
    with pytest.raises(ValueError, match="two distinct component names"):
        with_mirror(system, params, names=("a.b", "c"))
    with pytest.raises(ValueError, match="no component to mirror"):
        with_mirror(system, params, per_component=True)


def test_with_mirror_binds_an_own_read_modality_to_nothing_and_a_composite_to_its_leaves():
    from hybrid_aif import Emission

    system, params = pusher()
    own = dataclasses.replace(
        System(
            A_deps=((0,), (0,)),
            B_deps=((0, 1), ()),
            controls=(1,),
            action_names=("push",),
            state_names=("x",),
            control_names=("u_push",),
            modality_names=("see", "read"),
            emissions={"mode_x": Emission(("x",), "mode", "identity")},
            observes={"read": "mode_x"},
            policy_len=1,
            epistemic=False,
        )
    )
    table = system.A_deps and params.likelihood["see"]
    own_params = SystemParams(
        likelihood={"see": table, "read": ConditionalCategorical(table.log_table, 1)},
        transition=params.transition,
        initial_state_prior=params.initial_state_prior,
    )
    coupled, cparams = with_mirror(own, own_params)
    assert ("model.read", "model.mode_x") in coupled.observes
    assert "model.read" not in {slot for slot, v in coupled.observes if v.startswith("world.")}
    traj = run(coupled, cparams, 3, KEY, {"model": plan(), "world": sample()})
    assert traj["model"].present["read"].tolist() == [False, True, True, True]


# ----------------------------------------------------------- the review's fixes


def test_a_driven_node_reads_live_under_sequential_and_async_and_at_the_boundary_otherwise():
    """A drives W and B, W is listed before B, and B reads ``W.see``: W and B are driven by A, so
    they step right after it in listing order, and B reads this round's emission of W under
    ``"sequential"`` and ``"async"`` (every read stands), the boundary's under
    ``"simultaneous"``."""
    system, params = couple(
        {"A": pusher(), "W": mirror(*pusher(noise=0.0), lower=True), "B": pusher()},
        [
            Identify("A.push", "W.u_push"),
            Identify("A.push", "B.u_push"),
            Identify("W.see", "A.see"),
            Identify("W.see", "B.see"),
        ],
    )
    seen = {}
    for order in ("simultaneous", "async", "sequential"):
        log: list = []
        steps = {"A": level(1), "W": sample(), "B": recorder(log, "B", level(0))}
        traj = run(system, params, 5, KEY, steps, tick_order=order)
        seen[order] = [int(o["B.see"]) for _, o, *_ in log if o is not None]
        emitted = traj["W"].emitted["see"].tolist()
        start = int(traj["W"].start["see"])
        if order == "simultaneous":
            assert seen[order] == [start, *emitted[:4]]
        else:
            assert seen[order] == emitted
    assert seen["async"] == seen["sequential"] != seen["simultaneous"]


def test_async_reads_live_in_a_fresh_random_order_of_the_sources_each_round():
    """W (a process) and B (a pusher that reads ``W.see``) are both sources: under ``"async"`` B
    reads this round's value when W came before it in that round's drawn listing and the
    previous round's otherwise, checked against the permutation the run key draws."""
    system, params = couple(
        {"W": mirror(*pusher(noise=0.0), lower=True), "B": pusher()},
        [Identify("W.see", "B.see")],
    )
    log: list = []
    ticks = 12
    steps = {"W": sample(), "B": recorder(log, "B", level(0))}
    traj = run(system, params, ticks, KEY, steps, tick_order="async")
    seen = [int(o["B.see"]) for _, o, *_ in log if o is not None]
    emitted = traj["W"].emitted["see"].tolist()
    start = int(traj["W"].start["see"])
    sweep = runner._sweep_key(jax.random.split(KEY)[0], 2)
    live = []
    for t in range(ticks):
        #  round t's listing of the sources, 0 is W, 1 is B (design batched-run.md R9)
        order = [int(i) for i in jax.random.permutation(jax.random.fold_in(sweep, t), 2)]
        live.append(order.index(0) < order.index(1))
        want = emitted[t] if live[-1] else (start if t == 0 else emitted[t - 1])
        assert seen[t] == want, (t, order)
    assert any(live) and not all(live)  # both listings occur


def test_a_leaf_whose_own_action_node_drives_its_own_control_node_is_refused():
    system, params = _self_driven()
    with pytest.raises(ValueError, match="a cycle of length one"):
        run(system, params, 1, KEY, {"A": level(0)})


def _self_driven():
    leaf, leaf_params = two_node_leaf()
    return merge({"A": leaf}, {"A": leaf_params}, [Identify("A.a1", "A.u2")])


def test_a_control_node_with_no_action_node_and_no_driver_is_refused():
    model, mp = pusher()
    orphan = (dataclasses.replace(model, action_names=(None,)), mp)
    system, params = couple(
        {"model": orphan, "world": mirror(*pusher(), lower=True)},
        [Identify("world.see", "model.see")],
    )
    with pytest.raises(ValueError, match=r"control node 'model.u_push' has no action node"):
        run(system, params, 1, KEY, {"model": idle, "world": sample()})


def test_with_mirror_drives_the_worlds_copy_of_an_internally_driven_node_by_the_same_decision():
    """The zoo whose world's node is driven by the agent's action node inside the model: the
    mirror's two copies of the node both carry the agent's realised level, and ``sample_step``
    gives a driven node its internal driver's drawn level."""
    from hybrid_aif.composition import Link
    from tests.test_merge import OBSERVATION_BINDINGS

    (agent, ap), (world, wp) = agent_and_world(policy_len=1)
    composite, cp = merge(
        {"agent": agent, "world": world},
        {"agent": ap, "world": wp},
        [Link("agent.a_location", "world.u_location"), *OBSERVATION_BINDINGS],
    )
    system, params = with_mirror(composite, cp)
    assert dict(system.observes)["world.world.u_location"] == "model.agent.a_location"
    assert dict(system.observes)["world.agent.u_location"] == "model.agent.a_location"
    steps = {"model.agent": level(3), "model.world": plan(), "world": sample()}
    traj = run(system, params, 6, KEY, steps, record_carries=True)
    received = traj["world"].received
    assert received["agent.u_location"].tolist() == [3] * 7 == received["world.u_location"].tolist()
    final = traj["world"].carry
    assert int(final["world.world.location"]) == int(final["world.agent.location"]) == 3
    # sample_step of the model itself: its internally driven node follows its driver
    state = composite.sample_initial(KEY, cp)[1]
    for i in range(20):
        _, state = composite.sample_step(jax.random.fold_in(KEY, i), cp, state)
        assert int(state["agent.location"]) == int(state["world.location"])


def test_sample_step_refuses_a_driven_node_nothing_supplies():
    from hybrid_aif.distributions import ConditionalCategorical as CC

    model, mp = pusher()
    world, wp = mirror(model, mp, lower=True)
    system, params = couple(
        {"model": (model, mp), "world": (world, wp)},
        [Identify("world.see", "model.see"), Identify("model.push", "world.u_push")],
    )
    view, vparams = subsystem(system, params, "world")
    state = view.sample_initial(KEY, vparams)[1]
    with pytest.raises(ValueError, match="is driven .* and inputs does not supply it"):
        view.sample_step(KEY, vparams, state, {})
    view.sample_step(KEY, vparams, state, {"world.u_push": jnp.asarray(1)})
    del CC


def test_with_mirror_lower_false_keeps_the_declarations_and_shares_none_with_the_model():
    from hybrid_aif.distributions import ConditionalDirichlet

    model, mp = pusher()
    mp = dataclasses.replace(
        mp,
        likelihood={"see": ConditionalDirichlet(concentration=jnp.ones((3, 3)), parent_ndims=1)},
    )
    system, params = with_mirror(model, mp, lower=False)
    assert isinstance(params.emission["world.see"], ConditionalDirichlet)
    assert params.emission["world.see"] is not params.likelihood["model.see"]
    run(system, params, 2, KEY, {"model": plan(), "world": sample()})


def test_a_view_that_learns_its_own_table_writes_it_to_the_home_under_its_namespaced_key():
    from hybrid_aif.distributions import ConditionalDirichlet

    model, mp = pusher()
    mp = dataclasses.replace(
        mp,
        likelihood={
            "see": ConditionalDirichlet(
                concentration=jnp.ones((3, 3)) + 4 * jnp.eye(3), parent_ndims=1
            )
        },
    )
    system, params = with_mirror(model, mp)
    start = params.likelihood["model.see"].concentration
    kernel = params.emission["world.see"]
    gains = {}
    for name, learn in (("full", True), ("half", {"lr": 0.5})):
        traj = run(system, params, 4, KEY, {"model": plan(learn=learn), "world": sample()})
        home = traj.params
        gains[name] = home.likelihood["model.see"].concentration - start
        assert gains[name].sum() > 0
        assert jax.tree_util.tree_all(
            jax.tree.map(jnp.array_equal, home.emission["world.see"], kernel)
        )
    assert 0 < gains["half"].sum() < gains["full"].sum()


def test_a_learned_table_whose_shape_the_slicing_changed_is_refused_naming_the_variable():
    import types

    from hybrid_aif.distributions import ConditionalDirichlet

    b, bp = pusher(noise=0.0)
    bp = dataclasses.replace(
        bp,
        transition={"x": ConditionalDirichlet(concentration=jnp.ones((3, 3, 3)), parent_ndims=2)},
    )
    system, params = couple(
        {"A": pusher(), "B": (b, bp), "W": mirror(b, bp, lower=True)},
        [
            Identify("A.push", "B.u_push"),
            Identify("W.see", "B.see"),
            Identify("W.see", "A.see"),
            Identify("A.push", "W.u_push"),
        ],
    )
    view, given = subsystem(system, params, "B", inputs={"A.push": 1})
    grown = ConditionalDirichlet(
        concentration=given.transition["B.x"].concentration + 1.0, parent_ndims=2
    )
    learned = dataclasses.replace(given, transition={"B.x": grown})
    node = types.SimpleNamespace(name="B")
    with pytest.raises(ValueError, match=r"learned transition\['B.x'\].*sliced at \['A.push'\]"):
        runner._own_slots(view, node, given, learned, (("A.push", 1),), params)


def test_an_emission_is_checked_as_an_action_node_is():
    system, params = pair()

    def world(value, first):
        def step(view, key, obs, carry, p):
            if obs is None:
                return {"world.see": first}, carry, p
            return {"world.see": value}, carry, p

        return step

    def run_with(value):
        steps = {
            "model": level(0),
            "world": world(value, value if value.shape == (3,) else jnp.asarray(0)),
        }
        return run(system, params, 1, KEY, steps)

    with pytest.raises(ValueError, match=r"'world.see' = 7; the port has 3 levels"):
        run_with(jnp.asarray(7))
    with pytest.raises(ValueError, match=r"shape \(2,\); the port has 3 levels.*shape \(3,\)"):
        run_with(jnp.asarray([0.2, 0.8]))
    with pytest.raises(ValueError, match="one integer level"):
        run_with(jnp.asarray(1.5))
    soft = run_with(jnp.asarray([0.2, 0.3, 0.5]))
    assert soft["world"].emitted["see"].shape == (1, 3)


def _emission_reader_system():
    """``P`` cycles a three-level state ``x`` and emits it (``x_read``); ``D`` chooses an action
    node ``a_hand``; ``R`` has a control node ``u_go`` driven by ``D``'s action (so ``R`` is a
    phase-two node) and a control node ``u_x`` driven by ``P``'s emission."""
    from hybrid_aif import Categorical
    from hybrid_aif.reads import Emission

    roll = jnp.log(jnp.roll(jnp.eye(3), 1, axis=0) + 1e-12)
    cycler = System(
        B_deps=((0,),),
        controlled=(False,),
        state_names=("x",),
        modality_names=(),
        emissions=(("x_read", Emission(("x",), "mode", "identity")),),
        policy_len=1,
        epistemic=False,
    )
    cycler_params = SystemParams(
        transition={"x": ConditionalCategorical(log_table=roll, parent_ndims=1)},
        initial_state_prior={"x": Categorical.from_probs(jnp.eye(3)[0])},
    )
    mover = System(
        B_deps=((0,),),
        controlled=(True,),
        state_names=("hand",),
        modality_names=(),
        policy_len=1,
        epistemic=False,
    )
    move = jnp.zeros((3, 3, 3)).at[jnp.arange(3), :, jnp.arange(3)].set(1.0)
    mover_params = SystemParams(
        transition={
            "hand": ConditionalCategorical(log_table=jnp.log(move + 1e-12), parent_ndims=2)
        },
        initial_state_prior={"hand": Categorical.from_probs(jnp.eye(3)[0])},
    )
    copy = jnp.zeros((3, 3, 3, 3)).at[jnp.arange(3), :, :, jnp.arange(3)].set(1.0)
    reader = System(
        B_deps=((0, 1, 2), (), ()),
        controls=(1, 2),
        action_names=("a_go", "a_x"),
        state_names=("h",),
        control_names=("u_go", "u_x"),
        modality_names=(),
        policy_len=1,
        epistemic=False,
    )
    reader_params = SystemParams(
        transition={"h": ConditionalCategorical(log_table=jnp.log(copy + 1e-12), parent_ndims=3)},
        initial_state_prior={"h": Categorical.from_probs(jnp.eye(3)[0])},
    )
    return couple(
        {"P": (cycler, cycler_params), "D": (mover, mover_params), "R": (reader, reader_params)},
        [Identify("D.a_hand", "R.u_go"), Identify("P.x_read", "R.u_x")],
    )


@pytest.mark.parametrize("order, reads", [("simultaneous", "boundary"), ("sequential", "now")])
def test_a_phase_two_node_reads_an_emission_driven_control_node_at_the_boundary(order, reads):
    """``R`` is a phase-two node (its ``u_go`` is driven by ``D``'s action) and reads ``P``'s
    emission through ``u_x``: the value at the round's boundary under ``"simultaneous"``, the value
    as it stands (this round's, ``P`` being listed first) under ``"sequential"``, whereas ``u_go``
    carries ``D``'s action as realised this round."""

    def chooser(system, key, obs, carry, params):
        if obs is None:
            return {}, 0, params
        return {"D.a_hand": jnp.asarray(carry % 3, jnp.int32)}, carry + 1, params

    system, params = _emission_reader_system()
    steps = {"P": sample(), "D": chooser, "R": sample()}
    traj = run(system, params, 6, KEY, steps, tick_order=order, record_carries=True)
    x = [int(c["P.x"]) for c in traj["P"].carries]
    seen = np.asarray(traj["R"].received["u_x"][:6]).tolist()
    assert seen == (x[:6] if reads == "boundary" else x[1:7])
    np.testing.assert_array_equal(traj["R"].received["u_go"][:6], traj["D"].emitted["a_hand"])


@pytest.mark.parametrize("order", ["sequential", "async"])
def test_the_derived_order_is_depth_first_a_chain_follows_its_source_before_a_sibling(order):
    """``S`` drives ``A`` and ``B``, ``A`` drives ``C``, and ``B`` reads ``C``'s emission. Each
    driven node steps as soon as its last driver has realised, before the sweep returns to nodes
    made ready earlier: ``S, A, C, B``, so ``B`` reads this round's emission of ``C`` (a sweep by
    waves, ``S, A, B, C``, would give it the round before's)."""
    from tests.test_clocks import _counting, _leaf

    system, params = couple(
        {n: _leaf() for n in "SABC"},
        [
            Identify("S.a1", "A.u1"),
            Identify("S.a2", "B.u1"),
            Identify("A.a2", "C.u1"),
            Identify("C.e", "B.u2"),
        ],
    )
    calls: list = []
    steps = {n: _counting(n, calls) for n in "SABC"}
    traj = run(system, params, 4, KEY, steps, tick_order=order)
    assert calls == ["S", "A", "C", "B"] * 4
    # round t (from 1): C emits t % 3 and B reads that value, this round's
    assert np.asarray(traj["B"].received["u2"][:4]).tolist() == [1, 2, 0, 1]
