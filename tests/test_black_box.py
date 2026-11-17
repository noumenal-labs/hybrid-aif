"""A black box: an external process declared by its ports alone, coupled and run.

The pieces, in order: the constructor and what it holds; a black box coupled to a small planning
agent and run under the three orders; a black box with float ports; every refusal of a face
that needs the inside; ``describe`` then ``build``; the description length, the black box
unscored by name and left out of the totals; ``env_step``, the step of a positional
environment, against ``GenerativeEnv.wrap`` on the same environment.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Categorical, ConditionalCategorical
from hybrid_aif.composition import (
    Clamp,
    Identify,
    black_box,
    couple,
    env_step,
    is_black_box,
    joint_planner,
    merge,
    plan,
    run,
    sample,
    subsystem,
    with_mirror,
)
from hybrid_aif.controls import FLOAT, LEVEL
from hybrid_aif.description import build, describe
from hybrid_aif.description.mdl import description_length, log_star, parent_set_code
from hybrid_aif.envs import GenerativeEnv
from hybrid_aif.envs.generative import make
from hybrid_aif.reads import BLACK_BOX, Emission
from hybrid_aif.reference.envs.mountain_car import MountainCar
from hybrid_aif.system import System, SystemParams
from hybrid_aif.system.mirror import mirror
from tests.test_action_nodes import pusher

KEY = jax.random.PRNGKey(0)
ORDERS = ("simultaneous", "sequential", "async")


def box():
    """A black box with one level in-port and one level out-port, three levels each."""
    return black_box(ins={"u": (LEVEL, 3)}, outs={"o": (LEVEL, 3)})


def coupled():
    """A pusher (a one-step planner over its state ``x``) coupled to ``box()``: the pusher's
    action node drives the in-port, the out-port feeds the pusher's modality."""
    return couple(
        {"agent": pusher(), "world": box()},
        [Identify("agent.push", "world.u"), Identify("world.o", "agent.see")],
    )


def names_of(system):
    """A view's port names keyed by their own names."""
    return {n.rpartition(".")[2]: n for n in (*system.names.controls, *system.names.emissions)}


def drift(system, key, obs, carry, params):
    """The black box's step: a counter on three levels moved by the in-port (level 1 holds it),
    emitted as it stands; the carry is the counter."""
    port = names_of(system)
    if obs is None:
        start = jnp.asarray(0 if carry is None else carry, jnp.int32)
        return {port["o"]: start}, start, params
    x = ((carry + obs[port["u"]] - 1) % 3).astype(jnp.int32)
    return {port["o"]: x}, x, params


# ---------------------------------------------------------------------------- the declaration


def test_a_black_box_is_a_system_of_ports_alone_with_its_types_marked():
    system, params = black_box(
        ins={"v": (FLOAT, 2), "go": (LEVEL, 4)}, outs={"fix": (FLOAT, 3), "gauge": (LEVEL, 5)}
    )
    assert isinstance(system, System) and isinstance(params, SystemParams)
    assert is_black_box(system)
    assert system.names.states == () and system.names.modalities == ()
    assert system.names.controls == ("v", "go") and system.names.actions == (None, None)
    assert system.names.emissions == ("fix", "gauge")
    assert dict(system.emissions)["fix"] == Emission((), "sample", BLACK_BOX)
    assert dict(system.black_box) == {
        "v": (FLOAT, 2),
        "go": (LEVEL, 4),
        "fix": (FLOAT, 3),
        "gauge": (LEVEL, 5),
    }
    ports = system.ports(params)
    assert ports.ins == {"v": (FLOAT, 2), "go": (LEVEL, 4)}
    assert ports.outs == {"fix": (FLOAT, 3), "gauge": (LEVEL, 5)}
    assert params.transition == {} and params.control == {} and params.emission == {}
    #  the types are held in-ports first, in node order, then out-ports, whatever the order given
    given = System(
        B_deps=((),),
        state_names=(),
        control_names=("u",),
        controls=(0,),
        action_names=(None,),
        emissions={"o": Emission((), "sample", BLACK_BOX)},
        black_box={"o": (LEVEL, 2), "u": (FLOAT, 1)},
        policy_len=1,
    )
    assert given.black_box == (("u", (FLOAT, 1)), ("o", (LEVEL, 2)))
    assert not is_black_box(pusher()[0]) and not is_black_box(coupled()[0])


def test_the_constructor_refuses_what_is_not_a_set_of_typed_ports():
    with pytest.raises(ValueError, match="declares no port"):
        black_box()
    with pytest.raises(ValueError, match=r"names \['x'\] as both an in-port and an out-port"):
        black_box(ins={"x": (LEVEL, 2)}, outs={"x": (LEVEL, 2)})
    with pytest.raises(ValueError, match=r"a port's type is \('level', levels\)"):
        black_box(ins={"x": ("int", 2)})
    with pytest.raises(ValueError, match="has size 0"):
        black_box(outs={"x": (FLOAT, 0)})
    with pytest.raises(ValueError, match="without a dot"):
        black_box(outs={"a.b": (FLOAT, 1)})
    #  a System that states more than ports is no black box
    with pytest.raises(ValueError, match="a black box states nothing but its ports"):
        System(
            B_deps=((), ()),
            state_names=("x",),
            control_names=("u",),
            controls=(1,),
            action_names=(None,),
            black_box={"u": (LEVEL, 2)},
            policy_len=1,
        )
    with pytest.raises(ValueError, match="and black_box gives them no type"):
        System(B_deps=(), emissions={"o": Emission((), "sample", BLACK_BOX)}, policy_len=1)
    with pytest.raises(ValueError, match="has no record|names parents"):
        System(
            B_deps=((),),
            state_names=(),
            control_names=("u",),
            controls=(0,),
            action_names=(None,),
            emissions={"o": Emission(("u",), "sample", BLACK_BOX)},
            black_box={"u": (LEVEL, 2), "o": (LEVEL, 2)},
            policy_len=1,
        )


# ------------------------------------------------------------------------ coupled and run


def test_couple_binds_its_ports_by_identify_and_types_them():
    system, params = coupled()
    assert system._black_boxes == ("world",)
    assert dict(system.black_box) == {"world.u": (LEVEL, 3), "world.o": (LEVEL, 3)}
    assert dict(system.observes) == {"agent.see": "world.o", "world.u": "agent.push"}
    assert system.ports(params).outs["world.o"] == (LEVEL, 3)
    view, _ = subsystem(system, params, "world")
    assert is_black_box(view) and view.names.controls == ("world.u",)
    #  the types of the two ends must agree, as for any port
    with pytest.raises(ValueError, match="world.o"):
        couple(
            {"agent": pusher(), "world": black_box(ins={"u": (LEVEL, 3)}, outs={"o": (LEVEL, 4)})},
            [Identify("agent.push", "world.u"), Identify("world.o", "agent.see")],
        )


@pytest.mark.parametrize("order", ORDERS)
def test_a_black_box_coupled_to_a_planning_agent_runs_under_every_order(order):
    system, params = coupled()
    rec = run(
        system,
        params,
        6,
        KEY,
        {"agent": plan(), "world": drift},
        tick_order=order,
        carries={"world": 1},
        record_carries=True,
    )
    world, agent = rec["world"], rec["agent"]
    #  the reset call stands entering round 0: the seeded counter, emitted
    assert int(world.start["o"]) == 1 and int(world.carries[0]) == 1
    #  the in-port reads the agent's action as realised this round (the world steps after it)
    np.testing.assert_array_equal(world.received["u"][:6], agent.emitted["push"])
    assert bool(world.present["u"][:6].all())
    #  the counter moves by the in-port, and its out-port is the counter
    counts = np.asarray([int(c) for c in world.carries])
    pushes = np.asarray(agent.emitted["push"])
    np.testing.assert_array_equal(counts[1:], (counts[:-1] + pushes - 1) % 3)
    np.testing.assert_array_equal(world.emitted["o"], counts[1:])
    #  the agent reads the out-port as it stands at its turn, before the world steps: the
    #  counter the round began with
    np.testing.assert_array_equal(agent.received["see"], counts)
    assert bool(world.decided.all()) and rec.params is not None


def test_two_black_boxes_with_float_ports_drive_each_other():
    """A float out-port drives a float in-port of another black box, both ways: one box a
    controller, the other a point that integrates its command."""
    controller = black_box(ins={"x": (FLOAT, 2)}, outs={"force": (FLOAT, 2)})
    plant = black_box(ins={"force": (FLOAT, 2)}, outs={"x": (FLOAT, 2)})
    system, params = couple(
        {"brain": controller, "body": plant},
        [Identify("brain.force", "body.force"), Identify("body.x", "brain.x")],
    )
    assert system.ports(params).outs == {"brain.force": (FLOAT, 2), "body.x": (FLOAT, 2)}

    def pull(system, key, obs, carry, params):
        port = names_of(system)
        x = jnp.zeros(2) if obs is None or obs[port["x"]] is None else obs[port["x"]]
        return {port["force"]: -0.5 * x + jnp.asarray([1.0, 0.0])}, None, params

    def integrate(system, key, obs, carry, params):
        port = names_of(system)
        if obs is None:
            return {port["x"]: jnp.zeros(2)}, jnp.zeros(2), params
        x = carry + obs[port["force"]]
        return {port["x"]: x}, x, params

    for order in ORDERS:
        rec = run(system, params, 4, KEY, {"brain": pull, "body": integrate}, tick_order=order)
        brain, body = rec["brain"], rec["body"]
        assert body.emitted["x"].shape == (4, 2) and brain.received["x"].dtype == jnp.float32
        #  each box integrates or reacts to what its in-port received
        x = np.concatenate([np.asarray(body.start["x"])[None], np.asarray(body.emitted["x"])])
        np.testing.assert_allclose(x[1:], x[:-1] + np.asarray(body.received["force"][:4]))
        np.testing.assert_allclose(
            brain.emitted["force"], 1.0 * np.asarray([1.0, 0.0]) - 0.5 * brain.received["x"][:4]
        )
    #  under simultaneous both are sources, so each in-port reads the other's out-port at the
    #  round's boundary: the value it held when the round began
    rec = run(system, params, 3, KEY, {"brain": pull, "body": integrate})
    np.testing.assert_array_equal(rec["body"].received["force"][0], rec["brain"].start["force"])
    np.testing.assert_array_equal(
        rec["body"].received["force"][1:3], rec["brain"].emitted["force"][:2]
    )


# ------------------------------------------------------------------------------- refusals


def test_plan_and_sample_on_a_black_box_are_refused_up_front_naming_a_custom_step():
    system, params = coupled()
    with pytest.raises(ValueError, match=r"steps\['world'\] is plan\(\).*custom step"):
        run(system, params, 2, KEY, {"agent": plan(), "world": plan()})
    with pytest.raises(ValueError, match=r"steps\['world'\] is sample\(\).*env_step"):
        run(system, params, 2, KEY, {"agent": plan(), "world": sample()})

    def sliced(system, key, obs, carry, params):
        return drift(system, key, obs, carry, params)

    sliced.sliced = True
    with pytest.raises(ValueError, match="is marked sliced, and 'world' is a black box"):
        run(system, params, 2, KEY, {"agent": plan(), "world": sliced})


def test_a_joint_planner_over_a_black_box_is_refused():
    system, params = coupled()
    with pytest.raises(ValueError, match="holds the black box 'world'"):
        joint_planner(system, params)
    nested = couple({"both": (system, params)}, [])
    with pytest.raises(ValueError, match="planned as one mind, and the black box 'both.world'"):
        run(*nested, 2, KEY, {"both": plan()})


def test_an_undriven_in_port_and_a_clamped_one_are_refused():
    system, params = couple({"agent": pusher(), "world": box()}, [Identify("world.o", "agent.see")])
    with pytest.raises(ValueError, match="'world.u' is an in-port of a black box and no variable"):
        run(system, params, 2, KEY, {"agent": plan(), "world": drift})
    with pytest.raises(ValueError, match="in-port of a black box, which a Clamp"):
        couple({"agent": pusher(), "world": box()}, [Clamp("world.u", 1)])


def test_every_face_that_needs_the_inside_refuses_a_black_box_and_a_system_holding_one():
    bare, bare_params = box()
    system, params = coupled()
    for s, p, which in ((bare, bare_params, "is a black box"), (system, params, "holds")):
        with pytest.raises(ValueError, match=f"mirror: this System {which}"):
            mirror(s, p)
        with pytest.raises(ValueError, match=f"as_env: this System {which}"):
            s.as_env(p)
        with pytest.raises(ValueError, match=f"sample_initial: this System {which}"):
            s.sample_initial(KEY, p)
        with pytest.raises(ValueError, match=f"sample_step: this System {which}"):
            s.sample_step(KEY, p, {})
        with pytest.raises(ValueError, match=f"infer_states: this System {which}"):
            s.infer_states({}, p)
        with pytest.raises(ValueError, match=f"act: this System {which}"):
            s.act(KEY, p, {})
    with pytest.raises(ValueError, match="with_mirror: this System holds"):
        with_mirror(system, params)
    with pytest.raises(ValueError, match="with_mirror: this System is a black box"):
        with_mirror(bare, bare_params)
    with pytest.raises(ValueError, match="a sliced view .* nothing inside a black box"):
        subsystem(system, params, "world", inputs={"agent.push": 1})


# -------------------------------------------------------------------------------- describe


def test_describe_writes_the_ports_and_build_rebuilds_the_black_box_exactly():
    bare, bare_params = black_box(
        ins={"v": (FLOAT, 2), "go": (LEVEL, 4)}, outs={"fix": (FLOAT, 3), "gauge": (LEVEL, 5)}
    )
    document, arrays = describe(bare, bare_params)
    assert document["relations"]["black_box"] == {"ins": ["v", "go"], "outs": ["fix", "gauge"]}
    assert document["variables"]["v"] == {"domain": "real", "dim": 2}
    assert document["variables"]["gauge"] == {"domain": "categorical", "levels": 5}
    assert document["factors"] == {} and arrays == {}
    rebuilt, rebuilt_params = build(document, arrays=arrays)
    assert rebuilt == bare and rebuilt_params == bare_params
    assert describe(rebuilt, rebuilt_params)[0] == document

    system, params = coupled()
    document, arrays = describe(system, params)
    assert document["relations"]["black_box"] == {"ins": ["world.u"], "outs": ["world.o"]}
    assert document["relations"]["driven"]["world.u"] == "agent.push"
    assert "world.o" not in document["factors"]  # an out-port has no record
    rebuilt, rebuilt_params = build(document, arrays=arrays)
    assert rebuilt == system
    assert jax.tree.all(
        jax.tree.map(lambda a, b: bool(jnp.array_equal(a, b)), rebuilt_params, params)
    )
    assert describe(rebuilt, rebuilt_params)[0] == document
    #  the rebuilt coupling runs as the original: the step is the caller's, not the document's
    one = run(system, params, 3, KEY, {"agent": plan(), "world": drift})
    two = run(rebuilt, rebuilt_params, 3, KEY, {"agent": plan(), "world": drift})
    np.testing.assert_array_equal(one["world"].emitted["o"], two["world"].emitted["o"])


def test_a_document_with_a_record_on_a_port_or_a_box_on_an_in_port_is_refused():
    document, arrays = describe(*box())
    boxed = dict(document["variables"])
    boxed["u"] = {"domain": "real", "dim": 1, "low": -1.0, "high": 1.0}
    with pytest.raises(Exception, match="has no box; drop low and high"):
        build({**document, "variables": boxed})
    recorded = {
        **document,
        "factors": {"o": {"actuator": {"parents": [], "options": {"kernel": "identity"}}}},
    }
    with pytest.raises(Exception, match="is a port of a black box .* has a record"):
        build(recorded)


# ------------------------------------------------------------------------ description length


def test_the_description_length_prices_the_ports_and_lists_the_inside_unscored():
    system, params = coupled()
    dl = description_length(system, 100.0, params=params)
    link = dl.per_address["black_box:world"]
    assert (link.graph, link.family, link.prior, link.posterior) == (None, None, None, None)
    assert "unscored, not zero" in link.note and "world.u level 3" in link.note
    assert {name for address, name in dl.unscored if address == "black_box:world"} == {
        "graph",
        "family",
        "prior",
        "posterior",
    }
    assert dl.exact is False
    #  the totals are the other addresses' alone
    others = [v for k, v in dl.per_address.items() if k != "black_box:world"]
    assert dl.structure == pytest.approx(sum(v.structure for v in others), abs=1e-9)
    assert dl.total == pytest.approx(dl.structure + dl.parameters, abs=1e-9)
    #  the out-port is declared structure: an emission of the graph code with its size code; the
    #  driven in-port is a slot of the agent's action node, priced there
    port = dl.per_address["emission:world.o"]
    assert port.graph == pytest.approx(parent_set_code(0, 2) + log_star(3))
    assert "out-port of a black box" in port.note
    assert "control:world.u" not in dl.per_address
    assert "0 modalities, 1 control nodes, 1 emissions" in dl.per_address["graph"].note
    #  the coupling prices at or above the agent alone: every variable it observes is coded once
    agent = description_length(pusher()[0], 100.0, params=pusher()[1])
    assert dl.total >= agent.total
    #  the same from the document
    document, arrays = describe(system, params)
    again = description_length(document, 100.0, arrays=arrays)
    assert again.total == dl.total and again.unscored == dl.unscored


@pytest.mark.parametrize("form", ["bare", "coupled", "merged"])
def test_a_system_of_black_boxes_alone_totals_its_port_codes(form):
    """No graph record (no state factor to code), each box's inside unscored by name, and the
    totals its ports' codes: from the System and from its document alike."""
    if form == "bare":
        system, params = box()
        boxes = ["black_box"]
    elif form == "coupled":
        system, params = couple(
            {"a": box(), "b": box()}, [Identify("a.o", "b.u"), Identify("b.o", "a.u")]
        )
        boxes = ["black_box:a", "black_box:b"]
    else:
        (a, ap), (b, bp) = box(), box()
        system, params = merge({"a": a, "b": b}, {"a": ap, "b": bp})
        boxes = ["black_box:a", "black_box:b"]
    dl = description_length(system, 100.0, params=params)
    assert "graph" not in dl.per_address
    assert all(b in dl.per_address for b in boxes)
    assert {a for a, _ in dl.unscored} == set(boxes) and dl.exact is False
    ports = [v for k, v in dl.per_address.items() if not k.startswith("black_box")]
    assert ports and dl.total == pytest.approx(sum(v.graph for v in ports), abs=1e-12)
    assert dl.total > 0.0 and dl.parameters == 0.0
    document, arrays = describe(system, params)
    assert description_length(document, 100.0, arrays=arrays).total == dl.total


# ------------------------------------------------------------------------------ env_step


def _tabular_env():
    """A positional three-state process moved by a three-level action, seen through a noisy
    table: the environment both adapters run."""
    A = (0.8 * jnp.eye(3) + 0.2 / 3,)
    moves = jnp.stack([jnp.roll(jnp.eye(3), k - 1, axis=0) for k in range(3)], axis=-1)
    B = (moves * 0.9 + 0.1 / 3,)
    D = (jnp.asarray([0.5, 0.3, 0.2]),)
    return make(A, B, D)


def test_env_step_is_the_wrapped_environment_bitwise_on_the_same_keys():
    env, env_params = _tabular_env()
    named = System(
        A_deps=((0,),),
        B_deps=((0,),),
        controlled=(True,),
        state_names=("x",),
        modality_names=("see",),
        policy_len=1,
    )
    (a,), (b,), (d,) = env_params.likelihood, env_params.transition, env_params.initial_state_prior
    named_params = SystemParams(
        likelihood={"see": ConditionalCategorical(log_table=jnp.log(a), parent_ndims=1)},
        transition={"x": ConditionalCategorical(log_table=jnp.log(b), parent_ndims=2)},
        initial_state_prior={"x": Categorical(logits=jnp.log(d))},
    )
    wrapped = GenerativeEnv.wrap(env, named, named_params)
    system, params = couple(
        {"agent": pusher(), "world": black_box(ins={"a_x": (LEVEL, 3)}, outs={"see": (LEVEL, 3)})},
        [Identify("agent.push", "world.a_x"), Identify("world.see", "agent.see")],
    )
    keyed = named.from_positional("params", env_params)
    calls = []
    inner = env_step(env, env_params)

    def logged(view, key, obs, carry, view_params):
        out, new, back = inner(view, key, obs, carry, view_params)
        #  logged on the host: the rounds after the first run in one compiled scan
        jax.debug.callback(lambda *a: calls.append(a), key, obs, carry, out, new, ordered=True)
        return out, new, back

    run(system, params, 8, KEY, {"agent": plan(), "world": logged}, tick_order="sequential")
    assert len(calls) == 9
    key, obs, carry, out, state = calls[0]
    seen, held = wrapped.reset(key, keyed)
    assert int(out["world.see"]) == int(seen["see"]) and int(state[0]) == int(held["x"])
    for key, obs, carry, out, state in calls[1:]:
        before = {"x": carry[0]}
        seen, held = wrapped.step(key, before, {"a_x": obs["world.a_x"]}, keyed)
        assert int(out["world.see"]) == int(seen["see"]) and int(state[0]) == int(held["x"])


def test_env_step_runs_a_continuous_environment_through_float_ports():
    car = MountainCar()
    controller = black_box(ins={"x": (FLOAT, 2)}, outs={"force": (FLOAT, 1)})
    plant = black_box(ins={"force": (FLOAT, 1)}, outs={"x": (FLOAT, 2)})
    system, params = couple(
        {"brain": controller, "car": plant},
        [Identify("brain.force", "car.force"), Identify("car.x", "brain.x")],
    )

    def push_right(system, key, obs, carry, params):
        return {names_of(system)["force"]: jnp.ones(1)}, None, params

    rec = run(system, params, 5, KEY, {"brain": push_right, "car": env_step(car)})
    #  the same physics stepped by hand on the same start
    k_reset = jax.random.split(KEY)[1]
    _, state = car.reset(jax.random.fold_in(k_reset, 1))
    for t in range(5):
        _, state = car.step(None, state, jnp.ones(1))
        np.testing.assert_array_equal(rec["car"].emitted["x"][t], state[0])
    with pytest.raises(TypeError, match="pass its .inner"):
        env_step(GenerativeEnv.wrap(*_tabular_env()[:1], pusher()[0]))


def test_env_step_refuses_a_node_that_is_not_a_black_box_and_a_mismatched_environment():
    env, env_params = _tabular_env()
    system, params = couple(
        {
            "agent": pusher(),
            "world": black_box(ins={"a_x": (LEVEL, 3)}, outs={"see": (LEVEL, 3), "x": (LEVEL, 2)}),
        },
        [Identify("agent.push", "world.a_x"), Identify("world.see", "agent.see")],
    )
    with pytest.raises(ValueError, match=r"returned 1 observation\(s\), and the black box decl"):
        run(system, params, 1, KEY, {"agent": plan(), "world": env_step(env, env_params)})
    with pytest.raises(ValueError, match="and the node it was given is not one"):
        run(*coupled(), 1, KEY, {"agent": env_step(env, env_params), "world": drift})
    with pytest.raises(ValueError, match="names \\['nope'\\], which are no in-port"):
        run(
            *coupled(),
            1,
            KEY,
            {"agent": plan(), "world": env_step(env, env_params, actions={"nope": 0})},
        )


# ------------------------------------------------------------------- a host-side process


class _Sensor:
    """A stand-in for hardware outside the library's computation graph: a Python iterator of
    NumPy readings, and a Python list the step writes its commands to."""

    def __init__(self):
        self.readings = iter(np.arange(100) % 3)
        self.sent: list[int] = []


def host_step(sensor):
    """A black box's step in host Python only: NumPy and Python numbers, a side effect on a
    list, a read from an iterator, a carry that is a plain Python object."""

    def step(system, key, obs, carry, params):
        port = names_of(system)
        if obs is None:
            return {port["o"]: int(next(sensor.readings))}, {"log": []}, params
        command = int(np.asarray(obs[port["u"]]))  # a concrete value: the step is never traced
        sensor.sent.append(command)
        carry["log"].append(command)
        return {port["o"]: np.int64(next(sensor.readings))}, carry, params

    return step


@pytest.mark.parametrize("order", ORDERS)
def test_a_black_box_step_in_host_python_is_refused_naming_the_step(order):
    """Every round runs in one compiled scan from the reset (design batched-run.md R1, R8, and
    Phase C: round 0 included), so a step is a traced function: a black box's step that reads a
    value on the host (here ``int(np.asarray(...))`` of the command it is handed) is refused
    naming the step and the remedy, in a run of one round too. Its reset, unbatched, runs on
    concrete values and may read the host (here an iterator)."""
    system, params = coupled()
    for ticks in (5, 1):
        with pytest.raises(
            ValueError, match=r"the step of 'world' calls host-side code on a traced value"
        ):
            steps = {"agent": plan(), "world": host_step(_Sensor())}
            run(system, params, ticks, KEY, steps, tick_order=order)
    sensor = _Sensor()
    steps = {"agent": plan(), "world": host_step(sensor)}
    rec = run(system, params, 0, KEY, steps, tick_order=order)
    assert sensor.sent == [] and int(rec["world"].start["o"]) == 0


def test_a_host_step_returning_the_wrong_shape_or_type_is_refused_naming_the_port():
    system, params = coupled()

    def wrong(value):
        def step(system, key, obs, carry, params):
            return {names_of(system)["o"]: value}, carry, params

        return step

    level = r"black box's out-port of 3 levels, so return one integer level in \[0, 3\)"
    for value in (np.float64(1.0), 1.0, np.asarray([0, 1]), np.asarray([0.2, 0.3, 0.5]), True):
        with pytest.raises(ValueError, match=r"returned 'world\.o' as .*" + level):
            run(system, params, 1, KEY, {"agent": plan(), "world": wrong(value)})
    with pytest.raises(ValueError, match=r"returned 'world\.o' = 7; .*" + level):
        run(system, params, 1, KEY, {"agent": plan(), "world": wrong(7)})
    floats = couple(
        {
            "brain": black_box(outs={"force": (FLOAT, 2)}),
            "body": black_box(ins={"force": (FLOAT, 2)}),
        },
        [Identify("brain.force", "body.force")],
    )

    def push(value):
        def step(system, key, obs, carry, params):
            return {names_of(system)["force"]: value}, carry, params

        return step

    def idle(system, key, obs, carry, params):
        return {}, carry, params

    width = r"float out-port of width 2, so return a vector of shape \(2,\) of numbers"
    for value in (np.zeros(3), 1.0, np.asarray(["a", "b"]), np.ones((2, 1))):
        with pytest.raises(ValueError, match=r"returned 'brain\.force' as .*" + width):
            run(*floats, 1, KEY, {"brain": push(value), "body": idle})


def test_a_box_out_port_value_is_normalised_to_one_dtype_across_rounds():
    """Any integer type on a level port is stored as int32, an integer vector on a float port as
    the float dtype, one number on a width-1 float port as a vector; the reset's value, round 0's
    (concrete) and the compiled rounds' (traced) stack in one dtype."""
    system, params = coupled()

    def mixed(system, key, obs, carry, params):
        t = jnp.asarray(0) if carry is None else carry
        if obs is None:
            return {names_of(system)["o"]: np.uint8(1)}, t, params
        return {names_of(system)["o"]: ((t + 2) % 3).astype(jnp.int16)}, t + 1, params

    rec = run(system, params, 5, KEY, {"agent": plan(), "world": mixed})
    assert rec["world"].emitted["o"].dtype == jnp.int32
    np.testing.assert_array_equal(rec["world"].emitted["o"], [2, 0, 1, 2, 0])
    assert int(rec["world"].start["o"]) == 1
    floats = couple(
        {
            "brain": black_box(outs={"force": (FLOAT, 1)}),
            "body": black_box(ins={"force": (FLOAT, 1)}),
        },
        [Identify("brain.force", "body.force")],
    )

    def push(system, key, obs, carry, params):
        t = jnp.asarray(0) if carry is None else carry
        if obs is None:
            return {names_of(system)["force"]: np.asarray([3])}, t, params
        return {names_of(system)["force"]: (t + 1) * 0.5}, t + 1, params  # one number

    def idle(system, key, obs, carry, params):
        return {}, carry, params

    rec = run(*floats, 4, KEY, {"brain": push, "body": idle})
    emitted = rec["brain"].emitted["force"]
    assert emitted.dtype == jnp.float32 and emitted.shape == (4, 1)
    np.testing.assert_array_equal(emitted[:, 0], [0.5, 1.0, 1.5, 2.0])
    assert rec["body"].received["force"].dtype == jnp.float32
    np.testing.assert_array_equal(rec["brain"].start["force"], [3.0])


# ------------------------------------------------------------------------------- fix pass


def test_a_box_nested_in_a_merge_gives_the_coupling_a_directly_coupled_box_gives():
    (b, bp), (a, ap) = box(), pusher()
    nested = merge({"world": b}, {"world": bp})
    bindings = [Identify("agent.push", "env.world.u"), Identify("env.world.o", "agent.see")]
    system, params = couple({"agent": (a, ap), "env": nested}, bindings)
    direct, _ = coupled()
    assert system.epistemic == direct.epistemic is False
    document, arrays = describe(system, params)
    rebuilt, rebuilt_params = build(document, arrays=arrays)
    assert rebuilt.epistemic == system.epistemic
    assert describe(rebuilt, rebuilt_params)[0] == document


def test_a_merge_of_two_boxes_round_trips_exactly():
    (a, ap), (b, bp) = box(), box()
    system, params = merge({"a": a, "b": b}, {"a": ap, "b": bp}, [Identify("a.o", "b.u")])
    assert is_black_box(system)  # a System stating nothing but black-box ports
    document, arrays = describe(system, params)
    rebuilt, _ = build(document, arrays=arrays)
    assert rebuilt == system
    assert [sub.boundary for _, sub in rebuilt.subsystems] == ["none", "none"]


def test_the_faces_off_the_refusal_list_speak_to_a_box():
    from hybrid_aif.replay import replay

    bare, bare_params = box()
    with pytest.raises(ValueError, match="replay: this System is a black box"):
        replay(bare, bare_params, {}, {})
    for s, p in (box(), coupled()):
        with pytest.raises(ValueError, match="beliefs_from_states: this System"):
            s.beliefs_from_states(p, None)
    env, _ = _tabular_env()
    with pytest.raises(TypeError, match="env_step"):
        GenerativeEnv.wrap(env, bare)
    with pytest.raises(ValueError, match="infer_states: this System is a black box"):
        bare.compiled().infer_states(bare_params, {})


def test_clocks_naming_a_box_is_refused_with_a_sentence_for_a_box():
    system, params = coupled()
    with pytest.raises(ValueError, match="a black box, which declares no durations.*own step"):
        run(system, params, 2, KEY, {"agent": plan(), "world": drift}, clocks={"world": 2})


def test_env_step_refuses_a_level_out_of_range_and_an_uncontrolled_factor():
    from hybrid_aif.reference.envs.tmaze import tmaze

    env, env_params = tmaze()
    brain = black_box(outs={"a": (LEVEL, 7)})

    def say(level):
        def step(system, key, obs, carry, params):
            return {names_of(system)["a"]: level}, carry, params

        return step

    world = black_box(
        ins={"a_loc": (LEVEL, 7)}, outs={"loc": (LEVEL, 5), "r": (LEVEL, 3), "c": (LEVEL, 3)}
    )
    system, params = couple({"brain": brain, "world": world}, [Identify("brain.a", "world.a_loc")])
    # every round is traced (design batched-run.md Phase C): the level is checked as the round
    # runs, and the run stops with the sentence
    with pytest.raises(Exception, match="holds a level outside .*factor 0 .* takes 5 action"):
        run(system, params, 1, KEY, {"brain": say(6), "world": env_step(env, env_params)})
    uncontrolled = env_step(env, env_params, actions={"a_loc": 1})
    with pytest.raises(ValueError, match="factor 1, which the environment's transitions take no"):
        run(system, params, 1, KEY, {"brain": say(1), "world": uncontrolled})


def test_numpy_integer_sizes_and_a_box_reading_its_own_output():
    system, _ = black_box(ins={"u": (LEVEL, np.int64(3))}, outs={"o": (FLOAT, np.int32(2))})
    assert system.black_box == (("u", (LEVEL, 3)), ("o", (FLOAT, 2)))
    assert all(type(size) is int for _, (_, size) in system.black_box)
    with pytest.raises(ValueError, match="keeps in its step's carry"):
        couple({"world": box()}, [Identify("world.o", "world.u")])
