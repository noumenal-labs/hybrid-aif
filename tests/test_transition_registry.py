"""The registry of named transition functions: a nonlinear transition as data.

``register_transition("<package>.<name>", cls)`` lets :func:`describe` write a
:class:`~hybrid_aif.NonlinearGaussianTransition`'s ``f`` as its registered name and its fields,
and :func:`build` rebuild it from the registered class. These tests pin the registration rules,
the round trip (first order and unscented, discrete and continuous controls, Gaussian- and
mixture-carried, a merged composite), the refusals (an unregistered class, an unknown name, a
document whose fields differ from the registered class's, a field a document cannot write), the
static fields, and that only registered classes are instantiated. Every test runs against an
empty registry, restored afterwards.

Design: docs/design/nonlinear-transitions.md N5.
"""

import dataclasses
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import (
    Box,
    DescriptionError,
    Gaussian,
    NonlinearGaussianTransition,
    System,
    SystemParams,
    build,
    describe,
    register_transition,
    registered_transitions,
)
from hybrid_aif import description as _description
from hybrid_aif.composition import merge
from hybrid_aif.description import load, save
from hybrid_aif.description.mdl import description_length
from hybrid_aif.objectives import total
from tests.test_nonlinear_transition import (
    PLAN_H,
    PLAN_Q,
    PLAN_V,
    PLAN_W,
    Affine,
    RateUnicycle,
    SteeredUnicycle,
    Tabled,
    _affine_pair,
    _close,
    _noise,
    _steered,
    _steered_nonlinear,
)


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    """An empty registry for the test, the module's own restored afterwards."""
    monkeypatch.setattr(_description, "_TRANSITIONS", {})


class Drive(eqx.Module):
    """A map with a scalar leaf, an array leaf and static fields of every JSON kind."""

    h: float
    gains: jax.Array
    mode: str = eqx.field(static=True, default="rear")
    order: int = eqx.field(static=True, default=1)
    clip: float | None = eqx.field(static=True, default=None)
    axes: tuple[int, ...] = eqx.field(static=True, default=(0, 1))

    def __call__(self, x, controls):
        step = self.h * self.gains * self.order
        if self.clip is not None:
            step = jnp.clip(step, -self.clip, self.clip)
        return x + step * (1.0 if self.mode == "rear" else -1.0)


class OlderDrive(eqx.Module):
    """``Drive`` as an earlier version of a package wrote it: no ``gains``, a ``rate``."""

    h: float
    rate: float
    mode: str = eqx.field(static=True, default="rear")

    def __call__(self, x, controls):
        return x + self.h * self.rate


class StaticOnly(eqx.Module):
    """A map with no leaves: a document writes no parameter variable for it."""

    shift: float = eqx.field(static=True, default=0.25)

    def __call__(self, x, controls):
        return x + self.shift


class Inverted(eqx.Module):
    """``Drive``'s field names with ``mode`` a leaf and ``h`` static."""

    h: float = eqx.field(static=True)
    gains: jax.Array
    mode: jax.Array
    order: int = eqx.field(static=True, default=1)
    clip: float | None = eqx.field(static=True, default=None)
    axes: tuple[int, ...] = eqx.field(static=True, default=(0, 1))

    def __call__(self, x, controls):
        return x


def _single(f, propagation="first_order"):
    """An uncontrolled 3-d factor through ``f``."""
    system = System(B_deps=((),), factor_names=("pose",), policy_len=2)
    params = SystemParams(
        transition={
            "pose": NonlinearGaussianTransition(
                f, _noise(jnp.zeros((1, 3)), 0.1 * jnp.eye(3)), propagation
            )
        },
        initial_state_prior={"pose": Gaussian(loc=jnp.ones(3), chol=0.3 * jnp.eye(3))},
        state_targets={"pose": Gaussian(loc=jnp.zeros(3), chol=jnp.eye(3))},
    )
    return system, params


def _round_trip(system, params, **kwargs):
    """Describe, through JSON text, and build."""
    doc = describe(system, params, **kwargs)
    document = json.loads(json.dumps(doc.document))
    return document, build(document, arrays=doc.arrays)


def _same_model(system, params, built, built_params, actions, policies):
    """One step's beliefs and the scored policies' term trees agree."""
    beliefs = system.initial_priors(params)
    _close(built.initial_priors(built_params), beliefs)
    _close(
        built.update_empirical_prior(built_params, beliefs, actions),
        system.update_empirical_prior(params, beliefs, actions),
    )
    _close(
        built.score_policies(built_params, beliefs, policies),
        system.score_policies(params, beliefs, policies),
    )


# --------------------------------------------------------------------- registration


def test_names_are_namespaced():
    for name in ("unicycle", ".unicycle", "drive.", "drive..unicycle", "drive.uni-cycle", 3):
        with pytest.raises(ValueError, match="namespaced"):
            register_transition(name, Drive)
    assert register_transition("drive.unicycle", Drive) is Drive
    register_transition("acme.robots.drive_v2", OlderDrive)
    assert registered_transitions() == {"drive.unicycle": Drive, "acme.robots.drive_v2": OlderDrive}


def test_a_name_and_a_class_are_registered_once():
    register_transition("drive.unicycle", Drive)
    with pytest.raises(ValueError, match=r"'drive.unicycle' is already registered to .*Drive"):
        register_transition("drive.unicycle", OlderDrive)
    with pytest.raises(ValueError, match="already registered"):
        register_transition("drive.unicycle", Drive)
    with pytest.raises(ValueError, match=r"Drive is already registered as 'drive.unicycle'"):
        register_transition("other.unicycle", Drive)
    assert registered_transitions() == {"drive.unicycle": Drive}


def test_only_a_callable_equinox_module_class_registers():
    class Plain:
        def __call__(self, x, controls):
            return x

    class Inert(eqx.Module):
        h: float

    for bad in (Plain, Drive(0.1, jnp.ones(3)), lambda x, c: x, "Drive"):
        with pytest.raises(TypeError, match="equinox.Module subclass"):
            register_transition("drive.bad", bad)
    with pytest.raises(TypeError, match="defines no __call__"):
        register_transition("drive.inert", Inert)
    assert registered_transitions() == {}


def test_the_registration_records_the_field_schema():
    register_transition("drive.unicycle", Drive)
    entry = _description._TRANSITIONS["drive.unicycle"]
    assert entry.fields == (
        ("h", False),
        ("gains", False),
        ("mode", True),
        ("order", True),
        ("clip", True),
        ("axes", True),
    )


# ---------------------------------------------------------------------- round trip


@pytest.mark.parametrize("propagation", ["first_order", "unscented"])
def test_a_discrete_control_round_trips(propagation):
    """The steered unicycle (a discrete turn-rate level): the record names f and the
    propagation, and the built model's beliefs and term trees are the original's; the plan
    still turns toward the target behind."""
    register_transition("drive.steered", SteeredUnicycle)
    system, params = _steered(_steered_nonlinear(propagation))
    document, (built, built_params) = _round_trip(system, params)
    record = document["factors"]["pose"]["transition"]
    assert record["family"] == "NonlinearGaussianTransition"
    assert record["parents"] == ["pose", "u_pose"]
    assert record["options"] == {"f": {"name": "drive.steered"}, "propagation": propagation}
    assert set(document["variables"][record["parameters"]["f"]]) == {"h", "v", "rates"}
    rebuilt = built_params.transition["pose"]
    assert type(rebuilt.f) is SteeredUnicycle and rebuilt.propagation == propagation
    _same_model(
        system, params, built, built_params, {"a_pose": jnp.int32(2)}, system.policies(params)
    )
    q_pi = built.infer_policies(built_params, built.initial_priors(built_params))[0]
    assert int(built.policies(built_params)[int(jnp.argmax(q_pi.probs)), 0, 0]) == 2


@pytest.mark.parametrize("propagation", ["first_order", "unscented"])
def test_a_continuous_control_round_trips(propagation):
    register_transition("drive.rate", RateUnicycle)
    system, params = _steered(
        NonlinearGaussianTransition(
            RateUnicycle(PLAN_H, PLAN_V), _noise(jnp.zeros((1, 3)), PLAN_Q), propagation
        ),
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("w",),
        factor_names=("pose", "u_w"),
        planner="gradient",
    )
    params = dataclasses.replace(
        params, control={"u_w": Box(-PLAN_W * jnp.ones(1), PLAN_W * jnp.ones(1))}
    )
    _, (built, built_params) = _round_trip(system, params)
    plans = jnp.asarray([[[0.3], [-0.5], [0.9], [0.1]], [[0.0], [0.4], [-0.2], [1.2]]])
    _same_model(system, params, built, built_params, {"w": jnp.asarray([0.7])}, plans)
    beliefs = built.initial_priors(built_params)
    grad = jax.grad(lambda plan: total(built.score_policies(built_params, beliefs, plan)).midpoint)(
        jnp.full((4, 1), 0.3)
    )
    assert np.all(np.isfinite(np.asarray(grad)))


@pytest.mark.parametrize("propagation", ["first_order", "unscented"])
@pytest.mark.parametrize("carry", ["gaussian", "mixture"])
def test_discrete_and_continuous_controls_round_trip(carry, propagation):
    """The affine map with a discrete mode node (per-mode innovations) and a continuous node,
    Gaussian- and mixture-carried, its arrays through the arrays file."""
    register_transition("linear.affine", Affine)
    system, _, params = _affine_pair(carry, propagation)
    doc = describe(system, params, arrays="file")
    assert doc.arrays
    built, built_params = build(doc.document, arrays=doc.arrays)
    levels = jnp.asarray([[0, 1, 1], [1, 0, 1]], jnp.int32)
    values = jnp.asarray([[[0.3], [-0.5], [0.9]], [[0.0], [0.4], [-0.2]]])
    action = {"m": jnp.int32(1), "u": jnp.asarray([0.3])}
    _same_model(system, params, built, built_params, action, {"u_m": levels, "u_u": values})


def test_save_and_load_round_trip(tmp_path):
    register_transition("drive.steered", SteeredUnicycle)
    system, params = _steered(_steered_nonlinear())
    save(describe(system, params, inline_max=2), tmp_path / "robot")
    document, arrays = load(tmp_path / "robot.json")
    built, built_params = build(document, arrays=arrays)
    _same_model(
        system, params, built, built_params, {"a_pose": jnp.int32(0)}, system.policies(params)
    )


def test_static_fields_round_trip():
    """Every JSON kind of static field is written under options.f.static and read back as f
    held it (a tuple as a tuple), so the rebuilt f hashes and jits like the original."""
    register_transition("drive.drive", Drive)
    f = Drive(0.1, jnp.asarray([1.0, -2.0, 0.5]), mode="front", order=3, clip=0.2, axes=(2, 0))
    system, params = _single(f)
    document, (built, built_params) = _round_trip(system, params)
    record = document["factors"]["pose"]["transition"]
    assert record["options"]["f"] == {
        "name": "drive.drive",
        "static": {"mode": "front", "order": 3, "clip": 0.2, "axes": [2, 0]},
    }
    rebuilt = built_params.transition["pose"].f
    assert (rebuilt.mode, rebuilt.order, rebuilt.clip, rebuilt.axes) == ("front", 3, 0.2, (2, 0))
    _close(rebuilt, f)
    _same_model(system, params, built, built_params, {}, system.policies(params))
    step = jax.jit(lambda p, b: built.update_empirical_prior(p, b, {}))
    _close(
        step(built_params, built.initial_priors(built_params)),
        step(params, built.initial_priors(params)),
    )


def test_a_map_with_no_leaves_writes_no_parameter_variable():
    register_transition("drive.shift", StaticOnly)
    system, params = _single(StaticOnly(0.5))
    document, (built, built_params) = _round_trip(system, params)
    record = document["factors"]["pose"]["transition"]
    assert "f" not in record["parameters"]
    assert record["options"]["f"] == {"name": "drive.shift", "static": {"shift": 0.5}}
    _same_model(system, params, built, built_params, {}, system.policies(params))


def test_names_survive_a_merge():
    """A merged composite's record keeps its f's registered name under the prefixed child,
    and the composite rebuilds."""
    register_transition("drive.steered", SteeredUnicycle)
    robot, rp = _steered(_steered_nonlinear("unscented"), policy_len=1)
    twin, tp = _steered(_steered_nonlinear(), policy_len=1)
    composite, cp = merge({"robot": robot, "twin": twin}, {"robot": rp, "twin": tp})
    document, (built, built_params) = _round_trip(composite, cp)
    for name, propagation in (("robot", "unscented"), ("twin", "first_order")):
        record = document["factors"][f"{name}.pose"]["transition"]
        assert record["options"]["f"] == {"name": "drive.steered"}
        assert record["options"]["propagation"] == propagation
        assert record["parameters"]["f"] == f"{name}.pose_transition_f"
    action = {"robot.a_pose": jnp.int32(2), "twin.a_pose": jnp.int32(0)}
    _same_model(composite, cp, built, built_params, action, composite.policies(cp))


def test_description_length_reads_the_document():
    register_transition("drive.steered", SteeredUnicycle)
    system, params = _steered(_steered_nonlinear())
    doc = describe(system, params)
    from_document = description_length(doc.document, 50.0)
    from_system = description_length(system, 50.0, params=params)
    assert from_document.total == pytest.approx(from_system.total)


# ------------------------------------------------------------------------- refusals


def test_describe_refuses_an_unregistered_class():
    register_transition("drive.rate", RateUnicycle)
    system, params = _steered(_steered_nonlinear())
    with pytest.raises(
        DescriptionError,
        match=r"transition:pose/f: .*register_transition\('<package>.<name>', SteeredUnicycle\)",
    ):
        describe(system, params)


def test_describe_refuses_a_subclass_of_a_registered_class():
    """The registry is keyed by the exact class: a subclass is other code."""

    class Faster(SteeredUnicycle):
        pass

    register_transition("drive.steered", SteeredUnicycle)
    system, params = _steered(
        NonlinearGaussianTransition(
            Faster(PLAN_H, PLAN_V, jnp.zeros(3)), _noise(jnp.zeros((3, 3)), PLAN_Q)
        )
    )
    with pytest.raises(DescriptionError, match="register_transition"):
        describe(system, params)


def test_build_refuses_an_unknown_name(monkeypatch):
    register_transition("drive.steered", SteeredUnicycle)
    system, params = _steered(_steered_nonlinear())
    doc = describe(system, params)
    monkeypatch.setattr(_description, "_TRANSITIONS", {})
    with pytest.raises(
        DescriptionError,
        match=r"no transition function is registered as 'drive.steered'.*"
        r"register_transition\('drive.steered', <its class>\)",
    ):
        build(doc.document, arrays=doc.arrays)


def test_build_refuses_a_document_written_for_another_f(monkeypatch):
    """An older f under the same name: the field names that differ are named, and nothing is
    rebuilt."""
    register_transition("drive.drive", Drive)
    system, params = _single(Drive(0.1, jnp.ones(3)))
    doc = describe(system, params)
    monkeypatch.setattr(_description, "_TRANSITIONS", {})
    register_transition("drive.drive", OlderDrive)
    with pytest.raises(DescriptionError) as raised:
        build(doc.document, arrays=doc.arrays)
    message = str(raised.value)
    assert "differ from those of the class registered under it" in message
    assert "missing: ['rate']" in message
    assert "not fields of the class: ['axes', 'clip', 'gains', 'order']" in message
    assert "refused rather than rebuilt with code that means something else" in message


def test_build_refuses_a_static_and_leaf_mismatch(monkeypatch):
    register_transition("drive.drive", Drive)
    system, params = _single(Drive(0.1, jnp.ones(3)))
    doc = describe(system, params)
    monkeypatch.setattr(_description, "_TRANSITIONS", {})
    register_transition("drive.drive", Inverted)
    with pytest.raises(
        DescriptionError,
        match=r"written as leaves but static: \['h'\]; written as static but leaves: \['mode'\]",
    ):
        build(doc.document, arrays=doc.arrays)


def test_build_checks_the_rebuilt_f_against_the_layout():
    """The schema fixes names, not shapes: a leaf whose shape breaks f is caught when the built
    System is validated."""
    register_transition("drive.drive", Drive)
    system, params = _single(Drive(0.1, jnp.ones(3)))
    document = json.loads(json.dumps(describe(system, params).document))
    document["variables"]["pose_transition_f"]["gains"]["shape"] = [2]
    document["clamps"]["pose_transition_f"]["gains"] = [1.0, 1.0]
    with pytest.raises(DescriptionError, match="the System it describes is invalid.*f"):
        build(document)


def test_build_refuses_a_malformed_f_option():
    register_transition("drive.steered", SteeredUnicycle)
    system, params = _steered(_steered_nonlinear())
    document = json.loads(json.dumps(describe(system, params).document))
    record = document["factors"]["pose"]["transition"]
    for bad in ("drive.steered", {"name": "drive.steered", "code": "..."}, None):
        record["options"]["f"] = bad
        with pytest.raises(DescriptionError, match=r"options.f is \{'name'"):
            build(document)


def test_fields_a_document_cannot_write_are_refused():
    """An integer leaf, a static array and a static dict are named by field."""
    register_transition("drive.tabled", Tabled)
    system, params = _single(Tabled(0.1, jnp.ones((4, 3)), jnp.asarray(2, jnp.int32)))
    with pytest.raises(
        DescriptionError, match=r"transition:pose/f/index: f's leaf field holds a .* int32"
    ):
        describe(system, params)

    class Configured(eqx.Module):
        table: object = eqx.field(static=True)

        def __call__(self, x, controls):
            return x

    register_transition("drive.configured", Configured)
    for value in ({"a": 1}, (1, [2])):
        system, params = _single(Configured(value))
        with pytest.raises(
            DescriptionError, match=r"transition:pose/f/table: f's static field holds a"
        ):
            describe(system, params)


def test_build_refuses_a_malformed_clamp_and_static_values():
    """``f``'s clamp is field by field, and a static value must be one ``describe`` writes;
    both are refused as description errors naming the address."""
    register_transition("drive.drive", Drive)
    system, params = _single(Drive(0.1, jnp.asarray([1.0, 0.5, 0.0])))
    doc = describe(system, params)
    document = json.loads(json.dumps(doc.document))
    var = document["factors"]["pose"]["transition"]["parameters"]["f"]
    for bad in (1.0, [1.0, 2.0], {"ref": "arrays.npz#x"}):
        broken = json.loads(json.dumps(document))
        broken["clamps"][var] = bad
        with pytest.raises(DescriptionError, match="field by field"):
            build(broken, arrays=doc.arrays)
    for bad in ({"x": 1}, float("nan")):
        broken = json.loads(json.dumps(document))
        broken["factors"]["pose"]["transition"]["options"]["f"].setdefault("static", {})
        broken["factors"]["pose"]["transition"]["options"]["f"]["static"]["mode"] = bad
        with pytest.raises(DescriptionError, match="static field cannot take"):
            build(broken, arrays=doc.arrays)


def test_a_python_float_leaf_describes_the_same_after_a_rebuild():
    """A leaf is written at the precision the model computes with, so describing a rebuilt
    model writes the same document."""
    register_transition("drive.drive", Drive)
    system, params = _single(Drive(0.1, jnp.asarray([1.0, 0.5, 0.0])))
    document, (built, built_params) = _round_trip(system, params)
    again = json.loads(json.dumps(describe(built, built_params).document))
    assert again == document


def test_a_nonlinear_transition_reading_another_state_is_refused():
    """Its record reads its own previous value and control nodes only."""
    register_transition("drive.steered", SteeredUnicycle)
    robot, rp = _steered(_steered_nonlinear(), policy_len=1)
    twin, tp = _steered(_steered_nonlinear(), policy_len=1)
    composite, cp = merge({"robot": robot, "twin": twin}, {"robot": rp, "twin": tp})
    document = json.loads(json.dumps(describe(composite, cp).document))
    record = document["factors"]["robot.pose"]["transition"]
    record["parents"] = [record["parents"][0], "twin.pose", *record["parents"][1:]]
    with pytest.raises(DescriptionError, match="own previous value and control nodes only"):
        build(document)
