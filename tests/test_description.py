"""The model description: ``describe`` / ``build`` / ``schema``.

The format: ``docs/description-format.md``; design: ``docs/design/model-description-0.3.md``.

Three contracts. **Round trip**: ``build(*describe(system, params))`` is the declaration back,
bitwise (every static field, every array leaf, every dtype) on every declared model the repo
ships, from the T-maze to the reference agent with its nested preference model; a composite's
per-component record carries what the document shows. **The document is validated before
anything is built**: each fault is one sentence naming the address and what to write instead.
**The schema is the contract**: every document validates, and the checked-in copy equals what the
code generates.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import pathlib
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hybrid_aif as H
from hybrid_aif import description as D
from hybrid_aif.composition.driver import agent_respond, ancestral_respond, run
from hybrid_aif.composition.merge import subsystem
from hybrid_aif.description import (
    Description,
    DescriptionError,
    build,
    describe,
    load,
    save,
    schema,
)
from hybrid_aif.objectives import registry
from hybrid_aif.reference.raif.agent import raif_mountain_car
from hybrid_aif.reference.raif.world import tiny_rssm
from tests.test_d_deps import hierarchical_tmaze
from tests.test_h_deps import two_level
from tests.test_merge import hand_flattened, merged
from tests.test_system import tmaze_system

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_EXAMPLES = _ROOT / "examples"
KEY = jax.random.PRNGKey(0)


def _example(name: str):
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def models() -> dict[str, tuple[H.System, H.SystemParams]]:
    """Every shipped declaration: the T-maze, the probe's model, the linear-Gaussian and switching
    demos, the noise learner, the tiny RSSM, the reference agent and the merged composite."""
    compose = _example("compose_target_factor")
    switching = _example("hybrid_switching")
    noise = _example("noise_learning")
    agent = raif_mountain_car(KEY)
    return {
        "tmaze": tmaze_system(),
        "compose": (compose.system, compose.params),
        "hill_lg": _example("hill_lg").make_agent(),
        "switching_filter": _example("switching_filter").make_agent(),
        "hybrid_switching": (switching.SYSTEM, switching.PARAMS),
        "noise_learning": (noise.make_system(), noise.make_params(jnp.asarray([[[0.1]], [[0.2]]]))),
        "tiny_rssm": tiny_rssm(2, 1, key=KEY),
        "raif_mountain_car": (agent.system, agent.params),
        "merged": merged(),
        "hierarchical_tmaze": hierarchical_tmaze(),
        "two_level": two_level(),
    }


NAMES = (
    "tmaze",
    "compose",
    "hill_lg",
    "switching_filter",
    "hybrid_switching",
    "noise_learning",
    "tiny_rssm",
    "raif_mountain_car",
    "merged",
    "hierarchical_tmaze",
    "two_level",
)


def _shown(system: H.System) -> H.System:
    """``system`` with each component record cut to what a document shows: no relayed,
    terminated or identified nodes, and ``actions`` naming only the control nodes present."""
    if system.subsystems is None:
        return system
    present = set(system.names.controls)
    shown = tuple(
        (
            name,
            sub._replace(
                relayed=(),
                terminated=(),
                identified=(),
                actions=tuple(pair for pair in sub.actions if pair[0] in present),
            ),
        )
        for name, sub in system.subsystems
    )
    return dataclasses.replace(system, subsystems=shown)


def _assert_same(a, b) -> None:
    """Bitwise: identical treedef (every static field) and identical leaves, dtypes included."""
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb, strict=True):
        x, y = np.asarray(x), np.asarray(y)
        assert x.dtype == y.dtype
        assert np.array_equal(x, y)


def _through_json(document: dict) -> dict:
    return json.loads(json.dumps(document))


def _round_trip(system, params, **kwargs):
    document, arrays = describe(system, params, **kwargs)
    return build(_through_json(document), arrays=arrays)


# ================================================================== the round trip


@pytest.mark.parametrize("name", NAMES)
def test_the_round_trip_is_bitwise(models, name):
    system, params = models[name]
    rebuilt_system, rebuilt_params = _round_trip(system, params)
    _assert_same(_shown(system), rebuilt_system)
    _assert_same(params, rebuilt_params)


@pytest.mark.parametrize("name", NAMES)
def test_describing_a_built_document_writes_the_same_document(models, name):
    system, params = models[name]
    document, arrays = describe(system, params)
    document = _through_json(document)
    again = _through_json(describe(*build(document, arrays=arrays)).document)
    assert again == document


def test_a_uniform_anchor_round_trips():
    """A scalar ``alpha0`` (the uniform decay anchor) is a hyperparameter like any other."""
    from tests.test_learn import _sense_system

    system, params = _sense_system()
    sense = H.ConditionalDirichlet.from_table(
        params.likelihood["sense"].mean(), parent_ndims=1, strength=3.0, anchor="uniform"
    )
    params = dataclasses.replace(params, likelihood={**params.likelihood, "sense": sense})
    for mode in ("inline", "file"):
        _, rebuilt_params = _round_trip(system, params, arrays=mode)
        again = rebuilt_params.likelihood["sense"]
        assert again.alpha0.shape == () and float(again.alpha0) == 1.0
        _assert_same(params, rebuilt_params)


def test_a_dotted_system_without_its_record_comes_back_with_the_record_filled():
    """The dot is reserved to composite prefixes, so dotted names are a composite's: a
    hand-declared System with dotted names builds back with each component's record."""
    system, params = hand_flattened()
    assert system.subsystems is None
    rebuilt, rebuilt_params = _round_trip(system, params)
    assert [name for name, _ in rebuilt.subsystems] == ["agent", "world"]
    assert dict(rebuilt.subsystems)["agent"].factors == (0, 1)
    _assert_same(params, rebuilt_params)
    assert dataclasses.replace(rebuilt, subsystems=None) == system


@pytest.mark.parametrize("mode", ["inline", "file"])
def test_every_array_placement_round_trips(models, mode):
    system, params = models["tiny_rssm"]
    document, arrays = describe(system, params, arrays=mode)
    assert bool(arrays) == (mode == "file")
    assert ("arrays_file" in document) == (mode == "file")
    s2, p2 = build(_through_json(document), arrays=arrays)
    _assert_same(system, s2)
    _assert_same(params, p2)


def _leaves(clamp):
    """Every leaf clamp of a clamp tree."""
    if isinstance(clamp, dict) and not ({"ref", "init"} & set(clamp)):
        for v in clamp.values():
            yield from _leaves(v)
    elif isinstance(clamp, list) and clamp and all(isinstance(v, dict) for v in clamp):
        for v in clamp:
            yield from _leaves(v)
    else:
        yield clamp


def test_auto_placement_splits_on_the_element_threshold(models):
    system, params = models["raif_mountain_car"]
    document, arrays = describe(system, params, arrays="auto", inline_max=256)
    sizes = {k: int(np.prod(v.shape)) for k, v in arrays.items()}
    assert sizes and all(n > 256 for n in sizes.values())
    inline = [
        np.asarray(leaf)
        for clamp in document["clamps"].values()
        for leaf in _leaves(clamp)
        if not isinstance(leaf, dict)
    ]
    assert inline and all(a.size <= 256 for a in inline)
    #  the arrays-file keys name the variable and the leaf, readable on their own
    assert "deter_transition_map/gate/weight" in arrays
    assert document["clamps"]["deter_transition_map"]["gate"]["weight"] == {
        "ref": "arrays.npz#deter_transition_map/gate/weight"
    }


def test_the_threshold_is_a_writing_policy_and_build_accepts_either_form(models):
    """A tiny ref and a huge inline array both build: placement is describe's choice only."""
    system, params = models["tiny_rssm"]
    document, arrays = describe(system, params, arrays="auto", inline_max=256)
    document = _through_json(document)
    small = document["clamps"]["deter_transition_map"]["norms"][0]["bias"]
    assert not isinstance(small, dict)
    arrays["moved"] = np.broadcast_to(np.asarray(small, np.float32), (32,)).copy()
    document["clamps"]["deter_transition_map"]["norms"][0]["bias"] = {"ref": "arrays.npz#moved"}
    big = document["clamps"]["deter_transition_map"]["gate"]["weight"]
    document["clamps"]["deter_transition_map"]["gate"]["weight"] = arrays.pop(
        big["ref"].partition("#")[2]
    ).tolist()
    s2, p2 = build(document, arrays=arrays)
    _assert_same(system, s2)
    _assert_same(params, p2)


def test_the_document_is_the_parameter_enumeration(models):
    """Every array the System holds is a parameter variable with a domain and a shape."""
    system, params = models["compose"]
    document, _ = describe(system, params)
    variables = document["variables"]
    statics = {n for n, r in variables.items() if "shape" in r}
    assert statics == {
        "D_target", "B_target", "D_mode", "B_mode", "A_distance", "A_distance_concentration",
        "A_size", "A_size_concentration", "A_answer", "C_answer",
    }  # fmt: skip
    for name in statics:
        assert set(variables[name]) == {"domain", "shape"}
    assert variables["A_distance_concentration"]["domain"] == "positive"
    assert document["factors"]["A_distance"]["prior"]["family"] == "Dirichlet"
    assert "A_distance" not in document["clamps"]
    count = sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(params))
    clamped = sum(int(np.prod(variables[n]["shape"])) for n in document["clamps"] if n in statics)
    assert count == clamped


def test_names_a_document_carries_are_declared_names(models):
    """A System built without names describes with the names it was filled in with, and builds
    back equal: no record of which names were given."""
    system, params = models["hybrid_switching"]
    assert system.modality_names == ()
    s2, p2 = _round_trip(system, params)
    assert s2 == system and s2.factor_names == ("mode", "x", "u_mode")
    unnamed = H.System(B_deps=((0,),), controlled=(True,), A_deps=((0,),), policy_len=1)
    up = H.SystemParams(
        transition={"factor_0": H.ConditionalCategorical(jnp.zeros((2, 2, 2)), 2)},
        initial_state_prior={"factor_0": H.Categorical(jnp.zeros(2))},
        likelihood={"modality_0": H.ConditionalCategorical(jnp.zeros((2, 2)), 1)},
    )
    document, _ = describe(unnamed, up)
    assert list(document["variables"])[:4] == ["factor_0", "u_factor_0", "a_factor_0", "modality_0"]
    assert build(document)[0] == unnamed


def test_a_merged_composite_writes_its_bindings_as_structure(models):
    system, params = models["merged"]
    document, _ = describe(system, params)
    assert set(document) >= {"variables", "factors", "relations"}
    assert "subsystems" not in json.dumps(document)
    #  a bound modality is a likelihood whose parent lies under another prefix
    assert document["factors"]["agent.location_obs"]["likelihood"]["parents"] == ["world.location"]
    #  the relay: the agent's state read a step later by the world's transition
    assert document["factors"]["world.location"]["transition"]["parents"] == [
        "world.location",
        "agent.location",
    ]
    assert "world.u_location" not in document["variables"]
    assert all("." in name for name in document["relations"]["observed"])


def test_the_composite_record_is_reconstructed_so_run_routes_and_views_agree(models):
    system, params = models["merged"]
    rebuilt, rp = _round_trip(system, params)
    assert dict(rebuilt.subsystems).keys() == dict(system.subsystems).keys()
    assert rebuilt.subsystems == _shown(system).subsystems

    def minds(composite, cp):
        return {
            "agent": agent_respond(*subsystem(composite, cp, "agent")),
            "world": ancestral_respond(*subsystem(composite, cp, "world")),
        }

    for name in ("agent", "world"):
        a, ap = subsystem(system, params, name)
        b, bp = subsystem(rebuilt, rp, name)
        _assert_same(ap, bp)
        assert _shown(a) == _shown(b)
    want = run(system, params, minds(system, params), ticks=3, key=KEY)
    got = run(rebuilt, rp, minds(rebuilt, rp), ticks=3, key=KEY)
    for part in ("states", "observations", "actions", "realised"):
        _assert_same(getattr(want, part), getattr(got, part))


def test_the_nested_preference_model_is_one_grammar(models):
    system, params = models["raif_mountain_car"]
    document, _ = describe(system, params, arrays="file")
    term = next(
        t for t in document["objective"]["terms"] if t["term"] == "cosine_sim_to_pref_rollout"
    )
    model = term["model"]
    assert set(model) == {
        "variables",
        "factors",
        "relations",
        "clamps",
        "objective",
        "configuration",
    }
    assert model["relations"] == {"external": ["embed"]}
    assert model["factors"]["stoch"]["recognition"]["parents"] == ["deter", "embed"]
    ref = model["clamps"]["deter_initial_value"]["ref"]
    assert ref == "arrays.npz#cosine_sim_to_pref_rollout/deter_initial_value"


# ======================================================= the reader, record by record


@pytest.fixture
def tmaze_doc(models):
    system, params = models["tmaze"]
    return _through_json(describe(system, params, arrays="inline").document)


@pytest.fixture
def compose_doc(models):
    system, params = models["compose"]
    return _through_json(describe(system, params, arrays="inline").document)


def test_each_role_label_lowers_like_its_generic_spelling(models, tmaze_doc):
    """initial is steps "0" at the same step; transition "1+" a step back; likelihood and
    actuator "all" at the same step. Written generically, the same System builds."""
    system, params = models["tmaze"]
    params = dataclasses.replace(
        params, actuator={"a_location": jnp.full((5, 5), 0.02).at[jnp.diag_indices(5)].set(0.92)}
    )
    document = _through_json(describe(system, params, arrays="inline").document)
    want = build(document)
    factors = document["factors"]
    loc = factors["location"]
    loc["start"] = {**loc.pop("initial"), "steps": "0"}
    moves = loc.pop("transition")
    loc["moves"] = {**moves, "parents": [f"{p}[t-1]" for p in moves["parents"]], "steps": "1+"}
    cue = factors["cue"]
    sees = cue.pop("likelihood")
    cue["sees"] = {**sees, "parents": [f"{p}[t]" for p in sees["parents"]]}
    acts = factors["a_location"].pop("actuator")
    factors["a_location"]["slips"] = {**acts, "parents": ["u_location[t]"]}
    got = build(document)
    _assert_same(want[0], got[0])
    _assert_same(want[1], got[1])


def test_a_composites_uncontrolled_table_may_omit_the_size_one_axis(models):
    """Layout is never written: a composite's uncontrolled transition table with one axis per
    parent builds as the padded table does (the engine reads a composite's tables padded)."""
    system, params = models["merged"]
    document = _through_json(describe(system, params, arrays="inline").document)
    want = build(document)
    for name in ("agent.B_context", "world.B_context", "world.B_location"):
        shape = document["variables"][name]["shape"]
        assert shape[-1] == 1
        document["variables"][name]["shape"] = shape[:-1]
        document["clamps"][name] = np.asarray(document["clamps"][name])[..., 0].tolist()
    got = build(document)
    _assert_same(want[0], got[0])
    _assert_same(want[1], got[1])


@pytest.mark.parametrize(
    "parent, sentence",
    [
        ("location[t+1]", r"write name\[t\] for the child's step or name\[t-1\]"),
        ("location[", r"write name\[t\] for the child's step or name\[t-1\]"),
        ("location[s]", r"write name\[t\] for the child's step or name\[t-1\]"),
    ],
)
def test_a_bracket_other_than_the_two_forms_is_refused(tmaze_doc, parent, sentence):
    tmaze_doc["factors"]["location"]["transition"]["parents"][0] = parent
    with pytest.raises(DescriptionError, match=sentence):
        build(tmaze_doc)


def test_brackets_are_parsed_and_a_bare_parent_under_a_generic_label_is_refused(tmaze_doc):
    record = tmaze_doc["factors"]["location"].pop("transition")
    tmaze_doc["factors"]["location"]["moves"] = {**record, "steps": "1+"}
    with pytest.raises(
        DescriptionError,
        match=r"under the generic label 'moves' every parent is bracketed; write location\[t\]",
    ):
        build(tmaze_doc)
    tmaze_doc["factors"]["location"]["moves"]["parents"] = ["location[t-1]", "u_location[t-1]"]
    build(tmaze_doc)


def test_a_variable_name_with_a_bracket_is_refused(tmaze_doc):
    tmaze_doc["variables"]["odd[1]"] = {"domain": "categorical", "levels": 2}
    with pytest.raises(DescriptionError, match=r"variables\['odd\[1\]'\]: a variable name"):
        build(tmaze_doc)


def test_steps_default_to_all_and_a_first_step_record_may_not_read_the_step_before(tmaze_doc):
    record = tmaze_doc["factors"]["location"].pop("transition")
    record["parents"] = ["location[t-1]", "u_location[t-1]"]
    tmaze_doc["factors"]["location"]["moves"] = record  # no steps: "all"
    with pytest.raises(
        DescriptionError, match=r"applies at step 0 \(steps 'all'\) .* write steps \"1\+\""
    ):
        build(tmaze_doc)


def test_an_initial_record_with_a_parent_a_step_back_is_refused(tmaze_doc):
    tmaze_doc["factors"]["context"]["initial"]["parents"] = ["location[t-1]"]
    with pytest.raises(
        DescriptionError, match=r"a record under 'initial' reads its parents at the first step"
    ):
        build(tmaze_doc)


def test_a_role_label_with_other_steps_is_refused(tmaze_doc):
    tmaze_doc["factors"]["location"]["transition"]["steps"] = "all"
    with pytest.raises(DescriptionError, match=r"a 'transition' record applies at steps '1\+'"):
        build(tmaze_doc)
    del tmaze_doc["factors"]["location"]["transition"]["steps"]
    tmaze_doc["factors"]["reward"]["target"]["steps"] = "all"
    with pytest.raises(DescriptionError, match=r"steps does not apply to a 'target' record"):
        build(tmaze_doc)


def test_a_latent_record_at_every_step_is_refused_naming_the_two_records(tmaze_doc):
    record = tmaze_doc["factors"]["context"].pop("transition")
    tmaze_doc["factors"]["context"]["stays"] = {**record, "parents": ["context[t]"]}
    with pytest.raises(DescriptionError, match=r"write two records, steps \"0\" and \"1\+\""):
        build(tmaze_doc)


def test_two_every_step_conditionals_on_one_latent_are_refused(tmaze_doc):
    record = tmaze_doc["factors"]["context"]["transition"]
    tmaze_doc["factors"]["context"]["again"] = {
        **record,
        "parents": ["context[t-1]"],
        "steps": "1+",
    }
    with pytest.raises(DescriptionError, match=r"'context' already has a transition"):
        build(tmaze_doc)


def test_the_hybrid_mode_parent_overrides_the_transition_label(models):
    system, params = models["hybrid_switching"]
    document = _through_json(describe(system, params, arrays="inline").document)
    assert document["factors"]["x"]["transition"]["parents"] == ["x", "mode[t]"]
    document["factors"]["x"]["transition"]["parents"] = ["x", "mode"]
    with pytest.raises(
        DescriptionError, match=r"reads the latent 'mode' at the child step: write mode\[t\]"
    ):
        build(document)
    document["factors"]["x"]["transition"]["parents"] = ["mode[t]", "x"]
    with pytest.raises(DescriptionError, match=r"list 'x' first among its parents"):
        build(document)


def test_a_same_step_misorder_is_refused_naming_a_valid_order(models):
    system, params = models["hybrid_switching"]
    document = _through_json(describe(system, params, arrays="inline").document)
    variables = document["variables"]
    document["variables"] = {"x": variables.pop("x"), **variables}
    with pytest.raises(
        DescriptionError,
        match=r"so 'mode' must be declared before 'x'.*valid order .* \['mode', 'x'\]",
    ):
        build(document)


def test_a_same_step_cycle_is_refused(models):
    system, params = models["hybrid_switching"]
    document = _through_json(describe(system, params, arrays="inline").document)
    initial = document["factors"]["mode"]["initial"]
    initial.update(family="ConditionalCategorical", parents=["x"])
    with pytest.raises(DescriptionError, match=r"read one another at the same step in a cycle"):
        build(document)


def test_a_table_shape_names_the_child_first_rule(tmaze_doc):
    tmaze_doc["variables"]["B_context"]["shape"] = [2, 2, 2]
    tmaze_doc["clamps"]["B_context"] = -0.6931471805599453
    with pytest.raises(
        DescriptionError, match=r"tables are child-first, so 'context' \(2 levels\)"
    ):
        build(tmaze_doc)


def test_a_table_whose_control_axis_disagrees_with_the_control_node_is_refused(tmaze_doc):
    tmaze_doc["variables"]["u_location"]["levels"] = 4
    tmaze_doc["variables"]["a_location"]["levels"] = 4
    with pytest.raises(DescriptionError, match=r"'u_location': 4\}.*gives \[5, 5, 4\]"):
        build(tmaze_doc)


def test_an_action_node_has_its_control_node_s_size(tmaze_doc):
    tmaze_doc["variables"]["a_location"]["levels"] = 4
    with pytest.raises(
        DescriptionError, match=r"action node of 'u_location'.*differs from the control node"
    ):
        build(tmaze_doc)


def test_relations_are_checked(tmaze_doc):
    observed = tmaze_doc["relations"]["observed"]
    tmaze_doc["variables"]["smell"] = {"domain": "categorical", "levels": 2}
    tmaze_doc["relations"]["observed"] = [*observed, "smell"]
    with pytest.raises(DescriptionError, match=r"'smell' is observed .* has no likelihood"):
        build(tmaze_doc)
    tmaze_doc["relations"]["observed"] = [*observed, "u_location"]
    with pytest.raises(DescriptionError, match=r"as a control node .* and as observed"):
        build(tmaze_doc)


def test_an_external_input_is_read_by_recognition_records_only(models):
    system, params = models["raif_mountain_car"]
    document, arrays = describe(system, params)
    model = next(t for t in document["objective"]["terms"] if "model" in t)["model"]
    model["factors"]["deter"]["transition"]["parents"].append("embed")
    with pytest.raises(DescriptionError, match=r"reads the external input\(s\) \['embed'\]"):
        build(_through_json(document), arrays=arrays)


def test_a_variable_used_both_ways_or_neither_is_refused(tmaze_doc):
    doc = copy.deepcopy(tmaze_doc)
    doc["factors"]["cue"]["likelihood"]["parents"] = ["location", "B_context"]
    with pytest.raises(DescriptionError, match=r"'B_context' is static where .* and per-step"):
        build(doc)
    tmaze_doc["variables"]["spare"] = {"domain": "real", "dim": 2}
    with pytest.raises(DescriptionError, match=r"'spare' is used by no factor, relation or term"):
        build(tmaze_doc)


def test_a_parameter_needs_exactly_one_of_a_clamp_and_a_prior(tmaze_doc):
    doc = copy.deepcopy(tmaze_doc)
    del doc["clamps"]["B_context"]
    with pytest.raises(DescriptionError, match=r"'B_context' has neither a clamp nor a prior"):
        build(doc)
    tmaze_doc["factors"]["B_context"] = {
        "prior": {"family": "Dirichlet", "parameters": {"concentration": "D_context"}}
    }
    with pytest.raises(DescriptionError, match=r"'B_context' has a clamp and a prior record"):
        build(tmaze_doc)


def test_an_unknown_family_lists_the_menu(tmaze_doc):
    tmaze_doc["factors"]["location"]["transition"]["family"] = "ConditionalCategorial"
    with pytest.raises(DescriptionError, match=r"unknown family 'ConditionalCategorial'; the menu"):
        build(tmaze_doc)


def test_an_unknown_parameter_field_or_option_lists_the_real_ones(tmaze_doc):
    doc = copy.deepcopy(tmaze_doc)
    doc["factors"]["location"]["transition"]["parameters"] = {"table": "B_location"}
    with pytest.raises(DescriptionError, match=r"no parameter\(s\) \['table'\]; its parameters"):
        build(doc)
    tmaze_doc["factors"]["location"]["transition"]["options"] = {"parent_ndims": 2}
    with pytest.raises(DescriptionError, match=r"no option\(s\) \['parent_ndims'\]"):
        build(tmaze_doc)


def test_a_family_under_a_label_it_does_not_take_is_refused(tmaze_doc):
    tmaze_doc["factors"]["location"]["transition"]["family"] = "Gaussian"
    with pytest.raises(DescriptionError, match=r"a Gaussian record is not written under the label"):
        build(tmaze_doc)


def test_a_positive_domain_is_checked(compose_doc):
    compose_doc["clamps"]["A_distance_concentration"] = -1.0
    with pytest.raises(DescriptionError, match=r"strictly positive everywhere"):
        build(compose_doc)


def test_a_simplex_domain_is_checked_and_a_declared_domain_must_fit_its_field(tmaze_doc):
    doc = copy.deepcopy(tmaze_doc)
    doc["clamps"]["B_context"] = 0.0
    with pytest.raises(DescriptionError, match=r"normalised log-probabilities"):
        build(doc)
    doc["variables"]["B_context"]["domain"] = "logits"  # a table's unnormalised logs
    build(doc)
    tmaze_doc["variables"]["B_context"]["domain"] = "positive"
    with pytest.raises(DescriptionError, match=r"declares the domain 'positive' and its field"):
        build(tmaze_doc)


def _assert_close(a, b) -> None:
    """Identical treedef, leaves equal to within float rounding."""
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        np.testing.assert_allclose(np.asarray(x, float), np.asarray(y, float), rtol=1e-5, atol=1e-5)


def test_a_table_written_as_probabilities_is_lowered_to_its_logs(tmaze_doc):
    doc = copy.deepcopy(tmaze_doc)
    for name, record in doc["variables"].items():
        if record.get("domain") == "log_simplex":
            record["domain"] = "simplex"
            doc["clamps"][name] = np.exp(np.asarray(doc["clamps"][name], np.float64)).tolist()
    doc["variables"]["D_context"]["domain"] = "simplex"  # a logits field takes probabilities too
    doc["clamps"]["D_context"] = [0.5, 0.5]
    system, params = build(doc)
    _assert_close((system, params), build(tmaze_doc))
    document = _through_json(describe(system, params, arrays="inline").document)
    assert document["variables"]["B_context"]["domain"] == "log_simplex"
    assert "simplex" not in {r.get("domain") for r in document["variables"].values()}


def test_a_zero_probability_is_an_impossible_entry(tmaze_doc):
    tmaze_doc["variables"]["B_context"]["domain"] = "simplex"
    tmaze_doc["clamps"]["B_context"] = [[[1.0], [0.0]], [[0.0], [1.0]]]
    _, params = build(tmaze_doc)
    table = np.exp(np.asarray(params.transition["context"].log_table, float))
    np.testing.assert_allclose(table[..., 0], np.eye(2), atol=1e-6)


def test_a_joint_categorical_is_one_distribution_over_its_whole_event(tmaze_doc):
    doc = copy.deepcopy(tmaze_doc)
    doc["factors"]["reward"]["target"]["family"] = "JointCategorical"
    doc["variables"]["C_reward"] = {"domain": "simplex", "shape": [3, 1]}
    doc["clamps"]["C_reward"] = [[0.2], [0.3], [0.5]]
    _, params = build(doc)
    np.testing.assert_allclose(
        np.exp(np.asarray(params.obs_targets["reward"].logits, float)), [[0.2], [0.3], [0.5]]
    )
    doc["variables"]["C_reward"]["domain"] = "log_simplex"
    doc["clamps"]["C_reward"] = np.log([[0.2], [0.3], [0.5]]).tolist()
    build(doc)
    doc["variables"]["C_reward"]["domain"] = "simplex"
    doc["clamps"]["C_reward"] = [[1.0], [1.0], [1.0]]  # each row sums to one; the joint does not
    with pytest.raises(DescriptionError, match=r"clamps\['C_reward'\] must be probabilities"):
        build(doc)


def test_a_distribution_of_zeros_goes_the_way_of_one_of_minus_inf(tmaze_doc):
    """A table with a dead column passes the domain check under either spelling and meets the
    same later check, or builds the same, in both."""
    outcomes = []
    for name in ("B_location", "A_cue"):
        for spelling in ("log_simplex", "simplex"):
            doc = copy.deepcopy(tmaze_doc)
            table = np.exp(np.asarray(doc["clamps"][name], np.float64))
            table[:, 0, 0] = 0.0
            doc["variables"][name]["domain"] = spelling
            with np.errstate(divide="ignore"):
                doc["clamps"][name] = (
                    table.tolist() if spelling == "simplex" else np.log(table).tolist()
                )
            try:
                _, params = build(doc)
                outcomes.append(
                    np.asarray(
                        getattr(params, "transition" if name[0] == "B" else "likelihood")[
                            {"B_location": "location", "A_cue": "cue"}[name]
                        ].log_table
                    ).tolist()
                )
            except DescriptionError as exc:
                outcomes.append(str(exc))
        assert outcomes[-1] == outcomes[-2]
        assert "must be" not in str(outcomes[-1])  # the domain check passed under both


def test_a_small_probability_is_a_finite_log(tmaze_doc):
    tmaze_doc["variables"]["D_context"]["domain"] = "simplex"
    tmaze_doc["clamps"]["D_context"] = [1.0 - 1e-30, 1e-30]
    _, params = build(tmaze_doc)
    logits = np.asarray(params.initial_state_prior["context"].logits)
    assert logits.dtype == np.float32 and np.isfinite(logits).all()


def test_a_ref_in_an_integer_dtype_is_lowered_to_the_default_dtype(tmaze_doc):
    tmaze_doc["variables"]["B_context"]["domain"] = "simplex"
    tmaze_doc["clamps"]["B_context"] = {"ref": "x.npz#B"}
    _, params = build(tmaze_doc, arrays={"B": np.eye(2, dtype=np.int64)[..., None]})
    assert params.transition["context"].log_table.dtype == np.float32


def test_a_network_leaf_declares_no_domain_but_real(models):
    system, params = models["tiny_rssm"]
    document = _through_json(describe(system, params, arrays="inline").document)
    leaf = next(v for k, v in document["variables"].items() if "trunk" in k)
    while not D._is_leaf(leaf):
        leaf = leaf[0] if isinstance(leaf, list) else next(iter(leaf.values()))
    leaf["domain"] = "simplex"
    with pytest.raises(
        DescriptionError, match=r"a network's leaf is real, and its record declares"
    ):
        build(document)


def test_a_cholesky_domain_is_checked(models):
    system, params = models["hill_lg"]
    document = _through_json(describe(system, params, arrays="inline").document)
    chol = np.asarray(document["clamps"]["state_initial_chol"], float)
    chol[0, 1] = 0.5
    document["clamps"]["state_initial_chol"] = chol.tolist()
    with pytest.raises(DescriptionError, match=r"must be lower-triangular"):
        build(document)


def test_a_ref_without_its_arrays_names_the_file(models):
    system, params = models["tiny_rssm"]
    document, _ = describe(system, params, arrays="file")
    with pytest.raises(DescriptionError, match=r"no arrays= holding the key .*arrays_file"):
        build(_through_json(document))


def test_a_clamp_is_one_of_a_value_a_ref_and_an_init(tmaze_doc):
    tmaze_doc["clamps"]["B_context"] = {"ref": "arrays.npz#B_context", "init": {}}
    with pytest.raises(DescriptionError, match=r"a clamp is a value, \{'ref'"):
        build(tmaze_doc)


def test_a_scalar_clamp_broadcasts_to_the_declared_shape(tmaze_doc):
    tmaze_doc["clamps"]["A_cue"] = -1.0986122886681098
    _, params = build(tmaze_doc)
    table = params.likelihood["cue"].log_table
    assert table.shape == (3, 5, 2) and bool(jnp.all(table == jnp.float32(-1.0986122886681098)))


def test_describe_writes_a_constant_array_as_its_scalar(models):
    system, params = models["hybrid_switching"]
    document, _ = describe(system, params, arrays="inline")
    assert np.float32(document["clamps"]["x_transition_F"]) == np.float32(0.9)
    assert document["variables"]["x_transition_F"]["shape"] == [2, 1, 1]
    written = describe(system, params, arrays="file").document["clamps"]["x_transition_F"]
    assert written == {"ref": "arrays.npz#x_transition_F"}


def test_a_document_from_another_major_version_is_refused(tmaze_doc):
    tmaze_doc[D.KEY] = "1.0"
    with pytest.raises(DescriptionError, match=r"description version 1.0"):
        build(tmaze_doc)
    del tmaze_doc[D.KEY]
    with pytest.raises(DescriptionError, match=r"not a hybrid-aif description"):
        build(tmaze_doc)


def test_the_library_validator_runs_at_build_not_first_call(models, tmaze_doc):
    """A document whose records are well formed but whose System is invalid fails in build."""
    tmaze_doc["objective"] = {"terms": [{"term": "utility"}, {"term": "infogain", "on": "A_cue"}]}
    with pytest.raises(
        DescriptionError,
        match=r"the System it describes is invalid: epistemic: likelihood:cue holds a "
        r"ConditionalCategorical \(a point value\)",
    ):
        build(tmaze_doc)
    system, params = models["hybrid_switching"]
    document = _through_json(describe(system, params, arrays="inline").document)
    del document["configuration"]["inference"]  # a mixture-carried latent needs mixture_k
    with pytest.raises(
        DescriptionError, match=r"\['x'\]\['initial'\]: a MixtureGaussian .*mixture_k"
    ):
        build(document)


def test_a_state_and_an_observation_of_one_name_is_refused_by_the_system():
    with pytest.raises(ValueError, match=r"\['x'\] name both a node and a modality.*'x'_obs"):
        H.System(
            B_deps=((0,),),
            A_deps=((0,),),
            policy_len=1,
            factor_names=("x",),
            modality_names=("x",),
        )


# ================================================= learned parameters and their priors


def _lg_model(likelihood):
    system = H.System(
        A_deps=((0,),),
        B_deps=((),),
        controlled=(False,),
        policy_len=1,
        modality_names=("y",),
        factor_names=("x",),
        epistemic=("x",),
    )
    params = H.SystemParams(
        likelihood={"y": likelihood},
        transition={
            "x": H.LinearGaussianTransition(
                F=jnp.asarray([[1.0]]),
                innovations=H.ConditionalGaussian(
                    loc=jnp.zeros((1, 1)), chol=jnp.asarray([[[0.05]]]), parent_ndims=1
                ),
            )
        },
        initial_state_prior={"x": H.Gaussian(loc=jnp.zeros(1), chol=jnp.asarray([[1.0]]))},
    )
    return system, params


def test_the_mniw_family_is_written_as_two_priors_and_read_back_as_one():
    from hybrid_aif.distributions import MatrixNormal
    from hybrid_aif.factors import LinearGaussianObservationMNIW

    mniw = LinearGaussianObservationMNIW.from_head_noise(
        jnp.asarray([[1.5, 0.3]]), jnp.asarray([[0.4]]), strength=2.0, anchor=True
    )
    system, params = _lg_model(mniw)
    document, arrays = describe(system, params, arrays="inline")
    factors = document["factors"]
    assert factors["y"]["likelihood"]["parameters"] == {
        "head": "y_likelihood_head",
        "noise": "y_likelihood_noise",
    }
    assert factors["y_likelihood_noise"]["prior"]["family"] == "InverseWishart"
    head_prior = factors["y_likelihood_head"]["prior"]
    assert head_prior["family"] == "MatrixNormal"
    assert head_prior["parameters"]["row_cov"] == "y_likelihood_noise"
    s2, p2 = build(_through_json(document), arrays=arrays)
    assert isinstance(p2.likelihood["y"], LinearGaussianObservationMNIW)
    _assert_same(params, p2)
    #  with the noise clamped, the head's record alone is the standalone matrix-normal head
    doc = _through_json(document)
    del doc["factors"]["y_likelihood_noise"]
    for name in ("y_likelihood_noise_Psi", "y_likelihood_noise_nu", "y_likelihood_noise_Psi0",
                 "y_likelihood_noise_nu0"):  # fmt: skip
        del doc["variables"][name], doc["clamps"][name]
    doc["clamps"]["y_likelihood_noise"] = [[0.4]]
    _, p3 = build(doc)
    assert isinstance(p3.likelihood["y"].head, MatrixNormal)
    assert np.array_equal(np.asarray(p3.likelihood["y"].noise), np.asarray([[0.4]], np.float32))
    doc["factors"]["y_likelihood_head"]["prior"]["parameters"]["row_cov"] = "y_likelihood_head"
    with pytest.raises(DescriptionError, match=r"row_cov names 'y_likelihood_head'"):
        build(doc)


def test_an_independent_noise_prior_round_trips():
    from hybrid_aif.distributions import InverseGamma

    noise = InverseGamma(alpha=jnp.asarray([3.0]), beta=jnp.asarray([0.5]))
    sensor = H.LinearGaussianObservation(H=jnp.asarray([[1.0]]), c=jnp.zeros(1), noise=noise)
    system, params = _lg_model(sensor)
    document, _ = describe(system, params, arrays="inline")
    assert document["factors"]["y_likelihood_noise"]["prior"] == {
        "family": "InverseGamma",
        "parameters": {"alpha": "y_likelihood_noise_alpha", "beta": "y_likelihood_noise_beta"},
    }
    assert "y_likelihood_noise" not in document["clamps"]
    _, p2 = build(_through_json(document))
    _assert_same(params, p2)


def test_a_prior_a_field_does_not_take_is_refused(tmaze_doc):
    del tmaze_doc["clamps"]["B_context"]
    tmaze_doc["variables"]["B_context_Psi"] = {"domain": "spd", "shape": [2, 2]}
    tmaze_doc["clamps"]["B_context_Psi"] = [[1.0, 0.0], [0.0, 1.0]]
    tmaze_doc["factors"]["B_context"] = {
        "prior": {"family": "InverseWishart", "parameters": {"Psi": "B_context_Psi"}}
    }
    with pytest.raises(DescriptionError, match=r"the InverseWishart prior cannot learn a table"):
        build(tmaze_doc)


# ============================================================== networks and trees


def _recipe(record: dict) -> dict:
    """The init recipe that draws a network of the shapes a described tree variable lists."""
    if "gate" in record:  # a GRU cell
        biased = "bias" in record["input_proj"]
        return {
            "hidden": record["input_proj"]["weight"]["shape"][0],
            "init": "default" if biased else "trunc_normal_avg",
        }
    layers = record["layers"]
    return {
        "width": layers[0]["weight"]["shape"][0],
        "depth": len(layers) - 1,
        "init": "default" if "bias" in layers[0] else "trunc_normal_avg",
    }


def _strip_to_init(document):
    """Every network and learned vector clamped by an init recipe instead of its values."""
    for name, record in list(document["variables"].items()):
        if isinstance(record, dict) and "domain" not in record:
            document["clamps"][name] = {"init": _recipe(record)}
            document["variables"][name] = {"domain": "real"}
    document["clamps"]["deter_initial_value"] = {"init": {}}
    return document


def test_an_init_recipe_draws_fresh_values_of_the_declared_shapes(models):
    system, params = models["tiny_rssm"]
    document, _ = describe(system, params, arrays="file")
    fresh = _strip_to_init(_through_json(document))
    fresh.pop("arrays_file")
    s2, p2 = build(fresh, key=jax.random.PRNGKey(7))
    assert jax.tree_util.tree_structure(p2) == jax.tree_util.tree_structure(params)
    for a, b in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(p2), strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(
            jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(p2), strict=True
        )
    )
    _assert_same(system, s2)


def test_an_init_recipe_without_a_key_is_refused_naming_the_variable(models):
    system, params = models["tiny_rssm"]
    document, _ = describe(system, params, arrays="file")
    with pytest.raises(DescriptionError, match=r"clamps\['.*'\] is clamped by an init recipe"):
        build(_strip_to_init(_through_json(document)))


def test_an_init_recipe_checks_its_arguments_and_the_declared_leaves(models):
    system, params = models["tiny_rssm"]
    described = _through_json(describe(system, params, arrays="file").document)
    leaves = described["variables"]["embed_recognition_layers"]
    document = _strip_to_init(described)
    doc = copy.deepcopy(document)
    doc["clamps"]["embed_recognition_layers"] = {"init": {"width": 32, "hidden": 3}}
    with pytest.raises(DescriptionError, match=r"an init recipe for MLPPredictor takes"):
        build(doc, key=KEY)
    document["variables"]["embed_recognition_layers"] = leaves  # the leaves, listed
    document["clamps"]["embed_recognition_layers"] = {"init": {"width": 7, "depth": 1}}
    with pytest.raises(DescriptionError, match=r"draws a network of shapes .* declares"):
        build(document, key=KEY)
    width = leaves["layers"][0]["weight"]["shape"][0]
    document["clamps"]["embed_recognition_layers"] = {"init": {"width": width, "depth": 1}}
    build(document, key=KEY)  # the drawn leaves match the listed ones


def test_a_tree_clamp_is_read_leaf_by_leaf_and_must_match_its_tree(models):
    system, params = models["tiny_rssm"]
    document, arrays = describe(system, params, arrays="file")
    document = _through_json(document)
    assert isinstance(document["variables"]["stoch_transition_trunk"]["layers"], list)
    tree = document["clamps"]["stoch_transition_trunk"]
    assert set(tree) == {"layers", "norms"} and "ref" in tree["layers"][0]["weight"]
    s2, p2 = build(document, arrays=arrays)
    _assert_same(params, p2)
    del document["clamps"]["stoch_transition_trunk"]["norms"]
    with pytest.raises(DescriptionError, match=r"the clamp's tree .* differs from the variable's"):
        build(document, arrays=arrays)


# ================================================================= default names


def test_default_names_follow_the_letters_the_fields_and_the_prefixes(models):
    compose, _ = models["compose"]
    document, _ = describe(*models["compose"], arrays="inline")
    assert {"A_distance", "B_target", "D_mode", "C_answer", "A_size_concentration"} <= set(
        document["variables"]
    )
    document, _ = describe(*models["hybrid_switching"], arrays="inline")
    assert {"x_transition_F", "x_transition_innovations_loc", "x_target_log_weights"} <= set(
        document["variables"]
    )
    document, _ = describe(*models["merged"], arrays="inline")
    assert {"agent.B_location", "world.B_location", "agent.A_location_obs"} <= set(
        document["variables"]
    )
    document, _ = describe(*models["raif_mountain_car"], arrays="file")
    assert {"g_tail_critic", "model_uncertainty_ensemble_0", "u_deter_recognition_layers"} <= set(
        document["variables"]
    )
    del compose


# ===================================================================== composites


def test_the_dot_is_reserved_to_composite_prefixes(tmaze_doc, models):
    tmaze_doc["variables"]["lone.input"] = {"domain": "real", "dim": 2}
    tmaze_doc["relations"]["external"] = ["lone.input"]
    with pytest.raises(
        DescriptionError, match=r"\['lone.input'\] are dotted, which makes the document a composite"
    ):
        build(tmaze_doc)
    system, params = models["merged"]
    document = _through_json(describe(system, params, arrays="inline").document)
    document["variables"]["nowhere.D"] = document["variables"].pop("agent.D_location")
    document["clamps"]["nowhere.D"] = document["clamps"].pop("agent.D_location")
    document["factors"]["agent.location"]["initial"]["parameters"]["logits"] = "nowhere.D"
    with pytest.raises(DescriptionError, match=r"'nowhere' is not a component of this document"):
        build(document)


# =================================================================== the policy prior


def test_the_policy_prior_is_checked_against_the_policy_set(models):
    system, params = models["tmaze"]
    params = dataclasses.replace(params, policy_prior=H.Categorical(jnp.zeros(25)))
    document = _through_json(describe(system, params, arrays="inline").document)
    assert document["factors"]["u_location"]["policy_prior"] == {
        "family": "Categorical",
        "parameters": {"logits": "E"},
    }
    _, p2 = build(document)
    _assert_same(params, p2)
    doc = copy.deepcopy(document)
    doc["variables"]["E"]["shape"] = [24]
    with pytest.raises(
        DescriptionError, match=r"control nodes \{'u_location': 5\} at horizon 2 has 25 policies"
    ):
        build(doc)


def test_one_policy_prior_sits_under_the_first_control_node():
    system = H.System(B_deps=((0,), (1,)), controlled=(True, True), policy_len=1)
    table = jnp.log(jnp.full((2, 2, 2), 0.5))
    params = H.SystemParams(
        transition={f"factor_{i}": H.ConditionalCategorical(table, 2) for i in range(2)},
        initial_state_prior={f"factor_{i}": H.Categorical(jnp.zeros(2)) for i in range(2)},
        policy_prior=H.Categorical(jnp.zeros(4)),
    )
    document = _through_json(describe(system, params).document)
    assert list(document["factors"]["u_factor_0"]) == ["policy_prior"]
    _, p2 = build(document)
    _assert_same(params, p2)
    document["factors"]["u_factor_1"] = {
        "policy_prior": document["factors"].pop("u_factor_0")["policy_prior"]
    }
    with pytest.raises(DescriptionError, match=r"sits under the first control node, 'u_factor_0'"):
        build(document)


def test_a_raw_log_array_policy_prior_comes_back_as_a_categorical(models):
    system, params = models["tmaze"]
    raw = dataclasses.replace(params, policy_prior=jnp.zeros(25))
    _, rebuilt = _round_trip(system, raw)
    assert isinstance(rebuilt.policy_prior, H.Categorical)
    assert np.array_equal(np.asarray(rebuilt.policy_prior.logits), np.zeros(25, np.float32))


def test_control_and_action_nodes_are_named_freely_in_a_document(tmaze_doc):
    document = json.loads(
        json.dumps(tmaze_doc).replace('"u_location"', '"go"').replace('"a_location"', '"move"')
    )
    assert document["relations"]["controls"] == {"go": "move"}
    system, params = build(document)
    assert system.names.controls == ("go",) and system.names.actions == ("move",)
    again = _through_json(describe(system, params, arrays="inline").document)
    assert again == document


def test_configuration_values_are_read_as_the_system_stores_them(tmaze_doc):
    tmaze_doc["configuration"]["planning"]["gamma"] = 4
    system, _ = build(tmaze_doc)
    assert system.gamma == 4.0 and isinstance(system.gamma, float)


def test_a_learned_vector_drawn_by_init_needs_no_key():
    drawn = build(_examples()["amortized"], key=KEY)
    document = _through_json(describe(*drawn, arrays="inline").document)
    document["clamps"]["deter_initial_value"] = {"init": {}}
    _, params = build(document)  # no key: the zero vector draws nothing
    assert np.array_equal(np.asarray(params.initial_state_prior["deter"].value), np.zeros(8))


def test_an_independent_matrix_normal_head_stays_independent():
    """A matrix-normal head beside an independent inverse-Wishart noise is written without
    row_cov, and comes back as the same class, not the joint family."""
    from hybrid_aif.distributions import InverseWishart, MatrixNormal

    sensor = H.LinearGaussianObservation(
        head=MatrixNormal.from_head(jnp.asarray([[1.0, 0.0]]), strength=2.0),
        noise=InverseWishart(Psi=jnp.asarray([[0.5]]), nu=jnp.asarray(4.0)),
    )
    system, params = _lg_model(sensor)
    document, arrays = describe(system, params, arrays="inline")
    assert "row_cov" not in document["factors"]["y_likelihood_head"]["prior"]["parameters"]
    _, rebuilt = build(_through_json(document), arrays=arrays)
    assert type(rebuilt.likelihood["y"]) is H.LinearGaussianObservation
    _assert_same(params, rebuilt)


# ================================================================= recognition


def test_the_recognition_order_is_derived_from_the_parents(models):
    system, params = models["tiny_rssm"]
    document, arrays = describe(system, params)
    document = _through_json(document)
    factors = document["factors"]
    embed = factors.pop("embed")
    document["factors"] = {**factors, "embed": embed}  # the embedding's block written last
    s2, _ = build(document, arrays=arrays)
    assert [r.variable for r in s2.recognition] == ["embed", "stoch"]
    assert s2.recognition == system.recognition


def test_recognition_records_in_a_cycle_are_refused(models):
    system, params = models["tiny_rssm"]
    document, arrays = describe(system, params)
    document = _through_json(document)
    document["variables"]["code"] = {"domain": "real", "dim": 3}
    document["factors"]["code"] = {
        "recognition": copy.deepcopy(document["factors"]["embed"]["recognition"])
    }
    document["factors"]["embed"]["recognition"]["parents"] = ["code"]
    document["factors"]["code"]["recognition"]["parents"] = ["embed"]
    with pytest.raises(DescriptionError, match=r"read one another in a cycle"):
        build(document, arrays=arrays)


def test_a_recognition_parent_may_not_be_another_amortized_posterior(models):
    system, params = models["tiny_rssm"]
    document, arrays = describe(system, params)
    document = _through_json(document)
    document["factors"]["embed"]["recognition"]["parents"] = ["stoch"]
    with pytest.raises(DescriptionError, match=r"'stoch' is itself an amortized posterior"):
        build(document, arrays=arrays)


def test_the_policy_module_is_the_control_node_s_recognition_record(models):
    system, params = models["raif_mountain_car"]
    document, arrays = describe(system, params)
    record = document["factors"]["u_deter"]["recognition"]
    assert record["family"] == "MLPConditional" and record["parents"] == ["deter", "stoch"]
    _, p2 = build(_through_json(document), arrays=arrays)
    _assert_same(params.policy_prior, p2.policy_prior)
    doc = _through_json(document)
    doc["factors"]["u_deter"]["recognition"]["parents"] = ["deter"]
    with pytest.raises(DescriptionError, match=r"write parents \['deter', 'stoch'\]"):
        build(doc, arrays=arrays)


# =================================================================== the objective


def test_the_decomposition_sugar_expands_and_the_pair_reads_back(models, tmaze_doc):
    system, params = models["tmaze"]
    assert tmaze_doc["objective"] == {"decomposition": "utility_infogain"}
    for epistemic, want in (
        (("location",), [{"term": "utility"}, {"term": "infogain", "on": "location"}]),
        (False, [{"term": "utility"}]),
        (("obs:cue",), [{"term": "utility"}, {"term": "infogain", "on": "cue"}]),
    ):
        variant = dataclasses.replace(system, epistemic=epistemic)
        document = _through_json(describe(variant, params).document)
        assert document["objective"] == {"terms": want}
        assert build(document)[0] == variant
    risk = dataclasses.replace(system, decomposition="risk_ambiguity")
    document = _through_json(describe(risk, params).document)
    assert document["objective"] == {"decomposition": "risk_ambiguity"}
    risk = dataclasses.replace(risk, epistemic=False)
    document = _through_json(describe(risk, params).document)
    assert document["objective"] == {"terms": [{"term": "risk"}, {"term": "ambiguity"}]}
    assert build(document)[0] == risk


def test_infogain_on_a_learned_parameter_and_on_the_patterns_round_trips(models):
    system, params = models["compose"]
    for epistemic, on in (
        (("likelihood:distance",), "A_distance"),
        (("*",), "*"),
        (("obs:*",), "obs:*"),
        (("*:*",), "*:*"),
    ):
        variant = dataclasses.replace(system, epistemic=epistemic)
        document = _through_json(describe(variant, params).document)
        assert {"term": "infogain", "on": on} in document["objective"]["terms"]
        assert build(document)[0] == variant


@pytest.mark.parametrize(
    "terms",
    [
        [{"term": "utility"}, {"term": "ambiguity"}],
        [{"term": "risk"}],
        [{"term": "utility"}, {"term": "risk"}, {"term": "ambiguity"}],
    ],
)
def test_a_pair_the_term_tree_cannot_build_is_refused_naming_the_two_pairs(tmaze_doc, terms):
    tmaze_doc["objective"] = {"terms": terms}
    with pytest.raises(
        DescriptionError, match=r"the pairs are utility with infogain, or risk with ambiguity"
    ):
        build(tmaze_doc)


def test_the_sugar_beside_the_pair_s_own_terms_is_refused(tmaze_doc):
    tmaze_doc["objective"]["terms"] = [{"term": "infogain", "on": "location"}]
    with pytest.raises(DescriptionError, match=r"write the pair as terms instead"):
        build(tmaze_doc)


def test_an_unknown_term_is_refused_naming_the_registry(tmaze_doc):
    tmaze_doc["objective"] = {"terms": [{"term": "curiosity"}]}
    with pytest.raises(
        DescriptionError, match=r"unknown term 'curiosity'; the registered terms are .*infogain"
    ):
        build(tmaze_doc)


def test_ambiguity_and_utility_take_no_on(tmaze_doc):
    tmaze_doc["objective"] = {"terms": [{"term": "risk"}, {"term": "ambiguity", "on": "cue"}]}
    with pytest.raises(DescriptionError, match=r"ambiguity takes no \['on'\]"):
        build(tmaze_doc)


def test_rows_their_targets_and_their_parameters_round_trip(models):
    system, params = models["raif_mountain_car"]
    document, _ = describe(system, params)
    terms = {t["term"]: t for t in document["objective"]["terms"]}
    assert terms["predicted_reward"] == {"term": "predicted_reward", "on": "reward"}
    assert terms["model_uncertainty"]["on"] == "stoch"
    assert terms["model_uncertainty"]["parameters"] == {
        "ensemble": [f"model_uncertainty_ensemble_{i}" for i in range(4)]
    }
    assert terms["g_tail"]["parameters"] == {"critic": "g_tail_critic", "slow": "g_tail_slow"}
    assert terms["g_tail"]["options"]["critic"]["family"] == "PredictorHead"


def test_a_term_s_parameters_are_checked_against_its_declared_names(models):
    system, params = models["raif_mountain_car"]
    document, arrays = describe(system, params)
    document = _through_json(document)
    g_tail = next(t for t in document["objective"]["terms"] if t["term"] == "g_tail")
    g_tail["parameters"]["value"] = g_tail["parameters"].pop("critic")
    g_tail["options"]["value"] = g_tail["options"].pop("critic")
    with pytest.raises(
        DescriptionError, match=r"'g_tail' takes the parameter\(s\) \['critic', 'slow'\]"
    ):
        build(document, arrays=arrays)


def test_parametric_rows_declare_their_parameter_names():
    import hybrid_aif.reference  # noqa: F401  (registers the reference rows)

    assert registry.ROWS["g_tail"].parameters == ("critic", "slow")
    assert registry.ROWS["model_uncertainty"].parameters == ("ensemble",)
    assert registry.by_name("ensemble_disagreement").parameters == ("ensemble",)
    assert registry.ROWS["cosine_sim_to_pref_rollout"].parameters == ()


def test_a_scoring_entry_no_term_mounts_is_refused(models):
    system, params = models["raif_mountain_car"]
    unmounted = dataclasses.replace(system, terms=tuple(t for t in system.terms if t != "g_tail"))
    with pytest.raises(
        DescriptionError, match=r"\['g_tail'\], which no term in System.terms mounts"
    ):
        describe(unmounted, params)


def test_describe_refuses_a_system_the_validator_refuses(models):
    system, params = models["tmaze"]
    for epistemic in (("likelihood:zz",), ("likelihood:cue",)):
        invalid = dataclasses.replace(system, epistemic=epistemic)
        with pytest.raises(
            DescriptionError, match=r"^the System is invalid, so it has no document"
        ):
            describe(invalid, params)
    with pytest.raises(DescriptionError, match=r"^the System is invalid, so it has no document"):
        describe(dataclasses.replace(system, terms=("occupancy:likelihood:cue",)), params)


# ================================================================= configuration


def test_every_configuration_field_round_trips(models):
    system, params = models["tmaze"]
    variants = [
        dataclasses.replace(system, planner="sophisticated", branch_on=("cue",)),
        dataclasses.replace(system, gamma=4.0, policy_len=3),
        dataclasses.replace(system, num_iters=8),
    ]
    for variant in variants:
        document = _through_json(describe(variant, params).document)
        assert build(document)[0] == variant
    document = _through_json(describe(variants[0], params).document)
    assert document["configuration"]["planning"] == {
        "planner": "sophisticated",
        "horizon": 2,
        "gamma": 16.0,
        "branch_on": ["cue"],
    }
    assert "inference" not in document["configuration"]
    switching, sp = models["switching_filter"]
    document = _through_json(describe(switching, sp).document)
    assert document["configuration"]["inference"] == {
        "iterations": 16,
        "mixture_k": 8,
        "mixture_schedule": "mode_aligned",
    }
    assert build(document)[0] == switching


def test_learning_frozen_sets_the_gradient_families_flags(models):
    system, params = models["tiny_rssm"]
    document, arrays = describe(system, params)
    document = _through_json(document)
    assert "learning" not in document["configuration"]
    document["configuration"]["learning"] = {
        "frozen": ["embed_recognition_layers", "deter_initial_value", "deter_transition_map"]
    }
    _, p2 = build(document, arrays=arrays)
    assert p2.recognition["embed"].frozen
    assert p2.initial_state_prior["deter"].frozen and p2.transition["deter"].frozen
    again = describe(*build(document, arrays=arrays)).document
    assert again["configuration"]["learning"]["frozen"] == [
        "deter_initial_value",
        "deter_transition_map",
        "embed_recognition_layers",
    ]


def test_an_unknown_configuration_field_is_refused(tmaze_doc):
    tmaze_doc["configuration"]["planning"]["candidates"] = 64
    with pytest.raises(DescriptionError, match=r"planner-specific settings and learning rates"):
        build(tmaze_doc)


# ======================================================== the gaps, refused by row


def test_a_target_with_parents_is_a_conditional_family(tmaze_doc):
    """A target's parents are its conditioning variables (step-durations.md T5): an
    unconditional family under them is refused naming the conditional ones."""
    target = tmaze_doc["factors"]["reward"]["target"]
    target["parents"] = ["context"]
    with pytest.raises(DescriptionError, match=r"conditional preference.*ConditionalCategorical"):
        build(tmaze_doc)


def test_a_per_step_prior_on_a_control_node_is_refused_with_a_sentence(tmaze_doc):
    tmaze_doc["variables"]["P_u"] = {"domain": "logits", "shape": [5]}
    tmaze_doc["clamps"]["P_u"] = 0.0
    tmaze_doc["factors"]["u_location"] = {
        "prior": {"family": "Categorical", "parameters": {"logits": "P_u"}}
    }
    with pytest.raises(
        DescriptionError, match=r"per-step prior on the control node 'u_location'.*does not score"
    ):
        build(tmaze_doc)


def test_a_frozen_conjugate_parameter_is_refused_with_a_sentence(compose_doc):
    compose_doc["configuration"]["learning"] = {"frozen": ["A_distance_concentration"]}
    with pytest.raises(
        DescriptionError, match=r"'A_distance_concentration', a conjugate parameter.*lr_by_slot"
    ):
        build(compose_doc)
    compose_doc["configuration"]["learning"] = {"frozen": ["B_target"]}
    with pytest.raises(DescriptionError, match=r"'B_target', which no trainer moves"):
        build(compose_doc)


# ========================================================== the format document


def _format_documents() -> list:
    text = (_ROOT / "docs" / "description-format.md").read_text()
    blocks = re.findall(r"```json\n(.*?)\n```", text, re.S)
    return [json.loads(block) for block in blocks if block.lstrip().startswith("{")]


def _close(a, b) -> bool:
    """Equal as documents, numbers compared at the precision an inline clamp is read in."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float | int) and isinstance(b, float | int) and not isinstance(a, bool):
        return bool(np.isclose(np.float32(a), np.float32(b), rtol=1e-6, atol=0))
    return a == b


def _examples() -> dict[str, dict]:
    full = [d for d in _format_documents() if D.KEY in d]
    return dict(zip(("tabular", "hybrid", "amortized", "composite"), full, strict=True))


@pytest.mark.parametrize("name", ["tabular", "hybrid", "composite"])
def test_the_format_document_examples_build_and_describe_back(name):
    document = _examples()[name]
    system, params = build(document)
    again = _through_json(describe(system, params, arrays="inline").document)
    for section in ("variables", "factors", "relations", "clamps", "objective", "configuration"):
        assert _close(again[section], document[section]), section
    assert list(again["variables"]) == list(document["variables"])


def test_the_amortized_example_draws_its_networks_and_round_trips():
    document = _examples()["amortized"]
    system, params = build(document, key=KEY)
    assert [r.variable for r in system.recognition] == ["embed", "stoch"]
    assert system.terms == ("predicted_reward:obs:reward", "g_tail")
    again, arrays = describe(system, params)
    s2, p2 = build(_through_json(again), arrays=arrays)
    _assert_same(system, s2)
    _assert_same(params, p2)
    again = _through_json(again)
    assert again["relations"] == document["relations"]
    assert again["configuration"] == document["configuration"]
    #  a normalising describe: the drawn weights written, the leaves listed, every option
    #  filled; describing what that builds gives the same document
    third = _through_json(describe(*build(again, arrays=arrays)).document)
    assert third == again


def test_the_generic_spelling_of_the_switching_transition_builds_the_same_model():
    snippets = [d for d in _format_documents() if D.KEY not in d]
    (generic,) = [s for s in snippets if "x" in s]
    document = _examples()["hybrid"]
    variant = copy.deepcopy(document)
    variant["factors"]["x"] = generic["x"]
    want, got = build(document), build(variant)
    _assert_same(want[0], got[0])
    _assert_same(want[1], got[1])


# ========================================================== save, load, the arrays file


def test_save_and_load_carry_the_arrays_file(models, tmp_path):
    system, params = models["raif_mountain_car"]
    json_path, npz_path = save(describe(system, params), tmp_path / "agent")
    assert json_path.name == "agent.json" and npz_path.name == "agent.npz"
    written = json.loads(json_path.read_text())
    assert written["arrays_file"] == "agent.npz"
    assert written["clamps"]["deter_transition_map"]["gate"]["weight"]["ref"].startswith(
        "agent.npz#"
    )
    loaded = load(json_path)
    s2, p2 = build(loaded.document, arrays=loaded.arrays)
    _assert_same(system, s2)
    _assert_same(params, p2)


def test_save_without_refs_writes_one_file(models, tmp_path):
    system, params = models["tmaze"]
    json_path, npz_path = save(describe(system, params), tmp_path / "tmaze.json")
    assert npz_path is None and "arrays_file" not in json.loads(json_path.read_text())
    _, p2 = build(load(json_path).document)
    _assert_same(params, p2)


def test_load_nests_a_model_given_by_path(models, tmp_path):
    system, params = models["raif_mountain_car"]
    document, arrays = describe(system, params)
    term = next(t for t in document["objective"]["terms"] if "model" in t)
    prefix = "cosine_sim_to_pref_rollout/"
    inner = json.loads(json.dumps({D.KEY: D.VERSION, **term.pop("model")}).replace(prefix, ""))
    inner_arrays = {k[len(prefix) :]: v for k, v in arrays.items() if k.startswith(prefix)}
    outer_arrays = {k: v for k, v in arrays.items() if not k.startswith(prefix)}
    save(Description(inner, inner_arrays), tmp_path / "preference")
    term["model"] = {"ref": "preference.json"}
    with pytest.raises(DescriptionError, match=r"a model given by path is read by load\(\)"):
        build(_through_json(document), arrays=outer_arrays)
    path, _ = save(Description(document, outer_arrays), tmp_path / "agent")
    loaded = load(path)
    s2, p2 = build(loaded.document, arrays=loaded.arrays)
    _assert_same(system, s2)
    _assert_same(params, p2)


# ===================================================================== registry refs


def test_a_named_function_travels_by_registry_reference(models):
    system, params = models["tiny_rssm"]
    named = H.Deterministic(map=np.tanh)
    record = D._encode(D._Writer("inline", 0), named, "deter_transition", None, "x")
    assert record == {
        "family": "Deterministic",
        "options": {"rule": "carry", "map": {"impl": "numpy:tanh"}},
    }
    model = D._Model(D._Reader({}, None), {}, "")
    assert D._import("numpy:tanh", "x", model.fail) is np.tanh
    with pytest.raises(DescriptionError, match=r"could not be imported"):
        D._import("numpy:no_such_function", "x", model.fail)
    del system, params


def test_a_lambda_cannot_be_described():
    with pytest.raises(DescriptionError, match=r"lambda or local function"):
        D._encode(D._Writer("inline", 0), H.Deterministic(map=lambda x: x), "d", None, "x")


# =========================================================================== schema


jsonschema = pytest.importorskip("jsonschema")


def test_the_schema_is_a_valid_2020_12_schema_and_is_checked_in():
    generated = schema()
    jsonschema.Draft202012Validator.check_schema(generated)
    checked_in = json.loads(D._schema_path().read_text())
    assert checked_in == generated, (
        "the checked-in schema is stale; regenerate with "
        "`.venv/bin/python -m hybrid_aif.description --write-schema`"
    )


@pytest.mark.parametrize("name", NAMES)
def test_every_document_validates_against_the_schema(models, name):
    system, params = models[name]
    validator = jsonschema.Draft202012Validator(schema())
    for mode in ("auto", "file"):
        document, _ = describe(system, params, arrays=mode)
        errors = list(validator.iter_errors(_through_json(document)))
        assert not errors, errors[0].message


def test_the_format_document_examples_validate_against_the_schema():
    validator = jsonschema.Draft202012Validator(schema())
    for document in _examples().values():
        assert not list(validator.iter_errors(document))


def test_the_schema_rejects_what_build_rejects_at_its_level(tmaze_doc):
    validator = jsonschema.Draft202012Validator(schema())
    for edit in (
        lambda d: d["factors"]["location"]["transition"]["parents"].__setitem__(0, "location[t+1]"),
        lambda d: d["clamps"].__setitem__("B_context", {"ref": "arrays.npz#x", "init": {}}),
        lambda d: d["factors"]["location"]["transition"].__setitem__("steps", "all"),
        lambda d: d["factors"]["location"]["initial"].__setitem__("static", {}),
        lambda d: d["objective"].__setitem__("terms", [{"on": "location"}]),
    ):
        doc = copy.deepcopy(tmaze_doc)
        edit(doc)
        assert list(validator.iter_errors(doc))


def test_the_schema_refuses_an_unknown_top_level_key_too(tmaze_doc):
    tmaze_doc["zeroth_control"] = {}
    validator = jsonschema.Draft202012Validator(schema())
    assert any("zeroth_control" in e.message for e in validator.iter_errors(tmaze_doc))


# =========================================================================== exports


def test_the_top_level_exports():
    assert H.System is D.System and H.SystemParams is D.SystemParams
    assert H.describe is describe and H.build is build and H.schema is schema
    for name in ("System", "SystemParams", "describe", "build", "schema"):
        assert name in H.__all__


def test_a_document_is_plain_json(models):
    system, params = models["raif_mountain_car"]
    document, _ = describe(system, params, arrays="auto")
    text = json.dumps(document)  # raises on anything that is not JSON
    assert json.loads(text) == document


def test_dataclasses_replace_and_the_document_agree(models, compose_doc):
    """The document-level patch is the same edit as the Python one: add an observation."""
    system, params = models["compose"]
    table = np.log(np.asarray([[0.8, 0.1, 0.5], [0.2, 0.9, 0.5]])).astype(np.float32)
    compose_doc["variables"]["colour"] = {"domain": "categorical", "levels": 2}
    compose_doc["variables"]["A_colour"] = {"domain": "log_simplex", "shape": [2, 3]}
    compose_doc["clamps"]["A_colour"] = table.tolist()
    compose_doc["factors"]["colour"] = {
        "likelihood": {
            "family": "ConditionalCategorical",
            "parents": ["target"],
            "parameters": {"log_table": "A_colour"},
        }
    }
    compose_doc["relations"]["observed"].append("colour")
    s2, p2 = build(compose_doc)
    patched = dataclasses.replace(
        system, A_deps=system.A_deps + ((0,),), modality_names=system.modality_names + ("colour",)
    )
    patched_params = dataclasses.replace(
        params,
        likelihood={**params.likelihood, "colour": H.ConditionalCategorical(jnp.asarray(table), 1)},
    )
    _assert_same(patched, s2)
    _assert_same(patched_params, p2)


# ====================================== the reserved namespace and unknown keys


def test_an_unknown_top_level_key_is_refused_and_pointed_at_extensions(tmaze_doc):
    tmaze_doc["zeroth_control"] = {"bindings": {"transition:target": "manifest.target"}}
    with pytest.raises(DescriptionError, match=r"\['zeroth_control'\].*belongs under 'extensions'"):
        build(tmaze_doc)


def test_extensions_are_carried_and_never_read(models, tmp_path):
    system, params = models["compose"]
    ext = {"zeroth_control": {"bindings": {"transition:target": "manifest.target"}, "kernel": "v3"}}
    document, arrays = describe(system, params, extensions=ext)
    assert document["extensions"] == ext
    _, p2 = build(_through_json(document), arrays=arrays)
    _assert_same(params, p2)
    json_path, _ = save(Description(document, arrays), tmp_path / "with_ext")
    assert load(json_path).document["extensions"] == ext
    assert "extensions" not in describe(system, params).document


def test_extensions_must_be_objects_keyed_by_consumer(models, compose_doc):
    system, params = models["compose"]
    with pytest.raises(DescriptionError, match="keyed by consumer name"):
        describe(system, params, extensions={"zeroth_control": "not an object"})
    compose_doc["extensions"] = ["not", "an", "object"]
    with pytest.raises(DescriptionError, match="keyed by consumer name"):
        build(compose_doc)


# ============================================= documents in the earlier layout


def test_a_document_in_the_earlier_layout_is_refused_naming_the_release_that_reads_it():
    document = {D.KEY: "0.2", "structure": {}, "declarations": {}}
    with pytest.raises(DescriptionError, match=r"build it with hybrid-aif 0\.18\.0"):
        build(document)


def test_describe_writes_this_version_only(tmaze_doc):
    assert D.VERSION == "0.3" and tmaze_doc[D.KEY] == "0.3"
    assert "structure" not in tmaze_doc and "declarations" not in tmaze_doc
