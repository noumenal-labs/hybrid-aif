"""A document of release 0.19.0 for a System declared with ``lagged=`` is read as the emission plus
``observes`` declaration of release 0.20.0.

The fixtures under ``tests/data/`` were written by the release 0.19.0 source (``describe`` and
``replay_document`` of that release), each JSON its document and each ``.npz`` that release's own
replay of it on a recorded sequence:

- ``v019_lagged_sources``: ``_lagged_system()`` of 0.19.0's ``tests/test_structure_learning.py``
  (the document its ``test_lagged_refusals_and_description`` writes): a Gaussian source ``x`` and
  a categorical source ``kind``, read by ``prev_x`` and ``prev_type``; replayed on the trace of
  its ``test_replay_fills_lagged_data_from_its_carry`` plainly, learning, and with ``prev_x``
  flagged absent on one tick.
- ``v019_lagged_action``: one categorical state factor read by two modalities and the action node
  read by a third, replayed plainly and learning.
- ``v019_lagged_noise_map``: a Gaussian source read by a modality whose noise is a
  ``GridNoiseMap`` (the option ``noise_map`` beside ``datum``).
- ``v019_lagged_merge``: the merge of two ``_lagged_system`` components, with the lagged pairs of
  component ``a`` declared on the composite under its prefix.
- ``v019_lagged_reported``: the document the 0.20.0 release check reported refused.
"""

import copy
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Emission, replay
from hybrid_aif.description import (
    DescriptionError,
    build,
    describe,
    describe_parameters,
)
from hybrid_aif.description import _read_datum as read_datum
from hybrid_aif.replay import replay_document
from tests.test_structure_learning import _own_read_system

DATA = Path(__file__).parent / "data"


def document(name):
    return json.loads((DATA / f"v019_lagged_{name}.json").read_text())


def recorded(name):
    return np.load(DATA / f"v019_lagged_{name}.npz")


def leaves(prefix, out):
    flat = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(out):
        flat[prefix + jax.tree_util.keystr(path)] = np.asarray(leaf)
    return flat


def assert_recorded(got, want, err_msg=""):
    """``got`` is the float32 array release 0.19.0 recorded, bit for bit. Under JAX's 64-bit mode
    a document's inline values build in float64 and the replay computes in float64, so ``got``
    is the recording to float32 rounding: a tolerance of 1e-6 (measured, 3.6e-7)."""
    if jax.config.jax_enable_x64:
        assert np.asarray(got).dtype == np.float64, err_msg  # the replay computed in float64
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6, err_msg=err_msg)
    else:
        np.testing.assert_array_equal(got, want, err_msg=err_msg)


def assert_as_recorded(tag, run, ref):
    """Every array release 0.19.0 recorded under ``tag`` (:func:`assert_recorded`)."""
    got = leaves(f"{tag}.beliefs", run.beliefs)
    if run.free_energy is not None:
        got[f"{tag}.free_energy"] = np.asarray(run.free_energy)
    if run.params is not None:
        got.update(leaves(f"{tag}.params", run.params.likelihood))
    keys = [k for k in ref.files if k.startswith(f"{tag}.")]
    assert keys
    for k in keys:
        assert_recorded(got[k], ref[k], err_msg=k)


def assert_same(a, b):
    """Bitwise with JAX's 64-bit mode off. With it on, the document's numbers were written in
    float32 and a natively declared model computes in float64, so the two agree to float32
    rounding: a relative tolerance of 1e-6."""
    if jax.config.jax_enable_x64:
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-6)
    else:
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def sources_trace():
    rng = np.random.default_rng(2)
    xs = jnp.asarray(np.cumsum(rng.normal(size=(5, 2)), axis=0), jnp.float32)
    kinds = jnp.asarray(rng.integers(0, 3, size=5), jnp.int32)
    return xs, kinds


def no_datum(node):
    return "datum" not in json.dumps(node)


# ---------------------------------------------------------------- what the reader builds


def test_a_state_factor_source_becomes_an_identity_emission_realised_by_its_family():
    system, _ = build(document("sources"))
    assert system.emissions == (
        ("x_read", Emission(("x",), realise="mode", kernel="identity")),
        ("kind_read", Emission(("kind",), realise="stats", kernel="identity")),
    )
    assert system.observes == (("prev_x", "x_read"), ("prev_type", "kind_read"))


def test_the_built_system_is_the_native_declaration():
    """The same System and the same parameters as the model declared in the 0.20.0 form."""
    system, params = build(document("sources"))
    native, native_params = _own_read_system(
        emissions={
            "x_read": Emission(("x",), realise="mode", kernel="identity"),
            "kind_read": Emission(("kind",), realise="stats", kernel="identity"),
        },
        observes={"prev_x": "x_read", "prev_type": "kind_read"},
    )
    assert system == native
    for a, b in zip(
        jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(native_params), strict=True
    ):
        assert_same(a, b)
    xs, kinds = sources_trace()
    obs = {"obs_x": xs, "obs_kind": kinds}
    ours = replay(system, params, obs, {}, metrics=False)
    theirs = replay(native, native_params, obs, {}, metrics=False)
    for a, b in zip(
        jax.tree_util.tree_leaves(ours.beliefs),
        jax.tree_util.tree_leaves(theirs.beliefs),
        strict=True,
    ):
        assert_same(a, b)


def test_replay_document_gives_release_0_19_0s_own_replay():
    ref = recorded("sources")
    doc = document("sources")
    xs, kinds = sources_trace()
    obs = {"obs_x": xs, "obs_kind": kinds}
    assert_as_recorded("plain", replay_document(doc, obs, {}, metrics=False), ref)
    assert_as_recorded("learn", replay_document(doc, obs, {}, metrics=False, learn=True), ref)
    # the absence flag of a modality reading a source is keyed by the emission it observes
    present = {"x_read": jnp.asarray([True, True, False, True, True])}
    run = replay_document(doc, obs, {}, metrics=False, present=present)
    assert_as_recorded("present", run, ref)


def test_an_action_node_source_is_observed_and_each_reader_has_its_own_emission():
    doc = document("action")
    system, _ = build(doc)
    assert system.emissions == (
        ("x_read", Emission(("x",), realise="stats", kernel="identity")),
        ("x_read_2", Emission(("x",), realise="stats", kernel="identity")),
    )
    assert system.observes == (
        ("prev_x", "x_read"),
        ("prev_x_again", "x_read_2"),
        ("prev_a", "a_x"),
    )
    ref = recorded("action")
    obs = {"obs": jnp.asarray(ref["trace.obs"])}
    actions = {"a_x": jnp.asarray(ref["trace.actions"])}
    assert_as_recorded("by_node", replay_document(doc, obs, actions, metrics=False), ref)
    learned = replay_document(doc, obs, actions, metrics=False, learn=True)
    assert_as_recorded("learn", learned, ref)


def test_a_noise_map_beside_the_option_is_kept():
    doc = document("noise_map")
    system, params = build(doc)
    assert system.observes == (("prev_x", "x_read"),)
    ref = recorded("noise_map")
    run = replay_document(doc, {"obs_x": jnp.asarray(ref["xs"])}, {}, metrics=False)
    assert_recorded(np.asarray(run.beliefs["x"].loc), ref["loc"])
    assert_recorded(np.asarray(run.beliefs["x"].chol), ref["chol"])
    assert_recorded(np.asarray(run.free_energy), ref["free_energy"])
    redescribed, _ = describe(system, params, arrays="inline")
    assert redescribed["factors"]["prev_x"]["likelihood"]["options"] == {
        "noise_map": doc["factors"]["prev_x"]["likelihood"]["options"]["noise_map"]
    }


def test_a_composite_reads_its_components_sources_under_their_prefix():
    doc = document("merge")
    system, _ = build(doc)
    assert [name for name, _ in system.subsystems] == ["a", "b"]
    assert system.observes == (("a.prev_x", "a.x_read"), ("a.prev_type", "a.kind_read"))
    ref = recorded("merge")
    xs, kinds = sources_trace()
    obs = {
        "a.obs_x": xs,
        "a.obs_kind": kinds,
        "b.obs_x": xs[::-1],
        "b.obs_kind": kinds[::-1],
        "b.prev_x": xs,
        "b.prev_type": kinds,
    }
    run = replay_document(doc, obs, {}, metrics=False)
    for key in ref.files:
        if key == "free_energy":
            got = run.free_energy
        else:
            name, field = key.rsplit(".", 1)
            got = getattr(run.beliefs[name], field)
        assert_recorded(np.asarray(got), ref[key], err_msg=key)


def test_the_reported_document_builds():
    system, _ = build(document("reported"))
    assert system.observes == (("prev_x", "x_read"),)
    assert system.emissions == (("x_read", Emission(("x",), realise="stats", kernel="identity")),)


def test_the_emission_name_avoids_every_name_in_the_document():
    doc = document("reported")
    # a modality named x_read: the emission takes the next free name
    doc["variables"]["x_read"] = doc["variables"].pop("obs")
    doc["factors"]["x_read"] = doc["factors"].pop("obs")
    doc["relations"]["observed"] = ["x_read", "prev_x"]
    system, _ = build(doc)
    assert system.observes == (("prev_x", "x_read_2"),)


# ---------------------------------------------------------------- what the writer writes


@pytest.mark.parametrize("name", ["sources", "action", "noise_map", "merge", "reported"])
def test_describing_the_built_system_writes_the_0_20_0_form(name):
    system, params = build(document(name))
    redescribed, arrays = describe(system, params, arrays="inline")
    assert no_datum(redescribed)
    assert read_datum(redescribed, "") is redescribed
    again, again_params = build(redescribed, arrays=arrays)
    assert again == system
    for a, b in zip(
        jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(again_params), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_the_sources_document_redescribes_as_the_native_one():
    system, params = build(document("sources"))
    native, native_params = _own_read_system(
        emissions={
            "x_read": Emission(("x",), realise="mode", kernel="identity"),
            "kind_read": Emission(("kind",), realise="stats", kernel="identity"),
        },
        observes={"prev_x": "x_read", "prev_type": "kind_read"},
    )
    ours = describe(system, params, arrays="inline")
    theirs = describe(native, native_params, arrays="inline")
    if jax.config.jax_enable_x64:
        #  the document's numbers were written in float32 and the native model computes its
        #  tables in float64: the two documents agree to float32 rounding
        assert ours.arrays == theirs.arrays
        assert_documents_close(ours.document, theirs.document)
    else:
        assert ours == theirs


def assert_documents_close(a, b, path="document"):
    """``a == b`` with every float compared to a relative tolerance of 1e-6."""
    assert type(a) is type(b), path
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            assert_documents_close(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            assert_documents_close(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert a == pytest.approx(b, rel=1e-6), path
    else:
        assert a == b, path


def test_describe_parameters_reads_the_document_as_the_native_one():
    system, params = build(document("sources"))
    native, _ = describe(system, params, arrays="inline")
    assert describe_parameters(document("sources")) == describe_parameters(native)


def test_a_document_without_the_option_is_read_as_it_is():
    system, params = _own_read_system()
    doc, _ = describe(system, params, arrays="inline")
    assert read_datum(doc, "") is doc


# ---------------------------------------------------------------- refusals


def _with_datum(datum):
    doc = copy.deepcopy(document("reported"))
    doc["factors"]["prev_x"]["likelihood"]["options"]["datum"] = datum
    return doc


@pytest.mark.parametrize(
    "datum, sentence",
    [
        ("x[t]", "no variable at the previous tick"),
        ("x", "no variable at the previous tick"),
        ("nowhere[t-1]", "no variable at the previous tick"),
        (3, "no variable at the previous tick"),
        ("u_x[t-1]", "neither a state factor nor the action node"),
        ("obs[t-1]", "neither a state factor nor the action node"),
    ],
)
def test_a_datum_release_0_19_0_refused_is_refused_naming_the_option(datum, sentence):
    with pytest.raises(DescriptionError, match=r"options\.datum.*release 0\.19\.0") as info:
        build(_with_datum(datum))
    assert sentence in str(info.value)


def test_a_datum_beside_observes_is_refused():
    doc = _with_datum("x[t-1]")
    doc["factors"]["prev_x"]["likelihood"]["observes"] = "obs"
    with pytest.raises(DescriptionError, match="options, not both|option or observes"):
        build(doc)
