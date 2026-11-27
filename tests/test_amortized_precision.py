"""Float precision on the amortized path: the neural components, the replay format, the
reference environments, the trainers' made-up inputs and the R-AIF reference.

The rule (:mod:`hybrid_aif.dtypes`): a float array a caller passes keeps its dtype, arrays of
different float dtypes promote to the wider, and an array the package makes takes its dtype
from the values it is combined with, else the default float dtype. A network computes in the
dtype of its weights whatever feeds it (``dtype=`` on a constructor chooses the dtype they are
drawn in, ``cast`` changes it). The R-AIF reference runs in float32 in either mode.

Every dtype claim is checked under JAX's 64-bit mode (``with jax.enable_x64(True):``), where
float32 and float64 differ; the inputs' dtypes are asserted first, and each claim has a
contrast that gives the other dtype. A comparison with the default mode computes that side
under ``jax.enable_x64(False)``, so the file runs the same in a process of either mode.
"""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.controls import soft_clip
from hybrid_aif.description import build, describe
from hybrid_aif.distributions.neural import (
    GRUCell,
    LearnedVector,
    MLPCategorical,
    MLPConditional,
    MLPPredictor,
    PredictorHead,
    disc_bins,
    trunc_normal_avg,
    twohot,
    uniform_avg,
)
from hybrid_aif.dtypes import cast
from hybrid_aif.inference.amortized import action_parent
from hybrid_aif.objectives.amortized import _action_vector
from hybrid_aif.reference.envs.mountain_car import MountainCar
from hybrid_aif.reference.envs.tmaze import tmaze
from hybrid_aif.reference.raif import run as run_mc
from hybrid_aif.reference.raif.agent import raif_mountain_car, to_world_batch, train_step
from hybrid_aif.reference.raif.columns import raif_episode
from hybrid_aif.reference.raif.crspp import ROW as CRSPP_ROW
from hybrid_aif.reference.raif.crspp import observe_no_action, resolve
from hybrid_aif.reference.raif.revise import label
from hybrid_aif.reference.raif.seed import bang_bang, collect_episode
from hybrid_aif.reference.raif.world import CARRY_NAME, CODE_NAME, EMBED_VARIABLE, tiny_rssm
from hybrid_aif.train.buffers import (
    ColumnSpec,
    Episode,
    StaticBuffer,
    StreamBuffer,
    add,
    as_episode,
    episode,
    extend,
    sample,
    sample_windows,
    stack_episodes,
)
from hybrid_aif.train.ensemble import ensemble_loss, transitions
from hybrid_aif.train.imagine import _bc_inputs, _slow_update, lambda_return
from hybrid_aif.train.world import assemble, fit_world
from tests.test_description import _written_inline

KEY = jax.random.PRNGKey(3)
F32, F64 = jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)

#: A small R-AIF agent, as the trainer tests size it.
TINY = dict(
    horizon=2,
    head_units=8,
    head_layers=1,
    units=4,
    deter=8,
    num_vars=3,
    num_classes=4,
    embed=5,
    mlp_units=6,
    ensemble_size=2,
)


def _float_dtypes(tree) -> set:
    """The set of float dtypes among a tree's array leaves."""
    return {
        jnp.dtype(x.dtype)
        for x in jax.tree_util.tree_leaves(tree)
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
    }


def _not_float32(x) -> bool:
    """Whether no element of the float64 array ``x`` survives rounding to float32."""
    x = np.asarray(x)
    return x.dtype == np.float64 and bool(np.all(x.astype(np.float32).astype(np.float64) != x))


def _assert_bitwise(a, b) -> None:
    """Same tree structure, and every leaf of the same dtype and bitwise equal."""
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        assert np.asarray(x).dtype == np.asarray(y).dtype
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _networks(dtype):
    """One of each network family (a predictor head of each loss), all from ``KEY``, drawn in
    ``dtype``."""
    return {
        "cond": MLPConditional.from_shapes(
            (3, 2), 2, key=KEY, width=4, depth=1, norm=True, init="trunc_normal_avg", dtype=dtype
        ),
        "pred": MLPPredictor.from_shapes((3, 2), 2, key=KEY, width=4, depth=1, dtype=dtype),
        "cat": MLPCategorical.from_shapes((3, 2), 2, 3, key=KEY, width=4, dtype=dtype),
        "mse": PredictorHead.from_shapes(
            (3, 2), loss="symlog_mse", out_dim=2, key=KEY, width=4, depth=1, dtype=dtype
        ),
        "disc": PredictorHead.from_shapes(
            (3, 2), loss="symlog_disc", num_bins=5, key=KEY, width=4, depth=1, dtype=dtype
        ),
        "binary": PredictorHead.from_shapes(
            (3, 2), loss="binary", key=KEY, width=4, depth=1, dtype=dtype
        ),
        "gru": GRUCell.from_shapes(3, (3, 2), key=KEY, hidden=4, dtype=dtype),
    }


def _inputs():
    """Every input a network reads, as float64 arrays none of whose elements is a float32 value
    (64-bit mode must be on): the parents, a carry, a Gaussian child, a soft categorical code, a
    vector target, a scalar target and a soft binary target."""
    k = jax.random.split(jax.random.PRNGKey(11), 6)
    return {
        "parents": (
            jax.random.normal(k[0], (3,), jnp.float64),
            jax.random.normal(k[1], (2,), jnp.float64),
        ),
        "deter": jax.random.normal(k[2], (3,), jnp.float64),
        "child": jax.random.normal(k[3], (2,), jnp.float64),
        "code": jax.nn.softmax(jax.random.normal(k[4], (2, 3), jnp.float64), axis=-1),
        "vector": 3.0 * jax.random.normal(k[5], (2,), jnp.float64),
        "scalar": jnp.asarray(1.2345678901234, jnp.float64),
        "soft": jnp.asarray(0.1234567890123, jnp.float64),
    }


def _check_inputs(inputs) -> None:
    """The presupposition of every float64-input claim: the inputs are float64 and rounding them
    to float32 changes every element."""
    assert _float_dtypes(inputs) == {F64}
    assert all(_not_float32(x) for x in jax.tree_util.tree_leaves(inputs))


#: Every public method of the networks that takes an input, on the inputs of :func:`_inputs`.
CALLS = {
    "MLPConditional.condition_on": lambda n, i: n["cond"].condition_on(i["parents"]),
    "MLPConditional.log_prob": lambda n, i: n["cond"].log_prob(i["child"], i["parents"]),
    "MLPConditional.sample": lambda n, i: n["cond"].sample(KEY, i["parents"]),
    "MLPConditional.moments": lambda n, i: n["cond"].moments(i["parents"]),
    "MLPConditional.diag_moments": lambda n, i: n["cond"].diag_moments(i["parents"]),
    "MLPConditional.diag_nll": lambda n, i: n["cond"].diag_nll(i["child"], i["parents"]),
    "MLPConditional.nll": lambda n, i: n["cond"].nll(i["child"], i["parents"]),
    "MLPConditional.stats": lambda n, i: n["cond"].stats(i["child"], i["parents"]),
    "MLPPredictor.__call__": lambda n, i: n["pred"](i["parents"]),
    "MLPCategorical.logits": lambda n, i: n["cat"].logits(i["parents"]),
    "MLPCategorical.condition_on": lambda n, i: n["cat"].condition_on(i["parents"]),
    "MLPCategorical.probs": lambda n, i: n["cat"].probs(i["parents"]),
    "MLPCategorical.log_prob": lambda n, i: n["cat"].log_prob(i["code"], i["parents"]),
    "MLPCategorical.sample": lambda n, i: n["cat"].sample(KEY, i["parents"]),
    "MLPCategorical.mode": lambda n, i: n["cat"].mode(i["parents"]),
    "PredictorHead[symlog_mse].__call__": lambda n, i: n["mse"](i["parents"]),
    "PredictorHead[symlog_mse].nll": lambda n, i: n["mse"].nll(i["vector"], i["parents"]),
    "PredictorHead[symlog_mse].predict": lambda n, i: n["mse"].predict(i["parents"]),
    "PredictorHead[symlog_disc].__call__": lambda n, i: n["disc"](i["parents"]),
    "PredictorHead[symlog_disc].nll": lambda n, i: n["disc"].nll(i["scalar"], i["parents"]),
    "PredictorHead[symlog_disc].predict": lambda n, i: n["disc"].predict(i["parents"]),
    "PredictorHead[binary].__call__": lambda n, i: n["binary"](i["parents"]),
    "PredictorHead[binary].nll": lambda n, i: n["binary"].nll(i["soft"], i["parents"]),
    "PredictorHead[binary].predict": lambda n, i: n["binary"].predict(i["parents"]),
    "GRUCell.__call__": lambda n, i: n["gru"](i["deter"], *i["parents"]),
}

DTYPE_IDS = {F32: "float32", F64: "float64"}


def _outputs(nets, parents, deter):
    """Every network's forward outputs on ``parents`` (and ``deter`` for the cell)."""
    mean, scale = nets["cond"].diag_moments(parents)
    return {
        "cond_mean": mean,
        "cond_scale": scale,
        "pred": nets["pred"](parents),
        "cat_logits": nets["cat"].logits(parents),
        "head_predict": nets["disc"].predict(parents),
        "gru": nets["gru"](deter, *parents),
    }


# ============================================================ a network's dtype


def test_the_default_draws_are_the_recorded_float32_weights():
    """With ``dtype=None`` and 64-bit mode off the weights are the draws the code made before
    ``dtype=`` existed (recorded from it); with 64-bit mode on, ``dtype=jnp.float32`` gives
    the same numbers, and ``dtype=None`` draws float64 weights, which are other numbers."""
    recorded = {
        "cond_l0": [-0.2784402668, 0.0663948059, 0.3814038038, 0.5468961596, 0.0101953968],
        "cond_l1": [-0.1964230537, 0.8148119449, -0.6724193096, 0.4807152748],
        "pred_l0": [0.3241778016, 0.0165243186, 0.5688888430],
        "gru": [-0.6708178520, 0.0402058996, -0.0182570573, 0.1172281057, -0.1168641597],
        "tn": [-0.8663320541, 0.9143887162, 0.2391053587],
        "un": [-0.8518948554, 0.8762886524, 0.2979140281],
    }

    def draws(dtype):
        m = MLPConditional.from_shapes(
            (3, 2), 2, key=KEY, width=4, depth=1, norm=True, init="trunc_normal_avg", dtype=dtype
        )
        p = MLPPredictor.from_shapes((3,), 2, key=KEY, width=4, depth=1, dtype=dtype)
        g = GRUCell.from_shapes(3, (2,), key=KEY, hidden=4, init="trunc_normal_avg", dtype=dtype)
        return {
            "cond_l0": m.layers[0].weight[0],
            "cond_l1": m.layers[1].weight[0],
            "pred_l0": p.layers[0].weight[0],
            "gru": g.gate.weight[0, :5],
            "tn": trunc_normal_avg(KEY, (3,), 4, 2, dtype),
            "un": uniform_avg(KEY, (3,), 4, 2, dtype),
        }

    with jax.enable_x64(False):
        for name, value in draws(None).items():
            assert value.dtype == F32, name
            np.testing.assert_allclose(np.asarray(value), recorded[name], rtol=0, atol=1e-9)
    with jax.enable_x64(True):
        for name, value in draws(jnp.float32).items():
            assert value.dtype == F32, name
            np.testing.assert_allclose(np.asarray(value), recorded[name], rtol=0, atol=1e-9)
        for name, value in draws(None).items():
            assert value.dtype == F64, name
            assert not np.allclose(np.asarray(value), recorded[name], atol=1e-3), name


@pytest.mark.parametrize("weights", [F32, F64], ids=DTYPE_IDS.get)
def test_a_networks_dtype_is_its_weights_dtype(weights):
    """The read-only ``dtype`` of each network is the dtype its weights were drawn in, and no
    network carries a precision field of its own: its only float dtypes are its arrays'."""
    with jax.enable_x64(True):
        nets = _networks(weights)
        assert _float_dtypes(nets) == {weights}
        for name, net in nets.items():
            assert net.dtype == weights, name
            assert "dtype" not in {f.name for f in dataclasses.fields(net)}, name
        for name in ("cat", "mse", "disc", "binary"):
            assert nets[name].trunk.dtype == weights, name


@pytest.mark.parametrize("inputs", [F32, F64], ids=DTYPE_IDS.get)
@pytest.mark.parametrize("weights", [F32, F64], ids=DTYPE_IDS.get)
@pytest.mark.parametrize("call", sorted(CALLS))
def test_a_network_computes_in_its_weights_dtype_whatever_its_inputs(call, weights, inputs):
    """Weights and inputs each in float32 or float64: every float the method returns is in the
    weights' dtype. Float32 weights on float64 inputs give, bit for bit, what they give on the
    inputs rounded to float32: the network rounds its inputs to its own precision."""
    with jax.enable_x64(True):
        nets = _networks(weights)
        assert _float_dtypes(nets) == {weights}
        wide = _inputs()
        _check_inputs(wide)
        given = cast(wide, inputs)
        assert _float_dtypes(given) == {inputs}
        out = CALLS[call](nets, given)
        assert _float_dtypes(out) == {weights}
        if weights == F32 and inputs == F64:
            _assert_bitwise(out, CALLS[call](nets, cast(wide, jnp.float32)))


@pytest.mark.parametrize("weights", [F32, F64], ids=DTYPE_IDS.get)
def test_a_learned_vector_computes_in_its_values_dtype(weights):
    """A learned vector has no inputs: its ``dtype`` is its value's and its output is in it,
    whether :meth:`LearnedVector.zeros` made it or the value was given."""
    with jax.enable_x64(True):
        made = LearnedVector.zeros(3, dtype=weights)
        assert made.value.dtype == made.dtype == weights
        assert made().dtype == weights
        value = jnp.asarray([0.1234567890123, -0.3, 2.0], weights)
        given = LearnedVector(value=value)
        assert given.dtype == weights and given().dtype == weights
        assert LearnedVector.zeros(3).dtype == F64  # the default float dtype


def test_a_network_given_float32_computes_in_float32_on_float64_inputs():
    """Float32 weights under 64-bit mode: float64 parents are cast on the way in, so every
    output is float32 and equal to the outputs on the parents rounded to float32 first; the
    same weights cast to float64 compute in float64 on the same parents and give other numbers
    (the float64 inputs carry digits float32 drops)."""
    with jax.enable_x64(True):
        nets = _networks(jnp.float32)
        assert _float_dtypes(nets) == {F32}
        assert all(n.dtype == F32 for n in nets.values())
        wide = _inputs()
        _check_inputs(wide)
        parents, deter = wide["parents"], wide["deter"]
        narrow = _outputs(nets, parents, deter)
        for name, out in narrow.items():
            assert out.dtype == F32, name
        rounded = _outputs(
            nets, tuple(p.astype(jnp.float32) for p in parents), deter.astype(jnp.float32)
        )
        for name in narrow:
            np.testing.assert_array_equal(np.asarray(narrow[name]), np.asarray(rounded[name]))

        widened = cast(nets, jnp.float64)
        assert _float_dtypes(widened) == {F64} and all(n.dtype == F64 for n in widened.values())
        computed = _outputs(widened, parents, deter)
        for name, out in computed.items():
            assert out.dtype == F64, name
            np.testing.assert_allclose(np.asarray(out), np.asarray(narrow[name]), rtol=1e-4)
        assert all(
            not np.array_equal(np.asarray(computed[k]), np.asarray(narrow[k], np.float64))
            for k in computed
        )


def test_a_network_given_float32_scores_a_float64_child_and_target_in_float32():
    """The losses read the child, code or target as an input too: cast to the network's dtype,
    so the loss is float32; the same network cast to float64 scores them in float64."""
    with jax.enable_x64(True):
        nets = _networks(jnp.float32)
        wide = _inputs()
        _check_inputs(wide)
        parents = wide["parents"]
        assert nets["cond"].diag_nll(wide["child"], parents).dtype == F32
        assert nets["cat"].log_prob(wide["code"], parents).dtype == F32
        assert nets["disc"].nll(wide["scalar"], parents).dtype == F32
        assert nets["disc"].bins.dtype == F32
        widened = cast(nets["cond"], jnp.float64)
        assert widened.dtype == F64
        assert widened.diag_nll(wide["child"], parents).dtype == F64
        assert cast(nets["disc"], jnp.float64).bins.dtype == F64


def test_dtype_none_draws_in_the_default_float_dtype_which_the_network_computes_in():
    """``dtype=None`` under 64-bit mode draws float64 weights (the default float dtype), so the
    network computes in float64 and float32 parents widen on the way in; cast to float32, the
    same network computes in float32 on float32 and on float64 parents alike."""
    with jax.enable_x64(True):
        nets = _networks(None)
        assert _float_dtypes(nets) == {F64} and all(n.dtype == F64 for n in nets.values())
        wide = _inputs()
        _check_inputs(wide)
        parents = tuple(p.astype(jnp.float32) for p in wide["parents"])
        deter = wide["deter"].astype(jnp.float32)
        assert _float_dtypes((parents, deter)) == {F32}
        for name, out in _outputs(nets, parents, deter).items():
            assert out.dtype == F64, name
        narrow = cast(nets, jnp.float32)
        assert _float_dtypes(narrow) == {F32} and all(n.dtype == F32 for n in narrow.values())
        for name, out in _outputs(narrow, parents, deter).items():
            assert out.dtype == F32, name
        for name, out in _outputs(narrow, wide["parents"], wide["deter"]).items():
            assert out.dtype == F32, name


def test_a_network_whose_layers_differ_in_dtype_casts_to_its_first_layer_then_promotes():
    """One later layer cast to float64: the network's ``dtype`` is its first layer's (float32),
    the inputs are rounded to it, and the values widen where they meet the float64 layer, so
    the output is float64 but equal to the output on the inputs rounded to float32."""
    with jax.enable_x64(True):
        wide = _inputs()
        _check_inputs(wide)
        parents, deter = wide["parents"], wide["deter"]
        rounded = cast((parents, deter), jnp.float32)
        pred = MLPPredictor.from_shapes(
            (3, 2), 2, key=KEY, width=4, depth=1, norm=False, dtype=jnp.float32
        )
        mixed = cast(pred, jnp.float64, where=lambda n: n.layers[-1])
        assert (mixed.layers[0].weight.dtype, mixed.layers[-1].weight.dtype) == (F32, F64)
        assert mixed.dtype == F32
        out = mixed(parents)
        assert out.dtype == F64
        np.testing.assert_array_equal(np.asarray(out), np.asarray(mixed(rounded[0])))
        whole = cast(pred, jnp.float64)
        assert whole.dtype == F64
        assert not np.array_equal(np.asarray(whole(parents)), np.asarray(out))

        gru = GRUCell.from_shapes(3, (3, 2), key=KEY, hidden=4, norm=False, dtype=jnp.float32)
        mixed = cast(gru, jnp.float64, where=lambda c: c.gate)
        assert (mixed.input_proj.weight.dtype, mixed.gate.weight.dtype) == (F32, F64)
        assert mixed.dtype == F32
        out = mixed(deter, *parents)
        assert out.dtype == F64
        np.testing.assert_array_equal(np.asarray(out), np.asarray(mixed(rounded[1], *rounded[0])))


#: One call per network, for the cast tests.
ONE_CALL = {
    "cond": lambda m, i: m.diag_nll(i["child"], i["parents"]),
    "pred": lambda m, i: m(i["parents"]),
    "cat": lambda m, i: m.log_prob(i["code"], i["parents"]),
    "mse": lambda m, i: m.nll(i["vector"], i["parents"]),
    "disc": lambda m, i: m.nll(i["scalar"], i["parents"]),
    "binary": lambda m, i: m.nll(i["soft"], i["parents"]),
    "gru": lambda m, i: m(i["deter"], *i["parents"]),
    "vec": lambda m, i: m(),
}


def test_cast_sets_the_dtype_a_network_computes_in():
    """A network drawn in float64, then ``cast`` to float32: its ``dtype`` is float32 and it
    computes in float32 on float64 inputs, bitwise as on the inputs rounded to float32. Before
    the cast the same call is float64."""
    with jax.enable_x64(True):
        nets = _networks(jnp.float64)
        nets["vec"] = LearnedVector(value=jax.random.normal(KEY, (3,), jnp.float64))
        wide = _inputs()
        _check_inputs(wide)
        for name, net in nets.items():
            assert net.dtype == F64, name
            assert _float_dtypes(ONE_CALL[name](net, wide)) == {F64}, name
            narrow = cast(net, jnp.float32)
            assert _float_dtypes(narrow) == {F32}, name
            assert narrow.dtype == F32, name
            out = ONE_CALL[name](narrow, wide)
            assert _float_dtypes(out) == {F32}, name
            _assert_bitwise(out, ONE_CALL[name](narrow, cast(wide, jnp.float32)))


def _named_networks(params) -> dict:
    """Every network of a model's parameters, by its path."""
    nets = (MLPConditional, MLPPredictor, MLPCategorical, PredictorHead, GRUCell, LearnedVector)
    flat, _ = jax.tree_util.tree_flatten_with_path(params, is_leaf=lambda x: isinstance(x, nets))
    return {jax.tree_util.keystr(p): x for p, x in flat if isinstance(x, nets)}


@pytest.mark.parametrize("how", ["function", "method"])
def test_cast_with_where_moves_one_network_and_only_it(how):
    """A model drawn in float64 and cast to float32, then its encoder alone cast back to float64
    (``cast(params, ..., where=)`` or ``params.cast``): the encoder computes in float64, every
    other network and array stays float32, and the posterior, a float32 network reading the
    encoder's float64 embedding, computes in float32 on it."""
    with jax.enable_x64(True):
        _, wide = tiny_rssm(2, 1, key=KEY, deter=8, num_vars=3, num_classes=4, units=4, embed=5)
        assert _float_dtypes(wide) == {F64}
        params = cast(wide, jnp.float32)
        assert _float_dtypes(params) == {F32}

        def where(p):
            return p.recognition[EMBED_VARIABLE]

        if how == "function":
            mixed = cast(params, jnp.float64, where=where)
        else:
            mixed = params.cast(jnp.float64, where=where)
        encoder, posterior = mixed.recognition[EMBED_VARIABLE], mixed.recognition[CODE_NAME]
        assert _float_dtypes(encoder) == {F64} and encoder.dtype == F64
        rest = dataclasses.replace(mixed, recognition={CODE_NAME: posterior})
        assert _float_dtypes(rest) == {F32}
        named = _named_networks(mixed)
        assert len(named) == 8
        for path, net in named.items():
            assert net.dtype == (F64 if net is encoder else F32), path

        obs = jnp.asarray([0.1234567890123, -0.3], jnp.float32)
        embed = encoder((obs,))
        assert embed.dtype == F64
        assert _not_float32(embed)
        deter = jnp.zeros(8, jnp.float32)
        logits = posterior.logits((deter, embed))
        assert logits.dtype == F32
        np.testing.assert_array_equal(
            np.asarray(logits), np.asarray(posterior.logits((deter, embed.astype(jnp.float32))))
        )


def _first_weight(net):
    """The array whose dtype is the network's ``dtype``."""
    if isinstance(net, (MLPCategorical, PredictorHead)):
        net = net.trunk
    if isinstance(net, GRUCell):
        return net.input_proj.weight
    if isinstance(net, LearnedVector):
        return net.value
    return net.layers[0].weight


def _filter_outputs(system, params):
    """The world model's filter over three steps, and its decoder's prediction at each step."""
    core, _ = assemble(system, params)
    obs = jnp.asarray([[0.1, 0.2], [0.3, -0.1], [0.0, 0.5]], jnp.float32)
    actions = jnp.asarray([[0.0], [0.5], [-0.5]], jnp.float32)
    states, post, prior = core.observe(KEY, actions, obs)
    decoded = jax.vmap(lambda d, s: core.decoder.predict((d, s.reshape(-1))))(
        states.deter, states.stoch
    )
    return states, post, prior, decoded


@pytest.mark.parametrize("arrays", ["auto", "inline", "file"])
def test_describe_then_build_keeps_a_float32_network_in_default_mode(arrays):
    """A model whose networks were drawn with ``dtype=jnp.float32``, described and built back
    through JSON with 64-bit mode off: every built network's ``dtype`` is float32 and the
    filter and decoder outputs are bitwise the original's."""
    with jax.enable_x64(False):
        system, params = tiny_rssm(
            2, 1, key=KEY, deter=8, num_vars=3, num_classes=4, units=4, dtype=jnp.float32
        )
        document, stored = describe(system, params, arrays=arrays)
        built_system, built = build(json.loads(json.dumps(document)), arrays=stored)
        before, after = _named_networks(params), _named_networks(built)
        assert list(before) == list(after) and len(after) == 8
        for path, net in after.items():
            assert before[path].dtype == net.dtype == F32, path
        _assert_bitwise(_filter_outputs(system, params), _filter_outputs(built_system, built))


@pytest.mark.parametrize("arrays", ["auto", "inline", "file"])
def test_describe_then_build_under_64_bit_mode_follows_the_array_placement(arrays):
    """The same round trip with 64-bit mode on. The description writes a small array inline, as
    plain numbers, which build in the default float dtype (float64 here), and a large one by
    ref, in its own dtype (``docs/description-format.md``): so a built network computes in
    float32 only where its first weight went by ref, and in float64 otherwise. The values are
    the float32 values either way."""
    with jax.enable_x64(True):
        system, params = tiny_rssm(
            2, 1, key=KEY, deter=8, num_vars=3, num_classes=4, units=4, dtype=jnp.float32
        )
        assert _float_dtypes(params) == {F32}
        document, stored = describe(system, params, arrays=arrays)
        _, built = build(json.loads(json.dumps(document)), arrays=stored)
        before, after = _named_networks(params), _named_networks(built)
        assert list(before) == list(after) and len(after) == 8
        placed = set()
        for path, net in after.items():
            first = np.asarray(_first_weight(before[path]))
            assert first.dtype == np.float32
            want = F64 if _written_inline(first, arrays) else F32
            placed.add(want)
            assert net.dtype == want, path
            for a, b in zip(
                jax.tree_util.tree_leaves(before[path]),
                jax.tree_util.tree_leaves(net),
                strict=True,
            ):
                np.testing.assert_array_equal(np.asarray(a, np.float64), np.asarray(b))
        assert placed == {"auto": {F32, F64}, "inline": {F64}, "file": {F32}}[arrays]


def test_float64_is_refused_without_64_bit_mode():
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="64-bit mode"):
            MLPPredictor.from_shapes((2,), 1, key=KEY, width=3, dtype=jnp.float64)
        with pytest.raises(ValueError, match="64-bit mode"):
            LearnedVector.zeros(3, dtype=jnp.float64)
        with pytest.raises(ValueError, match="64-bit mode"):
            disc_bins(5, dtype=jnp.float64)
        with pytest.raises(ValueError, match="not a float dtype"):
            GRUCell.from_shapes(2, (1,), key=KEY, hidden=2, dtype=jnp.int32)


def test_the_bins_and_the_two_hot_take_the_dtype_they_are_read_in():
    with jax.enable_x64(True):
        assert disc_bins(5).dtype == F64
        bins = disc_bins(5, dtype=jnp.float32)
        assert bins.dtype == F32
        target = jnp.asarray(0.3, jnp.float32)
        assert target.dtype == F32
        assert twohot(target, bins).dtype == F32
        assert twohot(jnp.asarray(0.3, jnp.float64), bins).dtype == F64


# ===================================================== the action rescale and inputs


def test_soft_clip_keeps_a_float64_action_and_its_digits():
    """Old behaviour cast to float32; a float64 action just outside the box now comes back
    float64, with the digit float32 would round away."""
    with jax.enable_x64(True):
        action = jnp.asarray([0.1234567890123, -2.0], jnp.float64)
        bounds = (jnp.full((2,), -1.0, jnp.float64), jnp.full((2,), 1.0, jnp.float64))
        assert action.dtype == F64
        out = soft_clip(action, bounds)
        assert out.dtype == F64
        assert float(out[0]) == 0.1234567890123 and float(out[1]) == -1.0
        narrow = soft_clip(action.astype(jnp.float32), tuple(b.astype(jnp.float32) for b in bounds))
        assert narrow.dtype == F32
        assert float(narrow[0]) != 0.1234567890123


def test_a_continuous_action_enters_the_transition_and_the_ensemble_in_its_dtype():
    class Layout:  # the one method these helpers read
        def block(self, f, action):
            return action

    with jax.enable_x64(True):
        action = jnp.asarray([0.1234567890123], jnp.float64)
        assert action.dtype == F64
        (parent,) = action_parent(0, action, (True,), (1,), Layout())
        assert parent.dtype == F64 and float(parent[0]) == 0.1234567890123
        vector = _action_vector(action, (True,), (1,), Layout())
        assert vector.dtype == F64 and float(vector[0]) == 0.1234567890123
        (narrow,) = action_parent(0, action.astype(jnp.float32), (True,), (1,), Layout())
        assert narrow.dtype == F32


# ======================================================================= replay


def test_an_episode_keeps_the_dtype_it_is_built_from():
    with jax.enable_x64(True):
        obs = jnp.asarray([[0.1234567890123, 0.0], [0.2, 0.1]], jnp.float64)
        assert obs.dtype == F64
        ep = episode(
            obs, jnp.zeros((2, 1), jnp.float32), [0.0, -1.0], is_first=[1, 0], is_last=[0, 1]
        )
        assert _float_dtypes(ep) == {F64}
        assert float(ep.obs[0, 0]) == 0.1234567890123
        narrow = episode(
            obs.astype(jnp.float32),
            jnp.zeros((2, 1), jnp.float32),
            jnp.zeros(2, jnp.float32),
            is_first=[1, 0],
            is_last=[0, 1],
        )
        assert _float_dtypes(narrow) == {F32}  # the default cont follows the columns
        built = as_episode({"obs": np.zeros((3, 2), np.float32), "action": np.zeros((3, 1))})
        assert built.action.dtype == F64 and built.reward.dtype == F64


def test_a_buffer_converts_each_float_column_to_the_dtype_it_stores():
    """A buffer stores the four float columns in the dtype it was allocated in (by default the
    default float dtype) and a float task column in its declared dtype: a wider episode written
    to a float32 buffer is stored rounded to float32, a narrower one widens into a float64
    store, and a float64 store keeps a float64 value whole. An integer task column of another
    dtype is still refused."""
    with jax.enable_x64(True):
        value = 0.1234567890123
        spec = {"weight": ColumnSpec((), "float32"), "tag": ColumnSpec((), "int32")}
        wide = episode(
            jnp.full((3, 2), value, jnp.float64),
            jnp.zeros((3, 1), jnp.float64),
            jnp.zeros(3, jnp.float64),
            is_first=[1, 0, 0],
            is_last=[0, 0, 1],
            weight=jnp.full(3, value, jnp.float64),
            tag=jnp.zeros(3, jnp.int32),
        )
        assert _float_dtypes(wide) == {F64}
        for empty, write in ((StaticBuffer.empty, add), (StreamBuffer.empty, extend)):
            default = empty(4, 3, 2, 1, extras=spec)
            assert _float_dtypes((default.data.obs, default.data.cont)) == {F64}
            assert default.data.extras["weight"].dtype == F32
            stored = write(default, wide)
            assert stored.data.obs.dtype == F64
            assert float(stored.data.obs.reshape(-1, 2)[0, 0]) == value
            assert stored.data.extras["weight"].dtype == F32
            assert float(stored.data.extras["weight"].reshape(-1)[0]) == float(np.float32(value))
            narrow = empty(4, 3, 2, 1, extras=spec, dtype=jnp.float32)
            assert _float_dtypes((narrow.data.obs, narrow.data.cont)) == {F32}
            rounded = write(narrow, wide)
            assert _float_dtypes(rounded.data) == {F32}
            assert float(rounded.data.obs.reshape(-1, 2)[0, 0]) == float(np.float32(value))
            widened = write(
                default,
                jax.tree_util.tree_map(
                    lambda x: x.astype(jnp.float32) if x.dtype == F64 else x, wide
                ),
            )
            assert _float_dtypes((widened.data.obs, widened.data.reward)) == {F64}
            wrong = wide._replace(extras={**wide.extras, "tag": jnp.zeros(3, jnp.int64)})
            assert wrong.extras["tag"].dtype == jnp.int64
            with pytest.raises(ValueError, match="task column 'tag' is int64"):
                write(default, wrong)


def test_an_episode_of_numpy_float64_columns_is_stored_in_the_buffers_dtype():
    """An :class:`Episode` built straight from NumPy float64 arrays is written to a buffer and
    stored in the buffer's dtype: float32 with 64-bit mode off, the values the arrays hold
    rounded to float32."""
    t = 5
    ep = Episode(
        obs=np.full((t, 2), 0.1),
        action=np.zeros((t, 1)),
        reward=np.zeros(t),
        cont=np.ones(t),
        is_first=np.eye(1, t, 0, dtype=bool)[0],
        is_last=np.eye(1, t, t - 1, dtype=bool)[0],
        extras={},
    )
    assert ep.obs.dtype == np.float64
    with jax.enable_x64(False):
        for empty, write in ((StaticBuffer.empty, add), (StreamBuffer.empty, extend)):
            stored = write(empty(16, t, 2, 1), ep)
            assert stored.data.obs.dtype == F32 and stored.data.cont.dtype == F32
            assert float(stored.data.obs.reshape(-1, 2)[0, 0]) == float(np.float32(0.1))


def test_a_buffer_draws_the_same_indices_in_either_mode():
    """The draw is int32 in either mode, so the same key picks the same sequences."""

    def draw(stream: bool = False):
        eps = [
            episode(
                jnp.full((2, 2), float(i)),
                jnp.zeros((2, 1)),
                jnp.zeros(2),
                is_first=[1, 0],
                is_last=[0, 1],
            )
            for i in range(9)
        ]
        if stream:
            buf = StreamBuffer.empty(18, 2, 2, 1)
            for ep in eps:
                buf = extend(buf, ep)
            return np.asarray(sample_windows(buf, jax.random.PRNGKey(7), 16).obs[:, :, 0])
        buf = StaticBuffer.empty(9, 2, 2, 1)
        for ep in eps:
            buf = add(buf, ep)
        return np.asarray(sample(buf, jax.random.PRNGKey(7), 16).obs[:, 0, 0])

    with jax.enable_x64(False):
        default, windows = draw(), draw(stream=True)
    with jax.enable_x64(True):
        np.testing.assert_array_equal(draw(), default)
        np.testing.assert_array_equal(draw(stream=True), windows)


# ======================================================================== the envs


def test_mountain_car_draws_its_reset_in_its_dtype_and_steps_in_the_promoted_one():
    with jax.enable_x64(False):
        _, recorded = MountainCar().reset(KEY)
        assert recorded[0].dtype == F32
    with jax.enable_x64(True):
        _, state = MountainCar().reset(KEY)
        assert state[0].dtype == F64
        assert float(state[0][0]) != float(recorded[0][0])  # another draw, in float64
        env = MountainCar(dtype=jnp.float32)
        _, state32 = env.reset(KEY)
        assert state32[0].dtype == F32
        np.testing.assert_array_equal(np.asarray(state32[0]), np.asarray(recorded[0]))
        _, stepped, reward, _ = env.agent_step(None, state32, jnp.ones(1, jnp.float32))
        assert stepped[0].dtype == F32 and reward.dtype == F32
        _, stepped, reward, _ = env.agent_step(None, state32, jnp.ones(1, jnp.float64))
        assert stepped[0].dtype == F64 and reward.dtype == F64


def test_the_tmaze_tables_take_the_probabilities_dtype():
    with jax.enable_x64(True):
        _, params = tmaze(reward_probability=jnp.asarray(0.9, jnp.float32))
        assert _float_dtypes(params) == {F32}
        _, params = tmaze(reward_probability=0.9)
        assert _float_dtypes(params) == {F64}


# ================================================================ the trainers' inputs


def test_the_lambda_return_promotes_its_inputs_together():
    """A float32 critic value against a float64 reward: one float64 scan (the scan's carry
    starts at the value)."""
    with jax.enable_x64(True):
        value = jnp.asarray([1.0, 2.0, 3.0], jnp.float32)
        reward = jnp.asarray([0.1234567890123, 0.5], jnp.float64)
        disc = jnp.asarray([0.9, 0.9], jnp.float32)
        assert value.dtype == F32 and reward.dtype == F64
        out = lambda_return(value, reward, disc, 0.95)
        assert out.dtype == F64
        assert lambda_return(value, reward.astype(jnp.float32), disc, 0.95).dtype == F32


def test_the_self_imitation_inputs_take_the_start_states_dtype():
    with jax.enable_x64(True):
        flags, actions = _bc_inputs(None, None, 3, 1, jnp.float32)
        assert flags.dtype == F32 and actions.dtype == F32
        flags, actions = _bc_inputs(
            jnp.zeros((3, 1), jnp.float64), jnp.ones(3, jnp.float64), 3, 1, jnp.float32
        )
        assert flags.dtype == F64 and actions.dtype == F64


def test_the_slow_critic_mixes_at_the_weights_precision():
    """The mix fraction is not rounded to float32 before it meets float64 weights."""
    with jax.enable_x64(True):
        slow, fast = jnp.zeros(2, jnp.float64), jnp.ones(2, jnp.float64)
        assert slow.dtype == fast.dtype == F64
        mixed, _ = _slow_update(slow, fast, jnp.asarray(1, jnp.int32), 0.02, 1)
        assert mixed.dtype == F64 and float(mixed[0]) == 0.02
        mixed32, _ = _slow_update(
            slow.astype(jnp.float32), fast.astype(jnp.float32), jnp.asarray(1, jnp.int32), 0.02, 1
        )
        assert mixed32.dtype == F32 and float(mixed32[0]) != 0.02


def test_the_ensemble_weights_default_to_the_losses_dtype():
    with jax.enable_x64(True):
        members = tuple(
            MLPConditional.from_shapes((2, 3, 1), 3, key=k, width=4, depth=1)
            for k in jax.random.split(KEY, 2)
        )
        assert _float_dtypes(members) == {F64}
        batch = transitions(
            jnp.zeros((1, 3, 2)), jax.nn.one_hot(jnp.zeros((1, 3, 1), int), 3), jnp.ones((1, 3, 1))
        )
        _, out = ensemble_loss(members, batch)
        assert out.per_member.dtype == F64 and out.weights.dtype == F64
        narrow = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float32) if jnp.issubdtype(x.dtype, jnp.floating) else x,
            members,
        )
        assert _float_dtypes(narrow) == {F32}
        batch32 = cast(batch, jnp.float32)
        assert _float_dtypes(batch32) == {F32}
        _, out = ensemble_loss(narrow, batch32, member_weights=jnp.ones(2, jnp.float64))
        assert out.per_member.dtype == F32 and out.weights.dtype == F64  # kept, not narrowed


# ============================================================= the R-AIF reference


def test_the_reference_agent_is_float32_in_either_mode_with_the_same_weights():
    """Built under 64-bit mode, every network computes in float32 (its ``dtype``) and every
    parameter and optimizer moment is float32 and equal to the default-mode build;
    ``tiny_rssm`` alone, which asks for nothing, is float64 there."""
    with jax.enable_x64(False):
        recorded = raif_mountain_car(KEY, **TINY)
    with jax.enable_x64(True):
        agent = raif_mountain_car(KEY, **TINY)
        assert _float_dtypes(agent.params) == {F32}
        assert _float_dtypes(agent.learn_state) == {F32}
        networks = _named_networks(agent.params)
        #  the world model's 8, the actor, the critic pair, the ensemble and the CRSPP's 4
        assert len(networks) == 8 + 1 + 2 + TINY["ensemble_size"] + 4
        assert all(net.dtype == F32 for net in networks.values())
        for a, b in zip(
            jax.tree_util.tree_leaves((agent.params, agent.learn_state)),
            jax.tree_util.tree_leaves((recorded.params, recorded.learn_state)),
            strict=True,
        ):
            assert np.asarray(a).dtype == np.asarray(b).dtype
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        _, plain = tiny_rssm(2, 1, key=KEY, deter=8, num_vars=3, num_classes=4, units=4)
        assert _float_dtypes(plain) == {F64}
        assert all(net.dtype == F64 for net in _named_networks(plain).values())


def test_the_reference_collects_and_trains_in_float32_under_64_bit_mode():
    """A bang-bang episode and one training step under 64-bit mode: the episode, the stepped
    parameters, the optimizer carry and the losses are float32, and the numbers are the
    default mode's (the episode bitwise, the losses to float32 rounding)."""
    env = MountainCar(time_limit=12)

    def run(widen: bool = False):
        agent = raif_mountain_car(KEY, **TINY)
        ep = label(collect_episode(env, bang_bang, jax.random.PRNGKey(1), can_self_imitate=True))
        batch = stack_episodes([ep, ep], 8)
        if widen:  # a caller's float64 batch: the step casts it to the agent's float32
            batch = cast(batch, jnp.float64)
            assert _float_dtypes(batch) == {F64}
        stepped, losses = train_step(agent, batch, key=jax.random.PRNGKey(2), horizon=2)
        return ep, stepped, losses

    with jax.enable_x64(False):
        ep0, _, losses0 = run()
        assert _float_dtypes(losses0) == {F32}
    with jax.enable_x64(True):
        ep, stepped, losses = run(widen=True)
        assert _float_dtypes(ep) == {F32}
        for a, b in zip(jax.tree_util.tree_leaves(ep), jax.tree_util.tree_leaves(ep0), strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        assert _float_dtypes(stepped.params) == {F32}
        assert _float_dtypes(stepped.learn_state) == {F32}
        assert _float_dtypes(losses) == {F32}
        #  Not bitwise: optax's Adam bias correction computes ``1 - decay**count`` from a Python
        #  float and the int32 count, which is float64 under 64-bit mode before it is cast to the
        #  moments' float32, so the stepped weights, and the losses read after them, differ in
        #  the last bits (by up to about 1e-5 relative). With that correction computed in
        #  float32 in both modes the step is bitwise the same.
        for a, b in zip(
            jax.tree_util.tree_leaves(losses), jax.tree_util.tree_leaves(losses0), strict=True
        ):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)


def test_the_preference_filter_computes_in_its_networks_dtype_whatever_the_embeddings():
    """The CRSPP's filter (its cell, prior and posterior) computes in its networks' dtype: on
    float64 embeddings a float32 CRSPP keeps a float32 carry and float32 logits, and a float64
    CRSPP on float32 embeddings computes in float64. A cell cast wider than the initial carry
    widens the carry (the scan settles on the wider dtype) instead of breaking the scan."""
    with jax.enable_x64(True):
        agent = raif_mountain_car(KEY, **TINY)
        parts = resolve(agent.crspp, agent.crspp_params)
        assert _float_dtypes(parts.parts) == {F32}
        embeds = jax.random.normal(KEY, (4, 5), jnp.float64)
        _check_inputs(embeds)
        out = observe_no_action(parts, KEY, embeds)
        assert _float_dtypes(out) == {F32}
        _assert_bitwise(out, observe_no_action(parts, KEY, embeds.astype(jnp.float32)))

        wide = resolve(agent.crspp, cast(agent.crspp_params, jnp.float64))
        assert _float_dtypes(wide.parts) == {F64}
        assert _float_dtypes(observe_no_action(wide, KEY, embeds.astype(jnp.float32))) == {F64}

        cell = cast(agent.crspp_params, jnp.float64, where=lambda p: p.transition[CARRY_NAME])
        mixed = resolve(agent.crspp, cell)
        assert mixed.initial().dtype == F32 and mixed.gru.map.dtype == F64
        deters, codes, post, prior = observe_no_action(mixed, KEY, embeds)
        assert deters.dtype == F64  # the carry, widened by the float64 cell
        assert codes.dtype == post.dtype == prior.dtype == F32  # float32 networks read it


def test_the_world_model_filter_computes_in_its_networks_dtype_whatever_the_observations():
    """The world model's filter and open-loop rollout compute in its networks' dtype: float64
    observations and actions leave a float32 model's carry float32, and a float64 model on
    float32 ones computes in float64. A cell cast wider than the initial carry widens it."""
    with jax.enable_x64(True):
        agent = raif_mountain_car(KEY, **TINY)
        core, _ = assemble(agent.system, agent.params)
        assert _float_dtypes(core) == {F32}
        obs = jnp.full((3, 2), 0.1234567890123, jnp.float64)
        actions = jnp.full((3, 1), 0.3333333333333, jnp.float64)
        _check_inputs((obs, actions))
        states, post, prior = core.observe(KEY, actions, obs)
        assert _float_dtypes((states, post, prior)) == {F32}
        rolled, logits = core.open_loop(KEY, core.initial_state(), actions)
        assert _float_dtypes((rolled, logits)) == {F32}

        wide, _ = assemble(agent.system, cast(agent.params, jnp.float64))
        assert _float_dtypes(wide) == {F64}
        narrow_in = cast((actions, obs), jnp.float32)
        assert _float_dtypes(wide.observe(KEY, *narrow_in)) == {F64}

        cell = cast(agent.params, jnp.float64, where=lambda p: p.transition[CARRY_NAME])
        mixed, _ = assemble(agent.system, cell)
        assert mixed.initial().dtype == F32 and mixed.gru.map.dtype == F64
        states, post, prior = mixed.observe(KEY, actions, obs)
        assert states.deter.dtype == F64  # the carry, widened by the float64 cell
        assert _float_dtypes((states.stoch, post, prior)) == {F32}


def test_a_reference_episode_is_float32_whatever_it_is_built_from():
    with jax.enable_x64(True):
        obs = jnp.full((2, 2), 0.1234567890123, jnp.float64)
        assert obs.dtype == F64
        ep = raif_episode(
            obs=obs,
            action=jnp.zeros((2, 1), jnp.float64),
            reward=jnp.zeros(2, jnp.float64),
            is_success=[False, True],
            has_succeeded=[False, True],
            can_self_imitate=[1.0, 0.0],
            is_first=[True, False],
            is_last=[False, True],
        )
        assert _float_dtypes(ep) == {F32}
        assert float(ep.obs[0, 0]) == float(np.float32(0.1234567890123))
        assert float(ep.obs[0, 0]) != 0.1234567890123


def test_the_reference_harness_collects_in_float32_under_64_bit_mode(tmp_path):
    """``run.build`` under 64-bit mode: the seeds' state and their buffers are float32, and a
    collection chunk handed a plain (default-dtype) environment still records float32."""
    cfg = run_mc.Config(
        out=str(tmp_path / "run"),
        seeds=(0,),
        steps=8,
        chunk_steps=8,
        batch_size=2,
        capacity=256,
        small=True,
        checkpoint_every=1e9,
        time_limit=6,
    )
    with jax.enable_x64(True):
        system, _, env, state, buffers, _, _, _ = run_mc.build(cfg)
        assert _float_dtypes(state) == {F32}
        assert _float_dtypes([b.data for pair in buffers for b in pair]) == {F32}
        plain = MountainCar(time_limit=6)
        assert plain.dtype is None
        collect = run_mc.make_collect_fn(system, plain, 4)
        params = jax.tree_util.tree_map(lambda x: x[0], state.params)
        carry = jax.tree_util.tree_map(lambda x: x[0], state.collect_carry())
        carry, stream = collect(params, carry)
        assert _float_dtypes(stream) == {F32}
        assert _float_dtypes(carry) == {F32}


def test_the_preference_rollout_widens_a_float32_start_to_its_float64_parameters():
    """The cosine row rolls the CRSPP forward from the caller's start beliefs: float32 beliefs
    under a float64 CRSPP widen the rollout's carry instead of breaking its scan, and the
    row's value is float64."""
    from hybrid_aif.reference.raif.agent import to_world_batch
    from hybrid_aif.reference.raif.crspp import ROW, crspp_recognize, embeddings
    from hybrid_aif.train.imagine import imagine
    from hybrid_aif.train.world import posterior_states

    with jax.enable_x64(True):
        agent = raif_mountain_car(KEY, **TINY)
        ep = label(
            collect_episode(
                MountainCar(time_limit=6), bang_bang, jax.random.PRNGKey(1), can_self_imitate=True
            )
        )
        batch = stack_episodes([ep], 4)
        world = to_world_batch(batch)
        states = posterior_states(agent.system, agent.params, world, key=KEY)
        beliefs = agent.system.beliefs_from_states(agent.params, states)
        flat = {k: jnp.reshape(x, (4, *x.shape[2:])) for k, x in beliefs.items()}
        embeds = embeddings(agent.system, agent.params, world.obs)
        aux = crspp_recognize(agent.crspp, agent.crspp_params, embeds, key=KEY, is_first=None)
        aux = agent.crspp.from_positional(
            "beliefs", tuple(jnp.reshape(x, (4, *x.shape[2:])) for x in aux)
        )
        assert _float_dtypes(aux) == {F32} and _float_dtypes(flat) == {F32}
        params = cast(agent.params, jnp.float64, where=lambda p: p.scoring[ROW])
        assert _float_dtypes(params.scoring[ROW]) == {F64}
        assert _float_dtypes(params.transition) == {F32}
        _, _, metrics = imagine(
            agent.system,
            params,
            flat,
            agent.system.init_learn_state(params),
            key=KEY,
            horizon=2,
            aux_beliefs={ROW: aux},
        )
        assert metrics.per_term[ROW].dtype == F64
        assert bool(jnp.isfinite(metrics.per_term[ROW]))


# ================================================= an amortized model in one dtype


def _amortized_model(name: str):
    """``(system, params)`` of the tiny RSSM (no objective terms, drawn in the default float
    dtype) or of the R-AIF agent (drawn in float32)."""
    if name == "tiny_rssm":
        return tiny_rssm(
            2, 1, key=KEY, policy_len=2, deter=8, num_vars=3, num_classes=4, units=4, embed=5
        )
    agent = raif_mountain_car(KEY, **TINY)
    return agent.system, agent.params


@pytest.mark.parametrize("dtype", [F32, F64], ids=DTYPE_IDS.get)
@pytest.mark.parametrize("name", ["tiny_rssm", "agent"])
def test_an_amortized_model_in_one_dtype_computes_in_that_dtype(name, dtype):
    """Under 64-bit mode a model cast to one dtype, and the optimizer carry it makes, return that
    dtype only, from each face of one tick (the planning faces on the agent, which mounts
    objective terms) and from one training step (``fit_world``, or the reference's
    ``train_step``), every input it is handed being in the other dtype. The float64 case is the
    guard of an array the package makes in the default float dtype; the float32 case is the one
    that catches it."""
    other = F64 if dtype == F32 else F32
    with jax.enable_x64(True):
        system, params = _amortized_model(name)
        params = cast(params, dtype)
        learn_state = system.init_learn_state(params)
        assert _float_dtypes((params, learn_state)) == {dtype}
        assert all(net.dtype == dtype for net in _named_networks(params).values())
        results = {}
        priors = system.initial_priors(params)
        results["initial_priors"] = priors
        observed = system.from_positional(
            "observations", (jnp.asarray([0.1234567890123, -0.3], other), None, None)
        )
        beliefs = system.infer_states(params, observed, priors)
        results["infer_states"] = beliefs
        beliefs = beliefs[0]
        actions = jnp.full((2, 1), 0.3333333333333, other)
        if name == "agent":
            action = system.act(KEY, params, beliefs, plan_key=KEY)
            results["act"] = action
            action = action[0]
            results["infer_policies"] = system.infer_policies(params, beliefs, plan_key=KEY)
            results["infer_policies(metrics=False)"] = system.infer_policies(
                params, beliefs, metrics=False, plan_key=KEY
            )
        else:
            action = system.from_positional("actions", actions[0], params)
        results["update_empirical_prior"] = system.update_empirical_prior(params, beliefs, action)
        results["predict_beliefs"] = system.predict_beliefs(params, beliefs, actions)

        ep = label(
            collect_episode(
                MountainCar(time_limit=12), bang_bang, jax.random.PRNGKey(1), can_self_imitate=True
            )
        )
        batch = cast(stack_episodes([ep, ep], 8), other)
        assert _float_dtypes(batch) == {other}
        if name == "agent":
            agent = raif_mountain_car(KEY, **TINY)._replace(
                params=params,
                crspp_params=params.scoring[CRSPP_ROW],
                learn_state=learn_state,
            )
            results["train_step"] = train_step(agent, batch, key=KEY, horizon=2)
        else:
            results["fit_world"] = fit_world(
                system, params, learn_state, to_world_batch(batch), key=KEY, lr=1e-3
            )
        for face, result in results.items():
            assert _float_dtypes(result) == {dtype}, face


@pytest.mark.parametrize("weights", [F32, F64], ids=DTYPE_IDS.get)
def test_an_integer_or_boolean_input_becomes_a_float_of_the_networks_dtype(weights):
    """Integer parents, an integer carry and a boolean target are read as floats of the
    network's dtype: the outputs are in it and equal to those on the same values given as
    floats of it."""
    with jax.enable_x64(True):
        nets = _networks(weights)
        ints = (jnp.asarray([1, 0, -2], jnp.int64), jnp.asarray([3, 1], jnp.int32))
        deter = jnp.asarray([1, 0, 1], jnp.int32)
        assert all(jnp.issubdtype(x.dtype, jnp.integer) for x in (*ints, deter))
        floats = tuple(x.astype(weights) for x in ints)
        for name, out in _outputs(nets, ints, deter).items():
            assert out.dtype == weights, name
        _assert_bitwise(_outputs(nets, ints, deter), _outputs(nets, floats, deter.astype(weights)))
        target = jnp.asarray(True)
        assert target.dtype == jnp.bool_
        out = nets["binary"].nll(target, ints)
        assert out.dtype == weights
        _assert_bitwise(out, nets["binary"].nll(jnp.asarray(1.0, weights), floats))


# ============================================ the columns and the trainer carries the package makes


def test_a_float64_buffer_stores_the_action_column_in_float64():
    """A buffer allocated in float64 stores the action column in float64, as it stores the other
    float columns, and keeps a float64 action written to it whole."""
    with jax.enable_x64(True):
        value = 0.1234567890123
        ep = episode(
            jnp.zeros((3, 2), jnp.float64),
            jnp.full((3, 1), value, jnp.float64),
            jnp.zeros(3, jnp.float64),
            is_first=[1, 0, 0],
            is_last=[0, 0, 1],
        )
        assert ep.action.dtype == F64 and _not_float32(ep.action)
        for empty, write in ((StaticBuffer.empty, add), (StreamBuffer.empty, extend)):
            for allocated in (empty(4, 3, 2, 1), empty(4, 3, 2, 1, dtype=jnp.float64)):
                assert allocated.data.action.dtype == F64
                stored = write(allocated, ep)
                assert stored.data.action.dtype == F64
                assert float(stored.data.action.reshape(-1)[0]) == value
            narrow = write(empty(4, 3, 2, 1, dtype=jnp.float32), ep)
            assert narrow.data.action.dtype == F32
            assert float(narrow.data.action.reshape(-1)[0]) == float(np.float32(value))


def test_an_episode_of_integer_columns_fills_its_reward_and_cont_in_the_default_float_dtype():
    """``as_episode`` from integer columns, with the reward, the cont or both left to be filled
    in: the four float columns, the filled-in ones among them, are in the default float dtype
    (float64 under 64-bit mode, float32 without), nothing among the columns being a float to
    read one from."""
    base = {"obs": np.zeros((3, 2), np.int32), "action": np.ones((3, 1), np.int32)}
    cases = (
        base,
        {**base, "reward": np.asarray([0, 1, 0], np.int32)},
        {**base, "cont": np.asarray([1, 1, 0], np.int32)},
    )
    for columns in cases:
        assert all(np.issubdtype(c.dtype, np.integer) for c in columns.values())
        for x64, want in ((True, F64), (False, F32)):
            with jax.enable_x64(x64):
                built = as_episode(columns)
                core = (built.obs, built.action, built.reward, built.cont)
                assert {jnp.dtype(c.dtype) for c in core} == {want}, (sorted(columns), x64)
                want_cont = columns.get("cont", np.ones(3))
                np.testing.assert_array_equal(np.asarray(built.cont), want_cont)
                want_reward = columns.get("reward", np.zeros(3))
                np.testing.assert_array_equal(np.asarray(built.reward), want_reward)


def test_the_self_imitation_inputs_left_out_are_zeros_in_the_start_states_dtype():
    """With neither expert actions nor weights given, the zeros that stand in for them are in
    the start states' dtype, float64 as well as float32."""
    with jax.enable_x64(True):
        for dtype in (F32, F64):
            flags, actions = _bc_inputs(None, None, 3, 2, dtype)
            assert (flags.dtype, actions.dtype) == (dtype, dtype)
            assert flags.shape == (3,) and actions.shape == (3, 2)
            assert not np.any(np.asarray(flags)) and not np.any(np.asarray(actions))


def _imagine_inputs(agent):
    """Start beliefs for four imagined trajectories of ``agent`` and the preference row's
    auxiliary beliefs at them, from one labelled episode."""
    from hybrid_aif.reference.raif.crspp import crspp_recognize, embeddings
    from hybrid_aif.train.world import posterior_states

    ep = label(
        collect_episode(
            MountainCar(time_limit=6), bang_bang, jax.random.PRNGKey(1), can_self_imitate=True
        )
    )
    world = to_world_batch(stack_episodes([ep], 4))
    states = posterior_states(agent.system, agent.params, world, key=KEY)
    beliefs = agent.system.beliefs_from_states(agent.params, states)
    flat = {k: jnp.reshape(x, (4, *x.shape[2:])) for k, x in beliefs.items()}
    embeds = embeddings(agent.system, agent.params, world.obs)
    aux = crspp_recognize(agent.crspp, agent.crspp_params, embeds, key=KEY, is_first=None)
    aux = agent.crspp.from_positional(
        "beliefs", tuple(jnp.reshape(x, (4, *x.shape[2:])) for x in aux)
    )
    return flat, aux


@pytest.mark.parametrize("dtype", [F32, F64], ids=DTYPE_IDS.get)
def test_an_imagine_step_keeps_its_carries_and_metrics_in_the_agents_dtype(dtype):
    """The R-AIF agent cast to one dtype, its start beliefs in it: the learn state the agent
    seeds (the return normaliser's statistics among its moments) and what one ``imagine`` step
    returns, the params, the learn state and every metric (the self-imitation weight, made up
    when none is given, among them), are in that dtype."""
    from hybrid_aif.objectives.amortized import RETNORM_KEY
    from hybrid_aif.reference.raif.crspp import ROW
    from hybrid_aif.train.imagine import imagine

    with jax.enable_x64(True):
        agent = raif_mountain_car(KEY, **TINY)
        flat, aux = _imagine_inputs(agent)
        params = cast(agent.params, dtype)
        start, aux = cast(flat, dtype), cast(aux, dtype)
        assert _float_dtypes((params, start, aux)) == {dtype}
        learn_state = agent.system.init_learn_state(params)
        assert learn_state.moments[RETNORM_KEY].dtype == dtype
        assert _float_dtypes(learn_state) == {dtype}
        stepped, new_state, metrics = imagine(
            agent.system, params, start, learn_state, key=KEY, horizon=2, aux_beliefs={ROW: aux}
        )
        assert metrics.bc_weight.dtype == dtype
        assert new_state.moments[RETNORM_KEY].dtype == dtype
        for name, part in (("params", stepped), ("learn_state", new_state), ("metrics", metrics)):
            assert _float_dtypes(part) == {dtype}, name
        assert bool(jnp.isfinite(metrics.actor_loss)) and bool(jnp.isfinite(metrics.critic_loss))


@pytest.mark.parametrize("weights, given", [(F64, F32), (F32, F64)], ids=["wider", "narrower"])
def test_an_imagine_step_runs_from_start_beliefs_in_another_dtype_than_the_agents(weights, given):
    """The R-AIF agent cast to one dtype and start beliefs in the other: the networks compute in
    their weights' dtype, so the imagined rollout returns a carry of another dtype than the start
    it was handed, and the step runs all the same, the learned params and the learn state in the
    agent's dtype."""
    from hybrid_aif.reference.raif.crspp import ROW
    from hybrid_aif.train.imagine import imagine

    with jax.enable_x64(True):
        agent = raif_mountain_car(KEY, **TINY)
        flat, aux = _imagine_inputs(agent)
        params = cast(agent.params, weights)
        start, aux = cast(flat, given), cast(aux, given)
        assert _float_dtypes(params) == {weights} and _float_dtypes((start, aux)) == {given}
        learn_state = agent.system.init_learn_state(params)
        stepped, new_state, metrics = imagine(
            agent.system, params, start, learn_state, key=KEY, horizon=2, aux_beliefs={ROW: aux}
        )
        assert _float_dtypes(stepped) == {weights}
        assert _float_dtypes(new_state) == {weights}
        assert bool(jnp.isfinite(metrics.actor_loss)) and bool(jnp.isfinite(metrics.critic_loss))


def test_an_imagined_step_feeds_a_float64_action_to_the_transition_whole():
    """One sampled imagination step of a float64 world model on a float64 action not
    representable in float32: its carry is the one ``update_empirical_prior`` predicts from the
    same beliefs and action, bit for bit, and differs from the carry the action rounded to
    float32 gives."""
    from hybrid_aif.train.imagine import _code_and_carry, _sampled_step

    with jax.enable_x64(True):
        system, params = _amortized_model("tiny_rssm")
        params = cast(params, F64)
        assert _float_dtypes(params) == {F64}
        priors = system.initial_priors(params)
        beliefs = tuple(priors[n] for n in system.names.states)
        code, carry = _code_and_carry(system._amortized_structure(params, "imagine"), system)
        action = jnp.asarray([0.1234567890123], jnp.float64)
        assert _not_float32(action)

        def carried(a):
            new, _, _ = _sampled_step(params, system, carry, code, beliefs, a, KEY)
            return new[carry]

        got = carried(action)
        pushed = system.update_empirical_prior(
            params, priors, system.from_positional("actions", action, params)
        )
        want = pushed[system.names.states[carry]]
        assert got.dtype == want.dtype == F64
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        rounded = carried(action.astype(jnp.float32).astype(jnp.float64))
        assert not np.array_equal(np.asarray(rounded), np.asarray(got))
