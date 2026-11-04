"""A recognition entry and a neural likelihood beside analytic factors.

A sticky categorical ``terrain``, a mode-aligned mixture-carried 2-D Gaussian ``x`` riding it, a
tabular ``cue`` over ``terrain``, a linear-Gaussian ``fix`` over ``x``, and an image modality
``img`` whose likelihood is a frozen ``MLPConditional`` over ``x`` and ``terrain``. A frozen
``MLPConditional`` recognition entry reads ``img`` and measures ``x`` (design:
recognition-beside-analytic.md R2 to R5).
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalGaussian,
    MixtureGaussian,
    MLPConditional,
    MLPPredictor,
)
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianTransition, Recognition
from hybrid_aif.system import System, SystemParams

M, K = 3, 6
STICKY = jnp.log(jnp.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]))
CUE = jnp.log(jnp.asarray([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]))
INNOV = ConditionalGaussian(
    loc=jnp.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.5]]),
    chol=jnp.stack([0.3 * jnp.eye(2)] * 3),
    parent_ndims=1,
)
TERRAIN = Categorical.from_probs(jnp.asarray([0.5, 0.3, 0.2]))
_WITHIN = np.asarray([[0.6, 0.4], [0.5, 0.5], [0.3, 0.7]])
X_PRIOR = MixtureGaussian(
    log_weights=jnp.log(jnp.asarray((np.asarray(TERRAIN.probs)[:, None] * _WITHIN).reshape(-1))),
    means=jnp.asarray([[-1.0, 0.0], [-0.6, 0.3], [0.0, -0.2], [0.3, 0.1], [0.9, 0.4], [1.3, 0.6]]),
    chols=jnp.stack([0.4 * jnp.eye(2)] * K),
)
PRIORS = {"terrain": TERRAIN, "x": X_PRIOR}
KEY = jax.random.PRNGKey(0)
IMG = jax.random.normal(jax.random.PRNGKey(7), (16,))
OBS = {"cue": jnp.asarray(1), "fix": jnp.asarray([0.5, 0.2])}
IMG_MODULE = MLPConditional.from_shapes((2, 3), 16, key=KEY, width=8, depth=1, frozen=True)
REC_MODULE = MLPConditional.from_shapes(
    (16,), 2, key=jax.random.PRNGKey(5), width=8, depth=1, frozen=True, init_scale=0.3
)


IMG_MODULE_TAB = MLPConditional.from_shapes((3,), 16, key=KEY, width=8, depth=1, frozen=True)


def build(img=True, recognition=(), modules=None, lg_rec=None, extra_likelihood=None):
    """The model; ``lg_rec=(y, scale)`` adds a last linear-Gaussian modality ``rec`` over ``x``."""
    mods, deps = ["cue", "fix"], [(0,), (1,)]
    lik = {
        "cue": ConditionalCategorical(log_table=CUE, parent_ndims=1),
        "fix": LinearGaussianObservation(H=jnp.eye(2), c=jnp.zeros(2), noise=0.4 * jnp.eye(2)),
    }
    if img:
        mods.append("img")
        deps.append((1, 0))
        lik["img"] = IMG_MODULE
    if lg_rec is not None:
        mods.append("rec")
        deps.append((1,))
        lik["rec"] = LinearGaussianObservation(
            H=jnp.eye(2), c=jnp.zeros(2), noise=jnp.diag(lg_rec[1])
        )
    system = System(
        A_deps=tuple(deps),
        B_deps=((0,), (0,)),
        controlled=(False, False),
        policy_len=1,
        mixture_k=K,
        mixture_schedule="mode_aligned",
        modality_names=tuple(mods),
        factor_names=("terrain", "x"),
        recognition=tuple(recognition),
    )
    params = SystemParams(
        likelihood=lik,
        transition={
            "terrain": ConditionalCategorical(log_table=STICKY, parent_ndims=1),
            "x": LinearGaussianTransition(F=0.9 * jnp.eye(2), innovations=INNOV),
        },
        initial_state_prior={"terrain": TERRAIN, "x": X_PRIOR},
        recognition=dict(modules or {}),
    )
    return system, params


def _leaves(beliefs):
    return jax.tree_util.tree_leaves(beliefs)


def _entry_system():
    return build(recognition=(Recognition("x", ("obs:img",)),), modules={"x": REC_MODULE})


def _reference():
    """The model without ``img`` and without the entry."""
    system, params = build(img=False)
    return system.infer_states(params, OBS, priors=PRIORS, detail=True)


def _lg_reference(y, scale):
    system, params = build(img=False, lg_rec=(y, scale))
    return system.infer_states(params, {**OBS, "rec": y}, priors=PRIORS, detail=True)


def _close(a, b, atol=1e-5):
    for u, v in zip(_leaves(a), _leaves(b), strict=True):
        np.testing.assert_allclose(np.asarray(u), np.asarray(v), atol=atol, rtol=1e-5)


# ------------------------------------------------------------------ the acceptance list


def test_without_the_entry_and_with_img_absent_the_beliefs_are_the_libraries_bit_for_bit():
    ref, ref_detail = _reference()
    system, params = build()
    got, detail = system.infer_states(
        params, {**OBS, "img": None}, priors=PRIORS, detail=True, present={"img": False}
    )
    for u, v in zip(_leaves(got), _leaves(ref), strict=True):
        np.testing.assert_array_equal(np.asarray(u), np.asarray(v))
    np.testing.assert_array_equal(
        np.asarray(detail.free_energy), np.asarray(ref_detail.free_energy)
    )


def test_an_absent_img_leaves_the_entry_silent():
    ref, ref_detail = _reference()
    system, params = _entry_system()
    got, detail = system.infer_states(
        params, {**OBS, "img": None}, priors=PRIORS, detail=True, present={"img": False}
    )
    for u, v in zip(_leaves(got), _leaves(ref), strict=True):
        np.testing.assert_array_equal(np.asarray(u), np.asarray(v))
    np.testing.assert_array_equal(
        np.asarray(detail.free_energy), np.asarray(ref_detail.free_energy)
    )
    assert float(detail.recognition_log_evidence["recognition:x"]) == 0.0


def test_the_entry_is_a_linear_gaussian_modality_whose_observation_is_the_modules_mean():
    system, params = _entry_system()
    got, _ = system.infer_states(params, {**OBS, "img": IMG}, priors=PRIORS, detail=True)
    y, scale = REC_MODULE.diag_moments((IMG,))
    ref, _ = _lg_reference(y, scale)
    _close(got["x"], ref["x"])
    _close(got["terrain"], ref["terrain"])
    plain, _ = _reference()
    assert float(jnp.max(jnp.abs(got["x"].mean() - plain["x"].mean()))) > 1e-3


def _log_normal(y, mu, cov):
    y, mu, cov = (np.asarray(a, np.float64) for a in (y, mu, cov))
    d = y - mu
    return -0.5 * (
        len(y) * math.log(2 * math.pi) + np.linalg.slogdet(cov)[1] + d @ np.linalg.solve(cov, d)
    )


def _covered_img_term(q):
    """``- E_q[log p(img | x, terrain)]`` to second order at each component's mean, the terrain
    one-hot of the component's bank."""
    qw = np.exp(np.asarray(q.log_weights, np.float64))
    total = 0.0
    for j in range(K):
        onehot = jax.nn.one_hot(j // (K // M), M)
        mu = q.means[j]
        S = np.asarray(q.chols[j], np.float64)
        S = S @ S.T
        mean, sc = IMG_MODULE._head((mu, onehot))
        jac = np.asarray(
            jax.jacfwd(lambda x, oh=onehot: IMG_MODULE._head((x, oh))[0])(mu), np.float64
        )
        Rm = np.diag(np.asarray(sc, np.float64) ** 2)
        total += qw[j] * (
            -_log_normal(IMG, mean, Rm) + 0.5 * np.trace(np.linalg.solve(Rm, jac @ S @ jac.T))
        )
    return total


def test_the_free_energy_is_the_r5_formula():
    system, params = _entry_system()
    got, detail = system.infer_states(params, {**OBS, "img": IMG}, priors=PRIORS, detail=True)
    pre, pre_detail = _reference()
    y, scale = (np.asarray(a, np.float64) for a in REC_MODULE.diag_moments((IMG,)))
    R = np.diag(scale**2)
    # log Z: the mixture under the mode belief the tabular stage and fix left, bank by bank
    w = np.exp(np.asarray(pre["x"].log_weights, np.float64))
    bank = K // M
    terms = []
    for j in range(K):
        b = j // bank
        within = w[j] / w[b * bank : (b + 1) * bank].sum()
        S = np.asarray(pre["x"].chols[j], np.float64)
        S = S @ S.T
        terms.append(
            math.log(float(pre["terrain"].probs[b]) * within)
            + _log_normal(y, pre["x"].means[j], S + R)
        )
    log_z = float(jax.scipy.special.logsumexp(jnp.asarray(terms)))
    # E_q[log m]: components of the final belief weighted by its weights
    q = got["x"]
    qw = np.exp(np.asarray(q.log_weights, np.float64))
    expect = 0.0
    for j in range(K):
        S = np.asarray(q.chols[j], np.float64)
        S = S @ S.T
        expect += qw[j] * (_log_normal(y, q.means[j], R) - 0.5 * np.trace(np.linalg.solve(R, S)))
    nll = _covered_img_term(q)
    expected = float(pre_detail.free_energy) - log_z + expect + nll
    np.testing.assert_allclose(float(detail.free_energy), expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        float(detail.recognition_log_evidence["recognition:x"]), log_z, rtol=1e-4, atol=1e-4
    )


def test_an_auxiliary_chain_is_evaluated_in_the_order_its_parents_derive():
    embed = MLPPredictor.from_shapes((16,), 5, key=jax.random.PRNGKey(9), width=6, depth=1)
    module = MLPConditional.from_shapes(
        (5,), 2, key=jax.random.PRNGKey(5), width=8, depth=1, frozen=True, init_scale=0.3
    )
    system, params = build(
        recognition=(Recognition("x", ("recognition:embed",)), Recognition("embed", ("obs:img",))),
        modules={"x": module, "embed": embed},
    )
    got, _ = system.infer_states(params, {**OBS, "img": IMG}, priors=PRIORS)
    y, scale = module.diag_moments((embed((IMG,)),))
    ref, _ = _lg_reference(y, scale)
    _close(got["x"], ref["x"])
    _close(got["terrain"], ref["terrain"])


def test_a_categorical_entry_is_a_likelihood_row_on_the_terrain():
    module = MLPPredictor.from_shapes((16,), 3, key=jax.random.PRNGKey(3), width=6, depth=1)
    system, params = build(
        recognition=(Recognition("terrain", ("obs:img",)),), modules={"terrain": module}
    )
    obs = {**OBS, "img": IMG}
    got, detail = system.infer_states(params, obs, priors=PRIORS, detail=True)
    row = jax.nn.softmax(module((IMG,)))
    tilted = Categorical(logits=jnp.log(TERRAIN.probs) + jnp.log(row))
    ref, ref_detail = build(img=False)[0].infer_states(
        build(img=False)[1], OBS, priors={**PRIORS, "terrain": tilted}, detail=True
    )
    _close(got["terrain"], ref["terrain"])
    _close(got["x"], ref["x"])
    base, base_detail = _reference()
    pre = np.asarray(TERRAIN.probs, np.float64) * np.exp(np.asarray(CUE[1], np.float64))
    pre = pre / pre.sum()
    r = np.asarray(row, np.float64)
    log_z = math.log(float(pre @ r))
    q = np.asarray(got["terrain"].probs, np.float64)
    nll = _covered_img_term(got["x"])
    # fix's evidence is read under the terrain belief the sweep holds, which includes the row
    shift = float(base_detail.log_evidence["fix"]) - float(ref_detail.log_evidence["fix"])
    expected = float(base_detail.free_energy) + shift - log_z + float(q @ np.log(r)) + nll
    np.testing.assert_allclose(float(detail.free_energy), expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        float(detail.recognition_log_evidence["recognition:terrain"]), log_z, rtol=1e-4
    )


def test_an_uncovered_neural_likelihood_is_conditioned_on_and_its_evidence_leaves_the_free_energy():
    system, params = build()
    got, detail = system.infer_states(params, {**OBS, "img": IMG}, priors=PRIORS, detail=True)
    ref, ref_detail = _reference()
    assert float(jnp.max(jnp.abs(got["x"].mean() - ref["x"].mean()))) > 1e-3
    ev = float(detail.log_evidence["img"])
    assert np.isfinite(ev)
    np.testing.assert_allclose(
        float(detail.free_energy), float(ref_detail.free_energy) - ev, rtol=1e-5, atol=1e-5
    )


# ----------------------------------------------------------------------------- refusals


def test_a_cycle_among_the_entries_is_refused():
    a = MLPPredictor.from_shapes((4,), 4, key=KEY, width=4, depth=1)
    system, params = build(
        recognition=(Recognition("a", ("recognition:b",)), Recognition("b", ("recognition:a",))),
        modules={"a": a, "b": a.__class__.from_shapes((4,), 4, key=KEY, width=4, depth=1)},
    )
    with pytest.raises(ValueError, match="18.5c"):
        system.infer_states(params, {**OBS, "img": IMG}, priors=PRIORS)


def test_a_module_of_the_wrong_family_is_refused():
    wrong = MLPPredictor.from_shapes((16,), 2, key=KEY, width=4, depth=1)
    system, params = build(recognition=(Recognition("x", ("obs:img",)),), modules={"x": wrong})
    with pytest.raises(ValueError, match=r"does not fit the gaussian factor .*R2"):
        system.infer_states(params, {**OBS, "img": IMG}, priors=PRIORS)
    system, params = build(
        recognition=(Recognition("terrain", ("obs:img",)),), modules={"terrain": REC_MODULE}
    )
    with pytest.raises(ValueError, match=r"does not fit the categorical factor .*R2"):
        system.infer_states(params, {**OBS, "img": IMG}, priors=PRIORS)


def test_a_mistyped_variable_a_control_name_and_a_wrong_parent_width_are_refused():
    aux = MLPPredictor.from_shapes((16,), 2, key=KEY, width=4, depth=1)
    system, params = build(recognition=(Recognition("xx", ("obs:img",)),), modules={"xx": aux})
    with pytest.raises(ValueError, match="no entry reads"):
        system.infer_states(params, {**OBS, "img": IMG}, priors=PRIORS)
    narrow = MLPConditional.from_shapes((4,), 2, key=KEY, width=4, depth=1, frozen=True)
    system, params = build(recognition=(Recognition("x", ("obs:img",)),), modules={"x": narrow})
    with pytest.raises(ValueError, match="parent_dims"):
        system.infer_states(params, {**OBS, "img": IMG}, priors=PRIORS)


def test_a_factor_entry_is_not_a_parent_of_another_entry():
    code = MLPPredictor.from_shapes((16,), 3, key=KEY, width=4, depth=1)
    follow = MLPConditional.from_shapes((3,), 2, key=KEY, width=4, depth=1, frozen=True)
    system, params = build(
        recognition=(
            Recognition("terrain", ("obs:img",)),
            Recognition("x", ("recognition:terrain",)),
        ),
        modules={"terrain": code, "x": follow},
    )
    with pytest.raises(ValueError, match="R2"):
        system.infer_states(params, {**OBS, "img": IMG}, priors=PRIORS)


def test_a_nonzero_rate_on_a_recognition_module_is_refused_and_it_needs_no_learn_state():
    live = dataclasses.replace(REC_MODULE, frozen=False)
    system, params = build(recognition=(Recognition("x", ("obs:img",)),), modules={"x": live})
    assert not any(k.startswith("recognition:") for k in system.init_learn_state(params).opt_states)
    beliefs, _ = system.infer_states(params, {**OBS, "img": IMG}, priors=PRIORS)
    obs = {**OBS, "img": IMG}
    with pytest.raises(ValueError, match=r"recognition:x.*frozen.*R5"):
        system.learn(params, obs, beliefs, lr_by_slot={"recognition:x": 1e-3})
    with pytest.raises(ValueError, match="R5"):
        system.learn(params, obs, beliefs, lr=1e-3)
    out = system.learn(params, obs, beliefs, lr_by_slot={"recognition:x": 0.0})
    assert isinstance(out, SystemParams)
    frozen_system, frozen = _entry_system()
    with pytest.raises(ValueError, match="frozen MLPConditional"):
        frozen_system.learn(frozen, obs, beliefs, lr_by_slot={"recognition:x": 1e-3})


def test_a_concrete_array_zero_rate_passes_and_a_traced_rate_is_refused():
    live = dataclasses.replace(REC_MODULE, frozen=False)
    system, params = build(recognition=(Recognition("x", ("obs:img",)),), modules={"x": live})
    obs = {**OBS, "img": IMG}
    beliefs, _ = system.infer_states(params, obs, priors=PRIORS)
    zero = jnp.asarray(0.0)
    assert isinstance(
        system.learn(params, obs, beliefs, lr_by_slot={"recognition:x": zero}), SystemParams
    )
    assert isinstance(system.learn(params, obs, beliefs, lr=zero), SystemParams)
    with pytest.raises(ValueError, match="R5"):
        system.learn(params, obs, beliefs, lr_by_slot={"recognition:x": jnp.asarray(1e-3)})

    def traced(r):
        return system.learn(params, obs, beliefs, lr_by_slot={"recognition:x": r})

    with pytest.raises(ValueError, match="static zero"):
        jax.jit(traced)(zero)


def test_the_rate_refusal_reads_the_resolved_rates_however_a_key_is_spelled():
    live = dataclasses.replace(REC_MODULE, frozen=False)
    system, params = build(recognition=(Recognition("x", ("obs:img",)),), modules={"x": live})
    obs = {**OBS, "img": IMG}
    beliefs, _ = system.infer_states(params, obs, priors=PRIORS)
    with pytest.raises(ValueError, match=r"recognition:x.*R5"):
        system.learn(params, obs, beliefs, lr_by_slot={"recognition:*": 1.0})
    for zero in ({"*:x": 0.0}, {"recognition:*": 0.0}, {"*:*": 0.0}):
        assert isinstance(system.learn(params, obs, beliefs, lr_by_slot=zero), SystemParams)
    with pytest.raises(ValueError, match="R5"):
        system.learn(params, obs, beliefs, lr=1e-3, lr_by_slot={"likelihood:*": 1e-3})


def test_a_head_on_the_analytic_path_is_refused_by_the_family_not_a_mixed_case():
    from hybrid_aif.distributions import PredictorHead

    system, params = build()
    head = PredictorHead.from_shapes((2,), loss="symlog_mse", out_dim=4, key=KEY, width=4, depth=1)
    params = dataclasses.replace(params, likelihood={**params.likelihood, "img": head})
    with pytest.raises(ValueError, match="neural path"):
        system.infer_states(params, {**OBS, "img": IMG}, priors=PRIORS)


def test_a_replay_reads_the_entry_through_the_same_pass():
    from hybrid_aif.replay import replay

    system, params = _entry_system()
    T = 3
    obs = {
        "cue": jnp.asarray([1, 1, 0]),
        "fix": jnp.tile(OBS["fix"], (T, 1)),
        "img": jnp.tile(IMG, (T, 1)),
    }
    out = replay(system, params, obs, {}, metrics=False)
    assert out.free_energy.shape == (T,)
    assert bool(jnp.all(jnp.isfinite(out.free_energy)))


def test_a_statically_absent_categorical_entry_has_a_zero_normaliser_on_every_branch():
    module = MLPPredictor.from_shapes((16,), 3, key=jax.random.PRNGKey(3), width=6, depth=1)
    system, params = build(
        recognition=(Recognition("terrain", ("obs:img",)),), modules={"terrain": module}
    )
    _, detail = system.infer_states(
        params, {**OBS, "img": None}, priors=PRIORS, present={"img": False}, detail=True
    )
    assert float(detail.recognition_log_evidence["recognition:terrain"]) == 0.0
    # a purely tabular System: one categorical factor, a cue, a categorical entry
    tab = System(
        A_deps=((0,), (0,)),
        B_deps=((0,),),
        controlled=(False,),
        policy_len=1,
        modality_names=("cue", "img"),
        factor_names=("terrain",),
        recognition=(Recognition("terrain", ("obs:img",)),),
    )
    tab_params = SystemParams(
        likelihood={
            "cue": ConditionalCategorical(log_table=CUE, parent_ndims=1),
            "img": IMG_MODULE_TAB,
        },
        transition={"terrain": ConditionalCategorical(log_table=STICKY, parent_ndims=1)},
        initial_state_prior={"terrain": TERRAIN},
        recognition={"terrain": module},
    )
    _, detail = tab.infer_states(
        tab_params,
        {"cue": jnp.asarray(1), "img": None},
        priors={"terrain": TERRAIN},
        present={"img": False},
        detail=True,
    )
    assert float(detail.recognition_log_evidence["recognition:terrain"]) == 0.0


def test_a_categorical_observation_parent_of_an_entry_is_refused_by_name():
    aux = MLPPredictor.from_shapes((3,), 2, key=KEY, width=4, depth=1)
    module = MLPConditional.from_shapes((2,), 2, key=KEY, width=4, depth=1, frozen=True)
    system, params = build(
        recognition=(Recognition("x", ("recognition:z",)), Recognition("z", ("obs:cue",))),
        modules={"x": module, "z": aux},
    )
    with pytest.raises(ValueError, match=r"'obs:cue' is a categorical observation.*R2"):
        system.infer_states(params, {**OBS, "img": IMG}, priors=PRIORS)


def test_a_missing_external_input_is_a_named_error():
    ext = MLPConditional.from_shapes((2,), 2, key=KEY, width=4, depth=1, frozen=True)
    system, params = build(recognition=(Recognition("x", ("external:pose",)),), modules={"x": ext})
    system = dataclasses.replace(system, external_inputs=(("pose", 2),))
    with pytest.raises(ValueError, match=r"external input 'pose'.*no default"):
        system.infer_states(params, {**OBS, "img": IMG}, priors=PRIORS)
    out, _ = system.infer_states(
        params, {**OBS, "img": IMG}, priors=PRIORS, externals={"pose": jnp.asarray([0.1, 0.2])}
    )
    assert out["x"] is not None


def test_a_measurement_of_a_mixture_factor_under_the_runnalls_schedule_is_refused_at_validation():
    system, params = _entry_system()
    system = dataclasses.replace(system, mixture_schedule="runnalls")
    with pytest.raises(ValueError, match=r"mode_aligned.*R3"):
        system.infer_states(params, {**OBS, "img": IMG}, priors=PRIORS)


def test_the_replay_free_energy_matches_infer_states_tick_by_tick():
    from hybrid_aif.replay import replay

    system, params = _entry_system()
    cues = [1, 1, 0]
    obs = {
        "cue": jnp.asarray(cues),
        "fix": jnp.tile(OBS["fix"], (3, 1)),
        "img": jnp.tile(IMG, (3, 1)),
    }
    out = replay(system, params, obs, {}, metrics=False, priors=PRIORS)
    priors = PRIORS
    expected = []
    for t in range(3):
        tick = {"cue": jnp.asarray(cues[t]), "fix": OBS["fix"], "img": IMG}
        beliefs, detail = system.infer_states(params, tick, priors=priors, detail=True)
        expected.append(float(detail.free_energy))
        priors = system.update_empirical_prior(params, beliefs, {})
    np.testing.assert_allclose(np.asarray(out.free_energy), expected, rtol=1e-4, atol=1e-4)


def test_describe_and_build_round_trip_an_analytic_system_with_a_recognition_entry():
    import json

    from hybrid_aif.description import build, describe

    system, params = _entry_system()
    document, arrays = describe(system, params)
    document = json.loads(json.dumps(document))
    assert "recognition" in json.dumps(document)
    system2, params2 = build(document, arrays=arrays)
    assert system2.recognition == system.recognition
    assert jax.tree_util.tree_structure(params2) == jax.tree_util.tree_structure(params)
    for a, b in zip(_leaves(params), _leaves(params2), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    again = json.loads(json.dumps(describe(system2, params2).document))
    assert again == document
    obs = {**OBS, "img": IMG}
    got, _ = system2.infer_states(params2, obs, priors=PRIORS)
    ref, _ = system.infer_states(params, obs, priors=PRIORS)
    for u, v in zip(_leaves(got), _leaves(ref), strict=True):
        np.testing.assert_array_equal(np.asarray(u), np.asarray(v))


def test_an_entry_named_like_a_control_node_is_refused():
    module = MLPPredictor.from_shapes((2,), 2, key=KEY, width=4, depth=1)
    system = System(
        A_deps=((0,),),
        B_deps=((0, 1), ()),
        controls=(1,),
        action_names=("go",),
        factor_names=("terrain", "u_go"),
        modality_names=("cue",),
        policy_len=1,
        recognition=(Recognition("u_go", ("external:e",)),),
        external_inputs=(("e", 2),),
    )
    stay = jnp.log(jnp.asarray([[[0.8, 0.2], [0.2, 0.8]], [[0.2, 0.8], [0.8, 0.2]]]))
    params = SystemParams(
        likelihood={"cue": ConditionalCategorical(log_table=CUE[:2, :2], parent_ndims=1)},
        transition={"terrain": ConditionalCategorical(log_table=stay, parent_ndims=2)},
        initial_state_prior={"terrain": Categorical.from_probs(jnp.asarray([0.5, 0.5]))},
        recognition={"u_go": module},
    )
    with pytest.raises(ValueError, match=r"recognition:u_go.*control node.*R2"):
        system.infer_states(params, {"cue": jnp.asarray(0)}, externals={"e": jnp.zeros(2)})


def test_a_covered_gaussian_emission_is_not_inverted_and_its_expectation_is_recorded():
    offsets = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    noise = 0.5 * jnp.eye(2)
    emission = LinearGaussianObservation(H=jnp.zeros((3, 2, 0)), c=offsets, noise=noise)
    y = jnp.asarray([1.8, 0.1])
    module = MLPPredictor.from_shapes((2,), 3, key=jax.random.PRNGKey(3), width=6, depth=1)

    def with_emission(recognition, modules):
        system, params = build(img=False, recognition=recognition, modules=modules)
        system = dataclasses.replace(
            system, A_deps=(*system.A_deps, (0,)), modality_names=(*system.modality_names, "em")
        )
        return system, dataclasses.replace(params, likelihood={**params.likelihood, "em": emission})

    obs = {**OBS, "em": y}
    system, params = with_emission((Recognition("terrain", ("obs:em",)),), {"terrain": module})
    got, detail = system.infer_states(params, obs, priors=PRIORS, detail=True)
    # the entry's message stands for the emission: the posterior is the one the model without
    # the emission gives from the prior the message tilts
    row = jax.nn.softmax(module((y,)))
    tilted = Categorical(logits=jnp.log(TERRAIN.probs) + jnp.log(row))
    open_system, open_params = with_emission((), {})
    ref, _ = open_system.infer_states(
        open_params, obs, priors={**PRIORS, "terrain": tilted}, present={"em": False}
    )
    _close(got["terrain"], ref["terrain"])
    _close(got["x"], ref["x"])
    assert detail.log_evidence["em"] is None
    q = np.asarray(got["terrain"].probs, np.float64)
    nll = [-_log_normal(y, offsets[j], 0.25 * np.eye(2)) for j in range(3)]
    np.testing.assert_allclose(float(detail.covered_terms["em"]), q @ np.asarray(nll), rtol=1e-4)
    assert set(detail.covered_terms) == {"em"}
