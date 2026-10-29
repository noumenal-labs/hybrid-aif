"""Evidence no state explains (``docs/design/zero-support-evidence.md``).

Three layers, each pinned: the tempered FPI and the per-factor ``unsupported`` flag
at inference (Z1–Z4, and through ``replay``), per-modality support at ``build``
(Z5), and the budgeted exhaustive joint-support diagnostic (Z6).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hybrid_aif as H
from hybrid_aif.description import (
    DescriptionError,
    build,
    check_joint_support,
    check_observation_support,
    describe,
)
from hybrid_aif.inference.discrete import LOG_ZERO, fpi_support
from hybrid_aif.reference.envs.tmaze import CENTRE, CUE
from hybrid_aif.replay import replay
from hybrid_aif.system import System, SystemParams
from tests.test_replay import tmaze_trace
from tests.test_system import tmaze_system

_EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"


def _example(name: str):
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IMPOSSIBLE = {
    "location_obs": jnp.asarray(CENTRE, jnp.int32),
    "reward": jnp.asarray(0, jnp.int32),
    "cue": jnp.asarray(1, jnp.int32),
}
#   location observed exactly at CENTRE while the cue channel reports a cue, which only
#   the CUE location produces: no location explains both


def _hard_zeros(params):
    """The T-maze's accidental 1e-16 floors replaced by true -inf."""
    tables = {
        name: H.ConditionalCategorical(
            jnp.where(a.log_table < -30.0, -jnp.inf, a.log_table), a.parent_ndims
        )
        for name, a in params.likelihood.items()
    }
    return dataclasses.replace(params, likelihood=tables)


# ------------------------------------------------------ Z1/Z2/Z4: the flag, tempered


def test_a_contradiction_is_flagged_and_a_supported_tick_is_not():
    system, params = tmaze_system()
    _, detail = system.infer_states(params, IMPOSSIBLE, None, detail=True)
    location, context = detail.unsupported["location"], detail.unsupported["context"]
    assert bool(location)
    #  mean-field attribution: the cue message reaches `context` as an expectation under
    #  the location belief, and that expectation is uniformly impossible too — so the
    #  flag spills to the co-parent. A flag says "this factor's evidence was uniformly
    #  impossible under current beliefs about its co-parents"; exact attribution of
    #  WHICH factor the contradiction is about is check_joint_support's (Z1 caveat).
    assert bool(context)
    fine = {**IMPOSSIBLE, "cue": jnp.asarray(0, jnp.int32)}
    _, ok = system.infer_states(params, fine, None, detail=True)
    assert not any(bool(f) for f in ok.unsupported.values())
    assert set(ok.unsupported) == set(system.names.states)  # one per factor


def test_true_zeros_give_a_finite_flagged_posterior_not_nan():
    system, params = tmaze_system()
    beliefs, detail = system.infer_states(_hard_zeros(params), IMPOSSIBLE, None, detail=True)
    for b in beliefs.values():
        assert bool(jnp.all(jnp.isfinite(b.probs)))
    assert bool(detail.unsupported["location"])
    #  the arithmetic's own answer: the level with the least impossible evidence wins,
    #  and the floored and true-zero tables agree on it
    floored, _ = system.infer_states(params, IMPOSSIBLE, None)
    assert (
        int(jnp.argmax(beliefs["location"].probs))
        == int(jnp.argmax(floored["location"].probs))
        == CENTRE
    )
    action, q_pi, _ = system.act(jax.random.PRNGKey(0), _hard_zeros(params), beliefs)
    assert bool(jnp.all(jnp.isfinite(q_pi.probs)))


def test_uniform_impossibility_returns_the_prior_flagged():
    """Z3 in its pure form: when every level is equally impossible the evidence adds one
    constant to the row and the posterior *is* the prior — flagged, not replaced."""
    prior = H.Categorical(jnp.log(jnp.asarray([0.2, 0.5, 0.3])))
    table = jnp.log(
        jnp.asarray([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    )  # level 1 impossible everywhere
    (post,), (flag,) = fpi_support(
        (H.ConditionalCategorical(table, 1),), ((0,),), (prior,), (jnp.asarray(1),)
    )
    #  atol 1e-3: the tempering clip costs ~1e-3 nats of resolution among the impossible
    #  levels (why LOG_ZERO is -1e4 and not smaller), so the prior comes back to ~1e-4
    np.testing.assert_allclose(np.asarray(post.probs), [0.2, 0.5, 0.3], atol=1e-3)
    assert bool(flag)
    assert bool(jnp.all(jnp.isfinite(post.logits)))  # tempered, not -inf


def test_the_tempering_clip_is_an_exact_zero_that_keeps_resolution():
    assert float(jnp.exp(jnp.asarray(LOG_ZERO, jnp.float32))) == 0.0
    #  a finite term added to the clip is not lost to the ulp (why it is not -1e30)
    assert float(jnp.asarray(LOG_ZERO, jnp.float32) + 0.5) != float(
        jnp.asarray(LOG_ZERO, jnp.float32)
    )


def test_support_floor_is_overridable_per_call():
    system, params = tmaze_system()
    _, strict = system.infer_states(params, IMPOSSIBLE, None, detail=True, support_floor=-1.0)
    _, lax = system.infer_states(params, IMPOSSIBLE, None, detail=True, support_floor=-1e5)
    assert bool(strict.unsupported["location"]) and not bool(lax.unsupported["location"])


def test_a_masked_out_contradiction_does_not_flag():
    system, params = tmaze_system()
    mask = {"location_obs": True, "reward": True, "cue": False}
    _, detail = system.infer_states(params, IMPOSSIBLE, None, detail=True, present=mask)
    assert not bool(detail.unsupported["location"])


def test_the_hybrid_path_flags_its_categorical_factors_only():
    noise = _example("noise_learning")
    system = noise.make_system()
    params = noise.make_params(jnp.asarray([[[0.1]], [[0.2]]]))
    _, detail = system.infer_states(
        params, {"cell": jnp.asarray(1, jnp.int32), "pose": jnp.asarray([0.3])}, None, detail=True
    )
    assert detail.unsupported["x"] is None  # the continuous factor has no zero support
    assert detail.unsupported["mode"] is not None and not bool(detail.unsupported["mode"])


def test_replay_surfaces_the_flag_per_tick():
    system, params = tmaze_system(policy_len=3)
    observations, actions = tmaze_trace(system, params)
    T = int(actions["a_location"].shape[0])
    broken = {k: o.at[2].set(int(IMPOSSIBLE[k])) for k, o in observations.items()}
    out = replay(system, params, broken, actions)
    location, context = out.unsupported["location"], out.unsupported["context"]
    assert location.shape == (T,) and location.tolist() == [t == 2 for t in range(T)]
    assert context.tolist() == [t == 2 for t in range(T)]  # the mean-field spill, per tick
    assert bool(jnp.all(jnp.isfinite(out.beliefs["location"].probs)))


# ---------------------------------------------------------------- Z4/Z5: at build


def test_minus_inf_is_legal_in_a_document_and_nan_is_not():

    system, params = tmaze_system()
    document, arrays = describe(system, _hard_zeros(params), arrays="inline")
    s2, p2 = build(document)  # -inf round-trips and builds
    assert bool(jnp.any(jnp.isneginf(p2.likelihood["cue"].log_table)))
    document["clamps"]["A_cue"][0][0][0] = float("nan")
    with pytest.raises(DescriptionError, match="contains NaN.*-inf is legal"):
        build(document)


def test_a_dead_observation_level_is_refused_at_build_by_modality_and_level():
    compose = _example("compose_target_factor")
    system, params = compose.system, compose.params
    dead = params.likelihood["answer"].log_table.at[2].set(-jnp.inf)  # level 2 from no state
    broken = dataclasses.replace(
        params, likelihood={**params.likelihood, "answer": H.ConditionalCategorical(dead, 2)}
    )
    with pytest.raises(DescriptionError, match=r"likelihood:answer: observation level\(s\) \[2\]"):
        check_observation_support(system, broken)
    document, _ = describe(system, broken, arrays="inline")
    with pytest.raises(DescriptionError, match=r"produced by no state.*check_joint_support"):
        build(document)
    check_observation_support(system, params)  # the healthy model passes


# --------------------------------------------------------------- Z6: joint support


def _disjunction_model():
    """Two binned features over one target where (near, big) is jointly impossible."""
    system = System(
        B_deps=((0,),),
        controlled=(False,),
        policy_len=1,
        A_deps=((0,), (0,)),
        factor_names=("target",),
        modality_names=("distance", "size"),
    )
    near_only_t0 = jnp.log(jnp.asarray([[1.0, 0.0], [0.0, 1.0]]))  # near <-> target 0
    big_only_t1 = jnp.log(
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    )  # small <-> target 0, big <-> target 1
    params = SystemParams(
        transition={"target": H.ConditionalCategorical(jnp.log(jnp.eye(2))[:, :, None], 2)},
        initial_state_prior={"target": H.Categorical(jnp.zeros(2))},
        likelihood={
            "distance": H.ConditionalCategorical(near_only_t0, 1),
            "size": H.ConditionalCategorical(big_only_t1, 1),
        },
    )
    return system, params


def test_joint_support_finds_the_disjunction_case_per_modality_check_misses():
    system, params = _disjunction_model()
    check_observation_support(system, params)  # every level alone is fine
    bad = check_joint_support(system, params, max_combinations=100)
    assert bad == [{"distance": 0, "size": 1}, {"distance": 1, "size": 0}]
    #  and the flag agrees at inference time
    _, detail = system.infer_states(
        params,
        {"distance": jnp.asarray(0, jnp.int32), "size": jnp.asarray(1, jnp.int32)},
        None,
        detail=True,
    )
    assert bool(detail.unsupported["target"])


def test_joint_support_refuses_above_its_budget_with_the_count():
    system, params = _disjunction_model()
    with pytest.raises(
        ValueError, match="8 \\(observation tuple, state configuration\\) pairs.*max_combinations=4"
    ):
        check_joint_support(system, params, max_combinations=4)


def test_joint_support_on_the_tmaze_reports_the_known_contradictions():
    system, params = tmaze_system()
    bad = check_joint_support(system, params, max_combinations=10_000)
    assert {"location_obs": CENTRE, "reward": 0, "cue": 1} in bad
    assert {"location_obs": CUE, "reward": 0, "cue": 1} not in bad
    assert bad == check_joint_support(system, _hard_zeros(params), max_combinations=10_000)


def test_joint_support_skips_continuous_modalities():
    noise = _example("noise_learning")
    system = noise.make_system()
    params = noise.make_params(jnp.asarray([[[0.1]], [[0.2]]]))
    assert check_joint_support(system, params, max_combinations=1000) == []


# ---------------------------------------------- what the adversarial review found


def test_a_state_that_produces_no_observation_is_tempered_and_refused_at_build():
    """The clip must come BEFORE normalisation: an all--inf column normalises to NaN."""
    prior = H.Categorical(jnp.zeros(3))
    #  state 1 produces nothing; every level is still produced by SOME state
    table = jnp.log(jnp.asarray([[1.0, 0.0, 0.5], [0.0, 0.0, 0.5]]))
    (post,), (flag,) = fpi_support(
        (H.ConditionalCategorical(table, 1),), ((0,),), (prior,), (jnp.asarray(0),)
    )
    assert bool(jnp.all(jnp.isfinite(post.probs))) and not bool(flag)
    np.testing.assert_allclose(np.asarray(post.probs), [2 / 3, 0.0, 1 / 3], atol=1e-6)
    system = System(
        B_deps=((),),
        controlled=(False,),
        policy_len=1,
        A_deps=((0,),),
        factor_names=("s",),
        modality_names=("o",),
    )
    params = SystemParams(
        transition={"s": H.ConditionalCategorical(jnp.log(jnp.eye(2))[:, :, None], 2)},
        initial_state_prior={"s": prior},
        likelihood={"o": H.ConditionalCategorical(table, 1)},
    )
    with pytest.raises(
        DescriptionError, match=r"parent configuration\(s\) \[1\].*produce no observation"
    ):
        check_observation_support(system, params)


def test_many_noisy_modalities_do_not_flag_a_supported_tick():
    """The criterion is the min over messages, not their sum: width cannot cross the floor."""
    n = 30
    system = System(
        B_deps=((),),
        controlled=(False,),
        policy_len=1,
        A_deps=((0,),) * n,
        factor_names=("s",),
        modality_names=tuple(f"o{i}" for i in range(n)),
    )
    noisy = H.ConditionalCategorical(jnp.log(jnp.asarray([[0.4, 0.4], [0.6, 0.6]])), 1)
    params = SystemParams(
        transition={"s": H.ConditionalCategorical(jnp.log(jnp.eye(2))[:, :, None], 2)},
        initial_state_prior={"s": H.Categorical(jnp.zeros(2))},
        likelihood={f"o{i}": noisy for i in range(n)},
    )
    obs = system.from_positional("observations", (jnp.asarray(0, jnp.int32),) * n)
    _, detail = system.infer_states(params, obs, None, detail=True)
    assert not bool(detail.unsupported["s"])
    #  ... while one impossible message among thirty still fails it
    dead = H.ConditionalCategorical(jnp.log(jnp.asarray([[0.0, 0.0], [1.0, 1.0]])), 1)
    broken = dataclasses.replace(params, likelihood={**params.likelihood, f"o{n - 1}": dead})
    _, detail = system.infer_states(broken, obs, None, detail=True)
    assert bool(detail.unsupported["s"])


def test_the_three_layers_agree_on_a_dirichlet_modality():
    """A tiny pseudo-count is a dead level under the FPI's lowering; build says so too."""
    compose = _example("compose_target_factor")
    system, params = compose.system, compose.params
    conc = params.likelihood["distance"].concentration.at[2].set(1e-13)  # 'far' from no target
    broken = dataclasses.replace(
        params, likelihood={**params.likelihood, "distance": H.ConditionalDirichlet(conc, 1)}
    )
    with pytest.raises(
        DescriptionError, match=r"likelihood:distance: observation level\(s\) \[2\]"
    ):
        check_observation_support(system, broken)
    assert {"distance": 2, "size": 0, "answer": 0} in check_joint_support(
        system, broken, max_combinations=10_000
    )


def test_padding_ticks_are_never_unsupported():
    from hybrid_aif.replay import stack_traces

    system, params = tmaze_system(policy_len=3)
    observations, actions = tmaze_trace(system, params)
    broken = {k: o.at[2].set(int(IMPOSSIBLE[k])) for k, o in observations.items()}
    short = (
        {k: o[:3] for k, o in broken.items()},
        {k: a[:3] for k, a in actions.items()},
        None,
    )
    obs_b, act_b, mask_b, valid_b = stack_traces([(broken, actions, None), short])
    #  the second trace is 3 ticks long and zero-padded to 6; padding rows must not flag
    second = lambda tree: {k: v[1] for k, v in tree.items()}  # noqa: E731
    out = replay(
        system, params, second(obs_b), second(act_b), present=second(mask_b), valid=valid_b[1]
    )
    assert out.unsupported["location"].tolist() == [False, False, True, False, False, False]


def test_replay_document_forwards_the_floor_to_build():
    from hybrid_aif.replay import replay_document

    compose = _example("compose_target_factor")
    system, params = compose.system, compose.params
    dead = params.likelihood["answer"].log_table.at[2].set(-jnp.inf)
    broken = dataclasses.replace(
        params, likelihood={**params.likelihood, "answer": H.ConditionalCategorical(dead, 2)}
    )
    document, _ = describe(system, broken, arrays="inline")
    T = 3
    observations = {n: jnp.zeros(T, jnp.int32) for n in system.modality_names}
    actions = {"a_mode": jnp.zeros(T, jnp.int32)}
    with pytest.raises(DescriptionError, match="produced by no state"):
        replay_document(document, observations, actions)
    out = replay_document(document, observations, actions, support_floor=-2e4)  # below the clip
    assert out.unsupported["target"].shape == (T,)
