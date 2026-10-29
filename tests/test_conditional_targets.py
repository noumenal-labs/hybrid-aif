"""Per-step targets conditional on a control node or a state factor: ``System(C_deps=...)``.

A ``state_targets`` or ``obs_targets`` entry may be a ``ConditionalCategorical`` or a
``ConditionalGaussian`` whose parents ``C_deps`` names by the targeted node. At each step the
target is resolved under the parents, a clamped control node's level selecting its own
distribution and a state parent averaging mean-field, then scored as an unconditional one: the
KL in a flat planner, the cross-entropy in a branching one, the closed-loop cross-entropy under
``feedback="linear_quadratic"`` with the level's precision in the Riccati pass. These tests pin
that against the unconditional target of the chosen level on both carriers, the mean-field
resolution against the terminal resolver, the refusals, the document round trip, ``merge``'s
namespacing, the dwell equivalence of a slow step against the fast tier it stands for, and the
two-row payoff toy of the slow model.

Design: docs/design/step-durations.md T4–T6, acceptance 1–3.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import (
    Box,
    Categorical,
    ConditionalCategorical,
    ConditionalGaussian,
    Gaussian,
    LinearGaussianObservation,
    LinearGaussianTransition,
    System,
    SystemParams,
)
from hybrid_aif.composition.merge import ActionControl, BoundaryPrior, merge, subsystem
from hybrid_aif.controls import ControlStep
from hybrid_aif.description import DescriptionError, build, describe
from hybrid_aif.factors import GridNoiseMap
from hybrid_aif.objectives import total
from hybrid_aif.objectives.discrete import (
    contract_probs,
    resolve_conditional_targets,
    resolve_step_targets,
)
from hybrid_aif.planning.enumerate import evaluate_policies
from tests.test_system import _initial_beliefs, tmaze_system


def _leaf(tree, term, key) -> np.ndarray:
    iv = tree[term][key]
    np.testing.assert_allclose(np.asarray(iv.lower), np.asarray(iv.upper))
    return np.asarray(iv.lower)


def _replace_system(system: System, **overrides) -> System:
    """The same System, declared otherwise (the sugar re-read as node form)."""
    kw = dict(
        A_deps=system.A_deps,
        B_deps=system.B_deps,
        controls=system.controls,
        action_names=system.action_names,
        policy_len=system.policy_len,
        gamma=system.gamma,
        epistemic=system.epistemic,
        modality_names=system.modality_names,
        factor_names=system.factor_names,
        decomposition=system.decomposition,
    )
    return System(**{**kw, **overrides})


# ------------------------------------------------------------------- the tabular carrier

#: C(reward | u_location): each level of the control node prefers its own outcome pattern.
REWARD_BY_LEVEL = jnp.asarray(
    [[0.0, 1.0, 2.0, -1.0, 0.5], [4.0, -2.0, 0.0, 3.0, 1.0], [-6.0, 1.0, -3.0, 0.0, -1.0]]
)
#: C(location | u_location), (5 levels, 5 levels).
LOCATION_BY_LEVEL = jnp.log(0.7 * jnp.eye(5) + 0.1)


def _tmaze_conditional(policy_len=2, **overrides):
    """The T-maze with its reward preference conditional on the move taken."""
    system, params = tmaze_system(policy_len=policy_len, epistemic=False)
    system = _replace_system(system, C_deps={"reward": ("u_location",)}, **overrides)
    params = dataclasses.replace(
        params,
        obs_targets={"reward": ConditionalCategorical(log_table=REWARD_BY_LEVEL, parent_ndims=1)},
    )
    return system, params


def _tmaze_level(level: int, policy_len=2, **overrides):
    """The same T-maze with the unconditional preference of one level."""
    system, params = tmaze_system(policy_len=policy_len, epistemic=False)
    system = _replace_system(system, **overrides)
    params = dataclasses.replace(
        params, obs_targets={"reward": Categorical(logits=REWARD_BY_LEVEL[:, level])}
    )
    return system, params


def _held(level: int, policy_len: int) -> jnp.ndarray:
    return jnp.full((policy_len, 1), level, jnp.int32)


@pytest.mark.parametrize("level", [0, 1, 2, 3, 4])
def test_a_categorical_obs_target_with_a_control_parent_scores_the_levels_column(level):
    """A plan that holds one level is scored, leaf for leaf, as under the unconditional target
    equal to that level's column."""
    system, params = _tmaze_conditional()
    beliefs = _initial_beliefs(system, params)
    plan = _held(level, system.policy_len)
    got = system.score_policies(params, beliefs, plan)
    ref_s, ref_p = _tmaze_level(level)
    want = ref_s.score_policies(ref_p, beliefs, plan)
    assert set(got) == set(want)
    np.testing.assert_allclose(_leaf(got, "utility", "reward"), _leaf(want, "utility", "reward"))
    np.testing.assert_allclose(float(total(got).lower), float(total(want).lower), rtol=1e-6)


def test_a_plan_that_changes_level_reads_each_steps_own_column():
    """Per step, the utility is ``Σ_o q(o) log C(o | u_t)`` with the step's level."""
    system, params = _tmaze_conditional(policy_len=3)
    beliefs = _initial_beliefs(system, params)
    plan = jnp.asarray([[2], [0], [3]], jnp.int32)
    got = _leaf(system.score_policies(params, beliefs, plan), "utility", "reward")
    steps = system.predict_beliefs(params, beliefs, plan)
    A = jax.nn.softmax(params.likelihood["reward"].log_table, axis=0)
    logs = jax.nn.log_softmax(REWARD_BY_LEVEL, axis=0)
    want = 0.0
    for t, q in enumerate(steps):
        q_obs = contract_probs(A, (q["location"].probs, q["context"].probs), keep_child=True)
        want += float(jnp.sum(q_obs * logs[:, int(plan[t, 0])]))
    np.testing.assert_allclose(got, want, rtol=1e-5)


@pytest.mark.parametrize("decomposition", ["utility_infogain", "risk_ambiguity"])
@pytest.mark.parametrize("level", [0, 3])
def test_a_categorical_state_target_with_a_control_parent_scores_the_levels_column_as_a_kl(
    level, decomposition
):
    system, params = tmaze_system(policy_len=2, epistemic=False)
    beliefs = _initial_beliefs(system, params)
    plan = _held(level, 2)
    cond = _replace_system(
        system, C_deps={"location": ("u_location",)}, decomposition=decomposition
    )
    cond_p = dataclasses.replace(
        params,
        obs_targets={},
        state_targets={
            "location": ConditionalCategorical(log_table=LOCATION_BY_LEVEL, parent_ndims=1)
        },
    )
    flat = _replace_system(system, decomposition=decomposition)
    flat_p = dataclasses.replace(
        params,
        obs_targets={},
        state_targets={"location": Categorical(logits=LOCATION_BY_LEVEL[:, level])},
    )
    got = cond.score_policies(cond_p, beliefs, plan)
    want = flat.score_policies(flat_p, beliefs, plan)
    np.testing.assert_allclose(_leaf(got, "risk", "location"), _leaf(want, "risk", "location"))


def _branching_scorer(monkeypatch, plan):
    """A registered branching planner that scores ``plan`` through the branching closures (a
    state target's cross-entropy) and returns its term tree."""
    from hybrid_aif.planning import base, register_planner

    name = "scores_one_plan_in_a_branching_search"
    monkeypatch.setitem(base._PLANNERS, name, None)

    @register_planner(name, branching=True)
    def scorer(ctx, use_bound="midpoint", metrics=True):
        tree = evaluate_policies(
            plan[None],
            ctx.init_beliefs,
            ctx.transition_fn,
            ctx.step_terms_fn,
            ctx.terminal_terms_fn,
        )
        return Categorical(logits=-ctx.gamma * total(tree).midpoint), tree, plan[None]

    return name


@pytest.mark.parametrize("level", [0, 2])
def test_in_a_branching_search_the_levels_column_is_scored_by_its_cross_entropy(level, monkeypatch):
    """In a branching planner (``branch_on`` set) a state target is its cross-entropy; a plan
    holding one level scores as the unconditional target of that level's column does there."""
    system, params = tmaze_system(policy_len=2, epistemic=False)
    beliefs = _initial_beliefs(system, params)
    plan = _held(level, 2)
    kw = dict(planner=_branching_scorer(monkeypatch, plan), branch_on=("cue",))
    cond = _replace_system(system, C_deps={"location": ("u_location",)}, **kw)
    cond_p = dataclasses.replace(
        params,
        obs_targets={},
        state_targets={
            "location": ConditionalCategorical(log_table=LOCATION_BY_LEVEL, parent_ndims=1)
        },
    )
    flat = _replace_system(system, **kw)
    flat_p = dataclasses.replace(
        params,
        obs_targets={},
        state_targets={"location": Categorical(logits=LOCATION_BY_LEVEL[:, level])},
    )
    _, got = cond.infer_policies(cond_p, beliefs)
    _, want = flat.infer_policies(flat_p, beliefs)
    got_leaf, want_leaf = _leaf(got, "risk", "location"), _leaf(want, "risk", "location")
    np.testing.assert_allclose(got_leaf, want_leaf, rtol=1e-6)
    # the cross-entropy, not the KL: the belief's entropy is not subtracted
    ce = 0.0
    for q in system.predict_beliefs(params, beliefs, plan):
        probs = q["location"].probs
        ce -= float(jnp.sum(probs * jax.nn.log_softmax(LOCATION_BY_LEVEL[:, level])))
    np.testing.assert_allclose(got_leaf, ce, rtol=1e-5)


def test_a_categorical_target_with_a_state_parent_averages_mean_field():
    """``Σ_context C(reward | context) q_t(context)`` at each step, the terminal resolver's rule
    on the step's beliefs."""
    system, params = tmaze_system(policy_len=2, epistemic=False)
    beliefs = _initial_beliefs(system, params)
    table = jnp.asarray([[0.0, 1.0], [4.0, -3.0], [-6.0, 2.0]])
    cond = _replace_system(system, C_deps={"reward": ("context",)})
    cond_p = dataclasses.replace(
        params, obs_targets={"reward": ConditionalCategorical(log_table=table, parent_ndims=1)}
    )
    plan = jnp.asarray([[2], [1]], jnp.int32)
    got = _leaf(cond.score_policies(cond_p, beliefs, plan), "utility", "reward")
    A = jax.nn.softmax(params.likelihood["reward"].log_table, axis=0)
    want = 0.0
    for q in cond.predict_beliefs(cond_p, beliefs, plan):
        resolved = resolve_conditional_targets(
            {"reward": cond_p.obs_targets["reward"]},
            ((1,),),
            (q["location"].probs, q["context"].probs),
            ("reward",),
        )["reward"]
        q_obs = contract_probs(A, (q["location"].probs, q["context"].probs), keep_child=True)
        want += float(jnp.sum(q_obs * resolved.log_probs))
    np.testing.assert_allclose(got, want, rtol=1e-5)


def test_a_soft_control_entry_gives_the_probability_weighted_target():
    """A belief-valued step (a ``ControlStep`` with a probability vector) resolves to the
    mixture ``Σ_u q(u) C(· | u)``; a Gaussian target under one raises."""
    probs = jnp.asarray([0.1, 0.2, 0.3, 0.25, 0.15])
    step = ControlStep(entries=(probs,))
    target = ConditionalCategorical(log_table=REWARD_BY_LEVEL, parent_ndims=1)
    got = resolve_step_targets({"reward": target}, {"reward": (("control", 0),)}, step, ())
    want = jax.nn.softmax(REWARD_BY_LEVEL, axis=0) @ probs
    np.testing.assert_allclose(np.asarray(got["reward"].probs), np.asarray(want), rtol=1e-6)
    clamped = resolve_step_targets(
        {"reward": target}, {"reward": (("control", 0),)}, jnp.asarray([2], jnp.int32), ()
    )
    np.testing.assert_allclose(
        np.asarray(clamped["reward"].log_probs),
        np.asarray(jax.nn.log_softmax(REWARD_BY_LEVEL[:, 2])),
        rtol=1e-6,
    )
    gaussian = ConditionalGaussian(loc=jnp.zeros((5, 1)), chol=jnp.ones((5, 1, 1)), parent_ndims=1)
    with pytest.raises(ValueError, match="soft control step is a mixture"):
        resolve_step_targets({"x": gaussian}, {"x": (("control", 0),)}, step, ())


# -------------------------------------------------------------------- the hybrid carrier

LEVEL_SD = (0.2, 1.0)  # the lane preference's sd under each level of ``u``
LEVEL_LOC = (0.3, -0.2)


def _lg(policy_len=2, feedback=None, conditional=True, level=0, on_obs=False, **overrides):
    """One Gaussian factor ``x`` whose transition reads the two-level node ``u`` as a mode and,
    under ``feedback``, the continuous node ``v`` through ``G v``; a reading ``y`` every step;
    a state target on ``x`` (on the reading ``y`` under ``on_obs``) conditional on ``u``, or
    the unconditional one of ``level``."""
    with_v = feedback is not None
    system = System(
        A_deps=((0,),),
        B_deps=((1, 2), (), ()) if with_v else ((1,), ()),
        controls=(1, 2) if with_v else (1,),
        action_names=("a_u", "a_v") if with_v else ("a_u",),
        factor_names=("x", "u", "v") if with_v else ("x", "u"),
        modality_names=("y",),
        policy_len=policy_len,
        planning_belief="posterior" if with_v else "prior",
        feedback=feedback,
        epistemic=False,
        C_deps={("y" if on_obs else "x"): ("u",)} if conditional else (),
        **overrides,
    )
    if conditional:
        target = ConditionalGaussian(
            loc=jnp.asarray(LEVEL_LOC)[:, None],
            chol=jnp.asarray(LEVEL_SD)[:, None, None],
            parent_ndims=1,
        )
    else:
        target = Gaussian(
            loc=jnp.asarray([LEVEL_LOC[level]]), chol=jnp.asarray([[LEVEL_SD[level]]])
        )
    params = SystemParams(
        transition={
            "x": LinearGaussianTransition(
                F=jnp.asarray([[0.9]]),
                G=jnp.asarray([[0.5]]) if with_v else None,
                innovations=ConditionalGaussian(
                    loc=jnp.asarray([[0.1], [-0.1]]),
                    chol=jnp.asarray([[[0.1]], [[0.3]]]),
                    parent_ndims=1,
                ),
            )
        },
        initial_state_prior={"x": Gaussian(loc=jnp.asarray([0.5]), chol=jnp.asarray([[0.4]]))},
        likelihood={
            "y": LinearGaussianObservation(
                H=jnp.asarray([[1.0]]), c=jnp.zeros(1), noise=0.2 * jnp.eye(1)
            )
        },
        state_targets={} if on_obs else {"x": target},
        obs_targets={"y": target} if on_obs else {},
        control={"v": Box(-3.0 * jnp.ones(1), 3.0 * jnp.ones(1))} if with_v else {},
    )
    return system, params


@pytest.mark.parametrize("level", [0, 1])
def test_a_gaussian_state_target_with_a_control_parent_selects_the_levels_loc_and_chol(level):
    system, params = _lg()
    flat, flat_p = _lg(conditional=False, level=level)
    beliefs = system.initial_priors(params)
    plan = _held(level, 2)
    got = system.score_policies(params, beliefs, plan)
    want = flat.score_policies(flat_p, beliefs, plan)
    np.testing.assert_allclose(_leaf(got, "risk", "x"), _leaf(want, "risk", "x"), rtol=1e-6)
    other = flat.score_policies(_lg(conditional=False, level=1 - level)[1], beliefs, plan)
    assert not np.allclose(_leaf(got, "risk", "x"), _leaf(other, "risk", "x"))


def test_a_plan_that_changes_level_scores_each_step_against_its_own_gaussian():
    system, params = _lg(policy_len=2)
    beliefs = system.initial_priors(params)
    plan = jnp.asarray([[1], [0]], jnp.int32)
    got = _leaf(system.score_policies(params, beliefs, plan), "risk", "x")
    want = 0.0
    for t, q in enumerate(system.predict_beliefs(params, beliefs, plan)):
        k = int(plan[t, 0])
        target = Gaussian(loc=jnp.asarray([LEVEL_LOC[k]]), chol=jnp.asarray([[LEVEL_SD[k]]]))
        want += float(q["x"].kl_to(target).lower)
    np.testing.assert_allclose(got, want, rtol=1e-5)


@pytest.mark.parametrize("level", [0, 1])
def test_the_closed_loop_riccati_pass_takes_the_levels_precision(level):
    """Under ``feedback="linear_quadratic"`` a plan holding one level scores, leaf for leaf, as
    the unconditional closed-loop run whose target is that level's: the pass weighs each step
    by the level's ``Λ``."""
    system, params = _lg(policy_len=3, feedback="linear_quadratic")
    flat, flat_p = _lg(policy_len=3, feedback="linear_quadratic", conditional=False, level=level)
    beliefs = system.initial_priors(params)
    plan = {"u": _held(level, 3)[:, 0], "v": jnp.asarray([[0.2], [-0.1], [0.3]])}
    got = system.score_policies(params, beliefs, plan)
    want = flat.score_policies(flat_p, beliefs, plan)
    np.testing.assert_allclose(_leaf(got, "risk", "x"), _leaf(want, "risk", "x"), rtol=1e-5)
    open_loop = _lg(policy_len=3, feedback="linear_quadratic", conditional=False, level=level)
    other_level = _lg(policy_len=3, feedback="linear_quadratic", conditional=False, level=1 - level)
    other = open_loop[0].score_policies(other_level[1], beliefs, plan)
    assert not np.allclose(_leaf(got, "risk", "x"), _leaf(other, "risk", "x"))


@pytest.mark.parametrize("feedback", [None, "linear_quadratic"])
@pytest.mark.parametrize("level", [0, 1])
def test_a_gaussian_obs_target_with_a_control_parent_selects_the_level_open_and_closed_loop(
    level, feedback
):
    """A linear-Gaussian reading's conditional preference: the level's density open loop, and
    its precision through ``H`` in the closed-loop pass."""
    system, params = _lg(policy_len=3, feedback=feedback, on_obs=True)
    flat, flat_p = _lg(policy_len=3, feedback=feedback, conditional=False, level=level, on_obs=True)
    beliefs = system.initial_priors(params)
    held = _held(level, 3)
    plan = {"u": held[:, 0], "v": jnp.asarray([[0.2], [-0.1], [0.3]])} if feedback else held
    got = system.score_policies(params, beliefs, plan)
    want = flat.score_policies(flat_p, beliefs, plan)
    np.testing.assert_allclose(_leaf(got, "utility", "y"), _leaf(want, "utility", "y"), rtol=1e-5)
    other_p = _lg(policy_len=3, feedback=feedback, conditional=False, level=1 - level, on_obs=True)
    other = flat.score_policies(other_p[1], beliefs, plan)
    assert not np.allclose(_leaf(got, "utility", "y"), _leaf(other, "utility", "y"))


# ------------------------------------------------------------------------- the refusals


def test_a_gaussian_target_with_a_state_parent_raises_by_name():
    system = System(
        A_deps=((0,),),
        B_deps=((2,), (1,), ()),
        controls=(2,),
        action_names=("a",),
        factor_names=("x", "z", "u"),
        modality_names=("y",),
        policy_len=1,
        epistemic=False,
        C_deps={"x": ("z",)},
    )
    params = SystemParams(
        transition={
            "x": LinearGaussianTransition(
                F=jnp.asarray([[1.0]]),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((2, 1)), chol=0.1 * jnp.ones((2, 1, 1)), parent_ndims=1
                ),
            ),
            "z": ConditionalCategorical(log_table=jnp.log(jnp.eye(2)[..., None]), parent_ndims=2),
        },
        initial_state_prior={
            "x": Gaussian(loc=jnp.zeros(1), chol=jnp.ones((1, 1))),
            "z": Categorical(logits=jnp.zeros(2)),
        },
        likelihood={
            "y": LinearGaussianObservation(H=jnp.ones((1, 1)), c=jnp.zeros(1), noise=jnp.eye(1))
        },
        state_targets={
            "x": ConditionalGaussian(
                loc=jnp.zeros((2, 1)), chol=jnp.ones((2, 1, 1)), parent_ndims=1
            )
        },
    )
    with pytest.raises(NotImplementedError, match=r"conditional on the state factor\(s\) \['z'\]"):
        system.score_policies(params, system.initial_priors(params), jnp.zeros((1, 1), jnp.int32))


def test_a_conditional_family_without_c_deps_and_an_unconditional_one_with_it_are_refused():
    system, params = tmaze_system(policy_len=1, epistemic=False)
    beliefs = _initial_beliefs(system, params)
    plan = jnp.zeros((1, 1), jnp.int32)
    without = dataclasses.replace(
        params, obs_targets={"reward": ConditionalCategorical(REWARD_BY_LEVEL, parent_ndims=1)}
    )
    with pytest.raises(ValueError, match=r"obs_targets\['reward'\] is a ConditionalCategorical"):
        system.score_policies(without, beliefs, plan)
    cond = _replace_system(system, C_deps={"reward": ("u_location",)})
    with pytest.raises(ValueError, match=r"the target is a Categorical; a conditional target"):
        cond.score_policies(params, beliefs, plan)
    flat_gaussian, flat_p = _lg(conditional=False)
    with pytest.raises(ValueError, match=r"state_targets\['x'\] is a ConditionalGaussian"):
        flat_gaussian.score_policies(
            dataclasses.replace(flat_p, state_targets=_lg()[1].state_targets),
            flat_gaussian.initial_priors(flat_p),
            jnp.zeros((2, 1), jnp.int32),
        )


def test_the_declaration_is_checked_by_name_at_construction():
    system, params = tmaze_system(policy_len=1)
    for deps, sentence in (
        ({"nowhere": ("u_location",)}, "neither a state factor nor a modality"),
        ({"reward": ("cue",)}, "names the modality 'cue' as a parent"),
        ({"location": ("location",)}, "names 'location' itself"),
        ({"u_location": ("context",)}, "names the control node 'u_location'"),
        ({"reward": ()}, "names no parent"),
        ({"reward": ("u_location", "u_location")}, "twice"),
    ):
        with pytest.raises(ValueError, match=sentence):
            _replace_system(system, C_deps=deps)
    assert _replace_system(system, C_deps={"reward": ("u_location",)}).C_deps == (
        ("reward", ("u_location",)),
    )


def test_a_continuous_control_node_cannot_parent_a_target():
    system, params = _lg(feedback="linear_quadratic", conditional=False)
    system = dataclasses.replace(system, C_deps=(("x", ("v",)),))
    with pytest.raises(ValueError, match="a continuous control node"):
        system.score_policies(
            params,
            system.initial_priors(params),
            {"u": jnp.zeros((2,), jnp.int32), "v": jnp.zeros((2, 1))},
        )


def test_the_shape_of_a_conditional_target_is_checked_against_its_parents():
    system, params = _tmaze_conditional()
    beliefs = _initial_beliefs(system, params)
    wrong = dataclasses.replace(
        params,
        obs_targets={"reward": ConditionalCategorical(REWARD_BY_LEVEL[:, :3], parent_ndims=1)},
    )
    with pytest.raises(ValueError, match=r"parent axes \(3,\) but its C_deps parent"):
        system.score_policies(wrong, beliefs, _held(0, 2))


# ------------------------------------------------------------ the document and merge


def test_a_categorical_conditional_target_round_trips_through_the_document():
    system, params = _tmaze_conditional()
    document, arrays = describe(system, params, arrays="inline")
    record = document["factors"]["reward"]["target"]
    assert record["family"] == "ConditionalCategorical"
    assert record["parents"] == ["u_location"]
    built, built_params = build(document)
    assert built.C_deps == (("reward", ("u_location",)),)
    assert isinstance(built_params.obs_targets["reward"], ConditionalCategorical)
    again, _ = describe(built, built_params, arrays="inline")
    assert again == document
    beliefs = _initial_beliefs(system, params)
    plan = jnp.asarray([[2], [0]], jnp.int32)
    np.testing.assert_allclose(
        _leaf(built.score_policies(built_params, beliefs, plan), "utility", "reward"),
        _leaf(system.score_policies(params, beliefs, plan), "utility", "reward"),
    )


def test_a_gaussian_conditional_target_round_trips_through_the_document():
    system, params = _lg()
    document, arrays = describe(system, params, arrays="inline")
    record = document["factors"]["x"]["target"]
    assert record["family"] == "ConditionalGaussian"
    assert record["parents"] == ["u"]
    assert set(record["parameters"]) == {"loc", "chol"}
    built, built_params = build(document)
    assert built.C_deps == (("x", ("u",)),)
    target = built_params.state_targets["x"]
    assert isinstance(target, ConditionalGaussian) and target.parent_shape == (2,)
    again, _ = describe(built, built_params, arrays="inline")
    assert again == document


def test_a_document_target_with_parents_and_the_wrong_family_is_refused():
    system, params = _lg()
    document, _ = describe(system, params, arrays="inline")
    document["factors"]["x"]["target"]["family"] = "Gaussian"
    with pytest.raises(DescriptionError, match="conditional preference.*ConditionalGaussian"):
        build(document)
    document, _ = describe(system, params, arrays="inline")
    del document["factors"]["x"]["target"]["parents"]
    with pytest.raises(DescriptionError, match="the record names none"):
        build(document)


def test_merge_namespaces_c_deps_keys_and_parents():
    low_s, low_p = _tmaze_conditional()
    high_s, high_p = _lg()
    composite, composite_params = merge(
        {"high": high_s, "low": low_s}, {"high": high_p, "low": low_p}, []
    )
    assert composite.C_deps == (("high.x", ("high.u",)), ("low.reward", ("low.u_location",)))
    assert isinstance(composite_params.obs_targets["low.reward"], ConditionalCategorical)
    assert isinstance(composite_params.state_targets["high.x"], ConditionalGaussian)
    view, view_params = subsystem(composite, composite_params, "low")
    assert view.C_deps == (("low.reward", ("low.u_location",)),)
    assert isinstance(view_params.obs_targets["low.reward"], ConditionalCategorical)


def _relay_pair():
    a, ap = _tmaze_conditional(policy_len=3)
    b, bp = tmaze_system(policy_len=3, epistemic=False)
    return {"a": a, "b": b}, {"a": ap, "b": bp}


def test_merge_refuses_a_conditional_targets_control_parent_relayed_to_a_state():
    systems, params = _relay_pair()
    with pytest.raises(ValueError, match=r"relay.*reads the state a step later"):
        merge(systems, params, [ActionControl("b.location", "a.u_location")])


def test_a_conditional_targets_control_parent_identified_with_a_driver_follows_it():
    systems, params = _relay_pair()
    composite, composite_params = merge(
        systems, params, [ActionControl("b.u_location", "a.u_location")]
    )
    assert composite.C_deps == (("a.reward", ("b.u_location",)),)
    doc, _ = describe(composite, composite_params, arrays="inline")
    assert doc["factors"]["a.reward"]["target"]["parents"] == ["b.u_location"]
    built, built_params = build(doc)
    assert built.C_deps == composite.C_deps
    beliefs = composite.initial_priors(composite_params)
    plan = jnp.asarray([[2], [4], [1]], jnp.int32)
    got = composite.score_policies(composite_params, beliefs, plan)
    again = built.score_policies(built_params, beliefs, plan)
    leaf = _leaf(got, "utility", "a.reward")
    np.testing.assert_allclose(_leaf(again, "utility", "a.reward"), leaf, rtol=1e-6)
    # the hand computation: each step reads its own belief of a.location and a.context through
    # the likelihood, scored against the column of the driver's level at that step
    seq = composite.predict_beliefs(composite_params, beliefs, plan)
    A = jax.nn.softmax(composite_params.likelihood["a.reward"].log_table, axis=0)
    table = jax.nn.softmax(REWARD_BY_LEVEL, axis=0)
    want = 0.0
    for t, q in enumerate(seq):
        q_obs = contract_probs(A, (q["a.location"].probs, q["a.context"].probs), keep_child=True)
        column = jax.nn.log_softmax(jnp.log(table[:, int(plan[t, 0])]))
        want += float(jnp.sum(q_obs * column))
    np.testing.assert_allclose(leaf, want, rtol=1e-5)


def test_merge_refuses_a_conditional_targets_parent_terminated_by_a_boundary_prior():
    systems, params = _relay_pair()
    with pytest.raises(ValueError, match=r"conditional on the control node.*BoundaryPrior"):
        merge(systems, params, [BoundaryPrior("a.u_location", 1)])


# ------------------------------------------------------- acceptance 1: dwell equivalence

N_TICKS = 5
LAM = 4.0  # the fast tier's per-tick precision on the static state
RHO = 2.0  # the per-tick reading's precision
SIGMA0 = 0.6  # the static state's prior sd


def _dwell(n_steps: int, target_precision: float, reading_precision: float, **overrides):
    """A static scalar state (a process noise of 1e-4, below the comparisons' tolerance, since
    a belief's covariance must stay positive definite) read once per step, one one-level control
    node (the plan)."""
    conditional = overrides.pop("conditional", False)
    system = System(
        A_deps=((0,),),
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("a",),
        factor_names=("x", "u"),
        modality_names=("y",),
        policy_len=n_steps,
        planning_belief="posterior",
        epistemic=True,
        C_deps={"x": ("u",)} if conditional else (),
        **overrides,
    )
    sd = target_precision**-0.5
    target = (
        ConditionalGaussian(loc=jnp.zeros((1, 1)), chol=jnp.full((1, 1, 1), sd), parent_ndims=1)
        if conditional
        else Gaussian(loc=jnp.zeros(1), chol=jnp.full((1, 1), sd))
    )
    params = SystemParams(
        transition={
            "x": LinearGaussianTransition(
                F=jnp.eye(1),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 1)), chol=jnp.full((1, 1, 1), 1e-4), parent_ndims=1
                ),
            )
        },
        initial_state_prior={"x": Gaussian(loc=jnp.asarray([0.4]), chol=jnp.full((1, 1), SIGMA0))},
        likelihood={
            "y": LinearGaussianObservation(
                H=jnp.eye(1), c=jnp.zeros(1), noise=jnp.full((1, 1), reading_precision**-0.5)
            )
        },
        state_targets={"x": target},
    )
    return system, params


def _score(system, params):
    beliefs = system.initial_priors(params)
    return system.score_policies(params, beliefs, system.policies(params))


def test_a_slow_step_scores_the_information_of_the_fast_steps_it_stands_for():
    """``n`` readings of a static state pool: the slow System's one reading at the pooled
    precision gains exactly what the fast tier's ``n`` readings do, and its conditional target
    of precision ``n·Λ`` carries the fast tier's quadratic cost, the normalisers apart."""
    fast = _score(*_dwell(N_TICKS, LAM, RHO))
    slow = _score(*_dwell(1, N_TICKS * LAM, N_TICKS * RHO, conditional=True))
    gain_fast, gain_slow = _leaf(fast, "infogain", "y"), _leaf(slow, "infogain", "y")
    pooled = 0.5 * np.log(1.0 + N_TICKS * SIGMA0**2 * RHO)
    np.testing.assert_allclose(gain_fast, pooled, rtol=1e-5)
    np.testing.assert_allclose(gain_slow, pooled, rtol=1e-5)
    # the risk leaves agree up to the normaliser difference: one density of precision nΛ
    # against n densities of precision Λ, each KL carrying −½(d + log det Λ + log det Σ)
    risk_fast, risk_slow = _leaf(fast, "risk", "x"), _leaf(slow, "risk", "x")
    sigma2 = SIGMA0**2
    constants = N_TICKS * 0.5 * (1.0 + np.log(LAM) + np.log(sigma2)) - 0.5 * (
        1.0 + np.log(N_TICKS * LAM) + np.log(sigma2)
    )
    np.testing.assert_allclose(risk_slow - risk_fast, constants, rtol=1e-5)
    quad = 0.5 * N_TICKS * LAM * (sigma2 + 0.4**2)
    np.testing.assert_allclose(
        risk_fast + N_TICKS * 0.5 * (1.0 + np.log(LAM) + np.log(sigma2)), quad, rtol=1e-5
    )


def test_the_gap_between_the_decompositions_is_unchanged_by_the_conditional_target():
    """A two-level node with different precisions and a plan that uses both levels: the gap
    ``G_risk_ambiguity - G_utility_infogain`` of the conditional model equals the unconditional
    model's, whose target is one of the levels, since it does not depend on the target."""
    plan = jnp.asarray([[0], [1]], jnp.int32)

    def gap(conditional, level=0):
        g = {}
        for decomposition in ("utility_infogain", "risk_ambiguity"):
            system, params = _lg(conditional=conditional, level=level, decomposition=decomposition)
            tree = system.score_policies(params, system.initial_priors(params), plan)
            g[decomposition] = float(np.asarray(total(tree).lower).reshape(()))
        assert g["utility_infogain"] != g["risk_ambiguity"]
        return g["risk_ambiguity"] - g["utility_infogain"]

    np.testing.assert_allclose(gap(True), gap(False, level=0), rtol=1e-5)
    np.testing.assert_allclose(gap(True), gap(False, level=1), rtol=1e-5)


# ---------------------------------------------------- acceptance 2: the two-row payoff toy

DRIVE, ASK, IDLE = 0, 1, 2
LANE_PRECISION = 0.5  # Λ, the fast tier's per-tick lane precision
ROW, WAIT = 300.0, 40.0  # the durations the conditional target accumulates over (T4)
ROW_SD = 0.05  # the row's process noise on the lane error
OFFSET_SD = 0.3  # the prior sd of the row's offset ``b``
POSITION_SD = 0.02  # the prior sd of the position ``p = e + b``
GPS_SD = 0.05
C_ASK = 20.0  # the ask's fixed cost in nats, above one row's saving and below six rows'


def payoff_model(c_ask: float, horizon: int):
    """The slow model of a drive agent in the coordinates ``(e, b, a)``: the lane error, the
    row's offset and an indicator the ASK level sets, which the answer channel's noise map reads
    (small noise once asked, no information before). DRIVE moves one row under the steer's
    correction and clears the indicator; ASK holds and sets it; IDLE holds and gives no answer.
    The lane target is conditional on the option, precision ``300Λ`` under DRIVE and ``40Λ`` under
    ASK and IDLE; the ask's fixed cost is the node's own target, IDLE's is none."""
    system = System(
        A_deps=((0,), (0,)),
        B_deps=((1, 2), (), ()),
        controls=(1, 2),
        action_names=("option", "steer"),
        factor_names=("lane", "u_option", "u_steer"),
        modality_names=("gps", "answer"),
        policy_len=horizon,
        planning_belief="posterior",
        feedback="linear_quadratic",
        epistemic=True,
        C_deps={"lane": ("u_option",)},
    )
    reset_a = jnp.diag(jnp.asarray([1.0, 1.0, 0.0]))
    tiny = 1e-3

    def chol(e_sd, b_sd, a_sd):
        return jnp.diag(jnp.asarray([e_sd, b_sd, a_sd]))

    lane_sd = lambda duration: (duration * LANE_PRECISION) ** -0.5  # noqa: E731
    # the agent reads its position ``p = e + b`` well and the offset ``b`` poorly, so the lane
    # error ``e = p − b`` is uncertain by the offset, the two anti-correlated
    prior_cov = jnp.asarray(
        [
            [POSITION_SD**2 + OFFSET_SD**2, -(OFFSET_SD**2), 0.0],
            [-(OFFSET_SD**2), OFFSET_SD**2, 0.0],
            [0.0, 0.0, tiny**2],
        ]
    )
    params = SystemParams(
        transition={
            "lane": LinearGaussianTransition(
                F=jnp.stack([reset_a, reset_a, reset_a]),
                G=jnp.stack(
                    [jnp.asarray([[1.0], [0.0], [0.0]]), jnp.zeros((3, 1)), jnp.zeros((3, 1))]
                ),
                innovations=ConditionalGaussian(
                    loc=jnp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
                    chol=jnp.stack(
                        [chol(ROW_SD, tiny, tiny), chol(tiny, tiny, tiny), chol(tiny, tiny, tiny)]
                    ),
                    parent_ndims=1,
                ),
            )
        },
        initial_state_prior={"lane": Gaussian.from_cov(jnp.zeros(3), prior_cov)},
        likelihood={
            "gps": LinearGaussianObservation(
                H=jnp.asarray([[1.0, 1.0, 0.0]]), c=jnp.zeros(1), noise=GPS_SD * jnp.eye(1)
            ),
            "answer": LinearGaussianObservation(
                H=jnp.asarray([[0.0, 1.0, 0.0]]),
                c=jnp.zeros(1),
                noise=GridNoiseMap(
                    values=jnp.asarray([[[100.0]], [[0.02]]]), low=0.0, high=1.0, axes=(2,)
                ),
            ),
        },
        state_targets={
            "lane": ConditionalGaussian(
                loc=jnp.zeros((3, 3)),
                chol=jnp.stack(
                    [
                        chol(lane_sd(ROW), 1e3, 1e3),
                        chol(lane_sd(WAIT), 1e3, 1e3),
                        chol(lane_sd(WAIT), 1e3, 1e3),
                    ]
                ),
                parent_ndims=1,
            ),
            "u_option": Categorical(logits=jnp.asarray([0.0, -c_ask, 0.0])),
        },
        control={"u_steer": Box(-5.0 * jnp.ones(1), 5.0 * jnp.ones(1))},
    )
    return system, params


def _payoff_g(c_ask: float, rows: int) -> dict:
    """G of the two plans of the acceptance over ``rows + 1`` steps at a nominal steer of zero,
    ``never = IDLE · DRIVE^rows`` and ``ask_first = ASK · DRIVE^rows`` (both drive ``rows``
    rows and wait once)."""
    horizon = rows + 1
    system, params = payoff_model(c_ask, horizon)
    beliefs = system.initial_priors(params)
    never = (IDLE, *((DRIVE,) * rows))
    ask_first = (ASK, *((DRIVE,) * rows))
    levels = jnp.asarray(np.array([never, ask_first]), jnp.int32)
    ctx = system.planner_context(params, beliefs)
    plans = ControlStep(entries=(levels, jnp.zeros((levels.shape[0], horizon, 1))))
    tree = evaluate_policies(
        plans, ctx.init_beliefs, ctx.transition_fn, ctx.step_terms_fn, ctx.terminal_terms_fn
    )
    g = np.asarray(total(tree).lower)
    out = {
        tuple(int(x) for x in seq): float(v) for seq, v in zip(np.asarray(levels), g, strict=True)
    }
    return {"never": out[never], "ask_first": out[ask_first]}


def _crossing(c_ask: float, max_rows: int = 14) -> int:
    for rows in range(1, max_rows + 1):
        g = _payoff_g(c_ask, rows)
        if g["ask_first"] < g["never"]:
            return rows
    return max_rows + 1


def test_asking_pays_across_laps_and_the_crossing_rises_with_the_asks_cost():
    """Both plans drive ``L`` rows and wait once (``IDLE`` or ``ASK`` as the first step, at the
    same lane precision), so the gap between them is the ask's cost against what the answer
    saves: each row's correction runs on a tighter offset, a per-row saving that is positive
    and about constant in ``L``. The crossing, the row count at which asking starts to pay,
    measures ``c_ask`` over that saving, so it rises with the cost. With one row left the plan
    that never asks is better; with many the one that asks first is."""
    one = _payoff_g(C_ASK, 1)
    assert one["never"] < one["ask_first"]
    many = _payoff_g(C_ASK, 6)
    assert many["ask_first"] < many["never"]
    cheap, dear = _crossing(C_ASK), _crossing(2.25 * C_ASK)
    assert 1 < cheap < dear
    saving = {}
    for rows in range(1, 7):
        g = _payoff_g(0.0, rows)
        saving[rows] = g["never"] - g["ask_first"]
    per_row = np.diff([saving[rows] for rows in range(1, 7)])[1:]  # L = 2..5 -> L+1
    assert np.all(per_row > 0)
    assert per_row.max() < 1.2 * per_row.min(), per_row


# ------------------------------------------------------------ the family against its carrier


def test_a_conditional_gaussian_target_on_a_tabular_modality_is_refused_by_name():
    system, params = _tmaze_conditional(policy_len=1)
    params = dataclasses.replace(
        params,
        obs_targets={
            "reward": ConditionalGaussian(
                loc=jnp.zeros((3, 1)), chol=jnp.ones((3, 1, 1)), parent_ndims=1
            )
        },
    )
    with pytest.raises(ValueError, match=r"ConditionalGaussian target on 'reward'.*categorical"):
        system.score_policies(params, _initial_beliefs(system, params), _held(0, 1))


def test_a_conditional_categorical_target_on_a_gaussian_factor_is_refused_by_name():
    system, params = _lg()
    params = dataclasses.replace(
        params,
        state_targets={"x": ConditionalCategorical(log_table=jnp.zeros((3, 2)), parent_ndims=1)},
    )
    with pytest.raises(ValueError, match=r"ConditionalCategorical target on 'x'.*Gaussian"):
        system.score_policies(params, system.initial_priors(params), _held(0, 2))


# ----------------------------------------------------------- a hard zero in a state parent


def test_the_gradient_through_a_state_parents_one_hot_belief_is_finite():
    system, params = tmaze_system(policy_len=2, epistemic=False)
    table = jnp.log(jnp.asarray([[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]))
    targets = {"reward": ConditionalCategorical(log_table=table, parent_ndims=1)}
    step = jnp.asarray([2], jnp.int32)
    beliefs = (jnp.asarray([0.25, 0.25, 0.25, 0.25]), jnp.asarray([1.0, 0.0]))

    def utility(context):
        resolved = resolve_step_targets(
            targets, {"reward": (("state", 1),)}, step, (beliefs[0], context)
        )["reward"]
        return jnp.sum(resolved.log_probs * jnp.asarray([0.2, 0.3, 0.5]))

    grad = jax.grad(utility)(beliefs[1])
    assert np.isfinite(np.asarray(grad)).all()
    resolved = resolve_step_targets(targets, {"reward": (("state", 1),)}, step, beliefs)["reward"]
    np.testing.assert_allclose(np.asarray(resolved.probs), [0.5, 0.5, 0.0], atol=1e-6)
