"""Shared discrete controls (``System.B_control_deps``): a factor sliced by another's control.

The control chosen at step t for a controlled factor slices every listed factor's transition from
t to t+1, the same step, while a ``B_deps`` state parent is read a step late. Design:
docs/design/control-states.md (Timing; C6).
"""

import dataclasses
import json
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Categorical, ConditionalCategorical, ConditionalDirichlet
from hybrid_aif.composition.merge import BoundaryPrior, merge, subsystem
from hybrid_aif.description import DescriptionError, build, describe
from hybrid_aif.objectives import compiled, total
from hybrid_aif.objectives import hybrid as _hybrid
from hybrid_aif.objectives.discrete import expected_states
from hybrid_aif.reference.envs.tmaze import tmaze
from hybrid_aif.system import System, SystemParams
from tests.test_system import _initial_beliefs, tmaze_system

HELP, WHO, POSE = 0, 1, 2


def _help_table():
    """``help`` copies the control: P(help' | a) is the identity, control as the only parent."""
    return jnp.eye(2)  # (child, a)


def _who_table(noise=0.0):
    """``who`` stays put under a=0 and moves to level 1 under a=1: (child, who_prev, a)."""
    stay = jnp.eye(2)
    move = jnp.asarray([[0.0, 0.0], [1.0, 1.0]])
    t = jnp.stack([stay, move], axis=-1)
    return (1 - noise) * t + noise / 2


def _pose_table():
    """``pose`` copies the ``help`` *state* (a one-step relay): (child, pose_prev, help, ctrl)."""
    t = jnp.zeros((2, 2, 2, 1))
    for h in range(2):
        t = t.at[h, :, h, 0].set(1.0)
    return t


def _cc(probs, parent_ndims):
    return ConditionalCategorical(
        log_table=jnp.log(jnp.clip(probs, 1e-16)), parent_ndims=parent_ndims
    )


def shared_model(
    *,
    pose=False,
    learnable_who=False,
    policy_len=1,
    noise=0.0,
    epistemic=False,
    flat=False,
    declaration=False,
):
    """``help`` (controlled) and ``who``, whose transition reads ``help``'s control same-step."""
    who_B = (
        ConditionalDirichlet.from_table(_who_table(0.1), parent_ndims=2, strength=2.0)
        if learnable_who
        else _cc(_who_table(noise), 2)
    )
    B_deps = ((), (WHO,))
    controlled = (True, False)
    control_deps = ((HELP,), (HELP,))
    transition = [_cc(_help_table(), 1), who_B]
    D = [Categorical.from_probs(jnp.asarray([1.0, 0.0])), Categorical.from_probs(jnp.full(2, 0.5))]
    names = ("help", "who")
    A = [_cc(jnp.asarray([[0.9, 0.1], [0.1, 0.9]]), 1)]
    A_deps = ((WHO,),)
    if pose:
        B_deps += ((POSE, HELP),)
        controlled += (False,)
        control_deps += ((),)
        transition.append(_cc(_pose_table(), 3))
        D.append(Categorical.from_probs(jnp.asarray([1.0, 0.0])))
        names += ("pose",)
    declared = dict(
        B_deps=B_deps,
        controlled=controlled,
        policy_len=policy_len,
        A_deps=A_deps,
        epistemic=epistemic,
        modality_names=("who_obs",),
        factor_names=names,
        B_control_deps=control_deps,
    )
    if flat:  # the same model the pymdp way: who owns a second control, kept equal to help's
        declared = {
            **declared,
            "controlled": (True, True, *controlled[2:]),
            "B_control_deps": None,
        }
    params = SystemParams(
        likelihood=dict(zip(("who_obs",), A, strict=True)),
        transition=dict(zip(names, transition, strict=True)),
        initial_state_prior=dict(zip(names, D, strict=True)),
        obs_targets={"who_obs": Categorical(logits=jnp.asarray([0.0, 3.0]))},
    )
    return (declared if declaration else System(**declared)), params


def _redeclared(**overrides) -> System:
    """The default :func:`shared_model` System, declared afresh with ``overrides``."""
    declared = dict(
        B_deps=((), (WHO,)),
        controlled=(True, False),
        policy_len=1,
        A_deps=((WHO,),),
        epistemic=False,
        modality_names=("who_obs",),
        factor_names=("help", "who"),
        B_control_deps=((HELP,), (HELP,)),
    )
    return System(**{**declared, **overrides})


def flattened_model(**kwargs):
    """The same model the pymdp way: ``who`` owns a second control, kept equal to ``help``'s."""
    return shared_model(flat=True, **kwargs)


def _point(belief_levels):
    return tuple(Categorical.from_probs(jax.nn.one_hot(i, 2)) for i in belief_levels)


def _kb(system, beliefs):
    """Positional beliefs as the keyed mapping the faces take (a mapping passes through)."""
    if isinstance(beliefs, Mapping):
        return beliefs
    return system.from_positional("beliefs", beliefs)


def _pb(system, beliefs):
    """Keyed beliefs back in factor order, so the tests index them as before."""
    return system.to_positional("beliefs", beliefs)


def _ka(system, action):
    """A per-factor level vector, cut to the System's control columns, as the keyed action."""
    return system.from_positional("actions", _cols(system, action))


def _cols(system, action):
    """The control columns (the leading ones, in these models) of a per-factor action."""
    return jnp.asarray(action)[..., : len(system.names.controls)]


def _pp(system, params):
    """The positional view of keyed params, for the lower-tier functions that index it."""
    return system.to_positional("params", params)


# ============================================================== 1. the same-step read


def test_a_control_taken_at_t_moves_the_reader_at_t_plus_one():
    system, params = shared_model()
    beliefs = _point((0, 0))
    for action, want in (((1, 0), 1.0), ((0, 0), 0.0)):
        pushed = _pb(
            system, system.update_empirical_prior(params, _kb(system, beliefs), _ka(system, action))
        )
        assert float(pushed[WHO].probs[1]) == pytest.approx(want)
        predicted = system.predict_beliefs(params, _kb(system, beliefs), _cols(system, [action]))
        assert float(_pb(system, predicted[0])[WHO].probs[1]) == pytest.approx(want)


def test_the_planner_scores_the_same_step_outcome():
    system, params = shared_model()
    beliefs = _kb(system, _point((0, 0)))
    g = total(system.score_policies(params, beliefs, system.policies(params))).midpoint
    # who_obs prefers level 1; only help=1 gets there within the one-step horizon
    assert int(jnp.argmin(g)) == 1
    q_pi, _ = system.infer_policies(params, beliefs)
    assert int(system.policies(params)[jnp.argmax(q_pi.probs)][0, HELP]) == 1
    # policies carry one column per control node
    assert system.policies(params).shape == (2, 1, 1)


def test_the_generative_process_reads_the_shared_control():
    system, params = shared_model()
    env, env_params = system.as_env(params)
    _, state = env.step(
        jax.random.PRNGKey(0),
        _kb(system, (jnp.int32(0), jnp.int32(0))),
        _ka(system, [1, 0]),
        env_params,
    )
    assert int(_pb(system, state)[WHO]) == 1


# ======================================================= 2. equivalence with flattening


@pytest.mark.parametrize("learnable_who", [False, True])
def test_matches_the_flattened_model_on_the_tied_policies(learnable_who):
    kwargs = dict(policy_len=2, noise=0.05, epistemic=True, learnable_who=learnable_who)
    shared, params = shared_model(**kwargs)
    flat, flat_params = flattened_model(**kwargs)
    beliefs = (
        Categorical.from_probs(jnp.asarray([1.0, 0.0])),
        Categorical.from_probs(jnp.asarray([0.7, 0.3])),
    )
    policies = shared.policies(params)  # (P, 2, 1): help's node
    tied = jnp.concatenate([policies, policies], axis=-1)  # the flat model's two tied columns
    beliefs = _kb(shared, beliefs)
    g_shared = total(shared.score_policies(params, beliefs, policies)).midpoint
    g_flat = total(flat.score_policies(flat_params, beliefs, tied)).midpoint
    np.testing.assert_allclose(np.asarray(g_shared), np.asarray(g_flat), rtol=1e-6, atol=1e-6)
    for p in range(policies.shape[0]):
        a = shared.predict_beliefs(params, beliefs, policies[p])
        b = flat.predict_beliefs(flat_params, beliefs, tied[p])
        for qa, qb in zip(_pb(shared, a[-1]), _pb(flat, b[-1]), strict=True):
            np.testing.assert_allclose(np.asarray(qa.probs), np.asarray(qb.probs), atol=1e-6)


def test_the_hybrid_categorical_step_agrees_with_the_compiled_one():
    system, params = shared_model(noise=0.05)
    beliefs = (
        Categorical.from_probs(jnp.asarray([0.4, 0.6])),
        Categorical.from_probs(jnp.asarray([0.7, 0.3])),
    )
    pos = _pp(system, params)
    tables = compiled.compile_tables(pos.likelihood, pos.transition, {}, {}, {})
    table = system._table
    fused = compiled.make_transition(tables, table.B_deps, table)
    family = _hybrid.make_transition(
        pos.transition,
        table.B_deps,
        (_hybrid.CATEGORICAL,) * 2,
        None,
        table=table,
        space=system._space(params),
    )
    for action in ((0, 0), (1, 0)):
        a = _cols(system, action)
        got = family(beliefs, a)
        want = fused(tuple(b.probs for b in beliefs), a)
        for q, p in zip(got, want, strict=True):
            np.testing.assert_allclose(np.asarray(q.probs), np.asarray(p), atol=1e-6)


# ========================================================== 3. mixed semantics


def test_a_state_parent_keeps_the_relay_while_a_control_parent_reads_same_step():
    system, params = shared_model(pose=True)
    beliefs = _kb(system, _point((0, 0, 0)))
    first = _pb(system, system.update_empirical_prior(params, beliefs, _ka(system, [1, 0, 0])))
    assert float(first[WHO].probs[1]) == pytest.approx(1.0)  # same step
    assert float(first[POSE].probs[1]) == pytest.approx(0.0)  # reads help_t, still 0
    second = system.update_empirical_prior(params, _kb(system, first), _ka(system, [0, 0, 0]))
    second = _pb(system, second)
    assert float(second[POSE].probs[1]) == pytest.approx(1.0)  # one step later
    assert float(second[WHO].probs[1]) == pytest.approx(1.0)


# ================================================================== 4. learning


def test_a_shared_control_dirichlet_learns_on_the_taken_slice():
    system, params = shared_model(learnable_who=True)
    prev, now = _point((0, 0)), _point((1, 1))
    obs = {"who_obs": jnp.asarray(1)}
    before = params.transition["who"].concentration
    learned = system.learn(
        params,
        obs,
        _kb(system, now),
        prev_beliefs=_kb(system, prev),
        action=_ka(system, [1, 0]),
    )
    delta = np.asarray(learned.transition["who"].concentration - before)
    want = np.zeros_like(delta)
    want[1, 0, 1] = 1.0  # who'=1 | who=0, help's control=1
    np.testing.assert_allclose(delta, want, atol=1e-6)


# ============================================================ 5. no default change


def test_the_default_is_bitwise_the_explicit_own_control_entry():
    system, params = tmaze_system(policy_len=2)
    env, _ = tmaze(reward_probability=0.98, punishment_probability=0.98, cue_validity=0.98)
    explicit = System(
        A_deps=env.A_deps,
        B_deps=env.B_deps,
        controlled=(True, False),
        policy_len=2,
        gamma=16.0,
        epistemic=True,
        modality_names=("location_obs", "reward", "cue"),
        factor_names=("location", "context"),
        B_control_deps=((0,), ()),
    )
    beliefs = _kb(system, _initial_beliefs(system, params))
    _, tree = system.infer_policies(params, beliefs)
    _, tree_x = explicit.infer_policies(params, beliefs)
    assert np.array_equal(np.asarray(total(tree).midpoint), np.asarray(total(tree_x).midpoint))
    action = jnp.asarray([3, 0])
    got = _pb(system, system.update_empirical_prior(params, beliefs, _ka(system, action)))
    got_x = _pb(explicit, explicit.update_empirical_prior(params, beliefs, _ka(system, action)))
    beliefs = _pb(system, beliefs)
    transitions = _pp(system, params).transition
    # the pre-feature expression, inline: slice the trailing axis at action[f], contract
    for f, deps in enumerate(system._table.B_deps):
        sliced = jnp.take(transitions[f].log_table, action[f], axis=-1)
        cond = ConditionalCategorical(
            log_table=sliced, parent_ndims=transitions[f].parent_ndims - 1
        )
        want = cond.average(tuple(beliefs[g] for g in deps))
        assert np.array_equal(np.asarray(got[f].probs), np.asarray(want.probs))
        assert np.array_equal(np.asarray(got_x[f].probs), np.asarray(want.probs))
    assert np.array_equal(
        np.asarray(
            expected_states(beliefs, transitions, action[:1], system._table.B_deps, system._table)[
                0
            ].probs
        ),
        np.asarray(got[0].probs),
    )


# ============================================================ 6. the description


def test_a_shared_control_node_is_a_parent_of_each_reader_and_round_trips():
    system, params = shared_model(pose=True)
    document, arrays = describe(system, params)
    factors = document["factors"]
    assert factors["help"]["transition"]["parents"] == ["u_help"]
    assert factors["who"]["transition"]["parents"] == ["who", "u_help"]
    assert factors["pose"]["transition"]["parents"] == ["pose", "help"]
    rebuilt, _ = build(json.loads(json.dumps(document)), arrays=arrays)
    assert rebuilt == system and rebuilt.B_control_deps == system.B_control_deps

    plain, plain_params = tmaze_system()
    document, arrays = describe(plain, plain_params)
    again, _ = build(json.loads(json.dumps(document)), arrays=arrays)
    assert again.B_control_deps is None
    assert jax.tree_util.tree_structure(again) == jax.tree_util.tree_structure(plain)


def test_the_description_checks_shared_control_axes():
    system, params = shared_model()
    document, arrays = describe(system, params)
    document["factors"]["who"]["transition"]["parents"] = ["who"]
    with pytest.raises(DescriptionError, match=r"tables are child-first.*gives \[2, 2\]"):
        build(json.loads(json.dumps(document)), arrays=arrays)


# ================================================================= composition


def test_merge_offsets_the_entries_and_the_team_view_keeps_its_wiring():
    system, params = shared_model(policy_len=2, epistemic=True)
    plain, plain_params = tmaze_system()
    composite, composite_params = merge(
        {"maze": plain, "team": system}, {"maze": plain_params, "team": params}
    )
    assert composite.B_control_deps == ((0,), (), (2,), (2,))
    #  the team's view keeps its shared wiring, and plans as the team alone does
    view, vp = subsystem(composite, composite_params, "team")
    assert view.B_control_deps == system.B_control_deps == ((0,), (0,))
    q_view, _ = view.infer_policies(vp, view.initial_priors(vp))
    q_team, _ = system.infer_policies(params, system.initial_priors(params))
    assert jnp.array_equal(q_view.logits, q_team.logits)


def _terminated(decl, prior, axis):
    """``decl``'s control axis ``axis`` (counted from the end) sliced at an int level, or its
    table averaged over it in probability space, by hand; a trailing no-op axis when no control
    axis remains."""
    if isinstance(prior, int):
        table = jnp.take(decl.log_table, prior, axis=axis)
    else:
        moved = jnp.moveaxis(jnp.exp(decl.log_probs), axis, -1)
        table = jnp.log(jnp.einsum("...u,u->...", moved, prior.probs))
    return table[..., None] if axis == -1 else table


@pytest.mark.parametrize("prior", [1, Categorical(logits=jnp.asarray([0.3, -0.8]))])
def test_a_boundary_prior_terminates_every_reader_of_a_shared_node(prior):
    plain, plain_params = tmaze_system(policy_len=2)
    #  one node read by two factors, each through its only control axis
    system, params = shared_model(policy_len=2)
    composite, cp = merge(
        {"maze": plain, "team": system},
        {"maze": plain_params, "team": params},
        [BoundaryPrior("team.u_help", prior)],
    )
    assert composite.names.controls == ("maze.u_location",)
    assert dict(composite.subsystems)["team"].terminated == ("team.u_help",)
    for name in ("help", "who"):
        declared = params.transition[name]
        got = cp.transition[f"team.{name}"]
        assert got.parent_ndims == declared.parent_ndims
        assert jnp.array_equal(got.log_table, _terminated(declared, prior, -1)), name

    #  'who' reads two nodes: only the terminated one's axis goes, the other stays a control
    system, params = two_control_model(policy_len=2)
    composite, cp = merge(
        {"maze": plain, "team": system},
        {"maze": plain_params, "team": params},
        [BoundaryPrior("team.u_help", prior)],
    )
    assert composite.names.controls == ("maze.u_location", "team.u_mode")
    assert composite.B_control_deps == ((0,), (), (), (3,), (3,))
    help_table = cp.transition["team.help"].log_table
    assert jnp.array_equal(help_table, _terminated(params.transition["help"], prior, -1))
    who = cp.transition["team.who"]
    assert who.parent_ndims == params.transition["who"].parent_ndims - 1
    assert jnp.array_equal(who.log_table, _terminated(params.transition["who"], prior, -2))
    beliefs = composite.initial_priors(cp)
    q_pi, _ = composite.infer_policies(cp, beliefs)
    assert np.isclose(float(q_pi.probs.sum()), 1.0, atol=1e-5)


# ============================================================ 7. validation


def test_an_entry_must_name_a_controlled_factor():
    with pytest.raises(ValueError, match="not controlled"):
        _redeclared(B_control_deps=((HELP,), (WHO,)))


def test_an_entry_per_factor_and_valid_indices():
    with pytest.raises(ValueError, match="one entry per factor"):
        _redeclared(B_control_deps=((HELP,),))
    with pytest.raises(ValueError, match="not factor indices"):
        _redeclared(B_control_deps=((HELP,), (7,)))


def test_a_control_axis_must_match_the_controls_level_count():
    system, params = shared_model()
    wrong = dataclasses.replace(
        params,
        transition={**params.transition, "who": _cc(jnp.full((2, 2, 3), 0.5), 2)},
    )
    with pytest.raises(ValueError, match="different level counts"):
        system.update_empirical_prior(wrong, _kb(system, _point((0, 0))), _ka(system, [1, 0]))
    missing_axis = dataclasses.replace(
        params, transition={**params.transition, "who": _cc(jnp.full((2, 2, 1), 0.5), 2)}
    )
    with pytest.raises(ValueError, match="different level counts"):
        system.planner_context(missing_axis, _kb(system, _point((0, 0))))
    too_few = dataclasses.replace(
        params, transition={**params.transition, "who": _cc(jnp.full((2, 2), 0.5), 1)}
    )
    with pytest.raises(ValueError, match="parent_ndims"):
        system.as_env(too_few)


def _continuous_system(control_deps):
    return System(
        B_deps=((), ()),
        controlled=(True, False),
        policy_len=1,
        planner="cem",
        B_control_deps=control_deps,
    )


def test_a_control_shared_by_two_factors_is_wiring_whatever_its_kind():
    """Whether the shared node is continuous is a property of the params (its box), so the
    wiring itself is accepted and read back."""
    assert _continuous_system(((0,), ())).B_control_deps is None  # read back as the default
    assert _continuous_system(((0,), (0,))).B_control_deps == ((0,), (0,))


def test_the_amortized_path_refuses_a_shared_entry_at_every_entry_point():
    from hybrid_aif.reference.raif.world import tiny_rssm
    from hybrid_aif.train.world import assemble

    system, params = tiny_rssm(2, 1, key=jax.random.PRNGKey(0))
    assert dataclasses.replace(system, B_deps=system.B_deps) is not None  # the default wiring
    #  the node read by both factors (a native declaration)
    shared = dataclasses.replace(system, B_deps=((1, 2), (0, 2), ()))
    with pytest.raises(NotImplementedError, match="amortized path"):
        assemble(shared, params)
    with pytest.raises(NotImplementedError, match="amortized path"):
        shared._validate(params)


def test_learn_validates_the_shared_control_table():
    system, params = shared_model(learnable_who=True)
    wrong = dataclasses.replace(
        params,
        transition={
            **params.transition,
            "who": ConditionalDirichlet(concentration=jnp.ones((2, 2, 3)), parent_ndims=2),
        },
    )
    with pytest.raises(ValueError, match="different level counts"):
        system.learn(
            wrong,
            {"who_obs": jnp.asarray(1)},
            _kb(system, _point((1, 1))),
            _kb(system, _point((0, 0))),
            _ka(system, [1, 0]),
        )


# ================================================== two shared controls of different sizes

MODE = 1  # in the two-control model: help (2 controls), mode (3 controls), who
WHO2 = 2


def _two_axis_table(seed=0):
    """A random ``who`` table over (child, who_prev, help's control, mode's control)."""
    raw = np.random.default_rng(seed).uniform(0.05, 1.0, size=(2, 2, 2, 3))
    return jnp.asarray(raw / raw.sum(axis=0, keepdims=True), jnp.float32)


def two_control_model(
    *, learnable=False, policy_len=2, epistemic=True, terms=(), declaration=False
):
    table = _two_axis_table()
    who_B = (
        ConditionalDirichlet.from_table(table, parent_ndims=3, strength=3.0)
        if learnable
        else _cc(table, 3)
    )
    declared = dict(
        B_deps=((), (), (WHO2,)),
        controlled=(True, True, False),
        policy_len=policy_len,
        A_deps=((WHO2,),),
        epistemic=epistemic,
        terms=terms,
        modality_names=("who_obs",),
        factor_names=("help", "mode", "who"),
        B_control_deps=((0,), (MODE,), (0, MODE)),
    )
    params = SystemParams(
        likelihood={"who_obs": _cc(jnp.asarray([[0.8, 0.2], [0.2, 0.8]]), 1)},
        transition={"help": _cc(jnp.eye(2), 1), "mode": _cc(jnp.eye(3), 1), "who": who_B},
        initial_state_prior={
            "help": Categorical.from_probs(jnp.asarray([1.0, 0.0])),
            "mode": Categorical.from_probs(jnp.asarray([1.0, 0.0, 0.0])),
            "who": Categorical.from_probs(jnp.asarray([0.6, 0.4])),
        },
        obs_targets={"who_obs": Categorical(logits=jnp.asarray([0.0, 2.0]))},
    )
    return (declared if declaration else System(**declared)), params


def _flat_declared(declared):
    return {
        **declared,
        "controlled": (True, True, True),
        "B_control_deps": None,
    }


def _flatten_two(declared, params):
    """pymdp's flattening: ``who`` owns one combined control, index ``a_help * 3 + a_mode``."""
    who = params.transition["who"]
    if isinstance(who, ConditionalDirichlet):
        flat_who = ConditionalDirichlet(
            concentration=who.concentration.reshape(2, 2, 6), parent_ndims=2
        )
    else:
        flat_who = ConditionalCategorical(log_table=who.log_table.reshape(2, 2, 6), parent_ndims=2)
    return System(**_flat_declared(declared)), dataclasses.replace(
        params, transition={**params.transition, "who": flat_who}
    )


def _two_beliefs():
    return (
        Categorical.from_probs(jnp.asarray([0.3, 0.7])),
        Categorical.from_probs(jnp.asarray([0.2, 0.5, 0.3])),
        Categorical.from_probs(jnp.asarray([0.6, 0.4])),
    )


def test_two_control_axes_predict_against_numpy():
    system, params = two_control_model()
    table = np.asarray(_two_axis_table())
    beliefs = _two_beliefs()
    q_who = np.asarray(beliefs[WHO2].probs)
    for a_help in range(2):
        for a_mode in range(3):
            pushed = system.update_empirical_prior(
                params, _kb(system, beliefs), _ka(system, [a_help, a_mode, 0])
            )
            pushed = _pb(system, pushed)
            want = table[:, :, a_help, a_mode] @ q_who
            np.testing.assert_allclose(np.asarray(pushed[WHO2].probs), want, atol=1e-6)


@pytest.mark.parametrize("learnable", [False, True])
def test_two_control_axes_score_as_the_flattened_model(learnable):
    terms = ("occupancy:transition:who",)
    declared, params = two_control_model(learnable=learnable, terms=terms, declaration=True)
    system = System(**declared)
    flat, flat_params = _flatten_two(declared, params)
    policies = system.policies(params)  # (6, 2, 2): help's and mode's nodes
    tied = jnp.concatenate([policies, policies[..., 0:1] * 3 + policies[..., 1:2]], axis=-1)
    beliefs = _kb(system, _two_beliefs())
    shared_tree = system.score_policies(params, beliefs, policies)
    flat_tree = flat.score_policies(flat_params, beliefs, tied)
    np.testing.assert_allclose(
        np.asarray(total(shared_tree).midpoint),
        np.asarray(total(flat_tree).midpoint),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(shared_tree["occupancy"]["transition:who"].lower),
        np.asarray(flat_tree["occupancy"]["transition:who"].lower),
        atol=1e-5,
    )
    if learnable:
        np.testing.assert_allclose(
            np.asarray(shared_tree["infogain"]["transition:who"].lower),
            np.asarray(flat_tree["infogain"]["transition:who"].lower),
            atol=1e-5,
        )


def test_two_control_axes_learn_on_the_taken_slice():
    system, params = two_control_model(learnable=True)
    before = np.asarray(params.transition["who"].concentration)
    prev = (
        Categorical.from_probs(jnp.asarray([1.0, 0.0])),
        Categorical.from_probs(jnp.asarray([1.0, 0.0, 0.0])),
        Categorical.from_probs(jnp.asarray([1.0, 0.0])),
    )
    now = (
        Categorical.from_probs(jnp.asarray([0.0, 1.0])),
        Categorical.from_probs(jnp.asarray([0.0, 0.0, 1.0])),
        Categorical.from_probs(jnp.asarray([0.0, 1.0])),
    )
    learned = system.learn(
        params,
        {"who_obs": jnp.asarray(1)},
        _kb(system, now),
        _kb(system, prev),
        _ka(system, [1, 2, 0]),
    )
    delta = np.asarray(learned.transition["who"].concentration) - before
    want = np.zeros_like(delta)
    want[1, 0, 1, 2] = 1.0  # who'=1 | who=0, help's control 1, mode's control 2
    np.testing.assert_allclose(delta, want, atol=1e-6)


# ================================================ the leaves, against numpy, both paths


def _entropy_rows(probs):
    return -(probs * np.log(np.clip(probs, 1e-30, None))).sum(axis=0)


def _novelty_np(conc):
    """Exact Dirichlet novelty weights over the child axis (``spm_wnorm``, sign +)."""
    from scipy.special import digamma

    a = np.asarray(conc, np.float64)
    a0 = a.sum(axis=0, keepdims=True)
    return np.log(a0) - np.log(a) + 1 / a - 1 / a0 + digamma(a) - digamma(a0)


@pytest.mark.parametrize("hybrid", [False, True])
def test_occupancy_and_novelty_read_the_shared_control(hybrid):
    declared, params = shared_model(learnable_who=True, declaration=True)
    # the two control slices differ in spread and in counts
    noisy = np.stack(
        [2.0 * np.asarray(_who_table(0.1))[..., 0], 6.0 * np.asarray(_who_table(0.6))[..., 1]],
        axis=-1,
    )
    params = dataclasses.replace(
        params,
        transition={
            **params.transition,
            "who": ConditionalDirichlet(concentration=jnp.asarray(noisy), parent_ndims=2),
        },
    )
    declared = {
        **declared,
        "terms": ("occupancy:transition:who",),
        "epistemic": ("transition:who",),
    }
    beliefs = (
        Categorical.from_probs(jnp.asarray([1.0, 0.0])),
        Categorical.from_probs(jnp.asarray([0.7, 0.3])),
    )
    if hybrid:
        system, params, beliefs = _hybridize(declared, params, beliefs)
    else:
        system = System(**declared)
    tree = system.score_policies(params, _kb(system, beliefs), system.policies(params))
    q_prev = np.asarray(beliefs[WHO].probs)
    mean = noisy / noisy.sum(axis=0, keepdims=True)
    w = _novelty_np(noisy)
    for p, a in enumerate((0, 1)):
        q_next = mean[:, :, a] @ q_prev
        occupancy = _entropy_rows(mean[:, :, a]) @ q_prev
        novelty = q_next @ (w[:, :, a] @ q_prev)
        assert float(tree["occupancy"]["transition:who"].lower[p]) == pytest.approx(
            occupancy, abs=1e-5
        )
        assert float(tree["infogain"]["transition:who"].lower[p]) == pytest.approx(
            novelty, abs=1e-5
        )


# =================================================== a full hybrid System


def _hybridize(declared, params, beliefs):
    """Append an uncontrolled LG factor and an LG modality: every face takes the hybrid path."""
    from hybrid_aif.distributions import ConditionalGaussian, Gaussian
    from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianTransition

    n = len(declared["B_deps"])
    drift = LinearGaussianTransition(
        F=jnp.eye(1),
        innovations=ConditionalGaussian(
            loc=jnp.zeros((1, 1)), chol=0.1 * jnp.ones((1, 1, 1)), parent_ndims=1
        ),
    )
    bcd = declared["B_control_deps"]
    system = System(
        **{
            **declared,
            "B_deps": (*declared["B_deps"], ()),
            "controlled": (*declared["controlled"], False),
            "A_deps": (*declared["A_deps"], (n,)),
            "modality_names": (*declared["modality_names"], "drift_obs"),
            "factor_names": (*declared["factor_names"], "drift"),
            "B_control_deps": None if bcd is None else (*bcd, ()),
        }
    )
    prior = Gaussian(loc=jnp.zeros(1), chol=jnp.eye(1))
    params = dataclasses.replace(
        params,
        transition={**params.transition, "drift": drift},
        likelihood={
            **params.likelihood,
            "drift_obs": LinearGaussianObservation(
                H=jnp.eye(1), c=jnp.zeros(1), noise=0.2 * jnp.eye(1)
            ),
        },
        initial_state_prior={**params.initial_state_prior, "drift": prior},
    )
    return system, params, (*beliefs, prior)


def test_a_hybrid_system_steps_and_scores_the_shared_control():
    declared, params = two_control_model(declaration=True)
    beliefs = _two_beliefs()
    _, flat_params = _flatten_two(declared, params)
    flat_declared = _flat_declared(declared)
    system, params, h_beliefs = _hybridize(declared, params, beliefs)
    flat, flat_params, _ = _hybridize(flat_declared, flat_params, beliefs)
    assert system._is_hybrid(params)
    policies = system.policies(params)
    tied = jnp.concatenate([policies, policies[..., 0:1] * 3 + policies[..., 1:2]], axis=-1)
    h_beliefs = _kb(system, h_beliefs)
    np.testing.assert_allclose(
        np.asarray(total(system.score_policies(params, h_beliefs, policies)).midpoint),
        np.asarray(total(flat.score_policies(flat_params, h_beliefs, tied)).midpoint),
        rtol=1e-5,
        atol=1e-5,
    )
    pushed = system.update_empirical_prior(params, h_beliefs, _ka(system, [1, 2, 0, 0]))
    pushed = _pb(system, pushed)
    want = np.asarray(_two_axis_table())[:, :, 1, 2] @ np.asarray(beliefs[WHO2].probs)
    np.testing.assert_allclose(np.asarray(pushed[WHO2].probs), want, atol=1e-6)

    # the generative process: a deterministic table, distinct for every (help, mode) pair
    sure = np.zeros((2, 2, 2, 3), np.float32)
    for h in range(2):
        for m in range(3):
            sure[int(h == 1 and m == 2), :, h, m] = 1.0
    env, env_params = system.as_env(
        dataclasses.replace(
            params,
            transition={**params.transition, "who": _cc(jnp.asarray(sure), 3)},
        )
    )
    state = _kb(system, (jnp.int32(0), jnp.int32(0), jnp.int32(0), jnp.zeros(1)))
    for action, want_who in (((1, 2, 0, 0), 1), ((0, 2, 0, 0), 0), ((1, 1, 0, 0), 0)):
        _, nxt = env.step(jax.random.PRNGKey(1), state, _ka(system, action), env_params)
        assert int(_pb(system, nxt)[WHO2]) == want_who
