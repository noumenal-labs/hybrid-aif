"""Control states as factor-table nodes: the sugar's translation, the node form, and the faces.

Design: docs/design/control-states.md (C1, C2, C6, C10, C11); the build plan's Phase 1b.
"""

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from hybrid_aif import System
from hybrid_aif.reference.raif.world import tiny_rssm
from tests.test_mlp_conditional import _mlp_transition_system
from tests.test_shared_controls import shared_model, two_control_model
from tests.test_system import tmaze_system


def _same(a: System, b: System) -> None:
    """Equal, hash-equal, one treedef, and one jit cache entry."""
    assert a == b
    assert hash(a) == hash(b)
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    calls = []

    @jax.jit
    def f(system, x):
        calls.append(None)
        return x + len(system.B_deps)

    f(a, jnp.zeros(()))
    f(b, jnp.zeros(()))
    assert len(calls) == 1


def _tmaze_native(**kw) -> System:
    return System(
        A_deps=((0,), (0, 1), (0, 1)),
        B_deps=((0, 2), (1,), ()),
        factor_names=("location", "context", "u_location"),
        modality_names=("location_obs", "reward", "cue"),
        controls=(2,),
        action_names=("a_location",),
        policy_len=2,
        **kw,
    )


def test_tmaze_sugar_and_native_twin_are_one_system():
    sugar, _ = tmaze_system(policy_len=2)
    _same(sugar, _tmaze_native())


def test_shared_control_toys_twins():
    sugar, _ = shared_model(pose=True, learnable_who=True, policy_len=2, epistemic=True)
    native = System(
        B_deps=((3,), (1, 3), (2, 0), ()),
        A_deps=((1,),),
        policy_len=2,
        epistemic=True,
        modality_names=("who_obs",),
        factor_names=("help", "who", "pose", "u_help"),
        controls=(3,),
        action_names=("a_help",),
    )
    _same(sugar, native)
    sugar, _ = two_control_model(learnable=True, policy_len=2)
    native = System(
        B_deps=((3,), (4,), (2, 3, 4), (), ()),
        A_deps=((2,),),
        policy_len=2,
        epistemic=True,
        modality_names=("who_obs",),
        factor_names=("help", "mode", "who", "u_help", "u_mode"),
        controls=(3, 4),
        action_names=("a_help", "a_mode"),
    )
    _same(sugar, native)


def test_mlp_transition_twin():
    sugar, _ = _mlp_transition_system()
    native = System(
        A_deps=(),
        B_deps=((1,), ()),
        policy_len=1,
        factor_names=("x", "u_x"),
        epistemic=False,
        controls=(1,),
        action_names=("a_x",),
    )
    _same(sugar, native)


def test_continuous_control_twin():
    sugar, params = tiny_rssm(
        2, 1, key=jax.random.PRNGKey(0), units=4, deter=8, num_vars=3, num_classes=4
    )
    names = (*sugar.factor_names[:2], "u_" + sugar.factor_names[0])
    assert set(params.control) == {names[2]}  # the box lives in params, keyed by the node
    assert sugar.factor_names == names
    native = System(
        A_deps=sugar.A_deps,
        B_deps=((1, 2), (0,), ()),
        policy_len=sugar.policy_len,
        planner="cem",
        epistemic=False,
        factor_names=names,
        modality_names=sugar.modality_names,
        controls=(2,),
        action_names=("a_" + sugar.factor_names[0],),
        recognition=sugar.recognition,
    )
    _same(sugar, native)


def test_sugar_reads_back_and_levels_live_with_tables():
    s, p = tmaze_system(policy_len=2)
    assert s.controlled == (True, False)
    assert s.B_control_deps is None
    assert not hasattr(s, "continuous_controls")
    with pytest.raises(TypeError, match="unexpected keyword argument .num_controls."):
        System(B_deps=((), ()), num_controls=(2, 1), controlled=(True, False), policy_len=1)
    assert not hasattr(s, "num_controls")
    assert s.names.states == ("location", "context")
    assert s.names.controls == ("u_location",)
    assert s._control_levels(p) == (5,)


def test_shared_wiring_reads_back():
    s, _ = two_control_model(learnable=True, policy_len=2)
    assert s.controlled == (True, True, False)
    assert s.B_control_deps == ((0,), (1,), (0, 1))
    #  declared-but-default wiring collapses to None
    d = System(
        B_deps=((), ()),
        controlled=(True, False),
        B_control_deps=((0,), ()),
        policy_len=1,
    )
    assert d.B_control_deps is None
    assert d == System(B_deps=((), ()), controlled=(True, False), policy_len=1)


def test_generated_names():
    s = System(B_deps=((), ()), controlled=(True, False), policy_len=1)
    assert s.factor_names == ("factor_0", "factor_1", "u_factor_0")
    assert s.modality_names == ()
    assert dataclasses.replace(s, gamma=2.0).factor_names == s.factor_names


def test_replace_with_sugar_on_nodes_raises():
    s, _ = tmaze_system(policy_len=2)
    assert dataclasses.replace(s, planner="sophisticated", branch_on=("cue",)).controls == (2,)
    with pytest.raises(ValueError, match="construct the System afresh"):
        dataclasses.replace(s, controlled=(True, False))
    with pytest.raises(ValueError, match="factor_names=None on a System with control nodes"):
        dataclasses.replace(s, factor_names=None)


def test_sugar_validation_sentences():
    with pytest.raises(ValueError, match="the name\\(s\\) the control sugar gives"):
        System(
            B_deps=((), ()),
            controlled=(True, False),
            factor_names=("a", "u_a"),
            policy_len=1,
        )
    with pytest.raises(ValueError, match="give one declaration or the other"):
        System(
            B_deps=((1,), ()),
            controlled=(True, False),
            controls=(1,),
            factor_names=("a", "u_a"),
            policy_len=1,
        )
    with pytest.raises(ValueError, match="read by no factor"):
        System(
            B_deps=((), ()),
            controls=(1,),
            factor_names=("a", "u_go"),
            action_names=("go",),
            policy_len=1,
        )


def test_node_beliefs_at_the_faces():
    s, p = tmaze_system(policy_len=2)
    obs = {"location_obs": jnp.asarray(0), "reward": jnp.asarray(0), "cue": jnp.asarray(0)}
    priors = s.initial_priors(p)
    assert list(priors) == ["location", "context", "u_location"]
    assert priors["u_location"].probs.shape == (5,)
    beliefs, _ = s.infer_states(p, obs)
    assert list(beliefs) == ["location", "context", "u_location"]
    #  a node entry is optional on input and passes through infer_states untouched
    marked = {**priors, "u_location": priors["u_location"]}
    again, _ = s.infer_states(p, obs, priors=marked)
    assert again["u_location"] is marked["u_location"]
    no_node, _ = s.infer_states(p, obs, priors={k: priors[k] for k in ("location", "context")})
    assert jnp.array_equal(no_node["location"].probs, again["location"].probs)
    assert s.to_positional("beliefs", {k: beliefs[k] for k in ("location", "context")})


# ------------------------------------------------------------------ policies, one column per node


def test_policies_are_the_old_enumerations_controlled_columns():
    from hybrid_aif.planning import enumerate_policies

    for build, num_controls, controlled in (
        (lambda: tmaze_system(policy_len=2), (5, 1), (True, False)),
        (lambda: two_control_model(policy_len=2), (2, 3, 1), (True, True, False)),
    ):
        s, p = build()
        old = enumerate_policies(num_controls, s.policy_len, controlled)
        cols = [f for f, c in enumerate(controlled) if c]
        assert jnp.array_equal(s.policies(p), old[..., jnp.asarray(cols)])


def test_planner_context_plans_to_its_horizon():
    from hybrid_aif.planning import get_planner

    s, p = tmaze_system(policy_len=2)
    beliefs = s.initial_priors(p)
    q2, _, pol2 = get_planner("enumeration")(s.planner_context(p, beliefs))
    q1, _, pol1 = get_planner("enumeration")(s.planner_context(p, beliefs, horizon=1))
    assert pol2.shape == (25, 2, 1) and pol1.shape == (5, 1, 1)
    q_default, _ = s.infer_policies(p, beliefs)
    assert jnp.array_equal(q_default.probs, q2.probs)
    ctx = s.planner_context(p, beliefs)
    assert ctx.num_controls == (5,) and ctx.controlled == (True,)


def test_score_policies_takes_one_column_per_node():
    s, p = tmaze_system(policy_len=2)
    beliefs = s.initial_priors(p)
    tree = s.score_policies(p, beliefs, s.policies(p))
    q, ref = s.infer_policies(p, beliefs)
    leaf = jax.tree_util.tree_leaves(tree)[0]
    assert leaf.shape[0] == 25
    with pytest.raises(ValueError, match="one column per control node"):
        s.score_policies(p, beliefs, jnp.zeros((25, 2, 2), jnp.int32))


# ------------------------------------------------------------- the action, one column per node


def test_positional_actions_have_one_column_per_node():
    s, p = two_control_model(policy_len=2)
    action = {"a_help": jnp.asarray(1), "a_mode": jnp.asarray(2)}
    positional = s.to_positional("actions", action)
    assert positional.shape == (2,)
    assert jnp.array_equal(positional, jnp.asarray([1, 2]))
    back = s.from_positional("actions", positional)
    assert list(back) == ["a_help", "a_mode"]
    env, ep = s.as_env(p)
    assert set(env.null_action()) == {"a_help", "a_mode"}


def test_update_empirical_prior_reads_each_nodes_column():
    """The same action as the per-factor reference: each factor sliced by its readers' levels."""
    from hybrid_aif.objectives.discrete import expected_states

    s, p = two_control_model(policy_len=2)
    beliefs = s.initial_priors(p)
    pos = s.to_positional("params", p)
    qs = s.to_positional("beliefs", {k: beliefs[k] for k in ("help", "mode", "who")})
    for a in ((0, 0), (1, 2), (0, 1)):
        got = s.update_empirical_prior(p, beliefs, {"a_help": a[0], "a_mode": a[1]})
        want = expected_states(qs, pos.transition, jnp.asarray([*a, 0]), s._B, ((0,), (1,), (0, 1)))
        for name, q in zip(("help", "mode", "who"), want, strict=True):
            assert jnp.array_equal(got[name].probs, q.probs)
        assert got["u_help"].probs.shape == (2,) and got["u_mode"].probs.shape == (3,)


# ------------------------------------------------------------------------ native-only shapes


def _tmaze_variant(B_deps, location_table, context_table):
    """The T-maze declared natively with its tables rearranged; params by name."""
    from hybrid_aif import SystemParams
    from hybrid_aif.distributions import ConditionalCategorical

    sugar, p = tmaze_system(policy_len=2)
    native = System(
        A_deps=sugar.A_deps,
        B_deps=B_deps,
        factor_names=sugar.factor_names,
        modality_names=sugar.modality_names,
        controls=(2,),
        action_names=sugar.action_names,
        policy_len=2,
    )
    loc, ctx = p.transition["location"], p.transition["context"]
    params = SystemParams(
        likelihood=dict(p.likelihood),
        transition={
            "location": ConditionalCategorical(
                log_table=location_table(loc.log_table), parent_ndims=2
            ),
            "context": ConditionalCategorical(
                log_table=context_table(ctx.log_table),
                parent_ndims=context_table(ctx.log_table).ndim - 1,
            ),
        },
        initial_state_prior=dict(p.initial_state_prior),
        obs_targets=dict(p.obs_targets),
    )
    return sugar, p, native, params


def _same_faces(sugar, p, native, q) -> None:
    obs = {"location_obs": jnp.asarray(0), "reward": jnp.asarray(0), "cue": jnp.asarray(0)}
    b_s, _ = sugar.infer_states(p, obs)
    b_n, _ = native.infer_states(q, obs)
    q_s, tree_s = sugar.infer_policies(p, b_s)
    q_n, tree_n = native.infer_policies(q, b_n)
    assert jnp.array_equal(q_s.probs, q_n.probs)
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(jnp.array_equal, tree_s, tree_n))
    for level in range(5):
        a = {"a_location": jnp.asarray(level)}
        u_s = sugar.update_empirical_prior(p, b_s, a)
        u_n = native.update_empirical_prior(q, b_n, a)
        for name in ("location", "context"):
            assert jnp.array_equal(u_s[name].probs, u_n[name].probs)


def test_a_control_parent_need_not_trail():
    sugar, p, native, q = _tmaze_variant(
        ((2, 0), (1,), ()), lambda t: jnp.swapaxes(t, 1, 2), lambda t: t
    )
    with pytest.raises(AttributeError, match="not in that shape"):  # not sugar-shaped
        tuple(native.controlled)
    _same_faces(sugar, p, native, q)


def test_a_table_may_omit_the_size_one_axis():
    sugar, p, native, q = _tmaze_variant(((0, 2), (1,), ()), lambda t: t, lambda t: t[..., 0])
    assert native == sugar  # the declaration is the sugar's; only the context table differs
    _same_faces(sugar, p, native, q)


def test_a_document_writes_a_control_node_in_any_parent_position():
    from hybrid_aif.description import build, describe

    _, _, native, q = _tmaze_variant(
        ((2, 0), (1,), ()), lambda t: jnp.swapaxes(t, 1, 2), lambda t: t
    )
    document, arrays = describe(native, q)
    assert document["factors"]["location"]["transition"]["parents"] == ["u_location", "location"]
    assert build(document, arrays=arrays)[0] == native


def _shared_pair():
    """One control node ``u`` (3 levels) read by two factors, natively (``u`` first for ``b``)
    and as the sugar's shared control of ``a``."""
    from hybrid_aif import SystemParams
    from hybrid_aif.distributions import Categorical, ConditionalCategorical

    key_a, key_b = jax.random.split(jax.random.PRNGKey(3))
    table_a = jax.random.normal(key_a, (2, 2, 3))  # (a', a, u)
    table_b = jax.random.normal(key_b, (2, 2, 3))  # (b', b, u)
    common = dict(
        A_deps=((0, 1),),
        modality_names=("o",),
        policy_len=2,
        epistemic=True,
    )
    sugar = System(
        B_deps=((0,), (1,)),
        controlled=(True, False),
        B_control_deps=((0,), (0,)),
        factor_names=("a", "b"),
        **common,
    )
    native = System(
        B_deps=((0, 2), (2, 1), ()),
        factor_names=("a", "b", "u_u"),
        controls=(2,),
        action_names=("u",),
        **common,
    )

    def params(b_table):
        return SystemParams(
            likelihood={
                "o": ConditionalCategorical(
                    log_table=jax.random.normal(jax.random.PRNGKey(5), (3, 2, 2)), parent_ndims=2
                )
            },
            transition={
                "a": ConditionalCategorical(log_table=table_a, parent_ndims=2),
                "b": ConditionalCategorical(log_table=b_table, parent_ndims=2),
            },
            initial_state_prior={
                "a": Categorical(logits=jnp.asarray([0.3, -0.2])),
                "b": Categorical(logits=jnp.asarray([-0.1, 0.4])),
            },
            obs_targets={"o": Categorical(logits=jnp.asarray([1.0, 0.0, -1.0]))},
        )

    return sugar, params(table_b), native, params(jnp.swapaxes(table_b, 1, 2))


def test_a_node_read_by_two_factors_plans_as_the_enumeration_oracle():
    from hybrid_aif.objectives.base import total
    from hybrid_aif.objectives.discrete import expected_states

    sugar, p, native, q = _shared_pair()
    assert native.names.controls == ("u_u",)
    beliefs = native.initial_priors(q)
    assert native.policies(q).shape == (9, 2, 1)
    q_n, tree_n = native.infer_policies(q, beliefs)
    q_s, tree_s = sugar.infer_policies(p, sugar.initial_priors(p))
    assert jnp.array_equal(q_n.probs, q_s.probs)
    #  the oracle: each policy's predicted beliefs, pushed by hand through both tables
    pos = native.to_positional("params", q)
    qs0 = native.to_positional("beliefs", {k: beliefs[k] for k in ("a", "b")})
    for policy in native.policies(q):
        qs = qs0
        for step in policy:
            qs = expected_states(
                qs,
                (pos.transition[0], _swap(pos.transition[1])),
                jnp.asarray([step[0], step[0]]),
                ((0,), (1,)),
                ((0,), (1,)),
            )
        got = native.predict_beliefs(q, beliefs, policy)[-1]
        for name, want in zip(("a", "b"), qs, strict=True):
            assert jnp.allclose(got[name].probs, want.probs, atol=1e-6)
    assert jnp.argmax(q_n.probs) == jnp.argmin(total(tree_n).midpoint)


def _swap(decl):
    from hybrid_aif.distributions import ConditionalCategorical

    return ConditionalCategorical(log_table=jnp.swapaxes(decl.log_table, 1, 2), parent_ndims=2)


def test_timing_a_control_parent_moves_its_reader_next_step_an_echo_a_tick_later():
    """``x`` reads the control node same step; ``echo`` copies it; ``y`` reads ``echo``."""
    from hybrid_aif import SystemParams
    from hybrid_aif.distributions import Categorical, ConditionalCategorical

    copy_u = jnp.log(jnp.eye(2) + 1e-9)  # (child, u)
    copy_state = jnp.log(jnp.eye(2) + 1e-9)  # (child, parent)
    system = System(
        B_deps=((3,), (3,), (1,), ()),
        factor_names=("x", "echo", "y", "u_u"),
        controls=(3,),
        action_names=("u",),
        policy_len=1,
    )
    start = Categorical(logits=jnp.log(jnp.asarray([1.0, 1e-9])))
    params = SystemParams(
        transition={
            "x": ConditionalCategorical(log_table=copy_u, parent_ndims=1),
            "echo": ConditionalCategorical(log_table=copy_u, parent_ndims=1),
            "y": ConditionalCategorical(log_table=copy_state, parent_ndims=1),
        },
        initial_state_prior={"x": start, "echo": start, "y": start},
    )
    beliefs = system.initial_priors(params)
    one = system.update_empirical_prior(params, beliefs, {"u": jnp.asarray(1)})
    assert [int(jnp.argmax(one[n].probs)) for n in ("x", "echo", "y")] == [1, 1, 0]
    two = system.update_empirical_prior(params, one, {"u": jnp.asarray(0)})
    assert [int(jnp.argmax(two[n].probs)) for n in ("x", "echo", "y")] == [0, 0, 1]


def test_a_shared_node_learns_and_steps_by_its_parent_positions():
    import dataclasses as dc

    from hybrid_aif.distributions import ConditionalDirichlet

    sugar, p, native, q = _shared_pair()

    def learnable(params):
        b = params.transition["b"]
        return dc.replace(
            params,
            transition={
                **params.transition,
                "b": ConditionalDirichlet.from_table(jnp.exp(b.log_table), parent_ndims=2),
            },
        )

    p, q = learnable(p), learnable(q)
    obs = {"o": jnp.asarray(1)}
    learned = []
    for system, params in ((sugar, p), (native, q)):
        beliefs, _ = system.infer_states(params, obs)
        new = system.learn(
            params,
            obs,
            beliefs,
            prev_beliefs=system.initial_priors(params),
            action={system.names.actions[0]: jnp.asarray(2)},
        )
        learned.append(new.transition["b"].concentration)
    # equal up to the rounding of the two contraction orders
    assert jnp.allclose(jnp.swapaxes(learned[1], 1, 2), learned[0], rtol=1e-6, atol=0.0)
    env_s, ep_s = sugar.as_env(p)
    env_n, ep_n = native.as_env(q)
    key = jax.random.PRNGKey(0)
    _, state = env_s.reset(key, ep_s)
    o_s, s_s = env_s.step(key, state, {"a_a": jnp.asarray(2)}, ep_s)
    o_n, s_n = env_n.step(key, state, {"u": jnp.asarray(2)}, ep_n)
    assert {k: int(v) for k, v in s_s.items()} == {k: int(v) for k, v in s_n.items()}


# ------------------------------------------------------------------------------ composition


def test_composite_control_nodes_are_namespaced_like_any_node():
    from hybrid_aif.composition.merge import merge
    from tests.test_merge import merged

    composite, cp = merged()
    assert composite.names.controls == ("agent.u_location",)
    assert "agent.u_location" in composite.factor_names
    action, _, _ = composite.act(jax.random.PRNGKey(0), cp, composite.initial_priors(cp))
    assert list(action) == ["agent.a_location"]
    native, q = _tmaze_variant(((2, 0), (1,), ()), lambda t: jnp.swapaxes(t, 1, 2), lambda t: t)[2:]
    sugar, p = tmaze_system(policy_len=2)
    one, op = merge({"a": native}, {"a": q})
    twin, tp = merge({"a": sugar}, {"a": p})
    assert one.names.controls == twin.names.controls == ("a.u_location",)
    beliefs, _ = one.infer_states(op, {f"a.{m}": jnp.asarray(0) for m in sugar.modality_names})
    q_one, _ = one.infer_policies(op, beliefs)
    q_twin, _ = twin.infer_policies(tp, beliefs)
    assert jnp.array_equal(q_one.probs, q_twin.probs)
    for level in range(5):
        pushed = one.update_empirical_prior(op, beliefs, {"a.a_location": level})
        want = twin.update_empirical_prior(tp, beliefs, {"a.a_location": level})
        for name in ("a.location", "a.context"):
            assert jnp.array_equal(pushed[name].probs, want[name].probs)


def test_a_native_shared_node_replays_as_its_sugar_twin():
    from hybrid_aif.replay import replay

    sugar, p, native, q = _shared_pair()
    obs = {"o": jnp.asarray([0, 2, 1])}
    levels = jnp.asarray([2, 0, 1])
    r_s = replay(sugar, p, obs, {"a_a": levels})
    r_n = replay(native, q, obs, {"u": levels})
    for name in ("a", "b"):
        assert jnp.array_equal(r_s.beliefs[name].probs, r_n.beliefs[name].probs)
    assert jnp.array_equal(r_s.q_pi.probs, r_n.q_pi.probs)
    assert jnp.array_equal(r_s.chosen["a_a"], r_n.chosen["u"])
    assert jnp.array_equal(r_s.action_log_prob, r_n.action_log_prob)


def test_a_wrapped_outside_env_steps_a_native_node_in_its_first_readers_column():
    from hybrid_aif.envs import GenerativeEnv
    from hybrid_aif.envs._process import DiscreteProcess

    sugar, p, native, _ = _shared_pair()
    _, env_params = sugar.as_env(p)  # the outside env's own tables: per-factor layout
    outside = DiscreteProcess(A_deps=((0, 1),), B_deps=((0,), (1,)), B_control_deps=((0,), (0,)))
    key = jax.random.PRNGKey(4)
    wrapped_s = GenerativeEnv.wrap(outside, sugar)
    wrapped_n = GenerativeEnv.wrap(outside, native)
    _, state = wrapped_s.reset(key, env_params)
    assert set(wrapped_n.null_action()) == {"u"}
    for level in range(3):
        o_s, s_s = wrapped_s.step(key, state, {"a_a": jnp.asarray(level)}, env_params)
        o_n, s_n = wrapped_n.step(key, state, {"u": jnp.asarray(level)}, env_params)
        assert {k: int(v) for k, v in s_s.items()} == {k: int(v) for k, v in s_n.items()}
        assert {k: int(v) for k, v in o_s.items()} == {k: int(v) for k, v in o_n.items()}


def test_a_document_writes_a_table_with_or_without_its_size_one_axis():
    from hybrid_aif.description import build, describe

    sugar, p, native, q = _tmaze_variant(((0, 2), (1,), ()), lambda t: t, lambda t: t[..., 0])
    for system, params, shape in ((sugar, p, [2, 2, 1]), (native, q, [2, 2])):
        document, arrays = describe(system, params)
        assert document["variables"]["B_context"]["shape"] == shape
        rebuilt, rebuilt_params = build(document, arrays=arrays)
        assert rebuilt == system
        assert rebuilt_params.transition["context"].parent_ndims == len(shape) - 1


def test_a_term_s_own_model_writes_its_control_nodes_from_its_params():
    from hybrid_aif import description as D

    inner, ip = tmaze_system(policy_len=1)
    body = D._write_system(D._Writer("inline", 256), inner, inner._p(ip), "row/")
    assert body["variables"]["u_location"] == {"domain": "categorical", "levels": 5}
    assert body["relations"]["controls"] == {"u_location": "a_location"}


def test_initial_and_terminal_parents_are_declared_per_node():
    s, _ = tmaze_system(policy_len=2)
    assert dataclasses.replace(s, H_deps=((), (0,), ())).H_deps == ((), (0,), ())
    for field in ("D_deps", "H_deps"):
        with pytest.raises(ValueError, match="has 2 entries for 3 nodes"):
            dataclasses.replace(s, **{field: ((), ())})


def test_a_node_entry_in_input_beliefs_is_checked():
    from hybrid_aif.distributions import Categorical

    s, p = tmaze_system(policy_len=2)
    beliefs = s.initial_priors(p)
    obs = {"location_obs": jnp.asarray(0), "reward": jnp.asarray(0), "cue": jnp.asarray(0)}
    for bad, got in (
        (Categorical(logits=jnp.zeros(7)), "a Categorical over 7 level"),
        (jnp.zeros(5), "a ArrayImpl"),
    ):
        wrong = {**beliefs, "u_location": bad}
        with pytest.raises(ValueError, match=f"its 5 level\\(s\\); got {got}"):
            s.infer_states(p, obs, priors=wrong)
        with pytest.raises(ValueError, match="control node 'u_location'"):
            s.infer_policies(p, wrong)


def test_a_controlled_factor_whose_control_nobody_reads_is_refused_in_sugar_terms():
    with pytest.raises(ValueError, match=r"factor\(s\) \['b'\] are declared controlled but no"):
        System(
            B_deps=((), ()),
            controlled=(True, True),
            B_control_deps=((0,), (0,)),
            factor_names=("a", "b"),
            policy_len=1,
        )


def test_sugar_sentences_name_counts_and_the_face():
    with pytest.raises(ValueError, match=r"factor_names has 1 name\(s\) for the 2 factor\(s\)"):
        System(
            B_deps=((), ()),
            controlled=(True, True),
            B_control_deps=((0, 3), (1,)),
            factor_names=("a",),
            policy_len=1,
        )
    s, _ = tmaze_system(policy_len=2)
    with pytest.raises(ValueError, match="construct the System afresh with one name per node"):
        dataclasses.replace(s, B_deps=((0,), (1,)))
    native, q = _tmaze_variant(((2, 0), (1,), ()), lambda t: jnp.swapaxes(t, 1, 2), lambda t: t)[2:]
    with pytest.raises(NotImplementedError, match="^describe: |^random_policy: "):
        from hybrid_aif.rollout import random_policy

        random_policy(native, q)


# ------------------------------------------- clamp and belief, targets on control nodes (C3, C7)


def _probs_of(levels, n: int):
    """One-hot rows over ``n`` levels, as a Categorical (the clamp written as a belief)."""
    from hybrid_aif.distributions import Categorical

    return Categorical(logits=jnp.log(jax.nn.one_hot(levels, n)))


def _leaves_equal(a, b) -> bool:
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    same = jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    return same and all(bool(jnp.array_equal(x, y)) for x, y in zip(la, lb, strict=True))


def test_c7_a_control_node_accepts_and_emits_messages():
    """C7's fence: a non-degenerate distribution on a control node changes what
    score_policies scores and comes back out under the node's name in the emitted beliefs
    (its entry there is the message the node emits); the clamp is the one-hot case."""
    from hybrid_aif.distributions import Categorical

    s, p = tmaze_system(policy_len=2)
    beliefs = s.initial_priors(p)
    plan = jnp.asarray([3, 1], jnp.int32)  # cue, then a reward arm
    clamped = s.score_policies(p, beliefs, plan[:, None])
    #  the clamp, keyed by the node: the positional plan, bitwise
    assert _leaves_equal(s.score_policies(p, beliefs, {"u_location": plan}), clamped)
    #  the one-hot belief is the clamp
    assert _leaves_equal(s.score_policies(p, beliefs, {"u_location": _probs_of(plan, 5)}), clamped)
    #  a non-degenerate belief scores something else
    message = Categorical(logits=jnp.log(jnp.asarray([[0.1, 0.1, 0.1, 0.6, 0.1]] * 2)))
    mixed = s.score_policies(p, beliefs, {"u_location": message})
    assert jax.tree_util.tree_structure(mixed) == jax.tree_util.tree_structure(clamped)
    assert not _leaves_equal(mixed, clamped)
    #  and comes back out under the node's name
    q_u = Categorical(logits=jnp.log(jnp.asarray([0.1, 0.1, 0.1, 0.6, 0.1])))
    obs = {"location_obs": jnp.asarray(0), "reward": jnp.asarray(0), "cue": jnp.asarray(0)}
    out, _ = s.infer_states(p, obs, priors={**beliefs, "u_location": q_u})
    assert jnp.array_equal(out["u_location"].probs, q_u.probs)


def test_a_belief_valued_policy_mixes_the_linear_leaves_and_not_the_nonlinear_ones():
    """Utility is linear in the control belief: the belief's leaf is the mixture of the
    clamped leaves under the product of the per-step beliefs. Risk and info gain are not."""
    from hybrid_aif.distributions import Categorical

    s, p = tmaze_system(policy_len=2)
    target = Categorical(logits=jnp.asarray([0.0, -1.0, 0.5, 0.2, -2.0]))
    p = dataclasses.replace(p, state_targets={"u_location": target})
    beliefs = s.initial_priors(p)
    q = jnp.asarray([[0.05, 0.05, 0.2, 0.6, 0.1], [0.1, 0.4, 0.4, 0.05, 0.05]])
    belief = s.score_policies(p, beliefs, {"u_location": Categorical(logits=jnp.log(q))})
    policies = s.policies(p)
    grid = s.score_policies(p, beliefs, policies)
    weight = q[0][policies[:, 0, 0]] * q[1][policies[:, 1, 0]]  # (25,), sums to 1

    def mixture(leaf):
        return float(jnp.sum(weight * leaf.lower))

    assert jnp.allclose(belief["utility"]["reward"].lower, mixture(grid["utility"]["reward"]))
    for term, block in (("risk", "u_location"), ("infogain", "cue")):
        assert not jnp.allclose(belief[term][block].lower, mixture(grid[term][block]))
    #  the node's own risk is KL(q(u_t) ‖ C) summed over the steps
    log_c = jax.nn.log_softmax(target.logits)
    kl = jnp.sum(q * (jnp.log(q) - log_c))
    assert jnp.allclose(belief["risk"]["u_location"].lower, kl)


def test_a_belief_on_a_hybrid_categorical_reader_is_a_mixture_and_one_hot_is_the_clamp():
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "examples" / "hybrid_switching.py"
    spec = importlib.util.spec_from_file_location("hybrid_switching", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    s, p = module.SYSTEM, module.PARAMS
    beliefs = s.initial_priors(p)
    policies = s.policies(p)
    clamped = s.score_policies(p, beliefs, policies)
    one_hot = s.score_policies(p, beliefs, {"u_mode": _probs_of(policies[..., 0], 3)})
    assert _leaves_equal(one_hot, clamped)


def test_a_belief_read_by_a_continuous_factor_is_refused():
    import importlib.util
    import pathlib

    from hybrid_aif.distributions import Categorical

    path = pathlib.Path(__file__).resolve().parents[1] / "examples" / "hill_lg.py"
    spec = importlib.util.spec_from_file_location("hill_lg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    s, p = module.make_agent()
    belief = Categorical(logits=jnp.zeros((s.policy_len, 3)))
    with pytest.raises(
        NotImplementedError, match=r"mode parent of continuous factor\(s\) \['state'\].*bounds"
    ):
        s.score_policies(p, s.initial_priors(p), {"u_state": belief})


def test_infer_policies_and_act_refuse_a_belief_valued_policy():
    from hybrid_aif.controls import ControlStep
    from hybrid_aif.distributions import Categorical

    s, p = tmaze_system(policy_len=2)
    beliefs = s.initial_priors(p)

    def mixing(ctx, use_bound="midpoint", metrics=True):
        probs = jnp.full((4, ctx.policy_len, 5), 0.2)
        return Categorical(logits=jnp.zeros(4)), None, ControlStep(entries=(probs,))

    for call in (
        lambda: s.infer_policies(p, beliefs, planner=mixing),
        lambda: s.act(jax.random.PRNGKey(0), p, beliefs, planner=mixing),
    ):
        with pytest.raises(ValueError, match="selects among clamped policies.*score_policies"):
            call()


def test_a_target_on_a_control_node_is_a_risk_leaf_keyed_by_its_name():
    from hybrid_aif.distributions import Categorical

    s, p = tmaze_system(policy_len=2)
    beliefs = s.initial_priors(p)
    policies = s.policies(p)
    target = Categorical(logits=jnp.asarray([0.0, -1.0, 0.5, 0.2, -2.0]))
    log_c = jax.nn.log_softmax(target.logits)
    plain = s.score_policies(p, beliefs, policies)
    with_target_p = dataclasses.replace(
        p, state_targets={"u_location": target}, terminal_targets={"u_location": target}
    )
    with_target = s.score_policies(with_target_p, beliefs, policies)
    #  under a clamp the leaf is −log C(u), summed over the steps; terminal: the last step
    want = -(log_c[policies[:, 0, 0]] + log_c[policies[:, 1, 0]])
    assert jnp.allclose(with_target["risk"]["u_location"].lower, want)
    assert jnp.allclose(with_target["terminal_risk"]["u_location"].lower, -log_c[policies[:, 1, 0]])
    #  toggled by the target's presence alone: every other leaf is untouched
    assert "risk" not in plain and "terminal_risk" not in plain
    rest = {t: b for t, b in with_target.items() if t not in ("risk", "terminal_risk")}
    assert _leaves_equal(rest, plain)
    #  the target reaches the planner: the expensive levels lose mass
    q_plain, _ = s.infer_policies(p, beliefs)
    q_with_target, _ = s.infer_policies(with_target_p, beliefs)
    assert not jnp.allclose(q_plain.probs, q_with_target.probs)
    #  names.controls is the set that puts a target on every node
    every = {n: target for n in s.names.controls}
    assert set(
        s.score_policies(dataclasses.replace(p, state_targets=every), beliefs, policies)["risk"]
    ) == set(s.names.controls)
    #  the sophisticated planner reads the same leaf
    soph = dataclasses.replace(s, planner="sophisticated")
    q_soph, _ = soph.infer_policies(p, beliefs)
    q_soph_with_target, _ = soph.infer_policies(with_target_p, beliefs)
    assert not jnp.allclose(q_soph.probs, q_soph_with_target.probs)


def test_a_target_on_a_node_is_a_categorical_over_its_levels():
    from hybrid_aif.distributions import Categorical

    s, p = tmaze_system(policy_len=2)
    beliefs = s.initial_priors(p)
    bad = dataclasses.replace(p, state_targets={"u_location": Categorical(logits=jnp.zeros(3))})
    with pytest.raises(ValueError, match=r"Categorical over its 5 level\(s\); got a Categorical"):
        s.infer_policies(bad, beliefs)


def test_epistemic_naming_a_control_node_is_refused():
    s, p = tmaze_system(policy_len=2)
    chosen = dataclasses.replace(s, epistemic=("u_location",))
    with pytest.raises(ValueError, match="'u_location' is a control node.*no uncertainty"):
        chosen.infer_policies(p, s.initial_priors(p))


def test_a_node_prior_in_params_is_refused_naming_the_path_dynamics_build():
    s, p = tmaze_system(policy_len=2)
    beliefs = s.initial_priors(p)
    for slot in ("transition", "initial_state_prior"):
        container = getattr(p, slot)
        bad = dataclasses.replace(p, **{slot: {**container, "u_location": container["location"]}})
        with pytest.raises(NotImplementedError, match="path-dynamics build"):
            s.infer_policies(bad, beliefs)


def test_a_target_on_a_control_node_is_written_under_the_node():
    from hybrid_aif.description import build, describe
    from hybrid_aif.distributions import Categorical

    s, p = tmaze_system(policy_len=2)
    with_target = dataclasses.replace(p, state_targets={"u_location": Categorical(jnp.zeros(5))})
    document, arrays = describe(s, with_target)
    assert document["factors"]["u_location"]["target"] == {
        "family": "Categorical",
        "parameters": {"logits": "C_u_location"},
    }
    _, rebuilt = build(document, arrays=arrays)
    assert set(rebuilt.state_targets) == {"u_location"}


def test_a_hard_zero_target_scores_a_one_hot_belief_as_its_clamp():
    """A level the target forbids (C(u) = 0) and the belief never takes contributes nothing:
    the one-hot belief of a legal plan equals the clamp bitwise, and any belief with no mass on
    the forbidden levels is finite."""
    from hybrid_aif.distributions import Categorical

    s, p = tmaze_system(policy_len=2)
    target = Categorical(logits=jnp.log(jnp.asarray([0.4, 0.3, 0.0, 0.3, 0.0])))
    p = dataclasses.replace(p, state_targets={"u_location": target})
    beliefs = s.initial_priors(p)
    plan = jnp.asarray([3, 1], jnp.int32)
    clamped = s.score_policies(p, beliefs, {"u_location": plan})
    one_hot = s.score_policies(p, beliefs, {"u_location": _probs_of(plan, 5)})
    assert _leaves_equal(one_hot, clamped)
    assert bool(jnp.isfinite(clamped["risk"]["u_location"].lower))
    q = jnp.asarray([[0.5, 0.2, 0.0, 0.3, 0.0], [0.1, 0.6, 0.0, 0.3, 0.0]])
    mixed = s.score_policies(p, beliefs, {"u_location": Categorical(logits=jnp.log(q))})
    assert bool(jnp.isfinite(mixed["risk"]["u_location"].lower))


def test_control_risk_wraps_closures_that_take_more_arguments():
    """The amortized scoring closures take the previous beliefs after ``t``; the target wrapper
    passes them through and adds only its own leaves."""
    from hybrid_aif.controls import ControlTarget, control_risk
    from hybrid_aif.interval import Interval

    seen = []

    def inner(qs, action, t, prev=None, dist=None):
        seen.append((prev, dist))
        return {"utility": {"r": Interval.exact(jnp.asarray(1.0))}}

    log_c = jax.nn.log_softmax(jnp.asarray([0.0, -1.0, 0.5]))
    wrapped = control_risk(inner, (ControlTarget(0, "u_x", log_c, False),), horizon=2)
    tree = wrapped((), jnp.asarray([2], jnp.int32), 0, "prev", "dist")
    assert seen == [("prev", "dist")]
    assert float(tree["utility"]["r"].lower) == 1.0
    assert jnp.allclose(tree["risk"]["u_x"].lower, -log_c[2])
