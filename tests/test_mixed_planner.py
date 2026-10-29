"""The ``"mixed"`` planner: the discrete policies enumerated, the continuous controls of each
descended.

The planner's descents are the gradient planner's own: each discrete policy's optimised ``G`` is
what :func:`~hybrid_aif.planning.gradient_plan` returns on that policy's continuous subproblem,
the posterior is the enumerating planner's softmax over those scores, and every score is that of
a clamped plan (``System.score_policies``).

Design: docs/design/mixed-control-planning.md M1-M3.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.controls import FLOAT, LEVEL, ControlLayout, ControlStep
from hybrid_aif.objectives import total
from hybrid_aif.planning import (
    enumerate_policies,
    gradient_plan,
    make_mixed_planner,
    mixed_plan,
    policy_posterior,
)
from hybrid_aif.system import System, SystemParams
from tests.test_action_node_modalities import maze_with_cue_by_node, payoff_model_by_node
from tests.test_conditional_targets import C_ASK
from tests.test_continuous_nodes import G2, F, _box, _gauss, _lg
from tests.test_lq_control import lq

STEPS = 60


def payoff(horizon: int, **over):
    system, params = payoff_model_by_node(C_ASK, horizon)
    return dataclasses.replace(system, planner="mixed", **over), params


def toy(horizon: int = 1):
    """A state ``x`` (2-d) whose three modes (a discrete node) each read a continuous input
    through their own ``G``; the target is ``(1, 0)``, so the best input differs by mode."""
    system = System(
        B_deps=((1, 2), (), ()),
        controls=(1, 2),
        action_names=("d", "c"),
        factor_names=("x", "u_d", "u_c"),
        policy_len=horizon,
        epistemic=False,
        planner="mixed",
    )
    offsets = jnp.asarray([[0.0, 0.0], [1.0, -1.0], [0.0, 2.0]])
    Fs = jnp.stack([F, 0.5 * F, jnp.eye(2)])
    Gs = jnp.stack([G2, 2.0 * G2, -G2])
    params = SystemParams(
        transition={"x": _lg(offsets, G=Gs, F=Fs)},
        initial_state_prior={"x": _gauss()},
        state_targets={"x": _gauss((1.0, 0.0), 1.0)},
        control={"u_c": _box()},
    )
    return system, params


def planned(system, params):
    return system.planner_context(params, system.initial_priors(params))


def fix_levels(ctx, levels):
    """``ctx`` as a continuous-only problem with the discrete levels ``(H, K_discrete)`` fixed:
    the step index rides in the carry, so each step reads its own row of ``levels``."""
    boxes = [ctx.boxes[k] for k, kind in enumerate(ctx.kinds) if kind == FLOAT]
    levels = jnp.asarray(levels, jnp.int32)

    def step_of(t, action):
        entries, d_at, c_at = [], 0, 0
        for k, kind in enumerate(ctx.kinds):
            if kind == LEVEL:
                entries.append(levels[t, d_at])
                d_at += 1
            else:
                entries.append(action[c_at : c_at + ctx.sizes[k]])
                c_at += ctx.sizes[k]
        return ControlStep(entries=tuple(entries))

    def transition(carry, action):
        qs, t = carry
        return ctx.transition_fn(qs, step_of(t, action)), t + 1

    def step_terms(carry, action, t):
        return ctx.step_terms_fn(carry[0], step_of(t, action), t)

    terminal = (
        None if ctx.terminal_terms_fn is None else lambda carry: ctx.terminal_terms_fn(carry[0])
    )
    low = jnp.concatenate([jnp.asarray(b.low, jnp.float32) for b in boxes])
    high = jnp.concatenate([jnp.asarray(b.high, jnp.float32) for b in boxes])
    layout = ControlLayout(
        node_spans=((0, low.shape[0]),),
        factor_nodes=(),
        action_dim=low.shape[0],
        low=low,
        high=high,
    )
    return ctx._replace(
        transition_fn=transition,
        step_terms_fn=step_terms,
        terminal_terms_fn=terminal,
        init_beliefs=(ctx.init_beliefs, jnp.int32(0)),
        continuous_controls=layout,
        kinds=(FLOAT,),
        sizes=(low.shape[0],),
        boxes=tuple(boxes),
    )


# ------------------------------------------------------------------------------ the equalities


def test_each_discrete_policys_descent_is_the_gradient_planners_on_its_subproblem():
    """The optimised plan, ``G`` and traces of every discrete policy equal (to rounding) those of
    ``gradient_plan`` run by hand on that policy's continuous subproblem, and the returned
    iterate is the same, on the payoff model (whose steer has its optimum at the warm start) and
    on the toy (whose optima move)."""
    for system, params, count in (payoff(3) + (27,), toy(2) + (9,)):
        ctx = planned(system, params)
        result = mixed_plan(ctx, steps=STEPS)
        assert result.discrete.shape[0] == count and result.plan.shape[0] == count
        assert result.G.lower.shape == (count,) and result.trace.shape == (count, STEPS)
        assert result.gradient_trace.shape == (count, STEPS)
        assert result.best_step.shape == (count,)
        for p in range(count):
            by_hand = gradient_plan(fix_levels(ctx, result.discrete[p]), steps=STEPS)
            for got, want in (
                (result.plan[p], by_hand.plan),
                (result.G.lower[p], by_hand.G.lower[0]),
                (result.G.upper[p], by_hand.G.upper[0]),
                (result.trace[p], by_hand.trace),
                (result.gradient_trace[p], by_hand.gradient_trace),
            ):
                np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-6)
            assert int(result.best_step[p]) == int(by_hand.best_step)


def test_the_posterior_is_the_softmax_of_the_scores_and_the_selected_plan_is_scored_clamped():
    """``q_pi`` is ``policy_posterior`` of the optimised ``G``; the selected plan is the argmin;
    and ``G`` of the returned plan is what ``score_policies`` gives that clamped plan."""
    system, params = payoff(3)
    beliefs = system.initial_priors(params)
    ctx = planned(system, params)
    result = mixed_plan(ctx, steps=STEPS)
    want = policy_posterior(result.G, ctx.gamma)
    np.testing.assert_array_equal(np.asarray(result.q_pi.logits), np.asarray(want.logits))
    best = int(jnp.argmin(result.G.midpoint))
    assert int(jnp.argmax(result.q_pi.probs)) == best
    action, _, _ = system.act(jax.random.PRNGKey(0), params, beliefs, selection="deterministic")
    assert int(action["option"]) == int(result.discrete[best, 0, 0])
    np.testing.assert_array_equal(np.asarray(action["steer"]), np.asarray(result.plan[best, 0]))
    scored = system.score_policies(
        params,
        beliefs,
        {"u_option": result.discrete[best, :, 0], "u_steer": result.plan[best]},
    )
    np.testing.assert_allclose(
        np.asarray(total(scored).lower), np.asarray(result.G.lower[best]), rtol=1e-5
    )
    toy_system, toy_params = toy(1)
    moved = mixed_plan(planned(toy_system, toy_params))
    pick = int(jnp.argmin(moved.G.midpoint))
    toy_scored = toy_system.score_policies(
        toy_params,
        toy_system.initial_priors(toy_params),
        {"u_d": moved.discrete[pick, :, 0], "u_c": moved.plan[pick]},
    )
    np.testing.assert_allclose(
        np.asarray(total(toy_scored).lower), np.asarray(moved.G.lower[pick]), rtol=1e-6
    )


def test_a_supplied_discrete_set_is_the_full_enumeration_restricted_to_it():
    """The first step enumerated, the later steps held at DRIVE: the three plans and their
    posterior are those of the full enumeration's three rows, the prior restricted to them."""
    system, params = payoff(3)
    ctx = planned(system, params)
    full = enumerate_policies((3,), 3, (True,))
    held = jnp.asarray([[[a], [0], [0]] for a in range(3)], jnp.int32)
    rows = [int(jnp.argmax(jnp.all(full == h[None], axis=(1, 2)))) for h in held]
    prior = jnp.linspace(0.0, 2.0, 27)
    with_e = dataclasses.replace(params, policy_prior=prior)
    ctx_e = planned(system, with_e)
    everything = mixed_plan(ctx_e, steps=STEPS)
    some = mixed_plan(ctx_e, discrete_policies=held, steps=STEPS)
    np.testing.assert_array_equal(np.asarray(some.discrete), np.asarray(held))
    np.testing.assert_allclose(
        np.asarray(some.plan), np.asarray(everything.plan[jnp.asarray(rows)]), atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(some.G.lower), np.asarray(everything.G.lower[jnp.asarray(rows)]), rtol=1e-5
    )
    want = policy_posterior(some.G, ctx.gamma, prior[jnp.asarray(rows)])
    np.testing.assert_allclose(np.asarray(some.q_pi.probs), np.asarray(want.probs), rtol=1e-6)
    #  the same set through the registered planner's face, as a policy set
    face = make_mixed_planner(discrete_policies=held, steps=STEPS)
    q_pi, tree, policies = face(ctx_e)
    assert policies.entries[0].shape == (3, 3) and policies.entries[1].shape == (3, 3, 1)
    assert tree["risk"]["lane"].lower.shape == (3,)
    np.testing.assert_array_equal(np.asarray(q_pi.logits), np.asarray(some.q_pi.logits))
    #  without a prior the posterior carries none
    bare = mixed_plan(ctx, discrete_policies=held, steps=STEPS)
    np.testing.assert_array_equal(
        np.asarray(bare.q_pi.logits), np.asarray(policy_posterior(bare.G, ctx.gamma).logits)
    )


def test_legal_masks_a_discrete_policy():
    """A level the mask forbids gets exactly zero mass, as under the enumerating planner."""
    system, params = payoff(2)
    beliefs = system.initial_priors(params)
    q_all, _ = system.infer_policies(params, beliefs)
    legal = {"u_option": jnp.asarray([True, False, True])}
    q_pi, _ = system.infer_policies(params, beliefs, legal=legal)
    first = np.asarray(enumerate_policies((3,), 2, (True,))[:, 0, 0])
    probs = np.asarray(q_pi.probs)
    assert probs.shape == (9,) and np.all(probs[first == 1] == 0.0)
    assert probs.sum() == pytest.approx(1.0, rel=1e-5)
    logits_all, logits = np.asarray(q_all.log_probs), np.asarray(q_pi.log_probs)
    shift = logits[first != 1] - logits_all[first != 1]
    np.testing.assert_allclose(shift, shift[0], atol=1e-4)  # the mask only renormalises
    assert np.all(np.isneginf(logits[first == 1]))
    deterministic, _, _ = system.act(
        jax.random.PRNGKey(1), params, beliefs, selection="deterministic", legal=legal
    )
    assert int(deterministic["option"]) != 1


# ------------------------------------------------------------------------------- the refusals


def test_the_refusals_name_the_planner_to_use():
    discrete, discrete_params = maze_with_cue_by_node()
    with pytest.raises(ValueError, match='all discrete.*"enumeration" planner'):
        dataclasses.replace(discrete, planner="mixed").infer_policies(
            discrete_params, discrete.initial_priors(discrete_params)
        )
    continuous, continuous_params = lq.make_lq()
    with pytest.raises(ValueError, match='all continuous.*"gradient" planner'):
        dataclasses.replace(continuous, planner="mixed").infer_policies(
            continuous_params, continuous.initial_priors(continuous_params)
        )
    system, params = toy()
    beliefs = system.initial_priors(params)
    for planner in ("gradient", "cem"):
        with pytest.raises(NotImplementedError, match='plan it with the "mixed" planner'):
            dataclasses.replace(system, planner=planner).infer_policies(
                params, beliefs, plan_key=jax.random.PRNGKey(0)
            )
    with pytest.raises(NotImplementedError, match="does not search their product"):
        dataclasses.replace(system, planner="enumeration").infer_policies(params, beliefs)
    ctx = planned(system, params)
    with pytest.raises(ValueError, match="discrete_policies has shape"):
        mixed_plan(ctx, discrete_policies=jnp.zeros((2, 3, 1), jnp.int32))
    with pytest.raises(ValueError, match="outside the nodes' levels"):
        mixed_plan(ctx, discrete_policies=jnp.asarray([[[3]]], jnp.int32))
    with pytest.raises(ValueError, match="plain gradient step"):
        mixed_plan(ctx, optimizer="gradient")
    with pytest.raises(ValueError, match="holds 0 sizes and 0 boxes for 2 control node"):
        mixed_plan(ctx._replace(sizes=(), boxes=()))
    #  a policy module as E: the sentence names this planner
    module = dataclasses.replace(params, policy_prior=lambda beliefs, key: None)
    with pytest.raises(NotImplementedError, match="The mixed planner does not accept one"):
        mixed_plan(planned(system, module), steps=1)
    #  legal= naming a continuous node has no level set to mask
    with pytest.raises(ValueError, match=r"names the continuous control node\(s\) \['u_c'\]"):
        system.infer_policies(params, beliefs, legal={"u_c": jnp.asarray([True])})


# --------------------------------------------------------------------------------------- jit


def test_infer_policies_and_act_run_under_jit():
    system, params = toy(2)
    beliefs = system.initial_priors(params)
    q_eager, tree_eager = system.infer_policies(params, beliefs)
    q_jit, tree_jit = jax.jit(system.infer_policies)(params, beliefs)
    np.testing.assert_allclose(np.asarray(q_jit.probs), np.asarray(q_eager.probs), rtol=1e-5)
    assert tree_jit["risk"]["x"].lower.shape == (9,)
    act = jax.jit(lambda k, p, b: system.act(k, p, b)[0])
    for key in range(3):
        action = act(jax.random.PRNGKey(key), params, beliefs)
        assert set(action) == {"d", "c"} and action["d"].dtype == jnp.int32
        assert action["c"].shape == (1,) and bool(jnp.all(jnp.abs(action["c"]) <= 2.0))
    with pytest.raises(ValueError, match='select mode="policy"'):
        system.act(jax.random.PRNGKey(0), params, beliefs, mode="marginal")


# ------------------------------------------------------------------------- the optima differ


def test_the_planner_finds_the_best_continuous_control_of_each_discrete_policy():
    """One step, three modes whose inputs push the state in different ways: the optimum of ``G``
    over the input is read off a grid for each mode, and the planner's plan is within the grid's
    spacing of it, with the modes' optima distinct."""
    system, params = toy(1)
    beliefs = system.initial_priors(params)
    result = mixed_plan(planned(system, params))
    grid = jnp.linspace(-2.0, 2.0, 801)
    found = []
    for mode in range(3):
        scored = system.score_policies(
            params,
            beliefs,
            {
                "u_d": jnp.full((801, 1), mode, jnp.int32),
                "u_c": grid[:, None, None],
            },
        )
        g = np.asarray(total(scored).lower)
        best = float(grid[int(np.argmin(g))])
        got = float(result.plan[mode, 0, 0])
        assert got == pytest.approx(best, abs=0.02), (mode, got, best)
        assert float(result.G.lower[mode]) <= g.min() + 1e-4
        found.append(got)
    assert len({round(u, 1) for u in found}) == 3, found
    assert result.discrete[:, 0, 0].tolist() == [0, 1, 2]
    assert max(found) - min(found) > 0.5, found


def test_every_policy_starts_from_the_same_warm_start_and_the_traces_are_returned():
    system, params = toy(2)
    ctx = planned(system, params)
    result = mixed_plan(ctx, steps=20, init=jnp.asarray([0.5]))
    assert result.trace.shape == (9, 20) and result.gradient_trace.shape == (9, 20)
    assert int(jnp.max(result.best_step)) <= 20
    #  the shared warm start: a descent of zero steps returns it for every policy
    held = mixed_plan(ctx, steps=0, init=jnp.asarray([0.5]))
    np.testing.assert_array_equal(np.asarray(held.plan), np.full((9, 2, 1), 0.5, np.float32))
    centre = mixed_plan(ctx, steps=0)
    np.testing.assert_array_equal(np.asarray(centre.plan), np.zeros((9, 2, 1), np.float32))
