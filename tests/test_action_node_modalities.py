"""A modality over an action node, read by the planning faces and the hybrid inference faces.

A modality whose parent is an action node reads what the agent did, not what it intended. At a plan
step the node's belief is ``P(a | u_t)``; a tabular channel (or a Gaussian emission) contracts over
it, and a linear-Gaussian channel over a Gaussian-carried factor selects its head and noise at the
level, the node being a point. These tests pin the selection against the same model written with an
indicator dimension of the state (the payoff toy), the tabular channel against a cue location held
as a state factor (the T-maze), the efference copy at inference, the soft-step refusal, the
epistemic chain and the closed loop, and the faces that keep their refusals.

Design: docs/design/action-node-modalities.md S1-S6.
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
from hybrid_aif.controls import ControlStep
from hybrid_aif.objectives import total
from hybrid_aif.planning.enumerate import evaluate_policies
from tests.test_conditional_targets import (
    ASK,
    C_ASK,
    DRIVE,
    GPS_SD,
    IDLE,
    LANE_PRECISION,
    OFFSET_SD,
    POSITION_SD,
    ROW,
    ROW_SD,
    WAIT,
    payoff_model,
)

KEY = jax.random.PRNGKey(0)
ANSWER_SD = 0.02
MUTE_SD = 100.0


# ---------------------------------------------------------------- the payoff toy, declared


def payoff_model_by_node(c_ask: float, horizon: int):
    """The payoff toy of ``test_conditional_targets`` with the answer channel a linear-Gaussian
    observation over ``(lane, a_option)``: the lane factor is ``(e, b)`` with no indicator, and
    the answer's noise is wide under DRIVE and IDLE and the answer's sd under ASK."""
    system = System(
        A_deps=((0,), (0, 3)),
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
    hold = jnp.eye(2)
    tiny = 1e-3

    def chol(e_sd, b_sd):
        return jnp.diag(jnp.asarray([e_sd, b_sd]))

    lane_sd = lambda duration: (duration * LANE_PRECISION) ** -0.5  # noqa: E731
    prior_cov = jnp.asarray(
        [
            [POSITION_SD**2 + OFFSET_SD**2, -(OFFSET_SD**2)],
            [-(OFFSET_SD**2), OFFSET_SD**2],
        ]
    )
    sds = jnp.asarray([MUTE_SD, ANSWER_SD, MUTE_SD]).reshape(3, 1, 1)
    params = SystemParams(
        transition={
            "lane": LinearGaussianTransition(
                F=hold,
                G=jnp.stack([jnp.asarray([[1.0], [0.0]]), jnp.zeros((2, 1)), jnp.zeros((2, 1))]),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((3, 2)),
                    chol=jnp.stack([chol(ROW_SD, tiny), chol(tiny, tiny), chol(tiny, tiny)]),
                    parent_ndims=1,
                ),
            )
        },
        initial_state_prior={"lane": Gaussian.from_cov(jnp.zeros(2), prior_cov)},
        likelihood={
            "gps": LinearGaussianObservation(
                H=jnp.asarray([[1.0, 1.0]]), c=jnp.zeros(1), noise=GPS_SD * jnp.eye(1)
            ),
            "answer": LinearGaussianObservation(
                H=jnp.asarray([[0.0, 1.0]]), c=jnp.zeros(1), noise=sds
            ),
        },
        state_targets={
            "lane": ConditionalGaussian(
                loc=jnp.zeros((3, 2)),
                chol=jnp.stack(
                    [chol(lane_sd(ROW), 1e3), chol(lane_sd(WAIT), 1e3), chol(lane_sd(WAIT), 1e3)]
                ),
                parent_ndims=1,
            ),
            "u_option": Categorical(logits=jnp.asarray([0.0, -c_ask, 0.0])),
        },
        control={"u_steer": Box(-5.0 * jnp.ones(1), 5.0 * jnp.ones(1))},
    )
    return system, params


def _plan_totals(system, params, plans) -> np.ndarray:
    beliefs = system.initial_priors(params)
    ctx = system.planner_context(params, beliefs)
    levels = jnp.asarray(np.array(plans), jnp.int32)
    steps = ControlStep(entries=(levels, jnp.zeros((levels.shape[0], levels.shape[1], 1))))
    tree = evaluate_policies(
        steps, ctx.init_beliefs, ctx.transition_fn, ctx.step_terms_fn, ctx.terminal_terms_fn
    )
    return np.asarray(total(tree).lower)


def _two_plans(rows: int):
    return [(IDLE, *((DRIVE,) * rows)), (ASK, *((DRIVE,) * rows))]


def _crossing(model, c_ask: float, max_rows: int = 14) -> int:
    for rows in range(1, max_rows + 1):
        system, params = model(c_ask, rows + 1)
        never, ask_first = _plan_totals(system, params, _two_plans(rows))
        if ask_first < never:
            return rows
    return max_rows + 1


#: What the indicator form's third state dimension adds to every step's cross-entropy against a
#: target of sd 1e3, whatever the plan: ``½ log 2π + log 1e3``.
INDICATOR_NORMALISER = 0.5 * np.log(2.0 * np.pi) + np.log(1e3)


def test_the_payoff_toy_declared_over_the_action_node_crosses_where_the_indicator_form_does():
    """The answer as a channel over ``(lane, a_option)`` reads at the step the ask is taken, as
    the indicator form of ``test_conditional_targets`` does (3 rows at an ask cost of 20 nats and
    7 at 45). The two score every plan alike once the indicator dimension's per-step target
    normaliser, the same for every plan, is added back."""
    crossings, original = {}, {}
    worst = 0.0
    for c_ask in (C_ASK, 2.25 * C_ASK):
        crossings[c_ask] = _crossing(payoff_model_by_node, c_ask)
        original[c_ask] = _crossing(payoff_model, c_ask)
        for rows in range(1, 9):
            plans = [*_two_plans(rows), (ASK, IDLE, *((DRIVE,) * (rows - 1)))]
            declared = _plan_totals(*payoff_model_by_node(c_ask, rows + 1), plans)
            indicator = _plan_totals(*payoff_model(c_ask, rows + 1), plans)
            worst = max(
                worst,
                float(np.max(np.abs(declared + (rows + 1) * INDICATOR_NORMALISER - indicator))),
            )
    assert crossings == original == {C_ASK: 3, 2.25 * C_ASK: 7}
    assert worst < 1e-4, worst


# ------------------------------------------------------------- a cue read only by looking

LOOK, LEFT, RIGHT = 0, 1, 2
CENTRE = 0
VALIDITY = 0.9


def _cue_rows() -> jnp.ndarray:
    """``P(cue | context)`` while looking: the cue shows the context with ``VALIDITY``."""
    return jnp.asarray([[VALIDITY, 1.0 - VALIDITY], [1.0 - VALIDITY, VALIDITY]])


def _reward_table() -> jnp.ndarray:
    """``P(reward | context, arm)`` child-first ``(reward, context, arm)``: nothing at the
    centre; at an arm, a win if the context puts the reward there and a loss otherwise."""
    table = np.zeros((3, 2, 3))
    table[0, :, CENTRE] = 1.0
    for context, arm in ((0, LEFT), (1, RIGHT)):
        table[1, context, arm] = 0.95
        table[2, context, arm] = 0.05
        table[1, 1 - context, arm] = 0.05
        table[2, 1 - context, arm] = 0.95
    return jnp.asarray(table)


def _arm_table() -> jnp.ndarray:
    """``arm' = `` the arm the level goes to (LOOK stays at the centre), child-first
    ``(arm', arm, u)``."""
    goes = {LOOK: CENTRE, LEFT: LEFT, RIGHT: RIGHT}
    table = np.zeros((3, 3, 3))
    for u, arm in goes.items():
        table[arm, :, u] = 1.0
    return jnp.asarray(table)


def _cc(table, parent_ndims: int) -> ConditionalCategorical:
    return ConditionalCategorical(log_table=jnp.log(table), parent_ndims=parent_ndims)


def maze_with_cue_by_node():
    """A T-maze whose cue is read only under the LOOK level of the control node: the cue
    modality is over ``(context, a_act)``, its rows the informative ones under LOOK and uniform
    under the other levels."""
    system = System(
        A_deps=((0, 1), (0, 3)),
        B_deps=((0,), (1, 2), ()),
        controls=(2,),
        action_names=("act",),
        factor_names=("context", "arm", "u_act"),
        modality_names=("reward", "cue"),
        policy_len=2,
        epistemic=True,
        gamma=16.0,
    )
    cue = jnp.stack([_cue_rows(), jnp.full((2, 2), 0.5), jnp.full((2, 2), 0.5)], axis=-1)
    return system, _maze_params(cue, parent_ndims=2)


def maze_with_cue_by_state():
    """The same model with the cue location a state factor ``looking`` the LOOK level moves to:
    the cue modality is over ``(context, looking)``."""
    system = System(
        A_deps=((0, 1), (0, 2)),
        B_deps=((0,), (1, 3), (3,), ()),
        controls=(3,),
        action_names=("act",),
        factor_names=("context", "arm", "looking", "u_act"),
        modality_names=("reward", "cue"),
        policy_len=2,
        epistemic=True,
        gamma=16.0,
    )
    cue = jnp.stack([jnp.full((2, 2), 0.5), _cue_rows()], axis=-1)
    params = _maze_params(cue, parent_ndims=2)
    looks = jnp.asarray([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    transition = {
        "context": params.transition["context"],
        "arm": params.transition["arm"],
        "looking": _cc(looks, 1),
    }
    initial = {
        **params.initial_state_prior,
        "looking": Categorical(logits=jnp.asarray([0.0, -30.0])),
    }
    return system, dataclasses.replace(params, transition=transition, initial_state_prior=initial)


def _maze_params(cue: jnp.ndarray, parent_ndims: int) -> SystemParams:
    return SystemParams(
        likelihood={"reward": _cc(_reward_table(), 2), "cue": _cc(cue, parent_ndims)},
        transition={"context": _cc(jnp.eye(2), 1), "arm": _cc(_arm_table(), 2)},
        initial_state_prior={
            "context": Categorical(logits=jnp.zeros(2)),
            "arm": Categorical(logits=jnp.asarray([0.0, -jnp.inf, -jnp.inf])),
        },
        obs_targets={"reward": Categorical(logits=jnp.asarray([0.0, 4.0, -6.0]))},
    )


def test_a_cue_read_only_by_looking_is_looked_for_first_and_scores_as_a_cue_location_does():
    """The modality over ``(context, a_act)`` is the tabular channel contracted over the node's
    one-hot at each step: the enumerating planner looks first (the cue resolves the reward arm),
    and every plan's leaves equal those of the model that holds the cue location as a state
    factor the LOOK level moves to, bitwise."""
    by_node, node_params = maze_with_cue_by_node()
    by_state, state_params = maze_with_cue_by_state()
    node_beliefs = by_node.initial_priors(node_params)
    state_beliefs = by_state.initial_priors(state_params)
    plans = by_node.policies(node_params)
    assert jnp.array_equal(plans, by_state.policies(state_params))
    node_tree = by_node.score_policies(node_params, node_beliefs, plans)
    state_tree = by_state.score_policies(state_params, state_beliefs, plans)
    assert set(node_tree) == set(state_tree)
    for term, leaves in node_tree.items():
        assert set(leaves) == set(state_tree[term]), term
        for key, iv in leaves.items():
            assert np.array_equal(np.asarray(iv.lower), np.asarray(state_tree[term][key].lower)), (
                term,
                key,
            )
    q_pi, _ = by_node.infer_policies(node_params, node_beliefs)
    best = plans[int(jnp.argmax(q_pi.probs))]
    assert int(best[0, 0]) == LOOK


# -------------------------------------------------- a linear-Gaussian channel that selects

LEVEL_SD = (0.5, 0.05, 0.5)  # the reading's noise sd under each level of the node
DRIFT = (0.0, 0.1, -0.2)  # the level's drift of the state
PROCESS_SD = 0.1
PRIOR_SD = 0.4
PRIOR_MEAN = 0.3


def scalar_model(*, horizon: int = 3, kernel=None, feedback=None, belief="posterior"):
    """A scalar state ``x`` that a three-level control node ``u`` drifts, read through a channel
    over ``(x, a)`` whose noise sd is ``LEVEL_SD[level]``: the node's action selects the reading's
    noise, with no mixture declared."""
    system = System(
        A_deps=((0, 2),),
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("a",),
        factor_names=("x", "u"),
        modality_names=("y",),
        policy_len=horizon,
        planning_belief=belief,
        feedback=feedback,
        epistemic=True,
    )
    params = SystemParams(
        transition={
            "x": LinearGaussianTransition(
                F=jnp.eye(1),
                innovations=ConditionalGaussian(
                    loc=jnp.asarray(DRIFT).reshape(3, 1),
                    chol=jnp.full((3, 1, 1), PROCESS_SD),
                    parent_ndims=1,
                ),
            )
        },
        initial_state_prior={
            "x": Gaussian(loc=jnp.asarray([PRIOR_MEAN]), chol=jnp.full((1, 1), PRIOR_SD))
        },
        likelihood={
            "y": LinearGaussianObservation(
                H=jnp.eye(1),
                c=jnp.zeros(1),
                noise=jnp.asarray(LEVEL_SD).reshape(3, 1, 1),
            )
        },
        actuator={} if kernel is None else {"a": kernel},
    )
    return system, params


def _kalman(m, P, y, sd):
    """One scalar Kalman update of ``N(m, P)`` on a reading ``y`` of noise sd ``sd``."""
    gain = P / (P + sd**2)
    return m + gain * (y - m), (1.0 - gain) * P


def test_the_efference_copy_selects_the_readings_noise_at_inference():
    """``infer_states`` with the one-hot prior of the action node updates the state as a hand
    Kalman filter with that level's noise does, and returns the node's belief unchanged."""
    system, params = scalar_model()
    beliefs = system.initial_priors(params)
    y = jnp.asarray([0.55])
    for level in range(3):
        copy = Categorical.from_probs(jax.nn.one_hot(level, 3))
        posterior, _ = system.infer_states(params, {"y": y}, {**beliefs, "a": copy})
        mean, var = _kalman(PRIOR_MEAN, PRIOR_SD**2, 0.55, LEVEL_SD[level])
        np.testing.assert_allclose(np.asarray(posterior["x"].loc), [mean], rtol=1e-5)
        np.testing.assert_allclose(np.asarray(posterior["x"].cov), [[var]], rtol=1e-5)
        assert jnp.array_equal(posterior["a"].probs, copy.probs)
    #  under jit the prior is traced and the same level is selected
    run = jax.jit(
        lambda probs: (
            system.infer_states(params, {"y": y}, {**beliefs, "a": Categorical.from_probs(probs)})[
                0
            ]["x"].loc
        )
    )
    np.testing.assert_allclose(
        np.asarray(run(jax.nn.one_hot(1, 3))),
        [_kalman(PRIOR_MEAN, PRIOR_SD**2, 0.55, LEVEL_SD[1])[0]],
        rtol=1e-5,
    )


def test_a_prior_over_the_action_node_that_is_not_a_point_is_refused_at_inference():
    system, params = scalar_model()
    beliefs = system.initial_priors(params)
    y = {"y": jnp.asarray([0.55])}
    spread = Categorical.from_probs(jnp.asarray([0.2, 0.8, 0.0]))
    pattern = r"selects its head and noise by action node 'a'.*is a point.*MixtureGaussian"
    with pytest.raises(ValueError, match=pattern):
        system.infer_states(params, y, {**beliefs, "a": spread})
    with pytest.raises(ValueError, match=pattern):  # no prior given: uniform
        system.infer_states(params, y, beliefs)
    run = jax.jit(
        lambda probs: (
            system.infer_states(params, y, {**beliefs, "a": Categorical.from_probs(probs)})[0][
                "x"
            ].loc
        )
    )
    with pytest.raises(Exception, match="selects its head and noise by action node 'a'"):
        run(jnp.asarray([0.2, 0.8, 0.0]))


def _hand_chain(levels, belief="posterior"):
    """The information gain ``½ log(1 + P / R²)`` of each step's reading under a hand Kalman
    chain: predict through the process noise, gain from the predicted variance, then update."""
    P, gains = PRIOR_SD**2, []
    for level in levels:
        P_pred = P + PROCESS_SD**2
        gains.append(0.5 * np.log(1.0 + P_pred / LEVEL_SD[level] ** 2))
        if belief == "posterior":
            P = P_pred * LEVEL_SD[level] ** 2 / (P_pred + LEVEL_SD[level] ** 2)
        else:
            P = P_pred
    return gains


def _infogain_by_step(system, params, levels):
    """The ``infogain`` leaf of ``y`` at each step of a plan, from one-step prefixes' gaps."""
    beliefs = system.initial_priors(params)
    ctx = system.planner_context(params, beliefs)
    plan = jnp.asarray(levels, jnp.int32).reshape(1, len(levels), 1)
    carry, out = ctx.init_beliefs, []
    for t in range(len(levels)):
        carry = ctx.transition_fn(carry, plan[0, t])
        tree = ctx.step_terms_fn(carry, plan[0, t], jnp.asarray(t))
        out.append(float(tree["infogain"]["y"].lower))
    return out


@pytest.mark.parametrize("levels", [(0, 0, 0), (1, 1, 1), (0, 1, 2), (2, 1, 0), (1, 0, 1)])
def test_the_information_gain_telescopes_as_a_kalman_filter_with_the_per_step_noise(levels):
    """Under ``planning_belief="posterior"`` the selecting channel conditions the epistemic chain
    at the step's level, so each step's gain reads the variance the earlier steps' readings left;
    under ``"prior"`` it reads the predicted variance alone."""
    system, params = scalar_model(belief="posterior")
    np.testing.assert_allclose(
        _infogain_by_step(system, params, levels), _hand_chain(levels), rtol=2e-5
    )
    prior, prior_params = scalar_model(belief="prior")
    np.testing.assert_allclose(
        _infogain_by_step(prior, prior_params, levels), _hand_chain(levels, "prior"), rtol=2e-5
    )


def soft_capable_model(kernel=None):
    """A selecting channel beside a tabular factor ``s`` the node ``u`` moves, which a soft step
    can pass through: the scalar state ``x`` does not read ``u``, only the channel does."""
    from tests.test_action_nodes import _roll

    system = System(
        A_deps=((0, 3),),
        B_deps=((), (1, 2), ()),
        controls=(2,),
        action_names=("a",),
        factor_names=("x", "s", "u"),
        modality_names=("y",),
        policy_len=1,
        planning_belief="posterior",
        epistemic=True,
    )
    params = SystemParams(
        transition={
            "x": LinearGaussianTransition(
                F=jnp.eye(1),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 1)), chol=jnp.full((1, 1, 1), PROCESS_SD), parent_ndims=1
                ),
            ),
            "s": ConditionalCategorical(log_table=jnp.log(_roll(0.05)), parent_ndims=2),
        },
        initial_state_prior={
            "x": Gaussian(loc=jnp.asarray([PRIOR_MEAN]), chol=jnp.full((1, 1), PRIOR_SD)),
            "s": Categorical(logits=jnp.zeros(3)),
        },
        likelihood={
            "y": LinearGaussianObservation(
                H=jnp.eye(1), c=jnp.zeros(1), noise=jnp.asarray(LEVEL_SD).reshape(3, 1, 1)
            )
        },
        actuator={} if kernel is None else {"a": kernel},
    )
    return system, params


def _evaluate(system, params, plans):
    ctx = system.planner_context(params, system.initial_priors(params))
    return evaluate_policies(
        plans, ctx.init_beliefs, ctx.transition_fn, ctx.step_terms_fn, ctx.terminal_terms_fn
    )


def test_a_soft_step_is_refused_by_a_selecting_channel_and_contracted_by_a_tabular_one():
    """A probability vector over the node's levels at a step is no point: the channel that
    selects its noise by the node is refused with the sentence naming the mixture-carried route,
    and so is a declared actuator kernel. A tabular channel over the node contracts over the
    vector (the mixture of its columns), equal to the level's own at a one-hot."""
    from tests.test_action_nodes import FEEL, KERNEL, _hybrid_sensed

    system, params = soft_capable_model()
    soft = ControlStep(entries=(jnp.asarray([[[0.2, 0.8, 0.0]]]),))
    with pytest.raises(ValueError, match=r"selects its head and noise by action node 'a'.*"):
        _evaluate(system, params, soft)
    with pytest.raises(ValueError, match="a probability vector over its levels at this step"):
        _evaluate(system, params, soft)
    kerneled, kernel_params = soft_capable_model(KERNEL)
    with pytest.raises(ValueError, match="kernel declared in params.actuator"):
        _evaluate(kerneled, kernel_params, jnp.asarray([[[1]]], jnp.int32))
    #  the same plan, clamped under the delta kernel, runs
    assert jnp.isfinite(
        _evaluate(system, params, jnp.asarray([[[1]]], jnp.int32))["infogain"]["y"].lower
    ).all()

    #  S2: nothing is scored about the node, so a channel over it alone has no information-gain
    #  leaf; its ambiguity and risk leaves read the contraction over the step's belief
    tabular, tparams = _hybrid_sensed()
    tabular = dataclasses.replace(tabular, epistemic=True)
    weights = jnp.asarray([0.2, 0.5, 0.3])
    soft = ControlStep(entries=(weights.reshape(1, 1, 3),))
    clamp = jnp.asarray([[[1]]], jnp.int32)
    assert "infogain" not in _evaluate(tabular, tparams, soft)
    ambiguous = dataclasses.replace(tabular, decomposition="risk_ambiguity")

    def leaves(params, plans, belief):
        tree = _evaluate(ambiguous, params, plans)
        marginal = FEEL @ belief
        h = lambda p: -jnp.sum(p * jnp.log(p), axis=0)  # noqa: E731
        np.testing.assert_allclose(
            np.asarray(tree["ambiguity"]["feel"].lower),
            [jnp.sum(belief * h(FEEL))],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(tree["risk"]["feel"].lower), [-h(marginal)], rtol=1e-5
        )
        return tree

    leaves(tparams, soft, weights)
    one_hot = ControlStep(entries=(jnp.asarray([[[0.0, 1.0, 0.0]]]),))
    np.testing.assert_allclose(
        np.asarray(_evaluate(ambiguous, tparams, one_hot)["risk"]["feel"].lower),
        np.asarray(_evaluate(ambiguous, tparams, clamp)["risk"]["feel"].lower),
        rtol=1e-6,
    )
    #  a declared kernel mixes its columns under the soft step, and takes the level's column
    #  under a clamp
    actuated = dataclasses.replace(tparams, actuator={"go": KERNEL})
    assert "infogain" not in _evaluate(tabular, actuated, soft)
    leaves(actuated, soft, KERNEL @ weights)
    leaves(actuated, jnp.asarray([[[2]]], jnp.int32), KERNEL[:, 2])


def test_the_closed_loop_credits_the_reading_a_level_buys():
    """Under ``feedback="linear_quadratic"`` the chain's covariances feed the Riccati pass, so
    two plans that differ only in which step asks (the answer's noise is small under ASK alone)
    score different closed-loop risk, though their open-loop predictions, and so the open-loop
    risk, are the same."""
    system, params = payoff_model_by_node(C_ASK, 4)
    plans = [(ASK, DRIVE, DRIVE, DRIVE), (IDLE, DRIVE, DRIVE, DRIVE)]
    closed = {}
    for feedback in ("linear_quadratic", None):
        model = dataclasses.replace(system, feedback=feedback)
        levels = jnp.asarray(plans, jnp.int32)
        steps = ControlStep(entries=(levels, jnp.zeros((2, 4, 1))))
        tree = _evaluate(model, params, steps)
        closed[feedback] = np.asarray(tree["risk"]["lane"].lower)
    assert abs(closed["linear_quadratic"][0] - closed["linear_quadratic"][1]) > 1e-3
    np.testing.assert_allclose(closed[None][0], closed[None][1], rtol=1e-6)


# -------------------------------------------------------- the faces that keep their refusals


def test_the_faces_that_do_not_read_the_node_keep_their_refusals(monkeypatch):
    from hybrid_aif.composition.merge import merge
    from hybrid_aif.replay import replay

    system, params = scalar_model()
    beliefs = system.initial_priors(params)
    obs = {"y": jnp.asarray([0.1])}
    remedy = r"observe action node\(s\) \['a'\], which {face} does not read; drop the action"
    for face, call in (
        ("learn", lambda: system.learn(params, obs, beliefs)),
        (
            "replay",
            lambda: replay(
                system, params, {"y": jnp.zeros((2, 1))}, {"a": jnp.zeros(2, jnp.int32)}
            ),
        ),
        ("merge", lambda: merge({"m": system}, {"m": params})),
    ):
        with pytest.raises(NotImplementedError, match=remedy.format(face=face)):
            call()
    #  the amortized path reads the structured filter and the planner reads decoded codes
    monkeypatch.setattr(System, "_is_amortized", lambda self, params: True)
    for face, call in (
        ("infer_states", lambda: system.infer_states(params, obs, beliefs)),
        ("infer_policies", lambda: system.infer_policies(params, beliefs)),
    ):
        with pytest.raises(NotImplementedError, match=remedy.format(face=face)):
            call()


def test_the_cases_the_selection_does_not_cover_are_refused_with_a_sentence():
    #  a noise map on a selecting channel is not built, as on a switching one
    from hybrid_aif.factors import GridNoiseMap
    from tests.test_action_nodes import _hybrid_sensed

    system, params = scalar_model()
    mapped = LinearGaussianObservation(
        H=jnp.eye(1),
        c=jnp.zeros(1),
        noise=GridNoiseMap(values=jnp.ones((2, 1, 1)), low=-1.0, high=1.0, axes=(0,)),
    )
    with pytest.raises(NotImplementedError, match="noise map on a switching LG observation"):
        system.infer_states(
            dataclasses.replace(params, likelihood={"y": mapped}),
            {"y": jnp.zeros(1)},
            {**system.initial_priors(params), "a": Categorical.from_probs(jax.nn.one_hot(0, 3))},
        )
    #  a branch cannot read an action node: it has no action to read it from
    tabular, tparams = _hybrid_sensed()
    branching = dataclasses.replace(
        tabular, branch_on=("feel",), planner="sophisticated", epistemic=True
    )
    with pytest.raises(ValueError, match=r"branch_on names 'feel', which observes action node"):
        branching.planner_context(tparams, branching.initial_priors(tparams))


def test_the_world_draws_the_reading_at_the_realised_actions_level():
    """``as_env`` on the hybrid path generates the selecting channel's observation from the
    level of the action the actuator realised: its noise is that level's."""
    system, params = scalar_model()
    env, eparams = system.as_env(params)
    _, state = env.reset(KEY, eparams)
    for level in (0, 1):
        keys = jax.random.split(jax.random.PRNGKey(level), 400)
        draws = jax.vmap(
            lambda k, level=level: env.step_actuated(k, state, {"a": jnp.asarray(level)}, eparams)[
                0
            ]["y"]
        )(keys)
        spread = np.asarray(draws)[:, 0].std()
        #  the reading is of the next state, which the process noise has moved
        np.testing.assert_allclose(spread, np.hypot(LEVEL_SD[level], PROCESS_SD), rtol=0.15)


def test_a_channel_with_a_state_mode_parent_beside_the_node_keeps_the_mixture_rule():
    """Only a channel whose mode parents are all action nodes selects; a state mode parent keeps
    the existing rule, that the observed factor is mixture-carried."""
    system, params = soft_capable_model()
    mixed = dataclasses.replace(system, A_deps=((0, 1, 3),))
    noise = jnp.full((3, 3, 1, 1), 0.2)
    likelihood = LinearGaussianObservation(H=jnp.eye(1), c=jnp.zeros(1), noise=noise)
    with pytest.raises(ValueError, match="switching LG observation.*must be carried as a mixture"):
        mixed.planner_context(
            dataclasses.replace(params, likelihood={"y": likelihood}),
            system.initial_priors(params),
        )


def test_a_channel_over_a_state_and_a_stochastic_action_scores_its_gain_about_the_state():
    """Nothing is scored about the action node: under a declared kernel the channel's
    information gain is ``I(o; x)`` with the node's belief marginalised, as a selection of the
    state parent gives, and under the delta kernel ``I(o; x, a)`` is the same number."""
    from tests.test_action_nodes import KERNEL, _roll

    rng = np.random.default_rng(3)
    table = rng.dirichlet(np.ones(4), size=(3, 3)).transpose(2, 0, 1)  # (o, x, a)
    system = System(
        A_deps=((0, 2),),
        B_deps=((0, 1), ()),
        controls=(1,),
        action_names=("a",),
        factor_names=("x", "u"),
        modality_names=("m",),
        policy_len=1,
        epistemic=True,
    )

    def params(kernel):
        return SystemParams(
            likelihood={"m": ConditionalCategorical(log_table=jnp.log(table), parent_ndims=2)},
            transition={
                "x": ConditionalCategorical(log_table=jnp.log(_roll(0.05)), parent_ndims=2)
            },
            initial_state_prior={"x": Categorical(logits=jnp.log(jnp.asarray([0.5, 0.3, 0.2])))},
            actuator={} if kernel is None else {"a": kernel},
        )

    for kernel in (None, KERNEL):
        p = params(kernel)
        ctx = system.planner_context(p, system.initial_priors(p))
        qs = ctx.transition_fn(ctx.init_beliefs, jnp.asarray([1]))
        weights = jnp.eye(3)[1] if kernel is None else KERNEL[:, 1]
        per_state = jnp.einsum("oxa,a->ox", jnp.asarray(table), weights)
        h = lambda q: -jnp.sum(q * jnp.log(q), axis=0)  # noqa: E731
        want = h(per_state @ qs[0]) - jnp.sum(qs[0] * h(per_state))
        got = ctx.step_terms_fn(qs, jnp.asarray([1]), jnp.asarray(0))["infogain"]["m"].lower
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=2e-5)


def test_a_closed_loop_observation_target_on_a_selecting_channel_is_refused():
    system, params = payoff_model_by_node(C_ASK, 2)
    target = Gaussian(loc=jnp.zeros(1), chol=jnp.eye(1))
    with pytest.raises(ValueError, match="selects its head by an action node"):
        _evaluate(
            system,
            dataclasses.replace(params, obs_targets={"answer": target}),
            ControlStep(entries=(jnp.zeros((1, 2), jnp.int32), jnp.zeros((1, 2, 1)))),
        )


def test_a_gaussian_emission_over_the_action_node_alone_scores_alike_under_both_chains():
    """An emission has no Gaussian factor to condition, so the epistemic chain of
    ``planning_belief="posterior"`` leaves it as the prior chain does."""
    from tests.test_action_nodes import KERNEL, _hybrid_sensed

    tabular, tparams = _hybrid_sensed()
    emission = LinearGaussianObservation(
        H=jnp.zeros((3, 1, 0)),
        c=jnp.asarray([[0.0], [1.0], [2.0]]),
        noise=jnp.full((3, 1, 1), 0.3),
    )
    for actuator in ({}, {"go": KERNEL}):
        params = dataclasses.replace(tparams, likelihood={"feel": emission}, actuator=actuator)
        trees = {}
        for belief in ("prior", "posterior"):
            system = dataclasses.replace(tabular, epistemic=True, planning_belief=belief)
            trees[belief] = system.score_policies(
                params, system.initial_priors(params), system.policies(params)
            )
        assert jax.tree.structure(trees["prior"]) == jax.tree.structure(trees["posterior"])
        for want, got in zip(
            jax.tree.leaves(trees["prior"]), jax.tree.leaves(trees["posterior"]), strict=True
        ):
            np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-6, atol=1e-7)
