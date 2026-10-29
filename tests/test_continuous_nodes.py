"""Continuous control nodes and readers beyond tables.

A control node is continuous when its box is in ``params.control`` and its readers read a float
vector (a neural transition's input, a linear-Gaussian transition's ``G u``); a discrete node is
a table's control axis or a linear-Gaussian mode index. These tests cover the box in params, the
``G u`` term under a clamp and under beliefs, discrete and continuous nodes shared across
readers, and a mixed space as structure.

Design: docs/design/control-states.md C4 (Where the box lives; Discrete nodes and the LG factor;
Readers beyond tables).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.controls import Box, ControlStep
from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalGaussian,
    Gaussian,
    MLPConditional,
    TruncatedGaussian,
)
from hybrid_aif.factors import LinearGaussianTransition
from hybrid_aif.objectives import hybrid as _hybrid
from hybrid_aif.system import System, SystemParams

KEY = jax.random.PRNGKey(0)
F = jnp.asarray([[1.0, 0.5], [0.0, 1.0]])
Q = 0.1 * jnp.eye(2)
G2 = jnp.asarray([[0.1], [0.5]])


def _box(dim: int = 1, half: float = 2.0) -> Box:
    return Box(-half * jnp.ones(dim), half * jnp.ones(dim))


def _lg(loc, G=None, F=F, chol=Q) -> LinearGaussianTransition:
    """An LG transition with innovation offsets ``loc`` (``(*parent_shape, 2)``)."""
    loc = jnp.asarray(loc, jnp.float32)
    shape = loc.shape[:-1]
    chols = jnp.broadcast_to(chol, (*shape, 2, 2))
    return LinearGaussianTransition(
        F=F,
        innovations=ConditionalGaussian(loc=loc, chol=chols, parent_ndims=len(shape)),
        G=G,
    )


def _gauss(mean=(0.3, -0.2), sd=0.2) -> Gaussian:
    return Gaussian(loc=jnp.asarray(mean, jnp.float32), chol=sd * jnp.eye(2))


def _native_continuous():
    """A state ``x`` and a continuous node ``u`` declared only by ``x``'s ``G u`` reader."""
    system = System(
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("u",),
        factor_names=("x", "u_u"),
        policy_len=2,
        planner="gradient",
        epistemic=False,
    )
    params = SystemParams(
        transition={"x": _lg(jnp.zeros((1, 2)), G=G2)},
        initial_state_prior={"x": _gauss()},
        control={"u_u": _box()},
    )
    return system, params


# --------------------------------------------------------------------- A: the box lives in params


def test_a_native_continuous_node_is_declared_by_its_readers_and_its_box():
    system, params = _native_continuous()
    space = system._space(params)
    assert space.kinds == ("float",) and space.sizes == (1,)
    assert system.names.controls == ("u_u",)
    assert system.action_dim(params) == 1
    np.testing.assert_array_equal(np.asarray(space.layout.low), [-2.0])
    #  without the box the node is a mode index, and the reader's G has no input to read
    with pytest.raises(ValueError, match="declares G, which reads continuous control nodes"):
        system._validate(dataclasses.replace(params, control={}))
    #  a box keyed by no control node is refused, naming the nodes
    with pytest.raises(ValueError, match=r"key a box by one of the control nodes \['u_u'\]"):
        system._validate(dataclasses.replace(params, control={"x": _box()}))
    #  the dropped keyword is an ordinary unexpected-keyword error
    with pytest.raises(TypeError, match="unexpected keyword argument .continuous_controls."):
        System(B_deps=((),), controlled=(True,), continuous_controls=(None,), policy_len=1)
    with pytest.raises(TypeError, match="unexpected keyword argument .control_boxes."):
        System(B_deps=((1,), ()), controls=(1,), control_boxes=None, policy_len=1)


def test_readers_that_disagree_on_kind_or_dimension_are_refused():
    #  kind: one node read through a table and through an LG G term
    system = System(
        B_deps=((2,), (2,), ()),
        controls=(2,),
        action_names=("u",),
        factor_names=("s", "x", "u_u"),
        policy_len=1,
        epistemic=False,
    )
    params = SystemParams(
        transition={
            "s": ConditionalCategorical(log_table=jnp.zeros((2, 3)), parent_ndims=1),
            "x": _lg(jnp.zeros((1, 2)), G=G2),
        },
        initial_state_prior={"s": Categorical(logits=jnp.zeros(2)), "x": _gauss()},
        control={"u_u": _box()},
    )
    with pytest.raises(ValueError, match=r"\{'s': 'ConditionalCategorical'\} read it as a level"):
        system._validate(params)
    #  dimension: two neural readers at different widths
    system = System(
        B_deps=((2,), (2,), ()),
        controls=(2,),
        action_names=("u",),
        factor_names=("a", "b", "u_u"),
        policy_len=1,
        epistemic=False,
    )
    one = Gaussian(loc=jnp.zeros(1), chol=jnp.eye(1))
    params = SystemParams(
        transition={
            "a": MLPConditional.from_shapes((1, 1), 1, key=KEY, depth=0),
            "b": MLPConditional.from_shapes((1, 2), 1, key=KEY, depth=0),
        },
        initial_state_prior={"a": one, "b": one},
        control={"u_u": _box(1)},
    )
    with pytest.raises(ValueError, match=r"reader\(s\) \{'b': 2\} read it at another width"):
        system._validate(params)


# ------------------------------------------------------------------- C: the LG control input


def test_a_clamped_input_is_an_offset_and_beliefs_push_through_g():
    """``x' = F x + G u + b + ε``: a clamp shifts the mean by ``G u``; a Gaussian belief over
    ``u`` adds ``G Σ_u Gᵀ`` to the covariance exactly; a truncated one pushes through its
    closed mean and variance."""
    system, params = _native_continuous()
    q = _gauss()
    u = jnp.asarray([0.7])
    pushed = system.update_empirical_prior(params, {"x": q}, {"u": u})["x"]
    np.testing.assert_allclose(np.asarray(pushed.loc), np.asarray(F @ q.loc + G2 @ u), rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(pushed.cov), np.asarray(F @ q.cov @ F.T + Q @ Q.T), rtol=1e-5, atol=1e-7
    )

    pos = system._p(params)
    step = _hybrid.make_transition(
        pos.transition,
        system._B,
        system._families(pos),
        None,
        table=system._table,
        space=system._space(pos),
    )
    belief = Gaussian(loc=jnp.asarray([0.4]), chol=jnp.asarray([[0.3]]))
    (got,) = step((q,), ControlStep(entries=(belief,)))
    np.testing.assert_allclose(np.asarray(got.loc), np.asarray(F @ q.loc + G2 @ belief.loc))
    want = F @ q.cov @ F.T + Q @ Q.T + G2 @ belief.cov @ G2.T
    np.testing.assert_allclose(np.asarray(got.cov), np.asarray(want), rtol=1e-5, atol=1e-7)
    #  a point-mass belief is the clamp
    point = Gaussian(loc=u, chol=jnp.asarray([[1e-12]]))
    (as_clamp,) = step((q,), ControlStep(entries=(point,)))
    np.testing.assert_allclose(np.asarray(as_clamp.cov), np.asarray(pushed.cov), atol=1e-6)

    truncated = TruncatedGaussian(loc=jnp.asarray([0.4]), scale=0.5, low=-0.2, high=0.6)
    (moment,) = step((q,), ControlStep(entries=(truncated,)))
    np.testing.assert_allclose(
        np.asarray(moment.loc), np.asarray(F @ q.loc + G2 @ truncated.mean()), rtol=1e-6
    )
    want = F @ q.cov @ F.T + Q @ Q.T + G2 @ truncated.cov @ G2.T
    np.testing.assert_allclose(np.asarray(moment.cov), np.asarray(want), rtol=1e-5, atol=1e-7)


def test_score_policies_takes_a_belief_on_a_continuous_node():
    from hybrid_aif.objectives import total

    system, params = _native_continuous()
    target = _gauss((1.0, 0.0), 1.0)
    params = dataclasses.replace(params, state_targets={"x": target})
    beliefs = {"x": _gauss()}
    belief = Gaussian(loc=jnp.full((2, 1), 0.4), chol=jnp.full((2, 1, 1), 0.3))
    tree = system.score_policies(params, beliefs, {"u_u": belief})
    clamp = system.score_policies(params, beliefs, {"u_u": jnp.full((2, 1), 0.4)})
    got = float(total(tree).midpoint)
    assert got != pytest.approx(float(total(clamp).midpoint), rel=1e-3)  # the belief is read
    #  by hand: two exact pushes, the input's covariance added each step, risk = Σ KL(q_t ‖ C)
    q, want = beliefs["x"], 0.0
    for _ in range(2):
        cov = F @ q.cov @ F.T + Q @ Q.T + G2 @ belief.cov[0] @ G2.T
        q = Gaussian.from_cov(F @ q.loc + G2 @ belief.loc[0], cov)
        want += float(q.kl_to(target).lower)
    np.testing.assert_allclose(got, want, rtol=1e-5)
    #  a neural reader takes a value, so a belief on its node is refused
    mlp, mlp_params = _two_readers()
    with pytest.raises(NotImplementedError, match="whose transitions take a value; clamp it"):
        mlp.score_policies(
            mlp_params,
            mlp.initial_priors(mlp_params),
            {"u_u": Gaussian(loc=jnp.zeros((1, 1)), chol=jnp.ones((1, 1, 1)))},
        )


def test_a_factor_reads_a_discrete_mode_node_and_a_continuous_input():
    """``x' = F_j x + G_j u + b_j``: the discrete node selects ``j``, the continuous one enters
    through the per-mode ``G_j``; per-mode ``F_j`` and ``G_j`` follow the same rule."""
    system = System(
        B_deps=((1, 2), (), ()),
        controls=(1, 2),
        action_names=("d", "c"),
        factor_names=("x", "u_d", "u_c"),
        policy_len=1,
        epistemic=False,
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
    space = system._space(params)
    assert space.kinds == ("level", "float") and space.sizes == (3, 1)
    q = _gauss()
    for j in range(3):
        u = jnp.asarray([0.3 * (j + 1)])
        got = system.update_empirical_prior(params, {"x": q}, {"d": jnp.int32(j), "c": u})["x"]
        want = Fs[j] @ q.loc + Gs[j] @ u + offsets[j]
        np.testing.assert_allclose(np.asarray(got.loc), np.asarray(want), rtol=1e-6)
    #  the enumerating planner does not search a mixed space
    with pytest.raises(NotImplementedError, match="does not search their product"):
        system.infer_policies(params, {"x": q})
    #  but it scores, keyed by node: mode 2's dynamics under u = 0.5, scored against the target
    tree = system.score_policies(
        params, {"x": q}, {"u_d": jnp.asarray([[2]]), "u_c": jnp.asarray([[[0.5]]])}
    )
    pushed = Gaussian.from_cov(
        Fs[2] @ q.loc + Gs[2] @ jnp.asarray([0.5]) + offsets[2], Fs[2] @ q.cov @ Fs[2].T + Q @ Q.T
    )
    np.testing.assert_allclose(
        np.asarray(tree["risk"]["x"].midpoint),
        [float(pushed.kl_to(_gauss((1.0, 0.0), 1.0)).lower)],
        rtol=1e-5,
    )
    #  and a belief on the discrete mode parent is refused, naming the mixture-bounds work
    with pytest.raises(NotImplementedError, match="mixture-bounds"):
        system.score_policies(
            params,
            {"x": q},
            {"u_d": Categorical(logits=jnp.zeros((1, 3))), "u_c": jnp.asarray([[0.5]])},
        )
    #  the generative process steps the same way
    env, ep = system.as_env(params)
    _, state = env.reset(KEY, ep)
    action = {"d": jnp.int32(1), "c": jnp.asarray([0.5])}
    _, nxt = env.step(KEY, state, action, ep)
    noise = jax.random.split(jax.random.split(KEY)[1], 1)[0]
    draw = ConditionalGaussian(loc=offsets, chol=jnp.broadcast_to(Q, (3, 2, 2)), parent_ndims=1)
    want = Fs[1] @ state["x"] + Gs[1] @ action["c"] + draw.condition_on((1,)).sample(noise)
    np.testing.assert_allclose(np.asarray(nxt["x"]), np.asarray(want), rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------- D: readers beyond tables


def test_a_discrete_control_shared_with_an_lg_factor():
    """The sugar's shared control (``B_control_deps``): a table and an LG factor read one
    discrete node, the LG factor as a mode index after its state parent."""
    system = System(
        B_deps=((0,), (0,)),
        controlled=(True, False),
        B_control_deps=((0,), (0,)),
        factor_names=("s", "x"),
        policy_len=1,
        epistemic=False,
        mixture_k=4,
    )
    table = jnp.log(jnp.asarray([[[0.9, 0.2], [0.1, 0.8]], [[0.1, 0.8], [0.9, 0.2]]]))
    offsets = jnp.asarray(
        [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [-1.0, -1.0]]]
    )  # (s levels, u levels, 2)
    params = SystemParams(
        transition={
            "s": ConditionalCategorical(log_table=table, parent_ndims=2),
            "x": _lg(offsets),
        },
        initial_state_prior={
            "s": Categorical(logits=jnp.zeros(2)),
            "x": __import__("hybrid_aif").MixtureGaussian.from_gaussian(_gauss()),
        },
    )
    assert system._space(params).kinds == ("level",)
    beliefs = system.initial_priors(params)
    for level in (0, 1):
        pushed = system.update_empirical_prior(params, beliefs, {"a_s": jnp.int32(level)})
        s_next = jnp.exp(table[:, :, level]) @ jnp.full(2, 0.5)
        np.testing.assert_allclose(np.asarray(pushed["s"].probs), np.asarray(s_next), rtol=1e-5)
        mix = pushed["x"].moment_match()
        q = _gauss()
        want = F @ q.loc + s_next @ offsets[:, level]
        np.testing.assert_allclose(np.asarray(mix.loc), np.asarray(want), rtol=1e-5, atol=1e-6)


def test_a_native_discrete_node_read_by_an_lg_factor_at_any_position():
    """A control node before the state parent: the innovation axes follow the declared order
    (node, then state), and the mixture grows over the state parent alone."""
    system = System(
        B_deps=((0,), (2, 0), ()),
        controls=(2,),
        action_names=("u",),
        factor_names=("s", "x", "u_u"),
        policy_len=1,
        epistemic=False,
        mixture_k=4,
    )
    offsets = jnp.asarray(
        [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 3.0], [-2.0, 1.0]], [[5.0, 5.0], [0.0, 0.0]]]
    )
    params = SystemParams(
        transition={
            "s": ConditionalCategorical(log_table=jnp.log(jnp.eye(2) * 0.8 + 0.1), parent_ndims=1),
            "x": _lg(offsets),
        },
        initial_state_prior={
            "s": Categorical(logits=jnp.log(jnp.asarray([0.25, 0.75]))),
            "x": __import__("hybrid_aif").MixtureGaussian.from_gaussian(_gauss()),
        },
        state_targets={"x": _gauss((1.0, 0.0), 1.0)},
    )
    beliefs = system.initial_priors(params)
    pushed = system.update_empirical_prior(params, beliefs, {"u": jnp.int32(1)})
    s_next = pushed["s"].probs
    want = F @ _gauss().loc + s_next @ offsets[1]
    mix = pushed["x"].moment_match()
    np.testing.assert_allclose(np.asarray(mix.loc), np.asarray(want), rtol=1e-5, atol=1e-6)
    #  the enumeration planner plans over the node's 3 levels
    q_pi, _ = system.infer_policies(params, beliefs)
    assert q_pi.probs.shape == (3,)


def _two_readers():
    """One continuous node read by an LG factor (``G u``) and by a neural one (its input)."""
    system = System(
        B_deps=((2,), (2,), ()),
        controls=(2,),
        action_names=("u",),
        factor_names=("x", "y", "u_u"),
        policy_len=2,
        planner="gradient",
        epistemic=False,
    )
    params = SystemParams(
        transition={
            "x": _lg(jnp.zeros((1, 2)), G=G2),
            "y": MLPConditional.from_shapes((1, 1), 1, key=KEY, depth=0),
        },
        initial_state_prior={
            "x": _gauss(),
            "y": Gaussian(loc=jnp.zeros(1), chol=0.1 * jnp.eye(1)),
        },
        state_targets={"x": Gaussian(loc=jnp.asarray([1.0, 0.0]), chol=jnp.eye(2))},
        control={"u_u": _box()},
    )
    return system, params


def test_a_continuous_node_read_by_several_factors():
    system, params = _two_readers()
    beliefs = system.initial_priors(params)
    lo = system.update_empirical_prior(params, beliefs, {"u": jnp.asarray([-1.0])})
    hi = system.update_empirical_prior(params, beliefs, {"u": jnp.asarray([1.0])})
    np.testing.assert_allclose(
        np.asarray(hi["x"].loc - lo["x"].loc), np.asarray(2.0 * G2[:, 0]), rtol=1e-6
    )
    assert float(jnp.abs(hi["y"].loc - lo["y"].loc).max()) > 0.0  # the net reads it too
    #  the planners search it as one flat vector
    action, _, _ = system.act(KEY, params, beliefs, selection="deterministic")
    assert action["u"].shape == (1,)
    #  and the process moves both readers
    env, ep = system.as_env(params)
    _, state = env.reset(KEY, ep)
    _, a = env.step(KEY, state, {"u": jnp.asarray([-1.0])}, ep)
    _, b = env.step(KEY, state, {"u": jnp.asarray([1.0])}, ep)
    np.testing.assert_allclose(
        np.asarray(b["x"] - a["x"]), np.asarray(2.0 * G2[:, 0]), rtol=1e-5, atol=1e-6
    )


def _one_hot_reader() -> MLPConditional:
    """A depth-0 net over ``[y; one-hot(u)]`` whose mean is ``10·(level + 1)``: the one-hot's
    slots carry 10, 20, 30, and the previous ``y`` carries nothing."""
    import equinox as eqx

    net = MLPConditional.from_shapes((1, 3), 1, key=KEY, depth=0, min_scale=1e-3)
    weight = jnp.asarray([[0.0, 10.0, 20.0, 30.0], [0.0, 0.0, 0.0, 0.0]])
    net = eqx.tree_at(lambda m: m.layers[0].weight, net, weight)
    #  a scale far below the minimum: the draw is the mean to within min_scale
    return eqx.tree_at(lambda m: m.layers[0].bias, net, jnp.asarray([0.0, -30.0]))


@pytest.mark.parametrize("wiring", ["native", "shared sugar"])
def test_a_neural_transition_reads_a_discrete_node_as_its_one_hot_anywhere(wiring):
    """A discrete node read by a table and by a net: the net takes the level's one-hot in its
    control slots, whether the node is declared natively or shared through the sugar."""
    if wiring == "native":
        system = System(
            B_deps=((2,), (2,), ()),
            controls=(2,),
            action_names=("u",),
            factor_names=("s", "y", "u_u"),
            policy_len=1,
            epistemic=False,
        )
    else:
        system = System(
            B_deps=((), ()),
            controlled=(True, False),
            B_control_deps=((0,), (0,)),
            factor_names=("s", "y"),
            policy_len=1,
            epistemic=False,
        )
    action = system.names.actions[0]
    params = SystemParams(
        transition={
            "s": ConditionalCategorical(log_table=jnp.zeros((2, 3)), parent_ndims=1),
            "y": _one_hot_reader(),
        },
        initial_state_prior={
            "s": Categorical(logits=jnp.zeros(2)),
            "y": Gaussian(loc=jnp.asarray([5.0]), chol=0.1 * jnp.eye(1)),
        },
    )
    assert system._space(params).kinds == ("level",) and system._space(params).sizes == (3,)
    beliefs = system.initial_priors(params)
    for level in range(3):
        pushed = system.update_empirical_prior(params, beliefs, {action: jnp.int32(level)})
        np.testing.assert_allclose(np.asarray(pushed["y"].loc), [10.0 * (level + 1)], rtol=1e-6)
    #  the generative process reads it the same way
    env, ep = system.as_env(params)
    _, state = env.reset(KEY, ep)
    _, nxt = env.step(KEY, state, {action: jnp.int32(1)}, ep)
    assert abs(float(nxt["y"][0]) - 20.0) < 0.01


def test_a_target_on_a_continuous_node_is_a_density_over_its_values():
    system, params = _native_continuous()
    with_target = dataclasses.replace(
        params, state_targets={"u_u": Gaussian(loc=jnp.zeros(1), chol=0.5 * jnp.eye(1))}
    )
    plan = jnp.asarray([[0.2], [-0.4]])
    leaf = system.score_policies(with_target, {"x": _gauss()}, plan)["risk"]["u_u"]
    #  one leaf per policy, summed over its steps: Σ_t −log C(u_t)
    want = -Gaussian(loc=jnp.zeros(1), chol=0.5 * jnp.eye(1)).log_prob(plan).sum()
    np.testing.assert_allclose(np.asarray(leaf.midpoint), np.asarray(want), rtol=1e-6)
    with pytest.raises(ValueError, match="pass a Gaussian over its \\(1,\\) values"):
        system.score_policies(
            dataclasses.replace(params, state_targets={"u_u": Categorical(logits=jnp.zeros(2))}),
            {"x": _gauss()},
            plan,
        )


def test_the_document_writes_the_box_as_the_control_node_s_domain_and_builds_it_into_params():
    """A continuous control node's box is its domain's bounds; the reader puts it back in
    ``params.control``, and the document is unchanged by the trip."""
    import json

    from hybrid_aif.description import build, describe
    from hybrid_aif.reference.raif.world import tiny_rssm

    system, params = tiny_rssm(
        2, 1, key=KEY, units=4, deter=8, num_vars=3, num_classes=4, embed=5, mlp_units=6
    )
    doc = describe(system, params, arrays="inline")
    assert doc.document["variables"]["u_deter"] == {
        "domain": "real",
        "dim": 1,
        "low": [-1.0],
        "high": [1.0],
    }
    assert doc.document["variables"]["a_deter"] == {"domain": "real", "dim": 1}
    built, built_params = build(doc.document, arrays=doc.arrays)
    assert built == system
    box = built_params.control["u_deter"]
    np.testing.assert_array_equal(np.asarray(box.low), np.asarray(params.control["u_deter"].low))
    np.testing.assert_array_equal(np.asarray(box.high), np.asarray(params.control["u_deter"].high))
    again = describe(built, built_params, arrays="inline")
    assert json.dumps(again.document, sort_keys=True) == json.dumps(doc.document, sort_keys=True)


@pytest.mark.parametrize(
    "low, high, sentence",
    [
        (1.0, -1.0, "low >= high"),  # inverted
        (0.5, 0.5, "low >= high"),  # degenerate
        (-jnp.inf, jnp.inf, "non-finite bound"),  # infinite
        (-1.0, jnp.inf, "non-finite bound"),  # half-infinite
    ],
)
def test_a_box_that_is_not_one_is_refused_at_the_face_and_under_jit(low, high, sentence):
    system, params = _native_continuous()
    params = dataclasses.replace(params, state_targets={"x": _gauss((1.0, 0.0), 1.0)})
    beliefs = {"x": _gauss()}
    lo, hi = jnp.full(1, low), jnp.full(1, high)
    with pytest.raises(ValueError, match=sentence):
        Box(lo, hi)

    @jax.jit
    def act(lo, hi):
        bad = dataclasses.replace(params, control={"u_u": Box(lo, hi)})
        return system.act(KEY, bad, beliefs)[0]["u"]

    with pytest.raises(Exception, match=sentence):
        jax.block_until_ready(act(lo, hi))


def test_a_truncated_target_must_live_on_the_nodes_box():
    system, params = _native_continuous()  # the node's box is [-2, 2]
    beliefs, plan = {"x": _gauss()}, jnp.asarray([[0.2], [-0.4]])

    def with_target_box(low, high):
        target = TruncatedGaussian(loc=jnp.zeros(1), scale=1.0, low=low, high=high)
        return dataclasses.replace(params, state_targets={"u_u": target})

    on_box = system.score_policies(with_target_box(-2.0, 2.0), beliefs, plan)["risk"]["u_u"]
    assert np.isfinite(float(on_box.midpoint))
    with pytest.raises(ValueError, match="on another box than the node's"):
        system.score_policies(with_target_box(-1.0, 2.0), beliefs, plan)

    @jax.jit
    def score(low):
        return system.score_policies(with_target_box(low, 2.0), beliefs, plan)["risk"][
            "u_u"
        ].midpoint

    with pytest.raises(Exception, match="on another box than the node's"):
        jax.block_until_ready(score(jnp.full(1, -1.0)))


@pytest.mark.parametrize("order", ["continuous first", "discrete first"])
def test_a_net_reading_a_discrete_and_a_continuous_node_finds_each_slot(order):
    """The one-hot of a discrete node sits at its own input slot, whichever order the net's
    control parents are declared in."""
    import equinox as eqx

    cont_first = order == "continuous first"
    system = System(
        B_deps=((2,), ((3, 2) if cont_first else (2, 3)), (), ()),
        controls=(2, 3),
        action_names=("d", "c"),
        factor_names=("s", "y", "u_d", "u_c"),
        policy_len=1,
        epistemic=False,
    )
    dims = (1, 1, 3) if cont_first else (1, 3, 1)
    net = MLPConditional.from_shapes(dims, 1, key=KEY, depth=0, min_scale=1e-3)
    #  mean = 100·u + 10·(level + 1): the continuous slot carries 100, the one-hot 10/20/30
    row = [0.0, 100.0, 10.0, 20.0, 30.0] if cont_first else [0.0, 10.0, 20.0, 30.0, 100.0]
    net = eqx.tree_at(lambda m: m.layers[0].weight, net, jnp.asarray([row, [0.0] * 5]))
    net = eqx.tree_at(lambda m: m.layers[0].bias, net, jnp.asarray([0.0, -30.0]))
    params = SystemParams(
        transition={
            "s": ConditionalCategorical(log_table=jnp.zeros((2, 3)), parent_ndims=1),
            "y": net,
        },
        initial_state_prior={
            "s": Categorical(logits=jnp.zeros(2)),
            "y": Gaussian(loc=jnp.zeros(1), chol=0.1 * jnp.eye(1)),
        },
        control={"u_c": _box()},
    )
    assert system._space(params).sizes == (3, 1)
    beliefs = system.initial_priors(params)
    pushed = system.update_empirical_prior(
        params, beliefs, {"d": jnp.int32(2), "c": jnp.asarray([0.5])}
    )
    np.testing.assert_allclose(np.asarray(pushed["y"].loc), [80.0], rtol=1e-6)


def test_an_undeclared_table_axis_is_refused_at_every_face():
    """A table carrying a parent axis its factor does not declare would be read at level 0;
    prediction and the generative process refuse it, as validation does."""
    system = System(B_deps=((),), factor_names=("s",), policy_len=1)
    params = SystemParams(
        transition={"s": ConditionalCategorical(log_table=jnp.zeros((2, 2)), parent_ndims=1)},
        initial_state_prior={"s": Categorical(logits=jnp.zeros(2))},
    )
    sentence = r"has parent_ndims=1 \(parent axes \(2,\)\), and B_deps declares 0 parent"
    with pytest.raises(ValueError, match=sentence):
        system.update_empirical_prior(params, system.initial_priors(params), {})
    with pytest.raises(ValueError, match=sentence):
        system.as_env(params)


def test_an_empty_box_is_refused():
    with pytest.raises(ValueError, match="dim ≥ 1"):
        Box(jnp.zeros(0), jnp.zeros(0))
