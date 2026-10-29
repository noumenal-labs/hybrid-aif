"""The sophisticated planner and the branching it plans over.

The T-maze is the canonical case (Friston et al. 2021): without information
gain an open-loop agent never visits the cue, because going to the cue and then
to an arm fixed in advance earns no more than going to an arm now. Planned on
what the cue will say, the cue is worth what it lets the agent do next. In
this maze the cue is two moves from the arms (cue, centre, middle, arm), so
that takes a horizon of 4.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.description import build, describe
from hybrid_aif.distributions import Categorical, ConditionalCategorical
from hybrid_aif.factors import gaussian_emission
from hybrid_aif.objectives import hybrid as hybrid_terms
from hybrid_aif.objectives.discrete import terminal_cross_entropy_risk
from hybrid_aif.planning.branching import make_branch_fn
from hybrid_aif.planning.enumerate import enumerate_policies
from hybrid_aif.planning.tree import evaluate_branching_tree
from hybrid_aif.reference.envs.tmaze import CUE
from hybrid_aif.system import System, SystemParams
from tests.test_system import _initial_beliefs, tmaze_system


def _replanned(system: System, **overrides) -> System:
    """The same T-maze System, planning otherwise."""
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
    )
    return System(**{**kw, **overrides})


def _first_action(system, params, beliefs) -> int:
    q_pi, _ = system.infer_policies(params, beliefs)
    return int(system.policies(params)[int(jnp.argmax(q_pi.probs)), 0, 0])


def test_a_pragmatic_agent_goes_to_the_cue_when_it_plans_on_what_the_cue_will_say():
    open_loop, params = tmaze_system(policy_len=4, epistemic=False)
    beliefs = _initial_beliefs(open_loop, params)
    assert _first_action(open_loop, params, beliefs) != CUE
    planned = _replanned(open_loop, planner="sophisticated", branch_on=("cue",))
    assert _first_action(planned, params, beliefs) == CUE


def test_without_an_outcome_to_branch_on_the_cue_is_worth_nothing_to_it():
    open_loop, params = tmaze_system(policy_len=4, epistemic=False)
    beliefs = _initial_beliefs(open_loop, params)
    blind = _replanned(open_loop, planner="sophisticated")
    assert _first_action(blind, params, beliefs) != CUE


def test_at_one_step_it_is_enumeration():
    one, params = tmaze_system(policy_len=1)
    beliefs = _initial_beliefs(one, params)
    q_enum, _ = one.infer_policies(params, beliefs)
    soph = _replanned(one, planner="sophisticated", branch_on=("cue",))
    q_soph, _ = soph.infer_policies(params, beliefs)
    np.testing.assert_allclose(q_soph.probs, q_enum.probs, rtol=1e-5, atol=1e-7)
    assert soph.policies(params).shape == (5, 1, 1)


def test_without_branch_on_it_scores_a_state_target_as_enumeration_does():
    """A per-step state target is scored by its cross-entropy only when the plan branches; the
    sophisticated planner with no ``branch_on`` scores as enumeration does, by the KL."""
    eps = 1e-9

    def table(p, parents):
        return ConditionalCategorical(
            log_table=jnp.log(jnp.clip(jnp.asarray(p), eps)), parent_ndims=parents
        )

    B = np.zeros((2, 2, 2))
    B[0, :, 0] = 1.0  # action 0 makes the state 0
    B[:, :, 1] = 0.5  # action 1 makes it a coin flip
    params = SystemParams(
        likelihood={"o": table(np.full((2, 2), 0.5), 1)},
        transition={"s": table(B, 2)},
        initial_state_prior={"s": Categorical(logits=jnp.zeros(2))},
        state_targets={"s": Categorical(logits=jnp.zeros(2))},
    )
    q = {}
    for planner in ("enumeration", "sophisticated"):
        system = System(
            A_deps=((0,),),
            B_deps=((0,),),
            controlled=(True,),
            policy_len=1,
            gamma=4.0,
            planner=planner,
            modality_names=("o",),
            factor_names=("s",),
        )
        q[planner], _ = system.infer_policies(params, system.initial_priors(params))
    np.testing.assert_array_equal(np.asarray(q["sophisticated"].probs), q["enumeration"].probs)
    assert float(q["enumeration"].probs[1]) > 0.9  # the KL prefers the flat coin flip


def test_a_registered_branching_planner_can_branch_on_a_modality(monkeypatch):
    from hybrid_aif.planning import base, register_planner
    from hybrid_aif.planning.sophisticated import sophisticated_planner

    monkeypatch.setitem(base._PLANNERS, "wrapped_sophisticated", None)  # removed at teardown

    @register_planner("wrapped_sophisticated", branching=True)
    def wrapped(ctx, use_bound="midpoint", metrics=True):
        return sophisticated_planner(ctx, use_bound, metrics)

    system, params = tmaze_system(policy_len=4, epistemic=False)
    beliefs = _initial_beliefs(system, params)
    reference = _replanned(system, planner="sophisticated", branch_on=("cue",))
    custom = _replanned(system, planner="wrapped_sophisticated", branch_on=("cue",))
    q_ref, _ = reference.infer_policies(params, beliefs)
    q_custom, _ = custom.infer_policies(params, beliefs)
    np.testing.assert_array_equal(np.asarray(q_custom.probs), np.asarray(q_ref.probs))
    assert int(jnp.argmax(q_custom.probs)) == CUE


def test_the_policies_it_returns_are_the_first_steps():
    system, params = tmaze_system()
    soph = _replanned(system, planner="sophisticated", branch_on=("cue",))
    beliefs = _initial_beliefs(soph, params)
    action, q_pi, tree = soph.act(
        __import__("jax").random.PRNGKey(0), params, beliefs, selection="deterministic"
    )
    assert q_pi.probs.shape == (5,)
    assert tree == {}
    assert int(action["a_location"]) == int(jnp.argmax(q_pi.probs))


def test_branching_is_exact_over_the_parents():
    """A modality over two factors: p(o) and each parent's posterior after o
    against the joint computed by hand."""
    table = np.array(  # child-first: (o, a, b)
        [[[0.9, 0.3], [0.2, 0.6]], [[0.1, 0.7], [0.8, 0.4]]], dtype=np.float32
    )
    A = ConditionalCategorical.from_probs(jnp.asarray(table), 2)
    qa, qb, other = (
        Categorical.from_probs(jnp.asarray([0.7, 0.3])),
        Categorical.from_probs(jnp.asarray([0.4, 0.6])),
        Categorical.from_probs(jnp.asarray([0.5, 0.5])),
    )
    p, after = make_branch_fn([A], [(0, 1)], [0])((qa, qb, other))
    joint = table * np.array([0.7, 0.3])[None, :, None] * np.array([0.4, 0.6])[None, None, :]
    np.testing.assert_allclose(p, joint.sum(axis=(1, 2)), rtol=1e-5)
    for o in range(2):
        post = joint[o] / joint[o].sum()
        np.testing.assert_allclose(after[0].probs[o], post.sum(axis=1), rtol=1e-5)
        np.testing.assert_allclose(after[1].probs[o], post.sum(axis=0), rtol=1e-5)
        np.testing.assert_allclose(after[2].probs[o], [0.5, 0.5], rtol=1e-6)


def test_two_modalities_branch_on_their_product():
    A = ConditionalCategorical.from_probs(jnp.asarray([[0.9, 0.2], [0.1, 0.8]]), 1)
    q = (Categorical.from_probs(jnp.asarray([0.5, 0.5])),)
    p, after = make_branch_fn([A, A], [(0,), (0,)], [0, 1])(q)
    assert p.shape == (4,) and after[0].probs.shape == (4, 2)
    np.testing.assert_allclose(float(p.sum()), 1.0, rtol=1e-6)
    # Two agreeing outcomes: 0.81 against 0.04.
    np.testing.assert_allclose(float(after[0].probs[0, 0]), 0.81 / 0.85, rtol=1e-5)


def test_branch_on_is_a_branching_planners_and_names_a_modality():
    system, params = tmaze_system()
    with pytest.raises(ValueError, match="only a branching planner"):
        _replanned(system, branch_on=("cue",))
    soph = _replanned(system, planner="sophisticated", branch_on=("nowhere",))
    beliefs = _initial_beliefs(soph, params)
    with pytest.raises(ValueError, match="not a modality"):
        soph.infer_policies(params, beliefs)


def test_a_document_keeps_what_it_branches_on():

    system, params = tmaze_system()
    soph = _replanned(system, planner="sophisticated", branch_on=("cue",))
    doc = describe(soph, params)
    assert doc.document["configuration"]["planning"]["branch_on"] == ["cue"]
    rebuilt, _ = build(doc.document, arrays=doc.arrays)
    assert rebuilt.branch_on == ("cue",) and rebuilt.planner == "sophisticated"
    assert "branch_on" not in describe(system, params).document["configuration"]["planning"]


def test_branch_on_is_refused_on_the_amortized_path_rather_than_ignored():
    """The amortized context builds no branch closure; a declared ``branch_on``
    there must fail at the declaration's first use, not plan blind."""
    import dataclasses

    from tests.test_amortized_objective import agent as amortized_agent

    system, params, _ = amortized_agent()
    soph = dataclasses.replace(
        system, planner="sophisticated", branch_on=(system.modality_names[0],)
    )
    with pytest.raises(ValueError, match="amortized path"):
        soph.infer_policies(params, system.initial_priors(params))


# ------------------------------------------ state targets inside a branching planner (V11 d6)

LOG2 = float(np.log(2.0))
WAIT, READ = 0, 1


def _two_rooms(policy_len: int = 3, terminal: bool = False, **overrides):
    """A robot 50/50 between two identical rooms; ``read`` shows a sign naming the room, ``wait``
    shows a blank; nothing it does depends on the answer, and its state target over the room is
    flat."""
    eps = 1e-9
    sign = np.zeros((3, 2, 2))
    sign[2, :, WAIT] = 1.0  # waiting shows the blank
    sign[0, 0, READ] = sign[1, 1, READ] = 1.0  # reading names the room
    room = np.eye(2)[:, :, None]  # (child, parent, control): it stays where it is
    looking = np.eye(2)  # (child, control): what the robot does this tick
    kw = dict(
        A_deps=((0, 1),),
        B_deps=((0,), ()),
        controlled=(False, True),
        policy_len=policy_len,
        gamma=16.0,
        planner="sophisticated",
        branch_on=("sign",),
        modality_names=("sign",),
        factor_names=("room", "looking"),
    )
    system = System(**{**kw, **overrides})

    def table(p, parents):
        return ConditionalCategorical(
            log_table=jnp.log(jnp.clip(jnp.asarray(p), eps)), parent_ndims=parents
        )

    flat = Categorical(logits=jnp.zeros(2))
    params = SystemParams(
        likelihood={"sign": table(sign, 2)},
        transition={"room": table(room, 2), "looking": table(looking, 1)},
        initial_state_prior={
            "room": flat,
            "looking": Categorical(logits=jnp.log(jnp.asarray([1.0 - eps, eps]))),
        },
        state_targets={"room": flat},
        terminal_targets={"room": flat} if terminal else {},
    )
    return system, params


def _root_g(ctx) -> np.ndarray:
    actions = enumerate_policies(ctx.num_controls, 1, ctx.controlled)[:, 0, :]
    assert np.asarray(actions)[:, 0].tolist() == [WAIT, READ]
    return np.asarray(
        evaluate_branching_tree(
            actions,
            ctx.policy_len,
            ctx.gamma,
            ctx.init_beliefs,
            ctx.transition_fn,
            ctx.step_terms_fn,
            ctx.terminal_terms_fn,
            ctx.branch_fn,
        )
    )


@pytest.mark.parametrize("k", [3, 4])
def test_reading_the_sign_is_not_charged_for_what_it_tells(k):
    """Inside the sophisticated planner a per-step state target is its cross-entropy, ``log 2``
    on every belief over the room, so each of the ``k`` steps costs ``log 2`` and reading the sign
    earns its information gain, ``log 2``, wherever in the plan it happens. Nothing depends on the
    answer, so reading now and waiting then reading are worth the same, ``(k − 1) · log 2``.
    Scored by the KL, each of the ``k − 1`` steps after reading charges
    ``KL(q(room | sign) ‖ flat) = log 2`` and reading now is penalised."""
    system, params = _two_rooms(k)
    beliefs = system.initial_priors(params)
    ctx = system.planner_context(params, beliefs)
    g = _root_g(ctx)
    # float32 rounding, plus the γ = 16 softmax's 2^-16 weight on waiting at the last step
    tol = 2e-5
    assert g[READ] == pytest.approx((k - 1) * LOG2, abs=tol)
    assert g[WAIT] == pytest.approx((k - 1) * LOG2, abs=tol)
    q_pi, _ = system.infer_policies(params, beliefs)
    np.testing.assert_allclose(np.asarray(q_pi.probs), [0.5, 0.5], atol=1e-3)

    # the KL, by hand on the branches the planner reads: the sign's two outcomes, each leaving
    # the room known
    child = ctx.transition_fn(ctx.init_beliefs, jnp.asarray([READ], jnp.int32))
    p, after = ctx.branch_fn(child)
    kl = [
        float(Categorical(logits=jnp.log(after[0][o])).kl_to(params.state_targets["room"]).lower)
        for o in range(2)
        if float(p[o]) > 1e-6
    ]
    assert kl == pytest.approx([LOG2, LOG2], abs=1e-4)
    charged = -LOG2 + (k - 1) * float(np.dot(np.asarray(p)[:2], kl))  # = (k − 2) · log 2
    # the step terms built for an open-loop planner score the same tree by the KL: reading now
    # costs (k − 2) · log 2; waiting keeps the belief flat (KL 0) and reads at the last step,
    # which no later step charges, for −log 2
    kl_ctx = system._planner_context(
        params, system.to_positional("beliefs", beliefs), branching=False
    )
    kl_g = _root_g(kl_ctx)
    assert charged == pytest.approx((k - 2) * LOG2, abs=1e-4)
    assert kl_g[READ] == pytest.approx(charged, abs=1e-4)
    assert kl_g[WAIT] == pytest.approx(-LOG2, abs=1e-4)
    assert kl_g[READ] - kl_g[WAIT] == pytest.approx((k - 1) * LOG2, abs=1e-4)  # reading penalised


def test_the_terminal_target_is_scored_by_its_cross_entropy_in_the_tree():
    system, params = _two_rooms(3, terminal=True)
    beliefs = system.initial_priors(params)
    ctx = system.planner_context(params, beliefs)
    known = (jnp.asarray([1.0 - 1e-7, 1e-7]), jnp.asarray([1e-7, 1.0 - 1e-7]))
    unknown = (jnp.asarray([0.5, 0.5]), jnp.asarray([1e-7, 1.0 - 1e-7]))
    for qs in (known, unknown):  # flat: log 2 whatever the belief, where the KL is log 2 or 0
        terminal = ctx.terminal_terms_fn(qs)
        assert float(terminal["terminal_risk"]["room"].lower) == pytest.approx(LOG2, abs=1e-4)
    # reading first and reading at the last step pay the same terminal charge
    bare, bare_params = _two_rooms(3)
    without = _root_g(bare.planner_context(bare_params, beliefs))
    np.testing.assert_allclose(_root_g(ctx) - without, [LOG2, LOG2], atol=1e-4)
    # the KL there is read before the last branch: waiting and reading last escapes it
    kl_ctx = system._planner_context(
        params, system.to_positional("beliefs", beliefs), branching=False
    )
    kl_g = _root_g(kl_ctx)
    assert kl_g[READ] - kl_g[WAIT] == pytest.approx(3 * LOG2, abs=1e-4)
    # an open-loop score of the same System reads the KL, 0 on the unchanged 50/50 belief
    policies = jnp.asarray([[[READ], [WAIT], [WAIT]]], jnp.int32)
    tree = system.score_policies(params, beliefs, policies)
    np.testing.assert_allclose(np.asarray(tree["risk"]["room"].lower), [0.0], atol=1e-5)
    np.testing.assert_allclose(np.asarray(tree["terminal_risk"]["room"].lower), [0.0], atol=1e-5)
    enumerated = _replanned(system, planner="enumerate", branch_on=())
    np.testing.assert_array_equal(
        np.asarray(enumerated.score_policies(params, beliefs, policies)["risk"]["room"].lower),
        np.asarray(tree["risk"]["room"].lower),
    )


def _one_shot_sign(policy_len: int, terminal: bool):
    """:func:`_two_rooms` with a sign that can be read at the first step only: a clock counts
    down 2 → 1 → 0 and the sign names the room at 1, so a plan that waits never learns it."""
    eps = 1e-9
    sign = np.zeros((3, 2, 2, 3))  # (obs, room, looking, clock)
    sign[2] = 1.0
    sign[:, :, READ, 1] = 0.0
    sign[0, 0, READ, 1] = sign[1, 1, READ, 1] = 1.0
    clock = np.zeros((3, 3, 1))
    clock[0, 0] = clock[0, 1] = clock[1, 2] = 1.0
    system = System(
        A_deps=((0, 1, 2),),
        B_deps=((0,), (), (2,)),
        controlled=(False, True, False),
        policy_len=policy_len,
        gamma=16.0,
        planner="sophisticated",
        branch_on=("sign",),
        modality_names=("sign",),
        factor_names=("room", "looking", "clock"),
    )

    def table(p, parents):
        return ConditionalCategorical(
            log_table=jnp.log(jnp.clip(jnp.asarray(p), eps)), parent_ndims=parents
        )

    flat = Categorical(logits=jnp.zeros(2))
    params = SystemParams(
        likelihood={"sign": table(sign, 3)},
        transition={
            "room": table(np.eye(2)[:, :, None], 2),
            "looking": table(np.eye(2), 1),
            "clock": table(clock, 2),
        },
        initial_state_prior={
            "room": flat,
            "looking": Categorical(logits=jnp.log(jnp.asarray([1.0 - eps, eps]))),
            "clock": Categorical(logits=jnp.log(jnp.asarray([eps, eps, 1.0 - 2 * eps]))),
        },
        state_targets={"room": flat},
        terminal_targets={"room": flat} if terminal else {},
    )
    return system, params


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("k", [2, 3, 4])
def test_a_sign_read_once_is_worth_its_information_gain_at_every_depth(k, terminal):
    system, params = _one_shot_sign(k, terminal)
    beliefs = system.initial_priors(params)
    g = _root_g(system.planner_context(params, beliefs))
    assert g[READ] - g[WAIT] == pytest.approx(-LOG2, abs=1e-4)  # credited once
    # the per-branch KL charges the known room at each of the k − 1 later steps (and once
    # more at the horizon, when a terminal target is declared)
    kl_ctx = system._planner_context(
        params, system.to_positional("beliefs", beliefs), branching=False
    )
    kl_g = _root_g(kl_ctx)
    charged = (k - 1) + int(terminal)
    assert kl_g[READ] - kl_g[WAIT] == pytest.approx((charged - 1) * LOG2, abs=1e-4)


def test_a_conditional_terminal_target_is_linear_in_its_parents_belief():
    # h(s | p, r): child-first (s, p, r), unnormalised logits normalised over s; s is known, p is
    # learned in branches, r stays broad
    rng = np.random.default_rng(0)
    logits = jnp.asarray(rng.normal(size=(3, 4, 2)))
    h = ConditionalCategorical(log_table=logits, parent_ndims=2)
    log_h = np.asarray(jax.nn.log_softmax(logits, axis=0))
    targets, deps, names = {"s": h}, ((1, 2), (), ()), ("s", "p", "r")
    s = jnp.asarray([0.2, 0.0, 0.8])
    r = jnp.asarray([0.3, 0.7])

    def leaf(p_probs):
        qs = (s, jnp.asarray(p_probs), r)
        tree = terminal_cross_entropy_risk(qs, targets, deps, names)
        return float(tree["terminal_risk"]["s"].lower)

    p_mean = np.asarray([0.1, 0.2, 0.3, 0.4])
    by_hand = -np.einsum("i,ijk,j,k->", np.asarray(s), log_h, p_mean, np.asarray(r))
    assert leaf(p_mean) == pytest.approx(by_hand, rel=1e-5)
    branches = sum(w * leaf(np.eye(4)[j]) for j, w in enumerate(p_mean))
    assert branches == pytest.approx(leaf(p_mean), rel=1e-5)
    # the resolved target Σ h(s | p, r) q(p) q(r) would charge for learning p (Jensen)
    resolved = -np.einsum(
        "i,i->",
        np.asarray(s),
        np.log(np.einsum("ijk,j,k->i", np.exp(log_h), p_mean, np.asarray(r))),
    )
    assert resolved < branches - 1e-3


def test_a_hard_zero_in_a_conditional_terminal_target_is_not_a_nan():
    h = ConditionalCategorical(
        log_table=jnp.log(jnp.asarray([[1.0, 0.0], [0.0, 1.0]])), parent_ndims=1
    )
    qs = (jnp.asarray([1.0, 0.0]), jnp.asarray([1.0, 0.0]))
    tree = terminal_cross_entropy_risk(qs, {"s": h}, ((1,), ()), ("s", "p"))
    assert float(tree["terminal_risk"]["s"].lower) == 0.0


def test_both_carriers_score_the_terminal_cross_entropy_alike():
    system, params = _two_rooms(3, terminal=True)
    fast = system.planner_context(params, system.initial_priors(params)).terminal_terms_fn
    generic = hybrid_terms.make_terminal_terms(
        {"room": Categorical(logits=jnp.zeros(2))}, ("room", "looking"), branching=True
    )
    qs = (jnp.asarray([0.3, 0.7]), jnp.asarray([1e-7, 1.0 - 1e-7]))
    a = float(fast(qs)["terminal_risk"]["room"].lower)
    b = generic(tuple(Categorical(logits=jnp.log(q)) for q in qs))["terminal_risk"]["room"]
    assert a == pytest.approx(LOG2, abs=1e-5)
    assert float(b.lower) == pytest.approx(a, abs=1e-5)


def test_a_branching_system_on_the_generic_carrier_scores_the_terminal_cross_entropy():
    # a Gaussian emission the room does not move takes the System off the tabular path
    system, params = _two_rooms(
        3,
        terminal=True,
        A_deps=((0, 1), (0,)),
        modality_names=("sign", "glow"),
        epistemic=("room",),
    )
    params = dataclasses.replace(
        params,
        likelihood={**params.likelihood, "glow": gaussian_emission(jnp.zeros((2, 1)), jnp.eye(1))},
    )
    assert system._is_hybrid(params)
    beliefs = system.initial_priors(params)
    bare = _root_g(
        system.planner_context(dataclasses.replace(params, terminal_targets={}), beliefs)
    )
    g = _root_g(system.planner_context(params, beliefs))
    np.testing.assert_allclose(g - bare, [LOG2, LOG2], atol=1e-4)
    kl_ctx = system._planner_context(
        params, system.to_positional("beliefs", beliefs), branching=False
    )
    kl_g = _root_g(kl_ctx)
    assert kl_g[READ] - kl_g[WAIT] == pytest.approx(3 * LOG2, abs=1e-4)
