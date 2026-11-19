"""The ``"factorised"`` planner: one policy posterior per control node, by mean-field sweeps.

Each control node keeps a posterior over its own sequences, scored with every other node entered
as its per-step marginal over levels (a probability vector in a ``ControlStep``), the nodes
updated in node order; ``q_pi`` is over the joint first step, the product of the nodes' first-step
marginals. The pins are the design's (docs/design/factorised-planning.md, "What to pin" 1-5): with
nodes that share no factor it is the enumeration planner's action marginals after one sweep and
does not move after; with one node it is the enumeration planner; on the two-rover survey
scheduler (one field learned within the plan) the gap to the product is measured; the sweeps move
and settle; and what it does not plan is refused. Then: it runs inside ``plan()`` / ``run``.
"""

import dataclasses

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest

from hybrid_aif import Categorical, ConditionalCategorical
from hybrid_aif.composition import Trace, plan, run
from hybrid_aif.objectives import total
from hybrid_aif.planning import (
    FactorisedPolicies,
    action_marginals,
    enumerate_policies,
    factorised_planner,
    get_planner,
    make_factorised_planner,
    planner_takes_init,
    policy_posterior,
)
from hybrid_aif.planning.enumerate import sample_action
from hybrid_aif.planning.tree import evaluate_policy_tree
from hybrid_aif.reference.envs.tmaze import tmaze
from hybrid_aif.system import System, SystemParams
from tests.test_mixed_planner import toy
from tests.test_plan_warm_start import steered
from tests.test_run import COUNTS, shared_field
from tests.test_survey_run import example

KEY = jax.random.PRNGKey(0)
TOL = 1e-6


def _cc(table, parent_ndims):
    return ConditionalCategorical(
        log_table=jnp.log(jnp.clip(table, 1e-16)), parent_ndims=parent_ndims
    )


def two_mazes(policy_len=2):
    """Two T-mazes as one System: factors ``(l1, c1)`` and ``(l2, c2)``, modalities of each maze
    reading its own factors, one control node per maze (``u_l1``, ``u_l2``), no shared table; the
    second maze's reward target is weaker, so the two nodes' posteriors differ."""
    env, p = tmaze(reward_probability=0.98, punishment_probability=0.98, cue_validity=0.98)
    deps = tuple(env.A_deps)
    system = System(
        A_deps=deps + tuple(tuple(f + 2 for f in d) for d in deps),
        B_deps=((0,), (1,), (2,), (3,)),
        controlled=(True, False, True, False),
        policy_len=policy_len,
        gamma=16.0,
        epistemic=True,
        modality_names=("loc1", "reward1", "cue1", "loc2", "reward2", "cue2"),
        state_names=("l1", "c1", "l2", "c2"),
    )
    likelihood = tuple(_cc(a, len(d)) for a, d in zip(p.likelihood, deps, strict=True))
    transition = tuple(_cc(b, len(d) + 1) for b, d in zip(p.transition, env.B_deps, strict=True))
    prior = tuple(Categorical.from_probs(d) for d in p.initial_state_prior)
    params = SystemParams(
        likelihood=likelihood + likelihood,
        transition=transition + transition,
        initial_state_prior=prior + prior,
        obs_targets={
            "reward1": Categorical(logits=jnp.asarray([0.0, 4.0, -6.0])),
            "reward2": Categorical(logits=jnp.asarray([0.0, 2.0, -6.0])),
        },
        state_targets={},
        terminal_targets={},
    )
    params = system.from_positional("params", params)
    return system, params, dict(params.initial_state_prior)


def _plan(system, params, beliefs, planner=None, **kw):
    return system.infer_policies(params, beliefs, planner=planner, return_policies=True, **kw)


def _marginal_gap(system, params, beliefs, sweeps, **kw):
    """The largest absolute difference of the first-step marginals of the enumeration planner and
    of ``sweeps`` factorised sweeps, whether every node's argmax agrees, and the
    :class:`FactorisedPolicies`."""
    levels = system._control_levels(params)
    q, _, joint = _plan(system, params, beliefs, get_planner("enumeration"), **kw)
    want = action_marginals(q, joint, levels)
    qf, _, fp = _plan(system, params, beliefs, make_factorised_planner(sweeps=sweeps), **kw)
    got = action_marginals(qf, fp.first, levels)
    gap = max(float(np.abs(a.probs - b.probs).max()) for a, b in zip(want, got, strict=True))
    agree = all(
        int(jnp.argmax(a.probs)) == int(jnp.argmax(b.probs)) for a, b in zip(want, got, strict=True)
    )
    return gap, agree, fp


# --------------------------------------------------------------------- 1. independence is exact


def test_independent_nodes_give_the_enumeration_marginals_after_one_sweep_and_then_stay():
    system, params, beliefs = two_mazes()
    gap, agree, fp = _marginal_gap(system, params, beliefs, sweeps=1)
    assert gap < TOL and agree
    assert isinstance(fp, FactorisedPolicies)
    assert fp.first.shape == (25, 1, 2)
    assert [g.shape for g in fp.grids] == [(25, 2), (25, 2)]
    assert [m.shape for m in fp.marginals] == [(2, 5), (2, 5)]
    assert [q.probs.shape for q in fp.posteriors] == [(25,), (25,)]
    assert float(fp.movement) > 0.1  # the first sweep moves off the uniform start
    _, _, fp2 = _marginal_gap(system, params, beliefs, sweeps=2)
    assert float(fp2.movement) < TOL
    #  the trees are each node's own, aligned with its posterior
    assert all(total(t).midpoint.shape == (25,) for t in fp2.trees)


def test_q_pi_is_the_product_of_the_first_step_marginals_and_actions_draw_from_it():
    system, params, beliefs = two_mazes()
    qf, tree, fp = _plan(system, params, beliefs, make_factorised_planner())
    assert tree == {}
    first = [
        jnp.sum(q.probs[:, None] * (g[:, 0, None] == jnp.arange(5)), 0)
        for q, g in zip(fp.posteriors, fp.grids, strict=True)
    ]
    product = first[0][fp.first[:, 0, 0]] * first[1][fp.first[:, 0, 1]]
    np.testing.assert_allclose(qf.probs, product, atol=TOL)
    np.testing.assert_allclose(fp.marginals[0][0], first[0], atol=TOL)
    #  metrics off: the same posterior, no trees
    q_off, tree_off, fp_off = _plan(
        system, params, beliefs, make_factorised_planner(), metrics=False
    )
    assert tree_off is None and fp_off.trees is None
    np.testing.assert_allclose(q_off.probs, qf.probs, atol=TOL)
    #  both modes of sample_action act on the joint first-step grid
    for mode in ("marginal", "policy"):
        a = sample_action(
            KEY, qf, fp, system._control_levels(params), mode=mode, selection="deterministic"
        )
        assert a.shape == (2,) and a.dtype == jnp.int32
    action, q_act, _ = system.act(
        KEY, params, beliefs, planner=make_factorised_planner(), selection="deterministic"
    )
    best = fp.first[int(jnp.argmax(qf.probs)), 0]
    assert [int(action["a_l1"]), int(action["a_l2"])] == [int(best[0]), int(best[1])]
    np.testing.assert_allclose(q_act.probs, qf.probs, atol=TOL)


def test_the_registered_name_resolves_and_a_first_step_mask_applies_to_the_joint_grid():
    system, params, beliefs = two_mazes()
    assert get_planner("factorised") is factorised_planner
    assert not planner_takes_init(factorised_planner)
    named = dataclasses.replace(system, planner="factorised")
    q_named, _ = named.infer_policies(params, beliefs)
    q_made, _, _ = _plan(system, params, beliefs, make_factorised_planner(sweeps=3))
    np.testing.assert_allclose(q_named.probs, q_made.probs, atol=TOL)
    #  a first-step mask: the masked levels take no mass, and with independent nodes the
    #  masked marginals are the enumeration planner's
    legal = {"u_l1": jnp.asarray([True, True, False, True, True])}
    gap, agree, fp = _marginal_gap(system, params, beliefs, sweeps=1, legal=legal)
    assert gap < TOL and agree
    q_masked, _, fp = _plan(named, params, beliefs, legal=legal)
    assert float(jnp.sum(jnp.where(fp.first[:, 0, 0] == 2, q_masked.probs, 0.0))) == 0.0
    with pytest.raises(ValueError, match="factorised planner's posterior is over the first"):
        named.infer_policies(params, beliefs, legal={"u_l1": jnp.ones((2, 5), bool)})


def test_a_prior_per_node_gives_the_enumeration_planner_s_under_their_product():
    system, params, beliefs = two_mazes()
    grid = np.asarray(enumerate_policies((5,), 2, (True,))[:, :, 0])
    e1 = jnp.asarray(np.where(grid[:, 0] == 0, 3.0, 0.0), jnp.float32)  # node 1 likes to stay
    e2 = jnp.asarray(np.where(grid[:, 1] == 4, 2.0, 0.0), jnp.float32)
    joint = np.asarray(system.policies(params))
    index = joint[:, 0, :] * 5 + joint[:, 1, :]  # each node's sequence's row of its own grid
    product = e1[index[:, 0]] + e2[index[:, 1]]
    with_joint = dataclasses.replace(params, policy_prior=product)
    with_nodes = dataclasses.replace(params, policy_prior=(e1, Categorical(logits=e2)))
    levels = system._control_levels(params)
    q, _, pol = _plan(system, with_joint, beliefs, get_planner("enumeration"))
    qf, _, fp = _plan(system, with_nodes, beliefs, make_factorised_planner(sweeps=1))
    for a, b in zip(
        action_marginals(q, pol, levels), action_marginals(qf, fp.first, levels), strict=True
    ):
        np.testing.assert_allclose(a.probs, b.probs, atol=TOL)


# --------------------------------------------------------------- 2. one node is the enumeration


@pytest.fixture(scope="module")
def survey():
    return example("survey_scheduler"), example("survey_model")


def test_one_node_is_the_enumeration_planner(survey):
    """The one-rover scheduler, horizon 4, ``N_OPTIONS ** 4`` sequences (2,401 at seven options).
    The factorised planner scores a node's sequences as the enumeration planner scores its grid,
    by the prefix tree (:func:`~hybrid_aif.planning.tree.evaluate_policy_tree`) with the same
    nesting and child order, so with one node the two are the same computation: the node's G and
    posterior equal the tree's bit for bit, and ``q_pi`` differs from the enumeration planner's
    action marginal by 6.0e-8 (the marginal summed by ``action_marginals`` against the
    factorised planner's logsumexp; measured 2026-11-18), pinned at 1e-6. (Under the flat scan of
    the first build the novelty leaf, ``transition:map``, reassociated by 2.4e-4 nats and the
    gap was 2.0e-4; the flat scan is no longer this planner's score, so it is not compared.)"""
    ss, sm = survey
    grid, centres = sm.field_grid()
    system, params = ss.scheduler(grid, centres, durations=True, readings=True)
    assert system.policy_len == ss.HORIZON == 4
    beliefs = dict(params.initial_state_prior)
    q, tree, joint = _plan(system, params, beliefs, get_planner("enumeration"))
    n = sm.N_OPTIONS
    assert joint.shape == (n**4, 4, 1)
    qf, _, fp = _plan(system, params, beliefs, make_factorised_planner())
    want = action_marginals(q, joint, (n,))[0].probs
    assert qf.probs.shape == (n,)
    gap = float(jnp.abs(qf.probs - want).max())
    print(f"q_pi gap to the enumeration planner {gap:.3g}")
    np.testing.assert_allclose(qf.probs, want, atol=TOL)
    assert int(jnp.argmax(qf.probs)) == int(jnp.argmax(want))
    #  against the prefix tree of the same grid: the same G and posterior
    ctx = system.planner_context(params, beliefs)
    scored, order = evaluate_policy_tree(
        ctx.num_controls,
        (True,),
        system.policy_len,
        ctx.init_beliefs,
        ctx.transition_fn,
        ctx.step_terms_fn,
        ctx.terminal_terms_fn,
    )
    np.testing.assert_array_equal(order[:, :, 0], fp.grids[0])
    np.testing.assert_allclose(total(fp.trees[0]).midpoint, total(scored).midpoint, atol=TOL)
    q_tree = policy_posterior(total(scored), system.gamma)
    np.testing.assert_allclose(fp.posteriors[0].probs, q_tree.probs, atol=TOL)
    assert float(fp.movement) == 0.0  # one node: the second sweep scores what the first did


# -------------------------------------------------------- 3, 4. the coupled case, and the sweeps


def two_rovers(ss, sm, gamma=None):
    """The two-rover scheduler at horizon 2 (``N_OPTIONS ** 2`` sequences a rover, their square
    joint: 25 and 625 at five options, 49 and 2,401 at seven), the field learned
    within the plan, durations and readings as the demo declares them, and a field holding some
    counts (a third of the cells read three times, at seeded terrain levels), so that the novelty
    the two rovers share depends on where each goes. ``gamma`` replaces the System's (16)."""
    grid, centres = sm.field_grid()
    system, params = ss.scheduler(
        grid,
        centres,
        rovers=2,
        policy_len=2,
        durations=True,
        readings=True,
        plan_learning=("parameter:field",),
    )
    field = params.parameters["field"]
    rng = np.random.default_rng(0)
    conc = np.asarray(field.concentration).copy()
    for cell in rng.choice(conc.shape[-1], size=conc.shape[-1] // 3, replace=False):
        conc[rng.integers(conc.shape[0]), cell] += 3.0
    field = dataclasses.replace(field, concentration=jnp.asarray(conc))
    params = dataclasses.replace(params, parameters={**params.parameters, "field": field})
    if gamma is not None:
        system = dataclasses.replace(system, gamma=gamma)
    return system, params, dict(params.initial_state_prior)


def test_the_coupled_scheduler_against_the_product(survey):
    """Two rovers, one field learned within the plan, at the scheduler's ``gamma`` (16) and at
    ``gamma`` 1, where both posteriors are spread. Measured 2026-11-18, after three sweeps, the
    largest absolute difference of a first-step marginal from the enumeration planner's: 1.0e-9
    at ``gamma`` 16 (the posteriors near one-hot: rover 1 surveys sector 4, rover 2 charges;
    3.2e-6 after one sweep), 1.6e-4 at ``gamma`` 1 (1.1e-3 after one sweep, the rovers planned
    against each other's uniform marginals), at five options; at seven (a charge option per
    sunny region) 1.0e-9 and 4.1e-4. The argmax agrees in both. The gap is the mean
    field's: the field's novelty is evaluated at the other rover's expected readings, not
    averaged over its plans (F2)."""
    ss, sm = survey
    gaps = {}
    for gamma, bound in ((None, 1e-7), (1.0, 5e-4)):
        system, params, beliefs = two_rovers(ss, sm, gamma)
        n = sm.N_OPTIONS
        assert system.policies(params).shape == (n**4, 2, 2)
        gap, agree, fp = _marginal_gap(system, params, beliefs, sweeps=3)
        gaps[gamma] = gap
        print(f"gamma {gamma or system.gamma}: largest marginal gap after 3 sweeps {gap:.3g}")
        assert agree
        assert gap < bound
        assert [g.shape for g in fp.grids] == [(n**2, 2), (n**2, 2)]
    assert gaps[1.0] > 1e-5  # coupled: the gap is not zero


def test_the_sweeps_move_and_settle(survey):
    """On the coupled case at ``gamma`` 1, the movement (the largest change of a marginal in the
    last sweep) of one, two and three sweeps, measured 2026-11-18: 0.395, 9.0e-4, 8.9e-8 at five
    options; 0.377, 4.9e-4, 8.3e-7 at seven."""
    ss, sm = survey
    system, params, beliefs = two_rovers(ss, sm, 1.0)
    moves = []
    for sweeps in (1, 2, 3):
        _, _, fp = _plan(system, params, beliefs, make_factorised_planner(sweeps=sweeps))
        moves.append(float(fp.movement))
    print("movement by sweep:", moves)
    assert moves[0] > 0.1  # the first sweep leaves the uniform start
    assert moves[0] > moves[1] > moves[2]
    assert moves[1] < 5e-3 and moves[2] < 1e-5


def _intermediate_sizes(jaxpr, out):
    """Every intermediate's shape in ``jaxpr`` and the jaxprs nested in its equations."""
    for eqn in jaxpr.eqns:
        out.extend(tuple(getattr(v.aval, "shape", ())) for v in eqn.outvars)
        for param in eqn.params.values():
            for sub in param if isinstance(param, tuple | list) else (param,):
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    _intermediate_sizes(inner, out)
    return out


def test_the_other_nodes_marginals_are_read_once_per_step_not_once_per_sequence(survey):
    """The cost pin. Node ``k``'s sequences are scored as a prefix tree, the inner vmap over the
    node's ``n`` levels and the outer over the tree's nodes, with the other nodes' marginals closed
    over: a reader gathers node ``k``'s table by level once per level (``(n, 256, 256, 8)``) and
    contracts another node's table against its marginal once per step, both shared by every node
    of the tree. The first build vmapped the marginals and contracted the rover transition per
    sequence, ``(343, 256, 256, 8, 7)``: 30.8 s a decision (two rovers, horizon 3, seven
    options, three sweeps) against the enumeration planner's 0.03 s. The second closed them over
    but scored a flat scan vmapped over the sequences, which gathered the node's own table per
    sequence, ``(343, 256, 256, 8)`` (180M floats; 33 GB at 15,625 sequences): 0.46 s. The tree:
    0.01 s (measured 2026-11-18). Pinned on the jaxpr of one sweep: the largest intermediate
    batched over the sequences or over the deepest level's nodes is smaller than ``n`` copies of
    the largest unbatched one (measured 263,424 against 7 × 3,670,016; the flat scan's
    179,830,784), and none carries a table of ``cells²`` entries per sequence or node."""
    ss, sm = survey
    grid, centres = sm.field_grid()
    system, params = ss.scheduler(
        grid,
        centres,
        rovers=2,
        policy_len=ss.JOINT_HORIZON,
        durations=True,
        readings=True,
        plan_learning=("parameter:field",),
    )
    beliefs = dict(params.initial_state_prior)
    n = sm.N_OPTIONS
    count = n**ss.JOINT_HORIZON
    cells = int(centres.shape[0])
    closed = jax.make_jaxpr(
        lambda p, b: system.infer_policies(p, b, planner=make_factorised_planner(sweeps=1))
    )(params, beliefs)
    shapes = _intermediate_sizes(closed.jaxpr, [])
    lead = (count, count // n)  # the sequences, the deepest level's nodes
    per_node = [s for s in shapes if len(s) > 1 and s[0] in lead]
    batched = max(int(np.prod(s)) for s in per_node)
    unbatched = max(int(np.prod(s)) for s in shapes if not s or s[0] not in lead)
    print(f"largest per-sequence intermediate {batched:,}, largest unbatched {unbatched:,}")
    assert batched < n * unbatched
    assert not [s for s in per_node if int(np.prod(s[1:])) >= cells * cells]


# --------------------------------------------------------------------------- 5. the refusals


def test_a_joint_policy_prior_is_refused():
    system, params, beliefs = two_mazes()
    joint = dataclasses.replace(params, policy_prior=jnp.zeros((625,)))
    with pytest.raises(ValueError, match="one vector over the joint policy set"):
        _plan(system, joint, beliefs, make_factorised_planner())
    short = dataclasses.replace(params, policy_prior=(jnp.zeros((25,)),))
    with pytest.raises(ValueError, match="1 entries for 2 control node"):
        _plan(system, short, beliefs, make_factorised_planner())
    wrong = dataclasses.replace(params, policy_prior=(None, jnp.zeros((24,))))
    with pytest.raises(ValueError, match=r"policy_prior\[1\] has shape \(24,\)"):
        _plan(system, wrong, beliefs, make_factorised_planner())


def test_a_continuous_or_mixed_space_is_refused():
    system, params = toy()
    system = dataclasses.replace(system, planner="factorised")
    with pytest.raises(
        NotImplementedError, match="'factorised' planner was asked to search a space mixing"
    ):
        system.infer_policies(params, system.initial_priors(params))
    from hybrid_aif.composition import subsystem

    view, vparams = subsystem(*steered(), "mind")
    view = dataclasses.replace(view, planner="factorised")
    with pytest.raises(ValueError, match="factorised planner cannot serve a continuous"):
        view.infer_policies(vparams, view.initial_priors(vparams))


def test_a_warm_start_is_refused():
    with pytest.raises(ValueError, match=r"'factorised\(sweeps=3\)' takes no init"):
        plan(make_factorised_planner(), warm_start=True)
    system, params = shared_field(COUNTS)
    with pytest.raises(ValueError, match="takes no init"):
        run(system, params, 1, KEY, {"A": plan(factorised_planner, warm_start=True), "B": plan()})


# --------------------------------------------------------------------------- inside the runner


def test_it_plans_inside_run_with_a_trace_of_fixed_structure():
    """Two leaves of two control nodes each (``shared_field``), the first planning with the
    factorised planner at horizon 2 and learning, for three rounds of ``run`` (one compiled scan):
    the trace holds the :class:`FactorisedPolicies`, no enumerated set, and every round's carry
    has the structure of the first."""
    system, params = shared_field(COUNTS)
    target = Categorical(logits=jnp.log(jnp.asarray([0.6, 0.3, 0.1])))
    params = dataclasses.replace(params, state_targets={"A.s1": target, "B.s1": target})
    step = plan(
        make_factorised_planner(sweeps=2), learn=True, trace=True, replace={"policy_len": 2}
    )
    rec = run(
        system,
        params,
        3,
        KEY,
        {"A": step, "B": plan()},
        tick_order="sequential",
        record_carries=True,
    )
    carries = rec["A"].carries  # the reset's, then each round's
    assert len(carries) == 4
    structure = jtu.tree_structure(carries[1])
    assert all(jtu.tree_structure(c) == structure for c in carries[1:])
    trace = carries[-1].trace
    assert isinstance(trace, Trace) and trace.policies is None
    assert isinstance(trace.plans, FactorisedPolicies)
    assert trace.plans.first.shape == (9, 1, 2)
    assert [g.shape for g in trace.plans.grids] == [(9, 2), (9, 2)]
    assert trace.q_pi.probs.shape == (9,)
    assert np.isfinite(float(trace.plans.movement))
    action = carries[-1].action
    assert set(action) >= {"A.a1", "A.a2"}
    #  the registered name, through the view's planner, under jit of the round
    named = plan(trace=True, replace={"policy_len": 2, "planner": "factorised"})
    rec = run(system, params, 2, KEY, {"A": named, "B": plan()}, record_carries=True)
    trace = rec["A"].carries[-1].trace
    assert trace.policies is None and isinstance(trace.plans, FactorisedPolicies)
