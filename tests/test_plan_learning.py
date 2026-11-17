"""A parameter read within a plan: the tables ``System.plan_learning`` names are carried through
the planning scan as working counts, and a closed pair (a tabular factor reading through a cell
the mixture-carried factor that switches on it) is read jointly over the mixture's components.

Pinned by: the urn (a one-cell field with a sparse prior, a plan that stays in the cell: after an
imagined terrain ``j`` the component conditioned on it predicts ``j`` again with ``(α_j + 1) /
(α_0 + 1)``, and the frozen reading keeps the prior mean); the joint read (components in
different cells give the grow the per-component weights ``w_i q_i[j]`` and the tabular factor the
marginal ``Σ_i w_i q_i``; components in one cell reproduce the mean-field step); the merge of a
per-component side statistic through a reduce (identical components keep it, a mode's slot holds
its group's weight average, the moments bitwise as without it); the novelty of a re-read row
(the first step's leaf is the frozen one, then it declines); the mean-field form on a tabular
parent set (the working counts after ``T`` steps are ``T`` absorptions of the model's joint over
child and parents, which the product of marginals ``learn`` absorbs differs from);
no key set (no working carry); the description round trip and schema, the key under a ``merge``
composite and a ``subsystem`` view; the refusals (an unknown or non-learnable key, a parameter
variable read in a closed pair and by another learned reader). On the fused all-tabular path (A3):
the urn by ``learn`` again and the bare carry without a key; the novelty of a re-read row, and
under pooled readings its decline as ``n`` absorptions per step; under the sophisticated planner
each branch's counts (the exact conjugate update of a branched likelihood, the expected statistic
of a transition under the branch's beliefs) and a two-step Bayes-adaptive ``G`` rolled by hand;
the amortized path, which has no table to learn.

Design: docs/design/parameters-within-the-plan.md D1, D2, A3.
"""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Grid
from hybrid_aif.cells import cell_mass, component_cell_mass
from hybrid_aif.composition.merge import SEPARATOR, merge, subsystem
from hybrid_aif.controls import Box, ControlStep
from hybrid_aif.description import build, describe
from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalDirichlet,
    ConditionalGaussian,
    Gaussian,
    MixtureGaussian,
)
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianTransition, Parameter
from hybrid_aif.inference.mixture_rollout import propagate
from hybrid_aif.objectives import hybrid as _hybrid
from hybrid_aif.system import System, SystemParams, WorkingCarry

GRID = Grid(shape=(3, 3), low=-3.0, high=3.0, axes=(0, 1))
ONE_CELL = Grid(shape=(1, 1), low=-3.0, high=3.0, axes=(0, 1))
ALPHA = jnp.asarray(
    [0.1, 0.05, 0.15]
)  # a sparse prior: each cell nearly one terrain, unknown which
STAY = ControlStep(entries=(jnp.zeros(2),))


# ------------------------------------------------------------------------------- fixtures


def _rover(field, grid, plan_learning=(), epistemic=(), schedule="mode_aligned", policy_len=3):
    """A 2-D position mixture switched by three terrains (each its own drift) and moved by a
    velocity control; the terrain is drawn from the field row of the cell the position falls in,
    so ``terrain`` and ``x`` are a closed pair."""
    system = System(
        A_deps=((0,), (1,)),
        B_deps=((1,), (0, 2), ()),
        controls=(2,),
        action_names=("a",),
        modality_names=("tex", "fix"),
        state_names=("terrain", "x"),
        control_names=("u",),
        cells={"x": grid},
        policy_len=policy_len,
        epistemic=epistemic,
        plan_learning=plan_learning,
        mixture_k=3,
        mixture_schedule=schedule,
        planner="gradient",
    )
    drift = jnp.asarray([[0.3, 0.0], [0.0, 0.3], [-0.3, -0.3]])
    innovations = ConditionalGaussian(
        loc=drift, chol=jnp.broadcast_to(0.05 * jnp.eye(2), (3, 2, 2)), parent_ndims=1
    )
    params = SystemParams(
        likelihood={
            "tex": ConditionalCategorical(
                log_table=jnp.log(0.9 * jnp.eye(3) + 0.05), parent_ndims=1
            ),
            "fix": LinearGaussianObservation(H=jnp.eye(2), c=jnp.zeros(2), noise=0.3 * jnp.eye(2)),
        },
        transition={
            "terrain": ConditionalDirichlet(concentration=field, parent_ndims=1),
            "x": LinearGaussianTransition(F=jnp.eye(2), innovations=innovations, G=jnp.eye(2)),
        },
        initial_state_prior={
            "terrain": Categorical(logits=jnp.zeros(3)),
            "x": _at([[0.0, 0.0]] * 3),
        },
        control={"u": Box(-jnp.ones(2), jnp.ones(2))},
    )
    return system, params


def _at(means, sd=0.05, weights=None) -> MixtureGaussian:
    means = jnp.asarray(means, jnp.float32)
    k = means.shape[0]
    w = jnp.ones(k) / k if weights is None else jnp.asarray(weights)
    return MixtureGaussian(
        log_weights=jnp.log(w), means=means, chols=jnp.broadcast_to(sd * jnp.eye(2), (k, 2, 2))
    )


def _one_cell(**kw):
    return _rover(ALPHA[:, None], ONE_CELL, **kw)


def _field() -> jnp.ndarray:
    """Cell 0 prefers terrain 0, cell 8 terrain 2, cell 4 terrain 1, the rest uniform."""
    conc = jnp.ones((3, 9))
    conc = conc.at[:, 0].set(jnp.asarray([8.0, 1.0, 1.0]))
    conc = conc.at[:, 4].set(jnp.asarray([1.0, 6.0, 1.0]))
    return conc.at[:, 8].set(jnp.asarray([1.0, 1.0, 6.0]))


def _tabular(
    plan_learning=("transition:s",), epistemic=("transition:s",), planner="enumeration", **kw
):
    """A hybrid System whose categorical factor ``s`` has a learnable transition over a
    categorical parent ``c`` and a discrete control ``u``; ``y`` is a Gaussian beside them."""
    system = System(
        A_deps=((0,), (2,)),
        B_deps=((1, 3), (1,), (), ()),
        controls=(3,),
        action_names=("a",),
        state_names=("s", "c", "y"),
        control_names=("u",),
        modality_names=("o", "fix"),
        policy_len=3,
        planner=planner,
        epistemic=epistemic,
        plan_learning=plan_learning,
        **kw,
    )
    conc = jnp.full((3, 2, 2), 0.2).at[0, 0, 0].set(3.0)
    params = SystemParams(
        likelihood={
            "o": ConditionalCategorical(
                log_table=jnp.log(0.8 * jnp.eye(3) + 0.2 / 3), parent_ndims=1
            ),
            "fix": LinearGaussianObservation(H=jnp.eye(1), c=jnp.zeros(1), noise=0.3 * jnp.eye(1)),
        },
        transition={
            "s": ConditionalDirichlet(concentration=conc, parent_ndims=2),
            "c": ConditionalCategorical(
                log_table=jnp.log(jnp.asarray([[0.7, 0.4], [0.3, 0.6]])), parent_ndims=1
            ),
            "y": LinearGaussianTransition(
                F=jnp.eye(1),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 1)), chol=0.1 * jnp.eye(1)[None], parent_ndims=1
                ),
            ),
        },
        initial_state_prior={
            "s": Categorical(logits=jnp.zeros(3)),
            "c": Categorical(logits=jnp.log(jnp.asarray([0.8, 0.2]))),
            "y": Gaussian(loc=jnp.zeros(1), chol=jnp.eye(1)),
        },
    )
    return system, params


def _per_step_leaves(ctx, actions, key="transition:terrain"):
    """The novelty leaf of each step of one plan, from the context's own closures."""
    carry, leaves = ctx.init_beliefs, []
    for t, action in enumerate(actions):
        carry = ctx.transition_fn(carry, action)
        leaves.append(
            float(ctx.step_terms_fn(carry, action, jnp.asarray(t))["infogain"][key].lower)
        )
    return np.asarray(leaves)


# ---------------------------------------------------------------------------- 1. the urn


def test_after_an_imagined_terrain_the_component_conditioned_on_it_predicts_it_by_the_urn():
    alpha0 = float(ALPHA.sum())
    system, params = _one_cell(plan_learning=("transition:terrain",))
    ctx = system.planner_context(params, system.initial_priors(params))
    assert isinstance(ctx.init_beliefs, WorkingCarry)
    carry = ctx.transition_fn(ctx.init_beliefs, STAY)
    counts = np.asarray(carry.counts["transition:terrain"])  # (K, L, cells): one slot per mode
    assert counts.shape == (3, 3, 1)
    for j in range(3):
        np.testing.assert_allclose(counts[j, :, 0], np.asarray(ALPHA) + np.eye(3)[j], rtol=1e-5)
    #  the slots' weights are the modes' marginal, the prior mean
    weights = np.asarray(jax.nn.softmax(carry.beliefs[1].log_weights))
    np.testing.assert_allclose(weights, np.asarray(ALPHA) / alpha0, rtol=1e-5)
    #  each component's predictive of the next terrain, as the next step reads it
    qs = carry.beliefs
    learned = _hybrid.pair_read(
        jnp.log(counts / counts.sum(axis=1, keepdims=True)),
        1,
        0,
        1,
        system._B,
        _hybrid.cell_view(qs, {1: ONE_CELL}),
        qs[1],
        ONE_CELL,
        (),
    ).child
    for j in range(3):
        assert float(learned[j, j]) == pytest.approx((float(ALPHA[j]) + 1) / (alpha0 + 1), rel=1e-5)
    #  the urn is a martingale: the next step's marginal is still the prior mean
    after = ctx.transition_fn(carry, STAY)
    np.testing.assert_allclose(
        np.asarray(after.beliefs[0].probs), np.asarray(ALPHA) / alpha0, rtol=1e-4
    )
    #  read frozen, every component predicts the prior mean
    frozen_system, frozen_params = _one_cell()
    frozen = frozen_system.planner_context(frozen_params, frozen_system.initial_priors(params))
    assert not isinstance(frozen.init_beliefs, WorkingCarry)
    fq = frozen.transition_fn(frozen.init_beliefs, STAY)
    mean = jnp.log(ALPHA / alpha0)[:, None]
    child = _hybrid.pair_read(
        mean, 1, 0, 1, system._B, _hybrid.cell_view(fq, {1: ONE_CELL}), fq[1], ONE_CELL, ()
    ).child
    np.testing.assert_allclose(np.asarray(child), np.tile(np.asarray(ALPHA) / alpha0, (3, 1)))


# ------------------------------------------------------------------------ 2. the joint read


def _frozen_step(mixture, grid=GRID, schedule="mode_aligned"):
    system, params = _rover(_field(), grid, schedule=schedule)
    beliefs = {"terrain": Categorical(logits=jnp.zeros(3)), "x": mixture}
    ctx = system.planner_context(params, beliefs)
    return ctx.transition_fn(ctx.init_beliefs, STAY), params


def _grow(mixture, log_weights, params, schedule):
    B = params.transition["x"]
    grown = propagate(
        mixture,
        jnp.broadcast_to(jnp.eye(2), (3, 2, 2)),
        B.innovations.loc,
        B.innovations.chol,
        log_weights,
    )
    if schedule == "mode_aligned":
        return grown.reduce_mode_aligned(3, 3)
    return grown.reduce(3)


@pytest.mark.parametrize("schedule", ["mode_aligned", "runnalls"])
def test_components_in_different_cells_grow_with_their_own_terrain_beliefs(schedule):
    mixture = _at([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]], weights=[0.5, 0.3, 0.2])
    new, params = _frozen_step(mixture, schedule=schedule)
    table = np.asarray(_field() / _field().sum(axis=0, keepdims=True))
    c = np.asarray(component_cell_mass(mixture, GRID))  # (K, cells)
    q = c @ table.T  # (K, L): each component's terrain belief
    w = np.asarray([0.5, 0.3, 0.2])
    np.testing.assert_allclose(np.asarray(new[0].probs), w @ q, rtol=1e-5)
    #  the switching factor grew with the weights w_i q_i[j], not w_i q[j]
    want = _grow(mixture, jnp.log(jnp.asarray(q)), params, schedule)
    for got, ref in zip(jax.tree.leaves(new[1]), jax.tree.leaves(want), strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-5, atol=1e-6)
    mean_field = _grow(mixture, jnp.log(jnp.asarray(w @ q)), params, schedule)
    assert not np.allclose(np.asarray(new[1].means), np.asarray(mean_field.means), atol=1e-3)


def test_components_in_one_cell_reproduce_the_mean_field_step():
    mixture = _at([[-2.1, -2.0], [-1.9, -2.2], [-2.0, -1.8]], weights=[0.5, 0.3, 0.2])
    new, params = _frozen_step(mixture)
    table = _field() / _field().sum(axis=0, keepdims=True)
    q = table @ cell_mass(mixture, GRID)
    want = _grow(mixture, jnp.log(q), params, "mode_aligned")
    np.testing.assert_allclose(np.asarray(new[0].probs), np.asarray(q), rtol=1e-5)
    for got, ref in zip(jax.tree.leaves(new[1]), jax.tree.leaves(want), strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------------ 3. the merged statistic


def _mixture(n, key=0):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(key), 3)
    return MixtureGaussian(
        log_weights=jax.random.normal(k1, (n,)),
        means=jax.random.normal(k2, (n, 2)),
        chols=jnp.broadcast_to(jnp.eye(2), (n, 2, 2)) * (1.0 + jax.random.uniform(k3, (n, 1, 1))),
    )


def test_identical_components_keep_their_statistic_through_a_reduce():
    m = MixtureGaussian(
        log_weights=jnp.log(jnp.asarray([0.1, 0.2, 0.3, 0.4])),
        means=jnp.zeros((4, 2)),
        chols=jnp.broadcast_to(jnp.eye(2), (4, 2, 2)),
    )
    stat = jnp.broadcast_to(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]), (4, 2, 2))
    _, merged = m.reduce(2, aux=stat)
    np.testing.assert_allclose(np.asarray(merged), np.asarray(stat[:2]), rtol=1e-6)
    _, aligned = m.reduce_mode_aligned(2, 2, aux=stat)
    np.testing.assert_allclose(np.asarray(aligned), np.asarray(stat[:2]), rtol=1e-6)


def test_with_one_slot_per_mode_the_merged_statistic_is_the_group_weight_average():
    m = _mixture(12)  # component-major over 3 modes: slot s from mode s % 3
    stat = jax.random.uniform(jax.random.PRNGKey(7), (12, 3, 4))
    _, merged = m.reduce_mode_aligned(3, 3, aux=stat)
    w = np.asarray(jax.nn.softmax(m.log_weights))
    for g in range(3):
        members = np.arange(g, 12, 3)
        share = w[members] / w[members].sum()
        want = np.tensordot(share, np.asarray(stat)[members], axes=1)
        np.testing.assert_allclose(np.asarray(merged[g]), want, rtol=1e-5)


@pytest.mark.parametrize("reduce", ["runnalls", "mode_aligned"])
def test_a_side_statistic_leaves_the_moments_bitwise_as_without_it(reduce):
    m = _mixture(12, key=3)
    stat = jax.random.uniform(jax.random.PRNGKey(1), (12, 5))

    def run(m, aux):
        if reduce == "runnalls":
            return m.reduce(4, aux=aux)
        return m.reduce_mode_aligned(6, 3, aux=aux)

    plain = jax.jit(lambda m: run(m, None))(m)
    with_aux, merged = jax.jit(run)(m, stat)
    for a, b in zip(jax.tree.leaves(plain), jax.tree.leaves(with_aux), strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert merged.shape[1:] == (5,)
    #  batched mixtures carry the statistic per batch entry
    batched = jax.tree.map(lambda a: jnp.stack([a, a]), m)
    _, per = run(batched, jnp.stack([stat, stat]))
    np.testing.assert_allclose(np.asarray(per[0]), np.asarray(merged), rtol=1e-6)


# ------------------------------------------------------------ 4. novelty along a re-read row


def test_the_novelty_of_a_re_read_row_starts_at_the_frozen_leaf_and_declines():
    actions = [STAY] * 4
    learned_system, params = _one_cell(
        plan_learning=("transition:terrain",), epistemic=("transition:terrain",)
    )
    frozen_system, _ = _one_cell(epistemic=("transition:terrain",))
    priors = learned_system.initial_priors(params)
    learned = _per_step_leaves(learned_system.planner_context(params, priors), actions)
    frozen = _per_step_leaves(frozen_system.planner_context(params, priors), actions)
    assert learned[0] == pytest.approx(frozen[0], rel=1e-5)
    assert np.all(np.diff(learned) < 0.0)
    assert np.all(learned[1:] < frozen[1:])


def test_the_mean_field_novelty_of_a_re_read_row_declines_too():
    system, params = _tabular()
    frozen, _ = _tabular(plan_learning=())
    priors = system.initial_priors(params)
    actions = [jnp.asarray([0])] * 3
    learned = _per_step_leaves(system.planner_context(params, priors), actions, "transition:s")
    plain = _per_step_leaves(frozen.planner_context(params, priors), actions, "transition:s")
    assert learned[0] == pytest.approx(plain[0], rel=1e-6)
    assert np.all(learned[1:] < plain[1:])


# ------------------------------------------- 5. the mean-field form, the joint by hand


def _joint_absorptions(conc, steps):
    """``conc`` after one absorption per step of the model's joint over ``(s', c, u)``: the
    working mean ``B(s' | c, u)`` times ``q(c)`` before the step and the taken level's one-hot."""
    n = jnp.asarray(conc)
    for q_c, level in steps:
        mean = n / n.sum(axis=0, keepdims=True)
        n = n + mean * (q_c[:, None] * jax.nn.one_hot(level, n.shape[-1]))[None]
    return n


def _probs(q):
    return q.probs if isinstance(q, Categorical) else jnp.asarray(q)


def _belief(q):
    """A carried belief as ``learn`` takes it: a fused path's probability vector as a
    Categorical, a distribution as it is."""
    return _cat(q) if isinstance(q, jnp.ndarray) else q


def _absorbs_the_joint(system, params):
    """Drive ``system``'s plan over four steps; its counts against the joint absorbed by hand,
    and the product of marginals ``learn`` absorbs from the same beliefs."""
    ctx = system.planner_context(params, system.initial_priors(params))
    names = system.state_names
    carry = ctx.init_beliefs[0]  # (WorkingCarry, previous): a novelty leaf is mounted
    assert isinstance(carry, WorkingCarry)
    learned, prev, steps = params, carry.beliefs, []
    for level in (0, 1, 1, 0):
        action = jnp.asarray([level])
        steps.append((_probs(prev[1]), level))
        carry = ctx.transition_fn((carry, carry), action)[0]
        learned = system.learn(
            learned,
            {"o": jnp.asarray(0), "fix": jnp.zeros(1)}
            if "fix" in system.modality_names
            else {"o": jnp.asarray(0)},
            {n: _belief(q) for n, q in zip(names, carry.beliefs, strict=True)},
            prev_beliefs={n: _belief(q) for n, q in zip(names, prev, strict=True)},
            action={"a": action[0]},
        )
        prev = carry.beliefs
    want = _joint_absorptions(params.transition["s"].concentration, steps)
    counts = np.asarray(carry.counts["transition:s"])
    np.testing.assert_allclose(counts, np.asarray(want), rtol=1e-5)
    return counts, np.asarray(learned.transition["s"].concentration)


def test_the_working_counts_after_t_steps_are_t_absorptions_of_the_joint():
    counts, _ = _absorbs_the_joint(*_tabular())
    #  the joint leaves each column's mean where it was: only the concentration grows
    conc = _tabular()[1].transition["s"].concentration
    np.testing.assert_allclose(
        counts / counts.sum(axis=0, keepdims=True),
        np.asarray(conc / conc.sum(axis=0, keepdims=True)),
        rtol=1e-5,
    )


def test_the_product_of_marginals_learn_absorbs_differs_from_the_joint():
    """The closed-loop face's statistic ``q(s') ⊗ q(c) ⊗ u`` is the joint only where ``s'`` and
    ``c`` are independent under the step; the table of ``s`` depends on ``c`` under ``u = 0``,
    so the two differ there by the mutual information, and agree in the uniform ``u = 1``
    columns."""
    counts, by_learn = _absorbs_the_joint(*_tabular())
    assert not np.allclose(counts[..., 0], by_learn[..., 0], rtol=1e-3)
    np.testing.assert_allclose(counts[..., 1], by_learn[..., 1], rtol=1e-5)


def test_a_likelihood_learned_within_the_plan_absorbs_its_expected_observation():
    system, params = _tabular(plan_learning=("likelihood:o",), epistemic=("likelihood:o",))
    table = ConditionalDirichlet(
        concentration=jnp.full((3, 3), 0.2).at[0, 0].set(4.0), parent_ndims=1
    )
    params = dataclasses.replace(params, likelihood={**params.likelihood, "o": table})
    frozen = dataclasses.replace(system, plan_learning=())
    priors = system.initial_priors(params)
    ctx = system.planner_context(params, priors)
    carry, want = ctx.init_beliefs, table
    for _ in range(3):
        carry = ctx.transition_fn(carry, jnp.asarray([0]))
        s = carry.beliefs[0]
        want = want.absorb(want.mean() * s.probs[None, :])  # the joint A(o | s) q(s)
    np.testing.assert_allclose(
        np.asarray(carry.counts["likelihood:o"]), np.asarray(want.concentration), rtol=1e-5
    )
    actions = [jnp.asarray([0])] * 3
    learned = _per_step_leaves(ctx, actions, "likelihood:o")
    plain = _per_step_leaves(frozen.planner_context(params, priors), actions, "likelihood:o")
    assert learned[0] == pytest.approx(plain[0], rel=1e-6)
    assert np.all(learned[1:] < plain[1:])


def test_the_planners_run_through_the_working_carry():
    system, params = _tabular()
    priors = system.initial_priors(params)
    q, tree = system.infer_policies(params, priors, metrics=True)
    assert np.isfinite(np.asarray(q.probs)).all() and "transition:s" in tree["infogain"]
    branching, _ = _tabular(planner="sophisticated", branch_on=("o",))
    q, _ = branching.infer_policies(params, priors)
    assert np.isfinite(np.asarray(q.probs)).all()
    rover, rover_params = _rover(
        _field(), GRID, plan_learning=True, epistemic=("transition:terrain",), policy_len=2
    )
    q, tree = rover.infer_policies(rover_params, rover.initial_priors(rover_params), metrics=True)
    assert np.isfinite(np.asarray(tree["infogain"]["transition:terrain"].lower)).all()


# -------------------------------------------------------------------------- 6. no key set


def test_no_key_set_or_no_learnable_table_keeps_the_bare_carry():
    system, params = _tabular(plan_learning=(), epistemic=())
    ctx = system.planner_context(params, system.initial_priors(params))
    assert isinstance(ctx.init_beliefs, tuple) and not isinstance(ctx.init_beliefs, WorkingCarry)
    fixed = dataclasses.replace(
        params,
        transition={**params.transition, "s": params.transition["s"].point()},
    )
    every, _ = _tabular(plan_learning=True, epistemic=())
    ctx = every.planner_context(fixed, every.initial_priors(fixed))
    assert not isinstance(ctx.init_beliefs, WorkingCarry)
    assert every._plan_learning(fixed) == ()


# ------------------------------------------------- 7. the description, merge and subsystem


@pytest.mark.parametrize("plan_learning", [("transition:s",), True])
def test_the_setting_round_trips_through_the_description_under_inference(plan_learning):
    system, params = _tabular(plan_learning=plan_learning)
    document, arrays = describe(system, params, arrays="inline")
    document = json.loads(json.dumps(document))
    written = document["configuration"]["inference"]["plan_learning"]
    assert written == (True if plan_learning is True else list(plan_learning))
    built, _ = build(document, arrays=arrays)
    assert built.plan_learning == plan_learning
    plain, plain_params = _tabular(plan_learning=())
    plain_doc = json.loads(json.dumps(describe(plain, plain_params, arrays="inline")[0]))
    assert "plan_learning" not in plain_doc.get("configuration", {}).get("inference", {})
    jsonschema = pytest.importorskip("jsonschema")
    from hybrid_aif import schema

    validator = jsonschema.Draft202012Validator(schema())
    validator.validate(document)
    broken = json.loads(json.dumps(document))
    broken["configuration"]["inference"]["plan_learning"] = "transition:s"
    assert list(validator.iter_errors(broken))
    with pytest.raises(ValueError, match="plan_learning"):
        build(broken, arrays=arrays)


def test_the_keys_compose_under_merge_and_narrow_under_a_subsystem_view():
    a, pa = _tabular()
    b, pb = _tabular(plan_learning=())
    composite, params = merge({"a": a, "b": b}, {"a": pa, "b": pb})
    sep = SEPARATOR
    assert composite.plan_learning == (f"transition:a{sep}s",)
    reader = composite._plan_learning(composite._p(params))
    assert [key for key, _ in reader] == [f"transition:a{sep}s"]
    view_b, _ = subsystem(composite, params, "b")
    assert view_b.plan_learning == ()
    view_a, _ = subsystem(composite, params, "a")
    assert view_a.plan_learning == (f"transition:a{sep}s",)
    c, pc = _tabular()
    both, _ = merge({"a": a, "c": c}, {"a": pa, "c": pc})
    assert set(both.plan_learning) == {f"transition:a{sep}s", f"transition:c{sep}s"}


# ---------------------------------------------------------------------------- 8. refusals


@pytest.mark.parametrize(
    ("key", "match"),
    [
        ("transition:nope", "unknown key"),
        ("transition:y", "not a learnable table"),
        ("transition:c", "not a learnable table"),
        ("parameter:nope", "unknown parameter variable"),
        ("transition:*", "pattern"),
    ],
)
def test_a_key_naming_no_learnable_table_is_refused_with_a_sentence(key, match):
    system, params = _tabular(plan_learning=(key,))
    with pytest.raises(ValueError, match=match):
        system.infer_policies(params, system.initial_priors(params))


def test_a_malformed_setting_is_refused_at_construction():
    with pytest.raises(ValueError, match="plan_learning must be"):
        _tabular(plan_learning="transition:s")


def test_a_variable_read_in_a_closed_pair_and_by_another_learned_reader_is_refused():
    system, params = _rover(_field(), GRID, plan_learning=("parameter:field",))
    system = System(
        A_deps=((0,), (1,)),
        B_deps=((1,), (0, 3), (1,), ()),
        controls=(3,),
        action_names=("a",),
        modality_names=("tex", "fix"),
        state_names=("terrain", "x", "map"),
        control_names=("u",),
        cells={"x": GRID},
        epistemic=("transition:terrain",),
        plan_learning=("parameter:field",),
        mixture_k=3,
        mixture_schedule="mode_aligned",
        planner="gradient",
        policy_len=2,
    )
    field = params.transition["terrain"]
    params = dataclasses.replace(
        params,
        parameters={"field": field},
        transition={**params.transition, "terrain": Parameter("field"), "map": Parameter("field")},
        initial_state_prior={**params.initial_state_prior, "map": Categorical(logits=jnp.zeros(3))},
    )
    with pytest.raises(NotImplementedError, match="closed pair"):
        system.infer_policies(params, system.initial_priors(params))
    #  learning only the closed pair's reader builds
    one = dataclasses.replace(system, plan_learning=("transition:terrain",))
    q, _ = one.infer_policies(params, one.initial_priors(params))
    assert np.isfinite(np.asarray(q.probs)).all()


# ---------------------------------------------------------- A3. the fused all-tabular path


def _fused(
    plan_learning=("transition:s",),
    epistemic=("transition:s",),
    planner="enumeration",
    conc=None,
    **kw,
):
    """``_tabular`` without its Gaussian: an all-tabular System, planned on the fused path. ``s``
    has a learnable transition over the categorical parent ``c`` and the control ``u``."""
    system = System(
        A_deps=((0,),),
        B_deps=((1, 2), (1,), ()),
        controls=(2,),
        action_names=("a",),
        state_names=("s", "c"),
        control_names=("u",),
        modality_names=("o",),
        policy_len=3,
        planner=planner,
        epistemic=epistemic,
        plan_learning=plan_learning,
        **kw,
    )
    conc = jnp.full((3, 2, 2), 0.2).at[0, 0, 0].set(3.0) if conc is None else conc
    params = SystemParams(
        likelihood={
            "o": ConditionalCategorical(
                log_table=jnp.log(0.8 * jnp.eye(3) + 0.2 / 3), parent_ndims=1
            )
        },
        transition={
            "s": ConditionalDirichlet(concentration=conc, parent_ndims=2),
            "c": ConditionalCategorical(
                log_table=jnp.log(jnp.asarray([[0.7, 0.4], [0.3, 0.6]])), parent_ndims=1
            ),
        },
        initial_state_prior={
            "s": Categorical(logits=jnp.zeros(3)),
            "c": Categorical(logits=jnp.log(jnp.asarray([0.8, 0.2]))),
        },
    )
    return system, params


def _cat(q):
    return Categorical(logits=jnp.log(q))


def _wnorm(conc):
    """The novelty weights of a count table (child axis first), as ``ConditionalDirichlet``."""
    return ConditionalDirichlet(concentration=conc, parent_ndims=conc.ndim - 1).novelty()


def test_on_the_fused_path_the_working_counts_after_t_steps_are_t_absorptions_of_the_joint():
    system, params = _fused()
    assert not system._is_hybrid(params)
    counts, by_learn = _absorbs_the_joint(system, params)
    assert not np.allclose(counts[..., 0], by_learn[..., 0], rtol=1e-3)
    #  the fused path and the hybrid path absorb the same counts from the same plan
    hybrid, hybrid_params = _tabular()
    hctx = hybrid.planner_context(hybrid_params, hybrid.initial_priors(hybrid_params))
    hcarry = hctx.init_beliefs[0]
    for level in (0, 1, 1, 0):
        hcarry = hctx.transition_fn((hcarry, hcarry), jnp.asarray([level]))[0]
    np.testing.assert_allclose(counts, np.asarray(hcarry.counts["transition:s"]), rtol=1e-5)


def test_on_the_fused_path_without_a_key_the_carry_is_bare_and_g_reads_the_hoisted_tables():
    system, params = _fused(plan_learning=())
    priors = system.initial_priors(params)
    ctx = system.planner_context(params, priors)
    assert not isinstance(ctx.init_beliefs[0], WorkingCarry)
    learned, _ = _fused()
    q_plain, plain = system.infer_policies(params, priors, metrics=True)
    q_learned, tree = learned.infer_policies(params, priors, metrics=True)
    #  the plans differ only past their first step, where the learned plan reads its counts
    assert not np.allclose(np.asarray(q_plain.probs), np.asarray(q_learned.probs))
    assert set(tree["infogain"]) == set(plain["infogain"])


def test_on_the_fused_path_the_novelty_of_a_re_read_row_starts_at_the_hoisted_leaf_and_declines():
    system, params = _fused()
    frozen, _ = _fused(plan_learning=())
    #  c starts at its stationary belief, so every step reads the same rows of s's table
    stationary = Categorical(logits=jnp.log(jnp.asarray([4.0, 3.0]) / 7.0))
    params = dataclasses.replace(
        params, initial_state_prior={**params.initial_state_prior, "c": stationary}
    )
    priors = system.initial_priors(params)
    actions = [jnp.asarray([0])] * 4
    learned = _per_step_leaves(system.planner_context(params, priors), actions, "transition:s")
    plain = _per_step_leaves(frozen.planner_context(params, priors), actions, "transition:s")
    assert learned[0] == plain[0]  # the first step reads the posterior: the hoisted table
    assert np.all(np.diff(learned) < 0.0)
    assert np.all(learned[1:] < plain[1:])


def test_on_the_fused_path_a_pooled_reading_absorbs_as_its_count_of_readings():
    n = 3.0
    pooled, params = _fused(readings=(("s", "u"),))
    params = dataclasses.replace(params, readings={"s": jnp.asarray([n, 1.0])})
    single, single_params = _fused()
    priors = pooled.initial_priors(params)
    ctx = pooled.planner_context(params, priors)
    carry = ctx.init_beliefs
    leaves = []
    for t in range(3):
        action = jnp.asarray([0])
        prev = carry[0].beliefs
        carry = ctx.transition_fn(carry, action)
        now = carry[0]
        #  the step stands for n readings: n times the step's joint over (s', c, u)
        before = now.before["transition:s"]
        joint = _joint_absorptions(before, [(prev[1], 0)]) - before
        np.testing.assert_allclose(
            np.asarray(now.counts["transition:s"] - before),
            n * np.asarray(joint),
            rtol=1e-5,
            atol=1e-7,
        )
        tree = ctx.step_terms_fn(carry, action, jnp.asarray(t))
        leaf = float(tree["infogain"]["transition:s"].lower)
        #  the leaf is n times the novelty of the counts before the step
        w = _wnorm(before)[:, :, 0] @ prev[1]
        assert leaf == pytest.approx(n * float(now.beliefs[0] @ w), rel=1e-5)
        leaves.append(leaf / n)
    one = _per_step_leaves(
        single.planner_context(single_params, priors), [jnp.asarray([0])] * 3, "transition:s"
    )
    assert leaves[0] == pytest.approx(one[0], rel=1e-6)
    assert np.all(np.asarray(leaves[1:]) < one[1:])  # n absorptions a step: a faster decline


def _bandit(
    plan_learning=("likelihood:o",),
    conc=((2.0, 0.5), (0.5, 1.0)),
    planner="sophisticated",
    branch_on=("o",),
):
    """Two states, a control that stays or flips, a learnable likelihood branched on: the
    smallest Bayes-adaptive problem the sophisticated planner solves."""
    system = System(
        A_deps=((0,),),
        B_deps=((0,),),
        controlled=(True,),
        policy_len=2,
        state_names=("s",),
        modality_names=("o",),
        epistemic=("likelihood:o",),
        plan_learning=plan_learning,
        planner=planner,
        branch_on=branch_on,
    )
    flip = jnp.stack([jnp.eye(2), jnp.eye(2)[::-1]], axis=-1)  # (s', s, u)
    params = SystemParams(
        likelihood={"o": ConditionalDirichlet(concentration=jnp.asarray(conc), parent_ndims=1)},
        transition={
            "s": ConditionalCategorical(log_table=jnp.log(jnp.clip(flip, 1e-9)), parent_ndims=2)
        },
        initial_state_prior={"s": Categorical(logits=jnp.log(jnp.asarray([0.7, 0.3])))},
        obs_targets={"o": Categorical(logits=jnp.log(jnp.asarray([0.6, 0.4])))},
    )
    return system, params


def test_each_branch_absorbs_its_outcome_by_the_exact_conjugate_update():
    system, params = _bandit()
    ctx = system.planner_context(params, system.initial_priors(params))
    carry = ctx.transition_fn(ctx.init_beliefs, jnp.asarray([1]))
    p, after = ctx.branch_fn(carry)
    before = np.asarray(carry.before["likelihood:o"])
    mean = before / before.sum(axis=0, keepdims=True)
    q = np.asarray(carry.beliefs[0])
    np.testing.assert_allclose(np.asarray(p), mean @ q, rtol=1e-6)
    for o in range(2):
        post = mean[o] * q / (mean[o] @ q)
        want = before + np.eye(2)[o][:, None] * post[None, :]
        np.testing.assert_allclose(np.asarray(after.counts["likelihood:o"][o]), want, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(after.beliefs[0][o]), post, rtol=1e-6)
    #  the step's expected statistic, the joint A(o | s) q(s), is the outcome-average of the
    #  branches' exact updates; the product q(o) ⊗ q(s) is not
    np.testing.assert_allclose(np.asarray(carry.counts["likelihood:o"]) - before, mean * q[None, :])
    np.testing.assert_allclose(
        np.tensordot(np.asarray(p), np.asarray(after.counts["likelihood:o"]), axes=1),
        np.asarray(carry.counts["likelihood:o"]),
        rtol=1e-6,
    )
    assert not np.allclose(mean * q[None, :], np.outer(mean @ q, q), rtol=1e-3)


def test_the_sophisticated_g_is_the_two_step_bayes_adaptive_value():
    system, params = _bandit()
    ctx = system.planner_context(params, system.initial_priors(params))
    gamma = float(ctx.gamma)
    log_c = np.log(np.asarray([0.6, 0.4]))
    flip = [np.eye(2), np.eye(2)[::-1]]

    def step(q, n):
        """G of one step at belief ``q`` with counts ``n``: minus the utility, minus the
        likelihood's novelty."""
        mean = n / n.sum(axis=0, keepdims=True)
        q_o = mean @ q
        novelty = q_o @ (np.asarray(_wnorm(jnp.asarray(n, jnp.float32))) @ q)
        return -(q_o @ log_c) - novelty

    def value(q, n):
        g = np.asarray([step(flip[a] @ q, n) for a in range(2)])
        w = np.exp(-gamma * (g - g.min()))
        return (w / w.sum()) @ g

    q0 = np.asarray([0.7, 0.3])
    n0 = np.asarray([[2.0, 0.5], [0.5, 1.0]])
    want = []
    for a in range(2):
        q1 = flip[a] @ q0
        mean = n0 / n0.sum(axis=0, keepdims=True)
        tail = 0.0
        for o in range(2):
            p_o = mean[o] @ q1
            post = mean[o] * q1 / p_o
            tail += p_o * value(post, n0 + np.eye(2)[o][:, None] * post[None, :])
        want.append(step(q1, n0) + tail)
    q_pi, _ = system.infer_policies(params, system.initial_priors(params))
    want = np.asarray(want)
    np.testing.assert_allclose(
        np.asarray(q_pi.probs), np.exp(-gamma * want) / np.exp(-gamma * want).sum(), rtol=1e-5
    )
    #  read frozen, the tail is the same recursion over the declared counts
    frozen, _ = _bandit(plan_learning=())
    q_frozen, _ = frozen.infer_policies(params, frozen.initial_priors(params))
    assert not np.allclose(np.asarray(q_frozen.probs), np.asarray(q_pi.probs), rtol=1e-4)


def test_under_the_sophisticated_planner_a_learned_transition_absorbs_the_branch_beliefs():
    system, params = _fused(planner="sophisticated", branch_on=("o",))
    ctx = system.planner_context(params, system.initial_priors(params))
    carry = ctx.init_beliefs  # (WorkingCarry, previous)
    prev = carry[0].beliefs
    carry = ctx.transition_fn(carry, jnp.asarray([0]))
    p, after = ctx.branch_fn(carry)
    now = carry[0]
    table = np.asarray(0.8 * np.eye(3) + 0.2 / 3)
    before = np.asarray(now.before["transition:s"])
    mean = before / before.sum(axis=0, keepdims=True)
    #  the step's joint over (s', c, u = 0), conditioned on each outcome: exactly, since o reads
    #  s' alone, p(s', c | o) ∝ A(o | s') B(s' | c, u) q(c)
    joint = mean * (np.asarray(prev[1])[:, None] * np.eye(2)[0][None, :])[None]
    for o in range(3):
        post = table[o] * np.asarray(now.beliefs[0])
        post = post / post.sum()
        np.testing.assert_allclose(np.asarray(after[0].beliefs[0][o]), post, rtol=1e-5)
        conditioned = table[o][:, None, None] * joint
        want = before + conditioned / conditioned.sum()
        np.testing.assert_allclose(
            np.asarray(after[0].counts["transition:s"][o]), want, rtol=1e-5, atol=1e-7
        )
    #  averaged over the outcomes, the branches absorb the step's joint
    np.testing.assert_allclose(
        np.tensordot(np.asarray(p), np.asarray(after[0].counts["transition:s"]), axes=1),
        np.asarray(now.counts["transition:s"]),
        rtol=1e-5,
    )
    q, _ = system.infer_policies(params, system.initial_priors(params))
    assert np.isfinite(np.asarray(q.probs)).all()


def test_the_fused_path_plans_with_a_key_and_the_amortized_path_has_no_table_to_learn():
    system, params = _fused(plan_learning=True)
    for planner in ("enumeration", "sophisticated"):
        q, _ = dataclasses.replace(system, planner=planner).infer_policies(
            params, system.initial_priors(params)
        )
        assert np.isfinite(np.asarray(q.probs)).all()
    #  a likelihood learned within the plan under present= stays refused, as on the hybrid path
    learned, learned_params = _bandit(planner="enumeration", branch_on=())
    with pytest.raises(NotImplementedError, match="present="):
        learned.infer_policies(
            learned_params, learned.initial_priors(learned_params), present={"o": True}
        )
    #  the amortized path's transitions are networks: True resolves to no table, a key refuses
    from tests.test_amortized_objective import agent

    amortized, amortized_params, _ = agent()
    every = dataclasses.replace(amortized, plan_learning=True)
    assert every._plan_learning(every._p(amortized_params)) == ()
    named = dataclasses.replace(amortized, plan_learning=("likelihood:reward",))
    with pytest.raises(ValueError, match="not a learnable table"):
        named._plan_learning(named._p(amortized_params))


def test_on_the_fused_path_the_occupancy_row_reads_the_working_posterior():
    def leaves(plan_learning):
        system, params = _fused(plan_learning=plan_learning, terms=("occupancy:transition:s",))
        ctx = system.planner_context(params, system.initial_priors(params))
        carry, out = ctx.init_beliefs, []
        for t in range(3):
            action = jnp.asarray([0])
            carry = ctx.transition_fn(carry, action)
            before = (carry[0].before["transition:s"], carry[0].beliefs) if plan_learning else None
            tree = ctx.step_terms_fn(carry, action, jnp.asarray(t))
            out.append((float(tree["occupancy"]["transition:s"].lower), before))
        return out

    learned, frozen = leaves(True), leaves(())
    assert learned[0][0] == frozen[0][0]
    prev = _fused()[1].initial_state_prior["c"].probs
    for value, (before, beliefs) in learned:
        #  E_q(c) H[B(· | c, u=0)] over the mean of the counts before the step
        table = before[:, :, 0] / before[:, :, 0].sum(axis=0, keepdims=True)
        entropy = -(table * jnp.log(table)).sum(axis=0)
        assert value == pytest.approx(float(entropy @ prev), rel=1e-5)
        prev = beliefs[1]
    #  along an unbranched plan the joint leaves each column's mean where it was
    assert learned[1][0] == pytest.approx(frozen[1][0], rel=1e-5)
    #  after a branch the mean moves, and the row reads the branch's counts
    system, params = _fused(
        terms=("occupancy:transition:s",), planner="sophisticated", branch_on=("o",)
    )
    ctx = system.planner_context(params, system.initial_priors(params))
    action = jnp.asarray([0])
    carry = ctx.transition_fn(ctx.init_beliefs, action)
    _, branched = ctx.branch_fn(carry)
    carry = jax.tree.map(lambda x: x[0], branched)  # the branch that saw o = 0
    q_c = carry[0].beliefs[1]
    carry = ctx.transition_fn(carry, action)
    value = float(
        ctx.step_terms_fn(carry, action, jnp.asarray(1))["occupancy"]["transition:s"].lower
    )
    counts = carry[0].before["transition:s"][:, :, 0]
    table = counts / counts.sum(axis=0, keepdims=True)
    assert value == pytest.approx(float(-(table * jnp.log(table)).sum(axis=0) @ q_c), rel=1e-5)
    conc = params.transition["s"].concentration[:, :, 0]
    assert not np.allclose(np.asarray(table), np.asarray(conc / conc.sum(axis=0)), rtol=1e-3)
