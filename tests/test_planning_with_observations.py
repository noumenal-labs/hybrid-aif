"""Planning that expects to observe (``docs/design/planning-with-observations.md``).

D1/D2: under ``planning_belief="posterior"`` a Gaussian-carried factor carries two beliefs,
the predictive one every target reads and an epistemic one that conditions on the
observations expected along the plan, which only the information-gain leaf reads. Pinned
against a hand-written Kalman recursion, and against the ``"prior"`` setting leaf by leaf.
D4: ``present=`` zeroes an absent modality's leaves and skips its conditioning. D6: a
parametric row on the linear-Gaussian path reads the step, its parameters and the epistemic
chain.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import Gaussian
from hybrid_aif.factors import LinearGaussianObservation
from hybrid_aif.inference.linear_gaussian import condition, predict
from hybrid_aif.interval import Interval
from hybrid_aif.objectives import registry
from hybrid_aif.objectives.hybrid import BeliefPair
from tests.test_hybrid_lg_system import INNOVATIONS, R_CHOL, C, F, H, make_params, make_system

PLAN = (0, 2, 1, 2)  # one thrust level per step of a four-step plan


def _belief():
    return Gaussian(loc=jnp.array([0.3, -0.1]), chol=jnp.diag(jnp.array([0.2, 0.3])))


def _run(system, params, belief, present=None):
    """Drive the planning closures by hand along ``PLAN``: the carries after each transition
    and the step trees scored on them."""
    ctx = system.planner_context(params, {"state": belief}, present=present)
    carry, carries, trees = ctx.init_beliefs, [], []
    for t, u in enumerate(PLAN):
        action = jnp.asarray([u], jnp.int32)
        carry = ctx.transition_fn(carry, action)
        carries.append(carry)
        trees.append(ctx.step_terms_fn(carry, action, jnp.asarray(t, jnp.int32)))
    return ctx, carries, trees


def _kalman(belief, present=None):
    """The reference: the predictive chain, and the epistemic chain's pre-conditioning
    covariances with the gain each step's observation delivers."""
    present = (True,) * len(PLAN) if present is None else present
    pred, epi = belief, belief
    pred_chain, epi_chain, gains = [], [], []
    for t, u in enumerate(PLAN):
        innov = INNOVATIONS.condition_on((u,))
        pred = predict(pred, F, innov.loc, innov.chol)
        if t > 0 and present[t - 1]:  # the seed is a posterior already
            y = (H @ epi.loc[..., None])[..., 0] + C
            epi, _ = condition(epi, H, C, R_CHOL, y)
        epi = predict(epi, F, innov.loc, innov.chol)
        pred_chain.append(pred)
        epi_chain.append(epi)
        S = H @ epi.chol @ epi.chol.T @ H.T + R_CHOL @ R_CHOL.T
        gains.append(0.5 * np.linalg.slogdet(np.asarray(S))[1] - np.log(float(R_CHOL[0, 0])))
    return pred_chain, epi_chain, gains


def _leaf(tree, term, key):
    return float(tree[term][key].midpoint)


# ------------------------------------------------------------------ D1/D2: the two chains


def test_kalman_parity():
    """The predictive chain is the predict-only predict; the epistemic chain is predict after
    conditioning on each step's expected observation, the Kalman covariance recursion with
    the mean untouched; the information-gain leaf is the gain of that step's observation on
    the epistemic chain, and the leaves telescope to the chain's total entropy drop."""
    system = make_system(planning_belief="posterior")
    params = make_params()
    belief = _belief()
    _, carries, trees = _run(system, params, belief)
    pred_chain, epi_chain, gains = _kalman(belief)
    for t, carry in enumerate(carries):
        (pair,) = carry
        assert isinstance(pair, BeliefPair) and int(pair.step) == t + 1
        np.testing.assert_allclose(np.asarray(pair.predictive.loc), pred_chain[t].loc, atol=1e-6)
        np.testing.assert_allclose(np.asarray(pair.predictive.chol), pred_chain[t].chol, atol=1e-6)
        np.testing.assert_allclose(np.asarray(pair.epistemic.loc), epi_chain[t].loc, atol=1e-6)
        np.testing.assert_allclose(np.asarray(pair.epistemic.chol), epi_chain[t].chol, atol=1e-6)
        # the mean is untouched by conditioning on an expected observation
        np.testing.assert_allclose(np.asarray(pair.epistemic.loc), np.asarray(pair.predictive.loc))
        assert _leaf(trees[t], "infogain", "position") == pytest.approx(gains[t], abs=1e-5)
    total = sum(_leaf(tree, "infogain", "position") for tree in trees)
    telescoped = 0.0
    for epi in epi_chain:
        y = (H @ epi.loc[..., None])[..., 0] + C
        post, _ = condition(epi, H, C, R_CHOL, y)
        telescoped += 0.5 * (
            np.linalg.slogdet(np.asarray(epi.chol @ epi.chol.T))[1]
            - np.linalg.slogdet(np.asarray(post.chol @ post.chol.T))[1]
        )
    assert total == pytest.approx(telescoped, abs=1e-5)
    # the epistemic chain tightens where the predictive one does not
    assert float(jnp.linalg.det(epi_chain[-1].chol)) < float(jnp.linalg.det(pred_chain[-1].chol))


def test_targets_are_chain_blind_and_only_the_info_gain_moves():
    """Every target, utility, risk, ambiguity and terminal leaf is bitwise the same under the
    two settings; only the information-gain leaves differ, and only after the first step: the
    channel's leaf reads the predictive belief under ``"prior"`` and the epistemic one under
    ``"posterior"``, which are the same belief until an expected observation has conditioned
    it (V11)."""
    belief = _belief()
    params = dataclasses.replace(
        make_params(),
        terminal_targets={
            "state": Gaussian(loc=jnp.zeros(2), chol=jnp.diag(jnp.array([0.1, 30.0])))
        },
    )
    prior_ctx, prior_carries, prior_trees = _run(make_system(), params, belief)
    post_ctx, post_carries, post_trees = _run(
        make_system(planning_belief="posterior"), params, belief
    )
    for t, (a, b) in enumerate(zip(prior_trees, post_trees, strict=True)):
        assert set(a) == set(b)
        for term in a:
            assert set(a[term]) == set(b[term])
            for key in a[term]:
                same = np.array_equal(
                    np.asarray(a[term][key].lower), np.asarray(b[term][key].lower)
                )
                assert same == (term != "infogain" or t == 0), (t, term, key)
    assert prior_ctx.terminal_terms_fn is not None
    terminal_prior = prior_ctx.terminal_terms_fn(prior_carries[-1])
    terminal_post = post_ctx.terminal_terms_fn(post_carries[-1])
    for key in terminal_prior["terminal_risk"]:
        assert np.array_equal(
            np.asarray(terminal_prior["terminal_risk"][key].lower),
            np.asarray(terminal_post["terminal_risk"][key].lower),
        )


# ------------------------------------------------------------------ D4: availability


def test_an_absent_modality_is_neither_scored_nor_conditioned_on():
    """Absent at every step gives the ``"prior"`` chain and zero leaves; present only at
    step 2 conditions the epistemic chain once, and the step trees elsewhere are zero for
    that modality."""
    system = make_system(planning_belief="posterior")
    params = dataclasses.replace(
        make_params(), obs_targets={"position": Gaussian(loc=jnp.ones(1), chol=0.3 * jnp.eye(1))}
    )
    belief = _belief()
    never = {"position": jnp.zeros(len(PLAN), bool)}
    _, carries, trees = _run(system, params, belief, present=never)
    pred_chain, _, _ = _kalman(belief)
    for t, carry in enumerate(carries):
        (pair,) = carry
        np.testing.assert_allclose(np.asarray(pair.epistemic.chol), pred_chain[t].chol, atol=1e-6)
        assert _leaf(trees[t], "infogain", "position") == 0.0
        assert float(trees[t]["utility"]["position"].lower) == 0.0

    only_second = jnp.asarray([False, True, False, False])
    _, carries, trees = _run(system, params, belief, present={"position": only_second})
    _, epi_chain, gains = _kalman(belief, present=(False, True, False, False))
    for t, carry in enumerate(carries):
        (pair,) = carry
        np.testing.assert_allclose(np.asarray(pair.epistemic.chol), epi_chain[t].chol, atol=1e-6)
        leaf = _leaf(trees[t], "infogain", "position")
        assert leaf == (pytest.approx(gains[t], abs=1e-5) if t == 1 else 0.0)
        assert (float(trees[t]["utility"]["position"].lower) != 0.0) == (t == 1)
    # a bool spans the plan, and the seed is never conditioned
    _, carries_all, _ = _run(system, params, belief, present={"position": True})
    _, carries_none, _ = _run(system, params, belief)
    for a, b in zip(carries_all, carries_none, strict=True):
        np.testing.assert_allclose(np.asarray(a[0].epistemic.chol), np.asarray(b[0].epistemic.chol))


def test_present_is_validated():
    system, params = make_system(), make_params()
    with pytest.raises(ValueError, match="not modalities"):
        system.planner_context(params, {"state": _belief()}, present={"nope": True})
    with pytest.raises(ValueError, match="bool array"):
        system.planner_context(params, {"state": _belief()}, present={"position": jnp.ones(4)})
    with pytest.raises(ValueError, match="one entry per plan step"):
        system.planner_context(
            params, {"state": _belief()}, present={"position": jnp.ones(3, bool)}
        )
    # the public face threads it through, and the plan still comes out
    q_pi, tree = system.infer_policies(
        params, {"state": _belief()}, present={"position": jnp.zeros(4, bool)}
    )
    assert tree is not None and float(jnp.sum(q_pi.probs)) == pytest.approx(1.0)


# ------------------------------------------------------------------ the option itself


def test_the_option_is_validated_and_refused_where_it_cannot_apply():
    with pytest.raises(ValueError, match='"prior" or "posterior"'):
        make_system(planning_belief="expected")
    with pytest.raises(ValueError, match="declare one of the two"):
        make_system(planning_belief="posterior", branch_on=("position",), planner="sophisticated")
    from hybrid_aif.reference.raif.world import tiny_rssm

    system, params = tiny_rssm(2, 1, key=jax.random.PRNGKey(0))
    posterior = dataclasses.replace(system, planning_belief="posterior")
    beliefs = posterior.initial_priors(params)
    with pytest.raises(NotImplementedError, match="amortized path"):
        posterior.planner_context(params, beliefs)
    with pytest.raises(NotImplementedError, match="amortized path"):
        system.planner_context(params, beliefs, present={"reward": False})


def test_a_tabular_system_is_untouched_by_the_option():
    """No Gaussian-carried factor: the carry has no pairs and every number is the predict-only."""
    from tests.test_system import tmaze_system

    system, params = tmaze_system()
    posterior = dataclasses.replace(system, planning_belief="posterior")
    beliefs = system.initial_priors(params)
    a = system.infer_policies(params, beliefs)
    b = posterior.infer_policies(params, beliefs)
    np.testing.assert_array_equal(np.asarray(a[0].probs), np.asarray(b[0].probs))


# ------------------------------------------------------------------ D6: a row on the LG path


def _probe_row():
    """A parametric row that reads the step, its scoring entry and the epistemic chain."""
    if "epistemic_probe" in registry.ROWS:
        return registry.ROWS["epistemic_probe"]

    def prepare(st: registry.Statics) -> registry.Step:
        scale = st.scoring

        def step(ctx: registry.StepContext) -> dict[str, Interval]:
            assert ctx.t is not None
            qs = ctx.qs if ctx.qs_epistemic is None else ctx.qs_epistemic
            spread = jnp.sum(jnp.log(jnp.diagonal(qs[0].chol)))
            return {"epistemic_probe": Interval.exact(scale * ctx.t + spread)}

        return step

    return registry.register_row(
        registry.Estimator(
            "epistemic_probe",
            registry.SINGLE,
            prepare,
            term="epistemic_probe",
            parametric=True,
            derived=True,
            parameters=("scale",),
        )
    )


def test_a_parametric_row_runs_on_the_linear_gaussian_path():
    _probe_row()
    params = dataclasses.replace(make_params(), scoring={"epistemic_probe": jnp.asarray(2.0)})
    belief = _belief()
    for setting in ("prior", "posterior"):
        system = make_system(planning_belief=setting, terms=("epistemic_probe",))
        _, carries, trees = _run(system, params, belief)
        for t, (carry, tree) in enumerate(zip(carries, trees, strict=True)):
            q = carry[0]
            q = q.epistemic if isinstance(q, BeliefPair) else q
            want = 2.0 * t + float(jnp.sum(jnp.log(jnp.diagonal(q.chol))))
            assert _leaf(tree, "epistemic_probe", "epistemic_probe") == pytest.approx(
                want, abs=1e-5
            )
    stripped = dataclasses.replace(params, scoring={})
    with pytest.raises(ValueError, match="scoring carries no"):
        make_system(terms=("epistemic_probe",)).planner_context(stripped, {"state": belief})


def test_a_point_noise_is_the_same_everywhere():
    model = LinearGaussianObservation(H=H, c=C, noise=R_CHOL)
    np.testing.assert_array_equal(np.asarray(model.R_chol_at(jnp.ones(2))), np.asarray(R_CHOL))
