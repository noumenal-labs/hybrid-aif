"""Conditional terminal designations: ``H_deps`` (design ``nesting-systems.md`` N1).

``h`` is the prior over final/intended states, reciprocal to ``d``
(arXiv:2312.07547), so ``H_deps`` mirrors ``D_deps``: a terminal designation
conditional on parent factor(s), resolved mean-field. What these tests pin:

- **the resolution rule** — the effective target is the conditional table
  averaged under ``q(parents)`` at the **final predicted beliefs** of the rollout
  being scored (the symmetric counterpart of ``initial_priors``' averaging at the
  episode start), so a parent the plan moves moves the target;
- **both planning paths** — the compiled prob-space path and the hybrid
  family-dispatched one resolve through the same
  ``objectives.discrete.resolve_conditional_targets``, and their leaves agree
  with hand-rolled scoring;
- **the two-level acceptance** — a high System's factor emits a low System's
  terminal target through a ``merge()`` binding, and changing the high state
  changes the low level's terminal-risk leaf *and* its chosen action;
- **the validation** — every way of declaring half a conditional target raises,
  naming its fix.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Categorical, ConditionalCategorical, ConditionalDirichlet
from hybrid_aif.composition.merge import TargetEmission, merge
from hybrid_aif.objectives import total
from hybrid_aif.system import System, SystemParams
from tests.test_mixture_filtering import make_params as mixture_params
from tests.test_mixture_filtering import make_system as mixture_system
from tests.test_system import _initial_beliefs, tmaze_system

KEY = jax.random.PRNGKey(0)

#: h(pos | mode), child-first (3 levels, 2 parent configs): mode 0 intends pos 0,
#: mode 1 intends pos 2 — the supraordinate conditioning of the paper's p.13.
H_TABLE = jnp.log(jnp.asarray([[0.98, 0.01], [0.01, 0.01], [0.01, 0.98]]))


def _high(mode_probs=(1.0, 0.0), flip=False, **overrides):
    """The supraordinate level: one 2-level 'mode' factor, no modalities.

    ``flip`` makes the mode deterministically swap each step, which is what
    separates "resolved at the final beliefs" from "resolved at the initial ones".
    """
    defaults = dict(
        A_deps=(),
        B_deps=((0,),),
        controlled=(False,),
        policy_len=1,
        epistemic=False,
        modality_names=(),
        factor_names=("mode",),
    )
    system = System(**{**defaults, **overrides})
    table = jnp.asarray([[0.0, 1.0], [1.0, 0.0]]) if flip else jnp.eye(2)
    params = SystemParams(
        likelihood={},
        transition={
            "mode": ConditionalCategorical(log_table=jnp.log(table)[..., None], parent_ndims=2)
        },
        initial_state_prior={"mode": Categorical.from_probs(jnp.asarray(mode_probs))},
    )
    return system, params


def _low(**overrides):
    """The subordinate level: 3-level 'pos', controlled (the action *is* the next
    position), identity modality, conditional terminal target awaiting its parent."""
    defaults = dict(
        A_deps=((0,),),
        B_deps=((0,),),
        controlled=(True,),
        policy_len=1,
        epistemic=False,
        modality_names=("look",),
        factor_names=("pos",),
    )
    system = System(**{**defaults, **overrides})
    # pos' = u whatever the previous pos, smoothed off the corners: an exactly one-hot
    # predicted belief would give KL(q ‖ h) a 0·log 0 (Categorical.kl_to does not clip).
    sharp = 0.96 * jnp.eye(3) + 0.02
    b = jnp.broadcast_to(sharp[:, None, :], (3, 3, 3))
    params = SystemParams(
        likelihood={"look": ConditionalCategorical(log_table=jnp.log(jnp.eye(3)), parent_ndims=1)},
        transition={"pos": ConditionalCategorical(log_table=jnp.log(b), parent_ndims=2)},
        initial_state_prior={"pos": Categorical(logits=jnp.zeros(3))},
        terminal_targets={"pos": ConditionalCategorical(log_table=H_TABLE, parent_ndims=1)},
    )
    return system, params


def two_level(mode_probs=(1.0, 0.0), flip=False):
    """The N1 acceptance toy: high.mode emits low.pos's terminal target."""
    high_s, high_p = _high(mode_probs, flip)
    low_s, low_p = _low()
    return merge(
        {"high": high_s, "low": low_s},
        {"high": high_p, "low": low_p},
        [TargetEmission(target="low.pos", parents=("high.mode",))],
    )


def _beliefs(system, params):
    return system.initial_priors(params)


def _best_pos(system, params):
    beliefs = _beliefs(system, params)
    q_pi, tree = system.infer_policies(params, beliefs)
    policies = system.policies(params)
    return int(policies[jnp.argmax(q_pi.probs)][0, 1]), tree


# ------------------------------------------------ the two-level acceptance (N1)


def test_target_emission_moves_the_low_levels_goal_and_its_leaf():
    """Changing the high state changes the low System's terminal-risk leaf *and*
    the action it picks — target emission as declared structure, no leaf swap."""
    system0, params0 = two_level(mode_probs=(1.0, 0.0))
    system1, params1 = two_level(mode_probs=(0.0, 1.0))
    assert system0.H_deps == ((), (0,), ())  # the binding declared it (and a node's empty entry)
    best0, tree0 = _best_pos(system0, params0)
    best1, tree1 = _best_pos(system1, params1)
    assert best0 == 0 and best1 == 2  # mode 0 intends pos 0, mode 1 intends pos 2
    leaf0 = np.asarray(tree0["terminal_risk"]["low.pos"].midpoint)
    leaf1 = np.asarray(tree1["terminal_risk"]["low.pos"].midpoint)
    assert not np.allclose(leaf0, leaf1)
    # the leaf is the KL to the *resolved* target, per policy
    assert leaf0[0] < leaf0[2] and leaf1[2] < leaf1[0]


def test_resolution_is_the_mean_field_average_under_the_parents():
    """``Σ_parents h(·|parents) q(parents)`` — the belief-side twin of ``D_deps``'
    ancestral resolution, spelled out against a hand einsum."""
    system, params = two_level(mode_probs=(0.7, 0.3))
    beliefs = _beliefs(system, params)
    resolved = system.resolve_terminal_targets(params, beliefs)
    table = np.asarray(jax.nn.softmax(H_TABLE, axis=0))  # child axis is first
    expected = table @ np.asarray(beliefs["high.mode"].probs)
    np.testing.assert_allclose(np.asarray(resolved["low.pos"].probs), expected, atol=1e-6)
    assert "low.pos" in resolved and len(resolved) == 1


def test_resolution_reads_the_final_predicted_beliefs_not_the_initial_ones():
    """The documented choice, and the one that makes the plan-dependence real: the
    parents enter at the horizon end, so a mode that flips during the rollout
    intends the *flipped* level."""
    system, params = two_level(mode_probs=(1.0, 0.0), flip=True)
    best, _ = _best_pos(system, params)
    assert best == 2  # mode starts at 0 but ends at 1
    static, static_params = two_level(mode_probs=(1.0, 0.0))
    assert _best_pos(static, static_params)[0] == 0


def test_leaf_matches_hand_rolled_scoring_on_the_compiled_path():
    """The composite is fully tabular, so this is the compiled prob-space path;
    score every policy by hand through ``predict_beliefs`` + the resolution face."""
    system, params = two_level(mode_probs=(0.6, 0.4))
    assert not system._is_hybrid(params)
    beliefs = _beliefs(system, params)
    tree = system.score_policies(params, beliefs, system.policies(params))
    got = np.asarray(tree["terminal_risk"]["low.pos"].midpoint)
    for p, policy in enumerate(system.policies(params)):
        final = system.predict_beliefs(params, beliefs, policy)[-1]
        target = system.resolve_terminal_targets(params, final)["low.pos"]
        want = float(system.to_positional("beliefs", final)[1].kl_to(target).midpoint)
        assert np.isclose(got[p], want, atol=1e-6), p


def test_metrics_false_folds_the_conditional_terminal_term_too():
    """The scan-carry path must score what the tree path scores (H_deps included)."""
    system, params = two_level(mode_probs=(0.8, 0.2))
    beliefs = _beliefs(system, params)
    q1, tree = system.infer_policies(params, beliefs)
    q2, none = system.infer_policies(params, beliefs, metrics=False)
    assert none is None
    assert jnp.allclose(q1.logits, q2.logits, atol=1e-6)
    assert jnp.allclose(total(tree).midpoint, -q1.logits / system.gamma, atol=1e-5)


def test_conditional_terminal_target_works_under_jit():
    """Resolution is traced structure, not Python branching on values."""
    system, params = two_level(mode_probs=(0.9, 0.1))
    beliefs = _beliefs(system, params)
    f = jax.jit(lambda p, b: system.infer_policies(p, b)[0].logits)
    assert jnp.allclose(
        f(params, beliefs), system.infer_policies(params, beliefs)[0].logits, atol=1e-6
    )


# --------------------------------------------------------------- the hybrid path


def hybrid_two_level(mode_probs=(1.0, 0.0)):
    """A hybrid (mixture-carried) low level: the same declaration, the other path.

    ``low.mode`` is the designated *discrete* factor of a System whose ``x`` is
    mixture-carried and observed by an LG modality, so every face routes through
    ``objectives.hybrid`` — the terminal target still resolves against the high
    level's factor.
    """
    high_s, high_p = _high(mode_probs, mixture_schedule="mode_aligned")
    low_s = mixture_system(policy_len=1)
    low_p = dataclasses.replace(
        mixture_params(),
        terminal_targets={
            "mode": ConditionalCategorical(
                log_table=jnp.log(jnp.asarray([[0.9, 0.1], [0.1, 0.9]])), parent_ndims=1
            )
        },
    )
    return merge(
        {"high": high_s, "low": low_s},
        {"high": high_p, "low": low_p},
        [TargetEmission(target="low.mode", parents=("high.mode",))],
    )


@pytest.mark.parametrize("mode_probs", [(1.0, 0.0), (0.0, 1.0)])
def test_hybrid_path_resolves_the_same_conditional_target(mode_probs):
    system, params = hybrid_two_level(mode_probs)
    assert system._is_hybrid(params)
    beliefs = system.initial_priors(params)
    tree = system.score_policies(params, beliefs, system.policies(params))
    got = np.asarray(tree["terminal_risk"]["low.mode"].midpoint)
    for p, policy in enumerate(system.policies(params)):
        final = system.predict_beliefs(params, beliefs, policy)[-1]
        target = system.resolve_terminal_targets(params, final)["low.mode"]
        want = float(system.to_positional("beliefs", final)[1].kl_to(target).midpoint)
        assert np.isclose(got[p], want, atol=1e-6), p


def test_hybrid_high_state_flips_the_preferred_mode():
    q0 = _hybrid_action((1.0, 0.0))
    q1 = _hybrid_action((0.0, 1.0))
    assert q0 != q1  # the emitted target changes which mode the low level steers to


def _hybrid_action(mode_probs):
    system, params = hybrid_two_level(mode_probs)
    beliefs = system.initial_priors(params)
    action, _, _ = system.act(KEY, params, beliefs, selection="deterministic")
    return int(system.to_positional("actions", action)[1])


# ------------------------------------------------------------------- validation


def _unconditional_tmaze():
    system, params = tmaze_system()
    target = Categorical(logits=jnp.asarray([3.0, -3.0]))
    return system, dataclasses.replace(params, terminal_targets={"context": target})


def test_conditional_target_without_H_deps_raises():
    system, params = tmaze_system()
    cond = ConditionalCategorical(log_table=jnp.zeros((2, 5)), parent_ndims=1)
    bad = dataclasses.replace(params, terminal_targets={"context": cond})
    with pytest.raises(ValueError, match="no H_deps entry"):
        system.infer_policies(bad, _initial_beliefs(system, params))


def test_H_deps_without_a_conditional_target_raises():
    system, params = _unconditional_tmaze()
    beliefs = _initial_beliefs(system, params)
    with pytest.raises(ValueError, match="ConditionalCategorical"):
        dataclasses.replace(system, H_deps=((), (0,), ())).infer_policies(params, beliefs)
    plain, plain_params = tmaze_system()
    with pytest.raises(ValueError, match="carries no terminal_targets entry"):
        dataclasses.replace(plain, H_deps=((), (0,), ())).infer_policies(
            plain_params, _initial_beliefs(plain, plain_params)
        )


def test_H_deps_shape_and_parent_errors():
    system, params = tmaze_system()
    beliefs = _initial_beliefs(system, params)
    cond = ConditionalCategorical(
        log_table=jnp.log(jnp.asarray([[0.9, 0.8, 0.7, 0.6, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])),
        parent_ndims=1,
    )
    p = dataclasses.replace(params, terminal_targets={"context": cond})
    ok = dataclasses.replace(system, H_deps=((), (0,), ()))
    ok.infer_policies(p, beliefs)  # the reference declaration validates

    with pytest.raises(ValueError, match="one entry per"):
        dataclasses.replace(system, H_deps=((),)).infer_policies(p, beliefs)
    with pytest.raises(ValueError, match="not factor indices"):
        dataclasses.replace(system, H_deps=((), (7,), ())).infer_policies(p, beliefs)
    with pytest.raises(ValueError, match="the factor itself"):
        dataclasses.replace(system, H_deps=((), (1,), ())).infer_policies(p, beliefs)
    with pytest.raises(ValueError, match="parent_ndims"):
        dataclasses.replace(system, H_deps=((), (0, 0), ())).infer_policies(p, beliefs)
    transposed = dataclasses.replace(
        params,
        terminal_targets={
            "context": ConditionalCategorical(log_table=jnp.zeros((5, 2)), parent_ndims=1)
        },
    )
    with pytest.raises(ValueError, match="from_parent_leading"):
        ok.infer_policies(transposed, beliefs)


def test_learnable_conditional_target_defers_loudly():
    """A Dirichlet ``h`` (the paper's p.13 'planning becomes habitual') has no
    statistic in ``learn`` yet, so it raises instead of silently averaging counts."""
    system, params = tmaze_system()
    learnable = dataclasses.replace(
        params,
        terminal_targets={
            "context": ConditionalDirichlet(concentration=jnp.ones((2, 5)), parent_ndims=1)
        },
    )
    with pytest.raises(NotImplementedError, match="habitual"):
        dataclasses.replace(system, H_deps=((), (0,), ())).infer_policies(
            learnable, _initial_beliefs(system, params)
        )


def test_continuous_parent_is_rejected_on_the_hybrid_path():
    system, params = hybrid_two_level()
    # re-point the emission at the mixture-carried factor 'low.x' (index 2)
    bad = dataclasses.replace(system, H_deps=((), (2,), (), ()))
    with pytest.raises(ValueError, match="non-categorical"):
        bad.infer_policies(params, system.initial_priors(params))


def test_unconditional_terminal_targets_are_bitwise_unchanged():
    """Regression: declaring the empty H_deps must not perturb the existing path."""
    system, params = _unconditional_tmaze()
    beliefs = _initial_beliefs(system, params)
    base, _ = system.infer_policies(params, beliefs)
    empty, _ = dataclasses.replace(system, H_deps=((), (), ())).infer_policies(params, beliefs)
    assert jnp.array_equal(base.logits, empty.logits)
    assert system.H_deps is None
