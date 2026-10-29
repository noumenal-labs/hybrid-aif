"""Hierarchical conditional priors (D_deps): resolution, sampling, classification."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Categorical, ConditionalCategorical, System
from tests.test_system import _initial_beliefs, tmaze_system

# P(context'=ctx | higher) with a 3-level higher factor, child-first (2, 3)
COND = ConditionalCategorical(
    log_table=jnp.log(jnp.asarray([[0.9, 0.5, 0.2], [0.1, 0.5, 0.8]])),
    parent_ndims=1,
)
HIGHER_D = Categorical(logits=jnp.log(jnp.asarray([0.5, 0.25, 0.25])))


def hierarchical_tmaze():
    """T-maze plus a leading 3-level 'regime' factor whose level sets the context prior."""
    base_system, base_params = tmaze_system()
    states = base_system.names.states
    system = System(
        #  the regime persists: its identity table reads its own previous level
        B_deps=((0,),) + tuple(tuple(d + 1 for d in deps) for deps in base_system._table.B_deps),
        A_deps=tuple(tuple(d + 1 for d in deps) for deps in base_system.A_deps),
        controlled=(False, True, False),
        factor_names=("regime", *states),
        D_deps=((), (), (0,)),  # context's prior is set by the regime
        policy_len=base_system.policy_len,
        gamma=base_system.gamma,
        epistemic=base_system.epistemic,
        modality_names=base_system.modality_names,
    )
    regime_B = ConditionalCategorical(log_table=jnp.log(jnp.eye(3))[..., None], parent_ndims=2)
    params = dataclasses.replace(
        base_params,
        transition={"regime": regime_B, **base_params.transition},
        initial_state_prior={
            "regime": HIGHER_D,
            "location": base_params.initial_state_prior["location"],
            "context": COND,
        },
    )
    return system, params


def test_initial_priors_resolves_ancestral_marginal():
    system, params = hierarchical_tmaze()
    priors = system.initial_priors(params)
    expected = jnp.exp(COND.log_probs) @ HIGHER_D.probs
    assert jnp.allclose(priors["context"].probs, expected, atol=1e-6)
    assert jnp.allclose(priors["regime"].probs, HIGHER_D.probs)  # unconditional pass-through


def test_fully_tabular_hierarchical_model_is_discrete():
    system, params = hierarchical_tmaze()
    assert not system._is_hybrid(params)
    ctx = system.planner_context(params, system.initial_priors(params))
    assert all(isinstance(b, jnp.ndarray) for b in ctx.init_beliefs)  # compiled path


def test_infer_states_defaults_to_resolved_priors():
    system, params = hierarchical_tmaze()
    from hybrid_aif.inference.discrete import fpi

    obs = (jnp.asarray(0), jnp.asarray(0), jnp.asarray(0))
    got, _ = system.infer_states(params, system.from_positional("observations", obs))
    priors = system.to_positional("beliefs", system.initial_priors(params))
    likelihood = system.to_positional("params", params).likelihood
    ref = fpi(likelihood, system.A_deps, priors, obs, num_iters=16)
    for a, b in zip(system.to_positional("beliefs", got), ref, strict=True):
        assert jnp.allclose(a.probs, b.probs, atol=1e-6)


def test_as_env_samples_jointly_not_from_marginals():
    system, params = hierarchical_tmaze()
    env, env_params = system.as_env(params)
    reset = jax.jit(
        jax.vmap(lambda k: tuple(env.reset(k, env_params)[1][n] for n in ("regime", "context")))
    )
    regimes, contexts = reset(jax.random.split(jax.random.PRNGKey(0), 40_000))
    joint = np.zeros((3, 2))
    np.add.at(joint, (np.asarray(regimes), np.asarray(contexts)), 1.0 / regimes.shape[0])
    expected = np.asarray(HIGHER_D.probs)[:, None] * np.exp(np.asarray(COND.log_probs)).T
    np.testing.assert_allclose(joint, expected, atol=0.01)


def test_validation_errors():
    system, params = hierarchical_tmaze()
    beliefs = system.initial_priors(params)
    late = dataclasses.replace(system, D_deps=((2,), (), (0,), ()))  # parent after child
    with pytest.raises(ValueError, match="earlier"):
        late.planner_context(params, beliefs)
    plain = dataclasses.replace(
        params,
        initial_state_prior={
            "regime": HIGHER_D,
            "location": params.initial_state_prior["location"],
            "context": Categorical(logits=jnp.zeros(2)),
        },
    )
    with pytest.raises(ValueError, match="tabular"):
        system.planner_context(plain, beliefs)  # D_deps declared, D unconditional
    wrong_arity = dataclasses.replace(system, D_deps=((), (), (0, 1), ()))
    with pytest.raises(ValueError, match="parent_ndims"):
        wrong_arity.planner_context(params, beliefs)
    with pytest.raises(ValueError, match="initial_priors"):
        system.planner_context(  # raw conditional D as beliefs
            params, dict(params.initial_state_prior)
        )


def test_v01_systems_unaffected():
    system, params = tmaze_system()
    beliefs = _initial_beliefs(system, params)
    assert system.D_deps is None
    q1, _ = system.infer_policies(params, beliefs)
    assert q1.probs.shape[0] == system.policies(params).shape[0]
