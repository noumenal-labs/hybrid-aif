"""Dynamic action sets by ``policy_prior = -inf`` masking (design ``nesting-systems.md`` N2).

The pattern, adjudicated for the mission executor and worth stating once: to
remove a policy from consideration *without changing the model*, put ``-inf`` in
the habit prior. Since ``q_pi ∝ softmax(-γ·G + log E)``, a ``-inf`` entry gets
exactly zero posterior mass, and because ``policy_prior`` is an **array leaf** of
``SystemParams`` — not static structure — swapping the mask is a leaf swap: no
re-jit as the legal set changes within a run. (A change in policy-set
*cardinality* is a different action space, hence a different jit shape: that is
the per-shape caching precedent, not this mechanism.)

This is a guard, not a feature: nothing in the library special-cases ``-inf``, so
what these tests pin is that nothing NaNs on the way through
``infer_policies``/``act`` — including the action-marginal path, which takes
``log`` of a masked marginal.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_aif.reference.envs.tmaze import CUE
from tests.test_merge import _obs, merged
from tests.test_system import _initial_beliefs, tmaze_system

KEY = jax.random.PRNGKey(0)


def _mask_first_action(system, params, level):
    """``policy_prior`` masking every policy whose first action is ``level``."""
    policies = system.policies(params)
    masked = policies[:, 0, 0] == level
    return jnp.where(masked, -jnp.inf, 0.0), np.asarray(masked)


def test_masked_policies_get_zero_probability_and_no_nans():
    system, params = tmaze_system()
    policy_prior, masked = _mask_first_action(system, params, CUE)
    params = dataclasses.replace(params, policy_prior=policy_prior)
    beliefs = _initial_beliefs(system, params)
    q_pi, tree = system.infer_policies(params, beliefs)
    probs = np.asarray(q_pi.probs)
    assert np.all(np.isfinite(probs)) and np.isclose(probs.sum(), 1.0, atol=1e-6)
    assert np.all(probs[masked] == 0.0)
    assert probs[~masked].sum() > 0.999
    assert np.all(np.isfinite(np.asarray(tree["utility"]["reward"].midpoint)))


def test_act_never_selects_a_masked_action():
    """Both selection rules and the marginal path (which logs a zero marginal)."""
    system, params = tmaze_system()
    policy_prior, _ = _mask_first_action(system, params, CUE)
    params = dataclasses.replace(params, policy_prior=policy_prior)
    beliefs = _initial_beliefs(system, params)
    action, _, _ = system.act(KEY, params, beliefs, selection="deterministic")
    assert (
        int(action["a_location"]) != CUE
    )  # the unmasked agent goes to the cue (test_system pins that)
    for key in jax.random.split(KEY, 32):
        drawn, _, _ = system.act(key, params, beliefs)
        assert int(drawn["a_location"]) != CUE
        assert drawn["a_location"].dtype == jnp.int32
    policy_mode, _, _ = system.act(KEY, params, beliefs, mode="policy")
    assert int(policy_mode["a_location"]) != CUE


def test_metrics_free_path_masks_identically():
    system, params = tmaze_system()
    policy_prior, masked = _mask_first_action(system, params, CUE)
    params = dataclasses.replace(params, policy_prior=policy_prior)
    beliefs = _initial_beliefs(system, params)
    q1, _ = system.infer_policies(params, beliefs)
    q2, none = system.infer_policies(params, beliefs, metrics=False)
    assert none is None
    assert np.all(np.asarray(q2.probs)[masked] == 0.0)
    assert jnp.allclose(q1.probs, q2.probs, atol=1e-6)


def test_swapping_the_mask_is_a_leaf_swap_not_a_retrace():
    """The N2 claim: masks change within a run without paying a compile."""
    system, params = tmaze_system()
    traces = []

    @jax.jit
    def plan(p, b):
        traces.append(1)
        return system.infer_policies(p, b)[0].probs

    beliefs = _initial_beliefs(system, params)
    first = plan(
        dataclasses.replace(params, policy_prior=_mask_first_action(system, params, CUE)[0]),
        beliefs,
    )
    second = plan(
        dataclasses.replace(params, policy_prior=_mask_first_action(system, params, 0)[0]), beliefs
    )
    assert len(traces) == 1  # one trace, two different legal sets
    assert not np.allclose(np.asarray(first), np.asarray(second))
    assert np.all(np.isfinite(np.asarray(second)))


def test_masking_survives_the_merged_composite():
    """The same leaf on a ``merge()`` composite: masking is inherited, like every
    other face, because the composite is just a System."""
    composite, cp = merged()
    policy_prior, masked = _mask_first_action(composite, cp, CUE)
    cp = dataclasses.replace(cp, policy_prior=policy_prior)
    beliefs, _ = composite.infer_states(cp, _obs(composite))
    q_pi, _ = composite.infer_policies(cp, beliefs)
    probs = np.asarray(q_pi.probs)
    assert np.all(np.isfinite(probs)) and np.isclose(probs.sum(), 1.0, atol=1e-6)
    assert np.all(probs[masked] == 0.0)
    action, _, _ = composite.act(KEY, cp, beliefs, selection="deterministic")
    assert int(action["agent.a_location"]) != CUE


def test_almost_everything_masked_still_normalizes():
    system, params = tmaze_system()
    policies = system.policies(params)
    keep = 3
    policy_prior = jnp.where(jnp.arange(policies.shape[0]) == keep, 0.0, -jnp.inf)
    params = dataclasses.replace(params, policy_prior=policy_prior)
    beliefs = _initial_beliefs(system, params)
    q_pi, _ = system.infer_policies(params, beliefs)
    assert np.isclose(float(q_pi.probs[keep]), 1.0, atol=1e-6)
    action, _, _ = system.act(KEY, params, beliefs, selection="deterministic")
    assert np.array_equal(
        np.asarray(system.to_positional("actions", action)), np.asarray(policies[keep, 0])
    )
