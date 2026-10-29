"""``legal=``: the hard constraint on action selection (``docs/design/action-legality.md``).

One keyword, three spellings (keyed by control node: a first-step or a per-step mask over the
node's levels; or one bool per policy), one composition point after ``log E``. These tests
pin: an illegal level is never selected; the three spellings agree; a correlated
policy mask no per-node mask expresses is honoured; ``E`` and ``G`` are untouched; the
hybrid path behaves identically; the refusals are sentences; values swap without a
retrace; and ``replay`` carries the mask per tick. ``tests/test_action_masking.py``
(N2's ``log_E`` route) must keep passing untouched beside this file.
"""

from __future__ import annotations

import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.planning.enumerate import legal_logits
from hybrid_aif.reference.envs.tmaze import CUE
from hybrid_aif.reference.raif.world import tiny_rssm
from hybrid_aif.replay import replay
from tests.test_replay import tmaze_trace
from tests.test_system import _initial_beliefs, tmaze_system

KEY = jax.random.PRNGKey(0)
_EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"


def _example(name: str):
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree_equal(a, b) -> bool:
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(
        np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb, strict=True)
    )


@pytest.fixture(scope="module")
def tmaze():
    return tmaze_system()


def _levels(system, params):
    """The first control node's level count (read off the policy set: levels live in tables)."""
    return int(system.policies(params)[..., 0].max()) + 1


def _forbid_first(system, params, level):
    """The first-step spelling: forbid ``level`` on the location factor's control node."""
    mask = jnp.ones(_levels(system, params), bool).at[level].set(False)
    return {"u_location": mask}


# ------------------------------------------------------------- L1/L2: selection


def test_an_illegal_level_is_never_selected(tmaze):
    system, params = tmaze
    beliefs = _initial_beliefs(system, params)
    free, _, _ = system.act(KEY, params, beliefs, selection="deterministic")
    assert (
        int(free["a_location"]) == CUE
    )  # the unconstrained agent goes to the cue (test_system pins it)
    legal = _forbid_first(system, params, CUE)
    action, q_pi, _ = system.act(KEY, params, beliefs, selection="deterministic", legal=legal)
    assert int(action["a_location"]) != CUE
    first = system.policies(params)[:, 0, 0]
    assert np.all(np.asarray(q_pi.probs)[np.asarray(first) == CUE] == 0.0)
    assert np.isclose(float(q_pi.probs.sum()), 1.0, atol=1e-6)
    for key in jax.random.split(KEY, 32):
        drawn, _, _ = system.act(key, params, beliefs, legal=legal)
        assert int(drawn["a_location"]) != CUE
    policy_mode, _, _ = system.act(KEY, params, beliefs, mode="policy", legal=legal)
    assert int(policy_mode["a_location"]) != CUE


def test_the_three_spellings_agree(tmaze):
    system, params = tmaze
    beliefs = _initial_beliefs(system, params)
    policies = system.policies(params)
    n = _levels(system, params)
    first = jnp.ones(n, bool).at[CUE].set(False)
    per_step = jnp.ones((system.policy_len, n), bool).at[0, CUE].set(False)  # later rows all true
    as_policies = policies[:, 0, 0] != CUE
    q_first, _ = system.infer_policies(params, beliefs, legal={"u_location": first})
    q_step, _ = system.infer_policies(params, beliefs, legal={"u_location": per_step})
    q_set, _ = system.infer_policies(params, beliefs, legal=as_policies)
    assert _tree_equal(q_first, q_step) and _tree_equal(q_first, q_set)


def test_a_per_step_mask_reaches_later_steps(tmaze):
    system, params = tmaze
    beliefs = _initial_beliefs(system, params)
    n = _levels(system, params)
    per_step = jnp.ones((system.policy_len, n), bool).at[1, CUE].set(False)  # step 1 only
    q, _ = system.infer_policies(params, beliefs, legal={"u_location": per_step})
    second = np.asarray(system.policies(params)[:, 1, 0])
    assert np.all(np.asarray(q.probs)[second == CUE] == 0.0)
    assert np.asarray(q.probs)[second != CUE].sum() > 0.999


def test_a_correlated_policy_mask_no_per_node_mask_expresses_is_honoured(tmaze):
    system, params = tmaze
    beliefs = _initial_beliefs(system, params)
    policies = np.asarray(system.policies(params))
    #  forbid exactly the policies that go CUE then CUE; CUE alone at either step stays legal
    illegal = (policies[:, 0, 0] == CUE) & (policies[:, 1, 0] == CUE)
    assert illegal.sum() == 1
    q, _ = system.infer_policies(params, beliefs, legal=jnp.asarray(~illegal))
    probs = np.asarray(q.probs)
    assert probs[illegal].item() == 0.0
    assert probs[(policies[:, 0, 0] == CUE) & ~illegal].sum() > 0.0
    #  the expansion helper says the same thing
    ll = legal_logits(
        jnp.asarray(~illegal), (_levels(system, params), 1), system.policy_len, system.controlled
    )
    assert np.isneginf(np.asarray(ll)[illegal]).all() and np.all(np.asarray(ll)[~illegal] == 0.0)


def test_e_and_g_are_untouched_by_a_mask(tmaze):
    system, params = tmaze
    beliefs = _initial_beliefs(system, params)
    q0, tree0 = system.infer_policies(params, beliefs)
    q1, tree1 = system.infer_policies(params, beliefs, legal=_forbid_first(system, params, CUE))
    assert _tree_equal(tree0, tree1)  # G's leaves: bitwise the same, illegal policies included
    assert params.policy_prior is None  # E was never written
    #  and with a habit prior present the mask composes after it, leaving E alone
    import dataclasses

    prior = jnp.zeros(system.policies(params).shape[0]).at[3].set(2.0)
    with_e = dataclasses.replace(params, policy_prior=prior)
    q_e, _ = system.infer_policies(with_e, beliefs, legal=_forbid_first(system, params, CUE))
    assert np.array_equal(np.asarray(with_e.policy_prior), np.asarray(prior))
    assert not _tree_equal(q_e, q1)


def test_the_hybrid_path_masks_identically():
    hill = _example("hill_lg")
    system, params = hill.make_agent()
    beliefs = system.initial_priors(params)
    free, q_free, _ = system.act(KEY, params, beliefs, selection="deterministic", mode="policy")
    best = int(free["a_state"])
    n = _levels(system, params)
    legal = {"u_state": jnp.ones(n, bool).at[best].set(False)}
    action, q, tree = system.act(
        KEY, params, beliefs, selection="deterministic", mode="policy", legal=legal
    )
    assert int(action["a_state"]) != best
    first = np.asarray(system.policies(params)[:, 0, 0])
    assert np.all(np.asarray(q.probs)[first == best] == 0.0)
    _, tree_free = system.infer_policies(params, beliefs)
    assert _tree_equal(tree, tree_free)


# ---------------------------------------------------------------- L4/L5: refusals


def test_an_all_illegal_mask_is_a_sentence(tmaze):
    system, params = tmaze
    beliefs = _initial_beliefs(system, params)
    with pytest.raises(
        ValueError,
        match=r"legal\['u_location'\] masks every level of control node 'u_location' at "
        r"step\(s\) \[0\]",
    ):
        system.infer_policies(params, beliefs, legal={"u_location": jnp.zeros(5, bool)})
    with pytest.raises(ValueError, match="leaves no legal policy"):
        system.infer_policies(
            params, beliefs, legal=jnp.zeros(system.policies(params).shape[0], bool)
        )


def test_wrong_shapes_are_sentences(tmaze):
    system, params = tmaze
    beliefs = _initial_beliefs(system, params)
    with pytest.raises(
        ValueError,
        match=r"legal\['u_location'\] has shape \(4,\); control node 'u_location' takes \(5,\)",
    ):
        system.infer_policies(params, beliefs, legal={"u_location": jnp.ones(4, bool)})
    with pytest.raises(ValueError, match=r"one bool per enumerated policy, shape \(25,\)"):
        system.infer_policies(params, beliefs, legal=jnp.ones(7, bool))


def test_legal_is_refused_on_continuous_controls():
    system, params = tiny_rssm(2, 1, key=KEY)
    beliefs = system.initial_priors(params)
    with pytest.raises(ValueError, match="continuous, where no level set exists"):
        system.infer_policies(params, beliefs, legal={})


def test_legal_is_keyed_by_control_node(tmaze):
    """Partial (a node left out is unrestricted), unknown keys refused naming the nodes, the
    per-state-factor tuple refused naming the keyed form; the bare (P,) mask unchanged."""
    system, params = tmaze
    beliefs = _initial_beliefs(system, params)
    q_free, _ = system.infer_policies(params, beliefs)
    q_empty, _ = system.infer_policies(params, beliefs, legal={})
    np.testing.assert_allclose(np.asarray(q_empty.probs), np.asarray(q_free.probs), rtol=1e-6)
    with pytest.raises(ValueError, match=r"legal has unknown key \['context'\]; known keys \["):
        system.infer_policies(params, beliefs, legal={"context": jnp.ones(2, bool)})
    with pytest.raises(TypeError, match=r"legal= is keyed by control node.*'u_location'"):
        system.infer_policies(params, beliefs, legal=(jnp.ones(5, bool), None))
    with pytest.raises(TypeError, match=r"legal= is keyed by control node"):
        system.act(KEY, params, beliefs, legal=(jnp.ones(5, bool), None))
    every = jnp.ones(system.policies(params).shape[0], bool)
    q_set, _ = system.infer_policies(params, beliefs, legal=every)
    assert _tree_equal(q_empty, q_set)


def test_an_injected_planner_is_constrained_too(tmaze):
    """The composition point is _plan, after whichever planner returned — not inside
    the built-in planners — so a planner that never heard of legality is still bound."""
    from hybrid_aif.distributions import Categorical
    from hybrid_aif.planning.enumerate import enumerate_policies

    system, params = tmaze
    beliefs = _initial_beliefs(system, params)

    def uniform(ctx, use_bound="midpoint", metrics=True):
        policies = enumerate_policies(ctx.num_controls, ctx.policy_len, ctx.controlled)
        return Categorical(logits=jnp.zeros(policies.shape[0])), None, policies

    q, _ = system.infer_policies(
        params, beliefs, planner=uniform, legal=_forbid_first(system, params, CUE)
    )
    first = np.asarray(system.policies(params)[:, 0, 0])
    assert np.all(np.asarray(q.probs)[first == CUE] == 0.0)
    assert np.allclose(np.asarray(q.probs)[first != CUE], 1.0 / (first != CUE).sum())
    action, _, _ = system.act(
        KEY, params, beliefs, planner=uniform, legal=_forbid_first(system, params, CUE)
    )
    assert int(action["a_location"]) != CUE


def test_values_swap_without_a_retrace(tmaze):
    """N2's leaf-swap property, kept: the mask is a traced array argument."""
    system, params = tmaze
    beliefs = _initial_beliefs(system, params)
    traces = []

    @jax.jit
    def plan(p, b, legal):
        traces.append(1)
        return system.infer_policies(p, b, legal=legal)[0].probs

    first = plan(params, beliefs, _forbid_first(system, params, CUE))
    second = plan(params, beliefs, _forbid_first(system, params, 0))
    assert len(traces) == 1
    assert not np.allclose(np.asarray(first), np.asarray(second))
    assert np.all(np.isfinite(np.asarray(second)))


# ---------------------------------------------------------------------- L6: replay


def test_replay_carries_a_per_tick_mask():
    system, params = tmaze_system(policy_len=3)
    observations, actions = tmaze_trace(system, params)
    T = int(actions["a_location"].shape[0])
    plain = replay(system, params, observations, actions)
    #  at tick 1, forbid whatever the model chose there
    forbidden = int(plain.chosen["a_location"][1])
    per_tick = jnp.ones((T, _levels(system, params)), bool).at[1, forbidden].set(False)
    out = replay(system, params, observations, actions, legal={"u_location": per_tick})
    assert int(out.chosen["a_location"][1]) != forbidden
    others = [t for t in range(T) if t != 1]
    np.testing.assert_array_equal(
        np.asarray(out.chosen["a_location"])[others], np.asarray(plain.chosen["a_location"])[others]
    )
    #  the recorded action at tick 1 was the forbidden one: the model gives it zero probability
    if int(actions["a_location"][1]) == forbidden:
        assert np.isneginf(float(out.action_log_prob[1]))
    assert np.all(np.isfinite(np.asarray(out.action_log_prob)[others]))
    #  the bare (T, P) spelling
    as_set = jnp.ones((T, system.policies(params).shape[0]), bool)
    as_set = as_set.at[1].set(jnp.asarray(system.policies(params)[:, 0, 0] != forbidden))
    again = replay(system, params, observations, actions, legal=as_set)
    np.testing.assert_array_equal(
        np.asarray(again.chosen["a_location"]), np.asarray(out.chosen["a_location"])
    )
    with pytest.raises(
        ValueError, match=r"legal\['u_location'\] must carry a leading time axis of 6"
    ):
        replay(system, params, observations, actions, legal={"u_location": per_tick[:4]})
    with pytest.raises(ValueError, match=r"leading time axis of 6"):
        replay(system, params, observations, actions, legal={"u_location": per_tick[:, 0]})
    with pytest.raises(TypeError, match=r"legal= is keyed by control node"):
        replay(system, params, observations, actions, legal=(per_tick, None))
    with pytest.raises(ValueError, match=r"legal has unknown key \['u_context'\]"):
        replay(system, params, observations, actions, legal={"u_context": per_tick})
    #  a per-tick all-illegal row is refused concretely, by tick, before any scan runs
    dead = per_tick.at[3].set(False)
    with pytest.raises(
        ValueError, match=r"at tick\(s\) \[3\], legal\['u_location'\] masks every level"
    ):
        replay(system, params, observations, actions, legal={"u_location": dead})
    #  the per-step spelling with all-true later rows agrees with the first-step one
    per_step = jnp.ones((T, system.policy_len, _levels(system, params)), bool)
    per_step = per_step.at[1, 0, forbidden].set(False)
    stepped = replay(system, params, observations, actions, legal={"u_location": per_step})
    np.testing.assert_array_equal(
        np.asarray(stepped.chosen["a_location"]), np.asarray(out.chosen["a_location"])
    )
    dead_step = per_step.at[4, 2].set(False)
    with pytest.raises(ValueError, match=r"at tick\(s\) \[4\]"):
        replay(system, params, observations, actions, legal={"u_location": dead_step})


def test_agent_respond_takes_legal_keyed_by_node(tmaze):
    """The driver's responder passes the keyed mask to act, and refuses the tuple, an unknown
    key and an all-illegal entry eagerly, when it is built."""
    from hybrid_aif.composition.driver import MindCarry, agent_respond

    system, params = tmaze
    obs = system.from_positional("observations", (jnp.asarray(0), jnp.asarray(0), jnp.asarray(0)))
    free = agent_respond(system, params, selection="deterministic")
    chosen, _ = free(KEY, obs, MindCarry())
    forbid = jnp.ones(_levels(system, params), bool).at[int(chosen["a_location"])].set(False)
    bound = agent_respond(system, params, selection="deterministic", legal={"u_location": forbid})
    action, _ = bound(KEY, obs, MindCarry())
    assert int(action["a_location"]) != int(chosen["a_location"])
    with pytest.raises(TypeError, match="legal= is keyed by control node"):
        agent_respond(system, params, legal=(forbid, None))
    with pytest.raises(ValueError, match=r"legal has unknown key \['location'\]"):
        agent_respond(system, params, legal={"location": forbid})
    with pytest.raises(ValueError, match=r"legal\['u_location'\] masks every level"):
        agent_respond(system, params, legal={"u_location": jnp.zeros_like(forbid)})


def test_legal_sentences_name_the_face_and_a_list_is_the_policy_mask(tmaze):
    from hybrid_aif.composition.driver import agent_respond

    system, params = tmaze
    beliefs = _initial_beliefs(system, params)
    dead = {"u_location": jnp.zeros(5, bool)}
    with pytest.raises(ValueError, match=r"^act: legal\['u_location'\] masks every level"):
        system.act(KEY, params, beliefs, legal=dead)
    with pytest.raises(ValueError, match=r"^infer_policies: legal\['u_location'\] has shape"):
        system.infer_policies(params, beliefs, legal={"u_location": jnp.ones(4, bool)})
    with pytest.raises(ValueError, match=r"^agent_respond: legal\['u_location'\] masks every"):
        agent_respond(system, params, legal=dead)
    with pytest.raises(ValueError, match=r"^act: legal= leaves no legal policy"):
        system.act(KEY, params, beliefs, legal=jnp.zeros(25, bool))
    #  a list of P bools is the (P,) mask
    allowed = np.asarray(system.policies(params)[:, 0, 0] != CUE)
    q_list, _ = system.infer_policies(params, beliefs, legal=[bool(b) for b in allowed])
    q_array, _ = system.infer_policies(params, beliefs, legal=jnp.asarray(allowed))
    assert _tree_equal(q_list, q_array)
    #  a list of per-node masks is refused, naming the array form and the keyed one
    with pytest.raises(TypeError, match=r"^act: a list as legal= is the \(P,\) bool mask"):
        system.act(KEY, params, beliefs, legal=[jnp.ones(5, bool), None])


# ------------------------------------------------- a concrete mask closed over by a jitted call


def _tmaze():
    from tests.test_system import tmaze_system

    system, params = tmaze_system()
    return system, params, system.initial_priors(params)


def test_a_concrete_jax_mask_closed_over_by_jit_scores_as_it_does_eagerly():
    """A ``legal=`` mask held as a JAX array and closed over by ``jax.jit`` is concrete at trace
    time, so it masks the same policies as the same call made eagerly."""
    system, params, beliefs = _tmaze()
    n_pol = system.policies(params).shape[0]
    per_policy = jnp.ones(n_pol, bool).at[0].set(False)
    per_node = {"u_location": jnp.array([True, False, True, True, True])}
    for legal in (per_policy, per_node):
        eager = system.infer_policies(params, beliefs, legal=legal)[0].logits

        def scored(p, b, legal=legal):
            return system.infer_policies(p, b, legal=legal)[0].logits

        jitted = jax.jit(scored)
        np.testing.assert_array_equal(np.asarray(jitted(params, beliefs)), np.asarray(eager))


@pytest.mark.parametrize("kind", ["per_policy", "per_node"])
def test_a_concrete_mask_that_leaves_nothing_legal_is_refused_under_jit(kind):
    """The refusal of a mask that leaves no legal policy, or no legal level at a step, fires under
    ``jax.jit`` as it does eagerly when the mask is closed over."""
    system, params, beliefs = _tmaze()
    n_pol = system.policies(params).shape[0]
    legal = jnp.zeros(n_pol, bool) if kind == "per_policy" else {"u_location": jnp.zeros(5, bool)}
    match = "leaves no legal policy" if kind == "per_policy" else "masks every level"
    with pytest.raises(ValueError, match=match):
        jax.jit(lambda p, b: system.infer_policies(p, b, legal=legal)[0].logits)(params, beliefs)


def test_a_traced_mask_is_applied_without_the_concrete_refusals():
    """A mask passed as an argument of the jitted function is traced: it masks the policies, and
    the refusals, which need its value, do not run."""
    system, params, beliefs = _tmaze()
    n_pol = system.policies(params).shape[0]
    mask = jnp.ones(n_pol, bool).at[0].set(False)

    @jax.jit
    def logits(p, b, m):
        return system.infer_policies(p, b, legal=m)[0].logits

    eager = system.infer_policies(params, beliefs, legal=mask)[0].logits
    np.testing.assert_array_equal(np.asarray(logits(params, beliefs, mask)), np.asarray(eager))
