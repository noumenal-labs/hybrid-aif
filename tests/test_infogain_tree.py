"""The rebased term tree: one ``infogain`` term, keyed by variable (V3/V5).

Build step 2's acceptance. The old tree had four epistemic term names
(``state_info_gain``, ``param_info_gain``, ``transition_param_info_gain``,
``state_entropy``); the new one has a single ``infogain`` term whose blocks name
the variable scored — the observing channel for a state leaf, the
slot-qualified key for a parameter leaf. Since the old names are removed
outright, the old-tree/new-tree bijection (V5.1) is pinned by **reference values
captured on the pre-rebase implementation** and hardcoded here: every leaf must
come through numerically untouched under the new key, and G must match wherever
the fold is over the same Intervals in the same order. "Untouched" is to within
a few float32 ulps: the reference values were captured through the scan
evaluator, and the planner's evaluator may associate the same sums differently.

The continuous belief entropy is the opt-in ``uncertainty`` row (V11), a cost stored as
``+H``, the old ``state_entropy`` leaf value for value; ``infogain`` holds information gain
only. G is bitwise identical.
"""

import dataclasses
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import Categorical
from hybrid_aif.objectives import merge, total
from hybrid_aif.objectives.discrete import expected_obs, expected_states, utility
from hybrid_aif.planning.enumerate import evaluate_policies
from tests.test_compiled_parity import synthetic_system
from tests.test_hybrid_lg_system import make_params as hill_params
from tests.test_hybrid_lg_system import make_system as hill_system
from tests.test_switching_observation import _X1, make_planning_params, make_planning_system
from tests.test_system import _initial_beliefs, tmaze_system


def _keyed(system, beliefs):
    """A shared fixture's beliefs, keyed (fixtures in other modules may still be positional)."""
    return beliefs if isinstance(beliefs, Mapping) else system.from_positional("beliefs", beliefs)


# ---------------------------------------------------------------------------
# Reference values captured on the pre-rebase tree (commit 6338346), keyed by the
# OLD term name -> block. `-> new key` records the bijection each one migrates by.
# ---------------------------------------------------------------------------

SYNTHETIC_UTILITY_INFOGAIN = {
    ("utility", "m0"): ([-1.4996453523635864, -1.5162439346313477],) * 2,
    ("utility", "m1"): ([-1.4170119762420654, -1.402628779411316],) * 2,
    # state_info_gain -> infogain, channel-keyed (bare modality name)
    ("state_info_gain", "m0"): ([0.05555522441864014, 0.061383724212646484],) * 2,
    ("state_info_gain", "m1"): ([0.06818252801895142, 0.0656176209449768],) * 2,
    # param_info_gain -> infogain["likelihood:<modality>"]
    ("param_info_gain", "m0"): ([0.1614953577518463, 0.16352897882461548],) * 2,
    # transition_param_info_gain -> infogain["transition:<factor>"]
    ("transition_param_info_gain", "f1"): ([0.17248569428920746, 0.17248569428920746],) * 2,
    ("risk", "f0"): ([0.8318672180175781, 0.6502143740653992],) * 2,
    ("terminal_risk", "f1"): ([0.7207863330841064, 0.7207863330841064],) * 2,
}
SYNTHETIC_UTILITY_INFOGAIN_G = ([4.011591911315918, 3.826857566833496],) * 2

SYNTHETIC_RISK_AMBIGUITY = {
    ("risk", "m0"): ([0.11790365725755692, 0.13425755500793457],) * 2,
    ("risk", "m1"): ([0.39634063839912415, 0.37233179807662964],) * 2,
    ("risk", "f0"): ([0.8318672180175781, 0.6502143740653992],) * 2,
    ("ambiguity", "m0"): ([1.3261864185333252, 1.3206026554107666],) * 2,
    ("ambiguity", "m1"): ([0.9524888396263123, 0.9646794199943542],) * 2,
    ("param_info_gain", "m0"): ([0.1614953577518463, 0.16352897882461548],) * 2,
    ("transition_param_info_gain", "f1"): ([0.17248569428920746, 0.17248569428920746],) * 2,
    ("terminal_risk", "f1"): ([0.7207863330841064, 0.7207863330841064],) * 2,
}
SYNTHETIC_RISK_AMBIGUITY_G = ([4.011591911315918, 3.826857805252075],) * 2

_IG_CUE = [3.5526916559438723e-15] * 3 + [0.595108151435852, 3.5526916559438723e-15]
_IG_LOCATION = [1.4182021713677467e-14] * 5
_IG_REWARD = [6.930789492667899e-15] * 3 + [6.930788645634952e-15] * 2
TMAZE_DEFAULT = {
    ("utility", "reward"): ([-4.018194675445557] * 5,) * 2,
    ("state_info_gain", "cue"): (_IG_CUE,) * 2,
    ("state_info_gain", "location_obs"): (_IG_LOCATION,) * 2,
    ("state_info_gain", "reward"): (_IG_REWARD,) * 2,
}
TMAZE_DEFAULT_G = ([4.018194675445557] * 3 + [3.423086643218994, 4.018194675445557],) * 2

SWITCHING_CHANNEL = {
    ("risk", "x"): ([6.4373016357421875e-06, 6.4373016357421875e-06],) * 2,
    ("state_info_gain", "y"): (
        [0.9336977601051331, 0.2375805377960205],
        [0.9337233901023865, 0.23760485649108887],
    ),
}
SWITCHING_CHANNEL_G = (
    [-0.9337169528007507, -0.23759841918945312],
    [-0.9336913228034973, -0.23757410049438477],
)

HILL_ENTROPY = {
    ("risk", "state"): ([80.093505859375, 79.93385314941406, 79.93350219726562],) * 2,
    ("state_entropy", "state"): ([-0.38472485542297363, -0.3856799602508545, -0.38472485542297363],)
    * 2,
}
HILL_ENTROPY_G = ([79.70877838134766, 79.54817199707031, 79.54877471923828],) * 2

#: old term name -> the new ``(term, key)`` builder for its blocks (V5.1's bijection)
BIJECTION = {
    "utility": lambda block: ("utility", block),
    "risk": lambda block: ("risk", block),
    "ambiguity": lambda block: ("ambiguity", block),
    "terminal_risk": lambda block: ("terminal_risk", block),
    "state_info_gain": lambda block: ("infogain", block),
    "state_entropy": lambda block: ("uncertainty", block),  # V11: the opt-in row
    "param_info_gain": lambda block: ("infogain", f"likelihood:{block}"),
    "transition_param_info_gain": lambda block: ("infogain", f"transition:{block}"),
}
#: old leaves stored negated under their new key (none since V11 moved the entropy to its row)
NEGATED: set[str] = set()


def _assert_bijection(tree, reference):
    """Every old leaf appears exactly once under its new key, value-for-value."""
    expected = {}
    for (old_term, block), (lower, upper) in reference.items():
        term, key = BIJECTION[old_term](block)
        if old_term in NEGATED:
            lower, upper = [-u for u in upper], [-lo for lo in lower]
        expected[(term, key)] = (lower, upper)
    got = {(term, key) for term, blocks in tree.items() for key in blocks}
    assert got == set(expected), f"leaf keys differ: {sorted(got ^ set(expected))}"
    for (term, key), (lower, upper) in expected.items():
        leaf = tree[term][key]
        _ulps(np.asarray(leaf.lower), np.float32(lower), f"{term}/{key} lower")
        _ulps(np.asarray(leaf.upper), np.float32(upper), f"{term}/{key} upper")


def _ulps(got, want, name, nulp=4):
    try:
        np.testing.assert_array_almost_equal_nulp(got, want, nulp=nulp)
    except AssertionError as e:
        raise AssertionError(f"{name}: {e}") from None


def _assert_g(tree, reference_g, exact=True):
    g = total(tree)
    lower, upper = reference_g
    if exact:
        _ulps(np.asarray(g.lower), np.float32(lower), "G lower")
        _ulps(np.asarray(g.upper), np.float32(upper), "G upper")
    else:
        _close(np.asarray(g.lower), np.float32(lower))
        _close(np.asarray(g.upper), np.float32(upper))


def _close(got, want):
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-5)


# ------------------------------------------------------- V5.1: leaf bijection, discrete


@pytest.mark.parametrize(
    ("decomposition", "reference", "reference_g", "exact_g"),
    [
        ("utility_infogain", SYNTHETIC_UTILITY_INFOGAIN, SYNTHETIC_UTILITY_INFOGAIN_G, True),
        ("risk_ambiguity", SYNTHETIC_RISK_AMBIGUITY, SYNTHETIC_RISK_AMBIGUITY_G, False),
    ],
)
def test_discrete_leaves_survive_the_rebase(decomposition, reference, reference_g, exact_g):
    """Every leaf of the four-term tree, under its new key, to within float32 ulps.

    The fused path's fixture with everything on at once: two channels (one
    multi-parent), a learnable likelihood and a learnable transition, and all
    three designation kinds. Under ``utility_infogain`` the fold visits the same
    Intervals in the same order, so G matches to ulps; under ``risk_ambiguity`` the
    novelty leaves now precede the risk-over-states leaf instead of following it,
    which is pure reassociation (~2e-7 on one policy here). Leaves are exact
    either way.
    """
    system, params, beliefs = synthetic_system(decomposition, True, policy_len=1)
    _, tree = system.infer_policies(params, _keyed(system, beliefs))
    _assert_bijection(tree, reference)
    _assert_g(tree, reference_g, exact=exact_g)


def test_tmaze_default_settings_are_bitwise():
    """The default-settings claim, on the canonical fixture: utility + three
    channel-keyed state leaves, folded in the same order as before the rebase."""
    system, params = tmaze_system(policy_len=1)
    beliefs = _initial_beliefs(system, params)
    _, tree = system.infer_policies(params, beliefs)
    _assert_bijection(tree, TMAZE_DEFAULT)
    _assert_g(tree, TMAZE_DEFAULT_G)


# --------------------------------------------------------- V5.1: leaf bijection, hybrid


def test_hybrid_channel_leaf_survives_the_rebase():
    """A switching LG channel over a mixture-carried factor: the state leaf keeps
    its channel key, and G is bitwise (risk + one epistemic leaf, same order)."""
    system, params = make_planning_system(), make_planning_params()
    beliefs = {"mode": Categorical(logits=jnp.zeros(2)), "x": _X1}
    policies = jnp.asarray([[0], [1]], jnp.int32)[..., None]
    tree = system.score_policies(params, beliefs, policies)
    _assert_bijection(tree, SWITCHING_CHANNEL)
    _assert_g(tree, SWITCHING_CHANNEL_G)


def test_the_belief_entropy_is_the_uncertainty_row_and_g_is_bitwise():
    """``state_entropy`` is the ``uncertainty`` row's leaf value for value (V11 decision 4),
    and G is bitwise what it was, with the position channel's information gain ablated so
    the tree holds the pre-rebase terms only."""
    system = hill_system(policy_len=1, epistemic=False, terms=("uncertainty:state",))
    params = hill_params()
    policies = jnp.arange(3, dtype=jnp.int32)[:, None, None]
    start = dict(params.initial_state_prior)
    tree = system.score_policies(params, start, policies)
    _assert_bijection(tree, HILL_ENTROPY)
    _assert_g(tree, HILL_ENTROPY_G)
    # and it really is the belief entropy
    entropy = system.predict_beliefs(params, start, policies[0])[0]["state"]
    assert float(tree["uncertainty"]["state"].lower[0]) == pytest.approx(
        float(entropy.entropy().lower), abs=1e-6
    )


# ------------------------------------------- V5.2: epistemic=False is exactly no leaves


def test_epistemic_false_matches_a_hand_built_pragmatic_reference():
    """``epistemic=False`` ≡ no epistemic leaves at all: the tree is utility only,
    and G matches a hand-built utility-only fold bitwise."""
    system, params = tmaze_system(epistemic=False)
    beliefs = _initial_beliefs(system, params)
    _, tree = system.infer_policies(params, beliefs)
    assert set(tree) == {"utility"}
    positional_params = system.to_positional("params", params)

    def transition(qs, action):
        #  the T-maze by hand: location reads its own control, context none (pymdp layout)
        per_factor = jnp.stack([action[0], jnp.int32(0)])
        return expected_states(qs, positional_params.transition, per_factor, ((0,), (1,)))

    def step_terms(qs, action, t):
        del action, t
        q_obs = expected_obs(qs, positional_params.likelihood, system.A_deps)
        return merge(utility(q_obs, params.obs_targets, system.modality_names))

    positional = system.to_positional("beliefs", beliefs)
    reference = evaluate_policies(system.policies(params), positional, transition, step_terms, None)
    for block, leaf in reference["utility"].items():
        np.testing.assert_array_equal(
            np.asarray(tree["utility"][block].lower), np.asarray(leaf.lower)
        )
    np.testing.assert_array_equal(np.asarray(total(tree).lower), np.asarray(total(reference).lower))


# --------------------------------------------------------- V3: the tree's key grammar


def test_tree_shape_is_one_term_keyed_by_variable():
    """The shape itself (V3): epistemic leaves under one term, state leaves on bare
    channel names, parameter leaves on slot-qualified variable keys — and the two
    key kinds cannot collide even when a modality and its likelihood are both
    scored."""
    from tests.test_transition_novelty import _dual_learnable_system

    system, params, beliefs = _dual_learnable_system(epistemic=True)
    _, tree = system.infer_policies(params, _keyed(system, beliefs))
    assert set(tree) == {"utility", "infogain"}
    assert set(tree["infogain"]) == {"sense", "ident", "likelihood:sense", "transition:s"}
    bare = {k for k in tree["infogain"] if ":" not in k}
    qualified = {k for k in tree["infogain"] if ":" in k}
    assert bare == set(system.modality_names)  # channel-keyed state leaves
    assert qualified == {"likelihood:sense", "transition:s"}  # variable-keyed novelty


def test_metrics_free_fold_agrees_with_the_rebased_tree():
    """V5.4's other half: the leaf-free G path folds the same signs (one epistemic
    term instead of four), so the two agree to reassociation on the T-maze."""
    system, params = tmaze_system()
    beliefs = _initial_beliefs(system, params)
    q_tree, tree = system.infer_policies(params, beliefs)
    q_fold, none = system.infer_policies(params, beliefs, metrics=False)
    assert none is None
    _close(np.asarray(q_fold.logits), np.asarray(q_tree.logits))
    del tree


def test_registry_dispatch_is_addressable_and_loud():
    """The registry seam (V4): every mounted leaf comes from a table entry, and a
    family with no entry raises there instead of silently dropping a leaf."""
    from hybrid_aif.objectives import registry

    assert registry.get(registry.STATE, registry.CATEGORICAL).name == "predictive"
    assert registry.get(registry.STATE, registry.CATEGORICAL).keying == registry.CHANNEL
    assert registry.get(registry.STATE, registry.LINEAR_GAUSSIAN).keying == registry.CHANNEL
    # V11: a continuous factor's entropy is a row, not an infogain estimator
    assert registry.get(registry.STATE, "continuous") is None
    assert registry.get_row("uncertainty").keying == registry.BLOCK
    assert registry.get(registry.param_kind("likelihood"), registry.DIRICHLET).keying == (
        registry.VARIABLE
    )
    # the deliberately absent rows (LG noise and head blocks)
    for family in (registry.INVERSE_GAMMA, registry.MATRIX_NORMAL):
        assert registry.get(registry.param_kind("likelihood"), family) is None
    with pytest.raises(NotImplementedError, match="objectives registry"):
        registry.prepare(
            registry.param_kind("likelihood"),
            registry.MATRIX_NORMAL,
            registry.Statics(index=(0,)),
        )
    assert registry.get(registry.STATE, "posterior") is None  # step 2b


def test_jit_and_vmap_cross_the_rebased_tree():
    """The tree is still a pytree of Intervals: jit over the whole planning face."""
    system, params = tmaze_system(policy_len=1)
    beliefs = _initial_beliefs(system, params)
    f = jax.jit(lambda p, b: system.infer_policies(p, b)[1]["infogain"]["cue"].lower)
    _, tree = system.infer_policies(params, beliefs)
    np.testing.assert_allclose(
        np.asarray(f(params, beliefs)), np.asarray(tree["infogain"]["cue"].lower), atol=1e-6
    )
    assert dataclasses.replace(system, epistemic=("context",)) is not system
