"""Compiled prob-space path vs the distribution-level reference, leaf by leaf.

The reference closures below reproduce the pre-compile ``System.planner_context``
composition verbatim (Categorical beliefs, ``objectives.discrete`` term
functions, ``objectives.merge`` toggle semantics); the compiled path
(``objectives.compiled``, what ``planner_context`` now builds) must match
``q_pi`` logits and every TermTree leaf to float tolerance on (a) the TMaze
fixture and (b) a synthetic multi-factor system with cross-factor A/B deps and
every designation kind at once, under both decompositions and per-variable
``epistemic`` tuples. Parameter novelty (observation and transition alike) now flows
from *learnable declarations in the slots* (D3 — no ``pA`` sidecar): the
synthetic model carries a ``ConditionalDirichlet`` likelihood and a
``ConditionalDirichlet`` transition, so this file is also the compiled-parity
surface for the D2 substitutions (mean tables throughout — ambiguity
included, per the coherence rule) and for transition novelty through the
``(qs, qs_prev)`` planning carry.
"""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import Categorical, ConditionalCategorical, ConditionalDirichlet
from hybrid_aif.objectives import merge, total
from hybrid_aif.objectives.discrete import (
    ambiguity,
    expected_obs,
    expected_states,
    flat_risk,
    param_info_gain,
    risk,
    state_info_gain,
    terminal_risk,
    transition_param_info_gain,
    utility,
)
from hybrid_aif.planning.enumerate import evaluate_policies, policy_posterior
from hybrid_aif.system import System, SystemParams, _carry_prev
from tests.test_system import _initial_beliefs, tmaze_system

ATOL = 1e-5


def _reference_plan(system, params, beliefs):
    """The pre-compile pipeline: distribution-level closures over Categorical beliefs.

    Selection comes from the System's own resolver (``_term_targets``, the single
    resolution point); everything downstream of it is hand-composed here, so the
    fused path's leaf keys — one ``infogain`` term, channel keys for state leaves
    and slot-qualified keys for the novelty ones — are pinned independently.
    """
    keyed_params = params
    params = system.to_positional("params", params)  # the reference indexes by position
    #  the per-factor (pymdp) convention, read off the public declaration: each state factor's
    #  state parents, and the action widened to one level per factor (its own node's column)
    states = len(system.names.states)
    state_deps = tuple(tuple(g for g in d if g < states) for d in system.B_deps[:states])
    owners = [f for f, c in enumerate(system.controlled) if c]

    def per_factor(action):
        return jnp.zeros((states,), jnp.int32).at[jnp.asarray(owners, jnp.int32)].set(action)

    sig_mods, param_ig_mods, param_ig_facs = system._term_targets(params)
    # V9: a mounted channel with a partial parent selection scores relative to it, in its info
    # gain and its ambiguity alike
    masks = tuple(
        mask if m in sig_mods else None for m, mask in enumerate(system._channel_selections(params))
    )

    def transition(qs, action):
        return expected_states(
            qs, params.transition, per_factor(action), state_deps, system.B_control_deps
        )

    def step_terms(qs, action, t, qs_prev=None):
        del t
        q_obs = expected_obs(qs, params.likelihood, system.A_deps)
        trees = []
        ig_decomposition = system.decomposition == "utility_infogain"
        if ig_decomposition:
            if params.obs_targets:
                trees.append(utility(q_obs, params.obs_targets, system.modality_names))
        else:
            # V11: an undesignated channel's risk is its implicit flat preference, −H[q(o)];
            # the selector mounts ambiguity on exactly the channels it mounts
            if params.obs_targets:
                trees.append(risk(q_obs, params.obs_targets, system.modality_names))
            flat = tuple(
                m for m, name in enumerate(system.modality_names) if name not in params.obs_targets
            )
            if flat:
                trees.append(
                    flat_risk(
                        tuple(q_obs[m] for m in flat),
                        tuple(system.modality_names[m] for m in flat),
                    )
                )
            if sig_mods:
                trees.append(
                    ambiguity(
                        qs,
                        tuple(params.likelihood[m] for m in sig_mods),
                        tuple(system.A_deps[m] for m in sig_mods),
                        tuple(system.modality_names[m] for m in sig_mods),
                        tuple(masks[m] for m in sig_mods),
                    )
                )
        if ig_decomposition and sig_mods:
            trees.append(
                state_info_gain(
                    qs,
                    tuple(params.likelihood[m] for m in sig_mods),
                    tuple(system.A_deps[m] for m in sig_mods),
                    tuple(system.modality_names[m] for m in sig_mods),
                    tuple(masks[m] for m in sig_mods),
                )
            )
        if param_ig_mods:
            trees.append(
                param_info_gain(
                    tuple(params.likelihood[m] for m in param_ig_mods),
                    qs,
                    tuple(q_obs[m] for m in param_ig_mods),
                    tuple(system.A_deps[m] for m in param_ig_mods),
                    tuple(system.modality_names[m] for m in param_ig_mods),
                )
            )
        if param_ig_facs:
            trees.append(
                transition_param_info_gain(
                    tuple(params.transition[f] for f in param_ig_facs),
                    qs,
                    qs_prev,
                    per_factor(action),
                    param_ig_facs,
                    tuple(state_deps[f] for f in param_ig_facs),
                    tuple(system.names.states[f] for f in param_ig_facs),
                    system.B_control_deps,
                )
            )
        if params.state_targets:
            trees.append(risk(qs, params.state_targets, system.names.states))
        return merge(*trees)

    terminal = None
    if params.terminal_targets:

        def terminal(qs_final):
            return terminal_risk(qs_final, params.terminal_targets, system.names.states)

    if param_ig_facs:
        transition, step_terms, terminal, beliefs = _carry_prev(
            transition, step_terms, terminal, beliefs
        )
    policies = system.policies(keyed_params)
    tree = evaluate_policies(policies, beliefs, transition, step_terms, terminal)
    q_pi = policy_posterior(total(tree), system.gamma, params.policy_prior)
    return q_pi, tree


def _assert_parity(system, params, beliefs):
    if not isinstance(beliefs, Mapping):
        beliefs = system.from_positional("beliefs", beliefs)
    q_pi, tree = system.infer_policies(params, beliefs)
    ref_q_pi, ref_tree = _reference_plan(system, params, system.to_positional("beliefs", beliefs))
    np.testing.assert_allclose(  # logits are gamma * G: a relative term for their magnitude
        np.asarray(q_pi.logits),
        np.asarray(ref_q_pi.logits),
        rtol=1e-6,
        atol=ATOL,
        err_msg="q_pi logits",
    )
    assert {t: set(b) for t, b in tree.items()} == {t: set(b) for t, b in ref_tree.items()}
    for term, blocks in ref_tree.items():
        for block, ref_iv in blocks.items():
            got = tree[term][block]
            np.testing.assert_allclose(
                np.asarray(got.lower),
                np.asarray(ref_iv.lower),
                atol=ATOL,
                err_msg=f"{term}/{block} lower",
            )
            np.testing.assert_allclose(
                np.asarray(got.upper),
                np.asarray(ref_iv.upper),
                atol=ATOL,
                err_msg=f"{term}/{block} upper",
            )
    return tree


# ----------------------------------------------------------------- (a) TMaze fixture


def test_tmaze_compiled_path_matches_reference():
    for policy_len in (1, 2):
        system, params = tmaze_system(policy_len=policy_len)
        beliefs = _initial_beliefs(system, params)
        _assert_parity(system, params, beliefs)


# --------------------------------------------- (b) synthetic multi-factor, everything on


def _rand_table(key, shape, child_axis):
    x = jax.random.uniform(key, shape, minval=0.1, maxval=1.0)
    return x / x.sum(axis=child_axis, keepdims=True)


def synthetic_system(decomposition, epistemic, policy_len=2):
    """2 factors, 2 modalities (one on both factors), cross-factor B deps, all
    designation kinds at once, and learnable declarations in both slot kinds:
    ``m0``'s likelihood and ``f1``'s transition are ConditionalDirichlets (D3),
    so mean-table substitution (ambiguity included) and observation +
    transition novelty all cross the compiled/reference parity surface."""
    ns, no, nu = (3, 2), (4, 3), (2, 1)
    A_deps, B_deps = ((0,), (0, 1)), ((0, 1), (1,))
    keys = iter(jax.random.split(jax.random.PRNGKey(11), 16))
    A_shapes = [(no[0], ns[0]), (no[1], ns[0], ns[1])]
    B_shapes = [(ns[0], ns[0], ns[1], nu[0]), (ns[1], ns[1], nu[1])]
    A = (
        ConditionalDirichlet(
            concentration=jax.random.uniform(next(keys), A_shapes[0], minval=0.5, maxval=4.0),
            parent_ndims=len(A_shapes[0]) - 1,
        ),
        ConditionalCategorical(
            log_table=jnp.log(_rand_table(next(keys), A_shapes[1], 0)),
            parent_ndims=len(A_shapes[1]) - 1,
        ),
    )
    B = (
        ConditionalCategorical(
            log_table=jnp.log(_rand_table(next(keys), B_shapes[0], 0)),
            parent_ndims=len(B_shapes[0]) - 1,
        ),
        ConditionalDirichlet(
            concentration=jax.random.uniform(next(keys), B_shapes[1], minval=0.5, maxval=4.0),
            parent_ndims=len(B_shapes[1]) - 1,
        ),
    )
    system = System(
        A_deps=A_deps,
        B_deps=B_deps,
        controlled=(True, False),
        policy_len=policy_len,
        gamma=8.0,
        decomposition=decomposition,
        epistemic=epistemic,
        modality_names=("m0", "m1"),
        factor_names=("f0", "f1"),
    )
    params = system.from_positional(
        "params",
        SystemParams(
            likelihood=A,
            transition=B,
            initial_state_prior=tuple(
                Categorical(logits=jax.random.normal(next(keys), (n,))) for n in ns
            ),
            obs_targets={
                "m0": Categorical(logits=jax.random.normal(next(keys), (no[0],))),
                "m1": Categorical(logits=jax.random.normal(next(keys), (no[1],))),
            },
            state_targets={"f0": Categorical(logits=jax.random.normal(next(keys), (ns[0],)))},
            terminal_targets={"f1": Categorical(logits=jax.random.normal(next(keys), (ns[1],)))},
        ),
    )
    beliefs = tuple(Categorical(logits=jax.random.normal(next(keys), (n,))) for n in ns)
    return system, params, beliefs


@pytest.mark.parametrize("decomposition", ["utility_infogain", "risk_ambiguity"])
@pytest.mark.parametrize(
    "epistemic",
    [
        True,
        ("f1", "likelihood:m0"),
        ("f1", "transition:f1"),
        False,
    ],
)
def test_synthetic_multifactor_compiled_path_matches_reference(decomposition, epistemic):
    """``epistemic=("f1", ...)`` mounts only m1's channel: m0 observes f0 alone,
    so naming f1 is how one selects that channel now (V6's non-partition). m1 observes
    (f0, f1), so its leaf and its ambiguity are scored relative to f1 alone (V9), the
    compiled path recomputing that conditional entropy per step."""
    system, params, beliefs = synthetic_system(decomposition, epistemic)
    tree = _assert_parity(system, params, beliefs)
    assert "terminal_risk" in tree and set(tree["terminal_risk"]) == {"f1"}
    assert "f0" in tree["risk"]  # risk-over-states designation always present
    infogain = tree.get("infogain", {})
    state_leaves = {"m0", "m1"} if epistemic is True else {"m1"} if epistemic else set()
    if decomposition != "utility_infogain":
        state_leaves = set()  # ambiguity carries the epistemic half of Form D
    param_leaves = {
        True: {"likelihood:m0", "transition:f1"},
        ("f1", "likelihood:m0"): {"likelihood:m0"},
        ("f1", "transition:f1"): {"transition:f1"},
        False: set(),
    }[epistemic]
    assert set(infogain) == state_leaves | param_leaves
