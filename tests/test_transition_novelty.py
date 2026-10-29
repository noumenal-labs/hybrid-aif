"""Transition novelty (D6): pymdp parity, selector symmetry, both planning paths.

Acceptance 6 from ``docs/design/parameter-learning.md``: slot-qualified
``epistemic`` keys drive observation *and* transition novelty through one walk
over the ``Learnable`` declarations, with the selector's cost printed the way
workstream L phase 2 printed ``metrics=False``. Both novelty kinds are leaves of
the one ``infogain`` term, keyed by parameter variable (``per-variable-infogain``
V3): ``likelihood:<modality>`` and ``transition:<factor>``.
The term itself is pymdp's ``calc_negative_pB_info_gain`` in natural sign,
and it must appear identically through the compiled and hybrid paths (the
compiled path hoists the ``wB`` weights but gathers the taken action's slice
per step — it is not loop-invariant like A-novelty).
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pymdp.control import calc_negative_pB_info_gain

from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalDirichlet,
    ConditionalGaussian,
    Dirichlet,
    Gaussian,
)
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianTransition
from hybrid_aif.objectives.discrete import transition_param_info_gain
from hybrid_aif.system import System, SystemParams

KEY = jax.random.PRNGKey(0)


# ----------------------------------------------------------- pymdp parity


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_transition_novelty_parity_with_pymdp(seed):
    """Our per-factor leaves sum to the negation of pymdp's
    ``calc_negative_pB_info_gain`` given the same marginals and actions."""
    rng = np.random.default_rng(seed)
    n_f = int(rng.integers(2, 4))
    sizes = [int(rng.integers(2, 5)) for _ in range(n_f)]
    controls = [int(rng.integers(1, 4)) for _ in range(n_f)]
    B_deps = [[0, int(rng.integers(1, n_f))] if f == 0 else [f] for f in range(n_f)]
    pB = [
        np.float32(rng.gamma(2.0, size=(sizes[f], *[sizes[d] for d in B_deps[f]], controls[f])))
        for f in range(n_f)
    ]
    qs_t = [np.float32(rng.dirichlet(np.ones(s))) for s in sizes]
    qs_tm1 = [np.float32(rng.dirichlet(np.ones(s))) for s in sizes]
    action = [int(rng.integers(c)) for c in controls]

    B = tuple(
        ConditionalDirichlet(concentration=jnp.asarray(p), parent_ndims=p.ndim - 1) for p in pB
    )
    tree = transition_param_info_gain(
        B,
        tuple(Categorical(logits=jnp.log(jnp.asarray(q))) for q in qs_t),
        tuple(Categorical(logits=jnp.log(jnp.asarray(q))) for q in qs_tm1),
        jnp.asarray(action, jnp.int32),
        tuple(range(n_f)),
        tuple(map(tuple, B_deps)),
    )
    ours = sum(float(iv.lower) for iv in tree["infogain"].values())
    neg_ref = calc_negative_pB_info_gain(
        [jnp.asarray(p) for p in pB],
        [jnp.asarray(q) for q in qs_t],
        [jnp.asarray(q) for q in qs_tm1],
        B_deps,
        action,
    )
    assert np.isclose(ours, -float(neg_ref), atol=1e-5)


# ------------------------------------------- acceptance 6: one selector space


def _near_identity(n, eps=1e-3):
    table = np.full((n, n), eps, np.float32) + (1.0 - n * eps) * np.eye(n, dtype=np.float32)
    return ConditionalCategorical(log_table=jnp.log(jnp.asarray(table)), parent_ndims=1)


def _dual_learnable_system(epistemic=True, ns=3, no=4, nu=2, seed=5):
    """Learnable likelihood ('sense') and learnable transition ('s') in one model."""
    rng = np.random.default_rng(seed)
    system = System(
        A_deps=((0,), (0,)),
        B_deps=((0,),),
        controlled=(True,),
        policy_len=2,
        epistemic=epistemic,
        modality_names=("sense", "ident"),
        factor_names=("s",),
    )
    params = SystemParams(
        likelihood={
            "sense": ConditionalDirichlet(
                concentration=jnp.asarray(np.float32(rng.gamma(2.0, size=(no, ns)) + 0.1)),
                parent_ndims=1,
            ),
            "ident": _near_identity(ns),
        },
        transition={
            "s": ConditionalDirichlet(
                concentration=jnp.asarray(np.float32(rng.gamma(2.0, size=(ns, ns, nu)) + 0.1)),
                parent_ndims=2,
            ),
        },
        initial_state_prior={"s": Categorical(logits=jnp.zeros(ns))},
        obs_targets={"ident": Categorical(logits=jnp.zeros(ns))},
    )
    obs = {"sense": jnp.asarray(1, jnp.int32), "ident": jnp.asarray(0, jnp.int32)}
    beliefs, _ = system.infer_states(params, obs)
    return system, params, beliefs


def test_slot_qualified_selectors_cover_both_novelty_kinds():
    """One ``infogain`` term, one leaf per selected variable: channel-keyed state
    leaves for a bare state name, slot-qualified keys for the parameter ones."""
    cases = {
        True: {"sense", "ident", "likelihood:sense", "transition:s"},
        ("likelihood:sense",): {"likelihood:sense"},
        ("transition:s",): {"transition:s"},
        ("likelihood:sense", "transition:s"): {"likelihood:sense", "transition:s"},
        ("s", "transition:s"): {"sense", "ident", "transition:s"},
        False: set(),
    }
    for sel, want in cases.items():
        system, params, beliefs = _dual_learnable_system(epistemic=sel)
        _, tree = system.infer_policies(params, beliefs)
        assert set(tree.get("infogain", {})) == want, sel


def test_selector_naming_point_slot_raises():
    system, params, beliefs = _dual_learnable_system(epistemic=("likelihood:ident",))
    with pytest.raises(ValueError, match=r"holds a ConditionalCategorical \(a point value\)"):
        system.infer_policies(params, beliefs)


def test_selector_naming_initial_prior_raises():
    """A point-valued prior trips the shared point-value error; a *learnable*
    prior trips the term-does-not-exist error (planning never revisits t=0)."""
    import dataclasses

    system, params, beliefs = _dual_learnable_system(epistemic=("initial_state_prior:s",))
    with pytest.raises(ValueError, match=r"holds a Categorical \(a point value\)"):
        system.infer_policies(params, beliefs)
    ns = params.initial_state_prior["s"].num_levels
    learnable_d = dataclasses.replace(
        params, initial_state_prior={"s": Dirichlet(concentration=jnp.ones(ns))}
    )
    with pytest.raises(ValueError, match="initial-state prior"):
        system.infer_policies(learnable_d, beliefs)


def test_transition_novelty_favors_untried_actions():
    """Counts for untried actions stay at the prior, which novelty then makes
    attractive: an agent whose action-0 slice is heavily observed prefers
    action 1 under the novelty term alone."""
    ns, nu = 3, 2
    conc = np.ones((ns, ns, nu), np.float32)
    conc[..., 0] += 50.0  # action 0 thoroughly explored
    system = System(
        A_deps=((0,),),
        B_deps=((0,),),
        controlled=(True,),
        policy_len=1,
        epistemic=("transition:s",),
        modality_names=("ident",),
        factor_names=("s",),
    )
    params = SystemParams(
        likelihood={"ident": _near_identity(ns)},
        transition={"s": ConditionalDirichlet(concentration=jnp.asarray(conc), parent_ndims=2)},
        initial_state_prior={"s": Categorical(logits=jnp.zeros(ns))},
    )
    beliefs, _ = system.infer_states(params, {"ident": jnp.asarray(0, jnp.int32)})
    q_pi, tree = system.infer_policies(params, beliefs)
    leaf = tree["infogain"]["transition:s"]
    assert float(leaf.lower[1]) > float(leaf.lower[0])  # policy taking action 1 gains more
    assert int(np.argmax(np.asarray(q_pi.probs))) == 1


def test_toggle_cost_printed():
    """The toggle's cost, printed the way L phase 2 printed ``metrics=False``:
    same posterior surface, a known per-step gather when on."""
    system_on, params, beliefs = _dual_learnable_system(epistemic=True)
    system_off = _dual_learnable_system(epistemic=("s",))[0]  # states only

    def timed(system):
        fn = jax.jit(lambda p, b: system.infer_policies(p, b, metrics=False)[0].logits)
        fn(params, beliefs)  # compile
        t0 = time.perf_counter()
        for _ in range(20):
            fn(params, beliefs).block_until_ready()
        return (time.perf_counter() - t0) / 20

    t_on, t_off = timed(system_on), timed(system_off)
    print(
        f"\nparameter-novelty selector cost (T-shaped dual-learnable model, policy_len=2): "
        f"on={t_on * 1e6:.0f}us off={t_off * 1e6:.0f}us (x{t_on / t_off:.2f})"
    )


# ------------------------------------------------------- the hybrid path


def _hybrid_learnable_system(seed=5, ns=3, nu=2):
    """Categorical factor 's' with a learnable transition + an LG factor 'x'
    (forces the hybrid path); the tabular sub-model matches the discrete twin."""
    rng = np.random.default_rng(seed)
    conc = np.float32(rng.gamma(2.0, size=(ns, ns, nu)) + 0.1)
    system = System(
        A_deps=((0,), (1,)),
        B_deps=((0,), ()),
        controlled=(True, False),
        policy_len=2,
        modality_names=("ident", "y"),
        factor_names=("s", "x"),
    )
    params = SystemParams(
        likelihood={
            "ident": _near_identity(ns),
            "y": LinearGaussianObservation(H=jnp.eye(1), c=jnp.zeros(1), noise=jnp.eye(1) * 0.5),
        },
        transition={
            "s": ConditionalDirichlet(concentration=jnp.asarray(conc), parent_ndims=2),
            "x": LinearGaussianTransition(
                F=jnp.eye(1),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 1)), chol=jnp.eye(1)[None] * 0.3, parent_ndims=1
                ),
            ),
        },
        initial_state_prior={
            "s": Categorical(logits=jnp.zeros(ns)),
            "x": Gaussian(loc=jnp.zeros(1), chol=jnp.eye(1)),
        },
        obs_targets={"ident": Categorical(logits=jnp.zeros(ns))},
    )
    return system, params, conc


def test_hybrid_path_matches_discrete_twin_on_transition_novelty():
    """The same tabular sub-model planned through the hybrid closures produces
    the same transition-novelty leaf as the compiled discrete path (the LG
    factor is decoupled), so the term is path-invariant."""
    system_h, params_h, conc = _hybrid_learnable_system()
    system_d = System(
        A_deps=((0,),),
        B_deps=((0,),),
        controlled=(True,),
        policy_len=2,
        modality_names=("ident",),
        factor_names=("s",),
    )
    params_d = SystemParams(
        likelihood={"ident": params_h.likelihood["ident"]},
        transition={"s": params_h.transition["s"]},
        initial_state_prior={"s": params_h.initial_state_prior["s"]},
        obs_targets={"ident": params_h.obs_targets["ident"]},
    )
    beliefs_d, _ = system_d.infer_states(params_d, {"ident": jnp.asarray(1, jnp.int32)})
    beliefs_h = {"s": beliefs_d["s"], "x": params_h.initial_state_prior["x"]}

    _, tree_h = system_h.infer_policies(params_h, beliefs_h)
    _, tree_d = system_d.infer_policies(params_d, beliefs_d)
    assert "transition:s" in tree_h["infogain"]
    np.testing.assert_allclose(
        np.asarray(tree_h["infogain"]["transition:s"].lower),
        np.asarray(tree_d["infogain"]["transition:s"].lower),
        atol=1e-5,
    )


def test_hybrid_learnable_likelihood_is_consumed_by_inference():
    """Regression for the positive-filter hazard (D3 'Cost, honestly'): a
    learnable tabular modality in a *hybrid* model must reach the FPI sub-model
    (system.py's ``cat_m``) and every objective term (hybrid.py's
    ``cat_mods``) — never be silently dropped."""
    system, params, _ = _hybrid_learnable_system()
    learnable_ident = ConditionalDirichlet(
        concentration=jnp.asarray(np.eye(3, dtype=np.float32) * 8.0 + 0.5), parent_ndims=1
    )
    params = SystemParams(
        likelihood={"ident": learnable_ident, "y": params.likelihood["y"]},
        transition=params.transition,
        initial_state_prior=params.initial_state_prior,
        obs_targets=params.obs_targets,
    )
    obs = {"ident": jnp.asarray(2, jnp.int32), "y": jnp.zeros(1)}
    beliefs, _ = system.infer_states(params, obs)
    # the observation moved the posterior off the uniform prior toward state 2
    assert int(np.argmax(np.asarray(beliefs["s"].probs))) == 2
    assert float(np.asarray(beliefs["s"].probs)[2]) > 0.6
    _, tree = system.infer_policies(params, beliefs)
    assert "likelihood:ident" in tree["infogain"]
    # and learn advances it through the hybrid faces' outputs
    out = system.learn(params, obs, beliefs, lr_by_slot={"transition:s": 0.0})
    assert not np.array_equal(
        np.asarray(out.likelihood["ident"].concentration),
        np.asarray(learnable_ident.concentration),
    )
