"""Hybrid System validation: scope guards and loud errors (design D1-D3).

Covers the required cases: the within-step declaration-order error computes
and prints a valid order; cross-family modalities are rejected; unknown
``epistemic`` variable names are rejected; ``infer_states`` on an observed
mixture factor raises ``NotImplementedError`` naming workstream C. Plus the
guards around ``mixture_k``, innovation parent shapes, LG observation targets,
and the discrete-only dispatch regression (compiled path untouched).
"""

import jax.numpy as jnp
import pytest

from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalGaussian,
    Gaussian,
    MixtureGaussian,
)
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianTransition
from hybrid_aif.system import System, SystemParams

MODE_TABLE = jnp.log(jnp.asarray([[0.5, 0.92, 0.08], [0.5, 0.08, 0.92]]))  # (modes, actions)
MODE_B = ConditionalCategorical(log_table=MODE_TABLE, parent_ndims=1)
LG_B = LinearGaussianTransition(
    F=jnp.asarray([[0.9]]),
    innovations=ConditionalGaussian(
        loc=jnp.asarray([[-1.2], [1.2]]),
        chol=jnp.asarray([[[0.3]], [[0.45]]]),
        parent_ndims=1,
    ),
)
MIX_D = MixtureGaussian(
    log_weights=jnp.log(jnp.asarray([0.5, 0.5])),
    means=jnp.asarray([[-1.0], [1.0]]),
    chols=jnp.asarray([[[0.6]], [[0.6]]]),
)
TARGET = MixtureGaussian(
    log_weights=jnp.zeros(1), means=jnp.asarray([[3.0]]), chols=jnp.asarray([[[0.8]]])
)


def switching(**system_overrides):
    system = System(
        **{
            **dict(
                A_deps=(),
                B_deps=((), (0,)),
                controlled=(True, False),
                policy_len=2,
                mixture_k=4,
                modality_names=(),
                factor_names=("mode", "x"),
            ),
            **system_overrides,
        }
    )
    params = SystemParams(
        likelihood={},
        transition={"mode": MODE_B, "x": LG_B},
        initial_state_prior={"mode": Categorical(logits=jnp.zeros(2)), "x": MIX_D},
        obs_targets={},
        state_targets={"x": TARGET},
        terminal_targets={},
    )
    return system, params


def beliefs(system, params):
    return dict(params.initial_state_prior)  # the priors, keyed by factor, as beliefs


def test_declaration_order_error_prints_a_valid_order():
    params = SystemParams(
        likelihood={},
        # continuous factor first, naming the later mode factor
        transition={"x": LG_B, "mode": MODE_B},
        initial_state_prior={"x": MIX_D, "mode": Categorical(logits=jnp.zeros(2))},
        obs_targets={},
        state_targets={"x": TARGET},
        terminal_targets={},
    )
    system = System(
        A_deps=(),
        B_deps=((1,), ()),
        controlled=(False, True),
        policy_len=2,
        mixture_k=4,
        modality_names=(),
        factor_names=("x", "mode"),
    )
    with pytest.raises(ValueError, match=r"valid declaration order is: \('mode', 'x'\)"):
        system.planner_context(params, beliefs(system, params))


def test_continuous_self_or_continuous_parent_rejected():
    system, params = switching(B_deps=((), (1,)))  # x depends on itself
    with pytest.raises(ValueError, match="categorical"):
        system.planner_context(params, beliefs(system, params))


def test_cross_family_modalities_rejected():
    # categorical modality observing the continuous factor
    system, params = switching(A_deps=((1,),), modality_names=("cat_obs",))
    params_cat = SystemParams(
        likelihood={"cat_obs": ConditionalCategorical(log_table=jnp.zeros((2, 2)), parent_ndims=1)},
        transition=params.transition,
        initial_state_prior=params.initial_state_prior,
        obs_targets={},
        state_targets={"x": TARGET},
        terminal_targets={},
    )
    with pytest.raises(ValueError, match="family-pure"):
        system.planner_context(params_cat, beliefs(system, params_cat))

    # LG modality observing the categorical mode factor
    system2, _ = switching(A_deps=((0,),), modality_names=("lg_obs",))
    params_lg = SystemParams(
        likelihood={
            "lg_obs": LinearGaussianObservation(
                H=jnp.eye(1), c=jnp.zeros(1), noise=0.1 * jnp.eye(1)
            ),
        },
        transition=params.transition,
        initial_state_prior=params.initial_state_prior,
        obs_targets={},
        state_targets={"x": TARGET},
        terminal_targets={},
    )
    with pytest.raises(ValueError, match="family-pure"):
        system2.planner_context(params_lg, beliefs(system2, params_lg))


def test_unknown_epistemic_variable_names_rejected():
    system, params = switching(epistemic=("x", "typo"))
    with pytest.raises(ValueError, match="typo"):
        system.planner_context(params, beliefs(system, params))


def test_mixture_infer_states_raises_naming_the_mode_aligned_route():
    system, params = switching(A_deps=((1,),), modality_names=("y",))
    params = SystemParams(
        likelihood={
            "y": LinearGaussianObservation(H=jnp.eye(1), c=jnp.zeros(1), noise=0.1 * jnp.eye(1)),
        },
        transition=params.transition,
        initial_state_prior=params.initial_state_prior,
        obs_targets={},
        state_targets={"x": TARGET},
        terminal_targets={},
    )
    with pytest.raises(NotImplementedError, match="mode_aligned"):
        system.infer_states(params, {"y": jnp.asarray([0.3])})


def test_missing_mixture_k_rejected():
    system, params = switching(mixture_k=None)
    with pytest.raises(ValueError, match="mixture_k"):
        system.planner_context(params, beliefs(system, params))


def test_innovation_parent_shape_mismatch_rejected():
    # x is mode-coupled (2 modes) but declares a 3-mode innovation family
    bad = LinearGaussianTransition(
        F=jnp.asarray([[0.9]]),
        innovations=ConditionalGaussian(
            loc=jnp.zeros((3, 1)),
            chol=jnp.broadcast_to(0.3 * jnp.eye(1), (3, 1, 1)),
            parent_ndims=1,
        ),
    )
    system, params = switching()
    import dataclasses

    params = dataclasses.replace(params, transition={"mode": MODE_B, "x": bad})
    with pytest.raises(ValueError, match="parent_shape"):
        system.planner_context(params, beliefs(system, params))


def _lg_targeted(target):
    """An LG modality over the mixture-carried factor, carrying a C designation."""
    system = System(
        A_deps=((1,),),
        B_deps=((), (0,)),
        controlled=(True, False),
        policy_len=2,
        mixture_k=4,
        modality_names=("y",),
        factor_names=("mode", "x"),
    )
    params = SystemParams(
        likelihood={
            "y": LinearGaussianObservation(H=jnp.eye(1), c=jnp.zeros(1), noise=0.1 * jnp.eye(1)),
        },
        transition={"mode": MODE_B, "x": LG_B},
        initial_state_prior={"mode": Categorical(logits=jnp.zeros(2)), "x": MIX_D},
        obs_targets={"y": target},
        state_targets={},
        terminal_targets={},
    )
    return system, params


def test_categorical_obs_target_on_lg_modality_rejected():
    """The v0.2 blanket fence is gone (see ``test_lg_obs_targets.py``); what survives
    is a *typed* rejection — a continuous modality's C designation is a density over
    its observation space, and the error says which families qualify."""
    system, params = _lg_targeted(Categorical(logits=jnp.zeros(2)))
    with pytest.raises(ValueError, match="Gaussian or MixtureGaussian"):
        system.planner_context(params, beliefs(system, params))


def test_gaussian_obs_target_on_lg_modality_accepted():
    system, params = _lg_targeted(Gaussian(loc=jnp.zeros(1), chol=jnp.asarray([[0.4]])))
    ctx = system.planner_context(params, beliefs(system, params))
    assert ctx.step_terms_fn is not None


def test_mismatched_family_pairs_rejected():
    system, params = switching()
    import dataclasses

    # Gaussian prior on a categorical-transition factor
    bad_d = dataclasses.replace(
        params,
        initial_state_prior={"mode": Gaussian(loc=jnp.zeros(1), chol=jnp.eye(1)), "x": MIX_D},
    )
    with pytest.raises(ValueError, match="tabular prior"):
        system.planner_context(bad_d, beliefs(system, bad_d))
    # Categorical prior on an LG-transition factor
    bad_d2 = dataclasses.replace(
        params,
        initial_state_prior={
            "mode": Categorical(logits=jnp.zeros(2)),
            "x": Categorical(logits=jnp.zeros(2)),
        },
    )
    with pytest.raises(ValueError, match="Gaussian or MixtureGaussian"):
        system.planner_context(bad_d2, beliefs(system, bad_d2))


def test_discrete_only_dispatch_keeps_the_compiled_path():
    """Regression guard: a discrete-only System's planner_context still hands the
    planner raw probability vectors (the compiled prob-space path), not
    distribution objects — the hybrid dispatch must not touch it."""
    table = jnp.log(jnp.asarray([[0.9, 0.1], [0.1, 0.9]]))
    system = System(
        A_deps=((0,),),
        B_deps=((0,),),
        controlled=(True,),
        policy_len=2,
    )
    params = SystemParams(
        likelihood={"modality_0": ConditionalCategorical(log_table=table, parent_ndims=1)},
        transition={
            "factor_0": ConditionalCategorical(log_table=jnp.zeros((2, 2, 2)), parent_ndims=2)
        },
        initial_state_prior={"factor_0": Categorical(logits=jnp.zeros(2))},
        obs_targets={"modality_0": Categorical(logits=jnp.asarray([1.0, -1.0]))},
        state_targets={},
        terminal_targets={},
    )
    assert not system._is_hybrid(params)
    ctx = system.planner_context(params, {"factor_0": Categorical(logits=jnp.zeros(2))})
    assert all(isinstance(b, jnp.ndarray) for b in ctx.init_beliefs)


def test_conditional_prior_without_d_deps_is_rejected_loudly():
    """A ConditionalCategorical D with no D_deps entry has an unidentified
    conditioning parent — loud error naming the declaration, both paths."""
    import dataclasses

    system, params = switching()
    cond_D = ConditionalCategorical(
        log_table=jnp.zeros((2, 3)), parent_ndims=1
    )  # P(mode | higher-level state): the hierarchical-prior pattern
    bad = dataclasses.replace(
        params, initial_state_prior={**params.initial_state_prior, "mode": cond_D}
    )
    assert system._is_hybrid(bad)
    with pytest.raises(ValueError, match="D_deps"):
        system.planner_context(
            bad,
            {"mode": params.initial_state_prior["mode"], "x": params.initial_state_prior["x"]},
        )
