"""The tabular ``occupancy`` row: MOP's β term over a declared transition.

Design: ``objective-terms.md`` (the lattice — expected conditional entropy of the
transition, sign −1, Ramírez-Ruiz et al. 2024's β term) and
``amortization-debt-ledger.md`` entry 2, which this row and its per-family
eligibility dispatch pay. The amortized estimator of the same row is under test in
``test_amortized_objective.py``; what is under test here is the closed form, the
identity that makes it a *new leaf* rather than a sign flip, and the mean-table
discipline it inherits from D2.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import xlogy

from hybrid_aif.distributions import Categorical, ConditionalCategorical, ConditionalDirichlet
from hybrid_aif.objectives import compiled as _compiled
from hybrid_aif.objectives import hybrid as _hybrid
from hybrid_aif.objectives import registry
from hybrid_aif.objectives.base import total
from hybrid_aif.objectives.discrete import transition_occupancy_leaf
from hybrid_aif.system import System, SystemParams
from tests.test_system import tmaze_system

MOUNT = registry.Mount("occupancy:transition:x", "occupancy", (0,), registry.CATEGORICAL)


def _cc(table, parent_ndims):
    return ConditionalCategorical(log_table=jnp.log(table), parent_ndims=parent_ndims)


def _coin_transition(p: float, q: float):
    """One factor, two states, two actions: action 0 flips with prob ``p``, action 1
    with prob ``q``. Child-first ``(child, parent, control)``."""
    table = jnp.zeros((2, 2, 2))
    for a, flip in enumerate((p, q)):
        table = table.at[:, :, a].set(jnp.asarray([[1 - flip, flip], [flip, 1 - flip]]))
    return _cc(table, 2)


def _h(*probs) -> float:
    p = jnp.asarray(probs)
    return float(-xlogy(p, p).sum())


# ----------------------------------------------------------------- the closed form


def test_the_leaf_is_the_hand_computed_expected_conditional_entropy():
    """``E_q(s) H[B(·|s,a)]`` on a two-state case, against the entropy by hand: the
    kernel's *own* spread, contracted over the belief — not the belief's spread."""
    B_x = _coin_transition(0.25, 0.5)
    belief = (Categorical.from_probs(jnp.asarray([0.7, 0.3])),)
    for action, flip in ((0, 0.25), (1, 0.5)):
        got = transition_occupancy_leaf(B_x, belief, jnp.asarray(action))
        # every row of this table has the same entropy, so the contraction is that value
        assert float(got) == pytest.approx(_h(flip, 1 - flip), abs=1e-6)

    # an asymmetric table: the contraction weights each parent's row by its belief
    table = jnp.asarray([[[0.9], [0.5]], [[0.1], [0.5]]])  # (child, parent, control=1)
    got = transition_occupancy_leaf(_cc(table, 2), belief, jnp.asarray(0))
    assert float(got) == pytest.approx(0.7 * _h(0.9, 0.1) + 0.3 * _h(0.5, 0.5), abs=1e-6)


def test_a_deterministic_kernel_has_no_occupancy_however_spread_the_belief_is():
    """The point of the row's placement in the lattice: it reads the *conditional*,
    so a belief with all the entropy in the world scores zero on a sure kernel."""
    sure = _cc(jnp.asarray([[[1.0], [0.0]], [[0.0], [1.0]]]), 2)
    flat = (Categorical.from_probs(jnp.asarray([0.5, 0.5])),)
    assert float(transition_occupancy_leaf(sure, flat, jnp.asarray(0))) == pytest.approx(0.0)


def test_the_identity_that_makes_occupancy_a_new_leaf_and_not_a_sign_flip():
    """``H[q(s')] = E_q H[B] + I(s'; s)`` (``objective-terms.md``), numerically.

    The equation the lattice doc was written to catch: flipping the sign of the
    ``uncertainty`` row — the *marginal* entropy ``H[q(s')]`` — would fold
    ``I(s'; s)`` into the occupancy term, where the info-gain leaves count it a
    second time. The two coincide only when the parent belief is a Dirac.
    """
    rng = np.random.default_rng(0)
    table = rng.dirichlet(np.ones(4), size=(4, 3)).transpose(2, 0, 1)  # (child, parent, control)
    B_x = _cc(jnp.asarray(table, jnp.float32), 2)
    prior = jnp.asarray(rng.dirichlet(np.ones(4)), jnp.float32)
    belief = (Categorical.from_probs(prior),)
    for action in range(3):
        slab = jnp.asarray(table[..., action], jnp.float32)  # p(s'|s)
        q_next = slab @ prior
        joint = slab * prior  # p(s', s)
        mutual = float(jnp.sum(joint * (jnp.log(slab) - jnp.log(q_next)[:, None])))
        occupancy = float(transition_occupancy_leaf(B_x, belief, jnp.asarray(action)))
        marginal = float(-jnp.sum(xlogy(q_next, q_next)))
        assert marginal == pytest.approx(occupancy + mutual, abs=1e-5)
        assert mutual > 1e-3  # the two leaves are genuinely different numbers here


def test_a_learnable_transition_is_scored_through_its_mean_table():
    """D2's coherence rule on the transition side: the novelty leaf owns
    ``I(s'; θ | s, a)``, so this leaf takes the *mean*-table entropy. Substituting
    the expected entropy ``E_θ H[B]`` would remove that mutual information twice —
    the same discipline ``ambiguity`` states on the observation side."""
    counts = jnp.asarray([[[2.0], [1.0]], [[6.0], [3.0]]])  # (child, parent, control=1)
    learnable = ConditionalDirichlet(concentration=counts, parent_ndims=2)
    belief = (Categorical.from_probs(jnp.asarray([1.0, 0.0])),)
    got = float(transition_occupancy_leaf(learnable, belief, jnp.asarray(0)))
    assert got == pytest.approx(_h(0.25, 0.75), abs=1e-6)  # mean table 2/8, 6/8
    expected = float(learnable.expected_entropy()[0, 0])
    assert expected < got - 1e-3  # strictly smaller: the mean table is the wider one


# ------------------------------------------------------------------- the two carriers


def _both_carriers(B, action):
    """The same mount through the fused prob-space carrier and the distribution-object
    one, on one factor with a two-state belief."""
    A = (_cc(jnp.asarray([[[0.8], [0.3]], [[0.2], [0.7]]]), 2),)
    A_deps, B_deps, names = ((0,),), ((0,),), ("x",)
    probs = jnp.asarray([0.6, 0.4])
    terms = dict(
        decomposition="utility_infogain",
        state_infogain_modalities=(),
        param_infogain_modalities=(),
        param_infogain_factors=(),
        mounts=(MOUNT,),
    )
    fused = _compiled.make_step_terms(
        _compiled.compile_tables(A, B, {}, {}, {}),
        A_deps,
        B_deps,
        1,
        modality_names=("o",),
        factor_names=names,
        **terms,
    )((probs,), action, jnp.asarray(0), (probs,))
    objects = _hybrid.make_step_terms(
        A,
        B,
        A_deps,
        B_deps,
        (_hybrid.CATEGORICAL,),
        {},
        {},
        modality_names=("o",),
        factor_names=names,
        **terms,
    )(
        (Categorical.from_probs(probs),),
        action,
        jnp.asarray(0),
        (Categorical.from_probs(probs),),
    )
    return fused, objects


def test_both_carriers_compute_the_same_leaf_under_the_same_block_key():
    """One estimator, two carriers (the registry's ``(prepare, prepare_compiled)``
    discipline): the fused prob-space path and the distribution-object path agree,
    and both key the leaf by the variable they read, ``transition:<factor>``."""
    fused, objects = _both_carriers((_coin_transition(0.25, 0.5),), jnp.asarray([0]))
    assert set(fused["occupancy"]) == {"transition:x"} == set(objects["occupancy"])
    assert float(fused["occupancy"]["transition:x"].lower) == pytest.approx(
        float(objects["occupancy"]["transition:x"].lower), abs=1e-6
    )
    assert float(fused["occupancy"]["transition:x"].lower) == pytest.approx(
        _h(0.25, 0.75), abs=1e-6
    )


def test_both_carriers_agree_on_a_learnable_transition_too():
    """The fused path's ``B_probs`` is the posterior *mean* table for a learnable
    declaration, which is the same substitution ``_point`` makes on the other."""
    counts = jnp.asarray(
        [[[2.0, 1.0], [1.0, 5.0]], [[6.0, 3.0], [3.0, 5.0]]]  # (child, parent, control)
    )
    B = (ConditionalDirichlet(concentration=counts, parent_ndims=2),)
    fused, objects = _both_carriers(B, jnp.asarray([1]))
    assert float(fused["occupancy"]["transition:x"].lower) == pytest.approx(
        float(objects["occupancy"]["transition:x"].lower), abs=1e-6
    )


# ------------------------------------------------------ mounted on a declared System


def _slippery_tmaze(slip=0.2, **kwargs):
    """The shipped T-maze with a slip on the location transition: a *stochastic*
    kernel, without which every occupancy leaf is identically zero."""
    system, params = tmaze_system(**kwargs)
    B_loc = jnp.exp(params.transition["location"].log_probs)
    stay = jnp.eye(5)[:, :, None] * jnp.ones((1, 1, 5))
    slipped = (1 - slip) * B_loc + slip * stay
    return system, dataclasses.replace(
        params, transition={**params.transition, "location": _cc(slipped, 2)}
    )


def test_the_row_mounts_on_a_declared_tmaze_and_its_leaf_is_derived():
    """``derived=True`` on the leaf and through the fold: the tabular estimator is an
    exact functional of the declared table under the beliefs the planner carries, so
    G stays a bound (``amortization.md`` A4 — the flag is an AND, never an assert)."""
    system, params = _slippery_tmaze()
    system = dataclasses.replace(system, terms=("occupancy:transition:location",))
    _, tree = system.infer_policies(params, system.initial_priors(params))
    leaf = tree["occupancy"]["transition:location"]
    assert leaf.derived is True
    assert total(tree).derived is True
    assert float(jnp.max(leaf.lower)) > 0.1  # the slip is what it is measuring


def test_occupancy_and_infogain_coexist_on_one_variable():
    """V2's rule is *within* a term: two ``infogain`` estimators on one variable would
    be two computations of the same thing, but ``occupancy`` and ``infogain`` are
    different functionals — and mounting both on one variable is R-AIF's own
    configuration (``objective-terms.md``, "what the lattice does not constrain")."""
    system, params = _slippery_tmaze()
    system = dataclasses.replace(
        system, terms=("occupancy:transition:location",), epistemic=("location",)
    )
    _, tree = system.infer_policies(params, system.initial_priors(params))
    assert "transition:location" in tree["occupancy"]
    assert set(tree["infogain"]) == {"location_obs", "reward", "cue"}  # channel-keyed (V3)


def test_the_wildcard_ranges_over_every_transition_it_has_an_estimator_for():
    """``occupancy:transition:*`` mounts the categorical transitions — both of the
    T-maze's, since eligibility is by *family*. The context factor's kernel is sure,
    so its leaf is zero rather than absent: an omitted leaf means "not selected",
    never "computed to be nothing"."""
    system, params = _slippery_tmaze()
    system = dataclasses.replace(system, terms=("occupancy:transition:*",))
    (mount,) = system._row_mounts(params)
    assert mount.index == (0, 1) and mount.family == registry.CATEGORICAL
    _, tree = system.infer_policies(params, system.initial_priors(params))
    assert set(tree["occupancy"]) == {"transition:location", "transition:context"}
    assert float(jnp.max(tree["occupancy"]["transition:context"].lower)) == pytest.approx(0.0)


def test_mop_prefers_the_action_that_actually_moves():
    """The behavioural signature, minimally: with ``occupancy`` the whole of G, the
    ranked policies are the ones whose moves are legal — a blocked action's kernel is
    a Dirac, and MOP's β term is exactly the reward for not being in one."""
    system, params = _slippery_tmaze(policy_len=1)
    system = dataclasses.replace(system, terms=("occupancy:transition:location",), epistemic=False)
    params = dataclasses.replace(params, obs_targets={})
    q_pi, _ = system.infer_policies(params, system.initial_priors(params))
    best = int(system.policies(params)[int(jnp.argmax(q_pi.probs))][0, 0])
    assert best in (3, 4)  # from the centre: the cue or the middle, the two legal moves


def test_a_system_declaring_only_occupancy_needs_no_scoring_slot():
    """The tabular row is non-parametric and owns no declared structure, so the
    ``scoring`` slot stays empty — which is what makes ``terms=`` usable on a model
    that carries no nets at all."""
    system, params = _slippery_tmaze()
    system = dataclasses.replace(system, terms=("occupancy:transition:location",))
    assert params.scoring == {}
    system._validate(params)  # declaration-time resolution, no slot entry required
    assert isinstance(system, System) and isinstance(params, SystemParams)


def test_the_row_mounts_on_the_categorical_factor_of_a_hybrid_system():
    """The distribution-object carrier, end to end. The switching fixture's ``mode``
    factor is categorical (eligible) and its ``x`` factor a mixture (not), so the
    wildcard mounts exactly one of the two — and ``mode``'s kernel has *no parents*
    but a control axis, which is the contraction's degenerate case."""
    from tests.test_hybrid_switching_system import make_params, make_system

    params = make_params()
    system = make_system(terms=("occupancy:transition:*",))
    (mount,) = system._row_mounts(params)
    assert mount.index == (0,) and mount.family == registry.CATEGORICAL

    _, tree = system.infer_policies(params, system.initial_priors(params))
    leaf = tree["occupancy"]["transition:mode"]
    assert leaf.derived is True
    #  the "dither" action (0) mixes the modes 50/50; "commit-right" (2) is 0.92/0.08,
    #  and this kernel has no parents, so the leaf is that entropy once per step
    per_policy = {
        tuple(int(a) for a in policy[:, 0]): float(value)
        for policy, value in zip(np.asarray(system.policies(params)), leaf.lower, strict=True)
    }
    assert per_policy[(0, 0, 0)] == pytest.approx(3 * _h(0.5, 0.5), abs=1e-5)
    assert per_policy[(2, 2, 2)] == pytest.approx(3 * _h(0.92, 0.08), abs=1e-5)

    named = make_system(terms=("occupancy:transition:x",))
    with pytest.raises(NotImplementedError, match="is 'mixture'"):
        named._row_mounts(params)
