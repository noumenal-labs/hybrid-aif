"""Workstream D tests: EFE term trees — pymdp parity, decompositions, mechanics.

Parity tests feed identical inputs to pymdp's JAX reference
(``compute_expected_state``, ``compute_expected_obs``,
``compute_expected_utility``, ``compute_info_gain``,
``calc_negative_pA_info_gain``) and to our term functions. Sign convention:
pymdp returns *negative* pA info gain; our natural-sign leaf is its negation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pymdp.control import (
    calc_negative_pA_info_gain,
    compute_expected_obs,
    compute_expected_state,
    compute_expected_utility,
    compute_info_gain,
)

from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalDirichlet,
    ConditionalGaussian,
    Gaussian,
    MixtureGaussian,
)
from hybrid_aif.objectives import continuous, discrete, merge, total

KEY = jax.random.PRNGKey(0)
SEEDS = [0, 1, 2, 3, 4, 5]


def _cc(table) -> ConditionalCategorical:
    log_table = jnp.log(jnp.clip(jnp.asarray(table, jnp.float32), 1e-30))
    return ConditionalCategorical(log_table=log_table, parent_ndims=table.ndim - 1)


def _random_model(seed):
    """Factorized model with cross-factor B deps, multi-factor A deps, controls."""
    rng = np.random.default_rng(seed)
    n_f = int(rng.integers(2, 4))
    sizes = [int(rng.integers(2, 5)) for _ in range(n_f)]
    controls = [int(rng.integers(1, 4)) for _ in range(n_f)]
    B_deps, B = [], []
    for f in range(n_f):
        deps = [0, int(rng.integers(1, n_f))] if f == 0 else [f]
        B_deps.append(deps)
        g = rng.gamma(1.0, size=(sizes[f], *[sizes[d] for d in deps], controls[f]))
        B.append(np.float32(g / g.sum(axis=0, keepdims=True)))
    deps_multi = sorted(rng.choice(n_f, size=min(2, n_f), replace=False).tolist())
    A_deps = [deps_multi, [int(rng.integers(n_f))]]
    A, num_obs = [], []
    for deps in A_deps:
        n_o = int(rng.integers(2, 5))
        g = rng.gamma(1.0, size=(n_o, *[sizes[d] for d in deps]))
        A.append(np.float32(g / g.sum(axis=0, keepdims=True)))
        num_obs.append(n_o)
    qs = [np.float32(rng.dirichlet(np.ones(s))) for s in sizes]
    C = [np.float32(rng.dirichlet(np.ones(n_o)) + 0.02) for n_o in num_obs]
    C = [c / c.sum() for c in C]
    pA = [np.float32(rng.gamma(2.0, size=a.shape) + 0.1) for a in A]
    action = [int(rng.integers(c)) for c in controls]
    return sizes, B, B_deps, A, A_deps, num_obs, qs, C, pA, action


def _ours(B, B_deps, A, A_deps, qs, action):
    qs_c = tuple(Categorical(logits=jnp.log(jnp.asarray(q))) for q in qs)
    B_c = tuple(_cc(b) for b in B)
    A_c = tuple(_cc(a) for a in A)
    qs_next = discrete.expected_states(qs_c, B_c, jnp.asarray(action), tuple(map(tuple, B_deps)))
    q_obs = discrete.expected_obs(qs_next, A_c, tuple(map(tuple, A_deps)))
    return A_c, qs_next, q_obs


def _pymdp(B, B_deps, A, A_deps, qs, action):
    qs_next = compute_expected_state(
        [jnp.asarray(q) for q in qs], [jnp.asarray(b) for b in B], jnp.asarray(action), B_deps
    )
    q_obs = compute_expected_obs(qs_next, [jnp.asarray(a) for a in A], A_deps)
    return qs_next, q_obs


def _targets(C, names=None):
    names = names or [f"modality_{m}" for m in range(len(C))]
    return {n: Categorical(logits=jnp.log(jnp.asarray(c))) for n, c in zip(names, C, strict=True)}


def _leaf_sum(tree, term):
    return sum(float(iv.lower) for iv in tree[term].values())


# --------------------------------------------------- parity on random models


@pytest.mark.parametrize("seed", SEEDS)
def test_expected_states_and_obs_parity(seed):
    _, B, B_deps, A, A_deps, _, qs, _, _, action = _random_model(seed)
    _, qs_next, q_obs = _ours(B, B_deps, A, A_deps, qs, action)
    qs_ref, qo_ref = _pymdp(B, B_deps, A, A_deps, qs, action)
    for ref, q in zip(qs_ref, qs_next, strict=True):
        assert np.allclose(np.asarray(q.probs), np.asarray(ref), atol=1e-5)
    for ref, q in zip(qo_ref, q_obs, strict=True):
        assert np.allclose(np.asarray(q.probs), np.asarray(ref), atol=1e-5)


@pytest.mark.parametrize("seed", SEEDS)
def test_utility_parity(seed):
    _, B, B_deps, A, A_deps, _, qs, C, _, action = _random_model(seed)
    _, _, q_obs = _ours(B, B_deps, A, A_deps, qs, action)
    _, qo_ref = _pymdp(B, B_deps, A, A_deps, qs, action)
    targets = _targets(C)
    tree = discrete.utility(q_obs, targets)
    ref = compute_expected_utility(qo_ref, [jnp.log(jnp.asarray(c)) for c in C])
    assert np.isclose(_leaf_sum(tree, "utility"), float(ref), atol=1e-4)


@pytest.mark.parametrize("seed", SEEDS)
def test_state_info_gain_parity(seed):
    _, B, B_deps, A, A_deps, _, qs, _, _, action = _random_model(seed)
    A_c, qs_next, _ = _ours(B, B_deps, A, A_deps, qs, action)
    qs_ref, qo_ref = _pymdp(B, B_deps, A, A_deps, qs, action)
    tree = discrete.state_info_gain(qs_next, A_c, tuple(map(tuple, A_deps)))
    ref = compute_info_gain(qs_ref, qo_ref, [jnp.asarray(a) for a in A], A_deps)
    assert np.isclose(_leaf_sum(tree, "infogain"), float(ref), atol=1e-4)


@pytest.mark.parametrize("seed", SEEDS)
def test_param_info_gain_parity(seed):
    _, B, B_deps, A, A_deps, _, qs, _, pA, action = _random_model(seed)
    _, qs_next, q_obs = _ours(B, B_deps, A, A_deps, qs, action)
    qs_ref, qo_ref = _pymdp(B, B_deps, A, A_deps, qs, action)
    pA_c = tuple(
        ConditionalDirichlet(concentration=jnp.asarray(p), parent_ndims=p.ndim - 1) for p in pA
    )
    tree = discrete.param_info_gain(pA_c, qs_next, q_obs, tuple(map(tuple, A_deps)))
    neg_ref = calc_negative_pA_info_gain([jnp.asarray(p) for p in pA], qo_ref, qs_ref, A_deps)
    assert np.isclose(_leaf_sum(tree, "infogain"), -float(neg_ref), atol=1e-4)


# --------------------------------------------------------------- TMaze parity


@pytest.mark.parametrize("action", [0, 1, 3])
def test_tmaze_parity_all_terms(action):
    pytest.importorskip("pymdp.envs")
    from pymdp.envs import TMaze

    env = TMaze(reward_condition=0)
    A = [np.asarray(a, np.float32) for a in env.A]
    B = [np.asarray(b, np.float32) for b in env.B]
    A_deps, B_deps = env.A_dependencies, env.B_dependencies
    qs = [np.full(5, 0.2, np.float32), np.full(2, 0.5, np.float32)]
    u = [action, 0]
    A_c, qs_next, q_obs = _ours(B, B_deps, A, A_deps, qs, u)
    qs_ref, qo_ref = _pymdp(B, B_deps, A, A_deps, qs, u)
    for ref, q in zip(qs_ref + qo_ref, qs_next + q_obs, strict=True):
        assert np.allclose(np.asarray(q.probs), np.asarray(ref), atol=1e-5)
    ig = discrete.state_info_gain(qs_next, A_c, tuple(map(tuple, A_deps)))
    ig_ref = compute_info_gain(qs_ref, qo_ref, [jnp.asarray(a) for a in A], A_deps)
    assert np.isclose(_leaf_sum(ig, "infogain"), float(ig_ref), atol=1e-4)
    C = [np.float32(np.full(a.shape[0], 1.0 / a.shape[0])) for a in A]
    C[1] = np.float32([0.1, 0.8, 0.1])  # prefer reward outcome
    targets = _targets(C)
    util = discrete.utility(q_obs, targets)
    util_ref = compute_expected_utility(qo_ref, [jnp.log(jnp.asarray(c)) for c in C])
    assert np.isclose(_leaf_sum(util, "utility"), float(util_ref), atol=1e-4)
    pA = [np.float32(8.0 * a) for a in A]  # zero counts exercise the wnorm mask
    pA_c = tuple(
        ConditionalDirichlet(concentration=jnp.asarray(p), parent_ndims=p.ndim - 1) for p in pA
    )
    pg = discrete.param_info_gain(pA_c, qs_next, q_obs, tuple(map(tuple, A_deps)))
    neg_ref = calc_negative_pA_info_gain([jnp.asarray(p) for p in pA], qo_ref, qs_ref, A_deps)
    assert np.isclose(_leaf_sum(pg, "infogain"), -float(neg_ref), atol=1e-4)


# ---------------------------------------------------- decomposition identity


@pytest.mark.parametrize("seed", SEEDS[:4])
def test_decomposition_identity_normalized_targets(seed):
    """risk + ambiguity == −(utility + infogain) exactly (constant is zero,
    because Categorical targets are normalized — see discrete.py docstring)."""
    _, B, B_deps, A, A_deps, _, qs, C, _, action = _random_model(seed)
    A_c, qs_next, q_obs = _ours(B, B_deps, A, A_deps, qs, action)
    a_deps = tuple(map(tuple, A_deps))
    targets = _targets(C)
    tree = {
        **discrete.utility(q_obs, targets),
        **discrete.state_info_gain(qs_next, A_c, a_deps),
        **discrete.risk(q_obs, targets, tuple(f"modality_{m}" for m in range(len(A)))),
        **discrete.ambiguity(qs_next, A_c, a_deps),
    }
    lhs = total(tree, exclude=("utility", "infogain"))
    rhs = total(tree, exclude=("risk", "ambiguity"))
    assert np.isclose(float(lhs.lower), float(rhs.lower), atol=1e-4)


def test_decomposition_constant_for_unnormalized_preferences():
    """Against raw (unnormalized) log-preferences, the two totals differ by
    Σ_m logsumexp(C_raw_m) — independent of the action/policy."""
    _, B, B_deps, A, A_deps, num_obs, qs, _, _, _ = _random_model(2)
    rng = np.random.default_rng(99)
    C_raw = [np.float32(np.log(rng.gamma(1.0, size=n) + 0.2)) for n in num_obs]
    names = tuple(f"modality_{m}" for m in range(len(A)))
    targets = {n: Categorical(logits=jnp.asarray(c)) for n, c in zip(names, C_raw, strict=True)}
    a_deps = tuple(map(tuple, A_deps))
    diffs = []
    for action in ([0] * len(B), [c.shape[-1] - 1 for c in B]):
        A_c, qs_next, q_obs = _ours(B, B_deps, A, A_deps, qs, action)
        qs_ref, qo_ref = _pymdp(B, B_deps, A, A_deps, qs, action)
        ours = total(
            {
                **discrete.risk(q_obs, targets, names),
                **discrete.ambiguity(qs_next, A_c, a_deps, names),
            }
        )
        util_raw = compute_expected_utility(qo_ref, [jnp.asarray(c) for c in C_raw])
        ig = compute_info_gain(qs_ref, qo_ref, [jnp.asarray(a) for a in A], A_deps)
        diffs.append(float(ours.lower) + float(util_raw) + float(ig))
    const = sum(float(jax.scipy.special.logsumexp(jnp.asarray(c))) for c in C_raw)
    assert np.isclose(diffs[0], diffs[1], atol=1e-4)  # policy-independent
    assert np.isclose(diffs[0], const, atol=1e-4)


# ------------------------------------------------------- term-tree mechanics


def _ablation_model():
    """Three policies whose full-EFE / KL-control / expected-utility argmins differ.

    Factors: location (delta per policy), context (uniform). Observations
    {reward, neutral, cueA, cueB}: loc 0 → reward surely (utility's pick);
    loc 1 → 50/50 reward/neutral, context-independent (pure ambiguity — risk's
    pick, spread mass on preferred outcomes); loc 2 → context-revealing cue
    (pure info gain — full EFE's pick).
    """
    table = np.zeros((4, 3, 2), np.float32)
    table[0, 0, :] = 1.0
    table[(0, 1), 1, :] = 0.5
    table[2, 2, 0] = 1.0
    table[3, 2, 1] = 1.0
    table = np.clip(table, 1e-6, None)
    table /= table.sum(axis=0, keepdims=True)
    A_c = (_cc(table),)
    a_deps = ((0, 1),)
    targets = {"modality_0": Categorical(logits=jnp.log(jnp.float32([5, 4, 3, 3]) / 15.0))}
    trees = []
    for loc in range(3):
        qs = (
            Categorical(logits=jnp.log(jnp.clip(jax.nn.one_hot(loc, 3), 1e-30))),
            Categorical(logits=jnp.zeros(2)),
        )
        q_obs = discrete.expected_obs(qs, A_c, a_deps)
        trees.append(
            {
                **discrete.utility(q_obs, targets),
                **discrete.state_info_gain(qs, A_c, a_deps),
                **discrete.risk(q_obs, targets, ("modality_0",)),
                **discrete.ambiguity(qs, A_c, a_deps),
            }
        )
    return trees


def test_total_exclusions_reproduce_ablation_rankings():
    trees = _ablation_model()

    def argmin(exclude):
        return int(np.argmin([float(total(t, exclude=exclude).lower) for t in trees]))

    full_efe = argmin(("risk", "ambiguity"))  # utility + info-gain form
    assert full_efe == argmin(("utility", "infogain"))  # risk + ambiguity form
    kl_control = argmin(("utility", "infogain", "ambiguity"))  # risk only
    expected_utility = argmin(("risk", "ambiguity", "infogain"))  # utility only
    assert (full_efe, kl_control, expected_utility) == (2, 1, 0)


def test_toggles_are_omitted_leaves():
    _, B, B_deps, A, A_deps, _, qs, C, _, action = _random_model(1)
    _, _, q_obs = _ours(B, B_deps, A, A_deps, qs, action)
    targets = {"modality_1": Categorical(logits=jnp.log(jnp.asarray(C[1])))}
    tree = discrete.utility(q_obs, targets)
    assert set(tree["utility"]) == {"modality_1"}  # undesignated block absent, not zeroed
    assert set(discrete.risk(q_obs, targets)["risk"]) == set()  # default names are block_i


def test_block_names_override_and_validation():
    _, B, B_deps, A, A_deps, _, qs, C, _, action = _random_model(1)
    _, _, q_obs = _ours(B, B_deps, A, A_deps, qs, action)
    targets = {"vision": Categorical(logits=jnp.log(jnp.asarray(C[0])))}
    tree = discrete.utility(q_obs, targets, block_names=("vision", "reward"))
    assert set(tree["utility"]) == {"vision"}
    with pytest.raises(ValueError):
        discrete.utility(q_obs, targets, block_names=("vision",))


def test_terminal_risk_matches_risk_under_terminal_term_name():
    _, B, B_deps, A, A_deps, _, qs, C, _, action = _random_model(4)
    _, qs_next, q_obs = _ours(B, B_deps, A, A_deps, qs, action)
    names = tuple(f"modality_{m}" for m in range(len(A)))
    targets = _targets(C, names)
    tree = discrete.terminal_risk(q_obs, targets, names)
    ref = discrete.risk(q_obs, targets, names)
    assert set(tree) == {"terminal_risk"}
    for n in names:
        assert np.isclose(float(tree["terminal_risk"][n].lower), float(ref["risk"][n].lower))
    assert float(total(tree).lower) > 0.0  # registered sign (+1) folds into G
    state_targets = {"factor_0": Categorical(logits=jnp.log(jnp.asarray(qs[0])))}
    state_tree = discrete.risk(qs_next, state_targets, tuple(f"factor_{f}" for f in range(len(qs))))
    assert set(state_tree["risk"]) == {"factor_0"}  # risk-over-states designation


# ------------------------------------------------------------------ continuous


def _gaussians():
    g1 = Gaussian(loc=jnp.float32([0.5, -0.2]), chol=jnp.eye(2) * 0.8)
    g2 = Gaussian(loc=jnp.float32([-0.3, 0.4]), chol=jnp.float32([[1.0, 0.0], [0.3, 0.7]]))
    return g1, g2


def test_continuous_gaussian_risk_exact_interval():
    g1, g2 = _gaussians()
    tree = continuous.risk((g1,), {"latent": g2}, block_names=("latent",))
    leaf = tree["risk"]["latent"]
    assert float(leaf.width) == 0.0
    assert np.isclose(float(leaf.lower), float(g1.kl_to(g2).lower), atol=1e-6)


def _mixture(key, shift):
    means = jnp.float32([[0.0, 0.0], [2.5, 1.0], [-1.5, 2.0]]) + shift
    chols = jnp.stack([jnp.eye(2) * s for s in (0.7, 0.9, 0.6)])
    lw = jnp.log(jnp.float32([0.5, 0.3, 0.2]))
    return MixtureGaussian(log_weights=lw, means=means, chols=chols)


def test_continuous_mixture_risk_bounds_sandwich_monte_carlo():
    f, g = _mixture(KEY, 0.0), _mixture(KEY, 0.8)
    tree = continuous.risk((f,), {"belief": g}, block_names=("belief",))
    leaf = tree["risk"]["belief"]
    x = f.sample(KEY, (200_000,))
    mc = float((f.log_prob(x) - g.log_prob(x)).mean())
    assert float(leaf.lower) - 0.03 <= mc <= float(leaf.upper) + 0.03
    assert float(leaf.width) > 0.0


def test_continuous_uncertainty_term():
    """The ``uncertainty`` row's leaf is the belief's entropy, a cost (V11 decision 4): the
    mixture's Kolchinsky–Tracey pair as it is, folding into G as +H."""
    m = _mixture(KEY, 0.0)
    tree = continuous.uncertainty((m,), block_names=("belief",))
    leaf = tree["uncertainty"]["belief"]
    ref = m.entropy()
    assert float(leaf.lower) == float(ref.lower) and float(leaf.upper) == float(ref.upper)
    assert float(total(tree).lower) == float(ref.lower)  # G pays +H


def test_continuous_conditional_gaussian_ambiguity_matches_manual():
    locs = jnp.float32([[0.0, 0.0], [1.0, -1.0], [2.0, 3.0]])
    chols = jnp.stack([jnp.eye(2) * s for s in (0.5, 1.0, 2.0)])
    model = ConditionalGaussian(loc=locs, chol=chols, parent_ndims=1)
    q = Categorical(logits=jnp.log(jnp.float32([0.2, 0.5, 0.3])))
    tree = continuous.ambiguity((q,), model)
    manual = float(jnp.sum(q.probs * model.gaussians().entropy().lower))
    leaf = tree["ambiguity"]["modality_0"]
    assert float(leaf.width) == 0.0
    assert np.isclose(float(leaf.lower), manual, atol=1e-5)


# ------------------------------------------------------------ jit/vmap smoke


def test_discrete_terms_jit_vmap_over_policy_batch():
    _, B, B_deps, A, A_deps, _, qs, C, pA, _ = _random_model(3)
    qs_c = tuple(Categorical(logits=jnp.log(jnp.asarray(q))) for q in qs)
    B_c, A_c = tuple(_cc(b) for b in B), tuple(_cc(a) for a in A)
    b_deps, a_deps = tuple(map(tuple, B_deps)), tuple(map(tuple, A_deps))
    names = tuple(f"modality_{m}" for m in range(len(A)))
    targets = _targets(C, names)
    pA_c = tuple(
        ConditionalDirichlet(concentration=jnp.asarray(p), parent_ndims=p.ndim - 1) for p in pA
    )

    def terms(action):
        qs_next = discrete.expected_states(qs_c, B_c, action, b_deps)
        q_obs = discrete.expected_obs(qs_next, A_c, a_deps)
        # `merge`, not a dict literal: state info gain and parameter novelty are
        # leaves of the same `infogain` term now, so ** would drop the first one.
        return merge(
            discrete.utility(q_obs, targets),
            discrete.state_info_gain(qs_next, A_c, a_deps),
            discrete.param_info_gain(pA_c, qs_next, q_obs, a_deps),
            discrete.risk(q_obs, targets, names),
            discrete.ambiguity(qs_next, A_c, a_deps),
            discrete.terminal_risk(q_obs, targets, names),
        )

    actions = jnp.asarray([[0] * len(B), [c.shape[-1] - 1 for c in B], [0] * len(B)], jnp.int32)
    batched = jax.jit(jax.vmap(terms))(actions)
    g_batched = jax.jit(jax.vmap(lambda a: total(terms(a), exclude=("risk", "ambiguity")).lower))(
        actions
    )
    assert g_batched.shape == (3,)
    for i in range(3):
        single = terms(actions[i])
        for term, blocks in single.items():
            for block, leaf in blocks.items():
                got = batched[term][block]
                assert got.lower.shape == (3,) + leaf.lower.shape
                assert np.allclose(np.asarray(got.lower[i]), np.asarray(leaf.lower), atol=1e-5)
        assert np.isclose(
            float(g_batched[i]),
            float(total(single, exclude=("risk", "ambiguity")).lower),
            atol=1e-5,
        )


def test_continuous_terms_jit_vmap_over_policy_batch():
    locs = jnp.float32([[0.0, 0.0], [1.0, -1.0], [2.0, 3.0]])
    chols = jnp.stack([jnp.eye(2) * s for s in (0.5, 1.0, 2.0)])
    model = ConditionalGaussian(loc=locs, chol=chols, parent_ndims=1)
    target = Gaussian(loc=jnp.zeros(2), chol=jnp.eye(2))

    def terms(parent_logits, g_loc):
        q = Categorical(logits=parent_logits)
        g = Gaussian(loc=g_loc, chol=jnp.eye(2) * 0.9)
        return {
            **continuous.ambiguity((q,), model),
            **continuous.risk((g,), {"latent": target}, block_names=("latent",)),
        }

    logits = jnp.stack([jnp.zeros(3), jnp.float32([2.0, 0.0, -2.0])])
    g_locs = jnp.float32([[0.1, 0.2], [-1.0, 0.5]])
    batched = jax.jit(jax.vmap(terms))(logits, g_locs)
    assert batched["ambiguity"]["modality_0"].lower.shape == (2,)
    assert batched["risk"]["latent"].lower.shape == (2,)
    for i in range(2):
        single = terms(logits[i], g_locs[i])
        assert np.isclose(
            float(batched["ambiguity"]["modality_0"].lower[i]),
            float(single["ambiguity"]["modality_0"].lower),
            atol=1e-5,
        )
        assert np.isclose(
            float(batched["risk"]["latent"].lower[i]),
            float(single["risk"]["latent"].lower),
            atol=1e-5,
        )
