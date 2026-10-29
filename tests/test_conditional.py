"""Contract revision 1 tests: conditional + joint families and their algebra.

Contract-generic tests (pytree/jit/vmap/batch_shape) run over every new
family; correctness tests exploit the conversion algebra (chain-rule
roundtrip, joint ≅ flattened-Categorical isomorphism) and parity with the
array kernels and pymdp (fixtures reused from ``test_discrete_parity``).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import digamma
from pymdp.learning import update_obs_likelihood_dirichlet

from hybrid_aif import (
    Categorical,
    ConditionalCategorical,
    ConditionalDirichlet,
    ConditionalGaussian,
    JointCategorical,
    MixtureGaussian,
)
from hybrid_aif.distributions.conditional import _count_update
from hybrid_aif.inference import fpi, propagate
from tests.test_discrete_parity import _log_stable, _random_model, _run_both

KEY = jax.random.PRNGKey(0)


def _prob_table(shape, axis, seed=0):
    g = np.random.default_rng(seed).gamma(1.0, size=shape).astype(np.float32)
    return jnp.asarray(g / g.sum(axis=axis, keepdims=True))


def make_cond_categorical():
    return ConditionalCategorical.from_probs(_prob_table((3, 2, 4), axis=0), parent_ndims=2)


def make_cond_dirichlet():
    g = np.random.default_rng(1).gamma(2.0, size=(3, 2, 4)).astype(np.float32) + 0.1
    return ConditionalDirichlet(concentration=jnp.asarray(g), parent_ndims=2)


def make_cond_gaussian():
    loc = jnp.asarray([[0.0, 1.0], [2.0, -1.0], [0.5, 0.5]])  # parent_shape (3,), dim 2
    chol = jnp.broadcast_to(jnp.eye(2) * 0.7, (3, 2, 2))
    return ConditionalGaussian(loc=loc, chol=chol, parent_ndims=1)


def make_joint():
    g = np.random.default_rng(2).gamma(1.0, size=(2, 4)).astype(np.float32)
    return JointCategorical.from_probs(jnp.asarray(g / g.sum()), event_ndims=2)


def _beliefs():
    return (
        Categorical(logits=jnp.asarray([0.2, -0.1])),
        Categorical(logits=jnp.asarray([0.0, 0.3, -0.5, 1.0])),
    )


CASES = [
    (make_cond_categorical, lambda d: d.expected_log(_beliefs())),
    (make_cond_dirichlet, lambda d: d.expected_log_params()),
    (make_cond_gaussian, lambda d: d.gaussians().entropy().lower),
    (make_joint, lambda d: d.entropy().lower),
]


# ---------------------------------------------------------------- contract


@pytest.mark.parametrize("make,op", CASES)
def test_pytree_roundtrip(make, op):
    d = make()
    leaves, treedef = jax.tree_util.tree_flatten(d)
    d2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert jnp.allclose(op(d), op(d2))


@pytest.mark.parametrize("make,op", CASES)
def test_jit_with_argument(make, op):
    d = make()
    assert jnp.allclose(jax.jit(op)(d), op(d), atol=1e-6)


@pytest.mark.parametrize("make,op", CASES)
def test_vmap_lifts_unbatched_object(make, op):
    d = make()
    stacked = jax.tree_util.tree_map(lambda a: jnp.stack([a, a, a]), d)
    out = jax.vmap(op)(stacked)
    base = op(d)
    assert out.shape == (3,) + base.shape
    assert jnp.allclose(out, base, atol=1e-6)


@pytest.mark.parametrize("make,op", CASES)
def test_batch_shape_property(make, op):
    d = make()
    assert d.batch_shape == ()
    stacked = jax.tree_util.tree_map(lambda a: jnp.stack([a, a]), d)
    assert stacked.batch_shape == (2,)


def test_structure_metadata():
    cc = make_cond_categorical()
    assert (cc.num_levels, cc.parent_shape) == (3, (2, 4))
    cd = make_cond_dirichlet()
    assert (cd.num_levels, cd.parent_shape) == (3, (2, 4))
    cg = make_cond_gaussian()
    assert (cg.parent_shape, cg.dim) == ((3,), 2)
    j = make_joint()
    assert (j.event_ndims, j.event_shape) == (2, (2, 4))


# ------------------------------------------------- per-config normalization


def test_conditional_categorical_normalizes_per_parent_config():
    cc = make_cond_categorical()
    sums = np.asarray(jnp.exp(cc.log_probs).sum(axis=0))
    assert np.allclose(sums, 1.0, atol=1e-6)
    # gauge: a per-config offset in the stored table leaves log_probs fixed
    offset = jnp.asarray(np.random.default_rng(3).normal(size=(1, 2, 4)).astype(np.float32))
    shifted = ConditionalCategorical(log_table=cc.log_table + offset, parent_ndims=2)
    assert np.allclose(np.asarray(shifted.log_probs), np.asarray(cc.log_probs), atol=1e-5)


def test_is_normalized_helper():
    cc = make_cond_categorical()
    assert bool(cc.is_normalized())
    doubled = ConditionalCategorical(log_table=cc.log_table + jnp.log(2.0), parent_ndims=2)
    assert not bool(doubled.is_normalized())


def test_dirichlet_mean_cpd_is_normalized():
    cd = make_cond_dirichlet()
    assert np.allclose(np.asarray(cd.mean().sum(axis=0)), 1.0, atol=1e-6)
    assert bool(cd.point().is_normalized())


def test_average_is_a_simplex_and_matches_manual():
    cc = make_cond_categorical()
    q1, q2 = _beliefs()
    avg = cc.average((q1, q2))
    table = np.asarray(jnp.exp(cc.log_probs))
    manual = np.einsum("abc,b,c->a", table, np.asarray(q1.probs), np.asarray(q2.probs))
    assert np.allclose(np.asarray(avg.probs), manual, atol=1e-6)
    assert np.isclose(float(avg.probs.sum()), 1.0, atol=1e-6)


def test_condition_on_equals_direct_slice():
    cc = make_cond_categorical()
    cond = cc.condition_on((1, 2))
    assert np.allclose(np.asarray(cond.log_probs), np.asarray(cc.log_probs[:, 1, 2]), atol=1e-6)
    cd = make_cond_dirichlet()
    d = cd.condition_on((0, 3))
    assert np.allclose(np.asarray(d.concentration), np.asarray(cd.concentration[:, 0, 3]))


# ------------------------------------------------------------ chain rule


def test_chain_rule_roundtrip_and_marginals():
    cc = make_cond_categorical()
    g = np.random.default_rng(4).gamma(1.0, size=(2, 4)).astype(np.float32)
    marg = JointCategorical.from_probs(jnp.asarray(g / g.sum()), event_ndims=2)
    joint = cc.join(marg)
    assert (joint.event_ndims, joint.event_shape) == (3, (3, 2, 4))
    # joint is normalized
    assert np.isclose(float(joint.probs.sum()), 1.0, atol=1e-6)
    # roundtrip: renormalizing along the child recovers the conditional
    back = joint.to_conditional(0)
    assert np.allclose(np.asarray(back.log_probs), np.asarray(cc.log_probs), atol=1e-5)
    # marginalizing the child out recovers the parent marginal
    m = joint.marginalize(0)
    assert isinstance(m, JointCategorical)
    assert np.allclose(np.asarray(m.probs), np.asarray(marg.probs), atol=1e-6)


def test_chain_rule_single_parent_with_categorical_marginal():
    cc = ConditionalCategorical.from_probs(_prob_table((3, 4), axis=0, seed=5), parent_ndims=1)
    marg = Categorical(logits=jnp.asarray([0.0, 0.5, -0.5, 1.0]))
    joint = cc.join(marg)
    parent_back = joint.marginalize(0)
    assert isinstance(parent_back, Categorical)
    assert np.allclose(np.asarray(parent_back.probs), np.asarray(marg.probs), atol=1e-6)
    # marginalizing the parent out gives the Bayesian model average
    child = joint.marginalize(1)
    assert np.allclose(np.asarray(child.probs), np.asarray(cc.average((marg,)).probs), atol=1e-6)


def test_to_conditional_nonleading_child_axis():
    j = make_joint()  # event_shape (2, 4)
    cond = j.to_conditional(1)  # P(axis-1 | axis-0)
    assert (cond.num_levels, cond.parent_shape) == (4, (2,))
    probs = np.asarray(j.probs)
    manual = (probs / probs.sum(axis=1, keepdims=True)).T  # (4, 2)
    assert np.allclose(np.asarray(jnp.exp(cond.log_probs)), manual, atol=1e-6)


# ----------------------------------------- joint ≅ flattened categorical


def test_joint_isomorphic_to_flattened_categorical():
    j = make_joint()
    flat = Categorical(logits=j.logits.reshape(-1))
    assert np.isclose(float(j.entropy().lower), float(flat.entropy().lower), atol=1e-6)
    g = np.random.default_rng(6).gamma(1.0, size=(2, 4)).astype(np.float32)
    other = JointCategorical.from_probs(jnp.asarray(g / g.sum()), event_ndims=2)
    other_flat = Categorical(logits=other.logits.reshape(-1))
    assert np.isclose(float(j.kl_to(other).lower), float(flat.kl_to(other_flat).lower), atol=1e-6)
    assert float(j.entropy().width) == 0.0 and float(j.kl_to(other).width) == 0.0
    for i in range(2):
        for k in range(4):
            lp = float(j.log_prob(jnp.asarray([i, k], jnp.int32)))
            assert np.isclose(lp, float(flat.log_prob(jnp.asarray(i * 4 + k))), atol=1e-6)


def test_joint_sample_indices_and_frequencies():
    j = make_joint()
    x = j.sample(KEY, (20_000,))
    assert x.dtype == jnp.int32
    assert x.shape == (20_000, 2)
    freqs = np.zeros((2, 4))
    np.add.at(freqs, (np.asarray(x[:, 0]), np.asarray(x[:, 1])), 1.0)
    assert np.allclose(freqs / x.shape[0], np.asarray(j.probs), atol=0.01)


# --------------------------------------------------------------- dirichlet


def test_conditional_dirichlet_expected_log_matches_manual_digamma():
    cd = make_cond_dirichlet()
    a = np.asarray(cd.concentration)
    manual = digamma(a) - digamma(a.sum(axis=0, keepdims=True))
    assert np.allclose(np.asarray(cd.expected_log_params()), np.asarray(manual), atol=1e-6)


def test_conditional_dirichlet_expected_entropy_matches_monte_carlo():
    """Closed form ``ψ(α₀+1) − Σ_k (α_k/α₀)·ψ(α_k+1)`` per parent config equals
    the Monte-Carlo average of sampled-table column entropies. Pure math: the
    term tree does not consume this (ambiguity scores the mean table, keeping
    ``I(o; θ | s)`` as the one explicit novelty leaf); it is the *irreducible*
    ambiguity component, kept for a future no-novelty-leaf mode."""
    cd = make_cond_dirichlet()
    keys = jax.random.split(KEY, 4000)
    tables = jax.vmap(cd.sample)(keys)  # (N, 3, 2, 4)
    h = -np.asarray(jnp.sum(jax.scipy.special.xlogy(tables, tables), axis=1))  # (N, 2, 4)
    assert np.allclose(h.mean(axis=0), np.asarray(cd.expected_entropy()), atol=0.01)
    # strictly below the mean-table entropy: the gap is I(o; θ | s) >= 0
    mean = np.asarray(cd.mean())
    h_mean = -np.asarray(jax.scipy.special.xlogy(mean, mean)).sum(axis=0)
    assert np.all(np.asarray(cd.expected_entropy()) < h_mean)


def test_update_matches_array_kernel_and_manual_outer():
    cd = make_cond_dirichlet()
    q1, q2 = _beliefs()
    obs = jnp.asarray(1, jnp.int32)
    updated = cd.update(obs, (q1, q2), lr=0.5)
    kernel = _count_update(cd.concentration, obs, (q1.probs, q2.probs), 0.5)
    assert np.allclose(np.asarray(updated.concentration), np.asarray(kernel))
    manual = np.asarray(cd.concentration).copy()
    manual[1] += 0.5 * np.outer(np.asarray(q1.probs), np.asarray(q2.probs))
    assert np.allclose(np.asarray(updated.concentration), manual, atol=1e-6)
    # one-hot observation gives the same counts; lr scales unit mass
    onehot = jax.nn.one_hot(1, 3)
    assert np.allclose(
        np.asarray(cd.update(onehot, (q1, q2), lr=0.5).concentration),
        np.asarray(updated.concentration),
        atol=1e-6,
    )
    assert np.isclose(float((updated.concentration - cd.concentration).sum()), 0.5, atol=1e-6)


def test_conditional_dirichlet_update_pymdp_parity():
    _, A_deps, A, num_obs, priors, obs = _random_model(11)
    rng = np.random.default_rng(101)
    pA = [np.float32(rng.gamma(2.0, size=a.shape) + 0.1) for a in A]
    _, posteriors = _run_both(A_deps, A, num_obs, priors, obs)
    qA_ref, _ = update_obs_likelihood_dirichlet(
        [jnp.asarray(p) for p in pA],
        [jnp.asarray(a) for a in A],
        [jnp.asarray([o]) for o in obs],
        [q.probs[None] for q in posteriors],
        A_dependencies=A_deps,
        categorical_obs=False,
        num_obs=num_obs,
        lr=1.0,
    )
    for m, deps in enumerate(A_deps):
        cd = ConditionalDirichlet(concentration=jnp.asarray(pA[m]), parent_ndims=len(deps))
        upd = cd.update(jnp.asarray(obs[m], jnp.int32), tuple(posteriors[f] for f in deps), lr=1.0)
        assert np.allclose(np.asarray(upd.concentration), np.asarray(qA_ref[m]), atol=1e-5)


# ---------------------------------------------------------------- gaussian


def test_conditional_gaussian_condition_on_equals_indexing():
    cg = make_cond_gaussian()
    for j in range(3):
        g = cg.condition_on((j,))
        assert np.allclose(np.asarray(g.loc), np.asarray(cg.loc[j]))
        assert np.allclose(np.asarray(g.chol), np.asarray(cg.chol[j]))


def test_propagate_accepts_conditional_gaussian():
    M, dim = 3, 2
    rng = np.random.default_rng(7)
    Fs = jnp.asarray(
        np.float32(np.stack([np.eye(dim) + 0.1 * rng.normal(size=(dim, dim)) for _ in range(M)]))
    )
    bs = jnp.asarray(np.float32(rng.normal(size=(M, dim))))
    Q_chols = jnp.asarray(np.float32(np.stack([np.eye(dim) * s for s in (0.5, 0.8, 1.1)])))
    tlw = jnp.log(jnp.asarray([0.2, 0.3, 0.5]))
    m = MixtureGaussian(
        log_weights=jnp.log(jnp.asarray([0.4, 0.6])),
        means=jnp.asarray([[-1.0, 0.0], [1.0, 0.5]]),
        chols=jnp.stack([jnp.eye(dim) * 0.9, jnp.eye(dim) * 1.2]),
    )
    noise = ConditionalGaussian(loc=bs, chol=Q_chols, parent_ndims=1)
    via_arrays = propagate(m, Fs, bs, Q_chols, tlw)
    via_conditional = propagate(m, Fs, noise, transition_log_weights=tlw)
    for a, b in zip(
        jax.tree_util.tree_leaves(via_arrays),
        jax.tree_util.tree_leaves(via_conditional),
        strict=True,
    ):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    with pytest.raises(ValueError):
        propagate(m, Fs, noise, Q_chols, tlw)


# --------------------------------------------------------------------- fpi


def test_fpi_accepts_conditional_categorical_likelihoods():
    _, A_deps, A, num_obs, priors, obs = _random_model(0)
    prior_cats = [Categorical(logits=jnp.asarray(_log_stable(p))) for p in priors]
    obs_arrs = [jnp.asarray(o, dtype=jnp.int32) for o in obs]
    raw = fpi([jnp.asarray(_log_stable(a)) for a in A], A_deps, prior_cats, obs_arrs)
    ccs = [
        ConditionalCategorical.from_probs(jnp.asarray(a), parent_ndims=len(deps))
        for a, deps in zip(A, A_deps, strict=True)
    ]
    ours = fpi(ccs, A_deps, prior_cats, obs_arrs)
    assert isinstance(ours, tuple)
    for q_raw, q_cc in zip(raw, ours, strict=True):
        assert np.allclose(np.asarray(q_cc.probs), np.asarray(q_raw.probs), atol=1e-5)


def test_fpi_dirichlet_likelihood_uses_expected_log():
    _, A_deps, A, num_obs, priors, obs = _random_model(1)
    rng = np.random.default_rng(21)
    prior_cats = [Categorical(logits=jnp.asarray(_log_stable(p))) for p in priors]
    obs_arrs = [jnp.asarray(o, dtype=jnp.int32) for o in obs]
    cds = [
        ConditionalDirichlet(
            concentration=jnp.asarray(np.float32(rng.gamma(2.0, size=a.shape) + 0.1)),
            parent_ndims=len(deps),
        )
        for a, deps in zip(A, A_deps, strict=True)
    ]
    via_objects = fpi(cds, A_deps, prior_cats, obs_arrs)
    via_tables = fpi([cd.expected_log_params() for cd in cds], A_deps, prior_cats, obs_arrs)
    for q_o, q_t in zip(via_objects, via_tables, strict=True):
        assert np.allclose(np.asarray(q_o.probs), np.asarray(q_t.probs), atol=1e-6)


# ------------------------------------------------------------------ dtypes


def test_discrete_samples_are_int32():
    cat = Categorical(logits=jnp.asarray([0.3, -0.5, 1.1]))
    assert cat.sample(KEY).dtype == jnp.int32
    assert cat.sample(KEY, (4,)).dtype == jnp.int32
    j = make_joint()
    assert j.sample(KEY).dtype == jnp.int32
    cc = make_cond_categorical()
    assert cc.condition_on((0, 1)).sample(KEY).dtype == jnp.int32
