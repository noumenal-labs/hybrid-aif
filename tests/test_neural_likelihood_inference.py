"""A neural likelihood conditioned on in inference, and a recognition entry that replaces it.

An ``MLPConditional`` declared as a modality's likelihood is read at the tick's prior belief as
the linear-Gaussian model it linearises to (``inference.linearise``) and conditioned on like an
``LGObservation``; a modality a recognition entry covers is not conditioned on and enters the free
energy as ``- E_q[log p(o | s)]`` instead; ``learn`` lowers the same free energy. Design:
recognition-beside-analytic.md L2, L3 and L5.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import (
    ConditionalCategorical,
    ConditionalGaussian,
    Gaussian,
    MLPConditional,
)
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianTransition, Recognition
from hybrid_aif.system import System, SystemParams
from tests.test_recognition_analytic import (
    IMG,
    KEY,
    OBS,
    PRIORS,
    _entry_system,
    _log_normal,
    build,
)

PROPAGATIONS = ("first_order", "unscented")
X0 = Gaussian(loc=jnp.asarray([0.3, -0.2]), chol=0.4 * jnp.eye(2))
INNOV0 = ConditionalGaussian(loc=jnp.zeros((1, 2)), chol=0.2 * jnp.eye(2)[None], parent_ndims=1)
Y3 = jnp.asarray([0.5, -0.4, 0.8])


def affine(parent_dims, d_obs, propagation="first_order", key=KEY):
    """A depth-0 conditional: an affine mean and a constant scale (its scale rows zeroed)."""
    decl = MLPConditional.from_shapes(
        parent_dims, d_obs, key=key, depth=0, frozen=True, propagation=propagation
    )
    weight = decl.layers[-1].weight.at[d_obs:].set(0.0)
    return eqx.tree_at(lambda d: d.layers[-1].weight, decl, weight)


def lg_of(decl, d, n_modes=None):
    """The ``LinearGaussianObservation`` an affine ``decl`` over ``(x[, one-hot mode])`` is."""
    d_obs = decl.child_dim
    H = decl.layers[-1].weight[:d_obs, :d]
    zero = jnp.zeros(d)
    if n_modes is None:
        mean, scale = decl._head((zero,))
        return LinearGaussianObservation(H=H, c=mean, noise=jnp.diag(scale))
    heads = [decl._head((zero, jax.nn.one_hot(i, n_modes))) for i in range(n_modes)]
    c = jnp.stack([h[0] for h in heads])
    return LinearGaussianObservation(H=H, c=c, noise=jnp.diag(heads[0][1]))


def gaussian_system(decl, recognition=(), modules=None, names=("o",)):
    system = System(
        A_deps=((0,),),
        B_deps=((),),
        controlled=(False,),
        policy_len=1,
        modality_names=names,
        factor_names=("x",),
        recognition=tuple(recognition),
    )
    params = SystemParams(
        likelihood={names[0]: decl},
        transition={"x": LinearGaussianTransition(F=0.9 * jnp.eye(2), innovations=INNOV0)},
        initial_state_prior={"x": X0},
        recognition=dict(modules or {}),
    )
    return system, params


def _same_beliefs(a, b, atol=1e-5):
    for u, v in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        np.testing.assert_allclose(np.asarray(u), np.asarray(v), atol=atol, rtol=1e-4)


# ------------------------------------------------------- (a) affine, Gaussian-carried


@pytest.mark.parametrize("propagation", PROPAGATIONS)
def test_an_affine_decoder_over_a_gaussian_factor_is_the_lg_modality(propagation):
    decl = affine((2,), 3, propagation)
    neural, p_neural = gaussian_system(decl)
    lg, p_lg = gaussian_system(lg_of(decl, 2))
    q_n, d_n = neural.infer_states(p_neural, {"o": Y3}, priors={"x": X0}, detail=True)
    q_l, d_l = lg.infer_states(p_lg, {"o": Y3}, priors={"x": X0}, detail=True)
    _same_beliefs(q_n, q_l)
    np.testing.assert_allclose(
        float(d_n.log_evidence["o"]), float(d_l.log_evidence["o"]), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(float(d_n.free_energy), float(d_l.free_energy), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(float(d_n.free_energy), -float(d_n.log_evidence["o"]), rtol=1e-5)


# ------------------------------------------------- (b) affine, mode-aligned mixture


@pytest.mark.parametrize("propagation", PROPAGATIONS)
def test_an_affine_decoder_over_a_mixture_factor_with_a_mode_parent_is_the_lg_modality(
    propagation,
):
    decl = affine((2, 3), 3, propagation)
    system, params = build()
    neural = dataclasses.replace(params, likelihood={**params.likelihood, "img": decl})
    lg = dataclasses.replace(params, likelihood={**params.likelihood, "img": lg_of(decl, 2, 3)})
    obs = {**OBS, "img": Y3}
    q_n, d_n = system.infer_states(neural, obs, priors=PRIORS, detail=True)
    q_l, d_l = system.infer_states(lg, obs, priors=PRIORS, detail=True)
    _same_beliefs(q_n, q_l)
    np.testing.assert_allclose(
        float(d_n.log_evidence["img"]), float(d_l.log_evidence["img"]), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(float(d_n.free_energy), float(d_l.free_energy), rtol=1e-4, atol=1e-4)


def test_an_affine_decoder_over_a_mixture_without_a_mode_parent_is_the_lg_modality():
    decl = affine((2,), 3)
    system, params = build()
    system = dataclasses.replace(system, A_deps=(*system.A_deps[:2], (1,)))
    neural = dataclasses.replace(params, likelihood={**params.likelihood, "img": decl})
    lg = dataclasses.replace(params, likelihood={**params.likelihood, "img": lg_of(decl, 2)})
    obs = {**OBS, "img": Y3}
    q_n, d_n = system.infer_states(neural, obs, priors=PRIORS, detail=True)
    q_l, d_l = system.infer_states(lg, obs, priors=PRIORS, detail=True)
    _same_beliefs(q_n, q_l)
    np.testing.assert_allclose(float(d_n.free_energy), float(d_l.free_energy), rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------- (c) a curved decoder


def _importance_posterior(decl, y, prior, n=400_000):
    xs = prior.loc + jax.random.normal(jax.random.PRNGKey(3), (n, 2)) @ prior.chol.T

    def logp(x):
        mean, scale = decl._head((x,))
        return -0.5 * jnp.sum(((y - mean) / scale) ** 2) - jnp.sum(jnp.log(scale))

    w = jax.nn.softmax(jax.vmap(logp)(xs))
    mean = w @ xs
    cov = ((xs - mean) * w[:, None]).T @ (xs - mean)
    return np.asarray(mean), np.asarray(cov)


@pytest.mark.parametrize("propagation", PROPAGATIONS)
def test_a_curved_decoder_posterior_matches_importance_sampling_under_a_tight_prior(propagation):
    decl = MLPConditional.from_shapes(
        (2,),
        3,
        key=jax.random.PRNGKey(1),
        width=16,
        depth=2,
        frozen=True,
        init_scale=0.15,
        propagation=propagation,
    )
    prior = Gaussian(loc=jnp.asarray([0.2, -0.1]), chol=0.15 * jnp.eye(2))
    y = decl._head((jnp.asarray([0.3, 0.0]),))[0]
    system, params = gaussian_system(decl)
    post, _ = system.infer_states(params, {"o": y}, priors={"x": prior})
    mean, cov = _importance_posterior(decl, y, prior)
    sd = np.sqrt(np.diag(cov))
    assert np.all(np.abs(np.asarray(post["x"].loc) - mean) < 0.2 * sd + 0.01)
    got_cov = np.asarray(post["x"].chol @ post["x"].chol.T)
    np.testing.assert_allclose(np.diag(got_cov), np.diag(cov), rtol=0.3)


# ---------------------------------------- (d) the covered modality: the L3 acceptance


def _identity_module(scale):
    """A depth-0 conditional whose output is its parent, with the constant ``scale``."""
    d = len(scale)
    decl = MLPConditional.from_shapes((d,), d, key=KEY, depth=0, frozen=True)
    last = decl.layers[-1]
    weight = jnp.concatenate([jnp.eye(d), jnp.zeros((d, d))], axis=0)
    raw = jnp.log(jnp.expm1(jnp.asarray(scale) - decl.min_scale))
    bias = jnp.concatenate([jnp.zeros(d), raw])
    layer = eqx.tree_at(lambda m: (m.weight, m.bias), last, (weight, bias))
    return eqx.tree_at(lambda m: m.layers, decl, (layer,))


def test_an_entry_whose_message_is_the_lg_message_gives_the_uncovered_posterior_and_energy():
    scale = jnp.asarray([0.3, 0.5])
    lg = LinearGaussianObservation(H=jnp.eye(2), c=jnp.zeros(2), noise=jnp.diag(scale))
    y = jnp.asarray([0.6, -0.1])
    open_sys, open_params = gaussian_system(lg)
    ref, ref_detail = open_sys.infer_states(open_params, {"o": y}, priors={"x": X0}, detail=True)
    entry = (Recognition("x", ("obs:o",)),)
    system, params = gaussian_system(lg, entry, {"x": _identity_module(scale)})
    got, detail = system.infer_states(params, {"o": y}, priors={"x": X0}, detail=True)
    _same_beliefs(got, ref)
    np.testing.assert_allclose(
        float(detail.free_energy), float(ref_detail.free_energy), rtol=1e-5, atol=1e-5
    )


def test_a_covered_lg_modality_over_a_mixture_gives_the_uncovered_posterior_and_energy():
    ref, ref_detail = build(img=False)[0].infer_states(
        build(img=False)[1], OBS, priors=PRIORS, detail=True
    )
    scale = jnp.asarray([0.4, 0.4])
    system, params = build(
        img=False,
        recognition=(Recognition("x", ("obs:fix",)),),
        modules={"x": _identity_module(scale)},
    )
    got, detail = system.infer_states(params, OBS, priors=PRIORS, detail=True)
    _same_beliefs(got, ref)
    np.testing.assert_allclose(
        float(detail.free_energy), float(ref_detail.free_energy), rtol=1e-4, atol=1e-4
    )


# --------------------------------- (e) a covered modality is conditioned on only by the entry


def test_a_covered_modality_has_no_evidence_and_its_expectation_is_in_the_detail():
    system, params = build(
        img=False,
        recognition=(Recognition("x", ("obs:fix",)),),
        modules={"x": _identity_module(jnp.asarray([0.4, 0.4]))},
    )
    got, detail = system.infer_states(params, OBS, priors=PRIORS, detail=True)
    assert detail.log_evidence["fix"] is None
    q = got["x"]
    w = np.exp(np.asarray(q.log_weights, np.float64))
    R = 0.4**2 * np.eye(2)
    expected = 0.0
    for j in range(len(w)):
        S = np.asarray(q.chols[j], np.float64)
        S = S @ S.T
        expected += w[j] * (
            -_log_normal(OBS["fix"], q.means[j], R) + 0.5 * np.trace(np.linalg.solve(R, S))
        )
    np.testing.assert_allclose(float(detail.covered_terms["fix"]), expected, rtol=1e-4, atol=1e-4)


def test_a_covered_neural_modality_is_not_linearised_into_the_update():
    system, params = _entry_system()
    got, detail = system.infer_states(params, {**OBS, "img": IMG}, priors=PRIORS, detail=True)
    assert detail.log_evidence["img"] is None
    assert np.isfinite(float(detail.covered_terms["img"]))
    uncovered, _ = build()[0].infer_states(
        build()[1], {**OBS, "img": IMG}, priors=PRIORS, detail=True
    )
    assert float(jnp.max(jnp.abs(got["x"].mean() - uncovered["x"].mean()))) > 1e-4


# ---------------------------------------------------------------- (f) present= masks


def test_present_masks_a_neural_modality_statically_and_traced():
    decl = affine((2,), 3)
    system, params = gaussian_system(decl)
    prior = {"x": X0}
    got, detail = system.infer_states(
        params, {"o": None}, priors=prior, detail=True, present={"o": False}
    )
    _same_beliefs(got, prior)
    assert float(detail.free_energy) == 0.0

    @jax.jit
    def run(flag):
        return system.infer_states(
            params, {"o": Y3}, priors=prior, detail=True, present={"o": flag}
        )

    off, off_detail = run(jnp.asarray(False))
    _same_beliefs(off, prior)
    assert float(off_detail.log_evidence["o"]) == 0.0
    on, on_detail = run(jnp.asarray(True))
    ref, ref_detail = system.infer_states(params, {"o": Y3}, priors=prior, detail=True)
    _same_beliefs(on, ref)
    np.testing.assert_allclose(
        float(on_detail.free_energy), float(ref_detail.free_energy), rtol=1e-5
    )


# ------------------------------------------------------ (g) two continuous factors


def test_a_neural_modality_over_two_continuous_factors_is_refused_by_name():
    decl = MLPConditional.from_shapes((2, 2), 3, key=KEY, depth=0, frozen=True)
    system = System(
        A_deps=((0, 1),),
        B_deps=((), ()),
        controlled=(False, False),
        policy_len=1,
        modality_names=("o",),
        factor_names=("a", "b"),
    )
    transition = LinearGaussianTransition(F=0.9 * jnp.eye(2), innovations=INNOV0)
    params = SystemParams(
        likelihood={"o": decl},
        transition={"a": transition, "b": transition},
        initial_state_prior={"a": X0, "b": X0},
    )
    with pytest.raises(NotImplementedError, match=r"exactly one continuous factor.*L1"):
        system.infer_states(params, {"o": Y3}, priors={"a": X0, "b": X0})


# --------------------------------------------------------------------- (h) learning


def test_learn_lowers_the_free_energy_of_a_decoder():
    decl = MLPConditional.from_shapes(
        (2,), 3, key=jax.random.PRNGKey(4), width=8, depth=1, init_scale=0.3
    )
    system, params = gaussian_system(decl)
    state = system.init_learn_state(params)
    energies = []
    for _ in range(40):
        beliefs, detail = system.infer_states(params, {"o": Y3}, priors={"x": X0}, detail=True)
        energies.append(float(detail.free_energy))
        params, state = system.learn(params, {"o": Y3}, beliefs, lr=1e-2, learn_state=state)
    assert energies[-1] < energies[0] - 2.0
    assert all(np.isfinite(energies))
    assert np.mean(energies[-5:]) < np.mean(energies[:5])


def _flat(tree):
    return jnp.concatenate(
        [x.ravel() for x in jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))]
    )


@pytest.mark.parametrize("prior_sd", (0.1, 1.0))
@pytest.mark.parametrize("decl_kind", ("affine", "curved"))
def test_learn_on_a_decoder_matches_the_gradient_of_the_free_energy(decl_kind, prior_sd):
    if decl_kind == "affine":
        decl = dataclasses.replace(affine((2,), 3), frozen=False)
    else:
        decl = MLPConditional.from_shapes(
            (2,), 3, key=jax.random.PRNGKey(4), width=8, depth=1, init_scale=0.8
        )
    prior = Gaussian(loc=X0.loc, chol=prior_sd * jnp.eye(2))
    system, params = gaussian_system(decl)

    def free_energy(d):
        p = dataclasses.replace(params, likelihood={"o": d})
        return system.infer_states(p, {"o": Y3}, priors={"x": prior}, detail=True)[1].free_energy

    from hybrid_aif.system.faces import learn as _learn

    pin = system._params_in(params, "t")
    got = _learn._likelihood_stats(
        system, pin, 0, decl, (Y3,), system._beliefs_in(params, {"x": prior}, "t"), None, None, "t"
    )
    want = eqx.filter_grad(free_energy)(decl)
    np.testing.assert_allclose(
        np.asarray(_flat(got)), np.asarray(_flat(want)), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("deps", ((1,), (1, 0)))
def test_learn_on_a_curved_decoder_over_a_mixture_matches_the_gradient_of_the_free_energy(deps):
    parent_dims = (2, 3) if len(deps) == 2 else (2,)
    decl = MLPConditional.from_shapes(
        parent_dims, 3, key=jax.random.PRNGKey(5), width=8, depth=1, init_scale=0.8
    )
    system, params = build()
    system = dataclasses.replace(system, A_deps=(*system.A_deps[:2], deps))
    params = dataclasses.replace(params, likelihood={**params.likelihood, "img": decl})
    obs = {**OBS, "img": Y3}

    def free_energy(d):
        p = dataclasses.replace(params, likelihood={**params.likelihood, "img": d})
        return system.infer_states(p, obs, priors=PRIORS, detail=True)[1].free_energy

    from hybrid_aif.system.faces import learn as _learn

    pin = system._params_in(params, "t")
    got = _learn._likelihood_stats(
        system,
        pin,
        2,
        decl,
        tuple(obs[n] for n in system.modality_names),
        system._beliefs_in(params, PRIORS, "t"),
        None,
        None,
        "t",
    )
    want = eqx.filter_grad(free_energy)(decl)
    np.testing.assert_allclose(
        np.asarray(_flat(got)), np.asarray(_flat(want)), rtol=1e-4, atol=1e-5
    )


def test_learn_reads_the_tick_priors_it_is_given():
    decl = MLPConditional.from_shapes(
        (2,), 3, key=jax.random.PRNGKey(4), width=8, depth=1, init_scale=0.8
    )
    system, params = gaussian_system(decl)
    state = system.init_learn_state(params)
    wide = {"x": Gaussian(loc=X0.loc, chol=1.0 * jnp.eye(2))}
    beliefs, _ = system.infer_states(params, {"o": Y3}, priors=wide)
    by_default, _ = system.learn(params, {"o": Y3}, beliefs, lr=1e-2, learn_state=state)
    given, _ = system.learn(params, {"o": Y3}, beliefs, lr=1e-2, learn_state=state, priors=wide)
    apart = _flat(given.likelihood["o"]) - _flat(by_default.likelihood["o"])
    assert float(jnp.max(jnp.abs(apart))) > 1e-6


@pytest.mark.parametrize("mixture_k", (6, 18))
def test_a_neural_modality_after_a_switching_one_reads_the_regrown_mixture(mixture_k):
    decl = affine((2,), 3)
    system, params = build()  # img switches x on terrain, and runs first
    system = dataclasses.replace(
        system,
        A_deps=(*system.A_deps, (1,)),
        modality_names=(*system.modality_names, "late"),
        mixture_k=mixture_k,
    )
    obs = {**OBS, "img": IMG, "late": Y3}

    def run(likelihood):
        p = dataclasses.replace(params, likelihood={**params.likelihood, "late": likelihood})
        return system.infer_states(p, obs, priors=PRIORS, detail=True)

    got, got_detail = run(decl)
    ref, ref_detail = run(lg_of(decl, 2))
    _same_beliefs(got, ref)
    np.testing.assert_allclose(
        float(got_detail.free_energy), float(ref_detail.free_energy), rtol=1e-5, atol=1e-5
    )


# ------------------------------------------- a neural likelihood over categorical parents


def test_a_neural_likelihood_over_categorical_parents_is_a_row_in_the_sweep():
    tab = MLPConditional.from_shapes((3,), 4, key=KEY, width=8, depth=1, frozen=True)
    system, params = build(img=False)
    system = dataclasses.replace(
        system, A_deps=(*system.A_deps, (0,)), modality_names=(*system.modality_names, "t")
    )
    y = jnp.asarray([0.1, -0.2, 0.3, 0.0])
    row = jnp.stack([tab.log_prob(y, (jax.nn.one_hot(i, 3),)) for i in range(3)])
    # a table whose outcome-0 probability is exp(row - C) at every level, C one constant
    const = jnp.log(2.0 * jnp.max(jnp.exp(row)))
    other = jnp.log(jnp.exp(const) - jnp.exp(row))
    log_table = jnp.stack([row - const, other - const], axis=0)  # one column per parent level
    neural = dataclasses.replace(params, likelihood={**params.likelihood, "t": tab})
    table = dataclasses.replace(
        params,
        likelihood={
            **params.likelihood,
            "t": ConditionalCategorical(log_table=log_table, parent_ndims=1),
        },
    )
    q_n, d_n = system.infer_states(neural, {**OBS, "t": y}, priors=PRIORS, detail=True)
    q_t, d_t = system.infer_states(table, {**OBS, "t": jnp.asarray(0)}, priors=PRIORS, detail=True)
    _same_beliefs(q_n["terrain"], q_t["terrain"])
    np.testing.assert_allclose(
        float(d_t.free_energy), float(d_n.free_energy) + float(const), rtol=1e-4, atol=1e-4
    )
