"""Online structure learning (design: structure-learning.md).

S2, the Gaussian emission over categorical parents alone: its Student-t predictive against
quadrature over the matrix-normal inverse-Wishart posterior, its message against a NumPy mixture
responsibility, its conjugate update at zero regressor width against the hand update, and the
declaration through every face (validation, the environment, planning, the description).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, stats

from hybrid_aif import (
    ConditionalDirichlet,
    Dirichlet,
    LinearGaussianObservation,
    System,
    SystemParams,
    build,
    describe,
    gaussian_emission,
)
from hybrid_aif.description.mdl import description_length
from hybrid_aif.factors import LinearGaussianObservationMNIW
from hybrid_aif.inference.emission import emission_scores, is_emission

# -------------------------------------------------------------------------- the score


def test_the_alias_is_the_mniw_family_with_a_width_one_head():
    decl = gaussian_emission(jnp.asarray([[0.0, 1.0], [2.0, 3.0]]), 0.5 * jnp.eye(2), strength=2.0)
    assert isinstance(decl, LinearGaussianObservationMNIW) and is_emission(decl)
    assert decl.M.shape == (2, 2, 1) and decl.H.shape == (2, 2, 0)
    np.testing.assert_allclose(np.asarray(decl.c), [[0.0, 1.0], [2.0, 3.0]])
    np.testing.assert_allclose(np.asarray(decl.R_chol[1]), 0.5 * np.eye(2), rtol=1e-6)


def test_student_t_predictive_matches_quadrature_over_the_posterior():
    """One dimension: ``p(y) = ∫ N(y; m, r (1 + 1/v)) IW(r; ψ, ν) dr`` over the noise, the head
    integrated in closed form; the inverse-Wishart in one dimension is InvGamma(ν/2, ψ/2)."""
    m, v_inv, psi, nu = 0.7, 2.5, 1.8, 4.0
    decl = LinearGaussianObservationMNIW(
        M=jnp.asarray([[[m]]]),
        V_inv=jnp.asarray([[[v_inv]]]),
        Psi=jnp.asarray([[[psi]]]),
        nu=jnp.asarray([nu]),
    )
    for y in (-1.3, 0.2, 2.9):
        got = float(emission_scores(decl, jnp.asarray([y]))[0])

        def integrand(r, y=y):
            return stats.norm.pdf(y, m, np.sqrt(r * (1.0 + 1.0 / v_inv))) * stats.invgamma.pdf(
                r, nu / 2.0, scale=psi / 2.0
            )

        want, _ = integrate.quad(integrand, 0.0, np.inf)
        np.testing.assert_allclose(got, np.log(want), rtol=1e-5)


def test_student_t_predictive_matches_monte_carlo_in_two_dimensions():
    decl = gaussian_emission(jnp.asarray([[0.5, -0.5]]), jnp.asarray([[0.8, 0.0], [0.3, 0.6]]), 3.0)
    y = jnp.asarray([0.9, -0.2])
    keys = jax.random.split(jax.random.PRNGKey(3), 200_000)
    draws = jax.vmap(lambda k: decl.sample(k))(keys)
    dens = jax.vmap(
        lambda c, chol: jnp.exp(
            -0.5 * jnp.sum(jax.scipy.linalg.solve_triangular(chol, y - c, lower=True) ** 2)
            - jnp.sum(jnp.log(jnp.diagonal(chol)))
            - jnp.log(2.0 * jnp.pi)
        )
    )(draws.c[:, 0], draws.R_chol[:, 0])
    np.testing.assert_allclose(
        float(emission_scores(decl, y)[0]), float(jnp.log(jnp.mean(dens))), atol=0.03
    )


def test_a_point_emission_scores_the_gaussian_density():
    decl = LinearGaussianObservation(
        H=jnp.zeros((3, 2, 0)),
        c=jnp.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, -1.0]]),
        noise=0.5 * jnp.eye(2),
    )
    assert is_emission(decl)
    y = np.asarray([0.3, 0.9])
    got = np.asarray(emission_scores(decl, jnp.asarray(y)))
    want = [stats.multivariate_normal.logpdf(y, c, 0.25 * np.eye(2)) for c in np.asarray(decl.c)]
    np.testing.assert_allclose(got, want, rtol=1e-5)


# ---------------------------------------------------------------------- through faces


def _mixture_system(k=3, d=2, point=False, extra_tabular=False, levels_type=2):
    """``z`` memoryless over ``k`` levels, ``y`` a Gaussian emission over it; optionally a
    tabular modality ``type`` over ``z`` too."""
    mods = ("y", "type") if extra_tabular else ("y",)
    system = System(
        A_deps=((0,),) * len(mods),
        B_deps=((),),
        controlled=(False,),
        policy_len=1,
        modality_names=mods,
        factor_names=("z",),
        decomposition="risk_ambiguity",
        epistemic=False,
    )
    rng = np.random.default_rng(0)
    loc = jnp.asarray(rng.normal(size=(k, d)) * 2.0)
    chols = jnp.asarray(np.stack([0.5 * np.eye(d) for _ in range(k)]))
    if point:
        emission = LinearGaussianObservation(H=jnp.zeros((k, d, 0)), c=loc, noise=chols)
    else:
        emission = gaussian_emission(loc, chols, strength=2.0)
    likelihood = {"y": emission}
    if extra_tabular:
        table = rng.dirichlet(np.ones(levels_type), size=k).T  # (levels_type, k)
        likelihood["type"] = ConditionalDirichlet(
            concentration=jnp.asarray(3.0 * table), parent_ndims=1
        )
    weights = jnp.asarray(rng.dirichlet(np.ones(k)))
    params = SystemParams(
        likelihood=likelihood,
        transition={
            "z": ConditionalDirichlet(concentration=4.0 * weights[:, None], parent_ndims=1)
        },
        initial_state_prior={"z": Dirichlet(concentration=4.0 * weights)},
    )
    return system, params


def _numpy_responsibility(prior, scores):
    log_q = np.log(prior) + scores
    log_q -= log_q.max()
    q = np.exp(log_q)
    return q / q.sum()


@pytest.mark.parametrize("point", [False, True])
def test_the_message_is_the_exact_mixture_responsibility(point):
    system, params = _mixture_system(point=point)
    y = jnp.asarray([0.4, -1.1])
    beliefs, detail = system.infer_states(params, {"y": y}, None, detail=True)
    prior = np.asarray(system.initial_priors(params)["z"].probs)
    scores = np.asarray(emission_scores(params.likelihood["y"], y))
    np.testing.assert_allclose(
        np.asarray(beliefs["z"].probs), _numpy_responsibility(prior, scores), rtol=1e-5
    )
    # one categorical factor: −F is the exact log evidence log Σ_j p(j) p(y | j)
    want = np.log(np.sum(prior * np.exp(scores)))
    np.testing.assert_allclose(-float(detail.free_energy), want, rtol=1e-5)
    assert not bool(detail.unsupported["z"])  # a density supports every level


def test_the_emission_composes_with_a_tabular_modality_as_tabular_evidence():
    system, params = _mixture_system(extra_tabular=True)
    y, t = jnp.asarray([0.4, -1.1]), jnp.asarray(1, jnp.int32)
    beliefs, _ = system.infer_states(params, {"y": y, "type": t}, None)
    prior = np.asarray(system.initial_priors(params)["z"].probs)
    scores = np.asarray(emission_scores(params.likelihood["y"], y))
    table = np.asarray(params.likelihood["type"].expected_log_params())  # E[log p(type | z)]
    np.testing.assert_allclose(
        np.asarray(beliefs["z"].probs), _numpy_responsibility(prior, scores + table[1]), rtol=1e-5
    )


def test_an_absent_emission_leaves_the_prior_statically_and_traced():
    system, params = _mixture_system()
    prior = system.initial_priors(params)
    y = jnp.asarray([0.4, -1.1])
    static, _ = system.infer_states(params, {"y": None}, None, present={"y": False})
    np.testing.assert_allclose(np.asarray(static["z"].probs), np.asarray(prior["z"].probs))
    traced, _ = system.infer_states(params, {"y": y}, None, present={"y": jnp.asarray(False)})
    np.testing.assert_allclose(np.asarray(traced["z"].probs), np.asarray(prior["z"].probs))


def test_learn_at_zero_regressor_width_is_the_responsibility_weighted_update():
    system, params = _mixture_system()
    y = jnp.asarray([0.4, -1.1])
    beliefs, _ = system.infer_states(params, {"y": y}, None)
    learned = system.learn(params, {"y": y}, beliefs, lr_by_slot={"transition:*": 0.0})
    w = np.asarray(beliefs["z"].probs)
    before, after = params.likelihood["y"], learned.likelihood["y"]
    for j in range(3):
        # per level: V⁻¹ += w, (M V⁻¹) += w y, Φ += w y yᵀ, ν += w
        vinv = float(before.V_inv[j, 0, 0]) + w[j]
        nat = np.asarray(before.M[j]) * float(before.V_inv[j, 0, 0]) + w[j] * np.asarray(y)[:, None]
        m_new = nat / vinv
        phi = (
            np.asarray(before.Psi[j])
            + np.asarray(before.M[j]) @ np.asarray(before.M[j]).T * float(before.V_inv[j, 0, 0])
            + w[j] * np.outer(y, y)
        )
        psi_new = phi - nat @ m_new.T
        np.testing.assert_allclose(float(after.V_inv[j, 0, 0]), vinv, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(after.M[j]), m_new, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(after.Psi[j]), psi_new, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(float(after.nu[j]), float(before.nu[j]) + w[j], rtol=1e-6)
    # the weights' slot absorbs the posterior as a count when its rate is on
    learned = system.learn(params, {"y": y}, beliefs, prev_beliefs=beliefs)
    np.testing.assert_allclose(
        np.asarray(learned.transition["z"].concentration[:, 0]),
        np.asarray(params.transition["z"].concentration[:, 0]) + w,
        rtol=1e-6,
    )


def test_a_point_emission_with_a_learnable_noise_learns_the_scatter_per_level():
    from hybrid_aif.distributions import InverseWishart

    system, params = _mixture_system(point=True)
    decl = params.likelihood["y"]
    noise = InverseWishart(
        Psi=jnp.broadcast_to(2.0 * jnp.eye(2), (3, 2, 2)), nu=jnp.full((3,), 4.0)
    )
    params = dataclasses.replace(
        params, likelihood={"y": LinearGaussianObservation(head=decl.head, noise=noise)}
    )
    y = jnp.asarray([0.4, -1.1])
    beliefs, _ = system.infer_states(params, {"y": y}, None)
    learned = system.learn(params, {"y": y}, beliefs, lr_by_slot={"transition:*": 0.0})
    w = np.asarray(beliefs["z"].probs)
    for j in range(3):
        e = np.asarray(y) - np.asarray(decl.c[j])
        np.testing.assert_allclose(
            np.asarray(learned.likelihood["y"].noise.Psi[j]),
            2.0 * np.eye(2) + w[j] * np.outer(e, e),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(learned.likelihood["y"].noise.nu[j]), 4.0 + w[j], rtol=1e-6
        )


def test_the_environment_draws_from_the_selected_level():
    system, params = _mixture_system()
    env, env_params = system.as_env(params)
    keys = jax.random.split(jax.random.PRNGKey(1), 4000)
    obs, drawn = jax.vmap(lambda k: env.reset(k, env_params))(keys)
    ys, states = np.asarray(obs["y"]), np.asarray(drawn["z"])
    for j in range(3):
        assert np.sum(states == j) > 300
        np.testing.assert_allclose(
            ys[states == j].mean(0), np.asarray(params.likelihood["y"].c[j]), atol=0.08
        )


def test_planning_runs_with_an_emission_and_scores_its_ambiguity():
    """Selected (per-variable-infogain.md V11), the emission carries its ambiguity under
    ``risk_ambiguity``; selected or not, its implicit flat preference's risk ``−H[q(y)]``."""
    system, params = _mixture_system()
    beliefs = system.initial_priors(params)
    _, unselected = system.infer_policies(params, beliefs)
    assert set(unselected) == {"risk"}
    system = dataclasses.replace(system, epistemic=("z",))
    _, tree = system.infer_policies(params, beliefs)
    np.testing.assert_array_equal(
        np.asarray(tree["risk"]["y"].lower), np.asarray(unselected["risk"]["y"].lower)
    )
    leaf = tree["ambiguity"]["y"]
    probs = np.asarray(beliefs["z"].probs)
    # one tick ahead the prior is the weights' table; the leaf is their entropy average
    weights = np.asarray(params.transition["z"].mean()[:, 0])
    h = 0.5 * 2 * (1.0 + np.log(2 * np.pi)) + 2 * np.log(0.5)
    del probs
    np.testing.assert_allclose(float(leaf.lower[0]), float(np.sum(weights * h)), rtol=1e-5)


def test_refusals_name_the_emission():
    system, params = _mixture_system()
    wide = LinearGaussianObservation(
        H=jnp.zeros((3, 2, 1)), c=params.likelihood["y"].c, noise=0.5 * jnp.eye(2)
    )
    with pytest.raises(ValueError, match="gaussian_emission"):
        system.infer_states(
            dataclasses.replace(params, likelihood={"y": wide}), {"y": jnp.zeros(2)}, None
        )
    from hybrid_aif import Gaussian

    targeted = dataclasses.replace(
        params, obs_targets={"y": Gaussian(loc=jnp.zeros(2), chol=jnp.eye(2))}
    )
    with pytest.raises(NotImplementedError, match="Gaussian emission's observation"):
        system.infer_policies(targeted, system.initial_priors(params))
    # a continuous factor beside the categorical parent is not an emission's parent
    system2 = System(
        A_deps=((0, 1),),
        B_deps=((), ()),
        controlled=(False, False),
        policy_len=1,
        modality_names=("y",),
        factor_names=("z", "x"),
    )
    from hybrid_aif import Gaussian as G
    from hybrid_aif.distributions import ConditionalGaussian
    from hybrid_aif.factors import LinearGaussianTransition

    params2 = SystemParams(
        likelihood={"y": params.likelihood["y"]},
        transition={
            "z": params.transition["z"],
            "x": LinearGaussianTransition(
                F=jnp.eye(1),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 1)), chol=jnp.ones((1, 1, 1)), parent_ndims=1
                ),
            ),
        },
        initial_state_prior={
            "z": params.initial_state_prior["z"],
            "x": G(loc=jnp.zeros(1), chol=jnp.eye(1)),
        },
    )
    with pytest.raises(ValueError, match="whose parents are all categorical"):
        system2.infer_states(params2, {"y": jnp.zeros(2)}, None)


def test_the_emission_describes_round_trips_and_is_priced():
    system, params = _mixture_system(extra_tabular=True)
    document, arrays = describe(system, params, arrays="inline")
    rebuilt_system, rebuilt = build(document, arrays=arrays)
    assert isinstance(rebuilt.likelihood["y"], LinearGaussianObservationMNIW)
    assert rebuilt.likelihood["y"].M.shape == (3, 2, 1)
    for a, b in zip(
        jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(rebuilt), strict=True
    ):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert rebuilt_system == system
    priced = description_length(system, 10.0, params=params)
    assert np.isfinite(priced.total)


# ==================================================================== S3: growth


def _nw_predictive(m, kappa, psi, nu, y):
    """NumPy: the Student-t predictive of a normal-inverse-Wishart level at ``y``."""
    d = y.shape[-1]
    df = nu - d + 1.0
    scale = psi * (1.0 + 1.0 / kappa) / df
    return stats.multivariate_t.logpdf(y, loc=m, shape=scale, df=df)


def _nw_absorb(m, kappa, psi, nu, y, w):
    """NumPy: the conjugate update of a normal-inverse-Wishart level by ``y`` at weight ``w``."""
    kappa2 = kappa + w
    m2 = (kappa * m + w * y) / kappa2
    psi2 = psi + w * np.outer(y, y) + kappa * np.outer(m, m) - kappa2 * np.outer(m2, m2)
    return m2, kappa2, psi2, nu + w


def _growable_system(k_max=6, d=2, alpha=1.0, strength=0.1, prior_scale=1.0, **kwargs):
    """``z`` growable over ``k_max`` levels, one live; ``y`` a Gaussian emission over it with a
    broad prior centred at the origin."""
    system = System(
        A_deps=((0,),),
        B_deps=((),),
        controlled=(False,),
        policy_len=1,
        modality_names=("y",),
        factor_names=("z",),
        epistemic=False,
        **kwargs,
    )
    weights = jnp.zeros((k_max, 1)).at[0, 0].set(1.0)
    emission = gaussian_emission(
        jnp.zeros((k_max, d)),
        jnp.broadcast_to(prior_scale * jnp.eye(d), (k_max, d, d)),
        strength=strength,
    )
    params = SystemParams(
        likelihood={"y": emission},
        transition={
            "z": ConditionalDirichlet(
                concentration=weights, parent_ndims=1, growth=jnp.asarray(alpha)
            )
        },
        initial_state_prior={
            "z": Dirichlet(concentration=weights[:, 0], growth=jnp.asarray(alpha))
        },
    )
    return system, params


def _stream(n=60, seed=1):
    rng = np.random.default_rng(seed)
    centres = np.asarray([[8.0, 8.0], [-8.0, 8.0], [0.0, -10.0]])
    labels = rng.integers(0, 3, size=n)
    return (centres[labels] + 0.3 * rng.normal(size=(n, 2))).astype(np.float32), labels


def _numpy_algorithm_1(params, ys, alpha):
    """AXIOM Algorithm 1 with N = 1, E = 1 per datum, in NumPy: NIW levels, a Dirichlet over the
    weights with dead levels at zero, the first dead level woken when the best live level
    explains the datum worse than the prior by ``log α``; the woken level takes the datum hard
    and is seeded by its conjugate update, its weight ``α`` plus the datum."""
    decl = params.likelihood["y"]
    k = decl.M.shape[0]
    levels = [
        (
            np.asarray(decl.M[j, :, 0], np.float64),
            float(decl.V_inv[j, 0, 0]),
            np.asarray(decl.Psi[j], np.float64),
            float(decl.nu[j]),
        )
        for j in range(k)
    ]
    counts = np.asarray(params.transition["z"].concentration[:, 0], np.float64)
    for y in ys:
        y = np.asarray(y, np.float64)
        live = counts > 0
        scores = np.asarray([_nw_predictive(*levels[j], y) for j in range(k)])
        dead = np.flatnonzero(~live)
        grow = dead.size > 0 and scores[live].max() < scores[dead[0]] + np.log(alpha)
        if grow:
            q = np.zeros(k)
            q[dead[0]] = 1.0
        else:
            logits = np.where(
                live, np.log(np.where(live, counts, 1.0) / counts.sum()) + scores, -np.inf
            )
            q = np.exp(logits - logits.max())
            q /= q.sum()
        levels = [_nw_absorb(*levels[j], y, q[j]) for j in range(k)]
        counts = counts + q
        if grow:
            counts[dead[0]] += alpha
    return levels, counts


def test_the_growth_step_is_algorithm_1_on_a_stream():
    """Per tick through ``infer_states`` + ``learn``, against the NumPy reference: the same levels
    wake at the same ticks, and the level statistics agree."""
    system, params = _growable_system()
    ys, _ = _stream()
    levels, counts = _numpy_algorithm_1(params, ys, 1.0)
    p = params
    for y in ys:
        beliefs, _ = system.infer_states(p, {"y": jnp.asarray(y)}, None)
        p = system.learn(p, {"y": jnp.asarray(y)}, beliefs, prev_beliefs=beliefs)
    got_counts = np.asarray(p.transition["z"].concentration[:, 0])
    np.testing.assert_allclose(got_counts, counts, rtol=1e-4, atol=1e-5)
    assert int(np.sum(got_counts > 0)) == 3  # one level per cluster, three dead
    np.testing.assert_array_equal(np.asarray(p.initial_state_prior["z"].concentration), got_counts)
    decl = p.likelihood["y"]
    for j in range(6):
        m, kappa, psi, nu = levels[j]
        np.testing.assert_allclose(np.asarray(decl.M[j, :, 0]), m, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(float(decl.V_inv[j, 0, 0]), kappa, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(decl.Psi[j]), psi, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(float(decl.nu[j]), nu, rtol=1e-5)


def test_a_woken_level_is_seeded_from_the_datum_alone():
    """The woken level's statistics are the dead level's prior absorbing the datum at weight
    one, bitwise, and its weight is the growth propensity plus that one count."""
    system, params = _growable_system(alpha=0.5)
    y0, y1 = jnp.asarray([0.2, -0.1]), jnp.asarray([8.0, 8.0])
    beliefs, _ = system.infer_states(params, {"y": y0}, None)
    first = system.learn(params, {"y": y0}, beliefs, prev_beliefs=beliefs)
    assert int(np.sum(np.asarray(first.transition["z"].concentration[:, 0]) > 0)) == 1
    beliefs, _ = system.infer_states(first, {"y": y1}, None)
    learned = system.learn(first, {"y": y1}, beliefs, prev_beliefs=beliefs)
    before, after = first.likelihood["y"], learned.likelihood["y"]
    at = (jnp.arange(6) == 1)[:, None, None]
    seeded = before.absorb(
        before.stats(
            jnp.outer(y1, y1)[None] * at,
            (y1[:, None])[None] * at,
            jnp.ones((6, 1, 1)) * at,
            at[:, 0, 0].astype(jnp.float32),
        )
    )
    for name in ("M", "V_inv", "Psi", "nu"):
        assert np.array_equal(np.asarray(getattr(after, name)), np.asarray(getattr(seeded, name)))
    np.testing.assert_allclose(np.asarray(after.nu), np.asarray(before.nu) + (np.arange(6) == 1))
    counts = np.asarray(learned.transition["z"].concentration[:, 0])
    np.testing.assert_allclose(counts, [2.0, 1.5, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(learned.initial_state_prior["z"].concentration), counts)


def test_no_growth_without_a_dead_level_or_below_the_threshold():
    system, params = _growable_system(k_max=1)
    y = jnp.asarray([4.0, 4.0])
    beliefs, _ = system.infer_states(params, {"y": y}, None)
    learned = system.learn(params, {"y": y}, beliefs, prev_beliefs=beliefs)
    np.testing.assert_allclose(np.asarray(learned.transition["z"].concentration[:, 0]), [2.0])
    # a datum the live level explains better than the prior does keeps the live set
    system, params = _growable_system()
    y0 = jnp.asarray([0.1, -0.1])
    beliefs, _ = system.infer_states(params, {"y": y0}, None)
    first = system.learn(params, {"y": y0}, beliefs, prev_beliefs=beliefs)
    beliefs, _ = system.infer_states(first, {"y": y0}, None)
    second = system.learn(first, {"y": y0}, beliefs, prev_beliefs=beliefs)
    assert int(np.sum(np.asarray(second.transition["z"].concentration[:, 0]) > 0)) == 1


def test_the_batch_form_wakes_one_level_per_round_and_absorbs_the_window():
    """From a level sharp on one cluster, a window with a second cluster wakes one level for
    its worst-explained datum, which then takes that cluster; the window is absorbed once; a
    second round wakes nothing more; a later window with a third cluster wakes a third level."""
    system, params = _growable_system()
    ys, labels = _stream(n=60)
    p = params
    for y in ys[labels == 0][:5]:  # level 0 learns the first cluster tick by tick
        beliefs, _ = system.infer_states(p, {"y": jnp.asarray(y)}, None)
        p = system.learn(p, {"y": jnp.asarray(y)}, beliefs, prev_beliefs=beliefs)
    assert int(np.sum(np.asarray(p.transition["z"].concentration[:, 0]) > 0)) == 1
    window = jnp.asarray(ys[labels != 1][:20])
    one = system.grow(p, {"y": window}, "z", rounds=1)
    counts = np.asarray(one.transition["z"].concentration[:, 0])
    assert int(np.sum(counts > 0)) == 2
    before = float(p.transition["z"].concentration.sum())
    np.testing.assert_allclose(counts.sum(), before + 1.0 + 20, rtol=1e-5)  # once, plus α
    np.testing.assert_allclose(
        np.asarray(one.likelihood["y"].nu).sum(), float(p.likelihood["y"].nu.sum()) + 20, rtol=1e-5
    )
    np.testing.assert_allclose(np.asarray(one.likelihood["y"].c[1]), [0.0, -10.0], atol=0.5)
    np.testing.assert_array_equal(np.asarray(one.initial_state_prior["z"].concentration), counts)
    two = system.grow(p, {"y": window}, "z", rounds=2)
    assert int(np.sum(np.asarray(two.transition["z"].concentration[:, 0]) > 0)) == 2
    later = system.grow(one, {"y": jnp.asarray(ys[labels == 1][:10])}, "z", rounds=1)
    assert int(np.sum(np.asarray(later.transition["z"].concentration[:, 0]) > 0)) == 3
    np.testing.assert_allclose(np.asarray(later.likelihood["y"].c[2]), [-8.0, 8.0], atol=0.5)


def test_growth_refusals():
    system, params = _growable_system()
    with pytest.raises(ValueError, match="not a growable factor"):
        system.grow(params, {"y": jnp.zeros((3, 2))}, "w")
    with pytest.raises(ValueError, match="windows are missing"):
        system.grow(params, {}, "z")
    bad = dataclasses.replace(
        params,
        initial_state_prior={
            "z": Dirichlet(concentration=params.initial_state_prior["z"].concentration)
        },
    )
    with pytest.raises(ValueError, match="mirrors its weights"):
        system.learn(
            bad,
            {"y": jnp.zeros(2)},
            system.initial_priors(params),
            prev_beliefs=system.initial_priors(params),
        )
    with pytest.raises(ValueError, match="strictly positive"):
        ConditionalDirichlet(
            concentration=jnp.ones((2, 1)), parent_ndims=1, growth=jnp.asarray(0.0)
        )


# ================================================================= S4: reduction


def test_dirichlet_evidence_with_dead_levels_is_the_live_family():
    from hybrid_aif.learning import absorbed_count, log_evidence

    prior = Dirichlet(concentration=jnp.asarray([1.0, 0.0, 2.0, 0.0]), growth=jnp.asarray(0.5))
    post = Dirichlet(concentration=jnp.asarray([4.0, 0.0, 3.5, 0.0]), growth=jnp.asarray(0.5))
    live_prior = Dirichlet(concentration=jnp.asarray([1.0, 2.0]))
    live_post = Dirichlet(concentration=jnp.asarray([4.0, 3.5]))
    np.testing.assert_allclose(
        float(log_evidence(prior, post)), float(log_evidence(live_prior, live_post)), rtol=1e-6
    )
    np.testing.assert_allclose(float(absorbed_count(prior, post)), 4.5)
    # a woken level: its prior pseudo-count is the growth propensity
    woken = Dirichlet(concentration=jnp.asarray([4.0, 2.5, 3.5, 0.0]), growth=jnp.asarray(0.5))
    w_prior = Dirichlet(concentration=jnp.asarray([1.0, 0.5, 2.0]))
    w_post = Dirichlet(concentration=jnp.asarray([4.0, 2.5, 3.5]))
    np.testing.assert_allclose(
        float(log_evidence(prior, woken)), float(log_evidence(w_prior, w_post)), rtol=1e-6
    )
    np.testing.assert_allclose(float(absorbed_count(prior, woken)), 6.5)


def _assign(system, params, ys, level):
    """Absorb each datum of ``ys`` into ``level`` hard (growth off through a one-hot belief)."""
    from hybrid_aif.learning.structure import hard_assignment

    p = params
    for y in ys:
        belief = {"z": hard_assignment(6, jnp.asarray(level))}
        p = system._params_out(
            system._learn(
                system._p(p),
                (jnp.asarray(y),),
                (belief["z"],),
                prev_beliefs=(belief["z"],),
                grow=False,
            )
        )
    return p


def test_merge_equals_the_refit_on_the_pooled_data():
    from hybrid_aif.learning.structure import merge_levels

    system, params = _growable_system()
    weights = jnp.zeros((6, 1)).at[0, 0].set(1.0).at[1, 0].set(1.0)
    params = dataclasses.replace(
        params,
        transition={"z": dataclasses.replace(params.transition["z"], concentration=weights)},
        initial_state_prior={
            "z": dataclasses.replace(params.initial_state_prior["z"], concentration=weights[:, 0])
        },
    )
    ys, _ = _stream(n=12)
    a, b = ys[:5], ys[5:]
    separate = _assign(system, _assign(system, params, a, 0), b, 1)
    merged = system._params_out(
        merge_levels(
            system, system._p(separate), system._p(params), 0, jnp.asarray(0), jnp.asarray(1)
        )
    )
    refit = _assign(system, params, ys, 0)
    for name in ("M", "V_inv", "Psi", "nu"):
        np.testing.assert_allclose(
            np.asarray(getattr(merged.likelihood["y"], name)),
            np.asarray(getattr(refit.likelihood["y"], name)),
            rtol=1e-4,
            atol=1e-4,
        )
    # the killed level returns to its prior; the weights pool the data and keep level 0's prior
    for name in ("M", "V_inv", "Psi", "nu"):
        np.testing.assert_allclose(
            np.asarray(getattr(merged.likelihood["y"], name)[1]),
            np.asarray(getattr(params.likelihood["y"], name)[1]),
        )
    np.testing.assert_allclose(
        np.asarray(merged.transition["z"].concentration[:, 0]), [13.0, 0, 0, 0, 0, 0]
    )
    np.testing.assert_allclose(
        np.asarray(merged.initial_state_prior["z"].concentration), [13.0, 0, 0, 0, 0, 0]
    )


def test_reduce_merges_duplicate_levels_and_keeps_distinct_ones():
    system, params = _growable_system()
    weights = jnp.zeros((6, 1)).at[0, 0].set(1.0).at[1, 0].set(1.0).at[2, 0].set(1.0)
    params = dataclasses.replace(
        params,
        transition={"z": dataclasses.replace(params.transition["z"], concentration=weights)},
        initial_state_prior={
            "z": dataclasses.replace(params.initial_state_prior["z"], concentration=weights[:, 0])
        },
    )
    rng = np.random.default_rng(4)
    near = (np.asarray([4.0, 4.0]) + 0.3 * rng.normal(size=(30, 2))).astype(np.float32)
    far = (np.asarray([-4.0, 4.0]) + 0.3 * rng.normal(size=(15, 2))).astype(np.float32)
    fitted = _assign(
        system, _assign(system, _assign(system, params, near[:15], 0), near[15:], 1), far, 2
    )
    reduced = system.reduce(fitted, params)
    counts = np.asarray(reduced.transition["z"].concentration[:, 0])
    assert int(np.sum(counts > 0)) == 2
    assert counts[2] == 16.0  # the distinct level is untouched
    assert counts[0] + counts[1] == 31.0 and min(counts[0], counts[1]) == 0.0
    # the merged level sits on the shared cluster; the reduced model's evidence is no lower
    from hybrid_aif.learning.structure import factor_evidence

    assert float(factor_evidence(system, system._p(reduced), system._p(params), 0)) >= float(
        factor_evidence(system, system._p(fitted), system._p(params), 0)
    )
    # a second pass finds nothing to merge
    again = system.reduce(reduced, params)
    np.testing.assert_array_equal(
        np.asarray(again.transition["z"].concentration),
        np.asarray(reduced.transition["z"].concentration),
    )


def test_the_schedule_reduces_every_nth_learn_call():
    system, params = _growable_system(reduction_period=3)
    weights = jnp.zeros((6, 1)).at[0, 0].set(1.0).at[1, 0].set(1.0)
    params = dataclasses.replace(
        params,
        transition={"z": dataclasses.replace(params.transition["z"], concentration=weights)},
        initial_state_prior={
            "z": dataclasses.replace(params.initial_state_prior["z"], concentration=weights[:, 0])
        },
    )
    rng = np.random.default_rng(5)
    near = (np.asarray([4.0, 4.0]) + 0.3 * rng.normal(size=(30, 2))).astype(np.float32)
    fitted = _assign(system, _assign(system, params, near[:15], 0), near[15:], 1)
    state = system.init_learn_state(fitted)
    p = fitted
    y = jnp.asarray([4.0, 4.0])
    for call in range(1, 4):
        beliefs, _ = system.infer_states(p, {"y": y}, None)
        p, state = system.learn(
            p, {"y": y}, beliefs, prev_beliefs=beliefs, learn_state=state, prior=params
        )
        live = int(np.sum(np.asarray(p.transition["z"].concentration[:, 0]) > 0))
        assert live == (1 if call == 3 else 2)
    with pytest.raises(ValueError, match="pass prior="):
        system.learn(p, {"y": y}, beliefs, prev_beliefs=beliefs, learn_state=state)
    with pytest.raises(ValueError, match="counter rides learn_state"):
        system.learn(p, {"y": y}, beliefs, prev_beliefs=beliefs, prior=params)


# ============================================================ S6: the description


def test_a_growable_factor_with_dead_levels_round_trips_bitwise():
    system, params = _growable_system(reduction_period=5, reduction_pairs=3)
    ys, _ = _stream(n=20)
    p = system.grow(params, {"y": jnp.asarray(ys)}, "z", rounds=2)
    document, arrays = describe(system, p, arrays="inline")
    assert document["configuration"]["learning"] == {"reduction_period": 5, "reduction_pairs": 3}
    assert document["variables"]["B_z_concentration"]["domain"] == "nonnegative"
    assert "B_z_growth" in document["clamps"]
    rebuilt_system, rebuilt = build(document, arrays=arrays)
    assert rebuilt_system == system
    for a, b in zip(jax.tree_util.tree_leaves(p), jax.tree_util.tree_leaves(rebuilt), strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert float(rebuilt.transition["z"].growth) == 1.0


# ============================================================ S8: the lagged datum


def _lagged_system(lagged=(("prev_x", "x"), ("prev_type", "kind"))):
    """``x`` a 2-D random walk observed by ``obs_x``; ``kind`` a categorical factor observed by
    ``obs_kind``; ``z`` growable, emitting the previous ``x`` (a Gaussian emission) and the
    previous ``kind`` (a tabular modality), both read from the carry."""
    from hybrid_aif import Categorical, ConditionalCategorical, Gaussian
    from hybrid_aif.distributions import ConditionalGaussian
    from hybrid_aif.factors import LinearGaussianTransition

    system = System(
        A_deps=((0,), (1,), (2,), (2,)),
        B_deps=((), (1,), ()),
        controlled=(False, False, False),
        policy_len=1,
        modality_names=("obs_x", "obs_kind", "prev_x", "prev_type"),
        factor_names=("x", "kind", "z"),
        epistemic=False,
        lagged=lagged,
    )
    k = 4
    weights = jnp.zeros((k, 1)).at[0, 0].set(1.0)
    params = SystemParams(
        likelihood={
            "obs_x": LinearGaussianObservation(
                H=jnp.eye(2), c=jnp.zeros(2), noise=0.3 * jnp.eye(2)
            ),
            "obs_kind": ConditionalCategorical(jnp.log(jnp.eye(3) * 0.9 + 0.05), 1),
            "prev_x": gaussian_emission(
                jnp.zeros((k, 2)), jnp.broadcast_to(jnp.eye(2), (k, 2, 2)), strength=0.1
            ),
            "prev_type": ConditionalDirichlet(concentration=jnp.ones((3, k)), parent_ndims=1),
        },
        transition={
            "x": LinearGaussianTransition(
                F=jnp.eye(2),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 2)),
                    chol=jnp.broadcast_to(0.5 * jnp.eye(2), (1, 2, 2)),
                    parent_ndims=1,
                ),
            ),
            "kind": ConditionalCategorical(jnp.log(jnp.eye(3) * 0.8 + 0.1), 1),
            "z": ConditionalDirichlet(
                concentration=weights, parent_ndims=1, growth=jnp.asarray(1.0)
            ),
        },
        initial_state_prior={
            "x": Gaussian(loc=jnp.zeros(2), chol=jnp.eye(2)),
            "kind": Categorical(logits=jnp.zeros(3)),
            "z": Dirichlet(concentration=weights[:, 0], growth=jnp.asarray(1.0)),
        },
    )
    return system, params


def test_a_lagged_datum_is_the_previous_belief_mean():
    system, params = _lagged_system()
    obs = {"obs_x": jnp.asarray([1.0, -1.0]), "obs_kind": jnp.asarray(2, jnp.int32)}
    first, _ = system.infer_states(params, {**obs, "prev_x": None, "prev_type": None}, None)
    # without a carry the lagged modalities are absent: z keeps its prior
    np.testing.assert_allclose(np.asarray(first["z"].probs), [1.0, 0, 0, 0])
    nxt = {"obs_x": jnp.asarray([1.2, -0.8]), "obs_kind": jnp.asarray(2, jnp.int32)}
    second, _ = system.infer_states(
        params, {**nxt, "prev_x": None, "prev_type": None}, None, previous=first
    )
    # the same as handing the previous means over as observations
    explicit, _ = system.infer_states(
        params, {**nxt, "prev_x": first["x"].mean(), "prev_type": first["kind"].probs}, None
    )
    np.testing.assert_allclose(
        np.asarray(second["z"].probs), np.asarray(explicit["z"].probs), rtol=1e-6
    )
    # an explicit datum wins over the carry
    other, _ = system.infer_states(
        params, {**nxt, "prev_x": jnp.asarray([9.0, 9.0]), "prev_type": None}, None, previous=first
    )
    assert not np.allclose(np.asarray(other["z"].logits), np.asarray(second["z"].logits))


def test_learn_reads_the_lagged_datum_from_prev_beliefs():
    system, params = _lagged_system()
    obs = {"obs_x": jnp.asarray([1.0, -1.0]), "obs_kind": jnp.asarray(2, jnp.int32)}
    first, _ = system.infer_states(params, {**obs, "prev_x": None, "prev_type": None}, None)
    nxt = {"obs_x": jnp.asarray([1.2, -0.8]), "obs_kind": jnp.asarray(2, jnp.int32)}
    second, _ = system.infer_states(
        params, {**nxt, "prev_x": None, "prev_type": None}, None, previous=first
    )
    learned = system.learn(
        params, {**nxt, "prev_x": None, "prev_type": None}, second, prev_beliefs=first
    )
    explicit = system.learn(
        params,
        {**nxt, "prev_x": first["x"].mean(), "prev_type": first["kind"].probs},
        second,
        prev_beliefs=first,
    )
    for a, b in zip(
        jax.tree_util.tree_leaves(learned), jax.tree_util.tree_leaves(explicit), strict=True
    ):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    # the emission absorbed the carried mean at the live level (prior strength 0.1 at the origin)
    np.testing.assert_allclose(
        np.asarray(learned.likelihood["prev_x"].c[0]),
        np.asarray(first["x"].mean()) / 1.1,
        rtol=1e-5,
    )


def test_replay_fills_lagged_data_from_its_carry():
    from hybrid_aif import replay

    system, params = _lagged_system()
    T = 5
    rng = np.random.default_rng(2)
    xs = jnp.asarray(np.cumsum(rng.normal(size=(T, 2)), axis=0), jnp.float32)
    kinds = jnp.asarray(rng.integers(0, 3, size=T), jnp.int32)
    out = replay(system, params, {"obs_x": xs, "obs_kind": kinds}, {}, metrics=False)
    # tick 0: the lagged modalities are absent, z at its prior; tick t: the carry's means
    np.testing.assert_allclose(np.asarray(out.beliefs["z"].probs[0]), [1.0, 0, 0, 0])
    assert out.exercised is None
    beliefs_prev = {
        name: jax.tree_util.tree_map(lambda a: a[0], b) for name, b in out.beliefs.items()
    }
    want, _ = system.infer_states(
        params,
        {"obs_x": xs[1], "obs_kind": kinds[1], "prev_x": None, "prev_type": None},
        {name: jax.tree_util.tree_map(lambda a: a[1], b) for name, b in out.priors.items()},
        previous=beliefs_prev,
    )
    np.testing.assert_allclose(
        np.asarray(out.beliefs["z"].probs[1]), np.asarray(want["z"].probs), rtol=1e-5
    )
    # and learning along the replay absorbs the carried data on every tick that has a carry
    learned = replay(
        system, params, {"obs_x": xs, "obs_kind": kinds}, {}, metrics=False, learn=True
    )
    np.testing.assert_allclose(
        float(learned.params.likelihood["prev_x"].nu.sum()),
        float(params.likelihood["prev_x"].nu.sum()) + (T - 1),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        float(learned.params.likelihood["prev_type"].concentration.sum()),
        float(params.likelihood["prev_type"].concentration.sum()) + (T - 1),
        rtol=1e-5,
    )


def test_lagged_refusals_and_description():
    with pytest.raises(ValueError, match="neither a state factor"):
        _lagged_system(lagged=(("prev_x", "nowhere"),))
    with pytest.raises(ValueError, match="not one of"):
        _lagged_system(lagged=(("elsewhere", "x"),))
    system, params = _lagged_system(
        lagged=(("prev_x", "kind"),)
    )  # a categorical read by a Gaussian
    with pytest.raises(ValueError, match="tabular modality over 3 levels"):
        system.infer_states(
            params,
            {"obs_x": jnp.zeros(2), "obs_kind": jnp.asarray(0), "prev_x": None, "prev_type": None},
            None,
        )
    system, params = _lagged_system()
    document, arrays = describe(system, params, arrays="inline")
    assert document["factors"]["prev_x"]["likelihood"]["options"] == {"datum": "x[t-1]"}
    assert document["factors"]["prev_type"]["likelihood"]["options"] == {"datum": "kind[t-1]"}
    rebuilt_system, rebuilt = build(document, arrays=arrays)
    assert rebuilt_system == system
    for a, b in zip(
        jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(rebuilt), strict=True
    ):
        assert np.array_equal(np.asarray(a), np.asarray(b))


# ================================================================ the acceptance task


def test_the_gameworld_example_grows_and_reduces():
    """A short run of ``examples/gameworld_growth.py``: the model grows beyond its one level,
    the reduced run ends with no more live levels than the unreduced one, and every number is
    finite."""
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "examples" / "gameworld_growth.py"
    spec = importlib.util.spec_from_file_location("gameworld_growth", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    on = module.run(400, 250, window=100)
    off = module.run(400, None, window=100)
    assert on["live"].shape == (4,) and off["live"].shape == (4,)
    assert off["live"][-1] > 1 and on["live"][-1] >= 1
    assert on["live"][-1] <= off["live"][-1]
    assert np.all(np.isfinite(on["reward"])) and np.all(np.isfinite(off["reward"]))


# ======================================================== review pins (2026-10-15)


def test_no_growth_on_a_tick_without_data_even_when_alpha_exceeds_one():
    system, params = _growable_system(alpha=2.0)
    beliefs = system.initial_priors(params)
    learned = system.learn(
        params, {"y": jnp.zeros(2)}, beliefs, prev_beliefs=beliefs, lr_by_slot={"likelihood:y": 0.0}
    )
    # the datum at the origin is explained by the live level; and with the modality absent
    # through the replay mask, nothing is born
    from hybrid_aif import replay

    out = replay(
        system,
        params,
        {"y": jnp.zeros((5, 2))},
        {},
        present={"y": jnp.zeros(5, bool)},
        metrics=False,
        learn=True,
    )
    assert int(np.sum(np.asarray(out.params.transition["z"].concentration[:, 0]) > 0)) == 1
    del learned


def test_a_zero_rate_on_the_weights_freezes_structure():
    system, params = _growable_system()
    y0, y1 = jnp.asarray([0.2, -0.1]), jnp.asarray([8.0, 8.0])
    beliefs, _ = system.infer_states(params, {"y": y0}, None)
    first = system.learn(params, {"y": y0}, beliefs, prev_beliefs=beliefs)
    beliefs, _ = system.infer_states(first, {"y": y1}, None)
    for frozen in (
        system.learn(first, {"y": y1}, beliefs, prev_beliefs=beliefs, lr=0.0),
        system.learn(
            first, {"y": y1}, beliefs, prev_beliefs=beliefs, lr_by_slot={"transition:z": 0.0}
        ),
        system.learn(
            first,
            {"y": y1},
            beliefs,
            prev_beliefs=beliefs,
            lr_by_slot={"transition:z": jnp.asarray(0.0)},
        ),
    ):
        counts = np.asarray(frozen.transition["z"].concentration[:, 0])
        assert int(np.sum(counts > 0)) == 1
    grown = system.learn(first, {"y": y1}, beliefs, prev_beliefs=beliefs)
    assert int(np.sum(np.asarray(grown.transition["z"].concentration[:, 0]) > 0)) == 2


def test_replay_runs_the_reduction_schedule_against_the_given_prior():
    from hybrid_aif import replay

    system, params = _growable_system(reduction_period=4)
    rng = np.random.default_rng(6)
    ys = jnp.asarray(np.asarray([8.0, 8.0]) + 0.3 * rng.normal(size=(12, 2)), jnp.float32)
    out = replay(system, params, {"y": ys}, {}, metrics=False, learn=True)
    off_system, _ = _growable_system()
    off = replay(off_system, params, {"y": ys}, {}, metrics=False, learn=True)
    live_on = int(np.sum(np.asarray(out.params.transition["z"].concentration[:, 0]) > 0))
    live_off = int(np.sum(np.asarray(off.params.transition["z"].concentration[:, 0]) > 0))
    assert live_on <= live_off
    # a dead level's emission is at the model prior after a merge
    dead = np.flatnonzero(np.asarray(out.params.transition["z"].concentration[:, 0]) == 0)
    for name in ("M", "V_inv", "Psi", "nu"):
        np.testing.assert_allclose(
            np.asarray(getattr(out.params.likelihood["y"], name))[dead],
            np.asarray(getattr(params.likelihood["y"], name))[dead],
            rtol=1e-5,
            atol=1e-5,
        )


def test_unbuilt_readers_of_a_growable_factor_are_refused():
    from hybrid_aif import Gaussian, MixtureGaussian
    from hybrid_aif.distributions import ConditionalGaussian
    from hybrid_aif.factors import LinearGaussianTransition

    _, params = _growable_system()
    system = System(
        A_deps=((1, 0),),
        B_deps=((), (0,)),
        controlled=(False, False),
        policy_len=1,
        modality_names=("y",),
        factor_names=("z", "x"),
        epistemic=False,
        mixture_k=6,
    )
    switching = dataclasses.replace(
        params,
        likelihood={
            "y": LinearGaussianObservation(
                H=jnp.broadcast_to(jnp.eye(2), (6, 2, 2)), c=jnp.zeros((6, 2)), noise=jnp.eye(2)
            )
        },
        transition={
            "z": params.transition["z"],
            "x": LinearGaussianTransition(
                F=jnp.eye(2),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((6, 2)),
                    chol=jnp.broadcast_to(jnp.eye(2), (6, 2, 2)),
                    parent_ndims=1,
                ),
            ),
        },
        initial_state_prior={
            "z": params.initial_state_prior["z"],
            "x": MixtureGaussian.from_gaussian(Gaussian(loc=jnp.zeros(2), chol=jnp.eye(2))),
        },
    )
    with pytest.raises(NotImplementedError, match="parents the transition"):
        b = system.initial_priors(switching)
        system.learn(switching, {"y": jnp.zeros(2)}, b, prev_beliefs=b)
    # a point emission with a learnable noise
    from hybrid_aif.distributions import InverseWishart

    system, params = _growable_system()
    point = LinearGaussianObservation(
        H=jnp.zeros((6, 2, 0)),
        c=jnp.zeros((6, 2)),
        noise=InverseWishart(Psi=jnp.broadcast_to(jnp.eye(2), (6, 2, 2)), nu=jnp.full((6,), 4.0)),
    )
    point_params = dataclasses.replace(params, likelihood={"y": point})
    b = system.initial_priors(point_params)
    with pytest.raises(NotImplementedError, match="is a point emission"):
        system.learn(point_params, {"y": jnp.zeros(2)}, b, prev_beliefs=b)
    fixed = dataclasses.replace(
        params,
        likelihood={
            "y": LinearGaussianObservation(
                H=jnp.zeros((6, 2, 0)), c=jnp.zeros((6, 2)), noise=jnp.eye(2)
            )
        },
    )
    with pytest.raises(NotImplementedError, match="is a point emission"):
        system.reduce(fixed, fixed)


def test_an_emission_over_two_parents_is_the_mean_field_fixed_point():
    system = System(
        A_deps=((0, 1),),
        B_deps=((), ()),
        controlled=(False, False),
        policy_len=1,
        modality_names=("y",),
        factor_names=("z", "g"),
        epistemic=False,
        num_iters=64,
    )
    rng = np.random.default_rng(3)
    loc = jnp.asarray(rng.normal(size=(3, 2, 2)) * 2.0, jnp.float32)
    emission = gaussian_emission(loc, jnp.eye(2) * 0.7, strength=2.0)
    pz, pg = np.asarray([0.5, 0.3, 0.2]), np.asarray([0.6, 0.4])
    params = SystemParams(
        likelihood={"y": emission},
        transition={
            "z": ConditionalDirichlet(concentration=jnp.asarray(5 * pz)[:, None], parent_ndims=1),
            "g": ConditionalDirichlet(concentration=jnp.asarray(5 * pg)[:, None], parent_ndims=1),
        },
        initial_state_prior={
            "z": Dirichlet(concentration=jnp.asarray(5 * pz)),
            "g": Dirichlet(concentration=jnp.asarray(5 * pg)),
        },
    )
    y = jnp.asarray([0.5, -0.4])
    beliefs, _ = system.infer_states(params, {"y": y}, None)
    scores = np.asarray(emission_scores(emission, y))  # (3, 2)
    qz, qg = np.asarray(beliefs["z"].probs), np.asarray(beliefs["g"].probs)
    np.testing.assert_allclose(qz, _numpy_responsibility(pz, scores @ qg), rtol=1e-4)
    np.testing.assert_allclose(qg, _numpy_responsibility(pg, qz @ scores), rtol=1e-4)
    learned = system.learn(params, {"y": y}, beliefs, lr_by_slot={"transition:*": 0.0})
    np.testing.assert_allclose(
        np.asarray(learned.likelihood["y"].nu - emission.nu), np.outer(qz, qg), atol=1e-6
    )


def test_dead_level_evidence_against_enumeration():
    """The Dirichlet-multinomial over the live levels, against the sum over sequences of the
    Pólya predictive product, with a woken level whose prior pseudo-count is the growth
    propensity."""
    import itertools

    from hybrid_aif.learning import log_evidence

    alpha, prior_conc = 0.7, np.asarray([1.2, 0.0, 2.0, 0.0])
    counts = np.asarray([2.0, 1.0, 1.0, 0.0])  # level 1 woken, holding one datum
    effective = np.where((prior_conc == 0) & (counts > 0), alpha, prior_conc)
    live = np.flatnonzero(effective > 0)
    seq = np.repeat(np.arange(4), counts.astype(int))
    want = 0.0
    for order in set(itertools.permutations(seq)):
        c, lp = effective.copy(), 0.0
        for x in order:
            lp += np.log(c[x] / c[live].sum())
            c[x] += 1.0
        want = lp  # every order of one multiset has the same probability
        break
    prior = Dirichlet(concentration=jnp.asarray(prior_conc), growth=jnp.asarray(alpha))
    post = Dirichlet(concentration=jnp.asarray(effective + counts), growth=jnp.asarray(alpha))
    post = dataclasses.replace(
        post, concentration=jnp.where(jnp.asarray(effective + counts) > 0, post.concentration, 0.0)
    )
    np.testing.assert_allclose(float(log_evidence(prior, post)), want, rtol=1e-5)


def test_a_structural_zero_round_trips_without_growth():
    system = System(
        A_deps=((0,),),
        B_deps=((0,),),
        controlled=(False,),
        policy_len=1,
        modality_names=("o",),
        factor_names=("s",),
    )
    conc = jnp.asarray([[4.0, 0.0, 1.0], [0.0, 2.0, 1.0]])  # a hard zero in each column but one
    params = SystemParams(
        likelihood={"o": ConditionalDirichlet(concentration=conc, parent_ndims=1)},
        transition={"s": ConditionalDirichlet(concentration=jnp.eye(3) + 0.5, parent_ndims=1)},
        initial_state_prior={"s": Dirichlet(concentration=jnp.asarray([1.0, 0.0, 1.0]))},
    )
    document, arrays = describe(system, params, arrays="inline")
    assert document["variables"]["A_o_concentration"]["domain"] == "nonnegative"
    assert "growth" not in document["factors"]["A_o"]["prior"]["parameters"]
    _, rebuilt = build(document, arrays=arrays)
    np.testing.assert_array_equal(
        np.asarray(rebuilt.likelihood["o"].concentration), np.asarray(conc)
    )
    assert rebuilt.likelihood["o"].growth is None
    np.testing.assert_array_equal(
        np.asarray(rebuilt.initial_state_prior["s"].concentration), [1.0, 0.0, 1.0]
    )


def test_a_lagged_modality_may_be_left_out_of_the_observations():
    system, params = _lagged_system()
    obs = {"obs_x": jnp.asarray([1.0, -1.0]), "obs_kind": jnp.asarray(2, jnp.int32)}
    first, _ = system.infer_states(params, obs, None)
    second, _ = system.infer_states(params, obs, None, previous=first)
    explicit, _ = system.infer_states(
        params, {**obs, "prev_x": None, "prev_type": None}, None, previous=first
    )
    np.testing.assert_allclose(np.asarray(second["z"].probs), np.asarray(explicit["z"].probs))
    system.learn(params, obs, second, prev_beliefs=first)
