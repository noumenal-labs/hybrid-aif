"""Description length (``docs/design/description-length.md`` D1–D4, Acceptance).

Each closed form is pinned against something that does not share its algebra: the tabular
family cost against the exact multinomial NML normaliser by Kontkanen & Myllymäki's recursion;
the D1 identity between the two bookings of family + prior; the Gaussian extent against a Monte
Carlo estimate of the prior's Fisher-Rao entropy, with the Fisher determinant itself taken by
differentiating a KL; the graph code against its primitives in a verdict test. The invariances
the 0.3 note promises (names, order, placement, spelling) are pinned bitwise.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats
from scipy.special import logsumexp

import hybrid_aif as H
from hybrid_aif.description import describe
from hybrid_aif.description.mdl import (
    MENUS,
    count_code,
    description_length,
    dirichlet_fisher_rao_entropy,
    dirichlet_prior_link,
    family_code,
    gaussian_extent,
    inverse_gamma_fisher_rao_entropy,
    inverse_wishart_fisher_rao_entropy,
    kl_inverse_gamma,
    kl_inverse_wishart,
    kl_matrix_normal_inverse_wishart,
    log_star,
    parent_set_code,
    reference_head,
    reference_inverse_gamma,
    reference_noise,
    resolution_term,
    simplex_log_volume,
    tabular_family_cost,
)
from hybrid_aif.distributions import InverseGamma
from hybrid_aif.factors import LinearGaussianObservationMNIW
from hybrid_aif.replay import replay
from hybrid_aif.system import System, SystemParams
from tests.test_description import NAMES, _example, models  # noqa: F401  (the fixture)
from tests.test_learn import _sense_system, _tick_inputs

# ================================================================== D2: the codes


def test_log_star_is_rissanen_code_in_nats():
    """``log* n`` in bits is ``log₂ c₀`` plus the positive iterated logarithms; in nats, ``log* 1``
    is ``log c₀`` alone and ``log* 5`` is ``(log₂ 2.865 + log₂ 5 + log₂ log₂ 5 + log₂ log₂ log₂ 5)``
    times ``log 2``."""
    assert log_star(1) == pytest.approx(math.log(2.865064))
    bits = math.log2(2.865064) + math.log2(5) + math.log2(math.log2(5))
    bits += math.log2(math.log2(math.log2(5)))
    assert log_star(5) == pytest.approx(bits * math.log(2))
    with pytest.raises(ValueError, match="at least one"):
        log_star(0)
    assert count_code(5, cap=32) == pytest.approx(math.log(32))
    assert count_code(5) == log_star(5)
    assert parent_set_code(2, 3) == pytest.approx(log_star(3) + math.log(3))
    for label, menu in MENUS.items():
        assert family_code(label) == pytest.approx(math.log(len(menu)))
    assert "ConditionalCategorical" in MENUS["likelihood"]
    assert "JointCategorical" not in MENUS["likelihood"]  # an objective family only
    assert "MLPPredictor" not in MENUS["transition"]  # a recognition family only


# ================================================================== D3: the tabular family


def _log_multinomial_nml(n: int, levels: int) -> float:
    """``log C(n, L)``, the exact multinomial NML normaliser, by Kontkanen & Myllymäki's linear
    recursion ``C(n, k) = C(n, k−1) + (n / (k−2)) · C(n, k−2)`` from ``C(n, 1) = 1`` and the
    binomial sum ``C(n, 2) = Σ_h C(n, h) (h/n)^h ((n−h)/n)^{n−h}``, in the log domain."""
    h = np.arange(n + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = (
            stats.binom.logpmf(h, n, 0.5)
            + n * math.log(2)
            + np.where(h > 0, h * np.log(h / n), 0.0)
            + np.where(h < n, (n - h) * np.log((n - h) / n), 0.0)
        )
    log_c = {1: 0.0, 2: float(logsumexp(terms))}
    for k in range(3, levels + 1):
        log_c[k] = float(np.logaddexp(log_c[k - 1], math.log(n / (k - 2)) + log_c[k - 2]))
    return log_c[levels]


@pytest.mark.parametrize("levels", [2, 3, 5])
def test_tabular_family_cost_is_the_multinomial_nml_asymptote(levels):
    """Rissanen's two-term expansion of ``log C(n, L)`` is the family cost; the error shrinks as
    ``n`` grows and is below a twentieth of a nat at ``n = 10⁴``."""
    errors = [
        abs(_log_multinomial_nml(n, levels) - tabular_family_cost(levels, n))
        for n in (20, 200, 2000, 10000)
    ]
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 0.05


def test_the_d1_identity_on_a_tabular_family():
    """Family + prior is resolution plus the room the Jeffreys base admits plus the declared
    prior's KL from it, with the KL taken from the engine's own ``Dirichlet.kl_to``: zero for
    the Jeffreys prior itself, positive for a flat one and larger for a sharp one; and
    ``log V − KL(p ‖ Jeffreys)`` is ``h(p)``, the prior's entropy in Fisher-Rao volume."""
    n = 50.0
    costs = []
    for alpha in ([0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [10.0, 1.0, 0.3], [40.0, 40.0, 40.0]):
        a = np.asarray(alpha)
        levels = a.size
        kl = float(
            H.Dirichlet(concentration=jnp.asarray(a))
            .kl_to(H.Dirichlet(concentration=jnp.full(levels, 0.5)))
            .lower
        )
        assert dirichlet_prior_link(a) == pytest.approx(kl, rel=1e-4, abs=1e-5)
        assert simplex_log_volume(levels) - kl == pytest.approx(
            dirichlet_fisher_rao_entropy(a), rel=1e-4, abs=1e-5
        )
        costs.append(tabular_family_cost(levels, n) + dirichlet_prior_link(a))
    assert costs[0] == pytest.approx(tabular_family_cost(3, n))
    assert costs[0] < costs[1] < costs[3]
    assert dirichlet_prior_link(np.full(4, 0.5)) == pytest.approx(0.0, abs=1e-12)
    assert dirichlet_fisher_rao_entropy(np.full(4, 0.5)) == pytest.approx(simplex_log_volume(4))


def test_dirichlet_fisher_rao_entropy_against_monte_carlo():
    """``h(p) = −E_p[log p(θ) + ½ Σ log θ_i]`` (the simplex's Fisher-Rao density is
    ``Π θ_i^{−½}``) by sampling the Dirichlet."""
    rng = np.random.default_rng(3)
    alpha = np.asarray([3.0, 1.5, 0.8])
    theta = rng.dirichlet(alpha, size=400_000)
    estimate = -(stats.dirichlet.logpdf(theta.T, alpha) + 0.5 * np.log(theta).sum(-1)).mean()
    assert dirichlet_fisher_rao_entropy(alpha) == pytest.approx(estimate, abs=0.01)


# ================================================================== D4: the Gaussian extent


def _fisher_log_sqrt_det(mu: np.ndarray, sigma: np.ndarray) -> float:
    """``log √det I(μ, Σ)`` of the Gaussian family in its ``(μ, vech Σ)`` coordinates, the Fisher
    matrix taken as the Hessian of ``KL(N(θ) ‖ N(θ'))`` at ``θ' = θ``."""
    d = mu.size
    iu = np.triu_indices(d)

    def unpack(theta):
        m = theta[:d]
        s = jnp.zeros((d, d)).at[iu].set(theta[d:])
        s = s + jnp.triu(s, 1).T
        return m, s

    def kl(theta_q, theta_p):
        mq, sq = unpack(theta_q)
        mp, sp = unpack(theta_p)
        sp_inv = jnp.linalg.inv(sp)
        diff = mp - mq
        return 0.5 * (
            jnp.trace(sp_inv @ sq)
            + diff @ sp_inv @ diff
            - d
            + jnp.linalg.slogdet(sp)[1]
            - jnp.linalg.slogdet(sq)[1]
        )

    theta = jnp.concatenate([jnp.asarray(mu, jnp.float64), jnp.asarray(sigma, jnp.float64)[iu]])
    fisher = jax.hessian(kl, argnums=1)(theta, theta)
    return 0.5 * float(jnp.linalg.slogdet(fisher)[1])


@pytest.mark.parametrize("d", [1, 2])
def test_the_gaussian_fisher_rao_density(d):
    """``√det I(μ, Σ) = 2^{−d/2} |Σ|^{−(d+2)/2}``: the constant the extent formulas carry, from the
    KL Hessian rather than the formula."""
    jax.config.update("jax_enable_x64", True)
    try:
        rng = np.random.default_rng(1)
        a = rng.normal(size=(d, d))
        sigma = a @ a.T + d * np.eye(d)
        want = -0.5 * d * math.log(2) - 0.5 * (d + 2) * np.linalg.slogdet(sigma)[1]
        assert _fisher_log_sqrt_det(rng.normal(size=d), sigma) == pytest.approx(want, abs=1e-6)
    finally:
        jax.config.update("jax_enable_x64", False)


def test_gaussian_extent_one_dimension_against_monte_carlo():
    """``h(p)`` for a normal-inverse-gamma prior, ``σ² ~ IG(ν/2, ψ/2)``, ``μ | σ² ~ N(m, v σ²)``:
    ``−E[log p(μ, σ²)] + E[log(σ^{−3}/√2)]`` by sampling (the Acceptance pin)."""
    rng = np.random.default_rng(7)
    psi, nu, v, m = 2.5, 6.0, 0.7, 0.3
    size = 400_000
    var = stats.invgamma.rvs(nu / 2.0, scale=psi / 2.0, size=size, random_state=rng)
    mu = rng.normal(m, np.sqrt(v * var))
    log_p = stats.invgamma.logpdf(var, nu / 2.0, scale=psi / 2.0) + stats.norm.logpdf(
        mu, m, np.sqrt(v * var)
    )
    estimate = (-log_p - 1.5 * np.log(var) - 0.5 * math.log(2)).mean()
    assert gaussian_extent(v, np.asarray([[psi]]), nu) == pytest.approx(estimate, abs=0.02)


def test_gaussian_extent_two_dimensions_against_monte_carlo():
    """The same for a normal-inverse-Wishart prior in two dimensions: with ``μ = m + L ε``,
    ``L Lᵀ = v R``, the conditional normal's log density at the draw is
    ``−½ (d log 2π + log|v R| + ε·ε)``."""
    rng = np.random.default_rng(11)
    psi = np.asarray([[2.0, 0.3], [0.3, 1.0]])
    nu, v = 6.0, 0.5
    iw = stats.invwishart(df=nu, scale=psi)
    rs = iw.rvs(size=300_000, random_state=rng)
    eps = rng.normal(size=(rs.shape[0], 2))
    log_det = np.linalg.slogdet(rs)[1]
    log_normal = -0.5 * (
        2.0 * math.log(2.0 * math.pi) + 2.0 * math.log(v) + log_det + (eps**2).sum(-1)
    )
    log_p = iw.logpdf(np.moveaxis(rs, 0, -1)) + log_normal
    estimate = (-log_p - 2.0 * log_det - math.log(2)).mean()
    assert gaussian_extent(v, psi, nu) == pytest.approx(estimate, abs=0.05)


def test_noise_only_extents_against_monte_carlo():
    """``h(p)`` for an inverse-gamma variance and an inverse-Wishart covariance with a known
    mean (``√det I = 2^{−d/2} |R|^{−(d+1)/2}``), by sampling."""
    rng = np.random.default_rng(5)
    alpha, beta = np.asarray([2.5, 4.0]), np.asarray([1.2, 0.6])
    var = stats.invgamma.rvs(alpha, scale=beta, size=(300_000, 2), random_state=rng)
    log_p = stats.invgamma.logpdf(var, alpha, scale=beta)
    estimate = (-log_p - np.log(var) - 0.5 * math.log(2)).sum(-1).mean()
    assert inverse_gamma_fisher_rao_entropy(alpha, beta) == pytest.approx(estimate, abs=0.02)

    psi, nu = np.asarray([[1.5, 0.2], [0.2, 0.8]]), 5.0
    iw = stats.invwishart(df=nu, scale=psi)
    rs = iw.rvs(size=200_000, random_state=rng)
    log_det = np.linalg.slogdet(rs)[1]
    estimate = (-iw.logpdf(np.moveaxis(rs, 0, -1)) - 1.5 * log_det - math.log(2)).mean()
    assert inverse_wishart_fisher_rao_entropy(psi, nu) == pytest.approx(estimate, abs=0.05)


def test_the_gaussian_kls_against_monte_carlo():
    """The three KL closed forms the reference base needs, by sampling the first argument and
    scoring both densities with scipy."""
    rng = np.random.default_rng(13)
    psi1, nu1 = np.asarray([[1.5, 0.4], [0.4, 0.9]]), 7.0
    psi0, nu0 = np.asarray([[1.0, 0.0], [0.0, 1.0]]), 4.0
    iw1, iw0 = stats.invwishart(df=nu1, scale=psi1), stats.invwishart(df=nu0, scale=psi0)
    rs = iw1.rvs(size=200_000, random_state=rng)
    stacked = np.moveaxis(rs, 0, -1)
    estimate = (iw1.logpdf(stacked) - iw0.logpdf(stacked)).mean()
    assert kl_inverse_wishart(psi1, nu1, psi0, nu0) == pytest.approx(estimate, abs=0.02)
    assert kl_inverse_wishart(psi1, nu1, psi1, nu1) == pytest.approx(0.0, abs=1e-12)

    a1, b1, a0, b0 = np.asarray([3.0, 2.0]), np.asarray([0.8, 2.5]), 1.5, 0.5
    var = stats.invgamma.rvs(a1, scale=b1, size=(300_000, 2), random_state=rng)
    estimate = (
        (stats.invgamma.logpdf(var, a1, scale=b1) - stats.invgamma.logpdf(var, a0, scale=b0))
        .sum(-1)
        .mean()
    )
    assert kl_inverse_gamma(a1, b1, a0, b0) == pytest.approx(estimate, abs=0.02)

    # a 1 × 2 head: W | R ~ N(M, R V), R scalar, so the conditional is a bivariate normal
    m1, v_inv1 = np.asarray([[0.5, -0.2]]), np.asarray([[2.0, 0.3], [0.3, 1.0]])
    m0, v_inv0 = np.zeros((1, 2)), np.eye(2)
    p1, n1, p0, n0 = np.asarray([[1.2]]), 5.0, np.asarray([[1.0]]), 3.0
    r = stats.invwishart(df=n1, scale=p1).rvs(size=100_000, random_state=rng)
    v1 = np.linalg.inv(v_inv1)
    chol = np.linalg.cholesky(v1)
    w = m1[0] + np.sqrt(r)[:, None] * (rng.normal(size=(r.size, 2)) @ chol.T)
    log_q = stats.invwishart(df=n1, scale=p1).logpdf(r) + np.asarray(
        [
            stats.multivariate_normal.logpdf(w_i, m1[0], r_i * v1)
            for w_i, r_i in zip(w, r, strict=True)
        ]
    )
    log_p = stats.invwishart(df=n0, scale=p0).logpdf(r) + np.asarray(
        [
            stats.multivariate_normal.logpdf(w_i, m0[0], r_i * np.eye(2))
            for w_i, r_i in zip(w, r, strict=True)
        ]
    )
    estimate = (log_q - log_p).mean()
    got = kl_matrix_normal_inverse_wishart(m1, v_inv1, p1, n1, m0, v_inv0, p0, n0)
    assert got == pytest.approx(estimate, abs=0.03)
    assert kl_matrix_normal_inverse_wishart(m1, v_inv1, p1, n1, m1, v_inv1, p1, n1) == (
        pytest.approx(0.0, abs=1e-12)
    )


def test_the_reference_priors():
    psi0, nu0 = reference_noise(3, 2.0)
    np.testing.assert_allclose(psi0, 4.0 * np.eye(3))
    assert nu0 == 5.0  # the vaguest proper inverse-Wishart with a mean, scale² I
    assert reference_inverse_gamma(2.0) == (1.5, 2.0)  # its d = 1 case
    m0, v_inv0, psi, nu = reference_head(2, 3, 1.0)
    assert m0.shape == (2, 3) and np.array_equal(v_inv0, np.eye(3))
    np.testing.assert_allclose(psi, np.eye(2))


# ================================================================== the measure


def _prob_table(rng, shape):
    t = rng.random(shape) + 0.1
    return t / t.sum(axis=0, keepdims=True)


def _tabular_system(ns=(3,), no=(4,), nu=2, learned=(), seed=5):
    """State factors of ``ns`` levels (the first controlled with ``nu`` levels), modalities of
    ``no`` levels each reading every factor; the modalities named in ``learned`` are Dirichlet."""
    rng = np.random.default_rng(seed)
    f_count = len(ns)
    system = System(
        A_deps=tuple(tuple(range(f_count)) for _ in no),
        B_deps=tuple((f,) for f in range(f_count)),
        controlled=tuple(f == 0 for f in range(f_count)),
        policy_len=1,
        modality_names=tuple(f"o{m}" for m in range(len(no))),
        factor_names=tuple(f"s{f}" for f in range(f_count)),
    )
    likelihood = {}
    for m, levels in enumerate(no):
        table = _prob_table(rng, (levels, *ns))
        if f"o{m}" in learned:
            likelihood[f"o{m}"] = H.ConditionalDirichlet(
                concentration=jnp.asarray(4.0 * table + 0.5), parent_ndims=f_count
            )
        else:
            likelihood[f"o{m}"] = H.ConditionalCategorical(
                log_table=jnp.log(jnp.asarray(table)), parent_ndims=f_count
            )
    transition = {}
    for f, levels in enumerate(ns):
        shape = (levels, levels, nu) if f == 0 else (levels, levels, 1)
        transition[f"s{f}"] = H.ConditionalCategorical(
            log_table=jnp.log(jnp.asarray(_prob_table(rng, shape))), parent_ndims=2
        )
    initial = {f"s{f}": H.Categorical(logits=jnp.zeros(levels)) for f, levels in enumerate(ns)}
    params = SystemParams(
        likelihood=likelihood,
        transition=transition,
        initial_state_prior=initial,
        obs_targets={"o0": H.Categorical(logits=jnp.zeros(no[0]))},
    )
    return system, params


def test_the_links_of_a_tabular_model_are_what_d2_and_d3_say():
    """Every address of a small tabular model, recomputed from the primitives."""
    system, params = _tabular_system(ns=(3,), no=(4,), nu=2, learned=("o0",))
    n = 100.0
    dl = description_length(system, n, params=params)
    candidates = 2  # one state factor and one control node
    graph = dl.per_address["graph"]
    assert graph.graph == pytest.approx(log_star(1) + log_star(2) + log_star(2))
    assert dl.per_address["control:u_s0"].graph == pytest.approx(log_star(2))
    init = dl.per_address["initial_state_prior:s0"]
    # an initial prior can only read earlier factors: its pool is the factors alone
    assert init.graph == pytest.approx(family_code("initial") + parent_set_code(0, 1))
    assert init.family == pytest.approx(tabular_family_cost(3, n))
    assert init.prior is None and not init.exact
    trans = dl.per_address["transition:s0"]
    assert trans.graph == pytest.approx(
        family_code("transition") + parent_set_code(2, candidates) + log_star(3)
    )
    assert trans.family == pytest.approx(3 * 2 * tabular_family_cost(3, n))
    o0 = dl.per_address["likelihood:o0"]
    assert o0.graph == pytest.approx(
        family_code("likelihood") + parent_set_code(1, candidates) + log_star(4)
    )
    assert o0.family == pytest.approx(3 * tabular_family_cost(4, n))
    conc = np.moveaxis(np.asarray(params.likelihood["o0"].concentration), 0, -1)
    assert o0.prior == pytest.approx(dirichlet_prior_link(conc))
    assert o0.exact and o0.posterior is None
    assert dl.structure == pytest.approx(
        math.fsum(link.structure for link in dl.per_address.values())
    )
    assert dl.parameters == pytest.approx(o0.prior)
    assert dl.total == pytest.approx(dl.structure + dl.parameters)
    assert dl.bits == pytest.approx(
        (dl.structure / math.log(2), dl.parameters / math.log(2), dl.total / math.log(2))
    )
    assert not dl.exact
    assert ("likelihood:o0", "posterior") in dl.unscored
    assert ("initial_state_prior:s0", "prior") in dl.unscored


def test_verdict_a_factor_added_and_a_level_added_move_structure_by_their_own_amounts():
    """ "Add a state factor" and "add a level to one" move ``structure`` by amounts that differ
    and are each what the codes say: the first by the factor's two records, the count code's
    step and every parent set's wider candidate pool; the second by the level count's code
    and the family costs of every table that gained a row or a level."""
    n = 200.0
    system, params = _tabular_system(ns=(3,), no=(4,), nu=2)
    dl = description_length(system, n, params=params)
    sys_f, par_f = _tabular_system(ns=(3, 2), no=(4,), nu=2)
    with_factor = description_length(sys_f, n, params=par_f)
    sys_l, par_l = _tabular_system(ns=(4,), no=(4,), nu=2)
    with_level = description_length(sys_l, n, params=par_l)
    factor_delta = with_factor.structure - dl.structure
    level_delta = with_level.structure - dl.structure
    assert factor_delta != pytest.approx(level_delta)

    # the factor: counts, its own two records, and a wider candidate pool for every parent set
    want = log_star(2) - log_star(1)
    want += family_code("initial") + parent_set_code(0, 2) + tabular_family_cost(2, n)
    want += family_code("transition") + parent_set_code(1, 3) + log_star(2)
    want += 2 * tabular_family_cost(2, n)
    want += parent_set_code(0, 2) - parent_set_code(0, 1)  # s0's initial, one more factor
    want += parent_set_code(2, 3) - parent_set_code(2, 2)  # s0's transition
    want += parent_set_code(2, 3) - parent_set_code(1, 2)  # o0 now reads both factors
    want += 2 * 3 * tabular_family_cost(4, n) - 3 * tabular_family_cost(4, n)  # o0's rows
    assert factor_delta == pytest.approx(want, abs=1e-9)

    # the level: the count code and the tables that grew
    want = log_star(4) - log_star(3)
    want += tabular_family_cost(4, n) - tabular_family_cost(3, n)  # the initial prior
    want += 4 * 2 * tabular_family_cost(4, n) - 3 * 2 * tabular_family_cost(3, n)
    want += 4 * tabular_family_cost(4, n) - 3 * tabular_family_cost(4, n)  # o0's rows
    assert level_delta == pytest.approx(want, abs=1e-9)


def test_the_cap_rule_prices_declared_room_and_the_family_prices_live_levels():
    """A factor with five levels of which two are dead (no record puts mass on them) costs
    ``log 5`` for its count and the family costs of three live levels over live rows; the same
    factor declared with three levels costs ``log* 3`` and the same family costs."""
    n = 100.0

    def declared(levels, dead):
        rng = np.random.default_rng(2)  # the same live tables under either declaration
        live = levels - dead
        table = np.zeros((levels, levels, 2))
        table[:live, :live] = _prob_table(rng, (live, live, 2))
        conc = np.zeros((levels, levels, 2))
        conc[:live, :live] = 4.0 * table[:live, :live] + 0.5
        obs = np.zeros((4, levels))
        obs[:, :live] = _prob_table(rng, (4, live))
        initial = np.full(levels, -np.inf)
        initial[:live] = 0.0
        system = System(
            A_deps=((0,),),
            B_deps=((0,),),
            controlled=(True,),
            policy_len=1,
            modality_names=("o",),
            factor_names=("z",),
        )
        params = SystemParams(
            likelihood={
                "o": H.ConditionalCategorical(log_table=jnp.log(jnp.asarray(obs)), parent_ndims=1)
            },
            transition={
                "z": H.ConditionalDirichlet(concentration=jnp.asarray(conc), parent_ndims=2)
            },
            initial_state_prior={"z": H.Categorical(logits=jnp.asarray(initial))},
            obs_targets={"o": H.Categorical(logits=jnp.zeros(4))},
        )
        return description_length(system, n, params=params)

    capped, outright = declared(5, 2), declared(3, 0)
    cap_t, out_t = capped.per_address["transition:z"], outright.per_address["transition:z"]
    assert cap_t.graph - out_t.graph == pytest.approx(math.log(5) - log_star(3))
    assert (
        cap_t.family
        == pytest.approx(out_t.family)
        == pytest.approx(3 * 2 * tabular_family_cost(3, n))
    )
    assert cap_t.prior == pytest.approx(out_t.prior)
    assert np.isfinite(cap_t.prior)
    for key in ("initial_state_prior:z", "likelihood:o"):
        assert capped.per_address[key].family == pytest.approx(outright.per_address[key].family)


def test_the_posterior_link_reads_a_prior_beside_the_params():
    """With the params a slot was learned from passed as ``prior``, the posterior link is
    ``KL(Dir(after) ‖ Dir(before))`` summed over the rows, and the prior link is still the
    declared prior's distance from Jeffreys."""
    system, params = _sense_system()
    obs, beliefs = _tick_inputs(system, params)
    after = system.learn(params, obs, beliefs)
    dl = description_length(system, 10.0, params=after, prior=params)
    link = dl.per_address["likelihood:sense"]
    before, now = params.likelihood["sense"], after.likelihood["sense"]
    want = 0.0
    for j in range(now.concentration.shape[1]):
        q = H.Dirichlet(concentration=now.concentration[:, j])
        p = H.Dirichlet(concentration=before.concentration[:, j])
        want += float(q.kl_to(p).lower)
    assert link.posterior == pytest.approx(want, rel=1e-5)
    assert link.prior == pytest.approx(
        dirichlet_prior_link(np.moveaxis(np.asarray(now.concentration), 0, -1))
    )
    assert dl.parameters == pytest.approx(link.prior + link.posterior)
    assert (
        description_length(system, 10.0, params=after).per_address["likelihood:sense"].posterior
        is None
    )


def test_the_resolution_is_one_number_or_one_per_address():
    system, params = _tabular_system(ns=(3,), no=(4,), learned=("o0",))
    by_address = description_length(
        system, {"likelihood:o0": 50.0, "transition:s0": 10.0}, params=params
    )
    assert by_address.per_address["likelihood:o0"].n == 50.0
    assert by_address.per_address["transition:s0"].family == pytest.approx(
        6 * tabular_family_cost(3, 10.0)
    )
    unpriced = by_address.per_address["initial_state_prior:s0"]
    assert unpriced.family is None and unpriced.n is None
    assert ("initial_state_prior:s0", "family") in by_address.unscored
    none = description_length(system, None, params=params)
    assert all(link.family is None for link in none.per_address.values())
    assert ("transition:s0", "family") in none.unscored
    assert none.parameters == pytest.approx(by_address.parameters)
    for bad in (0.0, -1, "100", True, {"likelihood:o0": 0}):
        with pytest.raises(ValueError, match="must be positive"):
            description_length(system, bad, params=params)
    with pytest.raises(TypeError, match="needs the params"):
        description_length(system, 10.0)


def test_a_one_hot_initial_belief_is_not_dead_room(models):  # noqa: F811
    """The T-maze starts at a known location: its initial record puts mass on one level, but
    the transition reaches every level, so all five are live and priced; a level with no mass
    in either record is dead."""
    from hybrid_aif.description.mdl import _live_levels

    system, params = models["tmaze"]
    n = 100.0
    dl = description_length(system, n, params=params)
    trans = dl.per_address["transition:location"]
    assert trans.family == pytest.approx(25 * tabular_family_cost(5, n))
    assert trans.graph == pytest.approx(
        family_code("transition") + parent_set_code(2, 3) + log_star(5)
    )
    assert dl.per_address["initial_state_prior:location"].family == pytest.approx(
        tabular_family_cost(5, n)
    )
    initial = H.Categorical(logits=jnp.asarray([0.0, -jnp.inf, -jnp.inf]))
    table = jnp.log(jnp.asarray([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]))
    reaches_two = H.ConditionalCategorical(log_table=table, parent_ndims=1)
    assert _live_levels(initial, reaches_two).tolist() == [True, True, False]


def test_non_tabular_parameter_counts_follow_the_declared_shapes(models):  # noqa: F811
    """Every non-tabular family's ``k`` recomputed from its arrays: a shared field counts once
    and a per-mode field once per mode; a mixture adds its weights; a net counts its sizes."""
    n = 1000.0
    system, params = models["hill_lg"]
    dl = description_length(system, n, params=params)
    pos = system._params_in(params, "test")
    d = pos.initial_state_prior[0].loc.shape[-1]
    assert dl.per_address["initial_state_prior:state"].family == pytest.approx(
        resolution_term(d + d * (d + 1) // 2, n)
    )
    lgt = pos.transition[0]
    modes = int(np.prod(np.shape(lgt.innovations.chol)[:-2]))
    k = np.size(lgt.F) + np.size(lgt.innovations.loc) + modes * (d * (d + 1) // 2)
    if lgt.G is not None:
        k += np.size(lgt.G)
    assert dl.per_address["transition:state"].family == pytest.approx(resolution_term(k, n))
    lgo = pos.likelihood[0]
    d_obs = lgo.head.shape[-2]
    k = np.size(lgo.head) + int(np.prod(np.shape(lgo.noise)[:-2])) * (d_obs * (d_obs + 1) // 2)
    assert dl.per_address["likelihood:position"].family == pytest.approx(resolution_term(k, n))

    system, params = models["switching_filter"]
    dl = description_length(system, n, params=params)
    pos = system._params_in(params, "test")
    mix = pos.initial_state_prior[1]
    k_mix, d = mix.means.shape[-2], mix.means.shape[-1]
    assert dl.per_address["initial_state_prior:x"].family == pytest.approx(
        resolution_term(k_mix - 1 + k_mix * (d + d * (d + 1) // 2), n)
    )
    lgt = pos.transition[1]
    modes = int(np.prod(np.shape(lgt.innovations.chol)[:-2]))
    assert np.ndim(lgt.F) > 2  # a per-mode F, counted once per mode
    k = np.size(lgt.F) + np.size(lgt.innovations.loc) + modes * (d * (d + 1) // 2)
    if lgt.G is not None:
        k += np.size(lgt.G)
    assert dl.per_address["transition:x"].family == pytest.approx(resolution_term(k, n))

    example = _example("noise_learning")
    learner = example.make_params(
        InverseGamma.from_R_chol(jnp.full((2, 1, 1), 0.6), strength=2.0, anchor=True)
    )
    dl = description_length(example.make_system(), n, params=learner)
    pose = learner.likelihood["pose"]
    k = np.size(pose.head) + np.size(pose.noise.alpha)  # one shared head, two noise modes
    room = 2 * inverse_gamma_fisher_rao_entropy(*reference_inverse_gamma(1.0))
    assert dl.per_address["likelihood:pose"].family == pytest.approx(resolution_term(k, n) + room)

    system, params = models["tiny_rssm"]
    dl = description_length(system, n, params=params)
    pos = system._params_in(params, "test")
    k = sum(np.size(leaf) for leaf in jax.tree_util.tree_leaves(pos.transition[0]))
    assert dl.per_address["transition:deter"].family == pytest.approx(resolution_term(k, n))
    assert dl.per_address["control:u_deter"].graph == pytest.approx(log_star(1))


def test_a_conditional_initial_prior_codes_its_parent_set(models):  # noqa: F811
    system, params = models["hierarchical_tmaze"]
    dl = description_length(system, 100.0, params=params)
    link = dl.per_address["initial_state_prior:context"]
    assert link.graph == pytest.approx(family_code("initial") + parent_set_code(1, 3))
    assert link.family == pytest.approx(3 * tabular_family_cost(2, 100.0))


def _lg_system(decl, dx: int):
    """One continuous factor of ``dx`` dimensions (one when the emission reads no regressor)
    with one linear-Gaussian emission ``decl``; the emission's parents follow its head."""
    d = max(dx, 1)
    deps = (0,) if dx > 0 else ()
    system = System(
        A_deps=(deps,),
        B_deps=((0,),),
        controlled=(False,),
        policy_len=1,
        modality_names=("y",),
        factor_names=("x",),
    )
    params = SystemParams(
        likelihood={"y": decl},
        transition={
            "x": H.LinearGaussianTransition(
                F=jnp.eye(d), innovations=H.Gaussian(loc=jnp.zeros(d), chol=jnp.eye(d))
            )
        },
        initial_state_prior={"x": H.Gaussian(loc=jnp.zeros(d), chol=jnp.eye(d))},
    )
    return system, params


@pytest.mark.parametrize("modes", [1, 3])
def test_a_matrix_normal_inverse_wishart_prior_is_priced_from_the_reference(modes):
    """A declared MNIW prior's link is its KL from the reference at the scale given, summed
    over its modes; a sharper declaration costs more; with a regressor the room is unscored but
    the KL still scores."""
    from hybrid_aif.factors import LinearGaussianObservationMNIW

    def declared(strength):
        head = jnp.zeros((modes, 1, 2)) if modes > 1 else jnp.zeros((1, 2))
        return LinearGaussianObservationMNIW.from_head_noise(head, jnp.eye(1), strength=strength)

    def links(decl):
        system, params = _lg_system(decl, 1)
        if modes > 1:  # a categorical mode parent beside the regressor
            system = System(
                A_deps=((1, 0),),
                B_deps=((0,), (1,)),
                controlled=(False, False),
                policy_len=1,
                modality_names=("y",),
                factor_names=("x", "m"),
            )
            params = dataclasses.replace(
                params,
                transition={
                    **params.transition,
                    "m": H.ConditionalCategorical(
                        log_table=jnp.log(jnp.full((modes, modes, 1), 1.0 / modes)),
                        parent_ndims=2,
                    ),
                },
                initial_state_prior={
                    **params.initial_state_prior,
                    "m": H.Categorical(logits=jnp.zeros(modes)),
                },
            )
        return description_length(system, 100.0, params=params).per_address["likelihood:y"]

    vague, sharp = links(declared(1.0)), links(declared(20.0))
    d = declared(1.0)
    want = sum(
        kl_matrix_normal_inverse_wishart(m, v_inv, psi, float(nu), *reference_head(1, 2, 1.0))
        for m, v_inv, psi, nu in zip(
            np.asarray(d.M).reshape(-1, 1, 2),
            np.asarray(d.V_inv).reshape(-1, 2, 2),
            np.asarray(d.Psi).reshape(-1, 1, 1),
            np.asarray(d.nu).reshape(-1),
            strict=True,
        )
    )
    assert vague.prior == pytest.approx(want)
    assert sharp.prior > vague.prior
    assert not vague.exact and "unscored" in vague.note
    assert vague.family == pytest.approx(modes * resolution_term(2 + 1, 100.0))


def test_an_emission_over_categorical_parents_only_has_its_room_in_closed_form():
    """With the regressor block empty the family is a Gaussian over its categorical parents,
    and the room the reference admits is ``gaussian_extent`` of the reference, exact."""
    from hybrid_aif.factors import LinearGaussianObservationMNIW

    decl = LinearGaussianObservationMNIW.from_head_noise(jnp.zeros((1, 1)), jnp.eye(1), 2.0)
    system, params = _lg_system(decl, 0)
    link = description_length(system, 100.0, params=params).per_address["likelihood:y"]
    psi0, nu0 = reference_noise(1, 1.0)
    assert link.family == pytest.approx(
        resolution_term(1 + 1, 100.0) + gaussian_extent(1.0, psi0, nu0)
    )
    assert link.exact and "closed form" in link.note


def test_a_point_head_with_inverse_wishart_noise_and_an_independent_matrix_normal_head():
    """A point head beside an inverse-Wishart noise prior scores the noise's room and KL; an
    independent matrix-normal head leaves its own prior unscored and still scores the noise."""
    from hybrid_aif.distributions import InverseWishart, MatrixNormal

    noise = InverseWishart(Psi=jnp.asarray([[0.5]]), nu=jnp.asarray(4.0))
    point = H.LinearGaussianObservation(head=jnp.asarray([[1.0, 0.0]]), noise=noise)
    system, params = _lg_system(point, 1)
    link = description_length(system, 100.0, params=params).per_address["likelihood:y"]
    psi0, nu0 = reference_noise(1, 1.0)
    assert link.family == pytest.approx(
        resolution_term(2 + 1, 100.0) + inverse_wishart_fisher_rao_entropy(psi0, nu0)
    )
    assert link.prior == pytest.approx(kl_inverse_wishart(np.asarray([[0.5]]), 4.0, psi0, nu0))
    assert not link.exact and "point head" in link.note

    independent = H.LinearGaussianObservation(
        head=MatrixNormal.from_head(jnp.asarray([[1.0, 0.0]]), strength=2.0), noise=noise
    )
    system, params = _lg_system(independent, 1)
    link = description_length(system, 100.0, params=params).per_address["likelihood:y"]
    assert link.prior == pytest.approx(kl_inverse_wishart(np.asarray([[0.5]]), 4.0, psi0, nu0))
    assert not link.exact and "matrix-normal head" in link.note


# ================================================================== invariances


def _through_json(document):
    return json.loads(json.dumps(document))


def _renamed(document):
    """Every variable renamed (its leaf, under any composite prefix), every reference with it."""
    names = list(document["variables"])

    def ren(name):
        if name not in names:
            return name
        prefix, dot, leaf = name.rpartition(".")
        # the amortized path names a control node and its action node after their latent
        for tag in ("u_", "a_"):
            if leaf.startswith(tag) and f"{prefix}{dot}{leaf[2:]}" in names:
                return f"{prefix}{dot}{tag}v_{leaf[2:]}"
        return f"{prefix}{dot}v_{leaf}"

    def parent(spec):
        name, bracket, rest = spec.partition("[")
        return ren(name) + bracket + rest

    def params_of(value):
        if isinstance(value, str):
            return ren(value)
        if isinstance(value, dict):
            return {k: params_of(v) for k, v in value.items()}
        if isinstance(value, list):
            return [params_of(v) for v in value]
        return value

    out = copy.deepcopy(document)
    out["variables"] = {ren(k): v for k, v in document["variables"].items()}
    out["factors"] = {
        ren(child): {
            label: {
                **record,
                **(
                    {"parents": [parent(p) for p in record["parents"]]}
                    if "parents" in record
                    else {}
                ),
                **(
                    {"parameters": params_of(record["parameters"])}
                    if "parameters" in record
                    else {}
                ),
            }
            for label, record in records.items()
        }
        for child, records in document.get("factors", {}).items()
    }
    relations = document.get("relations", {})
    out["relations"] = {
        key: (
            {ren(u): ren(a) for u, a in value.items()}
            if isinstance(value, dict)
            else [ren(v) for v in value]
        )
        for key, value in relations.items()
    }
    out["clamps"] = {ren(k): v for k, v in document.get("clamps", {}).items()}
    objective = document.get("objective")
    if objective and "terms" in objective:
        out["objective"] = {
            **objective,
            "terms": [
                {
                    **term,
                    **({"on": ren(term["on"])} if "on" in term else {}),
                    **(
                        {"parameters": params_of(term["parameters"])}
                        if "parameters" in term
                        else {}
                    ),
                }
                for term in objective["terms"]
            ],
        }
    return out


def _reordered(document):
    """Observations reversed among themselves and static variables moved first (the layouts
    that cannot change what a record reads), factor records reversed."""
    observed = set(document.get("relations", {}).get("observed", ()))
    static = set(document.get("clamps", {})) | {
        child for child, records in document.get("factors", {}).items() if "prior" in records
    }
    items = list(document["variables"].items())
    first = [kv for kv in items if kv[0] in static]
    obs = [kv for kv in reversed(items) if kv[0] in observed]
    rest = [kv for kv in items if kv[0] not in static and kv[0] not in observed]
    out = copy.deepcopy(document)
    out["variables"] = dict(first + rest + obs)
    out["factors"] = {
        child: dict(reversed(list(records.items())))
        for child, records in reversed(list(document.get("factors", {}).items()))
    }
    return out


def _as_simplex(document):
    """Every inline ``log_simplex`` clamp written as probabilities under ``simplex``."""
    out = copy.deepcopy(document)
    for name, record in document["variables"].items():
        clamp = document.get("clamps", {}).get(name)
        inline = clamp is not None and not (isinstance(clamp, dict) and "ref" in clamp)
        if record.get("domain") == "log_simplex" and inline:
            out["variables"][name] = {**record, "domain": "simplex"}
            out["clamps"][name] = np.exp(np.asarray(clamp, np.float64)).tolist()
    return out


@pytest.mark.parametrize("name", NAMES)
def test_names_order_placement_and_spelling_cost_nothing(models, name):  # noqa: F811
    system, params = models[name]
    n = 100.0
    document, arrays = describe(system, params)
    document = _through_json(document)
    base = description_length(document, n, arrays=arrays)
    assert base.total == description_length(system, n, params=params).total
    for variant in (_renamed(document), _reordered(document), _as_simplex(document)):
        other = description_length(_through_json(variant), n, arrays=arrays)
        assert other.structure == base.structure
        assert other.parameters == base.parameters
    inline, _ = describe(system, params, arrays="inline")
    filed, filed_arrays = describe(system, params, arrays="file")
    placed = description_length(_through_json(inline), n)
    assert placed.structure == base.structure and placed.parameters == base.parameters
    placed = description_length(_through_json(filed), n, arrays=filed_arrays)
    assert placed.structure == base.structure and placed.parameters == base.parameters


def test_a_document_with_params_is_refused_and_an_unbuildable_one_says_why():
    system, params = _tabular_system()
    document, arrays = describe(system, params)
    with pytest.raises(TypeError, match="takes no params"):
        description_length(document, 10.0, params=params)
    with pytest.raises(H.DescriptionError):
        description_length({"hybrid_aif_description": "0.3"}, 10.0)


def test_every_shipped_declaration_is_priced(models):  # noqa: F811
    """Every shipped model has a length in both forms, with the links the design names: a net
    and a mixture an upper bound, a fixed Gaussian unscored for its extent, a point table
    priced by its family alone."""
    for name in NAMES:
        system, params = models[name]
        dl = description_length(system, 1000.0, params=params)
        assert np.isfinite(dl.total) and dl.structure > 0.0
        for addr, link in dl.per_address.items():
            assert addr == "graph" or addr.startswith("control:") or link.family is not None
    rssm = description_length(models["tiny_rssm"][0], 1000.0, params=models["tiny_rssm"][1])
    assert not rssm.per_address["transition:deter"].exact
    assert "upper bound" in rssm.per_address["transition:deter"].note
    assert rssm.per_address["initial_state_prior:stoch"].family == 0.0
    lg = description_length(models["hill_lg"][0], 1000.0, params=models["hill_lg"][1])
    assert lg.per_address["initial_state_prior:state"].prior is None
    assert ("initial_state_prior:state", "prior") in lg.unscored
    example = _example("noise_learning")
    learner = example.make_params(
        InverseGamma.from_R_chol(jnp.full((2, 1, 1), 0.6), strength=2.0, anchor=True)
    )
    noise = description_length(example.make_system(), 1000.0, params=learner)
    pose = noise.per_address["likelihood:pose"]
    assert pose.prior is not None and "noise prior" in pose.note and "scale 1" in pose.note
    ig = learner.likelihood["pose"].noise
    assert pose.prior == pytest.approx(
        kl_inverse_gamma(np.asarray(ig.alpha), np.asarray(ig.beta), *reference_inverse_gamma(1.0))
    )
    rescaled = description_length(
        example.make_system(), 1000.0, params=learner, reference_scale=0.6
    )
    assert rescaled.per_address["likelihood:pose"].prior == pytest.approx(
        kl_inverse_gamma(np.asarray(ig.alpha), np.asarray(ig.beta), *reference_inverse_gamma(0.6))
    )
    assert rescaled.per_address["likelihood:pose"].prior != pose.prior
    with pytest.raises(ValueError, match="scale"):
        description_length(example.make_system(), 1000.0, params=learner, reference_scale=0.0)


def test_a_gaussian_emission_prices_the_live_levels_of_its_parent_only():
    """A Gaussian emission over a factor with two dead levels takes the family and prior links
    of its three live levels' heads, the same as the factor declared with three levels: a dead
    level's head is room the cap declared, not a member of the family."""
    n = 50.0

    def declared(levels, dead):
        live = levels - dead
        weights = np.zeros(levels)
        weights[:live] = 1.0
        loc = np.zeros((levels, 2))
        loc[:live] = np.arange(2 * live, dtype=float).reshape(live, 2)
        system = System(
            A_deps=((0,),),
            B_deps=((),),
            controlled=(False,),
            policy_len=1,
            modality_names=("y",),
            factor_names=("z",),
            epistemic=False,
        )
        conc = jnp.asarray(weights, jnp.float32)
        params = SystemParams(
            likelihood={
                "y": H.gaussian_emission(
                    jnp.asarray(loc, jnp.float32),
                    jnp.broadcast_to(jnp.eye(2), (levels, 2, 2)),
                    strength=0.5,
                )
            },
            transition={"z": H.ConditionalDirichlet(concentration=conc[:, None], parent_ndims=1)},
            initial_state_prior={"z": H.Dirichlet(concentration=conc)},
        )
        return description_length(system, n, params=params).per_address["likelihood:y"]

    capped, outright = declared(5, 2), declared(3, 0)
    assert capped.family == pytest.approx(outright.family, rel=1e-9)
    assert capped.prior == pytest.approx(outright.prior, rel=1e-6)
    assert capped.family < declared(5, 0).family


def _total_trace(system, params, learn, ticks=6, seed=3):
    rng = np.random.default_rng(seed)
    observations = {"o0": jnp.asarray(rng.integers(0, 4, ticks), jnp.int32)}
    actions = {system.names.actions[0]: jnp.asarray(rng.integers(0, 2, ticks), jnp.int32)}
    return replay(system, params, observations, actions, metrics=False, learn=learn)


def test_a_replay_pays_for_each_parameter_once():
    """With learning off the total is the description length at each address's count along the
    trace (D1) plus the summed free energy. With learning on, an address the replay learned is
    paid for in the free energy and takes no family link; one it did not learn (a point table)
    keeps it."""
    system, params = _tabular_system(learned=("o0",))
    frozen = _total_trace(system, params, learn=False)
    total = description_length(system, replay=frozen, params=params)
    energy = float(np.sum(np.asarray(frozen.free_energy, np.float64)))
    n = {"likelihood:o0": 6.0, "transition:s0": 5.0, "initial_state_prior:s0": 1.0}
    want = description_length(system, n, params=params).total + energy
    assert total.learned == () and {a: total.per_address[a].n for a in n} == n
    assert total.data == pytest.approx(energy) and total.total == pytest.approx(want)

    learning = _total_trace(system, params, learn=True)
    total = description_length(system, replay=learning, params=params)
    energy = float(np.sum(np.asarray(learning.free_energy, np.float64)))
    assert total.learned == ("likelihood:o0",)
    links = total.per_address
    assert links["likelihood:o0"].family is None and links["likelihood:o0"].prior is not None
    full = description_length(system, n, params=params)
    assert links["transition:s0"] == full.per_address["transition:s0"]  # a point table
    want = full.total - full.per_address["likelihood:o0"].family + energy
    assert total.total == pytest.approx(want)
    assert ("likelihood:o0", "family") not in total.unscored
    with pytest.raises(ValueError, match="one or the other"):
        description_length(system, replay=learning, params=params, prior=params)


def test_a_model_with_nothing_to_learn_ignores_the_learning_flag():
    system, params = _tabular_system()
    on = description_length(system, replay=_total_trace(system, params, learn=True), params=params)
    off = description_length(
        system, replay=_total_trace(system, params, learn=False), params=params
    )
    assert on.learned == () and on.total == pytest.approx(off.total)


def test_a_replay_sums_valid_ticks_over_leading_axes():
    """Episodes replayed under ``vmap`` are one trace: their free energies over the valid ticks
    add, and each address's default resolution counts its data over all of them. Learning
    episodes under ``vmap`` are refused: each would pay for the parameters again."""
    system, params = _tabular_system(learned=("o0",))
    rng = np.random.default_rng(4)
    obs = {"o0": jnp.asarray(rng.integers(0, 4, (3, 5)), jnp.int32)}
    acts = {system.names.actions[0]: jnp.asarray(rng.integers(0, 2, (3, 5)), jnp.int32)}
    valid = jnp.asarray(np.arange(5) < 4)

    def episodes(learn):
        return jax.vmap(
            lambda o, a: replay(system, params, o, a, valid=valid, metrics=False, learn=learn)
        )(obs, acts)

    trace = episodes(False)
    total = description_length(system, replay=trace, params=params)
    energy = np.where(np.asarray(trace.valid), np.asarray(trace.free_energy, np.float64), 0.0)
    want = {"likelihood:o0": 12.0, "transition:s0": 9.0, "initial_state_prior:s0": 3.0}
    assert {a: total.per_address[a].n for a in want} == want
    assert total.data == pytest.approx(float(energy.sum()))
    with pytest.raises(ValueError, match="once per episode"):
        description_length(system, replay=episodes(True), params=params)


def test_a_switching_emission_prices_the_live_modes_only():
    """An emission over a continuous state and a mode factor is batched over the mode alone; its
    dead modes take no links, so two dead modes of three price as one mode declared outright."""

    def declared(levels, live):
        w = np.zeros(levels)
        w[:live] = 1.0
        conc = jnp.asarray(w, jnp.float32)
        system = System(
            A_deps=((1, 0),),
            B_deps=((), (0,)),
            controlled=(False, False),
            policy_len=1,
            modality_names=("y",),
            factor_names=("mode", "x"),
            mixture_k=2,
            epistemic=False,
        )
        head = jnp.concatenate(
            [jnp.ones((levels, 1, 1)), jnp.arange(levels, dtype=jnp.float32)[:, None, None]], -1
        )
        params = SystemParams(
            likelihood={
                "y": LinearGaussianObservationMNIW.from_head_noise(head, jnp.eye(1), strength=0.5)
            },
            transition={
                "mode": H.ConditionalDirichlet(concentration=conc[:, None], parent_ndims=1),
                "x": H.LinearGaussianTransition(
                    F=jnp.full((levels, 1, 1), 0.9),
                    innovations=H.ConditionalGaussian(
                        loc=jnp.zeros((levels, 1)),
                        chol=jnp.full((levels, 1, 1), 0.3),
                        parent_ndims=1,
                    ),
                ),
            },
            initial_state_prior={
                "mode": H.Dirichlet(concentration=conc),
                "x": H.MixtureGaussian(
                    log_weights=jnp.log(jnp.asarray([0.5, 0.5])),
                    means=jnp.zeros((2, 1)),
                    chols=jnp.ones((2, 1, 1)),
                ),
            },
        )
        return description_length(system, 50.0, params=params).per_address["likelihood:y"]

    capped, outright = declared(3, 1), declared(1, 1)
    assert capped.family == pytest.approx(outright.family, rel=1e-9)
    assert capped.prior == pytest.approx(outright.prior, rel=1e-9)
    assert capped.family < declared(3, 3).family


def test_a_replay_that_only_casts_a_point_table_learned_nothing():
    """A point table held in float64 comes back from a learning replay in float32: a cast, not
    learning, so it keeps its family link."""
    system, params = _tabular_system()
    table = np.asarray(params.likelihood["o0"].log_table, np.float64)
    params = dataclasses.replace(
        params,
        likelihood={"o0": H.ConditionalCategorical(log_table=table, parent_ndims=1)},
    )
    learning = _total_trace(system, params, learn=True)
    length = description_length(system, replay=learning, params=params)
    assert length.learned == () and length.per_address["likelihood:o0"].family is not None


def test_a_replay_that_learns_only_the_noise_keeps_the_point_heads_family_link():
    """A linear-Gaussian observation with a point head and a learnable noise: the replay pays
    for the noise, and the head's resolution term stays in the model part."""
    emission = H.LinearGaussianObservation(
        H=jnp.asarray([[1.0]]),
        c=jnp.zeros(1),
        noise=InverseGamma.from_R_chol(jnp.full((1, 1), 0.6), strength=2.0),
    )
    system = System(
        A_deps=((0,),),
        B_deps=((),),
        controlled=(False,),
        policy_len=1,
        modality_names=("y",),
        factor_names=("x",),
        epistemic=False,
    )
    params = SystemParams(
        likelihood={"y": emission},
        transition={
            "x": H.LinearGaussianTransition(
                F=jnp.asarray([[1.0]]),
                innovations=H.ConditionalGaussian(
                    loc=jnp.zeros((1, 1)), chol=jnp.asarray([[[0.3]]]), parent_ndims=1
                ),
            )
        },
        initial_state_prior={"x": H.Gaussian(loc=jnp.asarray([0.2]), chol=jnp.asarray([[1.0]]))},
    )
    ticks = 30
    y = jnp.asarray(np.random.default_rng(0).normal(size=(ticks, 1)), jnp.float32)
    learning = replay(system, params, {"y": y}, {}, learn=True, metrics=False)
    length = description_length(system, replay=learning, params=params)
    assert length.learned == ("likelihood:y",)
    assert length.per_address["likelihood:y"].family == pytest.approx(
        resolution_term(2, float(ticks))
    )


def test_a_level_a_replay_wakes_is_priced_from_its_declared_head():
    """A growable factor's dead levels are room until a replay wakes one: then the head it
    started from is part of the model, and sharpening a dead head towards the data costs."""
    k_max = 6

    def model(dead_loc):
        weights = np.zeros(k_max)
        weights[0] = 1.0
        loc = np.zeros((k_max, 2))
        loc[1:] = dead_loc
        system = System(
            A_deps=((0,),),
            B_deps=((),),
            controlled=(False,),
            policy_len=1,
            modality_names=("y",),
            factor_names=("z",),
            epistemic=False,
        )
        conc = jnp.asarray(weights, jnp.float32)
        params = SystemParams(
            likelihood={
                "y": H.gaussian_emission(
                    jnp.asarray(loc, jnp.float32),
                    jnp.broadcast_to(jnp.eye(2), (k_max, 2, 2)),
                    strength=0.1,
                )
            },
            transition={
                "z": H.ConditionalDirichlet(
                    concentration=conc[:, None], parent_ndims=1, growth=jnp.asarray(1.0)
                )
            },
            initial_state_prior={"z": H.Dirichlet(concentration=conc, growth=jnp.asarray(1.0))},
        )
        return system, params

    rng = np.random.default_rng(1)
    centres = np.asarray([[0.0, 0.0], [8.0, 8.0]])
    y = jnp.asarray(centres[rng.integers(0, 2, 40)] + 0.5 * rng.normal(size=(40, 2)), jnp.float32)
    lengths = []
    for dead_loc in ([0.0, 0.0], [8.0, 8.0]):
        system, params = model(dead_loc)
        learning = replay(system, params, {"y": y}, {}, learn=True, metrics=False)
        lengths.append(description_length(system, replay=learning, params=params))
        alone = description_length(system, params=params)
        assert lengths[-1].parameters > alone.parameters  # the woken level's head is priced
    assert lengths[1].parameters > lengths[0].parameters  # a sharper dead head costs more


def test_an_initial_prior_keeps_its_family_link_under_a_learning_replay():
    """The initial prior is read at the first tick only, before anything is learned, so the
    free energy never pays for it: it keeps its family link, at one draw."""
    system, params = _tabular_system(learned=("o0",))
    params = dataclasses.replace(
        params, initial_state_prior={"s0": H.Dirichlet(concentration=jnp.full(3, 2.0))}
    )
    length = description_length(
        system, replay=_total_trace(system, params, learn=True), params=params
    )
    assert "initial_state_prior:s0" not in length.learned
    assert length.per_address["initial_state_prior:s0"].family is not None
    assert length.per_address["initial_state_prior:s0"].n == 1.0


def test_a_lagged_modality_is_counted_from_an_episodes_second_tick():
    """A lagged modality reads its datum from the carry, so an episode's first tick has none:
    its default resolution is the valid ticks less one per episode."""
    from tests.test_structure_learning import _lagged_system

    system, params = _lagged_system()
    ticks = 7
    rng = np.random.default_rng(2)
    observations = {
        "obs_x": jnp.asarray(rng.normal(size=(ticks, 2)), jnp.float32),
        "obs_kind": jnp.asarray(rng.integers(0, 3, ticks), jnp.int32),
    }
    trace = replay(system, params, observations, {}, metrics=False)
    length = description_length(system, replay=trace, params=params)
    assert length.per_address["likelihood:obs_x"].n == float(ticks)
    assert length.per_address["likelihood:prev_x"].n == float(ticks - 1)
    assert length.per_address["likelihood:prev_type"].n == float(ticks - 1)
