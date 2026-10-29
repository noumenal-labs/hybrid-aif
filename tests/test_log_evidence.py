"""Closed-form evidence per conjugate family (``hybrid_aif.learning.evidence``).

Each family's number is pinned against something that does not share its algebra: the
Dirichlet against the Pólya chain rule and against the sum over every sequence; the
inverse-Gamma against quadrature; the inverse-Wishart and the matrix-normal inverse-Wishart
against the chain of Student-t predictives, and the scalar MNIW case against a double
quadrature over the head and the noise.
"""

import itertools

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, stats

from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalDirichlet,
    Dirichlet,
    InverseGamma,
    InverseWishart,
)
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianObservationMNIW
from hybrid_aif.learning import Evidence, absorbed_count, log_evidence
from hybrid_aif.system import System, SystemParams

# ------------------------------------------------------------------------------ Dirichlet


def test_dirichlet_evidence_is_the_polya_chain_rule():
    prior = Dirichlet(concentration=jnp.asarray([0.5, 1.5, 2.0]))
    post, chain = prior, 0.0
    for x in (0, 2, 2, 1, 0, 2):
        chain += float(jnp.log(post.mean()[x]))
        post = post.absorb(post.stats(jnp.asarray(x)))
    np.testing.assert_allclose(float(log_evidence(prior, post)), chain, rtol=1e-6)
    assert float(absorbed_count(prior, post)) == 6.0


def test_dirichlet_evidence_sums_to_one_over_every_sequence():
    prior = Dirichlet(concentration=jnp.asarray([0.7, 1.0, 2.5]))
    total = 0.0
    for seq in itertools.product(range(3), repeat=3):
        counts = jnp.asarray(np.bincount(seq, minlength=3), dtype=jnp.float32)
        total += float(jnp.exp(log_evidence(prior, prior.absorb(counts))))
    np.testing.assert_allclose(total, 1.0, rtol=1e-5)


def test_conditional_dirichlet_rows_are_independent_families():
    conc = jnp.asarray([[1.0, 2.0, 0.5], [3.0, 1.0, 1.0]])  # (levels=2, parents=3)
    prior = ConditionalDirichlet(concentration=conc, parent_ndims=1)
    counts = jnp.asarray([[2.0, 0.0, 1.0], [1.0, 0.0, 4.0]])
    post = ConditionalDirichlet(concentration=conc + counts, parent_ndims=1)
    rows = log_evidence(prior, post)
    assert rows.shape == (3,)
    for j in range(3):
        one = Dirichlet(concentration=conc[:, j])
        np.testing.assert_allclose(
            float(rows[j]), float(log_evidence(one, one.absorb(counts[:, j]))), rtol=1e-6
        )
    np.testing.assert_allclose(np.asarray(absorbed_count(prior, post)), [3.0, 0.0, 5.0])


# ---------------------------------------------------------------------- inverse-Gamma


def test_inverse_gamma_evidence_matches_quadrature():
    alpha, beta = 2.5, 1.2
    x = np.asarray([0.3, -1.1, 0.8, 0.2])
    prior = InverseGamma(alpha=jnp.asarray([alpha]), beta=jnp.asarray([beta]))
    post = prior.absorb(prior.stats(jnp.asarray([float(np.sum(x**2))]), jnp.asarray(4.0)))

    def integrand(s):
        return np.prod(stats.norm.pdf(x, 0.0, np.sqrt(s))) * stats.invgamma.pdf(
            s, alpha, scale=beta
        )

    want, _ = integrate.quad(integrand, 0.0, np.inf)
    np.testing.assert_allclose(float(log_evidence(prior, post)), np.log(want), rtol=1e-6)
    assert float(absorbed_count(prior, post)) == pytest.approx(4.0)


def test_inverse_gamma_dimensions_add():
    prior = InverseGamma(alpha=jnp.asarray([2.0, 3.0]), beta=jnp.asarray([1.0, 0.5]))
    post = prior.absorb(prior.stats(jnp.asarray([0.4, 2.0]), jnp.asarray(3.0)))
    per_dim = [
        float(
            log_evidence(
                InverseGamma(alpha=prior.alpha[i : i + 1], beta=prior.beta[i : i + 1]),
                InverseGamma(alpha=post.alpha[i : i + 1], beta=post.beta[i : i + 1]),
            )
        )
        for i in range(2)
    ]
    np.testing.assert_allclose(float(log_evidence(prior, post)), sum(per_dim), rtol=1e-6)


# -------------------------------------------------------------------- inverse-Wishart


def _iw_predictive_logpdf(x, psi, nu):
    d = psi.shape[-1]
    df = nu - d + 1.0
    return stats.multivariate_t(loc=np.zeros(d), shape=psi / df, df=df).logpdf(x)


def test_inverse_wishart_evidence_is_the_student_t_chain_rule():
    rng = np.random.default_rng(0)
    psi0 = np.asarray([[1.5, 0.3], [0.3, 0.8]])
    nu0 = 4.5
    prior = InverseWishart(Psi=jnp.asarray(psi0), nu=jnp.asarray(nu0))
    post, chain = prior, 0.0
    for _ in range(5):
        x = rng.normal(size=2)
        chain += _iw_predictive_logpdf(x, np.asarray(post.Psi), float(post.nu))
        post = post.absorb(post.stats(jnp.asarray(np.outer(x, x)), jnp.asarray(1.0)))
    np.testing.assert_allclose(float(log_evidence(prior, post)), chain, rtol=1e-5)
    assert float(absorbed_count(prior, post)) == pytest.approx(5.0)


def test_scalar_inverse_wishart_is_the_inverse_gamma():
    psi, nu, scatter, n = 1.3, 3.0, 2.2, 4.0
    iw = InverseWishart(Psi=jnp.asarray([[psi]]), nu=jnp.asarray(nu))
    iw_post = iw.absorb(iw.stats(jnp.asarray([[scatter]]), jnp.asarray(n)))
    ig = InverseGamma(alpha=jnp.asarray([nu / 2.0]), beta=jnp.asarray([psi / 2.0]))
    ig_post = ig.absorb(ig.stats(jnp.asarray([scatter]), jnp.asarray(n)))
    np.testing.assert_allclose(
        float(log_evidence(iw, iw_post)), float(log_evidence(ig, ig_post)), rtol=1e-6
    )


def test_point_head_with_learnable_noise_scores_the_noise():
    prior = LinearGaussianObservation(
        head=jnp.asarray([[1.0, 0.0]]),
        noise=InverseGamma(alpha=jnp.asarray([2.0]), beta=jnp.asarray([1.0])),
    )
    post_noise = prior.noise.absorb(prior.noise.stats(jnp.asarray([0.5]), jnp.asarray(2.0)))
    post = LinearGaussianObservation(head=prior.head, noise=post_noise)
    np.testing.assert_allclose(
        float(log_evidence(prior, post)), float(log_evidence(prior.noise, post_noise))
    )
    assert float(absorbed_count(prior, post)) == pytest.approx(2.0)


# ---------------------------------------------------------- matrix-normal inverse-Wishart


def _mniw_predictive_logpdf(y, x_aug, m, v_inv, psi, nu):
    d = psi.shape[-1]
    df = nu - d + 1.0
    v = np.linalg.inv(v_inv)
    scale = psi * (1.0 + x_aug @ v @ x_aug) / df
    return stats.multivariate_t(loc=m @ x_aug, shape=scale, df=df).logpdf(y)


def _absorb_pair(post, y, x_aug):
    return post.absorb(
        post.stats(
            jnp.asarray(np.outer(y, y)),
            jnp.asarray(np.outer(y, x_aug)),
            jnp.asarray(np.outer(x_aug, x_aug)),
            jnp.asarray(1.0),
        )
    )


def test_mniw_evidence_is_the_student_t_chain_rule():
    rng = np.random.default_rng(1)
    head = jnp.asarray([[1.0, 0.5], [-0.3, 0.2]])  # (d_obs=2, p=2): one regressor + offset
    r_chol = jnp.asarray([[0.7, 0.0], [0.2, 0.5]])
    prior = LinearGaussianObservationMNIW.from_head_noise(head, r_chol, strength=2.0)
    post, chain = prior, 0.0
    for _ in range(6):
        x_aug = np.asarray([rng.normal(), 1.0])
        y = rng.normal(size=2)
        step = _mniw_predictive_logpdf(
            y,
            x_aug,
            np.asarray(post.M),
            np.asarray(post.V_inv),
            np.asarray(post.Psi),
            float(post.nu),
        )
        own = post.predictive_log_prob(jnp.asarray(y), jnp.asarray(x_aug))
        np.testing.assert_allclose(float(own), step, rtol=1e-5)
        chain += step
        post = _absorb_pair(post, y, x_aug)
    np.testing.assert_allclose(float(log_evidence(prior, post)), chain, rtol=1e-5)
    assert float(absorbed_count(prior, post)) == pytest.approx(6.0)


def test_scalar_mniw_evidence_matches_double_quadrature():
    m, v, psi, nu = 0.4, 0.8, 1.5, 3.0
    xs = np.asarray([0.5, -1.0, 1.5])
    ys = np.asarray([0.3, -0.2, 0.9])
    prior = LinearGaussianObservationMNIW(
        M=jnp.asarray([[m]]),
        V_inv=jnp.asarray([[1.0 / v]]),
        Psi=jnp.asarray([[psi]]),
        nu=jnp.asarray(nu),
    )
    post = prior
    for x, y in zip(xs, ys, strict=True):
        post = _absorb_pair(post, np.asarray([y]), np.asarray([x]))

    def integrand(w, s):
        lik = np.prod(stats.norm.pdf(ys, w * xs, np.sqrt(s)))
        return (
            lik
            * stats.norm.pdf(w, m, np.sqrt(s * v))
            * stats.invgamma.pdf(s, nu / 2.0, scale=psi / 2.0)
        )

    want, err = integrate.dblquad(integrand, 1e-4, 40.0, -12.0, 12.0)
    assert err < 1e-6 * want
    np.testing.assert_allclose(float(log_evidence(prior, post)), np.log(want), rtol=1e-4)


# ---------------------------------------------------------------------- the system face


def test_evidence_by_slot_keys_and_unscored_slots():
    system = System(
        A_deps=((0,),),
        B_deps=((0,),),
        controlled=(False,),
        policy_len=1,
        modality_names=("sense",),
        factor_names=("s",),
    )
    like = ConditionalDirichlet(concentration=jnp.full((2, 2), 1.0), parent_ndims=1)
    trans = ConditionalCategorical(log_table=jnp.log(jnp.full((2, 2, 1), 0.5)), parent_ndims=2)
    d0 = Dirichlet(concentration=jnp.ones(2))
    prior = SystemParams(
        likelihood={"sense": like}, transition={"s": trans}, initial_state_prior={"s": d0}
    )
    counts = jnp.asarray([[3.0, 0.0], [1.0, 2.0]])
    posterior = SystemParams(
        likelihood={
            "sense": ConditionalDirichlet(concentration=like.concentration + counts, parent_ndims=1)
        },
        transition={"s": trans},
        initial_state_prior={"s": d0},
    )
    out = system.log_evidence(prior, posterior)
    assert set(out) == {"likelihood:sense", "transition:s", "initial_state_prior:s"}
    sense = out["likelihood:sense"]
    assert sense.exact and float(sense.count) == 6.0
    np.testing.assert_allclose(
        float(sense.nats),
        float(jnp.sum(log_evidence(like, posterior.likelihood["sense"]))),
        rtol=1e-6,
    )
    assert out["transition:s"] == Evidence(None, False, None)
    untouched = out["initial_state_prior:s"]
    assert untouched.nats is None and untouched.exact and float(untouched.count) == 0.0


def test_decay_toward_the_prior_keeps_the_slot_scored():
    """Anchored at the prior, decay only reweights the absorbed statistics, so the count is the
    mass retained and the closed form scores it; the same decay toward a uniform anchor can let
    the prior's own mass go and the count turns negative."""
    table = jnp.asarray([[0.7, 0.2], [0.3, 0.8]])
    prior = ConditionalDirichlet.from_table(table, parent_ndims=1, strength=6.0, anchor="prior")
    uniform = ConditionalDirichlet.from_table(table, parent_ndims=1, strength=6.0, anchor="uniform")
    stats = jnp.asarray([[1.0, 0.0], [0.0, 0.0]])
    post, post_u = prior, uniform
    for _ in range(5):
        post = post.absorb(stats, decay=0.5)
        post_u = post_u.absorb(stats, decay=0.5)
    count = absorbed_count(prior, post)
    assert float(count[0]) > 0.0 and float(count[1]) == pytest.approx(0.0, abs=1e-6)
    assert np.isfinite(float(jnp.sum(log_evidence(prior, post))))
    assert float(jnp.sum(absorbed_count(uniform, post_u))) < 0.0


def test_mass_let_go_under_decay_is_unscored():
    psi = jnp.asarray([[1.0]])
    prior = InverseWishart(Psi=psi, nu=jnp.asarray(5.0), Psi0=psi, nu0=jnp.asarray(3.0))
    post = prior.absorb(prior.stats(jnp.zeros((1, 1)), jnp.asarray(0.0)), decay=0.5)
    system = System(
        A_deps=((0,),),
        B_deps=((0,),),
        controlled=(False,),
        policy_len=1,
        modality_names=("y",),
        factor_names=("x",),
    )
    assert float(absorbed_count(prior, post)) < 0.0
    out = system.log_evidence(
        SystemParams(
            likelihood={"y": LinearGaussianObservation(head=jnp.ones((1, 2)), noise=prior)},
            transition={
                "x": ConditionalCategorical(log_table=jnp.zeros((2, 2, 1)), parent_ndims=2)
            },
            initial_state_prior={"x": Categorical.from_probs(jnp.ones(2) / 2)},
        ),
        SystemParams(
            likelihood={"y": LinearGaussianObservation(head=jnp.ones((1, 2)), noise=post)},
            transition={
                "x": ConditionalCategorical(log_table=jnp.zeros((2, 2, 1)), parent_ndims=2)
            },
            initial_state_prior={"x": Categorical.from_probs(jnp.ones(2) / 2)},
        ),
    )
    assert out["likelihood:y"].nats is None and out["likelihood:y"].exact


def test_positional_params_are_refused_by_name():
    system = System(
        A_deps=((0,),),
        B_deps=((0,),),
        controlled=(False,),
        policy_len=1,
        modality_names=("y",),
        factor_names=("x",),
    )
    d0 = Dirichlet(concentration=jnp.ones(2))
    table = ConditionalCategorical(log_table=jnp.zeros((2, 2, 1)), parent_ndims=2)
    positional = SystemParams(likelihood=(), transition=(table,), initial_state_prior=(d0,))
    with pytest.raises(TypeError, match="from_positional"):
        system.log_evidence(positional, positional)


def test_mismatched_families_are_refused():
    with pytest.raises(TypeError, match="one family"):
        log_evidence(Dirichlet(concentration=jnp.ones(2)), Categorical.from_probs(jnp.ones(2) / 2))
