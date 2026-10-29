"""Novelty over a matrix-normal inverse-Wishart emission (``docs/design/parameter-learning.md``
D10): the leaf ``I(θ; y)`` with the state marginalised, its closed form at a known state, its
error against nested Monte Carlo in both regimes, its mounting by ``epistemic``, the epistemic
chain it reads, and the refusals."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import digamma, gammaln
from scipy.stats import invwishart

from hybrid_aif.distributions import (
    ConditionalDirichlet,
    ConditionalGaussian,
    Dirichlet,
    Gaussian,
    MixtureGaussian,
)
from hybrid_aif.factors import (
    LinearGaussianObservation,
    LinearGaussianObservationMNIW,
    LinearGaussianTransition,
    gaussian_emission,
)
from hybrid_aif.objectives import continuous, registry
from hybrid_aif.system import System, SystemParams

D, K = 2, 2  # reading and state dimensions; the head's columns are [x; 1]


def _random_model(rng, offset_only: bool, nu=None) -> LinearGaussianObservationMNIW:
    """A posterior with every column of the head uncertain, or with the state columns nearly
    known and only the offset uncertain (the customer's pile head); ``nu`` drawn from
    ``[d + 3, 12]`` unless given."""
    if offset_only:
        v = np.diag([1e-3, 1e-3, rng.uniform(0.5, 4)])
    else:
        a = rng.normal(size=(K + 1, K + 1))
        v = a @ a.T / (K + 1) * rng.uniform(0.2, 3)
    b = rng.normal(size=(D, D))
    return LinearGaussianObservationMNIW(
        M=jnp.asarray(rng.normal(size=(D, K + 1)) * 1.5),
        V_inv=jnp.asarray(np.linalg.inv(v)),
        Psi=jnp.asarray(b @ b.T + 0.5 * np.eye(D)),
        nu=jnp.asarray(rng.uniform(D + 3, 12) if nu is None else nu),
    )


def _leaf(model, q) -> float:
    return float(continuous.param_info_gain(q, model, continuous.mniw_draws(model)))


def _monte_carlo(model, mu, sx, rng, n_out=2000, n_in=20000) -> float:
    """Nested Monte Carlo ``E[log p(y | θ) − log p(y)]``; ``y | θ`` is Gaussian, so only the
    marginal is averaged over draws."""
    m, v = np.asarray(model.M), np.linalg.inv(np.asarray(model.V_inv))
    psi, nu = np.asarray(model.Psi), float(model.nu)
    u = np.append(mu, 1.0)
    su = np.zeros((K + 1, K + 1))
    su[:K, :K] = sx

    def draw(n):
        s = invwishart.rvs(df=nu, scale=psi, size=n, random_state=rng)
        w = m + np.linalg.cholesky(s) @ rng.normal(size=(n, D, K + 1)) @ np.linalg.cholesky(v).T
        return w @ u, w @ su @ np.swapaxes(w, -1, -2) + s

    def log_density(y, mean, cov):
        a, b, c = cov[..., 0, 0], cov[..., 0, 1], cov[..., 1, 1]
        det = a * c - b * b
        r0, r1 = y[..., 0] - mean[..., 0], y[..., 1] - mean[..., 1]
        return (
            -0.5 * (c * r0 * r0 - 2 * b * r0 * r1 + a * r1 * r1) / det
            - 0.5 * np.log(det)
            - np.log(2 * np.pi)
        )

    mean, cov = draw(n_out)
    y = mean + (np.linalg.cholesky(cov) @ rng.normal(size=(n_out, D, 1)))[..., 0]
    inner_mean, inner_cov = draw(n_in)
    marginal = np.concatenate(
        [
            np.logaddexp.reduce(
                log_density(y[s : s + 100, None], inner_mean[None], inner_cov[None]), axis=1
            )
            for s in range(0, n_out, 100)
        ]
    ) - np.log(n_in)
    return float(np.mean(log_density(y, mean, cov) - marginal))


def test_a_known_state_gives_the_closed_form():
    """With the state pinned the leaf is exact: ``(d/2) log(1 + x̃ᵀVx̃)``, the head's share, plus
    the noise's, the Student-t predictive's entropy less the expected Gaussian entropy, ``log det
    Ψ`` cancelling; no draw enters. Checked at a prior's low degrees of freedom too."""
    for nu in (None, D + 2.0, D + 1.2):
        model = _random_model(np.random.default_rng(0), offset_only=False, nu=nu)
        x = jnp.asarray([0.4, -0.7])
        q = Gaussian(loc=x, chol=1e-6 * jnp.eye(K))
        u = np.append(np.asarray(x), 1.0)
        quad = u @ np.linalg.inv(np.asarray(model.V_inv)) @ u
        nu = float(model.nu)
        df = nu - D + 1
        student = (
            -0.5 * D * np.log(df)
            + gammaln(df / 2)
            + 0.5 * D * np.log(df * np.pi)
            - gammaln((df + D) / 2)
            + 0.5 * (df + D) * (digamma((df + D) / 2) - digamma(df / 2))
        )
        expected_log_det = -D * np.log(2) - sum(digamma((nu - i) / 2) for i in range(D))
        noise_share = student - 0.5 * (D * np.log(2 * np.pi * np.e) + expected_log_det)
        assert _leaf(model, q) == pytest.approx(D / 2 * np.log1p(quad) + noise_share, abs=1e-4), nu


@pytest.mark.parametrize(
    ("offset_only", "nu", "tolerance"),
    [(False, None, 0.2), (True, None, 0.1), (False, D + 2.0, 0.2), (True, D + 1.5, 0.2)],
    ids=["general", "offset", "general-prior", "offset-near-floor"],
)
def test_the_estimate_tracks_monte_carlo(offset_only, nu, tolerance):
    """Against nested Monte Carlo over pose spreads 0.3 to 3, at moderate degrees of freedom and
    at a prior's low ones (``strength=1`` gives ``ν = d + 2``): within the tolerance D10
    records, the Monte Carlo's own error about 0.03. In the offset regime the leaf falls as the
    belief widens."""
    rng = np.random.default_rng(7)
    for _ in range(2):
        model = _random_model(rng, offset_only, nu)
        mu = rng.normal(size=K)
        leaves = []
        for spread in (0.3, 1.0, 3.0):
            q = Gaussian(loc=jnp.asarray(mu), chol=spread * jnp.eye(K))
            leaf = _leaf(model, q)
            truth = _monte_carlo(model, mu, spread**2 * np.eye(K), rng)
            assert leaf == pytest.approx(truth, abs=tolerance), (spread, leaf, truth)
            leaves.append(leaf)
        if offset_only:
            assert leaves[0] > leaves[1] > leaves[2]


def test_a_mixture_belief_enters_through_its_moments():
    model = _random_model(np.random.default_rng(1), offset_only=False)
    mixture = MixtureGaussian(
        log_weights=jnp.log(jnp.asarray([0.3, 0.7])),
        means=jnp.asarray([[0.5, 0.0], [-0.4, 0.8]]),
        chols=jnp.stack([0.2 * jnp.eye(K), 0.5 * jnp.eye(K)]),
    )
    assert _leaf(model, mixture) == pytest.approx(_leaf(model, mixture.moment_match()), abs=1e-5)


def test_the_noise_mean_must_exist():
    model = _random_model(np.random.default_rng(2), offset_only=False)
    with pytest.raises(ValueError, match=r"nu > d \+ 1"):
        continuous.mniw_draws(dataclasses.replace(model, nu=jnp.asarray(2.5)))


# ------------------------------------------------------------------- mounted by epistemic


def _system(**overrides):
    """A pose observed by a GPS and by a pile head whose offset is being learned."""
    defaults = dict(
        A_deps=((0,), (0,)),
        B_deps=((),),
        controlled=(True,),
        policy_len=2,
        modality_names=("gps", "pile"),
        factor_names=("pose",),
        planning_belief="posterior",
    )
    return System(**{**defaults, **overrides})


def _params():
    steps = jnp.asarray([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]])
    pile = LinearGaussianObservationMNIW(
        M=jnp.asarray([[1.0, 0.0, 0.2], [0.0, 1.0, -0.1]]),
        V_inv=jnp.diag(jnp.asarray([1e3, 1e3, 0.5])),
        Psi=0.04 * 6.0 * jnp.eye(D),
        nu=jnp.asarray(6.0),
    )
    return SystemParams(
        likelihood={
            "gps": LinearGaussianObservation(H=jnp.eye(K), c=jnp.zeros(K), noise=0.05 * jnp.eye(K)),
            "pile": pile,
        },
        transition={
            "pose": LinearGaussianTransition(
                F=jnp.eye(K),
                innovations=ConditionalGaussian(
                    loc=steps, chol=jnp.broadcast_to(0.3 * jnp.eye(K), (3, K, K)), parent_ndims=1
                ),
            )
        },
        initial_state_prior={"pose": Gaussian(loc=jnp.zeros(K), chol=jnp.eye(K))},
    )


def _trees(system, params, present=None):
    belief = Gaussian(loc=jnp.zeros(K), chol=jnp.eye(K))
    ctx = system.planner_context(params, {"pose": belief}, present=present)
    carry, trees = ctx.init_beliefs, []
    for t in range(system.policy_len):
        action = jnp.asarray([1], jnp.int32)
        carry = ctx.transition_fn(carry, action)
        trees.append(ctx.step_terms_fn(carry, action, jnp.asarray(t, jnp.int32)))
    return trees


@pytest.mark.parametrize("epistemic", [True, ("likelihood:pile",)])
def test_epistemic_mounts_the_leaf_as_an_estimate(epistemic):
    trees = _trees(_system(epistemic=epistemic), _params())
    for tree in trees:
        leaf = tree["infogain"]["likelihood:pile"]
        assert not leaf.derived
        assert float(leaf.lower) == float(leaf.upper) > 0.0


def test_the_leaf_reads_the_epistemic_chain():
    """A GPS reading expected at the first step pins the pose the second step's pile reading
    is read at, so that reading says more about the head; with the GPS absent it says less.
    An absent pile modality zeroes its own leaf."""
    system = _system(epistemic=("likelihood:pile",))
    with_gps = _trees(system, _params())
    without_gps = _trees(system, _params(), present={"gps": jnp.asarray([False, True])})
    gain = float(with_gps[1]["infogain"]["likelihood:pile"].lower)
    assert gain > float(without_gps[1]["infogain"]["likelihood:pile"].lower) + 0.05
    no_pile = _trees(system, _params(), present={"pile": False})
    assert all(float(t["infogain"]["likelihood:pile"].lower) == 0.0 for t in no_pile)


def test_planning_scores_the_leaf_under_jit():
    system = _system(epistemic=True)
    scored = jax.jit(lambda p: _trees(system, p)[1]["infogain"]["likelihood:pile"].lower)
    assert float(scored(_params())) == pytest.approx(
        float(_trees(system, _params())[1]["infogain"]["likelihood:pile"].lower), abs=1e-5
    )


def test_an_emission_with_discrete_parents_is_refused():
    """A Gaussian emission over a categorical parent is an MNIW with a mode axis: its novelty
    has no estimator, named or under ``True``."""
    k = 3
    system = System(
        A_deps=((0,),),
        B_deps=((),),
        controlled=(False,),
        policy_len=1,
        modality_names=("y",),
        factor_names=("z",),
    )
    params = SystemParams(
        likelihood={
            "y": gaussian_emission(
                jnp.zeros((k, D)), jnp.broadcast_to(jnp.eye(D), (k, D, D)), strength=1.0
            )
        },
        transition={"z": ConditionalDirichlet(concentration=jnp.ones((k, k)), parent_ndims=1)},
        initial_state_prior={"z": Dirichlet(concentration=jnp.ones(k))},
    )
    for epistemic in (True, ("likelihood:y",)):
        with pytest.raises(NotImplementedError, match="discrete parents"):
            dataclasses.replace(system, epistemic=epistemic)._term_targets(params)
    # a mode axis on the noise alone, beside a shared head, is a discrete parent too
    shared_head = _params()
    pile = shared_head.likelihood["pile"]
    per_mode = dataclasses.replace(
        pile, Psi=jnp.broadcast_to(pile.Psi, (3, D, D)), nu=jnp.full((3,), 6.0)
    )
    switching = System(
        A_deps=((0,), (0, 1)),
        B_deps=((), ()),
        controlled=(True, False),
        policy_len=1,
        modality_names=("gps", "pile"),
        factor_names=("pose", "z"),
        epistemic=("likelihood:pile",),
    )
    switching_params = dataclasses.replace(
        shared_head,
        likelihood={"gps": shared_head.likelihood["gps"], "pile": per_mode},
        transition={
            **shared_head.transition,
            "z": ConditionalDirichlet(concentration=jnp.ones((3, 3)), parent_ndims=1),
        },
        initial_state_prior={
            **shared_head.initial_state_prior,
            "z": Dirichlet(concentration=jnp.ones(3)),
        },
    )
    with pytest.raises(NotImplementedError, match="discrete parents"):
        switching._term_targets(switching_params)
    assert registry.get(registry.param_kind("likelihood"), registry.MNIW).name == "blended_entropy"
