"""The truncated Gaussian, against numerical integration per dimension.

Every closed form (log-normaliser, log-density, mean, variance, entropy, KL on a shared box) is
checked against a dense quadrature in float64 of the same density, one dimension at a time: a box
around ``loc``, a one-sided box, and boxes far in either tail.

Design: docs/design/control-states.md C4 (Bounded Gaussian).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import TruncatedGaussian

KEY = jax.random.PRNGKey(0)

#: Per-dimension cases ``(loc, scale, low, high)``: inside, straddling an edge, a narrow box,
#: and boxes 14-20 standard deviations into the lower and upper tails.
CASES = np.asarray(
    [
        (0.0, 1.0, -1.0, 2.0),
        (1.0, 2.0, 5.0, 9.0),
        (0.3, 0.7, -0.2, 0.1),
        (-2.0, 0.5, -10.0, -9.0),
        (0.0, 1.0, 15.0, 20.0),
    ]
).T


def _tg(cases=CASES) -> TruncatedGaussian:
    loc, scale, low, high = (jnp.asarray(c, jnp.float32) for c in cases)
    return TruncatedGaussian(loc=loc, scale=scale, low=low, high=high)


def _quadrature(loc, scale, low, high, n=200_001):
    """``(x, log density, log Z)`` of one dimension on a dense grid, in float64 log space."""
    x = np.linspace(low, high, n)
    log_kernel = -0.5 * ((x - loc) / scale) ** 2 - 0.5 * np.log(2 * np.pi) - np.log(scale)
    top = log_kernel.max()
    log_z = top + np.log(np.trapezoid(np.exp(log_kernel - top), x))
    return x, log_kernel - log_z, log_z


def _moments(x, log_p):
    p = np.exp(log_p)
    mean = np.trapezoid(x * p, x)
    var = np.trapezoid((x - mean) ** 2 * p, x)
    entropy = -np.trapezoid(p * log_p, x)
    return mean, var, entropy


def test_the_closed_forms_match_quadrature_per_dimension_including_far_tails():
    tg = _tg()
    log_z, mean, var = tg.log_normalisers(), tg.mean(), tg.variance()
    entropies = []
    for d, (loc, scale, low, high) in enumerate(CASES.T):
        x, log_p, want_log_z = _quadrature(loc, scale, low, high)
        want_mean, want_var, want_entropy = _moments(x, log_p)
        np.testing.assert_allclose(float(log_z[d]), want_log_z, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(float(mean[d]), want_mean, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(float(var[d]), want_var, rtol=2e-3)
        entropies.append(want_entropy)
        one = TruncatedGaussian(
            loc=jnp.asarray([loc]), scale=jnp.asarray([scale]), low=[low], high=[high]
        )
        np.testing.assert_allclose(float(one.entropy().lower), want_entropy, atol=2e-4)
    #  the event is the product of the dimensions: the entropy sums
    np.testing.assert_allclose(float(tg.entropy().lower), sum(entropies), atol=1e-3)
    assert tg.entropy().lower == tg.entropy().upper  # exact, zero-width


def test_the_log_density_is_the_quadratures_and_minus_infinity_outside():
    tg = _tg()
    inside = jnp.asarray([0.5, 6.0, 0.0, -9.5, 16.0])
    want = 0.0
    for d, (loc, scale, low, high) in enumerate(CASES.T):
        _, _, log_z = _quadrature(loc, scale, low, high, n=20_001)
        z = (float(inside[d]) - loc) / scale
        want += -0.5 * z * z - 0.5 * np.log(2 * np.pi) - np.log(scale) - log_z
    np.testing.assert_allclose(float(tg.log_prob(inside)), want, rtol=1e-5)
    outside = inside.at[3].set(-8.0)
    assert float(tg.log_prob(outside)) == -np.inf
    #  batched: one value per batch member
    batched = tg.log_prob(jnp.stack([inside, outside]))
    assert batched.shape == (2,) and np.isfinite(batched[0]) and batched[1] == -np.inf


def test_kl_on_a_shared_box_matches_quadrature_and_vanishes_on_itself():
    box_low, box_high = jnp.asarray([-1.0, 4.0]), jnp.asarray([2.0, 7.0])
    p = TruncatedGaussian(loc=jnp.asarray([0.0, 1.0]), scale=[1.0, 2.0], low=box_low, high=box_high)
    q = TruncatedGaussian(loc=jnp.asarray([0.5, 9.0]), scale=[0.6, 1.5], low=box_low, high=box_high)
    want = 0.0
    for d in range(2):
        x, log_p, _ = _quadrature(
            float(p.loc[d]), float(p.scale[d]), float(box_low[d]), float(box_high[d])
        )
        _, log_q, _ = _quadrature(
            float(q.loc[d]), float(q.scale[d]), float(box_low[d]), float(box_high[d])
        )
        want += np.trapezoid(np.exp(log_p) * (log_p - log_q), x)
    kl = p.kl_to(q)
    np.testing.assert_allclose(float(kl.lower), want, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(float(p.kl_to(p).lower), 0.0, atol=1e-5)


def test_kl_between_different_boxes_is_refused_with_a_sentence():
    p = TruncatedGaussian(loc=jnp.zeros(1), scale=1.0, low=0.0, high=1.0)
    q = TruncatedGaussian(loc=jnp.zeros(1), scale=1.0, low=0.0, high=2.0)
    with pytest.raises(ValueError, match="different boxes is not finite: pass two members"):
        p.kl_to(q)
    #  and under a trace, when the computation runs
    kl = jax.jit(lambda a, b: a.kl_to(b).lower)
    with pytest.raises(Exception, match="different boxes is not finite"):
        jax.block_until_ready(kl(p, q))
    np.testing.assert_allclose(float(kl(p, p)), 0.0, atol=1e-5)


def test_sampling_is_reparameterised_inside_the_box_with_the_closed_moments():
    tg = _tg()
    draws = tg.sample(KEY, (100_000,))
    assert draws.shape == (100_000, 5)
    assert bool(jnp.all((draws >= tg.low) & (draws <= tg.high)))
    sd = np.sqrt(np.asarray(tg.variance()) / draws.shape[0])
    assert np.all(np.abs(np.asarray(draws.mean(0)) - np.asarray(tg.mean())) < 6 * sd)
    np.testing.assert_allclose(np.asarray(draws.var(0)), np.asarray(tg.variance()), rtol=0.05)

    #  the draw is a differentiable function of loc and scale (the inverse-CDF path)
    def first(loc):
        return TruncatedGaussian(loc, tg.scale, tg.low, tg.high).sample(KEY).sum()

    grad = jax.grad(first)(tg.loc)
    assert bool(jnp.all(jnp.isfinite(grad))) and float(jnp.abs(grad).sum()) > 0.0
    #  and so are the closed moments: d mean / d loc is the variance over scale²
    d_mean = jax.grad(lambda loc: TruncatedGaussian(loc, tg.scale, tg.low, tg.high).mean().sum())
    np.testing.assert_allclose(
        np.asarray(d_mean(tg.loc)), np.asarray(tg.variance() / tg.scale**2), rtol=2e-3, atol=1e-7
    )


def test_the_contract():
    one = TruncatedGaussian(
        loc=jnp.asarray([0.0, 1.5]), scale=1.0, low=[-1.0, 0.0], high=[1.0, 3.0]
    )
    assert one.batch_shape == () and one.dim == 2
    batched = TruncatedGaussian(loc=jnp.zeros((3, 2)), scale=1.0, low=-1.0, high=1.0)
    assert batched.batch_shape == (3,) and batched.mean().shape == (3, 2)
    assert batched.cov.shape == (3, 2, 2)
    #  a pytree of arrays: it crosses jit and vmap unchanged
    out = jax.jit(lambda t: t.mean())(batched)
    np.testing.assert_allclose(np.asarray(out), np.zeros((3, 2)), atol=1e-7)
    assert jax.vmap(lambda t: t.entropy().lower)(batched).shape == (3,)
    with pytest.raises(ValueError, match="give loc, scale, low and high a trailing dimension"):
        TruncatedGaussian(loc=0.0, scale=1.0, low=-1.0, high=1.0)


def _midpoint_quadrature(loc, scale, low, high, n=400_001):
    """Mean, variance and entropy of one dimension by quadrature about the box's midpoint
    (float64), so a narrow box keeps its digits."""
    mid = 0.5 * (low + high)
    t = np.linspace(low - mid, high - mid, n)
    log_k = -0.5 * ((mid + t - loc) / scale) ** 2
    p = np.exp(log_k - log_k.max())
    p /= np.trapezoid(p, t)
    mean = np.trapezoid(t * p, t)
    var = np.trapezoid((t - mean) ** 2 * p, t)
    entropy = -np.trapezoid(p * np.log(p), t)
    return mid + mean, var, entropy


@pytest.mark.parametrize(
    "loc, scale, low, high",
    [
        (0.0, 30.0, -0.5, 0.5),  # scale/width 30
        (0.0, 1e3, -0.5, 0.5),  # scale/width 1e3
        (0.2, 1e4, -0.5, 0.5),  # scale/width 1e4
        (0.0, 1.0, 0.3, 0.300001),  # a near-degenerate box
        (0.0, 1.0, 15.0, 15.00001),  # 15σ out, width 1e-5
        (0.0, 1.0, 2.9, 3.05),  # narrowish, near the tail switch
    ],
)
def test_a_narrow_box_keeps_its_moments(loc, scale, low, high):
    low, high = float(np.float32(low)), float(np.float32(high))  # the box float32 holds
    tg = TruncatedGaussian(loc=jnp.asarray([loc]), scale=[scale], low=[low], high=[high])
    want_mean, want_var, want_entropy = _midpoint_quadrature(loc, scale, low, high)
    width = high - low
    mean, var = float(tg.mean()[0]), float(tg.variance()[0])
    assert low <= mean <= high
    #  to float32 resolution at the box: two ulps of its location, or 1e-6 of its width
    tol = max(2 * float(np.spacing(np.float32(abs(want_mean)))), 1e-6 * width)
    np.testing.assert_allclose(mean, want_mean, atol=tol)
    np.testing.assert_allclose(var, want_var, rtol=1e-5)
    assert 0.0 < var <= width**2 / 4
    np.testing.assert_allclose(float(tg.entropy().lower), want_entropy, atol=1e-5)

    #  gradients through every closed form and the draw are finite
    def total(loc_):
        t = TruncatedGaussian(loc=loc_, scale=tg.scale, low=tg.low, high=tg.high)
        return (
            t.mean().sum()
            + t.variance().sum()
            + t.entropy().lower
            + t.log_prob(jnp.asarray([0.5 * (low + high)]))
            + t.sample(KEY).sum()
        )

    assert np.all(np.isfinite(np.asarray(jax.grad(total)(tg.loc))))
    draws = tg.sample(KEY, (1000,))
    assert bool(jnp.all((draws >= tg.low) & (draws <= tg.high)))


def test_an_unbounded_side_is_finite_and_differentiable():
    """An infinite bound is read far beyond the other: the one-sided box's moments match
    quadrature and every gradient is finite."""
    tg = TruncatedGaussian(
        loc=jnp.asarray([0.0, 0.5]), scale=1.0, low=[-jnp.inf, 1.0], high=[2.0, jnp.inf]
    )
    for d, (lo, hi) in enumerate(((-12.0, 2.0), (1.0, 14.0))):
        want_mean, want_var, _ = _midpoint_quadrature(float(tg.loc[d]), 1.0, lo, hi)
        np.testing.assert_allclose(float(tg.mean()[d]), want_mean, atol=1e-6)
        np.testing.assert_allclose(float(tg.variance()[d]), want_var, rtol=1e-5)

    def total(loc, scale):
        t = TruncatedGaussian(loc=loc, scale=scale, low=tg.low, high=tg.high)
        return (
            t.mean().sum()
            + t.variance().sum()
            + t.entropy().lower
            + t.log_prob(jnp.asarray([0.0, 2.0]))
            + t.sample(KEY).sum()
        )

    for grad in jax.grad(total, argnums=(0, 1))(tg.loc, tg.scale):
        assert np.all(np.isfinite(np.asarray(grad)))


@pytest.mark.parametrize("c, h", [(0.0, 0.9), (0.9, 0.5), (0.5, 0.9), (2.0, 0.3)])
def test_draws_from_a_moderately_narrow_box_have_its_moments(c, h):
    """Boxes up to about ±1σ wide are drawn through the inverse CDF, not a first-order tilt:
    the draws' mean and variance match quadrature. With N draws the variance's relative
    standard error is at most sqrt(2/N) (kurtosis ≤ 3), and the tolerance is five of them."""
    n = 40_000
    tg = TruncatedGaussian(loc=jnp.zeros(1), scale=1.0, low=[c - h], high=[c + h])
    draws = np.asarray(tg.sample(KEY, (n,)))[:, 0]
    want_mean, want_var, _ = _midpoint_quadrature(0.0, 1.0, c - h, c + h)
    assert abs(draws.var() / want_var - 1.0) < 5 * np.sqrt(2.0 / n)
    assert abs(draws.mean() - want_mean) < 5 * np.sqrt(want_var / n)
