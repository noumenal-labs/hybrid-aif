"""Regression tests for float64 log-normalizing constants and sample() dtype preservation."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_aif.distributions import (
    ConditionalDirichlet,
    Dirichlet,
    Gaussian,
    InverseGamma,
    InverseWishart,
    MixtureGaussian,
)
from hybrid_aif.inference import linear_gaussian


def test_float64_log_2pi_constants_preserve_double_precision_after_float32_import():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        g = Gaussian(loc=jnp.zeros((2,), dtype=jnp.float64), chol=jnp.eye(2, dtype=jnp.float64))
        expected_entropy = 0.5 * 2 * (1.0 + math.log(2.0 * math.pi))
        np.testing.assert_allclose(float(g.entropy().lower), expected_entropy, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(
            float(g.log_prob(jnp.zeros((2,), dtype=jnp.float64))),
            -math.log(2.0 * math.pi),
            rtol=0.0,
            atol=1e-15,
        )
        _, log_z = linear_gaussian.condition(
            g,
            H=jnp.eye(2, dtype=jnp.float64),
            c=jnp.zeros((2,), dtype=jnp.float64),
            noise=jnp.eye(2, dtype=jnp.float64),
            y=jnp.zeros((2,), dtype=jnp.float64),
        )
        expected_log_z = -math.log(2.0 * math.pi) - math.log(2.0)
        np.testing.assert_allclose(float(log_z), expected_log_z, rtol=0.0, atol=1e-15)
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_float32_distributions_sample_in_float32_when_x64_is_enabled():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        key = jax.random.PRNGKey(0)
        g = Gaussian(loc=jnp.zeros((2,), dtype=jnp.float32), chol=jnp.eye(2, dtype=jnp.float32))
        assert g.sample(key).dtype == jnp.float32

        mg = MixtureGaussian(
            log_weights=jnp.zeros((2,), dtype=jnp.float32),
            means=jnp.zeros((2, 2), dtype=jnp.float32),
            chols=jnp.broadcast_to(jnp.eye(2, dtype=jnp.float32), (2, 2, 2)),
        )
        assert mg.sample(key).dtype == jnp.float32

        d = Dirichlet(concentration=jnp.ones((3,), dtype=jnp.float32))
        assert d.sample(key).dtype == jnp.float32

        cd = ConditionalDirichlet(concentration=jnp.ones((3, 2), dtype=jnp.float32), parent_ndims=1)
        assert cd.sample(key).dtype == jnp.float32

        ig = InverseGamma(
            alpha=jnp.full((2,), 3.0, dtype=jnp.float32),
            beta=jnp.ones((2,), dtype=jnp.float32),
        )
        assert ig.sample(key).dtype == jnp.float32

        iw = InverseWishart(
            Psi=jnp.eye(2, dtype=jnp.float32),
            nu=jnp.array(4.0, dtype=jnp.float32),
        )
        assert iw.sample(key).dtype == jnp.float32
    finally:
        jax.config.update("jax_enable_x64", prev)
