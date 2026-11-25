"""Regression tests for narrow-box control roots and ill-conditioned Laplace posteriors."""

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_aif.distributions import Gaussian, TruncatedGaussian
from hybrid_aif.objectives.hybrid import _control_root
from hybrid_aif.planning.gradient import _laplace


def test_truncated_gaussian_control_root_is_finite_on_zero_variance_narrow_box():
    tg = TruncatedGaussian(
        loc=jnp.array([0.0, 0.1], dtype=jnp.float32),
        scale=jnp.array([1e-10, 1.0], dtype=jnp.float32),
        low=jnp.array([-1e-20, -1.0], dtype=jnp.float32),
        high=jnp.array([1e-20, 1.0], dtype=jnp.float32),
    )
    root = _control_root(tg)
    assert jnp.all(jnp.isfinite(root))
    np.testing.assert_allclose(root @ root.T, tg.cov, rtol=1e-6, atol=1e-12)

    grad_fn = jax.grad(
        lambda scale0: jnp.sum(
            _control_root(
                TruncatedGaussian(
                    loc=jnp.array([0.0, 0.1], dtype=jnp.float32),
                    scale=jnp.stack([scale0, jnp.array(1.0, dtype=jnp.float32)]),
                    low=jnp.array([-1e-20, -1.0], dtype=jnp.float32),
                    high=jnp.array([1e-20, 1.0], dtype=jnp.float32),
                )
            )
        )
    )
    assert jnp.isfinite(grad_fn(jnp.array(1e-10, dtype=jnp.float32)))


def test_laplace_posterior_avoids_nan_on_ill_conditioned_spd_float32_hessian():
    theta = 0.7
    q = jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]],
        dtype=jnp.float32,
    )
    hess = q @ jnp.diag(jnp.array([2e7, 1.0], dtype=jnp.float32)) @ q.T
    hess = 0.5 * (hess + hess.T)
    assert float(jnp.min(jnp.linalg.eigvalsh(hess))) > 0.0

    post = _laplace(lambda x: 0.5 * (x @ hess @ x), jnp.zeros((2,), dtype=jnp.float32), gamma=1.0)
    assert isinstance(post, Gaussian)
    assert jnp.all(jnp.isfinite(post.chol))
    assert jnp.all(jnp.diag(post.chol) > 0.0)
