"""Performance-review numerics: each change keeps its numbers and drops its cost.

Explicit inverses become Cholesky/triangular solves, ``from_natural`` factorises
once, the LG noise lowering is hoisted out of the per-step leaves, the mixture
sampler gathers before it broadcasts, and the composition driver's async sweep
draws its order host-side. Each is pinned against the arithmetic it replaced.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.linalg import cho_solve

from hybrid_aif import Gaussian, MixtureGaussian
from hybrid_aif.distributions import InverseWishart, MatrixNormal
from hybrid_aif.distributions.gaussian import cholesky_of_inverse
from hybrid_aif.distributions.noise_priors import _diag_embed
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianObservationMNIW, lg_point
from hybrid_aif.objectives.continuous import state_info_gain

KEY = jax.random.PRNGKey(0)


def _spd(key, batch, n):
    a = jax.random.normal(key, (*batch, n, n))
    return a @ jnp.swapaxes(a, -1, -2) + n * jnp.eye(n)


# ------------------------------------------------------------ inverses -> solves


def test_cholesky_of_inverse_is_the_cholesky_of_the_inverse():
    a = _spd(KEY, (3,), 4)
    reference = jnp.linalg.cholesky(jnp.linalg.inv(a))  # the replaced computation
    np.testing.assert_allclose(np.asarray(cholesky_of_inverse(a)), np.asarray(reference), rtol=1e-4)


def test_matrix_normal_sample_matches_the_explicit_inverse_route():
    mn = MatrixNormal.from_head(jax.random.normal(KEY, (2, 3, 4)), strength=2.5)
    row_chol = jnp.linalg.cholesky(_spd(jax.random.PRNGKey(1), (2,), 3))
    k = jax.random.PRNGKey(2)
    col_chol = jnp.linalg.cholesky(jnp.linalg.inv(mn.V_inv))
    z = jax.random.normal(k, mn.M.shape, dtype=mn.M.dtype)
    reference = mn.M + row_chol @ z @ jnp.swapaxes(col_chol, -1, -2)
    np.testing.assert_allclose(np.asarray(mn.sample(k, row_chol)), np.asarray(reference), rtol=1e-4)


def test_inverse_wishart_precision_and_sample_match_the_explicit_inverse_route():
    psi = _spd(KEY, (2,), 3)
    nu = jnp.asarray([5.0, 7.5])
    iw = InverseWishart(Psi=psi, nu=nu)
    np.testing.assert_allclose(
        np.asarray(iw.expected_precision()),
        np.asarray(nu[..., None, None] * jnp.linalg.inv(psi)),
        rtol=1e-4,
    )
    k = jax.random.PRNGKey(3)
    # The replaced sampler: Bartlett factor, then inv(Psi), inv(W), and a Cholesky.
    kg, kn = jax.random.split(k)
    i = jnp.arange(3, dtype=psi.dtype)
    diag = jnp.sqrt(2.0 * jax.random.gamma(kg, (nu[..., None] - i) / 2.0))
    a = jnp.tril(jax.random.normal(kn, (2, 3, 3)), k=-1) + _diag_embed(diag)
    la = jnp.linalg.cholesky(jnp.linalg.inv(psi)) @ a
    reference = jnp.linalg.cholesky(jnp.linalg.inv(la @ jnp.swapaxes(la, -1, -2)))
    np.testing.assert_allclose(np.asarray(iw.sample(k)), np.asarray(reference), rtol=2e-4)


def test_from_natural_matches_the_two_factorisation_route():
    g = Gaussian.from_cov(jax.random.normal(KEY, (2, 3)), _spd(jax.random.PRNGKey(4), (2,), 3))
    eta1, eta2 = g.natural()
    prec = -2.0 * eta2
    cov = cho_solve((jnp.linalg.cholesky(prec), True), jnp.broadcast_to(jnp.eye(3), prec.shape))
    reference = Gaussian.from_cov(loc=(cov @ eta1[..., None])[..., 0], cov=cov)
    got = Gaussian.from_natural(eta1, eta2)
    np.testing.assert_allclose(np.asarray(got.loc), np.asarray(reference.loc), rtol=1e-4)
    np.testing.assert_allclose(
        np.asarray(got.chol), np.asarray(reference.chol), rtol=1e-4, atol=1e-6
    )


# ------------------------------------------------ the LG noise lowering, hoisted


def _learnable_lg():
    psi = jnp.asarray([[2.0, 0.3], [0.3, 1.0]])
    noise = InverseWishart(Psi=psi, nu=jnp.asarray(6.0))
    return LinearGaussianObservation(H=jnp.eye(2), c=jnp.zeros(2), noise=noise)


def test_lg_point_lowers_once_to_the_same_numbers():
    model = _learnable_lg()
    point = lg_point(model)
    assert not point.learnable_noise and not point.learnable_head
    np.testing.assert_array_equal(np.asarray(point.R_chol), np.asarray(model.R_chol))
    np.testing.assert_array_equal(np.asarray(point.head), np.asarray(model.head))
    assert lg_point(point) is point
    mniw = LinearGaussianObservationMNIW.from_head_noise(jnp.ones((2, 3)), jnp.eye(2))
    np.testing.assert_array_equal(np.asarray(lg_point(mniw).R_chol), np.asarray(mniw.R_chol))


def test_state_info_gain_factorises_a_learnable_noise_once():
    model = _learnable_lg()
    q = MixtureGaussian(
        log_weights=jnp.log(jnp.asarray([0.6, 0.4])),
        means=jnp.asarray([[0.0, 0.0], [1.0, -1.0]]),
        chols=jnp.broadcast_to(jnp.eye(2), (2, 2, 2)),
    )

    def choleskys(jaxpr) -> int:
        n = 0
        for e in jaxpr.eqns:
            n += "cholesky" in str(e.primitive)
            for v in e.params.values():
                if isinstance(v, jax.extend.core.ClosedJaxpr):
                    n += choleskys(v.jaxpr)
                elif isinstance(v, jax.extend.core.Jaxpr):
                    n += choleskys(v)
        return n

    def count(m):
        return choleskys(jax.make_jaxpr(lambda m: state_info_gain((q,), (m,)))(m).jaxpr)

    assert count(model) == count(lg_point(model)) + 1
    a = state_info_gain((q,), (model,))["infogain"]["modality_0"]
    b = state_info_gain((q,), (lg_point(model),))["infogain"]["modality_0"]
    np.testing.assert_allclose(np.asarray(a.lower), np.asarray(b.lower), rtol=1e-6)


# ----------------------------------------------------- the nu floor, host-side


def test_nu_floor_validation_keeps_its_semantics():
    with pytest.raises(ValueError, match="propriety floor"):
        InverseWishart(Psi=jnp.eye(3), nu=jnp.asarray(1.5))
    with pytest.raises(ValueError, match="propriety floor"):
        LinearGaussianObservationMNIW(
            M=jnp.zeros((3, 2)), V_inv=jnp.eye(2), Psi=jnp.eye(3), nu=jnp.asarray(2.0)
        )
    InverseWishart(Psi=jnp.eye(3), nu=np.asarray(2.5))  # a host value is accepted as is

    @jax.jit
    def build(nu):
        return InverseWishart(Psi=jnp.eye(3), nu=nu).point()

    build(jnp.asarray(1.5))  # traced: unchecked, as before


# ------------------------------------------------------ mixture sampling gather


def _mixture_sample_reference(m: MixtureGaussian, key, sample_shape):
    """The replaced sampler: broadcast every component to the draw shape, then gather."""
    key_z, key_x = jax.random.split(key)
    z = jax.random.categorical(key_z, m.log_weights, shape=sample_shape + m.batch_shape).astype(
        jnp.int32
    )
    eps = jax.random.normal(key_x, sample_shape + m.batch_shape + (m.dim,))
    means = jnp.take_along_axis(
        jnp.broadcast_to(m.means, sample_shape + m.means.shape), z[..., None, None], axis=-2
    )[..., 0, :]
    chols = jnp.take_along_axis(
        jnp.broadcast_to(m.chols, sample_shape + m.chols.shape),
        z[..., None, None, None],
        axis=-3,
    )[..., 0, :, :]
    return means + (chols @ eps[..., None])[..., 0]


@pytest.mark.parametrize("sample_shape", [(), (5,), (4, 3)])
def test_mixture_sample_is_bitwise_unchanged(sample_shape):
    k1, k2, k3 = jax.random.split(KEY, 3)
    m = MixtureGaussian(
        log_weights=jax.random.normal(k1, (2, 3)),
        means=jax.random.normal(k2, (2, 3, 4)),
        chols=jnp.linalg.cholesky(_spd(k3, (2, 3), 4)),
    )
    key = jax.random.PRNGKey(9)
    np.testing.assert_array_equal(
        np.asarray(m.sample(key, sample_shape)),
        np.asarray(_mixture_sample_reference(m, key, sample_shape)),
    )


# ------------------------------------------------------ driver: async sweep order


def test_async_with_one_mind_is_bitwise_sequential():
    """The async order draws its permutation of the sources host-side and consumes no device key,
    so one mind coupled to the mirror of each component steps identically under both orders; and
    both are the composite form's runs (``tests/data/migration_driver.npz``, ``one_mind``) under
    its key layout, whose sequential order stepped each component's factors at its turn as the
    mirror per component does."""
    import pathlib

    from hybrid_aif.composition import plan, run, sample, with_mirror
    from tests.test_merge import merged
    from tests.test_run_parity import _old_layout, as_recorded, idle

    system, params = with_mirror(*merged(policy_len=3), per_component=True)
    steps = {
        "model.agent": plan(selection="deterministic"),
        "model.world": idle,
        "world.agent": sample(),
        "world.world": sample(),
    }
    fixture = np.load(pathlib.Path(__file__).parent / "data" / "migration_driver.npz")
    got = {}
    for order in ("sequential", "async"):
        a, b, c = _old_layout(1)
        with a, b, c:
            traj = run(
                system, as_recorded(params), 3, KEY, steps, tick_order=order, record_carries=True
            )
        held = {
            f: np.asarray([s[f"world.{f}"] for s in traj[f"world.{f.split('.')[0]}"].carries])
            for f in ("agent.location", "agent.context", "world.location", "world.context")
        }
        agent = traj["model.agent"]
        held["agent.a_location"] = np.asarray(agent.intended["a_location"])
        for name, value in held.items():
            part = "actions" if name == "agent.a_location" else "states"
            np.testing.assert_array_equal(value, fixture[f"one_mind/{order}/{part}/{name}"])
        np.testing.assert_array_equal(
            agent.emitted["a_location"], fixture[f"one_mind/{order}/realised/agent.a_location"]
        )
        for m, values in agent.received.items():
            np.testing.assert_array_equal(
                values, fixture[f"one_mind/{order}/observations/agent.{m}"], err_msg=m
            )
        got[order] = held
    for name, value in got["sequential"].items():
        np.testing.assert_array_equal(value, got["async"][name])
