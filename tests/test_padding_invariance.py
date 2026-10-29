"""The D2 padding-invariance battery (required, hybrid-system design doc).

Zero-weight padding has two real hazards: ``-inf`` log-weights can poison
*gradients* (NaN through logsumexp branches even when values are correct), and
the Runnalls criterion sees zero-weight components as zero-cost merges. So for
a reference mixture vs its zero-padded version we assert (to tolerance)
identical entropy bounds, ``kl_to`` bounds, risk leaves, ``moment_match`` and
``reduce`` output, plus finite gradients through entropy and ``kl_to`` w.r.t.
means/chols/log-weights of the padded object. ``PAD_LOG_WEIGHT`` is what the
System pads with; the battery is the evidence it is safe.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import MixtureGaussian
from hybrid_aif.inference import condition_mixture
from hybrid_aif.objectives.continuous import risk
from hybrid_aif.objectives.hybrid import PAD_LOG_WEIGHT, as_mixture, pad_mixture

K_PAD = 6  # reference K=3 padded to 6


def reference() -> MixtureGaussian:
    return MixtureGaussian(
        log_weights=jnp.log(jnp.asarray([0.5, 0.3, 0.2])),
        means=jnp.asarray([[-1.0, 0.5], [1.0, -0.5], [2.0, 2.0]]),
        chols=jnp.asarray(
            [
                jnp.diag(jnp.asarray([0.6, 0.4])),
                jnp.diag(jnp.asarray([0.8, 0.5])),
                0.5 * jnp.eye(2),
            ]
        ),
    )


def target() -> MixtureGaussian:
    return MixtureGaussian(
        log_weights=jnp.log(jnp.asarray([0.85, 0.15])),
        means=jnp.asarray([[3.0, 0.0], [-3.0, 0.0]]),
        chols=jnp.asarray([0.8 * jnp.eye(2), 1.0 * jnp.eye(2)]),
    )


def padded() -> MixtureGaussian:
    return pad_mixture(reference(), K_PAD)


def _assert_interval_close(a, b, atol=1e-5):
    np.testing.assert_allclose(np.asarray(a.lower), np.asarray(b.lower), atol=atol)
    np.testing.assert_allclose(np.asarray(a.upper), np.asarray(b.upper), atol=atol)


def test_pad_shape_and_dead_weights():
    p = padded()
    assert p.num_components == K_PAD
    w = jax.nn.softmax(p.log_weights)
    np.testing.assert_allclose(np.asarray(w[3:]), 0.0, atol=0.0)  # exactly dead
    np.testing.assert_allclose(
        np.asarray(w[:3]), np.asarray(jax.nn.softmax(reference().log_weights)), atol=1e-6
    )


def test_entropy_bounds_invariant():
    _assert_interval_close(reference().entropy(), padded().entropy())


def test_kl_bounds_invariant_both_sides():
    ref, tgt = reference(), target()
    pad_t = pad_mixture(tgt, K_PAD)
    _assert_interval_close(ref.kl_to(tgt), padded().kl_to(tgt))
    _assert_interval_close(ref.kl_to(tgt), ref.kl_to(pad_t))  # padded *target* too
    _assert_interval_close(ref.kl_to(tgt), padded().kl_to(pad_t))


def test_risk_leaves_invariant():
    tgt = target()
    leaf_ref = risk((reference(),), {"x": tgt}, ("x",))["risk"]["x"]
    leaf_pad = risk((padded(),), {"x": tgt}, ("x",))["risk"]["x"]
    _assert_interval_close(leaf_ref, leaf_pad)


def test_moment_match_invariant():
    mm_ref, mm_pad = reference().moment_match(), padded().moment_match()
    np.testing.assert_allclose(np.asarray(mm_ref.loc), np.asarray(mm_pad.loc), atol=1e-6)
    np.testing.assert_allclose(np.asarray(mm_ref.cov), np.asarray(mm_pad.cov), atol=1e-6)


def test_reduce_output_invariant():
    """The Runnalls loop must consume dead slots before merging live components."""
    r_ref, r_pad = reference().reduce(2), padded().reduce(2)
    np.testing.assert_allclose(
        np.asarray(r_ref.log_weights), np.asarray(r_pad.log_weights), atol=1e-5
    )
    np.testing.assert_allclose(np.asarray(r_ref.means), np.asarray(r_pad.means), atol=1e-5)
    np.testing.assert_allclose(np.asarray(r_ref.chols), np.asarray(r_pad.chols), atol=1e-5)


@pytest.mark.parametrize("field", ["log_weights", "means", "chols"])
@pytest.mark.parametrize("bound", ["lower", "upper"])
@pytest.mark.parametrize("quantity", ["entropy", "kl_to"])
def test_gradients_through_padding_are_finite(field, bound, quantity):
    p, tgt = padded(), target()

    def scalar(x):
        m = dataclasses.replace(p, **{field: x})
        iv = m.entropy() if quantity == "entropy" else m.kl_to(tgt)
        return getattr(iv, bound)

    g = jax.grad(scalar)(getattr(p, field))
    assert bool(jnp.all(jnp.isfinite(g))), f"NaN/inf gradient d{quantity}.{bound}/d{field}"


def test_pad_mixture_handles_both_directions_and_reduce_rejects_upward():
    """The doc asks which side handles K0 < k: ``reduce`` rejects k > K, so the
    explicit pad branch is load-bearing; over-budget input goes through reduce."""
    ref = reference()
    with pytest.raises(ValueError, match="reduce"):
        ref.reduce(5)  # k > num_components: not a no-op-plus-pad
    up = pad_mixture(ref, 5)
    assert up.num_components == 5
    assert bool(jnp.all(up.log_weights[3:] == PAD_LOG_WEIGHT))
    down = pad_mixture(padded(), 2)  # over budget: Runnalls reduce
    assert down.num_components == 2
    same = pad_mixture(ref, 3)
    assert same is ref


# ---------------------------------------------- condition_mixture (D1) extension

# An LG observation model on the reference mixture's 2-D event space.
_H = jnp.asarray([[1.0, 0.2], [0.0, 1.0]])
_C = jnp.asarray([0.1, -0.2])
_R_CHOL = jnp.diag(jnp.asarray([0.3, 0.4]))
_Y = jnp.asarray([0.7, -0.3])


def test_condition_mixture_posterior_and_evidence_invariant_to_padding():
    """Conditioning a padded mixture gives the same posterior (in its live
    slots) and the same log-evidence as conditioning the reference: dead
    components contribute nothing (``-inf`` + finite marginal ``= -inf``)."""
    post_ref, ev_ref = condition_mixture(reference(), _H, _C, _R_CHOL, _Y)
    post_pad, ev_pad = condition_mixture(padded(), _H, _C, _R_CHOL, _Y)
    np.testing.assert_allclose(np.asarray(ev_ref), np.asarray(ev_pad), atol=1e-5)
    # posterior K is unchanged by conditioning; the padded posterior's dead slots
    # stay dead and its live slots match the reference posterior slot-for-slot.
    assert post_ref.num_components == 3 and post_pad.num_components == K_PAD
    w_pad = jax.nn.softmax(post_pad.log_weights)
    np.testing.assert_allclose(np.asarray(w_pad[3:]), 0.0, atol=0.0)  # exactly dead
    np.testing.assert_allclose(
        np.asarray(w_pad[:3]), np.asarray(jax.nn.softmax(post_ref.log_weights)), atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(post_pad.means[:3]), np.asarray(post_ref.means), atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(post_pad.chols[:3]), np.asarray(post_ref.chols), atol=1e-6
    )


@pytest.mark.parametrize("field", ["log_weights", "means", "chols"])
def test_gradients_through_condition_mixture_padding_are_finite(field):
    """The gradient hazard the battery guards against, for the D1 kernel:
    differentiating the log-evidence w.r.t. the padded object's parameters must
    not leak NaN/inf through the dead components' ``-inf`` log-weights."""
    p = padded()

    def scalar(x):
        m = dataclasses.replace(p, **{field: x})
        _, log_evidence = condition_mixture(m, _H, _C, _R_CHOL, _Y)
        return log_evidence

    g = jax.grad(scalar)(getattr(p, field))
    assert bool(jnp.all(jnp.isfinite(g))), f"NaN/inf gradient d(log_evidence)/d{field}"


def test_as_mixture_promotes_gaussian():
    from hybrid_aif.distributions import Gaussian

    g = Gaussian(loc=jnp.asarray([0.5, -0.5]), chol=0.3 * jnp.eye(2))
    m = as_mixture(g, 4)
    assert m.num_components == 4
    _assert_interval_close(m.entropy(), g.entropy(), atol=1e-5)
    np.testing.assert_allclose(np.asarray(m.mean()), np.asarray(g.loc), atol=1e-6)
    np.testing.assert_allclose(np.asarray(m.moment_match().cov), np.asarray(g.cov), atol=1e-6)
    # already-mixture input passes through pad_mixture unchanged in the live slots
    m2 = as_mixture(m, 4)
    assert m2.num_components == 4
