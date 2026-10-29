"""The sign contract on mutual-information leaves, asserted rather than documented.

``objectives/base.py`` states it in prose — "Mutual-information leaves are ≥ 0" —
and until 2026-09-05 nothing enforced it. The ensemble leaf violated it by up to
−1.5 for two million agent steps, which under the ``model_uncertainty`` cost sign
became a large *bonus* for model uncertainty and cost a 42-hour parity run. The
tests here are the assertion that would have caught it in seconds, so they are
deliberately cheap, property-based over the estimator's own arithmetic, and run
in the default window.

The mathematical warrant for ``≥ 0`` on the ensemble leaf: for a mixture over
members with means ``μ_i`` and diagonal scales ``σ_i``, the law of total variance
gives the mixture's per-dimension variance as ``mean_i(σ_i²) + var_i(μ_i)``, so

    MI = ½·log(mean_i(σ_i²) + var_i(μ_i)) − mean_i(log σ_i)
       ≥ ½·log(mean_i(σ_i²))            − mean_i(log σ_i)      (variance ≥ 0)
       ≥ ½·mean_i(log σ_i²)             − mean_i(log σ_i)  = 0 (AM ≥ GM)

with equality exactly when the members agree and share one scale. The estimator
is a moment-matched Gaussian approximation to a mixture whose true entropy has no
closed form, so the leaf is an *upper* bound on the mutual information — never a
negative number.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.objectives.amortized import _diag_entropy


def _leaf(means: np.ndarray, scales: np.ndarray) -> float:
    """The ensemble leaf's arithmetic, per dimension, as the estimator computes it."""
    m, s = jnp.asarray(means), jnp.asarray(scales)
    mixture_scale = jnp.sqrt(jnp.mean(s**2, axis=0) + m.var(axis=0))
    return float((_diag_entropy(mixture_scale) - _diag_entropy(s).mean(axis=0)) / m.shape[-1])


@pytest.mark.parametrize("spread", [0.0, 1e-6, 1e-3, 0.01, 0.32, 1.0, 10.0])
@pytest.mark.parametrize("sigma", [0.1, 1.0, 5.0])
def test_the_ensemble_leaf_is_never_negative(spread: float, sigma: float) -> None:
    """Across the whole (spread, σ) plane, including σ ≫ spread — the failing regime.

    Their ``ig_head`` fixes σ ≡ 1.0 and four heads on one target converge to
    ``spread/σ ~ 1e-3``, which is where the old ``ent(std(μ)) − E[ent(σ)]`` leaf
    reported ``log(spread/σ) ≈ −6.5`` and kept falling as the ensemble improved.
    """
    rng = np.random.default_rng(0)
    means = rng.normal(0.0, spread, size=(4, 256)).astype(np.float32)
    scales = np.full((4, 256), sigma, dtype=np.float32)
    assert _leaf(means, scales) >= 0.0


def test_identical_members_carry_no_information() -> None:
    """Zero spread, one shared scale — the equality case, and the old leaf's pole."""
    scales = np.full((4, 128), 1.0, dtype=np.float32)
    assert _leaf(np.zeros((4, 128), np.float32), scales) == pytest.approx(0.0, abs=1e-6)


def test_unequal_member_scales_stay_non_negative() -> None:
    """AM ≥ GM is what carries the contract once the members disagree about σ."""
    rng = np.random.default_rng(1)
    scales = rng.uniform(0.05, 3.0, size=(4, 512)).astype(np.float32)
    for spread in (0.0, 1e-4, 0.5):
        means = rng.normal(0.0, spread, size=(4, 512)).astype(np.float32)
        assert _leaf(means, scales) >= 0.0


def test_the_leaf_matches_the_equal_sigma_closed_form() -> None:
    """With one shared σ the estimator must be exactly ``½·log(1 + s²/σ²)`` per dim.

    Averaged over dimensions *after* the log, matching their ``.mean(-1)`` — the
    two orders differ by a Jensen gap of a couple of percent, and the leaf takes
    the per-dimension one.
    """
    rng = np.random.default_rng(2)
    sigma, spread = 1.0, 0.32
    means = rng.normal(0.0, spread, size=(4, 4096)).astype(np.float32)
    scales = np.full((4, 4096), sigma, dtype=np.float32)
    per_dim_var = np.asarray(jnp.asarray(means).var(axis=0))
    expected = float(np.mean(0.5 * np.log1p(per_dim_var / sigma**2)))
    assert _leaf(means, scales) == pytest.approx(expected, rel=1e-4)


def test_information_grows_with_disagreement() -> None:
    """Monotone in the spread: the leaf must still *be* an epistemic signal.

    The fix must not buy non-negativity by flattening the term to zero — the whole
    point of the row is that it moves when the members disagree.
    """
    rng = np.random.default_rng(3)
    scales = np.full((4, 1024), 1.0, dtype=np.float32)
    leaves = [
        _leaf(rng.normal(0.0, spread, size=(4, 1024)).astype(np.float32), scales)
        for spread in (0.001, 0.01, 0.1, 0.5, 1.0)
    ]
    assert leaves == sorted(leaves)
    assert leaves[-1] > leaves[0] + 0.1


def test_the_old_leaf_would_have_failed_this_contract() -> None:
    """A regression pin on the actual defect, so the fix cannot be quietly undone.

    ``ent(std_i(μ_i)) − E_i[ent(σ_i)]`` at the run's operating point (σ = 1, four
    converged heads) is ≈ ``log(1e-3)``: not merely negative but unbounded below,
    which is why the pathology *grew* as training improved the ensemble.
    """
    rng = np.random.default_rng(4)
    scales = jnp.full((4, 512), 1.0, dtype=jnp.float32)
    means = jnp.asarray(rng.normal(0.0, 1e-3, size=(4, 512)).astype(np.float32))
    old = float((_diag_entropy(means.std(axis=0)) - _diag_entropy(scales).mean(axis=0)) / 512)
    assert old < -5.0
    assert _leaf(np.asarray(means), np.asarray(scales)) >= 0.0
