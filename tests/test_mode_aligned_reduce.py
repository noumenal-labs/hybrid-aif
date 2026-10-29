"""Mode-aligned (IMM/GPB2) grouped reduction (D2): kernel + schedule + fidelity.

``MixtureGaussian.reduce_mode_aligned(k, M)`` partitions the component-major
grow output by producing mode (slot ``s`` -> mode ``s % M``) and runs Runnalls
*within* each group, laying the survivors out as contiguous blocks of ``k // M``
slots per mode. Covered here:

- mode ownership: per-mode posterior mass is recoverable from the slot layout;
- moment preservation (Runnalls is exact-moment per group, and group masses are
  preserved, so the whole-mixture mean/cov are preserved too);
- the schedule wired through the System (transition + validation);
- THE §2(iv) ranking-fidelity test: ``mode_aligned`` vs ``runnalls`` vs exact
  enumeration on the switching demo. This empirically settles the de-risk
  report's open IMM question (report §2): the gaps are recorded in-test.
"""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import MixtureGaussian
from hybrid_aif.inference import propagate
from hybrid_aif.objectives import total
from hybrid_aif.system import System

_DEMO_PATH = Path(__file__).resolve().parents[1] / "examples" / "hybrid_switching.py"
_spec = importlib.util.spec_from_file_location("hybrid_switching", _DEMO_PATH)
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)


def _initial():
    """The demo's declared D as keyed beliefs (the positional tuple, converted)."""
    return dict(demo.PARAMS.initial_state_prior)  # the priors, keyed by factor, as beliefs


def _grown(seed, K, M, dim=1):
    """A component-major grown mixture (K parents through M modes -> K*M comps)."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    parent = MixtureGaussian(
        log_weights=jax.random.normal(k1, (K,)),
        means=2.0 * jax.random.normal(k2, (K, dim)),
        chols=0.5 * jnp.broadcast_to(jnp.eye(dim), (K, dim, dim)),
    )
    Fs = jnp.broadcast_to(0.9 * jnp.eye(dim), (M, dim, dim))
    bs = jnp.linspace(-1.5, 1.5, M)[:, None] * jnp.ones((M, dim))
    Qc = jnp.broadcast_to(0.3 * jnp.eye(dim), (M, dim, dim))
    tlw = jax.nn.log_softmax(jax.random.normal(jax.random.PRNGKey(seed + 99), (M,)))
    return propagate(parent, Fs, bs, Qc, tlw), M


# ----------------------------------------------------------- kernel invariants


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("K,M,k", [(4, 2, 4), (3, 2, 2), (2, 3, 3), (4, 2, 8)])
def test_mode_ownership_mass_recoverable(seed, K, M, k):
    """Per-mode posterior mass = mass of that mode's contiguous output block,
    equal to the grown mixture's per-mode mass (mode-aligned merges never move
    mass across modes)."""
    grown, M = _grown(seed, K, M)
    per = k // M
    gw = jax.nn.softmax(grown.log_weights)
    grown_mode_mass = np.array([float(gw[g::M].sum()) for g in range(M)])

    red = grown.reduce_mode_aligned(k, M)
    assert red.num_components == k
    rw = jax.nn.softmax(red.log_weights)
    block_mass = np.array([float(rw[g * per : (g + 1) * per].sum()) for g in range(M)])
    np.testing.assert_allclose(block_mass, grown_mode_mass, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("K,M,k", [(4, 2, 4), (4, 2, 2), (3, 2, 2)])
def test_mode_aligned_preserves_whole_mixture_moments(seed, K, M, k):
    """Within-group moment matching + preserved group masses => the full
    mixture mean and covariance are preserved (same strong invariant as
    ``reduce``, despite the cross-mode-merge ban)."""
    grown, M = _grown(seed, K, M, dim=2)
    red = grown.reduce_mode_aligned(k, M)
    np.testing.assert_allclose(np.asarray(red.mean()), np.asarray(grown.mean()), atol=2e-5)
    np.testing.assert_allclose(np.asarray(red.cov()), np.asarray(grown.cov()), atol=2e-4)


def test_reduce_mode_aligned_rejects_non_multiple_k():
    grown, M = _grown(0, 4, 2)
    with pytest.raises(ValueError, match="multiple of num_modes"):
        grown.reduce_mode_aligned(3, 2)


def test_mode_aligned_survives_dead_padding_and_grads():
    """Padding a mode group with dead components must not poison the reduction
    or its gradient (the group is normalized with a fully-dead guard)."""
    grown, M = _grown(1, 4, 2)
    lw = grown.log_weights.at[0].set(-jnp.inf).at[2].set(-jnp.inf)  # kill some mode-0 slots
    md = MixtureGaussian(log_weights=lw, means=grown.means, chols=grown.chols)
    red = md.reduce_mode_aligned(4, 2)
    assert bool(jnp.all(jnp.isfinite(red.means))) and bool(jnp.all(jnp.isfinite(red.chols)))
    assert not bool(jnp.any(jnp.isnan(red.log_weights)))

    def loss(x):
        m = MixtureGaussian(log_weights=x, means=grown.means, chols=grown.chols)
        return m.reduce_mode_aligned(4, 2).entropy().midpoint

    g = jax.grad(loss)(grown.log_weights)
    assert bool(jnp.all(jnp.isfinite(g)))


# --------------------------------------------------- schedule through the System


def _sys(schedule, k):
    return System(
        B_deps=((), (0,)),
        controlled=(True, False),
        policy_len=demo.HORIZON,
        mixture_k=k,
        mixture_schedule=schedule,
        factor_names=("mode", "x"),
        terms=("uncertainty:x",),  # the demo's G: risk + the unobserved x's entropy
    )


def test_default_schedule_is_runnalls_and_backward_compatible():
    """The System default must be bitwise identical to the pre-schedule path."""
    assert _sys("runnalls", 4).mixture_schedule == "runnalls"
    assert System(B_deps=((),), controlled=(False,), policy_len=1).mixture_schedule == "runnalls"
    full = demo.POLICIES[..., None]
    g_default = total(demo.SYSTEM.score_policies(demo.PARAMS, _initial(), full))
    g_explicit = total(
        _sys("runnalls", demo.K_BUDGET).score_policies(demo.PARAMS, _initial(), full)
    )
    np.testing.assert_array_equal(np.asarray(g_default.lower), np.asarray(g_explicit.lower))
    np.testing.assert_array_equal(np.asarray(g_default.upper), np.asarray(g_explicit.upper))


def test_mode_aligned_validation_fires_when_k_not_multiple_of_M():
    full = demo.POLICIES[..., None]
    with pytest.raises(ValueError, match="mode_aligned.*multiple of 2"):
        _sys("mode_aligned", 3).score_policies(demo.PARAMS, _initial(), full)


def test_mode_aligned_update_empirical_prior_keeps_static_K_and_mode_blocks():
    beliefs = {
        "mode": demo.PARAMS.initial_state_prior["mode"],
        "x": demo.PARAMS.initial_state_prior["x"],
    }
    action = {"a_mode": jnp.asarray(demo.POLICIES[2][0], jnp.int32)}  # commit-right
    pushed = _sys("mode_aligned", 4).update_empirical_prior(demo.PARAMS, beliefs, action)
    x_prior = pushed["x"]
    assert isinstance(x_prior, MixtureGaussian) and x_prior.num_components == 4
    # commit-right drives mode 1 (rightward): its mode block should carry ~0.92 mass
    w = jax.nn.softmax(x_prior.log_weights)
    right_block_mass = float(w[2:].sum())  # k//M = 2 slots per mode; block 1 = mode 1
    assert right_block_mass > 0.85


# ------------------------------------------- THE §2(iv) ranking-fidelity test


# Well-separated policy subset (min exact G gap >= 0.15, checked in-test) so
# ranking equality is a meaningful assertion, mirroring test_demo_gmm's set.
RANKING_POLICIES = jnp.asarray([[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 2, 2], [0, 0, 2]], jnp.int32)
MIN_GAP = 0.15


def _rank(v):
    return np.argsort(np.argsort(np.asarray(v)))


def _g_mid(schedule, k, policies):
    full = policies[..., None]
    return np.asarray(
        total(_sys(schedule, k).score_policies(demo.PARAMS, _initial(), full)).midpoint
    )


def test_ranking_fidelity_mode_aligned_vs_runnalls_vs_exact():
    """§2(iv): at an adequate budget the three pipelines rank policies
    identically; the recorded gaps settle the IMM question empirically.

    Empirical finding on the demo (M=2): mode_aligned is a strictly *coarser*
    approximation than runnalls at equal k (cross-mode merges forbidden). At the
    demo's tuned k=4 runnalls already achieves ranking fidelity while
    mode_aligned does not (its G error exceeds the inter-policy gaps and flips
    mid-table ranks); mode_aligned needs ~2x the budget (k=8) to catch up. So
    the schedule is ranking-faithful *given enough components*, but the
    de-risk report's IMM preference costs budget, not zero — the gradient-
    friendliness (no argmin) is not free."""
    g_exact = np.asarray(jax.vmap(demo.policy_scores_exact)(RANKING_POLICIES)[0].midpoint)
    gaps = np.diff(np.sort(g_exact))
    assert gaps.min() >= MIN_GAP, f"policy set degenerate: gaps {gaps}"
    rank_exact = _rank(g_exact)

    # equal small budget (k=4): mode_aligned is materially coarser than runnalls
    g_run4, g_ma4 = (
        _g_mid("runnalls", 4, RANKING_POLICIES),
        _g_mid("mode_aligned", 4, RANKING_POLICIES),
    )
    gap_run4 = float(np.max(np.abs(g_run4 - g_exact)))
    gap_ma4 = float(np.max(np.abs(g_ma4 - g_exact)))
    assert gap_run4 < MIN_GAP  # runnalls faithful at k=4 (observed ~0.07)
    assert gap_ma4 > gap_run4  # mode_aligned strictly coarser at equal k (observed ~0.82)
    assert np.array_equal(_rank(g_run4), rank_exact)

    # adequate budget (k=8): all three agree, gaps below the inter-policy gaps
    g_run8, g_ma8 = (
        _g_mid("runnalls", 8, RANKING_POLICIES),
        _g_mid("mode_aligned", 8, RANKING_POLICIES),
    )
    gap_ma8 = float(np.max(np.abs(g_ma8 - g_exact)))
    assert gap_ma8 < MIN_GAP  # observed ~0.05
    assert np.array_equal(_rank(g_ma8), rank_exact)
    assert np.array_equal(_rank(g_run8), rank_exact)
    assert np.array_equal(_rank(g_ma8), _rank(g_run8))

    # record the observed gap magnitudes (visible with -s): the report's answer.
    gap_run8 = float(np.max(np.abs(g_run8 - g_exact)))
    print(
        f"\n[ranking-fidelity] exact G gaps min={gaps.min():.3f}; "
        f"max|G-exact|: runnalls k4={gap_run4:.3f} k8={gap_run8:.3f}, "
        f"mode_aligned k4={gap_ma4:.3f} k8={gap_ma8:.3f}"
    )
