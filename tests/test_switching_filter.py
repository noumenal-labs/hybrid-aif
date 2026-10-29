"""Closed-loop switching-filter acceptance suite (workstream C phase 1, D5).

The load-bearing behavioral claim: with the hybrid channel ``infogain`` leaf on,
the agent *probes* — its first action separates the hidden mode-hypotheses so the
next observation disambiguates them — and its mode entropy collapses within a
tick; ablating it (``epistemic=False``) removes the probing,
the agent anchors immediately and stays ignorant of the mode. Alongside the
behavior: filtering parity against an independent exact-enumeration oracle (no
reduction) at 1e-4, the chosen k=8 vs the exact-fallback k=16 ranking fidelity,
and a full-episode no-NaN / no-dead-end guard.

The demo module is loaded from ``examples/`` the same way the other demo suites
load theirs (``importlib`` off the file path).
"""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_DEMO_PATH = Path(__file__).resolve().parents[1] / "examples" / "switching_filter.py"
_spec = importlib.util.spec_from_file_location("switching_filter", _DEMO_PATH)
sf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sf)

KEY = jax.random.PRNGKey(0)
TRUE_MODE = 1


# ------------------------------------------------------------------ probing emerges


def test_probing_emerges_with_info_gain():
    """Info gain on: first action is a probe (drift/run) and the mode is disambiguated.

    Quantified: the agent's first action separates the mode-hypotheses (it is in
    PROBE_ACTIONS, not the anchor), it probes before it ever anchors, and the
    mode entropy drops from a full bit (log 2) at t=0 to well under half a bit by
    t=1 — the probe's observation resolved the current.
    """
    system, params = sf.make_agent(epistemic=True)
    traj = sf.closed_loop(KEY, system, params, true_mode=TRUE_MODE)
    probe, anchor = sf._first_probe_and_commit(traj.actions)

    assert traj.actions[0] in sf.PROBE_ACTIONS  # first move is a probe
    assert probe == 0 and (anchor == -1 or probe < anchor)  # probes before settling
    assert traj.mode_entropies[0] > 0.69 - 1e-3  # starts maximally uncertain (log 2)
    assert traj.mode_entropies[1] < 0.35  # disambiguated after one probe
    # and it committed to the correct current (mode 1 flows right)
    assert traj.mode_probs[-1, TRUE_MODE] > 0.95


def test_ablation_removes_probing():
    """Info gain off: the agent anchors on tick 0 and never probes — so it never learns.

    Anchoring holds the dock but makes the two mode-branches identical, so the
    observations carry no information about the current: the ablated agent's mode
    entropy stays pinned at its maximum for the whole episode. Same env, same
    everything but the selector.
    """
    system, params = sf.make_agent(epistemic=False)
    traj = sf.closed_loop(KEY, system, params, true_mode=TRUE_MODE)

    assert traj.actions[0] == sf.COMMIT_ACTION  # anchors immediately
    assert not np.isin(traj.actions, np.asarray(sf.PROBE_ACTIONS)).any()  # never probes
    # never disambiguates: entropy essentially unchanged from the log-2 start
    assert traj.mode_entropies.min() > 0.69 - 1e-2
    assert abs(traj.mode_probs[-1, TRUE_MODE] - 0.5) < 1e-2


def test_probing_contrast_is_the_info_gain_terms_doing():
    """The two agents differ only in the selector, yet diverge on tick 0 — a clean ablation."""
    s_ig, p_ig = sf.make_agent(epistemic=True)
    s_ab, p_ab = sf.make_agent(epistemic=False)
    t_ig = sf.closed_loop(KEY, s_ig, p_ig, true_mode=TRUE_MODE)
    t_ab = sf.closed_loop(KEY, s_ab, p_ab, true_mode=TRUE_MODE)
    assert t_ig.actions[0] != t_ab.actions[0]
    # the prober ends far more certain than the ablation
    assert t_ig.mode_entropies[-1] < 0.2 < t_ab.mode_entropies[-1]


# ------------------------------------------------------------------ oracle parity


def test_oracle_parity_exact_fallback():
    """Reduced pipeline at the exact-fallback budget k=K0*M^H matches enumeration.

    Over a short horizon the mode-aligned pipeline does no live merges at
    k=K_EXACT, so filtering through the public faces (infer_states /
    update_empirical_prior) reproduces the independent textbook enumeration to
    ~1e-4 or tighter on both the mode posterior and the x posterior mean.
    """
    system, params = sf.make_agent(epistemic=True)
    traj = sf.closed_loop(KEY, system, params, true_mode=TRUE_MODE)
    acts = traj.actions[: sf.PARITY_STEPS]
    ys, _ = sf.env_obs_stream(KEY, params, acts, TRUE_MODE)
    mode_exact, x_exact = sf.exact_filter(params, acts, ys)
    mode_k16, x_k16 = sf.replay_filter(params, acts, ys, k=sf.K_EXACT)
    np.testing.assert_allclose(mode_k16, mode_exact, atol=1e-4)
    np.testing.assert_allclose(x_k16, x_exact, atol=1e-4)


def test_chosen_k_filtering_fidelity():
    """k=8 filtering stays faithful to the exact oracle (mode-aligned preserves bank mass)."""
    system, params = sf.make_agent(epistemic=True)
    traj = sf.closed_loop(KEY, system, params, true_mode=TRUE_MODE)
    acts = traj.actions[: sf.PARITY_STEPS]
    ys, _ = sf.env_obs_stream(KEY, params, acts, TRUE_MODE)
    mode_exact, _ = sf.exact_filter(params, acts, ys)
    mode_k8, _ = sf.replay_filter(params, acts, ys, k=sf.K_BUDGET)
    np.testing.assert_allclose(mode_k8, mode_exact, atol=1e-3)


def test_k8_planning_ranking_fidelity():
    """k=8 vs the exact fallback k=16 on a bimodal belief (reduction genuinely bites).

    Reproduces the design doc's D2 finding: at 2x the runnalls-tuned budget the
    mode-aligned schedule keeps the whole policy ranking and the G midpoints
    within a hair of exact enumeration.
    """
    ranks_agree, g_gap = sf.planning_fidelity()
    assert ranks_agree
    assert g_gap < 1e-3  # observed ~8e-6 nats


# ------------------------------------------------------------------ health / no dead-ends


def _finite(x) -> bool:
    return bool(jnp.all(jnp.isfinite(jnp.asarray(x))))


@pytest.mark.parametrize("info", [True, False])
def test_no_nan_or_dead_end_over_full_episode(info):
    """A full episode for either agent: every belief finite, probabilities valid, x bounded."""
    system, params = sf.make_agent(epistemic=info)
    traj = sf.closed_loop(KEY, system, params, true_mode=TRUE_MODE, num_steps=12)
    assert _finite(traj.mode_probs) and _finite(traj.x_means) and _finite(traj.mode_entropies)
    row_sums = traj.mode_probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)
    assert (traj.mode_probs >= -1e-6).all()
    assert np.abs(traj.x_means).max() < 20.0  # station-keeping never diverges


def test_true_mode_zero_also_disambiguated():
    """Symmetry: the prober also finds a left-flowing current (mode 0)."""
    system, params = sf.make_agent(epistemic=True)
    traj = sf.closed_loop(KEY, system, params, true_mode=0)
    assert traj.actions[0] in sf.PROBE_ACTIONS
    assert traj.mode_probs[-1, 0] > 0.9
