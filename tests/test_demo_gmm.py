"""GMM demo suite, incl. the de-risk report's §2(iv) ranking-fidelity test.

The load-bearing correctness claim: collapsed-pipeline EFE (grow -> Runnalls
reduce(k), k < K0 * M**H) matches exact-enumeration EFE within documented
tolerance AND the policy *rankings* agree — ranking fidelity is what planning
needs. Since the v0.2 rewrite the demo's reduced pipeline is the hybrid
System's transition/term closures (static-K: beliefs padded to mixture_k dead
components on entry); the exact reference stays a pure-numpy enumeration
(K0 * M**t components at step t, no reduction) outside that code path. Both
pipelines are scored with the same estimators (risk = Kolchinsky–Tracey
mixture-KL bound midpoints, uncertainty = KT entropy bound midpoints), so
the tolerance isolates *reduction* error, while separate sandwich tests pin
the estimators against Monte Carlo.

Documented tolerances (float32, demo model constants):
- exact pipeline vs numpy enumeration: machine precision (atol 1e-4 over H=3);
- reduced (k=4) vs exact G midpoints: observed max |diff| 0.066 nats at H=3
  (0.041 at H=2), asserted at atol=0.12 with min exact policy gap >= 0.15
  (observed 0.17 at H=2 / 0.24 at H=3) so ranking equality is meaningful;
- GPB1 (k=1 collapse): matches its numpy moment recursion to 1e-4, but its G
  error vs exact exceeds 1.0 nats and scrambles mid-table rankings — the
  documented reason the Runnalls schedule, not single-Gaussian collapse, is
  the planner default.
"""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import MixtureGaussian
from hybrid_aif.inference import rollout_gpb1
from hybrid_aif.interval import Interval
from hybrid_aif.objectives import total
from hybrid_aif.objectives.continuous import risk, uncertainty
from hybrid_aif.planning import policy_posterior

_DEMO_PATH = Path(__file__).resolve().parents[1] / "examples" / "hybrid_switching.py"
_spec = importlib.util.spec_from_file_location("hybrid_switching", _DEMO_PATH)
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)

# Demo model constants as numpy, for the enumeration reference.
F = np.asarray(demo.FS)[:, 0, 0]  # (M,)
B = np.asarray(demo.INNOVATIONS.loc)[:, 0]  # (M,)
Q = np.asarray(demo.INNOVATIONS.chol)[:, 0, 0] ** 2  # (M,)
T = np.exp(np.asarray(demo.ACTION_MODE_LOGW))  # (A, M) mode probs per action
W0 = np.exp(np.asarray(demo.INIT_BELIEF.log_weights))
MU0 = np.asarray(demo.INIT_BELIEF.means)[:, 0]
V0 = np.asarray(demo.INIT_BELIEF.chols)[:, 0, 0] ** 2

# Ranking test sets: >=4 policies with genuinely distinct G (min exact pairwise
# gap >= MIN_GAP, checked in-test). H=2 swaps late-right (duplicate of dither
# when truncated) for zigzag; H=3 drops zigzag (its exact G sits within
# reduction tolerance of commit-left's).
RANKING_POLICIES = {
    2: [[0, 0], [1, 1], [2, 2], [0, 2], [2, 1]],
    3: [[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 2, 2], [0, 0, 2]],
}
G_ATOL = 0.12  # reduced-vs-exact G midpoint tolerance (observed max 0.066)
MIN_GAP = 0.15  # required distinctness of exact per-policy G


def numpy_enumeration(policy):
    """Exact predictive mixtures per step, pure numpy: K0 * M**t components.

    Component order is propagate()'s component-major layout (slot k*M + j).
    """
    w, mu, v = W0.copy(), MU0.copy(), V0.copy()
    steps = []
    for a in policy:
        t = T[a]
        w = (w[:, None] * t[None, :]).reshape(-1)
        mu = (F[None, :] * mu[:, None] + B[None, :]).reshape(-1)
        v = (F[None, :] ** 2 * v[:, None] + Q[None, :]).reshape(-1)
        steps.append((w.copy(), mu.copy(), v.copy()))
    return steps


def score_numpy_steps(steps) -> Interval:
    """Score enumerated (w, mu, v) steps with the demo's own G = risk + entropy."""
    g = None
    for w, mu, v in steps:
        m = MixtureGaussian(
            log_weights=jnp.log(jnp.asarray(w, jnp.float32)),
            means=jnp.asarray(mu, jnp.float32)[:, None],
            chols=jnp.sqrt(jnp.asarray(v, jnp.float32))[:, None, None],
        )
        g_t = total(
            {
                **risk((m,), {"state": demo.TARGET}, ("state",)),
                **uncertainty((m,), ("state",)),
            }
        )
        g = g_t if g is None else g + g_t
    return g


# ------------------------------------------------- exact pipeline vs enumeration


def test_exact_pipeline_matches_numpy_enumeration():
    """Grow-without-reduce equals the numpy enumeration to machine precision."""
    policy = jnp.asarray([0, 2, 1], jnp.int32)
    beliefs = demo.exact_beliefs(policy)
    steps = numpy_enumeration(np.asarray(policy))
    assert beliefs[-1].num_components == len(steps[-1][0]) == 2 * 2**3
    for m, (w, mu, v) in zip(beliefs, steps, strict=True):
        np.testing.assert_allclose(
            np.exp(np.asarray(m.log_weights)) / np.exp(np.asarray(m.log_weights)).sum(),
            w / w.sum(),
            atol=1e-6,
        )
        np.testing.assert_allclose(np.asarray(m.means)[:, 0], mu, atol=1e-4)
        np.testing.assert_allclose(np.asarray(m.chols)[:, 0, 0] ** 2, v, atol=1e-4)


# --------------------------------------------- THE ranking-fidelity test (§2(iv))


@pytest.mark.parametrize("horizon", [2, 3])
def test_ranking_fidelity_reduced_vs_exact(horizon):
    """Collapsed-pipeline EFE within documented tolerance of exact enumeration,
    and policy rankings identical, across 5 policies with genuinely distinct G."""
    policies = RANKING_POLICIES[horizon]
    k = demo.K_BUDGET
    assert k < len(W0) * len(F) ** horizon  # genuine reduction, not a no-op

    g_exact = np.array([float(score_numpy_steps(numpy_enumeration(p)).midpoint) for p in policies])
    g_reduced = np.asarray(
        jax.vmap(lambda p: demo.policy_scores(p, k)[0].midpoint)(jnp.asarray(policies, jnp.int32))
    )

    # The policies genuinely disagree: exact G values are pairwise separated.
    gaps = np.diff(np.sort(g_exact))
    assert gaps.min() >= MIN_GAP, f"policy set degenerate: gaps {gaps}"

    np.testing.assert_allclose(g_reduced, g_exact, atol=G_ATOL)
    assert np.array_equal(np.argsort(g_reduced), np.argsort(g_exact)), (
        f"ranking mismatch: reduced {np.argsort(g_reduced)} vs exact {np.argsort(g_exact)}"
    )


# ------------------------------------------------------------------ GPB1 variant


def _gpb1_scores(policy) -> tuple[float, float, float]:
    """(G midpoint, final mean, final var) under per-step single-Gaussian collapse."""
    g = demo.INIT_BELIEF
    total_iv = None
    for a in policy:
        g, _ = rollout_gpb1(
            g,
            demo.FS,
            demo.INNOVATIONS.loc,
            demo.INNOVATIONS.chol,
            demo.ACTION_MODE_LOGW[int(a)],
            horizon=1,
        )
        m1 = MixtureGaussian(log_weights=jnp.zeros(1), means=g.loc[None], chols=g.chol[None])
        g_t = total(
            {
                **risk((m1,), {"state": demo.TARGET}, ("state",)),
                **uncertainty((m1,), ("state",)),
            }
        )
        total_iv = g_t if total_iv is None else total_iv + g_t
    return float(total_iv.midpoint), float(g.loc[0]), float(g.chol[0, 0]) ** 2


def _gpb1_numpy_moments(policy) -> tuple[float, float]:
    """Numpy reference: exact moment-matched (GPB1) mean/variance recursion."""
    mu = float(np.sum(W0 * MU0))
    v = float(np.sum(W0 * (V0 + (MU0 - mu) ** 2)))
    for a in policy:
        t = T[a]
        mj = F * mu + B
        vj = F**2 * v + Q
        mu = float(np.sum(t * mj))
        v = float(np.sum(t * (vj + (mj - mu) ** 2)))
    return mu, v


def test_gpb1_matches_numpy_moment_recursion():
    for policy in ([0, 0, 0], [2, 1, 2], [0, 2, 2]):
        _, mu, v = _gpb1_scores(policy)
        mu_ref, v_ref = _gpb1_numpy_moments(policy)
        np.testing.assert_allclose(mu, mu_ref, atol=1e-4)
        np.testing.assert_allclose(v, v_ref, atol=1e-4)


def test_gpb1_collapse_degrades_g_but_keeps_best_policy():
    """The k=1 ablation: far larger G error than Runnalls-k and scrambled
    mid-table rankings (multimodality scored as one huge Gaussian), though the
    clear winner survives — why the reduction *schedule* is load-bearing."""
    policies = np.asarray(demo.POLICIES)
    g_exact = np.array([float(score_numpy_steps(numpy_enumeration(p)).midpoint) for p in policies])
    g_gpb1 = np.array([_gpb1_scores(p)[0] for p in policies])
    g_reduced = np.asarray(jax.vmap(lambda p: demo.policy_scores(p)[0].midpoint)(demo.POLICIES))

    assert np.abs(g_reduced - g_exact).max() < G_ATOL
    assert np.abs(g_gpb1 - g_exact).max() > 1.0
    assert not np.array_equal(np.argsort(g_gpb1), np.argsort(g_exact))
    assert np.argmin(g_gpb1) == np.argmin(g_exact)


# ------------------------------------------------- bound sandwiches post-reduction


@pytest.mark.parametrize("policy_idx", [0, 3])  # dither, probe-then-right
def test_entropy_bounds_sandwich_mc_post_reduction(policy_idx):
    final = demo.reduced_beliefs(demo.POLICIES[policy_idx])[-1]
    assert final.num_components == demo.K_BUDGET
    iv = final.entropy()
    x = final.sample(jax.random.PRNGKey(7 + policy_idx), (200_000,))
    mc = float(-final.log_prob(x).mean())
    assert float(iv.lower) - 0.02 <= mc <= float(iv.upper) + 0.02
    assert float(iv.width) <= np.log(demo.K_BUDGET) + 1e-6


@pytest.mark.parametrize("policy_idx", [0, 3])
def test_risk_bounds_sandwich_mc_post_reduction(policy_idx):
    final = demo.reduced_beliefs(demo.POLICIES[policy_idx])[-1]
    iv = final.kl_to(demo.TARGET)
    x = final.sample(jax.random.PRNGKey(17 + policy_idx), (200_000,))
    mc = float((final.log_prob(x) - demo.TARGET.log_prob(x)).mean())
    assert float(iv.lower) - 0.03 <= mc <= float(iv.upper) + 0.03
    assert float(iv.lower) >= 0.0


# ------------------------------------------------------------------------- smoke


def test_policy_scores_jit_vmap_smoke():
    jitted = jax.jit(jax.vmap(demo.policy_scores))(demo.POLICIES)
    for i, policy in enumerate(demo.POLICIES):
        eager = demo.policy_scores(policy)
        for iv_batch, iv in zip(jitted, eager, strict=True):
            np.testing.assert_allclose(float(iv_batch.lower[i]), float(iv.lower), atol=1e-4)
            np.testing.assert_allclose(float(iv_batch.upper[i]), float(iv.upper), atol=1e-4)


def test_system_scoring_face_matches_per_policy_wrappers():
    """The demo's System path (score_policies over the 6-policy set) and its
    per-policy notebook wrappers agree leaf-for-leaf, and q_pi tracks G."""
    full = demo.POLICIES[..., None]
    beliefs = dict(demo.PARAMS.initial_state_prior)  # the priors, keyed by factor, as beliefs
    tree = demo.SYSTEM.score_policies(demo.PARAMS, beliefs, full)
    g = total(tree)
    q_pi = policy_posterior(g, demo.SYSTEM.gamma)
    g_w, risk_w, ent_w = jax.vmap(demo.policy_scores)(demo.POLICIES)
    np.testing.assert_allclose(np.asarray(g.lower), np.asarray(g_w.lower), atol=1e-5)
    np.testing.assert_allclose(np.asarray(g.upper), np.asarray(g_w.upper), atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(tree["risk"]["x"].lower), np.asarray(risk_w.lower), atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(tree["uncertainty"]["x"].lower), np.asarray(ent_w.lower), atol=1e-5
    )
    assert int(jnp.argmax(q_pi.probs)) == int(jnp.argmin(g.midpoint))


def test_demo_main_prints_ranked_table(capsys):
    demo.main()
    out = capsys.readouterr().out
    for label in demo.POLICY_LABELS:
        assert label in out
    assert "policy rankings agree, reduced vs exact: True" in out
    assert "best policy (min G midpoint): commit-right" in out
