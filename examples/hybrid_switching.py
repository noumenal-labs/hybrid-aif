"""Flagship GMM demo: EFE planning over a hybrid switching System.

The minimal model class where continuous-state epistemic active inference does
anything at all (Koudahl et al. 2021): a 1D state under M=2 mode-conditioned
linear-Gaussian dynamics. Mode 0 drifts left, mode 1 drifts right; actions
influence which mode fires (a "dither" action leaves the mode maximally
uncertain, "commit" actions make one mode dominant).

The model is *declared*, not hand-rolled: a two-factor :class:`System` — a
controlled discrete ``mode`` factor plus a 1-D mixture ``x`` factor whose
``B_deps`` name the mode — with the preference TARGET as a ``state_targets``
designation. The de-risk report §2 composition (exact grow through all M
transitions, then Runnalls reduction to ``mixture_k``) is the System's
transition closure; risk + belief-entropy per step is its term closure, the
entropy the ``uncertainty`` row on ``x`` (``terms=("uncertainty:x",)``: no
channel observes ``x``, so it has no information to gain). Every
leaf is a Kolchinsky–Tracey *bound pair*, so G arrives as an [lower, upper]
Interval — the table prints the bounds, not a point estimate that hides the
estimator bias.

The 6 hand-declared policies go through ``System.score_policies``, the
planner-free scoring face (the full 27-policy enumeration would be one
``SYSTEM.infer_policies(PARAMS, SYSTEM.initial_priors(PARAMS))`` away). Alongside, the demo scores
the exact enumeration — on ``propagate`` directly, outside the
System's budget and its reduce() code path, so it stays an independent
reference — and shows the policy rankings agree: ranking fidelity is what
planning needs (de-risk report §2(iv)).

Run: ``.venv/bin/python examples/hybrid_switching.py``
"""

import time

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif import (
    Categorical,
    ConditionalCategorical,
    ConditionalGaussian,
    Interval,
    LinearGaussianTransition,
    MixtureGaussian,
)
from hybrid_aif.inference import propagate
from hybrid_aif.objectives import total
from hybrid_aif.objectives.continuous import risk
from hybrid_aif.planning import policy_posterior
from hybrid_aif.system import System, SystemParams

# --------------------------------------------------------------------------- model
# M = 2 modes, 1D state: x' = F_j x + b_j + eps_j, per-mode maps and innovations.

NUM_MODES = 2
FS = jnp.asarray([[[0.9]], [[0.9]]])  # (M, 1, 1) per-mode F
INNOVATIONS = ConditionalGaussian(  # per-mode b (loc) and chol(Q)
    loc=jnp.asarray([[-1.2], [1.2]]),
    chol=jnp.asarray([[[0.3]], [[0.45]]]),
    parent_ndims=1,
)

# Actions influence the mode: rows are per-action mode log-probabilities.
ACTION_MODE_LOGW = jnp.log(
    jnp.asarray(
        [
            [0.50, 0.50],  # 0 dither: mode maximally uncertain
            [0.92, 0.08],  # 1 commit-left
            [0.08, 0.92],  # 2 commit-right
        ]
    )
)

# Initial belief: bimodal — the agent does not know which basin it is in.
INIT_BELIEF = MixtureGaussian(
    log_weights=jnp.log(jnp.asarray([0.5, 0.5])),
    means=jnp.asarray([[-1.0], [1.0]]),
    chols=jnp.asarray([[[0.6]], [[0.6]]]),
)

# Preference target: mostly "be right, tightly", with a small left basin.
TARGET = MixtureGaussian(
    log_weights=jnp.log(jnp.asarray([0.85, 0.15])),
    means=jnp.asarray([[3.0], [-3.0]]),
    chols=jnp.asarray([[[0.8]], [[1.0]]]),
)

HORIZON = 3
K_BUDGET = 4  # Runnalls budget: k < K0 * M**HORIZON = 16 (genuine reduction)

POLICIES = jnp.asarray(
    [
        [0, 0, 0],  # dither
        [1, 1, 1],  # commit-left
        [2, 2, 2],  # commit-right
        [0, 2, 2],  # probe-then-right
        [0, 0, 2],  # late-right
        [2, 1, 2],  # zigzag
    ],
    jnp.int32,
)
POLICY_LABELS = (
    "dither",
    "commit-left",
    "commit-right",
    "probe-then-right",
    "late-right",
    "zigzag",
)

# -------------------------------------------------------------------------- System
# The declaration IS the pipeline: mode's B is the action->mode table (natural
# actions-leading layout), x's B is the switching transition whose mode weights
# come from mode's post-transition belief each step.


def make_system(k: int = K_BUDGET) -> System:
    return System(
        B_deps=((), (0,)),  # mode: control-only parents; x rides the mode
        controlled=(True, False),
        policy_len=HORIZON,
        mixture_k=k,
        factor_names=("mode", "x"),
        terms=("uncertainty:x",),
    )


SYSTEM = make_system()
PARAMS = SystemParams(
    transition={
        "mode": ConditionalCategorical.from_parent_leading(ACTION_MODE_LOGW, parent_ndims=1),
        "x": LinearGaussianTransition(F=FS, innovations=INNOVATIONS),
    },
    initial_state_prior={"mode": Categorical(logits=jnp.zeros(NUM_MODES)), "x": INIT_BELIEF},
    state_targets={"x": TARGET},  # the preference, as a standing designation
)


def policy_scores(policy: Array, k: int = K_BUDGET) -> tuple[Interval, Interval, Interval]:
    """Reduced-pipeline (G, risk, entropy) Intervals for one policy; vmap/jit-safe."""
    full = policy[..., None]  # the one control node's column
    system = make_system(k)
    tree = system.score_policies(PARAMS, system.initial_priors(PARAMS), full)
    return total(tree), tree["risk"]["x"], tree["uncertainty"]["x"]


def reduced_beliefs(policy: Array, k: int = K_BUDGET) -> list[MixtureGaussian]:
    """Per-step x beliefs under the reduced pipeline: the predict_beliefs face."""
    full = policy[..., None]
    system = make_system(k)
    return [qs["x"] for qs in system.predict_beliefs(PARAMS, system.initial_priors(PARAMS), full)]


# ------------------------------------------------------------------ exact reference


def exact_beliefs(policy: Array) -> list[MixtureGaussian]:
    """Per-step beliefs with no reduction: exact grow to K0 * M**t components.

    Deliberately on ``propagate``, not a big-budget System: the §2(iv) reference
    must not share the reduced pipeline's reduce()/static-K-padding code path.
    """
    m, out = INIT_BELIEF, []
    for t in range(policy.shape[0]):
        m = propagate(m, FS, INNOVATIONS, None, ACTION_MODE_LOGW[policy[t]])
        out.append(m)
    return out


def policy_scores_exact(policy: Array) -> tuple[Interval, Interval, Interval]:
    """Exact-enumeration (G, risk, entropy) Intervals for one policy; vmap/jit-safe."""
    risk_sum = ent_sum = None
    for m in exact_beliefs(policy):
        r = risk((m,), {"x": TARGET}, ("x",))["risk"]["x"]
        h = m.entropy()
        risk_sum = r if risk_sum is None else risk_sum + r
        ent_sum = h if ent_sum is None else ent_sum + h
    return risk_sum + ent_sum, risk_sum, ent_sum


# --------------------------------------------------------------------------- table


def _rank(values: jnp.ndarray) -> jnp.ndarray:
    order = jnp.argsort(values)
    return jnp.argsort(order)  # rank of each entry, 0 = best (lowest G)


def main() -> None:
    t0 = time.perf_counter()
    full = POLICIES[..., None]
    tree = jax.jit(lambda p: SYSTEM.score_policies(p, SYSTEM.initial_priors(p), full))(PARAMS)
    g_red, risk_red, ent_red = total(tree), tree["risk"]["x"], tree["uncertainty"]["x"]
    g_ex, _, _ = jax.jit(jax.vmap(policy_scores_exact))(POLICIES)
    rank_red, rank_ex = _rank(g_red.midpoint), _rank(g_ex.midpoint)
    elapsed = time.perf_counter() - t0

    n_exact = INIT_BELIEF.num_components * NUM_MODES**HORIZON
    print("Hybrid switching demo: M=2 mode-conditioned linear-Gaussian dynamics, 1D state")
    print(
        f"System(mode, x), grow-then-reduce (Runnalls mixture_k={K_BUDGET}) vs "
        f"exact enumeration ({n_exact} components at H={HORIZON})"
    )
    print("G = risk (KL to target) + belief entropy, per-step Kolchinsky-Tracey bound pairs\n")

    header = (
        f"{'policy':<18}{'actions':<10}{'risk [lo, hi]':<19}{'H [lo, hi]':<17}"
        f"{'G [lo, hi]':<19}{'G mid':>7}{'exact':>8}{'rank':>6}"
    )
    print(header)
    print("-" * len(header))
    for i, label in enumerate(POLICY_LABELS):
        actions = ",".join(str(int(a)) for a in POLICIES[i])
        row = (
            f"{label:<18}{actions:<10}"
            f"[{float(risk_red.lower[i]):6.2f},{float(risk_red.upper[i]):6.2f}]   "
            f"[{float(ent_red.lower[i]):5.2f},{float(ent_red.upper[i]):5.2f}]   "
            f"[{float(g_red.lower[i]):6.2f},{float(g_red.upper[i]):6.2f}]   "
            f"{float(g_red.midpoint[i]):7.2f}{float(g_ex.midpoint[i]):8.2f}"
            f"{int(rank_red[i]):>4}/{int(rank_ex[i])}"
        )
        print(row)

    agree = bool(jnp.all(rank_red == rank_ex))
    best = POLICY_LABELS[int(jnp.argmin(g_red.midpoint))]
    q_pi = policy_posterior(g_red, SYSTEM.gamma)
    top = POLICY_LABELS[int(jnp.argmax(q_pi.probs))]
    print(f"\nbest policy (min G midpoint): {best}")
    print(f"policy posterior argmax (softmax(-gamma G), gamma={SYSTEM.gamma:g}): {top}")
    print(f"policy rankings agree, reduced vs exact: {agree}")
    print(
        f"components per step, reduced: {INIT_BELIEF.num_components} (pad to {K_BUDGET}) -> "
        + " -> ".join(f"{K_BUDGET * NUM_MODES}~>{K_BUDGET}" for _ in range(HORIZON))
        + "   (~> = Runnalls merge; pad = static-K dead components)"
    )
    print(f"elapsed (jit compile + eval, both pipelines): {elapsed:.2f}s")


if __name__ == "__main__":
    main()
