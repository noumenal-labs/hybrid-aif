"""Closed-loop probing: filtering a switching world to disambiguate a hidden mode.

The acceptance demo for mixture *filtering* (D5). Where
``hybrid_switching.py`` scored open-loop plans, here the loop is closed: a noisy
continuous observation arrives each tick, the agent filters it, and — crucially —
the hidden discrete *mode* is inferred from that continuous evidence, not chosen.

A boat wants to hold station at the dock (``x = 0``) in an unknown current. A
hidden mode ``j in {0, 1}`` fixes the current for the whole episode (mode 0 sets
it flowing left, mode 1 right). The boat has three controls on its position
``x`` under mode-conditioned linear-Gaussian dynamics ``x' = F(u) x + b(mode, u)
+ eps``:

- **anchor** (``u = 0``): a hard hold (contracting ``F``, no drift) — pulls ``x``
  back to the dock *regardless of the current*, so the two mode-hypotheses
  converge and the current stays unobservable. Safe, blind.
- **drift** (``u = 1``): let the current carry the boat (``b`` runs with the
  mode, ``F ≈ 1``) — the two mode-hypotheses separate in ``x``, so the next
  position reading is highly informative about the current. Informative, but off
  the dock.
- **run** (``u = 2``): motor downstream — the aggressive drift; more separation,
  more risk.

The epistemic value of the separated belief is the hybrid channel-keyed
``infogain`` leaf (D4): for the LG modality over the mixture-carried ``x``, the predictive
observation density is a K-component mixture, and ``I(x; y) = H[q(y)] - H[y|x]``
is a Kolchinsky-Tracey bound pair — the mutual information between the next
reading and the state, large exactly when the mode-branches are spread apart.
With it on, the planner values *drifting to read the current* and the agent
probes before it settles; ablate it (``epistemic=False``) and the
agent just anchors on tick 0, blind to the current. Filtering runs under
``mixture_schedule="mode_aligned"`` so mixture slots stay mode-owned: the mode
posterior is the mass of each mode's slot bank (``mode_evidence_message``, D3).

The whole agent is one declaration — a hidden ``mode`` factor (sticky, its own
parent) and a controlled mixture ``x`` factor riding it, an LG sensor on ``x``,
and the dock as a ``state_targets`` designation. ``main`` runs both agents
(probing vs ablation) from the same env and prints an honest per-tick table; an
exact-enumeration filter (no reduction) is the internal oracle.

Run: ``.venv/bin/python examples/switching_filter.py``
"""

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalGaussian,
    Gaussian,
    MixtureGaussian,
)
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianTransition
from hybrid_aif.inference.linear_gaussian import condition
from hybrid_aif.objectives import TermTree, total
from hybrid_aif.system import System, SystemParams

# --------------------------------------------------------------------------- model
# M = 2 hidden current modes, 1-D state x (boat position, dock at x = 0). Mode 0
# flows left, mode 1 right; the current is constant over an episode (a sticky,
# near-identity mode transition). Three controls on x, each with its own linear
# map F(u) and mode-conditioned drift b(mode, u): the mode-times-action structure
# (not a mere additive offset) is what lets an action change how *separable* — and
# so how informative — the two mode-hypotheses are.

NUM_MODES = 2
DRIFT = 1.1  # how far the current carries the boat per drift step
CURRENT = jnp.asarray([-1.0, +1.0])  # per-mode flow direction

# b[mode, action, d]: anchor holds (0), drift rides the current (±DRIFT), run
# motors downstream (±2·DRIFT).
_B_ACT = jnp.asarray([0.0, DRIFT, 2.0 * DRIFT])  # per-action drift magnitude
B_OFFSET = (CURRENT[:, None] * _B_ACT[None, :])[..., None]  # (mode=2, action=3, d=1)

# Per-mode-and-action linear map F(u): anchor contracts hard (holds the dock
# regardless of current), drift/run coast (F≈1). Shape (mode, action, d, d);
# F varies only by action here (broadcast across mode).
F_ROW = jnp.asarray([0.25, 0.85, 1.0])  # anchor, drift, run
F_MAP = jnp.broadcast_to(F_ROW[None, :, None, None], (NUM_MODES, 3, 1, 1))

# Innovation noise per action (mild) — the epistemic contrast comes from mode
# separation via F(u)/b(mode,u), not from a per-action noise lever.
INNOV_STD = jnp.asarray([0.2, 0.2, 0.3])  # anchor, drift, run
Q_CHOL = jnp.broadcast_to(INNOV_STD[None, :, None, None], (NUM_MODES, 3, 1, 1))

# Sticky hidden mode: near-identity self-transition, one trivial control column.
_STICK = jnp.log(jnp.asarray([[0.9975, 0.0025], [0.0025, 0.9975]]))  # (next, prev)
MODE_TABLE = _STICK[:, :, None]  # (mode_next, mode_prev, control=1)

# Noisy position sensor y = x + eps, eps ~ N(0, R).
OBS_H = jnp.asarray([[1.0]])
OBS_C = jnp.zeros(1)
OBS_R_CHOL = jnp.asarray([[0.3]])

# Preference: hold station at the dock x = 0, tightly. The width is deliberate:
# too wide and matching the target's spread would reward drifting even for the
# risk-only agent; this value makes anchoring the risk-optimal move, so the
# probing is unambiguously the info-gain term's doing.
TARGET = Gaussian(loc=jnp.zeros(1), chol=jnp.asarray([[0.55]]))

# Initial belief: uniform over the current mode, and a mode-aligned station guess —
# one component per mode bank, both at x = 0, so at t=0 the banks are identical
# and carry no spurious mode evidence (the current is not yet observable).
MODE_PRIOR = Categorical(logits=jnp.zeros(NUM_MODES))
X_PRIOR = MixtureGaussian(
    log_weights=jnp.log(jnp.asarray([0.5, 0.5])),
    means=jnp.asarray([[0.0], [0.0]]),
    chols=jnp.asarray([[[0.5]], [[0.5]]]),
)

HORIZON = 3
# Mode-aligned needs ~2x the runnalls-tuned budget (design doc Guidance): M=2 so
# use k=8. The exact fallback is K0 * M**H = 2 * 2**3 = 16 (genuine reduction at k=8).
K_BUDGET = 8
K_EXACT = X_PRIOR.num_components * NUM_MODES**HORIZON

ACTION_LABELS = ("anchor", "drift", "run")
PROBE_ACTIONS = (1, 2)  # drift / run: separate the mode-hypotheses (informative)
COMMIT_ACTION = 0  # anchor: hold the dock, blind to the current


# -------------------------------------------------------------------------- System


def make_agent(
    epistemic: bool = True,
    k: int = K_BUDGET,
    horizon: int = HORIZON,
    gamma: float = 8.0,
) -> tuple[System, SystemParams]:
    """Declare the switching-filter agent. ``epistemic`` is the ablation.

    Two factors: a hidden ``mode`` (categorical, its own sticky parent,
    uncontrolled) and the controlled mixture ``x`` riding it. One LG sensor
    ``y`` observes ``x``. The goal is a ``state_targets`` designation; the only
    epistemic leaf is the hybrid channel one, ``infogain["y"]`` — a factor carries
    no leaf of its own, so the contrast is a clean risk vs. risk+info-gain
    ablation.
    """
    system = System(
        A_deps=((1,),),  # y observes x
        B_deps=((0,), (0,)),  # mode is its own (sticky) parent; x rides the mode
        controlled=(False, True),
        policy_len=horizon,
        gamma=gamma,
        mixture_k=k,
        mixture_schedule="mode_aligned",
        epistemic=epistemic,
        modality_names=("y",),
        factor_names=("mode", "x"),
    )
    params = SystemParams(
        likelihood={"y": LinearGaussianObservation(H=OBS_H, c=OBS_C, noise=OBS_R_CHOL)},
        transition={
            "mode": ConditionalCategorical(log_table=MODE_TABLE, parent_ndims=2),
            "x": LinearGaussianTransition(
                F=F_MAP,
                innovations=ConditionalGaussian(loc=B_OFFSET, chol=Q_CHOL, parent_ndims=2),
            ),
        },
        initial_state_prior={"mode": MODE_PRIOR, "x": X_PRIOR},
        state_targets={"x": TARGET},
    )
    return system, params


# ---------------------------------------------------------------------- closed loop


@jax.jit
def tick(
    system: System,
    params: SystemParams,
    obs: dict[str, Array],
    priors: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, Array], TermTree]:
    """One agent tick: filter on ``obs``, plan and act, predict next priors.

    Returns ``(posteriors, next_priors, action, metrics_tree)``, keyed as the faces key
    them (the action ``{"a_x": level}``). Selection is the deterministic argmax-policy
    first step, so the probing is reproducible.
    """
    beliefs, _ = system.infer_states(params, obs, priors)
    action, _, tree = system.act(
        jax.random.PRNGKey(0), params, beliefs, selection="deterministic", mode="policy"
    )
    return beliefs, system.update_empirical_prior(params, beliefs, action), action, tree


def mode_entropy(mode_belief: Categorical) -> float:
    """Shannon entropy (nats) of the discrete mode posterior."""
    return float(mode_belief.entropy().midpoint)


class Trajectory(NamedTuple):
    """Closed-loop record, one row per tick (obs then action within a tick)."""

    xs: Array  # (T,) true state
    ys: Array  # (T,) observations
    mode_probs: Array  # (T, M) posterior P(mode) after filtering y_t
    x_means: Array  # (T,) posterior mean of x after filtering
    actions: Array  # (T,) action emitted from u_x
    mode_entropies: Array  # (T,) H[P(mode)] after filtering, nats


def closed_loop(
    key: Array,
    system: System,
    params: SystemParams,
    true_mode: int = 1,
    num_steps: int = 8,
) -> Trajectory:
    """Filter -> plan -> act for ``num_steps`` ticks under a pinned hidden wind.

    Ground truth is ``as_env``; the reset override pins the true mode
    (``true_mode``) and the start ``x = 0`` while the agent's belief stays the
    uniform-mode, station-guess prior.
    """
    env, env_params = system.as_env(params)
    key, k_reset = jax.random.split(key)
    obs, state = env.reset(
        k_reset, env_params, state={"mode": jnp.asarray(true_mode, jnp.int32), "x": jnp.zeros(1)}
    )
    priors = system.initial_priors(params)
    xs, ys, mode_probs, x_means, actions, ents = [], [], [], [], [], []
    for _ in range(num_steps):
        posteriors, priors, action, _ = tick(system, params, obs, priors)
        xs.append(float(state["x"][0]))
        ys.append(float(obs["y"][0]))
        mode_probs.append(np.asarray(posteriors["mode"].probs))
        x_means.append(float(posteriors["x"].mean()[0]))
        actions.append(int(action["a_x"]))
        ents.append(mode_entropy(posteriors["mode"]))
        key, k_env = jax.random.split(key)
        obs, state = env.step(k_env, state, action, env_params)
    return Trajectory(
        xs=np.asarray(xs),
        ys=np.asarray(ys),
        mode_probs=np.stack(mode_probs),
        x_means=np.asarray(x_means),
        actions=np.asarray(actions),
        mode_entropies=np.asarray(ents),
    )


# ------------------------------------------------------------------- exact oracle
# Parity strategy (D5): a fixed action stream is rolled once through the env to
# fix the observations, then filtered two ways — the reduced System pipeline
# (infer_states / update_empirical_prior at some budget k) and an independent
# textbook enumeration (grow through both modes, never merge) — and the belief
# trajectories are compared. At the exact-fallback budget K0*M^H the pipeline
# does no live merges over a short horizon, so the two agree to ~1e-6; at the
# chosen k=8 the gap is the honest reduction error (the fidelity check).


def env_obs_stream(
    key: Array, params: SystemParams, actions: np.ndarray, true_mode: int
) -> tuple[list[Array], np.ndarray]:
    """Roll a fixed action stream through ``as_env``; return the obs and true xs."""
    system, _ = make_agent()
    env, env_params = system.as_env(params)
    key, k_reset = jax.random.split(key)
    obs, state = env.reset(
        k_reset, env_params, state={"mode": jnp.asarray(true_mode, jnp.int32), "x": jnp.zeros(1)}
    )
    ys, xs = [], []
    for t in range(len(actions)):
        ys.append(obs["y"])
        xs.append(float(state["x"][0]))
        a = {"a_x": jnp.asarray(int(actions[t]), jnp.int32)}
        key, k_env = jax.random.split(key)
        obs, state = env.step(k_env, state, a, env_params)
    return ys, np.asarray(xs)


def replay_filter(
    params: SystemParams, actions: np.ndarray, ys: list[Array], k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Filter a fixed obs/action stream through the reduced System pipeline at budget ``k``.

    Uses the public faces only — ``infer_states`` (mode-aligned Gaussian-sum
    conditioning + mode-evidence message) then ``update_empirical_prior`` (grow
    -> reduce_mode_aligned) — so this exercises exactly the closed loop's
    machinery, just with the action stream pinned.
    """
    system, _ = make_agent(k=k)
    priors = system.initial_priors(params)
    mode_probs, x_means = [], []
    for t in range(len(actions)):
        post, _ = system.infer_states(params, {"y": ys[t]}, priors)
        mode_probs.append(np.asarray(post["mode"].probs))
        x_means.append(float(post["x"].mean()[0]))
        priors = system.update_empirical_prior(
            params, post, {"a_x": jnp.asarray(int(actions[t]), jnp.int32)}
        )
    return np.stack(mode_probs), np.asarray(x_means)


def _bank_layout(m: MixtureGaussian, num_modes: int) -> MixtureGaussian:
    """Reorder a component-major grown mixture (mode = slot % M) into contiguous banks."""
    k = m.num_components
    order = jnp.asarray(np.concatenate([np.arange(g, k, num_modes) for g in range(num_modes)]))
    return MixtureGaussian(
        log_weights=m.log_weights[order], means=m.means[order], chols=m.chols[order]
    )


def exact_filter(
    params: SystemParams, actions: np.ndarray, ys: list[Array]
) -> tuple[np.ndarray, np.ndarray]:
    """Textbook Gaussian-sum filter by enumeration — no reduction — the reference.

    Grows the x belief through both modes each step (K -> 2K, never merged),
    conditions on the real observation, and reads the mode posterior off the
    mode-owned banks. Deliberately runs the raw kernels (``condition`` /
    ``propagate``), not a big-budget System, so it never shares the reduced
    pipeline's reduce()/static-K-padding code path. Returns ``(mode_probs,
    x_means)`` per tick.
    """
    from hybrid_aif.inference.mixture_rollout import propagate

    x_bel = X_PRIOR
    mode_lp = MODE_PRIOR.log_probs
    mode_probs, x_means = [], []
    for t in range(len(actions)):
        # measurement update (exact Gaussian-sum), K unchanged
        post, log_marg = condition(x_bel.components, OBS_H, OBS_C, OBS_R_CHOL, ys[t][..., None, :])
        spb = x_bel.num_components // NUM_MODES  # slots per contiguous mode bank
        prior_lw = jax.nn.log_softmax(x_bel.log_weights)
        unnorm = prior_lw + log_marg
        x_bel = MixtureGaussian(
            log_weights=unnorm - jax.scipy.special.logsumexp(unnorm),
            means=post.loc,
            chols=post.chol,
        )
        # mode posterior from within-bank-normalized bank evidence
        within = jax.nn.log_softmax(prior_lw.reshape(NUM_MODES, spb), axis=-1)
        log_ev = jax.scipy.special.logsumexp(within + log_marg.reshape(NUM_MODES, spb), axis=-1)
        mode_lp = jax.nn.log_softmax(mode_lp + log_ev)  # mode posterior this tick
        mode_probs.append(np.asarray(jnp.exp(mode_lp)))
        x_means.append(float(x_bel.mean()[0]))
        # transition: grow x through both modes under the action, no reduction
        a = int(actions[t])
        Fs = jnp.take(F_MAP, a, axis=-3).reshape(NUM_MODES, 1, 1)
        loc = jnp.take(B_OFFSET, a, axis=-2).reshape(NUM_MODES, 1)
        chol = jnp.take(Q_CHOL, a, axis=-3).reshape(NUM_MODES, 1, 1)
        x_bel = _bank_layout(propagate(x_bel, Fs, loc, chol, mode_lp), NUM_MODES)
        # sticky mode transition for the next tick's prior (matches the pipeline's
        # update_empirical_prior, which pushes the mode through its own B).
        stick = jnp.exp(MODE_TABLE[..., 0])  # (next, prev)
        mode_lp = jnp.log(stick @ jnp.exp(mode_lp))
    return np.stack(mode_probs), np.asarray(x_means)


# --------------------------------------------------------------------------- table


def _first_probe_and_commit(actions: np.ndarray) -> tuple[int, int]:
    """(first tick the agent probes, first tick it anchors); -1 if never."""
    is_probe = np.isin(actions, np.asarray(PROBE_ACTIONS))
    probe = int(np.argmax(is_probe)) if is_probe.any() else -1
    commit = int(np.argmax(actions == COMMIT_ACTION)) if (actions == COMMIT_ACTION).any() else -1
    return probe, commit


def planning_fidelity(gamma: float = 8.0) -> tuple[bool, float]:
    """Ranking fidelity of the chosen k=8 vs the exact fallback k=K0*M^H (design doc D2).

    Scores the full enumerated policy set from the initial (uniform-mode) belief —
    where the belief stays genuinely bimodal so mode-aligned reduction bites at the
    last grow — at both budgets. Returns ``(rankings_agree, max|ΔG_midpoint|)``.
    """
    beliefs = {"mode": MODE_PRIOR, "x": X_PRIOR}
    g = {}
    for k in (K_BUDGET, K_EXACT):
        system, params = make_agent(k=k, gamma=gamma)
        tree = system.score_policies(params, beliefs, system.policies(params))
        g[k] = np.asarray(total(tree).midpoint)
    rank8 = np.argsort(np.argsort(g[K_BUDGET]))
    rank16 = np.argsort(np.argsort(g[K_EXACT]))
    return bool((rank8 == rank16).all()), float(np.max(np.abs(g[K_BUDGET] - g[K_EXACT])))


PARITY_STEPS = 4  # short horizon: at k=K_EXACT the pipeline does no live merges here


def oracle_report(
    key: Array, params: SystemParams, actions: np.ndarray, true_mode: int
) -> tuple[float, float, float]:
    """(exact-parity mode err, exact-parity x err, k=8 fidelity mode err) on a fixed stream."""
    acts = actions[:PARITY_STEPS]
    ys, _ = env_obs_stream(key, params, acts, true_mode)
    mode_exact, x_exact = exact_filter(params, acts, ys)
    mode_k16, x_k16 = replay_filter(params, acts, ys, k=K_EXACT)
    mode_k8, _ = replay_filter(params, acts, ys, k=K_BUDGET)
    parity_mode = float(np.max(np.abs(mode_k16 - mode_exact)))
    parity_x = float(np.max(np.abs(x_k16 - x_exact)))
    fidelity = float(np.max(np.abs(mode_k8 - mode_exact)))
    return parity_mode, parity_x, fidelity


def main() -> None:
    t0 = time.perf_counter()
    key = jax.random.PRNGKey(0)
    true_mode = 1  # the current flows right; the agent must discover this

    agent_ig, params_ig = make_agent(epistemic=True)
    agent_ab, params_ab = make_agent(epistemic=False)
    traj_ig = closed_loop(key, agent_ig, params_ig, true_mode=true_mode)
    traj_ab = closed_loop(key, agent_ab, params_ab, true_mode=true_mode)
    parity_mode, parity_x, fidelity = oracle_report(key, params_ig, traj_ig.actions, true_mode)
    ranks_agree, g_gap = planning_fidelity()
    elapsed = time.perf_counter() - t0

    print("Switching-filter demo: a hidden current inferred from a noisy position sensor")
    print(
        f"System(mode[hidden, sticky], x[mixture]) + LG sensor; mode_aligned "
        f"mixture_k={K_BUDGET} (exact fallback K0*M^H={K_EXACT}); planning H={HORIZON}"
    )
    print(
        f"true mode = {true_mode} (current flows {'right' if true_mode else 'left'}); "
        "the agent starts uniform over the current\n"
    )

    header = f"{'tick':<5}{'y':>7}{'  info-gain: action':>21}{'P(m1)':>8}{'H(mode)':>9}"
    header += f"{'  ||   ablation: action':>24}{'P(m1)':>8}{'H(mode)':>9}"
    print(header)
    print("-" * len(header))
    for t in range(len(traj_ig.actions)):
        print(
            f"{t:<5}{traj_ig.ys[t]:>7.2f}"
            f"{ACTION_LABELS[traj_ig.actions[t]]:>21}"
            f"{traj_ig.mode_probs[t, 1]:>8.3f}{traj_ig.mode_entropies[t]:>9.3f}"
            f"{'  ||   ' + ACTION_LABELS[traj_ab.actions[t]]:>24}"
            f"{traj_ab.mode_probs[t, 1]:>8.3f}{traj_ab.mode_entropies[t]:>9.3f}"
        )

    probe_ig, commit_ig = _first_probe_and_commit(traj_ig.actions)
    probe_ab, commit_ab = _first_probe_and_commit(traj_ab.actions)
    # entropy first crossing below half a bit (0.35 nats): "disambiguated"
    thr = 0.35

    def first_below(ent: np.ndarray) -> int:
        return int(np.argmax(ent < thr)) if (ent < thr).any() else -1

    probes_first = probe_ig == 0 and (commit_ig == -1 or probe_ig < commit_ig)
    print("\nprobing (info gain on):")
    print(
        f"  first probe (drift/run) at tick {probe_ig}; first anchor at tick {commit_ig}"
        f"  ->  probes before settling: {probes_first}"
    )
    print(
        f"  mode entropy {traj_ig.mode_entropies[0]:.3f} -> {traj_ig.mode_entropies[-1]:.3f} nats; "
        f"disambiguated (H<{thr}) by tick {first_below(traj_ig.mode_entropies)}"
    )
    print("ablation (info gain off):")
    print(
        f"  first probe (drift/run) at tick {probe_ab}; first anchor at tick {commit_ab}"
        f"  ->  anchors on tick 0 (no probe): {commit_ab == 0 and probe_ab != 0}"
    )
    print(
        f"  mode entropy {traj_ab.mode_entropies[0]:.3f} -> {traj_ab.mode_entropies[-1]:.3f} nats; "
        f"disambiguated (H<{thr}) by tick {first_below(traj_ab.mode_entropies)}"
    )
    print(
        f"\noracle parity (reduced pipeline at k={K_EXACT} vs exact enumeration, "
        f"{PARITY_STEPS} ticks): max |ΔP(mode)| = {parity_mode:.2e}, "
        f"max |Δ x-mean| = {parity_x:.2e}"
    )
    print(
        f"k={K_BUDGET} filtering fidelity vs exact (reduction error): "
        f"max |ΔP(mode)| = {fidelity:.2e}"
    )
    print(
        f"k={K_BUDGET} planning fidelity vs exact fallback k={K_EXACT} (bimodal belief, "
        f"reduction active): rankings agree = {ranks_agree}, max |ΔG| = {g_gap:.2e} nats"
    )
    print(f"elapsed (jit compile + eval, both agents + oracle): {elapsed:.2f}s")


if __name__ == "__main__":
    main()
