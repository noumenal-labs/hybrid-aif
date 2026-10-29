"""Hill climb, linear-Gaussian: the continuous closed loop as one hybrid System.

A 1D "hill" double-integrator — a mountain-car-class task in the linear regime.
State ``x = (position, velocity)``; the goal is the hilltop at ``p = 1`` where
an anti-spring gravity force ``slope * (p - 1)`` vanishes (unstable
equilibrium: gravity pulls the cart back down the slope everywhere below the
top). Control is a small discrete thrust set realized as a
:class:`ConditionalGaussian` innovation family ``N(b_j, Q_j)`` over the shared
linear dynamics ``x' = F x + b_j + eps`` — higher thrust is noisier, so the
per-mode ``Q_j`` genuinely differ and the belief's entropy has something to say.

Everything is one declaration (``docs/design/hybrid-system.md``): the dynamics
are a :class:`LinearGaussianTransition` and the sensor a
:class:`LinearGaussianObservation` in :class:`SystemParams`; the Gaussian goal
is a ``state_targets`` designation, the price of the uncertainty a plan injects
the ``uncertainty["state"]`` row (``terms=("uncertainty:state",)``), and the
information the expected position reading carries about the state the
``infogain["position"]`` leaf, mounted by the per-variable ``epistemic`` selector.
A tick runs the System's faces (``infer_states`` -> ``act`` ->
``update_empirical_prior``), and the ground truth is the same declaration run
forward via ``as_env``.

Run: ``python examples/hill_lg.py`` — filters, plans, reaches the goal region,
and prints the EFE metrics term tree from the first planning call.
"""

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hybrid_aif.distributions import ConditionalGaussian, Gaussian
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianTransition
from hybrid_aif.objectives import TermTree, total
from hybrid_aif.system import System, SystemParams

DT = 0.1  # integration step
SLOPE = 0.5  # gravity gain of the anti-spring force slope*(p - GOAL_POS)
GOAL_POS = 1.0  # hilltop position
THRUSTS = (-1.0, 0.0, 1.0)  # discrete thrust levels (the control set)
GOAL_TOL = (0.1, 0.25)  # goal region half-widths: |p - 1|, |v|


def make_agent(
    dt: float = DT,
    slope: float = SLOPE,
    thrusts: tuple[float, ...] = THRUSTS,
    obs_std: float = 0.05,
    policy_len: int = 4,
    gamma: float = 16.0,
    include_entropy: bool = True,
    epistemic: bool = True,
) -> tuple[System, SystemParams]:
    """Declare the hill System. Force on the cart: ``slope*(p - GOAL_POS) + u``.

    The linear-in-p part of gravity lives in ``F``; the constant part and the
    thrust fold into the per-mode innovation offset ``b_j``. Process noise
    scales with |thrust| (engine vibration), so the innovation family carries
    per-mode Cholesky factors, not one shared Q. ``D`` is the agent's prior
    (and ``as_env``'s default start — ``closed_loop`` overrides it with the
    exact point ``(0, 0)``); the goal is a ``state_targets`` designation on
    the single factor. ``include_entropy`` mounts the ``uncertainty`` row on the
    state, ``epistemic`` the position reading's information-gain leaf.
    """
    F = jnp.array([[1.0, dt], [slope * dt, 1.0]])
    b = jnp.stack([jnp.array([0.0, dt * (u - slope * GOAL_POS)]) for u in thrusts])
    Q_chol = jnp.stack([jnp.diag(jnp.array([1e-3, 5e-3 + 5e-3 * abs(u)])) for u in thrusts])
    system = System(
        A_deps=((0,),),
        B_deps=((),),
        controlled=(True,),
        policy_len=policy_len,
        gamma=gamma,
        epistemic=epistemic,
        terms=("uncertainty:state",) if include_entropy else (),
        modality_names=("position",),
        factor_names=("state",),
    )
    params = SystemParams(
        likelihood={
            "position": LinearGaussianObservation(
                H=jnp.array([[1.0, 0.0]]), c=jnp.zeros(1), noise=jnp.array([[obs_std]])
            ),
        },
        transition={
            "state": LinearGaussianTransition(
                F=F, innovations=ConditionalGaussian(loc=b, chol=Q_chol, parent_ndims=1)
            ),
        },
        initial_state_prior={"state": Gaussian(loc=jnp.zeros(2), chol=0.2 * jnp.eye(2))},
        state_targets={
            "state": Gaussian(
                loc=jnp.array([GOAL_POS, 0.0]), chol=jnp.diag(jnp.array([0.08, 0.25]))
            )
        },
    )
    return system, params


@jax.jit
def tick(
    system: System, params: SystemParams, obs: dict[str, Array], priors: dict[str, Gaussian]
) -> tuple[dict[str, Gaussian], dict[str, Gaussian], dict[str, Array], TermTree]:
    """One agent tick, three faces: filter on ``obs``, plan and act, predict.

    Returns ``(posteriors, next_priors, action, metrics_tree)``, keyed as the faces key them;
    the action ``{"a_state": level}`` is the deterministic argmax-policy first step (an int32
    thrust index). ``system`` holds only static structure, so jit specializes on it.
    """
    beliefs, _ = system.infer_states(params, obs, priors)
    action, _, tree = system.act(
        jax.random.PRNGKey(0), params, beliefs, selection="deterministic", mode="policy"
    )
    return beliefs, system.update_empirical_prior(params, beliefs, action), action, tree


def in_goal_region(x: Array) -> Array:
    """Whether a true state sits in the goal region at the hilltop."""
    return (jnp.abs(x[..., 0] - GOAL_POS) <= GOAL_TOL[0]) & (jnp.abs(x[..., 1]) <= GOAL_TOL[1])


class Trajectory(NamedTuple):
    """Closed-loop record, one row per tick (obs then action within a tick)."""

    xs: Array  # (T, 2) true states at each tick
    ys: Array  # (T, 1) observations
    means: Array  # (T, 2) posterior belief means (after conditioning on y_t)
    actions: Array  # (T,) chosen thrust indices
    first_tree: TermTree  # metrics term tree from the first planning call


def closed_loop(
    key: Array, system: System, params: SystemParams, num_steps: int = 60
) -> Trajectory:
    """Filter -> plan -> act for ``num_steps`` ticks from rest in the valley.

    Ground truth is ``as_env``; the reset override pins the true start at
    exactly ``(0, 0)`` while the agent's belief stays the uncertain prior ``D``.
    """
    env, env_params = system.as_env(params)
    key, k_reset = jax.random.split(key)
    obs, state = env.reset(k_reset, env_params, state={"state": jnp.zeros(2)})
    priors = system.initial_priors(params)
    xs, ys, means, actions = [], [], [], []
    first_tree = None
    for _ in range(num_steps):
        posteriors, priors, action, tree = tick(system, params, obs, priors)
        if first_tree is None:
            first_tree = tree
        xs.append(state["state"])
        ys.append(obs["position"])
        means.append(posteriors["state"].loc)
        actions.append(action["a_state"])
        key, k_env = jax.random.split(key)
        obs, state = env.step(k_env, state, action, env_params)
    return Trajectory(
        xs=jnp.stack(xs),
        ys=jnp.stack(ys),
        means=jnp.stack(means),
        actions=jnp.stack(actions),
        first_tree=first_tree,
    )


def print_metrics(tree: TermTree) -> None:
    """Print a term tree once: per-term/block leaf summaries plus the folded G."""
    print("EFE metrics term tree (per-policy leaves):")
    for term in sorted(tree):
        for block in sorted(tree[term]):
            iv = tree[term][block]
            mid, width = np.asarray(iv.midpoint), np.asarray(iv.width)
            print(
                f"  {term}/{block}: shape {mid.shape}, "
                f"min {mid.min():.3f} @ policy {mid.argmin()}, "
                f"max {mid.max():.3f}, max interval width {width.max():.2e}"
            )
    g = total(tree)
    gm = np.asarray(g.midpoint)
    print(f"  G (folded): min {gm.min():.3f} @ policy {gm.argmin()}, max {gm.max():.3f}")


def main() -> None:
    system, params = make_agent()
    start = time.perf_counter()
    traj = closed_loop(jax.random.PRNGKey(0), system, params)
    elapsed = time.perf_counter() - start

    print_metrics(traj.first_tree)
    print()
    print("closed loop (noisy position obs -> filter -> enumeration planner -> thrust):")
    for t in range(0, len(traj.actions), 5):
        x, m, u = traj.xs[t], traj.means[t], THRUSTS[int(traj.actions[t])]
        print(
            f"  t={t:3d}  p={float(x[0]):+.3f} v={float(x[1]):+.3f}  "
            f"belief p={float(m[0]):+.3f} v={float(m[1]):+.3f}  thrust={u:+.1f}"
        )
    reached = np.asarray(jax.vmap(in_goal_region)(traj.xs))
    first_hit = int(reached.argmax()) if reached.any() else None
    final = traj.xs[-1]
    print()
    print(f"goal region |p-{GOAL_POS}|<={GOAL_TOL[0]}, |v|<={GOAL_TOL[1]}")
    print(
        f"first reached at tick {first_hit}; final state p={float(final[0]):+.3f} "
        f"v={float(final[1]):+.3f}; in goal at end: {bool(reached[-1])}"
    )
    mse = float(jnp.mean((traj.means[:, 0] - traj.xs[:, 0]) ** 2))
    print(f"filter position MSE vs truth: {mse:.2e} (obs noise var {0.05**2:.2e})")
    print(f"runtime: {elapsed:.2f}s for {len(traj.actions)} ticks (incl. jit compile)")


if __name__ == "__main__":
    main()
