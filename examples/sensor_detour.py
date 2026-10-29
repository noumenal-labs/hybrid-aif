"""A rover that plans a detour because it expects to see better there.

A rover drives a 2-D position from a start waypoint to a dock straight ahead, under a velocity
control in a box. Its position fix is precise only near a survey beacon off to one side of the
straight route: the fix's noise is a :class:`~hybrid_aif.factors.GridNoiseMap`, a standard
deviation read off a smooth grid over the position (:func:`map_sigmas` gives the values the
map takes far from the beacon, where the straight route passes closest, and at the beacon).
The gradient planner chooses an open-loop plan of eight velocity steps.

The same model is planned under two objectives. One expects no observation: under
``planning_belief="prior"`` it charges the entropy of the predictive belief (the ``uncertainty``
row, ``terms=("uncertainty:position",)``) and carries no information gain, so the map cannot
enter it and it cannot prefer one route. The other expects to observe: under
``planning_belief="posterior"`` the planner carries a second belief, conditioned on the fix
expected at each step with the noise the map gives at that step's mean, and the fix's
information-gain leaf is the information that step's fix adds to the plan's earlier ones, so a
route past the beacon promises a sharper belief and the plan bends toward it before turning for
the dock. (The fix's information gain under ``"prior"``, the default, reads the map as well, at
the predictive belief: it counts each step's fix as though no earlier one had been taken.)

:func:`compare` returns both plans, the tracks they drive, the uncertainty each leaves after
each step's expected fix, and each plan's expected free energy under both objectives. Run as a
script it prints a summary.

Design: docs/design/planning-with-observations.md (D1–D3)
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from hybrid_aif.controls import Box
from hybrid_aif.distributions import ConditionalGaussian, Gaussian
from hybrid_aif.factors import GridNoiseMap, LinearGaussianObservation, LinearGaussianTransition
from hybrid_aif.objectives import total
from hybrid_aif.objectives.hybrid import BeliefPair
from hybrid_aif.planning.gradient import gradient_plan
from hybrid_aif.system import System, SystemParams

DT = 0.25
START = np.array([-1.0, 0.0])
DOCK = np.array([1.0, 0.0])
BEACON = np.array([0.0, 0.6])
SD_COARSE = 0.6  # the grid's value far from the beacon
SD_WELL = 0.57  # the depth of the well of precision the grid samples around the beacon
POLICY_LEN = 8
EXTENT = (-1.0, 1.0)
NODES = 9
#: Adam steps per plan: enough that the plan's printed score is stable under a small perturbation
#: of the starting plan (it is still descending in the fourth decimal: the descent is slow along
#: the stretch at the beacon)
PLAN_STEPS = 10_000


def noise_sd(nodes: int = NODES) -> tuple[np.ndarray, np.ndarray]:
    """The fix's standard deviation on the grid: coarse, with a smooth well of precision around
    the beacon (wide enough that the planner feels its slope from the straight route).
    Returns the node coordinates and the ``(nodes, nodes)`` grid, indexed ``[x, y]``."""
    xs = np.linspace(*EXTENT, nodes)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    well = np.exp(-((gx - BEACON[0]) ** 2 + (gy - BEACON[1]) ** 2) / 0.25)
    return xs, SD_COARSE - SD_WELL * well


def sensor_map() -> GridNoiseMap:
    """The fix's noise as a map over the position: an isotropic ``R_chol`` at every node."""
    _, sd = noise_sd()
    values = np.zeros((*sd.shape, 2, 2), np.float32)
    values[..., 0, 0] = sd
    values[..., 1, 1] = sd
    return GridNoiseMap(jnp.asarray(values), (EXTENT[0],) * 2, (EXTENT[1],) * 2, axes=(0, 1))


def sigma_at(points) -> np.ndarray:
    """The fix's standard deviation the map gives at ``points`` (``(..., 2)``), as the planner
    reads it: interpolated between the grid's nodes, held at the edge beyond them."""
    chols = sensor_map()(jnp.asarray(points, jnp.float32).reshape(-1, 2))
    return np.asarray(chols[:, 0, 0]).reshape(np.shape(points)[:-1])


def map_sigmas() -> dict[str, float]:
    """The map's σ far from the beacon (a corner), where the straight route passes closest to
    the beacon (``(0, 0)``), and at the beacon."""
    far, route, beacon = sigma_at(np.array([[-1.0, -1.0], [0.0, 0.0], BEACON]))
    return {"far": float(far), "route": float(route), "beacon": float(beacon)}


def make_agent(
    planning_belief: str = "posterior",
    epistemic: bool = True,
    policy_len: int = POLICY_LEN,
    uncertainty: bool = False,
) -> tuple[System, SystemParams]:
    """The rover: a position under a velocity control in ``[-2, 2]²``, observed through the
    map, with a terminal target at the dock. ``epistemic`` mounts the fix's information-gain
    leaf, ``uncertainty`` the position's ``uncertainty`` row."""
    system = System(
        A_deps=((0,),),
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("velocity",),
        modality_names=("fix",),
        factor_names=("position", "u"),
        policy_len=policy_len,
        planner="gradient",
        epistemic=epistemic,
        planning_belief=planning_belief,
        terms=("uncertainty:position",) if uncertainty else (),
    )
    params = SystemParams(
        likelihood={
            "fix": LinearGaussianObservation(H=jnp.eye(2), c=jnp.zeros(2), noise=sensor_map())
        },
        transition={
            "position": LinearGaussianTransition(
                F=jnp.eye(2),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 2)),
                    chol=jnp.broadcast_to(0.02 * jnp.eye(2), (1, 2, 2)),
                    parent_ndims=1,
                ),
                G=DT * jnp.eye(2),
            )
        },
        initial_state_prior={"position": Gaussian(loc=jnp.asarray(START), chol=0.1 * jnp.eye(2))},
        control={"u": Box(-2.0 * jnp.ones(2), 2.0 * jnp.ones(2))},
        terminal_targets={"position": Gaussian(loc=jnp.asarray(DOCK), chol=0.1 * jnp.eye(2))},
    )
    return system, params


def _context(system: System, params: SystemParams):
    return system.planner_context(params, system.initial_priors(params))


def choose_plan(
    system: System, params: SystemParams, steps: int = PLAN_STEPS, init=None
) -> np.ndarray:
    """The gradient planner's plan, ``(policy_len, 2)`` velocities, descended from ``init``
    (default: the box midpoint, zero velocity)."""
    return np.asarray(gradient_plan(_context(system, params), steps=steps, init=init).plan)


def rollout(system: System, params: SystemParams, plan) -> tuple[np.ndarray, np.ndarray]:
    """The predicted position at each step of ``plan`` and the standard deviation of the
    belief the rover expects to hold after that step's fix: ``(T, 2)`` and ``(T,)``. Read
    under ``planning_belief="posterior"``, whose conditioned belief is that expected posterior;
    under ``"prior"`` the second array is the predictive belief's."""
    ctx = _context(system, params)
    carry = ctx.init_beliefs
    means, sds = [], []
    for u in jnp.asarray(plan):
        carry = ctx.transition_fn(carry, u)
        q = carry[0]  # a BeliefPair under "posterior": the class is not re-exported
        held = q.conditioned if isinstance(q, BeliefPair) else q
        predictive = q.predictive if isinstance(q, BeliefPair) else q
        means.append(np.asarray(predictive.loc))
        sds.append(float(np.sqrt(np.mean(np.diag(np.asarray(held.cov))))))
    return np.asarray(means), np.asarray(sds)


def expected_free_energy(system: System, params: SystemParams, plan) -> float:
    """``G`` of ``plan`` under ``system``'s objective (the midpoint of its interval)."""
    tree = system.score_policies(params, system.initial_priors(params), jnp.asarray(plan))
    return float(total(tree).midpoint)


class Comparison(NamedTuple):
    plans: dict[str, np.ndarray]  # objective -> (T, 2) velocities
    tracks: dict[str, np.ndarray]  # objective -> (T + 1, 2) predicted positions
    sds: dict[str, np.ndarray]  # objective -> (T + 1,) expected posterior sd
    G: dict[tuple[str, str], float]  # (scored under, plan chosen under) -> G


def agents() -> dict[str, tuple[System, SystemParams]]:
    """The two objectives :func:`compare` plans under, by planning belief: ``"prior"`` charges
    the predictive belief's entropy and expects no observation, ``"posterior"`` carries the fix's
    information gain on the conditioned belief."""
    return {
        "prior": make_agent("prior", epistemic=False, uncertainty=True),
        "posterior": make_agent("posterior"),
    }


def compare(steps: int = PLAN_STEPS) -> Comparison:
    """Plan under each objective, then read both plans under the posterior belief (the
    uncertainty each leaves) and score both plans under both objectives."""
    beliefs = ("prior", "posterior")
    agents_ = agents()
    plans = {b: choose_plan(*agents_[b], steps=steps) for b in beliefs}
    tracks, sds = {}, {}
    for b in beliefs:
        means, sd = rollout(*agents_["posterior"], plans[b])
        tracks[b] = np.vstack([START, means])
        sds[b] = np.concatenate([[0.1], sd])
    G = {(s, b): expected_free_energy(*agents_[s], plans[b]) for s in beliefs for b in beliefs}
    return Comparison(plans, tracks, sds, G)


def main() -> None:
    sig = map_sigmas()
    print(
        f"fix σ: {sig['far']:.2f} far from the beacon, {sig['route']:.2f} where the straight "
        f"route passes closest, {sig['beacon']:.3f} at the beacon"
    )
    result = compare()
    for b in ("prior", "posterior"):
        track = result.tracks[b]
        print(
            f"planned under {b:>9}: farthest from the straight route {track[:, 1].max():.2f}, "
            f"ends at ({track[-1, 0]:.2f}, {track[-1, 1]:.2f}), expected sd at the dock "
            f"{result.sds[b][-1]:.3f}"
        )
    print("G of each plan (rows: scored under; columns: planned under prior, posterior)")
    for s in ("prior", "posterior"):
        row = [result.G[(s, b)] for b in ("prior", "posterior")]
        print(f"  {s:>9}: {row[0]:9.3f} {row[1]:9.3f}")


if __name__ == "__main__":
    main()
