"""Linear-quadratic control as active inference: a continuous control node through ``G u``.

A double integrator ``x = (position, velocity)`` moves by ``x' = F x + G u + ε``, its control a
continuous node ``u_u`` with action ``u``, declared natively (``controls=``, ``action_names=``),
read by the linear-Gaussian transition through its ``G u`` term, with its box in
``params.control``. The agent observes ``y = x + noise`` and prefers observations near a goal (a
Gaussian ``obs_targets`` entry) and small pushes (a Gaussian ``state_targets`` entry on the node
itself, a target on the control).

On this path the belief stays Gaussian and its covariance does not depend on the plan, so the
expected free energy splits into a quadratic in the plan (the utility's mean term and the
control target) and terms that do not move with it (the utility's trace term, the info gain).
Its minimiser is the linear-quadratic regulator's plan. The gradient planner descends ``G`` to it;
CEM samples its way close; the Laplace posterior around the plan is the inverse curvature of the
quadratic. ``tests/test_lq_control.py`` checks the plan against the finite-horizon Riccati
recursion.

Run: ``python examples/lq_control.py`` — plans with the gradient planner and with CEM, prints both
plans, their ``G`` and the Laplace standard deviations.
"""

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_aif.controls import Box
from hybrid_aif.distributions import ConditionalGaussian, Gaussian
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianTransition
from hybrid_aif.objectives import total
from hybrid_aif.planning import cem_plan, gradient_plan
from hybrid_aif.system import System, SystemParams

DT = 0.5  # integration step
HORIZON = 4
GOAL = (1.0, 0.0)  # the preferred observation: at position 1, at rest
GOAL_SD = (0.5, 1.0)  # its tolerance per observed dimension
CONTROL_TARGET_SD = 1.0  # the control target's scale: C(u) = N(0, CONTROL_TARGET_SD²)
BOX = 10.0  # the control's support [-BOX, BOX]: wide, so the optimum is interior


def make_lq(
    *,
    dt: float = DT,
    horizon: int = HORIZON,
    goal: tuple[float, float] = GOAL,
    goal_sd: tuple[float, float] = GOAL_SD,
    control_target_sd: float = CONTROL_TARGET_SD,
    box: float = BOX,
    process_sd: float = 0.05,
    obs_sd: float = 0.1,
    gamma: float = 1.0,
    planner: str = "gradient",
) -> tuple[System, SystemParams]:
    """The LQ System and its params.

    Nodes: the state ``x`` (2-D) and the control ``u_u`` (1-D, action ``u``), ``x`` reading
    ``u_u`` as a transition parent. ``gamma`` is the policy precision (1: the Laplace covariance
    is the inverse curvature of ``G`` itself).
    """
    system = System(
        A_deps=((0,),),
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("u",),
        factor_names=("x", "u_u"),
        modality_names=("y",),
        policy_len=horizon,
        planner=planner,
        gamma=gamma,
        epistemic=True,
    )
    F = jnp.asarray([[1.0, dt], [0.0, 1.0]])
    G = jnp.asarray([[0.5 * dt**2], [dt]])
    params = SystemParams(
        transition={
            "x": LinearGaussianTransition(
                F=F,
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 2)),
                    chol=process_sd * jnp.eye(2)[None],
                    parent_ndims=1,
                ),
                G=G,
            )
        },
        initial_state_prior={"x": Gaussian(loc=jnp.zeros(2), chol=0.1 * jnp.eye(2))},
        likelihood={
            "y": LinearGaussianObservation(H=jnp.eye(2), c=jnp.zeros(2), noise=obs_sd * jnp.eye(2))
        },
        obs_targets={"y": Gaussian(loc=jnp.asarray(goal), chol=jnp.diag(jnp.asarray(goal_sd)))},
        state_targets={"u_u": Gaussian(loc=jnp.zeros(1), chol=control_target_sd * jnp.eye(1))},
        control={"u_u": Box(-box * jnp.ones(1), box * jnp.ones(1))},
    )
    return system, params


def main() -> None:
    system, params = make_lq()
    beliefs = system.initial_priors(params)
    ctx = system.planner_context(params, beliefs)
    grad = gradient_plan(
        ctx, optimizer="gradient", lr=0.1, steps=400, posterior="laplace", metrics=True
    )
    cem = cem_plan(ctx, jax.random.PRNGKey(0), num_candidates=256, cem_iters=12, min_std=1e-3)
    G_grad = float(total(system.score_policies(params, beliefs, grad.plan)).midpoint)
    G_cem = float(total(system.score_policies(params, beliefs, cem.mean)).midpoint)
    sd = np.sqrt(np.diag(np.asarray(grad.curvature.cov)))
    print("gradient plan u_t:", np.round(np.asarray(grad.plan[:, 0]), 4), f"G = {G_grad:.5f}")
    print("CEM plan      u_t:", np.round(np.asarray(cem.mean[:, 0]), 4), f"G = {G_cem:.5f}")
    print("Laplace sd    u_t:", np.round(sd, 4))
    action, _, _ = system.act(jax.random.PRNGKey(1), params, beliefs, selection="deterministic")
    print("act (the mode's first step):", np.asarray(action["u"]))


if __name__ == "__main__":
    main()
