"""P1: state-based Mountain Car by search over a world model learned online.

The build plan's P1 (``docs/design/amortization.md``): "classical end on
state-based Mountain Car — A1(a) or mixture world model + A9 CEM +
designations; sparse-reward MC solved by search. (A2 lands here for world-model
learning.)" Every piece is a declaration or an injected planner; this file wires
nothing by hand.

- **World model**: one :class:`~hybrid_aif.distributions.MLPConditional` in the
  ``transition`` slot — ``(position, velocity) | previous state, action`` — the
  family dial's third notch (A1), predicted through by A1(a)'s linearized
  (EKF) row and learned by A2's gradient rule *online*, one adam step per agent
  step, with the :class:`~hybrid_aif.system.LearnState` carry threaded through
  the episode scan.
- **Observation**: the env hands over the state itself (addendum §1's ``state``
  key), so the likelihood is a :class:`LinearGaussianObservation` with ``H = I``
  and a tiny ``R`` — a near-exact data clamp. A *neural* likelihood is legal to
  declare but not to filter through (A1's fence: ``infer_states`` raises), and
  P1 does not need one.
- **Planner**: A9's CEM over continuous action sequences, with the
  receding-horizon convention — replan every tick, act on the refit proposal
  mean's first step.
- **Preference**: a ``state_targets`` designation, below.

**Their reward vs. our preference, exactly.** Their score is sparse: ``-1`` per
raw env step, ``0`` on the terminating step, ``position >= 0.45`` (addendum §1),
which is "minimize the number of steps until you reach the flag". Our agent does
not see a reward at all — it sees a preference designation
``state_targets["state"] = N([0.5, 0], diag([0.1, 10]))``: position concentrated
just past the flag, velocity left near-free (σ 10 — a target that
*preferred* zero velocity would forbid the swing the task requires). The per-step
risk ``KL(q(s_t) ‖ C)`` accumulated over the horizon is our analogue of their
per-step ``-1``: both are minimized by arriving sooner and staying.

The difference: a Gaussian target is **dense in position**, where their
reward is flat until the flag. So P1 tests A1+A2+A9 against Mountain Car's
*deception* (you must drive away from the goal to reach it) rather than against
reward sparsity — the flat-target variant is a designation change, not a code
change. Reported scores are
always **their** reward (ACR) and **their** success criterion, computed from the
env, never from the preference — so the numbers stay comparable to the oracle
run §7's P1 row asks for.

**Exploration.** There is no epistemic term here: ``epistemic=False``, because a
neural slot has no novelty estimator in the registry (A10 disposition 3 charters
``ensemble_disagreement`` for that) and the continuous state-entropy leaf rewards
*certainty*, which is not exploration. What drives search into new states is the
classical mechanism CEM already has: the ``min_std`` floor under the refit
proposal (see :mod:`hybrid_aif.planning.cem`) plus the fresh net's own wide,
wrong predictions early on. That is a **proposal-distribution** floor — it moves
where the search looks, never what G says about what it finds; no term is added
to the objective and no reward is shaped.

*Measured, and it is not enough.* Tripling the floor (0.2 → 0.6 action units,
150 episodes) moved the success rate from 0.13 to 0.17 — inside the noise. The
reason is visible in the logs: the one-step model error converges to ~1e-3 **on
the trajectories the agent actually takes**, which are the ones that hover in the
valley; the swing the task needs is off-distribution, and no amount of
*action*-level jitter makes a closed loop that keeps returning to the same basin
cover the state space. Wider action noise is the wrong instrument for a
state-coverage problem. The instruments that are right are both chartered and
both out of P1: replay-driven world-model fitting (A6's ``fit_world`` — R-AIF
trains from a 5e5-step buffer, not from the current tick) and a parameter-novelty
drive that pays for visiting states the model is unsure about (A10 disposition 3's
``ensemble_disagreement``).

**Protocol: defaults-first, tune-and-report**.
P1's acceptance is *matching an oracle number*, and a match produced by silent
knob-fitting claims nothing — so the classical end's hyperparameters are treated
as its design space, not as free parameters. Every knob below is fixed *before*
the reported run, by library default or by a stated principle, and any knob
changed after a full run is logged below (knob, old → new, why, which run
motivated it). The ideal knobs-touched list is empty.

**Defaults, and why each one is what it is:**

======================  =========  ==========================================
Knob                    Value      Rationale
======================  =========  ==========================================
MLP width / depth       96 / 2     library default shape (``from_shapes``
                                   defaults are 64/2); 96 is the one
                                   round step up for a 3→2 map
activation              tanh       ``MLPConditional`` default
``init_scale``          0.05       the fresh model's predictive sd: one
                                   scaled-velocity unit's worth of "I do not
                                   know yet", so early plans are visibly
                                   uncertain rather than confidently wrong
``min_scale``           1e-3       ``MLPConditional`` default (untouched)
state units             v / 0.07   the two output dimensions of one head must
                                   be commensurate; ``max_speed`` is the
                                   natural velocity unit, and the *sensor
                                   map* is where units are declared
observation ``R``       1e-3 I     the addendum's observation is exact; this
                                   is "exact up to float", not a tuned filter
CEM horizon             40         the horizon must span the *task*, not just
                                   a swing: their ACR −68.3 ⇒ ≈34 agent steps
                                   to the flag, so a plan shorter than that
                                   cannot see the goal at all (revised from
                                   20 — see the knob log below)
CEM candidates/iters    64 / 4     CEM's own conventional defaults (and this
                                   module's registry defaults)
``elite_frac``          0.125      = 8 of 64 elites, the CEM literature's
                                   usual tenth-ish
proposal init / floor   0.5 / 0.1  of the box width — :mod:`hybrid_aif.
                                   planning.cem`'s stated defaults, i.e. the
                                   exploration floor is *not* task-tuned
preference target       N([0.5,0], position sd 0.1 ≈ the flag's own width;
                        diag(0.1,   velocity sd 10 = "no preference", since a
                        10))       target over velocity would forbid the swing
``gamma``               4.0        only sharpens ``q_pi``; the reported action
                                   is the proposal mean, so it is inert here
model lr                3e-3 →     Robbins–Monro: single-sample stochastic
                        3e-4       approximation converges only with a
                                   decaying rate, so the rate decays
                                   geometrically over the episode budget
                                   (the rate grammar's documented curriculum
                                   form, A2)
======================  =========  ==========================================

**Knob log** (the protocol's audit trail; each entry names the run that motivated
the change):

1. *state units*: raw ``velocity`` → ``velocity / 0.07``, and the initial prior's
   velocity sd 0.01 → 0.1 with it. Motivated by an open-loop model-quality
   diagnostic (20-step rollout after 100 episodes of online learning): the
   velocity prediction error was ≈0.1, i.e. **larger than velocity's whole
   range**, because one Gaussian head cannot serve two dimensions three orders of
   magnitude apart in scale. Measured on model error, not on task score.
2. *CEM horizon*: 20 → 40. Motivated by a **frozen-model ablation** — train the
   world model offline to convergence (one-step error 5.4e-3 position, 8.3e-4
   velocity; 20-step rollout error 0.012), then run the controller with it
   frozen. At H=20: SR 0.00, every episode ending near p=−0.2. At H=40: SR 1.00,
   ACR −90. The horizon, not the learner, was the binding constraint. This one
   *is* a knob picked by looking at the task score, and is logged as such; the
   principle it settles on (H must span the addendum's own ≈34-agent-step
   time-to-goal) was available a priori and the original 20 was too
   short for an MC swing.

Measured but **not** adopted (an ablation, not a default): replacing the Gaussian
target with a broad+goal ``MixtureGaussian`` — flat away from the flag, peaked at
it — which is the *faithful* analogue of their reward ("−1 per step until you
arrive", and no penalty at all for driving left). On the frozen model it scores
better than the default (ACR −80.0 vs −90.0 at H=40, and SR 0.60 vs 0.00 at
H=20, since a flat target cannot be deceptive). It is not the default,
and it costs no new machinery: a
mixture target already works through ``kl_to``'s Kolchinsky–Tracey bounds.

Run: ``python examples/mountain_car.py --episodes 300`` (``--help`` for the
knobs). Prints a per-episode line (their ACR, success, wall clock, one-step model
error) and a rolling success rate over the last 100 episodes.
"""

import argparse
import time
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hybrid_aif.controls import Box
from hybrid_aif.distributions import Gaussian, MLPConditional
from hybrid_aif.factors import LinearGaussianObservation
from hybrid_aif.planning.cem import cem_plan, mean_action
from hybrid_aif.reference.envs.mountain_car import MountainCar
from hybrid_aif.system import LearnState, System, SystemParams

STATE_DIM = 2
TARGET_POSITION = 0.5  # just past their flag at 0.45
TARGET_SD = (0.1, 10.0)  # (position, velocity): velocity is near-free
VELOCITY_SCALE = 0.07  # their max_speed: the unit the state variable is declared in


def make_agent(
    key: Array,
    *,
    horizon: int = 40,
    width: int = 96,
    depth: int = 2,
    activation: str = "tanh",
    init_scale: float = 0.05,
    obs_std: float = 1e-3,
    gamma: float = 4.0,
    velocity_scale: float = VELOCITY_SCALE,
) -> tuple[System, SystemParams]:
    """The P1 System: neural transition, clamp observation, goal designation.

    ``init_scale`` centres the fresh net's predictive standard deviation. It is
    the one knob with an exploration flavour — a wide fresh model makes every
    plan look uncertain and none look good, which is exactly the "optimistic
    initialization" the classical end has available — but it is a *prior over
    dynamics noise*, declared in the model, not a bonus bolted onto G.

    **Units are a declaration too.** The state variable is
    ``(position, velocity / velocity_scale)``, not ``(position, velocity)``: the
    env's velocity lives in ±0.07 while its position spans ±1, and a net whose
    two output dimensions differ by that much in scale spends its capacity on the
    large one and predicts the small one as noise. The *observation model* is
    where the two unit systems meet — ``H = diag(1, velocity_scale)`` maps the
    declared state onto the env's raw ``(position, velocity)`` — which is exactly
    what a sensor map is for, and it keeps the change inside the declaration
    instead of wrapping the env. Set ``velocity_scale=1.0`` to declare the state
    in raw units and watch the model error stop converging.
    """
    system = System(
        A_deps=((0,),),
        B_deps=((),),
        controlled=(True,),
        policy_len=horizon,
        planner="cem",
        gamma=gamma,
        epistemic=False,  # see the module docstring: no estimator for a neural slot
        modality_names=("state_obs",),
        factor_names=("state",),
    )
    params = SystemParams(
        likelihood={
            "state_obs": LinearGaussianObservation(
                H=jnp.diag(jnp.asarray([1.0, velocity_scale])),
                c=jnp.zeros(STATE_DIM),
                noise=obs_std * jnp.eye(STATE_DIM),
            ),
        },
        transition={
            "state": MLPConditional.from_shapes(
                (STATE_DIM, 1),  # own previous value, then the continuous action
                STATE_DIM,
                key=key,
                width=width,
                depth=depth,
                activation=activation,
                init_scale=init_scale,
            ),
        },
        initial_state_prior={
            "state": Gaussian(loc=jnp.asarray([-0.5, 0.0]), chol=jnp.diag(jnp.asarray([0.1, 0.1])))
        },
        state_targets={
            "state": Gaussian(
                loc=jnp.asarray([TARGET_POSITION, 0.0]),
                chol=jnp.diag(jnp.asarray(TARGET_SD)),
            )
        },
        #  the push: one number in [-1, 1], the box of the control node u_state
        control={"u_state": Box(-jnp.ones(1), jnp.ones(1))},
    )
    return system, params


class EpisodeLog(NamedTuple):
    """One episode's record, in *their* units where they have any."""

    reward: Array  # their sparse reward, summed: the ACR contribution
    success: Array  # their is_success: the flag was reached
    agent_steps: Array  # steps until termination (== time_limit if never)
    final_position: Array
    model_error: Array  # mean |one-step prediction - observation|, in the env's units


class _Carry(NamedTuple):
    params: SystemParams
    learn_state: LearnState
    priors: dict
    prev_beliefs: dict
    prev_action: Array
    state: tuple
    plan_mean: Array
    key: Array
    done: Array
    reward: Array
    steps: Array
    error: Array
    lr: Array


def make_episode(
    system: System,
    env: MountainCar,
    *,
    num_candidates: int = 64,
    cem_iters: int = 4,
    elite_frac: float = 0.125,
    min_std: float = 0.2,
    lr: float = 3e-3,
):
    """A jitted episode: ``TimeLimit(100)`` agent steps of observe → learn → plan → act.

    One ``lax.scan``, so the whole episode compiles once: the params, the
    optimizer carry, the belief carry and the CEM warm start all ride in the
    scan carry. Ticks after termination are masked (their env stops; the scan
    cannot), which freezes the state, the reward and the learning rate alike —
    learning on a frozen transition would teach the model that the world stands
    still.

    ``lr`` is the *default* rate; the caller passes the episode's rate to
    ``episode(...)`` so a schedule lives in the caller's loop (the rate grammar
    takes traced rates precisely so a curriculum needs no new API).
    """

    def tick(carry: _Carry, t: Array) -> tuple[_Carry, None]:
        live = ~carry.done
        obs = system.from_positional("observations", env.observe(carry.state))
        beliefs, _ = system.infer_states(carry.params, obs, carry.priors)

        # A2, online: one adam step per agent step. Tick 0 has no previous belief,
        # and a terminated episode has no dynamics left to learn — both are traced
        # zero rates (learn's documented in-scan curriculum form), not skips, so the
        # LearnState pytree keeps one shape through the scan.
        rate = jnp.where((t > 0) & live, carry.lr, 0.0)
        params, learn_state = system.learn(
            carry.params,
            obs,
            beliefs,
            prev_beliefs=carry.prev_beliefs,
            action=system.from_positional("actions", carry.prev_action, carry.params),
            lr=rate,
            learn_state=carry.learn_state,
        )

        # A9: CEM, warm-started from the previous tick's plan shifted one step
        # (receding horizon). The action is the refit proposal mean's first step.
        key, k_plan = jax.random.split(carry.key)
        ctx = system.planner_context(params, beliefs)
        plan = cem_plan(
            ctx,
            k_plan,
            num_candidates=num_candidates,
            cem_iters=cem_iters,
            elite_frac=elite_frac,
            init_mean=carry.plan_mean,
            min_std=min_std,
            metrics=False,
        )
        action = mean_action(plan.candidates)

        obs_next, state_next, reward, done = env.agent_step(None, carry.state, action)
        priors = system.update_empirical_prior(
            params, beliefs, system.from_positional("actions", action, params)
        )
        # the one-step predictive error, read in the ENV's units (through the sensor
        # map, since the state variable is declared in scaled velocity)
        predicted = params.likelihood["state_obs"].H @ priors["state"].mean()
        error = jnp.mean(jnp.abs(predicted - obs_next[0]))
        return (
            _Carry(
                params=params,
                learn_state=learn_state,
                priors=priors,
                prev_beliefs=beliefs,
                prev_action=action,
                state=state_next,
                plan_mean=jnp.concatenate([plan.mean[1:], plan.mean[-1:]]),
                key=key,
                done=done,
                reward=carry.reward + reward,
                steps=carry.steps + live.astype(jnp.int32),
                error=carry.error + jnp.where(live, error, 0.0),
                lr=carry.lr,
            ),
            None,
        )

    @eqx.filter_jit
    def episode(
        key: Array,
        params: SystemParams,
        learn_state: LearnState,
        episode_lr: float | Array = lr,
    ) -> tuple[SystemParams, LearnState, EpisodeLog]:
        k_reset, k_run = jax.random.split(key)
        _, state = env.reset(k_reset)
        beliefs = system.initial_priors(params)
        action_dim = system.action_dim(params)
        init = _Carry(
            params=params,
            learn_state=learn_state,
            priors=beliefs,
            prev_beliefs=beliefs,
            prev_action=jnp.zeros((action_dim,)),
            state=state,
            plan_mean=jnp.zeros((system.policy_len, action_dim)),
            key=k_run,
            done=jnp.asarray(False),
            reward=jnp.zeros(()),
            steps=jnp.zeros((), jnp.int32),
            error=jnp.zeros(()),
            lr=jnp.asarray(episode_lr, jnp.float32),
        )
        out, _ = jax.lax.scan(tick, init, jnp.arange(env.time_limit))
        log = EpisodeLog(
            reward=out.reward,
            success=out.done,
            agent_steps=out.steps,
            final_position=out.state[0][0],
            model_error=out.error / jnp.maximum(out.steps, 1),
        )
        return out.params, out.learn_state, log

    return episode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=40, help="CEM planning horizon, agent steps")
    ap.add_argument("--candidates", type=int, default=64)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--min-std", type=float, default=0.2, help="CEM proposal floor (action units)")
    ap.add_argument("--lr", type=float, default=3e-3, help="world-model rate at episode 0")
    ap.add_argument(
        "--lr-final",
        type=float,
        default=3e-4,
        help="world-model rate at the last episode (geometric decay; Robbins-Monro)",
    )
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--init-scale", type=float, default=0.05)
    ap.add_argument("--every", type=int, default=1, help="print every N episodes")
    args = ap.parse_args()

    key = jax.random.PRNGKey(args.seed)
    key, k_agent = jax.random.split(key)
    system, params = make_agent(
        k_agent,
        horizon=args.horizon,
        width=args.width,
        depth=args.depth,
        init_scale=args.init_scale,
    )
    env = MountainCar()
    learn_state = system.init_learn_state(params)
    episode = make_episode(
        system,
        env,
        num_candidates=args.candidates,
        cem_iters=args.iters,
        min_std=args.min_std,
        lr=args.lr,
    )

    print(
        f"P1 Mountain Car: {args.episodes} episodes x {env.time_limit} agent steps "
        f"(ActionRepeat({env.action_repeat})), CEM {args.candidates}x{args.iters} over "
        f"H={args.horizon}, model lr {args.lr} -> {args.lr_final}"
    )
    print("episode  ACR      success  steps  final_p  model_err  wall")
    rewards: list[float] = []
    successes: list[float] = []
    first_success = None
    t0 = time.perf_counter()
    for ep in range(args.episodes):
        key, k_ep = jax.random.split(key)
        start = time.perf_counter()
        frac = ep / max(args.episodes - 1, 1)
        rate = args.lr * (args.lr_final / args.lr) ** frac
        params, learn_state, log = episode(k_ep, params, learn_state, jnp.float32(rate))
        elapsed = time.perf_counter() - start
        rewards.append(float(log.reward))
        successes.append(float(log.success))
        if first_success is None and log.success:
            first_success = ep
        if ep % args.every == 0 or ep == args.episodes - 1:
            print(
                f"{ep:7d}  {rewards[-1]:7.1f}  {bool(log.success)!s:>7}  "
                f"{int(log.agent_steps):5d}  {float(log.final_position):+.3f}  "
                f"{float(log.model_error):9.2e}  {elapsed:5.2f}s"
            )
    total = time.perf_counter() - t0

    window = min(100, len(successes))
    print()
    print(f"episodes to first success: {first_success}")
    print(
        f"success rate over the last {window} episodes: "
        f"{np.mean(successes[-window:]):.2f}  (ACR {np.mean(rewards[-window:]):.1f})"
    )
    print(f"wall clock: {total:.1f}s for {args.episodes} episodes")


if __name__ == "__main__":
    main()
