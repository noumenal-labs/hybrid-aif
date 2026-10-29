"""P1′ chunk 2c's acceptance: the whole loop runs on Mountain Car, and it learns.

Design: ``docs/design/amortization.md`` A12's scope split — "**2c** = the training
side … Acceptance: the full loop runs on MC and learns" — with the four criteria
the chunk was dispatched on:

    (a) every trainer's loss is finite and moves;
    (b) the world model's open-loop prediction improves over its untrained value;
    (c) the critic's return estimate tracks realized λ-returns better than a
        constant;
    (d) the actor's self-imitation term drives ``log π(a_expert)`` up on seed data.

**Not** reaching paper numbers: that is 2d, and it is compute rather than code
(4 seeds × 1M steps against the oracle). What is asserted here is that every
gradient path is connected and pointed the right way.

**Two deliberate departures from parity, both for the clock.** The model is
smaller than their ``tiny`` and the learning rates are raised
(:data:`FAST`) — their ``adam 3e-5`` on the actor and critic needs tens of
thousands of steps to move anything visibly, which is 2d's budget, not a test's.
Every *structural* number is theirs: the objective, the λ-return, the signs, the
gates, the batch shape's meaning. The full-size, parity-rate version of this loop
is :func:`test_the_loop_runs_at_parity_settings`, marked ``slow``.

Run the slow marker with ``pytest -m slow tests/test_raif_acceptance.py``
(``addopts`` deselects it, as it does the notebook gate).
"""

import time
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.reference.envs.mountain_car import MountainCar
from hybrid_aif.reference.raif.agent import (
    act,
    raif_mountain_car,
    to_world_batch,
    train_step,
)
from hybrid_aif.reference.raif.columns import EXTRAS
from hybrid_aif.reference.raif.metrics import evaluate
from hybrid_aif.reference.raif.revise import label
from hybrid_aif.reference.raif.run import alternate
from hybrid_aif.reference.raif.seed import bang_bang, collect_episode
from hybrid_aif.train.buffers import StaticBuffer, add, sample, seed_stats
from hybrid_aif.train.imagine import RETNORM_KEY, SLOW_KEY, imagine, rollout_returns
from hybrid_aif.train.world import assemble, posterior_states

#: Small enough to fit the budget, structurally theirs. Their ``tiny`` is
#: ``deter 128``, ``stoch 32×32``, ``.*units 32``, ``mlp_units 64``, actor/critic
#: ``layers 3, units 256``.
SMALL = dict(
    horizon=4,  # their imag_horizon is 8
    deter=16,
    num_vars=6,
    num_classes=6,
    units=8,
    mlp_units=16,
    embed=16,
    head_units=32,
    head_layers=2,
)

#: Raised rates, so "does it learn" is answerable in seconds. Their parity rates are
#: the defaults of every trainer (model 1e-4, actor/critic 3e-5, IG 1e-4).
FAST = dict(world_lr=3e-3, actor_lr=3e-3, critic_lr=3e-3, ensemble_lr=3e-3, crspp_lr=3e-3)

SEQ_LEN = 32  # their batch_length is 64
BATCH = 8  # their batch_size is 16
STEPS = 120
TIME_LIMIT = 90  # their TimeLimit(100) agent steps
SEED_EPISODES = 4
CRITIC_STEPS = 400  # the critic-fitting phase (criterion (c)), actor frozen
RETURN_SAMPLES = 2  # rollout keys averaged when measuring the return


class Run(NamedTuple):
    agent: object
    history: list  # one StepLosses per gradient step, BC term at their 0.5
    ablated: list  # the same loop with bc_scale=0 — criterion (d)'s control
    before: float  # the untrained open-loop ratio (model MSE / persistence MSE)
    after: float
    batch: object  # a held-out batch, for the critic's residual
    seed: object  # the seed policy's stats, reported beside the numbers (A12 item 6)
    collected: list  # episodes the trained habit collected, for the metrics harness
    critic: tuple  # (Residual before, Residual after) the critic-fitting phase


class Residual(NamedTuple):
    """How well a value predictor explains a set of λ-returns. Lower is better.

    ``critic`` is the trained critic's mean squared error against the returns;
    ``zero`` is the error of the *constant* predictor the untrained critic is
    (``out_scale 0.0`` ⇒ it answers ≈0 for every state); ``best_constant`` is the
    error of the best possible constant, the returns' own mean. Criterion (c) is
    asserted against ``zero`` and *reported* against ``best_constant`` — see
    :func:`test_the_critic_tracks_the_realized_lambda_returns_better_than_a_constant`.
    """

    critic: float
    zero: float
    best_constant: float
    ret_mean: float
    critic_mean: float


def _open_loop_ratio(agent, batch, *, warm: int = 6, horizon: int = 4) -> float:
    """``model MSE / persistence MSE`` on the batch's held-out tail (chunk 2a's metric).

    Warm up the posterior, roll the *prior* forward with the recorded actions and no
    further observations, and predict position better than holding the last one —
    the baseline a model that had learned nothing about how velocity integrates
    would tie. Below 1 is better.
    """
    core, _ = assemble(agent.system, agent.params)
    world_batch = to_world_batch(batch)

    def one(key, obs, action):
        states, _, _ = core.observe(key, action[:warm], obs[:warm])
        warmed = jax.tree_util.tree_map(lambda x: x[-1], states)
        rolled, _ = core.open_loop(key, warmed, action[warm : warm + horizon])
        predicted = jax.vmap(core.decode)(rolled)[:, 0]
        truth = obs[warm : warm + horizon, 0]
        persistence = jnp.broadcast_to(obs[warm - 1, 0], truth.shape)
        return jnp.mean((predicted - truth) ** 2), jnp.mean((persistence - truth) ** 2)

    keys = jax.random.split(jax.random.PRNGKey(11), world_batch.obs.shape[0])
    model, base = jax.vmap(one)(keys, world_batch.obs, world_batch.action)
    return float(jnp.mean(model) / jnp.mean(base))


@pytest.fixture(scope="module")
def run() -> Run:
    """Seed from bang-bang, train, then collect with the habit. One run, read below."""
    env = MountainCar(time_limit=TIME_LIMIT)

    #  1. seed M+ from the documented fallback controller (A12 item 6), labelled
    #     once per FINISHED episode — the schedule, not a convenience.
    seed_keys = jax.random.split(jax.random.PRNGKey(1), SEED_EPISODES)
    episodes = [label(collect_episode(env, bang_bang, k, can_self_imitate=True)) for k in seed_keys]
    stats = seed_stats(episodes, success="is_success")

    positive = StaticBuffer.empty(16, SEQ_LEN, 2, 1, extras=EXTRAS)
    general = StaticBuffer.empty(16, SEQ_LEN, 2, 1, extras=EXTRAS)
    for ep in episodes:
        positive, general = add(positive, ep), add(general, ep)  # success -> BOTH

    fresh = raif_mountain_car(jax.random.PRNGKey(2), **SMALL)
    held = sample(general, jax.random.PRNGKey(3), BATCH)
    before = _open_loop_ratio(fresh, held, horizon=SMALL["horizon"])

    #  2. the gradient loop: strict alternation with the documented fallback while
    #     M+ is short (A10 disposition 9 — both in the caller's loop, right here).
    #     Run twice from the same initialization and the same batch sequence, once
    #     with their BC scale and once with it ablated: criterion (d) is a claim
    #     about the *term*, and the advantage term dominates the actor's gradient in
    #     a 120-step run (their percentile-EMA normalizer needs far longer than that
    #     to warm up), so an absolute rise in log π would be luck rather than
    #     evidence. The paired difference is the evidence.
    step = eqx.filter_jit(
        lambda a, b, k, bc: train_step(a, b, key=k, horizon=SMALL["horizon"], bc_scale=bc, **FAST)
    )

    def loop(bc_scale: float):
        agent, out = fresh, []
        run_key = jax.random.PRNGKey(31)
        for i, buf in zip(range(STEPS), alternate(positive, general), strict=False):
            run_key, draw = jax.random.split(run_key)
            source = buf if buf.ready(BATCH) else general
            batch = sample(source, draw, BATCH)
            agent, losses = step(agent, batch, jax.random.fold_in(run_key, i), bc_scale)
            out.append(losses)
        return agent, out

    agent, history = loop(0.5)  # their loss_scales value
    _, ablated = loop(0.0)

    after = _open_loop_ratio(agent, held, horizon=SMALL["horizon"])

    #  3. a critic-fitting phase, for criterion (c): ``imagine`` alone, actor frozen,
    #     one fixed rollout key, and — the one substantive choice — ``discount=0``,
    #     which degenerates the λ-return to the per-step G and so switches the
    #     bootstrap off. Two reasons, both measured and both about the *budget*
    #     rather than the code:
    #
    #     (i) in the joint loop the target runs away — the info-gain leaf is a
    #         differential entropy, unbounded below, and it grows as the ensemble
    #         fits, so the value climbs faster than a short run can track (measured:
    #         ret mean 53 -> 199 over 150 joint steps);
    #     (ii) at their ``discount = 1 - 1/333`` the TD fixed point is ≈1/(1−γ)
    #         times the per-step value, so with G ≈ 35/step the critic's target is
    #         ≈10⁴ and the bootstrap raises the target as fast as the critic rises
    #         (measured: the residual sits at ≈Σ G and does not fall).
    #
    #     Both are faithful to their objective and are 2d's to live with; what this
    #     phase isolates is whether the critic's head, loss and optimizer actually
    #     fit the returns they are handed.
    starts = _flat_starts(agent, held)
    fit = eqx.filter_jit(
        lambda p, ls: imagine(
            agent.system,
            p,
            starts,
            ls,
            key=jax.random.PRNGKey(23),
            horizon=SMALL["horizon"],
            actor_lr=0.0,
            critic_lr=1e-2,
            discount=0.0,
        )
    )
    critic_before = _residual(agent.system, agent.params, starts)
    params, learn_state = agent.params, agent.learn_state
    for _ in range(CRITIC_STEPS):
        params, learn_state, _ = fit(params, learn_state)
    critic_after = _residual(agent.system, params, starts)

    #  3. the loop closes: the trained habit acts in the env, and what it collects
    #     goes through the same routing and the same metrics harness.
    collected = []
    for i in range(2):
        collected.append(
            label(collect_episode(env, _habit_policy(agent), jax.random.PRNGKey(40 + i)))
        )
    return Run(
        agent=agent,
        history=history,
        ablated=ablated,
        before=before,
        after=after,
        batch=held,
        seed=stats,
        collected=collected,
        critic=(critic_before, critic_after),
    )


def _habit_policy(agent):
    """``(key, obs) -> action`` from the habit: filter, then act (A12 item 3)."""
    system, params = agent.system, agent.params
    state = {"beliefs": None}

    def policy(key, obs):
        act_key, infer_key = jax.random.split(key)
        priors = system.initial_priors(params) if state["beliefs"] is None else state["beliefs"]
        observed = system.from_positional("observations", (obs, None, None))
        beliefs, _ = system.infer_states(params, observed, priors)
        action = act(agent, beliefs, act_key)
        state["beliefs"] = system.update_empirical_prior(params, beliefs, action)
        del infer_key
        return system.to_positional("actions", action, params)

    return policy


# ================================================================== (a) losses move


def test_every_trainers_loss_is_finite_throughout(run: Run):
    """Criterion (a), first half: four losses, every step, no NaN and no inf.

    The failure this catches is the one a training loop hides best — a single
    non-finite step poisons every parameter afterwards, and the loss curve then
    looks flat rather than broken.
    """
    for i, losses in enumerate(run.history):
        for name, value in (
            ("world", losses.world.total),
            ("actor", losses.imagine.actor_loss),
            ("critic", losses.imagine.critic_loss),
            ("ensemble", losses.ensemble.total),
            ("crspp", losses.crspp.total),
        ):
            assert bool(jnp.isfinite(value)), (i, name, float(value))


def test_every_trainers_loss_moves(run: Run):
    """Criterion (a), second half: every gradient path is *connected*.

    A loss that does not move over 120 steps is a disconnected path — a
    stop-gradient in the wrong place, a slot never written back, an optimizer state
    keyed wrong — and it is exactly what a smoke test that only checks finiteness
    misses.
    """
    first, last = run.history[0], run.history[-1]
    moved = {
        "world": (float(first.world.total), float(last.world.total)),
        "actor": (float(first.imagine.actor_loss), float(last.imagine.actor_loss)),
        "critic": (float(first.imagine.critic_loss), float(last.imagine.critic_loss)),
        "ensemble": (float(first.ensemble.total), float(last.ensemble.total)),
        "crspp": (float(first.crspp.total), float(last.crspp.total)),
    }
    still = {name: pair for name, pair in moved.items() if pair[0] == pair[1]}
    assert not still, still


def test_the_world_model_and_the_ensemble_losses_go_down(run: Run):
    """Both are supervised losses on replay, so "it learns" is simply "it descends".

    Stated separately from the movement test because *direction* is the claim: the
    actor's and the critic's losses need not fall (their targets move under them),
    but a rising reconstruction loss would be a sign error.
    """
    first, last = run.history[0], run.history[-1]
    assert float(last.world.total) < float(first.world.total)
    assert float(last.ensemble.total) < float(first.ensemble.total)


def test_the_preference_model_sees_positive_states_to_learn_from(run: Run):
    """Self-revision has signal: bang-bang solves the task, so ρ is positive over the
    approach and the CRSPP's gated KL is non-empty. A run where ``positive`` is 0
    throughout would be training the preference model on nothing — and, since the
    gate is a hard mask, doing so silently.
    """
    positive = np.array([float(x.crspp.positive) for x in run.history])
    assert positive.max() > 0.0
    assert np.isfinite(np.array([float(x.crspp.kl) for x in run.history])).all()


# =============================================== (b) the world model actually predicts


def test_the_open_loop_prediction_improves_over_its_untrained_value(run: Run):
    """Criterion (b). The chunk-2a metric — an open-loop rollout's position MSE
    against a persistence baseline — computed on the same held-out batch before and
    after training. The *ratio* is the statistic, so the baseline cancels out the
    batch's own difficulty.
    """
    assert run.after < run.before, (run.before, run.after)


# ==================================================== (c) the critic tracks the return


def test_the_critic_tracks_the_realized_lambda_returns_better_than_a_constant(run: Run):
    """Criterion (c). Over the critic-fitting phase, its squared error against the
    λ-returns it is regressed on falls by orders of magnitude and ends far below the
    error of a **constant** predictor — namely the one the untrained critic *is*,
    since ``out_scale 0.0`` makes a fresh critic answer ≈0 for every state.

    The returns come from :func:`hybrid_aif.train.imagine.rollout_returns` — the
    read-only face of the trainer's own rollout — so the diagnostic is not a second
    implementation of the λ-return that could drift from the one being fitted.

    **The tighter reading, now asserted (2026-09-02).** The tight reading of "better
    than a constant" is the *best-fit* constant — the returns' own mean, which no
    state-dependent critic can beat by luck. This test previously only *reported* it,
    because on the pre-2026-09-02 objective the margin was a hair (3.45 against 4.10,
    from 1820.7 untrained) and not robust across seeds. Closing divergences 10–13
    changed the target's scale, not just its level: the per-step ``weight`` and the
    hard continuation mode truncate a rollout at its predicted end, so the λ-return
    is both smaller (``ret_mean`` ≈0.59, was ≈50) and *more* state-dependent — and
    the critic now beats the best-fit constant by ≈2.3×, comfortably rather than
    narrowly.

    So the assertion moved to that reading, and the **zero**-predictor comparison
    relaxed from 100× to 10× in the same breath — deliberately, and this is the
    honest half: with the target an order of magnitude smaller, the error against a
    zero predictor is smaller too, and the old 100× headroom was a property of the
    *level* rather than of the fit. Asserting the level-free version and loosening
    the level-dependent one keeps the criterion's content while removing its
    dependence on a number the objective is allowed to move.
    """
    before, after = run.critic
    assert np.isfinite([before.critic, after.critic, after.zero]).all()
    assert after.critic < before.critic / 100.0, (before, after)
    #  the tight reading: better than the best possible constant
    assert after.critic < after.best_constant, (after.critic, after.best_constant)
    #  and the level-dependent one, at the headroom the post-fix scale actually has
    assert after.critic < after.zero / 10.0, (after.critic, after.zero)
    assert abs(after.critic_mean - after.ret_mean) < 0.05 * abs(after.ret_mean), after


# ============================================== (d) the actor imitates the seed data


def test_the_self_imitation_term_raises_the_log_probability_of_the_expert_action(run: Run):
    """Criterion (d). ``bc_logprob`` is ``log π(a_start | s_0)`` on the batch's own
    actions, and the seed data is all expert-flagged, so with the BC term active it
    must rise over training. This is the one criterion that reads a *policy-side*
    quantity, which is why ``imagine`` returns it as a metric rather than folding it
    into the loss and hoping (``objective-terms.md``: policy-side quantities have no
    term-tree home, so the trainer reports them).
    """
    on = np.array([float(x.imagine.bc_logprob) for x in run.history])
    off = np.array([float(x.imagine.bc_logprob) for x in run.ablated])
    weight = np.array([float(x.imagine.bc_weight) for x in run.history])
    assert weight.max() > 0.0, "no expert-flagged starts: the term was inert"
    #  same initialization, same batches, same keys — only the term's scale differs
    assert np.allclose(on[0], off[0], atol=1e-5), (on[0], off[0])
    assert on[-10:].mean() > off[-10:].mean(), (on[-10:].mean(), off[-10:].mean())


# ===================================================== the loop closes, and reports


def test_the_trained_habit_acts_in_the_environment_and_the_metrics_fold(run: Run):
    """End to end: the habit plans nothing (A12 item 3 — G trains, it does not
    select), acts through ``System.act``, and what it collects goes through the
    routing and the metrics harness unchanged. ACR/SR/R-S are *computed*, not
    asserted against paper numbers: 2d owns those.
    """
    #  RAGGED, and deliberately not stacked: since 2026-09-07 an episode ends at the
    #  flag, so its length is variable and its final row is the goal observation —
    #  and R-S's C drops exactly that row's reward, so padding to a rectangle would
    #  move which row it drops (`eval.stability_c`)
    rewards = [np.asarray(ep.reward) for ep in run.collected]
    successes = np.array([bool(ep.extras["has_succeeded"][-1]) for ep in run.collected])
    assert all(row.ndim == 1 and 1 <= row.shape[0] <= TIME_LIMIT + 1 for row in rewards)
    metrics = evaluate(rewards, successes, window=100)
    assert metrics.n == len(run.collected)
    assert np.isfinite([metrics.acr, metrics.sr, metrics.rs]).all()
    assert -2 * TIME_LIMIT <= metrics.acr <= 0.0


def test_the_seed_policys_quality_is_reported_beside_the_numbers(run: Run):
    """A12 item 6: "seed stats are reported next to the acceptance numbers either
    way — under the fixture that is a parity check, not a confession"."""
    assert run.seed.n_episodes == SEED_EPISODES
    assert 0.0 <= run.seed.success_rate <= 1.0
    assert run.seed.mean_return < 0.0


# ---------------------------------------------------------------------- the slow twin


@pytest.mark.slow
def test_the_loop_runs_at_parity_settings():
    """The same loop at their ``tiny`` sizes, their batch shape and their rates.

    Ten gradient steps only — this asserts that nothing about the *full-size*
    configuration is broken (shapes, the 255-bin critic, ``imag_horizon 8``,
    ``16 × 64`` batches), not that it learns, which at ``adam 3e-5`` needs 2d's
    budget. Deselected by default; run it with ``pytest -m slow``.
    """
    env = MountainCar()
    episodes = [
        label(collect_episode(env, bang_bang, jax.random.PRNGKey(i), can_self_imitate=True))
        for i in range(2)
    ]
    buf = StaticBuffer.empty(4, 64, 2, 1, extras=EXTRAS)
    for ep in episodes:
        buf = add(buf, ep)
    agent = raif_mountain_car(jax.random.PRNGKey(0))  # every default: their tiny
    step = eqx.filter_jit(lambda a, b, k: train_step(a, b, key=k))
    started = time.time()
    for i in range(10):
        agent, losses = step(agent, sample(buf, jax.random.PRNGKey(i), 16), jax.random.PRNGKey(i))
        assert bool(jnp.isfinite(losses.world.total))
        assert bool(jnp.isfinite(losses.imagine.actor_loss))
        assert bool(jnp.isfinite(losses.crspp.total))
    assert time.time() - started > 0.0


# -------------------------------------------------------------------------- helpers


def _residual(system, params, starts) -> "Residual":
    """The critic's error against the λ-returns, beside two constant baselines.

    The returns are averaged over :data:`RETURN_SAMPLES` rollout keys, because a
    single sampled λ-return carries the info-gain leaf's sampling noise and "does
    the critic track the return" is a question about the expectation.
    """
    rets, values = [], []
    for j in range(RETURN_SAMPLES):
        est = rollout_returns(
            system,
            params,
            starts,
            key=jax.random.PRNGKey(1000 + j),
            horizon=SMALL["horizon"],
            discount=0.0,  # the phase's setting: the per-step G, no bootstrap
        )
        rets.append(est.ret)
        values.append(est.critic)
    ret = jnp.mean(jnp.stack(rets), axis=0).reshape(-1)
    value = jnp.mean(jnp.stack(values), axis=0).reshape(-1)
    return Residual(
        critic=float(jnp.mean((value - ret) ** 2)),
        zero=float(jnp.mean(ret**2)),
        best_constant=float(jnp.mean((jnp.mean(ret) - ret) ** 2)),
        ret_mean=float(jnp.mean(ret)),
        critic_mean=float(jnp.mean(value)),
    )


def _flat_starts(agent, batch):
    states = posterior_states(
        agent.system, agent.params, to_world_batch(batch), key=jax.random.PRNGKey(5)
    )
    beliefs = agent.system.beliefs_from_states(agent.params, states)
    n = states.deter.shape[0] * states.deter.shape[1]
    return {k: jnp.reshape(x, (n, *x.shape[2:])) for k, x in beliefs.items()}


def test_the_normalizer_and_slow_critic_carries_survive_the_trainer_seam():
    """``moments`` must cross every trainer, not just the one that owns it.

    ``fit_world``, ``fit_ensemble`` and ``fit_crspp`` all run before ``imagine`` in
    :func:`~hybrid_aif.reference.raif.agent.train_step`, and each returns a fresh
    ``LearnState``; one that rebuilds it without ``moments`` resets the return
    normalizer's percentile EMA and the slow critic's update counter on every gradient
    step (``imagine`` now refuses a carry with the entries *missing*, but not one a
    caller re-seeds). The transcription tests over this machinery cannot see that: a
    port proves a *formula* and inherits its author's assumptions about the caller —
    one threads the EMA state by hand in a Python loop, the other takes the advantage
    as a given input. So this test is deliberately at the seam — N real ``train_step``
    calls — and asserts the two invariants a reset carry violates (divergence 29).
    """
    env = MountainCar(time_limit=TIME_LIMIT)
    keys = jax.random.split(jax.random.PRNGKey(7), 4)
    episodes = [label(collect_episode(env, bang_bang, k, can_self_imitate=True)) for k in keys]
    buf = StaticBuffer.empty(8, SEQ_LEN, 2, 1, extras=EXTRAS)
    for ep in episodes:
        buf = add(buf, ep)

    agent = raif_mountain_car(jax.random.PRNGKey(8), **SMALL)
    batch = sample(buf, jax.random.PRNGKey(9), BATCH)
    step = eqx.filter_jit(lambda a, b, k: train_step(a, b, key=k, horizon=SMALL["horizon"], **FAST))

    def band(a):
        low, high = (float(x) for x in a.learn_state.moments[RETNORM_KEY])
        return high - low

    def k(i):
        return jax.random.fold_in(jax.random.PRNGKey(10), i)

    one, _ = step(agent, batch, k(0))
    n = 7
    many = one
    for i in range(1, n):
        many, _ = step(many, batch, k(i))

    #  the slow critic's counter counts every step it has taken, so a wiped carry
    #  pins it at 1 — and `_slow_update`'s mix then saturates, hard-copying the critic.
    assert int(one.learn_state.moments[SLOW_KEY]) == 1
    assert int(many.learn_state.moments[SLOW_KEY]) == n

    #  A dropped carry makes every step the FIRST step, so the band is always
    #  `(1 - decay) x percentile` and barely moves between step 1 and step n. A live
    #  carry grows it by `(1 - decay**n) / (1 - decay)` — about 6.8x at n = 7, decay
    #  0.99 — on the same data. The ratio is the signature, so the test needs no
    #  absolute scale and cannot be fooled by the policy drifting.
    assert band(one) > 0.0, "the first step produced no band at all"
    assert band(many) > 3.0 * band(one), (
        f"retnorm band {band(many):.6g} after {n} steps vs {band(one):.6g} after one: "
        "the percentile EMA is restarting, so a trainer is dropping `moments`"
    )
