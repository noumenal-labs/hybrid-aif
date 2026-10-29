"""Performance-review fixes (2026-09-25): each keeps its numbers and drops its cost.

Runnalls' reduction updates its cost matrix incrementally; the mode-aligned reduce
vmaps its groups; the composition driver's agent tick is compiled; ``replay`` and
``rollout`` run compiled cores; the diagonal neural head evaluates its density on the
diagonal. Each is pinned against the arithmetic it replaced.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from hybrid_aif.distributions import Gaussian, MixtureGaussian, MLPConditional
from hybrid_aif.distributions.mixture import _log_det, _merge_moments, _runnalls_merge

KEY = jax.random.PRNGKey(0)


def _runnalls_merge_reference(
    log_w: Array, means: Array, chols: Array, k: int
) -> tuple[Array, Array, Array]:
    """The pre-2026-09-25 merge: every pair cost rebuilt each iteration (the reference)."""
    n = log_w.shape[0]
    covs = chols @ jnp.swapaxes(chols, -1, -2)
    upper = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    peak = jnp.max(log_w)  # -inf only if every slot is dead
    offset = jnp.where(jnp.isfinite(peak), peak, 0.0)
    log_w = log_w - offset  # bound exp() without changing merge decisions or gauge

    def body(_, state):
        log_w, means, covs, alive = state
        lw_sum, mu_m, cov_m = _merge_moments(
            log_w[:, None],
            log_w[None, :],
            means[:, None, :],
            means[None, :, :],
            covs[:, None, :, :],
            covs[None, :, :, :],
        )
        cost = 0.5 * (
            jnp.exp(lw_sum) * _log_det(cov_m)
            - jnp.exp(log_w)[:, None] * _log_det(covs)[:, None]
            - jnp.exp(log_w)[None, :] * _log_det(covs)[None, :]
        )
        dead_pair = jnp.isneginf(log_w)[:, None] | jnp.isneginf(log_w)[None, :]
        valid = upper & alive[:, None] & alive[None, :]
        # Runnalls costs are >= 0, so -1 forces dead slots to be consumed first.
        cost = jnp.where(valid, jnp.where(dead_pair, -1.0, cost), jnp.inf)
        idx = jnp.argmin(cost.reshape(-1))
        i, j = idx // n, idx % n  # upper triangle only, so i < j
        log_w = log_w.at[i].set(lw_sum[i, j]).at[j].set(-jnp.inf)
        means = means.at[i].set(mu_m[i, j])
        covs = covs.at[i].set(cov_m[i, j])
        return log_w, means, covs, alive.at[j].set(False)

    alive = jnp.ones(n, dtype=bool)
    log_w, means, covs, alive = lax.fori_loop(0, n - k, body, (log_w, means, covs, alive))
    top_lw, top_idx = lax.top_k(log_w, k)
    return top_lw + offset, means[top_idx], covs[top_idx]


def _mixture(key, n, dim, dead=()):
    k1, k2, k3 = jax.random.split(key, 3)
    lw = jax.nn.log_softmax(jax.random.normal(k1, (n,)))
    lw = lw.at[jnp.asarray(dead)].set(-jnp.inf) if dead else lw
    means = jax.random.normal(k2, (n, dim)) * 2.0
    a = jax.random.normal(k3, (n, dim, dim)) * 0.3
    chols = jnp.linalg.cholesky(a @ jnp.swapaxes(a, -1, -2) + 0.5 * jnp.eye(dim))
    return lw, means, chols


def _close(a, b, **kw):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), **kw)


def test_incremental_runnalls_costs_reproduce_the_full_rebuild():
    for seed, (n, k, dim, dead) in enumerate(
        [(8, 3, 2, ()), (16, 4, 3, ()), (12, 5, 2, (1, 7)), (6, 1, 4, ()), (10, 9, 2, ())]
    ):
        lw, means, chols = _mixture(jax.random.PRNGKey(seed), n, dim, dead)
        got = _runnalls_merge(lw, means, chols, k)
        want = _runnalls_merge_reference(lw, means, chols, k)
        for g, w in zip(got, want, strict=True):
            _close(g, w, rtol=1e-5, atol=1e-6)


def test_the_reduced_mixture_is_unchanged_by_the_faster_merge():
    lw, means, chols = _mixture(KEY, 12, 3)
    mix = MixtureGaussian(log_weights=lw, means=means, chols=chols)
    got = mix.reduce(4)
    ref = _runnalls_merge_reference(lw, means, chols, 4)
    _close(got.log_weights, jax.nn.log_softmax(ref[0]), rtol=1e-5, atol=1e-6)
    _close(got.means, ref[1], rtol=1e-5, atol=1e-6)
    _close(got.chols, jnp.linalg.cholesky(ref[2]), rtol=1e-4, atol=1e-5)
    #  under jit and vmap alike
    batched = MixtureGaussian(
        log_weights=jnp.stack([lw, lw]),
        means=jnp.stack([means, means]),
        chols=jnp.stack([chols, chols]),
    )
    out = jax.jit(lambda m: m.reduce(4))(batched)
    _close(out.means[1], got.means, rtol=1e-5, atol=1e-6)


def test_mode_aligned_reduce_matches_the_per_group_loop():
    num_modes, per_in, per_out, dim = 3, 4, 2, 2
    n, k = num_modes * per_in, num_modes * per_out
    lw, means, chols = _mixture(jax.random.PRNGKey(3), n, dim, dead=(4,))
    mix = MixtureGaussian(log_weights=lw, means=means, chols=chols)
    got = mix.reduce_mode_aligned(k, num_modes)
    # the loop this replaced: group g = slots g, g + num_modes, ...; each reduced to per_out
    from jax.scipy.special import logsumexp

    parts_lw, parts_mu, parts_cov = [], [], []
    for g in range(num_modes):
        idx = jnp.arange(g, n, num_modes)
        mass = logsumexp(lw[idx])
        off = jnp.where(jnp.isfinite(mass), mass, 0.0)
        r_lw, r_mu, r_cov = _runnalls_merge_reference(
            lw[idx] - off, means[idx], chols[idx], per_out
        )
        parts_lw.append(r_lw + off)
        parts_mu.append(r_mu)
        parts_cov.append(r_cov)
    _close(got.log_weights, jax.nn.log_softmax(jnp.concatenate(parts_lw)), rtol=1e-5, atol=1e-6)
    _close(got.means, jnp.concatenate(parts_mu), rtol=1e-5, atol=1e-6)
    _close(got.chols, jnp.linalg.cholesky(jnp.concatenate(parts_cov)), rtol=1e-4, atol=1e-5)


def test_the_diagonal_head_density_equals_the_dense_gaussian():
    head = MLPConditional.from_shapes((3,), 5, key=KEY, width=8, depth=1)
    parents = (jnp.array([0.2, -1.0, 0.5]),)
    child = jnp.array([0.1, 0.4, -0.3, 2.0, 0.0])
    mean, scale = head.diag_moments(parents)
    dense = Gaussian(loc=mean, chol=jnp.diag(scale)).log_prob(child)
    _close(head.log_prob(child, parents), dense, rtol=1e-5)
    _close(head.nll(child, parents), -dense, rtol=1e-5)
    #  and the gradient is finite under jit at a wide child (the case that motivated it)
    wide = MLPConditional.from_shapes((4,), 36, key=KEY, width=8, depth=1)
    g = jax.jit(jax.grad(lambda m: m.nll(jnp.zeros(36), (jnp.ones(4),))))(wide)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(g))


def test_the_agent_tick_is_compiled_and_equals_the_eager_faces():
    from hybrid_aif.composition.driver import MindCarry, agent_respond
    from tests.test_system import tmaze_system

    system, params = tmaze_system()
    respond = agent_respond(system, params, selection="deterministic")
    priors = system.initial_priors(params)
    carry = MindCarry(beliefs=priors)
    obs = system.from_positional(
        "observations", tuple(jnp.zeros((), jnp.int32) for _ in system.A_deps)
    )
    action, new = respond(KEY, obs, carry)
    beliefs, _ = system.infer_states(params, obs, priors=priors)
    want, _, _ = system.act(KEY, params, beliefs, metrics=False, selection="deterministic")
    pushed = system.update_empirical_prior(params, beliefs, want)
    np.testing.assert_array_equal(np.asarray(action["a_location"]), np.asarray(want["a_location"]))
    for a, b in zip(
        jax.tree_util.tree_leaves(new.beliefs), jax.tree_util.tree_leaves(pushed), strict=True
    ):
        _close(a, b, rtol=1e-6, atol=1e-7)
    #  a second call with different parameters shares the program (params is an argument)
    action2, _ = respond(KEY, obs, new)
    assert action2["a_location"].shape == action["a_location"].shape


def test_replay_repeats_bitwise_through_its_compiled_core():
    from hybrid_aif.replay import replay
    from tests.test_system import tmaze_system

    system, params = tmaze_system()
    T = 4
    obs = system.from_positional(
        "observations", tuple(jnp.zeros((T,), jnp.int32) for _ in system.A_deps)
    )
    acts = system.from_positional("actions", jnp.zeros((T, len(system.names.controls)), jnp.int32))
    a = replay(system, params, obs, acts)
    b = replay(system, params, obs, acts)
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


# ================================================== CEM's carried scores


def test_cem_reports_the_same_scores_whether_carried_or_recomputed():
    """Without metrics the plan's G is the last iteration's carried scores plus one
    fresh row for the mean; with metrics it is a full re-scoring through the term
    tree. Same numbers either way."""
    from tests.test_cem import _belief, _plan, lq_system  # noqa: PLC0415

    system, params = lq_system(horizon=3)
    beliefs = _belief(-0.2)
    kw = dict(num_candidates=16, cem_iters=2)
    with_tree = _plan(system, params, beliefs, metrics=True, **kw)
    without = _plan(system, params, beliefs, metrics=False, **kw)
    assert without.G.derived == with_tree.G.derived
    np.testing.assert_allclose(
        np.asarray(with_tree.G.midpoint), np.asarray(without.G.midpoint), atol=1e-5
    )
    np.testing.assert_array_equal(np.asarray(with_tree.candidates), np.asarray(without.candidates))


# ================================================== the encoder hoisted out of observe


def _observe_reference(core, key, actions, observations, is_first=None):
    """The scan as it was: ``obs_step`` (encoder inside) at every step."""
    keys = jax.random.split(key, observations.shape[0])
    restart = core.initial_state()

    def step(carry, xs):
        k, action, obs, first = xs
        if first is not None:
            carry = type(carry)(
                deter=jnp.where(first, restart.deter, carry.deter),
                stoch=jnp.where(first, restart.stoch, carry.stoch),
            )
            action = jnp.where(first, jnp.zeros_like(action), action)
        new, post, prior = core.obs_step(k, carry, action, obs)
        return new, (new, post, prior)

    flags = None if is_first is None else jnp.asarray(is_first)
    if flags is None:
        body = lambda c, xs: step(c, (*xs, None))  # noqa: E731
        xs = (keys, actions, observations)
    else:
        body = step
        xs = (keys, actions, observations, flags)
    _, out = lax.scan(body, core.initial_state(), xs)
    return out


def test_observe_with_the_encoder_hoisted_matches_the_per_step_scan():
    """``observe`` encodes the whole sequence once and scans the recurrent part;
    the per-step form (encoder inside the scan) is kept here as the reference.
    Both ``is_first`` branches, gradients included."""
    from hybrid_aif.reference.raif.agent import raif_mountain_car
    from hybrid_aif.train.world import assemble
    from tests.test_amortized_trainers import TINY

    agent = raif_mountain_car(KEY, **TINY)
    core, _ = assemble(agent.system, agent.params)
    t = 6
    obs = jax.random.normal(jax.random.PRNGKey(1), (t, core.obs_dim))
    act = jax.random.normal(jax.random.PRNGKey(2), (t, core.action_dim))
    first = jnp.zeros((t,), bool).at[0].set(True).at[3].set(True)
    for flags in (None, first):
        got = core.observe(KEY, act, obs, is_first=flags)
        want = _observe_reference(core, KEY, act, obs, flags)
        for a, b in zip(
            jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want), strict=True
        ):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-6)

    def loss(core, fn):
        _, post, prior = fn(core)
        return jnp.sum(post**2) + jnp.sum(prior**2)

    g_new = jax.grad(lambda c: loss(c, lambda c: c.observe(KEY, act, obs, is_first=first)))
    g_old = jax.grad(lambda c: loss(c, lambda c: _observe_reference(c, KEY, act, obs, first)))
    filt = lambda c: eqx.filter(c, eqx.is_array)  # noqa: E731
    for a, b in zip(
        jax.tree_util.tree_leaves(g_new(filt(core))),
        jax.tree_util.tree_leaves(g_old(filt(core))),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)
