"""Our hot formulas against **line-for-line ports** of theirs, on random inputs.

Design: ``docs/design/amortization.md``'s "A12 implementation note: chunk 2d",
§(a‴) — the semantic half of the 2026-09-02 parity audit, whose config half is
``tests/test_oracle_parity.py``.

**Why this file exists, and why it is written this way.** Chunk 2d's two worst bugs
— a 1024× info-gain leaf and an off-by-one in the ensemble's actions — were both
found by *transcribing* the authors' code and comparing, and both had been invisible
to a test that reimplemented our own expression (that is
``test_ensemble_disagreement_matches_its_formula``'s recorded failure mode: it could
only ever confirm that the row agreed with itself). So the ports below are **theirs,
not ours restated**: each is a paste of their source with `tfd`/`nj`/`sg` swapped for
their JAX equivalents and nothing else moved, with the file:line above it, and the
resolved config branch named where their code has an `if`.

The five formulas the 2026-09-02 sweep covered, and what it found:

======================================  =====================================
their λ-return recursion                 MATCHED (its ``disc`` input did not)
their ``Moments(impl='perc_ema')``       MATCHED
their actor-loss assembly                FIXED (two divergences)
their slow-critic ``logprob`` reg        FIXED (the first-update copy)
their ``dyn_loss``/``rep_loss`` mix      MATCHED (free bits before the mix)
======================================  =====================================

A port is a liability the moment it drifts from the source, so every one here is
*small* and every one carries its citation. Where the comparison needs a whole
rollout (the actor loss) the port takes the rollout's arrays as inputs rather than
re-deriving them: what is under test is the **assembly**, and the pieces it
assembles have tests of their own.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions.neural import (
    factored_categorical_kl,
    kl_balance,
    symlog_disc_mean,
    symlog_disc_nll,
)
from hybrid_aif.train.imagine import (
    _cont_mode,
    _slow_update,
    init_retnorm,
    lambda_return,
    perc_ema,
    rollout_weight,
)

#: The resolved config's values, for the ports (``logs/oracle_state_mc_seed0/config.yaml``).
HORIZON = 333
DISCOUNT = 1.0 - 1.0 / HORIZON
RETURN_LAMBDA = 0.95
ACTENT = 3e-4
WMENT = 3e-4
ACTOR_REFRESH = 0.5
SLOWREG = 1.0
SLOW_FRACTION = 0.02
SLOW_PERIOD = 1
FREE = 1.0
DYN_SCALE = 0.5
REP_SCALE = 0.1
IMAG_HORIZON = 8

sg = jax.lax.stop_gradient


# ============================================================ 1. the lambda-return


def their_score_return(G, value, cont):
    """``agent.py:568-576`` (``GFunction.score``), verbatim but for ``self.config``::

        discount = 1 - 1 / self.config.horizon
        disc = traj['cont'][1:] * discount
        value = self.net(traj).mean()
        vals = [value[-1]]
        interm = G + disc * value[1:] * (1 - self.config.return_lambda)
        for t in reversed(range(len(disc))):
          vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
        ret = jnp.stack(list(reversed(vals))[:-1])

    ``cont`` here is their ``traj['cont']``, length ``H+1``: ``[first_cont, c_1 … c_H]``.
    A Python loop, deliberately — a scan would be *our* idiom creeping into the port.
    """
    disc = cont[1:] * DISCOUNT
    vals = [value[-1]]
    interm = G + disc * value[1:] * (1 - RETURN_LAMBDA)
    for t in reversed(range(len(disc))):
        vals.append(interm[t] + disc[t] * RETURN_LAMBDA * vals[-1])
    return jnp.stack(list(reversed(vals))[:-1])


def test_the_lambda_return_is_their_recursion_on_random_inputs():
    """Ours against their loop, including the bootstrap leaf and which end is dropped.

    The three ways this recursion is usually got wrong — the off-by-one between
    ``value`` and ``reward``, what bootstraps at the horizon, and whether the first or
    the last element of the reversed list is dropped — are all decided by comparing
    against the loop rather than by re-deriving it.
    """
    rng = np.random.default_rng(0)
    for _ in range(20):
        h = int(rng.integers(1, 12))
        value = jnp.asarray(rng.normal(size=(h + 1,)) * 5.0, jnp.float32)
        g = jnp.asarray(rng.normal(size=(h,)) * 2.0, jnp.float32)
        cont_head = jnp.asarray(rng.random(size=(h,)) < 0.7, jnp.float32)  # their modes
        first_cont = jnp.asarray(float(rng.integers(0, 2)), jnp.float32)
        theirs = their_score_return(g, value, jnp.concatenate([first_cont[None], cont_head]))
        #  ours takes `disc` already formed, which is `discount * cont(s_1 … s_H)` —
        #  note their `traj['cont'][1:]` drops `first_cont`, so it never enters `disc`
        ours = lambda_return(value, g, DISCOUNT * cont_head, RETURN_LAMBDA)
        np.testing.assert_allclose(np.asarray(ours), np.asarray(theirs), rtol=1e-5, atol=1e-6)


def test_the_continuation_that_discounts_the_return_is_their_hard_mode():
    """``self.heads['cont'](traj).mode()`` (``agent.py:256``) — ``tfd.Bernoulli.mode()``.

    Divergence 10, closed 2026-09-02: ours passed ``sigmoid`` straight through, which
    *discounts* a trajectory the model believes has ended where theirs **truncates**
    it. The gap is not small — at a cont probability of 0.6 theirs discounts by 0.997
    and ours by 0.598 — and it also gave the actor a gradient path through ``disc``
    that theirs (a threshold, so gradient-free) does not have.
    """
    probability = jnp.asarray([0.0, 0.1, 0.49, 0.5, 0.51, 0.9, 1.0])
    #  theirs: tfd.Bernoulli(logits).mode() == (probs > 0.5)
    theirs = (probability > 0.5).astype(jnp.float32)
    np.testing.assert_array_equal(np.asarray(_cont_mode(probability)), np.asarray(theirs))
    #  and the mode carries no gradient, which is half of why it matters
    grad = jax.grad(lambda p: jnp.sum(_cont_mode(p)))(probability)
    np.testing.assert_array_equal(np.asarray(grad), np.zeros_like(np.asarray(grad)))


# ================================================================ 2. the retnorm


def their_moments_perc_ema(low, high, x):
    """``jaxutils.py:307-310`` (``update``) then ``:338-341`` (``stats``), verbatim::

        low, high = per(x, self.perclo), per(x, self.perchi)
        self.low.write(m * self.low.read() + (1 - m) * low)
        self.high.write(m * self.high.read() + (1 - m) * high)
        ...
        offset = self.low.read()
        invscale = jnp.maximum(1 / self.max, self.high.read() - self.low.read())

    Resolved: single device, so ``per = jnp.percentile`` over the **whole** array;
    ``impl = 'perc_ema'`` and not ``perc_ema_corr``, so there is deliberately **no**
    bias correction; ``m = decay = 0.99``, ``max = 1.0``, both variables init to zero.
    """
    m = 0.99
    lo, hi = jnp.percentile(x, 5.0), jnp.percentile(x, 95.0)
    low = m * low + (1 - m) * lo
    high = m * high + (1 - m) * hi
    return low, high, low, jnp.maximum(1 / 1.0, high - low)


def test_the_return_normalizer_is_their_percentile_ema_step_for_step():
    """Ours against their ``Moments``, over a *sequence* of updates.

    A single update would not distinguish the two readings that matter: whether the
    percentiles are taken over the whole array or per-something, and whether the
    first update is bias-corrected. Theirs is whole-array and **not** corrected — so
    ``invscale`` sits at its ``1/max`` floor of 1.0 for many early steps, which is a
    real property of their early training and not a bug to fix.
    """
    rng = np.random.default_rng(1)
    state = init_retnorm()
    low, high = jnp.zeros(()), jnp.zeros(())
    for step in range(12):
        x = jnp.asarray(rng.normal(size=(8, 64)) * (1.0 + step) + step, jnp.float32)
        low, high, t_offset, t_invscale = their_moments_perc_ema(low, high, x)
        state, offset, invscale = perc_ema(state, x, 5.0, 95.0, 0.99, 1.0)
        np.testing.assert_allclose(float(offset), float(t_offset), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(float(invscale), float(t_invscale), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(float(state[0]), float(low), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(float(state[1]), float(high), rtol=1e-6, atol=1e-7)
    #  the un-corrected EMA's own signature: after 12 steps of a spread that is
    #  never small, `high - low` is still far below where a corrected EMA would be
    assert float(state[1] - state[0]) < float(jnp.percentile(x, 95.0) - jnp.percentile(x, 5.0))


# ============================================================== 3. the actor loss


def their_weight(cont, first_cont):
    """``agent.py:257-259``, verbatim::

        traj['cont'] = jnp.concatenate([first_cont[None], cont[1:]], 0)
        traj['weight'] = jnp.cumprod(discount * traj['cont'], 0) / discount

    ``cont`` is their head's modes at ``s_1 … s_H``; the result is length ``H+1`` and
    the loss uses ``[:-1]``.
    """
    full = jnp.concatenate([first_cont[None], cont], 0)
    return jnp.cumprod(DISCOUNT * full, 0) / DISCOUNT


def their_actor_loss(adv, ent, wment_ent, weight, start_logpi, can_self_imitate):
    """``agent.py:463-484`` (``ImagActorCritic.loss``), verbatim from ``self.grad``::

        loss = {'backprop': -adv, 'reinforce': -logpi * sg(adv)}[self.grad]
        ent = policy.entropy()[:-1]
        loss -= self.config.actent * ent
        loss *= sg(traj['weight'])[:-1]
        wment = self.get_dist(traj).entropy()[1:]
        loss -= self.config.wment * wment
        policy2 = self.actor(sg(start))
        logpi2 = policy2.log_prob(sg(start['action']))
        actor_refresh = -logpi2 * sg(start['can_self_imitate'])
        actor_refresh *= self.config.loss_scales.actor_refresh
        loss *= self.config.loss_scales.actor
        return loss.mean() + actor_refresh.mean(), metrics

    Resolved: ``actor_grad_cont: backprop`` so the branch is ``-adv``;
    ``loss_scales.actor: 1.0``; ``loss_scales.actor_refresh: 0.5``. The
    **parenthesization** is the content — ``weight`` multiplies the advantage and the
    *policy* entropy but not ``wment``, because the multiply happens before that
    subtraction.
    """
    loss = -adv
    loss = loss - ACTENT * ent
    loss = loss * sg(weight)
    loss = loss - WMENT * wment_ent
    actor_refresh = -start_logpi * sg(can_self_imitate) * ACTOR_REFRESH
    return loss.mean() + actor_refresh.mean()


def our_actor_loss(adv, ent, wment_ent, weight, start_logpi, can_self_imitate):
    """``train/imagine.py``'s assembly, lifted out of ``loss_fn`` unchanged."""
    per_step = -adv - ACTENT * ent
    bc = -start_logpi * can_self_imitate
    return (
        jnp.mean(per_step * sg(weight)) - WMENT * jnp.mean(wment_ent) + ACTOR_REFRESH * jnp.mean(bc)
    )


def test_the_rollout_weight_is_their_cumulative_product():
    """``cumprod(discount·cont)/discount``, truncated — and the truncation is the point.

    A cumulative product, so one step whose continuation mode is 0 zeroes that step
    **and every later one**. Divergence 11: we took plain means.
    """
    rng = np.random.default_rng(2)
    for _ in range(20):
        n, h = 4, IMAG_HORIZON
        cont = jnp.asarray(rng.random(size=(n, h)) < 0.75, jnp.float32)
        first = jnp.asarray(rng.random(size=(n,)) < 0.9, jnp.float32)
        ours = rollout_weight(cont, DISCOUNT, first)
        theirs = jnp.stack([their_weight(cont[i], first[i])[:-1] for i in range(n)])
        np.testing.assert_allclose(np.asarray(ours), np.asarray(theirs), rtol=1e-6, atol=1e-7)
    #  the truncation, stated as a case: a zero at step 3 kills 3 onwards
    cont = jnp.asarray([[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    weight = np.asarray(rollout_weight(cont, DISCOUNT, jnp.ones((1,))))
    assert weight[0, 0] == 1.0 and weight[0, 2] > 0.0
    assert np.all(weight[0, 3:] == 0.0)


def test_the_actor_loss_assembly_is_theirs_on_random_inputs():
    """The whole assembly, term by term, against their four lines.

    What this pins that a per-term test could not: the *order* of the multiply.
    ``weight`` scales the advantage and ``actent`` but not ``wment``, and the
    self-imitation term is a separate mean over a different shape added outside
    ``loss_scales.actor``. Get any of those wrong and the scalar moves.
    """
    rng = np.random.default_rng(3)
    for _ in range(20):
        n, h = 5, IMAG_HORIZON
        adv = jnp.asarray(rng.normal(size=(n, h)), jnp.float32)
        ent = jnp.asarray(rng.normal(size=(n, h)) + 2.0, jnp.float32)
        wment_ent = jnp.asarray(rng.normal(size=(n, h)) + 5.0, jnp.float32)
        cont = jnp.asarray(rng.random(size=(n, h)) < 0.75, jnp.float32)
        first = jnp.asarray(rng.random(size=(n,)) < 0.9, jnp.float32)
        logpi = jnp.asarray(rng.normal(size=(n,)) - 1.0, jnp.float32)
        mask = jnp.asarray(rng.random(size=(n,)), jnp.float32)
        weight = rollout_weight(cont, DISCOUNT, first)
        theirs = jnp.mean(
            jnp.stack(
                [
                    their_actor_loss(
                        adv[i], ent[i], wment_ent[i], weight[i], logpi[i : i + 1], mask[i : i + 1]
                    )
                    for i in range(n)
                ]
            )
        )
        ours = our_actor_loss(adv, ent, wment_ent, weight, logpi, mask)
        np.testing.assert_allclose(float(ours), float(theirs), rtol=1e-5, atol=1e-6)


def test_the_weight_is_not_absorbable_into_a_plain_mean():
    """The divergence the fix closed, kept as a number so it cannot come back quietly."""
    rng = np.random.default_rng(4)
    n, h = 5, IMAG_HORIZON
    adv = jnp.asarray(rng.normal(size=(n, h)), jnp.float32)
    ent = jnp.asarray(rng.normal(size=(n, h)) + 2.0, jnp.float32)
    zero = jnp.zeros((n, h), jnp.float32)
    cont = jnp.asarray(rng.random(size=(n, h)) < 0.75, jnp.float32)
    weight = rollout_weight(cont, DISCOUNT, jnp.ones((n,)))
    weighted = our_actor_loss(adv, ent, zero, weight, jnp.zeros((n,)), jnp.zeros((n,)))
    unweighted = our_actor_loss(adv, ent, zero, jnp.ones((n, h)), jnp.zeros((n,)), jnp.zeros((n,)))
    assert abs(float(weighted) - float(unweighted)) > 1e-3


# =========================================================== 4. the slow critic


def their_slow_updater(dst, src, updates):
    """``jaxutils.py:483-495`` (``SlowUpdater.__call__``), verbatim::

        updates = self.updates.read()
        need_init = (updates == 0).astype(jnp.float32)
        need_update = (updates % self.period == 0).astype(jnp.float32)
        mix = jnp.clip(1.0 * need_init + self.fraction * need_update, 0, 1)
        self.dst.putm(tree_map(lambda s, d: mix * s + (1 - mix) * d, source, self.dst.getm()))
        self.updates.write(updates + 1)

    Resolved: ``slow_critic_fraction 0.02``, ``slow_critic_update 1``.
    """
    need_init = (updates == 0).astype(jnp.float32)
    need_update = (updates % SLOW_PERIOD == 0).astype(jnp.float32)
    mix = jnp.clip(1.0 * need_init + SLOW_FRACTION * need_update, 0, 1)
    return jax.tree_util.tree_map(lambda s, d: mix * s + (1 - mix) * d, src, dst), updates + 1


def test_the_slow_update_is_their_slow_updater():
    """Ours against their mixer over a run of steps, from ``updates = 0``.

    Divergence 13: ``need_init`` makes the first ``mix`` exactly ``1.0`` — a hard
    copy — and we EMA-ed from step one, so ``slowreg`` pulled the critic toward an
    unrelated net at full weight for the ~150 steps a 0.02 EMA needs to catch up.
    """
    rng = np.random.default_rng(5)
    slow = {"w": jnp.asarray(rng.normal(size=(4, 3)), jnp.float32)}
    theirs, ours = slow, slow
    t_updates = jnp.zeros((), jnp.int32)
    o_updates = jnp.zeros((), jnp.int32)
    for _ in range(8):
        fast = {"w": jnp.asarray(rng.normal(size=(4, 3)), jnp.float32)}
        theirs, t_updates = their_slow_updater(theirs, fast, t_updates)
        ours, o_updates = _slow_update(ours, fast, o_updates, SLOW_FRACTION, SLOW_PERIOD)
        np.testing.assert_allclose(
            np.asarray(ours["w"]), np.asarray(theirs["w"]), rtol=1e-6, atol=1e-7
        )
        assert int(o_updates) == int(t_updates)


def their_critic_loss(logits, target, slow_logits, bins, weight):
    """``agent.py:534-545`` (``GFunction.loss``), resolved to ``critic_slowreg: logprob``::

        dist = self.net(traj)
        loss = -dist.log_prob(sg(target))
        reg = -dist.log_prob(sg(self.slow(traj).mean()))
        loss += self.config.loss_scales.slowreg * reg
        loss = (loss * sg(traj['weight'])).mean()
        loss *= self.config.loss_scales.critic

    ``self.slow(traj).mean()`` is ``DiscDist.mean() = symexp(Σ softmax·bins)`` — a
    *scalar in the value's own units*, not a distribution — so the regularizer is the
    asymmetric "logprob of the slow copy's point estimate under the live critic".
    """
    slow_mean = jax.vmap(lambda lg: symlog_disc_mean(lg, bins))(slow_logits)
    nll = jax.vmap(lambda lg, y: symlog_disc_nll(lg, y, bins))
    loss = nll(logits, sg(target)) + SLOWREG * nll(logits, sg(slow_mean))
    return jnp.mean(loss * sg(weight))


def test_the_critic_loss_is_their_weighted_two_hot_plus_slowreg():
    """Ours against theirs, with the weight where theirs puts it.

    Theirs is ``mean((nll_target + slowreg·nll_slow) · weight)``, which is *not*
    ``mean(nll_target) + mean(nll_slow)`` — the form we had. Note also what does NOT
    diverge: the regularizer's shape (logprob of the slow **mean** under the live
    distribution, stop-gradiented) was already exact.
    """
    rng = np.random.default_rng(6)
    n, num_bins = 24, 255
    bins = jnp.linspace(-20.0, 20.0, num_bins)
    logits = jnp.asarray(rng.normal(size=(n, num_bins)), jnp.float32)
    slow_logits = jnp.asarray(rng.normal(size=(n, num_bins)), jnp.float32)
    target = jnp.asarray(rng.normal(size=(n,)) * 3.0, jnp.float32)
    weight = jnp.asarray(rng.random(size=(n,)), jnp.float32)
    slow_mean = jax.vmap(lambda lg: symlog_disc_mean(lg, bins))(slow_logits)
    nll = jax.vmap(lambda lg, y: symlog_disc_nll(lg, y, bins))
    ours = jnp.mean((nll(logits, target) + SLOWREG * nll(logits, slow_mean)) * weight)
    theirs = their_critic_loss(logits, target, slow_logits, bins, weight)
    np.testing.assert_allclose(float(ours), float(theirs), rtol=1e-6, atol=1e-6)


def test_their_two_hot_log_prob_is_our_symlog_disc_nll():
    """``jaxutils.py:201-219`` (``DiscDist.log_prob``), verbatim, against ours.

    The below/above/clip/interpolate block, which is the piece a reimplementation
    gets wrong at the bin edges and outside the range.
    """
    num_bins = 255
    bins = jnp.linspace(-20.0, 20.0, num_bins)

    def theirs(logits, x):
        symlog = lambda v: jnp.sign(v) * jnp.log1p(jnp.abs(v))  # noqa: E731
        x = symlog(x)
        below = (bins <= x[..., None]).astype(jnp.int32).sum(-1) - 1
        above = len(bins) - (bins > x[..., None]).astype(jnp.int32).sum(-1)
        below = jnp.clip(below, 0, len(bins) - 1)
        above = jnp.clip(above, 0, len(bins) - 1)
        equal = below == above
        dist_to_below = jnp.where(equal, 1, jnp.abs(bins[below] - x))
        dist_to_above = jnp.where(equal, 1, jnp.abs(bins[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            jax.nn.one_hot(below, len(bins)) * weight_below[..., None]
            + jax.nn.one_hot(above, len(bins)) * weight_above[..., None]
        )
        log_pred = logits - jax.scipy.special.logsumexp(logits, -1, keepdims=True)
        return (target * log_pred).sum(-1)

    rng = np.random.default_rng(7)
    logits = jnp.asarray(rng.normal(size=(32, num_bins)), jnp.float32)
    #  in range, at the extremes, and OUTSIDE the bins on both sides
    values = jnp.asarray(
        np.concatenate([rng.normal(size=(28,)) * 5.0, [0.0, -1e6, 1e6, 20.0]]), jnp.float32
    )
    ours = jax.vmap(lambda lg, y: symlog_disc_nll(lg, y, bins))(logits, values)
    np.testing.assert_allclose(
        np.asarray(ours), -np.asarray(theirs(logits, values)), rtol=1e-5, atol=1e-5
    )


# ============================================================ 5. kl_balance


def their_kl_mix(post_logit, prior_logit):
    """``nets.py:251-276`` (``dyn_loss``/``rep_loss``) then ``agent.py:212-219``::

        losses['dyn'] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
        losses['rep'] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
        scaled = {k: v * self.scales[k] for k, v in losses.items()}
        model_loss = sum(scaled.values())

    with, inside each::

        loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))   # dyn
        loss = self.get_dist(post).kl_divergence(self.get_dist(sg(prior)))   # rep
        if free:
          loss = jnp.maximum(loss, free)

    Resolved: ``classes: 32`` so ``get_dist`` is ``Independent(OneHotDist(logit), 1)``
    — the KL is **summed** over the 32 stochastic variables (and over classes inside
    each) to one scalar per element — and ``free: 1.0`` on both, so the clamp is
    **per term, before** the ``0.5``/``0.1`` mix. That order is the whole risk here.
    """

    def one_hot_kl(q_logit, p_logit):
        lq = jax.nn.log_softmax(q_logit, -1)
        lp = jax.nn.log_softmax(p_logit, -1)
        #  Independent(..., 1): sum over the variables axis after the per-categorical
        #  sum over classes
        return (jnp.exp(lq) * (lq - lp)).sum(-1).sum(-1)

    dyn = one_hot_kl(sg(post_logit), prior_logit)
    rep = one_hot_kl(post_logit, sg(prior_logit))
    if FREE:
        dyn = jnp.maximum(dyn, FREE)
        rep = jnp.maximum(rep, FREE)
    return DYN_SCALE * dyn + REP_SCALE * rep, dyn, rep


def test_kl_balance_clamps_before_the_mix_as_theirs_does():
    """The reviewer's named risk: free bits **per term, before** ``0.5``/``0.1``.

    Verified against their code rather than argued: ``dyn_loss`` and ``rep_loss`` each
    apply ``jnp.maximum(loss, free)`` to their own already-summed scalar, and
    ``agent.py``'s ``scaled = {k: v * self.scales[k]}`` happens afterwards. The
    scenarios below straddle the clamp deliberately — a random pair of logits gives
    a KL far above 1.0, so the interesting case (both terms *under* the floor) has to
    be constructed.
    """
    rng = np.random.default_rng(8)
    for scale in (1e-3, 0.05, 1.0, 4.0):  # small scale -> KL under the free floor
        post = jnp.asarray(rng.normal(size=(6, 32, 32)) * scale, jnp.float32)
        prior = jnp.asarray(rng.normal(size=(6, 32, 32)) * scale, jnp.float32)
        theirs, t_dyn, t_rep = their_kl_mix(post, prior)
        ours = jax.vmap(
            lambda a, b: kl_balance(a, b, dyn_scale=DYN_SCALE, rep_scale=REP_SCALE, free_nats=FREE)
        )(post, prior)
        np.testing.assert_allclose(np.asarray(ours.total), np.asarray(theirs), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(ours.dyn), np.asarray(t_dyn), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(ours.rep), np.asarray(t_rep), rtol=1e-5, atol=1e-6)
        if scale == 1e-3:  # the case that separates clamp-before from clamp-after
            assert float(jnp.max(ours.total)) == pytest.approx(0.6, abs=1e-5)


def test_the_categorical_kl_is_summed_over_variables_not_meaned():
    """``Independent(OneHotDist(logit), 1)``: **summed** over the 32 variables.

    The 1024× info-gain bug (divergence 1) was a sum where theirs meaned; this is the
    mirror-image risk in the KL, where theirs sums and a mean would be 32× too small.
    Pinned because the free-bits floor of 1.0 only means what it means at the right
    scale.
    """
    rng = np.random.default_rng(9)
    post = jnp.asarray(rng.normal(size=(32, 32)), jnp.float32)
    prior = jnp.asarray(rng.normal(size=(32, 32)), jnp.float32)
    lq = jax.nn.log_softmax(post, -1)
    lp = jax.nn.log_softmax(prior, -1)
    per_variable = (jnp.exp(lq) * (lq - lp)).sum(-1)  # (32,)
    ours = factored_categorical_kl(post, prior)
    np.testing.assert_allclose(float(ours), float(jnp.sum(per_variable)), rtol=1e-6)
    assert float(ours) > 10.0 * float(jnp.mean(per_variable))  # not a mean, by a factor of 32


# ================================================ 6. the actor / ig_head (MLP + Dist)


def their_linear(x, kernel, bias, norm_scale, norm_bias, act=jax.nn.silu):
    """``nets.py:659-690`` (``Linear.__call__``), verbatim, resolved to one layer of
    their actor/ig_head trunk: ``norm: layer`` (so ``self._bias = bias and norm ==
    'none'`` is **False** — no bias), then ``Norm('layer')`` — ``jax.nn.standardize``
    at ``epsilon=1e-3`` times a learned scale plus a learned bias — then ``act``::

        x = x @ kernel                       # no bias: norm != 'none'
        x = self.get('norm', Norm, self._norm)(x)
        x = self._act(x)

    ``bias`` is accepted and asserted ``None`` so the port *states* the dropped-bias
    branch instead of hiding it.
    """
    assert bias is None
    x = x @ kernel
    x = jax.nn.standardize(x, axis=-1, epsilon=1e-3)
    x = x * norm_scale + norm_bias
    return act(x)


def their_dist_normal(x, w_out, b_out, w_std, b_std, minstd, maxstd):
    """``nets.py:553-580`` (``Dist.inner``), the ``'normal'`` branch, verbatim::

        out = self.get('out', Linear, ...)(inputs)          # norm-free, bias kept
        std = self.get('std', Linear, ...)(inputs)
        lo, hi = self._minstd, self._maxstd
        std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
        dist = tfd.Normal(jnp.tanh(out), std)
        dist = tfd.Independent(dist, len(self._shape))

    Returns ``(mean, std)``; the Independent-Normal log-density is a separate
    two-liner below because the *formula* is what the comparison pins.
    """
    out = x @ w_out + b_out
    std = x @ w_std + b_std
    std = (maxstd - minstd) * jax.nn.sigmoid(std + 2.0) + minstd
    return jnp.tanh(out), std


def their_independent_normal_log_prob(value, mean, std):
    """``tfd.Independent(tfd.Normal(mean, std), 1).log_prob(value)``: the per-dimension
    Normal log-density, summed."""
    z = (value - mean) / std
    return jnp.sum(-0.5 * z**2 - jnp.log(std) - 0.5 * jnp.log(2.0 * jnp.pi))


def _ours_with_their_weights(rng, in_dim, units, layers, child, minstd, maxstd):
    """An ``MLPConditional(norm=True, head="bounded")`` and the verbatim-port
    parameters it was loaded with, drawn at random.

    Their trunk parameters are drawn as *their* shapes — ``kernel (in, out)``,
    per-layer norm scale/bias — and transposed into ours; the two head linears
    (``out``/``std``) are fused into our final ``2·child`` layer, mean half first.
    So the comparison is theirs-vs-ours on identical weights, not init-vs-init.
    """
    import equinox as eqx

    from hybrid_aif.distributions.neural import MLPConditional

    sizes = [in_dim] + [units] * layers
    kernels = [
        jnp.asarray(rng.normal(size=(a, b)) * 0.3, jnp.float32)
        for a, b in zip(sizes[:-1], sizes[1:], strict=True)
    ]
    scales = [
        jnp.asarray(rng.normal(size=(units,)) * 0.2 + 1.0, jnp.float32) for _ in range(layers)
    ]
    nbiases = [jnp.asarray(rng.normal(size=(units,)) * 0.1, jnp.float32) for _ in range(layers)]
    w_out = jnp.asarray(rng.normal(size=(units, child)) * 0.3, jnp.float32)
    b_out = jnp.asarray(rng.normal(size=(child,)) * 0.1, jnp.float32)
    w_std = jnp.asarray(rng.normal(size=(units, child)) * 0.3, jnp.float32)
    b_std = jnp.asarray(rng.normal(size=(child,)) * 0.1, jnp.float32)

    ours = MLPConditional.from_shapes(
        (in_dim,),
        child,
        key=jax.random.PRNGKey(0),
        width=units,
        depth=layers,
        activation="silu",
        norm=True,
        head="bounded",
        minstd=minstd,
        maxstd=maxstd,
    )
    new_layers = [
        eqx.tree_at(lambda m: m.weight, ours.layers[i], kernels[i].T) for i in range(layers)
    ]
    new_layers.append(
        eqx.tree_at(
            lambda m: (m.weight, m.bias),
            ours.layers[-1],
            (jnp.concatenate([w_out.T, w_std.T], 0), jnp.concatenate([b_out, b_std])),
        )
    )
    norms = tuple(
        eqx.tree_at(lambda n: (n.weight, n.bias), ours.norms[i], (scales[i], nbiases[i]))
        for i in range(layers)
    )
    ours = eqx.tree_at(lambda m: (m.layers, m.norms), ours, (tuple(new_layers), norms))
    port = dict(
        kernels=kernels,
        scales=scales,
        nbiases=nbiases,
        w_out=w_out,
        b_out=b_out,
        w_std=w_std,
        b_std=b_std,
    )
    return ours, port


def _their_forward(x, port, minstd, maxstd):
    h = x
    for kernel, scale, nbias in zip(port["kernels"], port["scales"], port["nbiases"], strict=True):
        h = their_linear(h, kernel, None, scale, nbias)
    return their_dist_normal(
        h, port["w_out"], port["b_out"], port["w_std"], port["b_std"], minstd, maxstd
    )


def test_the_bounded_head_is_their_mlp_plus_dist_normal_on_shared_weights():
    """Divergences 14 and 16, closed: ours against their ``MLP`` + ``Dist('normal')``
    end to end, on identical weights and random inputs.

    What the end-to-end form pins that a per-piece test could not: the **placement**
    — ``linear → norm → act`` with the hidden bias dropped, the head linears
    norm-free with their biases kept — plus the head formula with its ``+2.0``
    shift, the ``tanh`` on the mean, and the (minstd, maxstd) band. The actor's
    resolved numbers (3×256 shrunk to 3×16 for speed; ``minstd 0.1, maxstd 1.0``).
    """
    rng = np.random.default_rng(10)
    ours, port = _ours_with_their_weights(
        rng, in_dim=12, units=16, layers=3, child=1, minstd=0.1, maxstd=1.0
    )
    for _ in range(10):
        x = jnp.asarray(rng.normal(size=(12,)) * 2.0, jnp.float32)
        t_mean, t_std = _their_forward(x, port, 0.1, 1.0)
        o_mean, o_std = ours.diag_moments((x,))
        np.testing.assert_allclose(np.asarray(o_mean), np.asarray(t_mean), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(o_std), np.asarray(t_std), rtol=1e-5, atol=1e-6)
        assert float(t_std[0]) > 0.1 and float(t_std[0]) < 1.0  # inside the band, strictly
        value = jnp.asarray(rng.normal(size=(1,)), jnp.float32)
        np.testing.assert_allclose(
            float(ours.log_prob(value, (x,))),
            float(their_independent_normal_log_prob(value, t_mean, t_std)),
            rtol=1e-5,
            atol=1e-5,
        )


def test_the_fixed_sigma_head_is_their_ig_head_defaults():
    """Divergence 15, closed: their ``ig_head`` declares no ``minstd``/``maxstd``, so
    ``Dist.__init__``'s defaults (1.0, 1.0) make ``std = (1−1)·σ(raw+2) + 1 ≡ 1.0``
    — constant, with an exactly-zero gradient on the σ half — against a ``tanh``
    mean. Ours at ``minstd = maxstd = 1.0``, on shared weights, plus the gradient
    property their constant implies (their std Linear exists and is never learned).
    """
    import equinox as eqx

    rng = np.random.default_rng(11)
    child = 6
    ours, port = _ours_with_their_weights(
        rng, in_dim=9, units=8, layers=2, child=child, minstd=1.0, maxstd=1.0
    )
    x = jnp.asarray(rng.normal(size=(9,)), jnp.float32)
    t_mean, t_std = _their_forward(x, port, 1.0, 1.0)
    o_mean, o_std = ours.diag_moments((x,))
    np.testing.assert_array_equal(np.asarray(t_std), np.ones((child,), np.float32))
    np.testing.assert_array_equal(np.asarray(o_std), np.ones((child,), np.float32))
    np.testing.assert_allclose(np.asarray(o_mean), np.asarray(t_mean), rtol=1e-5, atol=1e-6)
    #  σ ≡ 1.0 makes the NLL a fixed-σ scaled MSE, so the σ-half rows of the head
    #  layer get an EXACTLY zero gradient: parameters present, never learned
    grads = eqx.filter_grad(lambda m: m.diag_nll(jnp.zeros(child), (x,)))(ours)
    head = np.asarray(grads.layers[-1].weight)
    assert np.all(head[child:] == 0.0)
    assert np.abs(head[:child]).max() > 0.0


def their_initializer_normal_avg(key, shape):
    """``nets.py:740-781`` (``Initializer``), the ``dist='normal', fan='avg'`` branch
    at ``scale = 1.0``, verbatim but for ``nj.rng()`` → ``key``::

        fanin, fanout = self._fans(shape)                      # len-2 shape: as-is
        denoms = {'avg': np.mean((fanin, fanout)), ...}
        scale = self.scale / denoms[self.fan]
        std = np.sqrt(scale) / 0.87962566103423978
        value = std * jax.random.truncated_normal(nj.rng(), -2, 2, shape, f32)
    """
    fanin, fanout = shape
    scale = 1.0 / np.mean((fanin, fanout))
    std = np.sqrt(scale) / 0.87962566103423978
    return std * jax.random.truncated_normal(key, -2, 2, shape, jnp.float32)


def their_initializer_uniform_avg(key, shape):
    """``nets.py:748-753`` (``Initializer``), the ``dist='uniform', fan='avg'`` branch
    at ``scale = 1.0``, verbatim but for ``nj.rng()`` → ``key``::

        fanin, fanout = self._fans(shape)
        denoms = {'avg': (fanin + fanout) / 2, ...}
        scale = self.scale / denoms[self.fan]
        limit = np.sqrt(3 * scale)
        value = jax.random.uniform(nj.rng(), shape, f32, -limit, limit)

    ``'uniform'`` is ``Initializer.__init__``'s **default** ``dist``, and this branch
    is therefore what their ``Dist`` output linears are drawn from: ``Dist.inner``
    builds them with ``kw = {'outscale': …, 'outnorm': …}`` and nothing else
    (``nets.py:556-565``), so the surrounding block's ``winit: normal`` reaches every
    hidden ``Linear`` in the stack and not the two head linears.
    """
    fanin, fanout = shape
    limit = np.sqrt(3 * (1.0 / ((fanin + fanout) / 2)))
    return jax.random.uniform(key, shape, jnp.float32, -limit, limit)


def test_the_dist_head_linears_are_drawn_uniform_as_theirs_are():
    """Divergence 26b, closed: their ``Dist`` head is **not** ``winit: normal``.

    This is the second of the two defects the fourth audit found in our fourteen
    transcription ports, and it is the same *kind* as the info-gain one: the port of
    the formula was faithful and the harness around it asserted something false
    about where the formula is called from. ``their_dist_normal``'s arithmetic was
    right — ``tanh`` mean, ``+2.0`` shift, the std band, biases kept, norm-free — and
    the surrounding initializer test pinned a **normal** draw for layers their code
    draws **uniform**, because ``Dist.inner`` forwards only ``outscale``/``outnorm``
    to ``Linear`` and the block's ``winit`` never gets there.

    Cosmetic in its numbers: ``U(−ℓ, ℓ)`` with ``ℓ = sqrt(3·scale)`` has std
    ``sqrt(scale)``, exactly as their scale-corrected truncated normal does. So the
    assertion that discriminates the two is the **support**, not the moment: a
    uniform draw is bounded strictly inside ``sqrt(3)·std ≈ 1.732·std`` while a
    truncated normal at their ``/0.8796`` correction runs out to ``2/0.8796 ≈
    2.27·std`` and, over a few hundred draws, gets there.
    """
    from hybrid_aif.distributions.neural import MLPConditional, uniform_avg

    #  the port, key for key
    for i, shape in enumerate([(64, 32), (7, 3), (256, 512)]):
        key = jax.random.PRNGKey(40 + i)
        theirs = their_initializer_uniform_avg(key, shape)
        ours = uniform_avg(key, shape, shape[0], shape[1])
        np.testing.assert_allclose(np.asarray(ours), np.asarray(theirs), rtol=1e-6, atol=1e-8)
        limit = float(np.sqrt(3.0 / np.mean(shape)))
        assert float(jnp.max(jnp.abs(ours))) <= limit
        if np.prod(shape) > 1000:  # a uniform's std is limit/sqrt(3) = sqrt(scale)
            want = np.sqrt(1.0 / np.mean(shape))
            assert abs(float(jnp.std(ours)) - want) < 0.05 * want

    #  and the built head really takes that branch, while the trunk does not
    child, width = 3, 200
    net = MLPConditional.from_shapes(
        (100,),
        child,
        key=jax.random.PRNGKey(5),
        width=width,
        depth=2,
        activation="silu",
        norm=True,
        head="bounded",
        init="trunc_normal_avg",
    )
    head = np.asarray(net.layers[-1].weight)  # (2·child, width), halves at fan (width+child)/2
    half_std = np.sqrt(1.0 / ((width + child) / 2))
    for half in (head[:child], head[child:]):
        assert abs(half.std() - half_std) < 0.15 * half_std  # the shared second moment
        #  the support is what separates uniform from their truncated normal
        assert np.abs(half).max() <= np.sqrt(3.0) * half_std
    trunk = np.asarray(net.layers[0].weight)  # (width, 100): winit: normal DOES reach here
    trunk_std = np.sqrt(1.0 / ((100 + width) / 2))
    assert np.abs(trunk).max() > np.sqrt(3.0) * trunk_std  # outside a uniform's support


def test_the_winit_normal_initializer_is_theirs_key_for_key():
    """Divergence 17, closed for the ``MLPConditional`` slots: ``trunc_normal_avg``
    against their ``Initializer('normal', fan='avg')``, same key ⇒ same array, plus
    the properties a same-key identity cannot certify alone (support, std, fans).

    The **head** layer is not covered here: their ``Dist`` draws it uniform, which
    :func:`test_the_dist_head_linears_are_drawn_uniform_as_theirs_are` pins.
    """
    from hybrid_aif.distributions.neural import MLPConditional, trunc_normal_avg

    for i, shape in enumerate([(64, 32), (7, 3), (256, 512)]):
        key = jax.random.PRNGKey(20 + i)
        theirs = their_initializer_normal_avg(key, shape)
        #  ours draws (out, in) for eqx's kernel layout; fans are the same pair
        ours = trunc_normal_avg(key, shape, shape[0], shape[1])
        np.testing.assert_allclose(np.asarray(ours), np.asarray(theirs), rtol=1e-6, atol=1e-8)
        std = float(np.sqrt(1.0 / np.mean(shape)) / 0.87962566103423978)
        assert float(jnp.max(jnp.abs(ours))) <= 2.0 * std + 1e-6  # truncated at ±2σ_param
        if np.prod(shape) > 1000:  # enough draws for a tight sample-std read
            assert abs(float(jnp.std(ours)) - np.sqrt(1.0 / np.mean(shape))) < 0.05 * np.sqrt(
                1.0 / np.mean(shape)
            )

    #  the built module: biases zeroed (theirs start at zeros), normed hidden layers
    #  carry no bias at all, and the fused head layer's halves are drawn at their own
    #  fan — fan_out = child_dim per half, their two separate Dist linears
    net = MLPConditional.from_shapes(
        (100,),
        2,
        key=jax.random.PRNGKey(3),
        width=200,
        depth=2,
        activation="silu",
        norm=True,
        head="bounded",
        init="trunc_normal_avg",
    )
    assert all(layer.bias is None for layer in net.layers[:-1])
    assert np.all(np.asarray(net.layers[-1].bias) == 0.0)
    w0 = np.asarray(net.layers[0].weight)  # (200, 100): fan denominator (100+200)/2
    assert abs(w0.std() - np.sqrt(1.0 / 150.0)) < 0.05 * np.sqrt(1.0 / 150.0)
    head = np.asarray(net.layers[-1].weight)  # each half (2, 200): denom (200+2)/2
    half_std = np.sqrt(1.0 / 101.0)
    assert abs(head[:2].std() - half_std) < 0.15 * half_std
    assert abs(head[2:].std() - half_std) < 0.15 * half_std
    #  and the two halves are DIFFERENT draws (a fused single draw would tie them)
    assert np.abs(head[:2] - head[2:]).max() > 0.0
    #  the head's *shape* is uniform, not this initializer's — see
    #  test_the_dist_head_linears_are_drawn_uniform_as_theirs_are
    assert np.abs(head).max() <= np.sqrt(3.0) * half_std
