"""R-AIF's world-model core: categorical latents, the deterministic node, ``fit_world``.

P1′ chunks 1 and 2a (``docs/design/amortization.md`` build plan; the architecture
and losses are pinned by ``amortization-p0-addendum.md`` §§1–3). Goal 2 is the
test this file pays for — *replicate R-AIF with the same architecture* — so the
assertions are about their semantics, not ours-in-general.

**Chunk 2a re-pointed everything downstream of the parts through the declared
System** (``worlds.tiny_rssm``): the trainer takes ``(system, params,
learn_state, batch)``, the trained fixture is a params pytree, and the prediction
claim is made twice — once through the assembly (their ``open_loop``) and once
through the faces (``predict_beliefs``). The slot-level machinery itself — A11's
recognition slot, its grammar, the loud fences — is ``test_recognition_slot.py``;
what stays here is the architecture, the losses and the trainer.

- **the categorical output head** (A10 disposition 1): 32×32 factored logits,
  ``unimix 0.01`` flooring every class probability, straight-through one-hot
  sampling, and the probability tensor standing in for ``moments``;
- **the loss pieces** (§3): KL balancing at ``dyn 0.5``/``rep 0.1`` — one number,
  two stop-gradient directions — with the one-nat free-bits floor, plus the
  ``symlog_mse``/``symlog_disc``/``binary`` head losses;
- **the ``Deterministic`` node** (A10 disposition 2): the ``system.md`` Factor
  union's second member, wrapping the GRU as a pure map with an explicit carry
  and learning through A2's ``LearnState`` path;
- **A1(b) amortized recognition**: posterior and prior as the two heads the KL
  balances, with ``observe`` a ``lax.scan`` whose carry is the posterior state;
- **``fit_world``** (A6's first trainer) on real Mountain Car replay: the loss
  falls, the free-bits floor visibly engages, and multi-step *open-loop*
  prediction beats a persistence baseline on held-out sequences.

What is deliberately absent is the next chunk: no actor, no critic, no CRSPP, no
self-revision, no ``imagine()``.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from hybrid_aif.distributions import Categorical, MLPCategorical, MLPPredictor, PredictorHead
from hybrid_aif.distributions.neural import (
    GRUCell,
    binary_nll,
    disc_bins,
    factored_categorical_kl,
    kl_balance,
    symexp,
    symlog,
    symlog_disc_mean,
    symlog_disc_nll,
    symlog_mse_nll,
    twohot,
    unimix_logits,
)
from hybrid_aif.factors import Deterministic
from hybrid_aif.inference.amortized import head_parents
from hybrid_aif.inference.rssm import PART_FIELDS, RSSMState
from hybrid_aif.reference.envs.mountain_car import MountainCar
from hybrid_aif.reference.raif.world import tiny_core, tiny_rssm
from hybrid_aif.system import LearnState
from hybrid_aif.train import WorldBatch, assemble, fit_world, posterior_states, world_loss

KEY = jax.random.PRNGKey(0)
VARS, CLASSES = 32, 32


def _head(**kwargs) -> MLPCategorical:
    opts = {"key": jax.random.PRNGKey(5), "width": 32, "depth": 1}
    opts.update(kwargs)
    return MLPCategorical.from_shapes((4,), VARS, CLASSES, **opts)


# ============================================================ the categorical head


def test_the_head_emits_a_factored_categorical_over_variables():
    """A10 disposition 1's shape, and the library's batch convention doing the
    factoring: ``(num_vars, num_classes)`` logits are a ``Categorical`` whose
    *batch* axis is the variable axis, so entropies and KLs come out per variable."""
    head = _head()
    parents = (jnp.asarray([0.3, -0.7, 1.0, 0.0]),)
    logits = head.logits(parents)
    assert logits.shape == (VARS, CLASSES)
    assert head.flat_dim == VARS * CLASSES

    dist = head.condition_on(parents)
    assert isinstance(dist, Categorical)
    assert dist.batch_shape == (VARS,)
    assert dist.num_levels == CLASSES
    probs = head.probs(parents)
    np.testing.assert_allclose(np.asarray(probs.sum(axis=-1)), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(probs), np.asarray(dist.probs), atol=0)
    assert head.parent_dim == 4 and head.learnable and not head.frozen


def test_the_probs_tensor_replaces_moments_for_a_categorical_child():
    """``moments`` has no referent here — there is no (mean, covariance) pair a
    Gaussian consumer could read off a one-hot code — so the family's moment view
    is the probability tensor, and the class deliberately has no ``moments``."""
    head = _head()
    assert not hasattr(head, "moments")
    parents = (jnp.zeros(4),)
    assert head.probs(parents).shape == (VARS, CLASSES)
    mode = head.mode(parents)
    assert mode.shape == (VARS, CLASSES)
    np.testing.assert_allclose(np.asarray(mode.sum(axis=-1)), 1.0, atol=1e-5)
    np.testing.assert_array_equal(
        np.asarray(jnp.argmax(mode, axis=-1)), np.asarray(jnp.argmax(head.logits(parents), axis=-1))
    )


def test_unimix_floors_every_class_probability_so_log_prob_is_finite_everywhere():
    """``unimix 0.01`` (addendum §2) is not cosmetic: it is what makes ``log_prob``
    of *any* code finite and the posterior/prior KL unable to diverge. Even a
    saturated logit vector keeps every class above ``unimix/num_classes``."""
    saturated = jnp.asarray([1e4, -1e4, 0.0, -1e4])
    mixed = jax.nn.softmax(unimix_logits(saturated, 0.01))
    assert float(mixed.min()) >= 0.01 / 4 * (1 - 1e-6)
    assert bool(jnp.all(jnp.isfinite(unimix_logits(saturated, 0.01))))

    head = _head()
    parents = (jnp.asarray([50.0, -50.0, 50.0, -50.0]),)  # push the trunk hard
    probs = head.probs(parents)
    assert float(probs.min()) >= 0.01 / CLASSES * (1 - 1e-5)
    worst = jax.nn.one_hot(jnp.argmin(head.logits(parents), axis=-1), CLASSES)
    lp = head.log_prob(worst, parents)
    assert bool(jnp.isfinite(lp)) and float(lp) < 0.0

    # with unimix switched off the same adversarial code can underflow to -inf
    bare = dataclasses.replace(head, unimix=0.0)
    bare_worst = jax.nn.one_hot(jnp.argmin(bare.logits(parents), axis=-1), CLASSES)
    assert float(bare.log_prob(bare_worst, parents)) < float(lp)


def test_log_prob_is_the_one_hot_inner_product_summed_over_variables():
    """The child value of this family is the one-hot *code*, so ``log_prob`` is
    ``Σ_v log p(class_v)`` — checked against the plain ``Categorical`` family on
    the same logits, and accepting the flattened code as well."""
    head = _head()
    parents = (jnp.asarray([0.1, 0.2, -0.3, 0.4]),)
    indices = jax.random.randint(jax.random.PRNGKey(1), (VARS,), 0, CLASSES)
    code = jax.nn.one_hot(indices, CLASSES)
    expected = jnp.sum(head.condition_on(parents).log_prob(indices))
    assert float(head.log_prob(code, parents)) == pytest.approx(float(expected), rel=1e-5)
    assert float(head.log_prob(code.reshape(-1), parents)) == pytest.approx(
        float(expected), rel=1e-5
    )
    with pytest.raises(ValueError, match="one-hot code"):
        head.log_prob(jnp.zeros(7), parents)


def test_sampling_is_a_hard_one_hot_whose_straight_through_gradient_flows():
    """Forward: a hard one-hot per variable (the discrete code the downstream nets
    were designed for). Backward: ``∂probs`` — which is the only reason a
    categorical latent is trainable by backprop at all. A parameter's gradient
    through a *sampled* code is nonzero, and equals the gradient through the
    probabilities."""
    head = _head()
    parents = (jnp.asarray([0.2, 0.0, -0.1, 0.3]),)
    draw = head.sample(jax.random.PRNGKey(2), parents)
    assert draw.shape == (VARS, CLASSES)
    np.testing.assert_allclose(np.asarray(draw.sum(axis=-1)), 1.0, atol=1e-5)
    assert set(np.unique(np.asarray(draw)).tolist()) <= {0.0, 1.0}

    weights = jax.random.normal(jax.random.PRNGKey(3), (VARS, CLASSES))

    def through_sample(h):
        return jnp.sum(weights * h.sample(jax.random.PRNGKey(2), parents))

    def through_probs(h):
        return jnp.sum(weights * h.probs(parents))

    g_sample = eqx.filter_grad(through_sample)(head)
    g_probs = eqx.filter_grad(through_probs)(head)
    leaf = g_sample.trunk.layers[-1].weight
    assert float(jnp.max(jnp.abs(leaf))) > 0.0
    np.testing.assert_allclose(
        np.asarray(leaf), np.asarray(g_probs.trunk.layers[-1].weight), rtol=1e-5, atol=1e-7
    )


# =========================================================== the KL loss pieces (§3)


def test_the_factored_kl_sums_over_variables_and_is_zero_at_equality():
    """Independent variables ⇒ the joint KL is the sum of per-variable KLs (their
    ``Independent(OneHotDist, 1)``), checked against the ``Categorical`` family."""
    a = jax.random.normal(jax.random.PRNGKey(4), (VARS, CLASSES))
    b = jax.random.normal(jax.random.PRNGKey(5), (VARS, CLASSES))
    expected = jnp.sum(Categorical(logits=a).kl_to(Categorical(logits=b)).midpoint)
    got = factored_categorical_kl(a, b)
    assert got.shape == ()
    assert float(got) == pytest.approx(float(expected), rel=1e-5)
    assert float(factored_categorical_kl(a, a)) == pytest.approx(0.0, abs=1e-5)

    stacked = factored_categorical_kl(jnp.stack([a, b]), jnp.stack([b, a]))
    assert stacked.shape == (2,)  # leading batch axes preserved: one KL per step


def test_free_bits_clamps_cheap_complexity_and_releases_expensive_complexity():
    """Their ``free: 1.0``. Below one nat the term is a *constant* — cheap
    complexity is free and its gradient is exactly zero — and above it the raw KL
    is paid in full. The clamped/released pair is visible in the returned
    decomposition (``dyn > dyn_raw`` ⇔ the floor is binding)."""
    same = jax.random.normal(jax.random.PRNGKey(6), (VARS, CLASSES))
    clamped = kl_balance(same, same)
    assert float(clamped.dyn_raw) == pytest.approx(0.0, abs=1e-5)
    assert float(clamped.dyn) == 1.0 and float(clamped.rep) == 1.0
    assert float(clamped.total) == pytest.approx(0.5 * 1.0 + 0.1 * 1.0, rel=1e-6)

    # a crafted expensive case: opposed near-deterministic codes
    sharp = 10.0 * jax.nn.one_hot(jnp.zeros(VARS, jnp.int32), CLASSES)
    other = 10.0 * jax.nn.one_hot(jnp.ones(VARS, jnp.int32), CLASSES)
    released = kl_balance(sharp, other)
    assert float(released.dyn_raw) > 1.0
    assert float(released.dyn) == pytest.approx(float(released.dyn_raw), rel=1e-6)
    assert float(released.total) == pytest.approx(
        0.5 * float(released.dyn) + 0.1 * float(released.rep), rel=1e-6
    )

    # the floor is a floor, not a scale: no gradient below it
    grad = jax.grad(lambda logits: kl_balance(same, logits).total)(same + 1e-3)
    assert float(jnp.max(jnp.abs(grad))) == pytest.approx(0.0, abs=1e-8)
    with pytest.raises(ValueError, match="free_nats must be non-negative"):
        kl_balance(same, same, free_nats=-1.0)


def test_kl_balancing_is_two_stop_gradient_directions_of_one_number():
    """``dyn`` and ``rep`` are the *same* KL with opposite stop-gradients (§3):
    equal in value, and each blind to one of the two heads. ``dyn`` (0.5) trains
    the prior toward the posterior; ``rep`` (0.1) trains the posterior toward the
    prior, one fifth as hard — the asymmetry that stops the posterior collapsing
    onto an unpredictable code."""
    post = jax.random.normal(jax.random.PRNGKey(7), (VARS, CLASSES)) * 2.0
    prior = jax.random.normal(jax.random.PRNGKey(8), (VARS, CLASSES)) * 2.0
    out = kl_balance(post, prior)
    assert float(out.dyn_raw) == pytest.approx(float(out.rep_raw), rel=1e-6)
    assert float(out.dyn_raw) > 1.0  # both released, so the scales are readable

    d_post_dyn, d_prior_dyn = jax.grad(lambda q, p: kl_balance(q, p).dyn, argnums=(0, 1))(
        post, prior
    )
    d_post_rep, d_prior_rep = jax.grad(lambda q, p: kl_balance(q, p).rep, argnums=(0, 1))(
        post, prior
    )
    assert float(jnp.max(jnp.abs(d_post_dyn))) == pytest.approx(0.0, abs=1e-9)  # sg(q)
    assert float(jnp.max(jnp.abs(d_prior_dyn))) > 0.0
    assert float(jnp.max(jnp.abs(d_prior_rep))) == pytest.approx(0.0, abs=1e-9)  # sg(p)
    assert float(jnp.max(jnp.abs(d_post_rep))) > 0.0

    # the weights are verifiably the ones applied to those two paths
    d_post_total, d_prior_total = jax.grad(lambda q, p: kl_balance(q, p).total, argnums=(0, 1))(
        post, prior
    )
    np.testing.assert_allclose(
        np.asarray(d_prior_total), 0.5 * np.asarray(d_prior_dyn), rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        np.asarray(d_post_total), 0.1 * np.asarray(d_post_rep), rtol=1e-5, atol=1e-7
    )


def test_the_three_head_losses_are_theirs():
    """``symlog_mse`` (vector decoder), ``symlog_disc`` (255 bins over [-20,20]) and
    ``binary`` (continuation), addendum §2/§3."""
    x = jnp.asarray([-3.0, 0.0, 17.5])
    np.testing.assert_allclose(np.asarray(symexp(symlog(x))), np.asarray(x), rtol=1e-5, atol=1e-6)

    target = jnp.asarray([2.0, -0.5])
    assert float(symlog_mse_nll(symlog(target), target)) == pytest.approx(0.0, abs=1e-10)
    assert float(symlog_mse_nll(jnp.zeros(2), target)) == pytest.approx(
        float(jnp.sum(symlog(target) ** 2)), rel=1e-6
    )

    bins = disc_bins(255, -20.0, 20.0)
    assert bins.shape == (255,) and float(bins[0]) == -20.0 and float(bins[-1]) == 20.0
    for value in (-1.0986, 0.0, 3.3, -7.7):
        weights = twohot(jnp.asarray(value), bins)
        assert float(jnp.sum(weights)) == pytest.approx(1.0, rel=1e-6)
        assert int(jnp.sum(weights > 0)) <= 2
        assert float(jnp.sum(weights * bins)) == pytest.approx(value, abs=1e-4)

    # a head whose logits *are* the two-hot target decodes back to the reward
    reward = jnp.asarray(-2.0)
    logits = jnp.log(twohot(symlog(reward), bins) + 1e-9)
    assert float(symlog_disc_mean(logits, bins)) == pytest.approx(-2.0, abs=1e-3)
    assert float(symlog_disc_nll(logits, reward, bins)) < float(
        symlog_disc_nll(jnp.zeros(255), reward, bins)
    )

    logit = jnp.asarray(1.3)
    assert float(binary_nll(logit, jnp.asarray(1.0))) == pytest.approx(
        float(-jnp.log(jax.nn.sigmoid(logit))), rel=1e-6
    )
    assert float(binary_nll(logit, jnp.asarray(0.0))) == pytest.approx(
        float(-jnp.log1p(-jax.nn.sigmoid(logit))), rel=1e-6
    )


# ============================================================ the Deterministic node


def _gru_node(**kwargs) -> Deterministic:
    opts = {"key": jax.random.PRNGKey(9), "hidden": 32}
    opts.update(kwargs)
    return Deterministic.gru(16, (8, 1), **opts)


def test_the_deterministic_node_wraps_a_pure_map_with_an_explicit_carry():
    """A10 disposition 2's first build. The node is the ``system.md`` Factor
    union's second member: a pure map, no density, and — being recurrent — a carry
    that is an *argument*, so the same node is a ``lax.scan`` body under jit and a
    Python loop outside it — same trajectory, up to XLA's fusion arithmetic."""
    node = _gru_node()
    assert isinstance(node.map, GRUCell)
    assert node.rule == "carry" and node.learnable
    deter = jnp.zeros(16)
    stoch, action = jnp.ones(8) * 0.1, jnp.asarray([0.5])
    once = node(deter, stoch, action)
    assert once.shape == (16,)

    steps = 5

    def scanned(node, deter):
        def body(carry, _):
            return node(carry, stoch, action), None

        out, _ = jax.lax.scan(body, deter, None, length=steps)
        return out

    eager = deter
    for _ in range(steps):
        eager = node(eager, stoch, action)
    np.testing.assert_allclose(
        np.asarray(jax.jit(scanned)(node, deter)), np.asarray(eager), rtol=1e-5, atol=1e-6
    )

    with pytest.raises(ValueError, match="expected \\(16,\\)"):
        node(jnp.zeros(4), stoch, action)
    with pytest.raises(ValueError, match="expected \\(8,\\)"):
        node(deter, jnp.zeros(3), action)


def test_deterministic_declaration_errors_are_loud():
    """A designed-but-unbuilt message rule is refused by name (``system.md``'s
    sigma-point/quadrature treatments), an unknown one lists what exists, and a
    non-callable map is not a map."""
    with pytest.raises(NotImplementedError, match="sigma_point"):
        Deterministic(map=lambda x: x, rule="sigma_point")
    with pytest.raises(NotImplementedError, match="quadrature"):
        Deterministic(map=lambda x: x, rule="quadrature")
    with pytest.raises(ValueError, match="unknown Deterministic rule"):
        Deterministic(map=lambda x: x, rule="magic")
    with pytest.raises(ValueError, match="must be a pure callable"):
        Deterministic(map=jnp.zeros(3))


def test_a_deterministic_node_learns_through_a2s_gradient_path():
    """Its gradient statistic cannot come from a likelihood of its own — a
    deterministic node has none — so it arrives from the consumer's loss and takes
    the same optax step through the same explicitly-carried optimizer state."""
    node = _gru_node()
    deter, stoch, action = jnp.zeros(16), jnp.ones(8) * 0.1, jnp.asarray([0.5])
    target = jnp.ones(16) * 0.3

    def downstream_loss(n):
        return jnp.sum((n(deter, stoch, action) - target) ** 2)

    opt = node.init_opt_state()
    before = float(downstream_loss(node))
    for _ in range(20):
        node, opt = node.absorb(eqx.filter_grad(downstream_loss)(node), opt, lr=0.05)
    assert float(downstream_loss(node)) < before

    frozen = _gru_node(frozen=True)
    assert not frozen.learnable
    with pytest.raises(NotImplementedError, match="decay"):
        frozen.absorb(eqx.filter_grad(downstream_loss)(frozen), opt, decay=0.9)


# ================================================================ the RSSM assembly


def test_the_tiny_core_is_their_tiny_config_row():
    """``rssm {deter 128, stoch 32, classes 32}``, ``unimix 0.01``, reward
    ``symlog_disc`` with 255 bins over [-20,20], ``mlp_units 64`` embedding
    (addendum §2's tiny row, §1's state observation). The three heads carry their
    losses as declarations (chunk 2a), and the initial carry is theirs: a
    trainable vector that *starts* at zeros."""
    core = tiny_core(2, 1, key=KEY)
    assert (core.deter_dim, core.num_vars, core.num_classes) == (128, 32, 32)
    assert core.prior.unimix == 0.01 and core.posterior.unimix == 0.01
    assert (core.num_bins, core.reward.bin_low, core.reward.bin_high) == (255, -20.0, 20.0)
    assert core.embed_dim == 64
    assert core.decoder.out_dim == 2 and core.cont.out_dim == 1
    assert (core.decoder.loss, core.reward.loss, core.cont.loss) == (
        "symlog_mse",
        "symlog_disc",
        "binary",
    )
    assert core.encoder.symlog_inputs  # their symlog_inputs, declared on the net
    assert core.gru.map.input_dims == (32 * 32, 1)
    assert sorted(core.parts()) == sorted(PART_FIELDS)
    assert core.initial.dim == 128 and core.initial.learnable

    state = core.initial_state()
    assert state.deter.shape == (128,) and state.stoch.shape == (32, 32)
    assert state.features.shape == (128 + 1024,)
    np.testing.assert_allclose(np.asarray(state.deter), 0.0, atol=0)  # tanh(0)


def test_parts_that_do_not_chain_are_refused_at_construction():
    """The bundle's widths are structure, so a mismatch is a declaration error
    naming both sides — not a shape error inside a scan on step 40. Since chunk
    2a the *loss* is structure too: a head whose loss says something else about
    its numbers is refused by name."""
    core = tiny_core(2, 1, key=KEY)
    wrong = MLPCategorical.from_shapes((7,), 32, 32, key=KEY, width=32)
    with pytest.raises(ValueError, match="prior parents"):
        dataclasses.replace(core, prior=wrong)
    with pytest.raises(ValueError, match="the same factored categorical"):
        dataclasses.replace(
            core, posterior=MLPCategorical.from_shapes((128, 64), 16, 32, key=KEY, width=32)
        )
    with pytest.raises(ValueError, match="decoder emits"):
        dataclasses.replace(
            core,
            decoder=PredictorHead.from_shapes(
                (128, 1024), loss="symlog_mse", out_dim=7, key=KEY, width=32
            ),
        )
    with pytest.raises(ValueError, match="predict one number each"):
        dataclasses.replace(
            core,
            reward=PredictorHead.from_shapes(
                (128, 1024), loss="symlog_mse", out_dim=3, key=KEY, width=32
            ),
        )
    with pytest.raises(ValueError, match="one logit per bin"):
        PredictorHead(
            trunk=MLPPredictor.from_shapes((128, 1024), 7, key=KEY, width=32), loss="symlog_disc"
        )


def test_observe_is_a_scan_whose_carry_is_the_posterior_state():
    """A1(b): the posterior *is* the inference strategy, and the state it produces
    is what the next GRU call receives — which is what makes ``observe`` a filter
    rather than T independent encodings. Equal to the hand-written loop."""
    core = tiny_core(2, 1, key=KEY)
    steps = 6
    actions = jax.random.uniform(jax.random.PRNGKey(1), (steps, 1), minval=-1.0, maxval=1.0)
    observations = jax.random.normal(jax.random.PRNGKey(2), (steps, 2)) * 0.3
    key = jax.random.PRNGKey(3)

    filtered = jax.jit(lambda k, a, o: core.observe(k, a, o))
    states, post, prior = filtered(key, actions, observations)
    assert states.deter.shape == (steps, 128)
    assert states.stoch.shape == (steps, 32, 32)
    assert post.shape == prior.shape == (steps, 32, 32)

    manual, keys = core.initial_state(), jax.random.split(key, steps)
    for t in range(steps):
        manual, post_t, prior_t = core.obs_step(keys[t], manual, actions[t], observations[t])
        np.testing.assert_allclose(np.asarray(post[t]), np.asarray(post_t), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(states.deter[-1]), np.asarray(manual.deter), rtol=1e-5, atol=1e-6
    )

    with pytest.raises(ValueError, match="ONE sequence"):
        core.observe(key, actions[None], observations[None])
    with pytest.raises(ValueError, match="disagree on T"):
        core.observe(key, actions[:-1], observations)


def test_is_first_restarts_the_carry_inside_the_sequence():
    """A spliced sequence filters exactly as its pieces filtered alone.

    The flat-stream contract, stated as the equality that has to hold for it to be
    worth anything: a window holding the tail of one episode and the head of the
    next, with ``is_first`` marking the join, must produce the same posterior states
    as filtering the two pieces separately. That is only true if the reset is total
    — the carry replaced by the learned initial state *and* the incoming ``a_{t−1}``
    zeroed, because a step that begins an episode has no previous action (their
    ``obs_step`` masks both, ``raif/aif/nets.py:131-141``).
    """
    core = tiny_core(2, 1, key=KEY)
    steps, split = 7, 3
    actions = jax.random.uniform(jax.random.PRNGKey(1), (steps, 1), minval=-1.0, maxval=1.0)
    obs = jax.random.normal(jax.random.PRNGKey(2), (steps, 2)) * 0.3
    is_first = jnp.zeros((steps,), bool).at[0].set(True).at[split].set(True)
    #  the a_{t−1} convention: whatever the stream held at a boundary is discarded
    keys = jax.random.split(jax.random.PRNGKey(3), steps)

    states, post, _ = jax.jit(lambda k, a, o, f: core.observe(k, a, o, is_first=f))(
        jax.random.PRNGKey(3), actions, obs, is_first
    )

    manual = core.initial_state()
    for t in range(steps):
        if t in (0, split):  # step 0 is a boundary too, and its action is zeroed as well
            manual = core.initial_state()
            action = jnp.zeros_like(actions[t])
        else:
            action = actions[t]
        manual, post_t, _ = core.obs_step(keys[t], manual, action, obs[t])
        np.testing.assert_allclose(np.asarray(post[t]), np.asarray(post_t), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(states.deter[-1]), np.asarray(manual.deter), rtol=1e-5, atol=1e-6
    )

    #  and the second piece really is independent of the first: perturb the head of
    #  the window and nothing after the boundary moves
    other = obs.at[:split].set(obs[:split] + 5.0)
    perturbed, _, _ = core.observe(jax.random.PRNGKey(3), actions, other, is_first=is_first)
    np.testing.assert_allclose(
        np.asarray(perturbed.deter[split:]), np.asarray(states.deter[split:]), rtol=1e-5, atol=1e-6
    )
    assert not np.allclose(np.asarray(perturbed.deter[:split]), np.asarray(states.deter[:split]))


def test_omitting_is_first_is_bit_identical_to_the_episode_aligned_scan():
    """The argument is *optional*, and optional has to mean free.

    Every caller that predates it hands ``observe`` an episode-aligned sequence, and
    for those the answer must not move by one ULP — so passing nothing builds the
    same scan body as before, and passing the flags an episode-aligned window
    actually carries (true at step 0, nowhere else) is a no-op on top of it: the
    scan already starts from the learned initial carry there, and the shifted
    ``a_{-1}`` is already zero.
    """
    core = tiny_core(2, 1, key=KEY)
    steps = 6
    actions = jax.random.uniform(jax.random.PRNGKey(1), (steps, 1), minval=-1.0, maxval=1.0)
    actions = actions.at[0].set(0.0)  # prev_actions()' zero at an episode's first step
    obs = jax.random.normal(jax.random.PRNGKey(2), (steps, 2)) * 0.3
    key = jax.random.PRNGKey(3)

    plain = core.observe(key, actions, obs)
    aligned = core.observe(key, actions, obs, is_first=jnp.zeros((steps,), bool).at[0].set(True))
    leaves = zip(jax.tree_util.tree_leaves(plain), jax.tree_util.tree_leaves(aligned), strict=True)
    for before, after in leaves:
        assert np.array_equal(np.asarray(before), np.asarray(after))

    with pytest.raises(ValueError, match="is_first has shape"):
        core.observe(key, actions, obs, is_first=jnp.zeros((steps, 1), bool))


def test_a_world_batch_threads_is_first_through_to_the_scan():
    """``WorldBatch.is_first`` is the seam: declared optional, validated when given,
    and carried by both readers of the recurrent scan (the loss and
    :func:`posterior_states`), so the two cannot disagree about where an episode
    begins."""
    system, params = tiny_rssm(2, 1, key=jax.random.PRNGKey(11))
    core, _ = assemble(system, params)
    b, t = 2, 6
    batch = WorldBatch(
        obs=jax.random.normal(jax.random.PRNGKey(12), (b, t, core.obs_dim)) * 0.3,
        action=jnp.zeros((b, t, core.action_dim)),
        reward=jnp.zeros((b, t)),
        cont=jnp.ones((b, t)),
    )
    spliced = batch._replace(is_first=jnp.zeros((b, t), bool).at[:, 0].set(True).at[:, 3].set(True))

    assert batch.check(core) == (b, t) == spliced.check(core)
    with pytest.raises(ValueError, match="is_first"):
        batch._replace(is_first=jnp.zeros((b, t + 1), bool)).check(core)

    key = jax.random.PRNGKey(13)
    flat = posterior_states(system, params, batch, key=key)
    reset = posterior_states(system, params, spliced, key=key)
    assert flat.deter.shape == reset.deter.shape == (b, t, core.deter_dim)
    #  the reset at step 3 is visible in the states and therefore in the loss
    assert not np.allclose(np.asarray(flat.deter[:, 3:]), np.asarray(reset.deter[:, 3:]))
    assert float(world_loss(core, spliced, key)[0]) != float(world_loss(core, batch, key)[0])


def test_open_loop_reads_no_observation():
    """The prior-only rollout is the world model predicting on its own: the same
    warmed state and actions give the same trajectory whatever the (unseen)
    observations were. Not ``imagine()`` — nobody proposes the actions here."""
    core = tiny_core(2, 1, key=KEY)
    state = core.initial_state()
    actions = jax.random.uniform(jax.random.PRNGKey(4), (5, 1), minval=-1.0, maxval=1.0)
    rolled, logits = jax.jit(lambda k, s, a: core.open_loop(k, s, a))(
        jax.random.PRNGKey(5), state, actions
    )
    assert rolled.deter.shape == (5, 128) and logits.shape == (5, 32, 32)
    decoded = jax.vmap(core.decode)(rolled)
    assert decoded.shape == (5, 2) and bool(jnp.all(jnp.isfinite(decoded)))
    rewards = jax.vmap(core.predict_reward)(rolled)
    conts = jax.vmap(core.predict_cont)(rolled)
    assert rewards.shape == (5,) and bool(jnp.all((conts >= 0.0) & (conts <= 1.0)))

    again, _ = core.open_loop(jax.random.PRNGKey(5), state, actions)
    np.testing.assert_allclose(
        np.asarray(again.deter), np.asarray(rolled.deter), rtol=1e-5, atol=1e-6
    )


# ==================================================== fit_world on real MC replay


def _episode(key: jax.Array, swing: bool, env: MountainCar) -> WorldBatch:
    """One episode's worth of columns, in their per-step convention.

    Index 0 is the reset observation with a zero "previous action"; index ``t`` is
    the observation ``a_{t-1}`` produced, with ``cont = 1 − done``. Two policies,
    both fixed-key: a noisy energy-pumping swing (which solves the task, so the
    data contains successes and terminations) and a temporally correlated random
    walk (which almost never does).
    """
    k_reset, k_pol = jax.random.split(key)
    obs0, state = env.reset(k_reset)

    def tick(carry, k):
        state, prev_action, done = carry
        noise = jax.random.uniform(k, (1,), minval=-1.0, maxval=1.0)
        action = (
            jnp.where(state[0][1] >= 0.0, 1.0, -1.0) * 0.9 + 0.1 * noise
            if swing
            else jnp.clip(0.7 * prev_action + 0.7 * noise, -1.0, 1.0)
        )
        obs, state, reward, done_next = env.agent_step(None, state, action)
        return (state, action, done_next), (obs[0], action, reward, 1.0 - done)

    _, (obs, action, reward, cont) = jax.lax.scan(
        tick,
        (state, jnp.zeros(1), jnp.zeros((), bool)),
        jax.random.split(k_pol, env.time_limit),
    )
    return WorldBatch(
        obs=jnp.concatenate([obs0[0][None], obs[:-1]]),
        action=jnp.concatenate([jnp.zeros((1, 1)), action[:-1]]),
        reward=jnp.concatenate([jnp.zeros(1), reward[:-1]]),
        cont=jnp.concatenate([jnp.ones(1), cont[:-1]]),
    )


def _collect(key: jax.Array, episodes: int, env: MountainCar) -> WorldBatch:
    """``episodes × 100`` agent steps of replay, alternating the two policies."""
    run = jax.jit(_episode, static_argnums=(1, 2))
    columns = [run(k, bool(i % 2 == 0), env) for i, k in enumerate(jax.random.split(key, episodes))]
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *columns)


@pytest.fixture(scope="module")
def trained():
    """Fit the tiny world model on 4,000 Mountain Car transitions (fixed keys).

    40 episodes × 100 agent steps, 32 for training and 8 held out; 300 gradient
    steps of ``fit_world`` on batches of 8 whole episodes. ~1 minute on CPU, and
    the assertions below all read this one run.

    Chunk 2a runs it **through the declared System**: the model is
    ``worlds.tiny_rssm``'s ``(System, SystemParams)`` pair, the optimizer carry is
    ``system.init_learn_state(params)``, and the trainer reads and writes the
    slots. Everything after this fixture is a statement about a declared model.
    """
    env = MountainCar()
    data = _collect(jax.random.PRNGKey(0), 40, env)
    train = jax.tree_util.tree_map(lambda x: x[:32], data)
    held = jax.tree_util.tree_map(lambda x: x[32:], data)

    system, params = tiny_rssm(2, env.action_dim, key=jax.random.PRNGKey(1))
    learn_state = system.init_learn_state(params)
    step = jax.jit(lambda p, s, b, k: fit_world(system, p, s, b, key=k, lr=3e-4))

    history = []
    for i in range(300):
        k_batch, k_step = jax.random.split(jax.random.PRNGKey(1000 + i))
        idx = jax.random.randint(k_batch, (8,), 0, train.obs.shape[0])
        batch = jax.tree_util.tree_map(lambda x, idx=idx: x[idx], train)
        params, learn_state, losses = step(params, learn_state, batch, k_step)
        history.append(losses)
    return (system, params), history, held, data


def test_the_replay_is_their_mountain_car_in_their_per_step_convention(trained):
    """The data the world model is fit on: sparse −2/−1/0 rewards, position inside
    their limits, successes present (the swing policy reaches the flag, so ``cont``
    goes to zero in those episodes), and index 0 the reset step."""
    _, _, _, data = trained
    assert data.obs.shape == (40, 100, 2)
    assert data.action.shape == (40, 100, 1)
    assert float(data.obs[..., 0].min()) >= -1.2 - 1e-6
    assert float(data.obs[..., 0].max()) <= 0.6 + 1e-6
    assert set(np.unique(np.asarray(data.reward)).tolist()) <= {0.0, -1.0, -2.0}
    np.testing.assert_array_equal(np.asarray(data.action[:, 0]), 0.0)
    np.testing.assert_array_equal(np.asarray(data.cont[:, 0]), 1.0)
    terminated = int(jnp.sum(jnp.any(data.cont < 0.5, axis=1)))
    assert 0 < terminated < 40  # both regimes present in replay


def test_fit_world_drives_the_loss_down(trained):
    """A6's trainer on their Eq.-4 objective: the total falls sharply, and so do
    the reconstruction and reward NLLs individually — the accuracy half is
    genuinely being fit, not traded against the complexity half."""
    _, history, _, _ = trained
    first, last = history[0], history[-1]
    tail = float(np.mean([float(h.total) for h in history[-20:]]))
    assert tail < 0.5 * float(first.total)
    assert float(last.decoder) < 0.1 * float(first.decoder)
    assert float(last.reward) < float(first.reward)
    assert all(bool(jnp.isfinite(h.total)) for h in history)


def test_the_free_bits_floor_engages_once_the_prior_tracks_the_posterior(trained):
    """The floor is *not* binding at initialization — two randomly-initialized
    heads disagree by ≈3 nats, well above ``free 1.0`` — and becomes binding once
    the ``dyn`` direction has pulled the prior onto the posterior, after which the
    complexity term is a constant and only accuracy is paid for. (Recorded: the
    build brief expected the reverse order; with 32×32 unimix latents the initial
    KL is above the floor, so the clamp arrives with learning rather than before
    it. Both regimes are exercised — this test for the transition, the crafted
    cases in ``test_free_bits_clamps_…`` for the floor's own semantics.)"""
    _, history, _, _ = trained
    first, last = history[0], history[-1]
    assert float(first.dyn_raw) > 1.0
    assert float(first.dyn) == pytest.approx(float(first.dyn_raw), rel=1e-5)  # released
    assert float(last.dyn_raw) < 1.0
    assert float(last.dyn) == pytest.approx(1.0, abs=1e-2)  # clamped
    assert float(last.dyn) > float(last.dyn_raw)
    # the two directions are one number: equal in value at every step
    assert float(last.dyn_raw) == pytest.approx(float(last.rep_raw), rel=1e-5)


def test_open_loop_prediction_beats_persistence_on_held_out_sequences(trained):
    """The point of a world model: after ten steps of posterior warm-up, roll the
    *prior* forward eight steps (their ``imag_horizon`` for MC) on held-out
    episodes with no further observations, and predict position better than
    holding the last one — which is the baseline a model that had learned nothing
    about how velocity integrates would tie."""
    (system, params), _, held, _ = trained
    core, _ = assemble(system, params)
    warm, horizon = 10, 8

    def evaluate(key, obs, action):
        states, _, _ = core.observe(key, action[:warm], obs[:warm])
        warmed = jax.tree_util.tree_map(lambda x: x[-1], states)
        rolled, _ = core.open_loop(key, warmed, action[warm : warm + horizon])
        predicted = jax.vmap(core.decode)(rolled)[:, 0]
        truth = obs[warm : warm + horizon, 0]
        persistence = jnp.broadcast_to(obs[warm - 1, 0], truth.shape)
        return jnp.mean((predicted - truth) ** 2), jnp.mean((persistence - truth) ** 2)

    keys = jax.random.split(jax.random.PRNGKey(7), held.obs.shape[0])
    model, persistence = jax.vmap(evaluate)(keys, held.obs, held.action)
    model_mse, persistence_mse = float(jnp.mean(model)), float(jnp.mean(persistence))
    assert np.isfinite(model_mse) and persistence_mse > 0.0
    assert model_mse < 0.6 * persistence_mse, (model_mse, persistence_mse)


def test_open_loop_through_the_system_faces_also_beats_persistence(trained):
    """The chunk-2a claim, on the trained model: the canonical tick loop
    (``update_empirical_prior`` → ``infer_states``) filters, ``predict_beliefs``
    rolls the world model forward on its own, and the prediction beats persistence
    — through the *faces*, with no assembly, no bundle and no key, reading the
    decoder as the declaration it now is.

    **Measured gap, recorded** (and documented in
    :mod:`hybrid_aif.inference.amortized`): the faces are keyless, so they
    propagate the code's *belief* (its expected one-hot) where ``open_loop``
    propagates a straight-through *sample*. On this run that is 0.0120 vs 0.0159
    (≈1.3x better than persistence) against the sampled path's ≈2.8x — mean-field
    propagation of a categorical latent trained on samples pays for the gap
    between ``f(E[c])`` and ``E[f(c)]``. Beating persistence is the claim here; the
    sampled rollout stays the sharper instrument and is what chunk 2b's
    ``imagine()`` (which has a key) should use."""
    (system, params), _, held, _ = trained
    warm, horizon = 10, 8
    decoder = params.likelihood["state"]

    def evaluate(obs, action):
        beliefs = system.initial_priors(params)
        for t in range(warm):  # the canonical tick loop, twice per step
            beliefs = system.update_empirical_prior(
                params, beliefs, system.from_positional("actions", action[t], params)
            )
            observed = system.from_positional("observations", (obs[t], None, None))
            beliefs, _ = system.infer_states(params, observed, beliefs)
        rolled = system.predict_beliefs(params, beliefs, action[warm : warm + horizon])
        predicted = jnp.stack(
            [
                decoder.predict(head_parents(system.to_positional("beliefs", state), (0, 1)))[0]
                for state in rolled
            ]
        )
        truth = obs[warm : warm + horizon, 0]
        persistence = jnp.broadcast_to(obs[warm - 1, 0], truth.shape)
        return jnp.mean((predicted - truth) ** 2), jnp.mean((persistence - truth) ** 2)

    scores = [evaluate(held.obs[i], held.action[i]) for i in range(held.obs.shape[0])]
    model_mse = float(np.mean([float(m) for m, _ in scores]))
    persistence_mse = float(np.mean([float(p) for _, p in scores]))
    assert np.isfinite(model_mse) and persistence_mse > 0.0
    assert model_mse < 0.85 * persistence_mse, (model_mse, persistence_mse)


def test_fit_world_needs_its_optimizer_carry_declared_explicitly():
    """A2's dropped-carry guard, at the trainer: ``None`` raises rather than
    silently restarting adam, and a carry whose keys are not this model's is stale,
    not a suggestion. The keys are the System's own slot addresses."""
    system, params = tiny_rssm(2, 1, key=KEY)
    batch = WorldBatch(
        obs=jnp.zeros((2, 3, 2)),
        action=jnp.zeros((2, 3, 1)),
        reward=jnp.zeros((2, 3)),
        cont=jnp.ones((2, 3)),
    )
    with pytest.raises(ValueError, match="init_learn_state"):
        fit_world(system, params, None, batch, key=KEY)
    good = system.init_learn_state(params)
    assert set(good.opt_states) == {
        "transition:deter",
        "transition:stoch",
        "recognition:stoch",
        "recognition:embed",
        "likelihood:state",
        "likelihood:reward",
        "likelihood:cont",
        "initial_state_prior:deter",
    }
    with pytest.raises(ValueError, match="stale"):
        fit_world(
            system,
            params,
            LearnState(opt_states={"likelihood:nope": good.opt_states["likelihood:state"]}),
            batch,
            key=KEY,
        )
    with pytest.raises(ValueError, match="WorldBatch shapes disagree"):
        fit_world(system, params, good, batch._replace(reward=jnp.zeros((2, 4))), key=KEY)


def test_a_frozen_slot_holds_no_optimizer_state_and_never_moves():
    """A1's trainable marking, now on a real slot: freezing the decoder drops its
    ``LearnState`` entry and its weights come back bitwise, while the rest of the
    model still trains. Chunk 1 promised the *structural* exclusion for when the
    parts became slots, and this is it — a frozen slot is not differentiated at
    all, so it does not even enter the global-norm clip."""
    system, params = tiny_rssm(2, 1, key=KEY)
    frozen = dataclasses.replace(
        params.likelihood["state"],
        trunk=dataclasses.replace(params.likelihood["state"].trunk, frozen=True),
    )
    params = dataclasses.replace(params, likelihood={**params.likelihood, "state": frozen})
    assert not params.likelihood["state"].learnable
    learn_state = system.init_learn_state(params)
    assert "likelihood:state" not in learn_state.opt_states

    batch = WorldBatch(
        obs=jnp.zeros((2, 4, 2)),
        action=jnp.zeros((2, 4, 1)),
        reward=-2.0 * jnp.ones((2, 4)),
        cont=jnp.ones((2, 4)),
    )
    updated, _, _ = fit_world(system, params, learn_state, batch, key=KEY, lr=1e-2)
    np.testing.assert_array_equal(
        np.asarray(updated.likelihood["state"].trunk.layers[-1].weight),
        np.asarray(params.likelihood["state"].trunk.layers[-1].weight),
    )
    moved = jnp.abs(
        updated.transition["stoch"].trunk.layers[-1].weight
        - params.transition["stoch"].trunk.layers[-1].weight
    )
    assert float(jnp.max(moved)) > 0.0


def test_fit_world_on_slots_reproduces_the_bundle_trainer_bitwise():
    """The chunk-2a rehoming is a *rehoming*: the same eight modules, the same one
    loss, the same one global-norm clip, the same optax steps — so training the
    declared System must reproduce the chunk-1 bundle trainer's trajectory exactly,
    not approximately. The reference below is chunk 1's ``fit_world`` body
    (differentiate the assembly, clip its global norm, step each part through its
    own carry); the assertion is bitwise equality of the losses *and* of every
    parameter after several steps.

    What makes it bitwise rather than 1e-7-close is the part *order*
    (:data:`PART_FIELDS`, shared by the assembly's fields and the trainer's
    differentiation target), since the clip's norm is a sum over leaves and float
    addition does not reassociate."""
    system, params = tiny_rssm(2, 1, key=KEY)
    core = assemble(system, params)[0]
    batch = WorldBatch(
        obs=jax.random.normal(jax.random.PRNGKey(21), (4, 12, 2)) * 0.3,
        action=jax.random.uniform(jax.random.PRNGKey(22), (4, 12, 1), minval=-1.0, maxval=1.0),
        reward=-jnp.ones((4, 12)),
        cont=jnp.ones((4, 12)),
    )

    def bundle_step(core, opt, key, lr=3e-4, clip=1000.0):
        grads, losses = eqx.filter_grad(lambda model: world_loss(model, batch, key), has_aux=True)(
            core
        )
        clipper = optax.clip_by_global_norm(clip)
        grads, _ = clipper.update(grads, clipper.init(grads))
        stepped = {}
        for name in PART_FIELDS:
            part = getattr(core, name)
            if part.learnable:
                stepped[name], opt[name] = part.absorb(getattr(grads, name), opt[name], lr=lr)
        return core.replace_parts(stepped), opt, losses

    opt = {name: part.init_opt_state() for name, part in core.parts().items() if part.learnable}
    learn_state = system.init_learn_state(params)
    for i in range(4):
        k = jax.random.PRNGKey(300 + i)
        core, opt, bundle_losses = bundle_step(core, opt, k)
        params, learn_state, slot_losses = fit_world(
            system, params, learn_state, batch, key=k, lr=3e-4
        )
        assert float(slot_losses.total) == float(bundle_losses.total)  # bitwise
    for a, b in zip(
        jax.tree_util.tree_leaves(core),
        jax.tree_util.tree_leaves(assemble(system, params)[0]),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_world_loss_is_the_sum_of_its_reported_pieces():
    """The decomposition is the objective, not a log line: ``total`` is exactly
    ``decoder + reward + cont + 0.5·dyn + 0.1·rep`` at the reported means, so a
    reader can see which term any change came from (and the scales are readable)."""
    core = tiny_core(2, 1, key=KEY)
    batch = WorldBatch(
        obs=jax.random.normal(jax.random.PRNGKey(11), (3, 5, 2)) * 0.2,
        action=jax.random.uniform(jax.random.PRNGKey(12), (3, 5, 1), minval=-1.0, maxval=1.0),
        reward=-2.0 * jnp.ones((3, 5)),
        cont=jnp.ones((3, 5)),
    )
    total, losses = world_loss(core, batch, jax.random.PRNGKey(13))
    assert float(total) == pytest.approx(float(losses.total), rel=1e-6)
    assert float(losses.total) == pytest.approx(
        float(losses.decoder + losses.reward + losses.cont + 0.5 * losses.dyn + 0.1 * losses.rep),
        rel=1e-5,
    )
    # a different scale changes exactly the term it names
    scaled, _ = world_loss(core, batch, jax.random.PRNGKey(13), reward_scale=0.0)
    assert float(scaled) == pytest.approx(float(total - losses.reward), rel=1e-5)


def test_the_state_carries_both_halves_of_their_state_vector():
    """Addendum §2, gap note 2: ``sim``, every head and the actor read
    ``[deter, flat stoch]``, so ``deter`` is part of "the state" — which is why it
    is a ``Deterministic`` factor rather than something bolted onto ``Beliefs``."""
    state = RSSMState(deter=jnp.arange(3.0), stoch=jnp.eye(2))
    assert state.features.shape == (3 + 4,)
    np.testing.assert_array_equal(np.asarray(state.features[:3]), np.asarray(jnp.arange(3.0)))
