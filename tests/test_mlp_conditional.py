"""The neural conditional family and gradient learning (amortization.md A1/A2, P1).

:class:`~hybrid_aif.distributions.MLPConditional` is the family dial's third
notch — a parametrized *function* in a declaration slot — and the first slot in
this library that learns by gradient rather than by conjugacy. The tests here
bind five things:

- **the family contract** (A1): conditional log-density, ancestral sampling, and
  moments for prediction, all three consistent with one another;
- **slot integration**: a non-frozen net is one learnable parameter variable in
  the one namespace, so ``lr_by_slot`` addresses it (wildcards included) and
  ``epistemic`` selecting it raises the registry's missing-estimator error — the
  same shape the LG noise block raises with, since a net has no novelty
  estimator either (V4, A10 disposition 3);
- **A2's learning rule**: an MLP transition trained tick-by-tick on ``(s, a,
  s')`` triples from a known linear system recovers it, with the optimizer state
  riding in an explicit :class:`~hybrid_aif.system.LearnState` carry;
- **frozen ⇔ no trainable marking** (A1): a frozen declaration comes back
  bitwise, is not rate-addressable, and does not change ``learn``'s return type;
- **the prediction row** (A1(a)): linearized moment propagation through the net,
  exact whenever the mean head is affine — validated against our own LG
  ``predict``, since A1(a) has no R-AIF oracle (A10 disposition 10).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import Categorical, ConditionalCategorical, Gaussian, MLPConditional
from hybrid_aif.distributions.learnable import Learnable
from hybrid_aif.inference.linear_gaussian import predict
from hybrid_aif.objectives import registry
from hybrid_aif.system import LearnState, System, SystemParams

KEY = jax.random.PRNGKey(0)
DIM, NUM_CONTROLS = 2, 3


def _net(**kwargs) -> MLPConditional:
    opts = {"width": 32, "depth": 2, "key": jax.random.PRNGKey(7)}
    opts.update(kwargs)
    return MLPConditional.from_shapes((DIM, NUM_CONTROLS), DIM, **opts)


def _mlp_transition_system(**system_kwargs):
    """One continuous factor 'x', controlled, with a neural transition and no modality."""
    system = System(
        A_deps=(),
        B_deps=((),),
        controlled=(True,),
        policy_len=1,
        factor_names=("x",),
        epistemic=False,  # a neural slot has no novelty estimator; see the tests below
        **system_kwargs,
    )
    params = SystemParams(
        likelihood={},
        transition={"x": _net()},
        initial_state_prior={"x": Gaussian(loc=jnp.zeros(DIM), chol=jnp.eye(DIM))},
    )
    return system, params


def _point(x, sd: float = 1e-3) -> Gaussian:
    """A sharp Gaussian belief at ``x`` — the point evidence a tick supplies."""
    x = jnp.asarray(x, jnp.float32)
    return Gaussian(loc=x, chol=sd * jnp.eye(x.shape[-1]))


# ------------------------------------------------------------------ the family contract


def test_log_prob_sample_and_moments_are_one_conditional():
    """A1's three members are three views of one diagonal Gaussian: ``log_prob``
    matches the closed form built from ``moments``, ``sample`` has the child's
    shape, and the sample mean/variance converge on the predicted moments."""
    net = _net()
    parents = (jnp.asarray([0.3, -0.7]), jax.nn.one_hot(1, NUM_CONTROLS))
    mean, cov = net.moments(parents)
    assert mean.shape == (DIM,)
    assert cov.shape == (DIM, DIM)
    np.testing.assert_allclose(np.asarray(cov - jnp.diag(jnp.diagonal(cov))), 0.0, atol=0.0)

    child = jnp.asarray([0.1, 0.2])
    var = jnp.diagonal(cov)
    manual = -0.5 * jnp.sum((child - mean) ** 2 / var + jnp.log(2 * jnp.pi * var))
    lp = net.log_prob(child, parents)
    assert lp.shape == ()
    assert float(lp) == pytest.approx(float(manual), rel=1e-5)

    keys = jax.random.split(jax.random.PRNGKey(3), 20000)
    draws = jax.vmap(lambda k: net.sample(k, parents))(keys)
    assert draws.shape == (20000, DIM)
    np.testing.assert_allclose(np.asarray(draws.mean(axis=0)), np.asarray(mean), atol=0.03)
    np.testing.assert_allclose(np.asarray(draws.var(axis=0)), np.asarray(var), rtol=0.05)


def test_scale_is_floored_and_the_head_starts_near_init_scale():
    """The softplus floor is what keeps the conditional log-density finite; the
    final layer's bias centres the initial scale on ``init_scale``."""
    floored = _net(min_scale=0.5, init_scale=1.0)
    parents = (jnp.asarray([5.0, -5.0]), jax.nn.one_hot(0, NUM_CONTROLS))
    _, cov = floored.moments(parents)
    assert float(jnp.min(jnp.sqrt(jnp.diagonal(cov)))) >= 0.5

    wide = _net(init_scale=2.0)
    _, cov2 = wide.moments((jnp.zeros(DIM), jax.nn.one_hot(0, NUM_CONTROLS)))
    np.testing.assert_allclose(np.asarray(jnp.sqrt(jnp.diagonal(cov2))), 2.0, atol=0.5)


def test_parent_widths_and_the_one_hot_convention_are_enforced():
    """Parents are unbatched vectors, one per declared parent, a discrete parent
    as a one-hot of its level count — mismatches are loud, not broadcast."""
    net = _net()
    good = (jnp.zeros(DIM), jax.nn.one_hot(2, NUM_CONTROLS))
    net.log_prob(jnp.zeros(DIM), good)  # the convention, spelled out
    with pytest.raises(ValueError, match="1 parent value.* for 2 declared parent"):
        net.log_prob(jnp.zeros(DIM), (jnp.zeros(DIM),))
    with pytest.raises(ValueError, match=r"parent 1 has shape \(2,\), expected \(3,\)"):
        net.log_prob(jnp.zeros(DIM), (jnp.zeros(DIM), jnp.zeros(2)))
    with pytest.raises(ValueError, match="batch with vmap"):
        net.log_prob(jnp.zeros(DIM), (jnp.zeros((4, DIM)), jax.nn.one_hot(0, NUM_CONTROLS)))


def test_construction_validates_shapes_and_options():
    with pytest.raises(ValueError, match="non-empty tuple of positive widths"):
        MLPConditional.from_shapes((), DIM, key=KEY)
    with pytest.raises(ValueError, match="unknown activation"):
        # (was "silu", which the P1′ chunk-1 heads added to the known set)
        MLPConditional.from_shapes((2,), 1, key=KEY, activation="mystery")
    with pytest.raises(ValueError, match="init_scale"):
        MLPConditional.from_shapes((2,), 1, key=KEY, min_scale=1.0, init_scale=0.5)
    net = _net()
    with pytest.raises(ValueError, match="final layer emits"):
        dataclasses.replace(net, child_dim=DIM + 1)
    with pytest.raises(ValueError, match="parents concatenate to"):
        dataclasses.replace(net, parent_dims=(DIM, NUM_CONTROLS + 1))


def test_it_deliberately_does_not_claim_the_learnable_protocol():
    """A net has no posterior over its weights, so ``point()`` /
    ``expected_log_params()`` / ``novelty()`` have no referents — and
    ``isinstance(decl, Learnable)`` must stay False so the conjugate paths
    (``as_env`` lowering, ``learn``'s conjugate branches) leave it alone."""
    assert not isinstance(_net(), Learnable)
    assert registry.param_family(_net()) == registry.MLP
    assert registry.novelty_estimator("transition", _net()) is None


# ------------------------------------------------------------------- slot integration


def test_a_non_frozen_net_is_one_learnable_parameter_variable():
    """V1's multiplicity rule: the weights are ONE posterior-free trainable block,
    so there is one variable and no sub-keys to address."""
    system, params = _mlp_transition_system()
    from hybrid_aif.system import _param_blocks, _slot_learnable

    decl = params.transition["x"]
    assert _slot_learnable(decl) and _param_blocks(decl) == ()
    variables = system._epistemic_variables(params)
    assert set(variables) == {"x", "x:dim0", "x:dim1", "transition:x"}
    assert variables["transition:x"] == ("param", 0, "transition", None)
    assert [k for k, _, _, _ in system._gradient_slots(params)] == ["transition:x"]


def test_frozen_net_is_not_learnable_and_not_rate_addressable():
    system, params = _mlp_transition_system()
    frozen = dataclasses.replace(params, transition={"x": dataclasses.replace(_net(), frozen=True)})
    assert set(system._epistemic_variables(frozen)) == {"x", "x:dim0", "x:dim1"}
    assert system._gradient_slots(frozen) == ()
    with pytest.raises(ValueError, match="frozen MLPConditional, which has no trainable"):
        system._resolve_rates({"transition:x": 1.0}, "lr_by_slot", frozen)
    with pytest.raises(ValueError, match="frozen MLPConditional"):
        dataclasses.replace(system, epistemic=("transition:x",))._term_targets(frozen)


def test_rate_keys_including_wildcards_resolve_to_the_net():
    """The rate grammar is family-agnostic: literal, class and cross-slot patterns
    all land on the neural variable, and the specificity ladder still decides."""
    system, params = _mlp_transition_system()
    for mapping in ({"transition:x": 0.3}, {"transition:*": 0.3}, {"*:*": 0.3}, {"*:x": 0.3}):
        assert system._resolve_rates(mapping, "lr_by_slot", params) == {"transition:x": 0.3}
    assert system._resolve_rates({"*:*": 0.1, "transition:x": 0.9}, "lr_by_slot", params) == {
        "transition:x": 0.9
    }
    # no independent blocks, so a third segment addresses nothing (V1)
    with pytest.raises(ValueError, match="single parameter posterior"):
        system._resolve_rates({"transition:x:head": 1.0}, "lr_by_slot", params)
    # ...and the block *class* through the wildcard is legitimately "all of none"
    assert system._resolve_rates({"*:*:*": 1.0}, "lr_by_slot", params) == {}


def test_epistemic_selection_raises_the_missing_estimator_error():
    """The registry has no novelty estimator for the ``mlp`` family and must not
    silently drop the variable — literal keys, ``True``, and every wildcard that
    *matches* it raise, exactly as the LG noise block does (V4/V7, and the pinned
    behaviour of test_epistemic_selector.py)."""
    system, params = _mlp_transition_system()

    def resolve(sel):
        return dataclasses.replace(system, epistemic=sel)._term_targets(params)

    with pytest.raises(NotImplementedError, match="must not silently ignore"):
        resolve(True)
    for sel in (("transition:x",), ("*:*",), ("transition:*",), ("*:x",)):
        with pytest.raises(NotImplementedError, match="deferred"):
            resolve(sel)
    # ...but the *block* class through its wildcard stays silent, because the net
    # has no independent blocks: "all of none", exactly as a single-posterior
    # ConditionalDirichlet's ':*' is (test_epistemic_selector's empty-class rule)
    assert resolve(("transition:x:*",)) == ((), (), ())
    # the state-only class stays expressible; x has no channel, so it selects nothing (V11)
    assert resolve(("*",)) == ((), (), ())
    assert resolve(False) == ((), (), ())


def test_declaration_shape_errors_name_the_slot_convention():
    """The class is role-free, so the SLOT supplies the parent order and widths —
    and that is where the two are checked against each other (A1)."""
    system, params = _mlp_transition_system()
    beliefs = {"x": _point([0.0, 0.0])}
    wrong_parents = dataclasses.replace(
        params, transition={"x": MLPConditional.from_shapes((DIM,), DIM, key=KEY, width=8, depth=1)}
    )
    with pytest.raises(ValueError, match="the one-hot of a discrete level"):
        system.infer_policies(wrong_parents, beliefs)
    wrong_child = dataclasses.replace(
        params,
        transition={
            "x": MLPConditional.from_shapes(
                (DIM, NUM_CONTROLS), DIM + 1, key=KEY, width=8, depth=1
            ),
        },
    )
    with pytest.raises(ValueError, match="child_dim=3 but the slot's child is 2-dimensional"):
        system.update_empirical_prior(wrong_child, beliefs, {"a_x": jnp.asarray(0)})


def test_neural_transition_rejects_discrete_parents_and_mixture_priors():
    """P1's prediction row carries ONE Gaussian: a mode parent would grow the
    belief through one net per mode, which is deferred rather than approximated."""
    system = System(
        A_deps=(),
        B_deps=((), (0,)),
        controlled=(False, True),
        policy_len=1,
        factor_names=("mode", "x"),
        epistemic=False,
        mixture_k=2,
    )
    params = SystemParams(
        likelihood={},
        transition={
            "mode": ConditionalCategorical(log_table=jnp.zeros((2, 1)), parent_ndims=1),
            "x": MLPConditional.from_shapes((DIM, NUM_CONTROLS), DIM, key=KEY, width=8, depth=1),
        },
        initial_state_prior={
            "mode": Categorical(logits=jnp.zeros(2)),
            "x": Gaussian(loc=jnp.zeros(DIM), chol=jnp.eye(DIM)),
        },
    )
    with pytest.raises(NotImplementedError, match="one mode-conditioned net per mode"):
        system._families(params)


def test_neural_likelihood_is_declarable_but_conditioning_through_it_is_unbuilt():
    """A1(a)/A1(b) split: the declaration is legal, rate-addressable and
    gradient-learnable, but ``infer_states`` through a net needs an inference
    strategy the registry does not have (structured filtering, or the P3
    recognition row)."""
    system = System(
        A_deps=((0,),),
        B_deps=((),),
        controlled=(True,),
        policy_len=1,
        modality_names=("y",),
        factor_names=("x",),
        epistemic=False,
    )
    obs_net = MLPConditional.from_shapes((DIM,), 1, key=KEY, width=8, depth=1)
    params = SystemParams(
        likelihood={"y": obs_net},
        transition={"x": _net()},
        initial_state_prior={"x": Gaussian(loc=jnp.zeros(DIM), chol=jnp.eye(DIM))},
    )
    assert set(system._epistemic_variables(params)) == {
        "x",
        "x:dim0",
        "x:dim1",
        "obs:y",
        "likelihood:y",
        "transition:x",
    }
    assert system._resolve_rates({"*:*": 0.5}, "lr_by_slot", params) == {
        "likelihood:y": 0.5,
        "transition:x": 0.5,
    }
    with pytest.raises(NotImplementedError, match="A1\\(b\\), the P3 track"):
        system.infer_states(params, {"y": jnp.asarray([0.3])})
    # ...while the FORWARD direction is fine: as_env samples observations through
    # the net (the direction a net is unproblematic in — A1's member (ii))
    env, ep = system.as_env(params)
    obs, _ = env.reset(jax.random.PRNGKey(4), ep, state={"x": jnp.zeros(DIM)})
    assert obs["y"].shape == (1,)
    # ...and the gradient path still learns it from beliefs supplied elsewhere
    beliefs = {"x": _point([0.2, -0.1])}
    out, _ = system.learn(
        params,
        {"y": jnp.asarray([0.3])},
        beliefs,
        prev_beliefs=beliefs,
        action={"a_x": jnp.asarray(1)},
        lr=0.05,
        lr_by_slot={"transition:x": 0.0},
        learn_state=system.init_learn_state(params),
    )
    assert not np.array_equal(
        np.asarray(out.likelihood["y"].layers[0].weight), np.asarray(obs_net.layers[0].weight)
    )
    # a C designation over a neural modality would be silently dropped, so it raises —
    # the one modality kind still fenced now that LG modalities are supported: a net has
    # no closed-form pushforward of the belief to score the target against (A1)
    targeted = dataclasses.replace(params, obs_targets={"y": Categorical(logits=jnp.zeros(2))})
    with pytest.raises(ValueError, match="obs_targets on neural modality"):
        system.update_empirical_prior(targeted, beliefs, {"a_x": jnp.asarray(0)})
    with pytest.raises(ValueError, match="amortization.md A1"):
        system.update_empirical_prior(targeted, beliefs, {"a_x": jnp.asarray(0)})


# -------------------------------------------------------------- the prediction row (A1(a))


def test_linearized_prediction_matches_the_lg_predict_for_an_affine_head():
    """A1(a)'s validation story (A10 disposition 10): no R-AIF oracle exists, so
    the row is pinned against our *own* exact LG path — a depth-0 net's mean head
    IS affine, so the linearization must reproduce
    :func:`~hybrid_aif.inference.linear_gaussian.predict` exactly."""
    affine = MLPConditional.from_shapes(
        (DIM, NUM_CONTROLS), DIM, key=jax.random.PRNGKey(11), depth=0, init_scale=0.3
    )
    system, params = _mlp_transition_system()
    params = dataclasses.replace(params, transition={"x": affine})
    q = Gaussian(loc=jnp.asarray([0.4, -0.6]), chol=jnp.asarray([[0.5, 0.0], [0.2, 0.3]]))
    action = {"a_x": jnp.asarray(2)}

    got = system.update_empirical_prior(params, {"x": q}, action)["x"]
    # the affine head, read off the single Linear layer restricted to the state part
    w = affine.layers[0].weight
    F = w[:DIM, :DIM]
    head = affine.condition_on((q.loc, jax.nn.one_hot(2, NUM_CONTROLS)))
    want = predict(q, F, head.loc - F @ q.loc, head.chol)
    np.testing.assert_allclose(np.asarray(got.loc), np.asarray(want.loc), atol=1e-5)
    np.testing.assert_allclose(np.asarray(got.cov), np.asarray(want.cov), atol=1e-5)


def test_prediction_propagates_the_belief_spread_through_the_net():
    """The nonlinear case: the mean is the net's mean at the belief mean, and the
    covariance is ``J Σ Jᵀ`` plus the head's own noise — so a wider prior
    predicts a wider posterior, and a point belief predicts the head alone."""
    system, params = _mlp_transition_system()
    net, action = params.transition["x"], {"a_x": jnp.asarray(1)}
    mu = jnp.asarray([0.5, -0.5])
    head = net.condition_on((mu, jax.nn.one_hot(1, NUM_CONTROLS)))

    sharp = system.update_empirical_prior(params, {"x": _point(mu, sd=1e-6)}, action)["x"]
    np.testing.assert_allclose(np.asarray(sharp.loc), np.asarray(head.loc), atol=1e-6)
    np.testing.assert_allclose(np.asarray(sharp.cov), np.asarray(head.cov), atol=1e-6)

    wide = system.update_empirical_prior(params, {"x": _point(mu, sd=0.4)}, action)["x"]
    np.testing.assert_allclose(np.asarray(wide.loc), np.asarray(head.loc), atol=1e-6)
    assert np.all(np.diagonal(np.asarray(wide.cov)) > np.diagonal(np.asarray(head.cov)))

    jac = jax.jacfwd(lambda x: net.condition_on((x, jax.nn.one_hot(1, NUM_CONTROLS))).loc)(mu)
    manual = jac @ (0.16 * jnp.eye(DIM)) @ jac.T + head.cov
    np.testing.assert_allclose(np.asarray(wide.cov), np.asarray(manual), atol=1e-5)


def test_planning_runs_through_a_neural_transition():
    """The prediction row is the *planning* seam too (the same closure the planner,
    ``predict_beliefs`` and ``update_empirical_prior`` share): a risk designation
    over the continuous factor scores every enumerated policy through the net, and
    ``act`` jits. This is the P1 shape, minus CEM."""
    from hybrid_aif.objectives import total

    system, params = _mlp_transition_system()
    system = dataclasses.replace(system, policy_len=3)
    params = dataclasses.replace(
        params,
        state_targets={"x": Gaussian(loc=jnp.asarray([1.0, 1.0]), chol=0.3 * jnp.eye(DIM))},
    )
    beliefs = {"x": _point([0.0, 0.0], sd=0.2)}
    q_pi, tree = system.infer_policies(params, beliefs)
    assert sorted(tree) == ["risk"] and sorted(tree["risk"]) == ["x"]
    scores = np.asarray(total(tree).lower)
    assert scores.shape == (NUM_CONTROLS**3,)
    assert np.all(np.isfinite(scores)) and scores.std() > 0.0
    assert np.all(np.isfinite(np.asarray(q_pi.logits)))
    action = jax.jit(lambda p, b: system.act(jax.random.PRNGKey(0), p, b))(params, beliefs)[0]
    assert system.to_positional("actions", action).shape == (1,)


def test_as_env_samples_ancestrally_through_the_net():
    """A1's contract member (ii) at System level: the same declaration run
    forward. No posterior over the weights means ``param_key`` randomizes
    nothing here — the env is the net itself."""
    system, params = _mlp_transition_system()
    env, ep = system.as_env(params)
    assert ep.transition["x"] is params.transition["x"]
    _, state = env.reset(jax.random.PRNGKey(1), ep, state={"x": jnp.zeros(DIM)})
    _, nxt = env.step(jax.random.PRNGKey(2), state, {"a_x": jnp.asarray(1)}, ep)
    assert nxt["x"].shape == (DIM,)
    _, ep_key = system.as_env(params, param_key=jax.random.PRNGKey(5))
    np.testing.assert_array_equal(
        np.asarray(ep_key.transition["x"].layers[0].weight),
        np.asarray(params.transition["x"].layers[0].weight),
    )


# ------------------------------------------------------- A2: gradient learning + LearnState


def _linear_world(seed=0, n=6000):
    """A known linear system ``s' = A s + b_a``, sampled at random states/actions."""
    rng = np.random.default_rng(seed)
    A = np.asarray([[0.9, 0.15], [-0.1, 0.8]], np.float32)
    b = np.asarray([[0.4, 0.0], [0.0, 0.4], [-0.3, -0.3]], np.float32)
    s = np.float32(rng.normal(scale=0.7, size=(n, DIM)))
    a = np.int32(rng.integers(NUM_CONTROLS, size=n))
    s_next = s @ A.T + b[a]
    return A, b, jnp.asarray(s), jnp.asarray(a), jnp.asarray(s_next)


def _mean_error(net, A, b, s, a):
    """RMS error of the net's mean head against the true next state."""
    truth = s @ jnp.asarray(A).T + jnp.asarray(b)[a]
    pred = jax.vmap(lambda si, ai: net.condition_on((si, jax.nn.one_hot(ai, NUM_CONTROLS))).loc)(
        s, a
    )
    return float(jnp.sqrt(jnp.mean((pred - truth) ** 2)))


def test_gradient_learning_recovers_a_known_linear_transition():
    """A2 end to end: ``stats`` = the gradient of the slot's negative conditional
    log-density on this tick's evidence, ``absorb`` = the optax step, optimizer
    state threaded through the explicit ``LearnState`` carry. Trained tick by tick
    inside ``lax.scan`` — the shape the P1 loop will use — under a traced,
    linearly-annealed rate, which is the in-scan curriculum form of the rate
    grammar applied to a neural slot.

    Acceptance is *held-out* prediction error of the mean head against the true
    ``A s + b_a``, since the state scale is ~0.7: under 0.05 RMSE, and at least
    a tenfold improvement over the freshly-initialized net."""
    system, params = _mlp_transition_system()
    n, lr0 = 8000, 8e-3
    A, b, s, a, s_next = _linear_world(seed=0, n=n)
    holdout = _linear_world(seed=101, n=2000)[2:]
    net0 = params.transition["x"]
    before = _mean_error(net0, A, b, *holdout[:2])

    def tick(carry, xs):
        decl, ls = carry
        t, s_t, a_t, s_next_t = xs
        p = dataclasses.replace(params, transition={"x": decl})
        p, ls = system.learn(
            p,
            {},
            {"x": _point(s_next_t)},
            prev_beliefs={"x": _point(s_t)},
            action={"a_x": a_t},
            lr_by_slot={"transition:x": lr0 * (1.0 - t / n)},
            learn_state=ls,
        )
        return (p.transition["x"], ls), None

    init = (net0, LearnState(opt_states={"transition:x": net0.init_opt_state()}))
    ts = jnp.arange(n, dtype=jnp.float32)
    (trained, final_state), _ = jax.lax.scan(tick, init, (ts, s, a, s_next))
    assert set(final_state.opt_states) == {"transition:x"}  # the carry survived the scan

    after = _mean_error(trained, A, b, *holdout[:2])
    assert after < 0.05, f"held-out RMSE {after:.4f} (started at {before:.4f})"
    assert after < before / 10.0
    # the learned head's own noise has tightened toward the (noiseless) truth
    scale = jnp.sqrt(jnp.diagonal(trained.moments((s[0], jax.nn.one_hot(a[0], NUM_CONTROLS)))[1]))
    assert float(jnp.max(scale)) < 0.5


def test_learn_state_round_trips_under_jit_and_is_not_optional_bookkeeping():
    """The carry's pytree structure is stable across calls (so it rides a scan),
    it is honoured (a resumed step differs from a restarted one, because adam's
    moments carry), and a stale carry is loud."""
    system, params = _mlp_transition_system()
    prev, cur, action = (
        {"x": _point([0.3, -0.2])},
        {"x": _point([0.25, -0.1])},
        {"a_x": jnp.asarray(1)},
    )

    @jax.jit
    def step(p, ls):
        return system.learn(p, {}, cur, prev_beliefs=prev, action=action, lr=0.02, learn_state=ls)

    p1, ls1 = system.learn(
        params,
        {},
        cur,
        prev_beliefs=prev,
        action=action,
        lr=0.02,
        learn_state=system.init_learn_state(params),
    )
    assert set(ls1.opt_states) == {"transition:x"}
    p2, ls2 = step(p1, ls1)
    assert jax.tree_util.tree_structure(ls1) == jax.tree_util.tree_structure(ls2)
    # an EXPLICIT restart is expressible (and loses adam's moments)...
    p3, _ = step(p1, system.init_learn_state(p1))
    assert not np.allclose(
        np.asarray(p2.transition["x"].layers[0].weight),
        np.asarray(p3.transition["x"].layers[0].weight),
    )
    # ...but a dropped carry is NOT a restart: it fails loudly (A2's guard)
    with pytest.raises(ValueError, match="init_learn_state"):
        system.learn(p1, {}, cur, prev_beliefs=prev, action=action, lr=0.02)
    with pytest.raises(ValueError, match="the carry is stale"):
        system.learn(
            params,
            {},
            cur,
            prev_beliefs=prev,
            action=action,
            learn_state=LearnState(opt_states={"likelihood:nope": ls1.opt_states["transition:x"]}),
        )


def test_return_type_follows_the_declarations_not_the_call():
    """The judgment call, pinned: ``learn`` returns bare params for a
    conjugate-only model (every v0.4 caller unchanged) and the ``(params,
    LearnState)`` pair exactly when a gradient-learnable slot exists — static
    pytree structure, hence decidable at the call site."""
    system, params = _mlp_transition_system()
    prev, cur, action = (
        {"x": _point([0.1, 0.1])},
        {"x": _point([0.2, 0.0])},
        {"a_x": jnp.asarray(0)},
    )
    out = system.learn(
        params,
        {},
        cur,
        prev_beliefs=prev,
        action=action,
        lr=0.01,
        learn_state=system.init_learn_state(params),
    )
    assert isinstance(out, tuple) and isinstance(out[1], LearnState)

    frozen = dataclasses.replace(params, transition={"x": dataclasses.replace(_net(), frozen=True)})
    bare = system.learn(frozen, {}, cur, prev_beliefs=prev, action=action, lr=0.01)
    assert isinstance(bare, SystemParams)
    # a static 0.0 rate skips the *update* but keeps the state entry (the carry's
    # structure is a function of the declarations, so it survives a scan)
    still_paired = system.learn(
        params,
        {},
        cur,
        prev_beliefs=prev,
        action=action,
        lr_by_slot={"transition:x": 0.0},
        learn_state=system.init_learn_state(params),
    )
    assert isinstance(still_paired, tuple)
    assert set(still_paired[1].opt_states) == {"transition:x"}


def test_frozen_declaration_comes_back_bitwise():
    """A1's frozen marking is structural: no gradient is taken, no optimizer state
    allocated, and the declaration is the same object on the way out."""
    system, params = _mlp_transition_system()
    frozen_net = dataclasses.replace(_net(), frozen=True)
    frozen = dataclasses.replace(params, transition={"x": frozen_net})
    prev, cur = {"x": _point([0.1, 0.1])}, {"x": _point([0.9, -0.4])}
    out = system.learn(frozen, {}, cur, prev_beliefs=prev, action={"a_x": jnp.asarray(2)}, lr=1.0)
    assert out.transition["x"] is frozen_net
    with pytest.raises(ValueError, match="frozen ⇔ no trainable marking"):
        frozen_net.stats(cur["x"].loc, (prev["x"].loc, jax.nn.one_hot(2, NUM_CONTROLS)))


def test_a_static_zero_rate_freezes_the_slot_and_a_traced_one_gates_it():
    """The rate grammar unchanged (A2): a static ``0.0`` skips the slot at trace
    time; a traced rate multiplies the step, which is what makes an in-scan
    curriculum expressible on a neural slot too."""
    system, params = _mlp_transition_system()
    net = params.transition["x"]
    prev, cur, action = (
        {"x": _point([0.4, 0.1])},
        {"x": _point([0.1, 0.5])},
        {"a_x": jnp.asarray(1)},
    )

    static_zero, _ = system.learn(
        params,
        {},
        cur,
        prev_beliefs=prev,
        action=action,
        lr_by_slot={"transition:x": 0.0},
        learn_state=system.init_learn_state(params),
    )
    assert static_zero.transition["x"] is net

    @jax.jit
    def step(p, r):
        return system.learn(
            p,
            {},
            cur,
            prev_beliefs=prev,
            action=action,
            lr_by_slot={"transition:*": r},
            learn_state=system.init_learn_state(p),
        )

    live, _ = step(params, jnp.asarray(0.05))
    dead, _ = step(params, jnp.asarray(0.0))
    np.testing.assert_array_equal(
        np.asarray(dead.transition["x"].layers[0].weight), np.asarray(net.layers[0].weight)
    )
    assert not np.allclose(
        np.asarray(live.transition["x"].layers[0].weight), np.asarray(net.layers[0].weight)
    )


def test_decay_on_a_neural_slot_defers_to_the_design_doc():
    """Decay semantics for network weights are deliberately not invented (A2)."""
    system, params = _mlp_transition_system()
    prev, cur, action = (
        {"x": _point([0.0, 0.0])},
        {"x": _point([0.1, 0.1])},
        {"a_x": jnp.asarray(0)},
    )
    ls = system.init_learn_state(params)
    for kwargs in ({"decay": 0.99}, {"decay_by_slot": {"transition:x": 0.99}}):
        with pytest.raises(NotImplementedError, match="amortization.md A2"):
            system.learn(
                params, {}, cur, prev_beliefs=prev, action=action, learn_state=ls, **kwargs
            )
    # decay=1.0 (the default) is the ordinary path
    system.learn(params, {}, cur, prev_beliefs=prev, action=action, decay=1.0, learn_state=ls)


def test_active_gradient_transition_requires_prev_beliefs_and_action():
    """The same carry requirements the conjugate transition path enforces, in the
    same words: the net is fit on (s_{t-1}, a_{t-1}) -> s_t."""
    system, params = _mlp_transition_system()
    cur = {"x": _point([0.2, 0.2])}
    with pytest.raises(ValueError, match="prev_beliefs"):
        system.learn(params, {}, cur)
    with pytest.raises(ValueError, match="needs action"):
        system.learn(params, {}, cur, prev_beliefs=cur)
    frozen_by_rate = system.learn(
        params,
        {},
        cur,
        lr_by_slot={"transition:x": 0.0},
        learn_state=system.init_learn_state(params),
    )
    assert frozen_by_rate[0].transition["x"] is params.transition["x"]
