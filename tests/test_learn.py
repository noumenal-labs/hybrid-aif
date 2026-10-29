"""Parameter learning: the ``learn`` face, polymorphic slots, rates, and decay.

Acceptance tests from ``docs/design/parameter-learning.md`` (build-plan steps
1-2): kernel parity A/B against pymdp (1, 2), draw-then-recover through
``as_env(params, param_key=...)`` (3), the structural no-sidecar test (5), the
frozen-but-distributional round trip (7), novelty without learning (8),
``decay=1.0`` bitwise conjugacy (9), plus implementation gaps 1 (tick-0 traced
rate inside ``lax.scan``) and 7 (the E[log A] Jensen divergence is monotone
decreasing in the concentration total). Parity binds at the kernel leaf, not
the loop (D2): a learning agent's posteriors deliberately diverge from pymdp's
because FPI takes ``E[log A]`` while pymdp syncs ``A = E[A]``.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pymdp.learning import update_obs_likelihood_dirichlet, update_state_transition_dirichlet_f

from hybrid_aif import (
    Categorical,
    ConditionalCategorical,
    ConditionalDirichlet,
    Dirichlet,
)
from hybrid_aif.envs import GenerativeEnvParams
from hybrid_aif.envs._process import HybridProcess
from hybrid_aif.inference import fpi
from hybrid_aif.system import System, SystemParams

KEY = jax.random.PRNGKey(0)


def _prob_table(rng, shape):
    g = rng.gamma(1.0, size=shape).astype(np.float32)
    return jnp.asarray(g / g.sum(axis=0, keepdims=True))


def _near_identity(n, eps=1e-3):
    table = np.full((n, n), eps, np.float32) + (1.0 - n * eps) * np.eye(n, dtype=np.float32)
    return ConditionalCategorical(log_table=jnp.log(jnp.asarray(table)), parent_ndims=1)


def _sense_system(ns=3, no=4, nu=2, seed=5, conc=2.0, **system_kwargs):
    """One factor; modality 'sense' is a learnable ConditionalDirichlet, modality
    'ident' a near-identity point table (sharp posteriors for recovery)."""
    rng = np.random.default_rng(seed)
    system = System(
        A_deps=((0,), (0,)),
        B_deps=((0,),),
        controlled=(True,),
        policy_len=1,
        modality_names=("sense", "ident"),
        factor_names=("s",),
        **system_kwargs,
    )
    params = SystemParams(
        likelihood={
            "sense": ConditionalDirichlet(concentration=jnp.full((no, ns), conc), parent_ndims=1),
            "ident": _near_identity(ns),
        },
        transition={
            "s": ConditionalCategorical(
                log_table=jnp.log(_prob_table(rng, (ns, ns, nu))), parent_ndims=2
            ),
        },
        initial_state_prior={"s": Categorical(logits=jnp.zeros(ns))},
        obs_targets={"ident": Categorical(logits=jnp.zeros(ns))},
    )
    return system, params


def _tick_inputs(system, params, seed=7):
    rng = np.random.default_rng(seed)
    ns = params.initial_state_prior["s"].num_levels
    no = params.likelihood["sense"].num_levels
    obs = {
        "sense": jnp.asarray(rng.integers(no), jnp.int32),
        "ident": jnp.asarray(rng.integers(ns), jnp.int32),
    }
    beliefs, _ = system.infer_states(params, obs)
    return obs, beliefs


# --------------------------------------------- acceptance 1: kernel parity A


def _multi_modality_model(seed=11):
    """Two modalities over two factors (one multi-parent), random posteriors/obs."""
    rng = np.random.default_rng(seed)
    sizes = (3, 4)
    A_deps = ((0, 1), (1,))
    num_obs = (4, 3)
    pA = [
        np.float32(rng.gamma(2.0, size=(n, *[sizes[f] for f in deps])) + 0.1)
        for n, deps in zip(num_obs, A_deps, strict=True)
    ]
    system = System(
        A_deps=A_deps,
        B_deps=((0,), (1,)),
        controlled=(False, False),
        policy_len=1,
    )
    params = system.from_positional(
        "params",
        SystemParams(
            likelihood=tuple(
                ConditionalDirichlet(concentration=jnp.asarray(p), parent_ndims=p.ndim - 1)
                for p in pA
            ),
            transition=tuple(
                ConditionalCategorical(
                    log_table=jnp.log(_prob_table(rng, (s, s, 1))), parent_ndims=2
                )
                for s in sizes
            ),
            initial_state_prior=tuple(Categorical(logits=jnp.zeros(s)) for s in sizes),
        ),
    )
    return system, params, sizes, num_obs, pA, rng


@pytest.mark.parametrize("T", [1, 25])
@pytest.mark.parametrize("categorical_obs", [False, True])
def test_likelihood_kernel_parity_over_sequence(T, categorical_obs):
    """Our per-tick ``learn`` folded over a sequence equals pymdp's summed
    ``update_obs_likelihood_dirichlet``: atol 1e-6 at T=1 (bitwise-adjacent),
    rtol 1e-5 over the sequence (incremental vs sum-then-add accumulation)."""
    system, params, sizes, num_obs, pA, rng = _multi_modality_model()
    qs_seq = [np.float32(rng.dirichlet(np.ones(s), size=T)) for s in sizes]
    obs_idx = [np.int32(rng.integers(n, size=T)) for n in num_obs]
    obs_oh = [np.float32(np.eye(n)[o]) for n, o in zip(num_obs, obs_idx, strict=True)]

    p = params
    for t in range(T):
        beliefs = tuple(Categorical(logits=jnp.log(jnp.asarray(q[t]))) for q in qs_seq)
        obs_t = tuple(
            jnp.asarray(o[t]) if categorical_obs else jnp.asarray(o[t], jnp.int32)
            for o in (obs_oh if categorical_obs else obs_idx)
        )
        p = system.learn(
            p,
            system.from_positional("observations", obs_t),
            system.from_positional("beliefs", beliefs),
        )

    qA_ref, _ = update_obs_likelihood_dirichlet(
        [jnp.asarray(a) for a in pA],
        [jnp.asarray(a / a.sum(axis=0, keepdims=True)) for a in pA],
        [jnp.asarray(o) for o in (obs_oh if categorical_obs else obs_idx)],
        [jnp.asarray(q) for q in qs_seq],
        A_dependencies=[list(d) for d in system.A_deps],
        categorical_obs=categorical_obs,
        num_obs=list(num_obs),
        lr=1.0,
    )
    ours_in_order = [p.likelihood[m] for m in system.names.modalities]
    for ours, ref in zip(ours_in_order, qA_ref, strict=True):
        if T == 1:
            np.testing.assert_allclose(np.asarray(ours.concentration), np.asarray(ref), atol=1e-6)
        else:
            np.testing.assert_allclose(np.asarray(ours.concentration), np.asarray(ref), rtol=1e-5)


# --------------------------------------------- acceptance 2: kernel parity B


def test_transition_kernel_parity():
    """``learn`` folded over (q_{t-1}, a_{t-1}, q_t) triples equals pymdp's
    ``update_state_transition_dirichlet_f`` given the same marginals and action
    one-hots, 1e-6 (kernel-level parity boundary, D4)."""
    rng = np.random.default_rng(3)
    ns, nu, T = 4, 3, 6
    pB = np.float32(rng.gamma(2.0, size=(ns, ns, nu)) + 0.1)
    system = System(
        A_deps=(),
        B_deps=((0,),),
        controlled=(True,),
        policy_len=1,
        factor_names=("s",),
    )
    params = SystemParams(
        likelihood={},
        transition={"s": ConditionalDirichlet(concentration=jnp.asarray(pB), parent_ndims=2)},
        initial_state_prior={"s": Categorical(logits=jnp.zeros(ns))},
    )
    q_seq = np.float32(rng.dirichlet(np.ones(ns), size=T + 1))
    actions = np.int32(rng.integers(nu, size=T))

    p = params
    for t in range(T):
        p = system.learn(
            p,
            {},
            {"s": Categorical(logits=jnp.log(jnp.asarray(q_seq[t + 1])))},
            prev_beliefs={"s": Categorical(logits=jnp.log(jnp.asarray(q_seq[t])))},
            action=system.from_positional("actions", jnp.asarray([actions[t]], jnp.int32)),
        )

    qB_ref, _ = update_state_transition_dirichlet_f(
        jnp.asarray(pB),
        jax.nn.one_hot(jnp.asarray(actions), nu),
        [jnp.asarray(q_seq[1:]), jnp.asarray(q_seq[:-1])],
        lr=1.0,
    )
    np.testing.assert_allclose(
        np.asarray(p.transition["s"].concentration), np.asarray(qB_ref), atol=1e-6
    )


# ------------------------------------------------- acceptance 3: recovery


def test_recovery_of_table_drawn_from_own_prior():
    """The forward model closes the loop: the env's true likelihood table is a
    draw from the agent's own prior (``as_env(params, param_key=...)``), and
    learning on data generated through it recovers that draw — no external
    fixture, no hand-written 'wrong' table."""
    k_draw, k_run = jax.random.split(jax.random.PRNGKey(42))
    system, params = _sense_system(conc=2.0)
    env, env_params = system.as_env(params, param_key=k_draw)
    true_table = np.asarray(env_params.likelihood["sense"])  # the sampled table (raw)
    prior_mean = np.asarray(params.likelihood["sense"].mean())
    assert not np.allclose(true_table, prior_mean, atol=0.02)  # the draw moved off the prior

    nu = 2  # _sense_system's default control levels
    _, state0 = env.reset(k_run, env_params)

    def tick(carry, _):
        key, state, conc = carry
        k_act, k_env, key = jax.random.split(key, 3)
        action = jax.random.randint(k_act, (1,), 0, nu, dtype=jnp.int32)
        obs, state = env.step(k_env, state, system.from_positional("actions", action), env_params)
        p = dataclasses.replace(
            params,
            likelihood={
                "sense": ConditionalDirichlet(concentration=conc, parent_ndims=1),
                "ident": params.likelihood["ident"],
            },
        )
        beliefs, _ = system.infer_states(p, obs)
        p = system.learn(p, obs, beliefs)
        return (key, state, p.likelihood["sense"].concentration), None

    init = (k_run, state0, params.likelihood["sense"].concentration)
    (_, _, conc), _ = jax.lax.scan(tick, init, None, length=3000)
    learned = np.asarray(ConditionalDirichlet(concentration=conc, parent_ndims=1).mean())
    assert np.abs(learned - true_table).max() < 0.05
    assert np.abs(learned - true_table).max() < 0.3 * np.abs(prior_mean - true_table).max()


# ------------------------------------------------ acceptance 5: structural


def test_no_parameter_duplicating_field_on_system_params():
    """The binding criterion for the representation change (D3): no field of
    ``SystemParams`` duplicates a declaration's parameters — fails if ``pA``
    (or any sidecar) is reintroduced. Asserted on the field set, not object
    identity (JAX gives none to assert on).

    ``recognition`` (amortization.md A11) is a *fourth slot*, not a sidecar: it
    holds the parameters of the amortized inference strategy for the variables it
    names, which no generative declaration holds — the criterion is "no field
    duplicates a declaration's parameters", and an encoder is not a duplicate of
    anything. ``scoring`` (A12) is the fifth on the same reading: the critic, the
    info-gain ensemble and the CRSPP are parameters no declaration holds. ``control`` holds
    each continuous control node's box, its support, which no declaration holds either, and
    ``actuator`` each action node's kernel, which the environment draws from, and
    ``durations`` a round's length per level of the one control node that sets the clock, and
    ``precision`` the policy precision, a scalar no declaration holds. Which
    *group* each field belongs to is ``test_scoring_slot.py``'s membership test."""
    fields = {f.name for f in dataclasses.fields(SystemParams)}
    assert fields == {
        "likelihood",
        "transition",
        "initial_state_prior",
        "recognition",
        "obs_targets",
        "state_targets",
        "terminal_targets",
        "scoring",
        "policy_prior",
        "control",
        "actuator",
        "durations",
        "precision",
    }


# --------------------------------- acceptance 7 + 8: frozen but distributional


def test_frozen_slot_round_trips_losslessly():
    """D1's capability/activation split: with activation off (static rate 0.0)
    ``learn`` is bitwise identity on the concentration, ``as_env(param_key=...)``
    still samples, novelty is still computed; rate 1.0 resumes from the same
    counts (freezing was lossless)."""
    system, params = _sense_system()
    obs, beliefs = _tick_inputs(system, params)

    frozen = system.learn(params, obs, beliefs, lr=0.0)
    assert np.array_equal(
        np.asarray(frozen.likelihood["sense"].concentration),
        np.asarray(params.likelihood["sense"].concentration),
    )

    # still a distribution: as_env samples a table from it, per-key
    _, ep1 = system.as_env(frozen, param_key=jax.random.PRNGKey(1))
    _, ep2 = system.as_env(frozen, param_key=jax.random.PRNGKey(2))
    assert not np.allclose(np.asarray(ep1.likelihood["sense"]), np.asarray(ep2.likelihood["sense"]))
    np.testing.assert_allclose(np.asarray(ep1.likelihood["sense"]).sum(axis=0), 1.0, atol=1e-5)

    # novelty still computed while frozen (acceptance 8 pairs with this)
    _, tree = system.infer_policies(frozen, beliefs)
    assert "infogain" in tree and "likelihood:sense" in tree["infogain"]

    # resuming continues from the same counts
    resumed = system.learn(frozen, obs, beliefs, lr=1.0)
    direct = system.learn(params, obs, beliefs, lr=1.0)
    assert np.array_equal(
        np.asarray(resumed.likelihood["sense"].concentration),
        np.asarray(direct.likelihood["sense"].concentration),
    )


def test_novelty_without_learning():
    """The ablation D1 protects: epistemic drive on, counts never move."""
    system, params = _sense_system()
    obs, beliefs = _tick_inputs(system, params)
    q_pi, tree = system.infer_policies(params, beliefs)  # epistemic defaults True
    assert "likelihood:sense" in tree["infogain"]
    assert np.all(np.isfinite(np.asarray(q_pi.logits)))
    after = system.learn(params, obs, beliefs, lr=0.0)
    assert np.array_equal(
        np.asarray(after.likelihood["sense"].concentration),
        np.asarray(params.likelihood["sense"].concentration),
    )


# ------------------------------------------------- acceptance 9: decay


def test_decay_one_is_bitwise_conjugate():
    system, params = _sense_system()
    obs, beliefs = _tick_inputs(system, params)
    a = params.likelihood["sense"]
    manual = a.concentration + a.stats(obs["sense"], (beliefs["s"],))
    for kwargs in ({}, {"decay": 1.0}):
        out = system.learn(params, obs, beliefs, **kwargs)
        assert np.array_equal(np.asarray(out.likelihood["sense"].concentration), np.asarray(manual))


def test_decay_relaxes_toward_anchor():
    """``α ← decay·α + (1−decay)·anchor + lr·stats``: a declaration without an anchor cannot
    decay, a uniform anchor is a scalar, ``alpha0`` is the prior when set; a *traced* zero rate
    keeps the walk running while a static 0.0 skips decay too (the frozen-but-drifting caveat,
    gap 4)."""
    system, params = _sense_system()
    obs, beliefs = _tick_inputs(system, params)
    conc = params.likelihood["sense"].concentration
    traced_zero = jnp.asarray(0.0)

    with pytest.raises(ValueError, match="likelihood:sense: a decay other than"):
        system.learn(params, obs, beliefs, lr=traced_zero, decay=0.9)

    uniform = ConditionalDirichlet.from_table(
        params.likelihood["sense"].mean(), parent_ndims=1, strength=8.0, anchor="uniform"
    )
    assert uniform.alpha0.ndim == 0
    p_uniform = dataclasses.replace(
        params, likelihood={"sense": uniform, "ident": params.likelihood["ident"]}
    )
    drift = system.learn(p_uniform, obs, beliefs, lr=traced_zero, decay=0.9)
    np.testing.assert_allclose(
        np.asarray(drift.likelihood["sense"].concentration),
        np.asarray(0.9 * uniform.concentration + 0.1 * jnp.ones_like(conc)),
        atol=1e-6,
    )

    anchored = ConditionalDirichlet.from_table(
        params.likelihood["sense"].mean(), parent_ndims=1, strength=8.0, anchor=True
    )
    p_anchor = dataclasses.replace(
        params, likelihood={"sense": anchored, "ident": params.likelihood["ident"]}
    )
    drift_a = system.learn(p_anchor, obs, beliefs, lr=traced_zero, decay=0.9)
    np.testing.assert_allclose(
        np.asarray(drift_a.likelihood["sense"].concentration),
        np.asarray(0.9 * anchored.concentration + 0.1 * anchored.alpha0),
        atol=1e-6,
    )
    # alpha0 survives the update (the anchor is a standing property, not one tick's)
    assert drift_a.likelihood["sense"].alpha0 is not None

    # static-zero rate skips the slot entirely — no drift, and no anchor is needed
    static = system.learn(params, obs, beliefs, lr=0.0, decay=0.9)
    assert np.array_equal(np.asarray(static.likelihood["sense"].concentration), np.asarray(conc))


def test_anchor_readings():
    """``from_table``'s ``anchor``: the prior is the concentration itself, uniform a scalar,
    an array kept as given, ``False`` none; an initial-prior Dirichlet without ``alpha0``
    refuses to decay in the same words."""
    from hybrid_aif.distributions.dirichlet import anchor_for

    table = jnp.asarray([[0.7, 0.2], [0.3, 0.8]])
    prior = ConditionalDirichlet.from_table(table, parent_ndims=1, strength=4.0, anchor="prior")
    assert prior.alpha0 is prior.concentration
    uniform = ConditionalDirichlet.from_table(table, parent_ndims=1, anchor="uniform")
    assert uniform.alpha0.shape == () and float(uniform.alpha0) == 1.0
    given = ConditionalDirichlet.from_table(table, parent_ndims=1, anchor=jnp.full((2, 2), 0.5))
    assert given.alpha0.shape == (2, 2)
    assert ConditionalDirichlet.from_table(table, parent_ndims=1).alpha0 is None
    assert anchor_for(table, True) is table
    from_numpy = ConditionalDirichlet.from_table(table, parent_ndims=1, anchor=np.ones((2, 2)))
    assert from_numpy.alpha0.dtype == table.dtype
    with pytest.raises(ValueError, match="anchor must be"):
        ConditionalDirichlet.from_table(table, parent_ndims=1, anchor="max-ent")
    with pytest.raises(ValueError, match="scalar or the concentration shape"):
        ConditionalDirichlet(concentration=table, parent_ndims=1, alpha0=jnp.ones(2))

    d = Dirichlet(concentration=jnp.asarray([2.0, 3.0]))
    with pytest.raises(ValueError, match="needs an anchor"):
        d.absorb(jnp.asarray([1.0, 0.0]), decay=0.9)
    d.absorb(jnp.asarray([1.0, 0.0]))  # the conjugate path reads no anchor
    out = Dirichlet(concentration=d.concentration, alpha0=jnp.asarray(1.0)).absorb(
        jnp.asarray([1.0, 0.0]), decay=0.5
    )
    np.testing.assert_allclose(
        np.asarray(out.concentration), [0.5 * 2.0 + 0.5 + 1.0, 0.5 * 3.0 + 0.5]
    )


# ------------------------------------------- gap 1: tick 0 inside lax.scan


def test_tick_zero_traced_rate_in_scan():
    """A learnable transition inside ``lax.scan``: tick 0 has no real previous
    belief, so the loop feeds a dummy and a traced ``jnp.where(t == 0, 0.0, 1.0)``
    rate (implementation gap 1) — counts must not move at t=0 and must move at
    t=1."""
    rng = np.random.default_rng(0)
    ns, nu = 3, 2
    system = System(
        A_deps=((0,),),
        B_deps=((0,),),
        controlled=(True,),
        policy_len=1,
        modality_names=("ident",),
        factor_names=("s",),
    )
    params = SystemParams(
        likelihood={"ident": _near_identity(ns)},
        transition={
            "s": ConditionalDirichlet(concentration=jnp.full((ns, ns, nu), 1.0), parent_ndims=2),
        },
        initial_state_prior={"s": Categorical(logits=jnp.zeros(ns))},
    )
    obs_seq = jnp.asarray(rng.integers(ns, size=4), jnp.int32)
    act_seq = jnp.asarray(rng.integers(nu, size=(4, 1)), jnp.int32)
    dummy = system.initial_priors(params)  # a face output: carries the control node's entry too

    def tick(carry, xs):
        t, obs_t, act_t = xs
        prev_beliefs, conc = carry
        p = dataclasses.replace(
            params, transition={"s": ConditionalDirichlet(concentration=conc, parent_ndims=2)}
        )
        beliefs, _ = system.infer_states(p, {"ident": obs_t})
        p = system.learn(
            p,
            {"ident": obs_t},
            beliefs,
            prev_beliefs=prev_beliefs,
            action=system.from_positional("actions", act_t),
            lr_by_slot={"transition:s": jnp.where(t == 0, 0.0, 1.0)},
        )
        return (beliefs, p.transition["s"].concentration), p.transition["s"].concentration

    init = (dummy, params.transition["s"].concentration)
    _, concs = jax.lax.scan(tick, init, (jnp.arange(4), obs_seq, act_seq))
    assert np.array_equal(np.asarray(concs[0]), np.asarray(params.transition["s"].concentration))
    assert not np.array_equal(np.asarray(concs[1]), np.asarray(concs[0]))


# ------------------------------- gap 7: the E[log A] divergence is a Jensen gap


def test_elog_divergence_monotone_decreasing_in_concentration():
    """FPI-under-E[log A] vs FPI-under-the-mean-table posterior divergence must
    shrink monotonically as the concentration total grows (E[log A] → log E[A]);
    the only cheap test that distinguishes the deliberate Jensen gap from a
    sign error or a wrong digamma."""
    rng = np.random.default_rng(9)
    ns, no = 3, 4
    base = np.float32(rng.gamma(2.0, size=(no, ns)) + 0.2)
    prior = (Categorical(logits=jnp.zeros(ns)),)
    obs = (jnp.asarray(1, jnp.int32),)
    divs = []
    for c in (1.0, 4.0, 16.0, 64.0):
        cd = ConditionalDirichlet(concentration=jnp.asarray(c * base), parent_ndims=1)
        q_avg = fpi((cd,), ((0,),), prior, obs)[0]
        q_mean = fpi((cd.point(),), ((0,),), prior, obs)[0]
        divs.append(float(q_avg.kl_to(q_mean).lower))
    assert divs[0] > divs[1] > divs[2] > divs[3] > 0.0


# ----------------------------------------------- rate/key error semantics


def test_rate_key_naming_point_slot_raises():
    system, params = _sense_system()
    obs, beliefs = _tick_inputs(system, params)
    with pytest.raises(ValueError, match=r"holds a ConditionalCategorical \(a point value\)"):
        system.learn(params, obs, beliefs, lr_by_slot={"likelihood:ident": 1.0})


def test_rate_key_unqualified_or_unknown_raises():
    system, params = _sense_system()
    obs, beliefs = _tick_inputs(system, params)
    with pytest.raises(ValueError, match="slot-qualified"):
        system.learn(params, obs, beliefs, lr_by_slot={"sense": 1.0})
    with pytest.raises(ValueError, match="likelihood:nope"):
        system.learn(params, obs, beliefs, decay_by_slot={"likelihood:nope": 0.9})


def test_active_learnable_transition_requires_prev_beliefs_and_action():
    rng = np.random.default_rng(1)
    ns, nu = 3, 2
    system = System(
        A_deps=((0,),),
        B_deps=((0,),),
        controlled=(True,),
        policy_len=1,
        modality_names=("ident",),
        factor_names=("s",),
    )
    params = SystemParams(
        likelihood={"ident": _near_identity(ns)},
        transition={
            "s": ConditionalDirichlet(concentration=jnp.full((ns, ns, nu), 1.0), parent_ndims=2),
        },
        initial_state_prior={"s": Categorical(logits=jnp.zeros(ns))},
    )
    obs = {"ident": jnp.asarray(rng.integers(ns), jnp.int32)}
    beliefs, _ = system.infer_states(params, obs)
    with pytest.raises(ValueError, match="prev_beliefs"):
        system.learn(params, obs, beliefs)
    with pytest.raises(ValueError, match="action"):
        system.learn(params, obs, beliefs, prev_beliefs=beliefs)
    # a static 0.0 rate deactivates the requirement (the slot is skipped at trace time)
    out = system.learn(params, obs, beliefs, lr_by_slot={"transition:s": 0.0})
    assert np.array_equal(
        np.asarray(out.transition["s"].concentration),
        np.asarray(params.transition["s"].concentration),
    )


# --------------------------------------------- the initial-state-prior slot


def _d_slot_system(ns=3):
    system = System(
        A_deps=((0,),),
        B_deps=((0,),),
        controlled=(False,),
        policy_len=1,
        modality_names=("ident",),
        factor_names=("s",),
    )
    params = SystemParams(
        likelihood={"ident": _near_identity(ns)},
        transition={"s": ConditionalCategorical(log_table=jnp.zeros((ns, ns, 1)), parent_ndims=2)},
        initial_state_prior={"s": Dirichlet(concentration=jnp.asarray([4.0, 2.0, 2.0]))},
    )
    return system, params


def test_learnable_initial_prior_slot():
    """A ``Dirichlet`` in ``initial_state_prior``: FPI takes ``log(mean())``
    (never the unnormalized expected_log_params, D2), ``as_env`` lowers it,
    and ``learn`` absorbs ``q(s_0)`` — the caller gates the rate to tick 0
    (D5's per-episode timing)."""
    system, params = _d_slot_system()
    d = params.initial_state_prior["s"]
    priors = system.initial_priors(params)
    np.testing.assert_allclose(np.asarray(priors["s"].probs), np.asarray(d.mean()), atol=1e-6)

    obs = {"ident": jnp.asarray(1, jnp.int32)}
    beliefs, _ = system.infer_states(params, obs)
    out = system.learn(params, obs, beliefs)
    np.testing.assert_allclose(
        np.asarray(out.initial_state_prior["s"].concentration),
        np.asarray(d.concentration + beliefs["s"].probs),
        atol=1e-6,
    )

    _, ep = system.as_env(params)
    np.testing.assert_allclose(
        np.asarray(ep.initial_state_prior["s"]), np.asarray(d.mean()), atol=1e-6
    )
    _, ep_drawn = system.as_env(params, param_key=jax.random.PRNGKey(3))
    np.testing.assert_allclose(np.asarray(ep_drawn.initial_state_prior["s"]).sum(), 1.0, atol=1e-5)


def test_hierarchical_learnable_prior_validates_but_update_is_step_4():
    """A ``ConditionalDirichlet`` D with a ``D_deps`` entry is a legal
    declaration (the widened ``_validate``); its *update* is D5's owed
    parent-weighted statistic (build-plan step 4), so an active rate raises
    loudly instead of absorbing the wrong thing."""
    ns0, ns1 = 2, 3
    system = System(
        A_deps=((0,), (1,)),
        B_deps=((0,), (1,)),
        controlled=(False, False),
        policy_len=1,
        D_deps=((), (0,)),
        modality_names=("m0", "m1"),
        factor_names=("ctx", "s"),
    )
    params = SystemParams(
        likelihood={"m0": _near_identity(ns0), "m1": _near_identity(ns1)},
        transition={
            name: ConditionalCategorical(log_table=jnp.zeros((n, n, 1)), parent_ndims=2)
            for name, n in (("ctx", ns0), ("s", ns1))
        },
        initial_state_prior={
            "ctx": Categorical(logits=jnp.zeros(ns0)),
            "s": ConditionalDirichlet(concentration=jnp.full((ns1, ns0), 2.0), parent_ndims=1),
        },
    )
    priors = system.initial_priors(params)  # log(mean) averaged over the parent
    np.testing.assert_allclose(float(np.asarray(priors["s"].probs).sum()), 1.0, atol=1e-5)

    obs = {"m0": jnp.asarray(0, jnp.int32), "m1": jnp.asarray(1, jnp.int32)}
    beliefs, _ = system.infer_states(params, obs)
    with pytest.raises(NotImplementedError, match="step 4"):
        system.learn(params, obs, beliefs)
    frozen = system.learn(params, obs, beliefs, lr_by_slot={"initial_state_prior:s": 0.0})
    assert np.array_equal(
        np.asarray(frozen.initial_state_prior["s"].concentration),
        np.asarray(params.initial_state_prior["s"].concentration),
    )


# ------------------------------------------------- env-side loud failures


def test_env_rejects_raw_parameter_distributions():
    """A Dirichlet reaching an env's sampling path is silent garbage (a simplex
    vector where a state index is expected), so hand-built env params fail
    loudly and point at ``as_env(params, param_key=...)``."""
    env = HybridProcess(A_deps=((0,),), B_deps=((0,),), controlled=(False,))
    ep = GenerativeEnvParams(
        likelihood=(_near_identity(3),),
        transition=(ConditionalCategorical(log_table=jnp.zeros((3, 3, 1)), parent_ndims=2),),
        initial_state_prior=(Dirichlet(concentration=jnp.ones(3)),),
    )
    with pytest.raises(ValueError, match="distribution over parameters"):
        env.reset(KEY, ep)


# ------------------- Form A coherence: the decompositions agree exactly


def test_decompositions_agree_exactly_under_learnable_likelihood():
    """The test that would have caught the spec's original D2 ambiguity row:
    with a learnable likelihood in the slot, the two decompositions' EFE
    totals agree to float tolerance for normalized targets, novelty leaf on
    *and* off. Both entropy-bearing terms score the mean table and
    ``I(o; θ | s)`` appears as exactly one explicit leaf (the novelty term),
    identical in both trees — Form A/D coherence. Expected-entropy ambiguity
    beside a novelty leaf (the spec's original prescription) would remove
    that mutual information twice and break this equality."""
    from hybrid_aif.objectives import total

    for epistemic in (True, ("s",)):  # states + novelty, then states only
        system, params = _sense_system(epistemic=epistemic)
        # normalized C designations on EVERY modality (the identity's precondition)
        params = dataclasses.replace(
            params,
            obs_targets={
                "sense": Categorical(logits=jnp.asarray([0.5, -0.2, 0.1, 0.0])),
                "ident": Categorical(logits=jnp.zeros(3)),
            },
        )
        _, beliefs = _tick_inputs(system, params)
        totals = {}
        for decomposition in ("utility_infogain", "risk_ambiguity"):
            sys_d = dataclasses.replace(system, decomposition=decomposition)
            _, tree = sys_d.infer_policies(params, beliefs)
            novelty = "likelihood:sense" in tree.get("infogain", {})
            assert novelty == (epistemic is True)
            totals[decomposition] = np.asarray(total(tree).lower)
        np.testing.assert_allclose(
            totals["utility_infogain"],
            totals["risk_ambiguity"],
            atol=1e-5,
            err_msg=f"epistemic={epistemic}",
        )


def test_ambiguity_under_learnable_likelihood_is_mean_table_entropy():
    """Pins Form A: the ambiguity leaf for a ConditionalDirichlet modality is
    the mean-table conditional entropy contraction, not the expected entropy
    (whose gap to it is exactly I(o; θ | s), the novelty term's referent)."""
    from jax.scipy.special import xlogy

    from hybrid_aif.objectives import discrete

    rng = np.random.default_rng(2)
    cd = ConditionalDirichlet(
        concentration=jnp.asarray(np.float32(rng.gamma(2.0, size=(4, 3)) + 0.1)),
        parent_ndims=1,
    )
    q = Categorical(logits=jnp.asarray(np.float32(rng.normal(size=3))))
    leaf = discrete.ambiguity((q,), (cd,), ((0,),))["ambiguity"]["modality_0"]
    mean = np.asarray(cd.mean())
    h_mean = -np.asarray(xlogy(mean, mean)).sum(axis=0)
    manual = float(np.asarray(q.probs) @ h_mean)
    assert np.isclose(float(leaf.lower), manual, atol=1e-6)
    # and the expected entropy is strictly below it (their gap is I(o; θ | s) ≥ 0)
    exp_ent = float(np.asarray(q.probs) @ np.asarray(cd.expected_entropy()))
    assert exp_ent < manual
