"""The G side of the R-AIF replication: the five rows, and acting from a habit (2b).

Design: ``docs/design/amortization.md`` A12 (the five rows, the habit planner, the
scope split), A4 (the ``derived`` warrant axis), A3 (E in every planner),
``objective-terms.md`` (the term lattice, and why ``occupancy`` is a new leaf
rather than a sign-flipped ``uncertainty``), and
``amortization-p0-addendum.md`` §§2–3 for every number.

**Nothing here is trained.** Random actor/critic/ensemble/CRSPP are exactly what
this chunk's acceptance asks for: what is under test is shape, math, sign,
honesty flag and grammar. The training loops are chunk 2c's.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import xlogy

from hybrid_aif.controls import soft_clip
from hybrid_aif.distributions import Categorical, ConditionalCategorical, MLPConditional
from hybrid_aif.distributions.neural import PredictorHead, symlog_disc_mean
from hybrid_aif.objectives.amortized import features
from hybrid_aif.objectives.base import INFOGAIN, total
from hybrid_aif.objectives.discrete import _mean_table_conditional_entropy
from hybrid_aif.planning import get_planner, make_cem_planner
from hybrid_aif.reference.raif.world import CODE_NAME, crspp_rssm, tiny_rssm

KEY = jax.random.PRNGKey(0)
DETER, NUM_VARS, NUM_CLASSES = 8, 3, 4
FLAT = NUM_VARS * NUM_CLASSES
FEATURE = DETER + FLAT
HORIZON = 2
ROWS = (
    "occupancy:transition:stoch",
    "predicted_reward:obs:reward",
    "cosine_sim_to_pref_rollout",
    "g_tail",
)


def agent(terms=ROWS, epistemic=("stoch",), **system_kwargs):
    """A ``tiny_rssm`` with A12's G side declared on it, at test sizes.

    The world model comes from :func:`hybrid_aif.reference.raif.world.tiny_rssm` unchanged — the
    G side is the *agent's* declaration, which is why ``worlds.py`` does not carry
    it — and every scoring module is random.
    """
    keys = jax.random.split(KEY, 6)
    system, params = tiny_rssm(
        2,
        1,
        key=keys[0],
        deter=DETER,
        num_vars=NUM_VARS,
        num_classes=NUM_CLASSES,
        units=4,
        mlp_units=6,
        embed=5,
        policy_len=HORIZON,
    )
    crspp, crspp_params = crspp_rssm(
        key=keys[1], deter=DETER, num_vars=NUM_VARS, num_classes=NUM_CLASSES, units=4
    )
    critic = PredictorHead.from_shapes(
        (FEATURE,), loss="symlog_disc", num_bins=9, key=keys[2], width=4, depth=1
    )
    slow = PredictorHead.from_shapes(
        (FEATURE,), loss="symlog_disc", num_bins=9, key=keys[3], width=4, depth=1
    )
    ensemble = tuple(
        MLPConditional.from_shapes((DETER, FLAT, 1), FLAT, key=k, width=4, depth=1)
        for k in jax.random.split(keys[4], 4)
    )
    actor = MLPConditional.from_shapes((FEATURE,), 1, key=keys[5], width=4, depth=1)
    system = dataclasses.replace(
        system,
        planner="habit",
        epistemic=epistemic,
        terms=terms,
        scoring=(("cosine_sim_to_pref_rollout", crspp),) if "cosine" in " ".join(terms) else (),
        **system_kwargs,
    )
    params = dataclasses.replace(
        params,
        policy_prior=actor,
        scoring={
            "g_tail": (critic, slow),
            "ensemble_disagreement": ensemble,
            "cosine_sim_to_pref_rollout": crspp_params,
        },
    )
    return system, params, (crspp, crspp_params, critic, slow, ensemble, actor)


def _habit_plan(system, params, beliefs=None):
    """The plan the habit chose, ``(1, H, action_dim)`` — the planner's own policy set."""
    beliefs = system.initial_priors(params) if beliefs is None else beliefs
    _, _, policies = get_planner("habit")(system.planner_context(params, beliefs, key=KEY))
    return policies


def _plan_and_tree(system, params):
    """The habit's plan, and the term tree **G** assigns it — through the teacher face.

    A habit planner selects without building a term tree (its selection *is* the
    actor's distribution), so ``infer_policies`` returns ``{}`` there and the tree
    is asked for separately: :meth:`System.score_policies` on the plan, which is
    A3's teacher face and takes float plans over a continuous action space (A9).
    The per-policy axis is kept, so every leaf is ``(1,)``.
    """
    beliefs = system.initial_priors(params)
    policies = _habit_plan(system, params, beliefs)
    return beliefs, policies, system.score_policies(params, beliefs, policies)


# ------------------------------------------------------------------- acceptance


def test_the_whole_tree_scores_and_every_amortized_leaf_is_unwarranted():
    """This chunk's acceptance, in one test: a ``tiny_rssm``-declared System with all
    five rows mounted scores a term tree whose leaves are all ``derived=False``, and
    the propagation poisons the total. Scored on the plan the habit chose, through
    ``score_policies`` (see :func:`_plan_and_tree`).

    A10 disposition 11, ratified: a fully-unwarranted tree in amortized mode is the
    flag working as designed. The decomposition is still the diagnostic value —
    every leaf is here, named, and separately readable."""
    system, params, _ = agent()
    _, _, tree = _plan_and_tree(system, params)
    assert set(tree) == {
        "occupancy",
        INFOGAIN,
        "predicted_reward",
        "cosine_sim_to_pref_rollout",
        "g_tail",
    }
    assert {tuple(blocks) for blocks in tree.values()} == {
        ("transition:stoch",),
        ("stoch",),
        ("reward",),
        ("crspp",),
        ("critic",),
    }
    for term, blocks in tree.items():
        for block, leaf in blocks.items():
            assert leaf.derived is False, (term, block)
            assert np.all(np.isfinite(np.asarray(leaf.midpoint)))
    G = total(tree)
    assert G.derived is False  # the AND: one unwarranted leaf is enough
    assert np.isfinite(float(G.midpoint[0]))


def test_act_returns_the_raw_actor_sample_and_leaves_the_box_to_its_consumers():
    """The other half of the acceptance, **re-aimed 2026-09-07** (acting-ledger D5).

    ``act`` samples the declared actor on the current beliefs and returns that draw
    **raw**. This test used to assert the opposite — that the action comes back
    inside the declared box — and that assertion was the divergence: their
    ``ImagActorCritic.policy`` is one actor evaluation with no clip at all
    (``agent.py:401-402``), and the box is applied by the action's *consumers*, in
    two different ways. ``ClipAction`` clips on the way into the env and builds a
    fresh dict, so the driver's recorded ``acts`` keep the raw sample
    (``wrappers.py:78-80``); ``obs_step``/``img_step`` rescale on the way into the
    transition, stop-gradiented (``nets.py:131-141``,
    :func:`hybrid_aif.controls.soft_clip`).

    It matters because two readers of the recorded column are not clip-invariant:
    the actor-refresh BC term evaluates a Gaussian log-density *at* ``start['action']``
    (``agent.py:477-479``), where clipping piles a third to a half of the batch's
    targets onto exactly ±1, and the ensemble's ``ig_head`` conditions on ``action``.
    Measured on their untrained actor, 25.7% of recorded actions lie outside the box,
    max 2.271 — and it does not vanish as the policy sharpens, because their ``tanh``
    mean approaches ±1 while σ has a 0.1 floor.

    A Gaussian head over a bounded action space **does** put mass outside the box, so
    the draw is asserted to be finite and of the declared shape rather than bounded,
    and a draw actually outside the box is exhibited so the property is visible
    rather than merely permitted.
    """
    system, params, _ = agent()
    beliefs = system.initial_priors(params)
    action, q_pi, tree = system.act(KEY, params, beliefs, plan_key=jax.random.PRNGKey(3))
    action = action["a_deter"]
    assert action.shape == (1,) and action.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(action)))
    assert q_pi.probs.shape == (1,)  # a habit named one plan; nothing left to weigh
    assert tree == {}  # this selection produced no term tree; see the next test

    #  the raw draw leaves the box, and `soft_clip` — not `act` — is what brings it in
    draws = jnp.stack(
        [
            system.act(k, params, beliefs, plan_key=jax.random.PRNGKey(3))[0]["a_deter"]
            for k in jax.random.split(jax.random.PRNGKey(21), 64)
        ]
    )
    outside = np.asarray(jnp.abs(draws) > 1.0)
    assert outside.any(), "a Gaussian actor over Box(-1, 1) must sometimes leave it"
    brought_in = soft_clip(draws, (jnp.full((1,), -1.0), jnp.full((1,), 1.0)))
    assert bool(jnp.all(jnp.abs(brought_in) <= 1.0 + 1e-6))
    #  and it is a RESCALE, not a clip: sign(a)·1 in value, `1/|a|` in gradient
    a = jnp.asarray([2.5])
    bounds = (jnp.full((1,), -1.0), jnp.full((1,), 1.0))
    assert float(soft_clip(a, bounds)[0]) == pytest.approx(1.0)
    grad = jax.grad(lambda x: soft_clip(x, bounds).sum())(a)
    assert float(grad[0]) == pytest.approx(1.0 / 2.5)
    assert float(jax.grad(lambda x: jnp.clip(x, -1.0, 1.0).sum())(a)[0]) == 0.0


def test_every_new_term_is_registered_with_the_value_sign():
    """A10 disposition 7: their ``G_scales`` reward 1.0 / sim 1.0 / IG −1.0 are
    *values* on a maximized objective, so ours are −1 terms on a minimized one, and
    the conversion happens at the registry boundary. ``total`` raising on an
    unregistered name is what makes this an assertion rather than a hope."""
    system, params, _ = agent()
    _, _, tree = _plan_and_tree(system, params)
    for term, blocks in tree.items():
        one = total({term: blocks})
        summed = sum(float(np.asarray(iv.midpoint)[0]) for iv in blocks.values())
        assert float(np.asarray(one.midpoint)[0]) == pytest.approx(-summed, rel=1e-5)


# ---------------------------------------------------------------------- occupancy


def test_occupancy_is_the_transitions_expected_conditional_entropy():
    """The row reads the code belief its own transition just produced, which on this
    path *is* ``p(stoch_t | deter_t)`` — the parent is a deterministic carry, so the
    expectation over ``q(s)`` is exact and the entropy is closed-form in the
    probability tensor."""
    system, params, _ = agent(terms=("occupancy:transition:stoch",), epistemic=False)
    beliefs, policies, tree = _plan_and_tree(system, params)
    rolled = system.predict_beliefs(params, beliefs, policies[0])
    per_step = [
        float(-xlogy(qs["stoch"], qs["stoch"]).sum()) for qs in rolled
    ]  # H of each step's code belief
    leaf = tree["occupancy"]["transition:stoch"]
    assert float(np.asarray(leaf.midpoint)[0]) == pytest.approx(sum(per_step), rel=1e-5)
    assert all(h >= 0.0 for h in per_step)


def test_occupancy_rewards_uncertainty_where_the_uncertainty_row_penalizes_it():
    """``ambiguity`` and ``occupancy`` are the same functional on different
    conditionals with opposite signs (``objective-terms.md``'s pair). Concretely
    here: the leaf stores ``+H`` under a −1 term, so G *rewards* an unpredictable
    successor — the exact opposite of what the continuous ``uncertainty`` row
    (which stores ``+H`` under a +1 term) contributes."""
    system, params, _ = agent(terms=("occupancy:transition:stoch",), epistemic=False)
    _, _, tree = _plan_and_tree(system, params)
    leaf = tree["occupancy"]["transition:stoch"]
    assert float(np.asarray(leaf.midpoint)[0]) > 0.0  # the entropy itself, not its negation
    assert float(np.asarray(total(tree).midpoint)[0]) < 0.0  # so G goes down as it goes up


def test_the_occupancy_identity_that_forbids_the_sign_flip():
    """``H[q(s')] = E_q H[p(s'|s,a)] + I(s';s)`` — the identity ``objective-terms.md``
    says the plan of record would have violated: flipping the sign on the
    ``uncertainty`` row (the *marginal* entropy ``H[q(x)]``) would fold ``I(s';s)``
    into the occupancy term, where the ensemble info-gain leaf counts it a second
    time.

    Checked on a tabular kernel, where all three quantities are separately
    computable and the mutual information is *not* defined as the difference:
    ``I = Σ_s q(s) KL(p(·|s) ‖ q(s'))``. It is the same functional the shipped
    ``ambiguity`` estimator computes, which is why the two rows are lattice
    neighbours. On the amortized path the parent is a Dirac, so ``I`` vanishes and
    the two coincide *numerically* — the identity is what says that is a property of
    the parent, not of the term."""
    table = jnp.asarray([[0.7, 0.1, 0.2], [0.2, 0.6, 0.3], [0.1, 0.3, 0.5]])  # p(s'|s)
    q = jnp.asarray([0.5, 0.3, 0.2])
    kernel = ConditionalCategorical(log_table=jnp.log(table), parent_ndims=1)
    expected_conditional_h = float(
        _mean_table_conditional_entropy(kernel, (Categorical(logits=jnp.log(q)),))
    )
    marginal = table @ q
    marginal_h = float(-xlogy(marginal, marginal).sum())
    info = float(sum(q[s] * jnp.sum(xlogy(table[:, s], table[:, s] / marginal)) for s in range(3)))
    assert marginal_h == pytest.approx(expected_conditional_h + info, rel=1e-5)
    assert info > 0.0  # the term the flip would have double-counted


# ----------------------------------------------------------- ensemble info gain


def _their_information_gain(means, scales, samples_per_member: int = 1, seed: int = 0) -> float:
    """``epistemics.py::InformationGain.__call__``, transcribed — the oracle, not a mirror.

    Their lines 34-35 and 41-43 (``raif/aif/epistemics.py``)::

        def _entropy(self, stddev):
          return 1.0/2 * jnp.log((2*jnp.pi * stddev**2).clip(1e-8)) + 1.0/2
        ...
        ent_avg = self._entropy(mixsigma).mean(-1)
        avg_ent = jnp.asarray([self._entropy(d.stddev()).mean(-1) for d in dists]).mean(0)
        params_info_gain = (ent_avg - avg_ent) * (1.0 / jnp.sqrt(config.rssm.stoch))

    Written out in *their* algebra rather than reusing
    :func:`~hybrid_aif.objectives.amortized._diag_entropy`, because this test
    previously reimplemented the row's own expression and so could only ever confirm
    that the row agreed with itself. It did — while both were **1024× off** at their
    ``tiny`` sizes, the row summing the per-dimension entropies where they average
    them. Found by P1′ chunk 2d's smoke (the term outweighed reward ~400× and the
    policy collapsed to standing still), not by this test. A10 disposition 7 makes
    the code the oracle; this now reads it.

    **And it did not read it closely enough** (2026-09-05). This transcription
    silently put ``means.std(0)`` where their line says ``preds.std(0)`` over
    ``preds = [dist.sample(...) for dist in dists]`` — sampled predictions, which
    carry each member's own σ. So the reference mirrored the row's defect a second
    time and pinned it for a 2M parity run. ``samples_per_member`` now makes the
    sampling explicit: ``1`` is their code exactly, and large values drive the
    sampling noise out to expose the estimand underneath.
    """
    rng = np.random.default_rng(seed)
    mu = np.asarray(means, np.float64)
    sd = np.asarray(scales, np.float64)

    def entropy(x):
        var = 2.0 * np.pi * np.asarray(x, np.float64) ** 2
        return 0.5 * np.log(np.clip(var, 1e-8, None)) + 0.5

    #  their `preds = jnp.asarray([dist.sample(seed=nj.rng()) for dist in dists])`,
    #  then `mixsigma = preds.std(0)` — the spread ACROSS members of one draw each.
    preds = mu[None] + sd[None] * rng.standard_normal((samples_per_member, *mu.shape))
    mixsigma = preds.reshape(-1, mu.shape[-1]).std(axis=0)
    ent_avg = entropy(mixsigma).mean(-1)
    avg_ent = np.mean([entropy(s).mean(-1) for s in sd], axis=0)
    return float((ent_avg - avg_ent) / np.sqrt(NUM_VARS))


def _moment_matched_information_gain(means, scales) -> float:
    """The same functional with the sampling noise removed — what the row computes.

    Their ``mixsigma`` is a plug-in estimate of the mixture's per-dimension spread,
    which the law of total variance gives in closed form as
    ``mean_i(σ_i²) + var_i(μ_i)``. Both are moment-matched Gaussian approximations
    to a mixture whose true entropy has none; this one is exact and keyless.
    """
    mu = np.asarray(means, np.float64)
    sd = np.asarray(scales, np.float64)

    def entropy(x):
        return 0.5 * np.log(np.clip(2.0 * np.pi * x**2, 1e-8, None)) + 0.5

    mixture = np.sqrt((sd**2).mean(axis=0) + mu.var(axis=0))
    return float(
        (entropy(mixture).mean(-1) - np.mean([entropy(s).mean(-1) for s in sd])) / np.sqrt(NUM_VARS)
    )


def _ensemble_moments(system, params, ensemble, beliefs, plan):
    """The four heads' ``(μ, σ)`` at ``(s_t, a_t)`` for every action in ``plan``.

    ``plan`` is one action per state to be scored, so it also decides *how many*
    states are scored and which: ``H`` actions give the ``H`` steps our rows fold
    into ``G``, and their ``H + 1`` give the whole array their
    ``epistemic_module(traj)`` returns before ``[1:]``. That is the parameter the
    old version of this helper did not have — it hard-coded ``range(HORIZON)`` at
    the row's own indexing, which is why it could certify the leaf's *formula* and
    was structurally blind to its *position*. See
    :func:`_their_igfn_from_the_call_site`.
    """
    steps = int(plan.shape[0])
    rolled = [beliefs, *system.predict_beliefs(params, beliefs, plan[: steps - 1])]
    for t in range(steps):
        parents = (rolled[t]["deter"], rolled[t]["stoch"].reshape(-1), plan[t])
        heads = [m.diag_moments(parents) for m in ensemble]
        yield jnp.stack([mu for mu, _ in heads]), jnp.stack([sd for _, sd in heads])


def _their_igfn_from_the_call_site(system, params, ensemble, actor, beliefs, plan):
    """``igfn`` at their **call site**, and ours beside it: ``(leaf, theirs, ours)``.

    Built from ``agent.py:356-358`` rather than from ``epistemics.py``'s expression,
    which is the whole point::

        rewfn   = lambda s: self.heads['reward'](s).mean()[1:]
        crsppfn = lambda s, p: self.crspp_traj_sim(s, p)[1:]
        igfn    = lambda s: self.epistemic_module(s)[1:]

    ``s`` is ``traj``, and ``traj`` is ``H + 1`` states with ``H + 1`` actions —
    ``imagine`` seeds ``start['action'] = policy(start)`` and every scan step
    returns ``{**img_step(prev, prev.pop('action')), 'action': policy(state)}``
    (``agent.py:244-251``), so their actor runs ``H + 1`` times and
    ``traj['action']`` comes out ``f32[9, 1024, 1]`` at their sizes. ``InformationGain``
    reads ``[deter, stoch, action]`` (``ig_head.inputs``; note the
    ``config.update({'ig_head.inputs': ['tensor']})`` on the line above does *not*
    reach it, because ``embodied.Config.update`` returns a new object), so their leaf
    array is ``L[k] = IG(s_k, a_k)`` for ``k = 0 … H``, and ``igfn``'s ``[1:]`` hands
    ``G[k]`` the entry ``L[k+1]``.

    So the two indexings are two slices of **one** array — theirs ``L[1:]``, ours
    ``L[:-1]`` — and returning all three is what puts the position inside a test's
    field of view. The last entry, ``L[H]``, is the one ours cannot express at all:
    it reads ``a_H``, an action drawn in a scan iteration *after* the step whose
    score would use it.
    """
    extended = _their_traj_plan(system, params, actor, beliefs, plan)
    moments = list(_ensemble_moments(system, params, ensemble, beliefs, extended))
    leaf = np.asarray([_moment_matched_information_gain(m, s) for m, s in moments])
    assert leaf.shape == (int(plan.shape[0]) + 1,)
    return leaf, leaf[1:], leaf[:-1]


def _their_traj_plan(system, params, actor, beliefs, plan):
    """``traj['action']``: the plan's ``H`` actions plus their ninth, ``policy(s_H)``.

    Their ``imagine``'s scan returns ``{**state, 'action': policy(state)}``, so the
    action at the *last* imagined state is sampled and kept even though ``logpi`` and
    ``ent`` both drop it with ``[:-1]``. Divergence 24: the only consumer of that
    ninth action is ``igfn``'s ``[1:]``, so it is inert with divergence 20 resolved
    our way and a prerequisite with it resolved theirs.
    """
    rolled = [beliefs, *system.predict_beliefs(params, beliefs, plan)]
    final = system.to_positional("beliefs", rolled[int(plan.shape[0])])
    last = actor.sample(jax.random.PRNGKey(99), (features(final),))
    return jnp.concatenate([jnp.asarray(plan), jnp.asarray(last)[None]], axis=0)


def test_ensemble_disagreement_matches_its_formula_and_reads_the_steps_inputs():
    """IDD §D: ``ent(mixture) − E_i[ent(σ_i)]``, scaled ``1/sqrt(num_vars)``
    (``epistemics.py:33-45``), where ``ent`` is the **per-dimension mean** entropy and
    the mixture's spread is ``mean_i(σ_i²) + var_i(μ_i)`` by the law of total
    variance — see :func:`_moment_matched_information_gain`. The inputs are
    ``[deter, flat stoch, action]`` at the step's *start* — the transition whose
    uncertainty the action resolves — which is why the row rides the
    ``(qs, qs_prev)`` planning carry."""
    system, params, parts = agent(terms=(), epistemic=("stoch",))
    beliefs, policies, tree = _plan_and_tree(system, params)
    wanted = sum(
        _moment_matched_information_gain(means, scales)
        for means, scales in _ensemble_moments(system, params, parts[4], beliefs, policies[0])
    )
    leaf = tree[INFOGAIN]["stoch"]
    assert float(np.asarray(leaf.midpoint)[0]) == pytest.approx(wanted, rel=1e-4)


def test_the_info_gain_leaf_is_read_one_step_earlier_than_their_call_site_reads_it():
    """The **position** of the info-gain leaf, pinned — an ARGUED divergence, measured.

    Training-ledger 20, adjudicated 2026-09-07 (Alex): **keep ours**. The reason is
    structural and is written out at the row itself
    (:func:`hybrid_aif.objectives.amortized._ensemble_disagreement`): their indexing
    needs ``a_{k+1}`` at index ``k``, an action sampled in the following scan
    iteration, so expressing it would mean either giving every row a look-ahead
    action — making a step's score depend on a decision not yet taken — or shifting
    one leaf array inside the rollout as a special case the term-tree grammar cannot
    express. Our contract is that at index ``k`` every row is a functional of the
    same event: taking ``a_k`` from ``s_k`` and arriving at ``s_{k+1}``.

    This estimator has now had **three** defects — a 1024× entropy sum, ``means.std(0)``
    standing in for ``preds.std(0)``, and this index — so it gets the most thorough
    port in the suite, and the port is built **from their call site**
    (:func:`_their_igfn_from_the_call_site`) rather than from their expression. That
    is the method finding of the two 2026-09-06 audits, and this test is its first
    application: the previous harness computed the leaf at *the row's own indexing*,
    so it could only ever certify that the formula agreed with itself, exactly as the
    pre-2026-09-05 harness had certified ``means.std(0)`` against itself. Five claims,
    in the order they would fail:

    1. their ``igfn`` and our row are two **slices of one array** — theirs ``L[1:]``,
       ours ``L[:-1]``;
    2. our row's actual folded leaf **is** ``sum(L[:-1])``, so its position is pinned
       against an array built at theirs;
    3. their slice is a different number, so the divergence is real rather than a
       relabelling that cancels;
    4. their last entry reads ``a_H``: perturb the ninth action and theirs moves
       while ours does not — the structural asymmetry, made a measurement;
    5. the **formula** still matches theirs at every index, which is what a
       transcription test is for in the first place.

    **Measured** at initialization over 64 independent rollouts at ``run_mc.SMALL``
    sizes with ``horizon=8`` (the numbers the ledger now carries): the leaf runs
    ≈0.031 and drops ≈6.3% from index 0 to index 1, then flat; the per-index gap
    ``|L[k+1] − L[k]|`` is 0.0039 mean / 0.016 max, ≈12.7% of the leaf's own
    magnitude; in ``G`` that is 0.0037 mean / 0.014 max per step, and in the actor's
    normalized advantage 0.0036 mean against a p5–p95 spread of 0.91 — **0.39%** of
    the advantage's own spread, roughly flat across the horizon rather than
    concentrated early. Sweeping the members' mean-spread ``s`` against their fixed
    ``σ ≡ 1.0``, the absolute gap peaks at ``s ≈ σ`` (0.0082 leaf, 1.64% of the
    advantage's spread) and falls away on both sides, to 0.2% of the leaf at
    ``s = 10σ``.
    """
    system, params, parts = agent(terms=(), epistemic=("stoch",))
    ensemble, actor = parts[4], parts[5]
    beliefs, policies, tree = _plan_and_tree(system, params)
    plan = policies[0]
    leaf, theirs, ours = _their_igfn_from_the_call_site(
        system, params, ensemble, actor, beliefs, plan
    )

    #  (1) two slices of one array
    np.testing.assert_allclose(theirs, leaf[1:], rtol=0, atol=0)
    np.testing.assert_allclose(ours, leaf[:-1], rtol=0, atol=0)
    assert leaf.shape == (HORIZON + 1,) and theirs.shape == ours.shape == (HORIZON,)

    #  (2) the row's own folded leaf is OURS, at the position this array says
    folded = float(np.asarray(tree[INFOGAIN]["stoch"].midpoint)[0])
    assert folded == pytest.approx(float(ours.sum()), rel=1e-4)

    #  (3) and it is NOT theirs: the shift changes the number
    assert float(theirs.sum()) != pytest.approx(float(ours.sum()), rel=1e-6)

    #  (4) their last entry reads `a_H`, which ours never sees. Perturbing the ninth
    #  action moves theirs and cannot move ours — the structural claim, measured
    class _ShiftedActor:
        def sample(self, key, parents):
            return actor.sample(key, parents) + 5.0

    shifted_leaf, shifted_theirs, shifted_ours = _their_igfn_from_the_call_site(
        system, params, ensemble, _ShiftedActor(), beliefs, plan
    )
    np.testing.assert_allclose(shifted_ours, ours, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(shifted_leaf[:-1], leaf[:-1], rtol=1e-6, atol=1e-8)
    assert shifted_theirs[-1] != pytest.approx(theirs[-1], rel=1e-6)

    #  (5) the formula, at every index, against their own expression in the limit
    #  where its one-draw-per-member plug-in has converged onto the estimand
    extended = _their_traj_plan(system, params, actor, beliefs, plan)
    moments = list(_ensemble_moments(system, params, ensemble, beliefs, extended))
    assert len(moments) == HORIZON + 1
    sampled = [_their_information_gain(m, s, 4096, seed=3) for m, s in moments]
    assert sampled == pytest.approx(list(leaf), abs=0.01)


def test_the_row_is_their_estimator_with_the_sampling_noise_removed():
    """The limit argument, proven rather than asserted (Alex, 2026-09-05).

    Their ``mixsigma`` is a **one-draw-per-member** plug-in for the mixture spread
    this row computes in closed form. Drive the draws up and their estimator has to
    converge onto the row; that is what makes the divergence a variance reduction
    rather than a different quantity — the claim whose earlier, sloppier version
    ("ours is theirs at σ → 0") blessed the defect that cost the first 2M run.

    At their actual ``samples_per_member=1`` the estimate is biased **low** — the
    ``ddof=0`` spread over four values has expectation ``s² + ¾σ²`` rather than
    ``s² + σ²``, and Jensen pushes ``E[log ŝ]`` down again — so the test also pins
    the direction of that gap rather than pretending the two agree exactly.
    """
    system, params, parts = agent(terms=(), epistemic=("stoch",))
    beliefs, policies, _ = _plan_and_tree(system, params)
    moments = list(_ensemble_moments(system, params, parts[4], beliefs, policies[0]))
    exact = [_moment_matched_information_gain(m, s) for m, s in moments]
    for n, tol in ((64, 0.05), (4096, 0.005)):
        sampled = [_their_information_gain(m, s, n, seed=7) for m, s in moments]
        assert sampled == pytest.approx(exact, abs=tol)
    theirs = [_their_information_gain(m, s, 1, seed=7) for m, s in moments]
    assert all(t < e for t, e in zip(theirs, exact, strict=True))


def test_the_disagreement_leaf_is_a_per_dimension_mean_not_a_joint_entropy():
    """The 2d regression guard: the leaf must not scale with the code's width.

    The bug this pins was invisible to every existing test because the row and its
    test shared one formula, and invisible to the type system because a joint entropy
    is a perfectly good number. What makes it *wrong* is that their ``G_scales`` are
    all ``1.0``, so ``model_uncertainty`` is meant to be commensurate with
    ``predicted_reward`` — and a leaf that grows linearly in ``num_vars·num_classes``
    is commensurate with nothing. Asserted as a bound rather than a value: a
    per-dimension mean of log-scales is O(1) for any width, a sum is O(width).
    """
    system, params, parts = agent(terms=(), epistemic=("stoch",))
    _, _, tree = _plan_and_tree(system, params)
    leaf = float(np.asarray(tree[INFOGAIN]["stoch"].midpoint)[0])
    flat = NUM_VARS * NUM_CLASSES
    assert abs(leaf) < 4.0 * HORIZON, (
        f"leaf {leaf} is too large to be a per-dimension mean over {flat} dimensions "
        "— it looks like a joint (summed) entropy again"
    )


def test_the_ensemble_row_is_selected_by_the_ordinary_epistemic_selector():
    """A10 disposition 3: a *sibling* row, not a variant of ``predictive`` — same
    functional, different conditional and estimator — so it is reached by naming the
    variable it scores, exactly like every other state estimator. Its leaf key is
    the variable (``objective-terms.md``'s table), and the carry is ineligible."""
    system, params, _ = agent(terms=(), epistemic=("stoch",))
    _, _, tree = _plan_and_tree(system, params)
    assert tuple(tree[INFOGAIN]) == (CODE_NAME,)
    quiet = dataclasses.replace(system, epistemic=("deter",))  # a carry: no uncertainty
    with pytest.raises(NotImplementedError, match="no objective row"):
        quiet.infer_policies(params, system.initial_priors(params), plan_key=KEY)


def test_a_wildcard_state_selector_finds_the_code_and_skips_the_carry():
    system, params, _ = agent(terms=(), epistemic=("*",))
    _, _, tree = _plan_and_tree(system, params)
    assert tuple(tree[INFOGAIN]) == (CODE_NAME,)


# -------------------------------------------------- the two off-lattice rows


def test_predicted_reward_decodes_the_declared_reward_head():
    """Their ``r_τ``: group-1 material read directly (the ``symlog_disc`` head the
    System already declares), decoded by ``symexp`` of the expected bin. No target
    distribution, hence no lattice coordinate and no ``utility``/``risk`` pair."""
    system, params, _ = agent(terms=("predicted_reward:obs:reward",), epistemic=False)
    beliefs, policies, tree = _plan_and_tree(system, params)
    rolled = system.predict_beliefs(params, beliefs, policies[0])
    head = params.likelihood["reward"]
    wanted = sum(
        float(symlog_disc_mean(head.trunk((qs["deter"], qs["stoch"].reshape(-1))), head.bins))
        for qs in rolled
    )
    assert float(np.asarray(tree["predicted_reward"]["reward"].midpoint)[0]) == pytest.approx(
        wanted, rel=1e-5
    )


def test_predicted_reward_on_a_non_scalar_channel_raises():
    system, params, _ = agent(terms=("predicted_reward:obs:state",), epistemic=False)
    with pytest.raises(ValueError, match="symlog_disc"):
        system.infer_policies(params, system.initial_priors(params), plan_key=KEY)


def test_cosine_sim_compares_the_rollout_to_the_crspps_own(caplog):
    """Their ``sim`` (``agent.py:165-179``): cosine similarity on ``[deter, flat
    stoch]`` against the CRSPP's imagined trajectory. The CRSPP is a declared System
    owned by this row, its rollout the shipped keyless ``predict_beliefs``, and —
    2b's documented scope — it starts from its own declared episode start, since a
    planner is handed beliefs and not the observation sequence."""
    system, params, parts = agent(terms=("cosine_sim_to_pref_rollout",), epistemic=False)
    crspp, crspp_params = parts[0], parts[1]
    beliefs, policies, tree = _plan_and_tree(system, params)
    ours = system.predict_beliefs(params, beliefs, policies[0])
    theirs = crspp.predict_beliefs(
        crspp_params, crspp.initial_priors(crspp_params), jnp.zeros((HORIZON, 0), jnp.int32)
    )
    wanted = 0.0
    for mine, pref in zip(ours, theirs, strict=True):
        a = features(system.to_positional("beliefs", mine))
        b = features(crspp.to_positional("beliefs", pref))
        wanted += float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
    leaf = float(np.asarray(tree["cosine_sim_to_pref_rollout"]["crspp"].midpoint)[0])
    assert leaf == pytest.approx(wanted, rel=1e-5)
    assert -HORIZON - 1e-5 <= leaf <= HORIZON + 1e-5  # a cosine per step


@pytest.mark.parametrize("keyed", [False, True])
def test_the_cosine_rows_preferred_rollout_is_the_stepwise_loop(keyed):
    """The row's per-step similarity equals the one computed against a Python-loop
    rollout of the CRSPP under the row's own key sequence, bitwise."""
    import zlib

    from hybrid_aif.objectives import registry
    from hybrid_aif.reference.raif.crspp import CRSPP_BLOCK, cosine_sim_to_pref_rollout

    system, params, parts = agent(terms=("cosine_sim_to_pref_rollout",), epistemic=False)
    crspp, cparams = parts[0], parts[1]
    name = "cosine_sim_to_pref_rollout"
    (mount,) = [m for m in system._row_mounts(params) if m.name == name]
    base = registry.Statics(
        index=(),
        key=KEY if keyed else None,
        scoring=params.scoring,
        declared=dict(system.scoring),
        structure=system._amortized_structure(params),
        policy_len=HORIZON,
        horizon=HORIZON,
    )
    st = registry.mount_statics(mount, base)
    step = cosine_sim_to_pref_rollout(st)
    keys = None if not keyed else jax.random.split(st.key, HORIZON)
    assert not keyed or bool(
        (st.key == jax.random.fold_in(KEY, zlib.crc32(name.encode()) & 0x7FFFFFFF)).all()
    )
    rolled = [crspp.initial_priors(cparams)]
    for t in range(HORIZON):
        k_t = None if keys is None else keys[t]
        rolled.append(crspp.update_empirical_prior(cparams, rolled[-1], {}, key=k_t))
    ours = system.predict_beliefs(params, system.initial_priors(params), jnp.zeros((HORIZON, 1)))
    for t in range(HORIZON):
        qs = system.to_positional("beliefs", ours[t])
        a, b = features(qs), features(crspp.to_positional("beliefs", rolled[t + 1]))
        want = jnp.sum(a * b) / jnp.maximum(jnp.linalg.norm(a) * jnp.linalg.norm(b), 1e-12)
        got = step(registry.StepContext(qs=qs, action=None, t=jnp.asarray(t)))[CRSPP_BLOCK]
        np.testing.assert_allclose(np.asarray(got.midpoint), np.asarray(want), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("horizon", [1, 3])
def test_the_cosine_row_rolls_its_preferred_trajectory_to_the_scored_horizon(horizon):
    """The row sizes its per-step table to ``Statics.horizon``, the plan's own length, not
    to ``policy_len``: every step of a plan shorter or longer than the System's default
    reads a finite preferred state, and the table has exactly the scored length."""
    from hybrid_aif.objectives import registry
    from hybrid_aif.reference.raif.crspp import CRSPP_BLOCK, cosine_sim_to_pref_rollout

    system, params, _ = agent(terms=("cosine_sim_to_pref_rollout",), epistemic=False)
    assert horizon != system.policy_len
    (mount,) = [m for m in system._row_mounts(params) if m.name == "cosine_sim_to_pref_rollout"]
    base = registry.Statics(
        index=(),
        scoring=params.scoring,
        declared=dict(system.scoring),
        structure=system._amortized_structure(params),
        policy_len=system.policy_len,
        horizon=horizon,
    )
    step = cosine_sim_to_pref_rollout(registry.mount_statics(mount, base))
    cells = dict(zip(step.__code__.co_freevars, step.__closure__, strict=True))
    assert cells["preferred"].cell_contents.shape[0] == horizon
    plan = jnp.zeros((horizon, 1))
    ours = system.predict_beliefs(params, system.initial_priors(params), plan)
    for t in range(horizon):
        qs = system.to_positional("beliefs", ours[t])
        got = step(registry.StepContext(qs=qs, action=None, t=jnp.asarray(t)))[CRSPP_BLOCK]
        assert np.isfinite(float(got.midpoint))
    #  a Statics with no scored horizon is refused rather than sized to policy_len
    with pytest.raises(ValueError, match="Statics.horizon"):
        cosine_sim_to_pref_rollout(registry.mount_statics(mount, base._replace(horizon=0)))


def test_the_cosine_row_without_its_declared_model_says_what_to_declare():
    system, params, _ = agent(terms=("cosine_sim_to_pref_rollout",), epistemic=False)
    orphan = dataclasses.replace(system, scoring=())
    with pytest.raises(ValueError, match="own a declared model and found none"):
        orphan.infer_policies(params, system.initial_priors(params), plan_key=KEY)


# ------------------------------------------------------------------- the g-tail


def test_g_tail_closes_the_horizon_with_the_critic_at_the_final_step():
    """A4's learned value tail, once, at the horizon end — not per step. The critic's
    decoded mean is a value of *negative* EFE (addendum §3), so the leaf is stored
    better-when-larger under a −1 term: G subtracts the value still to come."""
    system, params, parts = agent(terms=("g_tail",), epistemic=False)
    critic = parts[2]
    beliefs, policies, tree = _plan_and_tree(system, params)
    final = system.predict_beliefs(params, beliefs, policies[0])[-1]
    wanted = float(critic.predict((features(system.to_positional("beliefs", final)),)))
    assert float(np.asarray(tree["g_tail"]["critic"].midpoint)[0]) == pytest.approx(
        wanted, rel=1e-5
    )
    assert float(np.asarray(total(tree).midpoint)[0]) == pytest.approx(-wanted, rel=1e-5)


def test_only_the_critic_is_read_at_scoring_time():
    """``scoring['g_tail']`` holds ``(critic, slow)`` and the slow copy is a
    *training* target regularizer (``slow_critic_fraction 0.02``) — 2c's trainer's
    concern. Replacing it changes no score."""
    system, params, parts = agent(terms=("g_tail",), epistemic=False)
    critic = parts[2]
    _, _, tree = _plan_and_tree(system, params)
    swapped = dataclasses.replace(params, scoring={**params.scoring, "g_tail": (critic, "junk")})
    _, _, other = _plan_and_tree(system, swapped)
    assert float(np.asarray(other["g_tail"]["critic"].midpoint)[0]) == pytest.approx(
        float(np.asarray(tree["g_tail"]["critic"].midpoint)[0])
    )


# ------------------------------------------------------------- the habit planner


def test_the_habit_planner_acts_from_the_module_and_never_from_G():
    """A3's H=0 end: the actor *is* the policy. Two different actors on one world
    model give different plans; two different critics on one actor do not — G is
    scored for diagnostics and never consulted for selection."""
    system, params, parts = agent()
    beliefs = system.initial_priors(params)
    _, empty, plan = get_planner("habit")(system.planner_context(params, beliefs, key=KEY))
    assert empty == {}  # nothing was scored to choose this plan
    other_actor = MLPConditional.from_shapes(
        (FEATURE,), 1, key=jax.random.PRNGKey(99), width=4, depth=1
    )
    _, _, replanned = get_planner("habit")(
        system.planner_context(
            dataclasses.replace(params, policy_prior=other_actor), beliefs, key=KEY
        )
    )
    assert plan.shape == (1, HORIZON, 1)
    assert not np.allclose(np.asarray(plan), np.asarray(replanned))
    loud_critic = PredictorHead.from_shapes(
        (FEATURE,), loss="symlog_disc", num_bins=9, key=jax.random.PRNGKey(98), width=4, depth=1
    )
    _, _, same = get_planner("habit")(
        system.planner_context(
            dataclasses.replace(
                params, scoring={**params.scoring, "g_tail": (loud_critic, loud_critic)}
            ),
            beliefs,
            key=KEY,
        )
    )
    np.testing.assert_allclose(np.asarray(plan), np.asarray(same))


def test_the_habit_planner_plan_is_the_stepwise_loop():
    """The rolled plan equals the act-push-act Python loop bitwise: same key splits,
    same raw actions, the box only on the transition's side."""
    system, params, _ = agent()
    ctx = system.planner_context(params, system.initial_priors(params), key=KEY)
    _, _, plan = get_planner("habit")(ctx)
    bounds = ctx.continuous_controls.bounds
    carry, key, actions = ctx.init_beliefs, ctx.key, []
    for _ in range(ctx.policy_len):
        key, draw = jax.random.split(key)
        action = ctx.habit_fn(carry, draw)
        actions.append(action)
        carry = ctx.transition_fn(carry, soft_clip(action, bounds))
    np.testing.assert_allclose(
        np.asarray(plan), np.asarray(jnp.stack(actions)[None]), rtol=1e-6, atol=1e-6
    )


def test_the_habit_planner_returns_an_empty_tree_not_a_missing_one():
    """The rule ``metrics=`` obeys: it toggles whether the *by-products of the
    selection* come back, and their content is whatever **term tree** that
    selection produced. A search planner built one in order to select, so it
    returns it. This planner's selection is the actor's distribution — already
    returned in full as ``q_pi`` plus the plan — and it produces no term tree, so
    ``metrics=True`` gives ``{}``: "requested; there were none", which stays
    distinguishable from ``None``, "not requested".

    Not a gap in the tree's vocabulary but its subject matter: the term tree scores
    the *model*, and the policy is not part of the model
    (``objective-terms.md`` — ``log π``/``H[π]`` have no term-tree home by
    decision). The honest "why this action" under a habit is ``q_pi`` and the plan.
    """
    system, params, _ = agent()
    beliefs = system.initial_priors(params)
    _, asked = system.infer_policies(params, beliefs, plan_key=KEY)
    _, unasked = system.infer_policies(params, beliefs, metrics=False, plan_key=KEY)
    assert asked == {} and unasked is None
    _, _, from_act = system.act(KEY, params, beliefs, plan_key=KEY)
    assert from_act == {}


def test_folding_an_empty_tree_raises_which_is_the_documented_behaviour():
    """There is no G of nothing, and the zero ``total`` might otherwise return is a
    number nobody computed. A caller who wants the habit's plan scored asks the
    face that scores plans."""
    system, params, _ = agent()
    beliefs = system.initial_priors(params)
    _, tree = system.infer_policies(params, beliefs, plan_key=KEY)
    with pytest.raises(ValueError, match="no included leaves"):
        total(tree)


def test_the_distillation_gap_is_one_call_on_the_teacher_face():
    """A3's teacher face, and the reason the habit planner needs no scoring of its
    own: ``score_policies`` takes the plan the habit chose (float, over a
    continuous action space — A9) and returns the same tree any planner would
    report, so the habit-minus-search gap the slider is read off is a subtraction.
    """
    system, params, _ = agent()
    beliefs = system.initial_priors(params)
    plan = _habit_plan(system, params, beliefs)
    habit_G = total(system.score_policies(params, beliefs, plan))
    searcher = dataclasses.replace(system, planner="cem")
    searched = dataclasses.replace(params, policy_prior=None)  # CEM's E is its proposal
    _, tree = searcher.infer_policies(
        searched,
        beliefs,
        planner=make_cem_planner(num_candidates=4, cem_iters=1),
        plan_key=KEY,
    )
    gap = float(np.asarray(habit_G.midpoint)[0]) - float(np.asarray(total(tree).midpoint).min())
    assert np.isfinite(gap)
    assert habit_G.derived is False and total(tree).derived is False


def test_the_habit_planner_needs_its_own_key():
    """A habit is a stochastic policy: the draw is the caller's to control, never
    split off the action-selection key."""
    system, params, _ = agent()
    with pytest.raises(ValueError, match="plan_key"):
        system.infer_policies(params, system.initial_priors(params))


def test_the_habit_planner_also_serves_the_classical_vector_E():
    """E's other representation, in the same planner: ``q(π) = E(π)`` over the
    enumerated policy set, no γ and no G — A3's dial at its habit end on the
    discrete path."""
    from tests.test_system import tmaze_system

    system, params = tmaze_system(policy_len=1)
    E = jnp.log(jnp.asarray([0.5, 0.2, 0.1, 0.1, 0.1]))
    params = dataclasses.replace(params, policy_prior=E)
    system = dataclasses.replace(system, planner="habit")
    q_pi, tree = system.infer_policies(params, system.initial_priors(params), plan_key=KEY)
    np.testing.assert_allclose(np.asarray(q_pi.probs), np.exp(np.asarray(E)), atol=1e-6)
    assert tree == {}  # E selected; no term tree was computed to do it
    scored = system.score_policies(params, system.initial_priors(params), system.policies(params))
    assert total(scored).derived is True  # the tabular tree is untouched by any of this


def test_a_system_with_no_E_at_all_cannot_act_from_a_habit():
    from tests.test_system import tmaze_system

    system, params = tmaze_system(policy_len=1)
    system = dataclasses.replace(system, planner="habit")
    with pytest.raises(ValueError, match="acts from E and this System declares none"):
        system.infer_policies(params, system.initial_priors(params), plan_key=KEY)


def test_the_enumeration_planner_refuses_a_policy_module_by_name():
    """Deliverable of the unfence: what is still impossible stays loud and named.
    A3 says what a module E would mean under enumeration — the product of its
    per-step action probabilities — and it is deliberately unbuilt."""
    from tests.test_system import tmaze_system

    system, params = tmaze_system(policy_len=1)
    params = dataclasses.replace(params, policy_prior=lambda beliefs: None)
    with pytest.raises(NotImplementedError, match="prod_t"):
        system.infer_policies(params, system.initial_priors(params))


# ------------------------------------------------------- the other planners, unfenced


def test_cem_plans_over_the_amortized_system_and_the_metrics_free_path_agrees():
    """Continuous actions over a neural state space is CEM's home (A9), and it is
    unfenced by the same change. ``metrics=False`` exercises the accumulator whose
    warrant seed has to be read *statically* — an unwarranted step tree would
    otherwise change a ``lax.scan`` carry's pytree structure between the initial
    value and the first step."""
    system, params, _ = agent()
    system = dataclasses.replace(system, planner="cem")
    params = dataclasses.replace(params, policy_prior=None)  # CEM's E is its proposal (A3)
    beliefs = system.initial_priors(params)
    planner = make_cem_planner(num_candidates=4, cem_iters=1)
    q_with, tree = system.infer_policies(params, beliefs, planner=planner, plan_key=KEY)
    q_without, none = system.infer_policies(
        params, beliefs, planner=planner, metrics=False, plan_key=KEY
    )
    assert none is None
    assert total(tree).derived is False
    np.testing.assert_allclose(
        np.asarray(q_with.probs), np.asarray(q_without.probs), rtol=1e-4, atol=1e-6
    )


def test_score_policies_takes_a_hand_built_plan_on_the_amortized_path():
    """The planner-free face, unfenced with the rest: a continuous plan in, a term
    tree out, same closures."""
    system, params, _ = agent()
    beliefs = system.initial_priors(params)
    plan = jnp.zeros((HORIZON, 1), jnp.float32)
    tree = system.score_policies(params, beliefs, plan)
    assert set(tree) == {
        "occupancy",
        INFOGAIN,
        "predicted_reward",
        "cosine_sim_to_pref_rollout",
        "g_tail",  # the horizon end is scored here too: the same closures, no planner
    }
    assert total(tree).derived is False


# ------------------------------------------------------------------- the CRSPP


def test_the_crspp_is_a_declared_system_with_no_action_and_no_heads():
    """A12 item 2 / A10 disposition 5: their "RSSM without action encoded into the
    transition prior", sharing the architecture and not the parameters. No decoder
    of its own (addendum §2), hence no modality — so it is prediction-only, and its
    recognition side reads the *agent's* embedding as an external input rather than
    an observation of its own (P1′ chunk 2c, ledger entry 5)."""
    crspp, params = crspp_rssm(key=KEY, deter=DETER, num_vars=NUM_VARS, num_classes=NUM_CLASSES)
    assert crspp.A_deps == () and params.likelihood == {}
    assert crspp.controlled == (False, False)
    assert [r.parents for r in crspp.recognition] == [("deter", "external:embed")]
    assert crspp.external_inputs == (("embed", 64),)
    beliefs = crspp.initial_priors(params)
    assert [b.shape for b in beliefs.values()] == [(DETER,), (NUM_VARS, NUM_CLASSES)]
    rolled = crspp.predict_beliefs(params, beliefs, jnp.zeros((3, 0), jnp.int32))
    assert len(rolled) == 3
    np.testing.assert_allclose(float(rolled[-1]["stoch"].sum()), NUM_VARS, rtol=1e-5)
    #  it moves
    assert not np.allclose(np.asarray(rolled[0]["deter"]), np.asarray(rolled[-1]["deter"]))


def test_the_crspp_shares_the_architecture_but_not_the_parameters():
    system, params, parts = agent()
    crspp_params = parts[1]
    assert crspp_params.transition["stoch"].parent_dims == params.transition["stoch"].parent_dims
    assert not np.allclose(
        np.asarray(crspp_params.transition["stoch"].trunk.layers[0].weight),
        np.asarray(params.transition["stoch"].trunk.layers[0].weight),
    )
