"""The training side of the R-AIF replication: ``imagine``, the ensemble, the CRSPP.

Design: ``docs/design/amortization.md`` A6 (trainers are pure functions over
explicit state), A2 (the ``LearnState`` carry, explicit init), A4/A12 item 5 (the
critic), A10 dispositions 3/5/6/8/9, and ``amortization-p0-addendum.md`` §§3–5 for
every number. The two MUSTs of the chunk get a test each: ``imagine`` **samples**
the code, and the CRSPP posterior is **conditioned on the batch's observations**
(``amortization-debt-ledger.md`` entry 5).

Sizes are tiny and nothing here is trained to convergence — that is
``test_raif_acceptance.py``. What is under test is shape, sign, seam and carry.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import MLPConditional
from hybrid_aif.distributions.neural import (
    PredictorHead,
    factored_categorical_entropy,
    symlog_disc_mean,
)
from hybrid_aif.interval import Interval
from hybrid_aif.objectives import registry
from hybrid_aif.objectives.base import INFOGAIN, term_sign, total
from hybrid_aif.reference.raif.agent import raif_mountain_car, to_world_batch, train_step
from hybrid_aif.reference.raif.columns import EXTRAS, raif_episode
from hybrid_aif.reference.raif.crspp import (
    ROW as CRSPP_ROW,
)
from hybrid_aif.reference.raif.crspp import (
    STATE_KEY as CRSPP_STATE_KEY,
)
from hybrid_aif.reference.raif.crspp import (
    crspp_loss,
    crspp_recognize,
    embeddings,
    fit_crspp,
    observe_no_action,
    resolve,
)
from hybrid_aif.reference.raif.revise import label
from hybrid_aif.reference.raif.world import CODE_NAME, tiny_rssm
from hybrid_aif.system import ACTOR_KEY, LearnState
from hybrid_aif.train.buffers import StaticBuffer, add, sample
from hybrid_aif.train.ensemble import (
    EnsembleBatch,
    ensemble_loss,
    fit_ensemble,
    state_key,
    transitions,
)
from hybrid_aif.train.imagine import (
    CRITIC_KEY,
    RETNORM_KEY,
    SLOW_KEY,
    imagine,
    init_retnorm,
    lambda_return,
    perc_ema,
    term_value,
)
from hybrid_aif.train.world import posterior_states

KEY = jax.random.PRNGKey(0)
TINY = dict(
    horizon=2,
    head_units=8,
    head_layers=1,
    units=4,
    deter=8,
    num_vars=3,
    num_classes=4,
    embed=5,
    mlp_units=6,
    ensemble_size=4,
)
B, T = 2, 4

#: The row the reference agent mounts (R-AIF's own sign — paper Eq. 6 / 7b).
ENSEMBLE_ROW = "model_uncertainty"
ENSEMBLE_KEY = state_key(ENSEMBLE_ROW)


@pytest.fixture(scope="module")
def agent():
    return raif_mountain_car(KEY, **TINY)


@pytest.fixture(scope="module")
def batch():
    """A tiny replay batch in the :class:`Episode` format, with ρ filled."""
    t = T

    def one(i: int):
        return label(
            raif_episode(
                obs=jax.random.normal(jax.random.PRNGKey(10 + i), (t, 2)),
                action=jax.random.normal(jax.random.PRNGKey(20 + i), (t, 1)),
                reward=jnp.full((t,), -2.0),
                is_success=jnp.zeros((t,), bool).at[-1].set(i == 0),
                has_succeeded=jnp.zeros((t,), bool).at[-1].set(i == 0),
                can_self_imitate=jnp.full((t,), i == 0),
                is_first=jnp.zeros((t,), bool).at[0].set(True),
                is_last=jnp.zeros((t,), bool).at[-1].set(True),
            )
        )

    buf = StaticBuffer.empty(B, t, 2, 1, extras=EXTRAS)
    for i in range(B):
        buf = add(buf, one(i))
    return sample(buf, KEY, B)


@pytest.fixture(scope="module")
def starts(agent, batch):
    """The flattened posterior start states and the CRSPP's — computed once."""
    return _starts(agent, batch)


def _starts(agent, batch):
    """The flattened posterior start states (keyed by factor) and the CRSPP's, as ``imagine``
    wants them."""
    world_batch = to_world_batch(batch)
    states = posterior_states(agent.system, agent.params, world_batch, key=KEY)
    beliefs = agent.system.beliefs_from_states(agent.params, states)
    n = B * T
    flat = {k: jnp.reshape(x, (n, *x.shape[2:])) for k, x in beliefs.items()}
    embeds = embeddings(agent.system, agent.params, world_batch.obs)
    crspp = crspp_recognize(agent.crspp, agent.crspp_params, embeds, key=KEY, is_first=None)
    flat_crspp = agent.crspp.from_positional(
        "beliefs", tuple(jnp.reshape(x, (n, *x.shape[2:])) for x in crspp)
    )
    return flat, flat_crspp, states, embeds


# ================================================================ the pieces of the loss


def test_the_lambda_return_reduces_to_the_one_step_target_at_lambda_zero():
    """``lambda=0``: ``ret_k = g_k + disc_k · v_{k+1}`` — the TD(0) target, per state."""
    value = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    reward = jnp.asarray([10.0, 20.0, 30.0])
    disc = jnp.asarray([0.5, 0.5, 0.5])
    got = lambda_return(value, reward, disc, 0.0)
    np.testing.assert_allclose(np.asarray(got), [10.5, 21.0, 31.5], rtol=1e-6)


def test_the_lambda_return_reduces_to_the_full_monte_carlo_sum_at_lambda_one():
    """``lambda=1``: the discounted sum of rewards, bootstrapped once at the leaf."""
    value = jnp.asarray([0.0, 0.0, 0.0, 7.0])
    reward = jnp.asarray([1.0, 1.0, 1.0])
    disc = jnp.full((3,), 0.5)
    got = lambda_return(value, reward, disc, 1.0)
    # ret_2 = 1 + .5*7 = 4.5 ; ret_1 = 1 + .5*4.5 = 3.25 ; ret_0 = 1 + .5*3.25 = 2.625
    np.testing.assert_allclose(np.asarray(got), [2.625, 3.25, 4.5], rtol=1e-6)


def test_a_zero_continuation_stops_the_return_accumulating():
    """ "multiplied by predicted continuation": a trajectory the head believes has
    ended stops earning, rather than being truncated by hand."""
    value = jnp.asarray([0.0, 5.0, 5.0, 5.0])
    reward = jnp.asarray([1.0, 1.0, 1.0])
    got = lambda_return(value, reward, jnp.asarray([0.0, 1.0, 1.0]), 0.95)
    assert float(got[0]) == pytest.approx(1.0, abs=1e-6)  # nothing after step 0 counts


def test_the_return_normalizer_is_a_percentile_ema_that_never_amplifies():
    """``retnorm: perc_ema``, ``max 1.0`` ⇒ ``invscale = max(1, high − low)``: a
    small spread is left alone rather than blown up (IDD §E Eq. 4–5)."""
    state = init_retnorm()
    x = jnp.linspace(-1.0, 1.0, 101)
    new, offset, invscale = perc_ema(state, x, 5.0, 95.0, 0.99, 1.0)
    assert float(invscale) == pytest.approx(1.0)  # the true spread is 1.8 * 0.01 EMA weight
    assert float(new[0]) < 0.0 < float(new[1])
    # a big spread does scale, once the EMA has caught up
    warm = jnp.stack([jnp.asarray(-100.0), jnp.asarray(100.0)])
    _, _, big = perc_ema(warm, x * 100.0, 5.0, 95.0, 0.0, 1.0)
    assert float(big) > 1.0


def test_term_value_negates_our_g_once_and_scales_per_term():
    """A10 disposition 7 from the training side: their critic values *negative* EFE,
    so the fold negates once, and that is the whole of the conversion: there is no
    per-term weight argument, because a term's sign is registered once (their ``−IG``
    is the ``model_uncertainty`` row, not a scale on ``infogain``)."""
    tree = {
        "predicted_reward": {"reward": Interval.exact(jnp.asarray(3.0))},
        "model_uncertainty": {"stoch": Interval.exact(jnp.asarray(2.0))},
    }
    value, per_term = term_value(tree)
    assert term_sign("predicted_reward") == -1.0  # a value: G rewards it
    assert term_sign("model_uncertainty") == 1.0  # a cost: G penalizes it
    #  value = −sign·leaf summed: +3 for the reward, −2 for the uncertainty
    assert float(value) == pytest.approx(1.0)
    assert float(per_term["predicted_reward"]) == pytest.approx(3.0)
    assert float(per_term["model_uncertainty"]) == pytest.approx(-2.0)


def test_term_value_refuses_an_empty_tree():
    with pytest.raises(ValueError, match="empty fold"):
        term_value({})


# ================================================================== imagine, the MUSTs


def test_imagine_samples_the_code_along_the_rollout(agent, starts):
    """**MUST** (chunk 2a's measurement): the imagined carry holds a straight-through
    one-hot *draw*, never the probability tensor. Asserted where it is observable —
    two different keys give two different rollouts, and the code entering the heads
    is a one-hot, so the actor's own log-probability moves with the draw. A
    mean-field rollout would be a deterministic function of the start state.
    """
    flat, flat_crspp, _, _ = starts
    kwargs = dict(aux_beliefs={CRSPP_ROW: flat_crspp}, horizon=2)
    _, _, a = imagine(agent.system, agent.params, flat, agent.learn_state, key=KEY, **kwargs)
    _, _, b = imagine(
        agent.system,
        agent.params,
        flat,
        agent.learn_state,
        key=jax.random.PRNGKey(99),
        **kwargs,
    )
    assert float(a.value) != float(b.value)
    assert float(a.prior_entropy) > 0.0  # the *distribution*'s entropy, not the sample's


def test_the_state_prior_entropy_is_read_off_the_logits_not_the_sample(agent):
    """The ``wment`` quantity is the prior's entropy; a one-hot sample's is zero, so
    reading it off the carried belief would silently delete paper Eq. 10's ζ term."""
    onehot = jnp.zeros((3, 4)).at[:, 0].set(1.0)
    logits = jnp.log(jnp.maximum(onehot, 1e-30))
    assert float(factored_categorical_entropy(logits)) == pytest.approx(0.0, abs=1e-5)
    assert float(factored_categorical_entropy(jnp.zeros((3, 4)))) == pytest.approx(
        3 * float(jnp.log(4.0)), rel=1e-6
    )


def test_the_cosine_row_starts_from_the_crspp_posterior_when_it_is_supplied(agent, starts):
    """**MUST** — ledger entry 5. With ``aux_beliefs`` the preferred trajectory
    starts from the CRSPP's posterior over *this batch's* observations (their
    ``observe_no_action``); without it, from the CRSPP's declared episode start,
    which is a *generic* trajectory and is what made 2b's cosine numbers
    incomparable to theirs. The two give different similarities."""
    flat, flat_crspp, _, _ = starts
    args = (agent.system, agent.params, flat, agent.learn_state)
    _, _, seeded = imagine(*args, key=KEY, horizon=2, aux_beliefs={CRSPP_ROW: flat_crspp})
    _, _, generic = imagine(*args, key=KEY, horizon=2)
    row = "cosine_sim_to_pref_rollout"
    assert float(seeded.per_term[row]) != float(generic.per_term[row])


@pytest.mark.parametrize("horizon", [1, 3])
def test_the_aux_belief_seam_is_general_and_reaches_score_policies(agent, starts, horizon):
    """The seam is ``aux_beliefs={row: beliefs}`` on ``planner_context`` /
    ``score_policies``, not an ``imagine`` private: the same auxiliary beliefs change
    the tree the *teacher face* returns, which is where the diagnostic lives. The plan
    scored is shorter, then longer, than ``policy_len``: every row is built for the
    plan's own horizon, so every leaf is finite either way."""
    _, flat_crspp, _, _ = starts
    assert horizon != agent.system.policy_len
    beliefs = agent.system.initial_priors(agent.params)
    one = {k: x[0] for k, x in flat_crspp.items()}
    plan = jnp.zeros((horizon, 1))
    plain = agent.system.score_policies(agent.params, beliefs, plan)
    seeded = agent.system.score_policies(agent.params, beliefs, plan, aux_beliefs={CRSPP_ROW: one})
    for tree in (plain, seeded):
        for term, blocks in tree.items():
            for block, iv in blocks.items():
                assert bool(jnp.isfinite(iv.lower).all()), (term, block)
                assert bool(jnp.isfinite(iv.upper).all()), (term, block)
    assert float(plain[CRSPP_ROW]["crspp"].midpoint) != float(seeded[CRSPP_ROW]["crspp"].midpoint)
    #  the other rows do not read the carry
    for term in plain:
        if term != CRSPP_ROW:
            for block in plain[term]:
                assert float(plain[term][block].midpoint) == float(seeded[term][block].midpoint)


def test_aux_beliefs_on_a_non_amortized_system_raises():
    """No row on the tabular/LG paths owns a model, so the argument names nothing."""
    from hybrid_aif.distributions import Categorical, ConditionalCategorical
    from hybrid_aif.system import System, SystemParams

    system = System(B_deps=((0,),), controlled=(True,), policy_len=2, A_deps=((0,),))
    uniform = ConditionalCategorical.from_probs(jnp.full((2, 2, 2), 0.5), parent_ndims=2)
    params = SystemParams(
        transition={"factor_0": uniform},
        initial_state_prior={"factor_0": Categorical(logits=jnp.zeros((2,)))},
        likelihood={
            "modality_0": ConditionalCategorical.from_probs(jnp.full((2, 2), 0.5), parent_ndims=1)
        },
    )
    with pytest.raises(ValueError, match="auxiliary beliefs"):
        system.planner_context(params, system.initial_priors(params), aux_beliefs={"g_tail": ()})


# ============================================================= imagine, the objective


def test_imagine_steps_the_actor_and_the_critic_and_only_those(agent, starts):
    """A6: it steps its own two addresses and threads the rest through bitwise, which
    is what lets four trainers share one carry without knowing about each other."""
    flat, flat_crspp, _, _ = starts
    params, learn_state, _ = imagine(
        agent.system,
        agent.params,
        flat,
        agent.learn_state,
        key=KEY,
        horizon=2,
        aux_beliefs={CRSPP_ROW: flat_crspp},
    )
    assert not _same(params.policy_prior, agent.params.policy_prior)
    assert not _same(params.scoring["g_tail"][0], agent.params.scoring["g_tail"][0])
    #  the world model, the ensemble and the CRSPP are untouched
    assert _same(params.transition, agent.params.transition)
    assert _same(params.recognition, agent.params.recognition)
    assert _same(
        params.scoring[ENSEMBLE_ROW],
        agent.params.scoring[ENSEMBLE_ROW],
    )
    assert _same(params.scoring[CRSPP_ROW], agent.params.scoring[CRSPP_ROW])
    assert set(learn_state.opt_states) == set(agent.learn_state.opt_states)


def test_the_slow_critic_is_hard_copied_once_and_then_moves_by_ema(agent, starts):
    """Their ``SlowUpdater``: ``mix = clip(need_init + fraction·need_update, 0, 1)``.

    ``need_init`` is ``updates == 0``, so the **first** call is a hard copy whatever
    the fraction is — and only from the second does ``slow_critic_fraction 0.02``
    bite. That first branch was missing until 2026-09-02 (divergence 13); it is
    asserted here by the one case that separates the two readings, ``fraction = 0.0``,
    where a pure EMA would leave the slow copy untouched forever and theirs copies
    once. The counter rides in ``LearnState.moments``, so a resumed run does not
    re-copy.
    """
    flat, _, _, _ = starts
    args = (agent.system, agent.params, flat, agent.learn_state)
    #  fraction 0.0: theirs still copies on the first call (need_init), then freezes
    first, state, _ = imagine(*args, key=KEY, horizon=2, slow_fraction=0.0)
    assert _same(first.scoring["g_tail"][1], first.scoring["g_tail"][0])
    assert not _same(first.scoring["g_tail"][1], agent.params.scoring["g_tail"][1])
    assert int(state.moments[SLOW_KEY]) == 1
    #  the second call, with the counter carried, does NOT copy at fraction 0.0
    second, again, _ = imagine(
        agent.system, first, flat, state, key=KEY, horizon=2, slow_fraction=0.0
    )
    assert _same(second.scoring["g_tail"][1], first.scoring["g_tail"][1])
    assert int(again.moments[SLOW_KEY]) == 2
    #  and at fraction 1.0 it tracks the stepped critic every step
    copied, _, _ = imagine(agent.system, first, flat, state, key=KEY, horizon=2, slow_fraction=1.0)
    assert _same(copied.scoring["g_tail"][1], copied.scoring["g_tail"][0])


def test_the_retnorm_carry_rides_in_learn_state_moments(agent, starts):
    """Not optimizer state and nothing steps it, so it is a second field rather than
    an ``opt_states`` entry — and it is *resumable*, which is the point of A6."""
    flat, _, _, _ = starts
    _, learn_state, first = imagine(
        agent.system, agent.params, flat, agent.learn_state, key=KEY, horizon=2
    )
    assert RETNORM_KEY in learn_state.moments
    np.testing.assert_allclose(
        np.asarray(learn_state.moments[RETNORM_KEY]),
        [float(first.retnorm_low), float(first.retnorm_high)],
        rtol=1e-6,
    )
    _, again, second = imagine(agent.system, agent.params, flat, learn_state, key=KEY, horizon=2)
    assert float(second.retnorm_low) != float(first.retnorm_low)  # the EMA advanced


def test_init_learn_state_seeds_the_moments_so_the_carry_is_scan_stable(agent, starts):
    """The carry's pytree structure is fixed at init: ``init_learn_state`` seeds both of
    ``imagine``'s running statistics beside the ``g_tail`` row's optimizer state, and a
    step fills them without adding anything — the structure a ``lax.scan`` carry
    needs, and the whole legitimate first-call path."""
    fresh = agent.system.init_learn_state(agent.params)
    assert set(fresh.moments) == {RETNORM_KEY, SLOW_KEY}
    np.testing.assert_array_equal(np.asarray(fresh.moments[RETNORM_KEY]), np.zeros(2))
    assert fresh.moments[RETNORM_KEY].dtype == jnp.float32
    assert int(fresh.moments[SLOW_KEY]) == 0 and fresh.moments[SLOW_KEY].dtype == jnp.int32

    flat, _, _, _ = starts
    _, stepped, _ = imagine(agent.system, agent.params, flat, fresh, key=KEY, horizon=2)
    assert jax.tree_util.tree_structure(stepped) == jax.tree_util.tree_structure(fresh)
    assert int(stepped.moments[SLOW_KEY]) == 1

    #  with no g_tail row mounted there is nothing to seed
    world = {k: v for k, v in TINY.items() if k in ("units", "deter", "num_vars", "num_classes")}
    system, params = tiny_rssm(2, 1, key=KEY, policy_len=2, **world)
    assert system.init_learn_state(params).moments == {}


@pytest.mark.parametrize("dropped", [RETNORM_KEY, SLOW_KEY, "both"])
def test_a_learn_state_missing_a_moments_entry_makes_imagine_raise(agent, starts, dropped):
    """A supplied ``LearnState`` without one of ``imagine``'s entries is a dropped
    carry, never a first call: ``imagine`` refuses it by name rather than re-seeding
    the normalizer or the slow critic's counter."""
    flat, _, _, _ = starts
    keep = () if dropped == "both" else ({RETNORM_KEY, SLOW_KEY} - {dropped})
    moments = {k: v for k, v in agent.learn_state.moments.items() if k in keep}
    broken = LearnState(opt_states=agent.learn_state.opt_states, moments=moments)
    missing = [RETNORM_KEY, SLOW_KEY] if dropped == "both" else [dropped]
    with pytest.raises(ValueError, match=r"imagine: learn_state\.moments is missing") as info:
        imagine(agent.system, agent.params, flat, broken, key=KEY, horizon=2)
    message = str(info.value)
    assert repr(missing) in message
    assert "not re-seeded" in message
    assert "system.init_learn_state(params)" in message


@pytest.mark.parametrize("dropped", [ACTOR_KEY, CRITIC_KEY, "both"])
def test_a_learn_state_missing_an_adam_state_makes_imagine_raise(agent, starts, dropped):
    """The same refusal one layer down: a supplied ``LearnState`` without the actor's
    or the critic's adam state is a dropped carry, and re-seeding it would restart
    adam's moments on every step."""
    flat, _, _, _ = starts
    gone = {ACTOR_KEY, CRITIC_KEY} if dropped == "both" else {dropped}
    opt = {k: v for k, v in agent.learn_state.opt_states.items() if k not in gone}
    broken = LearnState(opt_states=opt, moments=agent.learn_state.moments)
    with pytest.raises(ValueError, match=r"imagine: learn_state\.opt_states is missing") as info:
        imagine(agent.system, agent.params, flat, broken, key=KEY, horizon=2)
    message = str(info.value)
    for k in sorted(gone):
        assert repr(k) in message
    assert "not re-seeded" in message
    assert "system.init_learn_state(params)" in message


def test_a_learn_state_missing_the_ensembles_adam_state_makes_fit_ensemble_raise(agent, batch):
    world_batch = to_world_batch(batch)
    states = posterior_states(agent.system, agent.params, world_batch, key=KEY)
    opt = {k: v for k, v in agent.learn_state.opt_states.items() if k != ENSEMBLE_KEY}
    broken = LearnState(opt_states=opt, moments=agent.learn_state.moments)
    with pytest.raises(ValueError, match=r"fit_ensemble: learn_state\.opt_states is missing"):
        fit_ensemble(
            agent.system,
            agent.params,
            broken,
            transitions(states.deter, states.stoch, world_batch.action),
        )


def test_the_bc_term_is_the_negative_log_probability_of_the_replay_action(agent, batch, starts):
    """A10 disposition 8: the term lives *inside* the actor loss, mixed. Its content
    is ``−log π(a_start)·can_self_imitate``, so raising the scale must raise the
    actor loss by exactly ``scale · mean(−log π · flag)``."""
    flat, _, _, _ = starts
    n = flat["deter"].shape[0]
    actions = agent.system.from_positional(
        "actions", jnp.reshape(batch.action, (n, 1)), agent.params
    )
    flags = jnp.ones((n,))
    args = (agent.system, agent.params, flat, agent.learn_state)
    kw = dict(key=KEY, horizon=2, start_actions=actions, bc_weights=flags)
    off, _, m0 = imagine(*args, bc_scale=0.0, **kw)
    on, _, m1 = imagine(*args, bc_scale=1.0, **kw)
    expected = -float(m0.bc_logprob)
    assert float(m1.actor_loss) - float(m0.actor_loss) == pytest.approx(expected, rel=1e-4)
    assert float(m0.bc_weight) == 1.0


def test_start_actions_are_keyed_by_the_action_node(agent, batch, starts):
    """``start_actions`` speaks action-node names; one keyed by the control node is refused
    naming the action node."""
    flat, _, _, _ = starts
    n = flat["deter"].shape[0]
    actions = agent.system.from_positional(
        "actions", jnp.reshape(batch.action, (n, 1)), agent.params
    )
    assert list(actions) == list(agent.system.names.actions) == ["a_deter"]
    with pytest.raises(ValueError, match="imagine: actions names 'u_deter', the control node"):
        imagine(
            agent.system,
            agent.params,
            flat,
            agent.learn_state,
            key=KEY,
            horizon=2,
            start_actions={"u_deter": actions["a_deter"]},
            bc_weights=jnp.ones((n,)),
        )


def test_imagine_ignores_a_declared_actuator_kernel(agent, starts):
    """The imagined rollouts read the control node, so a kernel changes nothing imagine returns."""
    flat, _, _, _ = starts
    params = dataclasses.replace(agent.params, actuator={"a_deter": jnp.eye(1)})
    plain = imagine(agent.system, agent.params, flat, agent.learn_state, key=KEY, horizon=2)
    stepped, state, metrics = imagine(
        agent.system, params, flat, agent.learn_state, key=KEY, horizon=2
    )
    with_kernel = (dataclasses.replace(stepped, actuator={}), state, metrics)
    for a, b in zip(
        jax.tree_util.tree_leaves(plain), jax.tree_util.tree_leaves(with_kernel), strict=True
    ):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_a_flag_without_an_action_raises(agent, starts):
    """Half a term is a mistake worth naming rather than defaulting past."""
    flat, _, _, _ = starts
    with pytest.raises(ValueError, match="together or neither"):
        imagine(
            agent.system,
            agent.params,
            flat,
            agent.learn_state,
            key=KEY,
            horizon=2,
            bc_weights=jnp.ones((flat["deter"].shape[0],)),
        )


def test_imagine_needs_its_optimizer_carry_explicitly(agent, starts):
    """A2's explicit-init rule: ``None`` raises rather than silently restarting adam."""
    flat, _, _, _ = starts
    with pytest.raises(ValueError, match="explicit-init"):
        imagine(agent.system, agent.params, flat, None, key=KEY, horizon=2)


def test_a_vector_policy_prior_has_nothing_to_backpropagate(agent, starts):
    """A3's two E representations: only a *module* is trainable by ``imagine``."""
    flat, _, _, _ = starts
    params = dataclasses.replace(agent.params, policy_prior=jnp.zeros((4,)))
    with pytest.raises(ValueError, match="declares no policy"):
        imagine(agent.system, params, flat, agent.learn_state, key=KEY, horizon=2)


def test_a_static_zero_rate_holds_a_head_still(agent, starts):
    """The rate grammar, inherited: ``scoring:g_tail`` is addressable *because* this
    trainer exists (ledger entry 6), and a static ``0.0`` skips its step."""
    flat, _, _, _ = starts
    params, _, _ = imagine(
        agent.system,
        agent.params,
        flat,
        agent.learn_state,
        key=KEY,
        horizon=2,
        lr_by_slot={CRITIC_KEY: 0.0},
    )
    assert _same(params.scoring["g_tail"][0], agent.params.scoring["g_tail"][0])
    assert not _same(params.policy_prior, agent.params.policy_prior)


def test_imagine_is_jit_able(agent, starts):
    flat, flat_crspp, _, _ = starts
    run = eqx.filter_jit(
        lambda p, ls, k: imagine(
            agent.system, p, flat, ls, key=k, horizon=2, aux_beliefs={CRSPP_ROW: flat_crspp}
        )
    )
    _, _, metrics = run(agent.params, agent.learn_state, KEY)
    assert bool(jnp.isfinite(metrics.actor_loss)) and bool(jnp.isfinite(metrics.critic_loss))


# ==================================================================== the ensemble


def test_the_ensemble_is_trained_on_the_transition_the_action_caused(agent, batch):
    """``(s_t, a_t) -> s_{t+1}``: inputs from step ``t``, the action from the *next*
    replay slot (which is the one that caused the move), the target from ``t+1``."""
    world_batch = to_world_batch(batch)
    states = posterior_states(agent.system, agent.params, world_batch, key=KEY)
    ens_batch = transitions(states.deter, states.stoch, world_batch.action)
    assert ens_batch.deter.shape == (B * (T - 1), 8)
    assert ens_batch.code.shape == (B * (T - 1), 12) and ens_batch.target.shape == (B * (T - 1), 12)
    np.testing.assert_allclose(
        np.asarray(ens_batch.action), np.asarray(batch.action[:, :-1]).reshape(-1, 1), rtol=1e-6
    )


def test_the_ensemble_loss_is_the_summed_member_nll_and_keeps_the_per_member_vector(agent, batch):
    """Their reduction is ``.sum()`` over members (``epistemics.py:61``), not a mean.

    Closed 2026-09-01 (chunk 2d divergence (c)). The reduction is a pure loss scale
    and adam is *nearly* invariant to one — but ``eps`` and a global-norm clip both
    see the raw scale, so ``nearly`` is not ``exactly`` and the transcription is what
    the file says rather than what is convenient. Asserted as ``sum``, and separately
    as ``M ×`` the mean, so a future change back to a mean fails loudly instead of
    drifting by a factor nobody re-derives.

    ``per_member`` stays the unscaled diagnostic: four members converging on the same
    NLL is the failure mode that silently zeroes the epistemic term, and it is
    invisible in the total.
    """
    world_batch = to_world_batch(batch)
    states = posterior_states(agent.system, agent.params, world_batch, key=KEY)
    members = agent.params.scoring[ENSEMBLE_ROW]
    total, out = ensemble_loss(members, transitions(states.deter, states.stoch, world_batch.action))
    assert out.per_member.shape == (4,)
    assert float(total) == pytest.approx(float(jnp.sum(out.per_member)), rel=1e-6)
    assert float(total) == pytest.approx(4.0 * float(jnp.mean(out.per_member)), rel=1e-6)
    assert len(set(np.asarray(out.per_member).round(4))) > 1  # they disagree at init


def test_fit_ensemble_steps_only_the_ensemble(agent, batch):
    world_batch = to_world_batch(batch)
    states = posterior_states(agent.system, agent.params, world_batch, key=KEY)
    params, learn_state, _ = fit_ensemble(
        agent.system,
        agent.params,
        agent.learn_state,
        transitions(states.deter, states.stoch, world_batch.action),
    )
    assert not _same(params.scoring[ENSEMBLE_ROW], agent.params.scoring[ENSEMBLE_ROW])
    assert _same(params.transition, agent.params.transition)
    assert _same(params.scoring["g_tail"], agent.params.scoring["g_tail"])
    assert ENSEMBLE_KEY in learn_state.opt_states


def test_member_weights_default_to_ones_and_fold_the_sum(agent, batch):
    """Without weights every member's loss enters the total at weight one — the plain
    sum — and the weights ride back in ``EnsembleLosses.weights``."""
    world_batch = to_world_batch(batch)
    states = posterior_states(agent.system, agent.params, world_batch, key=KEY)
    members = agent.params.scoring[ENSEMBLE_ROW]
    ens_batch = transitions(states.deter, states.stoch, world_batch.action)
    total, out = ensemble_loss(members, ens_batch)
    assert np.array_equal(np.asarray(out.weights), np.ones(4, np.float32))
    np.testing.assert_allclose(float(total), float(jnp.sum(out.per_member)), rtol=1e-6)
    explicit, _ = ensemble_loss(members, ens_batch, member_weights=jnp.ones(4))
    assert float(explicit) == float(total)


def test_a_zero_weighted_member_receives_no_gradient_and_the_others_are_unchanged(agent, batch):
    """The per-member weight is the general form of a member gate: weight zero
    detaches that member's loss from the step, weight one leaves the others exactly
    as the unweighted loss had them."""
    world_batch = to_world_batch(batch)
    states = posterior_states(agent.system, agent.params, world_batch, key=KEY)
    members = tuple(agent.params.scoring[ENSEMBLE_ROW])
    ens_batch = transitions(states.deter, states.stoch, world_batch.action)
    weights = jnp.array([1.0, 0.0, 1.0, 1.0])
    grads, out = eqx.filter_grad(
        lambda ms: ensemble_loss(ms, ens_batch, member_weights=weights), has_aux=True
    )(members)
    plain, _ = eqx.filter_grad(lambda ms: ensemble_loss(ms, ens_batch), has_aux=True)(members)
    assert np.array_equal(np.asarray(out.weights), np.asarray(weights))
    for i, (g, g0) in enumerate(zip(grads, plain, strict=True)):
        leaves, leaves0 = jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(g0)
        if i == 1:
            assert all(float(jnp.abs(x).max()) == 0.0 for x in leaves)
        else:
            for x, y in zip(leaves, leaves0, strict=True):
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_fit_ensemble_threads_member_weights_and_refuses_the_wrong_count(agent, batch):
    world_batch = to_world_batch(batch)
    states = posterior_states(agent.system, agent.params, world_batch, key=KEY)
    ens_batch = transitions(states.deter, states.stoch, world_batch.action)
    common = (agent.system, agent.params, agent.learn_state, ens_batch)
    off, _, off_out = fit_ensemble(*common)
    on, _, on_out = fit_ensemble(*common, member_weights=jnp.array([1.0, 0.0, 1.0, 1.0]))
    assert np.array_equal(np.asarray(off_out.weights), np.ones(4, np.float32))
    assert float(on_out.weights[1]) == 0.0
    #  the zero-weighted member did not move; a weighted one did
    before = agent.params.scoring[ENSEMBLE_ROW]
    for i in (0, 1):
        same = all(
            np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(
                jax.tree_util.tree_leaves(on.scoring[ENSEMBLE_ROW][i]),
                jax.tree_util.tree_leaves(before[i]),
                strict=True,
            )
        )
        assert same == (i == 1)
    with pytest.raises(ValueError, match="one weight per member"):
        ensemble_loss(tuple(before), ens_batch, member_weights=jnp.ones(3))


def test_a_member_with_the_wrong_parent_widths_is_named(agent, batch):
    world_batch = to_world_batch(batch)
    states = posterior_states(agent.system, agent.params, world_batch, key=KEY)
    ens_batch = transitions(states.deter, states.stoch, world_batch.action)
    members = agent.params.scoring[ENSEMBLE_ROW]
    with pytest.raises(ValueError, match=r"member\(s\)"):
        ensemble_loss(members, ens_batch._replace(deter=ens_batch.deter[:, :-1]))


# ======================================================================== the CRSPP


def test_the_crspp_posterior_reads_the_agents_embedding_as_an_external_input(agent):
    """Ledger entry 5's seam, at the declaration: one recognition entry over
    ``(its own carry, external:embed)``, the width declared statically and the value
    supplied per call. The agent's encoder is **not** in the CRSPP's slot —
    A11 refinement (ii)'s pytree-forking trap one level up."""
    parts = resolve(agent.crspp, agent.crspp_params)
    assert parts.embed_name == "embed"
    assert agent.crspp.external_inputs == (("embed", 5),)
    assert set(agent.crspp_params.recognition) == {"stoch"}
    encoder = agent.params.recognition["embed"]
    assert all(m is not encoder for m in agent.crspp_params.recognition.values())


def test_observe_no_action_is_action_free_and_carries_a_sample(agent):
    """Their ``observe_no_action``: no action parent anywhere, and the posterior code
    carried forward is a straight-through draw (a one-hot, so it sums to num_vars)."""
    parts = resolve(agent.crspp, agent.crspp_params)
    embeds = jax.random.normal(KEY, (4, 5))
    deters, codes, post, prior = observe_no_action(parts, KEY, embeds)
    assert deters.shape == (4, 8) and codes.shape == (4, 3, 4)
    np.testing.assert_allclose(
        np.asarray(jnp.sum(codes, axis=(-2, -1))), np.full(4, 3.0), rtol=1e-5
    )
    assert post.shape == prior.shape == (4, 3, 4)
    assert not np.allclose(np.asarray(post), np.asarray(prior))  # two parameter sets


def test_the_crspp_batch_functions_require_is_first():
    """Every CRSPP function that filters ``(B, T)`` windows takes ``is_first`` as a required
    argument with no default: a caller must hand it the batch's episode starts or declare,
    with ``None``, that each window is one episode. A preference model that silently carried
    across episode boundaries is what the actor-entropy diagnostic §29 traced."""
    import inspect

    for fn in (crspp_loss, crspp_recognize, fit_crspp):
        param = inspect.signature(fn).parameters["is_first"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, fn.__name__
        assert param.default is inspect.Parameter.empty, fn.__name__


def test_observe_no_action_restarts_at_is_first(agent):
    """An ``is_first`` inside the sequence restarts the filter from the CRSPP's own initial
    state: the restarted step's carry and prior equal the first step's bitwise, and an
    all-False mask is the unmasked scan."""
    parts = resolve(agent.crspp, agent.crspp_params)
    embeds = jax.random.normal(KEY, (6, 5))
    plain = observe_no_action(parts, KEY, embeds)
    unmasked = observe_no_action(parts, KEY, embeds, jnp.zeros(6, bool))
    for a, b in zip(plain, unmasked, strict=True):
        assert jnp.array_equal(a, b)
    first = jnp.asarray([True, False, False, True, False, False])
    deters, _, _, prior = observe_no_action(parts, KEY, embeds, first)
    assert jnp.array_equal(deters[3], deters[0]) and jnp.array_equal(prior[3], prior[0])
    assert not jnp.array_equal(plain[0][3], deters[3])
    with pytest.raises(ValueError, match="is_first"):
        observe_no_action(parts, KEY, embeds, jnp.zeros(5, bool))


def test_crspp_recognize_returns_belief_tuples_the_row_can_start_from(agent, batch):
    embeds = embeddings(agent.system, agent.params, to_world_batch(batch).obs)
    beliefs = crspp_recognize(agent.crspp, agent.crspp_params, embeds, key=KEY, is_first=None)
    assert [x.shape for x in beliefs] == [(B, T, 8), (B, T, 3, 4)]


def test_the_crspp_kl_is_gated_to_positive_states(agent, batch):
    """Paper Eq. 5: ``max(0, sgn ρ)``. With ρ ≤ 0 everywhere the KL contributes
    nothing at all, and the reported ``positive`` fraction says so."""
    parts = resolve(agent.crspp, agent.crspp_params)
    embeds = embeddings(agent.system, agent.params, to_world_batch(batch).obs)
    feats = jax.random.normal(jax.random.PRNGKey(5), (B, T, 20))
    _, none = crspp_loss(parts, KEY, embeds, -jnp.ones((B, T)), feats, is_first=None)
    _, all_of_it = crspp_loss(parts, KEY, embeds, jnp.ones((B, T)), feats, is_first=None)
    assert float(none.kl) == 0.0 and float(none.positive) == 0.0
    assert float(all_of_it.kl) > 0.0 and float(all_of_it.positive) == 1.0
    assert float(all_of_it.kl_raw) > 0.0  # the ungated KL is real either way


def test_the_contrastive_term_is_minus_rho_times_the_cosine_similarity(agent, batch):
    """``−ρ · sim(C_post, sg(post))`` with ``contrast 1.0``: a positive state pulls
    the preferred posterior toward the agent's state, a negative one pushes it away,
    and the agent's side is stop-gradiented (the CRSPP moves, the agent does not)."""
    parts = resolve(agent.crspp, agent.crspp_params)
    embeds = embeddings(agent.system, agent.params, to_world_batch(batch).obs)
    deters, codes, _, _ = jax.vmap(lambda k, e: observe_no_action(parts, k, e))(
        jax.random.split(KEY, B), embeds
    )
    feats = jax.vmap(jax.vmap(lambda d, c: jnp.concatenate([d, c.reshape(-1)])))(deters, codes)
    _, same = crspp_loss(parts, KEY, embeds, jnp.ones((B, T)), feats, is_first=None)
    #  compared against itself, the cosine is 1, so the term is −ρ·1 = −1
    assert float(same.contrast) == pytest.approx(-1.0, abs=1e-3)
    _, flipped = crspp_loss(parts, KEY, embeds, -jnp.ones((B, T)), feats, is_first=None)
    assert float(flipped.contrast) == pytest.approx(1.0, abs=1e-3)


def test_fit_crspp_steps_the_preference_model_through_its_own_nested_carry(agent, batch):
    """A row that owns a System owns its slot addresses too, so its optimizer carry
    is a nested ``LearnState`` over ``transition:*`` / ``recognition:*`` /
    ``initial_state_prior:*`` — and its rate grammar is the ordinary one."""
    embeds = embeddings(agent.system, agent.params, to_world_batch(batch).obs)
    nested = agent.learn_state.opt_states[CRSPP_STATE_KEY]
    assert isinstance(nested, LearnState)
    #  real features, not zeros: at initialization the CRSPP's balanced KL sits BELOW
    #  the one-nat free-bits floor, where the term is a constant with exactly zero
    #  gradient — so early preference learning is driven entirely by the contrastive
    #  term, and a zero-feature batch would make the whole loss flat.
    feats = jax.random.normal(jax.random.PRNGKey(5), (B, T, 20))
    stepped, state, losses = fit_crspp(
        agent.crspp,
        agent.crspp_params,
        nested,
        embeds,
        jnp.ones((B, T)),
        feats,
        key=KEY,
        is_first=None,
    )
    assert not _same(stepped.transition, agent.crspp_params.transition)
    assert not _same(stepped.recognition, agent.crspp_params.recognition)
    assert bool(jnp.isfinite(losses.total))
    held, _, _ = fit_crspp(
        agent.crspp,
        agent.crspp_params,
        nested,
        embeds,
        jnp.ones((B, T)),
        feats,
        key=KEY,
        is_first=None,
        lr_by_slot={"transition:*": 0.0, "recognition:*": 0.0, "initial_state_prior:*": 0.0},
    )
    assert _same(held.transition, agent.crspp_params.transition)


def test_a_crspp_with_an_action_parent_is_refused(agent):
    """ "An RSSM without action encoded into the transition prior" is the whole
    content of the model (paper §3.2), so a controlled factor is not a variant."""
    #  deter given a control node, declared natively (appended after the state factors)
    broken = dataclasses.replace(
        agent.crspp,
        B_deps=((1, 2), (0,), ()),
        factor_names=(*agent.crspp.names.states, "u_deter"),
        controls=(2,),
        action_names=("a_deter",),
    )
    #  the declaration is refused before ``resolve`` even sees it: a control node on the carry
    #  has no axis in its map to read it by (so no level count), the same rule speaking.
    with pytest.raises(ValueError, match="none of its readers .* carries an axis for it"):
        resolve(broken, agent.crspp_params)


# ================================================================== the assembled step


def test_the_whole_train_step_runs_and_moves_every_parameter_set(agent, batch):
    """The recipe's body, once: four losses, all finite, and all five parameter sets
    changed. This is the shape of the acceptance loop, without the budget."""
    stepped, losses = train_step(agent, batch, key=KEY, horizon=2)
    for value in (
        losses.world.total,
        losses.imagine.actor_loss,
        losses.imagine.critic_loss,
        losses.ensemble.total,
        losses.crspp.total,
    ):
        assert bool(jnp.isfinite(value)), losses
    assert not _same(stepped.params.transition, agent.params.transition)
    assert not _same(stepped.params.policy_prior, agent.params.policy_prior)
    assert not _same(stepped.params.scoring["g_tail"][0], agent.params.scoring["g_tail"][0])
    assert not _same(
        stepped.params.scoring[ENSEMBLE_ROW],
        agent.params.scoring[ENSEMBLE_ROW],
    )
    assert not _same(stepped.params.scoring[CRSPP_ROW], agent.params.scoring[CRSPP_ROW])
    #  the CRSPP's params travel in the slot AND on the Agent, and they agree
    assert _same(stepped.crspp_params, stepped.params.scoring[CRSPP_ROW])


def test_the_learn_state_carries_every_parameter_set_and_survives_a_step(agent, batch):
    """One carry for the whole agent (ledger entry 6): the world model's slots, the
    actor's bare key, and one entry per trained scoring row."""
    keys = set(agent.learn_state.opt_states)
    assert {ACTOR_KEY, CRITIC_KEY, ENSEMBLE_KEY, CRSPP_STATE_KEY} <= keys
    stepped, _ = train_step(agent, batch, key=KEY, horizon=2)
    assert set(stepped.learn_state.opt_states) == keys
    #  resumable: the second step accepts the first's carry
    train_step(stepped, batch, key=KEY, horizon=2)


def test_the_critics_decoded_value_is_the_symlog_disc_mean(agent):
    """A12 item 5 / A10 disposition 6: the head follows the code — 255 two-hot bins
    whose decoded expectation is the value, not paper Eq. 9's MSE regression."""
    critic = agent.params.scoring["g_tail"][0]
    assert critic.loss == "symlog_disc"
    feats = jax.random.normal(KEY, (critic.parent_dims[0],))
    logits = critic((feats,))
    np.testing.assert_allclose(
        float(critic.predict((feats,))), float(symlog_disc_mean(logits, critic.bins)), rtol=1e-6
    )


def _same(a, b) -> bool:
    """Whether two pytrees are bitwise equal on their array leaves."""
    left, right = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    if len(left) != len(right):
        return False
    return all(bool(jnp.array_equal(x, y)) for x, y in zip(left, right, strict=True))


# ============================================ the ensemble's two rows, and their signs


def test_the_two_ensemble_rows_share_an_estimator_and_differ_only_in_sign():
    """One estimator, two terms: curiosity under ``infogain``, pessimism under
    ``model_uncertainty``.

    R-AIF's own sign is the **cost** (adjudicated 2026-08-28 against the paper: Eq. 6
    writes the value as ``Σ γ^(t−τ)(r + sim − IG)`` and names the last term "the
    negative information gain −IG_τ"; Eq. 7b's prose has the policy "trained to reduce
    the model parameter's uncertainty … to maintain 'homeostasis'"). Expressed as a
    second registered row rather than as a scale on the first, because the sign of a
    term is a modelling claim and ``objectives.base`` registers it once.
    """
    row = registry.get_row("model_uncertainty")
    curiosity = registry.ESTIMATORS[(registry.STATE, registry.MLP_CATEGORICAL)]
    assert row.prepare is curiosity.prepare  # literally the same estimator function
    assert term_sign("model_uncertainty") == 1.0  # a cost
    assert term_sign(INFOGAIN) == -1.0  # a value
    assert row.derived is False and row.parametric is True
    assert row.trainer == curiosity.trainer  # the same NLL fits either


def test_the_row_is_mounted_by_a_bare_state_variable_key():
    """``terms=("model_uncertainty:stoch",)`` — V1's own spelling, the segment the
    ``epistemic`` selector takes. Not slot-qualified: the row reads a *belief*."""
    system, params, _ = agent_2b(
        terms=("model_uncertainty:stoch", "g_tail"),
        epistemic=False,
        ensemble_row="model_uncertainty",
    )
    tree = system.score_policies(params, system.initial_priors(params), jnp.zeros((2, 1)))
    assert "model_uncertainty" in tree and INFOGAIN not in tree
    assert set(tree["model_uncertainty"]) == {"stoch"}


def test_the_rows_key_grammar_is_checked():
    """A slot-qualified or unknown spelling raises, and a wildcard is the class form."""
    for terms, error, match in (
        (("model_uncertainty:transition:stoch",), ValueError, "one bare name"),
        (("model_uncertainty:nope",), ValueError, "unknown state variable"),
        #  a deterministic node has no successor: no ('model_uncertainty', 'carry') estimator
        (("model_uncertainty:deter",), NotImplementedError, "is 'carry'"),
    ):
        system, params, _ = agent_2b(
            terms=(*terms, "g_tail"), epistemic=False, ensemble_row="model_uncertainty"
        )
        with pytest.raises(error, match=match):
            system._row_mounts(params)
    #  the wildcard resolves to the code factors, silently skipping the carry
    system, params, _ = agent_2b(
        terms=("model_uncertainty:*", "g_tail"),
        epistemic=False,
        ensemble_row="model_uncertainty",
    )
    mounts = {m.name: m.index for m in system._row_mounts(params)}
    assert mounts["model_uncertainty"] == (1,)  # `stoch`, not `deter`


def test_the_cost_row_folds_into_g_with_the_opposite_sign():
    """The same nets, the same beliefs, the same leaf — opposite contributions to G.

    This is the whole content of the adjudication, asserted as arithmetic: mount the
    estimator as curiosity and as pessimism against *separate copies of one
    initialization*, and the leaf is identical while G moves by twice it.
    """
    shared = _ensemble(jax.random.PRNGKey(77))
    curious, c_params, _ = agent_2b(
        terms=("g_tail",),
        epistemic=("stoch",),
        ensemble=shared,
        ensemble_row="ensemble_disagreement",
    )
    pessimist, p_params, _ = agent_2b(
        terms=("model_uncertainty:stoch", "g_tail"),
        epistemic=False,
        ensemble=shared,
        ensemble_row="model_uncertainty",
    )
    plan = jnp.zeros((2, 1))
    c_tree = curious.score_policies(c_params, curious.initial_priors(c_params), plan)
    p_tree = pessimist.score_policies(p_params, pessimist.initial_priors(p_params), plan)
    leaf_c = float(c_tree[INFOGAIN]["stoch"].midpoint)
    leaf_p = float(p_tree["model_uncertainty"]["stoch"].midpoint)
    assert leaf_c == pytest.approx(leaf_p, rel=1e-6)  # one estimator
    #  G differs by exactly 2·leaf: −leaf under infogain, +leaf under model_uncertainty
    gap = float(total(p_tree).midpoint) - float(total(c_tree).midpoint)
    assert gap == pytest.approx(2.0 * leaf_c, rel=1e-5)


def test_both_rows_mounted_against_one_ensemble_raises():
    """A11's module-sharing commitment, one slot over: each mounted row owns its nets.

    The scoring slot is a pytree, so an apparently shared ensemble forks at the first
    optimizer step and the two copies drift apart silently — the same trap the
    recognition slot refuses. Separate nets are legal (see the next test).
    """
    shared = _ensemble(jax.random.PRNGKey(78))
    system, params, _ = agent_2b(
        terms=("model_uncertainty:stoch", "g_tail"),
        epistemic=("stoch",),
        ensemble=shared,
        ensemble_row="both_shared",
    )
    with pytest.raises(ValueError, match="same module object"):
        system.score_policies(params, system.initial_priors(params), jnp.zeros((2, 1)))


def test_both_rows_mounted_with_separate_ensembles_is_legal_and_carries_both_leaves():
    """Curious about one thing, pessimistic about another — two independent ensembles.

    Legal by construction, and the tree then carries both leaves with their own signs,
    which is what makes the pair an *ablation* rather than a switch.
    """
    system, params, _ = agent_2b(
        terms=("model_uncertainty:stoch", "g_tail"),
        epistemic=("stoch",),
        ensemble_row="both_separate",
    )
    tree = system.score_policies(params, system.initial_priors(params), jnp.zeros((2, 1)))
    assert INFOGAIN in tree and "model_uncertainty" in tree


def test_the_rows_parameters_rates_and_learn_state_key_under_its_own_name():
    """Ledger entry 6's grammar, for the new row: ``scoring:model_uncertainty``."""
    system, params, _ = agent_2b(
        terms=("model_uncertainty:stoch", "g_tail"),
        epistemic=False,
        ensemble_row="model_uncertainty",
    )
    key = state_key("model_uncertainty")
    assert key in system._slot_declarations(params)
    assert system._resolve_rates({key: 0.25}, "lr_by_slot", params)[key] == 0.25
    assert key in system.init_learn_state(params).opt_states


def test_fit_ensemble_trains_either_row_and_refuses_to_guess_between_two():
    """The loss does not depend on the sign, so one trainer serves both rows — but
    with two independent ensembles present, which to step is the caller's call."""
    system, params, _ = agent_2b(
        terms=("model_uncertainty:stoch", "g_tail"),
        epistemic=False,
        ensemble_row="model_uncertainty",
    )
    learn_state = system.init_learn_state(params)
    batch = _ensemble_batch()
    stepped, out_state, _ = fit_ensemble(system, params, learn_state, batch)
    assert not _same(stepped.scoring["model_uncertainty"], params.scoring["model_uncertainty"])
    assert state_key("model_uncertainty") in out_state.opt_states

    both, both_params, _ = agent_2b(
        terms=("model_uncertainty:stoch", "g_tail"),
        epistemic=("stoch",),
        ensemble_row="both_separate",
    )
    ls = both.init_learn_state(both_params)
    with pytest.raises(ValueError, match="both ensemble rows"):
        fit_ensemble(both, both_params, ls, batch)
    named, _, _ = fit_ensemble(both, both_params, ls, batch, row="model_uncertainty")
    assert not _same(named.scoring["model_uncertainty"], both_params.scoring["model_uncertainty"])
    assert _same(
        named.scoring["ensemble_disagreement"], both_params.scoring["ensemble_disagreement"]
    )


def test_the_reference_agents_parity_config_carries_the_cost_row_and_not_curiosity(agent, batch):
    """The R-AIF configuration, asserted term by term: ``reward + sim − IG`` closed by
    the critic. ``epistemic=False``, ``model_uncertainty`` mounted, and the ensemble's
    parameters under that name — so nothing in the parity run rewards disagreement."""
    assert agent.system.epistemic is False
    assert f"model_uncertainty:{CODE_NAME}" in agent.system.terms
    assert "model_uncertainty" in agent.params.scoring
    assert "ensemble_disagreement" not in agent.params.scoring
    beliefs = agent.system.initial_priors(agent.params)
    tree = agent.system.score_policies(agent.params, beliefs, jnp.zeros((2, 1)))
    assert set(tree) == {
        "predicted_reward",
        "cosine_sim_to_pref_rollout",
        "model_uncertainty",
        "g_tail",
    }
    assert INFOGAIN not in tree
    assert "occupancy" not in tree  # the actor's wment bonus, not the critic's value


def _ensemble(key):
    """Four ``MLPConditional`` heads over the next code, at this file's tiny sizes."""
    flat = TINY["num_vars"] * TINY["num_classes"]
    return tuple(
        MLPConditional.from_shapes((TINY["deter"], flat, 1), flat, key=k, width=4, depth=1)
        for k in jax.random.split(key, 4)
    )


def _ensemble_batch() -> EnsembleBatch:
    flat = TINY["num_vars"] * TINY["num_classes"]
    keys = jax.random.split(jax.random.PRNGKey(9), 4)
    return EnsembleBatch(
        deter=jax.random.normal(keys[0], (6, TINY["deter"])),
        code=jax.random.uniform(keys[1], (6, flat)),
        action=jax.random.normal(keys[2], (6, 1)),
        target=jax.random.uniform(keys[3], (6, flat)),
    )


def agent_2b(*, terms, epistemic, ensemble=None, ensemble_row="ensemble_disagreement"):
    """A ``tiny_rssm`` with a chosen G side — the 2b test idiom, parameterized.

    ``ensemble_row`` names how the ensemble is keyed in ``SystemParams.scoring``:
    one of the two row names, ``"both_shared"`` (the same tuple under both — the
    configuration that must raise) or ``"both_separate"`` (two independent
    ensembles, which is legal).
    """
    keys = jax.random.split(jax.random.PRNGKey(4), 6)
    system, params = tiny_rssm(
        2,
        1,
        key=keys[0],
        deter=TINY["deter"],
        num_vars=TINY["num_vars"],
        num_classes=TINY["num_classes"],
        units=TINY["units"],
        mlp_units=TINY["mlp_units"],
        embed=TINY["embed"],
        policy_len=2,
    )
    feature = TINY["deter"] + TINY["num_vars"] * TINY["num_classes"]
    critic = PredictorHead.from_shapes(
        (feature,), loss="symlog_disc", num_bins=9, key=keys[1], width=4, depth=1
    )
    slow = PredictorHead.from_shapes(
        (feature,), loss="symlog_disc", num_bins=9, key=keys[2], width=4, depth=1
    )
    actor = MLPConditional.from_shapes((feature,), 1, key=keys[3], width=4, depth=1)
    nets = _ensemble(keys[4]) if ensemble is None else ensemble
    scoring: dict[str, object] = {"g_tail": (critic, slow)}
    if ensemble_row == "both_shared":
        scoring["ensemble_disagreement"] = scoring["model_uncertainty"] = nets
    elif ensemble_row == "both_separate":
        scoring["ensemble_disagreement"] = nets
        scoring["model_uncertainty"] = _ensemble(keys[5])
    else:
        scoring[ensemble_row] = nets
    system = dataclasses.replace(system, planner="habit", epistemic=epistemic, terms=terms)
    params = dataclasses.replace(params, policy_prior=actor, scoring=scoring)
    return system, params, nets


# ================================================== the forward pass's states, reused


def test_fit_world_hands_back_the_states_of_the_pass_it_differentiated(agent, batch):
    """``return_states=True`` is the pre-step recognition with the loss's own draws:
    what ``posterior_states`` says at the *old* parameters under the same key (to
    rounding: the pass ran under the gradient's tracing), and not what it says at
    the stepped ones."""
    from hybrid_aif.reference.raif.agent import to_world_batch
    from hybrid_aif.train.world import fit_world, posterior_states

    world_batch = to_world_batch(batch)
    k = jax.random.PRNGKey(3)
    params, _, _, states = fit_world(
        agent.system,
        agent.params,
        agent.learn_state,
        world_batch,
        key=k,
        lr=1e-2,
        return_states=True,
    )
    before = posterior_states(agent.system, agent.params, world_batch, key=k)
    after = posterior_states(agent.system, params, world_batch, key=k)
    for got, want in zip(
        jax.tree_util.tree_leaves(states), jax.tree_util.tree_leaves(before), strict=True
    ):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-6)
    #  the step moved the carry; the one-hot codes may or may not flip under a small step
    assert not np.allclose(np.asarray(states.deter), np.asarray(after.deter), rtol=1e-5, atol=1e-6)


def test_fit_crspp_hands_back_the_posterior_of_the_pass_it_differentiated(agent, batch):
    """The CRSPP half of the same reuse: the fourth return is ``crspp_recognize``
    at the old parameters under the loss's key, in the row's declaration order."""
    from hybrid_aif.reference.raif.agent import CRSPP_STATE_KEY, to_world_batch
    from hybrid_aif.reference.raif.columns import PREF_LABEL
    from hybrid_aif.reference.raif.crspp import crspp_recognize, embeddings, fit_crspp
    from hybrid_aif.train.world import posterior_states

    world_batch = to_world_batch(batch)
    k = jax.random.PRNGKey(4)
    embeds = embeddings(agent.system, agent.params, world_batch.obs)
    states = posterior_states(agent.system, agent.params, world_batch, key=KEY)
    feats = jax.vmap(jax.vmap(lambda d, s: jnp.concatenate([d, s.reshape(-1)])))(
        states.deter, states.stoch
    )
    crspp_params, _, _, beliefs = fit_crspp(
        agent.crspp,
        agent.crspp_params,
        agent.learn_state.opt_states[CRSPP_STATE_KEY],
        embeds,
        batch.extras[PREF_LABEL],
        feats,
        key=k,
        is_first=world_batch.is_first,
        lr=1e-2,
        return_states=True,
    )
    first = world_batch.is_first
    before = crspp_recognize(agent.crspp, agent.crspp_params, embeds, key=k, is_first=first)
    after = crspp_recognize(agent.crspp, crspp_params, embeds, key=k, is_first=first)
    assert len(beliefs) == len(before)
    for got, want in zip(beliefs, before, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-6)
    #  the step moved at least the carry; the one-hot codes may or may not flip
    assert any(
        not np.allclose(np.asarray(got), np.asarray(moved), rtol=1e-5, atol=1e-6)
        for got, moved in zip(beliefs, after, strict=True)
    )


def test_imagine_refuses_a_categorical_target_on_a_continuous_control_node(agent, starts):
    """A target on a continuous node is a density over its values; the reference agent's node
    is continuous, so imagine refuses a Categorical target with the planning path's sentence."""
    from hybrid_aif.distributions import Categorical

    flat, flat_crspp, _, _ = starts
    with_target = dataclasses.replace(
        agent.params, state_targets={"u_deter": Categorical(logits=jnp.zeros(2))}
    )
    with pytest.raises(
        ValueError, match=r"^imagine: .*on continuous control node 'u_deter': pass a Gaussian"
    ):
        imagine(
            agent.system,
            with_target,
            flat,
            agent.learn_state,
            key=KEY,
            horizon=2,
            aux_beliefs={CRSPP_ROW: flat_crspp},
        )


def test_imagine_scores_a_target_on_a_continuous_control_node(agent, starts):
    """A Gaussian target on the reference agent's continuous node enters G as that node's
    ``risk`` leaf, ``−log C(a)`` at each imagined action: the per-term value carries it."""
    from hybrid_aif.distributions import Gaussian

    flat, flat_crspp, _, _ = starts
    target = Gaussian(loc=jnp.zeros(1), chol=0.1 * jnp.eye(1))
    with_target = dataclasses.replace(agent.params, state_targets={"u_deter": target})
    kw = dict(key=KEY, horizon=2, aux_beliefs={CRSPP_ROW: flat_crspp})
    _, _, plain = imagine(agent.system, agent.params, flat, agent.learn_state, **kw)
    _, _, metrics = imagine(agent.system, with_target, flat, agent.learn_state, **kw)
    assert "risk" in metrics.per_term and "risk" not in plain.per_term
    assert np.isfinite(float(metrics.per_term["risk"])) and float(metrics.per_term["risk"]) != 0.0
