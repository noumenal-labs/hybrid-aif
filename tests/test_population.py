"""The population contract: a population of agents is a ``vmap``, and a seed is an agent.

**The claim under test.** ONE static :class:`~hybrid_aif.system.System`, every
numeric carrier on a leading batch axis. A *population* of independent agents is a
``jax.vmap`` of the canonical loop's faces over that axis — nothing about a face
changes, no face learns that it might be one of many — and a **training seed is an
agent whose entire numeric state is batched**: parameters, the optimizer's moments,
the replay buffer, the environment's state, the PRNG key. That is what
``hybrid_aif.reference.raif.run`` runs four of at once, and it is why it needs no
per-seed process, no parameter server and no synchronisation: there is nothing to
synchronise, because a population has no interaction.

**Why it is a test and not a remark.** ``docs/project-brief.md`` commits to being
"vmapped across agents"; ``system/core.py``'s ``learn`` docstring states the
convention ("``learn`` is unbatched and lifts via ``vmap`` — per-agent rates
included, as traced arrays"); and until this file, *nothing in the suite vmapped a
single System face*. A convention no test exercises is a convention that decays at
the first shape-reading line inside a face. So each test here does the same thing:
build ``POPULATION`` members whose trainable state genuinely **differs** (divergent
``learn`` steps, or distinct init keys), run one ``vmap``, run the members
sequentially, and compare leaf for leaf.

**The comparison, and its exact-equality subset.** Floats are compared
``allclose`` at float32 tolerance, not bitwise: ``vmap`` rewrites a loop of small
contractions into one batched contraction, and XLA is free to pick a different
kernel and reduction order for it. Integer and boolean leaves are compared
**exactly** — a sampled action's level index, a buffer's ``size``/``cursor``, a
sampled clip index, an ``is_last`` flag. That split is the honest statement of the
contract: batching preserves *arithmetic* to float32 and *control flow* bitwise. A
face whose discrete output moved under batching would be a bug this catches; a
last-bit difference in a Dirichlet concentration is XLA doing its job.

**Population versus composition.** Orthogonal, and both are available at once —
see the dated paragraph beside ``docs/design/system.md``'s Faces section. A
population is ``vmap``: independent replicas of one System, no interaction, one
declaration. A composition is :func:`hybrid_aif.composition.merge`: several minds
inside one joint model, interacting, *a new System*. Nothing here touches merging;
a population **of** merged Systems is this file's ``vmap`` over ``merge()``'s
output, which needs no further machinery precisely because ``merge()`` returns a
System like any other.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalDirichlet,
    Gaussian,
    MLPConditional,
)
from hybrid_aif.reference.envs.mountain_car import MountainCar
from hybrid_aif.reference.raif.agent import act, raif_mountain_car
from hybrid_aif.reference.raif.columns import EXTRAS
from hybrid_aif.reference.raif.revise import label
from hybrid_aif.reference.raif.seed import bang_bang, collect_episode
from hybrid_aif.system import System, SystemParams
from hybrid_aif.train.buffers import StaticBuffer, add, sample

#: Three members: enough that a broadcast bug cannot masquerade as agreement (two
#: would let an accidental "member 0 for everyone" pass on a symmetric case) and
#: small enough to read in a failure message.
POPULATION = 3

#: Their ``tiny`` shrunk to test scale — structure preserved, sizes not. The
#: population claim is about the faces, and the faces do not know the widths.
SMALL = dict(
    horizon=3,
    deter=8,
    num_vars=4,
    num_classes=4,
    units=6,
    mlp_units=8,
    embed=8,
    head_units=16,
    head_layers=2,
)

NS, NO, NU = 3, 4, 2  # tabular: states, observation levels, control levels
DIM, NCTRL = 2, 3  # the gradient-slot System's factor width and control levels


# ============================================================== the comparison itself


def _stack(members: list):
    """The population as one pytree: every leaf gains a leading ``(POPULATION,)`` axis."""
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *members)


def _assert_population_matches(batched, members: list, *, what: str, tol: float = 1e-5) -> None:
    """``batched[i] == members[i]`` leaf for leaf — floats ``allclose``, discretes exact.

    See the module docstring for why the split is where it is. The treedefs are
    checked first, because a *structural* difference (a dropped ``None``, a dict key
    that only appears on one path) is a different failure from a numeric one and
    reads nothing like it in an ``allclose`` message.
    """
    want_leaves = [jax.tree_util.tree_leaves(m) for m in members]
    treedefs = {jax.tree_util.tree_structure(m) for m in members}
    assert len(treedefs) == 1, f"{what}: the members disagree on structure: {treedefs}"
    got_leaves = jax.tree_util.tree_leaves(batched)
    assert len(got_leaves) == len(want_leaves[0]), (
        f"{what}: the vmapped output has {len(got_leaves)} leaf/leaves, a member has "
        f"{len(want_leaves[0])}"
    )
    for j, leaf in enumerate(got_leaves):
        leaf = np.asarray(leaf)
        for i in range(len(members)):
            want = np.asarray(want_leaves[i][j])
            assert leaf.shape[1:] == want.shape, (
                f"{what}: leaf {j} is {leaf.shape} batched but {want.shape} per member — "
                "the leading axis is the population's and nothing else may move"
            )
            got = leaf[i]
            if want.dtype == np.bool_ or np.issubdtype(want.dtype, np.integer):
                assert np.array_equal(got, want), (
                    f"{what}: leaf {j}, member {i}: discrete leaves must match EXACTLY "
                    f"(got {got}, want {want})"
                )
            else:
                assert np.allclose(got, want, rtol=tol, atol=tol), (
                    f"{what}: leaf {j}, member {i}: max |Δ| "
                    f"{np.max(np.abs(got - want)):.3e} over tol {tol:.1e}"
                )


# ================================================================ the tabular System


def _prob_table(rng, shape):
    g = rng.gamma(1.0, size=shape).astype(np.float32)
    return jnp.asarray(g / g.sum(axis=0, keepdims=True))


def _near_identity(n: int, eps: float = 1e-3) -> ConditionalCategorical:
    table = np.full((n, n), eps, np.float32) + (1.0 - n * eps) * np.eye(n, dtype=np.float32)
    return ConditionalCategorical(log_table=jnp.log(jnp.asarray(table)), parent_ndims=1)


def _tabular(seed: int = 5, conc: float = 2.0):
    """One factor, a **learnable** Dirichlet likelihood, a controlled transition.

    ``tests/test_learn.py``'s ``_sense_system`` shape: modality ``sense`` is the
    conjugate slot ``learn`` advances, ``ident`` a near-identity point table so the
    posterior is sharp, and ``policy_len=2`` over ``num_controls=(2,)`` gives the
    enumeration planner a real four-policy set to score.
    """
    rng = np.random.default_rng(seed)
    system = System(
        A_deps=((0,), (0,)),
        B_deps=((0,),),
        controlled=(True,),
        policy_len=2,
        modality_names=("sense", "ident"),
        factor_names=("s",),
    )
    params = SystemParams(
        likelihood={
            "sense": ConditionalDirichlet(concentration=jnp.full((NO, NS), conc), parent_ndims=1),
            "ident": _near_identity(NS),
        },
        transition={
            "s": ConditionalCategorical(
                log_table=jnp.log(_prob_table(rng, (NS, NS, NU))), parent_ndims=2
            ),
        },
        initial_state_prior={"s": Categorical(logits=jnp.zeros(NS))},
        obs_targets={"sense": Categorical(logits=jnp.asarray([0.0, 2.0, -1.0, 0.0]))},
    )
    system._validate(params)
    return system, params


def _tabular_observation(i: int) -> dict:
    """Member ``i``'s tick: two int32 scalars, one per modality. Deliberately distinct."""
    return {
        "sense": jnp.asarray((i + 1) % NO, jnp.int32),
        "ident": jnp.asarray(i % NS, jnp.int32),
    }


@pytest.fixture(scope="module")
def tabular_population():
    """``(system, [params × POPULATION])`` whose conjugate posteriors genuinely differ.

    Divergence by construction, not by initialization: every member starts from the
    *same* declaration and then absorbs a different observation, so the Dirichlet
    concentrations that the faces below read are three different tables. A test that
    vmapped three identical members would pass over a face that ignored the batch
    axis entirely.
    """
    system, params = _tabular()
    members = []
    for i in range(POPULATION):
        obs = _tabular_observation(i)
        beliefs, _ = system.infer_states(params, obs)
        members.append(system.learn(params, obs, beliefs, lr=1.0 + i))
    concentrations = [np.asarray(m.likelihood["sense"].concentration) for m in members]
    for i in range(1, POPULATION):
        assert not np.allclose(concentrations[0], concentrations[i]), (
            "the members must differ before anything is claimed about batching them"
        )
    return system, members


# ------------------------------------------------------------------- infer_states


def test_infer_states_vmaps_over_a_population(tabular_population):
    """The perception face, three agents at once, against three sequential calls."""
    system, members = tabular_population
    observations = [_tabular_observation(i) for i in range(POPULATION)]

    def one(p, obs):
        return system.infer_states(p, obs)[0]

    batched = eqx.filter_jit(jax.vmap(one))(_stack(members), _stack(observations))
    sequential = [one(p, obs) for p, obs in zip(members, observations, strict=True)]
    _assert_population_matches(batched, sequential, what="infer_states/tabular")


# -------------------------------------------------------------------------- learn


def test_learn_vmaps_conjugate_slots_with_a_per_member_rate(tabular_population):
    """The conjugate ``learn``, with the rate itself batched — the docstring's own case.

    ``system/core.py``: "Batching follows the library-wide convention: ``learn`` is
    unbatched and lifts via ``vmap`` (**per-agent rates included, as traced
    arrays**)." A traced rate is the part worth pinning: a *static* Python ``0.0``
    skips a slot at trace time, so under ``vmap`` a per-member rate must be an array
    and the skip must not happen — which is exactly what makes three different rates
    expressible in one call.
    """
    system, members = tabular_population
    observations = [_tabular_observation(i) for i in range(POPULATION)]
    rates = [jnp.asarray(0.25 * (i + 1), jnp.float32) for i in range(POPULATION)]

    def one(p, obs, lr):
        beliefs, _ = system.infer_states(p, obs)
        return system.learn(p, obs, beliefs, lr=lr)

    batched = eqx.filter_jit(jax.vmap(one))(_stack(members), _stack(observations), jnp.stack(rates))
    sequential = [one(p, obs, lr) for p, obs, lr in zip(members, observations, rates, strict=True)]
    _assert_population_matches(batched, sequential, what="learn/conjugate")
    #  and the rates really were different: member 2 moved further than member 0
    moved = [
        float(
            np.max(
                np.abs(
                    np.asarray(out.likelihood["sense"].concentration)
                    - np.asarray(members[i].likelihood["sense"].concentration)
                )
            )
        )
        for i, out in enumerate(sequential)
    ]
    assert moved[0] < moved[-1], moved


def _gradient_system(**system_kwargs):
    """One continuous factor with a **neural** transition: the gradient-slot path.

    ``tests/test_mlp_conditional.py``'s ``_mlp_transition_system``, which is where
    ``learn`` returns the ``(params, LearnState)`` pair rather than bare params —
    so this is the test that the *optimizer carry* batches too, adam's moments
    included.
    """
    system = System(
        A_deps=(),
        B_deps=((),),
        controlled=(True,),
        policy_len=1,
        factor_names=("x",),
        epistemic=False,
        **system_kwargs,
    )
    params = SystemParams(
        likelihood={},
        transition={
            "x": MLPConditional.from_shapes(
                (DIM, NCTRL), DIM, key=jax.random.PRNGKey(7), width=8, depth=1
            ),
        },
        initial_state_prior={"x": Gaussian(loc=jnp.zeros(DIM), chol=jnp.eye(DIM))},
    )
    return system, params


def _point(x, sd: float = 1e-3) -> Gaussian:
    x = jnp.asarray(x, jnp.float32)
    return Gaussian(loc=x, chol=sd * jnp.eye(x.shape[-1]))


def test_learn_vmaps_gradient_slots_and_their_optimizer_carry():
    """The gradient ``learn``: divergent nets, divergent adam moments, one ``vmap``.

    Two steps per member so the carry is doing real work — a first step's moments
    are still the initialization's and would batch correctly by accident. The
    members diverge because each is handed a different transition target, which is
    what makes their nets *and* their moments three different pytrees by the time
    the batched call reads them.
    """
    system, params = _gradient_system()

    def step(p, ls, prev, cur, action):
        return system.learn(
            p,
            {},
            {"x": cur},
            prev_beliefs={"x": prev},
            action=system.from_positional("actions", action),
            lr=0.05,
            learn_state=ls,
        )

    targets = [
        (_point([0.3 * (i + 1), -0.2]), _point([0.25, -0.1 * (i + 1)]), jnp.asarray([i % NCTRL]))
        for i in range(POPULATION)
    ]
    members = []
    for prev, cur, action in targets:
        p, ls = params, system.init_learn_state(params)
        for _ in range(2):
            p, ls = step(p, ls, prev, cur, action)
        members.append((p, ls))
    weights = [np.asarray(p.transition["x"].layers[0].weight) for p, _ in members]
    for i in range(1, POPULATION):
        assert not np.allclose(weights[0], weights[i])

    def one(pair, prev, cur, action):
        p, ls = pair
        return step(p, ls, prev, cur, action)

    batched = eqx.filter_jit(jax.vmap(one))(
        _stack(members),
        _stack([t[0] for t in targets]),
        _stack([t[1] for t in targets]),
        _stack([t[2] for t in targets]),
    )
    sequential = [one(pair, *t) for pair, t in zip(members, targets, strict=True)]
    _assert_population_matches(batched, sequential, what="learn/gradient+LearnState")
    #  the carry's addresses are the declaration's, per member and after batching
    assert set(batched[1].opt_states) == {"transition:x"}


# ---------------------------------------------------------------------------- act


def test_act_vmaps_under_a_search_planner(tabular_population):
    """``act`` with the enumeration planner: the search itself is inside the ``vmap``.

    The planner already ``vmap``s over its own policy set
    (``planning/enumerate.py``), so this nests one ``vmap`` outside another — the
    case where a face that read a batch shape out of an array would give itself
    away. ``selection="deterministic"`` so the action is the argmax rather than a
    draw: the *level index* is then a discrete leaf and is asserted **exactly**,
    which is the sharper claim (a stochastic draw would only ever be compared
    allclose on its logits).
    """
    system, members = tabular_population
    observations = [_tabular_observation(i) for i in range(POPULATION)]

    def one(p, obs, key):
        beliefs, _ = system.infer_states(p, obs)
        action, q_pi, _ = system.act(key, p, beliefs, selection="deterministic", metrics=False)
        return action, q_pi

    keys = jax.random.split(jax.random.PRNGKey(3), POPULATION)
    batched = eqx.filter_jit(jax.vmap(one))(_stack(members), _stack(observations), keys)
    sequential = [one(p, obs, k) for p, obs, k in zip(members, observations, keys, strict=True)]
    _assert_population_matches(batched, sequential, what="act/enumeration")
    positional = system.to_positional("actions", batched[0])
    assert positional.shape == (POPULATION, 1) and positional.dtype == jnp.int32


# --------------------------------------------------------- update_empirical_prior


def test_update_empirical_prior_vmaps(tabular_population):
    """The carry face: beliefs pushed through the chosen action, three at a time.

    The one face whose output *is* the next tick's input, so a batching error here
    would not be a wrong number — it would be a population that silently shares a
    carry.
    """
    system, members = tabular_population
    observations = [_tabular_observation(i) for i in range(POPULATION)]
    actions = [jnp.asarray([i % NU], jnp.int32) for i in range(POPULATION)]

    def one(p, obs, action):
        beliefs, _ = system.infer_states(p, obs)
        return system.update_empirical_prior(p, beliefs, system.from_positional("actions", action))

    batched = eqx.filter_jit(jax.vmap(one))(_stack(members), _stack(observations), _stack(actions))
    sequential = [one(p, obs, a) for p, obs, a in zip(members, observations, actions, strict=True)]
    _assert_population_matches(batched, sequential, what="update_empirical_prior/tabular")


def test_the_whole_canonical_loop_vmaps_as_one_scan(tabular_population):
    """Four ticks of perceive → learn → act → carry, ``vmap``-ed and jitted together.

    The faces pass individually above; this is the claim that *composing* them does
    too, which is the only form the claim is used in — a population is a training
    loop, not a face call.
    """
    system, members = tabular_population
    streams = [
        [_tabular_observation((i + t) % POPULATION) for t in range(4)] for i in range(POPULATION)
    ]

    def one(p, stream, key):
        priors, prev, prev_action = None, None, None
        for t in range(4):
            obs = {"sense": stream[0][t], "ident": stream[1][t]}
            beliefs, _ = system.infer_states(p, obs, priors)
            p = system.learn(p, obs, beliefs, prev_beliefs=prev, action=prev_action, lr=0.5)
            action, _, _ = system.act(
                jax.random.fold_in(key, t), p, beliefs, selection="deterministic", metrics=False
            )
            priors = system.update_empirical_prior(p, beliefs, action)
            prev, prev_action = beliefs, action
        return p, priors

    packed = [
        (jnp.stack([o["sense"] for o in stream]), jnp.stack([o["ident"] for o in stream]))
        for stream in streams
    ]
    keys = jax.random.split(jax.random.PRNGKey(11), POPULATION)
    batched = eqx.filter_jit(jax.vmap(one))(_stack(members), _stack(packed), keys)
    sequential = [one(p, s, k) for p, s, k in zip(members, packed, keys, strict=True)]
    _assert_population_matches(batched, sequential, what="canonical loop/tabular")


# ============================================================== the amortized agent


@pytest.fixture(scope="module")
def rssm_population():
    """``POPULATION`` R-AIF agents at test sizes — one static System, three params.

    Distinct initialization keys, so the parameters differ everywhere at once
    (world model, actor, critic, ensemble, CRSPP). This is exactly what
    :mod:`hybrid_aif.reference.raif.run` holds for four training seeds; the fixture
    IS a four-seed run's state with three seeds.
    """
    agents = [raif_mountain_car(jax.random.PRNGKey(100 + i), **SMALL) for i in range(POPULATION)]
    system = agents[0].system
    for i in range(1, POPULATION):
        assert not np.allclose(
            np.asarray(agents[0].params.policy_prior.layers[0].weight),
            np.asarray(agents[i].params.policy_prior.layers[0].weight),
        )
    return system, agents


def test_the_amortized_recognition_and_carry_faces_vmap(rssm_population):
    """``infer_states`` + ``update_empirical_prior`` on the declared RSSM, batched.

    The amortized path is the one where the faces dispatch through
    ``_amortized_structure`` and read declaration shapes, so it is the path most
    likely to notice a batch axis. It does not.
    """
    system, agents = rssm_population
    observations = [jnp.asarray([-0.5 + 0.1 * i, 0.01 * i], jnp.float32) for i in range(POPULATION)]
    actions = [jnp.asarray([0.3 * i - 0.3], jnp.float32) for i in range(POPULATION)]

    def one(p, obs, action):
        observed = system.from_positional("observations", (obs, None, None))
        beliefs, _ = system.infer_states(p, observed, system.initial_priors(p))
        return beliefs, system.update_empirical_prior(
            p, beliefs, system.from_positional("actions", action, p)
        )

    params = _stack([a.params for a in agents])
    batched = eqx.filter_jit(jax.vmap(one))(params, _stack(observations), _stack(actions))
    sequential = [
        one(a.params, obs, act_) for a, obs, act_ in zip(agents, observations, actions, strict=True)
    ]
    _assert_population_matches(batched, sequential, what="infer_states+carry/amortized")


def test_act_vmaps_under_the_habit_planner(rssm_population):
    """``act`` from the E-designated actor (A12 item 3), three agents at once.

    The continuous case: the action is a *draw* from the actor's Gaussian, so it is
    a float leaf and compared ``allclose`` — the discrete-exactness claim above has
    nothing to say here, and the docstring's split says so.
    """
    system, agents = rssm_population
    observations = [jnp.asarray([-0.5 + 0.1 * i, 0.01 * i], jnp.float32) for i in range(POPULATION)]

    def one(agent_params, obs, key):
        agent = agents[0]._replace(params=agent_params)
        beliefs, _ = system.infer_states(
            agent_params,
            system.from_positional("observations", (obs, None, None)),
            system.initial_priors(agent_params),
        )
        return system.to_positional("actions", act(agent, beliefs, key), agent_params)

    keys = jax.random.split(jax.random.PRNGKey(5), POPULATION)
    batched = eqx.filter_jit(jax.vmap(one))(
        _stack([a.params for a in agents]), _stack(observations), keys
    )
    sequential = [
        one(a.params, obs, k) for a, obs, k in zip(agents, observations, keys, strict=True)
    ]
    _assert_population_matches(batched, sequential, what="act/habit")
    assert batched.shape == (POPULATION, 1)


def test_act_vmaps_under_the_cem_search_planner(rssm_population):
    """``act`` with CEM over the same agents — a *search* planner on the amortized path.

    Declared by two ``dataclasses.replace``s, which is the point: swapping the
    planner is a declaration change and the population axis is untouched by it. The
    second replace drops ``policy_prior`` because CEM refuses it by design — A3's
    ``e_proposal`` mode (E as the *proposal distribution* of a sampled planner) is
    P2 and deliberately unbuilt — so the search config here is the G-scoring one,
    which is what the goal-(3) slider prod will be.

    Compared at a **looser** tolerance than the rest of this file, and the reason is
    worth stating rather than burying in a number. CEM's inner loop is
    ``top_k`` over sampled candidate scores — a *discrete* selection over float
    quantities — so a last-bit difference in a score can promote a different elite
    and move the returned plan by a visible amount. That is CEM being stochastic
    search, not batching being wrong. The leaf-for-leaf claim this file makes is
    therefore asserted on the deterministic planners; here the assertion is that the
    two paths agree to search noise and that the population axis is exactly the
    leading one.
    """
    system, agents = rssm_population
    searching = dataclasses.replace(system, planner="cem")
    searched = [dataclasses.replace(a.params, policy_prior=None) for a in agents]
    observations = [jnp.asarray([-0.5 + 0.1 * i, 0.01 * i], jnp.float32) for i in range(POPULATION)]

    def one(p, obs, key):
        observed = searching.from_positional("observations", (obs, None, None))
        beliefs, _ = searching.infer_states(p, observed, searching.initial_priors(p))
        act_key, plan_key = jax.random.split(key)
        action, _, _ = searching.act(act_key, p, beliefs, plan_key=plan_key, metrics=False)
        return searching.to_positional("actions", action, p)

    keys = jax.random.split(jax.random.PRNGKey(9), POPULATION)
    batched = eqx.filter_jit(jax.vmap(one))(_stack(searched), _stack(observations), keys)
    sequential = [one(p, obs, k) for p, obs, k in zip(searched, observations, keys, strict=True)]
    assert batched.shape == (POPULATION, 1)
    assert bool(jnp.all(jnp.isfinite(batched)))
    _assert_population_matches(batched, sequential, what="act/cem", tol=2e-2)


# ==================================================================== replay, batched


@pytest.fixture(scope="module")
def episodes():
    """Three real Mountain Car episodes, labelled — one per population member."""
    env = MountainCar(time_limit=24)
    return [
        label(collect_episode(env, bang_bang, jax.random.PRNGKey(40 + i), can_self_imitate=True))
        for i in range(POPULATION)
    ]


def test_static_buffer_add_and_sample_vmap(episodes):
    """``add`` and ``sample`` under ``vmap``: the replay is part of the batched state.

    :class:`~hybrid_aif.train.buffers.StaticBuffer` exists so that ``sample`` is a
    pure function of a key (module docstring: "hence a training step that samples
    inside ``lax.scan``"). The population claim needs one thing more — that the
    *buffer itself* rides the batch axis, so four seeds hold four independent
    replays inside one pytree. ``size``, ``cursor`` and every flag column are
    integer or boolean leaves, so they are asserted **exactly**: a population whose
    ring cursors drifted would be a population sharing data.
    """
    empty = StaticBuffer.empty(6, 16, 2, 1, extras=EXTRAS)
    members = [empty] * POPULATION

    def one(buffer, ep, key):
        buffer = add(buffer, ep)
        buffer = add(buffer, ep)  # twice, so cursor and size are past their init
        return buffer, sample(buffer, key, 3)

    keys = jax.random.split(jax.random.PRNGKey(13), POPULATION)
    batched = eqx.filter_jit(jax.vmap(one))(_stack(members), _stack(episodes), keys)
    sequential = [one(b, ep, k) for b, ep, k in zip(members, episodes, keys, strict=True)]
    _assert_population_matches(batched, sequential, what="StaticBuffer add/sample")
    assert np.array_equal(np.asarray(batched[0].size), np.full(POPULATION, 2))
    #  and the members are genuinely different data, not three copies of one episode
    drawn = np.asarray(batched[1].obs)
    assert not np.allclose(drawn[0], drawn[1])


def test_a_population_shares_the_declaration_and_nothing_else(tabular_population, rssm_population):
    """The claim's negative half: one **static** System, and no numeric leaf in it.

    :class:`~hybrid_aif.system.System` is all-``static`` ``eqx.field``s, so it has
    no array leaves at all — which is *why* it is closed over rather than mapped,
    and why a population cannot accidentally share a parameter: everything numeric
    is in :class:`~hybrid_aif.system.SystemParams` and the carries, and every one of
    those is on the batch axis in the tests above. Stated as a test because it is
    the load-bearing structural fact, not a stylistic one.
    """
    for system, _ in (tabular_population, rssm_population):
        assert jax.tree_util.tree_leaves(system) == [], (
            "System must hold no numeric leaf: a population closes over it, so any "
            "array in it would be silently shared across members"
        )
    system, agents = rssm_population
    assert len(jax.tree_util.tree_leaves(agents[0].params)) > 0
    assert len(jax.tree_util.tree_leaves(agents[0].learn_state)) > 0
