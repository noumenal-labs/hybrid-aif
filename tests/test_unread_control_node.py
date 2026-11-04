"""A control node no factor's transition reads, sized by another declaration that names it.

The node may be named by a conditional per-step target (``C_deps``), by a modality that observes
its action node, or by a per-level ``durations`` entry; each gives its level count, and the
sources must agree. The planner reads sizes only, so the model scores as the same model with a
null parent (a constant axis) added to a transition.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Categorical, ConditionalCategorical
from hybrid_aif.composition.merge import ActionControl, merge, subsystem
from hybrid_aif.controls import Box
from hybrid_aif.description import build, describe
from hybrid_aif.objectives import total
from hybrid_aif.system import System, SystemParams

LEVELS = 2
TRANSITION = np.asarray([[0.9, 0.2], [0.1, 0.8]])
LIKELIHOOD = np.asarray([[0.8, 0.3], [0.2, 0.7]])
TARGET = np.asarray([[0.0, 1.0], [2.0, -1.0]])  # (child level, node level)
OBSERVED = np.asarray([[[0.8, 0.5], [0.3, 0.1]], [[0.2, 0.5], [0.7, 0.9]]])  # (obs, s, a)
PREFERENCE = Categorical(logits=jnp.asarray([0.0, 1.5]))


def _cc(table, parent_ndims):
    return ConditionalCategorical(
        log_table=jnp.log(jnp.clip(jnp.asarray(table, jnp.float32), 1e-12)),
        parent_ndims=parent_ndims,
    )


def model(source, *, null_parent=False, durations=None, target_levels=LEVELS):
    """A two-level state ``s`` and a node ``u_opt`` that no transition reads (or, with
    ``null_parent``, that ``s`` reads through a constant axis), named by ``source``."""
    sensed = source == "sensed"
    c_deps = (("s", ("u_opt",)),) if source == "target" else ()
    system = System(
        A_deps=((0, 2),) if sensed else ((0,),),
        B_deps=((0, 1), ()) if null_parent else ((0,), ()),
        controls=(1,),
        action_names=("a_opt",),
        factor_names=("s", "u_opt"),
        modality_names=("obs",),
        policy_len=2,
        planning_belief="posterior",
        C_deps=c_deps,
    )
    transition = np.stack([TRANSITION] * LEVELS, -1) if null_parent else TRANSITION
    state_targets = {}
    if source == "target":
        table = np.tile(TARGET[:, :1], (1, target_levels))
        table[:, :LEVELS] = TARGET[:, :target_levels]
        state_targets = {"s": ConditionalCategorical(jnp.asarray(table, jnp.float32), 1)}
    params = SystemParams(
        likelihood={"obs": _cc(OBSERVED if sensed else LIKELIHOOD, 2 if sensed else 1)},
        transition={"s": _cc(transition, 2 if null_parent else 1)},
        initial_state_prior={"s": Categorical.from_probs(jnp.asarray([0.6, 0.4]))},
        obs_targets={"obs": PREFERENCE},
        state_targets=state_targets,
        durations={} if durations is None else {"u_opt": jnp.asarray(durations)},
    )
    return system, params


def _scores(system, params):
    q = system.initial_priors(params)
    tree = system.score_policies(params, q, system.policies(params))
    return np.asarray(total(tree).lower)


@pytest.mark.parametrize("source", ["target", "sensed"])
def test_a_node_named_only_by_a_target_or_a_modality_scores_as_with_a_null_parent(source):
    system, params = model(source)
    assert system._table.readers == ((),)
    assert system._space(params).sizes == (LEVELS,)
    assert system.policies(params).shape == (LEVELS**2, 2, 1)
    reference = model(source, null_parent=True)
    np.testing.assert_allclose(_scores(system, params), _scores(*reference), rtol=1e-6)


def test_an_unread_node_plans_infers_and_acts():
    system, params = model("target")
    q, _ = system.infer_states(params, {"obs": jnp.asarray(1)})
    ref_system, ref_params = model("target", null_parent=True)
    ref_q, _ = ref_system.infer_states(ref_params, {"obs": jnp.asarray(1)})
    np.testing.assert_allclose(q["s"].probs, ref_q["s"].probs, rtol=1e-6)
    action, _, _ = system.act(jnp.asarray([0, 0], jnp.uint32), params, q, metrics=False)
    assert set(action) == {"a_opt"}


def test_a_node_sized_only_by_durations():
    system, params = model("durations", durations=[1.0, 2.0])
    assert system._table.readers == ((),)
    assert system._space(params).sizes == (2,)
    assert system.policies(params).shape == (2**2, 2, 1)
    _scores(system, params)
    assert float(system.elapsed({"a_opt": jnp.asarray(1)}, params)) == 2.0


def test_a_scalar_duration_gives_no_size():
    system, params = model("durations", durations=1.0)
    with pytest.raises(ValueError, match="read by no factor and nothing else gives its level"):
        system._space(params)


def test_sources_that_disagree_are_refused():
    system, params = model("target", durations=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match=r"read by no factor and sized differently.*durations"):
        system._space(params)


def test_a_transition_reader_sizes_the_node_and_a_target_that_disagrees_is_named():
    #  a reader's axis is the size; the target is checked against it where targets are
    system, params = model("target", null_parent=True, target_levels=3)
    assert system._space(params).sizes == (2,)
    with pytest.raises(ValueError, match=r"parent axes \(3,\) but its C_deps parent"):
        _scores(system, params)


def test_a_node_nothing_sizes_is_refused_with_the_ways_to_size_it():
    system, params = model("none")
    pattern = (
        r"control node 'u_opt' is read by no factor and nothing else gives its level count.*"
        r"C_deps target.*observing its action node.*durations"
    )
    with pytest.raises(ValueError, match=pattern):
        system._space(params)


def test_a_composite_containing_an_unread_node_merges_and_describes():
    system, params = model("target")
    other, other_params = model("durations", durations=[1.0, 2.0])
    cs, cp = merge({"a": system, "b": other}, {"a": params, "b": other_params})
    assert cs._space(cp).sizes == (LEVELS, LEVELS)
    document, arrays = describe(cs, cp)
    rebuilt, rebuilt_params = build(document, arrays=arrays)
    assert rebuilt._space(rebuilt_params).sizes == (LEVELS, LEVELS)
    view, view_params = subsystem(cs, cp, "a")
    assert view._table.readers == ((),)
    np.testing.assert_allclose(_scores(view, view_params), _scores(system, params), rtol=1e-6)


def test_identifying_an_unread_node_with_a_driver_keeps_the_target_on_the_driver():
    system, params = model("target")
    driver, driver_params = model("durations", null_parent=True, durations=[1.0, 2.0])
    cs, cp = merge(
        {"a": system, "b": driver},
        {"a": params, "b": driver_params},
        [ActionControl("b.u_opt", "a.u_opt")],
    )
    assert ("a.s", ("b.u_opt",)) in cs.C_deps
    assert cs._space(cp).sizes == (LEVELS,)


def test_a_node_named_only_by_a_modality_round_trips_through_a_description():
    system, params = model("sensed")
    document, arrays = describe(system, params)
    rebuilt, rebuilt_params = build(document, arrays=arrays)
    assert rebuilt._space(rebuilt_params).sizes == (LEVELS,)
    assert rebuilt._table.readers == ((),)


def _sugar_named(unread: bool):
    """A node named ``u_s`` after the factor ``s``, which reads it when ``unread`` is false and
    is sized by a C_deps target when it does not."""
    system = System(
        A_deps=((0,),),
        B_deps=((0, 1), ()) if not unread else ((0,), ()),
        controls=(1,),
        action_names=("a_s",),
        factor_names=("s", "u_s"),
        modality_names=("obs",),
        policy_len=2,
        planning_belief="posterior",
        C_deps=(("s", ("u_s",)),),
    )
    transition = np.stack([TRANSITION] * LEVELS, -1) if not unread else TRANSITION
    params = SystemParams(
        likelihood={"obs": _cc(LIKELIHOOD, 1)},
        transition={"s": _cc(transition, 1 if unread else 2)},
        initial_state_prior={"s": Categorical.from_probs(jnp.asarray([0.6, 0.4]))},
        obs_targets={"obs": PREFERENCE},
        state_targets={"s": ConditionalCategorical(jnp.asarray(TARGET, jnp.float32), 1)},
    )
    return system, params


def test_an_unread_node_named_after_a_factor_is_a_node_of_its_own():
    system, params = _sugar_named(unread=True)
    assert system._table.sugar is None and system._table.readers == ((),)
    reference = _sugar_named(unread=False)
    np.testing.assert_allclose(_scores(system, params), _scores(*reference), rtol=1e-6)
    q, _ = system.infer_states(params, {"obs": jnp.asarray(1)})
    action, _, _ = system.act(jnp.asarray([0, 0], jnp.uint32), params, q, metrics=False)
    assert set(action) == {"a_s"}
    env, env_params = system.as_env(params)
    _, state = env.reset(jax.random.PRNGKey(0), env_params)
    env.step(jax.random.PRNGKey(1), state, {"a_s": jnp.asarray(1)}, env_params)


def test_a_sensor_node_read_only_through_a_levelled_noise_map_is_sized_by_the_map():
    from hybrid_aif.distributions import ConditionalGaussian
    from tests.test_noise_map_mixture import _levelled, _showcase

    system, params = _showcase(_levelled())
    innovations = ConditionalGaussian(
        loc=jnp.zeros((2, 4)), chol=jnp.broadcast_to(0.05 * jnp.eye(4), (2, 4, 4)), parent_ndims=1
    )
    unread = dataclasses.replace(system, B_deps=((0,), (2, 0), (), ()))
    unread_params = dataclasses.replace(
        params,
        transition={
            "terrain": params.transition["terrain"],
            "x": dataclasses.replace(params.transition["x"], innovations=innovations),
        },
    )
    space = unread._space(unread_params)
    assert space.sizes == (2, 2)
    mixed = dataclasses.replace(unread, planner="mixed", num_iters=3)
    q_pi, _ = mixed.infer_policies(unread_params, mixed.initial_priors(unread_params))
    assert np.isfinite(np.asarray(q_pi.probs)).all()


def test_a_per_level_mniw_channel_sizes_its_node():
    from hybrid_aif.controls import _observation_axes
    from hybrid_aif.factors import LinearGaussianObservationMNIW

    head = jnp.zeros((3, 2, 4))
    mnw = LinearGaussianObservationMNIW.from_head_noise(
        head, jnp.broadcast_to(jnp.eye(2), (3, 2, 2))
    )
    assert _observation_axes(mnw, 2) == (None, 3)


def test_a_boxed_continuous_node_nothing_reads_is_refused():
    system = System(
        A_deps=((0,),),
        B_deps=((0,), ()),
        controls=(1,),
        action_names=("a_opt",),
        factor_names=("s", "u_opt"),
        modality_names=("obs",),
        policy_len=2,
    )
    params = SystemParams(
        likelihood={"obs": _cc(LIKELIHOOD, 1)},
        transition={"s": _cc(TRANSITION, 1)},
        initial_state_prior={"s": Categorical.from_probs(jnp.asarray([0.6, 0.4]))},
        obs_targets={"obs": PREFERENCE},
        control={"u_opt": Box(low=-jnp.ones(1), high=jnp.ones(1))},
    )
    pattern = (
        r"control node 'u_opt' has a box in params.control but no factor reads it.*drop its box"
    )
    with pytest.raises(ValueError, match=pattern):
        system._space(params)


def test_a_description_build_counts_only_a_per_level_duration_as_naming_a_node():
    system, params = model("durations", durations=[1.0, 2.0])
    document, arrays = describe(system, params)
    build(document, arrays=arrays)  # a per-level entry names the node
    document["variables"]["u_opt"]["duration"] = 1.0
    with pytest.raises(ValueError, match=r"'u_opt' is read by no transition.*per-level duration"):
        build(document, arrays=arrays)


@pytest.mark.parametrize(
    ("controlled", "deps"),
    [((True, True), ((1,), (0, 1))), ((True, False), ((), (0,))), ((True, True), ((0, 1), (1,)))],
    ids=["swapped", "crossed", "shared"],
)
def test_a_cross_wired_sugar_declaration_keeps_its_sugar_view(controlled, deps):
    #  every node is read by some factor, so the pymdp-style view stands and a per-factor
    #  positional action keeps control k in column k
    system = System(B_deps=((), ()), controlled=controlled, B_control_deps=deps, policy_len=1)
    sugar = system._table.sugar
    assert sugar is not None and sugar.B_control_deps == deps
    owners = tuple(f for f, c in enumerate(controlled) if c)
    assert system._names().controlled_factors == owners
