"""The one ``epistemic`` selector and its key grammar (design:
``per-variable-infogain.md``, build steps 1-3).

``System._term_targets`` is the single resolver: one per-variable selector (V1's
namespace, V2's shape) over the union of bare state names and slot-qualified
parameter keys, and ``lr_by_slot``/``decay_by_slot`` share the optional third key
segment (V1's multiplicity rule, closing the v0.3 rate-granularity gap). The
tests here bind three things:

- **the resolution** — ``True`` mounts every eligible variable, ``False``/``()``
  is the pragmatic agent (V5.2), and a tuple mounts exactly what it names;
- **the namespace** — bare state names, slot-qualified parameter keys, one
  sub-key per *independent posterior block* (two for an LG carrying both a noise
  and a head posterior, none for MNIW's joint one), with V5.3's loud errors;
- **the rate granularity** — a sub-block key moves its own block and leaves the
  sibling posterior bitwise untouched.

Term *leaves* stay coarse where the registry is coarse: a block-qualified key
resolves to its declaration's index (there is no per-block novelty estimator
yet), which the ``_epistemic_variables`` records make explicit.

Step 3 adds the last two pieces of the grammar, bound in the final two sections:

- **observation variables** (V8) — ``obs:<modality>`` mounts one channel's
  state-epistemic leaf, the granularity no bare state name can express;
- **segment wildcards** (V7) — ``*`` matches one whole segment, in ``epistemic``
  and in the rate mappings alike, with specificity = literal-segment count
  deciding rate precedence (and ties raising).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import InverseGamma, MatrixNormal
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianObservationMNIW
from tests.test_learn import _d_slot_system, _multi_modality_model, _sense_system
from tests.test_lg_head_learning import _mniw_system_and_params
from tests.test_mixture_filtering import make_params as mixture_params
from tests.test_mixture_filtering import make_system as mixture_system
from tests.test_switching_observation import make_switch_params, make_switch_system


def _discrete():
    """Two modalities over one factor 's'; 'sense' learnable, 'ident' a point table."""
    return _sense_system()


def _hybrid():
    """mode(categorical) + x(mixture), LG modality 'y' observing x."""
    return mixture_system(), mixture_params()


# --------------------------------------------------------- the default: True, False


@pytest.mark.parametrize(
    ("build", "expected"),
    [(_discrete, ((0, 1), (0,), ())), (_hybrid, ((0,), (), ()))],
    ids=["discrete", "hybrid"],
)
def test_default_true_mounts_every_eligible_variable(build, expected):
    """``epistemic=True`` (the default) mounts every eligible variable and
    ``False``/``()`` mounts none — the pragmatic-agent ablation (V5.2).

    The hybrid case shows the one-estimator-per-variable rule (V2, settled in
    build step 2): ``x`` is observed by the eligible LG channel ``y``, so it is
    scored through that channel, and a continuous factor carries no leaf of its
    own (V11: its entropy is the opt-in ``uncertainty`` row, not information gain).
    """
    system, params = build()
    assert system.epistemic is True
    assert system._term_targets(params) == expected
    for sel in (False, ()):
        assert dataclasses.replace(system, epistemic=sel)._term_targets(params) == ((), (), ())


def test_epistemic_must_be_a_bool_or_a_tuple():
    """A selector, not a switch: anything else is loud at construction (V2)."""
    system, _ = _discrete()
    with pytest.raises(ValueError, match="must be True, False, or a tuple"):
        dataclasses.replace(system, epistemic="s")


def test_named_variables_resolve_per_kind():
    """A bare state name and a parameter key select independently."""
    system, params = _discrete()
    named = dataclasses.replace(system, epistemic=("s",))
    assert named._term_targets(params) == ((0, 1), (), ())  # both channels observe 's'
    no_novelty = dataclasses.replace(system, epistemic=("s",))
    assert no_novelty._term_targets(params)[1:] == ((), ())


# ------------------------------------------------------------- the namespace (V1)


def test_state_variables_are_bare_and_parameter_variables_slot_qualified():
    system, params = _discrete()
    variables = system._epistemic_variables(params)
    assert set(variables) == {"s", "obs:sense", "obs:ident", "likelihood:sense"}
    assert variables["s"].kind == "state" and variables["s"].index == 0
    assert variables["likelihood:sense"] == ("param", 0, "likelihood", None)
    # V8: observation variables are in the namespace regardless of learnability —
    # 'ident' is a point table, but it is still a variable one can observe through
    assert variables["obs:ident"] == ("obs", 1, None, None)


def _two_block_lg():
    """An LG modality carrying BOTH posteriors: two independent blocks, two variables."""
    a = LinearGaussianObservation(
        head=MatrixNormal.from_head(jnp.asarray([[1.5, 0.3]]), strength=2.0),
        noise=InverseGamma.from_R_chol(jnp.asarray([[0.5]]), strength=2.0),
    )
    return _mniw_system_and_params(a)


def _noise_only_lg():
    a = LinearGaussianObservation(
        H=jnp.asarray([[1.0]]),
        c=jnp.zeros(1),
        noise=InverseGamma.from_R_chol(jnp.asarray([[0.5]]), strength=2.0),
    )
    return _mniw_system_and_params(a)


def _mniw_lg():
    m = LinearGaussianObservationMNIW.from_head_noise(
        jnp.asarray([[1.0, 0.0]]), jnp.asarray([[0.5]]), strength=2.0
    )
    return _mniw_system_and_params(m)


def test_multiplicity_rule_follows_the_posterior_factorization():
    """V1: one variable per *independent block* of the declaration's parameter
    posterior. Two learnable blocks give two sub-keys; one gives the single
    matching sub-key (aliasing the group); MNIW's joint posterior gives none."""
    both_s, both_p = _two_block_lg()
    assert set(both_s._epistemic_variables(both_p)) == {
        "x",
        "x:dim0",
        "obs:y",
        "likelihood:y",
        "likelihood:y:noise",
        "likelihood:y:head",
    }
    noise_s, noise_p = _noise_only_lg()
    assert set(noise_s._epistemic_variables(noise_p)) == {
        "x",
        "x:dim0",
        "obs:y",
        "likelihood:y",
        "likelihood:y:noise",
    }
    mniw_s, mniw_p = _mniw_lg()
    assert set(mniw_s._epistemic_variables(mniw_p)) == {"x", "x:dim0", "obs:y", "likelihood:y"}


def test_block_keys_resolve_to_their_declaration_until_step_2():
    """The group key and its sub-keys share slot and declaration index: the term
    tree has no sub-block leaves yet, so selection is coarse while *addressing*
    is already fine-grained (rates use it today)."""
    system, params = _two_block_lg()
    variables = system._epistemic_variables(params)
    assert variables["likelihood:y"] == ("param", 0, "likelihood", None)
    assert variables["likelihood:y:noise"] == ("param", 0, "likelihood", "noise")
    assert variables["likelihood:y:head"] == ("param", 0, "likelihood", "head")


# -------------------------------------------------- state variables mount modalities


def test_bare_state_name_mounts_every_modality_involving_it():
    """V6 part 1: a state variable mounts every eligible channel whose dependency
    set involves it — deliberately not a partition, so a two-parent modality is
    mounted by *either* parent's name."""
    system, params = _multi_modality_model()[:2]
    assert system.A_deps == ((0, 1), (1,))  # modality 0 has two parents

    def resolve(sel):
        return dataclasses.replace(system, epistemic=sel)._term_targets(params)

    assert resolve(("factor_0",)) == ((0,), (), ())
    assert resolve(("factor_1",)) == ((0, 1), (), ())
    assert resolve(("factor_0", "factor_1")) == ((0, 1), (), ())
    # naming states only leaves the parameter tuples empty, though both A slots
    # are learnable — the selector separates the two variable kinds
    assert resolve(("factor_0",))[1:] == ((), ())


def test_continuous_state_name_mounts_one_estimator():
    """With an eligible LG channel on ``x``, the channel leaf is the one that mounts
    (V2/V4); the switching modality observes both variables, so either name mounts
    it."""
    system, params = make_switch_system(), make_switch_params()
    assert system.A_deps == ((1, 0),)  # y observes x (continuous) gated by mode

    def resolve(sel):
        return dataclasses.replace(system, epistemic=sel)._term_targets(params)

    assert resolve(("x",)) == ((0,), (), ())
    assert resolve(("mode",)) == ((0,), (), ())


def test_unchannelled_continuous_variable_mounts_nothing_and_naming_it_raises():
    """The mirror: no eligible channel over ``x``, so there is no information about it to
    gain and nothing mounts (V11 decision 4). ``True`` and a wildcard pass over it; a literal
    key raises, naming the ``uncertainty`` row that carries the drive the old entropy block
    gave, so a selection never silently loses the leaf it asked for."""
    from tests.test_hybrid_switching_system import initial_beliefs, make_params, make_system

    system, params = make_system(), make_params()
    del initial_beliefs
    assert system.A_deps == ()  # open-loop model: nothing observes x
    assert system._term_targets(params) == ((), (), ())
    assert dataclasses.replace(system, epistemic=("*",))._term_targets(params) == ((), (), ())
    for key in ("x", "x:dim0"):
        with pytest.raises(ValueError, match=r'terms=\("uncertainty:x",\)'):
            dataclasses.replace(system, epistemic=(key,))._term_targets(params)
    assert dataclasses.replace(system, epistemic=("mode",))._term_targets(params) == (
        (),
        (),
        (),
    )


def test_unobserved_or_ineligible_state_name_mounts_nothing():
    system, params = _hybrid()
    assert system.A_deps == ((1,),)  # 'mode' is observed by no modality
    unified = dataclasses.replace(system, epistemic=("mode",))
    assert unified._term_targets(params) == ((), (), ())


# ------------------------------------------------------ parameter variables (V5.3)


def test_parameter_group_key_feeds_the_novelty_leaves():
    system, params = _discrete()

    def resolve(sel):
        return dataclasses.replace(system, epistemic=sel)._term_targets(params)

    assert resolve(("likelihood:sense",)) == ((), (0,), ())
    assert resolve(("s", "likelihood:sense")) == ((0, 1), (0,), ())


def test_sub_key_on_a_single_posterior_declaration_raises():
    """A ConditionalDirichlet is one variable (V1): its novelty leaf already
    aggregates over the per-column product, so there is nothing to address."""
    system, params = _discrete()
    with pytest.raises(ValueError, match="single parameter posterior"):
        dataclasses.replace(system, epistemic=("likelihood:sense:noise",))._term_targets(params)


def test_mniw_sub_keys_raise_pointing_at_the_joint_posterior():
    """MNIW couples head and noise in ONE posterior, so it is ONE variable — in
    the selector and in the rate mappings alike."""
    system, params = _mniw_lg()
    with pytest.raises(ValueError, match="ONE joint posterior"):
        dataclasses.replace(system, epistemic=("likelihood:y:head",))._term_targets(params)
    y = {"y": jnp.asarray([0.6])}
    beliefs, detail = system.infer_states(params, y, detail=True)
    with pytest.raises(ValueError, match="ONE joint posterior"):
        system.learn(params, y, beliefs, detail=detail, lr_by_slot={"likelihood:y:noise": 1.0})
    # the group key is the way to address it
    out = system.learn(params, y, beliefs, detail=detail, lr_by_slot={"likelihood:y": 1.0})
    assert float(out.likelihood["y"].nu) > float(params.likelihood["y"].nu)


def test_unknown_key_lists_state_and_parameter_candidates():
    system, params = _discrete()
    with pytest.raises(ValueError, match=r"state variables are \['s'\]"):
        dataclasses.replace(system, epistemic=("typo",))._term_targets(params)
    with pytest.raises(ValueError, match=r"parameter variables are \['likelihood:sense'\]"):
        dataclasses.replace(system, epistemic=("pose_cell",))._term_targets(params)
    # a modality name is not a variable: channels are not epistemic targets
    with pytest.raises(ValueError, match="unknown variable 'sense'"):
        dataclasses.replace(system, epistemic=("sense",))._term_targets(params)


def test_parameter_key_error_semantics_are_shared_with_the_rate_keys():
    """V5.3: the selector reuses ``_check_slot_key``, so a point-valued slot and
    an unknown slot read exactly as they do for ``lr_by_slot``."""
    system, params = _discrete()
    with pytest.raises(ValueError, match=r"holds a ConditionalCategorical \(a point value\)"):
        dataclasses.replace(system, epistemic=("likelihood:ident",))._term_targets(params)
    with pytest.raises(ValueError, match="unknown slot key 'likelihood:nope'"):
        dataclasses.replace(system, epistemic=("likelihood:nope",))._term_targets(params)


def test_initial_prior_key_has_no_planning_time_novelty_term():
    system, params = _d_slot_system()
    with pytest.raises(ValueError, match="no planning-time novelty term"):
        dataclasses.replace(system, epistemic=("initial_state_prior:s",))._term_targets(params)


def test_epistemic_true_keeps_the_lg_novelty_deferral_loud():
    """``epistemic=True`` must not silently drop a learnable slot: a learnable LG
    declaration's parameter family has no novelty estimator in the objectives
    registry yet (V4), so selecting it — by name or by ``True`` — raises."""
    system, params = _two_block_lg()
    unified = dataclasses.replace(system, epistemic=True)
    with pytest.raises(NotImplementedError, match="must not silently ignore"):
        unified._term_targets(params)
    with pytest.raises(NotImplementedError, match="deferred"):
        dataclasses.replace(system, epistemic=("likelihood:y:noise",))._term_targets(params)
    # naming only the state variable is the expressible ablation: x's channel leaf
    assert dataclasses.replace(system, epistemic=("x",))._term_targets(params) == ((0,), (), ())


# ----------------------------------------------- sub-block rates through `learn`


def _lg_tick():
    system, params = _two_block_lg()
    y = {"y": jnp.asarray([0.6])}
    beliefs, _ = system.infer_states(params, y)
    return system, params, y, beliefs


def test_noise_sub_key_moves_the_noise_block_only():
    """The v0.3 gap closed: with the group rate at a static 0.0, the ``:noise``
    key alone advances the noise posterior and the head posterior comes through
    bitwise (same object — nothing was rebuilt)."""
    system, params, y, beliefs = _lg_tick()
    out = system.learn(params, y, beliefs, lr=0.0, lr_by_slot={"likelihood:y:noise": 1.0})
    before, after = params.likelihood["y"].noise, out.likelihood["y"].noise
    assert float(after.alpha[0]) == pytest.approx(float(before.alpha[0]) + 0.5)
    assert out.likelihood["y"].head is params.likelihood["y"].head


def test_head_sub_key_governs_the_head_block_only():
    """The mirror: the ``:head`` rate alone decides whether the (unbuilt,
    deliberately loud) independent-head update is attempted, so freezing the head
    while the noise keeps learning is now expressible — it was not before, when
    only the whole declaration had a rate."""
    system, params, y, beliefs = _lg_tick()
    with pytest.raises(NotImplementedError, match="MatrixNormal head"):
        system.learn(params, y, beliefs, lr=0.0, lr_by_slot={"likelihood:y:head": 1.0})

    out = system.learn(params, y, beliefs, lr=1.0, lr_by_slot={"likelihood:y:head": 0.0})
    assert out.likelihood["y"].head is params.likelihood["y"].head
    assert float(out.likelihood["y"].noise.alpha[0]) == pytest.approx(
        float(params.likelihood["y"].noise.alpha[0]) + 0.5
    )
    # and the group key still sets both blocks: the head is active, hence the raise
    with pytest.raises(NotImplementedError, match="MatrixNormal head"):
        system.learn(params, y, beliefs, lr_by_slot={"likelihood:y": 1.0})


def test_sub_block_rate_may_be_traced():
    """Key resolution is static Python, the rate itself need not be: a sub-block
    key takes a traced rate under jit exactly as a group key does (the in-scan
    curriculum form)."""
    system, params, y, beliefs = _lg_tick()

    def step(p, r):
        return system.learn(p, y, beliefs, lr=0.0, lr_by_slot={"likelihood:y:noise": r})

    jitted = jax.jit(step)
    live = jitted(params, jnp.asarray(1.0)).likelihood["y"].noise
    frozen = jitted(params, jnp.asarray(0.0)).likelihood["y"].noise
    assert float(live.alpha[0]) == pytest.approx(float(params.likelihood["y"].noise.alpha[0]) + 0.5)
    np.testing.assert_array_equal(
        np.asarray(frozen.alpha), np.asarray(params.likelihood["y"].noise.alpha)
    )


def test_group_key_and_sub_key_are_equivalent_when_only_one_block_learns():
    """A noise-only LG: the ``:noise`` sub-key aliases the group, bitwise."""
    system, params = _noise_only_lg()
    y = {"y": jnp.asarray([0.6])}
    beliefs, _ = system.infer_states(params, y)
    group = system.learn(params, y, beliefs, lr_by_slot={"likelihood:y": 0.7})
    block = system.learn(params, y, beliefs, lr=0.0, lr_by_slot={"likelihood:y:noise": 0.7})
    np.testing.assert_array_equal(
        np.asarray(group.likelihood["y"].noise.beta), np.asarray(block.likelihood["y"].noise.beta)
    )


def test_decay_sub_key_rides_the_same_grammar():
    a = LinearGaussianObservation(
        head=MatrixNormal.from_head(jnp.asarray([[1.5, 0.3]]), strength=2.0),
        noise=InverseGamma.from_R_chol(jnp.asarray([[0.5]]), strength=2.0, anchor=True),
    )
    system, params = _mniw_system_and_params(a)
    y = {"y": jnp.asarray([0.6])}
    beliefs, _ = system.infer_states(params, y)
    out = system.learn(
        params,
        y,
        beliefs,
        lr=0.0,
        lr_by_slot={"likelihood:y:noise": 1.0},
        decay_by_slot={"likelihood:y:noise": 0.9},
    )
    noise, before = out.likelihood["y"].noise, params.likelihood["y"].noise
    assert float(noise.alpha[0]) == pytest.approx(
        0.9 * float(before.alpha[0]) + 0.1 * float(before.alpha0[0]) + 0.5
    )


def test_group_key_is_the_base_and_the_sub_key_the_override():
    """V7 revises step 1's group+sub-key *conflict* into precedence: specificity =
    number of literal segments, most specific wins, so the group key is the base
    rate for every block of the declaration and a sub-key overrides it. That is
    what makes ``{"likelihood:y": 0.0, "likelihood:y:noise": 1.0}`` the natural
    spelling of "freeze the head, learn the noise" — it used to raise."""
    system, params, y, beliefs = _lg_tick()
    out = system.learn(
        params, y, beliefs, lr_by_slot={"likelihood:y": 0.0, "likelihood:y:noise": 1.0}
    )
    before, after = params.likelihood["y"].noise, out.likelihood["y"].noise
    assert float(after.alpha[0]) == pytest.approx(float(before.alpha[0]) + 0.5)
    assert out.likelihood["y"].head is params.likelihood["y"].head  # the group 0.0 froze the head
    # decay rides the same ladder: the block key overrides the declaration's base
    resolved = system._resolve_rates(
        {"likelihood:y": 0.9, "likelihood:y:noise": 0.5}, "decay_by_slot", params
    )
    assert resolved == {
        "likelihood:y": 0.9,
        "likelihood:y:noise": 0.5,
        "likelihood:y:head": 0.9,
    }
    # the sibling sub-key is fine beside it (unchanged from step 1)
    system.learn(
        params,
        y,
        beliefs,
        lr=0.0,
        lr_by_slot={"likelihood:y:noise": 1.0, "likelihood:y:head": 0.0},
    )


def test_unknown_block_in_a_rate_key_lists_the_learnable_blocks():
    system, params = _noise_only_lg()
    y = {"y": jnp.asarray([0.6])}
    beliefs, _ = system.infer_states(params, y)
    with pytest.raises(ValueError, match=r"unknown block 'head'.*block\(s\) \['noise'\]"):
        system.learn(params, y, beliefs, lr_by_slot={"likelihood:y:head": 1.0})


# ------------------------------------------------- observation variables (V8)


def test_obs_key_mounts_exactly_its_own_channel():
    """V8: an observation variable is a variable, and selecting it mounts the one
    channel's ``I(o_m; s_deps)`` leaf — where a bare state name mounts *every*
    channel its factor is involved in."""
    system, params = _discrete()

    def resolve(sel):
        return dataclasses.replace(system, epistemic=sel)._term_targets(params)

    assert resolve(("obs:sense",)) == ((0,), (), ())
    assert resolve(("obs:ident",)) == ((1,), (), ())
    assert resolve(("obs:sense", "obs:ident")) == ((0, 1), (), ())
    assert resolve(("s",)) == ((0, 1), (), ())  # the state name cannot say "just one"
    # kinds compose: one channel plus the likelihood's parameter novelty
    assert resolve(("obs:sense", "likelihood:sense")) == ((0,), (0,), ())


def test_obs_keys_restore_the_cue_ablation_config():
    """The configuration V2's collapse made unreachable (V8's lesson): all channels
    *except* one. The T-maze's cue channel shares its parents with the reward
    channel, so no bare state name can drop it — ``obs:`` can."""
    from tests.test_system import tmaze_system

    system, params = tmaze_system()
    assert system.A_deps == ((0,), (0, 1), (0, 1))  # location, reward, cue

    def resolve(sel):
        return dataclasses.replace(system, epistemic=sel)._term_targets(params)

    assert resolve(True) == ((0, 1, 2), (), ())
    assert resolve(("obs:location_obs", "obs:reward")) == ((0, 1), (), ())  # cue ablated
    assert resolve(("obs:*",)) == ((0, 1, 2), (), ())  # the class = every channel
    # both bare state names are strictly coarser: 'context' cannot separate reward
    # from cue, and a duplicated state/obs name ('location') stays unambiguous
    assert resolve(("context",)) == ((1, 2), (), ())
    assert resolve(("location",)) == ((0, 1, 2), (), ())


def test_obs_key_on_an_ineligible_channel_resolves_empty():
    """Known-but-ineligible resolves empty, matching how a bare state name treats
    an unobserved variable: a neural likelihood has no state-information estimator, so
    naming it is a no-op, not an error. A Gaussian emission over categorical parents alone
    is eligible (per-variable-infogain.md V11), its information about those parents."""
    from hybrid_aif.distributions import MLPConditional
    from tests.test_structure_learning import _mixture_system

    system, params = _mixture_system()
    assert system._eligible_state_blocks(params) == ((0,), ())
    assert dataclasses.replace(system, epistemic=("obs:y",))._term_targets(params) == (
        (0,),
        (),
        (),
    )
    lg_s, lg_p = _two_block_lg()
    net = MLPConditional.from_shapes(
        (lg_p.initial_state_prior["x"].loc.shape[-1],), 1, key=jax.random.PRNGKey(0)
    )
    neural = dataclasses.replace(lg_p, likelihood={"y": net})
    assert dataclasses.replace(lg_s, epistemic=("obs:y",))._term_targets(neural) == ((), (), ())
    # an LG channel over a plain Gaussian factor is eligible (V11 decision 3): it mounts
    # its channel leaf, and the factor carries none of its own
    lg_s, lg_p = _two_block_lg()
    assert lg_s._eligible_state_blocks(lg_p) == ((0,), ())
    assert dataclasses.replace(lg_s, epistemic=("obs:y",))._term_targets(lg_p) == ((0,), (), ())
    # the same on an LG channel over a mixture-carried factor
    hyb_s, hyb_p = _hybrid()
    assert dataclasses.replace(hyb_s, epistemic=("obs:y",))._term_targets(hyb_p) == (
        (0,),
        (),
        (),
    )


def test_unknown_or_over_qualified_obs_key_raises():
    system, params = _discrete()
    with pytest.raises(ValueError, match=r"unknown observation variable 'obs:nope'"):
        dataclasses.replace(system, epistemic=("obs:nope",))._term_targets(params)
    with pytest.raises(ValueError, match="no third segment"):
        dataclasses.replace(system, epistemic=("obs:sense:noise",))._term_targets(params)
    # the reserved prefix is listed in the namespace error too
    with pytest.raises(ValueError, match=r"observation variables are \['obs:ident', 'obs:sense'\]"):
        dataclasses.replace(system, epistemic=("typo",))._term_targets(params)


def test_a_modality_named_like_a_slot_word_stays_unambiguous():
    """Keys are matched by exact table lookup, never parsed for meaning, so a
    modality may be named ``likelihood`` without confusing anything — and the
    wildcarded first segment still ranges over parameter slots only."""
    original, params = _discrete()
    system = dataclasses.replace(original, modality_names=("likelihood", "transition"))
    #  the renamed System keys its params by the new names
    params = system.from_positional("params", original.to_positional("params", params))
    assert set(system._epistemic_variables(params)) == {
        "s",
        "obs:likelihood",
        "obs:transition",
        "likelihood:likelihood",
    }

    def resolve(sel):
        return dataclasses.replace(system, epistemic=sel)._term_targets(params)

    assert resolve(("obs:likelihood",)) == ((0,), (), ())
    assert resolve(("likelihood:likelihood",)) == ((), (0,), ())
    assert resolve(("obs:*",)) == ((0, 1), (), ())
    assert resolve(("*:*",)) == ((), (0,), ())  # 'obs:...' is NOT in *:*


# ------------------------------------------------------ segment wildcards (V7)


def _dual():
    """Learnable likelihood 'sense' (modality 0) and learnable transition 's'."""
    from tests.test_transition_novelty import _dual_learnable_system

    system, params, _ = _dual_learnable_system()
    return system, params


def test_bare_star_is_every_state_variable_and_no_parameters():
    """``("*",)`` is pymdp's granularity — state info gain on, parameter novelty
    off — expressed without enumerating names, so a later-added variable is
    mounted rather than silently missed."""
    system, params = _dual()

    def resolve(sel):
        return dataclasses.replace(system, epistemic=sel)._term_targets(params)

    assert resolve(True) == ((0, 1), (0,), (0,))
    assert resolve(("*",)) == ((0, 1), (), ())
    assert resolve(("*",)) == resolve(("s",))
    # and the two classes together are exactly the default
    assert resolve(("*", "*:*")) == resolve(True)


def test_star_star_is_every_parameter_variable_at_declaration_granularity():
    system, params = _dual()

    def resolve(sel):
        return dataclasses.replace(system, epistemic=sel)._term_targets(params)

    assert resolve(("*:*",)) == ((), (0,), (0,))
    assert resolve(("likelihood:*",)) == ((), (0,), ())
    assert resolve(("transition:*",)) == ((), (), (0,))
    assert resolve(("*:s",)) == ((), (), (0,))  # the name class, across slots
    assert resolve(("*:*:*",)) == ((), (), ())  # no declaration here has blocks
    # composable with literals, and idempotent (selection is a union)
    assert resolve(("likelihood:*", "likelihood:sense")) == ((), (0,), ())
    assert resolve(("*", "likelihood:*")) == ((0, 1), (0,), ())


def test_initial_prior_class_raises_but_the_wildcarded_slot_is_silent():
    """The literal-projection rule at work on one model: ``initial_state_prior:*``
    *pins* the slot, so it raises exactly as the literal key does; ``*:*`` reaches
    the same variable *through* a wildcard, where having no planning-time novelty
    term makes it ineligible rather than a lie — "all of the ones that can carry
    one"."""
    system, params = _d_slot_system()
    assert system._epistemic_variables(params)["initial_state_prior:s"].kind == "param"
    with pytest.raises(ValueError, match="no planning-time novelty term"):
        dataclasses.replace(system, epistemic=("initial_state_prior:*",))._term_targets(params)
    assert dataclasses.replace(system, epistemic=("*:*",))._term_targets(params) == ((), (), ())


def test_a_wildcard_never_validates_less_than_its_literal_part():
    """Every literal segment is checked against its finite domain."""
    system, params = _discrete()

    def resolve(sel):
        return dataclasses.replace(system, epistemic=sel)._term_targets(params)

    with pytest.raises(ValueError, match=r"unknown first segment 'likelihod'.*legal first"):
        resolve(("likelihod:*",))
    with pytest.raises(ValueError, match=r"unknown declaration name 'nope'"):
        resolve(("*:nope",))
    with pytest.raises(ValueError, match=r"unknown block 'nope'"):
        resolve(("*:*:nope",))
    with pytest.raises(ValueError, match="4 segments"):
        resolve(("likelihood:sense:noise:*",))
    # a pattern pinning ONE declaration gets that declaration's participation check
    with pytest.raises(ValueError, match=r"holds a ConditionalCategorical \(a point value\)"):
        resolve(("likelihood:ident:*",))


def test_point_valued_declaration_raises_under_a_pinning_pattern_only():
    """A pattern that pins one declaration (slot AND name literal) carries its
    participation check; the *same* declaration reached through a wildcarded
    segment is a class member, and an empty class is silent."""
    both_s, both_p = _two_block_lg()
    with pytest.raises(ValueError, match=r"holds a LinearGaussianTransition \(a point value\)"):
        dataclasses.replace(both_s, epistemic=("transition:x:*",))._term_targets(both_p)
    system, params = _discrete()
    with pytest.raises(ValueError, match="point value"):
        dataclasses.replace(system, epistemic=("likelihood:ident:*",))._term_targets(params)
    # 'ident' through the wildcarded slot: no learnable declaration by that name,
    # hence nothing selected, hence silence
    assert dataclasses.replace(system, epistemic=("*:ident",))._term_targets(params) == (
        (),
        (),
        (),
    )


def test_empty_classes_resolve_silently():
    """Silence is reserved for classes empty *through the wildcarded* segment."""
    system, params = _d_slot_system()  # no learnable likelihood, no learnable transition

    def resolve(sel):
        return dataclasses.replace(system, epistemic=sel)._term_targets(params)

    assert resolve(("likelihood:*",)) == ((), (), ())
    assert resolve(("transition:*",)) == ((), (), ())
    assert resolve(("*:*:*",)) == ((), (), ())
    # a learnable single-posterior declaration has no blocks: ':*' is "all of none"
    sense_s, sense_p = _discrete()
    assert dataclasses.replace(sense_s, epistemic=("likelihood:sense:*",))._term_targets(
        sense_p
    ) == ((), (), ())


def test_wildcard_match_of_an_estimatorless_family_raises():
    """Judgment call (step 3), recorded: a *wildcard* match of a learnable family
    with no registered novelty estimator raises exactly as naming it literally
    does. Silent-skip would reintroduce the "toggle lies" failure — the class
    ``*:*`` says "every parameter variable", and quietly dropping a learnable one
    is the lie. (Contrast the initial prior, which has no such term *by design*:
    that one is ineligibility, not a missing implementation.)"""
    system, params = _two_block_lg()
    for sel in (("*:*",), ("likelihood:*",), ("*:*:*",), ("likelihood:y:*",)):
        with pytest.raises(NotImplementedError, match="deferred"):
            dataclasses.replace(system, epistemic=sel)._term_targets(params)
    # the state-only class stays expressible (and is the ablation to reach for)
    assert dataclasses.replace(system, epistemic=("*",))._term_targets(params) == ((0,), (), ())


# -------------------------------------------- wildcard rates + the specificity ladder


def test_wildcard_rates_are_base_plus_override():
    """``{"*:*": base, "likelihood:y:noise": override}``: the block key has three
    literal segments, the class none, so the override wins on the noise while the
    head keeps the base — bitwise identical to spelling the base literally."""
    system, params, y, beliefs = _lg_tick()
    out = system.learn(
        params, y, beliefs, lr=99.0, lr_by_slot={"*:*": 0.0, "likelihood:y:noise": 1.0}
    )
    explicit = system.learn(
        params, y, beliefs, lr=99.0, lr_by_slot={"likelihood:y": 0.0, "likelihood:y:noise": 1.0}
    )
    before = params.likelihood["y"].noise
    assert float(out.likelihood["y"].noise.alpha[0]) == pytest.approx(float(before.alpha[0]) + 0.5)
    assert out.likelihood["y"].head is params.likelihood["y"].head  # base 0.0 froze the head
    np.testing.assert_array_equal(
        np.asarray(out.likelihood["y"].noise.beta), np.asarray(explicit.likelihood["y"].noise.beta)
    )
    # the ladder, read off the resolver: every block class is less specific than
    # the block key, and the group key sits between them
    assert system._resolve_rates(
        {"*:*:*": 0.1, "likelihood:y": 0.2, "likelihood:y:head": 0.3}, "lr_by_slot", params
    ) == {"likelihood:y": 0.2, "likelihood:y:noise": 0.2, "likelihood:y:head": 0.3}


def test_equal_specificity_patterns_on_one_variable_raise():
    """``likelihood:*`` vs ``*:sense`` — one literal segment each, both matching
    the same variable: rates must be single-valued, so it is loud (V7)."""
    system, params = _dual()
    obs = system.from_positional("observations", (jnp.asarray(1), jnp.asarray(0)))
    beliefs, _ = system.infer_states(params, obs)
    with pytest.raises(ValueError, match=r"same specificity \(1 literal segment"):
        system.learn(
            params,
            obs,
            beliefs,
            prev_beliefs=beliefs,
            action=system.from_positional("actions", jnp.asarray([0])),
            lr_by_slot={"likelihood:*": 0.5, "*:sense": 0.7},
        )
    # disjoint classes are fine, and the tie is about the *pattern*, not the value
    assert system._resolve_rates({"likelihood:*": 0.5, "*:s": 0.7}, "lr_by_slot", params) == {
        "likelihood:sense": 0.5,
        "transition:s": 0.7,
    }
    with pytest.raises(ValueError, match="same specificity"):
        system._resolve_rates({"likelihood:*": 0.5, "*:sense": 0.5}, "decay_by_slot", params)


def test_wildcard_rate_value_may_be_traced():
    """Pattern matching is static Python at trace time; the rate it resolves to may
    be a traced array, so the in-scan curriculum form works through a class key."""
    system, params, y, beliefs = _lg_tick()

    def step(p, r):
        return system.learn(p, y, beliefs, lr=0.0, lr_by_slot={"*:*": 0.0, "likelihood:*:noise": r})

    jitted = jax.jit(step)
    live = jitted(params, jnp.asarray(1.0)).likelihood["y"].noise
    frozen = jitted(params, jnp.asarray(0.0)).likelihood["y"].noise
    assert float(live.alpha[0]) == pytest.approx(float(params.likelihood["y"].noise.alpha[0]) + 0.5)
    np.testing.assert_array_equal(
        np.asarray(frozen.alpha), np.asarray(params.likelihood["y"].noise.alpha)
    )


def test_rate_keys_reject_the_kinds_that_hold_no_parameters():
    """State and observation variables are in the ``epistemic`` namespace but not
    rate-addressable: they hold no parameters, so a rate key naming them is loud
    (and the wildcard forms say so in the same words)."""
    system, params, y, beliefs = _lg_tick()
    with pytest.raises(ValueError, match="hold no parameters"):
        system.learn(params, y, beliefs, lr_by_slot={"*": 1.0})
    with pytest.raises(ValueError, match=r"legal first segments are \[.*'transition'\]"):
        system.learn(params, y, beliefs, decay_by_slot={"obs:*": 1.0})
    with pytest.raises(ValueError, match="unknown slot key 'obs:y'"):
        system.learn(params, y, beliefs, lr_by_slot={"obs:y": 1.0})
    # empty classes stay silent here too (nothing learnable in the transition slot)
    assert system._resolve_rates({"transition:*": 0.5}, "lr_by_slot", params) == {}


def test_initial_prior_class_is_a_legal_rate_key():
    """The one asymmetry between the two key spaces: ``initial_state_prior:*`` has
    no planning-time novelty term, so it raises in ``epistemic`` — but the slot
    learns, so it takes a rate (D5's per-episode gating, now as a class)."""
    system, params = _d_slot_system()
    obs = system.from_positional("observations", (jnp.asarray(1, jnp.int32),))
    beliefs, _ = system.infer_states(params, obs)
    out = system.learn(params, obs, beliefs, lr=0.0, lr_by_slot={"initial_state_prior:*": 1.0})
    explicit = system.learn(params, obs, beliefs, lr=0.0, lr_by_slot={"initial_state_prior:s": 1.0})
    np.testing.assert_array_equal(
        np.asarray(out.initial_state_prior["s"].concentration),
        np.asarray(explicit.initial_state_prior["s"].concentration),
    )
    assert float(out.initial_state_prior["s"].concentration[1]) > float(
        params.initial_state_prior["s"].concentration[1]
    )
