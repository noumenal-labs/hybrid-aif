"""A12's ``scoring`` slot, the ``terms=`` key grammar, and the three groups.

Design: ``docs/design/amortization.md`` A12 item 1 (the slot + the selector), its
"A12 implementation note: the terms= key grammar" (the build-time grammar
decisions this file pins), A4 (the registry-row ``derived`` field) and
``system.md``'s 2026-08-21 addendum (the three groups of ``SystemParams`` and the
two enforcement points it asks for).

The mathematics of the five rows is ``test_amortized_objective.py``; this file is
about *addressing*: which keys are legal, what the illegal ones say, and which
field belongs to which group.
"""

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from hybrid_aif.composition.merge import merge
from hybrid_aif.distributions import Categorical, MLPConditional
from hybrid_aif.distributions.neural import PredictorHead
from hybrid_aif.interval import Interval
from hybrid_aif.objectives import registry
from hybrid_aif.reference.raif.world import crspp_rssm, tiny_rssm
from hybrid_aif.system import ACTOR_KEY, LearnState, SystemParams

KEY = jax.random.PRNGKey(0)

#: The three groups of ``system.md``'s addendum. A field is a **closed-loop
#: factor's own** iff its contents, evaluated at the loop boundary, *return the
#: distribution the loop then acts on* — the recognition module returns the belief
#: update, the E designation returns ``P(a|s)``. Everything else is material
#: consumed interior to one of those two, and the group it lands in says which.
GROUPS: dict[str, frozenset[str]] = {
    # 1: the declared model — positional, indexed by graph position, interior to Q
    #    (``control``: each continuous control node's support, the box its family lives on;
    #    ``actuator``: an action node's kernel P(a | u); ``durations``: a round's clock per
    #    level of one control node)
    "declared_model": frozenset(
        {"transition", "likelihood", "initial_state_prior", "control", "actuator", "durations"}
    ),
    # 2: the closed-loop factors — these two and no others
    "loop_factors": frozenset({"recognition", "policy_prior"}),
    # 3: scoring material — no kernel of its own; interior to P(a|s)
    #    (``precision``: the policy precision scaling G in q_pi, read where ``System.gamma`` is)
    "scoring_material": frozenset(
        {"obs_targets", "state_targets", "terminal_targets", "scoring", "precision"}
    ),
}


def _tiny(**kwargs):
    return tiny_rssm(
        2, 1, key=KEY, deter=8, num_vars=3, num_classes=4, units=4, mlp_units=6, embed=5, **kwargs
    )


def _agent(**system_kwargs):
    """A ``tiny_rssm`` with the whole G side declared on it: A12's fidelity config."""
    keys = jax.random.split(KEY, 6)
    system, params = _tiny()
    crspp, crspp_params = crspp_rssm(key=keys[0], deter=8, num_vars=3, num_classes=4, units=4)
    feat = 8 + 3 * 4
    critic = PredictorHead.from_shapes(
        (feat,), loss="symlog_disc", num_bins=9, key=keys[1], width=4, depth=1
    )
    ensemble = tuple(
        MLPConditional.from_shapes((8, 12, 1), 12, key=k, width=4, depth=1)
        for k in jax.random.split(keys[2], 4)
    )
    actor = MLPConditional.from_shapes((feat,), 1, key=keys[3], width=4, depth=1)
    declared = {
        "planner": "habit",
        "epistemic": ("stoch",),
        "terms": (
            "occupancy:transition:stoch",
            "predicted_reward:obs:reward",
            "cosine_sim_to_pref_rollout",
            "g_tail",
        ),
        "scoring": (("cosine_sim_to_pref_rollout", crspp),),
        "policy_len": 2,
    }
    declared.update(system_kwargs)
    if "cosine_sim_to_pref_rollout" not in declared["terms"]:
        declared["scoring"] = ()  # an unmounted structure entry raises, by design
    system = dataclasses.replace(system, **declared)
    params = dataclasses.replace(
        params,
        policy_prior=actor,
        scoring={
            "g_tail": (critic, critic),
            "ensemble_disagreement": ensemble,
            "cosine_sim_to_pref_rollout": crspp_params,
        },
    )
    return system, params


# ------------------------------------------------- the groups (the membership test)


def test_every_system_params_field_is_in_exactly_one_group():
    """``system.md``'s first enforcement point: the partition is asserted, so a
    future field has to be *classified* to pass rather than drifting in unnamed.

    The criterion is quoted at :data:`GROUPS`. Group membership is a role, not a
    type: the info-gain ensemble and the CRSPP are built out of kernels but
    consumed to produce a real number, so ``scoring`` is group 3 — and what a row
    returns (an ``Interval``, never a ``Distribution``) is the mechanical form of
    that test."""
    fields = {f.name for f in dataclasses.fields(SystemParams)}
    classified = frozenset().union(*GROUPS.values())
    assert fields == classified, (
        f"unclassified SystemParams field(s) {sorted(fields - classified)}: put each in one of "
        "the three groups of system.md's 2026-08-21 addendum (declared model / closed-loop "
        f"factors / scoring material), or drop {sorted(classified - fields)} if a field went away"
    )
    for a, first in GROUPS.items():
        for b, second in GROUPS.items():
            assert a == b or not (first & second), f"{a} and {b} overlap"


def test_the_two_loop_factors_are_exactly_recognition_and_the_policy_prior():
    """ "These two and no others": the group-2 claim is the load-bearing one, since
    it is what justifies ``recognition`` and ``scoring`` being separate fields at
    all (each group-1/3 field is interior to one of these two)."""
    assert GROUPS["loop_factors"] == frozenset({"recognition", "policy_prior"})


def test_policy_prior_carries_the_E_alias_and_no_log_E_remains():
    """A12 item 3: the field is ``policy_prior`` with a read-only ``E`` alias
    completing ``A``/``B``/``D`` — D the prior over initial states, E the prior over
    policies. No ``log_E`` shim anywhere: a module entry has no log-probability
    vector, so the old name would be ill-defined as well as stale."""
    _, params = _tiny()
    assert params.E is params.policy_prior
    assert not hasattr(params, "log_E")
    assert "log_E" not in {f.name for f in dataclasses.fields(SystemParams)}
    with pytest.raises(AttributeError):
        params.E = None  # read-only, like A/B/D


# ------------------------------------------------------------- the terms= grammar


def test_row_keys_and_variable_keys_share_one_grammar_without_ambiguity():
    """A row key is ``<row>`` or ``<row>:<variable key>``, and the first segment is
    resolved against the registry's *closed* table — which is what keeps the two
    selectors' namespaces disjoint without a collision checker."""
    system, params = _agent()
    resolved = {m.name: m.index for m in system._row_mounts(params)}
    assert resolved == {
        "occupancy": (1,),  # the code factor: transition:stoch
        "predicted_reward": (1,),  # the reward modality of (state, reward, cont)
        "cosine_sim_to_pref_rollout": (),  # no target: it scores the whole rollout
        "g_tail": (),
    }


def test_an_unknown_row_lists_the_rows_and_points_away_from_epistemic():
    system, params = _agent(terms=("noveltyish:transition:stoch",))
    with pytest.raises(ValueError, match="unknown objective row"):
        system._row_mounts(params)


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("g_tail:transition:stoch", "takes no target"),  # a row with no target
        ("occupancy", "scores a declared transition"),  # a slot row needs one
        ("occupancy:stoch", "scores a declared transition"),
        ("occupancy:*", "scores a declared transition"),  # the shape cannot address a kernel
        ("occupancy:*:*", "there is exactly one it can"),  # no wildcard in the slot segment
        ("occupancy:likelihood:state", "scores a declared transition"),
        ("occupancy:transition:nope", "unknown slot key"),
        ("predicted_reward", "one observation channel"),
        ("predicted_reward:obs:*", "no wildcard form"),
        ("predicted_reward:reward", "one observation channel"),
        ("predicted_reward:obs:nope", "unknown modality"),
    ],
)
def test_illegal_terms_keys_raise_and_say_what_the_row_scores(key, message):
    """The grammar's conservative half (the A12 implementation note, section b): the
    row segment is always literal, a wildcard is legal only where the row declares
    a target *class*, and a pattern whose shape cannot address the row's target kind
    raises rather than resolving empty — because the literal row name pins that
    kind, and V7's rule is that a wildcard never validates less than its literal
    part."""
    system, params = _agent(terms=(key,))
    with pytest.raises(ValueError, match=message):
        system._row_mounts(params)


def test_the_row_segment_takes_no_wildcard_at_all():
    """``terms=("*",)`` would mean "mount every registered row" — a set of
    independent modelling commitments, most needing parameters this System does not
    carry, and no finite domain to validate against. It raises."""
    system, params = _agent(terms=("*",))
    with pytest.raises(ValueError, match="unknown objective row"):
        system._row_mounts(params)


def test_a_class_wildcard_selects_the_kernels_that_can_carry_the_row():
    """``occupancy:transition:*`` is the legal wildcard: it ranges over the declared
    transitions and *silently* skips the deterministic carry, which has no
    conditional entropy — V7's "empty through the wildcarded segment" silence,
    beside the loud raise for naming that carry literally (above)."""
    system, params = _agent(terms=("occupancy:transition:*",))
    (mount,) = system._row_mounts(params)
    assert mount.name == "occupancy" and mount.index == (1,)  # the code, not the carry


def test_selection_is_a_union_so_a_row_named_twice_mounts_once():
    system, params = _agent(terms=("occupancy:transition:*", "occupancy:transition:stoch"))
    assert len(system._row_mounts(params)) == 1


@pytest.mark.parametrize(
    ("key", "family"),
    [
        ("occupancy:transition:deter", "carry"),  # a carry has no conditional entropy
        ("model_uncertainty:deter", "carry"),
    ],
)
def test_a_row_named_on_a_family_it_cannot_read_names_the_missing_estimator(key, family):
    """Eligibility is per row and per family (ledger entry 2): what is missing is one
    ``(row, family)`` entry, and the refusal says so — beside the wildcard's silence
    for the same member (V7's rule, tested above)."""
    system, params = _agent(terms=(key, "g_tail"))
    with pytest.raises(NotImplementedError, match=f"and .* is '{family}'"):
        system._row_mounts(params)


def test_an_amortized_only_row_is_refused_on_a_tabular_system_per_row():
    """The blanket "terms= on a non-amortized System" raise is gone: a row that reads
    neural material is refused by *itself*, naming the families it has an estimator
    for — while ``occupancy``, which now has a tabular estimator, mounts."""
    from tests.test_system import tmaze_system

    system, params = tmaze_system()
    for key in ("g_tail", "predicted_reward:obs:reward"):
        broken = dataclasses.replace(system, terms=(key,))
        with pytest.raises(NotImplementedError, match="carries an estimator for"):
            broken.infer_policies(params, broken.initial_priors(params))
    mounted = dataclasses.replace(system, terms=("occupancy:transition:location",))
    (mount,) = mounted._row_mounts(params)
    assert (mount.name, mount.index, mount.family) == ("occupancy", (0,), "categorical")


def test_terms_are_validated_at_declaration_time():
    """``_validate`` resolves the keys, so a typo fails where the model is written
    rather than at the first plan."""
    system, params = _agent(terms=("g_tail:oops",))
    with pytest.raises(ValueError, match="takes no target"):
        system._validate(params)


def test_mounting_is_declared_never_inferred_from_the_slot():
    """A12 item 1's invariant: two ``SystemParams`` against one ``System`` produce
    the same term-tree *shape*. So an empty ``scoring`` slot does not silently
    unmount the rows — it is a mismatch, and it is loud."""
    system, params = _agent()
    beliefs = system.initial_priors(params)
    stripped = dataclasses.replace(params, scoring={})
    with pytest.raises(ValueError, match="scoring carries no"):
        system.infer_policies(stripped, beliefs, plan_key=KEY)


def test_the_two_scoring_halves_pair_by_row_name_like_recognition_does():
    """The flipped shape (debt-ledger entry 3): ``System.scoring`` holds a row's
    declared *structure* and ``SystemParams.scoring`` its *parameters*, under the
    same key — which is ``System.recognition`` / ``SystemParams.recognition`` one
    slot over. A tuple of pairs on the static side, because static fields must hash
    (the ``subsystems`` precedent); ``dict(system.scoring)`` is the lookup form."""
    system, params = _agent()
    assert dict(system.scoring).keys() == {"cosine_sim_to_pref_rollout"}
    assert "cosine_sim_to_pref_rollout" in params.scoring
    declared = system._declared_structures()
    assert declared["cosine_sim_to_pref_rollout"].A_deps == ()  # the CRSPP: no heads


def test_a_system_carrying_a_declared_structure_is_still_hashable_and_static():
    """``System.scoring`` is a static field, so its contents must hash — a declared
    System does, being an all-static pytree, and a tuple of pairs keeps the field
    itself hashable where a dict would not."""
    system, _ = _agent()
    assert isinstance(hash(system), int)
    assert not jax.tree_util.tree_leaves(system)  # all-static: nothing to trace


def test_a_structure_entry_that_terms_does_not_mount_raises():
    """Mounting stays the ``terms=`` selector's job alone (A12 item 1), so a
    structure entry is never a second, quieter way to switch a row on."""
    system, params = _agent()
    crspp = dict(system.scoring)["cosine_sim_to_pref_rollout"]
    inert = dataclasses.replace(system, terms=("g_tail",))  # structure kept, mount removed
    with pytest.raises(ValueError, match="does not mount"):
        inert._row_mounts(params)
    unmounted = dataclasses.replace(
        system,
        terms=(),
        scoring=(
            (
                "cosine_sim_to_pref_rollout",
                crspp,
            ),
        ),
    )
    with pytest.raises(ValueError, match="mounts none of them"):
        unmounted._row_mounts(params)


def test_a_mounted_row_that_owns_a_model_must_find_it():
    system, params = _agent()
    with pytest.raises(ValueError, match="own a declared model and found none"):
        dataclasses.replace(system, scoring=())._row_mounts(params)


def test_the_structure_mapping_validates_its_own_keys():
    system, params = _agent()
    with pytest.raises(ValueError, match="unknown objective row"):
        dataclasses.replace(system, scoring=(("nope", None),))._row_mounts(params)
    doubled = (("cosine_sim_to_pref_rollout", None),) * 2
    with pytest.raises(ValueError, match="twice"):
        dataclasses.replace(system, scoring=doubled)._row_mounts(params)


# --------------------------------------------------- scoring: as an address, and not


def test_scoring_rows_are_addressable_in_the_one_slot_key_space():
    """The grammar inherits (A12): ``scoring:<row>`` joins the slot-qualified key
    space, so a trainer can read the model by key and write it back by the same
    key."""
    system, params = _agent()
    table = system._slot_declarations(params)
    assert {"scoring:g_tail", "scoring:ensemble_disagreement"} <= set(table)
    assert table["scoring:g_tail"][0] == "scoring"
    written = system._write_slots(params, {"scoring:g_tail": ("critic",)})
    assert written.scoring["g_tail"] == ("critic",)
    assert written.scoring["ensemble_disagreement"] is params.scoring["ensemble_disagreement"]


def test_a_literal_scoring_rate_key_resolves_once_its_row_has_a_trainer():
    """Ledger entry 6, paid by P1′ chunk 2c: A12 FLAG (4) fenced ``scoring:`` rates
    because nothing stepped them. Now something does, and the address behaves like
    any other — the rate reaches the trainer that owns the row's loss, never the
    per-tick ``learn`` face (which raises on an amortized System outright)."""
    system, params = _agent()
    resolved = system._resolve_rates({"scoring:g_tail": 0.5}, "lr_by_slot", params)
    assert resolved["scoring:g_tail"] == 0.5


def test_a_scoring_rate_key_for_a_row_with_no_trainer_still_raises():
    """The other half of the same payment: the fence is per-row, not lifted
    wholesale. A rate nothing would apply is still the "toggle lies" failure, and
    the error says what makes a row addressable."""
    system, params = _agent()
    # ``occupancy`` is a registered row whose leaf reads declared material only, so
    # nothing trains it and nothing ever will: a perfect stand-in for the fence.
    with_untrained = dataclasses.replace(params, scoring={**params.scoring, "occupancy": object()})
    with pytest.raises(NotImplementedError, match="no trainer"):
        system._resolve_rates({"scoring:occupancy": 0.5}, "lr_by_slot", with_untrained)


def test_a_wildcard_ranging_over_scoring_reaches_a_trained_row():
    """With the fence lifted per-row, a trained scoring row joins the class a
    wildcard ranges over — the ordinary rate semantics, no special case. An
    untrained row is still dropped silently through the wildcard (V7's rule) and
    loud when named literally, which the previous test asserts."""
    system, params = _agent()
    resolved = system._resolve_rates({"*:*": 0.9}, "lr_by_slot", params)
    assert resolved["scoring:g_tail"] == 0.9
    assert resolved["scoring:ensemble_disagreement"] == 0.9


def test_scoring_is_not_an_epistemic_target():
    """A11's argument, one slot over: planning scores what the *world model* does
    not know, and a critic is not a world claim. Both the literal key and a pinned
    wildcard raise."""
    system, params = _agent()
    for key in ("scoring:g_tail", "scoring:*"):
        broken = dataclasses.replace(system, epistemic=(key,))
        with pytest.raises(ValueError, match="scoring"):
            broken._term_targets(params)


def test_learn_state_carries_one_entry_per_trained_scoring_row():
    """Ledger entry 6's second half. Optimizer state for the critic/ensemble/CRSPP
    is still *owned* by the trainers that hold their losses (A6), but it rides in
    the one carry under the same addresses — including the actor's, whose
    ``policy_prior`` is not a slot and so takes the bare
    :data:`~hybrid_aif.system.ACTOR_KEY`. The CRSPP's is a nested ``LearnState``
    over its own declared slots, because a row that owns a System owns its
    addresses too."""
    system, params = _agent()
    opt = system.init_learn_state(params).opt_states
    assert {"scoring:g_tail", "scoring:ensemble_disagreement", ACTOR_KEY} <= set(opt)
    assert isinstance(opt["scoring:cosine_sim_to_pref_rollout"], LearnState)
    crspp = dict(system.scoring)["cosine_sim_to_pref_rollout"]
    nested = opt["scoring:cosine_sim_to_pref_rollout"].opt_states
    assert set(nested) == set(
        crspp.init_learn_state(params.scoring["cosine_sim_to_pref_rollout"]).opt_states
    )


# ------------------------------------------------------- the registry-row contract


def test_a_parametric_row_gets_no_derived_default():
    """A4's enforcement half, structurally: the only place a trust label can be
    wrong is the one reviewed line that registers a parameter-bearing row."""
    with pytest.raises(ValueError, match="no 'derived' default"):
        registry.Estimator("made_up", registry.SINGLE, lambda st: None, parametric=True)
    assert registry.Estimator("plain", registry.SINGLE, lambda st: None).derived is True


def test_a_row_name_may_not_collide_with_the_variable_key_grammar():
    """A row called ``obs`` would make ``obs:cue`` ambiguous between the two
    selectors' readings of its first segment, so the shape of the name is checked at
    registration."""
    for name in ("obs", "transition", "occupancy:extra"):
        with pytest.raises(ValueError, match="collides with the variable key grammar|already"):
            registry.register_row(
                registry.Estimator(name, registry.SINGLE, lambda st: None, derived=False)
            )


def test_a_second_estimator_under_one_row_name_must_be_a_disjoint_family_variant():
    """How ``occupancy`` holds two estimators (ledger entry 2): one bucket, one
    estimator per family. A duplicate registration that does not partition by family
    is the old mistake and still raises, and a variant may differ from its sibling
    only in its estimator and its warrant."""
    plain = dict(keying=registry.VARIABLE, term="occupancy", target=registry.SLOT_TARGET)
    for row, match in (
        (registry.Estimator("occupancy", **plain, prepare=lambda st: None), "already registered"),
        (
            registry.Estimator(
                "occupancy", **plain, prepare=lambda st: None, families=(registry.CATEGORICAL,)
            ),
            "already registered for family",
        ),
        (
            registry.Estimator(
                "occupancy",
                registry.SINGLE,
                lambda st: None,
                term="occupancy",
                target=registry.SLOT_TARGET,
                families=("gaussian",),
            ),
            "variants disagree on",
        ),
    ):
        with pytest.raises(ValueError, match=match):
            registry.register_row(row, slot="transition")
    #  the shipped pair, as registered: two families, two warrants, one row
    assert registry.row_families("occupancy") == ("categorical", "code")
    assert registry.row_variant("occupancy", "categorical").derived is True
    assert registry.row_variant("occupancy", "code").derived is False


def test_every_registered_row_is_marked_and_every_amortized_one_is_unwarranted():
    """A10 disposition 11 as a property of the table: the amortized rows all claim
    nothing, which is the flag working as designed rather than a defect to mitigate.
    The warrant is a property of the *estimator*, so it is read per family variant —
    ``occupancy``'s tabular leaf is a derivation, its neural namesake is not, and the
    ``uncertainty`` row on a declared Gaussian or mixture belief is one too."""
    declared = (registry.CATEGORICAL, registry.GAUSSIAN, registry.MIXTURE)
    for name, row in registry.ROWS.items():
        assert row.term in ("occupancy", "predicted_reward", name, "cosine_sim_to_pref_rollout")
        for family in registry.row_families(name):
            variant = registry.row_variant(name, family)
            assert variant.derived is (family in declared), (name, family)


def test_the_row_stamps_its_warrant_as_an_and_never_an_assignment():
    """``registry.stamp`` poisons, it does not launder: a derived row keeps whatever
    its own computation claimed."""
    unwarranted = registry.stamp(lambda ctx: {"x": Interval.exact(jnp.zeros(()))}, False)
    assert unwarranted(None)["x"].derived is False
    already = registry.stamp(lambda ctx: {"x": Interval.exact(jnp.zeros(()), derived=False)}, True)
    assert already(None)["x"].derived is False


# ------------------------------------------------------------------- composition


def test_merging_a_component_with_a_g_side_names_the_key_grammar():
    """The addendum's second enforcement point. The obstacle is addressing, not
    mathematics: a row is addressed by row *name*, which namespacing deliberately
    does not qualify, so two components' critics would both answer to ``g_tail``."""
    system, params = _agent()
    for broken_system, broken_params in (
        (system, params),  # the whole G side
        (dataclasses.replace(system, terms=()), params),  # structure alone
        (dataclasses.replace(system, terms=(), scoring=()), params),  # parameters alone
    ):
        with pytest.raises(NotImplementedError, match="objective rows"):
            merge({"mind": broken_system}, {"mind": broken_params})


def test_merging_a_scoring_entry_as_a_subsystem_names_the_kernel_criterion():
    """FLAG 5's real hazard, and its cheap sibling: a group-3 entry has no kernel at
    the loop boundary, so it cannot be a partner in the loop — it is material
    consumed interior to computing ``P(a|s)``."""
    _, params = _agent()
    with pytest.raises(TypeError, match="no kernel at the loop boundary"):
        merge({"mind": params}, {"mind": params})


def test_a_policy_module_does_not_compose_over_a_joint_policy_set():
    """A module E is a conditional prior over its *own* beliefs, so composing two is
    a question about a joint action distribution, not a log-space addition."""
    from tests.test_merge import agent_and_world

    (asys, ap), (wsys, wp) = agent_and_world()
    ap = dataclasses.replace(ap, policy_prior=lambda beliefs: None)
    with pytest.raises(NotImplementedError, match="policy module"):
        merge({"agent": asys, "world": wsys}, {"agent": ap, "world": wp})


def test_a_vector_habit_still_composes_as_it_always_did():
    """The rename is a rename: the classical E path is untouched."""
    from tests.test_merge import agent_and_world

    (asys, ap), (wsys, wp) = agent_and_world(policy_len=1)
    ap = dataclasses.replace(ap, policy_prior=jnp.zeros(5))
    wp = dataclasses.replace(wp, policy_prior=Categorical(logits=jnp.arange(5.0)))
    _, cp = merge({"agent": asys, "world": wsys}, {"agent": ap, "world": wp})
    assert cp.policy_prior.shape == (25,)
