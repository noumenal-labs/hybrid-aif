"""A11's ``recognition`` slot, and R-AIF's RSSM as a *declared* System (P1′ chunk 2a).

Design: ``docs/design/amortization.md`` **A11** — a fourth ``SystemParams`` slot
``recognition={variable: module}``, the structure on ``System.recognition``, and
"the registry names the strategy; the slot holds the strategy's parameters".
A11's refinement adopts the slot **as the restricted syntax** of a generic q-side
graph, which makes two of its guards load-bearing rather than fussy (no shared
module objects, no amortized variable in a conditioning set) — both asserted here.

What this file is about, in order:

- the **placement**: the eight parts in slots, and the faces that work over them
  (``initial_priors``, ``infer_states``, ``update_empirical_prior``,
  ``predict_beliefs``) reproducing the assembly's own steps;
- the **grammar inherited free**: rate keys, wildcards, freezing, ``LearnState``
  round-trip, ``merge()`` namespacing — none of it written for ``recognition``,
  all of it working on it;
- the **loud fences**: the ancestrality rule the code's within-step parent
  creates, a code with no recognition entry (A1(a)'s unbuilt row), A11's two
  restricted-syntax guards, and what each face still refuses.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.composition.merge import merge
from hybrid_aif.controls import Box
from hybrid_aif.distributions import MLPCategorical, MLPPredictor, PredictorHead
from hybrid_aif.distributions.neural import LearnedVector
from hybrid_aif.factors import DerivedInitial, Deterministic, Recognition
from hybrid_aif.reference.raif.world import tiny_rssm
from hybrid_aif.system import System, SystemParams
from hybrid_aif.train import WorldBatch, assemble, fit_world

KEY = jax.random.PRNGKey(0)
OBS = {"state": jnp.asarray([-0.4, 0.02]), "reward": None, "cont": None}


def _tiny(**kwargs):
    """A small-but-real RSSM System: their architecture, fewer units."""
    return tiny_rssm(
        2, 1, key=KEY, deter=8, num_vars=3, num_classes=4, units=4, mlp_units=6, embed=5, **kwargs
    )


def _batch(b=2, t=4):
    return WorldBatch(
        obs=jax.random.normal(jax.random.PRNGKey(1), (b, t, 2)) * 0.2,
        action=jax.random.uniform(jax.random.PRNGKey(2), (b, t, 1), minval=-1.0, maxval=1.0),
        reward=-jnp.ones((b, t)),
        cont=jnp.ones((b, t)),
    )


# ======================================================== the placement (A11 + P1′)


def test_the_rssm_is_eight_declarations_in_four_slots():
    """The whole of chunk 2a's rehoming, read off the declared model: a
    ``Deterministic`` carry and an ``MLPCategorical`` code in ``transition``, the
    three heads in ``likelihood``, the learned initial and its derived sibling in
    ``initial_state_prior``, and A11's two q-side factors in ``recognition`` —
    structure on the System, parameters in the params."""
    system, params = _tiny()
    assert isinstance(params.transition["deter"], Deterministic)
    assert params.transition["deter"].rule == "carry"
    assert isinstance(params.transition["stoch"], MLPCategorical)
    heads = [params.likelihood[m] for m in system.names.modalities]
    assert [head.loss for head in heads] == ["symlog_mse", "symlog_disc", "binary"]
    assert isinstance(params.initial_state_prior["deter"], LearnedVector)
    assert isinstance(params.initial_state_prior["stoch"], DerivedInitial)
    assert set(params.recognition) == {"embed", "stoch"}
    assert system.recognition == (
        Recognition(variable="embed", parents=("obs:state",)),
        Recognition(variable="stoch", parents=("deter", "recognition:embed")),
    )
    # the within-step edge that makes the ordering rule bite: stoch reads deter_t
    #  (deter's action parent is its control node, index 2, appended after the states)
    assert system.B_deps == ((1, 2), (0,), ())
    assert sorted(system._slot_declarations(params)) == [
        "initial_state_prior:deter",
        "initial_state_prior:stoch",
        "likelihood:cont",
        "likelihood:reward",
        "likelihood:state",
        "recognition:embed",
        "recognition:stoch",
        "transition:deter",
        "transition:stoch",
    ]


def test_the_faces_are_the_assembly_steps(monkeypatch):
    """``initial_priors`` is their initial state, ``update_empirical_prior`` +
    ``infer_states`` are ``obs_step`` split along the library's own seam (the
    action-conditioned half is prediction; reading the posterior off the net is
    inference), and ``predict_beliefs`` is ``open_loop``. Checked against the
    assembly's own arithmetic on the *deterministic* quantities — the carry and the
    logits — since the faces carry beliefs where the core draws codes."""
    system, params = _tiny()
    core, _ = assemble(system, params)

    initial = system.initial_priors(params)
    deter0, code0 = initial["deter"], initial["stoch"]
    np.testing.assert_allclose(np.asarray(deter0), np.asarray(core.initial()), atol=0)
    np.testing.assert_allclose(
        np.asarray(code0), np.asarray(core.prior.probs((deter0,))), rtol=1e-6
    )

    action = jnp.asarray([0.3])
    predicted = system.update_empirical_prior(
        params, {"deter": deter0, "stoch": code0}, {"a_deter": action}
    )
    expected_deter = core.gru(deter0, code0.reshape(-1), action)
    np.testing.assert_allclose(np.asarray(predicted["deter"]), np.asarray(expected_deter), atol=0)
    np.testing.assert_allclose(
        np.asarray(predicted["stoch"]),
        np.asarray(core.prior.probs((expected_deter,))),
        rtol=1e-6,
    )

    beliefs, detail = system.infer_states(params, OBS, predicted)
    assert detail is None  # a net's forward pass has no pre-reduction mixtures
    embed = core.encoder((OBS["state"],))
    np.testing.assert_allclose(np.asarray(beliefs["deter"]), np.asarray(predicted["deter"]), atol=0)
    np.testing.assert_allclose(
        np.asarray(beliefs["stoch"]),
        np.asarray(core.posterior.probs((predicted["deter"], embed))),
        rtol=1e-6,
    )

    rolled = system.predict_beliefs(params, beliefs, jnp.full((3, 1), 0.1))
    assert len(rolled) == 3
    step = system.update_empirical_prior(params, beliefs, {"a_deter": jnp.asarray([0.1])})
    np.testing.assert_allclose(np.asarray(rolled[0]["deter"]), np.asarray(step["deter"]), atol=0)
    assert rolled[-1]["stoch"].shape == (3, 4)


def test_the_recognition_module_is_what_inference_is():
    """A1(b), asserted rather than asserted-about: change the *module* in the slot
    and the posterior changes; freeze the generative side and train only the
    encoder — the whole reason an inference choice is not a world claim (A11's
    rejection of welding the encoder into the generative family)."""
    system, params = _tiny()
    beliefs, _ = system.infer_states(params, OBS, system.initial_priors(params))
    other = MLPCategorical.from_shapes((8, 5), 3, 4, key=jax.random.PRNGKey(9), width=4, depth=1)
    swapped = dataclasses.replace(params, recognition={**params.recognition, "stoch": other})
    changed, _ = system.infer_states(swapped, OBS, system.initial_priors(swapped))
    assert float(jnp.max(jnp.abs(changed["stoch"] - beliefs["stoch"]))) > 1e-3
    np.testing.assert_array_equal(  # same carry
        np.asarray(changed["deter"]), np.asarray(beliefs["deter"])
    )

    # freeze the model, train the encoder: expressible because they are two slots
    frozen_generative = {
        key: 0.0 for key in ("transition:*", "likelihood:*", "initial_state_prior:*")
    }
    updated, _, _ = fit_world(
        system,
        params,
        system.init_learn_state(params),
        _batch(),
        key=KEY,
        lr=1e-2,
        lr_by_slot=frozen_generative,
    )
    np.testing.assert_array_equal(
        np.asarray(updated.transition["stoch"].trunk.layers[-1].weight),
        np.asarray(params.transition["stoch"].trunk.layers[-1].weight),
    )
    moved = jnp.abs(
        updated.recognition["stoch"].trunk.layers[-1].weight
        - params.recognition["stoch"].trunk.layers[-1].weight
    )
    assert float(jnp.max(moved)) > 0.0


# ============================================== the grammar, inherited (V7/A2/A11)


def test_recognition_keys_take_rates_wildcards_and_freezing():
    """None of this was written for ``recognition``: the keys are slot-qualified, so
    ``recognition:stoch`` takes a rate, ``recognition:*`` is the class, the
    specificity ladder resolves base-plus-override, and a typo is loud."""
    system, params = _tiny()
    resolved = system._resolve_rates({"*:*": 1e-4, "recognition:stoch": 3e-4}, "lr_by_slot", params)
    assert resolved["recognition:stoch"] == 3e-4
    assert resolved["recognition:embed"] == 1e-4
    assert resolved["transition:deter"] == 1e-4
    with pytest.raises(ValueError, match="unknown slot key"):
        system._resolve_rates({"recognition:nope": 1.0}, "lr_by_slot", params)
    with pytest.raises(ValueError, match="unknown first segment"):
        system._resolve_rates({"recognitoin:*": 1.0}, "lr_by_slot", params)

    # the wildcard holds the whole inference side still for one step
    updated, _, _ = fit_world(
        system,
        params,
        system.init_learn_state(params),
        _batch(),
        key=KEY,
        lr=1e-2,
        lr_by_slot={"recognition:*": 0.0},
    )
    for name in ("embed", "stoch"):
        np.testing.assert_array_equal(
            np.asarray(updated.recognition[name].trunk.layers[-1].weight)
            if name == "stoch"
            else np.asarray(updated.recognition[name].layers[-1].weight),
            np.asarray(params.recognition[name].trunk.layers[-1].weight)
            if name == "stoch"
            else np.asarray(params.recognition[name].layers[-1].weight),
        )
    assert (
        float(
            jnp.max(
                jnp.abs(
                    updated.likelihood["state"].trunk.layers[-1].weight
                    - params.likelihood["state"].trunk.layers[-1].weight
                )
            )
        )
        > 0.0
    )


def test_a_frozen_recognition_module_holds_no_optimizer_state():
    """A1's trainable marking on an inference-side slot: a frozen encoder allocates
    no ``LearnState`` entry, never moves, and naming it in a rate mapping says so
    (workstream H's transducers are exactly this — a frozen module in this slot)."""
    system, params = _tiny()
    encoder = params.recognition["embed"]
    params = dataclasses.replace(
        params,
        recognition={**params.recognition, "embed": dataclasses.replace(encoder, frozen=True)},
    )
    learn_state = system.init_learn_state(params)
    assert "recognition:embed" not in learn_state.opt_states
    assert "recognition:stoch" in learn_state.opt_states
    with pytest.raises(ValueError, match="frozen MLPPredictor"):
        system._resolve_rates({"recognition:embed": 1.0}, "lr_by_slot", params)
    updated, _, _ = fit_world(system, params, learn_state, _batch(), key=KEY, lr=1e-2)
    np.testing.assert_array_equal(
        np.asarray(updated.recognition["embed"].layers[-1].weight),
        np.asarray(params.recognition["embed"].layers[-1].weight),
    )


def test_the_learn_state_round_trips_by_slot_key():
    """A2's explicit carry over the four slots: one entry per gradient-learnable
    slot, keys stable across steps, and a stale carry refused. Threading it is the
    whole of resumability — this is the loop a trainer writes."""
    system, params = _tiny()
    learn_state = system.init_learn_state(params)
    keys = set(learn_state.opt_states)
    assert "recognition:stoch" in keys and "initial_state_prior:deter" in keys
    assert "initial_state_prior:stoch" not in keys  # DerivedInitial has no parameters
    batch = _batch()
    for _ in range(2):
        params, learn_state, _ = fit_world(system, params, learn_state, batch, key=KEY, lr=1e-3)
        assert set(learn_state.opt_states) == keys
    with pytest.raises(ValueError, match="init_learn_state"):
        fit_world(system, params, None, batch, key=KEY)


def test_merge_namespaces_the_recognition_declaration_and_slot():
    """``merge()`` was written before this slot existed and namespaces it anyway,
    on both halves: the variable *and* every parent address it names, plus the
    module dict's keys. Amortized inference is per-variable, so there is nothing to
    merge — which is what makes nesting free here."""
    a, pa = _tiny()
    b, pb = _tiny()
    composite, params = merge({"one": a, "two": b}, {"one": pa, "two": pb})
    assert composite.recognition == (
        Recognition(variable="one.embed", parents=("obs:one.state",)),
        Recognition(variable="one.stoch", parents=("one.deter", "recognition:one.embed")),
        Recognition(variable="two.embed", parents=("obs:two.state",)),
        Recognition(variable="two.stoch", parents=("two.deter", "recognition:two.embed")),
    )
    assert set(params.recognition) == {"one.embed", "one.stoch", "two.embed", "two.stoch"}
    assert {"recognition:one.stoch", "recognition:two.embed"} <= set(
        composite._slot_declarations(params)
    )
    assert "recognition:two.stoch" in composite.init_learn_state(params).opt_states
    # the composite's faces still run: two independent RSSMs, one belief tuple
    obs = {f"{part}.{name}": value for part in ("one", "two") for name, value in OBS.items()}
    beliefs, _ = composite.infer_states(params, obs, composite.initial_priors(params))
    assert [tuple(x.shape) for x in beliefs.values()] == [(8,), (3, 4), (8,), (3, 4)]
    # the *assembly* is R-AIF's specific one, and says so rather than guessing
    with pytest.raises(ValueError, match="one carry node and one code factor"):
        assemble(composite, params)


def test_epistemic_treats_recognition_as_inference_not_as_a_world_claim():
    """A recognition module holds no belief about the world, so it carries no
    novelty leaf: naming it (or pinning the slot) raises, and a wildcard that
    *ranges over* slots drops it silently — exactly how the initial-prior slot
    already behaves."""
    system, params = _tiny()
    variables = system._epistemic_variables(params)
    assert "recognition:stoch" in variables
    named = dataclasses.replace(system, epistemic=("recognition:stoch",))
    with pytest.raises(ValueError, match="amortized recognition module"):
        named._validate(params)
    pinned = dataclasses.replace(system, epistemic=("recognition:*",))
    with pytest.raises(ValueError, match="pins the recognition slot"):
        pinned._validate(params)
    wide = dataclasses.replace(system, epistemic=("*:*",))
    expanded = wide._expand_epistemic("*:*", params, variables)
    assert expanded and all(not key.startswith("recognition:") for key in expanded)
    # what a class-wide wildcard *does* hit here is the generative side, whose
    # neural novelty estimator is itself deferred — and it says so, about that slot
    with pytest.raises(NotImplementedError, match="likelihood:state"):
        wide._validate(params)


# ================================================================== the loud fences


def test_a_code_declared_before_its_within_step_parent_is_refused():
    """The one ordering rule this path adds (D2's, extended to a ``Deterministic``
    parent): a code is read from its parents' values *at this step*, so a carry it
    reads must be declared earlier. Here the same model is declared with the two
    factors in the other order — code first — and the error prints a valid one."""
    system, params = _tiny()
    reversed_system = dataclasses.replace(
        system,
        factor_names=("stoch", "deter", "u_deter"),
        #  stoch <- deter_t, deter <- stoch_{t-1} and its control node: now stoch is first
        B_deps=((1,), (0, 2), ()),
        controls=(2,),
        A_deps=((1, 0), (1, 0), (1, 0)),
        recognition=(
            Recognition(variable="embed", parents=("obs:state",)),
            Recognition(variable="stoch", parents=("deter", "recognition:embed")),
        ),
    )
    reversed_params = dataclasses.replace(
        params,
        transition={"stoch": params.transition["stoch"], "deter": params.transition["deter"]},
        initial_state_prior={
            "stoch": params.initial_state_prior["stoch"],
            "deter": params.initial_state_prior["deter"],
        },
    )
    with pytest.raises(ValueError, match="within-step ordering"):
        reversed_system._validate(reversed_params)


def test_a_code_with_no_recognition_entry_names_the_unbuilt_structured_row():
    """A11's degenerate case is "fall back to the structured/exact row", and for a
    categorical code that row is A1(a)'s — designed, unbuilt, and without an oracle.
    So the fallback raises *by name* instead of silently doing something else."""
    system, params = _tiny()
    bare_system = dataclasses.replace(
        system, recognition=(Recognition(variable="embed", parents=("obs:state",)),)
    )
    bare_params = dataclasses.replace(params, recognition={"embed": params.recognition["embed"]})
    with pytest.raises(NotImplementedError, match="A1\\(a\\)|structured/exact inference row"):
        bare_system._validate(bare_params)


def test_a_shared_module_object_across_entries_is_refused():
    """A11 refinement (ii): two entries holding the *same* module object look like a
    shared torso and are not one — the pytree flatten duplicates the leaves and the
    first optimizer step forks them. Structurally identical but distinct objects are
    fine, because they genuinely are separate parameters."""
    system, params = _tiny()
    encoder = params.recognition["embed"]
    twin = MLPPredictor.from_shapes(
        (2,),
        encoder.out_dim,
        key=jax.random.PRNGKey(11),
        width=6,
        depth=1,
        out_activation=True,
        symlog_inputs=True,
    )
    shared_system = dataclasses.replace(
        system,
        recognition=(*system.recognition, Recognition(variable="embed2", parents=("obs:state",))),
    )
    with pytest.raises(ValueError, match="same module object"):
        shared_system._validate(
            dataclasses.replace(params, recognition={**params.recognition, "embed2": encoder})
        )
    shared_system._validate(
        dataclasses.replace(params, recognition={**params.recognition, "embed2": twin})
    )


def test_a_conditioning_set_may_not_name_another_amortized_variable():
    """A11 refinement (iii): this slot is the *restricted* syntax — disjoint
    per-variable stars — so structured Q (a q-factor over another q-factor's child)
    raises rather than being half-supported."""
    system, params = _tiny()
    structured = dataclasses.replace(
        system,
        recognition=(
            system.recognition[0],
            system.recognition[1],
            Recognition(variable="extra", parents=("stoch",)),
        ),
    )
    extra = MLPPredictor.from_shapes((12,), 4, key=jax.random.PRNGKey(12), width=4, depth=1)
    with pytest.raises(NotImplementedError, match="structured Q"):
        structured._validate(
            dataclasses.replace(params, recognition={**params.recognition, "extra": extra})
        )


def test_a_carry_cannot_be_amortized_and_a_module_needs_its_declaration():
    """Two correspondence errors, both loud: a recognition entry over a
    ``Deterministic`` carry has no referent (it is computed forward, never
    inferred), and a module without a declaration — or a declaration without a
    module — is the two halves of A11 disagreeing."""
    system, params = _tiny()
    carried = dataclasses.replace(
        system, recognition=(*system.recognition, Recognition(variable="deter", parents=("stoch",)))
    )
    module = MLPCategorical.from_shapes((12,), 3, 4, key=KEY, width=4, depth=1)
    with pytest.raises(ValueError, match="computed forward and carried"):
        carried._validate(
            dataclasses.replace(params, recognition={**params.recognition, "deter": module})
        )
    with pytest.raises(ValueError, match="module without a declaration"):
        system._validate(
            dataclasses.replace(params, recognition={**params.recognition, "spare": module})
        )
    with pytest.raises(ValueError, match="declared without a module"):
        dataclasses.replace(
            system, recognition=(*system.recognition, Recognition("spare", ("deter",)))
        )._validate(params)


def test_the_pairing_rules_between_a_transition_and_its_initial_prior():
    """A carry's initial prior is its episode-start *value* and a code's is derived
    from its own transition; anything else is a declaration error naming both
    sides."""
    system, params = _tiny()
    with pytest.raises(ValueError, match="LearnedVector"):
        system._validate(
            dataclasses.replace(
                params,
                initial_state_prior={
                    "deter": DerivedInitial(),
                    "stoch": params.initial_state_prior["stoch"],
                },
            )
        )
    with pytest.raises(ValueError, match="DerivedInitial"):
        system._validate(
            dataclasses.replace(
                params,
                initial_state_prior={
                    "deter": params.initial_state_prior["deter"],
                    "stoch": LearnedVector.zeros(12),
                },
            )
        )


def test_planning_with_no_row_mounted_says_so_and_learn_as_env_still_fence():
    """What each face refuses, now that chunk 2b has landed the G side.

    Planning no longer refuses *categorically* — it refuses a System that mounts no
    objective row, because G would then be an empty fold, and the error names the
    two declarations that fix it (``terms=`` and ``epistemic=``). The other two
    fences are unchanged and unrelated to scoring: the joint sequence loss is not a
    per-tick statistic ``learn`` could absorb, and running a world model forward as
    an environment is imagination (chunk 2c's ``imagine()``).

    (``plan_key`` is passed because this System's declared planner is ``"cem"``,
    whose stochasticity check comes first; the point under test is the next
    refusal.)"""
    system, params = _tiny()
    beliefs = system.initial_priors(params)
    for face in (
        lambda: system.infer_policies(params, beliefs, plan_key=KEY),
        lambda: system.score_policies(params, beliefs, jnp.zeros((2, 1))),
        lambda: system.act(KEY, params, beliefs, plan_key=KEY),
    ):
        with pytest.raises(NotImplementedError, match="no objective row"):
            face()
    with pytest.raises(NotImplementedError, match="fit_world"):
        system.learn(params, OBS, beliefs)
    with pytest.raises(NotImplementedError, match="imagine"):
        system.as_env(params)


def test_designations_on_an_amortized_system_point_at_the_scoring_rows():
    """A *target* designation over a predicted-through-a-net observation has nothing
    to take a KL against — a net has no closed-form pushforward of the belief (A1) —
    so declaring one raises rather than being silently dropped, and the error names
    what an amortized System uses instead: A12's scoring rows."""
    system, params = _tiny()
    from hybrid_aif.distributions import Gaussian

    with pytest.raises(NotImplementedError, match="scoring rows"):
        system._validate(
            dataclasses.replace(
                params, obs_targets={"state": Gaussian.from_cov(jnp.zeros(2), jnp.eye(2))}
            )
        )


def test_the_head_declaration_carries_the_loss_that_reads_its_numbers():
    """The predictor-head wrapper's whole reason to exist: 255 numbers are a
    255-dimensional observation *or* a binned scalar, and only the declared loss
    says which. Role-free (it never learns which slot it sits in), and both halves
    of the pair — ``nll`` for training, ``predict`` for reading — come from it."""
    head = PredictorHead.from_shapes(
        (3,), loss="symlog_disc", num_bins=41, bin_low=-5.0, bin_high=5.0, key=KEY, width=4, depth=1
    )
    parents = (jnp.asarray([0.1, -0.2, 0.3]),)
    assert head.out_dim == 41 and head.target_dim == 1
    assert head(parents).shape == (41,)
    assert float(head.nll(jnp.asarray(-1.5), parents)) > 0.0
    assert jnp.ndim(head.predict(parents)) == 0
    binary = PredictorHead.from_shapes((3,), loss="binary", key=KEY, width=4, depth=1)
    assert 0.0 <= float(binary.predict(parents)) <= 1.0
    vector = PredictorHead.from_shapes(
        (3,), loss="symlog_mse", out_dim=2, key=KEY, width=4, depth=1
    )
    assert vector.predict(parents).shape == (2,) and vector.target_dim == 2
    with pytest.raises(ValueError, match="out_dim"):
        PredictorHead.from_shapes((3,), loss="symlog_mse", key=KEY, width=8)
    with pytest.raises(ValueError, match="unknown head loss"):
        PredictorHead.from_shapes((3,), loss="poisson", key=KEY, width=8)


def test_the_learned_initial_is_a_declaration_that_trains():
    """Their ``initial: 'learned'``: a trainable vector behind ``tanh``, in the
    ``initial_state_prior:deter`` slot — so it takes a rate, holds a ``LearnState``
    entry, and moves. ``learned_initial=False`` is their ``initial: 'zeros'``, the
    same declaration frozen, and the configuration whose training trajectory is
    bitwise the chunk-1 bundle's."""
    system, params = _tiny()
    initial = params.initial_state_prior["deter"]
    assert initial.learnable and initial.activation == "tanh"
    np.testing.assert_array_equal(np.asarray(initial()), np.zeros(8))
    updated, _, _ = fit_world(
        system, params, system.init_learn_state(params), _batch(), key=KEY, lr=1e-2
    )
    assert float(jnp.max(jnp.abs(updated.initial_state_prior["deter"].value - initial.value))) > 0.0
    np.testing.assert_allclose(
        np.asarray(system.initial_priors(updated)["deter"]),
        np.asarray(jnp.tanh(updated.initial_state_prior["deter"].value)),
        atol=0,
    )

    zeros_system, zeros_params = _tiny(learned_initial=False)
    assert not zeros_params.initial_state_prior["deter"].learnable
    assert "initial_state_prior:deter" not in zeros_system.init_learn_state(zeros_params).opt_states


def test_a_half_amortized_declaration_fails_at_the_declaration():
    """``_is_amortized`` routes on the declaration families, and the amortized
    validator then insists the rest of the model is one it can express — so a
    tabular factor beside a code, or a recognition entry on a tabular System, is a
    declaration error rather than a silent trip down the wrong path."""
    tabular_transition = MLPCategorical.from_shapes((8,), 3, 4, key=KEY, width=4, depth=1)
    system = System(
        A_deps=(),
        B_deps=((),),
        controlled=(False,),
        policy_len=1,
        factor_names=("stoch",),
        recognition=(Recognition("stoch", ()),),
    )
    with pytest.raises(ValueError, match="parent_dims"):
        system._validate(
            SystemParams(
                transition={"stoch": tabular_transition},
                initial_state_prior={"stoch": DerivedInitial()},
                recognition={
                    "stoch": MLPCategorical.from_shapes(
                        (8,), 3, 4, key=jax.random.PRNGKey(3), width=4, depth=1
                    )
                },
            )
        )
    from hybrid_aif.distributions import Categorical, ConditionalCategorical

    with pytest.raises(ValueError, match="not a transition declaration of the amortized path"):
        System(
            A_deps=(),
            B_deps=((),),
            controlled=(False,),
            policy_len=1,
            factor_names=("x",),
            recognition=(Recognition("x", ()),),
        )._validate(
            SystemParams(
                transition={
                    "x": ConditionalCategorical(
                        log_table=jnp.log(jnp.ones((2, 1)) / 2), parent_ndims=1
                    ),
                },
                initial_state_prior={"x": Categorical(logits=jnp.zeros(2))},
                recognition={"x": MLPPredictor.from_shapes((2,), 2, key=KEY, width=2, depth=1)},
            )
        )


def test_a_continuous_action_reaches_the_carry_as_a_parent_vector():
    """A9's positive list gains its second member here, and it is the same rule it
    always was: the action must reach the transition as a *parent vector*, which a
    ``Deterministic`` map's input takes. The layout is the standing one — the
    declared block of the flat float action."""
    system, params = _tiny()
    assert system.action_dim(params) == 1
    assert isinstance(params.control["u_deter"], Box)
    beliefs = system.initial_priors(params)
    left = system.update_empirical_prior(params, beliefs, {"a_deter": jnp.asarray([-1.0])})
    right = system.update_empirical_prior(params, beliefs, {"a_deter": jnp.asarray([1.0])})
    #  the action is doing something
    assert float(jnp.max(jnp.abs(left["deter"] - right["deter"]))) > 0.0
    with pytest.raises(ValueError, match="integer dtype"):
        system.predict_beliefs(params, beliefs, jnp.zeros((2, 1), jnp.int32))
