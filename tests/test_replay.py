"""Recorded-trace replay (``docs/design/replay.md``): fidelity, the record winning,
the fitting objective, masks, padding, batches, the document entry, and learning.

Scan-versus-eager agreement is asserted at float32 tolerance, not bitwise: a
``lax.scan`` fuses differently from a Python loop of the same faces (the
population contract records the same caution for ``vmap``).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hybrid_aif as H
from hybrid_aif.description import DescriptionError, describe
from hybrid_aif.objectives.hybrid import pad_mixture
from hybrid_aif.objectives.hybrid import prepare_beliefs as pad_beliefs
from hybrid_aif.reference.envs.tmaze import CENTRE
from hybrid_aif.reference.raif.world import tiny_rssm
from hybrid_aif.replay import Replay, replay, replay_document, stack_traces
from tests.test_system import tmaze_system

_EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"


def _example(name: str):
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _close(a, b, **kw) -> None:
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    for x, y in zip(la, lb, strict=True):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-5, atol=1e-6, **kw)


def _stack(rows):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *rows)


def _equal(a, b) -> None:
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _length(observations, actions) -> int:
    return int(jax.tree_util.tree_leaves((observations, actions))[0].shape[0])


def _keyed_mask(system, mask):
    """A ``(T, M)`` (or ``(M,)``) flag array as the keyed mask, one entry per modality."""
    return system.from_positional("observations", tuple(jnp.moveaxis(mask, -1, 0)))


def _flags(system, flags):
    """One flag per modality, in modality order, keyed by modality name."""
    return system.from_positional("observations", flags)


# ------------------------------------------------------------- eager references


def eager_loop(system, params, observations, actions, *, learn=False, lr=1.0):
    """The canonical loop written out, with the recorded actions driving the carry."""
    T = _length(observations, actions)
    priors = system.initial_priors(params)
    if system._is_hybrid(params):  # the same seam replay pads at: mixture factors at mixture_k
        padded = pad_beliefs(
            system.to_positional("beliefs", priors), system._families(params), system.mixture_k
        )
        priors = system.from_positional("beliefs", padded)
    prev_b, prev_a = priors, {k: jnp.zeros_like(v[0]) for k, v in actions.items()}
    beliefs_rows, priors_rows, q_rows, chosen_rows = [], [], [], []
    for t in range(T):
        obs = {k: o[t] for k, o in observations.items()}
        act_t = {k: a[t] for k, a in actions.items()}
        beliefs, _ = system.infer_states(params, obs, priors)
        if learn and t > 0:
            params = system.learn(params, obs, beliefs, prev_b, prev_a, lr=lr)
        chosen, q_pi, _ = system.act(
            jax.random.PRNGKey(0), params, beliefs, selection="deterministic", mode="marginal"
        )
        beliefs_rows.append(beliefs)
        priors_rows.append(priors)
        q_rows.append(q_pi)
        chosen_rows.append(chosen)
        priors = system.update_empirical_prior(params, beliefs, act_t)
        prev_b, prev_a = beliefs, act_t
    return _stack(beliefs_rows), _stack(priors_rows), _stack(q_rows), _stack(chosen_rows), params


def tmaze_trace(system, params, ticks=6, seed=7, follow_model=True, override=None):
    """A trace from the real T-maze env, driven by the model's own deterministic choice —
    or, with ``follow_model=False``, by ``override`` (a fixed positional action vector)."""
    env, env_params = system.as_env(params)
    key = jax.random.PRNGKey(seed)
    k_reset, key = jax.random.split(key)
    obs, state = env.reset(k_reset, env_params)
    priors = None
    obs_rows, act_rows = [], []
    for _ in range(ticks):
        beliefs, _ = system.infer_states(params, obs, priors)
        k_act, k_env, key = tuple(jax.random.split(key, 3))
        action, _, _ = system.act(k_act, params, beliefs, selection="deterministic")
        if not follow_model:
            action = system.from_positional("actions", jnp.asarray(override, jnp.int32))
        obs_rows.append(obs)
        act_rows.append(action)
        priors = system.update_empirical_prior(params, beliefs, action)
        obs, state = env.step(k_env, state, action, env_params)
    return _stack(obs_rows), _stack(act_rows)


@pytest.fixture(scope="module")
def tmaze():
    return tmaze_system(policy_len=3)


@pytest.fixture(scope="module")
def own_trace(tmaze):
    return tmaze_trace(*tmaze)


# ----------------------------------------------------------------- R1/R2: fidelity


def test_replay_reproduces_the_eager_loop(tmaze, own_trace):
    system, params = tmaze
    observations, actions = own_trace
    out = replay(system, params, observations, actions)
    assert isinstance(out, Replay)
    beliefs, priors, q_pi, chosen, _ = eager_loop(system, params, observations, actions)
    _close(out.beliefs, beliefs)
    _close(out.priors, priors)
    _close(out.q_pi, q_pi)
    _equal(out.chosen, chosen)
    assert bool(jnp.all(out.valid))
    assert out.params is None
    # discrete path: the tick's free energy, and its negation as the evidence
    assert out.free_energy is not None and out.free_energy.shape == out.valid.shape
    np.testing.assert_array_equal(np.asarray(out.log_evidence), -np.asarray(out.free_energy))
    assert set(out.trees) >= {"infogain", "utility"}
    for row in out.trees.values():
        for leaf in row.values():
            assert leaf.midpoint.shape[0] == _length(observations, actions)


def test_the_trace_the_model_generated_is_its_own_choice(tmaze, own_trace):
    system, params = tmaze
    observations, actions = own_trace
    out = replay(system, params, observations, actions)
    _equal(out.chosen, actions)
    assert bool(jnp.all(out.action_log_prob <= 0.0))
    assert bool(jnp.all(jnp.isfinite(out.action_log_prob)))


def test_metrics_false_returns_no_trees(tmaze, own_trace):
    system, params = tmaze
    out = replay(system, params, *own_trace, metrics=False)
    assert out.trees is None


# ------------------------------------------------------------ R1/R3: the record wins


def test_the_record_drives_the_beliefs_not_the_model(tmaze):
    system, params = tmaze
    stay = jnp.asarray([CENTRE], jnp.int32)  # never visit the cue, never commit
    observations, actions = tmaze_trace(system, params, follow_model=False, override=stay)
    out = replay(system, params, observations, actions)
    beliefs, priors, _, chosen, _ = eager_loop(system, params, observations, actions)
    _close(out.beliefs, beliefs)
    _close(out.priors, priors)
    chosen = system.to_positional("actions", out.chosen)
    recorded = system.to_positional("actions", actions)
    #  the model would have done something else on at least one tick ...
    assert bool(jnp.any(jnp.any(chosen != recorded, axis=-1)))
    #  ... and the recorded action is less likely under it than its own choice is
    first = system.policies(params)[:, 0, :]
    own = jnp.log(
        jnp.sum(out.q_pi.probs * jnp.all(first[None] == chosen[:, None, :], axis=-1), axis=-1)
    )
    assert bool(jnp.all(out.action_log_prob <= own + 1e-6))
    assert bool(jnp.any(out.action_log_prob < own - 1e-3))


def test_action_log_prob_is_the_exact_first_step_probability(tmaze, own_trace):
    system, params = tmaze
    observations, actions = own_trace
    out = replay(system, params, observations, actions)
    first = system.policies(params)[:, 0, :]
    recorded = system.to_positional("actions", actions)
    for t in range(int(recorded.shape[0])):
        same = jnp.all(first == recorded[t][None, :], axis=-1)
        expect = jnp.log(jnp.sum(out.q_pi.probs[t] * same))
        np.testing.assert_allclose(float(out.action_log_prob[t]), float(expect), rtol=1e-6)


def test_the_true_model_explains_its_own_trace_better_than_a_blinded_one(tmaze, own_trace):
    """Workstream K's objective, before the fitting stack: Σ log P(a_t) ranks models."""
    system, params = tmaze
    observations, actions = own_trace
    cue = params.likelihood["cue"]
    flat = H.ConditionalCategorical(jnp.zeros_like(cue.log_table), cue.parent_ndims)
    blinded = dataclasses.replace(params, likelihood={**params.likelihood, "cue": flat})
    true_ll = float(replay(system, params, observations, actions).action_log_prob.sum())
    blind_ll = float(replay(system, blinded, observations, actions).action_log_prob.sum())
    assert true_ll > blind_ll


# ------------------------------------------------------------------ hybrid path


def test_the_hill_trace_replays_on_the_linear_gaussian_path():
    hill = _example("hill_lg")
    system, params = hill.make_agent()
    traj = hill.closed_loop(jax.random.PRNGKey(0), system, params, num_steps=12)
    observations = {"position": traj.ys}
    actions = {"a_state": traj.actions.astype(jnp.int32)}
    out = replay(system, params, observations, actions)
    np.testing.assert_allclose(
        np.asarray(out.beliefs["state"].loc), np.asarray(traj.means), rtol=1e-5, atol=1e-6
    )
    assert out.log_evidence is not None and out.log_evidence.shape == (12,)
    assert bool(jnp.all(jnp.isfinite(out.log_evidence)))
    assert out.action_log_prob.shape == (12,)


# ------------------------------------------------------------------- R6: masks


def test_an_all_true_mask_is_no_mask(tmaze, own_trace):
    system, params = tmaze
    observations, actions = own_trace
    T = _length(observations, actions)
    plain = replay(system, params, observations, actions)
    mask = _keyed_mask(system, jnp.ones((T, 3), bool))
    masked = replay(system, params, observations, actions, present=mask)
    _close(plain.beliefs, masked.beliefs)
    _close(plain.q_pi, masked.q_pi)


def test_a_fully_masked_tick_is_the_prior_pushed_through(tmaze, own_trace):
    """The predict-only step a dropped frame produces: posterior == prior at that tick."""
    system, params = tmaze
    observations, actions = own_trace
    T = _length(observations, actions)
    mask = _keyed_mask(system, jnp.ones((T, 3), bool).at[2].set(False))
    out = replay(system, params, observations, actions, present=mask)
    tick = lambda tree: jax.tree_util.tree_map(lambda x: x[2], tree)  # noqa: E731
    _close(tick(out.beliefs), tick(out.priors))
    #  and the garbage sitting in the record at that tick did not matter
    garbage = {k: o.at[2].set(0) for k, o in observations.items()}
    again = replay(system, params, garbage, actions, present=mask)
    _close(out.beliefs, again.beliefs)


def test_a_partially_masked_tick_is_inference_on_the_sub_model(tmaze, own_trace):
    """Masking the cue at a tick equals inferring with a System that has no cue modality."""
    system, params = tmaze
    observations, actions = own_trace
    obs_t = {k: o[1] for k, o in observations.items()}
    priors = system.initial_priors(params)
    masked, _ = system.infer_states(
        params, obs_t, priors, present=_flags(system, (True, True, False))
    )
    no_cue = dataclasses.replace(
        system, A_deps=system.A_deps[:2], modality_names=system.modality_names[:2]
    )
    no_cue_params = dataclasses.replace(
        params,
        likelihood={k: params.likelihood[k] for k in no_cue.modality_names},
        obs_targets={},
    )
    sub, _ = no_cue.infer_states(
        no_cue_params, {k: obs_t[k] for k in no_cue.modality_names}, priors
    )
    _close(masked, sub)
    #  a traced flag selects the same thing a static False skips
    traced_mask = _keyed_mask(system, jnp.asarray([True, True, False]))
    traced, _ = system.infer_states(params, obs_t, priors, present=traced_mask)
    _close(masked, traced)


def test_masks_on_the_hybrid_path_skip_the_lg_condition():
    noise = _example("noise_learning")
    system = noise.make_system()
    params = noise.make_params(jnp.asarray([[[0.1]], [[0.2]]]))
    priors = system.initial_priors(params)
    obs = {"cell": jnp.asarray(1, jnp.int32), "pose": jnp.asarray([0.3])}
    full, _ = system.infer_states(params, obs, priors)
    lg_masked, detail = system.infer_states(
        params, obs, priors, present=_flags(system, (True, False)), detail=True
    )
    #  the continuous factor kept its prior — at the static budget the switching
    #  observation's presence would have padded it to — and the categorical one
    #  still saw its own modality
    padded_prior = pad_mixture(priors["x"], system.mixture_k)
    _close(lg_masked["x"], padded_prior)
    assert not np.allclose(np.asarray(lg_masked["mode"].probs), np.asarray(priors["mode"].probs))
    assert detail.log_evidence["pose"] is None  # statically skipped
    all_masked, _ = system.infer_states(
        params, obs, priors, present=_keyed_mask(system, jnp.asarray([False, False]))
    )
    _close(all_masked["x"], padded_prior)
    np.testing.assert_allclose(
        np.asarray(all_masked["mode"].probs), np.asarray(priors["mode"].probs), atol=1e-6
    )
    #  static and traced absence agree, shape for shape
    static_all, _ = system.infer_states(params, obs, priors, present=_flags(system, (False, False)))
    _close(static_all, all_masked)
    _close(full, system.infer_states(params, obs, priors, present=_flags(system, (True, True)))[0])
    with pytest.raises(ValueError, match=r"present has unknown key \['extra'\]"):
        system.infer_states(
            params, obs, priors, present={**_flags(system, (True, True)), "extra": True}
        )


def test_masks_on_the_amortized_path_are_a_predict_only_tick():
    """Chunk two: the modalities a recognition module reads are absent together, and an
    absent tick returns the prior as the belief — statically or traced."""
    system, params = tiny_rssm(2, 1, key=jax.random.PRNGKey(0))
    n_mods = len(system.A_deps)
    priors = system.initial_priors(params)
    obs = system.from_positional("observations", (jnp.ones(2) * 0.2,) + (None,) * (n_mods - 1))
    seen, _ = system.infer_states(params, obs, priors)
    assert not all(
        np.allclose(np.asarray(a), np.asarray(b))
        for a, b in zip(
            jax.tree_util.tree_leaves(seen), jax.tree_util.tree_leaves(priors), strict=True
        )
    )
    static, _ = system.infer_states(
        params, obs, priors, present=_flags(system, (False,) + (True,) * (n_mods - 1))
    )
    _close(static, priors)
    traced, _ = system.infer_states(
        params,
        obs,
        priors,
        present=_keyed_mask(system, jnp.asarray([False] + [True] * (n_mods - 1))),
    )
    _close(traced, priors)
    #  a mask on the channels nothing reads is a no-op
    noop, _ = system.infer_states(
        params, obs, priors, present=_flags(system, (True,) + (False,) * (n_mods - 1))
    )
    _close(noop, seen)


# ------------------------------------------------------------ R7: padding & batches


def test_padding_leaves_the_valid_ticks_unchanged(tmaze, own_trace):
    system, params = tmaze
    observations, actions = own_trace
    T = _length(observations, actions)
    stacked_obs, _, _, valid = stack_traces([(observations, actions, None)])
    #  pad by hand to T + 3 to make the padding visible
    pad = 3

    def padded(x):
        return jnp.concatenate([x, jnp.zeros((pad, *x.shape[1:]), x.dtype)])

    obs_p = {k: padded(o) for k, o in observations.items()}
    acts_p = {k: padded(a) for k, a in actions.items()}
    valid_p = jnp.arange(T + pad) < T
    plain = replay(system, params, observations, actions)
    padded = replay(system, params, obs_p, acts_p, valid=valid_p)
    head = lambda tree: jax.tree_util.tree_map(lambda x: x[:T], tree)  # noqa: E731
    _close(head(padded.beliefs), plain.beliefs)
    _close(head(padded.q_pi), plain.q_pi)
    np.testing.assert_array_equal(np.asarray(padded.valid), np.asarray(valid_p))
    assert stacked_obs["location_obs"].shape[:2] == (1, T) and bool(jnp.all(valid[0]))


def test_stack_traces_and_vmap_replay_a_batch_of_different_lengths(tmaze):
    system, params = tmaze
    long = tmaze_trace(system, params, ticks=6, seed=1)
    short = tmaze_trace(system, params, ticks=4, seed=2)
    observations, actions, mask, valid = stack_traces([(*long, None), (*short, None)])
    assert actions["a_location"].shape[:2] == (2, 6)
    assert set(mask) == {"location_obs", "reward", "cue"}
    assert all(m.shape == (2, 6) for m in mask.values())
    assert valid.tolist() == [[True] * 6, [True] * 4 + [False] * 2]
    batched = jax.vmap(lambda o, a, m, v: replay(system, params, o, a, present=m, valid=v))(
        observations, actions, mask, valid
    )
    single = replay(system, params, *short)
    head = lambda tree: jax.tree_util.tree_map(lambda x: x[1, :4], tree)  # noqa: E731
    _close(head(batched.beliefs), single.beliefs)
    _close(head(batched.q_pi), single.q_pi)


def test_shape_mismatches_are_sentences(tmaze, own_trace):
    system, params = tmaze
    observations, actions = own_trace
    with pytest.raises(ValueError, match=r"observations are missing \['cue'\]"):
        replay(system, params, {k: observations[k] for k in ("location_obs", "reward")}, actions)
    with pytest.raises(ValueError, match="disagree on the trace length"):
        replay(system, params, observations, {k: a[:-1] for k, a in actions.items()})
    with pytest.raises(ValueError, match=r"present entries are \(6,\) bool"):
        replay(
            system,
            params,
            observations,
            actions,
            present=_keyed_mask(system, jnp.ones((5, 3), bool)),
        )


# --------------------------------------------------------------- R8: the document


def test_replay_document_equals_replay_of_the_built_declaration(tmaze, own_trace):

    system, params = tmaze
    observations, actions = own_trace
    observations = {("location_obs" if k == "location" else k): v for k, v in observations.items()}
    document, arrays = describe(system, params)
    via_doc = replay_document(document, observations, actions, arrays=arrays)
    direct = replay(system, params, observations, actions)
    _close(via_doc.beliefs, direct.beliefs)
    _close(via_doc.q_pi, direct.q_pi)
    _equal(via_doc.chosen, direct.chosen)


def test_an_ill_formed_document_fails_in_build_by_address(tmaze, own_trace):

    system, params = tmaze
    document, _ = describe(system, params)
    del document["factors"]["cue"]["likelihood"]  # a half-applied removal of the modality
    del document["variables"]["A_cue"], document["clamps"]["A_cue"]
    with pytest.raises(DescriptionError, match=r"'cue' is observed .* has no likelihood record"):
        replay_document(document, *own_trace)


# ------------------------------------------------------------------ R4: learning


def test_learn_along_the_trace_matches_the_eager_loop_and_moves_the_counts():
    compose = _example("compose_target_factor")
    system, params = compose.system, compose.params
    T = 5
    observations = system.from_positional(
        "observations",
        tuple(
            jnp.asarray(col, jnp.int32)
            for col in ([0, 0, 1, 0, 0], [0, 1, 0, 0, 1], [0, 0, 0, 0, 0])
        ),
    )
    actions = system.from_positional("actions", jnp.asarray([[2]] * T, jnp.int32))  # "ask"
    out = replay(system, params, observations, actions, learn=True)
    *_, eager_params = eager_loop(system, params, observations, actions, learn=True)
    _close(out.params, eager_params)
    before = params.likelihood["distance"].concentration
    after = out.params.likelihood["distance"].concentration
    assert float(after.sum()) > float(before.sum())  # Dirichlet counts absorbed evidence
    fixed = replay(system, params, observations, actions)
    assert fixed.params is None


def _old_replay_scan(system, params, observations, actions, mask, valid):
    """The scan with the parameters, the previous beliefs and the previous action all
    carried through ``lax.scan`` under ``learn=False`` — the reference for the carry
    that holds only the priors. Discrete path, ``metrics=True``, no legal/externals."""
    from hybrid_aif.objectives.hybrid import where_tree
    from hybrid_aif.planning.enumerate import sample_action

    select_key = jax.random.PRNGKey(0)
    start = system.initial_priors(params)

    def step(carry, xs):
        priors_t, prev_beliefs, prev_action, params_t, ls_t = carry
        obs_t, act_t, mask_t, valid_t = xs
        act_p = system.to_positional("actions", act_t)
        beliefs, detail = system.infer_states(
            params_t, obs_t, priors_t, detail=True, present=mask_t
        )
        q_pi, tree, policies = system._plan(
            params_t, system.to_positional("beliefs", beliefs), None, "midpoint", True, None, None
        )
        chosen = sample_action(
            select_key,
            q_pi,
            policies=policies,
            num_controls=system._control_levels(params_t),
            mode="marginal",
            selection="deterministic",
        )
        first = policies[:, 0, :]
        same = jnp.all(first == jnp.asarray(act_p, first.dtype)[None, :], axis=-1)
        action_log_prob = jnp.log(jnp.sum(q_pi.probs * same, axis=-1))
        log_evidence = -detail.free_energy
        priors_next = system.update_empirical_prior(params_t, beliefs, act_t)
        new_carry = (priors_next, beliefs, act_t, params_t, ls_t)
        carry = where_tree(valid_t, new_carry, carry)
        flags = {k: None if f is None else f & valid_t for k, f in detail.unsupported.items()}
        chosen = system.from_positional("actions", chosen)
        return carry, (beliefs, priors_t, q_pi, tree, chosen, action_log_prob, log_evidence, flags)

    carry0 = (start, start, {k: jnp.zeros_like(a[0]) for k, a in actions.items()}, params, None)
    xs = (observations, actions, mask, valid)
    (_, _, _, params_final, _), out = jax.lax.scan(step, carry0, xs)
    return params_final, out


def test_a_fixed_model_replays_as_if_its_parameters_rode_the_carry(tmaze, own_trace):
    system, params = tmaze
    observations, actions = own_trace
    T = _length(observations, actions)
    valid = jnp.arange(T) < T - 2
    mask = _keyed_mask(system, jnp.ones((T, len(observations)), bool).at[2, 1].set(False))
    for m, v in ((None, jnp.ones((T,), bool)), (mask, valid)):
        out = replay(system, params, observations, actions, present=m, valid=v)
        params_ref, ref = _old_replay_scan(system, params, observations, actions, m, v)
        got = (
            out.beliefs,
            out.priors,
            out.q_pi,
            out.trees,
            out.chosen,
            out.action_log_prob,
            out.log_evidence,
            out.unsupported,
        )
        _close(got, ref)
        _close(params, params_ref)  # the reference carry never moved the parameters
        assert out.params is None and out.learn_state is None


# ------------------------------------------------------------------- exports


def test_top_level_exports():
    for name in ("replay", "replay_document", "stack_traces", "Replay"):
        assert name in H.__all__ and getattr(H, name) is not None
    assert H.replay is replay


# ------------------------------------------- what the adversarial review found (2026-09-18)


def _noise_trace(T=5):
    noise = _example("noise_learning")
    system = noise.make_system()
    params = noise.make_params(jnp.asarray([[[0.1]], [[0.2]]]))
    cells = jnp.asarray([0, 1, 1, 0, 1][:T], jnp.int32)
    poses = jnp.linspace(-0.2, 0.4, T)[:, None]
    actions = {}  # both factors uncontrolled
    return system, params, {"cell": cells, "pose": poses}, actions


def test_a_hybrid_mixture_system_replays_with_a_padded_carry():
    """The carry seed must already sit at mixture_k, or the scan refuses after tick 0."""
    system, params, observations, actions = _noise_trace()
    out = replay(system, params, observations, actions)
    assert out.beliefs["x"].log_weights.shape == (5, system.mixture_k)
    beliefs, _, q_pi, chosen, _ = eager_loop(system, params, observations, actions)
    _close(out.beliefs["mode"], beliefs["mode"])
    _close(out.q_pi, q_pi)
    #  masked ticks on the mixture path, through the scan
    mask = _keyed_mask(system, jnp.ones((5, 2), bool).at[2].set(False))
    dropped = replay(system, params, observations, actions, present=mask)
    assert [bool(v) for v in dropped.exercised.values()] == [True, True]
    tick = lambda tree: jax.tree_util.tree_map(lambda x: x[2], tree)  # noqa: E731
    _close(tick(dropped.beliefs), tick(dropped.priors))


def test_exercised_reports_a_modality_the_trace_never_informed(tmaze, own_trace):
    system, params = tmaze
    observations, actions = own_trace
    T = _length(observations, actions)
    mask = _keyed_mask(system, jnp.ones((T, 3), bool).at[:, 2].set(False))  # cue never present
    out = replay(system, params, observations, actions, present=mask)
    assert out.exercised == {"location_obs": True, "reward": True, "cue": False}
    assert replay(system, params, observations, actions).exercised is None
    #  padding does not count as presence
    valid = jnp.arange(T) < T - 1
    late = _keyed_mask(system, jnp.zeros((T, 3), bool).at[T - 1].set(True))
    exercised = replay(system, params, observations, actions, present=late, valid=valid).exercised
    assert [bool(v) for v in exercised.values()] == [False, False, False]


def test_an_array_mask_of_the_wrong_length_is_a_sentence(tmaze, own_trace):
    system, params = tmaze
    observations, _ = own_trace
    obs_t = {k: o[0] for k, o in observations.items()}
    converter = r"from_positional\('observations', \.\.\.\)"
    with pytest.raises(TypeError, match=converter):
        system.infer_states(params, obs_t, None, present=jnp.ones(4, bool))
    with pytest.raises(TypeError, match=converter):
        system.infer_states(params, obs_t, None, present=jnp.ones(2, bool))
    with pytest.raises(ValueError, match=r"present has unknown key \['cues'\]"):
        system.infer_states(params, obs_t, None, present={"location_obs": True, "cues": False})


def test_actions_of_the_wrong_layout_are_a_sentence(tmaze, own_trace):
    system, params = tmaze
    observations, actions = own_trace
    with pytest.raises(ValueError, match=r"actions\['a_location'\] must be \(T,\)"):
        replay(system, params, observations, {k: a[:, None] for k, a in actions.items()})
    with pytest.raises(ValueError, match=r"actions \['a_location'\] are not integer"):
        replay(system, params, observations, {k: a.astype(jnp.float32) for k, a in actions.items()})
    T = _length(observations, actions)
    wrong_keys = {
        "location_obs": jnp.ones(T, bool),
        "reward": jnp.ones(T, bool),
        "cues": jnp.ones(T, bool),
    }
    with pytest.raises(ValueError, match=r"present has unknown key \['cues'\]"):
        replay(system, params, observations, actions, present=wrong_keys)


def test_a_static_absence_never_reads_the_observation(tmaze, own_trace):
    system, params = tmaze
    observations, _ = own_trace
    obs_t = {k: o[1] for k, o in observations.items()}
    priors = system.initial_priors(params)
    flags = _flags(system, (True, True, False))
    masked, _ = system.infer_states(params, obs_t, priors, present=flags)
    unread, _ = system.infer_states(params, {**obs_t, "cue": None}, priors, present=flags)
    _close(masked, unread)


def test_stack_traces_refuses_inconsistent_traces(tmaze, own_trace):
    system, params = tmaze
    observations, actions = own_trace
    two = {k: observations[k] for k in ("location_obs", "reward")}
    with pytest.raises(ValueError, match="every trace describes the same modalities"):
        stack_traces([(observations, actions, None), (two, actions, None)])
    with pytest.raises(ValueError, match=r"trace 0's observations and actions have leading axes"):
        stack_traces([(observations, {k: a[:5] for k, a in actions.items()}, None)])
    with pytest.raises(ValueError, match=r"trace 0's present entries .* are not \(6,\)"):
        stack_traces([(observations, actions, _keyed_mask(system, jnp.ones((5, 3), bool)))])


def test_where_tree_refuses_a_shape_mismatch():
    from hybrid_aif.objectives.hybrid import where_tree

    with pytest.raises(ValueError, match="differ in shape"):
        where_tree(jnp.asarray(True), (jnp.zeros(4),), (jnp.zeros(1),))
