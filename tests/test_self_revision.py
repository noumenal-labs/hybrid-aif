"""Self-revision: ρ and the imitation mask against the oracle (P1′ chunk 2c).

**The oracle is the code** (A10 disposition 7):
``raif/embodied/run/callbacks.py::lookback_on_episode``.
:func:`hybrid_aif.reference.raif.revise.pref_labels` is a line-for-line port of it, and
this file pins the port two ways — hand-computed cases derived from *that code*
for each of its clauses, and :func:`test_the_port_matches_the_numpy_reference`,
which runs the reference implementation verbatim against ours over random
episodes. ``amortization-p0-addendum.md`` §4 describes the same pass; where it
disagreed (the failure decay's exponent) the file won and §4 was corrected.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.reference.raif.columns import raif_episode
from hybrid_aif.reference.raif.revise import ALPHA, BETA, EPSILON, label, positive_gate, pref_labels
from hybrid_aif.train.buffers import Episode

KEY = jax.random.PRNGKey(0)


def _episode(
    is_success, has_succeeded=None, can_self_imitate=None, t: int | None = None
) -> Episode:
    """One episode carrying only the flags the pass reads; obs/action/reward are filler."""
    is_success = np.asarray(is_success, bool)
    t = is_success.size if t is None else t
    if has_succeeded is None:
        has_succeeded = np.maximum.accumulate(is_success)
    if can_self_imitate is None:
        can_self_imitate = np.zeros(t, bool)
    return raif_episode(
        obs=jnp.zeros((t, 2)),
        action=jnp.zeros((t, 1)),
        reward=jnp.zeros((t,)),
        is_success=is_success,
        has_succeeded=np.asarray(has_succeeded, bool),
        can_self_imitate=np.asarray(can_self_imitate, np.float32),
        is_first=np.eye(t, dtype=bool)[0],
        is_last=np.eye(t, dtype=bool)[-1],
    )


# ------------------------------------------------------- the reference, run verbatim


def reference(data: dict, disc=ALPHA, thr=EPSILON, fail=BETA) -> dict:
    """``lookback_on_episode`` copied out of ``callbacks.py``, numpy and all.

    Deliberately *not* refactored — not even the redundant ``neg_rates`` line or
    the in-place mutation order — so that a diff between this function and the
    file is a diff a human can check by eye. ``data`` is mutated in place, as
    theirs is, and returned for convenience.
    """
    T = len(data["is_first"])  # noqa: N806 — their name
    data["can_self_imitate"] = data["can_self_imitate"] * data["has_succeeded"][-1]
    carry = np.maximum(data["can_self_imitate"][-1], 0.0)
    imitation_mask = [carry]
    for t in range(T - 1, 0, -1):
        carry = np.maximum(carry * disc, data["is_success"][t])
        carry = carry * (carry >= thr).astype(np.float32)
        imitation_mask.insert(0, carry)
    imitation_mask = np.stack(imitation_mask)
    final_imitation_mask = np.clip(data["can_self_imitate"] + imitation_mask, 0, 1)
    positive_mask = np.concatenate([data["has_succeeded"][:1], final_imitation_mask[:-1]], 0)
    failure_decays = np.asarray([fail for i in range(T)]).cumprod(0)[::-1] / fail
    failure_decays = failure_decays * (1 - data["has_succeeded"][-1])
    neg_rates = -failure_decays
    neg_rates = neg_rates * (1 - data["can_self_imitate"])
    data["can_self_imitate"] = final_imitation_mask
    data["pref_label"] = positive_mask + neg_rates
    data["pref_label"][0] = 0
    data["pref_label"] = data["pref_label"].clip(-1, 1)
    return data


def test_the_port_matches_the_numpy_reference():
    """The port, against ``lookback_on_episode`` itself, over random episodes.

    Sixty-four episodes of lengths 1–12 with independently random flags — including
    the shapes hand-written cases do not reach: multiple successes, successes that
    are not at the end, expert flags scattered through a failure, ``has_succeeded``
    inconsistent with ``is_success`` (their wrapper's flags are separate keys, so
    nothing in the pass couples them). Both outputs are compared, because the
    callback writes both keys back.
    """
    rng = np.random.default_rng(0)
    for trial in range(64):
        t = int(rng.integers(1, 13))
        success = rng.random(t) < 0.25
        data = {
            "is_first": np.eye(t, dtype=bool)[0],
            "is_success": success.astype(np.float32),
            "has_succeeded": np.maximum.accumulate(success).astype(np.float32),
            "can_self_imitate": (rng.random(t) < 0.4).astype(np.float32),
        }
        ep = _episode(
            success,
            has_succeeded=data["has_succeeded"].astype(bool),
            can_self_imitate=data["can_self_imitate"],
        )
        want = reference({k: v.copy() for k, v in data.items()})
        got = pref_labels(ep)
        np.testing.assert_allclose(
            np.asarray(got.pref_label), want["pref_label"], atol=1e-6, err_msg=f"trial {trial}"
        )
        np.testing.assert_allclose(
            np.asarray(got.can_self_imitate),
            want["can_self_imitate"],
            atol=1e-6,
            err_msg=f"trial {trial}",
        )


# ------------------------------------------------------------------ the success branch


def test_the_backward_carry_credits_the_step_before_a_success_and_the_shift_is_forward():
    """A success at the last step of a 5-step episode, no expert flags.

    Their loop seeds the carry with the (success-gated) expert flag at ``T−1`` — 0
    here — and then credits index ``t−1`` from ``is_success[t]``::

        mask[4] = the seed                = 0
        mask[3] = is_success[4]           = 1
        mask[2] = α·mask[3]              = 0.4
        mask[1] = α·mask[2]              = 0.16
        mask[0] = α·mask[1]              = 0.064

    The shift is then **forward** — ``positive_mask[t] = final_mask[t−1]``, slot 0
    from ``has_succeeded[0]`` and immediately overridden to 0 — giving::

        rho  = [0(set), 0.064, 0.16, 0.4, 1.0]

    So the credit rises monotonically into the goal step, and the goal step itself
    carries ρ = 1. (This is the direction FLAG 11 got wrong before the file was
    read: the backward reading put the 1 one step early and left the goal at 0.)
    """
    got = pref_labels(_episode([0, 0, 0, 0, 1]))
    expected = [0.0, ALPHA**3, ALPHA**2, ALPHA, 1.0]
    np.testing.assert_allclose(np.asarray(got.pref_label), expected, atol=1e-6)


def test_the_epsilon_floor_quantizes_a_far_upstream_state_to_exactly_neutral():
    """With ``thr`` above the carry's value, credit is 0 rather than tiny.

    The claim is that the support is *finite*: a state far enough upstream of the
    goal is exactly neutral, not faintly preferred. ``epsilon = 0.5`` puts the floor
    between ``α = 0.4`` and ``1``, so only the credit at the step *before* the goal
    survives the recurrence — and after the forward shift only the goal step is
    positive.
    """
    got = pref_labels(_episode([0, 0, 0, 0, 1]), epsilon=0.5)
    np.testing.assert_allclose(np.asarray(got.pref_label), [0.0, 0.0, 0.0, 0.0, 1.0], atol=1e-6)


def test_the_first_step_is_never_labelled():
    """``pref_label[0] = 0``, and it is a *separate* statement for a reason.

    The forward shift fills slot 0 from ``has_succeeded[0]``, which is 1 whenever
    the episode succeeds at its very first step — so without the override that slot
    would be labelled. Asserted on exactly that case: the reset state is neither
    earned nor blamed.
    """
    ep = _episode([1, 0, 0, 0])
    assert float(ep.extras["has_succeeded"][0]) == 1.0  # the shift *would* put 1 at slot 0
    got = pref_labels(ep)
    assert float(got.pref_label[0]) == 0.0


def test_the_goal_step_keeps_its_credit_through_the_shift():
    """A mid-episode success, so the shift's direction is visible on both sides.

    ``is_success = [0, 0, 1, 0, 0]``: the carry gives ``mask[1] = 1`` and nothing
    after it (the recurrence only ever looks forward one step at a time from a
    later success), so the forward shift puts the 1 at index 2 — the goal step.
    """
    got = pref_labels(_episode([0, 0, 1, 0, 0]))
    rho = np.asarray(got.pref_label)
    assert rho[2] == pytest.approx(1.0, abs=1e-6)
    assert rho[1] == pytest.approx(ALPHA, abs=1e-6)  # one step of decay upstream
    assert rho[3] == pytest.approx(0.0, abs=1e-6)  # nothing is carried past a success


# ------------------------------------------------------------------ the expert branch


def test_expert_flags_are_gated_by_whether_the_episode_succeeded():
    """``can_self_imitate *= has_succeeded[-1]``, destructively and first.

    Same flags twice: with a success in the episode every expert step is positive
    (after the shift), and without one they contribute nothing — and because the
    failure branch's own expert exemption reads the *gated* flag, a failed expert
    episode is condemned like any other rather than exempted.
    """
    flags = [1, 1, 1, 1]
    won = pref_labels(_episode([0, 0, 0, 1], can_self_imitate=flags)).pref_label
    lost = pref_labels(_episode([0, 0, 0, 0], can_self_imitate=flags)).pref_label
    np.testing.assert_allclose(np.asarray(won), [0.0, 1.0, 1.0, 1.0], atol=1e-6)
    #  a failed expert episode: the gate zeroed the flags, so `neg` is NOT exempt
    np.testing.assert_allclose(np.asarray(lost), [0.0, -(BETA**2), -BETA, -1.0], atol=1e-6)


def test_the_returned_can_self_imitate_is_the_propagated_mask():
    """``data["can_self_imitate"] = final_imitation_mask`` — the overwrite.

    This is the correction the port surfaced, and it is what the actor's BC term is
    gated by: a step the agent authored itself acquires imitation weight from a
    later success, so "self-imitation" is mechanical rather than a manner of
    speaking. Here nothing is flagged expert and the episode still ends up with
    weight on its last four steps.
    """
    got = pref_labels(_episode([0, 0, 0, 0, 1]))
    mask = np.asarray(got.can_self_imitate)
    np.testing.assert_allclose(mask, [ALPHA**3, ALPHA**2, ALPHA, 1.0, 0.0], atol=1e-6)
    assert mask.dtype == np.float32  # a weight in [0, 1], not a flag


def test_a_flagged_step_keeps_full_weight_and_the_mask_is_clipped():
    """``clip(can_self_imitate + imitation_mask, 0, 1)``: the two sources add, then clip."""
    got = pref_labels(_episode([0, 0, 0, 1], can_self_imitate=[0, 0, 1, 1]))
    np.testing.assert_allclose(
        np.asarray(got.can_self_imitate), [ALPHA**2, ALPHA, 1.0, 1.0], atol=1e-6
    )


# ------------------------------------------------------------------ the failure branch


def test_a_failure_is_condemned_most_where_it_ended_up():
    """``failure_decays[t] = fail**(T−1−t)``, so the **last** step gets −1.

    Their construction is ``[fail]*T``'s cumprod, reversed, divided by ``fail`` —
    i.e. exponents ``T−1 … 0``, not ``T … 1``. On a 4-step failure with no expert
    flags::

        neg = [−β³, −β², −β, −1] = [−0.027, −0.09, −0.3, −1.0]
        rho = [0(set), −0.09, −0.3, −1.0]

    This is the one clause where ``amortization-p0-addendum.md`` §4's prose
    (``−β^(T−t)``) contradicted the file; §4 is corrected with the citation.
    """
    got = pref_labels(_episode([0, 0, 0, 0]))
    expected = [0.0, -(BETA**2), -BETA, -1.0]
    np.testing.assert_allclose(np.asarray(got.pref_label), expected, atol=1e-6)


def test_the_failure_branch_is_off_entirely_when_the_episode_succeeded():
    """``failure_decays *= (1 − has_succeeded[-1])``: one episode-level switch."""
    got = pref_labels(_episode([0, 0, 1]))
    assert float(jnp.min(got.pref_label)) >= 0.0


def test_the_expert_exemption_on_the_failure_branch_is_a_no_op():
    """``neg_rates *= (1 − can_self_imitate)`` reads the **gated** flag, so it cannot fire.

    In a failed episode the gate has already zeroed every expert flag, and in a
    successful one ``failure_decays`` is already zero — so this line, which §4
    describes as "zeroed … on expert steps", changes nothing either way. Ported
    verbatim and pinned here rather than dropped: it is what the file does, and a
    later change to the gate would give it teeth.
    """
    flagged = pref_labels(_episode([0, 0, 0, 0], can_self_imitate=[0, 1, 1, 0])).pref_label
    plain = pref_labels(_episode([0, 0, 0, 0])).pref_label
    np.testing.assert_allclose(np.asarray(flagged), np.asarray(plain), atol=0)


def test_rho_stays_in_the_unit_interval_by_construction():
    """``clip(positive_mask + neg_rates, −1, 1)`` — asserted over a random flag sweep."""
    keys = jax.random.split(KEY, 32)
    for k in keys:
        flags = jax.random.bernoulli(k, 0.3, (7,))
        got = pref_labels(_episode(np.asarray(flags), can_self_imitate=np.asarray(flags)))
        assert float(jnp.min(got.pref_label)) >= -1.0
        assert float(jnp.max(got.pref_label)) <= 1.0
        assert float(jnp.min(got.can_self_imitate)) >= 0.0
        assert float(jnp.max(got.can_self_imitate)) <= 1.0


def test_a_single_step_episode_degenerates_cleanly():
    """``T = 1``: the recurrence has no steps to run and ``pref_label[0] = 0`` is all
    of it. Pinned because the scan is skipped there, and the reference agrees."""
    for success in ([0], [1]):
        ep = _episode(success)
        got = pref_labels(ep)
        assert float(got.pref_label[0]) == 0.0
        want = reference(
            {
                "is_first": np.ones(1, bool),
                "is_success": np.asarray(success, np.float32),
                "has_succeeded": np.asarray(success, np.float32),
                "can_self_imitate": np.zeros(1, np.float32),
            }
        )
        np.testing.assert_allclose(
            np.asarray(got.can_self_imitate), want["can_self_imitate"], atol=1e-6
        )


# ------------------------------------------------------- the gate, and the schedule


def test_the_gate_is_a_hard_mask_on_the_sign():
    """``max(0, sgn ρ)``: the CRSPP KL is paid on positive states and told nothing
    about the rest, which is what makes it a model of *preferences* (paper Eq. 5)."""
    rho = jnp.asarray([-1.0, -0.01, 0.0, 0.01, 1.0])
    np.testing.assert_allclose(np.asarray(positive_gate(rho)), [0, 0, 0, 1, 1], atol=0)


def test_label_fills_both_columns_and_they_travel_with_a_stored_window():
    """The schedule: the pass runs once per *finished* episode, before the write.

    ``label`` is that call, and it writes **both** keys the callback writes — ρ and
    the overwritten imitation mask — so a stored window shorter than the episode
    still carries what the whole episode implied, which is precisely what
    recomputing at training time would lose.
    """
    ep = _episode([0, 0, 0, 0, 1], can_self_imitate=[1, 0, 0, 0, 0])
    labelled = label(ep)
    want = pref_labels(ep)
    np.testing.assert_allclose(
        np.asarray(labelled.extras["pref_label"]), np.asarray(want.pref_label), atol=0
    )
    np.testing.assert_allclose(
        np.asarray(labelled.extras["can_self_imitate"]), np.asarray(want.can_self_imitate), atol=0
    )
    assert not np.allclose(
        np.asarray(labelled.extras["can_self_imitate"]), np.asarray(ep.extras["can_self_imitate"])
    )

    from hybrid_aif.train.buffers import fit_length

    window = fit_length(labelled, 3)
    for column in ("pref_label", "can_self_imitate"):
        np.testing.assert_allclose(
            np.asarray(window.extras[column]),
            np.asarray(labelled.extras[column])[:3],
            atol=0,
        )


def test_pref_labels_is_jit_able_and_vmap_able():
    """A pure function of the flags: the backward carry is a ``lax.scan``, so the pass
    can run inside a scanned collection loop or over a whole batch at once."""
    ep = _episode([0, 0, 1, 0])
    once = jax.jit(pref_labels)(ep)
    np.testing.assert_allclose(
        np.asarray(once.pref_label), np.asarray(pref_labels(ep).pref_label), atol=1e-7
    )
    batched = jax.vmap(pref_labels)(jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), ep))
    assert batched.pref_label.shape == (2, 4)
    assert batched.can_self_imitate.shape == (2, 4)


def test_the_hyperparameters_are_validated():
    ep = _episode([0, 1])
    for kwargs in ({"alpha": 1.0}, {"beta": 1.5}, {"beta": 0.0}, {"epsilon": -1.0}):
        with pytest.raises(ValueError):
            pref_labels(ep, **kwargs)
