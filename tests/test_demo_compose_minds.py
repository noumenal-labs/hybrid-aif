"""``examples/compose_minds.py``, the notebook's composition section: the claims its prose makes."""

import importlib.util
import pathlib

import jax
import numpy as np

_PATH = pathlib.Path(__file__).resolve().parents[1] / "examples" / "compose_minds.py"
_SPEC = importlib.util.spec_from_file_location("compose_minds", _PATH)
cm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cm)

KEY = jax.random.PRNGKey(0)


def test_the_merged_system_is_bitwise_the_longhand_declaration():
    assert cm.identity_check() == {
        "static fields": True,
        "parameter arrays": True,
        "beliefs": True,
        "policy scores": True,
    }


def test_the_policy_counts():
    assert cm.policy_counts() == {"merged": 81, "coupled": 9}


def test_from_the_symmetric_start_the_merged_mind_splits_at_tick_two():
    a, b = cm.merged_episode(KEY, cm.SYMMETRIC)
    np.testing.assert_array_equal(a[:3], [2, 1, 0])
    np.testing.assert_array_equal(b[:3], [2, 3, 4])
    assert cm.split_tick(a, b) == 2
    by_tick, _ = cm.split_by_tick(cm.merged_episode, 32, start=cm.SYMMETRIC)
    np.testing.assert_array_equal(by_tick, [0, 0, 1, 1, 1, 1, 1, 1, 1])


def test_deterministic_coupled_minds_from_the_symmetric_start_never_part():
    a, b = cm.coupled_episode(KEY, cm.SYMMETRIC)
    np.testing.assert_array_equal(a, b)
    assert cm.split_tick(a, b) is None
    by_tick, _ = cm.split_by_tick(cm.coupled_episode, 32, start=cm.SYMMETRIC)
    np.testing.assert_array_equal(by_tick, np.zeros(9))


def test_each_coupled_mind_wants_a_dock_and_values_the_two_equally():
    r, rp = cm.robot(*cm.SYMMETRIC)
    obs = {"me_obs": 2, "other_obs": 2, "status": 2}
    beliefs, _ = r.infer_states(rp, {k: jax.numpy.asarray(v) for k, v in obs.items()})
    q, _ = r.infer_policies(rp, beliefs)
    probs = np.asarray(q.probs)
    plans = [tuple(p) for p in np.asarray(r.policies(rp))[:, :, 0].tolist()]
    left_left, right_right = probs[plans.index((0, 0))], probs[plans.index((2, 2))]
    np.testing.assert_allclose(left_left, right_right)
    assert round(float(left_left), 3) == 0.455  # the figure the section prints and cites


def test_stochastic_coupled_minds_from_the_symmetric_start():
    by_tick, first = cm.split_by_tick(
        cm.coupled_episode, 32, start=cm.SYMMETRIC, selection="stochastic"
    )
    assert int((first >= 0).sum()) == 32
    assert int((first == 2).sum()) == 15  # at tick 2, as early as the merged mind
    np.testing.assert_allclose(by_tick, np.asarray([0, 0, 15, 15, 27, 29, 31, 31, 32]) / 32)


def test_the_merged_mind_splits_its_mass_between_the_two_assignments():
    comp, cp = cm.merged()
    obs = {f"{r}.{m}": 2 for r in "ab" for m in ("me_obs", "other_obs", "status")}
    beliefs, _ = comp.infer_states(cp, {k: jax.numpy.asarray(v) for k, v in obs.items()})
    q, _ = comp.infer_policies(cp, beliefs)
    for marginal in comp.action_marginals(cp, q).values():
        np.testing.assert_allclose(np.asarray(marginal.probs), [0.5, 0.0, 0.5], atol=1e-3)


def test_acting_on_marginals_the_merged_mind_behaves_as_the_coupled_pair():
    a, b = cm.merged_episode(KEY, cm.SYMMETRIC, mode="marginal")
    np.testing.assert_array_equal(a, b)
    by_tick, _ = cm.split_by_tick(cm.merged_episode, 32, start=cm.SYMMETRIC, mode="marginal")
    np.testing.assert_array_equal(by_tick, np.zeros(9))
    by_tick, _ = cm.split_by_tick(
        cm.merged_episode, 32, start=cm.SYMMETRIC, selection="stochastic", mode="marginal"
    )
    np.testing.assert_allclose(by_tick, np.asarray([0, 0, 12, 12, 27, 30, 32, 32, 32]) / 32)


def test_a_stochastic_merged_mind_draws_a_coordinated_policy():
    by_tick, _ = cm.split_by_tick(cm.merged_episode, 32, start=cm.SYMMETRIC, selection="stochastic")
    np.testing.assert_array_equal(by_tick, [0, 0, 1, 1, 1, 1, 1, 1, 1])


def test_from_the_asymmetric_start_both_split_at_tick_two():
    assert cm.split_tick(*cm.merged_episode(KEY, cm.ASYMMETRIC)) == 2
    assert cm.split_tick(*cm.coupled_episode(KEY, cm.ASYMMETRIC)) == 2
