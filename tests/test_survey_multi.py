"""The multi-rover survey, step one (``examples/survey_*.py``): two rovers that are two readers of
one field.

The one-rover declarations are those the regression pin of ``tests/test_survey_run.py`` was recorded
from (that pin passes unchanged); here: the names of the one-rover scheduler and world; the
two-rover scheduler, planned by the factorised planner (one posterior per rover over its own option
sequences, mean-field sweeps between them) and recording per rover; a world of two vehicles, which
is two worlds of one stepped side by side; the field learned within the scheduler's plan, under
which a rover's planned readings of cells the other rover's plan reads earlier are worth less; and
short closed-loop runs of the two-rover settings, with and without the scheduler, recording the
coverage of the shared field.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_survey_run import GRID, STARTS, STEPS, example, small_camera

TICKS = 3


@pytest.fixture(scope="module")
def mods():
    return {
        n: example(n) for n in ("survey_run", "survey_scheduler", "survey_model", "survey_world")
    }


def test_the_one_rover_scheduler_and_world_keep_their_names(mods):
    ss, sm, wd = mods["survey_scheduler"], mods["survey_model"], mods["survey_world"]
    grid, centres = sm.field_grid(GRID)
    system, params = ss.scheduler(grid, centres, durations=True, readings=True)
    names = system.names
    assert names.states == ("visit", "energy", "map")
    assert names.controls == ("u_option",)
    assert names.modalities == ("cell_obs", "gauge")
    assert system.action_names == ("option",)
    assert system.epistemic == ("transition:map",)
    assert system.plan_learning == ()
    assert list(params.durations) == ["u_option"]
    world, _ = wd.world_system()
    assert set(world.names.controls) == {"v"}
    assert set(world.names.emissions) == {"fix", "img", "gauge", "band"}


def test_the_two_rover_scheduler_plans_each_rover_s_options_over_one_field(mods):
    ss, sm = mods["survey_scheduler"], mods["survey_model"]
    grid, centres = sm.field_grid(GRID)
    system, params = ss.scheduler(grid, centres, rovers=2, durations=True, readings=True)
    names = system.names
    assert names.states == ("visit_1", "visit_2", "energy_1", "energy_2", "map_1", "map_2")
    assert names.controls == ("u_option_1", "u_option_2")
    assert names.modalities == ("cell_obs_1", "cell_obs_2", "gauge_1", "gauge_2")
    assert system.epistemic == ("transition:map_1", "transition:map_2")
    #  one field and one energy table, each read by both rovers' factors
    assert params.transition["map_1"].name == params.transition["map_2"].name == "field"
    assert params.transition["energy_1"].name == params.transition["energy_2"].name
    #  the durations on the first option node alone; the second is held with the node
    assert list(params.durations) == ["u_option_1"]
    #  the second rover's visit starts at the second start's cell
    start = np.argmin(
        np.linalg.norm(centres - np.asarray(mods["survey_world"].SECOND_START), axis=1)
    )
    assert int(np.argmax(params.initial_state_prior["visit_2"].logits)) == start
    #  planned by the factorised planner, one posterior per rover over its own sequences; the
    #  one-rover scheduler, one node, by the enumeration planner
    assert system.planner == "factorised"
    assert ss.scheduler(grid, centres)[0].planner == "enumeration"
    #  the joint policy set, were it enumerated: the Cartesian product of the two nodes' levels
    s, p = ss.scheduler(grid, centres, rovers=2, policy_len=2)
    joint = np.asarray(s.policies(p))
    assert joint.shape == (sm.N_OPTIONS**4, 2, 2)
    assert len({tuple(q.ravel()) for q in joint}) == joint.shape[0]
    #  the two rovers' terminal penalty covers the novelty both could earn
    assert ss.terminal_penalty(3, rovers=2) == pytest.approx(2 * ss.terminal_penalty(3))


def _map_novelty(ss, sm, plan_learning):
    """The ``transition:map_1`` leaf (midpoint) of the two-rover scheduler at the mission's start,
    horizon 3, keyed by the two rovers' option sequences: the model's term on each joint sequence,
    scored by the enumeration planner (the scheduler plans with the factorised one, which keeps no
    term per joint sequence)."""
    grid, centres = sm.field_grid()
    system, params = ss.scheduler(
        grid, centres, rovers=2, readings=True, policy_len=3, plan_learning=plan_learning
    )
    system = dataclasses.replace(system, planner="enumeration")
    _, metrics = system.compiled().infer_policies(params, dict(params.initial_state_prior))
    leaf = metrics["infogain"]["transition:map_1"]
    mid = (np.asarray(leaf.lower) + np.asarray(leaf.upper)) / 2
    policies = np.asarray(system.policies(params))

    def at(first, second):
        want = np.stack([first, second], axis=-1)
        return float(mid[np.nonzero((policies == want).all(axis=(1, 2)))[0][0]])

    return at


def test_with_the_field_learned_in_the_plan_a_rover_s_readings_of_the_other_s_cells_earn_less(
    mods,
):
    """Rover 1 surveys sector 4 for three options; rover 2 starts in sector 4 and either surveys
    it too or leaves for sector 1. Read frozen, rover 1's novelty does not depend on rover 2's
    plan. Learned within the plan, rover 1's planned readings of the cells rover 2's plan reads at
    earlier steps are worth less (a step reads the cell the rover is in before it, so a step's
    absorption lowers the novelty of later steps, of either rover). The gap is small (about 0.17
    nats of 27): a rover 2 that leaves still reads its start cell and the path out of sector 4
    first, so it takes away some of the same cells' novelty either way."""
    ss, sm = mods["survey_scheduler"], mods["survey_model"]
    sector_4, sector_1 = [3, 3, 3], [0, 0, 0]
    frozen = _map_novelty(ss, sm, ())
    learned = _map_novelty(ss, sm, ("parameter:field",))
    assert frozen(sector_4, sector_4) == pytest.approx(frozen(sector_4, sector_1), rel=1e-6)
    assert learned(sector_4, sector_4) < learned(sector_4, sector_1) - 0.1
    #  and its own repeated readings are worth less than frozen either way (the urn)
    assert learned(sector_4, sector_1) < frozen(sector_4, sector_1)


def test_a_world_of_two_vehicles_is_two_worlds_of_one(mods):
    wd = mods["survey_world"]
    two = wd.world_stepper(vehicles=2)
    one = wd.world_stepper()
    two_sys, _ = wd.world_system(vehicles=2)
    one_sys, _ = wd.world_system()
    key = jax.random.PRNGKey(3)
    commands = jnp.asarray([[[0.8, 0.3], [0.5, -0.6]], [[1.0, 0.0], [0.2, 1.0]]], jnp.float32)
    out2, carry2, _ = two(two_sys, key, None, None, None)
    ones = []
    for k in (1, 2):
        given = {"x": jnp.asarray(wd.STARTS[k - 1], jnp.float32)}
        ones.append(list(one(one_sys, jax.random.fold_in(key, k), None, given, None)))
    for t, command in enumerate(commands):
        k_t = jax.random.fold_in(key, 10 + t)
        obs = {"v_1": command[0], "v_2": command[1]}
        for k in (1, 2):
            want_out, want_state = ones[k - 1][0], ones[k - 1][1]
            for port, value in want_out.items():
                np.testing.assert_array_equal(out2[f"{port}_{k}"], value, err_msg=f"{port}_{k}")
            for name, value in want_state.items():
                np.testing.assert_array_equal(carry2[f"{name}_{k}"], value, err_msg=f"{name}_{k}")
            ones[k - 1][:2] = one(
                one_sys, jax.random.fold_in(k_t, k), {"v": command[k - 1]}, want_state, None
            )[:2]
        out2, carry2, _ = two(two_sys, k_t, obs, carry2, None)
    assert not np.array_equal(carry2["x_1"], carry2["x_2"])


def _assert_per_rover_record(slow, sm, horizon):
    """A decision's record of the two-rover scheduler (``joint_decision_record``) holds, per
    rover, the sequence its posterior puts highest with its term tree, ``G`` and probability, the
    best ``G`` and the probability of each first option over its own sequences, the sequence count
    of each rover's posterior, and the last sweep's movement."""
    n = sm.N_OPTIONS
    assert slow["policies"] == [n**horizon, n**horizon]
    assert np.isfinite(slow["movement"]) and slow["movement"] >= 0.0
    assert len(slow["chosen"]) == 2 and len(slow["efe_by_option"]) == 2
    for k in range(2):
        chosen = slow["chosen"][k]
        assert len(chosen["options"]) == horizon
        assert all(0 <= o < n for o in chosen["options"])
        assert chosen["G"][0] <= chosen["G"][1]
        assert 0.0 < chosen["q_pi"] <= 1.0
        leaves = [name for leaves in chosen["terms"].values() for name in leaves]
        assert f"transition:mission.map_{k + 1}" in leaves, leaves
        q = slow["q_by_option"][k]
        assert len(q) == n and sum(q) == pytest.approx(1.0, abs=1e-5)
        efe = np.asarray(slow["efe_by_option"][k])
        assert efe.shape == (n,) and np.all(np.isfinite(efe))
        #  the chosen sequence's own G is no better than the best of its first option
        first = chosen["options"][0]
        assert efe[first] <= np.mean(chosen["G"]) + 1e-3


@pytest.mark.parametrize("sid", ["two-rover-two-level", "two-rover-one-level"])
def test_two_rovers_run_and_record_the_coverage_of_their_shared_field(mods, sid):
    sr, ss, wd = mods["survey_run"], mods["survey_scheduler"], mods["survey_world"]
    setting = dict(next(s for s in sr.settings_multi() if s["id"] == sid), grid=GRID)
    if setting["tiers"] == 2:
        penalty = ss.terminal_penalty(2, setting["field_strength"], rovers=2)
        setting.update(slow_ticks=2, horizon=2, terminal_penalty=penalty)
    result = sr.run_setting(setting, TICKS, small_camera(wd.IMG_PIXELS), STEPS, STARTS)
    rows = result["ticks"]
    assert [r["t"] for r in rows] == list(range(TICKS))
    assert set(result["rovers"]) == {"rover1", "rover2"}
    assert result["grouped"] is True
    #  a rover counts one terrain reading a round from its second round on, into the one field
    for t, r in enumerate(rows):
        assert sum(r["counts"]) == pytest.approx(2.0 * t, abs=1e-3)
    coverage = [r["coverage"] for r in rows]
    assert coverage == sorted(coverage) and coverage[-1] > 0
    first, second = (result["rovers"][k] for k in ("rover1", "rover2"))
    assert first[0]["true_x"][:2] == pytest.approx(list(wd.START), abs=1e-5)
    assert second[0]["true_x"][:2] == pytest.approx(list(wd.SECOND_START), abs=1e-5)
    if setting["tiers"] == 2:
        decisions = [r for r in rows if r["slow"]["decision"]]
        assert [r["t"] for r in decisions] == [0, 2]
        _assert_per_rover_record(decisions[0]["slow"], sm=mods["survey_model"], horizon=2)
        #  both rovers' options held between decisions
        assert rows[0]["options"] == rows[1]["options"]
        assert [r["option"] for r in first] == [r["options"][0] for r in rows]
        assert [r["option"] for r in second] == [r["options"][1] for r in rows]
