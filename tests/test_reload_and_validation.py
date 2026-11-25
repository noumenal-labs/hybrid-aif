"""Tests for register_transition(..., replace=True) reloads and parent_ndims type validation."""

import importlib.util
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import pytest

import hybrid_aif.description as desc_mod
from hybrid_aif.description import register_transition, registered_transitions
from hybrid_aif.distributions import ConditionalCategorical, ConditionalDirichlet


def test_register_transition_replace_rebinds_reloaded_class(monkeypatch):
    monkeypatch.setattr(desc_mod, "_TRANSITIONS", dict(desc_mod._TRANSITIONS))

    class StepV1(eqx.Module):
        def __call__(self, x, controls):
            return x

    class StepV2(eqx.Module):
        def __call__(self, x, controls):
            return x

    register_transition("reload_test.step", StepV1)
    with pytest.raises(ValueError, match="already registered"):
        register_transition("reload_test.step", StepV2)

    register_transition("reload_test.step", StepV2, replace=True)
    assert registered_transitions()["reload_test.step"] is StepV2


def test_survey_world_reload_updates_registered_vehicle_step(monkeypatch):
    monkeypatch.setattr(desc_mod, "_TRANSITIONS", dict(desc_mod._TRANSITIONS))
    path = Path(__file__).resolve().parents[1] / "examples" / "survey_world.py"
    for idx in range(2):
        spec = importlib.util.spec_from_file_location(f"_survey_world_reload_{idx}", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    assert registered_transitions()["survey.vehicle"] is mod.VehicleStep


@pytest.mark.parametrize("bad_ndims", [(3, 2), [2], True, 1.5])
def test_tabular_conditional_rejects_non_int_parent_ndims(bad_ndims):
    with pytest.raises(TypeError, match="parent_ndims must be an int count"):
        ConditionalCategorical(jnp.zeros((2, 3, 2)), parent_ndims=bad_ndims)
    with pytest.raises(TypeError, match="parent_ndims must be an int count"):
        ConditionalDirichlet(jnp.ones((2, 3, 2)), parent_ndims=bad_ndims)
