"""Acceptance-fix 3: SystemParams/System defaults for absent machinery.

An open-loop model (no modalities) used to need ``likelihood={}, obs_targets={},
state_targets={}, terminal_targets={}`` boilerplate. ``likelihood`` and the three
designation dicts now default (fresh dicts per instance via
``default_factory``), and ``System.A_deps`` defaults to ``()``. Existing
keyword constructions keep their meaning.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np

from hybrid_aif.distributions import Categorical
from hybrid_aif.system import System, SystemParams
from tests.test_hybrid_switching_system import (
    HORIZON,
    K_BUDGET,
    initial_beliefs,
    make_params,
    make_system,
)


def _beliefs():
    """The fixture's initial beliefs, keyed by factor as the faces take them."""
    beliefs = initial_beliefs()
    if isinstance(beliefs, Mapping):
        return beliefs
    return make_system().from_positional("beliefs", beliefs)


def _minimal_params():
    boilerplate = make_params()  # the fixture's full likelihood={}/empty-dict construction
    return SystemParams(
        transition=boilerplate.transition,
        initial_state_prior=boilerplate.initial_state_prior,
        state_targets=boilerplate.state_targets,
    )


def test_defaults_fill_absent_machinery():
    params = _minimal_params()
    assert params.likelihood == {}
    assert params.obs_targets == {} and params.terminal_targets == {}
    assert params.policy_prior is None


def test_aif_letter_aliases():
    params = make_params()
    assert params.A is params.likelihood
    assert params.B is params.transition
    assert params.D is params.initial_state_prior


def test_default_dicts_are_per_instance():
    p1, p2 = _minimal_params(), _minimal_params()
    p1.obs_targets["oops"] = Categorical(logits=jnp.zeros(2))
    assert p2.obs_targets == {}  # default_factory, not a shared mutable default


def test_minimal_construction_plans_identically_to_boilerplate():
    system = make_system()
    q_min, tree_min = system.infer_policies(_minimal_params(), _beliefs())
    q_full, tree_full = system.infer_policies(make_params(), _beliefs())
    np.testing.assert_array_equal(np.asarray(q_min.logits), np.asarray(q_full.logits))
    for term, blocks in tree_full.items():
        for block, iv in blocks.items():
            assert jnp.array_equal(tree_min[term][block].lower, iv.lower), (term, block)


def test_system_a_deps_defaults_to_empty():
    no_a_deps = System(
        B_deps=((), (0,)),
        controlled=(True, False),
        policy_len=HORIZON,
        mixture_k=K_BUDGET,
        modality_names=(),
        factor_names=("mode", "x"),
        terms=("uncertainty:x",),  # as the fixture it is compared with
    )
    assert no_a_deps.A_deps == ()
    q_pi, _ = no_a_deps.infer_policies(_minimal_params(), _beliefs())
    q_ref, _ = make_system().infer_policies(_minimal_params(), _beliefs())
    np.testing.assert_array_equal(np.asarray(q_pi.logits), np.asarray(q_ref.logits))
