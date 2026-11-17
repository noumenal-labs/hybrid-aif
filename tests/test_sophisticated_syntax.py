"""``System(sophisticated=...)``: the alternative syntax for the two halves of sophisticated
planning, in ``epistemic``'s selector grammar. An ``obs:<modality>`` key branches the plan on that
modality (``branch_on``); a ``transition:``/``likelihood:``/``parameter:`` key carries that
table's counts within the plan (``plan_learning``). The two fields are the canonical form.

Pinned by: the lowering (the canonical fields equal the two-field declaration's, and on the fused
all-tabular path and the hybrid path the plan's ``q(π)`` and ``G`` are bitwise equal); the guards
(both syntaxes, ``True``, a pattern, an unknown ``obs`` key, an unknown first segment); the
description (it writes the canonical fields alone and rebuilds them); ``merge`` and ``subsystem``
(the composite of two Systems declared through ``sophisticated=`` is the canonical declarations'
composite); ``dataclasses.replace`` and ``eqx.tree_at`` (the canonical fields follow
``sophisticated``, and a replace of a canonical field alone is refused).

Design: docs/design/parameters-within-the-plan.md A1.
"""

import dataclasses
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.composition.merge import merge, subsystem
from hybrid_aif.description import build, describe
from hybrid_aif.distributions import Categorical, ConditionalCategorical, ConditionalDirichlet
from hybrid_aif.objectives.base import total
from hybrid_aif.system import System, SystemParams
from tests.test_plan_learning import _tabular


def _fused(**kw):
    """An all-tabular System: ``s`` has a learnable transition over ``c`` and the control ``u``,
    ``o`` a learnable likelihood of ``s``; planned by the sophisticated planner."""
    system = System(
        A_deps=((0,),),
        B_deps=((1, 2), (1,), ()),
        controls=(2,),
        action_names=("a",),
        state_names=("s", "c"),
        control_names=("u",),
        modality_names=("o",),
        policy_len=2,
        epistemic=("transition:s", "likelihood:o"),
        **{"planner": "sophisticated", **kw},
    )
    params = SystemParams(
        likelihood={
            "o": ConditionalDirichlet(
                concentration=jnp.asarray([[2.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 3.0]]),
                parent_ndims=1,
            )
        },
        transition={
            "s": ConditionalDirichlet(
                concentration=jnp.full((3, 2, 2), 0.2).at[0, 0, 0].set(3.0), parent_ndims=2
            ),
            "c": ConditionalCategorical(
                log_table=jnp.log(jnp.asarray([[0.7, 0.4], [0.3, 0.6]])), parent_ndims=1
            ),
        },
        initial_state_prior={
            "s": Categorical(logits=jnp.zeros(3)),
            "c": Categorical(logits=jnp.log(jnp.asarray([0.8, 0.2]))),
        },
        obs_targets={"o": Categorical(logits=jnp.log(jnp.asarray([0.5, 0.3, 0.2])))},
    )
    return system, params


SOPH = ("obs:o", "transition:s", "likelihood:o")
CANON = {"branch_on": ("o",), "plan_learning": ("transition:s", "likelihood:o")}


def _canonical(system):
    return tuple(system.branch_on), system.plan_learning


def _keys(tree) -> set[str]:
    """Every key of a JSON document, at any depth."""
    if isinstance(tree, dict):
        return set(tree) | set().union(*(_keys(v) for v in tree.values()))
    if isinstance(tree, list):
        return set().union(*(_keys(v) for v in tree))
    return set()


def _same_plan(soph, canon, params, priors):
    """``q(π)`` (the softmax of the sophisticated planner's ``G``) and the scored term tree of
    every policy, bitwise between the two declarations."""
    q_s, _ = soph.infer_policies(params, priors)
    q_c, _ = canon.infer_policies(params, priors)
    np.testing.assert_array_equal(np.asarray(q_s.probs), np.asarray(q_c.probs))
    policies = canon.policies(params)
    g_s = total(soph.score_policies(params, priors, policies))
    g_c = total(canon.score_policies(params, priors, policies))
    np.testing.assert_array_equal(np.asarray(g_s), np.asarray(g_c))


# ------------------------------------------------------------------------------ the lowering


def test_the_lowering_fills_the_canonical_fields_and_keeps_what_was_written():
    soph, _ = _fused(sophisticated=SOPH)
    canon, _ = _fused(**CANON)
    assert soph.branch_on == canon.branch_on == ("o",)
    assert soph.plan_learning == canon.plan_learning == ("transition:s", "likelihood:o")
    assert soph.sophisticated == SOPH and canon.sophisticated is None
    assert "sophisticated=" in repr(soph)
    #  only the state half, or only the parameter half
    branch, _ = _fused(sophisticated=("obs:o",))
    assert branch.branch_on == ("o",) and branch.plan_learning == ()
    learn, _ = _fused(sophisticated=("transition:s",))
    assert learn.branch_on == () and learn.plan_learning == ("transition:s",)


def test_the_plan_is_bitwise_the_canonical_declarations_plan_on_the_fused_path():
    soph, params = _fused(sophisticated=SOPH)
    canon, _ = _fused(**CANON)
    priors = canon.initial_priors(params)
    _same_plan(soph, canon, params, priors)
    #  and the declaration did something: read frozen, the plan differs
    frozen, _ = _fused(branch_on=("o",))
    q_f, _ = frozen.infer_policies(params, priors)
    q_s, _ = soph.infer_policies(params, priors)
    assert not np.array_equal(np.asarray(q_f.probs), np.asarray(q_s.probs))


def test_the_plan_is_bitwise_the_canonical_declarations_plan_on_the_hybrid_path():
    soph, params = _tabular(
        plan_learning=(), planner="sophisticated", sophisticated=("obs:o", "transition:s")
    )
    canon, _ = _tabular(planner="sophisticated", branch_on=("o",))
    assert _canonical(soph) == _canonical(canon)
    priors = canon.initial_priors(params)
    _same_plan(soph, canon, params, priors)


# -------------------------------------------------------------------------------- the guards


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"branch_on": ("o",)}, "declare one syntax"),
        ({"plan_learning": ("transition:s",)}, "declare one syntax"),
        ({"plan_learning": True}, "declare one syntax"),
    ],
)
def test_declaring_both_syntaxes_is_refused(kw, match):
    with pytest.raises(ValueError, match=match):
        _fused(sophisticated=("obs:o",), **kw)


@pytest.mark.parametrize(
    ("sophisticated", "match"),
    [
        (True, "plan_learning=True"),
        ("obs:o", "tuple of keys"),
        (("transition:*",), "pattern"),
        (("obs:*",), "pattern"),
        (("obs:nope",), r"no modality.*\['o'\]"),
        (("s",), "unknown key"),
        (("obs:o:x",), "unknown key"),
    ],
)
def test_a_malformed_declaration_is_refused_with_a_sentence(sophisticated, match):
    with pytest.raises(ValueError, match=match):
        _fused(sophisticated=sophisticated)


def test_the_existing_checks_reach_the_lowered_fields_and_say_where_they_came_from():
    system, params = _fused(sophisticated=("transition:c",))  # c's table is not learnable
    with pytest.raises(ValueError, match=r"declared through sophisticated=.*not a learnable"):
        system.infer_policies(params, system.initial_priors(params))
    with pytest.raises(ValueError, match="declared through sophisticated="):
        _fused(sophisticated=("obs:o",), planner="enumeration")


# ------------------------------------------------------------------- describe, merge, replace


def test_the_description_writes_the_canonical_fields_and_rebuilds_them():
    soph, params = _fused(sophisticated=SOPH)
    canon, _ = _fused(**CANON)
    document, arrays = describe(soph, params, arrays="inline")
    document = json.loads(json.dumps(document))
    assert "sophisticated" not in _keys(document)  # a planner's name is a value, not a key
    assert (
        document["configuration"]
        == json.loads(json.dumps(describe(canon, params, arrays="inline")[0]))["configuration"]
    )
    built, _ = build(document, arrays=arrays)
    assert _canonical(built) == _canonical(canon) and built.sophisticated is None


def test_merge_and_subsystem_read_the_canonical_fields():
    a, pa = _fused(sophisticated=SOPH)
    b, pb = _fused(sophisticated=("transition:s",))
    ca, _ = _fused(**CANON)
    cb, _ = _fused(plan_learning=("transition:s",))
    soph, params = merge({"a": a, "b": b}, {"a": pa, "b": pb})
    canon, _ = merge({"a": ca, "b": cb}, {"a": pa, "b": pb})
    assert soph == canon
    assert _canonical(soph) == _canonical(canon)
    view_s, _ = subsystem(soph, params, "a")
    view_c, _ = subsystem(canon, params, "a")
    assert _canonical(view_s) == _canonical(view_c)


def test_replace_lowers_afresh_and_a_canonical_field_alone_is_refused():
    system, params = _fused(sophisticated=SOPH)
    same = dataclasses.replace(system)
    assert _canonical(same) == _canonical(system) and same.sophisticated == SOPH
    fewer = dataclasses.replace(system, sophisticated=("transition:s",))
    assert fewer.branch_on == () and fewer.plan_learning == ("transition:s",)
    with pytest.raises(ValueError, match="declare one syntax.*sophisticated=None"):
        dataclasses.replace(system, plan_learning=("transition:s",))
    plain = dataclasses.replace(system, sophisticated=None, plan_learning=("transition:s",))
    assert plain.sophisticated is None and plain.branch_on == ("o",)
    assert type(plain.branch_on) is tuple  # the lowering mark goes with sophisticated
    #  a System's fields are static: flattening and unflattening (as eqx.tree_at and jit do)
    #  bypasses the constructor and keeps them as lowered, and the lowered System is the same
    #  static key under jit as the canonical one up to the declared field
    leaves, treedef = jax.tree_util.tree_flatten(system)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert _canonical(rebuilt) == _canonical(system) and rebuilt.sophisticated == SOPH
    p2 = eqx.tree_at(lambda p: p.initial_state_prior["s"].logits, params, jnp.ones(3))
    q, _ = rebuilt.infer_policies(p2, rebuilt.initial_priors(p2))
    assert np.isfinite(np.asarray(q.probs)).all()
