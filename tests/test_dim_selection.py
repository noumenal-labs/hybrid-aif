"""Selecting dimensions within a continuous factor (design: per-variable-infogain.md V10).

``System(dims={"pose": (...)})`` names a continuous factor's dimensions, and ``epistemic``
selects them as ``<factor>:<dim>``. On a Gaussian-carried factor each channel's leaf is its
information about the selected dimensions, the others marginalised, and the factor's
``observation_interaction`` is ``TC(Y) − TC(Y | x_S)``, so that the channels' leaves less the
interaction are the information the step's readings carry about ``x_S`` together, the entropy
drop of the marginal under the epistemic chain (V11); under ``risk_ambiguity`` each channel
observing the factor takes ``E H[y | x_S]``. Pinned against hand computation on the
consumer's shape: a pose ``(x, y, yaw)`` with two static map offsets ``(b1, b2)``, a pile reading
``y = n·p + b1 − n·a + e`` and a GPS on ``(x, y)``.
"""

from __future__ import annotations

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.composition.merge import merge
from hybrid_aif.controls import ControlStep
from hybrid_aif.description import build, describe
from hybrid_aif.distributions import ConditionalGaussian, Gaussian, MixtureGaussian
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianTransition
from hybrid_aif.objectives import hybrid as _hybrid
from hybrid_aif.system import System, SystemParams
from tests.test_continuous_nodes import _box

NAMES = ("x", "y", "yaw", "b1", "b2")
D = len(NAMES)
NORMAL = np.asarray([0.6, 0.8])  # the pile's normal, n
ANCHOR = np.asarray([1.0, 2.0])  # the pile's anchor, a
SIGMA_R = 0.2
Q_DIAG = (0.1, 0.1, 0.05, 1e-3, 1e-3)  # the offsets are constants: a small process noise
PRIOR_SD = (0.5, 0.5, 0.2, 1.0, 1.5)
H_PILE = np.asarray([[NORMAL[0], NORMAL[1], 0.0, 1.0, 0.0]])
H_GPS = np.eye(2, D)
R_GPS = 0.3


def _system(**overrides):
    defaults = dict(
        A_deps=((0,), (0,)),
        B_deps=((),),
        controlled=(True,),
        policy_len=2,
        modality_names=("gps", "pile"),
        factor_names=("pose",),
        dims={"pose": NAMES},
        planning_belief="posterior",
        epistemic=("pose:b1",),
    )
    return System(**{**defaults, **overrides})


def _transition(q_diag=Q_DIAG) -> LinearGaussianTransition:
    return LinearGaussianTransition(
        F=jnp.eye(D),
        innovations=ConditionalGaussian(
            loc=jnp.zeros((2, D)),
            chol=jnp.broadcast_to(jnp.diag(jnp.asarray(q_diag)), (2, D, D)),
            parent_ndims=1,
        ),
    )


def _params(q_diag=Q_DIAG) -> SystemParams:
    return SystemParams(
        likelihood={
            "gps": LinearGaussianObservation(
                H=jnp.asarray(H_GPS, jnp.float32), c=jnp.zeros(2), noise=R_GPS * jnp.eye(2)
            ),
            "pile": LinearGaussianObservation(
                H=jnp.asarray(H_PILE, jnp.float32),
                c=jnp.asarray([-float(NORMAL @ ANCHOR)]),
                noise=SIGMA_R * jnp.eye(1),
            ),
        },
        transition={"pose": _transition(q_diag)},
        initial_state_prior={"pose": _belief()},
    )


def _belief() -> Gaussian:
    return Gaussian(loc=jnp.zeros(D), chol=jnp.diag(jnp.asarray(PRIOR_SD)))


def _run(system, params, present=None, belief=None):
    """Drive the planning closures by hand: the carries after each transition and the step
    trees scored on them."""
    ctx = system.planner_context(params, {"pose": belief or _belief()}, present=present)
    carry, carries, trees = ctx.init_beliefs, [], []
    for t in range(system.policy_len):
        action = jnp.asarray([0], jnp.int32)
        carry = ctx.transition_fn(carry, action)
        carries.append(carry)
        trees.append(ctx.step_terms_fn(carry, action, jnp.asarray(t, jnp.int32)))
    return carries, trees


def _leaf(tree, term, key) -> float:
    leaf = tree[term][key]
    assert float(leaf.lower) == float(leaf.upper)
    return float(leaf.lower)


def _info(tree) -> float:
    """The information the step's readings carry about the selection together: the channels'
    leaves less the factor's interaction information (per-variable-infogain.md V11 decision 3)."""
    leaves = sum(_leaf(tree, "infogain", name) for name in tree["infogain"])
    return leaves - _leaf(tree, "observation_interaction", "pose")


def _cov(q: Gaussian) -> np.ndarray:
    chol = np.asarray(q.chol, np.float64)
    return chol @ chol.T


def _half_logdet(cov: np.ndarray, dims) -> float:
    idx = np.asarray(dims)
    return 0.5 * np.linalg.slogdet(cov[np.ix_(idx, idx)])[1]


def _kalman(cov: np.ndarray, H: np.ndarray, R: np.ndarray) -> np.ndarray:
    """The covariance after conditioning on ``y = H x + e``, ``e ~ N(0, R)``."""
    S = H @ cov @ H.T + R
    return cov - cov @ H.T @ np.linalg.solve(S, H @ cov)


def _same_trees(a, b) -> None:
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    for x, y in zip(la, lb, strict=True):
        assert np.array_equal(np.asarray(x), np.asarray(y))


B1 = NAMES.index("b1")


# ------------------------------------------------------------------- the Gaussian leaf


def test_the_leaf_is_the_marginal_information_gain_under_the_epistemic_chain():
    """For one pile reading (the GPS absent, so its leaf is zeroed and it drops out of the
    interaction), the pile's leaf is the entropy drop of the offset's marginal; at the first step
    the offset and the pose are uncorrelated, so it is the closed form ½ log(1 + σ_b² / (σ_r² +
    nᵀ Σ_pose n)); at the second the first reading has correlated them, and the leaf is the
    general marginal drop."""
    carries, trees = _run(_system(), _params(), present={"gps": False})
    R = np.asarray([[SIGMA_R**2]])
    for t, ((pair,), tree) in enumerate(zip(carries, trees, strict=True)):
        before = _cov(pair.epistemic)
        after = _kalman(before, H_PILE, R)
        want = _half_logdet(before, (B1,)) - _half_logdet(after, (B1,))
        assert _leaf(tree, "infogain", "gps") == 0.0
        assert _leaf(tree, "observation_interaction", "pose") == pytest.approx(0.0, abs=1e-6)
        assert _leaf(tree, "infogain", "pile") == pytest.approx(want, abs=1e-5)
        assert _info(tree) == pytest.approx(want, abs=1e-5)
        if t == 0:
            assert np.allclose(before[B1, :3], 0.0)
            pose = before[:2, :2]
            closed = 0.5 * np.log(1.0 + before[B1, B1] / (SIGMA_R**2 + NORMAL @ pose @ NORMAL))
            assert _leaf(tree, "infogain", "pile") == pytest.approx(closed, abs=1e-5)
        else:
            assert not np.allclose(before[B1, :2], 0.0)  # the reading correlated them
    # the reading says nothing about the other offset, which it does not observe
    _, other = _run(_system(epistemic=("pose:b2",)), _params(), present={"gps": False})
    assert all(_info(tree) == pytest.approx(0.0, abs=1e-6) for tree in other)


def test_a_two_dimension_sub_block_with_both_readings():
    """Both readings present: the channels' leaves less the interaction are the drop of the
    ``(x, b1)`` marginal's entropy."""
    system = _system(epistemic=("pose:x", "pose:b1"))
    carries, trees = _run(system, _params())
    H = np.vstack([H_GPS, H_PILE])
    R = np.diag([R_GPS**2, R_GPS**2, SIGMA_R**2])
    for (pair,), tree in zip(carries, trees, strict=True):
        before = _cov(pair.epistemic)
        want = _half_logdet(before, (0, B1)) - _half_logdet(_kalman(before, H, R), (0, B1))
        assert _info(tree) == pytest.approx(want, abs=1e-5)


def test_without_the_epistemic_chain_the_leaves_read_the_predictive_belief():
    """Under ``"prior"`` the channels' leaves less the interaction are the information the
    step's readings carry about ``(x, b1)`` given the predictive belief, and the marginal's
    entropy is the ``uncertainty`` row on the same sub-block (V11 decision 4)."""
    system = _system(
        planning_belief="prior",
        epistemic=("pose:x", "pose:b1"),
        terms=("uncertainty:pose:x", "uncertainty:pose:b1"),
    )
    carries, trees = _run(system, _params())
    H = np.vstack([H_GPS, H_PILE])
    R = np.diag([R_GPS**2, R_GPS**2, SIGMA_R**2])
    for (q,), tree in zip(carries, trees, strict=True):
        cov = _cov(q)
        drop = _half_logdet(cov, (0, B1)) - _half_logdet(_kalman(cov, H, R), (0, B1))
        assert _info(tree) == pytest.approx(drop, abs=1e-5)
        want = _half_logdet(cov, (0, B1)) + (1.0 + np.log(2.0 * np.pi))  # d = 2
        assert _leaf(tree, "uncertainty", "pose") == pytest.approx(want, abs=1e-5)


@pytest.mark.parametrize("planning_belief", ["prior", "posterior"])
@pytest.mark.parametrize("decomposition", ["utility_infogain", "risk_ambiguity"])
def test_every_dimension_is_the_factor_bit_for_bit(planning_belief, decomposition):
    def trees(epistemic):
        system = _system(
            epistemic=epistemic, planning_belief=planning_belief, decomposition=decomposition
        )
        return _run(system, _params())[1]

    whole = trees(("pose",))
    _same_trees(trees(tuple(f"pose:{n}" for n in NAMES)), whole)
    _same_trees(trees(("pose:*",)), whole)
    _same_trees(trees(("pose", "pose:b1")), whole)  # a union with the whole factor


def test_undeclared_dimensions_are_named_by_position():
    declared = _run(_system(), _params())[1]
    positional = _run(_system(dims={}, epistemic=(f"pose:dim{B1}",)), _params())[1]
    _same_trees(positional, declared)
    system = _system(dims={"pose": ("x", "y", "dim2", "b1", "b2")}, epistemic=("pose:dim2",))
    _run(system, _params())  # dimN may name its own position


def test_an_earlier_position_reading_raises_the_offset_s_later_leaf():
    """A GPS reading expected at the first step pins the pose the second step's pile reading
    is read at, so that reading says more about the offset."""
    system = _system()
    with_gps = _run(system, _params(), present={"gps": jnp.asarray([True, False])})[1]
    without = _run(system, _params(), present={"gps": jnp.asarray([False, False])})[1]
    assert _info(with_gps[1]) > _info(without[1]) + 0.01


def test_wildcards():
    """``pose:*`` is every dimension; a wildcarded first segment ranges over parameter slots
    only, so ``*:*`` picks up no dimension."""
    system, params = _system(), _params()
    variables = system._epistemic_variables(params)
    assert {k for k, v in variables.items() if v.kind == "dim"} == {f"pose:{n}" for n in NAMES}
    assert system._expand_epistemic("pose:*", params, variables) == tuple(
        f"pose:{n}" for n in NAMES
    )
    assert system._expand_epistemic("*:*", params, variables) == ()
    assert system._expand_epistemic("*", params, variables) == ("pose",)


# --------------------------------------------------------------------- the ambiguity


def _two_factor_system(**overrides):
    defaults = dict(
        A_deps=((0,), (0,), (1,)),
        B_deps=((), ()),
        controlled=(True, False),
        policy_len=2,
        modality_names=("gps", "pile", "alt"),
        factor_names=("pose", "z"),
        dims={"pose": NAMES},
        decomposition="risk_ambiguity",
    )
    return System(**{**defaults, **overrides})


def _two_factor_params() -> SystemParams:
    params = _params()
    noise = 0.4 * jnp.eye(1)
    return dataclasses.replace(
        params,
        likelihood={
            **params.likelihood,
            "alt": LinearGaussianObservation(H=jnp.ones((1, 1)), c=jnp.zeros(1), noise=noise),
        },
        transition={
            **params.transition,
            "z": LinearGaussianTransition(
                F=jnp.eye(1),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 1)), chol=0.1 * jnp.ones((1, 1, 1)), parent_ndims=1
                ),
            ),
        },
        initial_state_prior={
            **params.initial_state_prior,
            "z": Gaussian(loc=jnp.zeros(1), chol=jnp.ones((1, 1))),
        },
    )


def _run_two(system, params):
    ctx = system.planner_context(params, {"pose": _belief(), "z": params.initial_state_prior["z"]})
    carry, carries, trees = ctx.init_beliefs, [], []
    for t in range(system.policy_len):
        action = jnp.asarray([0], jnp.int32)
        carry = ctx.transition_fn(carry, action)
        carries.append(carry)
        trees.append(ctx.step_terms_fn(carry, action, jnp.asarray(t, jnp.int32)))
    return carries, trees


@pytest.mark.parametrize("planning_belief", ["prior", "posterior"])
def test_ambiguity_is_the_entropy_given_the_selected_dimensions(planning_belief):
    """``E H[y | x_S] = ½ log|2πe (H Σ_{x|x_S} Hᵀ + R)|`` on the predictive belief, for each
    channel observing the selected factor; a channel on another factor is untouched."""
    params = _two_factor_params()
    system = _two_factor_system(epistemic=("pose:b1", "z"), planning_belief=planning_belief)
    carries, trees = _run_two(system, params)
    for carry, tree in zip(carries, trees, strict=True):
        q = carry[0].predictive if planning_belief == "posterior" else carry[0]
        cov = _cov(q)
        cond = cov - np.outer(cov[:, B1], cov[B1, :]) / cov[B1, B1]
        for name, H, R in (
            ("pile", H_PILE, np.asarray([[SIGMA_R**2]])),
            ("gps", H_GPS, R_GPS**2 * np.eye(2)),
        ):
            S = H @ cond @ H.T + R
            want = 0.5 * np.linalg.slogdet(2.0 * np.pi * np.e * S)[1]
            assert _leaf(tree, "ambiguity", name) == pytest.approx(want, abs=1e-5)
    whole = _run_two(
        _two_factor_system(epistemic=("pose", "z"), planning_belief=planning_belief), params
    )[1]
    for tree, base in zip(trees, whole, strict=True):
        assert np.array_equal(
            np.asarray(tree["ambiguity"]["alt"].lower), np.asarray(base["ambiguity"]["alt"].lower)
        )
        # selecting the factor whole leaves its channels at E H[y | x], the noise's entropy
        want = 0.5 * np.log(2.0 * np.pi * np.e * SIGMA_R**2)
        assert _leaf(base, "ambiguity", "pile") == pytest.approx(want, abs=1e-5)


# ----------------------------------------------------------------------- refusals


def test_a_sub_block_of_a_mixture_carried_factor_is_refused():
    belief = MixtureGaussian.from_gaussian(_belief())
    params = dataclasses.replace(_params(), initial_state_prior={"pose": belief})
    system = _system(mixture_k=2, planning_belief="prior")
    with pytest.raises(NotImplementedError, match="carried as a Gaussian mixture"):
        system.planner_context(params, {"pose": belief})
    # every dimension is the whole factor, which a mixture-carried factor takes
    whole = _system(mixture_k=2, planning_belief="prior", epistemic=("pose:*",))
    whole.planner_context(params, {"pose": belief})


@pytest.mark.parametrize(
    "key, match",
    [
        ("pose:zz", r"'pose' has no dimension 'zz'.*\['x', 'y', 'yaw', 'b1', 'b2'\]"),
        ("pose:b1:x", "a dimension of a factor is spelled 'pose:<dim>'"),
        ("pose:*:*", "has two segments"),
    ],
)
def test_an_unknown_dimension_is_refused(key, match):
    with pytest.raises(ValueError, match=match):
        _system(epistemic=(key,)).planner_context(_params(), {"pose": _belief()})


def test_a_dimension_of_a_categorical_factor_is_refused():
    from tests.test_hybrid_switching_system import make_params as mixture_params
    from tests.test_hybrid_switching_system import make_system as mixture_system

    system, params = mixture_system(), mixture_params()
    categorical = next(
        name
        for f, name in enumerate(system._facs)
        if system._families(params)[f] == _hybrid.CATEGORICAL
    )
    with pytest.raises(ValueError, match="categorical factor"):
        dataclasses.replace(system, epistemic=(f"{categorical}:dim0",))._validate(params)
    with pytest.raises(ValueError, match="categorical factor"):
        dataclasses.replace(system, dims={categorical: ("a",)})._validate(params)


@pytest.mark.parametrize(
    "dims, match",
    [
        ({"pose": ("x", "x", "yaw", "b1", "b2")}, r"repeats \['x'\]"),
        ({"pose": ("x", "y", "yaw", "b:1", "b2")}, "one non-empty key segment"),
        ({"pose": ("x", "y", "yaw", "b.1", "b2")}, "one non-empty key segment"),
        ({"pose": ("x", "y", "yaw", "b*", "b2")}, "one non-empty key segment"),
        ({"pose": ("x", "y", "yaw", "b[1]", "b2")}, "one non-empty key segment"),
        ({"pose": ("x", "y", "yaw", "", "b2")}, "one non-empty key segment"),
        ({"pose": ("dim1", "y", "yaw", "b1", "b2")}, "may name position N only"),
        ({"pose": ()}, "names no dimensions"),
        ({"nope": ("a",)}, r"unknown factor\(s\) \['nope'\]"),
        ({"pose": "xyz"}, "tuples of dimension names"),
    ],
)
def test_malformed_dims_are_refused_at_construction(dims, match):
    with pytest.raises((ValueError, TypeError), match=match):
        _system(dims=dims)


def test_a_dims_count_that_does_not_match_the_factor_is_refused():
    system = _system(dims={"pose": ("x", "y")}, epistemic=("pose",))
    with pytest.raises(ValueError, match=r"declares 2 name\(s\).*dimension 5"):
        system.planner_context(_params(), {"pose": _belief()})


def test_dims_on_a_control_node_is_refused():
    system = _system(dims={"u_pose": ("a",)}, epistemic=("pose",))
    with pytest.raises(ValueError, match="control node"):
        system.planner_context(_params(), {"pose": _belief()})


@pytest.mark.parametrize("name", ["likelihood", "obs", "transition"])
def test_a_factor_named_after_a_kind_word_is_refused(name):
    with pytest.raises(ValueError, match="reserved"):
        _system(factor_names=(name,), dims={}, epistemic=True)


def test_dims_are_stored_hashably_in_factor_order():
    system = _system()
    assert system.dims == (("pose", NAMES),)
    assert hash(system) == hash(_system(dims=(("pose", list(NAMES)),)))
    assert dataclasses.replace(system, gamma=2.0).dims == system.dims


# --------------------------------------------------------------- description, merge


def test_a_description_carries_declared_names_and_round_trips():
    params = _params(q_diag=(0.1, 0.1, 0.05, 1e-3, 1e-3))  # the document takes a positive noise
    system = _system()
    document, arrays = describe(system, params)
    document = json.loads(json.dumps(document))
    assert document["variables"]["pose"] == {"domain": "real", "dim": D, "dims": list(NAMES)}
    infogain = [t for t in document["objective"]["terms"] if t["term"] == "infogain"]
    assert [t["on"] for t in infogain] == ["pose:b1"]
    rebuilt, _ = build(document, arrays=arrays)
    assert rebuilt.dims == system.dims and rebuilt.epistemic == ("pose:b1",)
    # undeclared names are not written, and the positional key round-trips as written
    plain = _system(dims={}, epistemic=("pose:dim3",))
    document, arrays = describe(plain, params)
    document = json.loads(json.dumps(document))
    assert "dims" not in document["variables"]["pose"]
    rebuilt, _ = build(document, arrays=arrays)
    assert rebuilt.dims == () and rebuilt.epistemic == ("pose:dim3",)


def test_a_description_refuses_malformed_dims():
    from hybrid_aif.description import DescriptionError

    params = _params(q_diag=(0.1, 0.1, 0.05, 1e-3, 1e-3))
    document, arrays = describe(_system(), params)
    document = json.loads(json.dumps(document))
    for bad, match in (
        (["x", "y"], r"has 2 name\(s\) for dim 5"),
        (["x", "y", "yaw", "b:1", "b2"], "one non-empty key segment"),
    ):
        broken = json.loads(json.dumps(document))
        broken["variables"]["pose"]["dims"] = bad
        with pytest.raises(DescriptionError, match=match):
            build(broken, arrays=arrays)


def test_merge_namespaces_the_factor_segment_and_the_dims():
    system, params = _system(), _params()
    other = _system(epistemic=True, dims={"pose": ("a", "b", "c", "d", "e")})
    composite, composite_params = merge(
        {"arm": system, "leg": other}, {"arm": params, "leg": params}
    )
    assert composite.dims == (("arm.pose", NAMES), ("leg.pose", ("a", "b", "c", "d", "e")))
    assert "arm.pose:b1" in composite.epistemic
    assert "leg.pose" in composite.epistemic  # True names a factor whole, not its dimensions
    assert not any(key.startswith("leg.pose:") for key in composite.epistemic)
    composite._validate(composite_params)


# ---------------------------------------------------------------- static dimensions


def test_the_innovation_factor_tolerates_a_static_dimension():
    """The innovation after a belief-valued control is the factor ``[Q_chol | G S^½]``, whose
    product with its transpose is ``Q + G S Gᵀ`` exactly, singular or not, with finite
    gradients: on a dimension with no process noise that ``G`` does not reach, and on a sum
    singular off the axes."""
    Q = jnp.diag(jnp.asarray([0.1, 0.2, 0.0]))
    G = jnp.asarray([[0.1], [0.5], [0.0]])
    root = jnp.asarray([[0.3]])
    loc = jnp.zeros(3)

    def product(q, g, r):
        factor = _hybrid._push_input(loc, q, g, jnp.asarray([0.4]), r)[1]
        return factor @ jnp.swapaxes(factor, -1, -2)

    for q, g in ((Q, G), (jnp.asarray([[1.0, 0, 0], [1.0, 0, 0], [0, 0, 0]]), jnp.zeros((3, 1)))):
        want = q @ q.T + g @ root @ root.T @ g.T
        np.testing.assert_allclose(np.asarray(product(q, g, root)), np.asarray(want), atol=1e-7)
        grad = jax.grad(lambda r, q=q, g=g: jnp.sum(product(q, g, r)))(root)
        assert np.all(np.isfinite(np.asarray(grad)))
    batched = jnp.stack([Q, jnp.diag(jnp.asarray([0.1, 0.2, 0.3]))])
    factors = _hybrid._push_input(jnp.zeros((2, 3)), batched, G, jnp.asarray([0.4]), root)[1]
    assert factors.shape == (2, 3, 4)
    np.testing.assert_allclose(
        np.asarray(factors[1] @ factors[1].T),
        np.asarray(batched[1] @ batched[1].T + G @ root @ root.T @ G.T),
        atol=1e-7,
    )


def test_a_belief_valued_control_beside_a_constant_plans_finitely():
    """A state with a near-constant dimension, pushed by a continuous control carried as a
    belief: every transition and score is finite, and that dimension gains only its own small
    noise. A dimension with no process noise at all is refused."""
    q_static = jnp.diag(jnp.asarray([0.1, 0.1, 1e-3]))
    system = System(
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("u",),
        factor_names=("x", "u_u"),
        policy_len=2,
        planner="gradient",
        epistemic=False,
    )
    transition = LinearGaussianTransition(
        F=jnp.eye(3),
        innovations=ConditionalGaussian(loc=jnp.zeros((1, 3)), chol=q_static[None], parent_ndims=1),
        G=jnp.asarray([[0.1], [0.5], [0.0]]),
    )
    prior = Gaussian(loc=jnp.zeros(3), chol=0.5 * jnp.eye(3))
    params = SystemParams(
        transition={"x": transition},
        initial_state_prior={"x": prior},
        state_targets={"x": Gaussian(loc=jnp.ones(3), chol=jnp.eye(3))},
        control={"u_u": _box()},
    )
    pos = system._p(params)
    step = _hybrid.make_transition(
        pos.transition,
        system._B,
        system._families(pos),
        None,
        table=system._table,
        space=system._space(pos),
    )
    belief = Gaussian(loc=jnp.asarray([0.4]), chol=jnp.asarray([[0.3]]))
    (got,) = step((prior,), ControlStep(entries=(belief,)))
    assert np.all(np.isfinite(np.asarray(got.chol)))
    assert float(got.cov[2, 2]) == pytest.approx(0.25 + 1e-6, rel=1e-5)
    beliefs = Gaussian(loc=jnp.full((2, 1), 0.4), chol=jnp.full((2, 1, 1), 0.3))
    tree = system.score_policies(params, {"x": prior}, {"u_u": beliefs})
    assert all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves(tree))
    noiseless = dataclasses.replace(
        transition,
        innovations=ConditionalGaussian(
            loc=jnp.zeros((1, 3)), chol=jnp.diag(jnp.asarray([0.1, 0.1, 0.0]))[None], parent_ndims=1
        ),
    )
    bad = dataclasses.replace(params, transition={"x": noiseless})
    with pytest.raises(ValueError, match="zero on its Cholesky factor's diagonal"):
        system.score_policies(bad, {"x": prior}, {"u_u": beliefs})
    # under jit the factor is traced, and the refusal is raised when the computation runs
    with pytest.raises(Exception, match="zero on its Cholesky factor's diagonal"):
        jax.jit(lambda p: system.score_policies(p, {"x": prior}, {"u_u": beliefs}))(bad)
    # a generative process may be noiseless: only beliefs need positive process noise
    env, env_params = system.as_env(bad)
    assert env is not None and env_params is not None
