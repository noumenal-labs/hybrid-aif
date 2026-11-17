"""Switching nonlinear dynamics: a :class:`~hybrid_aif.factors.NonlinearGaussianTransition`
with discrete state parents.

The mode reaches ``f`` as an integer level in its declared (``B_deps``) place among
``controls``, and the innovations take one axis per discrete parent in that order, the
:class:`~hybrid_aif.factors.LinearGaussianTransition` convention. The factor is carried as a
mixture: each component through ``f`` once per joint mode, weighted by the mode's belief, then
reduced as the linear switching path reduces.

Pinned by: an affine ``f`` per mode against the per-mode linear bank, to float tolerance, under
both propagations and both reduction schedules, with a discrete control node declared before
or after the mode; the mode evidence a fix sends back to the discrete parent; a plan; the
description round trip; the generative process sampling through the mode; the declaration-order
and shape refusals.

Design: docs/design/nonlinear-transitions.md N4 (amended 2026-11-15).
"""

import dataclasses
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import (
    Box,
    Categorical,
    ConditionalCategorical,
    ConditionalGaussian,
    Gaussian,
    LinearGaussianObservation,
    LinearGaussianTransition,
    MixtureGaussian,
    NonlinearGaussianTransition,
    System,
    SystemParams,
    describe,
    register_transition,
)
from hybrid_aif import description as _description
from hybrid_aif.description import build
from hybrid_aif.objectives import hybrid as _hybrid
from hybrid_aif.objectives import total

KEY = jax.random.PRNGKey(0)
D = 2
#  per-mode maps: mode 0 drifts forward and turns a little, mode 1 drifts back and damps
F = jnp.asarray([[[1.0, 0.1], [-0.1, 1.0]], [[0.8, 0.0], [0.2, 0.7]]], jnp.float32)
G = jnp.asarray([[[1.0, 0.0], [0.0, 1.0]], [[0.4, 0.0], [0.0, 0.4]]], jnp.float32)
C = jnp.asarray([[0.5, 0.0], [-0.5, 0.2]], jnp.float32)
Q = jnp.asarray([[[0.05, 0.0], [0.01, 0.04]], [[0.1, 0.0], [0.0, 0.08]]], jnp.float32)
#  a discrete control node scales the offset (k = 0 or 1 picks a gain)
GAIN = jnp.asarray([1.0, -2.0], jnp.float32)


class ModeAffine(eqx.Module):
    """``F[m] x + G[m] u + c[m]``: the switching linear bank written as one map over the mode
    level ``m`` (the controls tuple is ``(m, u)``)."""

    F: jax.Array
    G: jax.Array
    c: jax.Array

    def __call__(self, x, controls):
        mode, u = controls
        assert jnp.issubdtype(mode.dtype, jnp.integer), mode.dtype
        return self.F[mode] @ x + self.G[mode] @ u + self.c[mode]


class GainModeAffine(eqx.Module):
    """``F[m] x + gain[k] c[m]``, the controls tuple ``(k, m)``: a discrete control node
    declared before the mode."""

    F: jax.Array
    c: jax.Array
    gain: jax.Array

    def __call__(self, x, controls):
        k, mode = controls
        return self.F[mode] @ x + self.gain[k] * self.c[mode]


def _prior(K: int) -> MixtureGaussian:
    means = jnp.asarray([[0.0, 0.0], [1.0, -0.5], [-0.8, 0.6], [0.3, 1.2]], jnp.float32)[:K]
    chols = jnp.stack([0.2 * jnp.eye(D), 0.3 * jnp.eye(D), 0.15 * jnp.eye(D), 0.25 * jnp.eye(D)])
    return MixtureGaussian(
        log_weights=jnp.log(jnp.asarray([0.4, 0.3, 0.2, 0.1][:K]) / sum([0.4, 0.3, 0.2, 0.1][:K])),
        means=means,
        chols=chols[:K],
    )


def _mode_table() -> ConditionalCategorical:
    return ConditionalCategorical(
        log_table=jnp.log(jnp.asarray([[0.8, 0.3], [0.2, 0.7]])), parent_ndims=1
    )


def _continuous(kind: str, propagation: str = "first_order", schedule="mode_aligned", K=4, **over):
    """The mode, then ``x`` switched by it and moved by a continuous ``u``; ``kind`` is
    ``"linear"`` (the per-mode bank) or ``"nonlinear"`` (one affine map over the mode)."""
    system = System(
        A_deps=((1,),),
        B_deps=((0,), (0, 2), ()),
        controls=(2,),
        action_names=("a_u",),
        control_names=("u",),
        modality_names=("fix",),
        state_names=("mode", "x"),
        policy_len=2,
        mixture_k=K,
        mixture_schedule=schedule,
        **over,
    )
    if kind == "linear":
        noise = ConditionalGaussian(loc=C, chol=Q, parent_ndims=1)
        x = LinearGaussianTransition(F=F, innovations=noise, G=G)
    else:
        noise = ConditionalGaussian(loc=jnp.zeros((2, D)), chol=Q, parent_ndims=1)
        x = NonlinearGaussianTransition(ModeAffine(F, G, C), noise, propagation)
    params = SystemParams(
        likelihood={
            "fix": LinearGaussianObservation(H=jnp.eye(D), c=jnp.zeros(D), noise=0.2 * jnp.eye(D))
        },
        transition={"mode": _mode_table(), "x": x},
        initial_state_prior={
            "mode": Categorical(logits=jnp.log(jnp.asarray([0.6, 0.4]))),
            "x": _prior(K),
        },
        state_targets={"x": Gaussian(loc=jnp.asarray([1.0, 0.0]), chol=0.5 * jnp.eye(D))},
        control={"u": Box(-jnp.ones(D), jnp.ones(D))},
    )
    return system, params


def _discrete(kind: str, propagation: str = "first_order", K=4):
    """``x`` reads a discrete control node ``k`` declared before its mode parent: its
    innovations' axes are ``(k, mode)``, and ``f`` takes ``(k, mode)``."""
    system = System(
        B_deps=((0,), (2, 0), ()),
        controls=(2,),
        action_names=("a_k",),
        control_names=("k",),
        state_names=("mode", "x"),
        policy_len=1,
        mixture_k=K,
        mixture_schedule="mode_aligned",
    )
    chol = jnp.broadcast_to(Q, (2, 2, D, D))
    if kind == "linear":
        loc = GAIN[:, None, None] * C[None]  # (k, mode, d)
        x = LinearGaussianTransition(
            F=jnp.broadcast_to(F, (2, 2, D, D)),
            innovations=ConditionalGaussian(loc=loc, chol=chol, parent_ndims=2),
        )
    else:
        noise = ConditionalGaussian(loc=jnp.zeros((2, 2, D)), chol=chol, parent_ndims=2)
        x = NonlinearGaussianTransition(GainModeAffine(F, C, GAIN), noise, propagation)
    params = SystemParams(
        transition={"mode": _mode_table(), "x": x},
        initial_state_prior={
            "mode": Categorical(logits=jnp.log(jnp.asarray([0.6, 0.4]))),
            "x": _prior(K),
        },
    )
    return system, params


def _close_mixture(a: MixtureGaussian, b: MixtureGaussian, atol=1e-5):
    np.testing.assert_allclose(np.exp(a.log_weights), np.exp(b.log_weights), atol=atol)
    np.testing.assert_allclose(a.means, b.means, atol=atol)
    cov = lambda m: np.asarray(m.chols @ jnp.swapaxes(m.chols, -1, -2))  # noqa: E731
    np.testing.assert_allclose(cov(a), cov(b), atol=atol)


# ---------------------------------------------------------- the per-mode linear bank, exactly


@pytest.mark.parametrize("propagation", ["first_order", "unscented"])
@pytest.mark.parametrize("schedule", ["mode_aligned", "runnalls"])
def test_an_affine_map_per_mode_is_the_switching_linear_bank(propagation, schedule):
    lin, lin_p = _continuous("linear", schedule=schedule)
    nl, nl_p = _continuous("nonlinear", propagation, schedule)
    assert nl._families(nl_p) == ("categorical", "mixture")
    beliefs = lin.initial_priors(lin_p)
    action = {"a_u": jnp.asarray([0.3, -0.6])}
    for _ in range(3):  # three steps: the reduced mixture feeds the next grow
        want = lin.update_empirical_prior(lin_p, beliefs, action)
        got = nl.update_empirical_prior(nl_p, beliefs, action)
        np.testing.assert_allclose(got["mode"].probs, want["mode"].probs, atol=1e-6)
        _close_mixture(got["x"], want["x"])
        beliefs = want


@pytest.mark.parametrize("propagation", ["first_order", "unscented"])
def test_a_discrete_control_declared_before_the_mode_keeps_its_place(propagation):
    """The innovations' axes and ``f``'s controls follow the declared order, so a control node
    declared before the mode comes first, as on the linear bank."""
    lin, lin_p = _discrete("linear")
    nl, nl_p = _discrete("nonlinear", propagation)
    beliefs = lin.initial_priors(lin_p)
    for level in (0, 1):
        action = {"a_k": jnp.int32(level)}
        _close_mixture(
            nl.update_empirical_prior(nl_p, beliefs, action)["x"],
            lin.update_empirical_prior(lin_p, beliefs, action)["x"],
        )


def test_a_gaussian_prior_grows_into_a_mode_mixture():
    """A single-Gaussian prior on a switched factor is carried as a mixture: one grown
    component per mode, weighted by the mode belief, as the linear path carries it."""
    lin, lin_p = _continuous("linear", K=2)
    nl, nl_p = _continuous("nonlinear", K=2)
    start = Gaussian(loc=jnp.asarray([0.2, -0.1]), chol=0.3 * jnp.eye(D))
    lin_p = dataclasses.replace(
        lin_p, initial_state_prior={**lin_p.initial_state_prior, "x": start}
    )
    nl_p = dataclasses.replace(nl_p, initial_state_prior={**nl_p.initial_state_prior, "x": start})
    assert nl._families(nl_p) == ("categorical", "mixture")
    action = {"a_u": jnp.asarray([0.0, 0.5])}
    want = lin.update_empirical_prior(lin_p, lin.initial_priors(lin_p), action)["x"]
    got = nl.update_empirical_prior(nl_p, nl.initial_priors(nl_p), action)["x"]
    _close_mixture(got, want)
    np.testing.assert_allclose(np.exp(got.log_weights), want_modes(nl_p), atol=1e-5)


def want_modes(params) -> np.ndarray:
    """The mode belief after one step of the mode table from the prior."""
    prior = np.asarray(params.initial_state_prior["mode"].probs)
    return np.exp(np.asarray(params.transition["mode"].log_table)) @ prior


# -------------------------------------------------------------------------------- inference


def test_a_fix_sends_mode_evidence_back_to_the_parent():
    """The modes predict different positions, so a fix near mode 1's prediction moves the mode
    belief toward mode 1, on the nonlinear path exactly as on the linear bank."""
    lin, lin_p = _continuous("linear")
    nl, nl_p = _continuous("nonlinear")
    beliefs = lin.initial_priors(lin_p)
    action = {"a_u": jnp.asarray([0.0, 0.0])}
    prior = nl.update_empirical_prior(nl_p, beliefs, action)
    means = np.asarray(prior["x"].means).reshape(2, -1, D)  # mode-owned blocks
    weights = np.exp(np.asarray(prior["x"].log_weights)).reshape(2, -1)
    mode1 = (weights[1, :, None] * means[1]).sum(0) / weights[1].sum()
    obs = {"fix": jnp.asarray(mode1)}
    got, _ = nl.infer_states(nl_p, obs, prior)
    want, _ = lin.infer_states(lin_p, obs, lin.update_empirical_prior(lin_p, beliefs, action))
    before = float(prior["mode"].probs[1])
    assert float(got["mode"].probs[1]) > before + 0.05
    np.testing.assert_allclose(got["mode"].probs, want["mode"].probs, atol=1e-5)
    _close_mixture(got["x"], want["x"], atol=1e-4)


# --------------------------------------------------------------------------------- planning


def test_plans_through_the_switched_map_under_the_prior_belief():
    system, params = _continuous(
        "nonlinear", "unscented", planning_belief="prior", epistemic=True, planner="gradient"
    )
    beliefs = system.initial_priors(params)
    q_pi, tree = system.infer_policies(params, beliefs)[:2]
    assert np.all(np.isfinite(np.asarray(q_pi.probs)))
    assert np.all(np.isfinite(np.asarray(total(tree).midpoint)))
    action = system.act(KEY, params, beliefs, selection="deterministic")[0]
    assert np.all(np.isfinite(np.asarray(action["a_u"])))


# ------------------------------------------------------------------------------- the surface


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(_description, "_TRANSITIONS", {})


def test_the_description_round_trips_the_switched_map(registry):
    register_transition("tests.mode_affine", ModeAffine)
    system, params = _continuous("nonlinear", "unscented")
    doc = describe(system, params)
    document = json.loads(json.dumps(doc.document))
    built, built_params = build(document, arrays=doc.arrays)
    decl = built_params.transition["x"]
    assert isinstance(decl.f, ModeAffine) and decl.innovations.parent_shape == (2,)
    beliefs = system.initial_priors(params)
    action = {"a_u": jnp.asarray([0.3, -0.6])}
    _close_mixture(
        built.update_empirical_prior(built_params, beliefs, action)["x"],
        system.update_empirical_prior(params, beliefs, action)["x"],
        atol=1e-6,
    )


def test_the_generative_process_samples_through_the_sampled_mode():
    """``as_env`` draws the mode first and steps ``x`` through ``f`` at that level: with the
    process noise set tiny, the next state is the drawn mode's map."""
    system, params = _continuous("nonlinear")
    tiny = ConditionalGaussian(
        loc=jnp.zeros((2, D)), chol=1e-6 * jnp.broadcast_to(jnp.eye(D), (2, D, D)), parent_ndims=1
    )
    decl = params.transition["x"]
    params = dataclasses.replace(
        params, transition={**params.transition, "x": NonlinearGaussianTransition(decl.f, tiny)}
    )
    env, env_params = system.as_env(params)
    x0 = jnp.asarray([0.5, -0.5])
    u = jnp.asarray([0.3, 0.1])
    seen = set()
    for i in range(12):
        _, state = env.reset(KEY, env_params, state={"mode": jnp.int32(0), "x": x0})
        _, nxt = env.step(jax.random.PRNGKey(i), state, {"a_u": u}, env_params)
        m = int(nxt["mode"])
        seen.add(m)
        np.testing.assert_allclose(nxt["x"], F[m] @ x0 + G[m] @ u + C[m], atol=1e-4)
    assert seen == {0, 1}


# -------------------------------------------------------------------------------- refusals


def test_the_mode_parent_must_precede_and_the_shape_must_count_it():
    system, params = _continuous("nonlinear")
    late = System(
        B_deps=((1, 2), (1,), ()),
        controls=(2,),
        action_names=("a_u",),
        control_names=("u",),
        state_names=("x", "mode"),
        policy_len=1,
        mixture_k=4,
        mixture_schedule="mode_aligned",
    )
    with pytest.raises(ValueError, match="within-step ordering"):
        bare = dataclasses.replace(params, likelihood={})
        late.update_empirical_prior(bare, late.initial_priors(bare), {"a_u": jnp.zeros(D)})
    flat = ConditionalGaussian(loc=jnp.zeros((1, D)), chol=Q[:1], parent_ndims=1)
    wrong = dataclasses.replace(
        params,
        transition={
            **params.transition,
            "x": NonlinearGaussianTransition(params.transition["x"].f, flat),
        },
    )
    with pytest.raises(ValueError, match=r"innovations parent_shape \(1,\) != \(2,\)"):
        system.update_empirical_prior(wrong, system.initial_priors(wrong), {"a_u": jnp.zeros(D)})
    #  mode_aligned needs mixture_k a multiple of the mode count
    odd = System(
        B_deps=((0,), (0, 2), ()),
        controls=(2,),
        action_names=("a_u",),
        control_names=("u",),
        state_names=("mode", "x"),
        policy_len=1,
        mixture_k=3,
        mixture_schedule="mode_aligned",
    )
    p3 = dataclasses.replace(
        bare, initial_state_prior={**params.initial_state_prior, "x": _prior(3)}
    )
    with pytest.raises(ValueError, match="multiple of 2"):
        odd.update_empirical_prior(p3, odd.initial_priors(p3), {"a_u": jnp.zeros(D)})


def test_the_mixture_step_without_a_mode_is_the_per_component_step():
    """``M = 1``: the generalised step reduces to the per-component push it replaced."""
    _, params = _continuous("nonlinear")
    decl = NonlinearGaussianTransition(
        ModeAffine(F, G, C), ConditionalGaussian(loc=jnp.zeros((1, D)), chol=Q[:1], parent_ndims=1)
    )
    q = _prior(4)
    inputs = _hybrid._NLInputs(jnp.zeros(D), Q[0], (jnp.int32(1), jnp.asarray([0.2, 0.1])), ())
    got = _hybrid._nonlinear_mixture_step(q, decl, lambda levels: inputs, (), (), 4, "mode_aligned")
    for c in range(4):
        one = _hybrid._nonlinear_step(Gaussian(loc=q.means[c], chol=q.chols[c]), decl, inputs)
        np.testing.assert_allclose(got.means[c], one.loc, atol=1e-6)
