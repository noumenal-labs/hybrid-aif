"""The data link along a replay (``docs/design/description-length.md`` D6, D7).

``Replay.free_energy`` is each tick's variational free energy and ``Replay.log_evidence`` its
negation: on one categorical factor the exact log evidence of the tick's observations under
its prior, and summed over a trace the prequential description length of the record under
the model as it stood at each tick, with learning off (a frozen model) and on (the model
learning as it goes). Pinned against a NumPy loop that lowers the tables as the engine does.
``resolution_from_evidence`` carries each slot's absorbed count into ``description_length``
as its resolution, and ``describe_parameters`` is the document's parameter envelope.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import digamma, logsumexp

import hybrid_aif as H
from hybrid_aif.description import describe, describe_parameters
from hybrid_aif.description.mdl import description_length, resolution_from_evidence
from hybrid_aif.inference.discrete import LOG_ZERO
from hybrid_aif.replay import replay
from tests.test_learn import _sense_system

_EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"


def _example(name: str):
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trace(system, params, T=7, seed=3):
    rng = np.random.default_rng(seed)
    ns = params.initial_state_prior["s"].num_levels
    no = params.likelihood["sense"].num_levels
    observations = {
        "sense": jnp.asarray(rng.integers(no, size=T), jnp.int32),
        "ident": jnp.asarray(rng.integers(ns, size=T), jnp.int32),
    }
    actions = {"a_s": jnp.asarray(rng.integers(2, size=T), jnp.int32)}
    return observations, actions


def _lowered(table) -> np.ndarray:
    """The engine's tempered log-likelihood of a table: the expected log parameters of a
    Dirichlet, a point table's normalised log table."""
    if isinstance(table, H.ConditionalDirichlet):
        a = np.asarray(table.concentration, np.float64)
        return np.maximum(digamma(a) - digamma(a.sum(0, keepdims=True)), LOG_ZERO)
    raw = np.asarray(table.log_table, np.float64)
    norm = np.maximum(raw, LOG_ZERO)
    norm = norm - logsumexp(norm, axis=0, keepdims=True)
    return np.where(raw <= LOG_ZERO, LOG_ZERO, norm)


def _numpy_prequential(system, params, observations, actions, learn: bool, present=None):
    """The canonical loop on one categorical factor, in NumPy: infer exactly, learn the sense
    table from the second tick on where it was observed, step the prior through the
    transition on the recorded action; the free energies and the counts at the end."""
    conc = np.asarray(params.likelihood["sense"].concentration, np.float64)
    ident = _lowered(params.likelihood["ident"])
    B = np.exp(np.asarray(params.transition["s"].log_table, np.float64))
    prior = np.exp(np.asarray(params.initial_state_prior["s"].log_probs, np.float64))
    prior = prior / prior.sum()
    T = int(observations["sense"].shape[0])
    energies = []
    for t in range(T):
        sense = _lowered(H.ConditionalDirichlet(concentration=jnp.asarray(conc), parent_ndims=1))
        o_s, o_i = int(observations["sense"][t]), int(observations["ident"][t])
        seen = True if present is None else bool(present[t])
        joint = np.maximum(np.log(prior), LOG_ZERO) + (sense[o_s] if seen else 0.0) + ident[o_i]
        log_z = logsumexp(joint)
        energies.append(-log_z)
        q = np.exp(joint - log_z)
        if learn and t > 0 and seen:
            conc[o_s] += q  # the conjugate update, one count split by the posterior
        prior = B[:, :, int(actions["a_s"][t])] @ q
    return np.asarray(energies), conc


# ------------------------------------------------------------------ the tick's evidence


def test_one_factor_the_free_energy_is_the_exact_evidence():
    """On one categorical factor the sweep is exact, so ``−free_energy`` is the tick's log
    evidence ``log Σ_s prior(s) Π_m A_m[o_m | s]`` under the engine's lowering, tick by tick."""
    system, params = _sense_system()
    observations, actions = _trace(system, params)
    out = replay(system, params, observations, actions)
    want, _ = _numpy_prequential(system, params, observations, actions, learn=False)
    np.testing.assert_allclose(np.asarray(out.free_energy), want, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(out.log_evidence), -np.asarray(out.free_energy))
    # the same number from the face, tick by tick
    for t in range(3):
        priors = {k: jax.tree_util.tree_map(lambda x, t=t: x[t], v) for k, v in out.priors.items()}
        obs = {k: v[t] for k, v in observations.items()}
        _, detail = system.infer_states(params, obs, priors=priors, detail=True)
        assert float(detail.free_energy) == pytest.approx(want[t], abs=1e-5)
    assert system.infer_states(params, obs, priors=priors)[1] is None


def test_two_factors_the_free_energy_bounds_the_evidence():
    """Mean field over two factors: ``−free_energy`` is at most the exact log evidence of the
    joint, computed by enumeration."""
    from tests.test_system import tmaze_system

    system, params = tmaze_system()
    priors = system.initial_priors(params)
    obs = {
        "location_obs": jnp.asarray(1, jnp.int32),
        "reward": jnp.asarray(0, jnp.int32),
        "cue": jnp.asarray(1, jnp.int32),
    }
    _, detail = system.infer_states(params, obs, priors=priors, detail=True)
    names = ("location_obs", "reward", "cue")
    tables = {n: _lowered(params.likelihood[n]) for n in names}
    deps = {n: system.A_deps[m] for m, n in enumerate(names)}
    log_prior = [
        np.maximum(np.asarray(priors[f].log_probs, np.float64), LOG_ZERO)
        for f in ("location", "context")
    ]
    total = -np.inf
    for s0 in range(len(log_prior[0])):
        for s1 in range(len(log_prior[1])):
            value = log_prior[0][s0] + log_prior[1][s1]
            for n in names:
                index = (int(obs[n]),) + tuple((s0, s1)[g] for g in deps[n])
                value += tables[n][index]
            total = np.logaddexp(total, value)
    assert -float(detail.free_energy) <= total + 1e-5


# ------------------------------------------------------------------ the prequential sum


@pytest.mark.parametrize("learn", [False, True])
def test_the_prequential_sum_matches_a_numpy_loop(learn):
    """``Σ_t free_energy[t]`` over a replay is the prequential code length: with learning off
    every tick is scored under the frozen model; with learning on, under the model as it stood
    before that tick, the counts moving with the record."""
    system, params = _sense_system()
    observations, actions = _trace(system, params, T=9)
    out = replay(system, params, observations, actions, learn=learn)
    want, counts = _numpy_prequential(system, params, observations, actions, learn=learn)
    np.testing.assert_allclose(np.asarray(out.free_energy), want, rtol=1e-5, atol=1e-5)
    assert float(jnp.sum(out.free_energy)) == pytest.approx(float(want.sum()), abs=1e-4)
    if learn:
        np.testing.assert_allclose(
            np.asarray(out.params.likelihood["sense"].concentration), counts, rtol=1e-5, atol=1e-5
        )
        frozen = replay(system, params, observations, actions)
        assert not np.allclose(np.asarray(frozen.free_energy), np.asarray(out.free_energy))


def test_padding_and_absence_leave_the_free_energy_consistent():
    """An absent modality contributes nothing to the tick's free energy; a padded tick's value
    is unspecified and never summed by a caller who honours ``valid``."""
    system, params = _sense_system()
    observations, actions = _trace(system, params, T=6)
    absent = {"ident": jnp.asarray([True, False, True, True, True, True])}
    out = replay(system, params, observations, actions, present=absent)
    full = replay(system, params, observations, actions)
    # at tick 1 only the sense modality is scored
    prior = {k: jax.tree_util.tree_map(lambda x: x[1], v) for k, v in out.priors.items()}
    conc = np.asarray(params.likelihood["sense"].concentration, np.float64)
    sense = _lowered(H.ConditionalDirichlet(concentration=jnp.asarray(conc), parent_ndims=1))
    joint = np.maximum(np.asarray(prior["s"].log_probs, np.float64), LOG_ZERO)
    joint = joint + sense[int(observations["sense"][1])]
    assert float(out.free_energy[1]) == pytest.approx(-logsumexp(joint), abs=1e-5)
    assert float(out.free_energy[0]) == pytest.approx(float(full.free_energy[0]), abs=1e-6)


def test_an_absent_observation_is_neither_scored_nor_learned_from():
    """Along a replay with learning on, a tick whose ``sense`` is absent neither scores that
    modality nor moves its counts, so the prequential sum under a padded or masked trace is
    the NumPy loop's."""
    system, params = _sense_system()
    observations, actions = _trace(system, params, T=8, seed=11)
    seen = np.array([True, True, False, True, True, True, False, True])
    out = replay(
        system, params, observations, actions, learn=True, present={"sense": jnp.asarray(seen)}
    )
    want, counts = _numpy_prequential(system, params, observations, actions, True, seen)
    np.testing.assert_allclose(np.asarray(out.free_energy), want, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(out.params.likelihood["sense"].concentration), counts, rtol=1e-5, atol=1e-5
    )


def _log_normal(y, mean, var):
    return -0.5 * ((y - mean) ** 2 / var + np.log(2 * np.pi * var))


_A_O = H.ConditionalCategorical(
    log_table=jnp.log(jnp.asarray([[0.7, 0.2], [0.2, 0.3], [0.1, 0.5]])), parent_ndims=1
)  # a 3-level report of the 2-level mode


def test_a_switching_channel_beside_a_tabular_one_is_exact_by_enumeration():
    """A categorical mode read by a tabular modality and gating a switching LG observation of
    a mixture-carried factor: ``−free_energy`` is ``log Σ_j p(j) A[o|j] p(y|j)`` enumerated
    over modes and components; absence of either modality drops its factor of the joint."""
    from tests import test_switching_observation as SW

    system = SW.make_switch_system(A_deps=((1, 0), (0,)), modality_names=("y", "o"))
    emission = H.LinearGaussianObservation(H=SW.H_PER_MODE, c=SW.C_PER_MODE, noise=SW.R_PER_MODE)
    params = SW.make_switch_params(likelihood={"y": emission, "o": _A_O})
    y, o = 0.8, 2
    priors = {"mode": SW.MODE_PRIOR, "x": SW.X_PRIOR}
    obs = {"y": jnp.asarray([y]), "o": jnp.asarray(o)}
    _, detail = system.infer_states(params, obs, priors=priors, detail=True)
    w = np.asarray(jax.nn.softmax(SW.X_PRIOR.log_weights), np.float64)
    mu = np.asarray(SW.X_PRIOR.means, np.float64)[:, 0]
    var = np.asarray(SW.X_PRIOR.chols, np.float64)[:, 0, 0] ** 2
    h = np.asarray(emission.H, np.float64)[:, 0, 0]
    c = np.asarray(emission.c, np.float64)[:, 0]
    r = np.asarray(emission.R_chol, np.float64)[:, 0, 0] ** 2
    pm = np.asarray(SW.MODE_PRIOR.probs, np.float64)
    table = _lowered(_A_O)
    log_py = np.array(
        [
            logsumexp(
                [
                    np.log(w[k]) + _log_normal(y, h[j] * mu[k] + c[j], h[j] ** 2 * var[k] + r[j])
                    for k in range(len(w))
                ]
            )
            for j in range(2)
        ]
    )
    joint = logsumexp(np.log(pm) + table[o] + log_py)
    assert -float(detail.free_energy) == pytest.approx(joint, abs=1e-5)
    _, only_o = system.infer_states(params, obs, priors=priors, detail=True, present={"y": False})
    assert -float(only_o.free_energy) == pytest.approx(logsumexp(np.log(pm) + table[o]), abs=1e-5)
    _, only_y = system.infer_states(params, obs, priors=priors, detail=True, present={"o": False})
    assert -float(only_y.free_energy) == pytest.approx(logsumexp(np.log(pm) + log_py), abs=1e-5)


def test_a_mode_aligned_mixture_beside_a_tabular_channel_is_exact_by_enumeration():
    """The mode-aligned mixture path: the evidence of ``y`` enters the tick's free energy under
    the mode belief the tabular stage left, not the mixture's own bank masses, so
    ``−free_energy`` is the enumerated log joint."""
    from tests import test_mixture_filtering as MF

    system = MF.make_system(A_deps=((1,), (0,)), modality_names=("y", "o"))
    emission = H.LinearGaussianObservation(H=MF.H, c=MF.C, noise=MF.R_CHOL)
    params = MF.make_params(likelihood={"y": emission, "o": _A_O})
    y, o = 0.8, 2
    priors = {"mode": MF.MODE_PRIOR, "x": MF.X_PRIOR}
    obs = {"y": jnp.asarray([y]), "o": jnp.asarray(o)}
    _, detail = system.infer_states(params, obs, priors=priors, detail=True)
    w = np.asarray(jax.nn.softmax(MF.X_PRIOR.log_weights), np.float64)
    mu = np.asarray(MF.X_PRIOR.means, np.float64)[:, 0]
    var = np.asarray(MF.X_PRIOR.chols, np.float64)[:, 0, 0] ** 2
    r = float(np.asarray(MF.R_CHOL)[0, 0]) ** 2
    pm = np.asarray(MF.MODE_PRIOR.probs, np.float64)
    spb = len(w) // 2
    log_py = np.zeros(2)
    for j in range(2):
        bank = w[j * spb : (j + 1) * spb] / w[j * spb : (j + 1) * spb].sum()
        log_py[j] = logsumexp(
            [
                np.log(bank[i]) + _log_normal(y, mu[j * spb + i], var[j * spb + i] + r)
                for i in range(spb)
            ]
        )
    table = _lowered(_A_O)
    joint = logsumexp(np.log(pm) + table[o] + log_py)
    assert -float(detail.free_energy) == pytest.approx(joint, abs=1e-5)
    # the detail's evidence of y reads the same mode belief the free energy does
    q1 = np.log(pm) + table[o] - logsumexp(np.log(pm) + table[o])
    assert float(detail.log_evidence["y"]) == pytest.approx(logsumexp(q1 + log_py), abs=1e-5)


def test_a_plain_lg_factor_beside_a_categorical_one_adds_its_evidence():
    emission = H.LinearGaussianObservation(
        H=jnp.asarray([[1.0]]), c=jnp.zeros(1), noise=jnp.asarray([[0.5]])
    )
    report = H.ConditionalCategorical(
        log_table=jnp.log(jnp.asarray([[0.6, 0.1, 0.3], [0.3, 0.6, 0.2], [0.1, 0.3, 0.5]])),
        parent_ndims=1,
    )
    system = H.System(
        A_deps=((0,), (1,)),
        B_deps=((), ()),
        controlled=(False, False),
        policy_len=1,
        modality_names=("o", "y"),
        factor_names=("s", "x"),
    )
    params = H.SystemParams(
        likelihood={"o": report, "y": emission},
        transition={
            "s": H.ConditionalCategorical(
                log_table=jnp.log(jnp.full((3, 1), 1 / 3)), parent_ndims=1
            ),
            "x": H.LinearGaussianTransition(
                F=jnp.asarray([[1.0]]),
                innovations=H.ConditionalGaussian(
                    loc=jnp.zeros((1, 1)), chol=jnp.asarray([[[0.3]]]), parent_ndims=1
                ),
            ),
        },
        initial_state_prior={
            "s": H.Categorical(logits=jnp.asarray([0.3, -0.2, 0.9])),
            "x": H.Gaussian(loc=jnp.asarray([0.2]), chol=jnp.asarray([[1.0]])),
        },
    )
    priors = system.initial_priors(params)
    obs = {"o": jnp.asarray(1), "y": jnp.asarray([0.4])}
    _, detail = system.infer_states(params, obs, priors=priors, detail=True)
    ps = np.asarray(priors["s"].probs, np.float64)
    log_po = logsumexp(np.log(ps) + _lowered(report)[1])
    log_py = _log_normal(0.4, 0.2, 1.0 + 0.25)
    assert -float(detail.free_energy) == pytest.approx(log_po + log_py, abs=1e-5)


def test_the_parent_axes_are_contracted_in_dependency_order():
    """Equal cardinalities everywhere, so a wrong axis order would be silent: the free energy
    equals an explicit einsum over the parents in ``A_deps`` order."""
    from hybrid_aif.inference.discrete import mean_field_free_energy

    rng = np.random.default_rng(5)
    L = 3
    tables = [
        H.ConditionalCategorical(
            log_table=jnp.log(
                jnp.asarray(rng.dirichlet(np.ones(L), size=(L, L)).transpose(2, 0, 1))
            ),
            parent_ndims=2,
        ),
        H.ConditionalCategorical(
            log_table=jnp.log(
                jnp.asarray(rng.dirichlet(np.ones(L), size=(L, L, L)).transpose(3, 0, 1, 2))
            ),
            parent_ndims=3,
        ),
    ]
    deps = ((1, 0), (2, 0, 1))
    priors = tuple(H.Categorical(logits=jnp.asarray(rng.normal(size=L))) for _ in range(3))
    posts = tuple(H.Categorical(logits=jnp.asarray(rng.normal(size=L))) for _ in range(3))
    obs = (jnp.asarray(2), jnp.asarray(0))
    got = float(mean_field_free_energy(tuple(tables), deps, priors, posts, obs))
    q = [np.asarray(b.probs, np.float64) for b in posts]
    want = sum(
        float(np.sum(q[f] * (np.log(q[f]) - np.maximum(np.asarray(priors[f].log_probs), LOG_ZERO))))
        for f in range(3)
    )
    want -= np.einsum("ab,a,b->", _lowered(tables[0])[2], q[1], q[0])
    want -= np.einsum("abc,a,b,c->", _lowered(tables[1])[0], q[2], q[0], q[1])
    assert got == pytest.approx(want, abs=1e-5)


# ------------------------------------------------------------------ the linear-Gaussian path


def test_on_the_linear_gaussian_path_the_free_energy_is_minus_the_evidence():
    hill = _example("hill_lg")
    system, params = hill.make_agent()
    traj = hill.closed_loop(jax.random.PRNGKey(0), system, params, num_steps=6)
    out = replay(system, params, {"position": traj.ys}, {"a_state": traj.actions.astype(jnp.int32)})
    assert out.free_energy.shape == (6,)
    for t in range(6):
        priors = {k: jax.tree_util.tree_map(lambda x, t=t: x[t], v) for k, v in out.priors.items()}
        _, detail = system.infer_states(
            params, {"position": traj.ys[t]}, priors=priors, detail=True
        )
        assert float(out.log_evidence[t]) == pytest.approx(
            float(detail.log_evidence["position"]), abs=1e-5
        )
        assert float(detail.free_energy) == pytest.approx(-float(detail.log_evidence["position"]))


def test_the_amortized_path_has_no_free_energy():
    from hybrid_aif.reference.raif.world import tiny_rssm

    system, params = tiny_rssm(2, 1, key=jax.random.PRNGKey(0))
    beliefs = system.initial_priors(params)
    obs = {"state": jnp.zeros(2), "reward": jnp.zeros(1), "cont": jnp.ones(1)}
    _, detail = system.infer_states(params, obs, priors=beliefs, detail=True)
    assert detail is None or detail.free_energy is None


# ------------------------------------------------------------------ the resolution bridge


def test_the_absorbed_counts_become_the_measures_resolution():
    """After a replay with learning on, each learned slot's absorbed count is its address's
    ``n``; a slot that absorbed nothing, or has no closed form, is left out and its family link
    unscored."""
    system, params = _sense_system()
    observations, actions = _trace(system, params, T=9)
    out = replay(system, params, observations, actions, learn=True)
    evidence = system.log_evidence(params, out.params)
    n = resolution_from_evidence(evidence)
    assert set(n) == {"likelihood:sense"}
    assert n["likelihood:sense"] == pytest.approx(8.0, abs=1e-4)  # tick 0 learns nothing
    assert evidence["likelihood:sense"].nats is not None
    dl = description_length(system, n, params=out.params, prior=params)
    sense = dl.per_address["likelihood:sense"]
    assert sense.n == pytest.approx(8.0, abs=1e-4) and sense.family is not None
    assert sense.posterior is not None and sense.posterior > 0.0
    assert dl.per_address["transition:s"].family is None
    assert ("transition:s", "family") in dl.unscored


# ------------------------------------------------------------------ the parameter envelope


def test_describe_parameters_reads_the_envelope_off_the_document():
    from hybrid_aif.reference.raif.world import tiny_rssm
    from tests.test_system import tmaze_system

    system, params = tmaze_system()
    document, arrays = describe(system, params)
    records = describe_parameters(json.loads(json.dumps(document)), arrays)
    by_name = {r.address: r for r in records}
    assert by_name["B_location"].shape == (5, 5, 5)
    assert by_name["B_location"].domain == "log_simplex"
    assert by_name["B_location"].source == "clamp" and by_name["B_location"].dtype == "float32"
    assert all(r.source == "clamp" for r in records)
    assert len(records) == len(
        [v for v in document["variables"] if v in document.get("clamps", {})]
    )

    compose = _example("compose_target_factor")
    document, arrays = describe(compose.system, compose.params)
    records = {r.address: r for r in describe_parameters(document, arrays)}
    learned = [r for r in records.values() if r.source == "prior"]
    assert learned and all(r.domain == "log_simplex" for r in learned)

    system, params = tiny_rssm(2, 1, key=jax.random.PRNGKey(0))
    document, arrays = describe(system, params)
    records = describe_parameters(document, arrays)
    refs = [r for r in records if r.source == "ref"]
    assert refs and all(
        r.dtype == str(np.asarray(arrays[r.address]).dtype) for r in refs if r.address in arrays
    )
    assert any("/" in r.address for r in records)  # a tree variable's leaves
    without = describe_parameters(document)  # no arrays: a ref keeps the declared shape
    assert len(without) == len(records)
    with pytest.raises(H.DescriptionError):
        describe_parameters({"hybrid_aif_description": "0.3"})
    with pytest.raises(H.DescriptionError):
        describe_parameters({"variables": {}})
