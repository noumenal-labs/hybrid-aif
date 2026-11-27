"""An own-belief read is an emission the System observes; where no key enters, its datum equals the
datum of a read of the previous round's belief taken from the carry.

``tests/data/own_read_datum.npz`` holds what a model that took such a datum from the carry
computed: the datum of a read of a Gaussian factor (its mean), of a categorical factor (its
probability vector) and of the same Gaussian through a grid (the largest cell, the cell mass), the
posterior one tick later with those data, the per-modality log evidence, and a replay of five
ticks that fed the data from its carry. The emission version, which the caller (or ``replay``)
feeds, reproduces every number bit for bit.
"""

import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_aif import (
    Categorical,
    ConditionalCategorical,
    Emission,
    Gaussian,
    Grid,
    LinearGaussianObservation,
    LinearGaussianTransition,
    System,
    SystemParams,
    replay,
)
from hybrid_aif.distributions import ConditionalGaussian

FIXTURE = np.load(Path(__file__).parent / "data" / "own_read_datum.npz")
GRID = Grid(shape=(2, 2), low=-2.0, high=2.0)
CELL = np.array([[0.6, 0.2, 0.1], [0.1, 0.6, 0.2], [0.1, 0.1, 0.6], [0.2, 0.1, 0.1]])
CELL = CELL / CELL.sum(axis=0)
READS = ("read_x", "read_kind", "cell_mode", "cell_stats")
KEY = jnp.zeros((2,), jnp.uint32)


def model():
    """Two factors, ``x`` a 2-D random walk and ``kind`` a sticky categorical, observed through
    ``obs_x`` and ``obs_kind``; four modalities read the factors' own beliefs: ``read_x`` (the
    mean of ``x``), ``read_kind`` (the probability vector of ``kind``), ``cell_mode`` and
    ``cell_stats`` (the cell of a grid ``x`` falls in, as the largest cell and as the mass)."""
    system = System(
        A_deps=((0,), (1,), (0,), (1,), (1,), (1,)),
        B_deps=((), (1,)),
        controlled=(False, False),
        policy_len=1,
        modality_names=("obs_x", "obs_kind", *READS),
        state_names=("x", "kind"),
        epistemic=False,
        emissions={
            "x_mean": Emission(("x",), realise="mode", kernel="identity"),
            "kind_vector": Emission(("kind",), realise="stats", kernel="identity"),
            "cell_largest": Emission(("x",), realise="mode", kernel=GRID),
            "cell_mass": Emission(("x",), realise="stats", kernel=GRID),
        },
        observes={
            "read_x": "x_mean",
            "read_kind": "kind_vector",
            "cell_mode": "cell_largest",
            "cell_stats": "cell_mass",
        },
    )
    eye3 = jnp.eye(3)
    cell = jnp.asarray(CELL, jnp.float32)
    params = SystemParams(
        likelihood={
            "obs_x": LinearGaussianObservation(
                H=jnp.eye(2), c=jnp.zeros(2), noise=0.3 * jnp.eye(2)
            ),
            "obs_kind": ConditionalCategorical(jnp.log(eye3 * 0.9 + 0.05), 1),
            "read_x": LinearGaussianObservation(
                H=jnp.eye(2), c=jnp.zeros(2), noise=0.2 * jnp.eye(2)
            ),
            "read_kind": ConditionalCategorical(jnp.log(eye3 * 0.7 + 0.15), 1),
            "cell_mode": ConditionalCategorical(jnp.log(cell), 1),
            "cell_stats": ConditionalCategorical(jnp.log(cell), 1),
        },
        transition={
            "x": LinearGaussianTransition(
                F=jnp.eye(2),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 2)),
                    chol=jnp.broadcast_to(0.5 * jnp.eye(2), (1, 2, 2)),
                    parent_ndims=1,
                ),
            ),
            "kind": ConditionalCategorical(jnp.log(eye3 * 0.8 + 0.1), 1),
        },
        initial_state_prior={
            "x": Gaussian(loc=jnp.zeros(2), chol=jnp.eye(2)),
            "kind": Categorical(logits=jnp.zeros(3)),
        },
    )
    #  the fixture was recorded in float32: the model is float32 in either mode (cast under
    #  JAX's 64-bit mode, where it builds in float64)
    from tests.test_run_parity import as_recorded

    return system, as_recorded(params)


def trace(T):
    rng = np.random.default_rng(7)
    xs = jnp.asarray(np.cumsum(rng.normal(size=(T, 2)), axis=0), jnp.float32)
    kinds = jnp.asarray(rng.integers(0, 3, size=T), jnp.int32)
    return xs, kinds


def test_the_emissions_equal_the_data_taken_from_the_carry():
    system, params = model()
    xs, kinds = trace(5)
    first, _ = system.infer_states(params, {"obs_x": xs[0], "obs_kind": kinds[0]}, None)
    emitted = system.emit(KEY, params, first)
    for name, variable in (
        ("datum_read_x", "x_mean"),
        ("datum_read_kind", "kind_vector"),
        ("datum_cell_mode", "cell_largest"),
        ("datum_cell_stats", "cell_mass"),
    ):
        np.testing.assert_array_equal(np.asarray(emitted[variable]), FIXTURE[name], err_msg=name)


def test_the_posterior_with_the_supplied_data_equals_the_one_from_the_carry():
    system, params = model()
    xs, kinds = trace(5)
    first, _ = system.infer_states(params, {"obs_x": xs[0], "obs_kind": kinds[0]}, None)
    datum = system.emit(KEY, params, first)
    second, info = system.infer_states(
        params, {"obs_x": xs[1], "obs_kind": kinds[1], **datum}, None, detail=True
    )
    np.testing.assert_array_equal(np.asarray(second["x"].loc), FIXTURE["x_loc_1"])
    np.testing.assert_array_equal(np.asarray(second["x"].chol), FIXTURE["x_chol_1"])
    np.testing.assert_array_equal(np.asarray(second["kind"].probs), FIXTURE["kind_probs_1"])
    recorded = [m for m in ("obs_x", "obs_kind", *READS) if f"log_evidence_{m}" in FIXTURE.files]
    assert "read_x" in recorded
    for m in recorded:
        np.testing.assert_array_equal(
            np.asarray(info.log_evidence[m]), FIXTURE[f"log_evidence_{m}"], err_msg=m
        )


def test_replay_feeds_the_emissions_it_realises_and_records_them():
    system, params = model()
    xs, kinds = trace(5)
    out = replay(system, params, {"obs_x": xs, "obs_kind": kinds}, {}, metrics=False)
    if jax.config.jax_enable_x64:
        #  float32 arrays throughout, but the Python constants in the program are taken in
        #  float64 under 64-bit mode before they meet one, and the replay differs from the
        #  recording by up to an ulp of float32 at the last tick
        assert out.beliefs["kind"].probs.dtype == jnp.float32
        same = functools.partial(np.testing.assert_allclose, rtol=1e-6, atol=1e-7)
    else:
        same = np.testing.assert_array_equal
    same(np.asarray(out.beliefs["x"].loc), FIXTURE["replay_x_loc"])
    same(np.asarray(out.beliefs["kind"].probs), FIXTURE["replay_kind_probs"])
    same(np.asarray(out.free_energy), FIXTURE["replay_free_energy"])
    # what is recorded at tick t is the emission of that tick's posterior
    np.testing.assert_array_equal(
        np.asarray(out.emitted["x_mean"]), np.asarray(out.beliefs["x"].loc)
    )
    np.testing.assert_array_equal(
        np.asarray(out.emitted["kind_vector"]), np.asarray(out.beliefs["kind"].probs)
    )
    assert out.emitted["cell_largest"].dtype == jnp.int32
    assert out.emitted["cell_mass"].shape == (5, GRID.n_cells)
