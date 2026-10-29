"""Micro-benchmark: hybrid-aif vs pymdp (JAX) on identical T-maze-class models.

Both libraries run the *same algorithms* (mean-field FPI for state inference;
exhaustive policy enumeration scored by expected free energy) on the *same
model tensors* (the classic 5-location / 2-context T-maze), both jitted, both
on the same backend. This is therefore a benchmark of implementation overhead,
not of algorithmic advantage — the interesting result is that the term-tree
metrics layer and the batch (multi-agent) path cost little.

What is measured:

(a) State inference — our ``System.infer_states`` (a thin wrapper over
    :func:`hybrid_aif.inference.fpi`) vs pymdp's ``run_factorized_fpi`` (the
    kernel behind ``Agent.infer_states``), identical priors/observations,
    16 sweeps each.
(b) Policy inference — our ``System.infer_policies`` (which also assembles the
    per-term/per-block Interval metrics tree) vs pymdp's
    ``update_posterior_policies``, same policy set, same gamma, C matched to
    our normalized log-prob targets. Outputs are asserted equal first.
(c) Batch scaling — one full tick (state + policy inference) vmapped over
    1/10/100 agents with per-agent priors and observations. On the pymdp side
    we vmap its functional core directly — the same thing pymdp's ``Agent``
    does internally with its batch axis — so neither side pays Python wrapper
    overhead. That is the fairest comparison we could construct.

Methodology: jit warm-up call excluded, every timed call ends in
``jax.block_until_ready``, median of >= 20 repetitions via
``time.perf_counter``. Run: ``.venv/bin/python examples/benchmark_pymdp.py``
(add ``--quick`` for a fast smoke run).
"""

import argparse
import platform
import statistics
import time
from importlib import metadata
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from pymdp.algos import run_factorized_fpi
from pymdp.control import update_posterior_policies

from hybrid_aif import Categorical, ConditionalCategorical
from hybrid_aif.reference.envs.tmaze import tmaze
from hybrid_aif.system import System, SystemParams

_EPS = 1e-16
NUM_ITERS = 16  # FPI sweeps, both sides
GAMMA = 16.0
POLICY_LEN = 2
NUM_OBS = (5, 3, 3)  # per-modality outcome counts (location, reward, cue)


def _cc(table: jax.Array, parent_ndims: int) -> ConditionalCategorical:
    return ConditionalCategorical(
        log_table=jnp.log(jnp.clip(table, _EPS)), parent_ndims=parent_ndims
    )


class Workload(NamedTuple):
    """One T-maze model, prepared for both libraries."""

    system: System
    params: SystemParams
    policies: jax.Array
    # pymdp-convention inputs (adapters per tests/test_discrete_parity.py):
    A_py: list[jax.Array]  # likelihood tensors
    B_py: list[jax.Array]  # transition tensors
    C_py: list[jax.Array]  # log-preferences == our targets' normalized log_probs
    E_py: jax.Array  # uniform habit prior over policies
    A_deps: list[list[int]]
    B_deps: list[list[int]]


def build_workload(policy_len: int = POLICY_LEN) -> Workload:
    """The tests' tmaze_system, plus the same tensors in pymdp's conventions."""
    env, p = tmaze(reward_probability=0.98, punishment_probability=0.98, cue_validity=0.98)
    system = System(
        A_deps=env.A_deps,
        B_deps=env.B_deps,
        controlled=(True, False),
        policy_len=policy_len,
        gamma=GAMMA,
        epistemic=True,
        modality_names=("location_obs", "reward", "cue"),
        factor_names=("location", "context"),
    )
    # Designate all three modalities (reward informative, others uniform) so both
    # libraries evaluate utility + info gain over the same three modalities.
    obs_targets = {
        "location_obs": Categorical(logits=jnp.zeros(5)),
        "reward": Categorical(logits=jnp.asarray([0.0, 4.0, -6.0])),
        "cue": Categorical(logits=jnp.zeros(3)),
    }
    #  the pymdp-shaped tables enter by position, through the one converter
    params = system.from_positional(
        "params",
        SystemParams(
            likelihood=tuple(_cc(a, len(d)) for a, d in zip(p.likelihood, env.A_deps, strict=True)),
            transition=tuple(
                _cc(b, len(d) + 1) for b, d in zip(p.transition, env.B_deps, strict=True)
            ),
            initial_state_prior=tuple(Categorical.from_probs(d) for d in p.initial_state_prior),
            obs_targets=obs_targets,
            state_targets={},
            terminal_targets={},
        ),
    )
    #  one column, the location control; pymdp's policies carry a column per factor, so the
    #  context factor gets its no-op 0
    ours = system.policies(params)
    policies = jnp.concatenate([ours, jnp.zeros_like(ours)], axis=-1)
    return Workload(
        system=system,
        params=params,
        policies=policies,
        A_py=[jnp.asarray(a) for a in p.likelihood],
        B_py=[jnp.asarray(b) for b in p.transition],
        C_py=[obs_targets[n].log_probs for n in ("location_obs", "reward", "cue")],
        E_py=jnp.ones(policies.shape[0]) / policies.shape[0],
        A_deps=[list(d) for d in env.A_deps],
        B_deps=[list(d) for d in env.B_deps],
    )


# ------------------------------------------------------------------- the two sides


def make_fns(w: Workload) -> dict[str, Any]:
    """Jitted callables for both libraries; inputs adapted per side's conventions.

    Both sides take the same positional (pymdp-shaped) arrays; ours enter and leave the
    name-keyed faces through ``System.from_positional`` / ``to_positional``, at trace time.
    """
    sys_, params = w.system, w.params

    def ours_state(prior_logits: tuple[jax.Array, ...], obs: tuple[jax.Array, ...]):
        priors = sys_.from_positional(
            "beliefs", tuple(Categorical(logits=lg) for lg in prior_logits)
        )
        posterior, _ = sys_.infer_states(params, sys_.from_positional("observations", obs), priors)
        return tuple(q.probs for q in sys_.to_positional("beliefs", posterior))

    def pymdp_state(prior_probs: list[jax.Array], obs_onehot: list[jax.Array]):
        return run_factorized_fpi(w.A_py, obs_onehot, prior_probs, w.A_deps, num_iter=NUM_ITERS)

    def ours_policy(belief_logits: tuple[jax.Array, ...]):
        beliefs = sys_.from_positional(
            "beliefs", tuple(Categorical(logits=lg) for lg in belief_logits)
        )
        return sys_.infer_policies(params, beliefs)[0].probs

    def ours_policy_no_metrics(belief_logits: tuple[jax.Array, ...]):
        beliefs = sys_.from_positional(
            "beliefs", tuple(Categorical(logits=lg) for lg in belief_logits)
        )
        return sys_.infer_policies(params, beliefs, metrics=False)[0].probs

    def pymdp_policy(qs: list[jax.Array]):
        return update_posterior_policies(
            w.policies,
            qs,
            w.A_py,
            w.B_py,
            w.C_py,
            w.E_py,
            None,
            None,
            w.A_deps,
            w.B_deps,
            gamma=GAMMA,
        )[0]

    def ours_tick(prior_logits, obs):
        return ours_policy(tuple(jnp.log(q) for q in ours_state(prior_logits, obs)))

    def pymdp_tick(prior_probs, obs_onehot):
        return pymdp_policy(pymdp_state(prior_probs, obs_onehot))

    return {
        "ours_state": jax.jit(ours_state),
        "pymdp_state": jax.jit(pymdp_state),
        "ours_policy": jax.jit(ours_policy),
        "ours_policy_no_metrics": jax.jit(ours_policy_no_metrics),
        "pymdp_policy": jax.jit(pymdp_policy),
        "ours_tick_batched": jax.jit(jax.vmap(ours_tick)),
        "pymdp_tick_batched": jax.jit(jax.vmap(pymdp_tick)),
    }


def make_inputs(w: Workload, batch: int | None = None, seed: int = 0) -> dict[str, Any]:
    """Identical random priors/observations, in each side's input convention.

    ``batch=None`` gives single-agent inputs; an int adds a leading agent axis.
    """
    rng = np.random.default_rng(seed)
    shape = () if batch is None else (batch,)
    sizes = [w.params.initial_state_prior[n].probs.shape[0] for n in w.system.names.states]
    prior_logits = tuple(
        jnp.asarray(rng.normal(scale=0.5, size=(*shape, n)).astype(np.float32)) for n in sizes
    )
    obs = tuple(jnp.asarray(rng.integers(n, size=shape or None), jnp.int32) for n in NUM_OBS)
    prior_probs = [jax.nn.softmax(lg, axis=-1) for lg in prior_logits]
    obs_onehot = [jax.nn.one_hot(o, n) for o, n in zip(obs, NUM_OBS, strict=True)]
    return {
        "prior_logits": prior_logits,
        "obs": obs,
        "prior_probs": prior_probs,
        "obs_onehot": obs_onehot,
    }


def verify_parity(w: Workload, fns: dict[str, Any]) -> float:
    """Assert both libraries produce the same posteriors; return max |dq_pi|."""
    x = make_inputs(w, seed=7)
    qs_ours = fns["ours_state"](x["prior_logits"], x["obs"])
    qs_py = fns["pymdp_state"](x["prior_probs"], x["obs_onehot"])
    for a, b in zip(qs_ours, qs_py, strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-5)
    q_ours = fns["ours_policy"](tuple(jnp.log(q) for q in qs_ours))
    q_py = fns["pymdp_policy"](list(qs_py))
    np.testing.assert_allclose(np.asarray(q_ours), np.asarray(q_py), atol=1e-5)
    q_ours_nm = fns["ours_policy_no_metrics"](tuple(jnp.log(q) for q in qs_ours))
    np.testing.assert_allclose(np.asarray(q_ours_nm), np.asarray(q_py), atol=1e-5)
    return float(np.abs(np.asarray(q_ours) - np.asarray(q_py)).max())


# ------------------------------------------------------------------------- timing


def median_time_us(fn, *args, reps: int = 20) -> float:
    """Median wall time in microseconds; one untimed warm-up (jit compile) first."""
    jax.block_until_ready(fn(*args))  # warm-up: compilation excluded
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e6


def run_benchmarks_rounds(
    rounds: int = 1, reps: int = 20, batch_sizes: tuple[int, ...] = (1, 10, 100)
) -> list[tuple[str, list[float], list[float]]]:
    """Rows of (label, ours_us_per_round, pymdp_us_per_round); parity asserted first.

    Everything is jitted once, then each round repeats every measurement through
    the same compiled functions — so the spread across rounds is OS-scheduling and
    cache noise, which on microsecond workloads is large (single medians swing 2-3x
    run-to-run). Report medians with the min-max spread, never one-shot numbers.
    """
    w = build_workload()
    fns = make_fns(w)
    verify_parity(w, fns)
    x = make_inputs(w)
    qs = fns["ours_state"](x["prior_logits"], x["obs"])
    log_qs = tuple(jnp.log(q) for q in qs)
    qs_py = [jnp.asarray(q) for q in qs]
    n_pol = w.policies.shape[0]

    measurements: list[tuple[str, Any, Any]] = [
        (
            f"state inference (FPI, {NUM_ITERS} sweeps)",
            lambda: median_time_us(fns["ours_state"], x["prior_logits"], x["obs"], reps=reps),
            lambda: median_time_us(
                fns["pymdp_state"], x["prior_probs"], x["obs_onehot"], reps=reps
            ),
        ),
        (
            f"policy inference ({n_pol} policies, H={POLICY_LEN})",
            lambda: median_time_us(fns["ours_policy"], log_qs, reps=reps),
            lambda: median_time_us(fns["pymdp_policy"], qs_py, reps=reps),
        ),
        (
            "policy inference, metrics=False",
            lambda: median_time_us(fns["ours_policy_no_metrics"], log_qs, reps=reps),
            lambda: median_time_us(fns["pymdp_policy"], qs_py, reps=reps),
        ),
    ]
    for batch in batch_sizes:
        xb = make_inputs(w, batch=batch, seed=batch)
        measurements.append(
            (
                f"full tick, vmap x {batch} agent{'s' if batch > 1 else ''}",
                lambda xb=xb: median_time_us(
                    fns["ours_tick_batched"], xb["prior_logits"], xb["obs"], reps=reps
                ),
                lambda xb=xb: median_time_us(
                    fns["pymdp_tick_batched"], xb["prior_probs"], xb["obs_onehot"], reps=reps
                ),
            )
        )

    rows: list[tuple[str, list[float], list[float]]] = [
        (label, [], []) for label, _, _ in measurements
    ]
    for _ in range(rounds):
        for (_, f_ours, f_py), (_, ours, ref) in zip(measurements, rows, strict=True):
            ours.append(f_ours())
            ref.append(f_py())
    return rows


def run_benchmarks(
    reps: int = 20, batch_sizes: tuple[int, ...] = (1, 10, 100)
) -> list[tuple[str, float, float]]:
    """All rows: (label, ours_us, pymdp_us). Parity is asserted before timing."""
    rows = run_benchmarks_rounds(rounds=1, reps=reps, batch_sizes=batch_sizes)
    return [(label, ours[0], ref[0]) for label, ours, ref in rows]


def format_variance_table(rows: list[tuple[str, list[float], list[float]]]) -> str:
    """Median [min-max] per side across measurement rounds; ratio of medians."""
    n = len(rows[0][1])
    header = f"{'benchmark (T-maze, CPU)':<38}{'hybrid-aif':>18}{'pymdp':>18}{'ratio':>8}"
    lines = [f"median [min-max] us across {n} measurement rounds", header, "-" * len(header)]
    for label, ours, ref in rows:
        mo, mr = statistics.median(ours), statistics.median(ref)
        lines.append(
            f"{label:<38}"
            f"{f'{mo:.0f} [{min(ours):.0f}-{max(ours):.0f}]':>18}"
            f"{f'{mr:.0f} [{min(ref):.0f}-{max(ref):.0f}]':>18}"
            f"{mo / mr:>7.2f}x"
        )
    lines.append("-" * len(header))
    lines.append("ratio = ratio of medians (< 1 means hybrid-aif is faster)")
    return "\n".join(lines)


def format_table(rows: list[tuple[str, float, float]]) -> str:
    header = f"{'benchmark (T-maze, CPU)':<42}{'hybrid-aif':>12}{'pymdp':>12}{'ratio':>8}"
    lines = [header, "-" * len(header)]
    for label, ours, ref in rows:
        lines.append(f"{label:<42}{ours:>9.0f} us{ref:>9.0f} us{ours / ref:>7.2f}x")
    lines.append("-" * len(header))
    lines.append("ratio = hybrid-aif / pymdp wall time (< 1 means hybrid-aif is faster)")
    return "\n".join(lines)


def machine_report() -> str:
    try:
        pymdp_version = metadata.version("inferactively-pymdp")
    except metadata.PackageNotFoundError:  # pragma: no cover - installed in dev env
        pymdp_version = "unknown"
    return (
        f"machine: {platform.platform()} / {platform.machine()}; "
        f"backend: {jax.default_backend()}; "
        f"jax {jax.__version__}, pymdp {pymdp_version}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="smoke run: 5 reps, batches (1, 10)")
    ap.add_argument("--reps", type=int, default=None, help="override repetition count")
    ap.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="measurement rounds through the same jitted fns; > 1 reports min-max spread",
    )
    args = ap.parse_args()
    reps = args.reps if args.reps is not None else (5 if args.quick else 20)
    batch_sizes = (1, 10) if args.quick else (1, 10, 100)

    print(machine_report())
    print(f"median of {reps} reps, jit warm-up excluded, block_until_ready per call\n")
    t0 = time.perf_counter()
    if args.rounds > 1:
        print(format_variance_table(run_benchmarks_rounds(args.rounds, reps, batch_sizes)))
    else:
        print(format_table(run_benchmarks(reps=reps, batch_sizes=batch_sizes)))
    print(
        "\ncaveats: same algorithms on both sides (mean-field FPI + enumerated-policy EFE),\n"
        "both JAX, both jitted, tiny CPU workloads -- differences are dispatch/implementation\n"
        "overhead, not algorithmic. hybrid-aif's default policy row builds its per-term\n"
        "Interval metrics tree (no counterpart in pymdp); the metrics=False row scores the\n"
        "same posterior without it and its residual vs pymdp is mostly the G bound-pair.\n"
        "pymdp's batched path is its own functional core vmapped (what Agent does\n"
        "internally), so neither side pays Python wrapper overhead."
    )
    print(f"total benchmark wall time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
