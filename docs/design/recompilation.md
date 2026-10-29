# Recompilation under the intended agent loop

Status: audit 2026-10-23; R1–R5 SETTLED (Alex, 2026-10-23); R6–R7 deferred.

## The audit

Used as the documentation and examples show (the faces wrapped once per model shape in
`eqx.filter_jit` or `jax.jit`, the System passed as an argument, values changing between ticks),
every face traced once over twenty ticks on every model measured: the T-maze, a learnable T-maze,
linear-Gaussian models under the enumerating, gradient, CEM and mixed planners, the sophisticated
planner, a switching mixture and an amortized agent. A changing number of available options is not
a retrace when the System declares the maximum and a `legal=` mask of fixed shape is passed every
tick. What does cause recompilation:

- a Python scalar passed to an `eqx.filter_jit` function is static, so a schedule passed as a
  Python float (`lr`, `decay`, `alpha`) recompiles per value, and a Python int observation per
  value combination; `replay` takes `lr` the same way;
- tick-0 structure that differs from tick 1: parameter leaves built with weak dtypes, an initial
  mixture smaller than the one the first push produces, `priors=None` on the first tick, `legal=`
  passed on some ticks only (each one extra compile);
- every static field of `System` (`gamma`, `policy_len`, `planner`, `branch_on`, ...): one compile
  per value, so annealing the precision recompiles at every value;
- input dtype drift (uint8 against int32 observations, int against bool masks, the two PRNG key
  forms).

JAX's `jax_explain_cache_misses` names the changed argument under `jax.jit`, but under
`eqx.filter_jit` its paths are opaque and a static change can be misreported.

## Decisions

**R1 — Tick-0 structure is fixed where it is produced.** Parameters and initial priors hold strong
dtypes; an initial mixture is padded to the component count the first push produces.

**R2 — A loop-invariance test.** Twenty ticks on each representative model under a counting
`eqx.filter_jit`, asserting one trace per face and the same pytree structure, shapes, dtypes and
weak types in and out of `update_empirical_prior` and `learn`.

**R3 — Small fixes.** `replay` turns a Python-float `lr` into an array unless it is the
deliberately static zero; the mountain-car example passes its decaying rate as an array; an
observation given as a list, a scalar or with a batch axis is refused with a sentence.

**R4 — Documentation.** A short section on compiling an agent loop: what is static, schedules as
arrays, start from `initial_priors`, always pass `legal=`, how to diagnose.

**R5 — A traced precision, in the parameters.** `SystemParams.precision`, an optional array,
is the policy precision γ when present; `System.gamma` remains the default when it is absent, so
existing models are unchanged. Annealing writes a new value into the parameters, which flow traced
through every face, so it recompiles nothing; it is also the slot a learned precision would
occupy. The description format writes it as any parameter. It was the only value of this shape:
the other numeric fields of `System` fix shapes or schedules (`num_iters`, `mixture_k`,
`reduction_period`, `policy_len`); `act`'s `alpha` and the learning rate are call arguments,
traced when passed as arrays; planner options bound by `make_*_planner` are closure constants, a
different planner when changed.

## Deferred

**R6 — A library-owned compiled step** (`System.stepper(...)`): a `jit` boundary the library owns,
with inputs canonicalised in front of it, so the input-type traps above cannot occur. The
composition driver's internal one-tick function already follows this pattern.

**R7 — An opt-in retrace watcher** (`hybrid_aif.debug.watch`): warns on a repeat trace with the
same static configuration, naming the argument whose structure, dtype or shape changed, or the
System field that changed, which `jax_explain_cache_misses` does not do under `eqx.filter_jit`.

## Build notes (2026-10-23)

What was built, per decision, and what was measured on the audit's loops (20 ticks, one counting
`eqx.filter_jit` per face, values only changing; the audit's scripts, before at b96d9f8).

**R1.** `SystemParams` holds every pytree field strong-typed: a field converter
(`system/core.py` `_strong`) rebuilds a field only when it holds a weak-typed array leaf, with
the same dtype and values, and returns the field itself otherwise. Parameters and `initial_priors`
(which passes the declared leaves through) therefore enter the faces as `update_empirical_prior`
and `learn` return them. Measured: the scalar linear-Gaussian model's `infer_states` 2 traces to 1
(its initial prior held a weak float leaf); a learnable T-maze with a weak Dirichlet
concentration, `learn` fed its own output, 2 traces to 1. **Deviation: the initial mixture is not
padded.** A switching mixture's `initial_priors` still holds 2 components against the 8 the first
push produces, so its tick 0 is one extra trace (`switching_filter.tick`, two agents: 4 caches,
unchanged). The size is derivable (`mixture_k`), but padding the declared prior inside
`initial_priors` changed numerics: eight tests of `tests/test_switching_filter.py` failed, the
first action of the probing agent among them, because under `mixture_schedule="mode_aligned"` the
first filtering step grows and reduces by slot ownership, and a prior padded to eight components
fills slots a two-component prior does not. A caller who wants one trace pads the prior with
`hybrid_aif.objectives.hybrid.pad_mixture(q, mixture_k)` and accepts that first step;
`tests/test_loop_invariance.py` pins both the extra trace and the remedy. `priors=None` on the
first tick keeps its extra trace (the T-maze, `infer_states` 2), documented in R4.

**R2.** `tests/test_loop_invariance.py`: eight models (T-maze, learnable T-maze, scalar
linear-Gaussian, payoff with the mixed planner, `small_lg` with the gradient planner, the
sophisticated T-maze, the switching mixture, `tiny_rssm`), each one trace per face over twenty
ticks from `initial_priors`, the same pytree structure, shapes, dtypes and weak types in and out
of `update_empirical_prior`, and of the parameters in and out of `learn`. The switching mixture
runs padded, as above.

**R3.** `replay` passes a nonzero Python-number `lr` to the compiled scan as a float32 array and
keeps a Python `0.0` static (`replay.py` `_traced_rate`): three Python-float rates, 3 compiles to
1. `examples/mountain_car.py` passes the episode's decaying rate as `jnp.float32(rate)`: the
episode program's cache holds 1 entry over a run where it held one per rate; the test runs the
example's `main` at a tiny size and counts. `infer_states` and `learn` refuse, naming the
modality, a linear-Gaussian observation that is a Python list or number, a scalar array, or has a
leading batch axis (before: `AttributeError`, `IndexError` or a broadcasting `ValueError`), and a
categorical observation that is a Python list (`system/validation.py` `_check_observations`).

**R4.** `docs/capabilities.md`, "Compiling an agent loop", under Plan; the test counts of its
footer (2,788 tests, 2,780 in the default selection).

**R5.** `SystemParams.precision`, an optional scalar array held float and strong-typed, a
`ValueError` for any other shape. The three planner-context constructors in `system/faces/plan.py`
read it through one helper, falling back to `System.gamma`; every planner's use of `ctx.gamma`
(enumeration's and the mixed planner's posterior, the sophisticated tree's soft value and
posterior, the cem posterior, the gradient planner's Laplace precision) is arithmetic and takes
the array. With the precision present, `infer_policies` and `act` equal those of
`System(gamma=value)` bitwise on the enumerating, sophisticated and gradient (Laplace draw)
planners; annealing it across ten values under `eqx.filter_jit` is one trace. The description
writes it as a clamped parameter variable `precision` (domain `positive`, shape `[]`) that
`configuration.planning.precision` names; the schema, `docs/description-format.md` and the
round trip are updated, and the description length does not price it. `merge` carries one
precision: components that all declare it with one value give the composite's, and a mix of
declared and undeclared, or different values, raises; `subsystem` views take the composite's.
`tests/test_scoring_slot.py` and `tests/test_learn.py`, which list the fields of `SystemParams`,
gain `precision` (scoring material).
