# Recompilation under the intended agent loop

Status: audit 2026-10-23; R1–R5 SETTLED (Alex, 2026-10-23); R6–R7 BUILT 2026-11-04 (see the
build notes at the end).

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

## Built

**R6: a library-owned compiled step** (`System.compiled()`): a `jit` boundary the library owns,
with inputs canonicalised in front of it, so the input-type traps above cannot occur. The
composition driver's internal one-tick function already follows this pattern. Status: built
2026-11-04.

**R7: a retrace watcher** (`hybrid_aif.debug.watch`): warns on a repeat trace with the same
static configuration, naming the argument whose structure, dtype or shape changed, or the System
field that changed, which `jax_explain_cache_misses` does not do under `eqx.filter_jit`. Status:
built 2026-11-04; on by default in the compiled object, opt-in for any other `eqx.filter_jit`
function.

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

## Build notes (2026-11-04)

**R6.** `System.compiled(watch=True)` returns a `CompiledSystem` (`system/compiled.py`; the name
follows the object's job, a System with its faces compiled, where `stepper` would suggest one
fused tick, which it is not: the faces stay separate, as the loops that call them are). It exposes
`infer_states`, `infer_policies`, `act`, `update_empirical_prior` and `learn` with the eager
signatures and return structures, each an `eqx.filter_jit` boundary with the System, the planner
callable and the non-array arguments (`metrics`, `detail`, `selection`, `mode`, `use_bound`,
`support_floor`) static and only arrays traced; any other attribute (`names`, `initial_priors`,
`as_env`, ...) is the System's. The eager faces are unchanged. Canonicalisation, ahead of the
boundary (the module docstring is the precise statement):

- parameters, beliefs, priors and learn carries: NumPy leaves become JAX arrays and weak types
  become strong, each leaf keeping its dtype (so NumPy, committed and weak-typed parameters share
  one program).
- observations, `previous`, `externals`, `action`: strong arrays; an integer takes the default
  integer dtype, a float keeps its dtype unless narrower than float32, a Python float takes the
  default float dtype (so x64 keeps its precision). An observation of the wrong form is refused
  eagerly by the eager faces' own check (`validation._check_observations`), naming the modality.
- `lr`, `decay`, `alpha` and the per-slot rates: Python numbers become arrays of the default
  float dtype, except a Python `lr` of `0.0` and a Python `decay` of `1.0`, which `learn` reads
  at trace time (a slot skipped, no decay and so no anchor) and which stay static, as `replay`
  keeps its `0.0`. *Note, 2026-11-27.* Under JAX's 64-bit mode a Python rate is a weakly typed
  array instead, so it takes the dtype of each table it scales and widens none (a number and an
  array rate are then two programs); with that mode off it is the float32 array it was.
- `legal`: a list is read as the eager faces read it, each mask becomes a bool array (an integer
  mask as bool; a float mask is refused with a sentence, where the eager faces fail with a NumPy
  error), and a concrete mask the eager faces refuse (no legal level at some step, a bare mask of
  the wrong length) is refused before the boundary with the eager sentence, as
  `composition/driver.py` does, because under the trace the concrete refusals cannot run. That
  pre-check runs when every control node is discrete and, for a bare mask, the planner is the
  System's enumeration; under a planner with a policy set of its own a bare mask is not checked
  before the boundary. `present` flags become bool arrays.
- PRNG keys: typed keys become raw `uint32` key data.
- `priors=None` becomes `initial_priors(params)`, completed to the key set
  `update_empirical_prior` returns: each discrete control node's entry (uniform) and each observed
  action node's entry (the one-hot at `previous[name]` when `previous` names it, else uniform),
  which `initial_priors` omits. Without the completion a model with a modality on an action node,
  or a prior missing a control node's entry, traced twice, because the first tick's pytree had
  fewer keys than the later ones. A mixture prior declared below `mixture_k` still differs from
  a pushed one, as it does eagerly, and the remedy is unchanged (`pad_mixture`); it is a separate
  cause from the missing keys.

The jitted face functions are module-level, so equal Systems share compiled programs: two
compiled objects of one System compile once, and an object's trace counts count the traces its
own calls caused (a counter bumped by the traced body, read before and after each call; a NumPy
against a JAX argument, a committed array, or another dispatch signature is a dispatch-cache entry
and not a trace). The watcher keeps the last traced signature per face and System by value in a
module table, so a real retrace through any compiled object is named against the last trace
anyone made of that face. `infer_policies` takes the eager face's trailing `return_policies`.

Measured (`tests/test_compiled_system.py`, twenty ticks, a counter on `System.infer_states`,
`infer_policies`, `act`, `update_empirical_prior` and `learn` that counts calls with a tracer
among the arguments): the T-maze, the payoff model under the mixed planner and `small_lg` under
the gradient planner each trace every face once; the learnable T-maze traces `learn` once over
nineteen calls with `lr` a Python float taking four values. Beliefs, policy posteriors, actions
and priors equal the eager faces' to `atol=2e-4` (the gradient planner's iterates differ in the
fifth digit between eager and compiled execution). A view from `subsystem` and its composite,
with a modality observing an action node, run from `priors=None` with one trace of each face.

**R7.** `hybrid_aif.debug` (`src/hybrid_aif/debug.py`): `RetraceWarning(RuntimeWarning)` and
`watch(fn)`, which wraps an `eqx.filter_jit` function and counts traces: a jaxpr trace announced
to `jax.monitoring` during the call (a cache-size read stands in where that is unavailable;
`Watched.traces`; the compiled object passes its own trace counter as `clock`). On the second and
later trace it warns, naming the first difference from the previous trace in this order: a static
field of an argument that is an `equinox.Module` (`static field 'gamma' of 'system' changed`, a
callable such as the planner by identity), an argument that was `None` and is not (or the
converse), the first path whose pytree structure differs, then the first leaf whose shape, dtype
or weak type differs (`observations['cue'] changed from int32[] to float32[3]`) or whose static
value differs; when none differs it says that the compilation cache was cleared or the program
evicted, or that a static argument or field changed in a way it does not name. The compiled
object watches every face by default (`compiled(watch=False)` counts without warning).

The eager nudge: a face (`infer_states`, `infer_policies`, `act`, `update_empirical_prior`,
`learn`) called eagerly more than ten times on one System object, counted per face and only for
the outermost call outside any trace, emits one `RetraceWarning` per System object, saying that
eager dispatch re-traces and runs operation by operation every call and pointing to
`System.compiled()`. (Measured 2026-11-04: a mixed-planner call took 40 ms compiled and 4 s
eagerly, flat in the number of gradient steps.) The count lives in a module-level table keyed by
object identity with a weak reference (the System's hash is by value, so equal Systems are counted
apart) and the System is not mutated. pytest does not turn warnings into errors in this
repository, so no configuration changed.
