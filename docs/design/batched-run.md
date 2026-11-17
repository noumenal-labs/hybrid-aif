# A batch axis for `run`

*Opened 2026-11-17 (Alex: "I'm kinda shocked we were not vmapping over multiple seeds though.
That needs fixing right away"). Decisions R1–R14 below are the builder's proposal; those marked
*Alex to confirm* are open. Customer: the multi-rover survey (two rovers under one scheduler,
`terrain-field.md`) and every multi-seed gate the project runs.*

## What is wanted

`run` (`composition/runner.py`, design `couple-and-run.md` §4–6) is the simulation path for a
composed System: a closed System, an antichain of stepped nodes, a tick order, clocks. It is a
Python loop over rounds. Only the two compiled tick functions inside `plan()` and `sample()` are
traced (`_plan_tick`, `_sample_tick`); everything around them runs on the host: holds as Python
ints, realised levels cast to ints that key a cache of distinct Systems, write-backs decided by
object identity and concrete equality checks, per-round output validation, the record as lists
stacked at the end. Replicas of a closed System (seeds, starts, parameter draws) therefore run
one after another, and ten 800-tick survey runs cost ten times one.

The only vmapped simulation in the library is `rollout` (`rollout.py`), the pymdp-style scan of
an `Env` against a `policy_fn` with a sliding history window. It is a test-only utility: nothing
in `src/`, `examples/` or the notebooks imports it, and the project brief's "the pymdp-style
fused rollout remains the default for simulation" (brief line 298) and `design/system.md`'s
"through the existing `rollout` driver" are stale. The R-AIF driver (`reference/raif/run.py`)
vmaps seeds inside one jitted program by a hand-written collect loop and uses neither.

Two things are wanted, and one traced round body serves both:

- **Replicas.** `run(..., batch=B)`: `B` copies of the closed System, one key each, optionally
  per-replica carries and params leaves, stepped by one program, the `Trajectory` with a leading
  replica axis.
- **Groups.** `N` structurally identical leaves (the two rovers) stepped by one vmapped call
  within a round, their write-backs to a shared variable summed at the boundary: a vectorised
  implementation of simultaneous order among the copies.

## R1 — One traced round body; the Python loop goes

The round becomes a pure function `tick(state, key) -> (state, record)` (a round is one tick,
§4.5) under `lax.scan`, and `run` without `batch=` scans it unvectorised. There is one code path: the host
loop is not kept as a second one. What a host loop offered was custom steps with host-side
Python (the survey's `timed` wrapper, `np.asarray` on black-box outputs); under R8 those are
refused naming the step, and the survey's per-tick timing moves to its separate harness
(`survey_run.timing`), where it already lives for the rover's parts. *Alex to confirm:* one
path. The alternative, `run(..., host=True)` keeping today's loop, is a maintenance burden on
the engine's core loop for a wrapper the examples can do without.

The whole-run program compiles once per (System structure, steps, ticks); the per-node tick
functions were compiled anyway. Compile time is recorded in the gate beside tick cost (R13).

## R2 — A structure pass on the host, then the body

Everything that reads structure, not values, runs once before the scan, as today's refusals
do: the antichain check, the derived sequential order, the clock table (rounds per level of
every durations table, every level validated), which variables each learner writes, the
two-writers refusal for a non-Dirichlet variable, the static shape and dtype of every step's
outputs (R7). The body receives these as static structure and makes no decision from a Python
value.

## R3 — Fixed-structure carries

`MindCarry` fields start as `None` and fill on round 0, so rounds 0 and 1 compile different
programs and the carry cannot be a scan state. The carry gains fixed-structure fields from the
start (zeros of the belief's shape, an empty learn state, a zero action) and a `started` flag;
`_plan_tick`'s branch on `carry.previous is not None` becomes a `jnp.where` on the flag;
`learned` and `planned` are always returned, with fixed structure. The same for `_Sample`'s
point carry and for `_Held` (R5). The reset (the first call with `obs=None`, §4.4) stays a
distinct call, but a traced one, vmapped under `batch=`: `plan()` returns nothing, `sample()`
returns `sample_initial`, and which emissions are present entering round 0 is static (R7).

## R4 — A view sliced at a traced level

Between two integer levels of a driven discrete node a sliced view has identical static
structure and table shapes; only values differ, by a gather (`merge._cut_axes`,
`_drop_control_axis`, `_cut_gaussian`, `_contract_target`, all `jnp.take` at the level). The
slice is static today only because `_termination` casts the level to a Python int and `_Views`
caches a System per level. The build: one System per sliced view, built once in the structure
pass, whose gathers take the level as a traced index; the level-keyed cache goes. The unsliced
(averaged) view is a different structure and remains the reset view where `subsystem` admits
it, with the level-0 fallback under `MixtureRefusal` decided in the structure pass. A `plan()`
handed a sliced view keeps the `sliced` mark; nothing a step sees changes.

## R5 — Holds as arrays

`_Held.remaining` becomes an int32 array in the scan state; the duration-to-rounds table of
each clocked node is computed and validated on the host (R2) and indexed at the chosen level in
the body. Decide-or-hold is `lax.cond` with both branches returning the same pytree: on a held
round the step's outputs are its standing values (every out-port keeps its value, §5), no
write-back, no realisation; `Trajectory.decided` is an array. Under `batch=` replicas hold
independently, which is the point of the array form.

*The one padding-like cost.* No shape is padded anywhere in this design: every round's outputs
have the shapes of the structure pass, and absence is a static mask. But `vmap` lowers a
`lax.cond` whose predicate differs across replicas to a `select`, so both branches run: a
replica on a held round computes the step it then discards. For a clocked node this is the
slow tier's step once per round instead of once per option; on the survey the scheduler's step
is cheap beside the rover's and its holds are in lockstep across replicas with equal durations,
so the waste is nil there, and the structure pass makes it so by construction: a clocked node
whose hold schedule does not depend on the realised level (every level of its durations table
equal, the common case) has a schedule that is a function of the round alone, and its
decide-or-hold is a `cond` on an unbatched predicate, which `vmap` leaves a `cond`. Only a
node whose durations differ across levels has a per-replica predicate and pays the `select`.
Hoisting that case out of the batch too (grouping replicas on their hold state, or a
compaction of the deciding replicas) is a deferral, not a ruling: it is not built in this
round and is the first follow-on if a System with level-dependent durations and an expensive
slow tier appears.

## R6 — Write-backs traced

Every step returns a full params pytree of the view (today: the object handed in when it did
not learn). The body takes each learner's delta against the round-start home and applies §4.6:
one writer, `where(wrote, new, old)` per leaf, `wrote` a static fact of the structure pass; a
Dirichlet variable several nodes wrote, round-start plus the sum of deltas. The sequential and
async rule (a live write-back: the next view is taken from the home as it then stands) is the
same arithmetic applied after each node's turn in the derived order, which is a static order.
The two refusals that read values at a boundary, a non-positive Dirichlet sum and two custom
steps that both wrote a non-Dirichlet variable in one round, cannot raise inside a scan: the
first becomes a recorded fault (`Trajectory.faults`, one flag per round per variable, the run
continuing with the round-start value), the second is caught in the structure pass where the
writers are `plan()` steps and otherwise becomes the same recorded fault. *Alex to confirm:*
recorded faults rather than `checkify` (which would cost a host sync per round under `batch=`).

## R7 — Output validation once, by shape

`_check_out`'s per-round checks (each key an out-port, a level in range, a vector of the
port's width, every emission present) move to the structure pass over `jax.eval_shape` of each
step at the reset and at round 0: shape, dtype and presence are static facts. A level's range is
the step's own contract (`plan()` and `sample()` satisfy it by construction; a custom step that
does not gets a recorded fault, R6, not a refusal).

## R8 — Steps are traced functions

A step must be jittable: its outputs JAX arrays of fixed structure. `plan()`, `sample()` and the
survey's world step already are (`survey_world.world_step` is `jnp`-pure). A black box whose
step returns host values, or a wrapper that reads the clock, is refused at the structure pass
naming the step and the host-side call (`_box_value`'s `np.asarray` goes). `Trace` is built
inside the body as arrays; `_enumerated`'s Python cache becomes a structure-pass constant.

## R9 — Orders under `batch=`

Simultaneous and sequential orders are static orders and batch as they are. Async draws a
permutation of the sources per round from the run key alone, never from the state, so under the
scan every round's order is drawn by `jax.random.permutation` from the same fold (which changes
async's draws from the host generator's, a pinned break, R13), and the unbatched scan keeps
async from Phase A. Under `batch=` the order differs per replica, and a per-replica order is a
`lax.switch` over the distinct derived orders with a batched index, which `vmap` lowers to a
`select` over all of them: the cost is the number of distinct orders per round (two sources,
two orders, twice the step work; `k` sources up to `k!`). Built in Phase B with that cost
stated in the docstring and a refusal above a stated count of sources (eight orders, four
sources, is the proposed ceiling), since the end state has no other runner for async to live in.
*Alex to confirm* the ceiling.

## R10 — Keys

Replica `b` runs under `fold_in(key, b)`; inside, the derivation of §7 is unchanged. Pin: the
trajectory of replica `b` under `batch=B` equals, array for array, the unbatched run under
`fold_in(key, b)`. A group (R12) steps its copies with the keys their nodes would have had,
stacked, so a grouped run equals the ungrouped simultaneous one.

## R11 — The record

`Trajectory` keeps its keys and gains a leading replica axis under `batch=` on every array,
`decided` and `faults` included; carries are the stacked scan outputs, no longer a tuple of
objects; `present` is a static mask from the structure pass. `batch=None` gives no replica
axis and a record unchanged in form, so every reader of a record (`rover_record`,
`decision_record`, the tests' fixtures) keeps working; `batch=1` gives an axis of length one,
the array convention, so a reader written for a batch never special-cases one replica.

*Memory.* The record is the memory: every stacked output over `ticks × B`, and a
`plan(trace=True)` step's trace (the policy posterior and the tree) is the largest of them. Two
provisions. `record=` names which outputs are stacked (default: what today's `Trajectory`
holds; `trace` opt-in as now; a step's whole carry under `record_carries` as now), so a
ten-seed gate that reads coverage keeps beliefs and actions only. `chunk=` splits the scan into
a host loop over compiled chunks of that many rounds (a Python loop over one program, not a
nested scan), bounding live memory to a chunk and giving the host a place between chunks for
timing and progress; `chunk=None` is one scan. The scan over rounds sits above the planner's
own scans (the gradient planner's iterations over its horizon, the mixture propagation), one
level deeper than today; no further nesting is added, and the gate measures it (R13 pin 7).

*Working memory under `batch=` is linear in `B`.* `vmap` saves nothing: a replica's peak is its
most expensive step's working set, and `B` replicas hold `B` of them at once. The sophisticated
planner materialises its deepest level in one vmap already (`sophisticated-memory.md`: 576 MB
per replica at eight actions, three outcomes, horizon five; 3 GB with a second branched
modality), so ten such replicas are 6 GB or 30 GB. The scan over rounds adds nothing to this,
since a round's intermediates are freed before the next; only the record persists. Provision:
`batch=` is processed in `passes` of at most `replicas_per_pass` replicas, a `lax.map` over
sub-batches each vmapped (an outer sequential map, not a nested scan over rounds); the
structure pass sizes it from the planner's own estimator (`branching_tree_memory`, which
evaluates nothing) against a stated budget, and reports the choice. Replicas in different
passes are independent, so this changes no number.

## R12 — Groups: a tuple key in `steps`

`steps={("agent.rover1", "agent.rover2"): rover_plan}` steps the named nodes by one vmapped
call of the step over their stacked views, observations, carries and keys. Refused: nodes whose
views differ in static structure (the check `plan()` already makes to share a program), a group
mixing clocked and unclocked nodes, a group under sequential order that is not contiguous in
the derived order. Semantics: the copies are simultaneous among themselves (each reads the
round-start home, or under sequential the home as it stood when the group's turn began; the
copies' write-backs are summed per §4.6) and the group is one node to the order. The record
is unstacked to the nodes' own names. *Alex to confirm* the syntax.

## R13 — What to pin

1. **Parity of the refactor.** The traced body reproduces today's loop array for array on the
   existing fixtures (`run_parity.npz`, seven cases, 171 arrays; `clock_parity.npz`; the three
   B6 fixtures) and the survey regression pins, under simultaneous and sequential. Async is
   re-pinned (its permutation moves to a JAX draw, R9).
2. **Replica parity** (R10), and `batch=B` of one System with identical params and keys gives
   `B` identical trajectories.
3. **Group parity** (R12): the grouped survey agent equals the ungrouped one under simultaneous.
4. **Cost.** Unbatched tick cost on the survey rover within noise of today's 0.10 s; `batch=10`
   of the one-level survey within 3× one run per tick (not 10×); compile time of the whole-run
   program recorded.
5. **Holds and faults**: a replica set holding different options; a non-positive Dirichlet sum
   recorded, the run continuing.
6. `examples/demo.ipynb` re-executed (an adoption gate; `run` is public API).
7. **Scale.** The two-tier survey, 800 ticks, `batch=10`, traces recorded, on the gate machine:
   peak memory and wall time against ten runs today, under `chunk=None` and `chunk=100`; and the
   corridor arena (`compose_minds`) under async with `batch=4`. Bumps are expected here, and
   this is where they show; a design that only passes the toys has not passed.

## R14 — `rollout`

Demoted now to an `Env`-level utility in the docs (`project-brief.md` line 298, `design/system.md`
lines 183 and 345, `capabilities.md` 557–582 name `run` as the simulation path). Retired once
batched `run` lands: its parity tests (`test_rollout.py`, `test_keyed_faces.py` 376–378) move to
batched `run`, and a `History` window is a carry a step can keep. Subsumed, concretely: a
`rollout` of `as_env(params)` against `policy_fn` is `run` over `with_mirror(system, params)`
under `batch=B`, the policy a custom step carrying the window; a `rollout` of a bare `Env`
(`MountainCar`) is a `run` whose one leaf is the `black_box` of that `Env` under `batch=`, so
black boxes with `jnp` outputs must batch (R8). `Env`, `GenerativeEnv`,
`as_env` and `broadcast_params` stay: `sample()`, `black_box` and the examples depend on them.

## Out of scope, noted

Nested rounds (a schedule as a step over a non-leaf node, per-subsystem tick orders) need the
structure-pass/body split of R2 and an open round body fed its in-ports; they are the next
build, not this one. The multi-rover survey model is the demo build that follows.

## Build plan

- **Phase A, no behaviour change.** R2–R8 for the unbatched run: structure pass, fixed carries,
  traced levels, array holds, traced write-backs, shape-checked outputs, the scanned body. Gate:
  pin 1, pin 4's unbatched cost, pin 6.
- **Phase B.** `batch=` (R9–R11) and groups (R12). Gate: pins 2–5; ten survey seeds in one call.
- **Phase C.** Docs (R14), the capabilities entry, the roadmap row, a minor version.

## As built (Phase A, 2026-11-17)

*Built under the delegation of the opening note; no commit. Phase A only: no `batch=`, no
groups, no `record=` or `chunk=` (R11's arguments did not fall out for free and were left to
Phase B).*

**What was built.** `composition/runner.py`: `_run` is the structure pass, the reset, round 0 and
one `lax.scan` of the rounds after it, all through one round body, `_Engine.round(state, t)`;
the Python loop of rounds is gone. `_State` (key, values, carries, home, held options, the async
key) is the scan's state; the record is the scan's stacked outputs, assembled into the same
`Trajectory` tree as before (`received`, `present`, `emitted`, `intended`, `decided`,
`carries` as a tuple of per-round carries, unstacked on the host). `Trajectory` gains `faults` at
the root. `_Views`, `_hashable`, `_Held`, `_chosen_duration`, `_durations_node`, `_stack` and the
host generator `_sweep_generator` are deleted; `_sweep_key` replaces the last.

- *R2.* Everything structural runs before the reset, as before, plus the clock table:
  `_clock_of` reads each clocked node's durations off its reset view and validates the rounds of
  **every** level (`_option_rounds(..., level=k)`), where the loop validated the chosen level when
  it was chosen. A durations table with a bad entry at a level never chosen is now refused up
  front, with the same sentence naming the level.
- *R4.* `_termination` (merge.py) passes a traced integer level through as the gathers' index.
  `subsystem` / `joint_planner` are called every turn on the home as it stands; inside the scan
  that is a cost of the trace only, and one System serves every level, so no view is cached and
  no level is cast to `int` there. The reset view and its `MixtureRefusal` level-0 fallback are
  unchanged.
- *R5.* A clocked node's option is `(remaining: int32, intended action)` in the state;
  decide-or-hold is `lax.cond`, both branches returning the out-ports, the carry and the params
  (on hold: the standing values, the carry and the params handed in). A node whose levels all
  last `R` rounds decides on `t % R == 0` (the lockstep predicate, a function of the round
  alone); otherwise on `remaining == 0`. Realisation is masked by the decision flag, which keeps
  every other action node's draw (each draws from its own subkey).
- *R6.* Write-backs are arithmetic on values: `merge._write_back` (the body of `write_back`,
  which now delegates to it) takes a `flag`; `_own_slots`, `_check_parameters_only` and
  `_at_the_boundary` compare by identity first, then by value, and a failure on concrete values
  (the reset, round 0) raises the sentence it always raised, while one on traced values is
  recorded in `Trajectory.faults` under the variable (or `"<node>:params"`, `"<node>:<table>"`)
  and the variable keeps its round-start value. The boundary sum reproduces the loop's arithmetic
  exactly (a single writer takes its value; several sum their masked deltas in turn order).
- *R7.* Shapes and dtypes are checked as the step returns them, which inside the scan is once on
  the trace; a traced level out of range is a fault (`_check_ranges`), a concrete one refused.
- *R8.* A step that raises a JAX concretisation error on a traced value is refused naming it
  (`run: the step of 'world' calls host-side code on a traced value (...)`, with the remedy).
  `env_step` (black_box.py) is now traceable (its in-port level is `jnp`, its range checked on
  concrete values). `_box_value` checks dtypes without `np.asarray` of the value.
  `_Plan._enumerated` no longer caches a traced policy set.
- *R9.* Async draws `jax.random.permutation(fold_in(fold_in(key, N), t), sources)` per round and
  `lax.switch`es over the distinct derived orders of every permutation (computed in the
  structure pass); a concrete round picks its order in Python.

**Changed from the proposal, and why.**

1. *R3 is not built as fixed carries with a `started` flag; round 0 runs on concrete values
   instead, and is the scan's template.* A custom step's contract hands it `None` for a variable
   with no value yet, which is round 0 only, so round 0 is a different program whatever the
   carries do; and `plan()`'s carry is not the only one that takes its form on round 0 (a mixture
   belief grows from 2 components to 4 between rounds 0 and 1 in the tiered rover). So the round
   body runs on concrete values while the state's shapes still change (`_Unsettled`, at most
   `_SETTLE = 3` rounds after round 0, then refused with the sentence naming the leaf path that
   changed), and the scan starts from the first settled state. Cost: round 0's steps compile as
   they did before (`_plan_tick` per view, cached across runs), and the scan compiles once.
   Under Phase B's `batch=` the concrete rounds are vmapped calls, not a scan.
2. *R6's two value refusals are refusals on round 0 and faults after it*, since round 0 is
   concrete; the existing refusal tests (non-positive sum, two non-Dirichlet writers, learned
   sliced table, params changed outside the variables) all trip on round 0 and pass unchanged.
3. *A run in which no step wrote back returns the params object handed in* (`Trajectory.params is
   params`), as before: whether any turn wrote is a fact of the trace.
4. *Async is a switch over orders, not per-replica*; under `batch=` the R9 cost statement and
   ceiling stand. No ceiling is enforced unbatched (the switch runs one branch; the compile is
   per distinct order).

**Refusals added**, each a sentence naming the fix: a step that calls host-side code on a traced
value; a carry, out-port or params leaf that changes shape or structure between rounds after
the state has settled ("a value keeps its shape from round to round (each round is one step of a
scan)"); a carry whose non-array leaf changes ("hold a count or a flag in the carry as a JAX
array"); a durations level whose rounds are not whole, at any level.

**Re-pins and test changes.** `tests/data/run_parity_async.npz` (new): the two direct-wire async
cases of `test_run_parity.py`, 14 arrays of which 4 changed, recorded from `run`; the other 43
async arrays of `run_parity.npz` still match the previous drivers bitwise, as does every
simultaneous and sequential array, `clock_parity.npz` (async included) and the B6 fixtures. The
async seed patch moves from `_sweep_generator` to `_sweep_key`. Test steps that logged into
Python lists, counted calls, or indexed a tuple by a carried count now log through
`jax.debug.callback` (`tests/host_log.py`) or index a `jnp` array; no assertion and no tolerance
changed except: `test_plan_warm_start` compares a held plan by value where it compared identity;
`test_black_box`'s host-Python step test becomes the R8 refusal test (a one-round run still
takes it), and its dtype-normalisation test returns traced values from a counted carry.

**Re-pinned: the survey's two-tier regression pin** (`survey_run.npz`, `two-level`), which
failed one element at `rtol=1e-5` (a late step of one start's plan at tick 2, 5e-5). Ticks 0
and 1 are bitwise; at tick 2 the beliefs differ by one float32 ulp (relative 8e-8, the camera's
log evidence by 1 ulp of 9e4) and the three-step gradient plan amplifies it. The one-tier pin
passes. On the 200-tick one-level survey the old and new runners agree to 1 ulp at tick 1 (the
world step, run op by op before, is compiled inside the scan now) and the closed loop amplifies
it: actions differ by 2e-5 at tick 11 and by order 0.1 from tick 29. Nothing semantic differs;
the two-tier half of the pin was re-recorded on the owner's go-ahead (2026-11-17: 59 of its 725 arrays changed; the 177 one-tier arrays are bitwise as recorded); the page's long runs change when regenerated.

*The cause, pinned.* The world step's inputs at the first ticks of that run were recorded from
the previous runner and `survey_world`'s step called on them eagerly and under `jax.jit`: the
two differ on the same inputs, by 1 ulp in the state `x` and up to 23 ulp in the image at tick 1
(1 ulp in `fix` and `gauge` at tick 0). The world step ran op by op under the loop and runs
fused inside the scan now; the divergence is XLA fusion, reproduced without `run`.

**Cost** (pin 4, unbatched; the one-level survey, 200 ticks, sequential, `record_carries=True`,
this machine; two runs each):

| | before (host loop) | after (scan) |
|---|---|---|
| tick | 0.130 / 0.131 s (median of the `plan()` call, `timed`) | 0.110 / 0.113 s (scan execution / 199 rounds, world step included) |
| whole `run` call | 49.7 / 45.5 s | 42.1 / 38.2 s |
| compile | ~7.4–8.0 s (the first `plan()` call) | scan program: 3.1–3.2 s trace + 4.7–4.8 s XLA compile, beside round 0's own `plan()` compile |

`run` compiles its scan ahead of time and records the host's wall time of its parts at the root
of the record, `Trajectory.timing` (`compile`, `scan`, the `scanned` and `concrete` rounds).
The survey's tick time is the scan's execution over its rounds, compile excluded
(`survey_run.run_seconds`): 0.112 / 0.115 s on the run above (scan 22.3 / 22.9 s over 199
rounds; compile, trace included, 8.0 / 8.5 s), where the run timed whole gave 0.214 / 0.196 s.
`timed` is gone and the rover's parts are still timed apart by `timing`.

## As built (Phase B, 2026-11-17)

*Built under the delegation of the opening note; no commit. R9 (async under `batch=`), R10, R11
and R12, and pins 2–5 and 7 of R13. Docs, the capabilities entry and the rollout retirement
(R14, Phase C) are not touched.*

**R3, settled: the concrete rounds are vmapped as they are (option a).** Under `batch=` the
reset and the rounds run on concrete values before the scan (round 0, and up to `_SETTLE = 3`
more while the state's shapes change) are `eqx.filter_vmap` calls of the same round body over
the pass's replicas; the scan is `jax.jit(jax.vmap(lax.scan(body)))`. Structure-change
detection is the unbatched one: lowering the vmapped scan traces the body on one replica's
abstract shapes, and `_Unsettled` sends one more vmapped concrete round, as unbatched. Fixed
carries (option b) were not built, for Phase A's reasons (round 0 hands a custom step `None`
for a variable with no value; a mixture belief grows after round 0), and because (a) proved
clean: three adjustments, all local. (1) Round 0 always decides: under vmap a clocked node's
count is traced, so `t == 0` is read off the Python round index. (2) The async switch passes the
round index by closure, not as an operand, so round 0 stays a Python int inside every branch.
(3) An async round 0 under `batch=` dispatches by a traced switch whose branches can read a
variable before or after it has a value (`None` in one order, a value in another): `_alike`
gives every branch the record of one structure, absent reads as zeros with a per-replica
presence flag (`present`), every fault label in every branch. The carry's dtype normalisation
keys on a `scanned` flag of the round rather than on a traced key, since every key is traced
under `batch=`. The concrete rounds are not jitted: each is a different program and a jitted
round of the survey costs the scan's ~8 s compile; eager vmap compiles only the steps' own
jitted ticks under vmap (round 0 of the one-level survey: 10.2 s unbatched, 16.7 s at
`batch=10`, compiles included).

**What was built.** `run(..., *, batch=None, record=None, chunk=None, replicas_per_pass=None,
per_replica=())` (keyword-only; `run`'s and the module's docstrings state it).

- *Keys (R10).* Replica `b` runs under `fold_in(key, b)`; a stack of `B` keys is used as
  given (so identical keys give identical replicas, pin 2's second half).
- *The record (R11).* Every array leads with the replica axis (`decided`, `present`, `faults`,
  `start`, the final `carry` and a written-back `params` included); `carries` stays a tuple over
  rounds whose carries' arrays lead with it (round 0's carry and later ones differ in
  structure, so a single stacked array is impossible). `batch=None` is unchanged in form,
  `batch=1` has an axis of one. A run in which no step wrote back returns the params handed in.
- `record=` names the stacked fields from `received` (with `present`), `emitted`, `intended`,
  `carries`; default all but `carries`, which `record_carries=True` adds; `decided` and `faults`
  always; an unrecorded field is `None` at every node, and the scan does not stack it.
- `chunk=`: a host loop over one compiled program of `chunk` rounds; a last chunk shorter than
  `chunk` runs padded rounds under a mask (`where(t < ticks, new, old)` on the state) whose
  records are dropped, so there is one program, not two. `Trajectory.timing`: `compile` (once),
  `scan` (all chunks and passes), `scanned`, `concrete`, `concrete_seconds` (resets and
  concrete rounds, step compiles included), `chunks` (each chunk's `pass`, `start`, `rounds`,
  `seconds`), `passes`, `replicas_per_pass`. `chunk=` works unbatched too.
- *Passes.* A host loop over passes of `replicas_per_pass` (my call over `lax.map`: it composes
  with the concrete rounds and the chunk loop, and times each pass). The last pass is filled with
  copies of the last replica (dropped), so one compiled program serves every pass. No sizing
  from `branching_tree_memory` was built (optional).
- *Async (R9).* Per-replica permutations of the sources (`state.sweep` is per replica) and the
  Phase A switch over distinct derived orders, lowered to a select under vmap. Refused under
  `batch=` above four source units (a group is one unit).
- *Groups (R12).* A tuple key of `steps`. The members' views are one view up to their names, so
  one trace cannot take both as they are: each member's keys, observations, carry, view params
  and option are renamed to the first member's names (`_swapper`: the swap of the two prefixes in
  every dict key and string, the parameter variables both views share left as they are), stacked,
  stepped by one `eqx.filter_vmap` of the turn (step, checks, decide-or-hold), unstacked and
  renamed back. The structure check is the one `plan()` makes to share a program, applied up to
  the names: the renamed first view's System treedef (all its fields are static) equals the
  member's, and the params' structure and shapes match. The members read the round-start home
  (sequential/async: the home as the group's turn began); their write-backs are combined by
  `_at_the_boundary` (Dirichlet deltas summed) at the round's boundary or at the group's turn's
  end. Two `plan()` learners of a non-counts variable in one group are refused under every order.
- *Parallel replicas (added).* Where several local devices exist and divide the pass (a CPU host
  given `XLA_FLAGS=--xla_force_host_platform_device_count=N`, or several accelerators) the scan's
  state is sharded over them on the replica axis (`_spread`); with one device nothing changes.
  This was added because of pin 4, below.

**Changed from the proposal, and why.**

1. *Per-replica inputs are named (`per_replica=`), not detected.* Telling a leaf with a replica
   axis from a shared one needs its unbatched shape, and none is at hand: every table of the
   three-level pusher pair leads with 3, so at `batch=3` shared params look batched. `per_replica`
   names `"params"` (every array with the axis: `broadcast_params`, then the differing leaves
   replaced) and nodes whose `carries` seed has it; everything else is shared. Per-replica
   durations or actuator kernels are refused (they are the structure pass's).
2. *Round 0's value refusals are recorded faults under `batch=`*, since round 0 is traced there
   (unbatched it still refuses). Pin 5 shows a replica faulting in round 0.
3. *Async replica parity is to float32 rounding, not bitwise*; simultaneous and sequential are
   bitwise on the fixtures. The vmapped switch is a select over all branches and XLA fuses the
   batched program differently (largest deviation on the pusher pair 4.8e-7 in a belief logit).
4. *Pin 4 does not hold on one CPU device* (numbers below): the survey tick is compute-bound,
   so vmap amortises nothing; it holds once the replicas are spread over host devices.
5. *A group under simultaneous must be contiguous among the driven nodes too*, not only under
   sequential (a driven group steps at one turn); a group may not mix driven and undriven nodes.

**Refusals added**, each a sentence naming the fix: `batch`, `chunk`, `replicas_per_pass` not
positive integers; `replicas_per_pass`/`per_replica` without `batch`; `per_replica` naming an
input without the leading axis ("give every array of it a leading replica axis
(envs.generative.broadcast_params(tree, batch) for the shared ones, then replace those that
differ), or drop it from per_replica to share it") or naming nothing; per-replica durations or
actuator kernels; a key that is neither one key nor a stack of `B`; `record` naming no field;
async above the ceiling ("this System has 5 sources, up to 120 orders, above the ceiling of 4
sources (24 orders): run it under "sequential" or "simultaneous", group identical sources into
one node (steps={(a, b): step}), or run the replicas one call each without batch="); a group key
that is not a tuple of two or more distinct names; a node named twice; a group's views differing
beyond their names; a group mixing driven and undriven, or clocked and unclocked, nodes, or
clocked on different durations; a group not contiguous in the sequential, driven or an async
order (naming the nodes between); a group whose members are handed values of different
structure in a round; under `batch=`, a clocked node with level-dependent durations whose carry
still changes form while replicas disagree on deciding; and the R8 host-code refusal now also
covers the reset under `batch=` (every call is traced).

**Tests.** `tests/test_batched_run.py` (20 tests); `test_run.py`'s signature pin gains the five
keywords. No other test changed.

**Measurements** (this machine, CPU, 16 cores, 128 GB; one configuration per process).

- *Pin 2.* Pusher pair (`wire()`), 6 ticks, `batch=3`: every replica bitwise its unbatched run
  under simultaneous and sequential (carries included); async to ≤4.8e-7. Clocked robots
  (durations 4/2/4), 20 ticks, `batch=4`: bitwise under simultaneous and async. One-level
  survey, 6 ticks, `batch=2`: max deviation 8.3e-7 (3 of 10 record leaves bitwise); the closed
  loop amplifies this over long runs as Phase A's re-pin describes. Identical keys: identical
  replicas.
- *Pin 3.* Shared-Dirichlet leaves with `plan(learn=True)`, grouped vs ungrouped simultaneous:
  bitwise (record, carries, learned home), unbatched and at `batch=2`; the pusher pair grouped:
  bitwise. Two survey rovers sharing the field: equal to `rtol=1e-4` (test).
- *Pin 4.* One-level survey, 200 ticks, sequential, `record_carries`: unbatched 0.112 s/tick
  (compile 7.8 s, round 0 10.2 s); `batch=10` on one device 0.991 s/tick (8.9×; compile 9.9 s,
  concrete 17.7 s), linear in B (B=1,2,4: 0.116, 0.211, 0.420). With
  `--xla_force_host_platform_device_count=10`: unbatched 0.115, `batch=10` 0.195 s/tick
  (1.7×, within 3×; compile 12.4 s, concrete 16.7 s; wall 71 s against ~45 s for one run).
  Peak RSS 1.32 GB unbatched, 1.75 GB (`batch=10`, one device), 1.89 GB (ten devices).
- *Pin 5.* Holds differ per replica and match the unbatched runs bitwise; the non-positive sum
  is a per-replica fault (`[[F,T,T,T],[F,F,F,T],[T,T,T,T]]`), the run continuing.
- *Pin 7.* Two-tier survey, 800 ticks, `batch=10`, traces recorded. One device: 1.00 s/tick
  for the ten replicas, 842 s wall, peak RSS 4.1 GB (`chunk=100`: the same speed in eight even
  chunks, 4.5 GB, no second compile; chunking frees nothing on a CPU host, where the chunks'
  records are kept and joined). Ten host devices: 0.21 s/tick, 214 s wall, 4.4 GB (`chunk=100`:
  212 s, 4.7 GB). Ten sequential runs: 1099 s, 0.125 s/tick, 2.7 GB. So 5.1× over sequential on
  ten devices, 1.3× on one. The corridor arena under async with `batch=4` equals four
  sequential runs exactly (3.6 s against 3.9 s); the two survey rovers grouped are bitwise equal
  to the ungrouped run on all 37 leaves. The device-scaling section below has the curve.

## Device scaling on the CPU host (measured 2026-11-17, after Phase B)

One-level survey, 200 ticks, replica-ticks per second of scan execution (unbatched: 8.1/s,
one replica keeping about 1.7 cores busy). Independent processes, one replica each, give the
machine's ceiling for this workload (an M3 Max, 16 cores):

| configuration | replica-ticks/s | against one replica |
|---|---|---|
| 8 processes | 57.9 | 7.2× |
| 16 processes | 89.0 | 11.0× |
| `batch=8` sharded over 8 host devices | 42.8 | 5.3× |
| `batch=16` sharded over 16 host devices | 52.1 | 6.5× |
| `batch=16` sharded over 8 host devices | 56.1 | 7.0× |
| `batch=10` with 16 host devices (16 does not divide 10: one device) | 10.2 | 1.3× |

Two-tier survey, 800 ticks, `batch=10` on 10 devices: 214 s against 1099 s for ten sequential
runs (5.1×; one device 842 s).

**Reading.** The CPU's ceiling for independent replicas is about 11× at sixteen cores, and the
sharded scan reaches about 60% of it at every device count; the process-level CPU utilisation of
the sharded run is 6.5 cores at sixteen devices, so the gap is idle cores, not contention: the
per-round dispatch and synchronisation of one program whose state is sharded over host devices.
Nothing crosses replicas inside a round, so the sharded form pays a synchronisation it does not
need. Two Phase C items follow:

- **Independent per-device scans.** Execute the replica axis as a `shard_map` over the devices
  with the whole scan inside (each device runs its share's scan to completion, no per-round
  rendezvous), in place of sharding the state of one scan. Expected to approach the process
  ceiling. Two replicas per device beat one (7.0× against 6.5×), so the per-device share need not
  be one.
- **The divisibility trap.** `_spread` shards only when the device count divides the pass;
  otherwise it silently runs on one device (`batch=10` on 16 devices: 1.3×). Use the largest
  device count that divides the pass, or pad the pass to a multiple of the device count and drop
  the padding, and report the device count in `Trajectory.timing`.
