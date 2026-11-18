# Design: nested rounds, a schedule as a step

*Opened 2026-11-17 (Fable, for Alex), after the batched run (`batched-run.md`, Phases A–C,
0.22.0). Rulings S1–S12 are proposed; the ones marked for Alex are open until he confirms.
Shelved 2026-11-18: groups (`batched-run.md` R12) already give the vmapped simultaneous
update of identical components, and the remaining motive is architectural, unobservable
unless inner members read each other; kept as a record, no build planned.*

## Why

`run` schedules a closed System with one tick order: every stepped node takes its turn under
`"simultaneous"`, `"sequential"` or `"async"`, chosen once for the whole System. Alex's picture
(2026-11-17): a System A with components B and C inside it, coupled to a System D, where A
updates B and C in parallel internally and D interacts with A sequentially without knowing A's
internal schedule, because A's Markov blanket hides it. The library has the blanket (C13, N1) and
the view of a non-leaf node as one stepped node, and lacks the schedule inside it.

What exists. The keys of `steps` are an antichain over the System's components; a non-leaf
node may be a key, stepped by `plan()` as one mind (the joint view, `joint_planner`) or by a
custom step on its subsystem view (`_take_view`: `subsystem(system, home, node.name)`). The
runner hands that step its in-port values once per round and takes back its out-ports, its carry
and its params; what the step does inside is invisible to the other nodes. So A as one node of D's
round exists. What does not exist is a step that is itself a round over B and C with its own
order, its own groups and its own clocks: the inner engine (`_Engine`, `_structure`) requires a
closed System, and A's view is open, fed by D through identifications that live in the outer
System.

For the multi-rover survey this is not needed in step one (the rovers read only the shared field,
so "rovers simultaneous inside a sequential outer round" changes nothing there); it is justified
by the architecture, and the demo's second step is its natural exhibit (two rovers nested
simultaneous, the scheduler and the world sequential around them).

## The rulings

**S1. A nested node is one node of the outer round.** Its blanket is the outer runner's whole
view of it: one turn per outer round, at the place the outer order gives it; it reads its in-ports
as any stepped node does (the standing values or the boundary's, by the outer order's rule);
its out-ports are realised once per outer round, at its turn; its write-back flows up through the
outer runner's ordinary write-back check (live under the outer `"sequential"` and `"async"`, at
the outer boundary under `"simultaneous"`, counts summed with other writers of the same variable
by the simultaneity rule). Nothing outside the node can tell what order it runs inside.

**S2. One inner round per outer round.** *(Alex to confirm.)* The nested node's turn runs one
round of its inner engine: every inner node takes its turn under the inner order, inner clocks
count outer rounds. An inner round is the outer round seen from inside, so the inner and outer
clocks are the same clock, and "different timescales" stay what they are today: durations and
holds. A sub-cycle (`k` inner rounds per outer round, the in-ports constant across them and the
out-ports those of the last) is named as a deferral, not built, because its semantics for the
inner reads of the outer in-ports (constant, or stale by a fraction of a round) are not settled.

**S3. The inner order is declared per nested node.** `rounds(steps, *, tick_order=
"simultaneous", clocks=None)` returns a step for a non-leaf node whose `steps` is an antichain
over the node's components that covers every leaf beneath it (groups as tuple keys, `plan()`,
`sample()`, custom steps, and nested `rounds()` again: the construction is recursive, and the
structure pass runs once per level). Its `tick_order` is one of the three; its `clocks` scale the
inner nodes' durations as `run`'s do. The outer `steps` lists the nested node by its name with this
step as its value, and the antichain rule of the outer runner is unchanged: the outer `steps`
names A, not B or C.

**S4. The inner engine runs an open view with supplied variables.** This is the refactor.
`_structure` and `_Engine` take a set of *supplied* slots: the modalities of the view that no
component inside it feeds, and the control nodes inside it that a variable outside drives. Today
both are refused ("the System is open"); supplied, each is a variable of the inner state whose
value is set from the step's `obs` at the start of every inner round and never realised inside.
Every inner read of a supplied variable is a boundary read (the value is the outer round's at the
node's turn, which is what the outer order already determined). The top-level `run` supplies
nothing, so its closedness check is unchanged in effect: open slots minus supplied slots must be
empty, with the same sentences.

**S5. Keys.** The nested node's step key is split as the top-level round key is (`_round_keys`
over the inner nodes: step keys, realisation keys, the sweep key for an inner `"async"` draw), so
an inner node's draws are a function of the outer key, the node's place and the round, as a
top-level node's are. No inner draw is a function of the outer order.

**S6. Realisation happens inside.** *(Alex to confirm.)* The inner engine owns the kernels of the
action nodes of its nodes and realises them as the top-level engine does (`_actuate` per turn).
An action node that crosses the nested node's blanket (an out-port of A identified with a slot of
D) is realised inside once and handed up as a value: the outer runner marks a nested node's
action nodes as realised by the step and does not draw them again. The outer structure pass
knows a nested node by its step's type, as it knows `plan()` and `sample()`. The alternative, the
inner engine returning intended levels and the outer realising them, would draw an action node
that an inner node also reads twice, once inside and once outside, with two values; so it is not
taken.

**S7. The inner record is lifted out of the carry.** The `Step` protocol stays `(out, carry,
params)`. The nested step's carry is the inner state less the home (the inner carries, the held
options, the presence flags, the sweep key) plus the inner round's record (`received`,
`present`, `intended`, `decided`, `emitted`, `carries`, `faults` per inner node). The outer
runner, knowing the node is nested, moves that record out of the carry into the `Trajectory`
tree under the node's path, where `Trajectory.__getitem__(path)` already descends, so
`rec["agent.rover1"]` of a nested rover reads as it does for a flat one; the carry that
`record_carries` stacks holds no record. `Trajectory.faults` holds the inner faults under the
inner node's path. `Trajectory.decided` for the nested node itself is the outer hold (S9).

**S8. Write-backs inside and up.** The inner home is the nested node's view params as the outer
runner handed them (taken from the outer home by the outer order's rule). Inner write-backs
apply as the inner order says. The step returns the inner home after the round; the outer
runner's write-back check lifts every parameter variable that changed into the outer home, as it
does for any step that learned. Two nested nodes that both write one variable in one outer
round are governed by the outer rule (counts summed; a form that is not counts refused before
the first round). Inside one nested node, two inner writers are governed by the inner rule.
The sibling-belief read (N3) across the nested node's blanket is a modality of D fed by an
emission of A's component, realised inside (S6), which is exactly today's `Link`.

**S9. Clocks at both levels.** A nested node may itself declare durations on a control node it
chooses (its view's `params.durations`): then the outer runner holds the whole nested node for
the option as it holds any clocked node, no inner node steps on a held outer round, and the inner
counts do not advance. Inner clocks are the inner engine's own and count the rounds the nested
node stepped. The group rules (`batched-run.md` R12) apply inside unchanged.

**S10. Under `batch=` and in the scan, nothing new.** The nested step is a traced function of
arrays: one round of an engine whose round is already the traced `tick`. The outer scan traces
it inline, replicas vmap over it, a group of nested nodes vmaps over it. The settle from the
reset (Phase C) applies inside: an inner variable the inner reset leaves without a value holds
zeros and a flag, and a supplied variable that the outer round hands as a `Maybe` is handed on as
one. The inner reset is the nested step's reset call (`obs is None`): each inner node's reset in
inner order with the inner reset keys, the state it leaves being the nested node's reset carry
(settled by the inner engine, with the same `started`-style gates). A nested node refused inside
(a host-side step, a carry of changing form) is refused naming the inner node's full path.

**S11. Costs stated.** An inner `"async"` order under `batch=` is a switch over the inner derived
orders, with the same ceiling of four sources per nested node (R9). A nested node costs one more
trace of its round at settle (or none once the static settle covers it). No cached view is
shared between levels: each level takes its views from its own home every turn (R4).

**S12. What is refused, each with a sentence naming the fix.** A `rounds()` step given to a leaf
(nothing to schedule beneath it); an inner `steps` that is not an antichain covering the node's
leaves; an inner key naming a component outside the node; a supplied slot that an inner node
also feeds (two sources); a nested node whose inner `steps` names a node with `sample()` beneath
a `plan()` key (the joint-view rule, as today); an outer `clocks` entry for an inner node (clocks
are declared at the level that steps the node).

## The build, in two phases

**N1. The open engine (no behaviour change).** `_structure(system, params, steps, supplied=())`
and `_Engine(..., supplied)`: supplied slots become variables of the state, set from `obs` each
round; the closedness check is "open minus supplied is empty". The top-level `run` passes nothing.
Pins: every parity fixture (`run_parity.npz`, `run_parity_async.npz`, `clock_parity.npz`, the
B6 fixtures, `test_batched_run.py`'s bitwise pins, the survey pins at their tolerance) unchanged,
not re-recorded; the refusal sentences for an open System unchanged.

**N2. `rounds()`.** The step constructor, the reset, the record lifted (S7), realisation inside
(S6), the write-back up (S8), the refusals (S12), the docs (`run`'s docstring, the
composing-systems guide's `run` section, the capabilities entry), the version. Pins:

1. *The blanket, testable.* Two inner nodes B and C that do not read each other, nested in A
   under `"simultaneous"` and again under `"sequential"`, coupled to D under an outer
   `"sequential"` order: D's record is bitwise the same for both inner orders.
2. *The flat equivalence.* A nested A over B and C under inner `"sequential"`, in an outer
   `"sequential"` order [A, D], equals the flat `run` under `"sequential"` [B, C, D] on every
   belief, action and emission to float32 rounding, with the keys aligned by construction (S5
   gives inner node j the key a flat node at the same place would get) — bitwise where the key
   derivation is identical, and the record says which.
3. *Mixed orders by hand.* Two tabular counters B and C that read each other, nested under
   `"simultaneous"` inside an outer `"sequential"` order with a third counter D that reads A:
   the record equals the values computed by hand for three rounds (B and C see each other's
   boundary values, D sees A's values of the same round).
4. *Clocks at both levels.* A nested node with durations held for its option while an inner
   clocked node has its own clock: the inner count does not advance on a held outer round.
5. *Batch.* The nested run under `batch=4` equals four unbatched nested runs replica for replica
   (bitwise under `"simultaneous"` and `"sequential"`, as R10 pins the flat run).
6. *The exhibit.* The two-rover survey (demo step two): rovers grouped and nested under
   `"simultaneous"` inside a `"sequential"` outer order with the scheduler and the world; equal to
   the flat grouped run on the beliefs and coverage to float32 rounding (the rovers read only the
   shared field), and the page says which order ran.

## Deferred, named

- The sub-cycle (`k` inner rounds per outer round; S2).
- A nested node run on another host (the deployment cut of N6's reading: the inner engine as a
  process of its own with the supplied variables as its wire; the design of the wire is the
  count-report of N6 for parameters and the action-or-sensation invariant of N1 for the rest).
- An inner order that changes at run time (an outer node choosing the inner order as an action):
  a static fact of the structure pass today, by R2.
