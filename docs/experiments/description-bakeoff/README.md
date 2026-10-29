# Description bake-off: 0.2 against 0.3, agents writing from prose (2026-10-13)

**Verdict.** Description 0.3 is easier for an agent to write than 0.2 and no harder to read.
Across four models and two agent types, the 0.3 role spelling built 7 cells of 8 against 0.2's
4, and no 0.3 cell was refused for a layout rule; the reading test scored 5.50 (0.2), 5.75
(0.3 role) and 6.00 (0.3 generic) out of six. The run changed the reader three times and the
format document twice, and left the default `describe` spelling role (Alex, 2026-10-13). The
rulings of `docs/design/model-description-0.3.md` stand; this folder is the record.

**What it is.** The acceptance test of the 0.3 reader (`model-description-0.3.md`, Q17,
"Acceptance bake-off"). It is a test of the reader and of the format document, and the
refusal sentences are under test as much as the agents are. It is not a referendum on the
format's rulings, which were adjudicated before it ran.

## Protocol

| arm | reference the agent got | spelling |
|---|---|---|
| 0.2 | the composing-systems excerpt of the last pre-0.3 commit (5ee7dad: shape rules, family table, bindings, document sections) and `schemas/description-0.2.json` from the same commit | the only one |
| 0.3 role | `docs/description-format.md` as committed with this record | role labels (`initial`, `transition`, `likelihood`, ...) |
| 0.3 generic | the same document | generic labels, every parent bracketed, `steps` stated |

Four models × three arms × two agent types (Sonnet, Haiku) = 24 writing cells, one run per
cell, fresh context every run, no tool but a file write and nothing to read outside the two
texts. A refused document came back to the same agent with the refusal sentence verbatim and
nothing else, at most three correction rounds. The record per cell is the round that built, or
"did not build" in four writes. The checker agents reported the reader's verdicts during the
run; every document was then rebuilt locally with `build`, and the tables here are the local
rebuild's (the two agree).

**Writing prompt.** "You are given a model in prose and a format document. Write the model as a
JSON document in that format, exactly, with every number the prose gives. Use the {role |
generic} spelling. Write only the document, as a single JSON file at {path}. Do not write any
other file, do not run any code, and do not read anything outside the two texts below." Then
the prose, then the format reference.

**Correction prompt.** "Your document was checked by the library's reader and refused. The
refusal, verbatim: {sentence}. Write the document again at {path}, fixing what the refusal
names and nothing else."

**Reading test.** Twelve fresh agents (two per arm and agent type) each got one document of M3
in its arm's format, written by us (`reading/`), and answered six questions in one line each:
which variable the rat chooses; what the cue depends on; which parameters are learned and from
what prior; where the slip in movement is written; what the end-of-horizon preference depends
on; how many steps ahead the rat plans. Key: `u_location`; `location` and `context`; `A_cue`
under a Dirichlet prior of concentration one; the actuator record `K_a_location` (0.3) or the
location transition table (0.2, which has no actuator record); `context`; three.

## The four models, as given to the agents

**M1. The T-maze.** A rat in a T-maze. Its location is one of four places: centre, left arm,
right arm, cue arm. The reward's side is a hidden context, left or right, fixed for the
episode. The rat chooses where to go next (four options matching the places); from any place
it reaches the chosen place with certainty, except that the arms are absorbing: once in the
left or right arm it stays there whatever it chooses. The context never changes. The rat sees
two things: its position, exactly; and a cue with three values, "left", "right" or "nothing":
in the cue arm the cue names the context's side with certainty, elsewhere it is "nothing". The
rat prefers a third observation, the outcome, with three values "reward", "no reward",
"neutral": in the arm matching the context it gets reward with probability 0.98, in the other
arm no reward with 0.98, elsewhere neutral; it values reward at +3 and no reward at −3 in log
preference, neutral at 0. It starts at the centre, with the context uniform. Horizon two.

**M2. A switching linear system.** A particle on a line with position and velocity (a
two-dimensional continuous state). It is in one of two regimes, calm or windy: calm keeps
velocity (position advances by velocity each step, velocity unchanged), windy adds a drift of
+0.5 to velocity each step. The regime is sticky: it stays with probability 0.9. The process
noise is Gaussian, standard deviation 0.1 on both coordinates in calm, 0.3 in windy. The regime
at the step being predicted is what chooses the dynamics (the regime is read at the same step
as the new state; the old state one step back). A sensor reads position with Gaussian noise of
standard deviation 0.2 and nothing else. A second sensor reports the regime correctly 80
percent of the time. The particle starts at the origin at rest with unit uncertainty, regime
uniform. No control. Horizon one.

**M3. A learned sensor and a slippery actuator.** As M1's maze, but: the cue's reliability is
unknown and learned, with a Dirichlet prior of concentration one on every entry of the cue's
table; and the rat's movement is unreliable, so that the place it ends up in is the chosen one
with probability 0.9 and each other place with probability 0.1/3 (the slip happens in the
world, after the choice). The outcome preference is replaced by a goal at the end of the
horizon: a preference over the final location that depends on the context, left arm when the
context is left, right arm when right, each with log preference +3 and every other place 0.
Horizon three.

**M4. A composite: a rat and its maze.** Two components, named `rat` and `maze`. The `maze` is
M1's world with no preferences, no cue, and an extra influence: a "door" with two states, open
or shut, which nobody chooses, and which, when shut, keeps the rat at the centre whatever it
chooses; the door is drawn open or shut with equal probability. The `rat` is M1's model as
written. The two are merged so that the rat's chosen move drives the maze's move, the rat's
position sensor reads the maze's location rather than its own, and the door is closed off with
its equal-probability prior. Write the merged model as one document.

## Rounds to build, per cell (– = did not build in four writes)

The run produced three reader fixes (below). "Before" is the reader the agents wrote against,
"after" rebuilds the same documents under the fixed reader; a later round written in answer to
a refusal that no longer fires is still counted at the first round that builds.
`verdicts-before-fixes.json` and `verdicts.json` carry both, one line per document.

| model | agent | 0.2 | 0.3 role | 0.3 generic |
|---|---|---|---|---|
| M1 | sonnet | 1 → 1 | 1 → 1 | 1 → 1 |
| M1 | haiku | 1 → 1 | 1 → 1 | – → – |
| M2 | sonnet | – → – | 2 → 2 | 2 → 2 |
| M2 | haiku | – → – | 2 → 2 | 2 → 2 |
| M3 | sonnet | 1 → 1 | 1 → 1 | 1 → 1 |
| M3 | haiku | – → – | 3 → 3 | – → – |
| M4 | sonnet | 3 → 3 | 3 → 1 | 2 → 1 |
| M4 | haiku | – → – | – → – | – → 4 |

| arm | built | first try | mean rounds (– = 4) | sonnet built / mean | haiku built / mean |
|---|---|---|---|---|---|
| 0.2 | 4/8 | 3/8 | 2.75 | 3/4, 2.25 | 1/4, 3.25 |
| 0.3 role (before → after) | 7/8 | 3/8 → 4/8 | 2.12 → 1.88 | 4/4, 1.75 → 1.25 | 3/4, 2.50 |
| 0.3 generic (before → after) | 5/8 → 6/8 | 2/8 → 3/8 | 2.50 → 2.38 | 4/4, 1.50 → 1.25 | 1/4 → 2/4, 3.50 |

## Reading test (M3, our own document per arm; six questions; twelve runs)

| arm | agent | run | q1 | q2 | q3 | q4 | q5 | q6 | total |
|---|---|---|---|---|---|---|---|---|---|
| 0.2 | sonnet | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| 0.2 | sonnet | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| 0.2 | haiku | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 5 |
| 0.2 | haiku | 2 | 1 | 1 | 1 | 1 | 0 | 1 | 5 |
| 0.3 role | sonnet | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| 0.3 role | sonnet | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| 0.3 role | haiku | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 5 |
| 0.3 role | haiku | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| 0.3 generic | sonnet | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| 0.3 generic | sonnet | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| 0.3 generic | haiku | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| 0.3 generic | haiku | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 6 |

Means out of six: 0.2 = 5.50 (sonnet 6.0, haiku 5.0); 0.3 role = 5.75 (sonnet 6.0, haiku
5.5); 0.3 generic = 6.00. The answers are in `reading/answers.json`.

The 0.2 misses were both Haiku runs naming the wrong parent of the terminal goal: in 0.2 the
parent sits in `structure.states[].terminal_parents`, a section away from the declaration,
whose `parent_ndims: 1` they read instead and answered "location". In 0.3 the parent is in the
record (`parents: ["context"]`) and every run found it. The one 0.3 role miss was a Haiku run
that took a deterministic transition's near-zero entries (−20.7) for the slip. Under the
generic spelling the slip's record was named `slips` by us, so the generic arm's clean sweep is
partly a naming advantage of the test document, not of the spelling.

## What the refusals were about

- **Shape rules (0.2 only).** Every 0.2 failure but M2's was the trailing control axis and
  `parent_ndims`: three Haiku cells cycled through "parent_ndims must be 2", "trailing control
  axes have sizes (4,) but the control parents [] declare num_controls (1,)", "declared
  controlled but no factor reads their control", and never converged. 0.3 has no layout to
  write, and no refusal of this kind fired in a 0.3 arm.
- **The composite's uncontrolled tables (0.3; reader fixed).** Three of the four Sonnet M4
  refusals were the engine's composite validator demanding the size-1 trailing axis on
  `rat.B_context` that the plain path accepts without. The reader now pads an uncontrolled
  transition table on the composite path, so the document is layout-free on both; M4 Sonnet
  builds first try in both 0.3 arms.
- **`mixture_k` (0.3; format document amended).** All four 0.3 M2 cells were refused once for
  a missing `configuration.inference.mixture_k` and fixed it in one round. The format document
  named the requirement only in the family table and section 9; section 3.5 now states it
  where the switching transition is described.
- **A transition reading the action node (0.3; reader and format document fixed).** Both
  Haiku M3 cells wrote the location transition over `a_location`, the natural reading of "the
  slip happens in the world". The refusal was the indirect "control node 'u_location' is read
  by no transition"; it is now "'a_location' is the action node of 'u_location'; a transition
  reads the control node, so write 'u_location' in its place", and the format document says a
  transition never reads an action node.
- **The legacy reader's crash (0.2; reader fixed).** Sonnet's M2 wrote `innovations` as an
  array under `arrays`; the legacy reader accepted it and the System validator crashed with
  `'ArrayImpl' object has no attribute 'parent_shape'`, three rounds running. The legacy reader
  now refuses a module field under `arrays` by name, pointing at `fields`.
- **Normalised log-probabilities (0.3; not changed).** Haiku's remaining 0.3 failures (M1
  generic, M3 generic, M4 role) were hand-written tables under `log_simplex` that were
  approximate logs (`-0.1, -4.0, -4.0, -4.0`) or mis-shaped nested lists. The sentence is
  right and names the two fixes; each round fixed the one clamp named, and the next clamp was
  refused the next round. Two things an agent-facing format could do, both rulings rather than
  fixes: a `simplex` domain the reader takes probabilities in, and a refusal that lists every
  clamp failing the domain check at once. Neither is done here.
- **Non-standard JSON.** Two Sonnet documents wrote `-Infinity` for an impossible entry;
  Python's reader accepts it, a strict JSON parser would not. Not the reader's concern; noted.

## Confounds

- **0.2 cannot write M3 as posed.** Description 0.2 has no actuator record, and `describe` at
  5ee7dad refuses a System with an actuator kernel. The 0.2 arm could only put the slip in the
  transition table, and the 0.2 reading document (`reading/m3-02.json`) was written that way.
  In 0.2's favour on difficulty.
- **The layout prior.** The 0.2 layout mirrors pymdp's, a prior every agent already has: one
  record per state and per observation with the transition, likelihood and initial prior as
  slots on it. Both arms carry pymdp's letters where a table is named, so the letters are not
  the difference. A 0.2 advantage on the tabular models would be partly this prior; none
  appeared.
- **One refusal at a time.** The reader refuses the first failing address, and the correction
  prompt said to fix what the refusal names and nothing else, so a document with three bad
  clamps costs three rounds whatever the format. The round counts measure the refusal loop as
  much as the format.
- **Reference size.** The 0.2 arm's reference was about 240 lines of prose and a schema; the
  0.3 arm's was the full format document. Neither arm was told anything about the engine.
- **One run per cell.** With one run per cell the per-cell numbers are anecdotes; the
  per-arm aggregates over eight cells carry the comparison.

## What the result decided

- **The default `describe` spelling stays role (Alex, 2026-10-13).** The bake-off plan's
  criterion was to change the default if the generic arm read better. It did by a single
  Haiku answer out of twenty-four, on a record the test document had named `slips`; discount
  the naming and the two spellings read the same. On writing, generic was level for Sonnet and
  worse for Haiku. And the question only has content for a document a person writes: `describe`
  has no names to draw on, so a generic default would emit the role label's own word with every
  parent bracketed and `steps` stated, the role document made longer and nothing more. The
  reading advantage the generic arm showed came from a human choosing verbs like `slips` and
  `sees`, which `describe` cannot do.
- **Which spelling agents reach for.** Under the generic prompt the labels were `start`,
  `moves`, `persist`, `advance`, `slips`, `sees`, `reads`: verbs for the factor's job, never a
  layout word. Recorded for the factor-graph lift.
- **Changed by the run:** the two reader fixes and the legacy refusal above; two sentences in
  `docs/description-format.md` (section 3.5: `mixture_k`, and a transition never reads an
  action node); three tests (`tests/test_description.py`: composite padding, a module field
  under `arrays` in the earlier layout; `tests/test_description_refusals.py`: an action node as
  a transition parent).
- **Open rulings:** the `simplex` domain and the all-clamps refusal, both agent-facing
  conveniences rather than fixes. Both adopted the same day as Q18.6 of
  `docs/design/model-description-0.3.md`: a `simplex` domain lowered to logs on read, and a
  refusal that lists every failing record of a section.

## Files

- `runs/<model>-<arm>-<agent>/r<k>.json` — every document the agents wrote, round `k` from 0.
  Arms are `02`, `03role`, `03generic`.
- `verdicts-before-fixes.json`, `verdicts.json` — the local rebuild of every document under
  the reader the agents wrote against and under the fixed reader: cell, round, `BUILT` or
  `REFUSED`, the refusal sentence, and whether it was a sentence or a crash.
- `reading/m3-02.json`, `reading/m3-03role.json`, `reading/m3-03generic.json` — the reading
  test's documents. The 0.3 pair are `describe`'s output for M3 built natively, the generic one
  derived from the role one and checked to describe back to it; the 0.2 one is the 5ee7dad
  writer's output with the slip in the transition table.
- `reading/answers.json` — the twelve agents' answers, scored above.
