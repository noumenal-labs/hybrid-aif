# A modality over an action node, in planning and on the hybrid path

Status: SETTLED (Alex, 2026-10-20), built the same day.

## Why

A model whose sensor channel depends on what the agent did this round (an answer that arrives
only when the agent asked; a cue read only when the agent looked) is, in the library's own
grammar, a modality whose parent is the action node: the control-states ruling of 2026-10-09
gives every control node `u` an action node `a` with the actuator kernel `P(a | u)`, a delta
unless declared, and lets a modality read `a` as it reads a state factor, `P(o | a, ...)`. The
declaration exists and round-trips through the description format. What was missing is the
reading of it: every planning face refused a modality over an action node, and so did the
hybrid path's inference faces, so the only way to write "the answer exists under ASK" was an
indicator dimension of the state that the ask's kernel sets and a noise map reads
(`step-durations.md`, build notes). This record lifts the refusals where the reading is a
selection or a contraction the engine already does, and keeps them where it is not.

A control node is already a parent of transitions (`B_control_deps`) and of per-step targets
(`C_deps`). A modality's parent is the action node rather than the control node because a
reading depends on what was done, not on what was intended; under the delta kernel the two
coincide in a plan, and under a declared actuator the difference is the kernel.

The sibling idea of a presence table per level (a channel declared present under some levels
of a control node) was parked on 2026-10-20: for a categorical channel the null outcome is a
deterministic reading and needs no presence concept, and for a Gaussian channel the absent
level is written as the noise limit, which is exact for information gain and a constant for
ambiguity. The exact Gaussian null outcome, which a designated channel would need, is the
queued follow-on, and it would extend `present=` rather than add a declaration (built
2026-10-21 as a per-level entry of `present=`; planning-with-observations.md D4).

## Decisions

**S1 — Which faces read it.** The planning faces (`infer_policies`, `score_policies`,
`planner_context`) read a modality over an action node on the tabular and hybrid paths; the
hybrid path's `infer_states`, `update_empirical_prior` and `as_env` read it as the tabular path's
already do. `learn`, `replay`, `merge`, `subsystem` and the amortized path keep their refusals,
each naming the face, as before. (`learn` and `replay` carry beliefs across ticks and would need
the action node's prior per tick from the trace; `merge` and `subsystem` renumber `A_deps`; none
is needed by the models that asked.)

**S2 — The action node at a plan step.** At step `t` of a plan the action node's belief is
`P(a | u_t)`: the one-hot of the level under the delta kernel, the kernel's column at the level
under a declared actuator, and under a probability-vector step (a soft
:class:`~hybrid_aif.controls.ControlStep`) the mixture of columns over the step's probabilities.
It is a parent at the step, not a belief the plan carries: the step's belief tuple is extended
by it only for the channels' parent lookups, nothing is scored about it (no information-gain
leaf is keyed by an action node; the gain of a channel reading `(a, x)` is about `x`), and
nothing is carried to the next step. The existing `_sensed_predictions` is this rule under a
clamped level. (Reaffirmed 2026-10-21: the gain about `a` is not scored because nothing in the
plan reads `a`, transitions reading the control node; a model that wants to value knowing what
it did declares its effector as a state factor, as the control-states ruling says. Learning the
kernel `P(a | u)` itself would be a parameter gain, and is a queued item.)

**S3 — A Gaussian-carried factor selects, a categorical parent contracts.** A linear-Gaussian
channel observing a Gaussian-carried factor whose mode parents are all action nodes takes the
level's `(H, c, R_chol)` at the step when the mode belief is a point, the delta kernel under a
clamped level; the factor stays Gaussian-carried, with no mixture declaration, since a certain
mode grows nothing. A mode belief that is not a point (a declared stochastic actuator, a
probability-vector step) is refused with a sentence naming the mixture-carried route, which is
the existing switching path (a mode parent that is a state factor keeps the existing rule: the
observed factor is mixture-carried). A tabular channel, or a Gaussian emission, over an action
node contracts over the node's categorical, point or not, as it does over a state factor. The
same holds at inference on the hybrid path: the action node's prior, given with the beliefs as
the tabular path takes it (the decision's one-hot, the efference copy), selects when it is a
point and is refused otherwise. A noise map on a selecting channel is not built, as on a
switching channel.

**S4 — The epistemic chain and the closed loop.** A selecting channel is a conditioner of the
epistemic chain under `planning_belief="posterior"`, its head and noise taken at the step's
level, so the chain's covariances carry what the reading at that step would do. The closed-loop
pass reads those covariances and so credits such a channel with no further rule, as the
customer's payoff model needs. A channel with a state-factor mode parent stays outside the chain
(planning-with-observations.md D2).

**S5 — The declaration is unchanged.** `A_deps` names the action node as it does today
(positionally, as the siblings do; the keyed migration will carry it). `describe` and `build`
round-trip as before. The information-gain, ambiguity, risk and utility leaves of such a channel
are keyed and scored exactly as any channel's. The absent level of a Gaussian channel is written
as the noise limit at that level (a wide `R_chol` in the mode's slot).

**S6 — Acceptance.**

- *The customer's model, declared.* The payoff toy of `step-durations.md` with the answer channel
  as a linear-Gaussian observation over `(lane, a_option)`, `R_chol` per level (wide under DRIVE
  and IDLE, the answer's sd under ASK), and the lane factor `(e, b)` without the indicator: the
  crossing rows are those of the indicator form (3 at an ask cost of 20 nats, 7 at 45), and every
  plan's score matches the indicator form's to float tolerance, the indicator having been read
  at a mean of exactly 0 or 1.
- *A cue by looking, tabular.* A T-maze whose cue is read only under the LOOK level of a control
  node: the modality over `(context, a_look)` with the cue's rows under LOOK and a uniform row
  otherwise; the enumerating planner prefers LOOK first when the cue resolves the reward arm, and
  the scores equal those of the same model written with the cue location as a state factor the
  LOOK level moves to, bitwise.
- *The efference copy at inference.* `infer_states` on the hybrid path with the one-hot prior for
  the action node matches a hand-written Kalman update with the level's `R`; a soft prior is
  refused with the S3 sentence.
- *A probability-vector step.* The gradient planner over a discrete node, relaxing to
  probability vectors, is refused with the S3 sentence on a selecting channel and runs on a
  contracting one.
- *The chain.* Under `planning_belief="posterior"` the information-gain leaves of a selecting
  channel telescope as a Kalman filter with the per-step `R` would, and under
  `feedback="linear_quadratic"` the closed-loop terms change with the level sequence.
- *Refusals kept.* `learn`, `replay`, `merge`, `subsystem` and the amortized path refuse as
  before, naming the face.

## Build notes (2026-10-20)

- **A closed-loop observation target on a selecting channel is refused.** Under
  `feedback="linear_quadratic"` the pass would take the selecting channel's head at each step's
  level; that is not built, and the target is refused with a sentence. The channel's
  conditioning of the epistemic chain, which S4 describes, is unaffected.
- **The payoff fixture clears its indicator under DRIVE.** The indicator form of
  `test_conditional_targets` first held the indicator through the DRIVE rows after an ASK, so
  the answer was re-read on every later row. DRIVE now clears it (ASK's innovation still sets
  it), the declared form is compared against that fixture directly, and the crossings are
  unchanged at 3 rows (20 nats) and 7 (45 nats); the corrected table is in `step-durations.md`.
- **A Gaussian emission over an action node alone is admitted on both chains.** It has no
  Gaussian factor to condition, so the epistemic chain of `planning_belief="posterior"` leaves
  it out and scores it as `"prior"` does, under the delta kernel and under a declared actuator.
- **S2 is enforced for a channel over action nodes alone.** Under a declared actuator the first
  build scored such a channel's information gain as `I(o; a)`; it now mounts no
  information-gain leaf, on the tabular and the hybrid path alike. Its ambiguity, risk and
  utility leaves are unchanged, and a channel over a state factor and an action node scores its
  gain about the state factor only.
