# Design: `legal=` — a hard constraint on action selection, separate from E

*Status: ADJUDICATED 2026-09-18 (Alex): L1–L5 as written after the reshape (one interface for level and policy masks; no continuous analogue for now); L6 amended to land in this chunk. **LANDED 2026-09-18:** `legal=` on `infer_policies`, `act` and `replay`; `planning.enumerate.legal_logits` is the one expansion and `System._plan` the one composition point, after the planner, over the policies it returned (no `PlannerContext` field — see L5 as built); refusals on continuous controls and module E; `tests/test_action_legality.py`; N2 carries a dated note; the roadmap item is marked landed. *As built:* `score_policies` does **not** take `legal=` — it scores G, and L2 says G is never modified by legality, so there is nothing for it to do there.
Companion to `system.md` (faces), `amortization.md` A3 (the E designation),
`nesting-systems.md` N2 (the −inf-in-E pattern this replaces).*

## Why

Every embodiment has hard preconditions — a verb the manifest forbids in this
state, a gripper that cannot close twice. Today the only way to tell the planner
is to put −inf into the habit prior `E` (the N2 pattern the mission build used),
and the roadmap flagged that on 2026-08-19: "masking currently rides the habit
prior — fine while E is static, a collision once E is *learned*. Compose
additively at the policy softmax. E's legitimate tenants are habits and declared
action priors; the mask is the squatter." Its gate was "with or before
`distill()`", the first thing that writes E.

The second arrival moves the gate. zeroth-control (2026-09-18): "`act` cannot be
told an action is illegal, so the controller calls `infer_policies` and does the
argmax itself over legal controls. Every embodiment has hard preconditions, and a
hard constraint is not a preference the planner may trade away." A controller
re-implementing selection outside the face is the footgun: the term tree, the
marginals and the recorded `q_pi` then describe a selection that never happened.

## What was adjudicated before, quoted

**N2** (`nesting-systems.md`, resolved by Alex 2026-08-17): "Settled tasks are
masked with `log_E = −inf` — the habit prior as the mask carrier: a *leaf swap*,
so no re-jit as tasks settle within a mission. … Needs a guard test (−inf
through the policy softmax), not a feature." The guard exists:
`tests/test_action_masking.py`, six tests.

**Entry 045** (2026-08-19, the downstream mission review): "log_E triple-duty:
refined to two-and-a-half tenants — escalate-cost is E's birthright (a declared
prior over actions), the MASK is the squatter; explicit action_mask input
flagged on the hybrid-aif roadmap, gated to land with or before amortization
A3's distill() (the first writer of E)."

So the decision that the mask gets its own input beside E is already made; this
round amends its **gate** (the second independent arrival) and settles its
**shape**. N2's route stays valid and tested and the downstream executor is
untouched; N2 gets a dated note that the recommended carrier for a hard
constraint is now `legal=`, and E is the soft prior alone.

## The decisions

### L1. One keyword, `legal=`, whose general object is a mask over the policy set

Two prior uses want two granularities — an embodiment's preconditions mask
**levels** of a factor at this tick; a mission masks **settled tasks**, a level
at every step; and a correlated constraint ("left then right is illegal, each
alone is fine") masks **policies**. They are one object: a mask over the
enumerated policy set. A per-step, per-factor level mask is its
product-structured special case — a policy is legal iff every step uses a legal
level — so every factored mask **expands** to a policy mask. The reverse does
not hold: "mask every action of this policy" would remove every policy sharing
a step with it, so a policy mask is not sugar over level masks; the expansion
runs from factored to set, and that is the direction the code takes.

```python
system.infer_policies(params, beliefs, legal=legal)
system.act(key, params, beliefs, legal=legal)
```

(`score_policies` takes no `legal=`: it scores G, which legality never touches.)

Three spellings, one meaning:

- **a tuple, one entry per state factor** — `None` (every level legal) or a bool
  array: `(num_controls[f],)` masks the **first step** only (this tick's
  preconditions); `(policy_len, num_controls[f])` masks **per step index**
  (N2's "this level at every step", or any step-indexed schedule);
- **a bare bool array `(P,)`** over the enumerated policy set — the general
  form, for constraints no product expresses.

A 1-D entry *inside the tuple* is a first-step level mask; a bare 1-D array is a
policy mask; the two cannot be confused because one is a tuple element and the
other is not.

### L2. Composition: one per-policy log-indicator, added at the softmax after E

Every spelling expands to `log 𝟙[legal(π)]` over the policy set, and
`policy_posterior` adds it once: `q(π) ∝ softmax(−γG + log E + log 𝟙[legal(π)])`.
`E` is untouched, so a learned habit and a mask never collide; the first-step
action marginals — hence `sample_action` in both modes — inherit exact zeros, so
neither stochastic nor deterministic selection can return an illegal action.
**G is not modified**: a hard constraint removes options, it does not re-price
them, and the term tree an illegal policy carries is still its honest score.
N2's leaf-swap property survives because `legal` is a traced array argument:
changing its *values* never retraces (switching between `None` and an array
does, being a change of static structure — L5 says so).

### L3. Per-step masks are for step-indexed constraints — state-dependent legality is not this feature

Preconditions are facts about the *current* state, which the caller knows, so
the common spelling masks `π₀`. A per-step mask is legitimate only where the
constraint is a function of the **step index** (a settled task, a schedule).
Whether an imagined action three steps out would be legal in the *state* it is
imagined from is not something the caller knows or the library can ask the
embodiment about, and `legal` does not pretend to. A state-dependent legality *model* — "this
action is impossible in states where …" — is a fact about the world and belongs
in the generative model (a transition under which the action changes nothing,
or a preference against its consequences), or, later, in a per-step legality
callable evaluated inside the scoring scan, which is ledger entry 12's shape
(the module-E-under-enumeration seam). Recorded, not built.

### L4. Which planners

- **Enumeration:** L2 as written.
- **Habit with a vector E:** the same −inf composition on `q(π) = E(π)`.
- **Habit with a policy module** (a continuous actor): refused with a sentence —
  a legality set over a continuous action space is not a level mask; the box
  bounds already declared on `ContinuousControl` are the continuous constraint
  the library has.
- **CEM / continuous controls:** refused likewise. `legal=` is a discrete-control
  feature; naming it on a continuous System is an error, not a silent no-op.

All-illegal for some controlled factor is the caller's contract violation: a
sentence when the mask is concrete; under `jit` the softmax over an all-`−inf`
row is `NaN`, and the doc says so rather than hiding it.

### L5. Plumbing — amended as built: enforcement at `_plan`, over the returned set

The draft put an expanded `legal` field on `PlannerContext` for each planner to
add. The adversarial review before commit found the hole: an *injected* planner
that never heard of legality would silently ignore it, and `act` would return
an illegal action while every returned quantity said the constraint held — the
exact footgun of the "Why". So there is no context field. `System._plan`
composes the mask **once, after whichever planner returned**, over the policies
that planner offered: `legal_logits(legal, …, policies=returned)` and
`q_pi.logits += log 𝟙[legal]`. Built-in and injected planners are constrained
identically; a bare `(P,)` mask that does not match the returned set is refused
by shape. `policy_posterior(..., legal=)` keeps the parameter for a caller
composing a posterior by hand. No `System` field: a mask is per call, not per
declaration, and a traced array, so changing its *values* never retraces
(switching between `None` and an array does — pass an all-true array to mask
intermittently within one trace). An entry for an uncontrolled factor is
refused: it has no action to constrain.

### L6. Replay carries it too — in this chunk (Alex: "we do want to land this now")

`replay(..., legal=)` takes the same spellings with a leading time axis — a
per-factor tuple of `(T, levels)` or `(T, H, levels)`, or a bare `(T, P)` — so
`chosen` and `action_log_prob` are computed under the constraints the
controller actually had at each tick. Workstream K's likelihood of a recorded
action is conditional on what was legal; without this, a recorded action the
mask forbade would read as a model failure.

## Acceptance

- On the T-maze, masking the argmax action makes `act` return the runner-up,
  deterministically and stochastically, and `q_pi` puts exactly zero on every
  policy starting with the masked level.
- The three spellings agree: a first-step level mask, its `(H, levels)`
  expansion with all-true later rows, and its explicit `(P,)` policy mask give
  bitwise the same `q_pi`; a correlated `(P,)` mask no tuple can express is
  honoured.
- Every test in `tests/test_action_masking.py` still passes unchanged: N2's
  route is not broken, only no longer recommended.
- `E` is bitwise unchanged by masking; a masked policy's G leaves are bitwise
  unchanged.
- The hybrid path (`hill_lg`) behaves identically.
- An all-illegal concrete mask is a sentence naming the factor; `legal=` on a
  continuous System is a sentence.
- `replay(..., legal=)` with a per-tick mask changes `chosen` where the model's
  argmax was illegal and leaves `action_log_prob` finite for legal recorded actions.
- The composer's reference gains the keyword; the roadmap item is marked landed;
  N2 carries a dated note.
