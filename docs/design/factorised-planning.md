# Design: factorised planning, a policy posterior per control node by mean-field sweeps

*Opened 2026-11-18 (Fable, for Alex), from the multi-rover survey (`terrain-field.md`, "The
survey after 0.21.0"; the charger rework of the same day). Rulings F1–F9 are proposed; the ones
marked for Alex are open until he confirms. Companions: `parameters-within-the-plan.md` D1, A3
(the working counts a plan learns within it, the message's carrier here); `nesting-and-coupling.md`
N8 (a control node consumed as a probability vector); `mixed-control-planning.md` (the
`ControlStep` policy set, which this planner reuses); `nested-rounds.md` (shelved; a sweep here
is the round it wanted).*

## Why

A System with several control nodes plans over their product: `enumerate_policies` lays out
every sequence of joint actions and the enumeration planner scores them all, `(Π_k n_k)^H` of
them. The survey's two-rover scheduler has two option nodes of five levels and a horizon of three:
15,625 joint sequences, 0.22 s a compiled round and about 5 GB resident. One charge option per
sunny region takes each node to seven levels: 117,649 sequences, the same speed, 19 GB resident,
and a five-replica batch no longer fits a 128 GB host. The cost is the trace of every joint
sequence held for the round, and it grows as the product of the nodes' level counts to the
horizon. A third rover would be out of reach at any option count.

The product is not where the model's structure is. The rovers' visit and energy factors depend on
their own option alone; the one coupling is the shared field, learned within the plan
(`plan_learning`), where a cell one rover's plan reads is worth less to the other's. The joint
posterior over policies, `q(π_1, π_2) ∝ exp(−γ G(π_1, π_2))`, is a two-variable factor graph:
a unary factor per node (its own transitions, preferences and novelty) and one shared factor (the
field's novelty, which reads both plans' imagined counts). Mean-field variational inference over
that graph keeps one posterior per node and updates each against the others' expectations, and
its cost is the sum, not the product, of the nodes' sequence counts.

## The rulings

**F1. One posterior per control node, updated by sweeps.** The planner `"factorised"` holds, for
each discrete control node `k`, a posterior `q(π_k)` over that node's own sequences, the `n_k^H`
rows of `enumerate_policies((n_k,), H)`. A sweep updates the nodes in node order: node `k`'s
sequences are scored with every other node's policy entered as its current per-step marginal over
levels (F2), and `q(π_k) ∝ softmax(−γ E_{−k}[G] + log E_k)`. The sweep count is static
(`make_factorised_planner(sweeps=3)`; the registry name `"factorised"` takes the default), so
the program is one compiled unrolled loop; sweep 0 scores each node against the others' uniform
marginals, so with one sweep every node plans as if the others were unknown. Its cost is
`sweeps · Σ_k n_k^H` sequence evaluations, 2 · 3 · 343 for the scheduler at seven options against
117,649.

**F2. The message is the expected sufficient statistic.** Node `j`'s policy enters node `k`'s
evaluation as a `ControlStep` whose entry for `j` at step `t` is `j`'s marginal over its levels at
`t` under `q(π_j)` (`action_marginals` per step of the grid). Every reader of a control node
already consumes a probability vector: a transition's control axis is contracted against it
(`take_controls`, N8), a learned table's parent one-hot becomes the vector
(`_table_parents`), a pooled reading's count becomes the expected count (`count_at`), an
observed action node's belief is pushed through its kernel (`action_node_beliefs`). So the
belief push is exact under the mixture, and the nonlinear leaves (novelty, risk) are evaluated at
the expected statistic rather than averaged over `q(π_j)`: the standard variational message of a
conjugate-exponential factor, moment-matched, not an exact expectation. Stated as such in the
docstring; the gap is measured in the pin (below).

**F3. What the planner returns.** *(Alex to confirm.)* The `Planner` protocol returns `(q_pi,
tree, policies)`. The factorised planner returns `q_pi` as the posterior over the joint
*first-step* actions, the product of the nodes' first-step marginals over the grid
`enumerate_policies(num_controls, 1)` (shape `(A, 1, K)`); so `sample_action` in its
`"marginal"` mode and the face's legal-mask application work unchanged, and `"policy"` mode draws
a joint first step. `tree` is `{}` (the protocol's form for a planner that builds no aligned
tree). `policies` is a `FactorisedPolicies` NamedTuple: `first`, the `(A, 1, K)` grid `q_pi`
is over; `grids`, the per-node sequence sets `(n_k^H, H)`; `posteriors`, the per-node
`Categorical`s; `trees`, the per-node term trees (leading `n_k^H` axis, under `metrics=True`,
else `None`); `marginals`, the per-node `(H, n_k)` step marginals of the last sweep; and
`movement`, the largest change of any marginal in the last sweep (a scalar, the convergence
diagnostic a record can keep). `act_and_policies` and `sample_action` branch on the type as they
do for `PlanPosterior` and `ControlStep`, reading `first`. The runner's `Trace` keeps it in
`plans`, its `policies` `None`, as for the other non-enumerating planners.

**F4. The policy prior factorises or is refused.** `E` is `None`, or a tuple of per-node
log-weight vectors over each node's own sequences (`SystemParams.policy_prior` as a tuple of
arrays or `Categorical`s in node order). A single vector over the joint set is refused with a
sentence naming the planner and the form it takes, since no factorised posterior can carry it. A
policy module acts through the `"habit"` planner, as for enumeration.

**F5. Discrete nodes only, exhaustive per node.** A continuous or mixed control space is refused
as the enumeration planner refuses it (`refuse_mixed`, the same sentences). Each node's sequences
are scored by `evaluate_policies` / `evaluate_policies_total` over a `ControlStep` policy set
(`(n_k^H, H)` levels for node `k`, `(n_k^H, H, n_j)` broadcast marginals for every other node),
the flat scan, since the sets are small; the prefix tree is not used. A node with one level
contributes a constant marginal and is never swept.

**F6. Legality.** A first-step `legal=` mask on the joint grid is applied by the face after the
planner returns, as now, on `first`. A per-depth mask is not read (`legal_depth=False`); a node
whose own levels are all illegal at the first step raises as the enumeration planner does.

**F7. Where the sweep starts, and warm starts.** Sweep 0 uses uniform marginals for the other
nodes (F1). A warm start from the previous round's marginals shifted one step is named and not
built: the scheduler's rounds are durations apart and its plans are short, so the gain is
unmeasured; `plan(warm_start=True)` with this planner is refused by the existing check, which
requires a planner that takes `init`.

**F8. Convergence is reported, not enforced.** The sweep count is fixed and static. `movement`
(F3) says how far the last sweep moved; a record that wants to know whether more sweeps would
change the plan reads it. Damping is not built: the scheduler's coupling is weak (one shared
table) and the pin below measures what three sweeps leave.

**F9. Symmetry.** With symmetric nodes and a cold start, a mean-field sweep can leave two nodes
on the same plan each assuming the other's stale one. The sweep order is node order, so node 2
always sees node 1's updated posterior within a sweep (Gauss–Seidel, not Jacobi), which breaks
the tie the moment one node moves. Not otherwise handled; the survey's rovers start at different
cells and the pin (below) looks for it.

## Where it plugs in

- `planning/factorised.py` (new): `FactorisedPolicies`, `factorised_plan(ctx, sweeps, use_bound,
  metrics)`, `make_factorised_planner(sweeps=3, **kw)`, registered as `"factorised"`. Reads
  `ctx.num_controls`, `ctx.policy_len`, `ctx.transition_fn`, `ctx.step_terms_fn`,
  `ctx.terminal_terms_fn`, `ctx.gamma`, `ctx.policy_prior`. `planning/__init__.py` exports them.
- `planning/enumerate.py`: `sample_action` reads `FactorisedPolicies.first`; `action_marginals`
  unchanged.
- `system/faces/plan.py` `act_and_policies`: the `FactorisedPolicies` branch, before the
  `ControlStep` test. `System(planner="factorised")` resolves by the registry as any name does.
- `composition/runner.py`: `_Plan._enumerated` already returns `None` for a planner that is not
  `"enumeration"`; `Trace.plans` holds the NamedTuple. `_acted_plan` is reached only under
  `warm_start`, which F7 refuses.
- `examples/survey_scheduler.py`: `scheduler(..., rovers=2)` declares `planner="factorised"`;
  `JOINT_HORIZON` and its comment stay (the horizon is a model choice; the sequence count is now
  `2 · 7^3`). `examples/survey_run.py` `joint_decision_record` reads the chosen sequence, its
  terms and `G` per rover from `trace.plans` (`posteriors`, `trees`, `grids`), `efe_by_option`
  becomes per rover, and `movement` is recorded. The one-rover scheduler keeps `"enumeration"`
  (one node: the two planners agree exactly, pinned below, and the one-level fixture stays
  bitwise).
- `docs/capabilities.md`: the planners paragraph gains the factorised planner.
  `docs/project-brief.md`: the roadmap entry `[flagged 2026-11-18]` for this document, marked
  landed when built.

## What to pin

1. *Independence is exact.* Two independent T-mazes declared as one System with two control
   nodes (disjoint factors, no shared table): the factorised planner's first-step marginals equal
   the enumeration planner's `action_marginals` to float tolerance after one sweep, and
   `movement` is zero after the second.
2. *One node is the enumeration planner.* On the one-rover survey scheduler (`rovers=1`,
   horizon 4, 625 sequences) `q_pi` of `"factorised"` equals `"enumeration"`'s to float
   tolerance, and the one-level fixture `tests/data/survey_run.npz` is bitwise unchanged.
3. *The coupled case against the product.* The two-rover scheduler at `JOINT_HORIZON` 2 and five
   options (625 joint sequences, cheap to enumerate): the factorised planner's joint first-step
   marginals against the enumeration planner's `action_marginals`, after three sweeps. The pin
   records the largest absolute difference and the agreement of the argmax; the expected gap is
   small and not zero (F2), and the test states the measured number.
4. *The sweep moves.* On the same case the first sweep's marginals differ from the uniform
   start and `movement` falls across sweeps (a decreasing sequence, printed in the test).
5. *Refusals.* A joint `policy_prior` vector; a continuous node; `warm_start=True`.
6. *The demo.* Five seeds of the two-rover arm under the factorised planner at seven options:
   coverage within the seed spread of the enumeration planner's (0.863 ± 0.016 at 800 ticks,
   measured 2026-11-18 in the charger's scratch runs), stall ticks 0, resident memory back near
   the five-option figure. The two-level fixture is re-recorded (the first decision's
   `q_pi` is no longer the joint one) with Alex's approval, as every re-record is.

## Decisions for Alex

- F3, the returned form: `q_pi` over the joint first step, the per-node detail in
  `FactorisedPolicies`. The alternative, a `q_pi` with a leading node axis, breaks every consumer
  of a `Categorical` over one set.
- Whether the two-rover scheduler adopts the planner now (the plug-in above) or the library
  change lands first and the demo follows in its own commit. The build below assumes both, the
  library first.

## Out of scope, named

A warm start from the previous round (F7); damping (F8); a per-depth legal mask (F6); the
prefix tree per node (F5, unneeded at these sizes); structured mean field over subsets of nodes
(a pair of nodes planned jointly against the rest), the natural extension when two nodes are
tightly coupled; the exact expectation of a nonlinear leaf under the other nodes' posteriors
(F2), which would cost the product again and is what the pin in item 3 bounds.

## As built (2026-11-18)

F1–F9 confirmed by Alex (F3 as written); the library piece is built, the survey's adoption (the
two-rover scheduler declaring `"factorised"`, pin 6) is the next commit's.

- `planning/factorised.py`: `FactorisedPolicies` (the F3 fields), `factorised_plan(ctx, sweeps,
  use_bound, metrics) -> (q_pi, FactorisedPolicies)`, `make_factorised_planner(sweeps=3)`, and
  `factorised_planner` registered as `"factorised"` (three sweeps). Each node's sequences are
  scored by `_score_node`, the flat scan of `evaluate_policies` (under `metrics`) or
  `evaluate_policies_total` over a `ControlStep` holding the node's own level and every other
  node's step marginal `(n_j,)`, closed over (see below); Gauss–Seidel in node order, the sweeps an unrolled Python loop.
  `q_pi` is over `enumerate_policies(num_controls, 1)`, its logits the sum of the nodes'
  first-step log-marginals (by logsumexp over each posterior). Exported from `hybrid_aif.planning`.
- `planning/enumerate.py` `sample_action` reads a `FactorisedPolicies` as its `first`, so both
  modes act on the joint first-step grid. `system/faces/plan.py`: `_plan` applies a first-step
  `legal=` to `first` (F6) and returns the NamedTuple; `act_and_policies` branches on it before
  the `ControlStep` test (mode `"marginal"` by default). `System(planner="factorised")` resolves
  through the registry, and `plan()` / `run` needed no change: `_Plan._enumerated` gives `None`,
  `Trace.plans` holds the NamedTuple, and the carry's structure is that of every round (pinned
  over three rounds of `run`'s scan).
- Tests: `tests/test_factorised_planner.py`, 12 tests.
- *The other nodes' marginals are read once per step* (fixed the same day): scoring through
  `evaluate_policies` broadcast them to `(n_k^H, H, n_j)` and vmapped them, so every reader of
  node j contracted its table once per sequence (the rover transition's `(343, 256, 256, 8, 7)`),
  and one two-rover decision at seven options and horizon 3 took 30.8 s for three sweeps (6.5 s
  for one) against the enumeration planner's 0.03 s. `_score_node` now writes the flat scan
  (both the tree and the folded-total forms) with the marginals closed over, unbatched under the
  vmap (policy-independent-leaves.md H2): 0.46 s for three sweeps, 0.17 s for one, the
  posteriors bit-identical, pinned on the jaxpr (no per-sequence intermediate as large as `P`
  copies of the table).

**The pins, measured.**

1. Two T-mazes as one System (two control nodes of five levels, horizon 2, no shared factor):
   first-step marginals equal the enumeration planner's to 3e-10 after one sweep; the second
   sweep's `movement` is 3e-10 (pinned < 1e-6). A per-node prior pair gives the enumeration
   planner's marginals under the joint prior that is their sum, to 1e-6.
2. The one-rover scheduler (horizon 4, 625 sequences): the node's posterior equals the
   softmax over the flat scan of the same grid to 1e-6, `movement` 0.0. Against the enumeration
   planner itself `q_pi` differs by 2.3e-4: the enumeration planner scores by the prefix tree,
   whose `transition:map` novelty leaf (about 44 nats, summed over 256 cells) differs from the
   flat scan's by 2.4e-4 nats (5e-6 relative, float reassociation), and γ = 16 carries that into
   `q_pi`. Pinned at 1e-3 against the enumeration planner, argmax equal. The one-level fixture is
   untouched (the examples are not changed in this commit).
3. The two-rover scheduler at horizon 2, five options, the field learned within the plan, a
   field holding seeded counts (a third of the cells read three times): after three sweeps the
   largest absolute gap of a first-step marginal from the enumeration planner's is **1.0e-9** at
   the scheduler's γ = 16 (3.2e-6 after one sweep). At γ = 16 both posteriors are near one-hot
   (rover 1 surveys sector 4, rover 2 charges), so the pin also measures γ = 1, where both are
   spread: **1.6e-4** after three sweeps (1.1e-3 after one). The argmax agrees in both; pinned
   at 1e-7 and 5e-4, and the γ = 1 gap above 1e-5 (the coupling is live).
4. `movement` on the γ = 1 case, by sweep: **0.395, 9.0e-4, 8.9e-8** (at γ = 16: 0.8, 3.2e-6,
   0.0). Beyond three sweeps it sits at float noise (1e-7), which is why the pin stops at three.
5. Refusals: a joint `policy_prior` vector (and a tuple of the wrong length or a wrong-shaped
   entry), a mixed space (`refuse_mixed`'s sentence), an all-continuous space (the enumeration
   planner's sentence, naming the factorised planner), and `plan(warm_start=True)` by the
   existing `takes_init` check, for a made planner and for the registered one.

**Deviations and what the design left open.**

- *A per-step `legal=` mask is refused*, not ignored (F6 said "not read"): `q_pi` is over the
  first step alone, and silently dropping the later rows would leave them unenforced. The
  sentence names the `(n,)` first-step form and the enumeration planner.
- *The evaluator is one per call*: under `metrics=True` every sweep scores with
  `evaluate_policies` (the tree), not only the last, so `movement` never measures the
  reassociation between the tree and the folded total (it did, at 4e-6, in the first draft).
  The cost is the tree's leaves in every sweep, small at these sizes.
- *A one-level node* is not swept; under `metrics` it is scored once in the last sweep so that
  `trees` has an entry per node.
- `make_factorised_planner` binds `sweeps` alone (the design's `**kw`): `use_bound` and
  `metrics` are per call, as the protocol passes them, and there is no other knob. The made
  planner's `__name__` is `factorised(sweeps=N)`, which the warm-start refusal names.
- A policy module in `policy_prior` raises the enumeration planner's module sentence, naming
  the factorised planner.
