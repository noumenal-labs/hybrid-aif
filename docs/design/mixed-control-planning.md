# Planning over a control space that mixes discrete and continuous nodes

Status: SETTLED (Alex, 2026-10-22).

## Why

A control space may mix discrete and continuous nodes: an option node beside a continuous steer,
a mode selector beside an effort. Every face but planning already takes such a space: the
enumerating and tree planners score a mixed plan that is given, conditional targets and
durations read a discrete node's level beside a continuous one, and `score_policies` evaluates
any mixed plan. No built-in planner searches it; the gradient planner, the CEM planner and the
habit planner refuse a mixed space by name.

## Decisions

**M1 — Exact: enumerate the discrete part, descend the continuous part.** The planner takes the
discrete policy set the System already defines (full enumeration over the discrete nodes'
levels, the order pymdp uses; pruned by `legal=` or supplied by the caller) and, for each
discrete policy, descends `G` over the continuous controls with the gradient planner's own
descent (its floor and its best-iterate return included). All descents run in parallel, vmapped
over the discrete policies. The cost is that of discrete exhaustive search times the number of
descent steps. The posterior over discrete policies is the softmax of `−γ` times each policy's
optimised `G` with the policy prior, as for the enumerating planner (pymdp's
`control.update_posterior_policies`); the selected plan is the discrete policy and its
continuous controls. Every score is expected free energy of a clamped plan. Truncated schemes
("enumerate the first two steps, then hold the level") are policy sets, written as constructors
of the discrete set, not planner modes.

**M2 — A planner of its own, `"mixed"`.** A new registered planner with its own result type,
whose plan, traces and returned-iterate index carry a leading discrete-policy axis. The gradient
planner keeps refusing a mixed space and its sentence names `"mixed"`; a caller who asks for
`"mixed"` has opted into the enumeration's cost.

**M3 — Routine choices.** Every discrete policy's descent starts from the same warm start for the
continuous controls (the caller's `init`, else the box centre); the per-policy traces are
returned so a caller can see which descents had not settled. A space with no continuous node
is refused with a sentence naming the enumerating planner, and one with no discrete node with a
sentence naming the gradient planner.

## Not built: a planner that searches over control beliefs

A planner may roll a plan out under a belief over a discrete control node, a probability vector
per step, and descend over the beliefs jointly with the continuous controls; the engine's
closures already evaluate such a step. It is not built, and it is not on the queue: it would be
built when a model needs a horizon too long to enumerate. If it is, its labels must be honest. The
score of a plan under control beliefs is not expected free energy, for three reasons:

- `G` is nonlinear in the predicted belief, so the score of a soft plan is not the average score
  of the plans it mixes. Risk, a KL divergence convex in the belief, is at most the average of the
  pure plans' risks: a mixture of two plans whose outcomes straddle a target can sit on the target
  that neither reaches. Entropy and information-gain terms grow on a mixture, crediting the
  planner for resolving uncertainty its own indecision created. Both favour the interior, so the
  relaxed optimum can beat every clamped plan and rounding can land on a poor one with no bound
  on the gap.
- The rollout mixes the outcomes at every step before the next, so under mode-dependent dynamics
  it collapses the branches each step and is not a distribution over policies either.
- It is exact when `G` is linear in the relaxed probabilities (a fixed per-level cost on the
  node, a cross-entropy target, a node that selects additive terms): the optimum over the simplex
  is then a vertex and rounding costs nothing.

Such a planner would therefore be a search heuristic: it would warn or refuse where the objective
has convex terms in the relaxed node, re-score its rounded result clamped, and report only the
clamped `G`. What it buys is cost polynomial rather than exponential in the horizon, and
gradients through the discrete choice.

## Build notes (2026-10-22)

Built as `src/hybrid_aif/planning/mixed.py` (`mixed_plan`, `MixedResult`, `make_mixed_planner`,
the registered `"mixed"`), with the descent of the gradient planner factored out as
`planning.gradient.descend` and shared; `gradient_plan`'s outputs are bitwise unchanged.

- `PlannerContext` gains `sizes` and `boxes` (each node's width or level count and each
  continuous node's box), which a mixed space has no flat layout to carry.
- A planner may now return a clamped `ControlStep` policy set (integer levels for a discrete
  node, a float block for a continuous one); `System._plan` accepts it, a belief-valued one still
  raises, and `legal=` reads the discrete nodes' levels of it. `sample_action` draws one policy's
  first step from such a set (`mode="policy"`; `"marginal"` raises, since a discrete marginal
  would pair a level with another policy's continuous controls).
- `evaluate_policies_total` takes a pytree policy set, as `evaluate_policies` already did.
- The policy prior is read over the full enumeration of the discrete nodes and, for a supplied
  set, restricted by each policy's index in that enumeration; without a prior the posterior has
  none.
- Each discrete policy's descent equals `gradient_plan` on its subproblem bitwise (the
  vmapped and the single descents round alike on CPU float32, on the payoff model and on a
  three-mode toy).

Measurements (jitted `infer_policies(metrics=False)`, default 200 steps, payoff model with a
three-level option node and a one-dimensional steer, median of five calls after compilation on one
laptop CPU session; first call including compilation about 0.9 s throughout):

| policy_len | discrete policies | time per call |
|---|---|---|
| 2 | 9 | 6 ms |
| 3 | 27 | 16 ms |
| 4 | 81 | 40 ms |
