# Design: closed-loop planning for linear-Gaussian factors

*Status: SETTLED 2026-10-19 (Alex), L1–L7, walked the same day; built the same day, see "Build
notes". Customer:
the drive agent of a separate, Noumenal-internal repository, whose note of 2026-10-19 is restated
under "Why" so this document stands alone. Companions: `planning-with-observations.md` (the two
belief chains, D1, and targets on the predictive chain, D5, which L5 annotates),
`per-variable-infogain.md` V11 decisions 6 and 8 (state targets inside a branching search),
`nonlinear-transitions.md` (the linearisation L4 reuses), `nesting-systems.md` (the tick
semantics a slow tier lives under). Sources: Bar-Shalom & Tse 1974, *Dual effect, certainty
equivalence, and separation in stochastic control* (IEEE TAC); van den Berg, Abbeel & Goldberg
2011, *LQG-MP* (IJRR); Platt, Tedrake, Kaelbling & Lozano-Pérez 2010, *Belief space planning
assuming maximum likelihood observations* (RSS); Chen, Georgiou & Pavon 2016, *Optimal steering
of a linear stochastic system to a final probability distribution* (IEEE TAC); Liu, Rapakoulias
& Tsiotras, *Optimal covariance steering for discrete-time linear stochastic systems* (arXiv
2022, a lossless convex relaxation).*

## Why

The drive agent decides whether to ask an operator where a pile row is (its sideways offset `b`)
by scoring, outside its planner, the plan it would follow with the answer against the plan it
follows without. Measured over 26,822 steps, a question's advantage fell by about 6.6 nats per
nat the answer would tell, close to the 7 a per-step KL charges (six step risks and the terminal)
against the one credit of its information gain (`per-variable-infogain.md`, V11 decision D). Put
inside the plan, an answer is credited once and charged nowhere, but it is then worth only its
information gain: what an answer buys, steering onto the real row on every later pass, is
invisible to the planner.

That is a property of open-loop evaluation, not of the decomposition. Two loops need telling
apart:

- *Execution.* Every planner here runs in a loop that plans, acts on the first step, observes and
  plans again; the agent's control is closed-loop (receding horizon, model-predictive control).
- *Evaluation.* Scoring a plan, does the agent imagine its future self reacting to what it will
  observe? An open-loop evaluation scores a fixed action sequence; a closed-loop one scores a
  policy, an action for each thing it might see.

Re-planning built from open-loop evaluations is open-loop feedback control, and its known blind
spot is the one the customer measured: an evaluation that assumes no future reaction cannot see
the value of learning something to act on it (the dual effect). Information gain recovers the
epistemic part of that value, knowing for its own sake; the pragmatic part, acting better once
known, needs a closed-loop evaluation. The library has one, the sophisticated planner, which
branches on categorical outcomes only. This design is its continuous counterpart for
linear-Gaussian factors, where the tree over outcomes need not be enumerated: the separation
principle gives its value in closed form. `planning-with-observations.md` D1 anticipated it ("a
continuous version would branch over sigma points of the outcome and is a later item, if a
customer needs it"); this is the analytic special case.

## The derivation

Linear dynamics `x_{τ+1} = F_τ x_τ + G_τ u_τ + w_τ`, `w ~ N(0, Q_τ)`, readings `y_τ = H_τ x_τ +
v_τ`. Write the state as estimate plus error, `x_τ = x̂_τ + e_τ`. The error's covariance `Σ_τ` is
the filter's, the epistemic chain's posterior covariance under `planning_belief="posterior"`
(D1), and no policy changes it (separation). The estimate moves as `x̂_τ = F x̂_{τ−1} + G u_{τ−1}
+ L_τ ν_τ`, the innovation term spreading the estimate by `M_τ = Σ⁻_τ − Σ_τ` (the predicted less
the posterior covariance). A future self that corrects deviations from the plan by linear
feedback, `u_τ = ū_τ + K_τ (x̂_τ − m̄_τ)`, leaves the estimate a spread

```
C_τ = (F + G K)_{τ−1} C_{τ−1} (F + G K)ᵀ_{τ−1} + M_τ,     C_0 = 0
```

and the state a closed-loop distribution `N(m̄_τ, S_τ)`, `S_τ = Σ_τ + C_τ`, the nominal mean
`m̄_τ` being the plan's predictive mean as today. Open loop is `K = 0`, where `S_τ` is the
predictive covariance.

A Gaussian target `P̃(z) = N(μ, Λ⁻¹)` on `z = T x` scored by its cross-entropy on that
distribution costs

```
−E log P̃(z) = ½ (T m̄ − μ)ᵀ Λ (T m̄ − μ) + ½ tr(Λ T S_τ Tᵀ) + const
```

The gains minimising the sum over the horizon, with a control target's precision `R` charging
the corrections `½ tr(R K_τ C_τ K_τᵀ)`, are the linear-quadratic regulator's, from the backward
Riccati pass over `W_τ = Tᵀ Λ T` (summed over the targets at `τ`, the terminal target's at the
last step):

```
P_T = W_T
K_τ = −(R + Gᵀ P_{τ+1} G)⁺ Gᵀ P_{τ+1} F
P_τ = W_τ + Fᵀ P_{τ+1} F + Fᵀ P_{τ+1} G K_τ
```

The code indexes `W` and `M` by the step whose successor state they weigh, so this document's `W_τ`
is the code's `W_{τ−1}`; the recursion is the same.

That is linear-quadratic-Gaussian control, exact for these models: certainty equivalence gives
the gains, separation the filter. The closed-loop score differs from the open-loop
cross-entropy by `½ tr(Λ T (S_τ − Σ_pred,τ) Tᵀ)`: the part of the predictive spread that
feedback removes, which is exactly the pragmatic value of what the plan's readings will reveal.
Where no control reaches a direction the gains do nothing there and the closed-loop score is the
open-loop one, so knowledge of something no action moves earns nothing beyond its information
gain.

## Decisions (walked with Alex, 2026-10-19)

**L1 — Riccati, not free feedback.** A simpler form, deviations corrected instantly and for
nothing (`Σ_pred → Σ_post` wherever the target weighs), is wrong where the controls cannot
reach: it credits knowing something no action moves, which is the information gain's value
counted a second time. The Riccati pass gets both cases right by construction and contains the
free form as its limit (`R → 0` on controllable directions).

**L2 — The control cost is the control target's precision, else zero.** Costs in G are declared
preferences, so `R` is the precision of a Gaussian target on the control node, `½ (u − μ_u)ᵀ R
(u − μ_u)`. A control with no target costs nothing to correct with: `R = 0`, solved through the
pseudo-inverse, corrects a reachable deviation in one step. The correction does not see the
control's box (the regulator is unconstrained), so a model whose corrections would saturate it
overvalues information; the remedy is a target on the control. Refusing a target-less control
would push users to add targets that also change the nominal plan's objective; a default ridge or
a box-derived `R` would put an undeclared cost into G.

**L3 — A System-level switch, `feedback="linear_quadratic"`, requiring `planning_belief=
"posterior"`.** The switch changes how targets are scored, not how plans are searched, so every
planner can use it. `feedback=None` (default) is open loop; `"linear_quadratic"` names what the
future self is assumed to do, correct deviations by the optimal linear rule for the preferences'
quadratic cost. Under `"prior"` there is no conditioned chain, `Σ_post = Σ_pred`, and the score
would silently become an open-loop cross-entropy, losing the KL's breadth term without saying so:
refused, naming the setting. `"covariance_steering"` is reserved for the KL form (L6).

**L4 — What the switch covers.**
- Gaussian state targets, per step and terminal, on Gaussian-carried factors: scored as above.
- Gaussian observation targets on linear-Gaussian channels: the reading is `H x + noise`, so its
  closed-loop distribution is `N(H m̄ + c, H S Hᵀ + R_obs)` and the utility is its expected
  log-preference. Feedback changes the utility only: the predictive entropy `H[q(y)]` and the
  ambiguity `E H[y | x]` are the information side, read on the belief chains, and no correction
  alters them. So each decomposition carries the change in the leaf that holds the utility: under
  `risk_ambiguity` the closed-loop risk is `−utility_cl − H[q(y)]`, so per channel `risk +
  utility` is unchanged by the switch and the two decompositions' G differ by what they differed
  by before it. *(Corrected on the build: the first draft claimed the V11 identity `risk +
  ambiguity = −(utility + infogain)` holds under the switch; it holds under `"prior"` only, V11's
  acceptance, and the switch requires `"posterior"`, where the two forms differ by design.)*
  *(On the review, Alex:)*
  the alternatives were a KL on the closed-loop reading distribution, `−utility_cl −
  H[q_cl(y)]`, which exceeds the `utility_infogain` G by `H[q(y)] − H[q_cl(y)] ≥ 0`, a breadth
  term charging the concentration the linear-quadratic future produces (the knowledge charge L6
  finds in a KL beside that future) and so moves the two forms apart by it; refusing the switch under `risk_ambiguity`, which would
  make the decomposition a choice of model again, against V11; and changing the ambiguity, which
  feedback does not touch.
- Categorical targets keep their scoring (the KL on the predictive belief; the cross-entropy
  inside a branching search, decisions 6 and 8). No open-loop planner has a feedback model for
  them, so in a mixed model under an open-loop planner G mixes a closed-loop continuous part and
  an open-loop categorical part, which the docs say. This is a property of today's planners: the
  mixed discrete × continuous planner, branching on categorical outcomes and optimising
  continuous controls in each branch with this scoring, would make a mixed model closed-loop
  throughout, and composes with decision 8.
- A target on a mixture-carried factor is refused (a Riccati pass per component is its own
  design).
- Nonlinear transitions use the propagation's own linearisation: the Jacobians under
  `first_order`, the statistical linearisation from the sigma points under `unscented`
  (`nonlinear-transitions.md`). Neural and deterministic transitions are refused.
- A factor no control reaches needs nothing special: the score reduces to the open-loop
  cross-entropy.
- The information-gain leaves are unchanged (they already read the epistemic chain). The
  closed-loop score adds the pragmatic value of knowing beside the epistemic value they carry,
  as a sophisticated tree does; L1 keeps the two from overlapping.

**L5 — Targets on the future posterior: a note in D5.** D5 kept targets on the predictive chain
because a KL on the future posterior rewards not observing and cancels the information gain. The
closed-loop form reads the future posterior for a target, but by a cross-entropy on the
closed-loop distribution and only under a feedback model, so D5 stands for open-loop evaluation.
The note added there:

> *(Note 2026-10-19: under `System(feedback="linear_quadratic")` a Gaussian target is scored on
> the closed-loop state distribution, the future posterior plus the spread a linear-quadratic
> correction leaves, by its cross-entropy and not a KL; `closed-loop-planning.md`. The rule above
> stands for open-loop evaluation, which is what it addressed. The "small difference in E log C"
> between the two beliefs, which an open-loop plan cannot act on, is the pragmatic value of
> knowing once feedback is modelled. No KL is scored on the future posterior, so the
> cancellation shown above does not arise.)*

**L6 — The KL form: principled, later, and not by the clamp.** EFE's state risk is a KL on each
step's marginal, and with feedback modelled it keeps its breadth term without charging knowledge
(Alex: KL control is part of EFE). Over policies of linear feedback plus Gaussian dither, `u =
ū + K (x̂ − m̂) + ξ`, `ξ ~ N(0, D)`, the reachable spreads grow by `G D Gᵀ` and shrink by
feedback, and the KL-optimal policy matches the preference's breadth where it can, dithering to
spread the state where the preference is loose. Information can be ignored, so the value of
information is never negative. This is covariance steering (Chen, Georgiou & Pavon), convex over
covariances (Liu, Rapakoulias & Tsiotras) but without a closed form unless the controls reach
every targeted direction in one step. Three realisations were validated (below):
- *The Riccati floor with an eigenwise clamp against the preference:* closed form, exact under
  one-step authority, unreliable otherwise (up to 19 nats below the optimum on random partially
  controllable systems, where it takes breadth as reachable when it is not). Rejected.
- *Feedback gains and dither as plan variables*, optimised by the gradient planner and
  warm-started from the Riccati gains: within 1e-3 nats of the optimum on 97 of 110 systems, the
  worst 4.4 nats, the worst relative 8.8 %; local minima exist (from random starts one system stopped at 103 against
  59). The KL form's route, gradient planner only, with a guard against local minima before it
  ships.
- *A convex solver per plan evaluation:* exact, but heavy and without gradients for the planner;
  kept as the test oracle.

Open with the KL form: whether execution samples the dither (a plan scored as if the future self
dithers is only realised by an agent that does), and per-target form choice (cross-entropy or KL
on the one closed-loop distribution), which becomes meaningful with it. Under the KL form the
closed-loop reading distribution's breadth `−H[q_cl(y)]` becomes intended, and how the two
decompositions then split `H[q_cl(y)]` against the information gain is to be settled with it
(L4 keeps one G under the cross-entropy form only).

**L7 — Selection, recorded not built.** Choosing per target whether feedback applies does not
hold together: the agent ends up in one state distribution and every target is scored on it.
What can coherently be selected is which targets the future self steers for, the targets that
enter the Riccati pass, every target still scored on the resulting distribution; it would be
written in the `epistemic` key grammar when a customer needs it. The first build steers for
every covered target.

## Beyond the horizon: the hierarchy, not a terminal estimate

What an answer buys mostly lies after the plan's horizon (this is often true and is true of 
the customer's model and is one of their asks). A terminal value counting the end belief's 
lane cost for the rest of the mission was considered and set aside (Alex): it bolts onto an 
estimate of how useful knowledge will be later onto the existing objective, but in general
this kind of knowledge should live at some other, slower level in a hierarchical model that 
conditions the fast one. The pattern is a slow planner whose step is a macro-step (a drive 
segment, a lap, an ask-and-wait), called by the agent at its decision points, a loop outside 
the engine that the tick semantics allow (`nesting-systems.md`, amendment 2026-10-14), handing 
its choice down as a target or prior (`H_deps`). It needs this design: planning open loop it 
would also see no pragmatic value in knowing. It also needs steps of different durations 
(process noise and preference costs scaled by a step's duration, the information gain not), 
a separate item. A slow tier held exactly between its boundaries is expressible today with a 
clock factor as a parent (identity off the boundary).

## Description

`configuration.planning` gains `belief` (`"prior"`, the default, or `"posterior"`) and `feedback`
(absent, or `"linear_quadratic"`). Writing `belief` also fixes a defect found while designing
this: `describe` drops `planning_belief`, so a System planning under `"posterior"` is rebuilt
under `"prior"` and plans differently. A backend that cannot realise `feedback` (a sampled
planning-as-inference backend) refuses it, as each backend declares what it accepts.

## Validation (offline, 2026-10-19)

The exact optimum over linear feedback plus dither, a semidefinite program (cvxpy, Clarabel),
against the realisations, with the solution's policy replayed through the dynamics to confirm its
value. Cases: a scalar random walk; a double integrator with a position preference; the lane
linearised, `(p, θ, b)` with an estimator reading every step, pile readings of `p − b` and an
optional answer of `b`, preference sd 0.02–0.5 on `p − b`; 120 random systems (3 states, 1–2
controls often reaching only some states, readings on random steps, 1–2-dimensional
preferences).

| form | result |
|---|---|
| cross-entropy, Riccati (L1–L4) | the optimum on every case, with and without control cost (largest relative gap 5e-4, solver precision) |
| KL, Riccati floor + clamp | exact on the scalar, double-integrator and lane cases at sd ≤ 0.2; 0 against 0.16 on the lane at sd 0.5; on the random systems from 19 nats below to 0.27 above |
| KL, gains and dither from the Riccati start | 97 of 110 within 1e-3, worst 4.4 nats |

On the lane at sd 0.02 an answer is worth 224 nats under the cross-entropy and 221 under the KL;
open loop gives it no pragmatic value.

## Acceptance

- The closed-loop score equals the Monte Carlo cost of executing the plan with a Kalman filter
  and the Riccati corrections in the true linear-Gaussian system (in-repo, no solver), and the
  harness's numbers for the cross-entropy cases.
- `feedback=None` leaves every score unchanged.
- A factor no control reaches scores as the open-loop cross-entropy.
- On a lane model, an expected answer of `b` lowers the closed-loop score by the predicted amount;
  open loop credits only its information gain.
- On the customer's unicycle (`NonlinearGaussianTransition`), the linearised score against Monte
  Carlo of the corrected closed loop, with the tolerance stated.
- Per channel, `risk + utility` is unchanged by the switch, the information-side leaves are
  unchanged, and the gap between the two decompositions is unchanged, under full and partial
  selection.
- Refusals: `"prior"`; a target on a mixture-carried factor; neural and deterministic
  transitions; the amortized path.
- Descriptions write and read `belief` and `feedback`; a `"posterior"` System round-trips.
- The planner's cost under the switch measured on the lane model and stated.
- The demo notebook re-executed (public API).

## Build notes

*2026-10-19, the first build.*

**Where the pass lives.** `hybrid_aif.objectives.closed_loop`. The planners score a plan by
summing per-step trees and adding the terminal tree once, and treat the carry as opaque, so the
pass needs no planner change: under the switch `System._context_without_control_risk` wraps the
hybrid closures (after the `(qs, qs_prev)` wrapper, before `control_risk`) so the carry is
`(inner, Record)`. The transition writes row `τ` of the record: the step's linearisation on the
joint state of the covered factors (`F_τ`, block-diagonal) and the joint input of every continuous
control node (`G_τ`), read at the pre-step predictive belief, and the covered factors' three
covariances after the step (predictive, epistemic before the step's observations, conditioned
after). The step terms score each covered target open loop, by its cross-entropy on the
predictive belief (a covered state or terminal target switches from its KL to that
cross-entropy; an observation target's utility is already one). The terminal terms run the
Riccati pass backward over the record, the estimate spread `C_τ` forward, and add to each
covered leaf its correction `½ tr(Λ T (S_τ − Σ_pred,τ) Tᵀ)` (negated for a utility) and to each
continuous control target's leaf `½ tr(R K_τ C_τ K_τᵀ)`, under the same keys, which the planners
sum with the step leaves. The tree is therefore per plan, not per step: a covered leaf's
correction arrives through the terminal tree. Every planner reads it, including the tree
planners and CEM; `score_policies` too. With `feedback=None` no wrapper is built and the
`cross_entropy_targets` argument the closures gained is empty, so the code path is the old one:
156 leaves of the term trees of eight fixture families (LQ control and the sensor detour under
both beliefs, the affine/nonlinear pair under every carry and propagation, the T-maze, the
switching model, the two-sensor model under both decompositions) were compared against `HEAD`
bit for bit, all equal.

**Gradients.** The gains are held fixed under differentiation (`stop_gradient`). They minimise the
closed-loop cost given the record, so the derivative of the total through them is zero (envelope
theorem) and the gradient of `G` is exact without differentiating the pseudo-inverse, whose
derivative is unstable where `R + Gᵀ P G` is singular (`R = 0`). Two consequences: individual
leaves' gradients are not each exact, only their sum; and the gradient planner's Laplace
curvature omits the gains' response to the plan, which is zero for linear-Gaussian factors (the
gains do not depend on the plan) and second order for nonlinear ones.

**Interpretations.**
- *Coverage.* A Gaussian state or terminal target on a Gaussian-carried factor; a Gaussian
  observation target on a linear-Gaussian channel observing one Gaussian-carried factor (`T` its
  `H`, counted at the steps the channel is present); a Gaussian or truncated-Gaussian target on a
  continuous control node gives `R` (a truncated target's underlying precision, `diag(1/scale²)`,
  whose quadratic its log-density is on the box). The targets on one step sum into `W_τ`; the
  pass runs on the joint state of the covered factors, so a control shared by two factors is
  regulated for both.
- *Refusals* (each a `ValueError` naming the setting, factor or modality): `feedback` under
  `planning_belief="prior"` (at construction); a value other than `"linear_quadratic"`; a target
  on a mixture-carried factor or on a channel observing one; a covered factor with a neural
  transition (`MLPConditional`); the amortized path (in the planning context and the amortized
  scoring that `imagine` shares). Deterministic transitions live on the amortized path only, so
  that refusal covers them. Not in the spec's list: a `MixtureGaussian` target (state or
  observation) is refused under the switch, since it has no quadratic cost for the regulator; a
  neural transition is refused only where a covered target sits on it. An all-tabular System
  accepts the switch and is unchanged (nothing is covered).
- *Branching.* `planning_belief="posterior"` is already refused beside `branch_on`, so the switch
  never meets a branching search: the sophisticated planner without `branch_on` scores as an
  open-loop planner does, and the switch applies to it through the same carry. With discrete
  controls only, `G_τ` has no columns and no correction acts, so a covered target scores its
  cross-entropy on `Σ_τ + C_τ` with `K = 0`.
- *Nonlinear transitions.* The linearisation is the Jacobian of the propagation's mean map in the
  belief's mean and in the control's value (a belief-valued control: its mean), at the predictive
  belief: under `first_order` `f`'s Jacobians at the mean, under `unscented` the sigma-point
  average of `f`'s Jacobians (by Stein's lemma the regression coefficient `Cov(f(x), x) Σ⁻¹` for
  a Gaussian belief, computed as the propagation computes its mean). A factor no control reaches
  scores exactly its open-loop cross-entropy for linear-Gaussian and first-order factors; under
  `unscented` `Σ + C` with `K = 0` differs from the predictive covariance by the difference
  between the two chains' sigma-point propagations.
- *The V11 identity.* Under `planning_belief="posterior"`, which the switch requires, the two
  decompositions already differ by design (`per-variable-infogain.md` V11: information gain reads
  the epistemic chain, risk and ambiguity the predictive one), so `risk + ambiguity = −(utility +
  infogain)` does not hold literally with or without the switch. What the build holds, and the
  tests pin under full selection, a channel's leaf alone and a sub-block: per covered channel
  `risk + utility` is unchanged by the switch (`−H[q(y)]`, the predictive entropy), the switch
  leaves every information-side leaf unchanged, and the gap between the two forms' G is the same
  with and without it. The literal identity holds wherever it held before.
- *A linear-Gaussian model with a constant noise* gets a correction that does not depend on the
  continuous plan (certainty equivalence): the switch changes which plan is best only where plans
  differ in what they observe (availability, a noise map, a discrete choice such as asking) or
  through nonlinear dynamics. In the lane benchmark below the compiled planner hoists the whole
  correction out of the optimisation loop for this reason.

**Acceptance, as built** (`tests/test_closed_loop.py`, 27 tests).
- Monte Carlo, linear-Gaussian (3-dimensional state, one continuous control with a target, two
  channels on different steps, a state, a terminal and an observation target, a nonzero plan and
  initial mean): every covered leaf equals the numpy implementation of the formulas to 1e-4
  relative, and the Monte Carlo cost of executing the plan with a Kalman filter and `u = ū + K
  (x̂ − m̄)` in the true system (200,000 draws, float64) within four standard errors: state risk
  14.3867 against 14.3806 ± 0.0105, terminal 24.7251 against 24.7036 ± 0.0186, observation
  utility −5.0237 against −5.0386 ± 0.0138, control risk 4.9866 against 4.9860 ± 0.0007 (all
  within 1.2 standard errors). The gains match a numpy Riccati pass to 1e-4.
- The harness's cross-entropy numbers (`cl_validate`, exact LQG): the scalar walk read every
  step and every third (65.0391, 74.1236), the double integrator (16.3640), the lane at sd 0.02
  without and with the answer (610.3206, 386.3012), each to 1e-4 relative after the target's
  normaliser. The lane's state target is on the whole factor `(e, θ, b)` with `e = p − b`, the
  heading and offset held at sd 1000, where the harness weighs `p − b` alone.
- The lane: the answer of `b` lowers the closed-loop risk by 224.016 at sd 0.02 (harness 224.019)
  and 2.2402 at sd 0.2 (harness 2.2402); open loop the risk is the same with and without it and
  G differs by the information leaves alone.
- A factor no control reaches scores its open-loop cross-entropy (to 1e-5), beside a steered
  factor whose leaf takes the corrections.
- The unicycle (`NonlinearGaussianTransition`, steered by a continuous turn rate, its lateral
  position read every step and targeted, eight steps along a curving plan): the score is the
  numpy first-order linearisation's to 1e-4, and within 0.5 % of Monte Carlo of the corrected
  plan executed in the true nonlinear system with an extended Kalman filter, under either
  propagation (measured 0.17 % first order, 0.20 % unscented, 129.71 and 129.67 against 129.94 ±
  0.018; open loop scores 159.9). The tolerance is the linearisation's error at a heading spread
  of about 0.1 rad, two orders above the sampling error.
- `feedback=None`: the bit-for-bit comparison above, and in the suite the plain carry and the KL
  on a state target.
- Refusals: each of the above; descriptions write and read `belief` and `feedback` (written only
  off their defaults; `build` refuses `feedback` without `belief` `"posterior"`), and the sensor
  detour's `"posterior"` agent rebuilds as `"posterior"` and scores as it did.
  `tests/test_noise_map.py` had pinned the dropped belief (`planning_belief == "prior"` after a
  round trip, "a planning option is not a document's"); it now asserts the fix.
- Cost, on a laptop CPU, jitted with the parameters traced: the lane (3 states, 6 steps) runs 200
  gradient steps in 0.44 ms with and without the switch and scores 512 plans in 0.16 ms either way
  (the correction is plan-independent there and hoisted); the unicycle (3 states, 8 steps,
  first order) 15.4 ms against 9.5 ms for 200 gradient steps and 2.3 ms against 1.5 ms for 512
  plans, about 1.6 times; compilation 0.25 s and 0.7 s.
- **On the review (2026-10-19).** `merge` built the composite without `planning_belief` or
  `feedback`, so a component planning on the posterior chain in closed loop merged into a
  composite planning open loop on the prior, silently (the belief had been dropped before this
  feature); both are now `_agree`d like the other planning options, and the `subsystem` view
  copies them. A mutation study killed 11 of 12 single-point mutants; the survivor, the
  linearisation read at the post-step belief instead of the pre-step one, was invisible to the
  unicycle score at its tolerance, so the recorded `F` of the first turn is now asserted against
  the Jacobian at the pre-step mean. Envelope argument confirmed numerically: `jax.grad` of G
  against central differences to 1.5e-4 on the unicycle under both propagations, and scaling the
  gains by 1e-2 moves G by 2e-4, second order. Two docstrings saying every target reads the
  predictive belief were qualified.
