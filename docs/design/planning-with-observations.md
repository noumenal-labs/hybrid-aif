# Design: planning that expects to observe — belief-space planning on the linear-Gaussian path

*Status: SETTLED 2026-10-15 (Alex), D1–D7; built the same day, see the "As built" notes. Customer: the drive agent of a separate,
Noumenal-internal repository, whose note of 2026-10-15 (against `main` at c5de9b1) asked for
four features and two small fixes; its asks are restated under "Why" so this document stands
alone. Companions: `objective-terms.md` (the two pairs), `per-variable-infogain.md` V9 (the
linear-Gaussian info-gain leaf), `mixture-filtering.md` (mixture-carried factors), `replay.md`
(the `present` mask at inference). Sources: Friston et al. 2015, *Active inference and
epistemic value* (the two decompositions of expected free energy); Friston et al. 2021,
*Sophisticated inference* (beliefs that condition on expected outcomes along a plan);
Bar-Shalom, Li & Kirubarajan 2001 (the Kalman covariance recursion is data-independent; the
extended filter's evaluation at the predicted mean).*

## Why

On a linear-Gaussian model, planning predicts and never conditions. The belief carried through
the planning scan is the predictive one, `Σ' = F Σ Fᵀ + Q`, so its covariance does not depend
on the plan, and the information-gain leaf of every linear-Gaussian modality is the same number
for every plan the planner considers. An agent cannot prefer a route because it will see
better there. The discrete path does not have this problem: its expected posteriors differ by
state, and the sophisticated planner branches on outcomes. The gap is specific to the
continuous path, and it is general: a localising robot, an active-sensing task, a sensor
placement choice under linear-quadratic-Gaussian control all need it.

What the customer asks, restated: (1) a planning belief that conditions on each step's expected
observation, so the information-gain leaf is plan-dependent; (2) an observation noise that
depends on the observed state, a precision map, since with a constant noise even the
conditioned covariance ignores the controls; (3) per-step observation availability in planning,
for sensors slower than the plan step and known outages; (4) a cross-entropy form for state and
terminal targets, because a KL target rewards a broad belief once the covariance moves; (5) two
small fixes so a custom term row can run on the linear-Gaussian path. Their open questions:
whether (1) is a System option or a planner option; whether evaluating the noise map at the
predicted mean is acceptable as a first version; whether any of this exists on a branch (it
does not).

Of these, (1) and (2) are one design, (3) is a rider on the same closures, (5) is folded in, and
(4) is declined with the reason recorded (D5): the trap it would fix is an artefact of the
one-chain implementation the note sketched, and the two-chain design below cannot produce it.

## D1 — Two belief chains (SETTLED, Alex 2026-10-15)

Both beliefs below are variational posteriors over the same hidden state at a future step `τ`
of an open-loop plan `π = (u_1, …, u_T)`; they differ in what they condition on:

```
q(s_τ | π)            the predictive belief: the current posterior pushed through the dynamics
                      under the plan, marginal over the observations y_1..y_τ not yet seen
q(s_τ | π, y_1:τ)     the future posterior: the belief the agent will hold at τ having seen
                      y_1..y_τ, a random object now because the y's are unknown
```

They are tied by the tower property, `E_y[q(s_τ | π, y)] = q(s_τ | π)`, and their entropies by
the chain rule of mutual information:

```
H[q(s_τ | π)]  =  E_y H[q(s_τ | π, y_1:τ)]  +  I(s_τ ; y_1:τ | π)
```

Expected free energy puts each where it belongs. In `G = E_{q(o,s|π)}[log q(s|π) − log p(o,s)]`
the entropy term is of the predictive `q(s|π)`, and the future posterior `q(s|o,π)` appears only
inside the information gain. Under the risk-and-ambiguity split the predictive's entropy inside
`KL(q(s_τ|π) ‖ C)` *is* the epistemic value, which is why a loose goal rewarding a broad
predictive belief is the semantics of a loose goal and not a defect (Alex: "an agent with a
maximum-entropy prior will actively seek uncertainty, and that is fine").

**The rule.** The planning scan carries two chains. The *predictive chain* is today's
`transition_fn`, predict only; every target, every utility, risk and ambiguity leaf, every
terminal term reads it, exactly as today. The *epistemic chain* predicts from the previous
step's future posterior and conditions on the step's expected observations (D2); only the
information-gain leaves read it. The step-`τ` leaf of a linear-Gaussian modality becomes the
conditional information gain

```
IG_τ  =  I(y_τ ; s_τ | y_1:τ−1, π)  =  H[q(s_τ | π, y_1:τ−1)] − E H[q(s_τ | π, y_1:τ)]
      =  ½ log|H Σ⁻_τ Hᵀ + R| − ½ log|R|,     Σ⁻_τ the epistemic chain's predicted covariance
```

the same closed form `state_info_gain` computes today, evaluated on the epistemic chain's
belief instead of the predictive one. Summed over the horizon the leaves telescope to the total
information the plan's observations deliver about the state. Today's leaf is `I(y_τ ; s_τ | π)`
on the open-loop predictive, which counts at every step information earlier observations of
the same plan would already have delivered.

**Why this is exact and cheap on this path.** For a linear-Gaussian model the posterior
covariance does not depend on which observation arrives, only on the fact that one does, so the
whole outcome tree of sophisticated inference collapses to one covariance chain, the Kalman
recursion. One chain stands for the tree. Conditioning on the *expected* observation
`y = H μ + c` has zero innovation, so the chain's mean equals the predictive mean and only its
covariance moves.

**What this is and is not.** Sophisticated inference has two parts: beliefs that condition on
the outcomes expected along the plan, and a recursion in which the action after each outcome
is chosen knowing that outcome. This design is the first part under an open-loop plan: the
covariance half of sophisticated inference, exact for linear-Gaussian factors. It leaves out
actions that react to outcomes inside the horizon (the discrete sophisticated planner's
territory; a continuous version would branch over sigma points of the outcome and is a later
item, if a customer needs it) *(2026-10-19: its analytic special case for linear-Gaussian factors is
`closed-loop-planning.md`.)* and the mean-dependence of a noise map (D3, an extended-filter
grade approximation).

**A consequence worth stating.** With a constant noise the Kalman covariance recursion ignores
the controls, so even the conditional leaf is the same for every plan. D1 corrects the numbers;
plan dependence comes from D3 (noise that varies over the state) or D4 (observations only
sometimes available). D1 and D3 are therefore one design.

## D2 — The epistemic chain, and where it lives (SETTLED, Alex 2026-10-15)

**A System option.** `System(planning_belief="prior")` is today's behaviour; `"posterior"`
switches the epistemic chain on. A System option rather than a planner option, as the customer
guessed: it changes what the closures carry, and every registered planner should see the same
objective for the same System.

**What the chain carries.** Under `"posterior"`, each Gaussian-carried state factor's entry in
the planning belief tuple becomes a pair, `(predictive, epistemic)`, both seeded from the
current posterior. The planners pass belief tuples through opaquely (they call `transition_fn`,
`step_terms_fn` and `terminal_terms_fn` only), so none changes. `make_transition` steps the
predictive member as today and the epistemic member by predict then condition.
`make_step_terms` unwraps: the predictive members feed every existing consumer and
`StepContext.qs`; the epistemic members feed the linear-Gaussian info-gain leaf and a new
`StepContext.qs_epistemic` (``None`` under `"prior"`), so a custom row can read either chain.
The habit planner and `branch_fn` read the predictive member.

**Conditioning.** At step `τ`, for each linear-Gaussian modality whose observed factor is
Gaussian-carried and which is present at `τ` (D4): `condition(q_epi, H, c, R_chol, y = H μ + c)`
from `inference/linear_gaussian`, the Joseph form the filter already uses, with `R_chol` the
noise at that step's mean (D3). Several modalities over one factor condition in sequence. The
mean is unchanged by construction; the log-marginal is discarded.

**Not in the first build.** A mixture-carried factor, and a switching modality (one with
categorical mode parents): conditioning on an expected observation would move the mode
posterior by an amount that depends on the observation's value, so the expected posterior is a
genuine average over outcomes. These keep today's leaf on the predictive chain, and the
docstring of the leaf says which modalities took the epistemic chain. A later item is the
sigma-point average.

*As built.* The pair carries three beliefs: `predictive`; `epistemic`, predicted from the
previous step's conditioned belief, the belief before this step's observations; and
`conditioned`, after them; plus the step count. The leaf a Gaussian-carried factor takes is
not the channel-keyed `state_info_gain` (the selector mounts that only for a mixture-carried
factor) but the block-keyed entropy-lowering leaf, `−H[q(x)]` on the predictive belief; under
the epistemic chain that block leaf becomes the entropy drop `H[epistemic] − H[conditioned]`,
the information gain of all the factor's expected observations at the step
(`objectives.continuous.entropy_drop`), which telescopes over the horizon as D1 says. The
channel-keyed leaf reads the epistemic member where it is mounted. Nothing conditions the
seed, which is a posterior already: conditioning follows the predict inside the transition,
so the carried `conditioned` member is always this step's.

*(Note, 2026-10-18: superseded in part by per-variable-infogain.md V11 decisions 3 and 4. The
block leaf is gone: a Gaussian-carried factor's channels each carry a channel-keyed information
gain `I(y_m; x_S)`, read on the epistemic member, and the factor-keyed `observation_interaction`
holds what their sum miscounts, so the two together equal the entropy drop above. The chain-less
entropy `−H[q(x)]`, which was not information gain, moved to the opt-in `uncertainty` row,
charged as `H[q(x)]`.)*

*(Note, 2026-10-20: a linear-Gaussian channel whose mode parents are all action nodes, over a
Gaussian-carried factor, joins the chain by selection: `_conditioners` admits it, and the
conditioner takes its head and noise at the step's level of each node, which must be a point
(the delta kernel under a clamped level). A channel with a state-factor mode parent stays
outside the chain, as above. Design: action-node-modalities.md S3, S4.)*

## D3 — A noise that depends on the state (SETTLED, Alex 2026-10-15)

**The gap.** `LinearGaussianObservation.noise` is a point `R_chol`, or an inverse-Gamma or
inverse-Wishart posterior lowered to one, and every consumer reads `R_chol` as a constant. The
nearest existing route, a switching modality gated by a categorical, cannot say "precision
depends on where `x` is", because mode parents are categorical; the neural emission's
heteroscedastic head can, but it is not a linear-Gaussian map and is not what a filter
conditions on.

**The declaration.** `noise` admits a third kind: a *noise map*, a `NoiseMap` subclass with
`__call__(loc) -> R_chol` whose parameters are leaves of `SystemParams` (traced, so a new map
does not re-jit) and whose output is lower-triangular with a positive diagonal. The family
gains `R_chol_at(loc)`, which a point or conjugate noise answers with its constant and a map
answers by evaluation; `R_chol` stays as the constant lowering for the consumers that do not
have a location (and raises on a map, naming `R_chol_at`). The ship-with map is
`GridNoiseMap(grid, low, high)`: bilinear interpolation on a blurred grid over a box, the
customer's sky-occlusion map, smooth so the gradient planner sees it; the grid is a parameter
variable in the description (`"family": "GridNoiseMap"`, `parameters: {"grid": ...}`,
`options: {"low": ..., "high": ...}`), and a custom module that is not a shipped family cannot
be described, which `describe` refuses with a sentence. A map is a point: no prior, no
learning; a learnable map is a later item.

**Where the noise is evaluated.** At the mean of the belief the consumer holds: `infer_states`
at the prior mean before conditioning, the planning scan at the predicted mean of the step
(the predictive and epistemic means coincide under D2, so there is no ambiguity), a mixture
belief per component mean. This is the extended Kalman filter's convention: it ignores the
noise's variation across the belief's spread. Documented as the first version; an unscented
evaluation is the later item. Consumers that change: `predictive_obs`,
`conditional_obs_entropy`, `selected_obs_entropy`, `condition` at inference, the ambiguity
leaf (`E_q H[y|x]` is no longer constant in `x`; evaluated at the mean), and the evidence of a
tick.

**Gradients.** The gradient planner differentiates through the map, so the map must be
differentiable in `loc`; bilinear on a blurred grid is, almost everywhere. The acceptance check
compares `jax.grad` of the objective with finite differences through the map.

*As built.* `GridNoiseMap(values, low, high, axes)` in `hybrid_aif.factors`; the family's
`noise_map` capability and `R_chol_at(loc)`, with `R_chol` refusing a map by name; a map is
admitted on a non-switching modality over a Gaussian-carried factor and refused elsewhere with
the design pointer; the description writes it as `noise: {"values": <variable>}` with
`options.noise_map`. The acceptance detour needs a blurred grid: a precise node whose influence
ends one cell away has zero gradient on the straight path, so the planner never feels it, which
is the note's own caveat.
The map is axis-aligned with the state's coordinates; `values`, `low` and `high` are leaves
and only `axes` is static, so a map that lives in a frame moving with the agent (a world-fixed
sensor-quality map read by a model rebuilt each step in the body frame) is resampled onto a
fixed body-frame grid each step, and nothing re-jits. An optional rigid transform on the map,
a traced rotation and offset applied to the mean before indexing with its two variables in
the description's noise record, is the item to build if a second user needs it; the first
user's resampling route is sound.

## D4 — Per-step availability in planning (SETTLED, Alex 2026-10-15)

`infer_states` takes `present`; planning scores every modality at every step. The planner
context gains `present`, a mapping from modality name to a bool or a `(policy_len,)` bool array
(traced), defaulting to all present, passed through `infer_policies(present=)` and
`planner_context(present=)` with the same spelling `infer_states` uses. A modality absent at a
step is not conditioned on in the epistemic chain and its leaves at that step are zero (a
`where` on the step's term tree, so the tree's shape is static). Every modality on every path,
not only linear-Gaussian ones, since the mask is a `where` on leaves.

*As built.* The mask zeroes the leaves keyed by the modality's name and by its likelihood
slot. A Gaussian-carried factor's block leaf is keyed by the factor, so under `"prior"` its
`−H[q]` stands whatever the mask says, carrying no observation; under `"posterior"` the same
leaf is zero at an absent step by construction, since nothing was conditioned on. A float
mask is refused: availability is a bool or a bool array.

*Extension (Alex, 2026-10-21): presence under a control node's levels.* An entry of `present=`
may also be a mapping from one discrete control node's name to a bool per level,
`present={"answer": {"u_option": (False, True, False)}}`, read at each plan step at that step's
level of the node, as a conditional target's level is read (`C_deps`): the plan's decision, so a
declared actuator kernel on the node does not spread it. The mask is then exact per plan, which
is what a channel with an observation target needs when it is absent under some levels: the
noise-limit form of an absent Gaussian channel (`action-node-modalities.md`) is exact for an
undesignated channel under both decompositions, its ambiguity and risk cancelling at every absent
step, but a wide predictive distribution has a large divergence from any target, and a Gaussian
channel has no null symbol a target could give mass to. A per-level entry is resolved inside the
step and transition closures from the step's action, so the enumerating and tree planners and the
closed-loop pass, which reads its level sequences, agree with the per-step form for every clamped
plan; a probability-vector step is refused with a sentence by the step-terms closures on every
path and by the transition where it reads the step (the posterior chain, the closed-loop pass),
as a float mask is. `infer_states` and `replay` take the flag forms only and refuse the per-level
form naming the face (`learn` takes the positional tuple form). The mask is a select, not a
multiply, so a leaf that is not finite where the modality is read is an exact zero where it is
absent. Found on the same day: the tabular path dropped `present=` entirely (the mask reached
the hybrid closures only), which this extension fixes; D4 holds on every path as written.

## D5 — Targets are untouched; the cross-entropy form is declined as a fix (SETTLED, Alex 2026-10-15)

The customer's feature (4) observed that `∂KL(q ‖ C)/∂Σ = ½(Λ⁻¹ − Σ⁻¹)` is negative wherever `q`
is tighter than the target, so with the covariance plan-dependent a loose target would pay the
planner to stay uncertain, and proposed cross-entropy (expected-utility) state and terminal
targets, perhaps as the default.

The cross-entropy form is not a hack: it is the expected-utility half of the
utility-and-information-gain decomposition, which the library already uses for observation
targets. But it is not the fix either, because the trap comes from scoring the wrong belief.
Scoring a state KL on the future posterior rewards not having observed and double counts
against the information-gain leaf:

```
KL(q(s_τ|π,y) ‖ C) − IG_τ  =  −H[post] − E log C − (H[pred] − H[post])
                            =  −H[pred] − E log C  =  KL(q(s_τ|π) ‖ C)
```

up to the small difference in `E log C` between the two beliefs: the information gain cancels
and the plan-independent objective comes back, which is the symptom the customer started from.
Under D1 targets read the predictive chain only, where the KL's entropy is the breadth of
visitation and, through the identity above, the epistemic value expected free energy already
contains. The trap cannot arise, no default changes, and the KL semantics of a loose goal are
kept. Under an open-loop plan the predictive belief is where the agent will be given what it
knows, because observations change knowledge without changing the control sequence; the
distinction between the two chains is the correct bookkeeping, not a trick.

*(Note 2026-10-19: under `System(feedback="linear_quadratic")` a Gaussian target is scored on
the closed-loop state distribution, the future posterior plus the spread a linear-quadratic
correction leaves, by its cross-entropy and not a KL; `closed-loop-planning.md`. The rule above
stands for open-loop evaluation, which is what it addressed. The "small difference in E log C"
between the two beliefs, which an open-loop plan cannot act on, is the pragmatic value of knowing
once feedback is modelled. No KL is scored on the future posterior, so the cancellation shown
above does not arise.)*

## D6 — Custom rows on the linear-Gaussian path (SETTLED, Alex 2026-10-15)

Two fixes, no design: the linear-Gaussian `step_terms` passes `t` into `StepContext` instead of
discarding it, and the linear-Gaussian path's `Statics` carries `scoring` so a parametric row
mounts. With `StepContext.qs_epistemic` (D2) a caller can prototype a feature like D3 as a
registered row rather than a wrapper.

## D7 — API, summarised (SETTLED, Alex 2026-10-15)

- `System(planning_belief="prior" | "posterior")`, a static field; `"prior"` is today.
- `LinearGaussianObservation(noise=GridNoiseMap(values, low, high, axes))`, or any `NoiseMap`
  subclass; `R_chol_at(loc)` on the family; `GridNoiseMap` describable.
- `infer_policies(..., present=)` and `planner_context(..., present=)`: a mapping from modality
  name to a bool or a `(policy_len,)` bool array.
- `StepContext.qs_epistemic`; `t` and `scoring` on the linear-Gaussian path.

## Acceptance

- **Kalman parity.** A linear-Gaussian system under `planning_belief="posterior"` on a fixed
  plan: the epistemic chain's covariances equal the Kalman filter's posterior recursion step for
  step, and the info-gain leaves sum to `½ log det Σ_pred,T − ½ log det Σ_post,T` telescoped over
  the horizon. Under `"prior"` every number is bitwise today's.
- **Targets are chain-blind.** On the same system every target, utility, risk, ambiguity and
  terminal leaf is bitwise equal under the two settings; only the info-gain leaves differ.
- **A plan-dependent choice.** A point mass with a tight goal and a region off the direct path
  where a `GridNoiseMap` is precise: with info gain on, the gradient plan detours through the
  region; with info gain off, or with the region's noise equal to elsewhere, it does not.
- **Gradient check.** Finite differences of the objective with respect to the plan through the
  map match `jax.grad`.
- **Availability.** A modality absent at every step gives today's numbers; present only at
  step `k` gives one nonzero info-gain leaf, at `k`, and zero utility and ambiguity elsewhere.
- **The noise map at inference.** `infer_states` with a map evaluates it at the prior mean and
  matches a hand-written Kalman update with that `R`; the tick's evidence likewise.
- **Description.** A `GridNoiseMap` declaration round-trips bitwise; a custom module is refused
  by `describe` naming the shipped family.
- **A custom row.** A parametric row mounted on the linear-Gaussian path reads `t`, its
  `scoring` entry and `qs_epistemic`.
- **The trap, by construction.** A terminal target with a huge spread on one axis: under
  `"posterior"` the terminal leaf is bitwise the `"prior"` leaf for every plan.

## Out of scope

The closed-loop half of sophisticated inference on the continuous path (actions that react to
outcomes within the horizon; sigma-point branching); the epistemic chain for mixture-carried
factors and switching modalities (today's leaf, flagged); an unscented evaluation of a noise map
across the belief's spread; learnable noise maps; cross-entropy state and terminal targets as a
default (declined, D5; an opt-in form can be added when someone asks for the expected-utility
reading of a state preference and says why).
