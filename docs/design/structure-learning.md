# Design: online structure learning — levels that are born and merged

*Status: SETTLED 2026-10-14. S1–S8 walked with Alex (2026-10-13/14); build items listed per
section, the first build's scope under §Acceptance. Companion to `mixture-filtering.md` (the belief-side mixture, its budget and
schedule), `conditional-families.md` (the conditional core), `parameter-learning.md` D3/D6/D7/D8
(slots, novelty, forgetting, the LG conjugate families), `per-variable-infogain.md`,
`model-description-0.3.md` (sizes, domains). Source: AXIOM (Heins et al. 2025,
arXiv:2505.24784, §2.1, A.6–A.8, A.10, Algorithms 1–2); Friston et al., Bayesian model
reduction (arXiv:1805.07092). Roadmap: project-brief "Bayesian model reduction & online structure
learning".*

## Why

A declared model has fixed sizes: a categorical factor has its levels, a mixture its budget, a
table its shape. AXIOM's result on Gameworld comes from models that are not fixed: every mixture
starts small, adds a component when a datum is explained by none of the live ones better than by
an empty one, and merges components when the merged model's evidence is no lower. The two rules
are the same for all four of its modules, and the components they operate on are, in our
vocabulary, levels of a categorical variable with conjugate emissions. Nothing in that layer needs
pixels. The pixel layer (the slot mixture over tokens whose components are parametrised by latent
states, the presence and moving gates, the unused counter, the nearest-object relation) is a
separate arc, recorded as a roadmap row in §Out of scope, with no priority set (Alex,
2026-10-13).

The general-purpose test (Alex, 2026-10-13): expand the engine only where the expansion is of
more general use than AXIOM. Every item below is stated for an arbitrary categorical factor with
conjugate emissions, tabular ones included; AXIOM's modules are the worked instance.

## What was adjudicated before, quoted

- `mixture-filtering.md`, guidance: "Semantics live in factors, never in components. Mixture
  components are anonymous approximation slots." — the belief-side mixture keeps that; a learned
  component is a level.
- `mixture-filtering.md` D1: "Conditioning never changes K. Dead (padding) components stay
  dead." — the idiom a dead level follows.
- `conditional-families.md`, joint families: "a joint-Dirichlet (one Dirichlet on the product
  simplex; what BMR over joint counts consumes) is not the product of conditional-Dirichlets."
- `parameter-learning.md` D6: "novelty must track the learnable set"; continuous-parameter novelty
  is deferred, and `epistemic` naming such a slot raises.
- `parameter-learning.md` D8: per-mode statistics are responsibility-weighted, `ν_j += q(parent = j)`.
- `parameter-learning.md` O3: "no knowingly-incorrect update" (`InferenceDetail.pre_reduce`).
- `model-description-0.3.md` Q18.6: the `simplex` domain allows a zero entry; the `log_simplex`
  domain allows `-inf`; a row of zeros is a dead row.

## S1 — A learned mixture is a categorical factor with a conditional emission (SETTLED)

A mixture model of a variable `y` with `K` learned components is declared as a categorical
factor `z` with `K` levels and a Dirichlet prior on its weights, and a conditional emission of
`y` given `z` whose per-level parameters carry conjugate priors. The components are the
emission's per-level distributions, indexed by the levels; the mixture weights are `z`'s prior;
the component parameters are the emission's per-level parameters. This is the switching bank's existing stance applied to a learned emission, and it
holds for every emission family the table has: a tabular modality over `z` is a mixture of
categoricals, a Gaussian emission over `z` (S2) is a Gaussian mixture, several modalities sharing
the parent `z` are a product-form component (AXIOM's recurrent mixture, Eq. 30). A mixture's
output may be a latent as well as an observation: a continuous state factor with `z` in its
`B_deps` is the switching transition, already built.

Rejected: a self-contained mixture family with an internal component axis, per-component
parameters and the weights as one of its parameters, with no variable in the graph for the
component index (the belief-side mixture's anonymous slots, as a model). It would rebuild
weights, dead slots, novelty and the description on an axis the rest of the engine cannot see,
against the guidance quoted above.

What stays as is: the belief-side `MixtureGaussian`, its `mixture_k` budget and schedule, Runnalls
reduction. They represent a density over one continuous variable; they are not the object of
structure learning.

## S2 — The Gaussian emission over categorical parents only (SETTLED)

**What it is.** A Gaussian modality today is `y = H·x + c + noise`, a categorical parent, if any,
selecting which `H`, `c` and noise apply. With no continuous parent at all it is `y = c + noise`
with `c` and the noise covariance selected by `z`: a Gaussian mixture, `z` the component label.
No family is missing. A Gaussian with a Normal-Inverse-Wishart prior on its mean and covariance
is `LinearGaussianObservationMNIW` with an empty regressor block (head width one, the offset
column only), and its responsibility-weighted update (D8) is the mixture M-step.

**Gap.** `objectives/hybrid.py` refuses an LG modality whose `A_deps` holds no continuous
factor ("an LG modality observes one continuous factor (optionally followed by categorical
switching parents)"), and no message exists for the case "`y` observed, `z` unknown, no
continuous state", which is the responsibility computation.

**Build.** (i) The validation rule admits zero continuous factors; `_validate_lg_obs_shapes`
checks `head (*parent_shape, d_obs, 1)`. (ii) A message routine for the no-latent case: the
per-level score `ℓ_j(y)` feeds `q(z) ∝ prior(z) · exp ℓ_j(y)`, an exact categorical message
composed with the other modalities' messages as tabular evidence is today; no Kalman step, no
mixture belief, `mixture_k` not involved. (iii) The learn step's LG dispatch handles the empty
regressor block (`_lg_joint_stats` with `dx = 0`). (iv) A constructor alias so a declaration
reads as what it is (a Gaussian emission over a categorical parent), returning the MNIW family
with a width-one head; sugar, no duplicated arithmetic.

**The score.** `ℓ_j(y)` is the posterior predictive log-density of `y` under level `j`'s MNIW
posterior: a multivariate Student-t with `ν_j − d + 1` degrees of freedom, location the posterior
mean offset, scale from `Ψ_j` inflated by the offset's posterior variance. Rejected for this
modality: the point head (mean head, expected-precision noise), which the switching path uses and
keeps; the expected log-likelihood under the parameter posterior (AXIOM's coordinate-ascent form).
The predictive is the exact conjugate answer, it is the quantity S3's threshold and S4's merge
test are defined on, and no existing number moves because no existing path computes this case.
The three agree as a level's counts grow; they differ early, which is when S3 needs the honest one.
(Alex, 2026-10-14: "the constraints clearly point in the direction you suggest".)

**Worked instance.** AXIOM's identity mixture: `z = type` with `V` levels, `y = (colour, shape)`
a 5-D observation, one such modality.

## S3 — A categorical factor with a cap, dead levels and a growth step (SETTLED)

**Gap.** Static shapes fix a factor's level count at construction. The engine's idioms for a dead
entry exist piecemeal: a zero Dirichlet concentration masks a level's weight (`_wnorm`), the
support floor masks its evidence, a dead mixture component is a `-inf` weight. There is no traced
live count and no operation that wakes a level.

**Proposal.** A growable factor declares its cap `K_max` as its size and starts with `K_0` live
levels; the live set is the levels whose Dirichlet concentration is positive, so liveness is a
property of params, traced, carried by the learn step, and written by the description as a
zero/`-inf` weight entry (S6). A dead level's emission parameters sit at their prior. The growth
step is one operation of the learn face, run after the tick's posterior, per datum:

```
ℓ_j     = predictive log-density of y under live level j        (S2's score)
ℓ_empty = predictive log-density of y under the prior            (an empty level)
τ       = ℓ_empty + log α                                        (α: the growth propensity)
grow    = max_j ℓ_j < τ   and   live < K_max
```

On `grow`, the first dead level is woken: its concentration set to one count, its emission
statistics absorbed from `y` alone (AXIOM Algorithm 1, line 12: `κ ← 1`, `ν ← d + 2`, mean at the
datum, scale at the prior), and the tick's posterior over `z` recomputed with the new level
hard-assigned to `y` before the ordinary responsibility-weighted update runs. Without `grow`, the
ordinary update runs. The rule is the maximum-a-posteriori form of the Chinese-restaurant
assignment under a truncated stick-breaking prior `Dir(1, …, 1, α)`. The woken level is seeded
from the datum (Alex, 2026-10-14); left at the prior it would separate only slowly, and seeding
is what makes a single event a cluster.

**Static shapes.** `K_max` is the shape everywhere; `grow` is a `where` over params; the
hard-assignment is a `where` over the posterior. No recompilation, no shape change.

**Which data.** One form, Algorithm 1 over a batch of `N` data: `E` rounds, each scoring every
datum under every live level (a batch computation), taking the worst-explained datum and waking
a level if it falls below `τ`; the rounds are sequential because whether a second level is
needed depends on the first one woken, and `E` is small and static. The acting face (`replay`,
the driver's learn call) is `N = 1, E = 1`; a replay window is a batch (Alex, 2026-10-14: the
batch form is not deferred).

**`α`.** A parameter of the model as the tables and heads are, stored in params, on the
weights' Dirichlet beside its concentration anchor (S6): the pseudo-count the prior gives a
level not yet live. There is no conjugate update for a Dirichlet concentration, so it takes the
engine's idiom for a non-conjugate parameter: a point, clamped in the first build (AXIOM fixes it
per module), movable by a gradient step on the weights' evidence (`log_evidence`, S4) once that
step exists; the step is a general item, serving the Dirichlet anchors and the MNIW priors
equally, and is listed under §Out of scope. The point is the first build's form, not the
engine's stance (Alex, 2026-10-14): non-conjugate parameters are to carry real posteriors as a
general facility (a Gamma on `α`, or a Gaussian on its log by a Laplace step at the evidence
maximum, the same gradient with a curvature attached), so `α` is held where a posterior's
location would be, under the parameter's record with the anchor, and nothing in the growth rule
reads it other than through `log α`, which becomes the expected log under a posterior. Whether
this `α` ever needs its uncertainty is not the test; that the engine wants Laplace-form updates
for such cases is.

## S4 — Reduction: merging two levels by evidence (SETTLED)

**Gap.** No closed-form evidence function exists for any conjugate family (the survey found
`Dirichlet._log_beta` and the LG state-level `log_marginal` only). Merging two levels'
statistics is already what the families' natural coordinates allow.

**Proposal.** Two operations and one evidence function per family.
- `merge(j, k)`: the merged level's natural parameters are `η_j + η_k − η_prior` (prior mass
  counted once; AXIOM A.10), written into level `j`; level `k` returns to dead (concentration
  zero, emission at the prior). Dirichlet counts add; MNIW natural coordinates `(V⁻¹, MV⁻¹, Φ, ν)`
  add. Shapes unchanged.
- `log_evidence(family, posterior)`: the marginal likelihood of the statistics a posterior has
  absorbed, relative to its prior: for a Dirichlet the Dirichlet-multinomial ratio of
  log-Beta functions; for an MNIW the ratio of the matrix-normal inverse-Wishart normalisers.
  This is the function the zeroth-harness brief asks for as `log_evidence(document, traces)`
  (per address, `exact` flag) and the one the description-length work pairs with; it is built
  once here.
- Accept `merge(j, k)` when the reduced model's evidence is no lower than the full model's:
  `log_evidence(merged) ≥ log_evidence(j) + log_evidence(k)` on the retained statistics (Friston's
  reduction in closed form), on the emissions and the weights together.

**Candidate pairs.** Score pairs by mutual expected log-likelihood (each level's mean under the
other's predictive), keep the top `n_pair`, test greedily. Under static shapes: a fixed `n_pair`
and a `fori_loop`; pairs scored over `K_max × K_max` with dead rows masked.

**When.** Every `ΔT` learn calls (AXIOM: 500 frames), or on demand. The engine has no periodic
hook; `train/imagine.py`'s `_slow_update` keeps a traced counter in `LearnState.moments` and
fires on `updates % period == 0`. That becomes one primitive, a schedule: "every `n`-th call of
this face, run this operation", a period and a counter in the carry. Reduction is its first
customer (the learn face, the merge test). Not a customer: a hierarchical model's slow tier,
which under the tick semantics of `nesting-systems.md` (amendment 2026-10-14) is a sticky mind
that responds every tick, not one that skips ticks; a colleague's multi-timescale coupling (the slow
states setting the fast ones at each option and reading them back) is a modelling pattern on
that rule and belongs to the coupling row.

**Rejected in the first build.** AXIOM's acceptance test, an ELBO on ancestrally sampled data
scored through the downstream multinomials over reward and next switch. It is a task-specific
proxy for the evidence over the whole model; the closed form on the retained statistics is the
general one. If the proxy turns out to matter on the acceptance task, it returns as a scoring
option, not as the rule.

## S5 — The weights (SETTLED: nothing new)

The mixture weights are `z`'s Dirichlet prior. For an assignment variable drawn afresh each tick,
`z` is memoryless (`B_deps = ()`) with a learnable Dirichlet in `initial_state_prior`, which the
learn step already absorbs from the posterior each tick. To verify at build: the learn face
updates that slot on every tick, not only the first. For a persistent `z` (a type that does not
change), `z` carries a transition as any factor does.

## S6 — The description (SETTLED)

A growable variable's `size` is `K_max`. Its weights clamp carries the dead levels as zeros under
`simplex` or `-inf` under `log_simplex` (Q18.6 already admits both); the emission's per-level
arrays carry the prior at dead levels. The factor's cap is its size, so nothing new there. `α`
is a hyperparameter of the weights' Dirichlet, the `D` record of `z`: beside the concentration
anchor (`D_z_concentration`) a second field, written as any prior hyperparameter is, a clamp or
a learned variable (Alex, 2026-10-14). The reduction period is a schedule and goes under
`configuration.learning` with `frozen`, a stored default the learn step reads. To test at build:
a System with dead levels round-trips bitwise through `describe` and `build`; the bake-off style
reading test is not needed.

## S7 — Planning (SETTLED: nothing new)

Information gain over Dirichlet tables has its rows (`("param", likelihood/transition, DIRICHLET)`,
`wnorm`), which covers AXIOM's choice of what to be curious about: Eq. 57's information gain is
the KL on the recurrent mixture's Dirichlet posteriors only. The Gaussian emission's parameter
novelty stays deferred (D6); AXIOM uses none. A dead level contributes no novelty: its zero concentration is masked by `_wnorm` already.

## S8 — Recurrence: a lagged datum, and AXIOM's recurrent mixture (SETTLED)

The recurrent mixture is `z = s_rmm`, a per-tick assignment variable (S5), read by several
modalities sharing it: a Gaussian emission (S2) over the slot's previous continuous state and the
displacement to the nearest other slot, and tabular emissions over the previous type, the
previous interaction partner, the previous switch, the action and the reward. Its output to the
dynamics is `p(s_tmm | s_rmm)`, a conditional Dirichlet table, which makes `s_rmm` a parent of the
switch's prior. Everything in that sentence is S1–S5 plus existing families, except one thing.

**The lagged datum.** The assignment at tick `t` explains, as its data, values from tick
`t − 1`: the continuous state, the type, the switch, the action. The direction matters: they are
emissions of `z`, not its parents, since the component explains the tuple. A previous-tick latent
as a modality's datum is not in the engine, though the engine carries what it needs: the filter
carries the previous beliefs and `learn` takes `prev_beliefs`. Decided (Alex, 2026-10-14): a
modality's datum source may be the carry, a named factor at lag one, shape from that factor,
read at the filter step. A point datum first, the carried belief's mean (AXIOM treats its
filtered state as data); the expected score under the full belief is closed under the point head
and not under the Student-t, and is an option later. This is a general device (autoregressive
observation models, anything that explains what it just believed), not a stopgap.

Rejected: the discriminative direction, a conditional table from previous-tick parents to `z`,
which has no row for a continuous parent and loses the mixture reading; a deterministic lag copy
of the state with `z` emitting the copy, which gives the copy two factors (a copy and an
emission) that the one-transition-per-factor structure cannot hold. Not this device: one
System's state emitted to another System next tick across a blanket, which is the Env recast row
(`control-overhaul-build-plan.md`) and stays there.

First build: the recurrent mixture over features supplied by the driver, which is what an
acceptance task with object state given provides; the lagged datum when the state is inferred.

## Acceptance

Pins, each against the arithmetic it replaces or a reference it does not share code with:
- S2: the Student-t predictive against numerical integration of the Gaussian over the MNIW
  posterior on small cases; the categorical message against a NumPy mixture responsibility.
- S3: the growth step against a NumPy implementation of Algorithm 1 on a stream; the woken
  level's statistics bitwise.
- S4: `merge` against a refit on the pooled data (the conjugate update is exact, so the merged
  posterior equals the posterior of the union); `log_evidence` against brute-force enumeration
  for small Dirichlet cases and against a known closed form for the MNIW.
- End to end: a Gameworld-style task with object state given (Gameworld is permissively
  licensed; AXIOM's code is not read), the model growing from one level per factor, reward
  curve and final level counts reported; a second run with reduction off, to show what merging
  buys.

## Out of scope (roadmap rows)

- **The pixel layer** (arc 2, no priority set): a mixture likelihood whose component parameters
  are variables rather than clamps (the slot mixture over pixel tokens), presence and moving
  gates with proxy likelihoods, the unused counter, the nearest-object relation. Engine
  requirement if taken up: a likelihood whose per-component parameters are latent states.
- **CSCG tools** ([arXiv:2212.01508](https://arxiv.org/abs/2212.01508)): clone-structured graphs, Viterbi and Baum-Welch
  over them; later.
- **Non-conjugate parameters in the learn face**: a gradient step on the evidence
  (`log_evidence`, S4) moving a clamped hyperparameter, then the Laplace posterior around that
  maximum as the general facility for parameters without a conjugate form; `α` the first
  customer, the Dirichlet anchors and the MNIW priors the others; the description ruling 18.5e
  names the gap.
- **A colleague's multi-timescale coupling** (the set-and-read exchange between a slow and a fast
  tier), on the coupling row.
- AXIOM's sampled-ELBO acceptance (S4) as a scoring option; the expected score under a full
  carried belief for a lagged datum (S8).

## As-built notes (2026-10-15)

Built in one pass, S2, S3, S4 and S8, with the acceptance task under `examples/gameworld_growth.py`
and the pins under `tests/test_structure_learning.py`. What the build settled beyond the text
above, each a reading of the rulings rather than a change to them:

- **S2.** `gaussian_emission(loc, R_chol)` is the alias, returning the MNIW family with a
  width-one head; a point `LinearGaussianObservation` with an empty regressor block is admitted
  too and scores the Gaussian density. The emission joins the mean-field sweep as a one-row
  tensor of its scores with the observation index `0`, so its message composes with the tabular
  ones exactly; its scores are densities, so the support check (`zero-support-evidence.md` Z1)
  exempts it. Planning: an emission carries its own ambiguity leaf (the per-configuration noise
  entropy under the parents' beliefs) under `risk_ambiguity`; it is not a channel for state
  information gain (its mutual information with the parents has no closed form; AXIOM uses
  none), and a preference over its observation is refused (the predictive is a mixture over the
  parents' configurations). The environment draws it from the sampled parents. *(Superseded in
  part by per-variable-infogain.md V11 decision 7, 2026-10-18: the emission is now a channel for
  state information gain, `I(y; z)` with the predictive mixture's entropy a Kolchinsky–Tracey
  pair, and follows the `epistemic` selector like every channel.)*
- **S3, the weights.** For a memoryless factor the per-tick prior is its transition column
  `(K_max, 1)`, so the weights, the growth propensity and liveness live on that
  `ConditionalDirichlet` (`growth`), not on the initial prior as S5 wrote; the initial prior is a
  `Dirichlet` over `K_max` with the same `growth`, zeros at the dead levels, kept live as the
  weights are (a woken level's count copied into it, a merge pooling it), so an episode starts
  with the levels the weights hold. The description writes `growth` as a hyperparameter of the
  `Dirichlet` prior record on both, and a concentration with a zero per dead level under the
  `nonnegative` domain.
- **S3, the seed is the absorb.** AXIOM's Algorithm 1 seeds the woken component for its E-step
  and recomputes every component from the prior in the M-step; in the engine's incremental form
  the woken level is left at its prior, the datum hard-assigned to it, and the ordinary
  responsibility-weighted update seeds it from the datum alone (the prior absorbing one datum,
  `M` at the datum as the prior strength vanishes, `ν` up by one). A woken level's weight count
  is the growth propensity plus the datum: the prior pseudo-count the rule gives a level not yet
  live, so the evidence of the weights reads a woken level's data as nonnegative. A live level
  still at its prior scores what an empty one scores, so the rule compares them by `log α`
  alone, as the stick-breaking prior `Dir(1, …, 1, α)` it is the MAP form of does: at `α ≤ 1`
  the first datum goes to the live level and growth begins when live levels have moved; at
  `α > 1` the first datum wakes a level. A starting level declared at weight 1 then takes only
  soft counts, and a merge removing it buys little Dirichlet evidence (of the order of
  `log((A + N)/A)` at its pseudo-count `A = 1`), so it can outlast the data's need for it; a
  starting level declared at weight `α`, as if woken, does not (`examples/growing_mixture.py`). (Corrected 2026-10-16:
  this note first said the first datum always goes to the live level.)
- **S3, the batch form** (`System.grow(params, observations, factor, rounds)`): each round scores
  the window under the working copy, wakes the first dead level for the worst-explained datum
  while it falls below the threshold (seeded from that datum, which it takes hard), and absorbs
  the window into the pre-window params under the responsibilities the working copy assigns;
  the absorbed params are the next round's working copy. Restricted to the modalities over the
  factor alone; `learn` is the one-datum, one-round case.
- **S4, the evidence.** `log_evidence` scores a Dirichlet with dead levels over its live ones, a
  woken level's prior pseudo-count being `growth`. A merge pools the killed level's data (its
  statistics beyond its prior) into the kept level in every slot the factor is an axis of, in
  natural coordinates (AXIOM's `η_j + η_k − η_prior_k`), the killed level returning to zero on a
  child axis (the weights, the initial prior) and to its prior on a parent axis (the emissions).
  The merge test sums the evidence of the weights and the emissions; the factor's own initial
  prior mirrors the weights and is not scored twice. Candidate pairs: mutual expected
  log-likelihood over the modalities over the factor alone, the `n_pair` best tested greedily
  under a `fori_loop`. The schedule: `System(reduction_period=…, reduction_pairs=…)`, the counter
  in `LearnState.moments["reduction"]`, `learn(prior=…)` supplying what the evidence is relative
  to; `System.reduce(params, prior)` is the explicit call; `replay(learn=True)` passes its
  starting params as the prior.
- **S4, what the weights' term does.** Pooling two levels always raises the weights'
  Dirichlet-multinomial evidence by about the label entropy of the split (one level explains the
  assignments trivially), so a merge is accepted when the emissions lose less than that: a small
  level survives only while its emissions differ from the big one's by more than roughly
  `log(n/n_k) + 1` nats per datum. This is the full-model evidence S4 chose over AXIOM's
  sampled proxy; it merges more freely early and settles as counts grow.
- **S8.** `System(lagged=(("modality", "source"), …))`: the modality's datum is the source's
  previous value, a factor's carried belief mean (a Gaussian's location, a categorical's
  probability vector as a soft observation) or the previous action's level, read at the filter
  step: `infer_states(previous=…)`, `learn` from its `prev_beliefs`/`action`, `replay` from its
  carry (the first tick absent). An explicit observation wins over the carry. The description
  writes it as the likelihood record's `options.datum`, `"<source>[t-1]"`. The expected score
  under the full belief stays open, as the text says.
- **Acceptance.** The Gameworld-style task has object state given: a player and food on a
  plane, four moves, the reward the sign of the change in distance to the nearest food. The
  model's tuple is the direction to the nearest food (a Gaussian emission), the move and the
  reward (tabular emissions, sharp priors so that a level that has seen one move or reward
  explains another badly). The driver chooses the move by the model's own reward prediction
  (the posterior over the level for the tuple without its reward, the reward table under it),
  with an exploratory draw falling over the first half of the run; no planner path was added
  (S7). Over 2,000 ticks from one level (seed 0): without reduction the reward per window of
  250 rises from −0.03 to about 0.9 with 14 live levels; with a reduction every 250 learn calls
  it rises the same way with 7; with one every 100 calls it stays near zero with 6. The
  diagnosis is the weights' term above: the first live level, seeded while the levels were at
  their prior, is fat and mixed in move and reward, and a young sharp level is a cheap merge
  into it (the Gaussian and the mixed tables lose little, the weights gain the label entropy)
  until its own counts make the loss exceed the gain; a reduction on a mature model is
  harmless (greedy reward 0.28 before, 0.29 after, at 800 ticks). The schedule is therefore a
  real parameter of the method, and the acceptance records it rather than tuning the rule;
  AXIOM's sampled proxy, which scores only the reward and switch predictions, would reject
  those merges, and it stays the scoring option §Out of scope names.
- **Acceptance over three seeds (2026-10-16).** The schedule finding above is one seed's.
  Final-window reward per tick (live levels) over seeds 0, 1, 2: without reduction 0.56 after a
  peak of 0.90 (14), 0.94 (16), 0.00 (12); a reduction every 250 calls 0.90 (7), 0.89 (9), 0.07
  (7); every 100 calls 0.04 (6), 0.86 (7), 0.00 (3). Seed 2 learns under none of the three, so the task
  is fragile at this budget whatever the reduction does; every 100 calls fails one seed the
  other schedules pass, and learns on another. A reduction every 250 calls is the most stable
  setting seen, with about half the levels. Whether the schedule is a parameter of the method
  or the fragility showing through it is open; the diagnosis above holds for seed 0.
- **After review (2026-10-15).** Growth and reduction move with the weights: a tick where no
  modality over the factor is present grows nothing (with `α > 1` an all-absent tick would
  otherwise beat every live level), and the weights' rate gates both, a static zero freezing
  the factor's structure, a traced zero skipping it. A merge's statistics are the posterior's
  beyond the *model's* prior and a killed level returns to it, so `prior=` is the model's prior,
  not a window's starting params; `replay` takes `prior=` (default its starting params, right
  only when they are the prior) and seeds the counter's carry when a reduction is scheduled.
  What growth and reduction do not handle is refused at the growable factor: a factor it parents
  (a transition, an initial prior), a switching or neural modality over it, a point emission (its
  offsets are fixed, so a woken level is not seeded from its datum and a merge has no emission
  evidence to weigh against the weights', which always favour merging). A lagged modality may
  be left out of `observations`. The schedule is a reduction counter in
  `LearnState.moments`, not the general "every n-th call, run this operation" primitive S4
  named; `train/imagine.py`'s slow-target counter stays its own, and folding the two is a later
  item. Two readings stand as they are: emission scores enter the sweep tempered at `LOG_ZERO`
  like every log term, so a gross outlier ties the levels; and an initial-prior rate gated to
  tick 0 leaves the initial prior's counts behind the weights' (the woken levels are copied, the
  data is not), which only the start of the next episode reads.
- **What a zero concentration means (2026-10-16).** A Dirichlet over `K` levels needs every
  concentration positive; a zero is read as the limit `ε → 0`, the family on the simplex's face
  where that level's probability is zero. The engine computes that limit throughout: a count
  landing on the level gives `0 + n`, the limit of `ε + n`; the evidence scores a level still at
  zero over the face and a level data landed on at `−∞`, a structural zero contradicted; the
  expected log-probability is `−∞`, which `_wnorm` masks. Growth is not a limit but a change of
  model: a woken level's prior pseudo-count is `α`, not zero. A dead level of a growable factor
  is therefore a structural zero that `growth` may wake, and a zero without `growth` stays absent
  until data lands on it. The description admits a zero in any Dirichlet's concentration
  (domain `nonnegative`), with or without `growth`. The strictly positive alternative (an `ε`
  everywhere, dead levels by an explicit mask) was set aside: it buys orthodoxy at the cost of
  exact parity with hard-zero tables and a second encoding of liveness.

## Open: growth as inference, and the prior's reserve (2026-10-16, not adjudicated)

Two readings raised after the build, recorded for a later design round; neither changes S3.

**Growth as soft inference.** Nothing in the records makes a factor growable: `growth` is a
hyperparameter of the weights' prior, the pseudo-count it holds in reserve for a level that is
zero now, as the concentration of a Chinese-restaurant prior does. Without it (`α = 0`) a zero
level has prior probability zero and no datum reaches it. The growth step is that prior's
assignment rule under two approximations, both AXIOM's: a hard assignment, and one new level a
tick. `max ℓ < ℓ_empty + log α` compares an existing level with a new one at its prior
predictive. The soft form gives the first dead level the predictive mass `α / (N + α)` inside
the ordinary sweep: the ordinary update then seeds it, the datum shared rather than taken, and
the growth step stops being separate code. Only the frontier level gets the mass. Identical dead
levels would otherwise receive identical shares and learn identically, a symmetry the mean-field
posterior has no way to break. Open: whether the soft form grows as AXIOM's acceptance does,
and how the frontier's mass is read when inference, not learning, is the face called.

**Smoothing and growth as one reserve.** A Dirichlet's prior concentration is already its
smoothing: a scalar or per-level pseudo-count is written into the prior (`from_table(strength=)`
is the scalar's sugar), and the engine needs no smoothing mechanism of its own. Growth differs in
when the pseudo-count is in force: a smoothing count enters the predictive of every level all the
time; a growth count is held at a zero level until the level is woken. The two are
parameterisations of one prior mass on unseen levels. A symmetric `Dirichlet(α/K, …, α/K)`
converges to a Dirichlet process of concentration `α` as `K → ∞` (the finite approximation),
with total predictive mass on the unused levels near `α / (N + α)`: the same mass, spread over
every dead level rather than held at the frontier. A generalisation is a per-level reserve, the
pseudo-count each level receives when woken (`growth` an array, not a scalar): the frontier `α`
is one profile, `α/K` everywhere another (with the symmetry problem above), and a decreasing,
stick-breaking profile a third. Open: whether a per-level reserve earns its place over a
scalar, and whether a smoothing helper at construction (a pseudo-count added per level) is
wanted at all, given that a prior already writes it.
