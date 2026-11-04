# Recognition beside analytic factors, and a mind that acts once per option

*Status: DECIDED 2026-11-03 (Alex + Fable, walked over one session; revised the same day after an
adversarial review, R1 to R9 below). Build for 0.19.0. Amends `amortization.md` A11 (routing, and
the reading of a recognition entry on the analytic path) and `step-durations.md` (T9, the
held-action mind). A1, 18.5c and T1 to T8 are unchanged.*

## Why

Two promises of the brief are met today only on the neural state-space path: a frozen pretrained
encoder as a first-class sensor, and amortized inference for a variable. Any recognition entry
routes the whole System to that path, whose validator admits only deterministic carries,
categorical codes and predictor heads. A categorical or Gaussian factor inferred by the
structured rules cannot sit beside a recognized variable, although `capabilities.md` says it can
("the rest fall back to the structured row") and A11 says the same ("a variable without an entry
falls back to the structured/exact row"). The showcase model for the public launch needs exactly
that: a hidden terrain regime and a mixture-carried position, inferred analytically from a fix
and a cue, with a frozen encoder reading a camera image. This record says what the analytic path
does with a recognition entry, and what the engine does not do.

A second, smaller gap is in the driver: a two-tier model is two Systems with one clock each (T2),
and the driver calls every mind each round, so a slow mind that decides once per option has had
to be driven by a hand-written outer loop.

## What was settled before, quoted

**Transducers** (`project-brief.md`, "Frozen pretrained world models"): a frozen encoder "lives on
the System boundary, owned by the System"; "a transducer has no inference semantics. The engine
never inverts it, differentiates through it, or trains it (only its precision is learnable)";
"it arrives as an observation on a latent modality [...] and becomes a belief by uncertainty
injection through machinery we already have: fixed/learned precision on the pseudo-observation
node". The scope line: "no inference semantics for foreign models". The brief's three-tier
section states the hierarchical form of the same reading: "pseudo-observations upward,
empirical priors downward". `test_recognition_slot.py` records that "workstream H's transducers
are exactly this, a frozen module in this slot".

**A1** (`amortization.md`): `MLPConditional` is one role-free family whose contract is
"(i) conditional log-density evaluation, (ii) ancestral sampling, (iii) an inference strategy
named in the registry". The analytic path admits it in any slot; `infer_states` through a neural
likelihood is refused today, the refusal naming the two strategies the registry lacks:
structured filtering through the net (A1(a)) and an amortized recognition model (A1(b)).

**A11**: the recognition slot is "a fourth slot on `SystemParams`, `recognition={variable: module}`,
with `System` carrying only the structural declaration"; "each recognition entry's declared
semantics IS a q-side factor (child = variable, parents = conditioning set)"; "a variable without
an entry falls back to the structured/exact row"; the generic `infer(query, clamped)` layer (E) is
the recorded horizon, and the dict form is "E's restricted syntax, never a rival".

**18.5c** (`model-description-0.3.md`): a recognition entry is written under its variable with the
role label `recognition`, "the third partition beside the generative and objective labels";
"in a `recognition` record a latent parent denotes its prediction at that step"; "evaluation
order is derived from the explicit parents, not declared".

**T2** (`step-durations.md`): "Two tiers are two Systems; one clock per System." The record's
preamble: "The library has no clock. Every factor steps once per round".

## Decisions

**R1, routing.** A System takes the neural state-space path if and only if it declares a factor
the analytic engine cannot hold as a belief: a `Deterministic` carry or an `MLPCategorical` code.
A `PredictorHead` likelihood, a `LearnedVector` or `DerivedInitial` initial prior presuppose such a
factor and are refused on the analytic path with a sentence naming the family they need. A
recognition entry is not a routing trigger: it is admitted on both paths. A declaration holding
both kinds of factor is the mixed case E, refused with a sentence naming it.

*Why this test and no other.* The two engines differ in what a state is. The analytic engine
holds distributions updated by evidence in a fixed-point sweep with closed-form rules. The neural
engine is a recurrent state space in the RSSM shape: a carry is a value computed from the previous
round, a code's prior is a net of that value and its posterior is a recognition module, and
nothing iterates. A carry has no belief and no evidence update, so the analytic engine has no
object for it; a code has no closed-form empirical prior and its only children are heads. Heads
and initial families are consequences of those two, and recognition is meaningful on either
engine. Routing on "any neural family or any recognition entry", the rule the code has had, was
convenience, not a boundary. The one test that pins the old rule,
`test_a_half_amortized_declaration_fails_at_the_declaration`'s tabular case, changes with it.

**R2, admitted entries on the analytic path.** A recognition entry names a state factor or an
auxiliary variable. For a Gaussian or mixture factor the module is an `MLPConditional`, whose
output is a diagonal Gaussian over the factor's dimensions; for a categorical factor it is an
`MLPCategorical` or an `MLPPredictor` with a softmax, whose output is a probability vector over the
levels; an auxiliary variable's module is any of these, its output read by later entries. Parents
are `obs:<modality>`, `external:<name>` and `recognition:<name>`; an empty parent set is refused.
The evaluation order is derived from the parents, as 18.5c says, and a cycle is refused. Entries
are evaluated before the sweep, once per tick. A factor-valued parent, the prior's prediction at
the step under 18.5c's reading, is refused on the analytic path in this release: under R3 the
module's output is read as a measurement, and a measurement that reads the prior it is multiplied
into would count that prior twice. The existing A11 checks apply: no shared module objects, no
amortized variable as a parent of another's module.

**R3, the message: a measurement.** On the analytic path a recognition entry's output is read as
a measurement of its variable, the brief's pseudo-observation: a Gaussian output is an observation
of the factor through the identity map with the module's scale as the noise; a categorical output
is a likelihood row over the levels. The message is applied after every modality, conditioned by
`condition` for a Gaussian belief and by `condition_mixture_components` for a mixture, every
component updated and the component weights re-scored, and for a mixture with categorical mode
parents it sends the same mode evidence a non-switching linear-Gaussian modality over that factor
sends. The categorical row joins the tabular sweep as a support-exempt row the way a Gaussian
emission's predictive scores do (structure-learning.md S2), and is left out of the generative
free-energy call. The message's normaliser is recorded in the inference detail under the entry's
name and does not join the modality evidence terms.

Under `mixture_schedule="runnalls"` a recognition entry on a mixture-carried factor is refused
at validation, as a modality over it is: a cross-mode Runnalls merge destroys the per-mode evidence
the update needs, so the measurement is expressible only under `"mode_aligned"`.

*This is a change to A11's wording, stated plainly.* On the neural path the module's output is the
variable's posterior, because it is the only evidence a code has. On the analytic path it is a
measurement combined with the analytic evidence as a product. The two differ by the prior: a
module trained as a posterior under some training prior carries that prior in its output, and
multiplying it into the analytic prior would count a prior twice. The user therefore calibrates
the module as a likelihood: trained with a flat prior over the variable's range, as a renderer
sampled uniformly over a field gives, or with its training prior divided out. The record says so
where the entry is declared, and the capabilities entry says so. Under E this is a structured q
with one likelihood-type factor, so the restricted syntax still retracts nothing.

**R4, absence.** An entry whose observed parent is absent at a tick contributes nothing, the rule
a masked modality follows; the `present` argument covers the modalities recognition entries read.
Absence is a property of data, so a belief is never absent, and an external parent is present
whenever it is supplied.

*Amended 2026-11-04 (L2, L5 below): the plug-in neural term is replaced by the linearised reading.*

**R5, free energy and learning.** A neural likelihood on the analytic path is admitted in
`infer_states`: it adds nothing to the update, which is what "never inverted" means in code, and
contributes its term to the free energy. The analytic detail's free energy today is the tabular
free energy minus the linear-Gaussian log evidences, exact when q is the exact posterior. With a
message and a neural term it becomes

    F = F_analytic - log Z_message + E_q[log m(s)] - E_q[log p(o_neural | s)]

where `m` is the message and `Z` its normaliser, each closed form for a Gaussian belief, for each
mixture component and for a categorical, and the neural term is the expected log-likelihood
evaluated at the belief's point: a Gaussian's mean, a categorical's probability vector (the one-hot
convention, A1) and a mixture's mean, the evaluation `learn` already makes through `_belief_value`.
The plug-in is not the expectation, so the free energy is no longer a bound where a neural term is
present, and the replay's data part is flagged inexact (R7). For a categorical measurement the
tabular sweep is a joint fixed point, so its normaliser is taken against the sweep's posterior
without the row and its expectation under the final belief; the modality evidences read under a
belief that already carries the row keep that shift inside F_analytic. The Gaussian and mixture
measurements have an exact normaliser, being applied after every modality. Per-component evaluation
for a mixture is recorded as a follow-on, not built, since `learn` would have to change with it. A
recognition entry contributes no generative term and is paid for in the data part through the free
energy its measurement leaves. The module is frozen on the analytic path in this release: a nonzero
rate on `recognition:*` there is refused. Training it means minimising the free energy through the
sweep, amortized variational inference proper, recorded as a follow-on.

*Superseded 2026-11-04 (L4, L6 below): planning channels are removed; planning reads the likelihood linearised at the belief.*

**R6, planning.** Rollouts never evaluate a recognition module or a neural likelihood: the engine
has no closed-form pushforward of a belief through a net (A1). In planning, a neural modality's
observation model is its planning channel, `SystemParams.planning_channels[name]`, a
`LinearGaussianObservation` over the modality's parents, per level where a parent is categorical
(the switching observation form), or a `ConditionalCategorical` where every parent is categorical.
The channel is the modality's likelihood for every planning purpose: the epistemic rows, the
undesignated-channel risk, the cross-modality rows over a shared factor, and the expected
observation under `planning_belief="posterior"`, so eligibility and keys follow the modality's
without a second rule. A neural modality without a planning channel carries no row, as today.
The channel is the user's model of what the sensor will resolve, and the natural value for a
frozen encoder is its calibrated covariance. A state-dependent covariance, the noise-map form, is
a follow-on.

*Why not the linearised decoder.* A mixture of the decoder at each component's mean with a
Jacobian covariance is an estimator without a bracket guarantee and costs a covariance the size of
the image per component per step. It stays recorded as the alternative, flagged `derived=False`
if built. Sampling through the decoder is the test oracle.

*Amended 2026-11-04 (L6, L8 below): the planning role label goes with the channels.*

**R7, the document and the price.** `describe` writes every recognition entry under its variable
with the role label `recognition` and the module as a tree variable (18.5c, built for the neural
path, extended to the analytic one); `build` reads it back. A planning channel is written under its
modality beside the `likelihood` record with the role label `planning`, a new label of the objective
partition, with the likelihood record's shape; a categorical channel has no document form, the
modality's document variable being continuous, and `describe` refuses it by name. The description
length prices the generative partition: a neural likelihood as a net at the resolution term with the
inexact flag (description-length.md D3); a recognition entry and a planning channel are not in the
model part. With a replay, the data part carries an inexact flag where a neural term enters it (R5).
The general rule, stated once: the model part prices p, and q is paid for in the data part, since a
worse recognition model raises the replay's free energy.

**Composition.** `merge` carries recognition entries, external inputs and planning channels under
the component's prefix as it carries likelihood records, with parents renamed; `subsystem` slices
each to the component's own, and a composite on the analytic path with recognition entries is no
longer refused.

**R8, the held-action mind (T9).** `composition.driver.hold` wraps a `RespondFn`: on a decision
round it calls the inner mind, stores the action and reads its duration with `System.elapsed`; on
every other round it returns the stored action and counts down, with no inference and no planning,
so the slow mind's factors step once per option and the fast tier reads the level in force. The
count is host-side, since `run` is a Python loop; the wrapped carry is a named tuple of the inner
`MindCarry`, the remaining count and the held action, and the first round always decides. Durations
are in the preference's unit (T1); `hold` takes `ticks_per_unit`, default 1, and refuses a duration
whose product with it is not a whole number of rounds, since a mind held by the fast clock holds
whole rounds. A held System with a non-identity actuator kernel on any of its action nodes is
refused: the driver realises every action every round, against the once-per-choice draw, and T8
reads the duration off the realised level, so even a deterministic permutation kernel would hold a
level the mind did not choose. Holding on the realised level is a follow-on. In the composite form
the slow System owns no world process: its factors are beliefs and the world is a separate process,
or the wired form is used; otherwise the composite environment would apply the option's whole-step
kernel (T3) once per fast round. The option node carries no target and no score in the fast tier,
where it is read as a kernel or target parent, the state reading "is driving to the dock"; a cost
per option is the slow tier's. Observations the slow mind does not read mid-option are not lost to
the model: the fast tier reads them, which is T7's division. Alongside: `System._durations` checks
positivity by converting a boolean that is a tracer under any `jit`, although `elapsed` is
documented as jit-legal; the check reads the value concretely where it can and skips otherwise.

**R9, release.** 0.19.0. The capabilities entry records R1 to R8 and the calibration requirement
of R3; the sentence "the rest fall back to the structured row" in the inference section is
replaced by the R2/R3 statement; A11 gains a dated amendment pointing here; `step-durations.md`
gains T9 pointing here.

## Build

1. Router and validation (R1): `System._is_amortized` tests carries and codes only; the analytic
   validator refuses heads and neural initials by name; `validate_structure` unchanged; the one
   test named in R1 updated.
2. Neural modality admitted in the analytic `infer_states` (R5): the refusal lifted; no update;
   its free-energy term.
3. Analytic recognition (R2 to R4): a `recognize` pass in `faces/infer.py` evaluating entries in
   derived order on observations, externals and earlier entries; measurements applied last as R3
   says; detail carries their normalisers; presence honoured; `replay` and `learn` read the
   entries through the same pass; `predict_beliefs` untouched, since it never infers.
4. Free energy (R5): the formula in `_detail` and in `mean_field_free_energy`'s caller; the
   inexact flag on the replay's data part.
5. Planning channel (R6): the slot, validated against the modality's parents, substituted for the
   neural likelihood wherever planning reads `params.likelihood`; `describe`/`build` and the
   `planning` label (R7).
6. Composition: `merge` and `subsystem` for recognition entries, external inputs and planning
   channels.
7. Driver (R8): `hold`, the `_durations` guard, tests under both forms.
8. Docs (R9) and the notebook gate.

Acceptance: a System with a sticky categorical, a mode-aligned mixture-carried Gaussian, a tabular
cue, a linear-Gaussian fix, an image modality whose likelihood is a frozen decoder and a frozen
`MLPConditional` recognition entry for the Gaussian factor over the image: (a) with the recognition
entry removed and the image absent, every belief equals the current library's bit for bit;
(b) with the entry present, the beliefs equal those of the same System with the entry replaced
by a linear-Gaussian modality over the factor whose observation is the module's mean and whose
noise is the module's scale, to float tolerance, and the terrain belief likewise; (c) the detail's
free energy equals the hand-written formula of R5; (d) the R-AIF `tiny` System still routes
neural and its oracle tests pass unchanged; (e) `describe` round-trips the entry and the channel;
(f) `description_length` prices the decoder and nothing of the entry or the channel; (g) every
planning row of the image modality equals the row of its planning channel declared as an ordinary
modality; (h) the held mind re-decides only at option boundaries under both driver forms, and the
refusals of R8 fire.

Review before build: an adversarial read of this record against the code and the quoted records
(done once; the findings are folded in above). Review before commit: the usual.

## Follow-ons, recorded and not built

- A1(a): conditioning through a neural likelihood by linearisation, the extended Kalman row per
  component, and the linearised predictive as an alternative planning channel.
- The same-tick factor link ("cross-factor continuous parents", queued 2026-10-18): a continuous
  child of a continuous parent at the same tick, with the upward soft-observation message and the
  downward empirical prior resolved in the sweep. With it a free latent between an image and the
  state is declarable, and factor-valued recognition parents (R2) become admissible.
- Training a recognition module on the analytic path by the free energy through the sweep.
- Per-component evaluation of a neural term under a mixture belief, in F and in `learn` together.
- State-dependent planning channels, and a recognition module's own scale as the channel.
- Holding under a non-identity actuator kernel, on the realised level.
- The mixed case E: carries and codes beside analytic factors.

## Amendment 2026-11-04: a neural likelihood is a likelihood

*Status: DECIDED 2026-11-04 (Alex + Fable, walked over one session). Build for 0.19.0, before
the release commit. Supersedes R6, amends R5 and R7, and the sentence of A1 that read "never
inverted" on the analytic path.*

**Why.** As first built for 0.19.0, a neural likelihood on the analytic path was scored and
never used: not inverted in inference (R5's "adds nothing to the update"), and in planning
replaced by a planning channel (R6), a second observation model declared beside the
likelihood that nothing derived from it and nothing bounded. A target on a neural modality was
refused on the assumption that only the epistemic term matters for an image, which was never
put to Alex and is wrong: an image is a preference over pixels, a Gaussian target over the
modality's output like any other. Alex's ruling: a declared likelihood is used to infer and to
plan, whatever its family; a stand-in is not an approximation. The library already pushes
beliefs through networks and user-written maps on the transition side (A1's first-order
propagation; nonlinear-transitions.md N2's `first_order` and `unscented`), and an observation
map is the same object read the other way. The decisions below give the likelihood that
reading, remove the channels, and give the bracket machinery a word for the approximation.

**L1, the reading: linearised at the belief.** A neural likelihood `o = m(x, parents) + eps`,
`eps ~ N(0, diag sigma(x, parents)^2)` (an `MLPConditional`), over a Gaussian belief
`N(mu, Sigma)` on its one continuous parent `x`, is read as the affine model `o = H x + c +
eps'`, `eps' ~ N(0, R)`, by the declaration's static `propagation`, the same field and values
a `NonlinearGaussianTransition` carries, default `"first_order"`:

- `"first_order"`: `H = dm/dx` at `mu`, `c = m(mu) - H mu`, `R = diag sigma(mu)^2`. The
  extended Kalman reading; one evaluation and one Jacobian; exact for `m` affine in `x`.
- `"unscented"`: the `2d + 1` sigma points of the belief through `m`; `H = P_xy^T Sigma^-1`
  (the statistical linearisation), `c = y_bar - H mu`, `R = P_yy - H Sigma H^T` with the averaged
  noise `sum_i w_i diag sigma(x_i)^2` inside `P_yy`; exact for affine `m`, moment-matching to
  degree three otherwise, so `H mu + c` and `H Sigma H^T + R` are the sigma points' mean and
  covariance of `o`. `R` is a Schur complement of a positive semidefinite matrix and is
  symmetrised before its Cholesky factor is taken.

The other parents are held fixed: a categorical parent at a level is its one-hot, so a
modality with categorical parents is linearised once per level and takes the switching form
`(H_l, c_l, R_l)` the engine already conditions on and plans with; a mixture-carried factor is
linearised once per component at the component's mean (and, with mode parents, per level), the
per-component form a noise map already presents. A modality naming two continuous factors is
refused by name (not built). The helper is `inference/linearise.py`, `linearise(decl, loc,
chol, parents, index, propagation) -> (H, c, R_chol)`, vmapped by its callers.

*What the approximation is.* The belief's push through `m` is not Gaussian; both rules
replace it by the Gaussian with matched moments and differ in how the moments are found: a
tangent at the mean, or a fixed quadrature rule over a few points. The first-order error is the
curvature of `m` within the belief's spread, so a tight belief is read nearly exactly and a
wide one over a curved `m` is not; `unscented` is the rule to reach for there, as N2 says for a
heading that stays uncertain. Along a plan each step is linearised at its own predicted
belief, the nominal trajectory, and re-taken on every gradient iteration, as N2 does.

**L2, inference.** A neural modality no recognition entry covers (L3) is conditioned on by
`condition` (Gaussian-carried) or `_condition_mixture` (mixture-carried, per component) with
`(H, c, R_chol)` linearised at the belief the factor holds when the modality is reached (the
sequential extended Kalman reading: the tick's prior for the first modality over the factor,
the running posterior after an earlier one, so a mixture regrown by a switching modality is
read at its current components; identical for an affine network), after the tabular stage and
in modality order with the linear-Gaussian ones; a noise map is read at the same belief; its log evidence is the innovation density, the
same quantity an LG modality's is, and enters `F_analytic` and the detail's evidence terms in
the same way; `present=` masks it as it masks an LG modality. The plug-in term (R5's
`- E_q[log p(o_neural | s)]` at the belief's point, `neural_terms` in `infer.py`) is removed.
Validation admits a neural modality over a Gaussian-carried or a mixture-carried factor with
categorical mode parents: a state mode parent takes the switching rule, action-node parents the
selection rule (action-node-modalities.md S3: a point prior selects the level, a spread prior
is refused by name). Under `mixture_schedule="runnalls"` the modality over a mixture-carried
factor is refused as an LG one is.

**L3, a recognition entry replaces inversion, per modality.** A recognition entry replaces the
engine's inversion of exactly the modalities it covers: the `obs:` parents of the entry and,
transitively through `recognition:` parents, of the entries it reads. A covered modality,
analytic or neural, is not conditioned on by the engine; the entry's message conditions the
variable as R3 says (a message is a posterior under a flat prior; the calibration clause
stands). A modality no entry covers is inverted by the engine, exactly if analytic, by L1 if
neural. Nothing is counted twice and nothing is dropped. A covered modality still enters the
free energy, as `- E_q[log p(o | s)]` under the tick's final belief: for an LG modality the
Gaussian expectation of a quadratic, closed form; for a neural one the second-order expansion
at the belief's mean, `-log p(o | mu) + (1/2) tr(J^T R^-1 J Sigma)` per component, weighted by
the component weights; recorded in the detail under the modality's name. It is absent at a
tick when the modality is (R4). Acceptance: an entry whose message equals the exact LG message
of the modality it covers gives the exact posterior and the same free energy as the uncovered
model, bit for bit up to arithmetic order.

**L4, planning.** At every plan step a neural modality presents to the objective layer as the
LG model linearised at the belief the term reads: the epistemic chain's belief for the
chain's conditioning and the information-gain and ambiguity leaves (the pair the two
decompositions equate), the predictive belief for the risk and utility leaves and the
expected observation (the two coincide in the mean, so `first_order`
reads one model; `unscented` reads one per belief), per component for a mixture, per level for
categorical parents. A neural modality over categorical parents alone is the Gaussian emission
with `c_j = m(e_j)` and `R_j = diag sigma(e_j)^2` per parent configuration, read exactly and not
stamped approximate. A non-default `propagation` on an `MLPConditional` outside the
`likelihood` slot is refused by name. The `Conditioner`
gains the belief-dependent model (`model_at(q) -> (H, c, R_chol)`, the LG case returning its
constants and the noise-map case its `R_chol_at` read, as today), and every term reads it:
information gain, ambiguity, risk, utility, the cross-modality rows, the expected observation
under `planning_belief="posterior"`, `present=` per level, the closed-loop pass (which reads the
step's `H` as it reads the transition's `F`), the gradient planner (the Jacobian is taken
under the gradient; forward over reverse), the enumeration and sophisticated planners with no
change of structure. An `obs_targets` entry on a neural modality is admitted as on an LG one:
a Gaussian, MixtureGaussian or ConditionalGaussian over `R^child_dim`, checked against
`child_dim`; the refusal in `validate_structure` goes. The risk term reads the linearised
predictive, whose per-pixel variances are the diagonal of `H Sigma H^T + R`.

**L5, free energy and learning.** `F_analytic` keeps its structure, the tabular free energy
minus the log evidences, a linearised modality's among them; R5's message terms
(`- log Z_message + E_q[log m(s)]`) are unchanged; a covered modality contributes as L3 says;
the plug-in term is gone. R5's "the free energy is no longer a bound where a neural term is
present" becomes "is a bound on the linearised model", the `approximate` status of L7.
`learn` on the analytic path trains an `MLPConditional` likelihood by the gradient of that
free energy through the inference pass itself, so it takes the tick's prior beliefs as
`infer_states` does (`priors=`, the initial priors by default) and differentiates the reported
free energy with respect to the decoder (the plug-in objective in `learn.py` goes with the
term); the replay score reads the same free energy. The detail carries each covered
modality's term in its own field, `covered_terms`, keyed by modality. A recognition module stays frozen on the
analytic path (R5, unchanged).

**L6, planning channels removed.** `SystemParams.planning_channels`, `_planning_view`,
`_validate_planning_channels`, the gate in `faces/plan.py`, the carry in `merge` and
`subsystem`, the `planning` role label and its records in `describe` and `build`
(`OBJECTIVE_LABELS` loses it), and `tests/test_planning_channels.py` are removed; the field
list in `test_learn.py` and `test_scoring_slot.py` follows. A user who wants planning to read
a different model of a sensor declares that model as the likelihood.

**L7, the `approximate` status.** `Interval` gains a static `approximate: bool = False`
beside `derived`: `derived` says the enclosure is guaranteed on the model the planner was
handed; `approximate` says that model is a linearised or quadrature reading of the declared
one. A result is approximate iff any input is (OR, beside `derived`'s AND). The registry
stamps it per leaf from a static map of the System: a factor is approximate if its transition
is an `MLPConditional` or a `NonlinearGaussianTransition`, or any factor it reads through
`B_deps` is, transitively; a modality is approximate if its likelihood is an `MLPConditional`
or it observes an approximate factor; a leaf keyed by a factor or modality carries that
factor's or modality's status, and the total carries the OR. It is reported wherever
`derived` is: the term tree, the trace fields, the planner's report. The mixture reductions
stay unflagged, the Runnalls bound bracketing them. The replay's and description length's
"inexact" flag is the same status under its new name: the data part is approximate where a
linearised modality or transition enters it.

**L8, environment, description, composition.** `as_env` draws through the network, as
today. `describe` writes `propagation` with the `MLPConditional` likelihood record as it does
with a nonlinear transition's, and `build` reads it back; a target on a neural modality is
written as one on an LG modality. `merge` and `subsystem` carry a neural likelihood and its
target as they carry an LG one. The description length's pricing is unchanged.

**Follow-ons, recorded and not built.** The encoder-route information gain (imagine
observations through the decoder, invert each with the encoder, average the posterior
entropies), equal to L4's term when the encoder is Bayes' rule for the decoder: a
`derived=False` estimator and the calibration diagnostic, not a planner reading. A neural
modality over two continuous factors. A neural transition with discrete parents (unchanged).
State-dependent decoder noise beyond its value at the mean (the unscented rule averages it;
first order does not).

**Build.** Four chunks on disjoint files, then one adversarial review: (A) `linearise.py`'s
callers in inference and learning, L2, L3, L5; (B) the objective layer and the planners, L4;
(C) the `approximate` status, L7; (D) the removal and the surfaces, L6, L8. Acceptance: an
affine `MLPConditional` (depth 0, constant scale) over a Gaussian-carried and a
mixture-carried factor gives the LG path's posteriors, evidences, leaves and plans under both
propagations; a nonlinear decoder's posterior and predictive against Monte Carlo within the
N6 figures, first order and unscented; the covered-modality acceptance of L3; a target on a
neural modality moves the plan toward the preferred image; the two 0.19 showcases' scores
unchanged where no neural modality is declared; `describe`/`build` round-trip `propagation`
and the target; `learn` lowers the free energy of a decoder; the status is `approximate`
exactly on the leaves L7 names and nowhere else; the demo notebook re-executes.
