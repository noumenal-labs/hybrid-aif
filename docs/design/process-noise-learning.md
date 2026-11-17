# Learnable process noise

*Status: design, 2026-11-17 (Alex + Claude); build queued behind the fused-path build of
`parameters-within-the-plan.md`. Customer: a researcher's port of a predictive-coding toy
(REBUS/SEBUS, gist agourakis82/e8d7b64959eaa1a2a926950b3959dea8), whose "weaker prior precision" is
a larger process noise Q, and who could learn the noise of the observation model but had to declare
the noise of the transition. Companion: `parameter-learning.md` D8 (the observation noise families
and their statistic, which this mirrors).*

## What is wanted

A linear-Gaussian observation's noise learns: `LinearGaussianObservation.noise` may be an
`InverseGamma` or `InverseWishart` prior, the learn face absorbs each tick's expected residual
scatter `E_q[(y − Hx − c)(y − Hx − c)ᵀ]` with the `HΣHᵀ` term carried in full (D8), the rate key is
`likelihood:<m>:noise`, the evidence has a closed form, and `describe` prices and round-trips the
prior. A linear-Gaussian transition's innovations are a point `ConditionalGaussian`: `F` and `Q`
are declared, never learned. The conjugate update for `Q` is the same family and the same shape of
statistic; what differs is that the residual `x' − F x − G u − b` is between two latents, so its
expected scatter needs the pairwise posterior over `(x, x')`, cross-covariance included.

## D1 — The declaration mirrors the observation side

`LinearGaussianTransition` and `NonlinearGaussianTransition` gain an optional field

    noise_prior: InverseGamma | InverseWishart | None = None

one prior over the innovations' covariance, shared across modes, or batched over the innovations'
`parent_shape` (one prior per mode, as `innovations.chol` is per mode). When set, the innovations'
`chol` is lowered from the prior's `point()` (`chol(Ψ/ν)`, the expected-precision substitution the
observation side uses) at every consumer: prediction, planning, `as_env`, the closed loop, the
description's point form. The declared `innovations.chol` is then ignored and `describe` says so;
a builder may want the declaration to require `chol=None` there to make the single source of
truth explicit (builder's choice, recorded). `learnable_noise` is true exactly when the prior is
set; the rate key is `transition:<factor>:noise` in the existing grammar, and `lr_by_slot` freezes
or rates it as it does the observation's.

## D2 — The statistic is the exact pairwise residual scatter

At a learn step the face holds the posterior at `t − 1` (`prev_beliefs`, filtered), the posterior
at `t` (`beliefs`), the action taken and the declaration. For a Gaussian-carried factor with the
linear map `x' = F x + G u + b + ε` under the taken control, the one-step smoother gives the
pairwise posterior from these alone:

    μ⁻ = F μ + G u + b,   Σ⁻ = F Σ Fᵀ + Q          (the prediction, Q the current point)
    J  = Σ Fᵀ (Σ⁻)⁻¹                                (the smoother gain)
    μₛ = μ + J (μ' − μ⁻),   Σₛ = Σ + J (Σ' − Σ⁻) Jᵀ   (x given y up to t)
    C  = Σ' Jᵀ                                      (Cov(x', x) given y up to t)
    S  = (μ' − F μₛ − G u − b)(…)ᵀ + Σ' − F Cᵀ − C Fᵀ + F Σₛ Fᵀ

`S` is `E_q[(x' − F x − G u − b)(…)ᵀ]` under the pairwise posterior, the statistic the inverse-
Wishart absorbs with count one (the EM M-step for `Q`, exact for linear-Gaussian filtering).
Dropping the cross term `C` and using the two marginals would overstate `Q`; the design takes
the pairwise form and a test pins the difference. An `InverseGamma` takes the diagonal of `S`.

- A discrete control node selects the mode's slice: the taken level's one-hot is the
  responsibility, as the Dirichlet transition's control axis and the emission's parent levels are
  read; under a belief-valued step, its probability vector.
- A nonlinear map is linearised at `μ` (the previous posterior mean), `F` its Jacobian and `b`
  the remainder, consistent with first-order propagation; the unscented rule is refused in the
  first build (the pairwise moments under sigma points are a different computation).
- A mixture-carried factor is refused with a sentence in the first build (the pairwise posterior
  per component is lost at the reduce, the O3 loss of `parameter-learning.md`); a factor with
  discrete state parents is therefore out, control modes are in.

## D3 — Everything else is the observation side's

Evidence: the inverse-Wishart / inverse-Gamma closed form of `learning/evidence.py`, keyed as the
rates are. Description: the prior written and read as the emission's noise prior is, the schema
extended, the point form lowered for a reader without the prior. Description length: priced as the
emission's noise prior. Novelty: none in this build; selecting `transition:<f>:noise` under
`epistemic` is refused with a sentence (the expected information about `Q` from a planned step is
the transition-side twin of the D10 estimator and can follow). `plan_learning`: not a table, and
the existing "not a learnable table" refusal stands.

## D4 — The noise is a parameter variable (the shared-parameters arc)

`shared-parameters.md` (amendment of 2026-11-05) makes a parameter a named variable in
`SystemParams.parameters` read by `Parameter(name)`, builds tabular homes first and says later
steps generalise the home. A noise prior is such a variable, so this build adds the noise home
kind beside `TabularHome`:

- `SystemParams.parameters` may hold an `InverseGamma` or `InverseWishart`; `noise_prior` on a
  transition, and `noise` on a `LinearGaussianObservation`, may be `Parameter(name)` reading it.
  Several factors or modalities may read one noise; the learn face sums the readers' statistics
  and updates the home once, as it does for a table (the S2 ruling); the rate key of a home is
  `parameter:<name>` for every reader, the reader's own key for one.
- The inline prior (a field on the declaration) stays and is the implicit single-reader home,
  the status an inline table has today. The reference form is what sharing and the description's
  variable record need.
- A reference here sits inside a declaration (one field), where today's references replace a
  whole `transition` or `likelihood` entry: `keyed.params_positional` resolves the field, the
  description writes the reader with the reference and the variable with its record, and
  `_refuse_aliases` treats one prior object under two readers as it treats a table.
- The observation side gains the reference in the same move, so that R and Q are one mechanism:
  a noise variable read by a likelihood or by a transition.

## What to pin

1. The statistic against a hand-rolled lag-one smoother on a two-dimensional model, exact.
2. Recovery: a world with a known `Q`, filtered and learned over `T` ticks from a broad prior,
   the posterior mean within tolerance; the marginal-only statistic recovers a larger `Q`.
3. The control slice (one-hot per mode) and a `G u` input in the residual.
4. An affine `f` with `noise_prior` equals the linear case bitwise at first order.
5. `replay(..., learn=True, lr_by_slot={"transition:z:noise": 1.0})` on a sixteen-dimensional
   model of the customer's shape, started from a scaled `Q`, moves toward the generating `Q`.
6. Description round trip and schema; description length; evidence.
7. Refusals: mixture-carried factor, unscented, the epistemic key, `plan_learning` on it.
9. The home: two factors reading one `Parameter` noise learn one posterior from the sum of their
   statistics; a likelihood and a transition sharing one noise variable; the description records
   the variable once with two readers; an aliased prior object under two readers is refused.
8. Every existing test, with the notebook gate (the declaration changes a public field set).

## As built (2026-11-16)

Built on `switching-transition`; pinned by `tests/test_process_noise.py` (items 1 to 7 and 9, plus
the lowering, the rate keys, the carry, `merge`/`subsystem`/`write_back` and learning under `jit`).
The judgement calls the record left open, and the ones the build met:

- **The declared `innovations.chol` is kept and ignored, not required to be `None`.** A
  `ConditionalGaussian` reads its parent shape from `loc` and checks `chol` against it, so a
  `chol=None` would need a change of that class everywhere it is read. With a prior set the
  declared factor keeps only its shape (the prior's leading axes must be none or
  `innovations.parent_shape`, its dimension the innovations'); `describe` writes no value for it
  (the innovations' `chol` is a learned variable under the prior's record), and `build` fills it
  with the prior's point. A rebuilt declaration therefore differs from the original in that
  ignored leaf only; the documents are equal.
- **The lowering is at the consumers, not at construction.** `factors.transition_point(decl)`
  returns the declaration with `chol` set to the prior's point and the prior dropped (as
  `lg_point` does for an observation); `_lg_inputs`, `_nl_inputs`, `_with_process_noise` and the
  generative process read through it, which covers prediction, every planner, the closed loop
  and `as_env`. `as_env` lowers through `_lower_learnable` (the point, or one draw per noise
  variable under `param_key`, the readers of one variable sharing it). A `Parameter` left
  unresolved raises at `transition_point` and at `LinearGaussianObservation.R_chol`, naming the
  faces as the resolver.
- **A field reference is recorded apart from a whole-entry one.** `SystemParams.noise_references`
  (static, `(slot, index, name)`, empty in the keyed form) beside `references`, so no existing
  reader of `references` sees a field reader as a whole-entry reader. `params_positional`
  resolves the field and checks the variable against the reader (a noise family, the reader's
  dimension, the transition's mode axes); `params_keyed` puts the token back. A whole entry
  naming a noise variable, and a field naming a table, are refused with a sentence. The inline
  prior is the implicit single-reader home, as designed.
- **The statistic enumerates the modes statically.** For each configuration of the discrete
  control nodes, `F_j`, `b_j + G_j u` and `Q_j` are read by the same `_lg_inputs` / `_nl_inputs`
  the prediction uses, the scatter is `conjugate_stats.pairwise_scatter` (the RTS one-step
  smoother, `Q` the current point), and the configuration's responsibility weights it: the taken
  level's one-hot, or a `Categorical` / probability-vector entry's probabilities (the
  belief-valued step; reachable by the statistic, not through `learn`'s keyed action, which takes
  levels, so it is unpinned). A shared prior sums the modes' weighted scatter and mass; a
  per-mode prior takes them per mode. `noise_scale="per_coordinate"` is supported, beyond the
  record: the smoother runs with `diag(s) Q diag(s)` at the previous mean and the scatter is
  standardised by `s` there, so the prior stays over the unscaled `Q`.
- **The refusals.** Mixture-carried (by the factor's family or a non-Gaussian belief),
  `propagation="unscented"` and a belief over a continuous control input at learn time are
  `NotImplementedError`s naming the rate key that freezes the noise. Under `epistemic` the noise
  is refused with the D3 sentence whether named (`transition:<f>`, `transition:<f>:noise`) or
  reached by `True`, as an observation noise without an estimator is. `plan_learning`'s existing
  refusal stands, for the reader key and for `parameter:<name>`. `durations.compose` refuses a
  learned noise (the composed `Q_n` is no conjugate variable) and names `transition_point`.
- **`replay` takes `lr_by_slot=`**, passed to every tick's learn step (pin 5 calls it; there was
  no such argument).
- **Rates.** A reader of a noise variable answers to `parameter:<name>`; its own
  `<slot>:<name>:noise` (and its group key when the noise is its only posterior) takes only a
  static `0.0`, which keeps its statistic out of the sum, as a table reader's does.
- **Evidence and pricing.** `log_evidence` and `absorbed_count` read a transition's prior as the
  observation's; `evidence_by_slot` now also reports every parameter variable both params hold
  under `parameter:<name>` (tables included). The description length prices a transition's
  noise prior with the emission's links (`mdl._noise_links`: its count, the reference prior's
  Fisher-Rao room, the KL from the reference), a noise variable once under `parameter:<name>` at
  its readers' summed resolution, and a reader of one without its noise; under a learning replay
  a transition whose noise learned keeps its other parts' family link unpaid.
- **The document.** A noise variable's record is `parameters[<name>] = {"family":
  "InverseWishart" | "InverseGamma", "parameters": {<hyperparameter>: <clamped variable>}}`; a
  reader writes `{"parameter": <name>}` in its field (`noise`, `innovations.chol`); the schema
  gains `$defs/reference`. A learned head beside a noise variable is refused by `build`.
- **Composition.** `merge` namespaces noise variables and `Identify` joins two; `subsystem`
  carries the ones a view's readers read and `write_back` writes them back. A binding that
  rewrites a control axis of a reader of a per-mode noise variable is refused. `mirror` without
  lowering keeps a transition's reference; a likelihood's becomes an emission kernel holding the
  variable's prior as it stands (no face learns an emission). `carry` carries the noise block
  (`transition:<f>:noise`) and a noise variable whole.
