# Design: nonlinear transitions, propagated along the plan

*Status: SETTLED 2026-10-17 (Alex), N1–N6; built the same day, see "As built". Customer: the drive agent of a
separate, Noumenal-internal repository, whose note of 2026-10-17 reported a failure that this
design removes; it is restated under "Why" so this document stands alone. Companions:
`hybrid-system.md` (the linear-Gaussian and mixture paths), `amortization.md` A1 (first-order
moment propagation through a neural transition, the precedent), `mixture-filtering.md`
(mixture-carried factors), `per-variable-infogain.md` V10 (positive process noise),
`model-description-0.3.md` (descriptions). Sources: Julier & Uhlmann 2004, *Unscented filtering
and nonlinear estimation*; Petersen, Hoffmann & Rostalski 2018, *On approximate nonlinear
Gaussian message passing on factor graphs* ([arXiv:1903.09136](https://arxiv.org/abs/1903.09136): the forward rule as
quadrature moment matching, and an RTS-type backward rule); Bar-Shalom, Li & Kirubarajan
2001 (the extended Kalman filter);
Li & Todorov 2004 (iLQR: linearising about the nominal trajectory).*

## Why

A continuous transition is linear-Gaussian, `x' = F x + G u + b + ε`, or neural. A robot whose
dynamics are nonlinear in its state, a unicycle `x' = x + h v cos θ, y' = y + h v sin θ, θ' =
θ + h ω`, must be linearised by the caller, once, before planning. The customer linearises
about the current heading each step and rebuilds the params in a frame centred on the robot.
Every step of a plan then uses that one linearisation. When the target is behind the robot the
plan needs a turn of about 180°, which the frozen linearisation predicts as a large sideways
move; a lane preference charges for that move more than the heading preference charges for
facing away, so the planner drives straight on, away from the target. In a pilot the robot drove
40 m off its route. The customer works around it with a creep speed until the robot faces the
target, but the predicted state is wrong wherever a plan turns far from the current heading,
route corners and the information-gain terms along a turning plan included.

Measured on the unicycle (h = 0.1 s, v = 0.5 m/s, a 180° turn over 2 s, final position against
200,000 Monte Carlo draws): the frozen linearisation is 1.3 m off whatever the heading
uncertainty; propagating through the true dynamics at each step's own belief is 0.001 m off at
a heading sd of 0.05 rad, 0.013 m at 0.2 and 0.075 m at 0.5 to first order, and within 0.002 m
throughout by the unscented transform.

## N1 — the declaration (Alex)

`NonlinearGaussianTransition(f, innovations, propagation="first_order")`. `f` is an equinox
module the user writes, `class Unicycle(eqx.Module): h: float; def __call__(self, x,
controls): ...`, returning the mean of the next state; its numeric fields are pytree leaves, so
they jit, vmap and carry, and its code is static structure, as `MLPConditional`'s is. The
process noise `innovations` is a `ConditionalGaussian` as on the linear path, its parent axes
the factor's discrete control parents, its factor's diagonal strictly positive (V10). A plain
function with a separate parameter pytree was set aside: a second convention beside the
modules the library already uses.

## N2 — the approximation (Alex)

`propagation` is static, `"first_order"` (the default) or `"unscented"`.

- **First order.** `μ' = f(μ, u)`, `Σ' = J_x Σ J_xᵀ + Q` with `J_x = ∂f/∂x` at the belief's
  mean and the step's control. Exact for `f` affine in `x`, as `_mlp_step` is for a neural
  transition, and cheap: one evaluation and one Jacobian.
- **Unscented.** `2n + 1` sigma points of the belief through `f` (`n` the state's dimension,
  `distributions/gaussian.py`'s `sigma_points`), their weighted mean and covariance, plus `Q`.

The unscented transform is one quadrature rule among several (cubature, Gauss–Hermite, sparse
grids) that approximate the belief's push through `f` by a weighted sum over points and
moment-match the result (Petersen et al.); with α = 1, κ = 0 its points and mean weights are
the spherical-radial cubature rule's, every weight non-negative. A further rule would be a
further value of `propagation`, not new machinery. These degree-3 rules miss the covariance of
an `f` quadratic in its inputs, a product of two uncertain quantities included (an uncertain
speed times the cosine of an uncertain heading); that is the regime where a higher-order rule
would earn its cost.

Each step is linearised (or sigma-pointed) at its own belief, which along a planned rollout is
the plan's nominal trajectory; the gradient planner recomputes the rollout on every iteration,
so the linearisation is re-taken there too, the iLQR pattern. First order is the default for
its cost and its agreement with the neural path; unscented is for agents whose heading, or
other nonlinearly entering coordinate, stays uncertain, where first order mis-shapes the spread
(at a heading sd of 0.5 rad its sideways sd was 0.112 m against Monte Carlo's 0.150 and the
unscented 0.118). The information-gain and ambiguity leaves read that spread.

## N3 — controls (Alex)

- **Signature.** `f(x, controls)`, `controls` a tuple with one entry per control parent of the
  factor, in declared order.
- **A discrete control arrives as its integer level.** Every transition receives the level: the
  neural path builds its one-hot inside the step because a network takes a vector, and a
  user's `f` indexes by the level itself. Within a rollout a discrete control is one level per
  policy step, never a belief; uncertainty over it lives in the policy posterior.
  *2026-11-06: a held control node (`nesting-and-coupling.md` N7) supplied with `stats`, a
  probability vector over levels (N8), cannot reach `f` as a level, and a nonlinear `f` has no table
  to average over the vector, so such a wire into a nonlinear reader is refused naming `sample` or
  `mode`, N8's last case.*
- **A continuous control carried as a belief** (the gradient planner) enters as on the linear
  path, generalised: to first order its spread through the Jacobian in `u`, `J_u S J_uᵀ`, added;
  unscented, the sigma points are drawn over the joint `(x, u)`. A truncated control belief
  enters through its mean and covariance.

## N4 — scope (Alex)

A Gaussian-carried factor, and a mixture-carried one whose mixture comes from its prior: each
component is propagated through the same `f`, then the mixture is reduced as on the linear
path. A nonlinear transition with discrete *state* parents (switching dynamics) is refused: it
would need `f` to take the mode, or one `f` per mode, an API with no consumer yet.
*2026-11-15: built, with `f` taking the mode, for a consumer (the survey rover's position
switched by its terrain, `terrain-field.md`); see "Amendment 2026-11-15" below. The refusal
above stood until then.*

## N5 — the surface (Alex)

- **Descriptions.** A model description is data and `f` is code. A registry of named
  transition functions, `register_transition("<package>.<name>", cls)`, lets a document record
  the name and `f`'s numeric fields, and `build` look the name up; a document can then refer
  only to code the reading process installed, and nothing in it is executed. Names are
  namespaced so two packages cannot collide silently, and a registration records `f`'s field
  schema (names and shapes), which `build` checks a document against, so a document written for
  an older `f` fails rather than rebuilding with code that means something else. *Sequenced
  (Alex): the registry is the next commit, the same release; until then `describe` refuses a
  nonlinear transition, naming `register_transition`.* Describing Python code compositionally
  or quoting its source was set aside: Python is already the compositional language for `f`,
  and a document must not carry code.
- **The generative process.** `as_env` samples `x' = f(x, u) + ε`.
- **Validation.** `f` is checked at validation (as built: the control layout is the System's)
  with `jax.eval_shape`: it must return the
  factor's dimension for the factor's control layout.
- **Learning.** `f`'s fields are not learned: it is not `Learnable`, so `learn` and
  `System.carry` leave it alone, as they do a frozen neural transition. Fitting `f` is its own
  feature.
- **Description length.** `f`'s numeric fields count as parameters, point values with no
  prior, as other point declarations are counted.
- **Composition.** `merge` carries the declaration through; a binding that would slice `f`'s
  inputs (remove a control node it reads) is refused, since `f`'s signature is fixed code. Under `couple` nothing is sliced: a control node `f` reads may be held and supplied by a wire (`nesting-and-coupling.md` N7), with the level arriving as from a planner.

## N6 — acceptance

- An affine `f` equal to a linear-Gaussian transition gives the linear path's beliefs and leaves,
  under both propagations (exact for affine `f`), Gaussian- and mixture-carried.
- The unicycle's 180° turn: final mean and covariance against Monte Carlo within the figures
  above, first order and unscented; the frozen linearisation's error reproduced for contrast.
- The customer's failure, reproduced as a planning test: with a lane preference and a target
  behind the robot, the plan turns toward the target instead of driving away.
- A belief-valued continuous control: first order adds `J_u S J_uᵀ` (checked by hand on an `f`
  linear in `u`); unscented against Monte Carlo over `(x, u)`.
- Discrete control levels reach `f` as integers; every planner (enumeration, gradient,
  sophisticated) runs on a nonlinear factor, gradients finite.
- Refusals: switching dynamics; `f` returning the wrong shape; a zero process noise; `describe`
  until the registry lands; a merge binding that slices `f`'s inputs.
- `as_env` samples through `f`; description length counts `f`'s fields; `learn` and `carry`
  leave `f` alone.

## As built (2026-10-17)

`tests/test_nonlinear_transition.py` holds N6.

- **Declaration** (`factors.py`, exported from `hybrid_aif`):
  `NonlinearGaussianTransition(f, innovations, propagation="first_order")`. `f` must be a
  callable `equinox.Module` (anything else is a `TypeError`); `propagation` is static and checked
  against `("first_order", "unscented")`. `dim` is the innovations'. The innovations' `loc` is
  added to `f`'s output, `x' = f(x, controls) + ε`, `ε ~ N(loc_j, chol_j chol_jᵀ)`, as `b_j` is on
  the linear path; its parent axes are the factor's discrete control nodes in declared order
  (the trivial `(1,)` for none), which is also where a discrete node's level count is read.
- **Propagation** (`objectives/hybrid.py`). The state and every continuous control carried as a
  belief form one joint `z` with a block-diagonal covariance root (control beliefs independent of
  the state and of each other). First order: `f(μ_z)` and `J Σ_z Jᵀ`, `J = jacfwd` in `z`, which
  is `J_x Σ J_xᵀ + J_u S J_uᵀ`. Unscented: `sigma_points(loc_z, root_z, kappa=0)` (points at
  `μ ± √n` root columns, mean weights `0` and `1/(2n)`) with `β = 2` added to the centre's
  covariance weight (`α = 1`, `κ = 0`, `β = 2`: every covariance weight positive), the weights of
  the reference script. Then `Q` and the offset are added and the covariance factorised. A
  truncated control belief enters through its mean and covariance. A step is unbatched (the
  planners vmap), as the neural step is.
- **Mixture-carried** (prior a `MixtureGaussian`): each component through the same step under
  `vmap`, weights kept, then `reduce(k)` (or `reduce_mode_aligned(k, 1)`), which at `k`
  components and one mode leaves the mixture as it is.
- **Epistemic chain**: the Gaussian branch's `BeliefPair` stepping was factored into one helper
  shared by the linear, nonlinear and neural branches (no change to the linear path). A neural
  transition under `planning_belief="posterior"` had received the pair unstepped and raised; it
  now steps both members like the others.
- **Validation** runs in `validate_structure` (every face's `_validate`), not at construction:
  the control layout (each node's kind and width) is the System's, so a declaration alone cannot
  be checked against it. Checked: the innovations' parent shape, the prior's dimension, and
  `jax.eval_shape(f)` on a `(d,)` float state and the layout (an `int32` scalar per discrete
  node, `float32[width]` per continuous one), which must give a floating `(d,)`; a failure inside
  `f` is re-raised naming the layout. Discrete state parents raise `NotImplementedError`
  (switching dynamics) from `System._space`, before any face reads the control layout, so the
  refusal is the one raised whether or not the factor also has a control node (a review found
  the layout's own error raised first in that case). The zero-process-noise refusal (V10) applies unchanged;
  its sentence now reads "a Gaussian (linear or nonlinear) transition's process noise …".
- **Surface.** `as_env` steps `f(x, controls) + ε_j`, a discrete entry cast to `int32`.
  `describe` writes it through the registry (below).
  `description_length` prices the address at `resolution_term(k, n)` with `k` = the size of
  `f`'s floating-point leaves (`eqx.is_inexact_array_like`: a Python float counts one; integer
  arrays and static fields are structure) + the innovations' `loc` + one Cholesky triangle per
  mode; no prior link (a point). `learnable_blocks` gives `()`, so `learn` and `System.carry`
  pass it through, and a rate on it is refused as a point value. `merge` carries it; a binding
  that would remove a control node `f` reads (a `BoundaryPrior`, a state relayed into the node,
  a `subsystem` view cutting a partner's node) raises `NotImplementedError` in
  `_drop_control_axis`; an `ActionControl`, which supplies one component's control node from another's (the acting binding of a nested component: the driver's decision is the one node, read as a supplied level, and no belief is identified; `nesting-and-coupling.md` N2 and N7), keeps `f`'s inputs whole and is allowed. A belief over a continuous control node is admitted for a nonlinear
  reader in `score_policies` (it was refused for every reader but a linear-Gaussian one).
- **The registry** (`description/__init__.py`, exported from `hybrid_aif`; tests in
  `tests/test_transition_registry.py`). `register_transition(name, cls)` takes a name of at
  least two dot-separated identifier segments (`"drive.unicycle"`) and a callable
  `equinox.Module` subclass (else `ValueError` / `TypeError`). As `register_format` and
  `register_row` do, it refuses a second registration of a name, the same class included, and
  also a class already registered under another name, since `describe` finds the name by the
  class. `registered_transitions()` lists `{name: class}`; there is no unregister (tests swap the
  registry for an empty one). The registration records the schema: each field of `cls`
  (`dataclasses.fields`) with whether it is static (`eqx.field(static=True)`) or a leaf. Shapes
  are not part of it: a class does not fix its leaves' shapes, each instance does. `build` checks
  shapes after it reconstructs `f`, when it validates the System, by the `jax.eval_shape` check
  against the factor's control layout described above; a leaf whose shape breaks `f` is refused
  there.
  - *`describe`.* The record is `{"family": "NonlinearGaussianTransition", "parents": [x,
    *control nodes], "parameters": {"f": <tree variable>, "innovations": {"loc", "chol"}},
    "options": {"f": {"name": ..., "static": {...}}, "propagation": ...}}`. `f`'s class is
    looked up by identity (a subclass of a registered class is unregistered); an unregistered
    one is refused naming `register_transition`. The leaves are one tree variable,
    `<child>_transition_f`, keyed by field name, each leaf a floating-point array (a Python float
    included) in the usual clamp encoding, inline or in the arrays file; an integer or boolean
    leaf is refused naming the field, since a document's numbers carry no dtype. A map with no
    leaves writes no `f` parameter. Static fields are written under `static` when they are
    `None`, a bool, an int, a finite float, a string or a tuple of these (a JSON list, read
    back as a tuple); any other is refused naming the field. The innovations are written as a
    linear-Gaussian transition's.
  - *`build`.* The name is looked up; an unknown one is refused naming the name and
    `register_transition`. The document's field names (the tree variable's keys and `static`'s)
    are compared with the registered schema, and missing, extra, and leaf/static-swapped fields
    are named together in one refusal, so a document written for an older `f` fails rather than
    rebuilding with code that means something else. `f` is then assembled from those values
    without calling the class's `__init__` (as the library's own families are, so a custom
    `__init__` signature does not matter), its `__check_init__` run. Only a registered class is
    instantiated, and nothing a transition record names is imported or executed (a
    `Deterministic` map's `impl` import path is a separate, older mechanism). A tree variable for
    `f` takes
    no prior and no `init` recipe.
  - *Composition and pricing.* A merged composite's records keep their names under the prefixed
    child (`robot.pose_transition_f`). The family joins the transition menu, so the graph code's
    `log |menu|` for a transition is `log 6` where it was `log 5`.
- **Measured** (the library, float32, against 200,000 draws): final position error at a heading
  sd of 0.2 rad, first order 0.013 m, unscented 0.0005 m, frozen 1.29 m; at 0.5 rad, 0.075 m,
  0.002 m and 1.34 m. The planning test: with a lane tolerance of 0.4 m and a terminal target
  at `(−1, 0.3)` facing back, enumeration through `f` turns left first under both propagations
  and the frozen linearisation goes straight; the gradient planner's first turn rate is `+π/2`
  through `f` and `−0.18` frozen.
- **Unscented accuracy and dimension** (measured in review). With `α = 1`, `κ = 0` the sigma
  points sit at `±√n` standard deviations, and `n` counts every dimension of the joint `z`,
  control beliefs and tracked constants included, even those `f` treats linearly. One step of
  the heading's cosine and sine at a heading sd of 1 rad errs by 0.003 at `n = 3` and 0.10 at
  `n = 10`; on the 20-step turn at a heading sd of 0.5, padding the pose with 3 or 7 unused
  dimensions moved the unscented error from 0.0020 to 0.0011 and 0.0041 m, still far inside
  first order's 0.075. `κ = 3 − n` keeps the error near 0.003 at every `n` but gives up the
  non-negative weights; a rule that does not widen with `n` (Gauss–Hermite on the nonlinear
  coordinates only) is the further `propagation` value N2 anticipates.

## Amendment 2026-11-13: process noise scaled by the map

*Status. Ruled by the owner on 2026-11-12: (i) a per-coordinate, state- and control-dependent
scale on the nonlinear transition, the registered map returning the mean and the scale (the
decoder-error design, R3.1 (a): "`scale` multiplying the declared innovation factor per
coordinate"; "first order reads the scale at the mean; unscented averages it over the points");
and (ii) one path shared with the neural transition's input-dependent noise where the
mathematics is the same, ruled in conversation on 2026-11-12 ("Decision 2"; the design file
does not record it). Every point below marked* **(builder's choice)** *was settled in the build
and awaits the owner's confirmation; the closing* **finding** *is for the owner to read, not
settled design.*

The innovations above are a fixed Gaussian: `x' = f(x, controls) + ε`, `ε ~ N(loc_j, Q_j)`. The
amendment lets the noise depend on the previous state and the controls without a new family:

    x' = f(x, u) + loc_j + diag(s(x, u)) L_j ξ,    ξ ~ N(0, I),    Q_j = L_j L_jᵀ

so the step's covariance is `diag(s) Q_j diag(s)`, entry `s_i s_k (Q_j)_ik`. The scale changes
each coordinate's variance and leaves the correlations of `Q_j` as declared. The decoder-error
design's test 12 writes the first-order noise as `diag(scale)² ∘ Q`, which keeps only `Q`'s
diagonal; what is built is `Q ∘ s sᵀ = diag(s) Q diag(s)`, the form R3.1 (a) states ("scale
multiplying the declared innovation factor"), which keeps `Q`'s correlations. The two agree for
a diagonal `Q`.

- **The offset is not scaled** **(builder's choice).** `loc_j` is added as declared. Scaling it
  would make the predictive carry a cross term, `Cov(m(z), s(z) ∘ loc)`.
- **The declaration** **(the field's name, its values and `None` for off: the builder's choice,
  confirmed by the owner on 2026-11-13).** A
  static field, `NonlinearGaussianTransition(f, innovations, propagation, noise_scale=None)`, with
  `noise_scale="per_coordinate"` declaring that `f` returns the pair `(mean, scale)`, both
  `float[d]`. Validation (the `jax.eval_shape` check of N5, which needs the System's control
  layout and so runs there, not at construction) checks the return against the declaration
  both ways: a pair from a map declared `None` is refused naming `noise_scale`, as is a mean
  alone, or a scale of the wrong shape, from one declared `"per_coordinate"`. No code path reads
  the return's structure to decide what it means; each reads the declared field through one
  private accessor, which raises at trace time if the two disagree. Detecting the pair by its
  structure alone was set aside: a map that returns a tuple by accident would be read as
  scaled, and the document would not say so. A string value, not a bool, so a matrix-valued
  form (below) is a further value.
- **What a scale must be.** Strictly positive and finite; not checked under `jit`. A zero on a
  coordinate whose mean has no spread makes the predicted covariance singular and its Cholesky
  factor NaN (V10 refuses a zero in a declared factor but cannot see one in a scale); a negative
  entry flips the sign of that coordinate's correlations. For gradients (the gradient planner,
  and gradient learning later) the scale must also be differentiable wherever it is evaluated,
  the unscented centre point included: its mean weight is zero, but `0 · NaN` is NaN. The form
  the decoder-error design needs, `sqrt(1 − rho(d)²)` with `rho(0) = 1`, has an infinite or NaN
  derivative at zero displacement, and from a start at rest the gradient planner returns a NaN
  plan. Guard it: `sqrt(1 − rho² + ε)`, or a double `where` that keeps the root's argument away
  from zero on the branch not taken. Tests pin the guarded form.
- **A scale of ones** gives the plain map's results: bit for bit to first order (the noise is
  `Q ∘ 1`), to rounding under the unscented rule (the weights' sum), bit for bit in `as_env`
  (the same draw from the same key). A map declared `None` takes the plain path bit for bit
  (checked on the survey example's regression quantities against the tree without this
  amendment).
- **Propagation.** First order (ruled): `μ' = f(μ_z)`, `Σ' = J Σ_z Jᵀ + Q ∘ s(μ_z) s(μ_z)ᵀ`, the
  scale read at the joint mean of the state and any control belief. Unscented
  **(builder's choice: the reading of "averages it over the points")**: the predictive
  covariance is the law of total covariance, `Cov E[x'|z] + E[Cov(x'|z)]`, and the second term is
  `E[diag(s) Q diag(s)] = Q ∘ E[s sᵀ]`, estimated over the same sigma points with the mean
  weights, `Σ_i w_i s_i s_iᵀ`; not `diag(E[s]) Q diag(E[s])`, which drops the scale's own spread
  (`Q ∘ Cov(s)`). For a scale affine in the state the unscented rule is exact (the second moment
  is quadratic) and first order is short by exactly `Q ∘ B Σ Bᵀ` (`s = c + B x`). A control
  carried as a belief enters the scale as it enters the mean, through the joint `z`. The second
  moment is positive semidefinite because every mean weight is non-negative (`κ = 0`); a
  negative `κ` would need this revisited.
- **The path shared with the neural transition** (ruled (ii)). One function,
  `objectives/hybrid.py::_push_conditional(g, loc, root, propagation)`, takes `g(z) -> (m(z),
  s(z))` and returns the mean and covariance of `m(z)` and `E[s sᵀ]` under either rule; the
  caller adds the noise, `Q ∘ E[s sᵀ]`. The nonlinear transition calls it with its map and its
  innovations' `Q`; the neural transition (`_mlp_step`) with its head's mean and scale and
  `Q = I`, which is its `diag σ(μ)²`. What is shared: the Jacobian or the sigma points, the
  spread, and the scale's second moment. What is not: the neural transition steps to first
  order only (its `propagation` field is read on the likelihood path, by
  `inference/linearise.py`); the neural likelihood's unscented reading keeps its own code,
  since it returns a statistical linearisation `(H, c, R)` for conditioning rather than
  predictive moments, but its noise, `Σ_i w_i σ(x_i)²`, is the diagonal case of the same
  expectation, so the two rules agree. Through the shared step the neural transition's spread
  is formed as `(J L)(J L)ᵀ` instead of `J (L Lᵀ) Jᵀ`: its mean agrees bit for bit and its
  covariance to relative rounding, a few units in the last place (2.9e-7 relative in float32,
  2.2e-16 in float64, measured in review).
- **Sampling.** `as_env` draws `ε_j = loc_j + diag(s) L_j ξ` with `s` at the sampled previous
  state and the step's controls, exactly.
- **Planning.** Every planner predicts through the step above, as inference does
  (`make_transition`); the gradient planner differentiates through the scale, in the state, the
  control and the map's leaves (subject to the differentiability above). Under
  `feedback="linear_quadratic"` **(builder's choice)** the gains read the mean's Jacobians and
  the scaled noise reaches the pass through the recorded covariances; the Riccati pass ignores
  the noise's dependence on the state, so it does not credit a correction for the noise it
  would avoid.
- **The record** **(builder's choice: omission at the default, refusal of a stripped
  document).** `describe` writes `options.noise_scale` when it is `"per_coordinate"` and omits
  it at its default, so the document of a map without a scale carries no trace of the option,
  and a document without it builds a map declared `None`. A reader of the document sees that
  the innovations are scaled by the named code. A document whose option is removed while its
  map returns a pair fails validation rather than building a model whose noise is unscaled; an
  unknown value is refused. `registered_transitions` and replay need nothing of their own
  (replay runs inference).
- **The description length** **(builder's choice).** The map is priced as N5 says: the scale is
  computed by `f`'s code from `f`'s leaves, already counted; the option itself is unpriced, as
  `propagation` is, so a scaled noise is model capacity at the plain map's price. The note says
  the noise is scaled.
- **Refusals** apply to a scaled map with the plain map's sentences: discrete state parents
  (switching dynamics), a zero on the declared factor's diagonal, a merge binding that removes a
  control node `f` reads, an unregistered map at `describe`. A mixture-carried factor steps each
  component with the scale at its own moments, as the plain map does; `merge` carries the field.
- **Scope: a per-coordinate scale only, as ruled.** A matrix-valued factor, `f` returning
  `(mean, M(x, u))` with the noise `M L Lᵀ Mᵀ`, would be a clean generalisation: the shared step
  would return `E[N Nᵀ]` with `N = M L` in place of `Q ∘ E[s sᵀ]` (`Σ_i w_i N_i N_iᵀ` unscented,
  `N(μ) N(μ)ᵀ` to first order), sampling would draw `M L ξ`, and the declaration would take a
  further value of `noise_scale`; it costs `d²` numbers per point and needs `M` kept invertible.

**Finding for the owner (not settled).** A per-coordinate scale keeps the correlations of `Q`.
A persistent error `e' = R e + w` with per-group persistences `R = diag(ρ_g)` and stationary
covariance `P` needs `Cov(w) = P − R P R`, entries `P_ik (1 − ρ_i ρ_k)`; a scaled fixed `Q` gives
`s_i s_k Q_ik`, whose correlation `Q_ik / sqrt(Q_ii Q_kk)` does not move. The two agree for every
`ρ` only if `(1 − ρ_i ρ_k)² = (1 − ρ_i²)(1 − ρ_k²)` wherever `P_ik ≠ 0`, which (Cauchy-Schwarz)
holds only for `ρ_i = ρ_k`. In the whitened declaration of R3.1 (`e = L_e ε`, persistence and
the diagonal scale `sqrt(1 − ρ²)` on `ε`), the persistence on `e` is `L_e R L_e⁻¹`, which acts
group by group only when `L_e` is block-diagonal by group. So per-group persistence is exact
when the error's factor is block-diagonal by group (the groups uncorrelated) or the groups share
one persistence; for correlated groups with different persistences it needs the matrix form
above or a declaration that accepts the mixing.

As built: `factors.py` (`noise_scale`, `NOISE_SCALES`, private `_scaled`, `_mean_and_scale`,
`_mean`), `objectives/hybrid.py` (`_validate_nonlinear`, `_push_conditional`,
`_nonlinear_moments`, `_nonlinear_step`, `_mlp_step`), `objectives/closed_loop.py` (the mean
alone), `envs/_process.py` (the draw), `description/__init__.py` (the option),
`description/mdl.py` (the note); `tests/test_scaled_innovations.py`. Measured there: a filter
whose world's speed noise grows with the speed covers the true speed in its 90 % interval 0.885
to 0.902 of the time at commanded speeds 0 to 3; the constant-noise model, right at speed 1.5,
covers 1.000 at rest and 0.605 at speed 3.

## Amendment 2026-11-15: switching dynamics (N4)

*Status. Built in a background step for the survey rover (`terrain-field.md`: "x: position and
velocity, a mode-aligned mixture switched by terrain"); the API choice below is the builder's
and awaits the owner's confirmation.*

A state factor whose `NonlinearGaussianTransition` names discrete state parents in `B_deps` is
switched by them, as a `LinearGaussianTransition` with per-mode `F_j` is.

- **How `f` receives the mode** **(builder's choice).** The mode reaches `f` as an integer
  level in the parent's declared place among `controls`: `controls` has one entry per parent in
  declared (`B_deps`) order, a discrete state parent's level, a discrete control node's level, a
  continuous node's value. The innovations' parent axes are the factor's discrete parents in
  the same order, the linear transition's convention (`innovations: ... its discrete state
  parents and any discrete control node, each a mode index j`). A factor with no state parent
  sees exactly the `controls` it saw before, so no existing map changes. Two alternatives were
  weighed. *One `f` per mode* (a tuple of modules, or a stacked module vmapped over a mode
  axis) reads naturally for a bank of unrelated maps, but costs a second declaration form for
  one family (a tuple of `f`s in the factor, in validation, in `describe` and in `merge`'s
  binding refusal), forces every mode to share one class and static structure anyway (a vmap
  over modes needs it), and cannot express the common case in which the modes share the map's
  form and differ in a coefficient (the rover's per-terrain speed), which one `f` indexing a
  table by the level writes in one line; a bank of unrelated maps is still one `f` that
  switches by `lax.switch` on the level. *The modes as leading entries of `controls`, ahead of
  the control nodes* (the order the build was first offered) agrees with the declared order
  whenever the state parents are listed first, which is the common case, but splits the
  innovations' axis order from the linear transition's when a control node is declared before
  a mode, and the control space's reading of a reader's level axis
  (`controls._level_axis`) counts axes in declared order for every innovations-carrying family;
  the declared order needs no change there.
- **Propagation.** Each of the `K` components is pushed through `f` once per joint mode `j`
  (the `M` levels of the state parents, row-major over `B_deps`), to first order or unscented
  as declared, the innovations conditioned on `(mode, controls)` in declared order; the grown
  component's log weight is the component's plus the mode's (the parents' post-transition
  belief, `_mode_log_weights`, as the linear bank reads it); the `K M` components are laid
  out component-major, `propagate`'s layout, and reduced by `reduce_mode_aligned(k, M)` under
  `mixture_schedule="mode_aligned"` or Runnalls otherwise. With no state parent `M = 1` and the
  step is the per-component one it replaced. An affine `f` per mode reproduces the per-mode
  linear bank to float tolerance under both rules and both schedules (the unscented rule is
  exact for an affine map).
- **Family.** A Gaussian prior on a switched factor makes it mixture-carried, as on the linear
  path; `mixture_k` must be set, and a multiple of `M` under `"mode_aligned"`. The parents must
  be categorical and declared earlier (the linear path's within-step ordering refusal).
- **Inference, information gain, planning.** Nothing of their own: the mode evidence a channel
  on the factor sends to its parents (`infer_states`), the Kolchinsky–Tracey bracket of a
  mixture-carried factor's information gain, and every planner read the mixture family and
  `B_deps`, not the transition's class. `feedback="linear_quadratic"` still refuses a
  mixture-carried factor (L4).
- **Sampling.** `as_env` reads the parent's level sampled at this step (the linear path's
  ancestral order) and steps `f` at it.
- **The record.** `describe` writes the factor's parents as for any transition; the reader
  admits a categorical latent as a nonlinear transition's parent (the mode) and refuses a
  continuous one, which remains not built (cross-factor continuous parents).
- **Refusals that stay:** a neural (`MLPConditional`) transition with discrete parents; a
  `merge` binding that removes a control node `f` reads. The 2026-11-13 amendment's list of
  refusals that apply to a scaled map loses "discrete state parents (switching dynamics)": a
  scaled map is switched the same way, its scale read per mode.

As built: `factors.py` (docstring), `objectives/hybrid.py` (`factor_families`,
`_validate_nonlinear`, `_nl_inputs`'s `modes`, `_nonlinear_mixture_step`, `make_transition`;
`refuse_switching_nonlinear` removed), `system/core.py` (`_space`), `envs/_process.py`,
`description/__init__.py` (the parent check); `tests/test_nonlinear_switching.py`.
