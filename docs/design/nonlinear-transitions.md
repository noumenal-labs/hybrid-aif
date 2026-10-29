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
- **A continuous control carried as a belief** (the gradient planner) enters as on the linear
  path, generalised: to first order its spread through the Jacobian in `u`, `J_u S J_uᵀ`, added;
  unscented, the sigma points are drawn over the joint `(x, u)`. A truncated control belief
  enters through its mean and covariance.

## N4 — scope (Alex)

A Gaussian-carried factor, and a mixture-carried one whose mixture comes from its prior: each
component is propagated through the same `f`, then the mixture is reduced as on the linear
path. A nonlinear transition with discrete *state* parents (switching dynamics) is refused: it
would need `f` to take the mode, or one `f` per mode, an API with no consumer yet.

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
  inputs (remove a control node it reads) is refused, since `f`'s signature is fixed code.

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
  `_drop_control_axis`; an `ActionControl` identifying two control nodes keeps `f`'s inputs
  whole and is allowed. A belief over a continuous control node is admitted for a nonlinear
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
