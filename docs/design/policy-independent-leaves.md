# Leaves that do not depend on the plan, and what a declaration has to say for JAX to see it

Status: SETTLED (Alex, 2026-10-21). The rule H1 was settled first; the build that followed it
found the mechanism already present in JAX, and the item became the declaration below.

## Why

A plan is scored by a vmap over policies around a scan over steps. Under the default
`planning_belief="prior"` a Gaussian-carried factor's belief covariance follows the predictive
recursion `Σ_{t+1} = F Σ_t Fᵀ + Q`, and a control enters the mean through `G u` only; when `F` and
`Q` carry no mode axis the covariance sequence over the horizon is the same for every policy, and
so is every leaf that reads the covariance alone: a linear-Gaussian channel's information gain,
the interaction and correlation of several channels on the factor, and the `uncertainty` row.
Risk, utility, a noise map's reading and the parameter-novelty rows read the mean and are per
policy.

## H1 — When the covariance is plan-independent (strict, exact)

A factor's covariance-only leaves are plan-independent when all of the following hold:

- the factor is Gaussian-carried from a `Gaussian` initial prior (no mixture, no `mixture_k`
  growth on it);
- its transition is a `LinearGaussianTransition` with a shared `(d, d)` `F` and an innovation
  noise with no mode axis; a per-mode `F`, a per-mode noise or a `NonlinearGaussianTransition`
  disqualifies it (a per-mode innovation mean and a per-mode `G` do not: they move the mean only);
- planning is on the `"prior"` chain (the posterior chain conditions on the plan's readings;
  `feedback=` requires it and is excluded with it);
- every channel reading the factor is a plain `LinearGaussianObservation`: fixed noise (a noise
  map reads the mean), no mode parents, not over an action node;
- no channel reading the factor has a per-level `present=` entry (the set of channels present at
  a step would then depend on the plan, and with it the interaction and correlation).

No approximation is admitted (a nonlinear transition linearised once at the initial mean would
be one).

## H2 — The mechanism is JAX's, and the declaration has to let it work

`jax.vmap` never batches a value that does not depend on the mapped axis, and `lax.scan`'s
batching rule keeps a carry leaf unbatched when nothing batched feeds it. So for a factor under
H1 the covariance carry and every leaf computed from it are already evaluated once per step,
outside the policy loop, with no help from the library. A build on 2026-10-21 that tabulated the
leaves per step (a scan of the recursion in the planner context, `jnp.take` on the step index in
the closure) measured the residual it could add: on a model with two qualifying factors and
1296 policies, 0.12 → 0.11 ms per `infer_policies` call and 74.5k → 37.3k flops; on the
benchmark model of `per-variable-infogain.md`'s 2026-10-21 note, nothing. It was not kept.

What does stop JAX is the declaration. A `ConditionalGaussian` innovation carries its `loc` and
its `chol` with the same parent axes, so a control that moves the mean per level obliges the
model to stack one noise matrix per level, the transition selects the step's by `jnp.take` on the
level, and the covariance carry is batched even when every level's noise is the same value. On
the benchmark model (`pos` driven by a three-level thrust node, per-level innovation mean, 1296
policies), handing the predictor one mode-free noise matrix instead of the selected slice, with
no other change, took a call from 0.23 ms and 322k flops to 0.12 ms and 90k, the scores bitwise
equal; tabulating on top of that reached 53k flops and the same time.

**Decision.** A `ConditionalGaussian` may declare its `chol` without the parent axes, `(d, d)`
beside a `loc` of `(*parent_shape, d)`: one noise shared across the modes while the mean is per
mode. The shape is static, so the transition step hands the predictor the shared matrix with no
selection, and H1's second clause reads "an innovation noise with no mode axis" off the
declaration. The description format writes and reads the form. A stacked `chol` of equal slices
stays valid and stays batched: the library does not inspect values. No setting, no engine path:
the saving is JAX's once the declaration says what is shared.

## H3 — Acceptance

- A model declared with the shared noise scores bitwise as the same model with the noise
  stacked per level, on the enumerating planner's whole policy set, through the tree planner and
  under the gradient planner (G and its gradient), with the posterior chain and with feedback.
- `describe`/`build` round-trip the shared form; the families table and the validation
  sentences name it.
- On the benchmark model with the shared declaration the lowered flops of `infer_policies` fall
  from the per-level figure and are bounded in `tests/test_plan_evaluation_cost.py`.

## Build notes

Built 2026-10-21. `ConditionalGaussian` accepts `chol` of shape `(*batch, d, d)` beside a `loc` of
`(*batch, *parent_shape, d)` (`shared_chol`); the parent axes are read from `loc`, and a `chol`
carrying some of them is refused with a sentence. `condition_on` returns the shared matrix
unselected, `gaussians()` and `stacked_chol()` broadcast it over the parent axes for a consumer
that needs the stacked shape. `hybrid._lg_inputs` skips the `jnp.take` of `chol` on a control
level, so the transition step hands `predict` a mode-free matrix; the mixture-carried step
broadcasts it over the modes, `mixture_rollout.propagate` reads `stacked_chol()`, a conditional
Gaussian target (`C_deps`) skips the selection, `durations.compose` keeps a shared noise shared
under a shared `F` (a per-mode `F` makes the composed noise per mode), and `discretise`, which
writes its kernel per mode, is unchanged. `description.describe` writes the variable at the shape
it has and `build` accepts `[d, d]` or `[*modes, d, d]` for an innovations record and a
conditional target; `docs/description-format.md` names the form. The learnable paths read no
innovation noise (a transition's innovations are clamped in a description and unlearned in
`learn`), so nothing there changed. `tests/test_shared_innovation_noise.py` pins H3: bitwise
equality with the stacked form on the enumerating planner (prior and posterior chains, whole tree
and posterior), the branching planner, the gradient planner (score and gradient) and
`feedback="linear_quadratic"`, a mixture-carried switching factor, and the round trip.

Measured with the bisect method on the benchmark model (median of 40 calls after warm-up, fresh
process, CPU; `infer_policies`). "Per level" is the shared matrix written as equal slices per
level; the benchmark's own per-level noise (different slice per level) lowers to the same count.
The scores of the shared and the per-level declarations agree to the printed sum.

| `infer_policies` | per level | shared |
|---|---|---|
| 1296 policies, prior chain | 0.226 ms, 320k flops | 0.120 ms, 90k flops |
| 1296 policies, posterior chain | 0.349 ms, 408k flops | 0.169 ms, 112k flops |
| 81 policies, prior chain | 0.160 ms, 28.2k flops | 0.087 ms, 13.6k flops |
| 81 policies, posterior chain | 0.134 ms, 40.1k flops | 0.111 ms, 18.5k flops |

`description_length` prices the shared form as the smaller declaration it is (130.66 against 138.96
nats on the benchmark model at n = 100, the stacked form carrying two more copies of the noise factor).

`tests/test_plan_evaluation_cost.py` bounds the shared 1296-policy call at 117k flops (1.3 times
the measured value).

