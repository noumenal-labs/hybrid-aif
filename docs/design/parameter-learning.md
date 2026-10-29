# Parameter learning — design requirements

*Status: requirements, all decided (D1–D8, O2–O4). Proposed-API samples use
the O4 factor names (`likelihood`/`transition`/`initial_state_prior`); the
AIF letters appear where the mathematics or today's code is quoted.
Sequencing per `nesting-systems.md` ("parameter learning first
(waiting customer: p_fix moving inside the coverage agent)"). Source
citations are verified against main @ `6338346`.*

*Two things shape everything below. **`pA` as a parallel tuple is out**,
replaced by polymorphic declaration slots — the parameter distribution
lives in the slot it describes (D3). And **"learnable" is two properties,
not one**: *capability*, carried structurally by the declaration type, and
*activation*, which is policy (D1 requirement 1). Collapsing them makes a
frozen-but-sampled parameter distribution unrepresentable and makes
freezing lossy. Activation is a **rate** (`lr`, per slot, static float or
traced array), not a selector. Read D1 requirement 1 and D3 first; the rest
depends on both.*

## The point

A `System` cannot learn. Every face returns beliefs or actions; none
returns parameters, so `SystemParams` is immutable for the lifetime of a
run. The discrete conjugate math for likelihood learning is already
built and parity-tested — it is simply unreachable from the top.

Three things follow, and the second and third are why this doc is longer
than "wire up the update":

1. The deliverable is a **seam**, not a kernel.
2. The waiting customer **changed families** while this item sat in the
   queue. `p_fix` is no longer a Dirichlet problem.
3. The two layers that would meet at that seam **disagree about how a
   parameter distribution is represented**, and the seam cannot be cut
   until that is settled (D3).

## What exists (audit)

| Piece | Where | State |
|---|---|---|
| `ConditionalDirichlet` — `update`, `expected_log`, `mean`, `to_categorical_table`, `condition_on` | `distributions/conditional.py:223` | built |
| Count kernel `α + lr·(onehot(o) ⊗ q(pa₁) ⊗ …)` | `distributions/conditional.py:_count_update` | built |
| `update_likelihood_dirichlet` (per-modality wrapper) | `learning/dirichlet_updates.py` | built, pymdp-parity tested (`test_dirichlet_update_parity`, `test_discrete_parity.py:189`) |
| FPI averaging over likelihood parameters (`expected_log`) | `inference/discrete.py:34` | built, tested (`test_conditional.py:338`) |
| `pA` → exact `spm_wnorm` novelty weights `wA` | `objectives/discrete.py:186`, `compiled.py:96`, `hybrid.py:567` | built, both paths — **migrates** under D3 |
| Polymorphic likelihood slot: `Array \| ConditionalCategorical \| ConditionalDirichlet` lowered through one `_as_log_likelihood` | `inference/discrete.py:28,31` | built, tested — **the representation D3 adopts** |

So the discrete half's math is done. Everything that would let a
*running* System use it is missing:

- **no learning face** — nothing calls the count update; `pA` is
  novelty-only;
- **no learnable transitions or initial priors, and no transition
  novelty** — `SystemParams` carries `pA` alone, and
  `include_param_info_gain` covers the likelihood only;
- **no conjugate family for any continuous parameter** — `distributions/`
  has no inverse-Wishart, inverse-Gamma, or matrix-normal, so D8's three
  priors are three genuinely new classes;
- **two layers disagree about how a parameter distribution is
  represented** (D3) — and that disagreement, not a missing kernel, is
  what the error below actually is.

The seam gap is reproducible today. A `ConditionalDirichlet` in
`params.A` — the one declaration that expresses "average over my
likelihood parameters", and the form FPI accepts and is tested on — trips
the negative condition in `_is_hybrid` and dies in the family classifier:

```
ValueError: modality 'modality_0': unknown observation declaration type
ConditionalDirichlet
```

This is the same shape as the `predict_beliefs` seam-dip: capability
present one layer down, unreachable through the faces.

## The customer moved families

The coverage port's `p_fix` — "does this cell afford a precise fix" — was
the reason parameter learning was scheduled first. Its formulation
changed between stages:

- **Stage 1** (discrete): `p_fix` set a categorical position
  likelihood's confusion precision, `p_fix·I + (1-p_fix)·uniform`. The
  AIF doc §4 frames learning it as Beta–Bernoulli over counts —
  literally `ConditionalDirichlet` on a 2-level modality.
- **Stage 2** (landed, workstream C phase 1.5): the pose is a continuous
  LG factor observed through `A_deps = (pose, cell)` with **per-cell
  observation noise** `R_cell = 0.8 m / clip(p_fix, 0.1, 1)`. `p_fix` is
  now a parameter of a *Gaussian* likelihood.

So the parameter the customer wants learned is an LG observation
precision, and no conjugate machinery for it exists. The brief already
charters the family ("Categorical–Dirichlet, **Linear-Gaussian–NIW**,
GMM via indicator CAVI, … conjugate closed forms as fast paths",
`project-brief.md:121`); nothing has been built.

The unifying observation that makes this one item rather than two: both
updates have the same shape. The Dirichlet accumulates
`onehot(o) ⊗ q(parents)`; the Gaussian-noise posterior accumulates
`residual_second_moment ⊗ q(parents)`. Same responsibility weighting by
the parents' posterior, same "one tick's sufficient statistic added to a
conjugate prior". **One face, one dispatch rule, per-family sufficient
statistics.** That is the design.

## D1 — **RESOLVED**: one face, family-dispatched

```python
System.learn(
    params: SystemParams,
    observations: tuple[Array, ...],
    beliefs: tuple[Belief, ...],
    prev_beliefs: tuple[Belief, ...] | None = None,
    action: Array | None = None,   # the action that PRODUCED this tick: a_{t-1}
    lr: float | Array = 1.0,       # broadcast rate; 0.0 freezes everything
    decay: float | Array = 1.0,    # D7 forgetting; 1.0 is exact conjugacy
    lr_by_slot: Mapping[str, float | Array] | None = None,      # per-slot override
    decay_by_slot: Mapping[str, float | Array] | None = None,
) -> SystemParams
```

`learn` is the **first** face that returns params; every existing face
returns beliefs or actions (`update_empirical_prior` returns
`tuple[Belief, ...]`, `system.py:641` — it is the belief-carry step, not a
params-shaped one). So `learn` adds a new leaf to the carry rather than
joining an existing pattern:

```python
beliefs = system.infer_states(params, obs, priors)
params = system.learn(params, obs, beliefs, prev_beliefs, prev_action)  # new
action, _, tree = system.act(key, params, beliefs)     # note: key is first
priors = system.update_empirical_prior(params, beliefs, action)
prev_beliefs, prev_action = beliefs, action            # carry
```

The transition evidence available at tick *t* is
`(s_{t-1}, a_{t-1}, s_t)` — so `learn` takes the **previous** action
from the carry, not the one `act` is about to return; passing this
tick's action is an off-by-one that pairs `a_t` with the wrong
transition. Placing `learn` before `act` also means the tick's own
planning already sees the updated counts.

Requirements on the face:

1. **Capability is structural; activation is policy.** Two questions, two
   answers.

   *Can* a parameter be learned? Read off the graph: iff the slot holds a
   distribution over it rather than a point value. A point table is a delta
   prior and can never be learned — not by a flag, but because there is no
   posterior to advance. No `learn_A: bool` statics, per `hybrid-system.md`
   D1, and no parallel parameter-prior tuple (D3).

   *Is* it being learned on this call, and how fast? A **rate**, not a
   selector: `lr` broadcasts to every capable slot, and `lr_by_slot`
   overrides it per slot with keys inheriting the broadcast value when
   absent. Deactivation is `lr=0.0`. There is no separate on/off mechanism,
   because a rate already carries one — `0.0` is a perfectly ordinary rate,
   and a boolean would be a degenerate encoding of it. Naming a
   point-valued slot in `lr_by_slot` raises at validation:
   *"modality_0 holds a ConditionalCategorical (a point value); declare a
   ConditionalDirichlet to make it learnable."*

   The two common shapes both read naturally: freeze one thing with
   `lr=1.0, lr_by_slot={"transition:cell": 0.0}`; learn only one thing
   with `lr=0.0, lr_by_slot={"likelihood:pose": 1.0}`.

   **Static rates skip the work; traced rates schedule it.** The type is
   `float | Array` per slot, and which one you pass decides what happens:

   - A **Python float** is static at trace time, so `if lr == 0.0: return
     params` is an ordinary Python conditional evaluated during tracing —
     the einsum is never emitted into the jaxpr and a frozen slot costs
     literally nothing. This is the same trace-time dispatch the library
     already relies on (`_is_hybrid`, `_as_log_likelihood`), and it is why
     deactivation is free rather than merely cheap.
   - A **traced array** cannot be branched on in Python, so it multiplies
     the statistic instead. That is what makes a curriculum inside
     `lax.scan` expressible — `lr_by_slot={"transition:x": jnp.where(t < 1000,
     1.0, 0.0)}` — and what lets a `vmap`ped batch of agents sit on different
     schedules.

   Both are wanted, and letting the *type* choose gets both without a mode
   flag. Two hazards to document: `jnp.float32(0.0)` silently takes the
   traced path and pays the FLOPs where `0.0` would not; and rates are
   arguments rather than `SystemParams` fields precisely so the static
   option survives — a Python float placed in a jitted pytree carry becomes
   a traced leaf, which would foreclose the trace-time skip. Per-call policy
   arguments are also the established pattern: `act` already takes six
   (`selection`, `alpha`, `mode`, `planner`, `use_bound`, `metrics`), while
   `SystemParams` stays "generative tensors + preference designations".

   **Collapsing the two costs three things**, which is why they are
   separate:

   - **A frozen distribution over parameters is a legitimate model.**
     Sampling an `A` table from a Dirichlet without learning is domain
     randomization; it is also the honest generative reading of a CPD whose
     uncertainty is by-design rather than to-be-resolved. D3 wrinkle 3
     specifies `as_env(params, param_key=key)` for exactly this case.
   - **Novelty-without-learning must be expressible.** Parameter info gain
     is computed from the concentration, so collapsing the two makes "does
     epistemic drive help even when the counts never move?" unaskable — an
     ablation that isolates the exploration bonus from the learning it
     motivates, and the kind Champion's contrarian-information-gatherer
     ladder needs (`project-brief.md` open question 11).
   - **Freezing by type conversion is lossy.** With capability alone as the
     switch, freezing means converting
     `ConditionalDirichlet → ConditionalCategorical`, which **discards the
     concentration**. A curriculum — learn for 1000 ticks, freeze to
     evaluate, resume — cannot round-trip: resuming either restarts from a
     fabricated prior or needs the counts stashed outside the declaration,
     which is the sidecar D3 deletes. Deactivating freezes losslessly and
     needs no graph surgery.

   **Why a rate is not the `learn_A: bool` that D1 forbids.** The
   prohibition is on statics that *duplicate* what the declarations
   already say — "is this thing learnable at all" is derivable, so
   encoding it twice invites the two copies to disagree. "How fast do I
   update on this call" is not derivable from the model; it is additive
   information owned by the call site. That is the line, and it is the same
   line D3 draws against `pA`: never represent one fact twice.

   It also keeps the departures from Bayesian purity to **one**. A learning
   rate is already a non-conjugate free parameter; making activation a value
   of that same parameter adds no second departure, whereas a separate
   on/off switch would have been a new one.

   **The key space must be slot-qualified, and this is a defect in the
   existing selectors too.** `lr_by_slot` and D6's
   `include_param_info_gain` should share one namespace, but bare
   declaration names cannot serve as one:

   - `modality_names` and `factor_names` are independent statics
     (`system.py:123-124`) with **no uniqueness and no cross-disjointness
     check** anywhere in `src/`. Name a modality and a factor both
     `"pose"` and any selector over "declaration names" is ambiguous.
   - Structurally worse: `B` and `D` are *both* keyed by factor names, so
     `("factor_1",)` cannot distinguish "learn factor_1's transition" from
     "learn factor_1's initial prior". That is not a collision, it is a
     **missing coordinate**. Selectors must be slot-qualified —
     `("likelihood:pose", "transition:cell")` — or keyed by a per-slot
     mapping. The qualifier is the slot's field name (O4).
   - And the two disagree on error semantics for the same input:
     `lr_by_slot` naming a point-valued slot must raise, while
     `include_param_info_gain` silently omits the leaf today
     (`compiled.py:157`, `hybrid.py:567`: `... if pA is not None else ()`).
     Make novelty raise too.
   - Acceptance 6's own example, `include_param_info_gain=("factor_1",)`,
     **raises today**: `_validate` checks that selector against modality
     names only (`system.py:347-350`).

   Slot-qualified keys are cheaper to introduce now than after
   `include_param_info_gain=("modality_0",)` has users, and doing it once
   gives D6's selectors and the later per-variable refactor one key space.
2. **Family dispatch, not family tags.** Which sufficient statistic is
   accumulated follows from the declaration type, inspected the way
   `planner_context` inspects the pytree; unknown declaration types raise
   loud, specific errors naming the extension point.
3. **Returns `SystemParams`, mutates nothing.** Learning becomes part of
   the agent's carry. Shapes are static, so a learning tick scans and
   jits like any other.
4. **No new inference machinery.** `learn` consumes only what the faces
   already return: this tick's observations, this tick's and the
   previous tick's posteriors, and the action that connected them — all
   already in the loop's carry. `prev_beliefs`/`action` are required
   exactly when a learnable transition is an active target (loud error if
   missing), ignored otherwise. Anything needing a smoother is out of
   scope (D4).
5. **This is the third rung of the existing timescale ladder** —
   `data-clamp (frozen) → parameters (slow) → states (fast)`, named in
   `nesting-systems.md`. Learning is the rung the library declares but
   has never run.

**The scope boundary, stated so it stays a choice.** In a generic
factor-graph engine `learn` would not exist: parameters are variables,
learning them is inferring them, and one `infer` face would cover both.
We are deliberately not that library — `infer_states` and `learn` stay
distinct faces with a visible parameter/state split at the API level,
patterning after pymdp, because the split is what buys the static shapes,
the compiled discrete path, and the timescale ladder above. What we take
from the generic treatment is its *representation* (D3) and its
*learnability rule* (requirement 1), not its uniform interface.

This is a **sequencing** boundary, not a rejection: the generic
`infer(query, clamped)` layer is on the roadmap as a **[deferred]** item
gated behind the per-variable info-gain refactor, and the fork is
`project-brief.md` open question 13. Nothing in this pass forecloses it —
D3's polymorphic slots are the generic representation, and D1's
per-declaration rates are a soft query set (`0.0` = not in the query,
 intermediate values = damping). The two rules that keep the
option cheap are that node-type dispatch resolves at trace time and that
schedules stay static; both hold throughout this design.

## D2 — **RESOLVED**: substitute by the kind of quantity, per consumer

Two coherent answers existed while `A` and `pA` were separate tensors:
mean-sync (`E[A]` everywhere, pymdp's choice — `A` is a cache) versus
posterior-authoritative (`E[log A]` in inference, the VFE-minimizing
mean-field update, which our FPI already implements).

**Decision: `E[log A]`, for correctness.** With the standing rationale
worth recording as a general library principle, not a local one: *parity with
pymdp is valuable wherever we can have it, and where we deliberately diverge
the divergence is recorded and tested.*

The cost is recorded because it is a **parity divergence, not a bug**:
`E[log A] ≠ log E[A]` (Jensen, with `E[log A] ≤ log E[A]`), so a learning
agent's state posteriors will *not* match pymdp's end-to-end even when
both count kernels agree to 1e-7. Parity therefore binds at the kernel
leaf, not the loop — the same boundary the compiled-path parity tests
already draw.

**Under D3 the *staleness* question stops existing** — there is no second
tensor that can disagree with the first. But every consumer that wants "the
table" must still say what it substitutes when the table is uncertain, and
these are **not interchangeable**: each consumer has a determinate right
answer, set by what kind of quantity it is computing.

| Consumer | Computes | Takes | Why |
|---|---|---|---|
| Inference (FPI) | an expected log-likelihood | `E[log A]` (digamma) | The VFE-minimizing mean-field update. Already implemented. |
| Predicted observations `q(o)` | a probability | `E[A]` (the mean table) | **Correct, not a compromise.** Marginalizing the parameters out of `P(o \| s, θ)` *is* the mean table, so the predictive uses it by derivation. Utility and risk inherit this. |
| Ambiguity `E_q(s)[H[·\|s]]` | an entropy | `H[E_θ[A]]` — the **mean-table** entropy | **Corrected 2026-07-27** (the row originally prescribed the expected entropy — a sign-error, see below). With an explicit novelty leaf in the tree, mean-table ambiguity is what makes the decomposition exact: the reducible part it contains is cancelled by the novelty leaf. |
| Novelty | an information gain | `_wnorm` on the concentration | Unchanged (D6). Every appearance of `I(o;θ\|s)` in the tree is exactly this one leaf. |
| `as_env` forward sampling | a table to sample through | `point()`, or a draw under `param_key` | D3 wrinkle 3. |
| A learnable `D` as an FPI prior | a log-prior | `log(mean())` | `Dirichlet.expected_log()` is *unnormalized* — digamma differences sum below zero (`dirichlet.py:48`) — and FPI adds priors as log-probabilities (`inference/discrete.py:67`), so passing it would add an improper prior. Use the normalized mean; the parameter-averaging form belongs on likelihood slots, not prior slots. |

**The ambiguity row is the one that matters — and this section originally
got it wrong.** (Correction 2026-07-27, caught during the step-1–2 build
when the implementation surfaced the two decompositions disagreeing; kept
here in full because the error is instructive.) Entropy decomposes exactly:

```
H[E_θ[A](·|s)]  =  E_θ[H[A(·|s)]]  +  I(o; θ | s)
```

The second term *is* parameter information gain, and the original row
reasoned: mean-table ambiguity "folds novelty in, and then the novelty
term adds it again." That argument has a **sign error**. The folded copy
enters ambiguity with a *plus*; the novelty leaf enters the EFE with a
*minus*; in the sum they **cancel**. Enumerating the assignments, with
`I := I(o;θ|s)` and exact `G = −utility − I_mean(o;s) − I`:

| Form | Terms | Total |
|---|---|---|
| D | −utility − state-IG (mean table) − novelty | `G` ✓ (pymdp; our `utility_infogain`) |
| A | risk + ambiguity (**mean table**) − novelty | `G` ✓ (the cancellation above) |
| B | risk + ambiguity (expected entropy), **no novelty leaf** | `G` ✓ (novelty absorbed) |
| C | risk + ambiguity (expected entropy) − novelty | `G − I` ✗ — the double count |

The original row prescribed Form C — the very defect it claimed to
prevent. **The decision is Forms A + D**: with an explicit novelty leaf,
*both* decompositions take the mean table everywhere, both equal `G`, and
both equal **each other exactly** — even under the `_wnorm` approximation,
because the identical novelty leaf appears in both, so the approximation
error cancels between them. Toggling `include_param_info_gain` off removes
the same leaf from both sides, and they agree again in the non-learning
form. Per-term toggle honesty holds throughout: `I(o;θ|s)` appears in the
tree exactly once, as the novelty leaf.

The expected entropy is not wasted: `amb_mean = amb_exp + I` is a
meaningful split — *irreducible* observation noise versus uncertainty that
learning can remove — so `expected_entropy()` remains on the declarations
as the irreducible-ambiguity diagnostic (closed form for a Dirichlet:
`ψ(α₀+1) − Σ_k (α_k/α₀)·ψ(α_k+1)`), and Form B remains available later as
a documented no-novelty-leaf mode.

So the rule is: **substitute by the kind of quantity, not by a single
global convention** — probabilities and entropies take the mean table,
log-likelihoods take `E[log A]` — **and every appearance of `I(o;θ|s)` in
the term tree must be one explicit leaf.** The binding test is
decomposition equality: with a learnable likelihood, the
`utility_infogain` and `risk_ambiguity` totals agree to float tolerance,
toggle on and toggle off. That test would have caught this error at
specification time; it is now part of the acceptance surface.

## D3 — **RESOLVED**: the parameter distribution lives in the slot it is a distribution over

Per D1's capability rule — a parameter *can* be learned iff its slot holds
a distribution rather than a point value — the real question is not *where
`pA` goes* but **whether that distribution sits in the slot it describes,
or in a tuple parallel to it.**

**Option (a) — parallel tuple** (what `SystemParams` has today, pymdp's
shape):

```python
params = SystemParams(
    A=(ConditionalCategorical(log_table=jnp.log(A0), parent_ndims=1),),
    pA=(ConditionalDirichlet(concentration=alpha, parent_ndims=1),),
    B=..., D=...,
)
params = system.learn(params, obs, beliefs)   # pA advances...
env, env_params = system.as_env(params)       # ...and A is now stale
```

Two objects describe one CPD. `_validate` must check they agree in count,
in `parent_ndims`, and in shape — three new failure modes that are
representable only because the representation allows disagreement. Every
consumer must know which wins (that was D2). And `learn` must choose
between refreshing `A` from `pA` on every tick — a redundant leaf in the
carry, kept in sync by hand — or leaving it stale, in which case `as_env`
samples from a table the agent no longer believes.

**Option (b) — polymorphic slot:**

```python
params = SystemParams(
    likelihood=(ConditionalDirichlet(concentration=alpha, parent_ndims=1),),  # capable
    transition=(ConditionalCategorical(log_table=jnp.log(B0), parent_ndims=2),),  # point value
    initial_state_prior=...,
)
params = system.learn(params, obs, beliefs)   # likelihood[0].concentration advances
env, env_params = system.as_env(params)       # takes likelihood[0].point(); cannot be stale
```

One object per CPD, so nothing can disagree with anything. Capability is
`isinstance(a, Learnable)`; note that the transition here is not "frozen" but
*incapable* — a point value with no posterior to advance, which is a
different thing from the capable-but-not-currently-advancing case
the rates express (D1 requirement 1). Mixed capability across modalities
is free and needs no selector grammar to express.

**Recommendation: (b)**, and the decisive
evidence is that **the library already does this one layer down.**
`inference/discrete.py:28` declares

```python
Likelihood = Array | ConditionalCategorical | ConditionalDirichlet
```

and lowers all three through one `_as_log_likelihood` (`discrete.py:31`),
taking `expected_log()` for the Dirichlet branch. FPI has accepted a
polymorphic declaration slot since it was written, its module docstring
documents it, and there is a test named for exactly this property —
`test_fpi_dirichlet_likelihood_uses_expected_log`
(`test_conditional.py:338`).
`System`'s classifiers are the only layer that rejects the form, and `pA`
is the only place in the library where a parameter distribution sits
*beside* its parameter instead of *in place of* it. So the `ValueError` in
"The point" is not a missing feature — it is two layers disagreeing about
the representation, and option (b) is the one that already has two of the
three layers behind it.

`_is_hybrid`'s "negative conditions" comment is not an obstacle here: it
warns against exempting new *families* from the classifiers, and a
Dirichlet over a categorical CPD is not a new family — it is the same
tabular family with an uncertain table. The predicate widens to a named
union rather than gaining a special case:

```python
TabularObservation = ConditionalCategorical | ConditionalDirichlet
TabularTransition  = ConditionalCategorical | ConditionalDirichlet
TabularPrior       = Categorical | ConditionalCategorical | Dirichlet | ConditionalDirichlet
```

### The protocol

Consumers need four things from a declaration, and `ConditionalDirichlet`
already provides all four — three as methods, one as a free function:

| Protocol member | Purpose | Today |
|---|---|---|
| `expected_log_params()` | inference averages over parameters | built as `expected_log()` (`conditional.py:245`), consumed by FPI — **must be renamed**, see below |
| `point()` | posterior-mean declaration, for `as_env` and the compiled path | built as `to_categorical_table()` (`:254`) |
| `stats(...)` + `absorb(stats, lr, decay)` | compute one tick's sufficient statistic, then absorb it | `update(child_obs, parent_beliefs, lr)` (`:260`) does **both at once**; must be split |
| `novelty()` | parameter-info-gain weights, better-when-larger | exists as `_wnorm` (`objectives/discrete.py:186`); moves onto the declaration |
| `sample(key)` | draw a table from the posterior | **does not exist** — new method, and D3 wrinkle 3 plus acceptances 3 and 7 all depend on it |

Three qualifications on that table, each of them real work:

- **`expected_log` is a name collision, not a shared member.**
  `ConditionalCategorical.expected_log(parent_beliefs)` (`:180`) means
  "E over *parents* of log P" and takes an argument;
  `ConditionalDirichlet.expected_log()` (`:245`) means "E over
  *parameters*" and takes none. Same name, different arity, different
  semantics. The genuinely uniform lowering is `_as_log_likelihood`, which
  *dispatches* precisely because the names collide. Rename the
  parameter-averaging one.
- **`update()` is not `absorb(stats, …)`.** It takes raw evidence and
  computes the statistic internally via `_count_update`. The protocol needs
  the split, which is also what makes D7's bitwise claim structurally
  guaranteed rather than incidental: both paths then add the *same*
  `stats` expression.
- **`point()`'s return type is polymorphic per slot** and the doc must say
  which: `to_categorical_table()` returns a `ConditionalCategorical`, but
  `as_env`'s discrete path wants a raw probability array and the compiled
  path wants `log_table`. For D8 it is genuinely ambiguous — does the noise
  slot's `point()` return a covariance or its Cholesky factor? (The
  existing field is `R_chol`.) Decide per slot, in the docstring.

`Learnable` is the protocol of the last two.

**`Learnable` is protocol satisfaction, not a boolean** (a second
question). There is no `is_delta` predicate and no `learnable` field to
read: a declaration is capable of learning iff its *type* provides
`absorb`, which `ConditionalDirichlet` does and `ConditionalCategorical`
does not. The distinction is carried by the class, so `isinstance(d,
Learnable)` is resolved statically at trace time and costs nothing at
runtime.

Deriving a boolean from an `is_delta()` check on a common base would have
been the other design, and it is worse for a specific reason: it implies a
single type spanning both cases, whose delta-ness is a *value* property —
so a "point" declaration would be a distribution with a degenerate
concentration, and every consumer would need a numerically-guarded branch
on how degenerate is degenerate enough. Typed capability puts that
distinction where it cannot drift.

The corollary is that **capability is declaration-time and activation is
per-call** (D1 requirement 1), and only the second is meant to vary during
a run. A frozen point table and a frozen-but-sampled Dirichlet are
different *models*; a Dirichlet you are and aren't currently updating is
one model under two policies.

### Cost, honestly

Roughly twenty sites widen, across `system.py`, `objectives/hybrid.py`,
`objectives/discrete.py`, `objectives/compiled.py`, and
`envs/generative.py` — the negative classifiers (`_is_hybrid`,
`factor_families`, the observation classifier) plus every reader of
`.log_table` / `.H` on a declaration. Enumerate them from the source at
build time rather than trusting a count here.

**The real hazard is not the loud sites but four *positive* `isinstance`
filters that fail silently.** D1 requirement 2 promises that unknown
declaration types raise specific errors, and widening the negative
classifiers delivers that — but a positive selection like

```python
cat_m = tuple(m for m, a in enumerate(params.A)
              if isinstance(a, ConditionalCategorical))    # system.py:270
```

does not raise when it *misses* a `ConditionalDirichlet`; it drops the
modality. Widen `hybrid.py:279` and forget `system.py:270` and a learnable
modality's observation is **never consumed by inference, with no error**.
The sibling at `hybrid.py:546` drops it from the term closures: no utility,
no info gain, no ambiguity, no novelty, silently. Under D5, two more:
`initial_priors` (`system.py:163`) passes a non-`ConditionalCategorical` D
straight through *as a belief*, and `envs/generative.py:153` would call
`Dirichlet.sample`, which returns a simplex vector where a state index is
expected — silent garbage rather than a crash.

**So step 1's acceptance must include an exhaustive audit of positive
`isinstance` filters on declaration types, not just the classifiers.** The
novelty path is by contrast the *safe* part: it reads only
`.concentration` and `.parent_ndims`, verified across every site.

Three further wrinkles:

1. **`pA` is deleted, not deprecated.** Fifteen source lines across four
   files, all novelty, all reading only `.concentration` and
   `.parent_ndims`; they take the likelihood tuple and filter for `Learnable`
   instead. **Thirty-five lines across five test files** also depend on its
   shape, including `test_objectives.py` — the pymdp parity surface for
   `param_info_gain`, which passes `pA` positionally to both
   implementations, so that parity test changes too. The notebook does not
   reference `pA` in any cell, so the adoption gate is unaffected. This is
   the cheapest moment in the library's life to make the change.

   It also removes a **fourth sidecar failure mode**, worse than the three
   above: `hybrid.py:608` indexes `pA` by *modality index*, so in a hybrid
   model `pA` must be a full-length tuple over **all** modalities including
   LG ones — you must supply a placeholder `ConditionalDirichlet` for an LG
   modality purely so the positional index lands. That trap gets worse the
   moment anyone builds a mixed-modality hybrid with novelty on.
2. **Capability is a declaration-time choice; only activation varies at
   run time.** Swapping a point declaration for a distributional one
   changes the pytree *structure*, so it retraces — and it is lossy in the
   freezing direction (D1 requirement 1). That is what the rates are for:
   setting `lr` to zero loses nothing and, with a static float, costs
   nothing, whereas toggling capability rebuilds the model. Say both in the
   docstring; the failure mode to warn against is someone reaching for type
   conversion when they wanted `lr=0.0`.
3. **`as_env` must now answer a question the sidecar hid:** with a
   distribution in the slot, does the environment use the mean table or a
   table drawn from it? Default `point()` — deterministic, and what every
   current test assumes. But the sampled reading is the honest generative
   one, and it makes acceptance 3 strictly better: draw the true table
   from the prior, then check learning recovers *it*, instead of hand-
   writing a "deliberately wrong `A`". Worth adding as
   `as_env(params, param_key=key)` in the same pass.

## D4 — **RESOLVED**: learnable transitions and the two-slice joint

The transition count update needs `q(s_t, s_{t-1}) ⊗ onehot(a_{t-1})`
(pymdp's `update_state_transition_dirichlet_f`, which takes the pairwise
belief from its caller). Under a mean-field posterior that factorizes
over time, the two-slice joint would *be* the outer product of the
marginals. So the kernel is `q(s_t) ⊗ q(s_{t-1}) ⊗ onehot(a)`, with the
previous tick's posterior threaded in the carry.

**But that family does not exist here, so the kernel is an approximation.**
The library has no variational posterior over trajectories: `fpi` computes
a per-tick mean field *across factors*, with the previous tick's
pushed-forward belief as a prior. That is a filtering recursion, so
`q_t(s_t)` conditions on `o_{1:t}` while `q_{t-1}(s_{t-1})` conditions on
`o_{1:t-1}`, and the outer product of two **differently-conditioned**
marginals is not the pairwise marginal of any single `q`. The smoother is
therefore a **correctness** upgrade, not merely a parity one. Say so in the
docstring.

The stated kernel is also the single-parent case only. `B_deps` supports
cross-factor dependencies, so the general form is
`q(s_{f,t}) ⊗ (⊗_g q(s_{g,t-1})) ⊗ onehot(a_f)`, with the
product-of-marginals approximation compounding once per parent.

Two honest caveats to document, not to fix:

- Our marginals are **filtered**, not smoothed; pymdp infers over a
  window, so its `q(s_{t-1})` has seen `o_t`. End-to-end episode
  learning will therefore differ from pymdp by more than D2's Jensen
  gap. **Accepted** — not a concern, on the condition that
  the smoother closes the gap when it lands. So the requirement this
  section leaves behind is on *that* pass, not this one: when the
  smoother arrives (workstream C, pending), the learning face must take
  the pairwise joint it produces and the divergence must measurably
  shrink. Write that as a test then, and note here that it is owed.
- Only the taken action's slice updates. Counts for untried actions stay
  at the prior — which is precisely what transition novelty should then
  make attractive (D6).

**Parity boundary:** kernel-level, given the same joint and action
one-hot as input. Stated in the test, not discovered later.

## D5 — **RESOLVED**: learnable initial priors are in scope

Initial-state counts accumulate once per *episode*, not per tick, so a
per-tick `learn` is the wrong home for them: they want an episode-boundary
entry point. pymdp-jax does not currently include an `update_D`, so there is **no parity
oracle** — the acceptance test is recovery, not parity.

**Decision: in scope.** The reasoning is worth keeping because it
explains *why* the dedicated slot works: in an unrolled factor-graph
message-passing setting there is only one copy of the initial prior's
parameters per rollout, so a `Dirichlet` sitting in the initial-state-prior slot just
works — the episode boundary is where that single copy gets its evidence,
and there is no per-tick ambiguity to resolve.

Under D3 this is nearly free: the slot already accepts a union
(`Categorical | Gaussian | MixtureGaussian`), so a learnable initial prior
is a `Dirichlet` (unconditional) or `ConditionalDirichlet` (hierarchical,
with a `D_deps` entry) in the same slot — and **both classes already
exist and are exported**. No new field, no new distribution, and the count
kernel is the existing one with no parents. What remains is the **timing**.

**The update belongs at tick 0 of the ordinary `learn`, gated to fire once
per episode — not in a face called at episode end.** At episode end the
belief in hand is `q(s_T)`, while the update needs `q(s_0)` (pymdp's
`update_state_prior_dirichlet` (not enabled in the current release) takes `qs` at the *first*
timepoint of its window). An episode-end face would therefore either absorb
the wrong belief or oblige the caller to stash `q(s_0)` for a whole
episode. Under filtering `q(s_0 | o_0)` is already in hand at tick 0, so
gating there needs no new face and adds no fourth rung to the ladder.

Two items this section owes, neither worked out: a
`ConditionalDirichlet` `D` with a `D_deps` entry needs
`q(s_0) ⊗ q(parents at t=0)`, but `initial_priors` *averages* the
conditional prior over its parents and returns marginals
(`system.py:159-164`), so what the weighting needs is not what the face
returns; and `_validate` currently requires `isinstance(d,
ConditionalCategorical)` when `D_deps` is present (`system.py:372`), so a
hierarchical learnable prior **raises today**.

## D6 — **RESOLVED**: novelty must track the learnable set

`include_param_info_gain` currently means "A-novelty". Once transitions
are learnable, the term must cover them or the toggle lies:

- **Transition novelty**: the same `_wnorm` (exact `spm_wnorm`,
  sign-flipped) applied to the transition declaration's concentration over
  the child axis, contracted against `q(s_{t-1})` and the policy's action.
  It shares `_wnorm` with A-novelty and nothing else: A-novelty is
  contracted at the observation and its weights are hoisted *out* of the
  planning scan as loop-invariant (`compiled.py:96`), whereas a per-action
  transition-novelty slice is not loop-invariant. So this is a new term at
  a new point in the loop, and it costs the compiled path a per-step
  gather — cheap, but not free.
- **Novelty for a continuous parameter (D8) has no ported form** and needs
  deriving — the expected information gain about an inverse-Wishart/Gamma
  posterior from one observation. Options: closed form (available for the
  scalar/diagonal case), or the existing Kolchinsky–Tracey bound-pair
  machinery, which would land it as an `Interval` rather than
  `Interval.exact` — the first non-exact leaf in the term tree, which is
  what the bound-pair design is for.
  *Resolved for the matrix-normal inverse-Wishart emission in D10 (2026-10-17).*

**Decision: per-parameter-type selectors, and per-parameter where cheap.**
Under D3 that lands almost for free, and lands in the better form. The
existing grammar is `bool | tuple[str, ...]` over *modality* names, which
was adequate only while "parameter" meant "A". Once any declaration can
carry a parameter distribution, the natural selector is over **declaration
names** — modality names for observations, factor names for transitions
and initial priors — and the implementation is one walk over the
declarations, filtering `Learnable`, rather than one code path per
parameter kind — though per the bullet above they remain two *terms*, with
only the weight computation shared.

**A deeper question, recorded not resolved.** In a generic
factor-graph treatment parameters are just variables, so there is no
`include_param_info_gain` at all: there is one `info_gain` term per
variable, and a parameter's novelty is the same term as a state's. That is
the better factoring, and this section is the second place the generic
reading pays (D3 was the first). We are not taking it in this pass — the
term tree is organized by EFE decomposition (`utility_infogain` /
`risk_ambiguity`) with static per-term toggles, and re-basing it on
per-variable terms is an objectives-layer redesign with its own parity
surface. But note what the per-declaration selector above *does* buy: it
is the same shape as the generic answer, one level short of it, so the
generic version is a later refactor of one walk rather than a rewrite of
several code paths. Deliberate stepping stone, not a dead end.
*Note (2026-10-17):* a consumer case makes the cost concrete. An unknown row offset coupled to
the pose by the reading is best inferred jointly with the pose, which the library does only
within a state block, so it is declared as a pose dimension; as a state it then needs a small
process noise to keep beliefs positive definite (per-variable-infogain.md V10, decision 4).
Under the generic treatment it would be a parameter inferred jointly with the states it is
coupled to, an exact constant needing neither.

**This deferral is a recorded debt with a release gate**, not a shrug:
`project-brief.md` → "Post-v0.1 roadmap candidates", the **[deferred]**
per-variable info-gain item. The gate is *before any public release*, on the
reasoning that an API-shape debt gets harder to pay the moment the current
shape has users. It in turn gates the generic message-passing layer listed
beneath it, so this refactor is on the critical path for both.
Anyone building on this section should read that entry as a constraint on
how much surface the per-declaration selectors are allowed to grow.

Worth stating as an upside rather than a risk: the coverage port runs
`include_param_info_gain=False` today (`coverage_aif_stage2.py:235`), so
parameter novelty is *new behavior* for the customer — an agent that
seeks out cells whose localizability it is uncertain about, instead of
only exploiting the map it was handed.

## D7 — **RESOLVED**: forgetting, as an opt-in

Pure count accumulation cannot unlearn: after 10⁴ ticks a cell's
concentration is immovable, and a world that changes is mismodeled
forever. pymdp's update exposes `lr`; we add `decay` and a prior-anchored form. The coverage agent already models
staleness as volatility (`ε` on the per-cell features), so the customer
has the concept.

**Decision: yes, as an option — and explicitly flagged as
"questionably Bayesian".** That flag is the design constraint, so it goes
in the docstring rather than in a footnote: exponential forgetting is not
a conjugate update to any generative model we have declared. It is a
tracking heuristic that stands in for a hierarchical model we have not
built (parameters as a slow random walk rather than as constants), and the
principled version is a change of *model*, not of update rule. Default
`1.0` therefore means the library ships Bayesian and lets you opt out
knowingly; the docstring names the model that would make it principled, so
the shortcut stays visible as a shortcut.

`decay` rides alongside `lr` on the face, with the same broadcast-plus-
per-slot-override shape (D1 requirement 1) — the two are independent rates
over the same key space, which is what lets a slot be frozen while still
drifting:

```
α ← decay·α + (1 − decay)·α₀ + lr·stats        decay ∈ (0, 1], default 1.0
```

the standard leaky-count / exponential-forgetting form, `decay=1.0`
recovering exact conjugacy and pymdp parity. Written in this form
deliberately: at `decay=1.0` the update is `1.0·α + 0.0·α₀ + lr·stats`,
which is **bitwise** the no-forgetting path on positive concentrations —
what acceptance 9 demands. The equivalent `α₀ + decay·(α − α₀)` form
re-derives `α` through a subtract-then-add round trip and is not
bitwise-safe. Cheap, one line, and it is the difference between
parameter learning and parameter *tracking*.

**`α₀` needs a home** — `ConditionalDirichlet` has exactly two fields
(`concentration`, `parent_ndims`, `conditional.py:232-233`) and the D3
protocol's `absorb(stats, lr, decay)` has no slot for a prior.

**Decision: an optional anchor on the declaration, chosen by the
declaration when it decays.**

```python
alpha0: Array | None = None    # decay anchor: the prior, a scalar (uniform), or absent
```

Decay relaxes toward the *anchor*. The rationale is not convenience but
modelling: for a realistic (e.g. biological) agent, forgetting should
relax toward a *constrained*, phenotypic prior — what the agent
constitutively expects — not toward maximum entropy. This also sharpens
the "questionably Bayesian" note above: an anchored decay is the
discrete-time relaxation of exactly the hierarchical model the docstring
names (parameters as a slow mean-reverting walk whose stationary point is
the anchor), so the anchor is what makes the shortcut *closest* to
principled.

*Amended 2026-10-15 (Alex).* The first ruling here read an unset anchor as
uniform `1.0`, the leaky-count convention. That default was retired: a
`decay < 1` on a declaration without an anchor now raises and names the
choices. The reason is the evidence measure
(`description-length.md` D7): anchored at the prior, decay only reweights
the absorbed statistics, `α_t − α₀ = Σ_s decay^(t−s)·lr·stats_s`, so the
slot's absorbed count is the mass it retains and its closed-form evidence
stays defined; anchored anywhere else, the prior's own mass above the
anchor is forgotten too, the count can turn negative and the slot is
unscored (the prequential evidence then carries it). The prior is
therefore the principled anchor, and it costs a second table of
parameters, because learning overwrites the concentration in place and
the prior is not recoverable from it. That cost is charged by choice, not
silently: `anchor="prior"` stores the concentration; `anchor="uniform"`
stores a scalar `1.0`, which broadcasts in the kernel and costs nothing;
an array is kept as given (the resume case: a declaration holding a saved
posterior, decaying toward the prior it was learned from, which is neither
itself nor uniform). A run without decay never reads the anchor, so none
of this reaches it. Stabilised forgetting with the prior as the
alternative distribution is the standard construction (Kulhavý & Zarrop
1993); a flat alternative is its other common form.

The no-decay and no-anchor cases pay nothing:

- `None` is structural in the pytree — no leaf, no shape, no runtime
  cost; every existing construction keeps working unchanged.
- `absorb` reads `self.alpha0` internally, so the face is untouched (the
  rejected route — α₀ as a `learn` argument — stays rejected).
- The `None`-vs-supplied branch resolves at trace time, and a static
  `decay=1.0` skips the whole term anyway, so acceptance 9 (bitwise) is
  unaffected.
- α₀ is genuinely not derivable from α, so the field is not the
  duplication D3 forbids.

Construction validates the anchor — shape equal to `concentration` or a
scalar, values finite and strictly positive — which retires the `inf·0.0 → nan`
hazard by construction. Ergonomics: the `from_table(A, strength=)`
constructor (implementation gap 2) takes `anchor=` with the readings
above (`"prior"`, `"uniform"`, an array, or `False`), for "forget toward
where I started". The same anchor pattern applies per family in step 3
(inverse-Gamma/IW anchors on their own parameters, required whenever the
family decays).

One numerical caveat survives any route: `absorb` needs a floor — with
`decay < 1` and no evidence,
concentrations fall toward α₀, and if α₀ is near zero then
`digamma(α) → −∞`. For inverse-Wishart, `ν` must stay above `d − 1` for
the mean to exist and `Ψ` must stay positive-definite — decaying `ν`
violates that silently. No family currently documents a floor; each must.

## D8 — **RESOLVED**: the LG conjugate families — the customer's path

New conjugate family: a prior over `LinearGaussianObservation`'s
parameters, per categorical parent (per cell). The sufficient statistic
for one tick, with the state uncertainty carried properly:

```
S = (y − H μ − c)(y − H μ − c)ᵀ + H Σ Hᵀ
```

accumulated with the same responsibility weighting as the Dirichlet —
`S ⊗ q(parents)`, so an uncertain cell splits the evidence across cells
exactly as an uncertain state splits counts. (The degrees-of-freedom
parameter accumulates the responsibility mass itself,
`ν_j += q(parent = j)` — the LG analogue of the Dirichlet's count; a
family that grows `S` without growing `ν` never sharpens.)

**`S ⊗ q(parents)` is an approximation, and it is worst exactly where the
customer lives.** For the Dirichlet, `onehot(o) ⊗ q(pa)` is the *exact*
expected statistic because the statistic is linear in the parent
indicator. For LG with **per-parent** `(H_j, c_j, R_j)` — the whole point
of `R_cell` — the exact per-mode statistic uses the **mode-conditional**
posterior, `q(j)·[(y − H_jμ_j − c_j)(·)ᵀ + H_jΣ_jH_jᵀ]`, not one marginal
`(μ, Σ)` scaled by `q(j)`: under an uncertain cell the joint over (pose,
cell) is a mixture, and `q(x)` alone is not sufficient for it. So "exactly
as an uncertain state splits counts" is false for the switching case.

Worse, it is *avoidable* in principle and blocked in practice:
`condition_mixture_switching` already computes the grown, mode-resolved
posterior, and then `pad_mixture(grown, mixture_k)` Runnalls-reduces it
(`system.py:296-302`) — destroying the mode-conditional pairing before
`infer_states` returns. So D1 requirement 4 ("`learn` consumes only what
the faces already return") genuinely binds here, and for the switching LG
case what the faces return has already thrown away what `learn` needs.
**Decide explicitly:** either `infer_states` gains a way to surface the
pre-reduce mixture, or the continuous update is knowingly biased under mode
uncertainty and says so in its docstring. This is the `mixture_k`-versus-
learning interaction, and it is not hypothetical.

### The three priors, all from the start

**Ship the standard set, matching prior factor-graph work: MNIW,
independent Normal × inverse-Wishart, and inverse-Gamma.** All three are
binding rather than sequenced, because "scalar first, widen later" would
set the seam's shape from the narrowest case and then re-cut it twice:

| Prior | Over | Shape of `R` | Use |
|---|---|---|---|
| **inverse-Gamma** | `R` only, head point-valued | diagonal (per-dimension), or one scalar × fixed shape | The customer's case — `R_cell = 0.8/clip(p_fix, 0.1, 1)` is isotropic. Cheapest, trivially static. |
| **inverse-Wishart** | `R` only, head point-valued | full | Correlated noise with a known, trusted sensor model. The floor, if MNIW proves painful. |
| **matrix-normal–inverse-Wishart (MNIW)** | head and `R` **jointly** | full | Learning the sensor model itself: head given `R` is `N(M, R ⊗ V)`, with `R ~ IW(Ψ, ν)`. |

The three are not a ladder but a **factorization choice**, which is what
makes the seam right the first time. Independent `N × IW` asserts that the
coefficient and noise posteriors factorize; MNIW asserts they do not,
coupling the coefficient scale to `R`.

Under D3 the declaration reads per field, straight from the capability
rule — each parameter independently either has a point value or carries a
distribution over itself:

```python
class LinearGaussianObservation(eqx.Module):
    head:  Array | MatrixNormal          # the augmented (H, c), one object
    noise: Array | InverseGamma | InverseWishart
```

Two fields, not three. `H` and `c` must **not** be separately
distributional: two independent matrix-normals would assert that their
posteriors factorize, and for a linear model with an intercept they do not
— they are coupled through `Σxx`. Since the point of this representation is
that it states which posteriors factorize, a per-field `H`/`c` split would
state something false. One matrix-normal over the augmented head is the
standard object and the correct one.

The three priors are then combinations of two fields: the customer's case
is `noise: InverseGamma` with a point head; independent `N × IW` is
`head: MatrixNormal, noise: InverseWishart`; a fully point-valued head and
noise is exactly today's class. **MNIW is the one case that does not
factorize per field** — it is a genuinely joint prior over head and noise —
so it needs one named type of its own:

```python
class LinearGaussianObservationMNIW(eqx.Module):
    """Head and noise under one matrix-normal–inverse-Wishart prior."""
```

That asymmetry is the representation doing its job: the joint prior looks
different because it *is* different.

Per-field capability composes with the activation policy without
interacting with it — a head can carry a noise posterior and a point
`head`, and the policy decides whether this call advances it. Which fields
are capable is a modelling choice; which are advancing is a scheduling one.

**Ordering within the pass:** inverse-Gamma first because it unblocks the
customer and its recovery test is the sharpest (two cells with
known-distinct `R_j`), then inverse-Wishart (same statistic, no
diagonalization), then MNIW (needs the coupled update and its own
recovery test on `H`). All three behind one seam, cut once.

**Two qualifications on the statistic, both of them real work.** The
statistic above is correct for `noise` **with the head known** — the
algebra is exact, `E_q(x)[(y−Hx−c)(·)ᵀ] = (y−Hμ−c)(·)ᵀ + HΣHᵀ`. It does
**not** cover the MNIW case: with `H` unknown there is no residual to form
around `Hμ`, and the update needs the triple `(S_yy, S_yx̃, S_x̃x̃)` with
`x̃ = [x; 1]` and `S_x̃x̃ = μ̃μ̃ᵀ + Σ̃` — a structurally larger statistic and
a different `absorb` arity. And `ν_j += q(parent=j)` is the inverse-Wishart
convention; for inverse-Gamma parameterized as (shape α, rate β) it is
`α += 1/2`, `β += S/2`. Both need the per-family factor stated.

**Two build-time findings (2026-07-27, step 3c), recorded as constraints:**

- **Joint (head, noise) learning needs an identification source.** With a
  latent state observed once per tick, the marginal is
  ``y ~ N(c, H²P + R)`` — any ``(H, R)`` on that ridge fits, so MNIW
  recovery *requires* a calibrated co-sensor pinning the state (or
  known-state episodes). This is a property of the model, not the update;
  the acceptance test uses an anchor modality, and the coverage stack's
  calibrated odometry/marker fixes play that role in production. Not a 1-D
  artifact — in ``d`` dims the ridge is ``{(H,R): HΣ₀Hᵀ + R = S}`` and, even
  at known ``R``, ``H`` is identified only up to a rotation of the whitened
  state (the factor-analysis ambiguity) — higher dimensions are strictly
  worse. Noise-only learning (3b) is immune — the head is known there.
- **The independent ``N × IW`` head learner is deferred, loudly.** The
  representation and joint sampling ship (``head: MatrixNormal`` beside any
  noise — domain randomization over sensor geometry works), but its
  shared-``V`` CAVI update under non-spherical noise is a genuine
  approximation, so ``learn`` raises pointing at MNIW (the coupled family,
  which needs no such approximation) or a frozen slot. Revisit only with a
  customer whose head↔noise factorization assumption is load-bearing.

**Naming:** the coupled prior here is matrix-normal–inverse-Wishart,
**MNIW**. The symmetry table's continuous-`D` row uses **NIW** for the
Normal–inverse-Wishart over an initial mean and covariance — a fourth
class, not among the three named as new here. Keep the names distinct in
code.

Note what is genuinely new here versus merely wired: `distributions/`
currently exports `Categorical`, `Dirichlet`, `Gaussian`,
`JointCategorical`, `MixtureGaussian`, and the three conditionals — so
`InverseGamma`, `InverseWishart`, and `MatrixNormal` are **three new
distribution classes**, not just new declaration types. They are the only
new leaves in this whole pass; everything in D1–D7 is representation and
wiring over families that already exist. That is the honest cost estimate
for D8, and the reason it is step 3 rather than step 1.

### Two routes to the customer's number

| Route | Update | Trade |
|---|---|---|
| Learn `R_j` directly | conjugate inverse-Gamma on the residual above | Principled; inverts the actual likelihood; gives per-cell noise straight to the model. |
| Learn a Beta on `p_fix`, push through the deterministic map | Beta–Bernoulli on the thresholded hit `σ ≤ 0.8 m` | The AIF doc §4 formulation; reuses the *existing* Dirichlet kernel with no new family; but learns from a lossy binary summary of the residual, and cannot represent a cell that is *biased* rather than imprecise. |

The second is a same-day change with today's kernels; the first is the
honest inversion and the one the library should own. They are not
exclusive — the Beta route is available immediately as the customer's
interim, and does not need this doc.

## D9 — **RESOLVED** (2026-10-16): carrying posteriors across a rebuild

A customer agent rebuilds its params every step (the dynamics linearised about the current
heading, in a frame centred on the robot, targets and noise maps re-expressed in it) and
must keep what learning put into them from step to step. `System.carry(fresh, learned,
slots=None)` returns the fresh params with every learned posterior block taken from the
learned ones. A block is one posterior `learn` updates, keyed as its rates are: a Dirichlet or
conditional Dirichlet, a matrix-normal inverse-Wishart emission (one joint block, head and
noise), a linear-Gaussian observation's learnable noise (its own block, so a re-expressed head
beside it is rebuilt while the noise carries), a trainable neural conditional (its optimizer
state is the `LearnState` the agent already carries). A block is carried whole, its anchors
and growth propensity included. Point values come from the fresh params, and so does a
separate `MatrixNormal` head, which `learn` does not update (D8). It refuses a block whose
family, shape or dtype differs between the two, a selected slot that holds a posterior on one
side only, and a selection that splits a growable factor's blocks (its weights, initial prior
and the emissions over it, which growth and reduction keep in step), rather than copy or drop
anything silently.

The carry is a plain copy, exact when what was learned means the same thing in the rebuilt
params. A posterior over a parameter the rebuild re-expresses is carried in its old
expression and is then wrong: a head in a frame that moves with the agent, as the customer's
pile head is under its per-step rebuild. A carry that maps a posterior through the change of
frame was considered (matrix-normal posteriors stay closed form under affine maps of their
input and output) and set aside (Alex, 2026-10-16): its only use is undoing the per-step
frame rebuild, and relinearising nonlinear transitions along the plan's nominal trajectory,
the customer's third request, removes that rebuild, letting the model live in the world frame
where the plain carry is exact. Until then the customer bridges the frame by hand for the one
head, or brings that work forward; their call.

## D10 — **RESOLVED** (2026-10-17): novelty over a matrix-normal inverse-Wishart emission

The customer's agent should seek information about a learned head itself (the pile map's
row offset) where its other sensors pin the pose. The `likelihood:<modality>` novelty leaf of
a `LinearGaussianObservationMNIW` is the information one observation carries about the head
and noise together, `θ = (W, Σ)`, for the emission `y = W x̃ + e`, `x̃ = [x; 1]`.

**At a known state** it is exact and closed form: `I(θ; y | x) = (d/2) log(1 + x̃ᵀVx̃) +
c(ν, d)`, `V = V_inv⁻¹`, where `c` (the noise's share, the same at every state) is the
entropy of the Student-t predictive with `ν − d + 1` degrees of freedom less the expected
Gaussian entropy, `log det Ψ` cancelling. Checked against Monte Carlo.

**Which quantity, with the state a belief (Alex, 2026-10-17).** The leaf is `I(θ; y)`, the
state marginalised under the belief the linear-Gaussian state leaf reads (the epistemic chain
when there is one), not `E_q(x) I(θ; y | x)`. Three reasons. It is what per-variable-infogain.md
V9 gives: a selection buys information about what it selects, the unselected parents
marginalised. It is the chain-rule partner of the channel's state leaf, which is computed at
the posterior-mean head, `I(y; x | θ)`: `I(y; x, θ) = I(y; θ) + I(y; x | θ)`, where the
averaged form would count the overlap twice (the identity is approximate, since the state leaf
reads the mean head and the expected-precision noise rather than averaging over `θ`). And
where the uncertain part of the head is its offset, as in the customer's case, it falls as the
state belief widens, since a reading that state error could explain says less about the
offset; the averaged form grows instead (on one random case from 1.6 to 4.0 nats as the pose
spread went from 0 to 3, where the truth fell from 1.6 to 0.5), rewarding the agent for being
lost. Where the head's state columns are uncertain too, the true value can rise with the
spread, since a wide spread of readings carries information about those columns.

**Estimator (Alex, 2026-10-17).** `I(θ; y) = H[y] − E_θ H[y | θ]`. At a fixed `θ` the
reading is exactly Gaussian, `N(W μ̃, W_x Σ_x W_xᵀ + Σ)`, so the second term is
`E[log det Σ]`, exact, plus an average over sampled heads of what the state's spread adds to
it (256 draws from fixed keys: the leaf is deterministic, and policies are compared on the
same draws). `H[y]` has no closed form, but `y`'s covariance does: `A + B`, with `A = (1 +
μ̃ᵀVμ̃ + tr(V_xx Σ_x)) E[Σ]` the noise's part and `B = M_x Σ_x M_xᵀ` the state's spread
through the mean head. At a known state `y` is the Student-t predictive with `ν − d + 1`
degrees of freedom; the state's spread adds a Gaussian part, which makes it less
heavy-tailed. So `H[y]` is the Student-t entropy at `y`'s covariance moved toward the
Gaussian entropy there by `B`'s share of it, `tr(B (A + B)⁻¹) / d`. At a known state the
leaf is exact, the closed form above. The leaf is reported as a zero-width `derived=False`
interval, as a learned row's estimate is (amortization.md A4).

Measured against a precise reference (the head and noise integrated out analytically, so only
the state is averaged, by importance sampling; its own error about 0.03 nats), on `d = 2` and
`d = 4`, `ν` from `d + 1.5` to 10, pose spreads 0 to 30, three random cases per cell:

| | `d = 2` | `d = 4` |
|---|---|---|
| only the offset uncertain, `ν ≥ d + 2` | within 0.10 | within 0.17 (0.08 for `ν ≥ 8`) |
| only the offset uncertain, `ν = d + 1.5` | within 0.22 | within 0.33 |
| every column uncertain | within 0.35, mean +0.02 to +0.13 | within 0.47, mean +0.13 to +0.20 |

Where every column is uncertain the leaf reads high, increasingly so with the state's spread:
the reading given the state is a Student-t whose scale grows with `x̃ᵀVx̃`, so averaging over
a wide state belief mixes scales and makes `y` heavier-tailed than the blend allows. In 4 of
the 48 cases that reversed the order of two spreads whose true values differed by more than
0.1 nats. A research round with a criterion fixed in advance (every cell within 0.15, mean
within ±0.05, no reversals) tested two further stand-ins, an entropy-power combination of the
state's Gaussian part with the noise's Student-t part and the decomposition `H[y | x] + I(x;
y)` with a Gaussian channel; both did worse than the blend everywhere and neither passed, so
the blend stands with the scope above, and a stand-in that accounts for the mixing of scales
is open. The 256 draws carry noise of up to about 0.075 nats; the leaf is clipped at zero,
which that noise could otherwise cross where the true value is near zero. Low `ν` matters
most: it is where learning starts. Four alternatives were set aside:

- *The Gaussian entropy at `y`'s covariance alone*, a ceiling on `H[y]` (adopted first and
  replaced the same day, Alex, after review). It overstates by the Student-t's heavy tails,
  and so does most at low `ν`: at a known state by 0.43 nats at `d = 2` with `ν = d + 2`,
  2.2 at `d = 8`; against Monte Carlo by up to 0.9 at `ν = 3.5`, by an amount that varies with
  the state and so distorts the ranking of policies.
- *The Student-t entropy alone*, exact at a known state, reads low as the state's Gaussian
  part grows, up to 0.8 nats under at `ν = 3.5`, and can go negative.

- *An interval with `E_q(x) I(θ; y | x)` as its ceiling.* That ceiling grows where the truth
  falls, so the interval is widest exactly where policies must be told apart, and its
  midpoint inherits the wrong trend.
- *Linearising the reading about the belief's mean* (closed form, the head's share exact at a
  known state, the noise's share shrunk by the noise's fraction of the reading's spread). It
  is within 0.05 nats where only the offset is uncertain but misses the information a wide
  state belief's spread of readings carries about uncertain state columns of the head, up to
  1.5 nats under. Dispatching to it in that regime was weighed and set aside: under the
  planners' `vmap` both branches run, the gain is within the Monte Carlo's own error, and a
  threshold would put a step into the leaf.

A Gaussian-mixture state belief enters through its mean and covariance. `E[Σ]` must exist
(and the Student-t's covariance with it), so `ν > d + 1` is required (a concrete `ν` below it raises; a traced one raises when the
computation runs).

**Discrete parents (Alex, 2026-10-17): refused.** With a mode parent, the reading at a fixed
`θ` is a mixture over modes, so the sampled term would need a Kolchinsky–Tracey bracket per
draw, and per-variable-infogain.md V9 records those pairs as loose at `d ≈ 4`. Weighting
per-mode values by `q(mode)` would not fall with mode uncertainty, the flaw rejected above.
Selecting the novelty of an emission with discrete parents raises, naming the sampled-bracket
estimator as the way forward.

## Symmetry check (per history 021)

Every slot, its point form and its distributional form, and where each
stands after this pass. Under D3 the third column is the whole design: the
distributional form *is* the declaration, in the same slot. Per D1 it
confers *capability*; how fast a given call advances it is `lr`, and
neither column is "the frozen one".

| Slot | Point form | Distributional form | This pass |
|---|---|---|---|
| `likelihood` (A) tabular | `ConditionalCategorical` | `ConditionalDirichlet` | seam + face (kernel exists) |
| `transition` (B) tabular | `ConditionalCategorical` | `ConditionalDirichlet` | new kernel + novelty |
| `initial_state_prior` (D) tabular | `Categorical` / `ConditionalCategorical` | `Dirichlet` / `ConditionalDirichlet` | D5, tick-0 gated |
| `likelihood` LG — `noise` | `Array` | `InverseGamma` / `InverseWishart` | D8, all three priors |
| `likelihood` LG — `head` | `Array` | `MatrixNormal`, or MNIW jointly with `noise` | **in**, last (D8's MNIW row) |
| `transition` LG — `F`, `b`, `Q` | `Array` | *not* the dual of the likelihood LG row: `Q` is not a field of `LinearGaussianTransition` but lives inside `innovations: ConditionalGaussian` as `.chol` (`factors.py:36`), so a learnable `Q_j` needs a *conditional* prior over a parent-indexed stack | **out** — a decision, not an oversight, and a harder problem than the likelihood LG row rather than its mirror image |
| `initial_state_prior` continuous | `Gaussian` / `MixtureGaussian` | NIW on the initial mean/covariance | **out** — needs D5's episode timing *and* D8's families; no customer |
| `MixtureGaussian` components | — | indicator CAVI | **out** — the next pass |

The `head` row moved in as a consequence of D8: once MNIW is in scope,
learning the coefficients is not additional machinery, it is the prior
that was already chosen. Only the LG *transition* dual stays out.

## Acceptance (binding)

1. **Kernel parity, A**: our per-tick update summed over a sequence
   equals pymdp's `update_obs_likelihood_dirichlet` (which sums a leading
   time axis), categorical and index observations both. Extends
   `test_dirichlet_update_parity` (`test_discrete_parity.py:189`) to the
   face. **Tolerance: `rtol=1e-5`, or `atol=1e-6` only for `T=1`.** A flat
   `atol=1e-6` over a sequence is not achievable: pymdp sums the statistic
   then adds once, we add incrementally, so float32 accumulation diverges —
   exact at `T=1`, but ~1.5e-3 by `T=2000` with counts of order 10³.
2. **Kernel parity, transitions**: equals
   `update_state_transition_dirichlet_f` given the same pairwise joint and
   action one-hot, 1e-6.
3. **Recovery, discrete**: a System draws its true `A` table *from its own
   prior* (`as_env(params, param_key=key)`, D3 wrinkle 3), generates data
   through it, and recovers it — the forward model closes the loop, no
   external fixture and no hand-written "wrong" table.
4. **Recovery, continuous**: the same for `R` on a two-cell pose model
   with known-distinct `R_j`, including the `HΣHᵀ` correction (drop it
   and the recovered noise is biased low — make that a test, not a
   comment). Separately for `H` under NIW.
5. **No stale-parameter representation exists.** A *structural* test on
   `SystemParams`' field set: no field duplicates a declaration's
   parameters, so the test fails if `pA` is reintroduced. Assert on the
   field set, not on object identity — JAX gives no identity to assert on,
   and under `param_key` sampling `as_env` returns a drawn array rather
   than the declaration. This is the binding criterion for the
   representation change.
6. **Novelty symmetry**: per-declaration selectors
   (`include_param_info_gain=("factor_1",)`) work for observation *and*
   transition novelty through one walk over `Learnable` declarations, not
   one code path each; the toggle's cost is printed the way L phase 2
   printed `metrics=False`.
7. **Frozen-but-distributional round-trips losslessly.** With a
   `ConditionalDirichlet` in the likelihood slot and activation off: `learn` is the identity
   on the concentration (bitwise), `as_env(params, param_key=key)` still
   samples a table from it, and parameter novelty is still computed. Turning
   activation back on resumes from the *same* counts. This is D1's
   capability/activation split as a test.
8. **Novelty without learning is expressible**: `include_param_info_gain`
   on with activation off runs, and the counts do not move — the ablation
   that isolates the epistemic drive from the learning it motivates.
9. **`decay=1.0` is exactly conjugate** — bitwise identical to the
   no-forgetting path.
10. **Notebook gate**: `pytest -m notebook` re-executed clean. `learn`
   is public API, so the adoption gate applies.
11. **Recovery, learnable initial prior** (D5): a `Dirichlet` in the initial-state-prior slot
    recovers a known initial-state distribution across repeated episodes.
    Recovery, not parity — pymdp does not currently include an `update_D`.
12. **The customer** (downstream-side, follows this pass): `p_fix` learned
    online in place of the monitor's `pf()`, with the σ-panel behavior at
    least as good as stage 2's.
13. **Decomposition equality** (D2, added with the 2026-07-27 correction):
    with a learnable likelihood, the `utility_infogain` and
    `risk_ambiguity` EFE totals agree to float tolerance, with
    `include_param_info_gain` on and off. The test that would have caught
    the original D2 error at specification time.

**Step mapping:** step 1 → 1, 3, 5, 7, 8, 13; step 2 → 2, 6, 9;
step 3 → 4; step 4 → 11; step 5 → 10, 12.

## Decision log: O2–O4

All now decided. O2 is resolved inside D2's consumer table, where
it belongs; the activation mechanism, formerly O1, is in D1 requirement 1.

### O2 — **RESOLVED** (corrected 2026-07-27): substitute by the kind of quantity

Resolved in D2's consumer table, where it belongs — and then **corrected
during the build**: the original resolution put the expected entropy in
ambiguity alongside an explicit novelty leaf, which is itself the double
count it was meant to prevent (a sign error; D2 now derives the four forms
and adopts A + D). Final rule: probabilities *and entropies* take the mean
table; log-likelihoods take `E[log A]`; `I(o;θ|s)` appears in the tree
exactly once, as the novelty leaf. The binding acceptance is decomposition
equality across `utility_infogain` and `risk_ambiguity` under a learnable
likelihood, toggle on and off.

### O3 — **RESOLVED**: surface the pre-reduction posterior; do not accept the bias

**Decision: no knowingly-incorrect update.** The `S ⊗ q(parents)` weighting
was never introduced as an approximation — it was proposed as *the* update —
so shipping it as one would be a silent downgrade rather than a documented
trade. It is also worst exactly where the waiting customer lives (per-cell
`R_j` under an uncertain cell), which is the opposite of the profile that
would justify a shortcut.

So `learn` must receive the mode-conditional posterior that
`condition_mixture_switching` computes and `pad_mixture` then destroys. The
remaining question is the mechanism, and the constraint is that it should
read as an independently motivated feature rather than a learning-specific
back door.

**Recommended: an optional structured second return from `infer_states`,
symmetric with `act`'s term tree.** The library already has exactly this
pattern one face over — `infer_policies` returns
`(q_pi, tree if metrics else None)` (`system.py:534-535`), a primary result
plus an opt-in structured record of what happened inside. The inference-side
counterpart is the missing half of that symmetry:

```python
infer_states(params, observations, priors=None, detail: bool = False)
    -> tuple[tuple[Belief, ...], InferenceDetail | None]
```

where `InferenceDetail` carries, per factor and modality, the things
inference computes internally and currently throws away: the pre-reduction
mixture components and their weights, the per-modality mode messages, and
the integrated evidence per bank. Learning's sufficient statistic is then
**one consumer** of a channel that is worth having anyway:

- **Diagnostics**: "which mode explained this observation?" is the first
  question anyone debugging a switching model asks, and today it is
  unanswerable from outside.
- **The notebook** already visualizes mode posteriors by reaching past the
  public faces; this gives it a supported route (and the notebook is an
  adoption gate, so that matters).
- **Mixture-collapse debugging**: seeing what was merged is how you tell a
  `mixture_k` that is too small from a genuinely multimodal posterior.
- It makes D1 requirement 4 (`learn` consumes only what the faces return)
  true again rather than something learning has to work around.

Why not the alternatives: requiring `mixture_schedule="mode_aligned"` for
learning would regress phase 1.5's deliberate generality (the switching path
reads its mode message pre-reduce precisely so it works under any schedule);
and computing the statistic *inside* `infer_states` would put learning
concerns in the inference layer and duplicate the `stats`/`absorb` split D3
already requires.

**Cost:** `infer_states` gains a second return value, which touches its 35
call sites. Mechanical, and pre-release. Follow `act`'s convention exactly —
always return the pair, `None` in the detail slot when `detail=False`, and
state that the beliefs are bitwise identical either way.

### O4 — **RESOLVED**: rename the declaration slots after their factors

**The objection.** In active inference `A` *is* the likelihood parameters —
the CPD tensor itself. So `A=(ConditionalDirichlet(...),)` says "the
likelihood parameters are a Dirichlet", which is a category error: a
Dirichlet is not a CPD. D3's polymorphic slot is semantically at odds with
the letter it fills.

**The representation survives; the name does not.** The only representation
in which `A` strictly denotes the parameters is the sidecar (`A` + `pA`) —
the one D3 deletes for four demonstrated defects. Followed through, the
objection forces a worse library, which means it is properly an objection
to the *name*. The fix is to name each slot after its **factor** rather than
its parameters: a factor may have known or uncertain parameters, and both
are legitimate contents.

| Slot | Proposed | Holds |
|---|---|---|
| `A` | `likelihood` | the observation factor, parameters known or uncertain |
| `B` | `transition` | the transition factor, likewise |
| `D` | `initial_state_prior` | the initial-state prior factor, likewise |

**The objection bites hardest on `D`, which is worth noticing.** For `A`
and `B`, the point and distributional forms are both objects over one index
space (a CPD table), so the tension is mild. For `D`, a `Categorical` *is*
a distribution over states while a `Dirichlet` is a distribution over the
parameters of that distribution — two different kinds of thing in one slot.
Only the factor reading makes that coherent, so `D` needs the rename most.

**This completes an existing convention rather than introducing one.**
`SystemParams` already names the preference designations semantically and
keeps the AIF letter in the comment: `obs_targets` is commented "standing C
designations", `terminal_targets` is "H designations" (`system.py:86-94`).
Letters survived only for `A`/`B`/`D`. Follow the same pattern — semantic
field name, letter in the comment — and add read-only `A`/`B`/`D`
properties for continuity with the literature and with ported pymdp code.

A pleasing consequence: the letter's strict referent becomes exactly what
`point()` returns. `A` stops being a slot you fill and becomes a value you
can compute, which is what the AIF convention always meant by it.

**Cost:** about 160 mechanical sites (34 reads and 3 constructions in
`src/`, ~121 in tests, 4 notebook cells). **Recommended timing: inside
step 1, as its own commit** — that step already rewrites every declaration
site for the union widening and filter audit, so the marginal cost is
small, and pre-release there is no deprecation obligation.

**The env-params classes are in scope** (adjudicated during the build,
reversing a first-pass exclusion). The exclusion rested on current
contents — `GenerativeEnvParams.A` holds the raw lowered table today, where
the letter is exactly right. But `HybridGenerativeEnvParams` passes the
declaration objects straight through, so under D3 a `ConditionalDirichlet`
will legitimately sit in its slot — the same category error one class over.
And the symmetry is package philosophy, not accident: `as_env` is "the same
tensors run forward", so envs built from agent models are precisely the
subset where agent/env isomorphism is the point. Both env-params classes
take the factor names (letters in comments; no alias properties — envs are
not a literature-facing surface). A payoff this leaves room for: a
Dirichlet in the env's likelihood slot with reset-time sampling is
per-episode domain randomization, the environment-side twin of
acceptance 3's draw-then-recover test.

**Explicitly not in scope here:** `A_deps` / `B_deps` / `D_deps`, at 336
sites. Those name a factor's parents rather than its parameters, so they
carry no semantic error — and factor-vocabulary accessors are part of the
generic-layer item (`project-brief.md` question 13), which will be touching
that surface anyway. `log_E` (the policy habit prior) is the one remaining
letter-named field and could follow the same pattern whenever it is next
touched.

## Implementation gaps

Concrete, and covered by none of the sections above.

1. **Tick 0 has no `prev_beliefs`.** With a learnable transition, the first
   tick has neither previous belief nor previous action. Inside `lax.scan`
   you cannot skip the call, so the loop must feed a dummy and pass a
   traced rate — `lr_by_slot={"transition:x": jnp.where(t == 0, 0.0, 1.0)}`. Cover
   this in the demo loop, since every caller hits it.
2. **Initialization is never discussed.** What is α₀ for a learnable `A` —
   uniform `1.0`, Jeffreys `0.5`, or a scaled hand-authored table
   (`8·A`, as `test_objectives.py:170` does)? The choice sets the learning
   rate in practice and decides whether `_wnorm`'s positive-concentration
   mask fires. Needs a constructor
   (`ConditionalDirichlet.from_table(A, strength=)`) and a documented
   default, or every user hand-rolls one.
3. **Missing observations.** `infer_states` requires exactly one
   observation per modality (`strict=True` zips) with no
   "not-observed-this-tick" mask, so any tick with a held-out or dropped
   observation absorbs a fabricated one into the counts, silently. This is
   the ordinary case in deployment. The rate mechanism makes the fix a
   one-liner — a traced `lr_by_slot` value of `lr · observed_m` — so what
   remains owed is only that the loop compute it.
4. **`decay` during a frozen window needs a documented convention.** D7
   justifies decay as standing in for a slow random walk, under which the
   prior should widen whether or not evidence arrives. Because `lr` and
   `decay` are separate rates, both readings are now expressible — `lr=0`
   with `decay` live keeps the walk running; zeroing both freezes
   completely — so state which one the default means. Note the interaction
   with the static fast path: a Python-float `lr=0.0` skips the update
   entirely, which also skips decay, so a frozen-but-drifting slot needs a
   *traced* zero rather than a static one. Say so, or someone will get
   silent non-drift.
5. **`learn` under `vmap`.** Never mentioned. The kernel batches cleanly
   (`_count_update` uses `...`-leading einsum specs and negative axis
   indexing throughout), and the whole inference path already assumes vmap
   rather than native batching (`_observed_log_likelihood` uses `axis=0`),
   so `learn` must inherit that convention explicitly.
6. **Compiled-path materialization becomes per-tick.** `compile_tables`
   hoists the probability tables, entropies, and novelty weights once per
   planning call. With learning on they change every tick, so this is
   intended behavior but a new per-tick cost — and any code caching a
   `PlannerContext` across ticks would silently plan against stale tables.
   Worth an explicit note where the hoisting is documented.
7. **No criterion bounds the `E[log A]` divergence.** D2 accepts a parity
   divergence but never pins it, and a test that merely asserts "differs"
   is worthless. Assert instead that the divergence is **monotone
   decreasing in the concentration total** (since `E[log A] → log E[A]` as
   counts grow) — cheap, and the only test that distinguishes a deliberate
   Jensen gap from a sign error or a wrong digamma.

## Out of scope

GMM-CAVI / indicator CAVI (the next pass, and the reason this one stays
tabular + LG); MNLR / PG-CAVI; BMR and structure learning;
gradient/optax parameter learning; smoothing; learnable LG *transition*
parameters and continuous initial priors (see the symmetry table —
flagged, not forgotten); re-basing the objectives layer on per-variable
info-gain terms (D6's closing note).

## Build plan

Sequenced so the customer is unblocked as early as possible:

1. **The representation** (D1, D2, D3): polymorphic slots, the `Learnable`
   protocol including the new `sample`, `pA` deleted and its novelty sites
   migrated, the classifier/union sites widened **and the positive
   `isinstance` filters audited**, the per-slot rate mechanism, the O4
   rename (its own commit inside this step), `learn`
   face — acceptance 1, 3, 5, 7, 8. This is the piece that must be right;
   everything else hangs off it, and its cost rises with every release
   that ships `pA`.
2. **Learnable transitions + novelty + `decay`** (D4, D6, D7):
   acceptance 2, 6, 9.
3. **The LG families** (D8): inverse-Gamma, then inverse-Wishart, then
   MNIW — acceptance 4. Novelty for a continuous parameter lands here or is
   explicitly deferred with the toggle raising rather than silently
   ignoring.
4. **Learnable initial priors** (D5): tick-0 gating, not an
   episode-end face — acceptance 11.
5. **Demo + notebook section**, then the downstream-side `p_fix` swap
   (acceptance 10, 12).

Steps 1–2 are one agent's brief (they share the declaration unions and
the face signature); step 3 is independently dispatchable once step 1's
protocol is fixed, and its three priors are three sequential commits
rather than three briefs. Per the worktree protocol: every brief states
its expected base SHA and orders a `git log -1` reconcile first.

**Step 1 is the invasive one, and that is the intended trade:** deleting
`pA` touches fifteen source lines, five test files, and around twenty
classifier and filter sites, in exchange for making the three later steps
additive rather than each re-cutting the seam. If that trade proves wrong —
if the filters are more fragile than the audit suggests — the fallback is
to keep `pA` for observations only and accept the sidecar's staleness
question. That is a materially worse library, so it should be a deliberate
retreat rather than a discovery mid-build.
