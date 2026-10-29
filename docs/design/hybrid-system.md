# Hybrid System design — v0.2 addendum

*Status: confirmed (Alex, 2026-07-22; review resolutions folded in). Extends
`system.md` (confirmed 2026-07-21); nothing here revises a confirmed
decision — it generalizes the concrete `System`/`SystemParams`, which
v0.1 hardwired to `ConditionalCategorical`, to the typed factor menu the
design always intended. Decisions D1–D5 confirmed in-session 2026-07-22.*

## Scope

**In:** typed per-factor families (categorical / Gaussian / mixture) for
transitions, observations, and initial priors; mode-coupled mixture
transitions with a static Runnalls budget; hybrid `planner_context`
closures behind the same faces; per-block state conditioning (discrete
FPI + linear-Gaussian `condition`); hybrid forward sampling for `as_env`;
the two demo rewrites as binding acceptance tests.

**Out (scope guards):** mixture *filtering* against evidence (IMM /
GMM-CAVI — workstream C; `infer_states` raises `NotImplementedError`
naming C for mixture factors); hybrid `state_info_gain` (needs the
mixture-KL machinery decisions; `state_entropy` is the epistemic term
both acceptance tests use); `ambiguity` beyond discrete modalities; any
compiled fast path for the hybrid closures (workstream L, only if
measured to matter).

## D1 — Typed factors, dispatch by declaration type

`System` statics are unchanged (`A_deps`, `B_deps`, `num_controls`,
`controlled`, `policy_len`, planner name, toggles, names) plus one new
static: `mixture_k: int | None` — the Runnalls budget shared by all
mixture factors (per-factor budgets only if a real model ever needs
them). No family-tag fields: **the family of each factor/modality is the
type of its declaration in `SystemParams`**, inspected at
`planner_context` build time (pytree structure is static under jit, so
`isinstance` dispatch is trace-legal). Discrete-only Systems construct
and behave byte-identically to v0.1, compiled fast path included.

New declaration types (in `src/hybrid_aif/factors.py`, the node layer's
first concrete module — these are `Stochastic` Factor payloads in the
`system.md` tagged-union sense):

```python
class LinearGaussianTransition(eqx.Module):
    """x' = F x + b_j + eps_j: shared linear map, conditioned innovations.

    ``innovations`` is a ConditionalGaussian whose parent tuple is
    (*discrete B_deps parents, control-if-controlled) — hill declares
    (control,), the switching model (mode,).
    """
    F: Array                       # (d, d) — Kalman-standard name, kept
    #   deliberately: free energy never appears as bare F in this codebase
    #   (EFE is G), and T would collide with the horizon length.
    innovations: ConditionalGaussian


class LinearGaussianObservation(eqx.Module):
    """y = H x + c + eps: an LG modality over one Gaussian factor."""
    H: Array                       # (d_obs, d)
    c: Array                       # (d_obs,)
    R_chol: Array                  # (d_obs, d_obs)
```

`SystemParams` fields widen (no new fields):

- `B[f]`: `ConditionalCategorical | LinearGaussianTransition`
- `A[m]`: `ConditionalCategorical | LinearGaussianObservation`
- `D[f]`: `Categorical | Gaussian | MixtureGaussian`
- targets/pA/log_E: unchanged (`risk` is already family-generic).

Belief per factor: `Categorical | Gaussian | MixtureGaussian`, matching
`D[f]`'s family. The hybrid planner path carries **distribution objects**
through the scan (the seam is opaque); only the discrete-only path uses
the compiled raw-prob representation.

## D2 — Mode-coupled transitions and within-step order

A continuous factor's `B_deps` may name discrete parents. Transition
semantics per family, per step:

- categorical: `expected_states` contraction (v0.1, unchanged);
- Gaussian, no discrete parents: `predict(q, F, b(u), chol(u))` — exact;
- Gaussian or mixture with discrete parents: the switching composition —
  mode weights are the parent's **post-transition belief this step**,
  then exact grow through all M mode-conditioned transitions, then
  Runnalls `reduce(mixture_k)`.

**Within-step ordering rule:** factors are computed in declaration order,
and a factor's `B_deps` may only name earlier factors (self excluded for
continuous factors; discrete factors keep v0.1 semantics). Validated at
`_validate` time — parents-before-children or a loud error. This is
exactly the demo's semantics: `ACTION_MODE_LOGW[u]` is the mode factor's
`B` (parents = control only), consumed by the continuous factor in the
same step. No auto-toposort — silently diverging declared vs execution
order is the worse footgun — but the error message must *compute and
print* a valid order, so the fix is copy-paste. No `Schedule` abstraction:
per-step weight schedules were an enclosure-level workaround that the
System composition subsumes (weights come from the mode factor's belief
each step); within-step cycles / iterative message passing is the
deferred factor-graph question, not a v0.2 type.

**Static shapes:** a mixture factor's belief always carries exactly
`mixture_k` components. `D` and `infer_states` outputs are padded to
`mixture_k` with zero-weight components on entry to planning; grow→reduce
returns exactly `mixture_k`. (K0·M ≤ mixture_k pads instead of reducing —
`reduce` already handles k ≥ K as a no-op-plus-pad or we pad explicitly;
the agent verifies which and tests both directions.)

**Padding-invariance test battery (required, agent 1):** zero-weight
padding has two real hazards — `-inf` log-weights can poison *gradients*
(NaN through logsumexp branches even when values are correct), and the
Runnalls criterion sees zero-weight components as zero-cost merges. So:
for a reference mixture vs its zero-padded version, assert (tolerance)
identical entropy bounds, `kl_to` bounds, risk leaves, `moment_match`,
and `reduce` output, and assert finite gradients through entropy and
`kl_to` w.r.t. means/chols/log-weights of the padded object. If `-inf`
fails any of these, use a finite floor (e.g. `-1e30`) and test that.

## D3 — infer_states (v0.2 scope)

Per-block conditioning, one mean-field sweep, evidence tuple typed per
modality (`int32` index for categorical, float array for LG):

- categorical factors ← discrete FPI over the categorical modalities
  (v0.1 path, restricted to the discrete sub-model);
- Gaussian factors ← `condition` per LG modality that observes them
  (sequential over modalities; exact for one, documented fixed order for
  several);
- mixture factors ← `NotImplementedError` naming workstream C.

Cross-family explaining-away (a categorical modality whose deps include a
Gaussian factor, or vice versa) is rejected at `_validate` with a clear
message — v0.2 supports family-pure modalities only.

`update_empirical_prior` pushes each factor through its own transition
(the hill demo's `predicted`), same face, family-dispatched.

## D4 — Objectives

- `risk` / `terminal_risk`: already family-generic via `kl_to`
  (mixture leaves arrive as KT bound pairs automatically).
- `state_entropy`: becomes a System-managed term with the standard
  block-grain toggle `include_state_entropy: bool | tuple[str, ...] =
  True` — scored for every continuous factor block unless toggled off.
  It is the hybrid epistemic term both acceptance tests use (and the
  quantity whose per-policy spread the notebook's section 3 displays).
- `utility` / `state_info_gain` / `param_info_gain` / `ambiguity`:
  discrete modalities only, unchanged.

*Terms note (why entropy, not info gain, for hybrid blocks in v0.2):*
`state_info_gain` is I(s; o) under the predictive distribution — expected
entropy *reduction* from the next observation, H[q(s)] − E_o H[q(s|o)] —
and needs an observation model; `state_entropy` is H[q(s)] itself —
uncertainty-aversion (KL-control lineage), no observation channel needed.
The demos' imagination is open-loop (no A in the rollout), where info
gain is identically zero and entropy is the honest epistemic handle; it
is also the term Koudahl proves constant in LG. Identity worth keeping:
risk + entropy = KL(q‖C) + H[q] = −E_q[ln C] — expected preference.
Hybrid info gain (expected entropy reduction, KT bound pairs) becomes
computable once workstream C lands mixture filtering, and joins as a
separate registered term then.

## D5 — Acceptance tests (binding)

Two demo rewrites, each an agent in a worktree, each reporting:
line count before/after, a clarity self-assessment the human can audit,
and numerical parity (tolerance where composition order changes,
bitwise where pure refactor). `hill_lg.py` should lose `HillModel`,
`make_context`, `tick`, and `env_step`/`observe` (via `as_env`) to
`System` + designations; `hybrid_switching.py` should lose its module-level
model constants and hand-rolled scoring loop to a `System` with an
injected 6-policy planner (showcasing the injection seam). **If either
rewrite is not strictly shorter and clearer, stop and report — System
gets rethought, not defended.**

## as_env

Hybrid forward sampling: ancestral per family in declaration order —
categorical sample, then `x' = F x + b_j + eps` with `j` the sampled
discrete parent (exact, no mixtures on the env side — the env holds a
point state). Extends `GenerativeEnv` or a sibling — agent 1's call; the hill
rewrite's `env_step`/`observe` must come out of it.

## Build split (phase 2/3)

- **Agent 1 (core):** `factors.py`, `SystemParams` widening, hybrid
  `planner_context` closures + within-step ordering + validation,
  `infer_states`/`update_empirical_prior` dispatch, static-K padding,
  `as_env` hybrid sampling; tests incl. a hand-checkable 1-D hybrid
  fixture and discrete-regression (v0.1 T-maze suite untouched and
  bitwise on the compiled path).
- **Agent 2 (machinery, parallel, disjoint files):** per-step
  transition-weight schedules in `rollout_reduced`; Gaussian↔Mixture KL
  convenience (both logged by the v0.1 demo agents as gaps).
- **Agents 3 ∥ 4 (after core):** the two acceptance-test rewrites.

## Amendments (landed 2026-07-23)

- **`System.predict_beliefs(params, beliefs, policy)`** — open-loop
  imagination as a face: per-step predicted beliefs under one policy, the
  same per-step transition the planner and `update_empirical_prior` use.
  Closes the last acceptance-audit seam-dip (belief trajectories reaching
  into `planner_context().transition_fn`).
- **`D_deps` (hierarchical conditional priors)** — `System.D_deps` names
  each factor's initial-prior parents (earlier, tabular); `D[f]` may then
  be a child-first `ConditionalCategorical`. `initial_priors` resolves
  ancestral mean-field marginals (used by `infer_states` when
  `priors=None`); `as_env` samples the *joint* ancestrally. A fully
  tabular hierarchical model classifies **discrete** (compiled fast
  path), dissolving the interim mis-routing; planning faces require
  marginal beliefs and point at `initial_priors` otherwise. Conditional
  priors on continuous families remain out of scope.

## Amendment: LG modality C designations (landed 2026-08-19)

D4's "`utility` / `ambiguity`: discrete modalities only" no longer
holds. `obs_targets` accepts a `Gaussian` or `MixtureGaussian` over an
LG modality's **observation space** (covariance-as-tolerance, the
`state_targets` reading), and the leaves join the existing terms keyed
by modality name — no new term names, so `total`'s signs already cover
them. Origin: friction log finding 6, two customer asks (the mission
model's "prefer observing my pose at the committed site" — a switching
modality with per-mode offsets and a zero-centred target — and its
coverage sibling), whose `state_targets` workaround needed a
calibration dance with two measured failure modes.

- `q(o)` is the LG pushforward of the belief,
  `objectives.continuous.predictive_obs`: `N(Hμ_k + c, HΣ_kHᵀ + RRᵀ)`
  per belief component, times one per mode configuration when the
  modality switches — the same `(K·M)` predictive the filtering side
  integrates against a datum, read here in imagination.
- `utility_infogain` → `utility[m] = E_q(o)[log C]`, as `−CE(q(o), C)`
  via the new `MixtureGaussian.cross_entropy_to` (which `kl_to` is now
  built from). **Exact** for a Gaussian target however many components
  `q(o)` has — linearity, one closed form per component — and a genuine
  Kolchinsky–Tracey Interval for a mixture target.
- `risk_ambiguity` → `risk[m] = KL(q(o) ‖ C)` through the unchanged
  `kl_to` dispatch, **plus** a new `ambiguity[m] = E_q(x)[H[y|x]] =
  Σ_j q(j)·½log|2πe R_jR_jᵀ|`, mounted for every LG modality exactly as
  the discrete builder mounts one per tabular modality (a channel's
  ambiguity is a property of its likelihood, not of any designation).
  That leaf is what the decomposition was missing.
- **Coherence.** With the channel's `infogain` leaf mounted, the two
  decompositions fold to the same G *at both interval endpoints* — the
  CE and entropy pieces are literally shared between the leaves. The one
  documented wedge is `kl_to`'s clamp at zero, reachable only when both
  the CE and the entropy are bound pairs (mixture `q(o)` *and* mixture
  `C`).
- **Which form à la carte** (friction log 11-13, measured in the
  field). Risk-family scoring carries `−H(q(o))` — the **certainty
  tax**, penalizing a belief for being *tighter* than the target's
  tolerance — and is coherent only inside the full `risk_ambiguity`
  decomposition, where the channel's ambiguity and info-gain leaves
  return it. The **utility** form is the correct à-la-carte piece:
  `E_q(o)[log C] = const − (bias² + Var_q)/2σ_C²` per dimension for a
  Gaussian target, monotone-improving as the belief sharpens, never
  taxing precision; epistemic value then enters *only* through
  explicitly mounted `infogain` leaves. This is D4's identity
  `risk + entropy-leaf = −E_q[ln C]` read at the observation instead of
  the state — the customer measured the tax alone moving an escalate
  crossover from attempt 4 to attempt 8 at the state-target level.
- Scope caveat: a **Gaussian-carried** factor's LG channel is not
  channel-eligible (V2 gives it the entropy lowering instead, the
  LG-degeneracy decision), so for that configuration the two
  decompositions differ by the unmounted channel info gain — a
  pre-existing eligibility consequence, unchanged by this work.
- Still fenced: `obs_targets` on a **neural** (`MLPConditional`)
  modality. A net has no closed-form pushforward of the belief, so
  there is nothing to score the target against until the amortized
  predictive lands (`amortization.md` A1); the error now says so
  instead of citing the retired v0.2 restriction.

## Amendment: the family-purity rule is current (2026-09-18)

D3's "v0.2 scope" heading reads as superseded history to a reader of the
v0.7 library; it is not. Cross-family explaining-away — a categorical modality
reading a Gaussian factor, or a linear-Gaussian modality reading a categorical
one — is still refused by `objectives.hybrid.validate_structure`, and the
error messages now say so without the version stamp. The two routes that work
are recorded for composers in `../composing-systems.md` (bin the features; or
carry a `MixtureGaussian` factor switched by the discrete one, which is the
mixture likelihood the direct declaration would compile to). Making that
direct declaration first-class is a roadmap candidate in `../project-brief.md`
(flagged 2026-09-18). Nothing in D3 or D4 is otherwise changed.
