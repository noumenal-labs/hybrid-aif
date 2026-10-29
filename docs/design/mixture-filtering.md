# Mixture filtering design — workstream C, phase 1

*Status: confirmed (Alex, 2026-07-23; D2 per this section's concrete
unpacking, D5 per the better-model criterion). Grounded in
`de-risk-report.md` §2 (Runnalls/IMM recipe, KT bounds) and
`hybrid-system.md` (whose D3 scope guard this lifts). Build is planned as
a post-reset fan-out; this doc is the agents' brief.*

## The point

v0.2's hybrid System *imagines* mixtures but cannot *perceive* them:
`infer_states` refuses evidence on mixture-carried factors. Closing that
gives the library's thesis end-to-end — a closed-loop agent over a
switching world — and unlocks the true hybrid epistemic term (info gain
about continuous state through observations), which the terms note in
`hybrid-system.md` deferred.

## D1 — Gaussian-sum conditioning (exact, K unchanged)

New kernel `condition_mixture(m, H, c, R_chol, y) -> (MixtureGaussian,
log_evidence)`: per-component Kalman `condition` (existing kernel) plus
the weight update `log w'_j = log w_j + log N(y; H mu_j + c, H Sigma_j
H^T + R)`, renormalized; `log_evidence` is the logsumexp — the integrated
evidence, exact. Conditioning never changes K. Dead (padding) components
stay dead (`-inf + finite = -inf`); the padding-invariance battery
extends to this kernel.

## D2 — Mode-aligned reduction (the IMM convention)

New per-System static: `mixture_schedule: Literal["runnalls",
"mode_aligned"] = "runnalls"` — the choice of how grow→reduce compresses,
defaulting to today's v0.2 behavior (backward compatible). Concretely, on
the switching demo (M=2, k=4) each step grows 4×2=8 components:
`runnalls` merges 8→4 greedily *anywhere*, so a merged component can
blend left- and right-mode children — best compression, fine for pure
imagination; `mode_aligned` partitions the 8 by producing mode and runs
Runnalls *within* each group (4→2 per mode; `mixture_k` a multiple of M),
so slots stay mode-owned forever. That bookkeeping is what filtering
needs: per-mode evidence is the posterior mass of that mode's slots, a
quantity a cross-mode merge destroys. It is a *System static* (not a
per-call flag) because the schedule determines what the carried belief
means — prediction, planning rollouts, and filtering must agree on it.
Cost: worse compression at equal k (cross-mode merges forbidden) —
quantified by the §2(iv)-style ranking-fidelity test, which settled the
de-risk report's unconfirmed-IMM open question empirically: **mode-owned
slots are not free.** On the switching demo (M=2), `mode_aligned` at the
demo's tuned k=4 has bound-midpoint error (0.82 nats) exceeding the
inter-policy gaps and flips mid-table ranks; at k=8 (~2× budget) all
three rankings (mode_aligned, runnalls, exact) agree (errors 0.051 /
0.028 nats). Rule of thumb recorded in the test: under `mode_aligned`,
budget ~2× the runnalls-tuned k. Whole-mixture mean/cov are preserved
either way. No argmin across modes → the gradient-friendliness the
report promised.

## D3 — infer_states over mixtures (one structured sweep)

For an LG modality observing a mixture-carried factor:

- factor posterior: `condition_mixture` on the (predicted) prior — exact.
- if the factor has a declared discrete mode parent AND the schedule is
  `mode_aligned`: the mode posterior updates from bank evidence —
  `q(mode=j) ∝ q_prior(mode=j) · exp(log_evidence_j)` with per-bank
  log-evidence from the mode-j component group, where `log_evidence_j`
  is the bank's **within-group-normalized** evidence `log p(y | mode=j)`
  (build-time correction: the carried mixture weights already encode the
  mode prior mass, so unnormalized bank evidence would double-count it —
  caught in implementation, verified by parity at ~6e-8). One sweep, declaration
  order, documented as structured mean-field (exact for a single LG
  modality on one factor; fixed order otherwise). This is the
  integrated-evidence-up boundary message the brief calls "the duct tape
  that is the product".
- the `NotImplementedError` remains only for genuinely unsupported
  shapes: mixture evidence under `runnalls` scheduling (alignment lost —
  the error says to declare `mode_aligned`), and non-LG modalities on
  continuous factors (MNLR territory, workstream C phase 2).

## D4 — Hybrid state info gain (the epistemic payoff)

With conditioning in hand, the deferred term is cheap: for an LG modality
over a mixture factor, the predictive observation density is itself a
K-component mixture (`H mu_j + c`, `H Sigma_j H^T + R`), so

    state_info_gain = H[q(y)] − E_q(x) H[y|x]
                    = KT-bound-pair( predictive mixture ) − exact Gaussian entropy

an Interval leaf like every other mixture term, registered under the
existing name, block-toggleable, and — per Koudahl — genuinely
policy-dependent only because the mixture is. The closed-loop demo should
show it: an agent that *probes* to disambiguate its mode.

## D5 — Acceptance (binding, same rules as v0.2)

A closed-loop switching demo: noisy continuous observations each tick,
the agent infers its mode, acts, and (with info gain on) demonstrably
probes before committing; ablating info gain removes the probing. Oracle
strategy: exact filtering by enumeration (no reduction) on small cases as
the internal reference; `dynamax`'s RBPF-SLDS as an *external*
cross-check test dependency if its API cooperates (investigate, don't
force). The demo folds in `D_deps` iff the slow regime factor makes a
**better model** of the demo's world — i.e., the story genuinely has two
timescales (a regime setting the mode prior) — never for feature
coverage (Alex's criterion). Notebook gains one section,
re-executed as the adoption gate.

## Guidance — k, semantics, and the exact fallback (build findings, 2026-07-23)

**Why grow-then-reduce at all.** Conditioning on an observation never grows
the mixture: a likelihood multiplies term-by-term, so the measurement update
is the classic responsibilities rule (`condition_mixture`, D1) with K
unchanged. Prediction through *mode-conditioned* dynamics is different: with
the mode unresolved, K components through M dynamics is a K·M-component
mixture — that IS the exact predictive, not an approximation choice — and in
imagination no observations fire, so growth compounds to K0·M^H. Some
compression is therefore forced; grow-exact-then-Runnalls does the exact step
first and compresses with a principled merge criterion (vs GPB1-style
collapse-early, rejected by ablation in the v0.1 demo). Reading: each grown
component is a mode-*history* remnant (anonymous, safe to merge); the mode
*factor* carries the semantic present. `mode_aligned` merges histories only
within the same current mode — which is why bank mass = mode posterior.

**Semantics live in factors, never in components.** Mixture components are
anonymous approximation slots for the continuous conditional; anything with a
name ("which regime", "which hypothesis") belongs in a categorical factor,
where its identity is exact regardless of k. `mode_aligned` is the guarantee
that the approximation never blurs a factor-level boundary; the ~2× budget
cost (D2) is the price of keeping semantics out of the approximation.

**The ~2× is one data point, not a constant.** The cost of mode-owned slots
grows with cross-mode component overlap, mode count, and per-mode complexity
imbalance (equal k/M allocation cannot adapt), and "ranking-faithful" is
relative to the model's inter-policy G gaps. Calibrate per model against the
exact fallback below. Also note the mode-evidence message (D3) inherits
reduction coarseness through the within-bank conditional, so the budget
guidance applies to filtering accuracy, not just planning ranks.

**Exact fallback, in the standard code path.** The merge core forces dead
padding slots to merge first, so `mixture_k = K0 · M^H` makes every reduction
a no-op on live components: propagation is exact through the ordinary
System/planner path (verified on the switching demo vs the independent
enumeration reference: max|G − exact| = 1.9e-6 at k=16, vs 6.5e-2 at k=4).
Exponential in horizon but static-shaped and jit-clean — affordable whenever
M^H is small; use it directly, or to calibrate k where it isn't.

## Phase 1.5 addendum — switching observation models (DRAFT, 2026-07-23)

*Status: confirmed (Alex, 2026-07-23). Surfaced by the library's first
external adopter (the downstream coverage agent, stage 2): localizability
there is cell-dependent fix noise — the switching structure on the
OBSERVATION side, where phase 1 landed it on the transition side. This
is the symmetric completion of v0.2's per-mode `F_j` (SLDS transitions),
which the fix bundle added for B and nobody demanded for A until now.
It is NOT MNLR: given the discrete parent the likelihood stays
linear-Gaussian — conjugate, exact per hypothesis — so it is strictly
easier than phase 2 and does not lift family purity.*

- **E1 — Conditioned-LG declaration.** `LinearGaussianObservation` gains
  discrete-parent conditioning: `A_deps` may name one continuous factor
  plus categorical parents; `H`, `c`, `R_chol` accept leading parent
  axes (broadcast, exactly the per-mode `F_j` convention in
  `LinearGaussianTransition`). Coverage needs only `R_j`; per-mode `H_j`
  and `c_j` cost nothing and cover mode-dependent sensor geometry.

- **E2 — Measurement-grow conditioning.** With the observing mode
  uncertain, conditioning is exact but *grows*: each (component k, mode
  j) pair gets the Kalman condition under `(H_j, c_j, R_j)` with joint
  weight `w_k · q(j) · N(y; H_j mu_k + c_j, S_kj)` — a K·M mixture born
  at the measurement (phase 1's mixtures are born at prediction), then
  reduced back to k. The mode message is read from the grown weights
  BEFORE the reduce, with the phase-1 normalization finding applying
  verbatim (within-hypothesis-normalized evidence, or the prior
  double-counts). Consequence worth stating loudly: because evidence is
  read pre-reduce, observation-side switching needs NO mode-owned
  persistent slots — the reduce can follow the System's declared
  schedule, `runnalls` included, at modest k. (Phase 1's
  `mode_aligned` + k-multiple-of-M constraint is only needed when the
  TRANSITION is mode-conditioned and per-mode mass must survive between
  steps; for coverage, M = number of cells ≈ 36, and that constraint
  would have been prohibitive.)

- **E3 — infer_states + the info-gain term.** One-sweep threading as in
  D3: grow-condition, mode message pre-reduce (exact), reduce to k. A
  Gaussian-family factor observed by a switching modality must be
  declared mixture-family (mixture_k set) for filtering — loud
  validation. D4 generalizes cleanly: the predictive observation density
  is the (K·M)-component mixture over (k, j); the conditional entropy is
  exact, `Σ_j q(j) · ½ log|2πe R_j|`. Planning-side special case (the
  coverage "refix" payoff): when the discrete parent's belief along an
  imagined rollout is concentrated (cell trajectories under a candidate
  policy are near-deterministic), the term collapses to the exact
  closed form with per-step `R_cell` — so "go re-fix when uncertain"
  emerges from the EFE with no mixtures in imagination at all.

- **E4 — Acceptance (binding, D5 rules).** The downstream coverage agent
  stage 2 is the demo: the v1 σ-gate wrapper (refix/escalate) removed or
  demonstrably redundant, refix emerging from the objective; closed-loop
  filtering under genuine cell uncertainty vs the exact-enumeration
  oracle on small grids. In-repo: exactness vs brute force, the padding
  battery extended to the grow-conditioning, per-mode-broadcast tests,
  and filtering fidelity at the chosen k. Effort: one build agent for
  the library side; coverage stage 2 rides downstream afterwards.

## Out of scope (phase 2+)

GMM-CAVI (learning-side), MNLR/PG-CAVI (discrete-given-continuous
likelihoods — lifts family purity), smoothing, soft-merge gradients
through Runnalls, RBPF sampling path.

## Build plan (updated 2026-07-23)

D1–D4 are landed (`d42f685`, 507 tests; findings folded into D2/D3 and
the Guidance section above). D5 — the closed-loop probing demo, notebook
section, and the `-m notebook` execution-gate test — is in flight.
Remaining dispatches, in order:

1. Phase 1.5 (once the addendum is confirmed): one build agent for
   E1–E3 + the in-repo half of E4 — conditioned-LG declaration,
   measurement-grow conditioning with the pre-reduce mode message,
   infer_states/info-gain threading, exactness + padding + fidelity
   tests.
2. Coverage agent stage 2 (downstream-side, imports 1.5): replace the v1
   σ-gate wrapper with the modeled pose — the binding E4 acceptance.
