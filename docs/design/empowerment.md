# Empowerment and the span grammar — scoping (E-round)

*Status: **scoped, unadjudicated** (Alex + Fable, 2026-09-02). No build
dispatched. Four decisions (E1–E4) are open below; everything else is
the map. This doc exists because the empowerment row was scoped twice
in conversation and nothing survived into the repo; it is also the
designated **second customer** that opens `register_estimator`
(`objective-terms.md`, "The open registration API", adjudicated
2026-08-20).*

## The row, placed on the lattice

Empowerment at a state is a channel capacity:

```
Emp_k(s_t) = max_ω I( a_{t:t+k} ; X_T | s_t )
```

— the mutual information between an action sequence and a future
outcome, maximized over an **open-loop source distribution** ω on
action sequences. On `objective-terms.md`'s three axes:

- **functional**: mutual information — already in the closed set of
  four. Capacity is MI evaluated at the maximizing source, and both
  estimator families below perform that maximization internally
  (Blahut–Arimoto *is* alternating maximization; the variational
  estimator trains ω to maximize a bound). The precedent is
  `ensemble_disagreement`, which is "MI, estimated by MC over an
  ensemble": the maximization is an estimator property, not a fifth
  functional. The naming rule gains one sentence: `empowerment` names
  MI ∘ argmax-over-source, with the argmax delegated to the row's
  derivation (tabular) or trainer (amortized).
- **sign**: −1 (value).
- **conditional**: the k-step **composed channel**
  `p(X_T | a_{t:t+k}, s_t)` — see the next section, because this is
  where the row is genuinely new.

Doctrine check: `objective-terms.md` keeps policy entropy out of the
tree because "the tree scores the model, and the policy is not part of
the model." ω is **not the policy** — it is row-owned scoring
machinery, living in `SystemParams.scoring["empowerment"]` under A11's
rule (the registry names the strategy, the slot holds its parameters,
exactly like the critic and the ensemble). Empowerment does not reopen
that adjudication.

## The two genuinely new pieces

**1. Spans, and the composed conditional.** Every shipped row reads a
single declared kernel. Empowerment's conditional is a *path through*
the declared kernels: k transitions composed, optionally capped by a
likelihood channel when the outcome is an observation. Nothing new is
declared — the composition is derivable from the model — but axis 1 of
the lattice gains a new kind of value. The unit is a **span**:

```
span = (variable-or-modality, contiguous time window)
```

and the composed conditional links an action span to an outcome span,
`p(outcome-span | action-span, s_t)`. This is exactly the generality
the row needs: outcomes may be state factors or observation modalities,
at any single index or contiguous range on either side, with classical
empowerment the point `outcome = obs @ t+k`. An outcome *range* means
MI with the **joint** over the window — never a sum of per-step MIs,
which would double-count the inter-step information the same way the
occupancy sign-flip would have (`objective-terms.md`, "What this
already caught").

**Contiguous windows only** (Alex, 2026-09-02): disjoint index sets are
expressible in the same grammar but deferred indefinitely; the tabular
variant raises `NotImplementedError` naming the unsupported span shape,
the refusal type the registry already uses for a missing
`(row, family)` variant.

**2. Mount-time row parameters.** Rows so far are selected by
`name:block` with nothing to configure; `empowerment` needs
`(k, outcome, at)` chosen at mount time. Small in code, but the first
row whose term identity has arguments, so the grammar choice sets
precedent. Two requirements regardless of syntax: the leaf key carries
the parameter signature (so two empowerment rows at different horizons
co-mount without colliding), and the registered sign still belongs to
the term name alone.

## What the same pieces buy (the generality audit)

The kit is: **span designation × composed conditional × the four
existing functionals × sign × optional source ownership**. Points
reachable without any further machinery:

- **`empowerment`** — MI(action-span; outcome-span), source-owned, −1.
- **Influence rows** — MI between spans of two *different* variables
  with nobody owning a source (evaluated at the model's own
  marginals): intrinsic-control / social-influence objectives are this
  coordinate.
- **Reachability / goal probability** — cross-entropy vs a target on a
  *future* span: "maximize the conditional probability linking two
  spans" is `utility`'s functional mounted at a span instead of at the
  per-step channel.
- **k-step predictive info gain** — the existing `infogain` functional
  at an offset outcome.
- **MOP's path reading** — expected conditional entropy of the composed
  transition over a window is the *path*-entropy form of MOP; the
  shipped `occupancy` row is its per-step term. Observation only.

**And the H observation (recorded, no action — see E4):** the
`terminal_targets` designation (H, with `H_deps`) is a declared target
object with its own conditioning structure — *not* sugar, it stays.
But the **row** `terminal_risk` is, in span terms, just `risk` against
the H target mounted `at = last`: the bespoke "at horizon end" time
pinning dissolves into the grammar as one point of it. Nothing
migrates now; the observation is recorded so that when the span
grammar exists, `terminal_risk`'s special-casing is understood as
sugar rather than as a fifth kind of row.

What the kit does **not** buy, stated so the boundary is honest:
anything needing a fifth functional, and policy entropy (which stays in
the trainer per the adopted doctrine).

## Estimators — one row, two family variants

Dispatch by `registry.row_variant`, the occupancy precedent (ledger 2's
mechanism).

**Tabular — exact, by Blahut–Arimoto.** Compose the posterior-**mean**
B tables k steps from the current belief (mean tables per D2's
coherence rule: the novelty leaf owns `I(·; θ)`, the same argument as
tabular occupancy), optionally contract through A, and run
Blahut–Arimoto on the resulting `|A|^k × |X|` channel. BA iterations
are monotone lower bounds on capacity with a computable optimality gap
(the classical sandwich stopping criterion), so a fixed iteration
count — which jit wants anyway — is honest. `|A|^k` is the real limit:
fine for the T-maze at k ≤ 3–4, which is all the vignette needs.
Acceptance: brute-force capacity on small random channels, and the gap
asserted under the chosen iteration count on the acceptance worlds.

**Amortized — the variational lower bound** (Barber–Agakov, as used by
Mohamed & Rezende 2015; Karl et al. 2017 for continuous control):

```
I(a-span; X_T)  ≥  H(ω)  +  E_{ω, channel}[ log q_ξ(a-span | X_T, s_t) ]
```

with learned source ω and learned planning decoder q, both trained to
**maximize** the bound — the training *is* the capacity maximization.
Runs over imagined rollouts where `ensemble_disagreement` runs
(`imagine`'s `aux_beliefs`-keyed-by-row seam was built for this shape);
outcome-as-state reads the latent code, outcome-as-modality decodes
through the declared `PredictorHead`. `derived=False`. Parameters in
`scoring["empowerment"]`; `Estimator.trainer` gates the key (ledger
6's mechanism), so mounting without the trainer raises at the key.
Acceptance: transcription-style test of the bound against a straight
NumPy implementation on random inputs.

## `register_estimator` goes public here

Per the adjudicated sequencing (2026-08-20: "open the registration API
deliberately when there is a second customer"), the amortized
empowerment variant registers **through the public path**, as its own
acceptance test of the API. The two commitments and the honest price
(`derived` never defaulted for user rows; a custom row means G bounds
nothing) are already written in `objective-terms.md` and carry over
unchanged.

## Open decisions

- **E1 — how generic to build now.** Options: (a) the span kit as a
  real internal representation (span designation + composed-conditional
  builder), with only the points empowerment and the vignette need
  actually implemented and every other coordinate raising the
  `(row, family)`-style refusal; (b) empowerment-specific plumbing with
  the grammar merely reserved. Recommendation: **(a)** — the audit
  above shows at least three further rows on the same pieces, and (b)
  would re-derive the kit per row.
- **E2 — row-parameter grammar.** Concrete syntax for
  `(k, outcome, at)` in `terms=` / the mount expression; leaf-key
  signature rule for co-mounted parameterizations. Needs a proposal in
  the discussion round; constraint: sign stays on the bare term name.
- **E3 — the trust label for certified-but-inexact estimators.**
  BA truncated at fixed N is neither exact arithmetic (`derived=True`
  overclaims) nor unwarranted (`derived=False` discards a real
  certificate): every iterate is a **lower bound on capacity with a
  computable optimality gap** (the sandwich criterion). The boolean
  cannot say that, and this row is the first where the loss is
  material. Proposal (Alex, 2026-09-02, refined): a `guarantee`
  vocabulary on the **registry row** —
  `{"derived", "bound" (with direction), "asymptotic", None}` — where
  `"bound"` is a directional bound with a finite-N certificate (BA;
  also the Barber–Agakov row, direction known, gap uncontrolled →
  `"asymptotic"` is arguable there) and `"asymptotic"` is consistency
  with no finite-N statement. Placement per A4's rule (the trust label
  lives on the one reviewed registration line): a field on
  `Estimator`, alongside `trainer`. **`Interval.derived` stays exactly
  as A12 shipped it** — the boolean fold flag, now defined as
  `guarantee == "derived"`, AND-propagation untouched, no leaf or fold
  migration. Direction is recorded because it composes with the term's
  sign (a leaf under-estimating a value term makes G conservative) —
  but any *quantitative* fold of gaps through G is interval arithmetic
  and explicitly deferred; payloads (iteration count, tested gap) live
  in the registration and its tests. Under this proposal E3 dissolves:
  tabular BA registers `guarantee="bound"` (lower, gap asserted at the
  chosen N on acceptance worlds) and its leaf is honestly
  `derived=False`. Alternative if the vocabulary is declined: the
  original boolean dilemma, with `derived=False` as the conservative
  pick.

  *Scope note (Alex, 2026-09-02): the `guarantee` vocabulary is a
  **repo-wide** proposal about the trust system, not an empowerment
  decision, and is adjudicated on its own — "consider it in the
  context of what's best for the repo overall, not just this
  addition." The composition analysis (same discussion) settles the
  shape of that adjudication. On the chain
  `derived > asymptotic > None` the fold is a plain meet
  (`derived ∧ asymptotic = asymptotic`: exact leaves contribute zero
  error). But `bound` breaks the chain: what G inherits is the
  direction **after per-leaf sign adjustment**, so same-direction
  bounds fold to a directional bound on G, mixed directions collapse
  (to `asymptotic` if both bounds are convergent, else `None`), and
  `bound ∧ asymptotic` loses the direction. The honest enum fold is
  therefore a small algebra with real ways to be confidently wrong —
  and the boolean is precisely its **largest trivially-foldable
  quotient**: exactness is closed under signed sums, so AND is its
  complete and always-correct propagation rule. Standing
  recommendation: `Interval.derived` stays the folded flag (defined as
  `guarantee == "derived"`, never independent state); `guarantee` —
  `{"derived", "bound" (direction, convergent?), "asymptotic", None}` —
  lives on the registry row per A4's reviewed-line rule, for humans,
  docs and diagnostics; folding the enum is deferred until a consumer
  wants "G is a directional bound" as a runtime fact badly enough to
  pay for the direction algebra. Adopting the vocabulary invites a
  one-time audit re-registering every existing row. **Empowerment does
  not wait on any of it**: E3-for-this-chunk is simply `derived=False`
  for tabular BA, with the registration upgraded to
  `guarantee="bound"` if and when the vocabulary lands.*
- **E4 — the H observation.** Adopt as recorded observation only (this
  doc + one pointer in `objective-terms.md`), no migration of
  `terminal_risk` or the H designation. Recommendation: yes, and revisit
  only if the span grammar's existence makes the special-casing cost
  something.

## Cost and shape

One chunk, discussion round first: (a) adjudicate E1–E4; (b) tabular
variant + BA with exactness tests; (c) amortized variant + trainer;
(d) `register_estimator` public with empowerment as its acceptance;
(e) the vignette column — with a ready-made payoff: the MOP paper
positions itself *against* empowerment explicitly, so
`examples/objectives_a_la_carte.ipynb` would then run, inside one
declared model over the same beliefs, the comparison that paper drew
on paper. Rough size: the occupancy row plus the ensemble trainer,
combined.

## References

- Klyubin, Polani & Nehaniv (2005), "Empowerment: a universal
  agent-centric measure of control" — the objective.
- Salge, Glackin & Polani (2014), "Empowerment — an introduction" —
  survey, exact computation and its limits.
- Mohamed & Rezende (2015), "Variational Information Maximisation for
  Intrinsically Motivated Reinforcement Learning" — the variational
  lower-bound estimator.
- Karl et al. (2017), "Unsupervised real-time control through
  variational empowerment" — continuous-control form of the bound.
- Blahut (1972); Arimoto (1972) — the capacity algorithm and its
  monotone bounds.
- `objective-terms.md` — the lattice, the staged registration API, the
  MOP-vs-empowerment comparison hook (its "Provenance" section).
- `parameter-learning.md` D2; `amortization.md` A4, A11;
  `amortization-debt-ledger.md` entries 2, 6 — the rules cited above.
