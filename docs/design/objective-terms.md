# Objective terms — the lattice under the term tree

*Status: adopted framing (Alex + Opus, 2026-08-20). Arrived sideways: the
amortization program's chunk 2b needs R-AIF's term 7c (`wment`), which
**rewards** state entropy, and the question "ship it plainly or scope it
replication-only?" turned out to be answerable only by writing down what
the objective layer's rows already have in common. No API changes are
adopted here beyond the one `occupancy` row 2b builds; the rest is the
map, recorded because the map changed the build (see "What this
already caught").*

## Why this doc exists

The term tree (`objectives/base.py`) is a two-level mapping
`term_name -> block_name -> Interval`, with each term name carrying a
registered sign in the fold to G. That is a bookkeeping contract, not a
theory of what a term *is* — so every new row has arrived as a
one-off judgment about whether it is "kosher," and the answers have been
inconsistent in tone: `ambiguity` is unremarkable, an entropy *bonus*
felt like a hack. It isn't. The rows we ship are already points in a
small lattice, and a row's respectability is a property of where it
sits, not of which literature it came from.

## The three axes

Every leaf in the term tree is fixed by three choices:

1. **Which declared conditional** it reads — a likelihood channel
   `p(o|s)`, a transition `p(s'|s,a)`, a recognition factor `q(x|·)`,
   a parameter posterior, the policy `π(a|s)`.
2. **Which functional** of that conditional it computes.
3. **Which sign** it takes in G (`+1` cost, `-1` value).

The functional vocabulary is closed, and small:

| functional | form | notes |
|---|---|---|
| marginal entropy | `H[q(child)]` | the belief's own spread |
| expected conditional entropy | `E_parents H[p(child\|parents)]` | the kernel's own spread |
| mutual information | `H[q(child)] − E H[p(child\|parents)]` | the difference of the two above |
| cross-entropy vs a target | `−E_q[log C(child)]` | the only one needing a designation |

Two composites are worth naming because we ship them and they are
*not* primitives: `KL(q ‖ C) = cross-entropy − marginal entropy` (which
is why `risk` carries the certainty tax — the `−H(q)` inside it
penalizes a belief tighter than its target), and the info-gain
identity `risk + ambiguity = −(utility + state info gain)`, which is
what makes the two shipped decompositions duals rather than rivals.
*(Qualified by `per-variable-infogain.md` V11, 2026-10-18:)* the identity
holds per channel wherever the channel's epistemic leaf is mounted, a channel
without a designation carrying an implicit flat preference (risk `−H[q(o)]`,
the normaliser dropped); with everything selected the two forms give one G,
and where a channel's leaf is ablated they differ by exactly its `H[q(o)]`.
On a Gaussian-carried factor read by several channels the joint risk is
`Σ_m KL(q(y_m) ‖ C_m) + TC(Y)`, the joint ambiguity `Σ_m E H[y_m | x_S] −
TC(Y | x_S)` and the joint information `Σ_m I(y_m; x_S) − (TC(Y) − TC(Y |
x_S))`, so the readings' total correlation is its own term,
`observation_correlation` (risk side), and the interaction information
`TC(Y) − TC(Y | x_S)` another, `observation_interaction` (epistemic side;
under `risk_ambiguity` it holds the ambiguity's `−TC(Y | x_S)`, zero on the
whole factor). Both are costs; G is the joint expected free energy in both
forms, and the two agree under full selection.

## Every shipped row, placed

| term | conditional | functional | sign | leaf key |
|---|---|---|---|---|
| `utility` | likelihood channel | cross-entropy vs C | −1 | modality |
| `risk` | state or channel | KL vs target (composite); a per-step state target inside a branching planner: cross-entropy vs target (V11) | +1 | block |
| `terminal_risk` | state, at horizon end | KL vs target (composite); inside a branching planner: cross-entropy vs target (V11) | +1 | factor |
| `ambiguity` | likelihood channel | expected conditional H | **+1** | modality |
| `infogain` (`predictive`) | likelihood channel | mutual information | −1 | channel |
| `infogain` (`posterior`, planned) | state | mutual information | −1 | variable |
| `infogain` (`wnorm`) | parameter posterior | mutual information | −1 | `slot:name` |
| `infogain` (`entropy_lowering`, **retired by V11**) | state | marginal entropy | −1 on a `−H` leaf ⇒ `+H` in G | factor |
| **`uncertainty`** (V11) | **state** | **marginal entropy** | **+1** | factor |
| **`observation_correlation`** (V11) | **likelihood channels jointly, on one Gaussian factor** | **total correlation `TC(Y) = Σ_m H[q(y_m)] − H[q(Y)]` (marginal entropies less the joint's)** | **+1** | factor |
| **`observation_interaction`** (V11) | **likelihood channels jointly, given a state block** | **interaction information `TC(Y) − TC(Y \| x_S)` (under `risk_ambiguity`: `−TC(Y \| x_S)`), signed: > 0 redundancy, < 0 synergy** | **+1** | factor |
| `infogain` (`ensemble_disagreement`, 2b) | transition (sampled ensemble) | mutual information (MC) | −1 | variable |
| **`model_uncertainty` (2c)** | **transition (sampled ensemble)** | **mutual information (MC)** | **+1** | variable |
| **`occupancy`** (2b: neural; 2026-08-31: tabular, closed form) | **transition** | **expected conditional H** | **−1** | `slot:name` |
| `occupancy` over the policy (**not in the tree**) | policy | marginal entropy | −1 | — see below |

The bolded rows are the new ones, and they are what this doc was
written to justify. *(V11, 2026-10-18:)* `entropy_lowering` was filed under
`infogain` without being information gain; its leaf moved to the opt-in
`uncertainty` row, the same marginal entropy stored as `+H` under a `+1` term,
a sibling of `occupancy` in the entropy family: the opposite sign on a
different quantity, the belief's entropy rather than the transition's
expected conditional entropy. `observation_correlation` and
`observation_interaction` are the lattice's first leaves over several
channels at once; they are what make a per-channel risk and a per-channel
`infogain` (and ambiguity) exact on a Gaussian factor read by overlapping
sensors. The correlation is a marginal-entropy composite on the joint
reading, a property of the predicted readings alone, so it sits with risk and
the selector never touches it; the interaction is a difference of
conditional total correlations, the same functional as the mutual
information it corrects, and it follows the selection. On the whole factor
`TC(Y | x) = 0` (the sensors' noises are independent given the state), so
the interaction is the total correlation itself and is never negative; on a
sub-block `x_S` it is `Σ_m I(y_m; x_S) − I(Y; x_S)` (for two readings exactly
their interaction information with `x_S`, `I(y_1; y_2) − I(y_1; y_2 | x_S)`),
negative when the readings carry information about `x_S` only together
(synergy).

`model_uncertainty` is the lattice's **third same-functional-opposite-sign
pair** (after `ambiguity`/`occupancy` — see the next section — and the
`utility`/`risk` reparameterization the identity below records). It is
*literally the same estimator* as
`infogain`/`ensemble_disagreement`: the same four-net Monte-Carlo mutual
information over the transition's successor code, entered under a term
of sign **+1** instead of −1, so that the ensemble's Jensen gap is a
**cost**. Curiosity and pessimism, one functional.

R-AIF is the customer for the cost sign, and it is deliberate on their
part rather than incidental: the paper's Eq. 6 writes the value as
`Σ γ^(t−τ)(r + sim − IG)` and names the last term "the negative
information gain `−IG_τ`", while Eq. 7b's prose has the policy
"trained to reduce the model parameter's uncertainty (Equation 7b) or
information gain (IG) … takes more certain actions to maintain
**'homeostasis'**". Exploration in their agent is carried by 7c (which
is `occupancy`) and 7a instead — so the pair is not a bug in their
derivation but a division of labour between the terms, and the lattice
is what makes that readable at a glance.

Two consequences of shape worth stating. The sign belongs to the
**term**, registered once, and never to a trainer argument or a config
weight: a knob that flipped it would make G mean two things under one
name and force every consumer of a term tree (`total`, the `derived`
fold, a diagnostic reading the leaves, this table) to be told which.
And because the two rows are peers, choosing between them is an
**ablation** the selector already expresses — mount either, or both
against separate ensembles (never one shared ensemble: the scoring slot
is a pytree, and apparent sharing forks at the first optimizer step).

## The pair that makes the point

(There are three such pairs now — `infogain`/`model_uncertainty` above is
the same functional on the *same* conditional with opposite signs, which
is the sharper case. This one is the original and the more instructive,
because the conditionals differ too.)

`ambiguity` and `occupancy` are **the same functional on different
conditionals with opposite signs.**

- AIF: the expected conditional entropy of the *observation channel* is
  a cost — an ambiguous sensor is bad, you want states that report
  themselves crisply.
- MOP: the expected conditional entropy of the *transition* is the
  entire objective — a state whose successors are unpredictable is
  where the future lives.

Neither is a hack; they are two coordinates in the same lattice. Once
that is on the page, "should we ship an entropy-seeking row?" stops
being a question about orthodoxy and becomes a question about whether
the coordinate is reachable — and it is, from machinery we already have.

## Provenance of the two new rows

**`occupancy` — the maximum occupancy principle.** Ramírez-Ruiz,
Grytskyy, Mastrogiuseppe, Habib & Moreno-Bote, "Complex behavior from
intrinsic motivation to occupy future action-state path space," *Nature
Communications* 15 (2024), doi:10.1038/s41467-024-49711-1
([doi:10.1038/s41467-024-49711-1](https://doi.org/10.1038/s41467-024-49711-1)). Their state-value function is

```
V^π(s) = E[ Σ_t γ^t ( α·H(A|s_t) + β·H(S'|s_t,a_t) ) ]
```

with an axiomatic backing: weighted action-state *path* entropy is the
unique occupancy measure satisfying additivity per time step,
positivity and smoothness. The `β` term is our `occupancy` row exactly.
The paper positions MOP explicitly against FEP and empowerment (arguing
both collapse toward deterministic policies with little behavioral
variability), which makes the row's presence in an AIF library a
feature rather than an embarrassment: the comparison the paper draws
becomes runnable inside one declared model, with both objectives
scored over the same beliefs.

**And R-AIF already contains MOP.** Their G (addendum §3, paper
Eq. 7a–7d) rewards policy entropy (7a, `actent`) and rewards
`E H[Q(s|θ,π)]` (7c, `wment`) — MOP's `α` and `β` terms, at
`α = β = 3e-4`, alongside reward, `sim` and parameter info gain. The
term we were about to mark "replication-only" is one half of a
principled objective the replication target had adopted wholesale.

**`ensemble_disagreement`** stays as adjudicated (`amortization.md` A10
disposition 3): workstream H's ensemble info gain, a mutual information
estimated by Monte Carlo over an ensemble rather than derived in closed
form, hence `derived=False`. Its place in the lattice is why it is a
*sibling* row and not a variant of `predictive`: same functional,
different conditional and different estimator, so it earns a registry
row and not a branch.

## What this already caught

Writing the lattice down changed the build, which is the argument for
having written it down.

The plan of record was to mount R-AIF's 7c by **flipping the sign** on
the existing `entropy_lowering` leaf. That is wrong. `entropy_lowering`
stores `−H[q(x)]` — the *marginal* entropy of the belief. MOP and R-AIF
want `E_{q(s)} H[p(s'|s,a)]` — the *expected conditional* entropy of the
transition kernel. The two differ by exactly a mutual information:

```
H[q(s')] = E_{q(s)} H[p(s'|s,a)]  +  I(s'; s)
```

So the sign flip would have silently folded `I(s'; s)` into the
occupancy term — where it would then be counted a second time against
the ensemble info-gain leaf (7b). This is the same failure mode D2
names on the observation side ("every appearance of `I(o; θ | s)` in the
tree is exactly one explicit leaf"), arriving on the transition side
for the first time. `occupancy` is a new leaf, not a sign flip.

## What the lattice does and does not constrain

**Naming rule.** A term name states the functional and the sign; block
keys state the conditional. `occupancy:{transition:stoch}` is legible
without a lookup table; `state_entropy` (the name the retired cost term
used) is not, which is why the row is not called that.

**One estimator per variable is a within-term rule.** V2 forbids two
`infogain` estimators on one variable because they would be two
computations of the same thing. It says nothing about mounting
`occupancy` and `infogain` on the same variable — those are different
functionals, and both being present is the R-AIF configuration.

**Coherence is documented, not enforced.** The lattice tells you where a
row sits; it cannot tell you that your selection of rows adds up to a
bound on anything. The two shipped decompositions are packages
(`risk + ambiguity` vs `utility + infogain`) and mixing halves of them
is incoherent even though every individual row is well-placed. That
discipline lives in D2 and in the registry's docs, and the `derived`
flag records the consequence rather than preventing it.

**Where MOP's `α` term lives: the trainer, not the tree.** Policy
entropy is not a functional of a *declared conditional* — for the
enumeration planner it is induced by γ inside the softmax
(`q(π) ∝ exp(ln E − γG)`), and for an amortized actor it is an explicit
loss term over the policy module (`imagine()`, chunk 2c). So it is the
one lattice coordinate with no term-tree home, and the asymmetry is
principled: the tree scores the *model*, and the policy is not part of
the model. Recorded here so the omission reads as a decision.

## The open registration API (staged, not adopted)

The lattice suggests the general capability, and the general capability
is nearly free: `register_term(name, sign)` is already public, and the
estimator registry's `prepare(statics) -> step(ctx)` contract is
already the right shape for a user-supplied JAX function. Making
user-defined objectives a first-class capability is two commitments:

1. A public `register_estimator(...)` plus a mount point on `System`
   for rows that are not driven by an existing designation.
   Parameter-bearing user rows get their parameters in the same
   row-name-keyed slot that holds the critic and the ensemble
   (A11's rule: the registry names the strategy, the slot holds the
   strategy's parameters).
2. User rows get **no `derived` default** — the same rule A4 gives
   module-bearing rows, for the same reason: the only place a trust
   label can be wrong should be the one reviewed line that registers
   the row, and the propagation AND does the rest.

The honest price, which the docs must state at that seam: a custom row
means G is no longer a bound on anything. The `derived=False`
propagation records exactly that, and the coherence discipline above
cannot be enforced for a row the library has never seen.

**Sequencing (adjudicated 2026-08-20):** build the `occupancy` row in
chunk 2b, keep this map current, and open the registration API
deliberately when there is a second customer for it — a general hook
with one user is a guess about the second one. *(2026-09-02: the second
customer is now scoped — `empowerment`, `docs/design/empowerment.md`,
E-round open. It brings the lattice's first **composed conditional** —
a span-to-span channel derived from the declared kernels rather than a
single kernel — and the first row with mount-time parameters; that doc
also records the observation that `terminal_risk` is, in span terms,
`risk` against the H target mounted at the last index.)* The ambition it serves
is on the record: hybrid-aif is meant to support intelligent Bayesian
ML generally, with AIF as the default framing rather than the only one,
and the à-la-carte term tree is already a departure from orthodox AIF
in that direction.

## Build note: the tabular `occupancy` row (2026-08-31)

*Status: **build-time note, not adjudication** (Opus, 2026-08-31, demo
vignette 1). Recorded here rather than in `amortization.md`'s A12 notes
because the subject is a **lattice coordinate**, not the amortization
arc: the row this doc placed in 2b now exists on the other side of the
`ambiguity`/`occupancy` pair, and the identity written down above is
what the build had to implement. `amortization.md`'s `terms=`-grammar
note carries the one-paragraph grammar amendment and points here.*

**The row.** `registry._occupancy_transition` (distribution-object
carrier) and `_occupancy_transition_compiled` (the fused prob-space one,
reading `CompiledTables.B_probs`), behind
`discrete.transition_occupancy_leaf` — the taken action's slice of the
(mean) table, its per-parent-config child entropy, contracted against
the **previous** step's parent beliefs, since those are the parents of
the transition that produced this step. Leaf key
`transition:<factor>` and term `occupancy`, both identical to the
amortized estimator's: same bucket, same address, `derived=True` here
because the arithmetic is exact in the declared table under the
mean-field beliefs the planner already carries.

**Mean table, not expected entropy.** A `ConditionalDirichlet`
transition is scored through its posterior *mean* table, which is D2's
coherence rule arriving on the transition side: the novelty leaf owns
`I(s'; θ | s, a)`, and substituting `E_θ H[B]` here would remove it a
second time. `ConditionalDirichlet.expected_entropy()` stays available
for the documented no-novelty-leaf mode `ambiguity`'s docstring names.

**Eligibility.** One row name, several estimators, dispatched by
variable family (`registry.row_variant`) — the mechanism this doc's
"one principle" section asked for, and what pays
`amortization-debt-ledger.md` entry 2. Three consequences worth stating:
a wildcard now filters by family in both regimes and a literal name
still raises; the refusal is a `NotImplementedError` naming the
`(row, family)` pair that would make the key legal, which is the type
`registry.prepare` already uses for a missing estimator; and a row
variant may differ from its sibling **only** in `prepare`/`derived`,
because every other field is read off the row without knowing the
family.

**Two things the vignette showed that are worth keeping.** (i) On a
*deterministic* world the leaf is identically zero — MOP has nothing to
say about a maze with no slip — which is the honest reading of the
functional and not a defect; the demo declares a slip for that reason.
(ii) `occupancy` alone is a complete, well-formed objective that bounds
nothing in particular, and the tree says so through the term's sign and
the `derived` flag rather than through a warning. The first customer is
`examples/objectives_a_la_carte.ipynb`, column 3.

## References

- Ramírez-Ruiz, Grytskyy, Mastrogiuseppe, Habib & Moreno-Bote (2024),
  *Nat Commun* 15, doi:10.1038/s41467-024-49711-1 — maximum occupancy
  principle; `occupancy`'s axiomatization and the FEP/empowerment
  comparison.
- `amortization.md` A4 (the `derived` warrant axis), A10 dispositions 3
  and 7 (ensemble info gain; sign conversion at the registry boundary),
  A11 (the slot rule quoted above).
- `amortization-p0-addendum.md` §3 — R-AIF's G, term by term, with the
  `actent`/`wment` weights.
- `per-variable-infogain.md` V2 (one estimator per variable), V3 (one
  epistemic term), V4 (the estimator registry).
- `parameter-learning.md` D2 — the coherence rule this doc can only
  point at.
- `empowerment.md` — the staged second registration customer, the span
  grammar, and the composed-conditional extension of axis 1 (E-round,
  unadjudicated).
