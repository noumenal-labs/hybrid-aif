# Design: description length — what a model costs, in nats, from equilibrium up

*Status: DRAFT 2026-10-14, for adjudication. The measure's form was settled in dialogue (Alex,
2026-10-14): a chain of KL divergences from equilibrium, with each family's cost its parametric
complexity (the log count of the distributions it can distinguish at the data's resolution).
D1–D4 are settled (D1's per-address `n`: Alex, 2026-10-14; the prior link's reading and the Gaussian base: Alex, 2026-10-15); D5–D7 are proposed. Customer: the model-search brief of a separate,
Noumenal-internal repository, whose asks are restated under "Why" so this document stands alone.
Companions: `model-description-0.3.md` §"Description length" (what the canonical form keeps for
this use), `structure-learning.md` S4 (`log_evidence` per conjugate family, built once, shared),
`parameter-learning.md`. Sources: Rissanen 1996 (stochastic complexity, the NML expansion);
Balasubramanian 1997 (Occam's razor as statistical mechanics on the space of distributions);
Myung, Balasubramanian & Pitt 2000 (counting distributions; the multinomial and Gaussian
volumes); Kontkanen & Myllymäki 2007 (the exact multinomial NML normaliser, a pin); Grünwald
2007, *The Minimum Description Length Principle*, ch. 11 (NML with infinite parametric
complexity: the constrained, conditional and luckiness forms); de Rooij & Grünwald 2006 (those
forms compared empirically on Gaussian and Poisson selection); Watanabe 2009 (singular models,
the learning coefficient); Blier & Ollivier 2018 (prequential coding of deep nets); Hinton & van
Camp 1993 (bits-back: KL as the parameter cost).*

## Why

Model search needs a complexity term that is honest across families. A count of parameters is
not: it prices a five-level table as any four-parameter family, it cannot see a prior, and it
charges a net for weights it does not use. The 0.3 note asks that names, placement and spelling
cost nothing and that parameters cost by what they say. The library's own quantity already is
a two-part code: the variational free energy is a model cost (the KL of the parameter posterior
from its prior) plus a data cost (the surprise under the posterior), and for a conjugate family
the two collapse into the closed-form evidence. The measure here is that quantity, made
intrinsic by choosing the base of each family's chain to be the prior that is uninformative *in
the distributions* rather than in the parameters, so that a family's cost is the log of how
many different things it could say.

The physical reading (Alex, 2026-10-14): the equilibrium state has no distinctions; every link
below is information held away from it, and nats become joules by Landauer's constant. The
measure does not reach an ideal metric; it reaches one where every term is readable that way
and none is a count.

What the customer asks of the library, restated: `description_length(document)` split into a
structure part and a parameter part, computable on a document alone, so that candidates can be
priced before any is fitted; `log_evidence` per address with an exact-or-bound flag, an address
the data never informed reported unscored rather than zero; a replay that carries its per-tick
free energy and, on the discrete path, its per-tick log evidence. Everything about how
candidates are proposed, pruned and ranked is the customer's; the library's part ends at the
numbers.

## D1 — The chain (SETTLED)

The description length of a model and a trace is a sum of five links, each in nats:

| link | what it prices | form |
|---|---|---|
| graph | the alphabet: how many factors and modalities, which families, which parents, how many levels and controls | a prefix code over the canonical structure (D2) |
| family | how many distributions the declared family can distinguish at resolution `n` | parametric complexity: `k/2 · log(n/2π) + log ∫ √det I(θ) dθ` (D3) |
| prior | a declared prior that is not the family's base | `KL(p_declared ‖ p_base)`; where no proper base exists, the prior's own extent (D4) |
| posterior | what learning put in | `KL(q(θ) ‖ p_declared)` for a family with a posterior (D5) |
| data | the trace's surprise | `−log p(trace ‖ model)`: the evidence for a conjugate family, the free-energy bound elsewhere, the prequential sum along a replay (D6) |

The family and prior links are the two parts of one code. Every family has a *base*, the
prior that is uninformative in the distributions: the Jeffreys prior where it is proper
(tabular), a reference prior at a declared scale where it is not (Gaussian, D4). The family
link is the resolution term plus the room the base admits, `log ∫ √det I` for Jeffreys and
`h(base)` in general, with `h(p) = −∫ p(θ) log (p(θ) / √det I(θ)) dθ` a prior's entropy
measured in Fisher-Rao volume; the prior link is `KL(p_declared ‖ base)`, what the declaration
holds away from that base. A sharper prior costs more, and the room costs what the room is:
ten levels cost more than three at the same sharpness (Alex, 2026-10-15).

*The reading set aside.* The Bayesian mixture code under prior `p` has expected regret
`k/2 · log(n/2π) + h(p)`, which is `res + log V − KL(p ‖ Jeffreys)`: under it a sharp prior is
a credit, the saving it would make on the data if it were right, averaged over its own mass.
That reading gives one finite number for every family, Gaussians included, and it is close to
a sparsity prior, for which nature offers evidence; it was set aside because it books an
expected saving on the data before any data has earned it, so a search that proposes priors
could lower a document's price by sharpening them (Alex, 2026-10-15: "the two-part reading is
more honest"). Both are one sign in `dirichlet_prior_link` apart, and the question is open to
be revisited by trying the tool under each and settling on one.

The customer's two halves are reported as: `structure_bits` = graph + family, both functions of
the document and `n` alone; `parameter_bits` = prior + posterior, functions of the document's
arrays. Each is also reported per address, with an `exact` flag per family, so "added a state
factor" and "added levels to one" stay distinguishable in a verdict. Nats are the unit; bits are
nats over `log 2`, offered as a convenience.

`n` is the resolution the family is priced at: the amount of data it is judged against, the
same `n` as BIC's, entering only the resolution term `k/2 · log(n/2π)`. Where a trace exists
it is not the caller's: each address has its own `n`, the count its conjugate statistics
absorbed (the responsibility-weighted mass that family saw, which the engine already tracks),
so a transition table is priced at the ticks it learned from and an initial-state prior at
the episodes. Only the document-only measure, with no trace to read, takes `n` from the caller,
and a search must fix one `n` for every candidate it compares: between two candidates the
`n`-dependence is `(k₂ − k₁)/2 · log n`, so a ranking can change with `n` only between
candidates of different dimension, which is the trade the term exists to make, and never by
the caller's choice within one comparison (Alex, 2026-10-14).

What the chain measures is `describe(build(doc))` in its canonical form, so renaming, reordering
and spelling cannot move the number (the 0.3 note). Nothing is computed on text.

## D2 — The graph code (SETTLED)

The alphabet is the combinatorial part no family owns: `F` state factors, `M` modalities, a
family for each from the fixed menu, a level count or a width for each, a parent set for each
factor and modality, a control count. Its code:

- a count with no declared bound costs the universal integer code, Rissanen's `log* n`
  (`log n + log log n + … + log c₀`, `c₀ ≈ 2.865`), the prior over integers that assumes no
  bound and is as uninformative as a proper prior over integers can be; computed in bits, where
  Rissanen's normaliser makes it a proper code, and converted to nats;
- a count under a declared cap costs `log cap`, the uniform prior over what the declaration made
  possible (a growable factor's `K_max`, `structure-learning.md` S3/S6). This prices the room
  declared, not the levels live: the count link is `log cap` whichever `K ≤ cap` is live, while
  the family link (D3) prices the live levels only. Declaring room for 32 and using 5 costs
  `log 32 ≈ 3.47` nats here against `log* 5 ≈ 3.70` for declaring 5 outright: the universal
  code pays for not knowing the bound, so a modest cap is a little cheaper than the count it
  covers and a generous one dearer (`log 1024 ≈ 6.9`), which is what a cap is;
- a family costs `log |menu|`, the menu being the families a record under that label may take
  (the format's own tables: an initial, a transition and a likelihood each have their own);
  *Note (2026-10-18), to be revisited:* the menu is the format's table in a given release, so it
  grows when a family is added (the transition menu went from 5 to 6 with
  `NonlinearGaussianTransition`, adding `log(6/5)` per transition record), and a description
  length is comparable across releases only after that shift. The code is also crude: it is
  relative to an enumeration rather than to what distinguishes one family's structure from
  another's, so it charges every record of a label the same. The direction to examine is a
  structural code, a family's cost the cost of its derivation in a small grammar of choice
  points (a table, a linear map or a general map; read controls or not; noise shared or per
  level; each slot fixed or learned), pricing only what the parameter part does not already
  price; which structural differences remain once the parameters are paid for needs working
  out first.
- a parent set over `P` candidates costs `log*(|parents| + 1) + log C(P, |parents|)`, `P` the
  parents a record of that kind can read (an initial prior: the state factors; a transition or
  a likelihood: the factors and the control nodes). A count that may be zero is coded as the
  count plus one throughout, here and for the modality and control counts;
- names, declaration order, role labels, `decomposition` sugar, inline against file, `simplex`
  against `log_simplex`: nothing (the 0.3 note).

Alternatives considered: a per-candidate-parent bit (a Bernoulli edge prior; costs `P log 2`
regardless of sparsity, so a sparse graph is overpriced); a fixed-width code for every count
(assumes a bound the document did not state). Settled (Alex, 2026-10-14): the universal integer
code as the base, with the cap rule.

## D3 — The family cost (SETTLED in form; per-family closed forms are build items)

A family's cost is its parametric complexity, the log of the number of distributions it can
distinguish at resolution `n`: `k/2 · log(n/2π)` for the resolution, plus the log of the
family's Fisher-Rao volume for its extent. The resolution term alone is Rissanen's count; the
volume term is what makes a table cost what a table costs.

- **Tabular** (a categorical over `L` levels, per parent combination): `k = L − 1`, and the
  Fisher-Rao volume of the simplex is `π^{L/2} / Γ(L/2)`, the normaliser of the Jeffreys
  Dirichlet with `½` per level. Closed form; a table over `R` parent combinations costs `R`
  times the one-row cost (the rows are independent families). Pin: Kontkanen & Myllymäki's
  exact multinomial NML normaliser, computed by its linear recursion for small `n` and `L`,
  against the two-term asymptote.
- **Gaussian** (a mean and a covariance in `d` dimensions): `k = d + d(d+1)/2`; the Fisher-Rao
  volume of a location-scale family is infinite, so the Jeffreys base is improper and the room
  is the one a reference prior admits (D4), `h(reference)`, closed form under the Gaussian
  family's Fisher-Rao density `2^{−d/2} |Σ|^{−(d+2)/2}` (and `2^{−d/2} |Σ|^{−(d+1)/2}` for a
  noise family with a known mean). For an emission over categorical parents only
  (`structure-learning.md` S2, the regressor block empty) the family is this one. For the
  regression families (a head over a regressor) the Fisher metric carries the regressor's
  second moment, which is the parent's and not the family's, so the room is unscored and the
  address is `exact=False`; the prior link, a KL between two declared priors, needs no metric
  and is scored regardless.
- **Singular families** (a net, a mixture read as one family): Watanabe's result replaces
  `k/2` by the learning coefficient `λ ≤ k/2`, usually far less, which is the precise form of
  "a count overprices a net". `λ` is a property of the family and the data together and is
  estimated from a fitted model (the local learning coefficient by stochastic-gradient Langevin
  sampling, Lau et al. 2023), so it is available at the prequential grain (D6), not at proposal
  time. Until then a net is reported at `k/2 · log(n/2π)` with `exact=False`, an upper bound,
  never silently averaged with an exact term.
- **A factor that is arbitrary code**: nothing intrinsic to count; a code length for the
  program is the floor, out of scope here.

## D4 — The Gaussian base (SETTLED: a reference prior at a declared scale)

A location family has infinitely many distinguishable distributions unless something bounds its
extent. The literature's three answers (Grünwald 2007 ch. 11): constrain the parameter range
(Rissanen's and Myung et al.'s practice: a box for the mean, an interval for the scale);
condition on the first few observations, which fix a location and a scale from the data
(conditional NML); or declare a proper prior over the parameters and let its spread be the
extent (luckiness NML; the Bayesian universal code). The third is the one the document
already carries: a linear-Gaussian family declares a matrix-normal inverse-Wishart prior
(`from_head_noise(head, noise, strength)`), and a continuous variable's record has a dimension
and no bounds. Under the two-part reading (D1) a prior's cost is its distance from the family's
base, and for a location family the flat base is improper: every proper prior is infinitely far
from it. What a location family needs is a scale, and that is the one hyperparameter the
measure declares (Alex, 2026-10-15: the prior precision is a hyperparameter, as in L2
regularisation). The base is a *reference prior* of the declaration's shape at unit scale: a
zero head with unit column covariance over a noise covariance `IW(scale² I, d + 2)`, the
vaguest proper inverse-Wishart whose mean is `scale² I` (its one-dimensional case
`IG(3/2, scale²/2)`). The family link adds the room this reference admits, `h(reference)`,
closed form where the Fisher metric is known (D3); the prior link is `KL(p_declared ‖
reference)`, closed form for the inverse-Wishart, the inverse-Gamma and the matrix-normal
inverse-Wishart pair, a sharper or displaced declaration costing more. The scale (the
argument `reference_scale`) defaults to one in the units the observations are declared in and
the caller may override it; the per-address report names it, so the convention is visible
rather than silent.

A Gaussian declared without a prior over its parameters (a fixed head and noise; a continuous
factor's fixed initial state) has nothing to measure from the base: its prior link is reported
unscored, never a defaulted box. Rejected: a range or box per continuous parameter (the same
arbitrariness as the scale, declared once per variable instead of once, and a truncation the
engine's Gaussians do not carry); a hierarchical prior over the hyperparameters (the same move
one level up; nothing needs it until a customer wants the scale learned). Set aside, open to
revisit: the luckiness reading, under which the declared prior carries its own extent as
`h(p_declared)` and no scale is needed (D1).

## D5 — The posterior link, and what has a posterior (PROPOSED)

`KL(q ‖ p)` is closed-form for the conjugate families (`Dirichlet.kl_to`, the matrix-normal
inverse-Wishart KL to be built beside `log_evidence`, S4). A family fitted to a point has no
posterior and no KL; it is priced by its family cost alone with `exact=False`, which is the
Rissanen reading and the honest one. The general facility that gives non-conjugate parameters
real posteriors (a Laplace step at the evidence maximum; `structure-learning.md` S3, Alex
2026-10-14) is what makes a net's posterior link available later, and nothing here is designed
around its absence. For a conjugate family the posterior link plus the data link equals the
closed-form evidence exactly (the bound is tight), so `log_evidence(family, posterior)` is the
one function, and the chain reports it split.

## D6 — The data link: evidence, bound, or prequential (PROPOSED)

Three readings of the same quantity, reported with a flag:

- **Evidence**, for a conjugate family: the marginal likelihood of the statistics the trace
  absorbed, `log_evidence` per address (S4; `exact=True`); an address the trace never informed
  is reported unscored, not zero (`exercised` carries the fact).
- **Bound**, elsewhere: the variational free energy summed along the trace, `exact=False`.
- **Prequential**, along a replay: `−Σ_t log p(o_t | o_{<t})`, each tick's observation scored
  under the model as it stood before seeing it; with learning off it is the trace's surprise
  under a frozen model, with learning on it includes what the parameters cost to learn, which
  is the description length that needs no posterior at all (Blier & Ollivier) and the one that
  ranks a batch of fitted candidates. In the engine this is the tick's variational free energy,
  `InferenceDetail.free_energy` (the mean-field posteriors' KL from their priors less the
  expected tabular log-likelihoods, less every LG modality's integrated log evidence), surfaced
  stacked as `Replay.free_energy`, and `Replay.log_evidence` its negation: exact where the
  tick's posterior is (every modality LG, or one categorical factor), a lower bound otherwise,
  on every path but the amortized one. The customer's two replay fields, seen as one thing.
  Built (`inference.discrete.mean_field_free_energy`; pinned on one factor against the exact
  evidence by enumeration, on two against the enumerated joint as a bound, on a switching and
  on a mode-aligned mixture beside a tabular channel against the enumerated joint, and along a
  replay against a NumPy loop with learning off and on). On the mode-aligned mixture path the
  evidence of `y` is read under the mode belief the tabular stage left, as the switching path
  reads it, rather than under the mixture's own bank masses, which still carry the prior mode
  mass; a replay with learning on teaches a slot nothing at a tick its modality is absent,
  padding included.

The data link is never scaled; a free energy balances its own halves.

*Paying for the parameters once (2026-10-16).* The family link and a learning replay's data
link price the same thing. The family link is what stating the parameters at resolution `n`
costs; a replay with learning on never states them, and its free energy pays for them instead,
each tick scored under the parameters learned from the ticks before it, so the early ticks
cost more. For a model starting from the family's base the two agree to leading order
(`k/2 · log(n/2π)` plus the room the base admits), which is the identity that makes the
prequential sum a description length on its own. So the total takes, part by part, the family
link where the replay did not learn the part (a point table, a slot it left as it found it, a
linear-Gaussian observation's point head beside a noise it learned) and leaves it out where it
did; an initial prior keeps it, being read at the first tick only, before anything is learned.
The graph and prior links are in the total either way, being what the declaration knew before
the trace, and they cover every level the trace puts to use: a dead level is room until a
replay wakes it, and then the head or column it started from is part of the model, so a dead
level's declaration cannot shape the data part unpriced. A family link takes its `n` from the
trace as D1 says: an observation table at the valid ticks, a transition at the ticks with a
step, an initial prior at the episodes. `description_length(model, replay=...)` composes the
total, the model part plus the data part, reading which addresses learned from the replay
(their parameters changed along it); it refuses episodes that learn under `vmap`, each of
which would pay for the parameters again from the start, and a posterior link beside a
replay, whose free energy already holds it. Built. The D7 recipe that reads `n` from a learning
replay's absorbed counts is to be checked against the same rule.

A Gaussian emission over a factor with dead levels takes the family and prior links of its
live levels' heads only, as a table takes its live rows only (D2's cap rule): a dead level's
head is room, not a member of the family. Built (2026-10-16).

## D7 — API (PROPOSED)

- `description_length(document, n) -> DescriptionLength` with fields `structure` (graph +
  family, nats), `parameters` (prior + posterior, nats), `total`, `per_address` (each link per
  address with its `exact` flag; an unscored link is `None`), `bits` as a view. Document-only:
  no engine state, no trace; `n` is the one resolution for every candidate in a comparison.
  `description_length(system, params)` after a replay with learning on reads each address's
  `n` from its absorbed count and takes none from the caller (D1). Built
  (`hybrid_aif.description.mdl`): `n` is one number or a mapping from address to its own
  resolution (the bridge from `Evidence.count` to that mapping rides the replay), `prior=`
  the params a slot was learned from fills the posterior link, `reference_scale=` the
  reference prior's width (D4; `scale=` until 2026-10-16), `replay=` a replay of a trace
  from the declared params, which adds the data part (D6).
- `log_evidence(prior, posterior)` per conjugate family, and
  `System.log_evidence(prior, posterior) -> {address: Evidence(nats, exact, count)}`: S4's
  function. Learning never stores the prior beside the posterior, so both are passed; `count`
  is the mass the slot absorbed, D1's `n` for that address; a slot with no closed form, or with
  no positive count (nothing absorbed, or mass let go under decay), is unscored. Decay
  anchored at the prior only reweights the absorbed statistics, so such a slot stays scored;
  decay toward any other anchor can forget the prior's own mass, and the prequential evidence
  carries that case (`parameter-learning.md` D7, amended). The customer's
  `log_evidence(document, traces)` is this after a replay with learning on. Built
  (`hybrid_aif.learning.evidence`).
- `Replay.free_energy` stacked per tick; `Replay.log_evidence` filled on the discrete path.
  Built (D6). `description_length.resolution_from_evidence(System.log_evidence(prior,
  posterior))` is the bridge from each slot's absorbed count to the per-address `n` the measure
  takes after a replay with learning on (D1). Built.
- `describe_parameters(document, arrays=None) -> (ParameterRecord(address, dtype, shape,
  domain, source), …)`: the envelope, read off the document's own records without
  constructing the model, one record per array (a tree variable's leaves by path). Built
  (`hybrid_aif.description.describe_parameters`).

## Acceptance

- Tabular family cost against the exact NML normaliser (Kontkanen & Myllymäki's recursion) on
  small `(n, L)`; the asymptote's error shrinking as `n` grows.
- The D1 identity on a tabular family: family + prior computed as volume plus `KL(p ‖
  Jeffreys)` and as resolution plus `h(p)` agree, for a Jeffreys prior, a flat one and a peaked
  one.
- The Gaussian extent (one dimension, normal-inverse-gamma prior) against a Monte Carlo
  estimate of `h(p)` over draws from the prior.
- `log_evidence` for a Dirichlet against brute-force enumeration; for the matrix-normal
  inverse-Wishart against the known closed form (shared with S4's pins).
- Prequential sum against a NumPy loop over a replayed trace, learning off and on.
- Invariance: renaming every variable, reordering declarations, switching inline and file
  arrays, `simplex` and `log_simplex`, leaves every number bitwise unchanged.
- A verdict test: "add a state factor" and "add a level to one" move `structure` by amounts
  that differ and are each what D2 and D3 say.

## Out of scope

- The learning coefficient estimator for singular families (D3); the regression families'
  Fisher volume (D3); an extent for a Gaussian declared without a prior (D4, reported
  unscored); the search loop, which is the customer's; program-length costs for code factors.
