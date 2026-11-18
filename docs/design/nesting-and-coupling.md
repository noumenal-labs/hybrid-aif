# Design: nesting and coupling. What crosses a blanket, and what does not

*Status: ADOPTED 2026-11-06 (Alex), N1 to N10, walked Alex + Fable the same day, superseding an
earlier draft of the same day. Built so far: the N10 audit, N3 (A11) and N4 with N9 (A12); N5 to
N8 are not built. Customer: the two-tier survey rover of the project page (`terrain-field.md`,
"The slow tier"), whose runner is now one `run` call on the particle coupled to the world.
Companions: `nesting-systems.md` principle 6 (no telepathy, no telekinesis),
`control-states.md` C12 and C13 (the action node, coupling through blankets),
`step-durations.md` T2 and T8 (two tiers are two Systems, the outer loop's clock),
`structure-learning.md` S8 (the lagged modality), `action-node-modalities.md` S2 (nothing scored
about the action node), `shared-parameters.md` amendment of 2026-11-05 (parameter variables),
`terrain-field.md` F1 (the cell indicator node), `predictive-coding-reading.md` (the actuator as
reflex arc). Sources: Friston 2019, *A free energy principle for a particular physics*
(arXiv:1906.10184, the partition and the synchronisation map); Friston, Da Costa,
Sakthivadivel, Heins, Pavliotis, Ramstead & Parr 2023, *Path integrals, particular kinds, and
strange things* (Physics of Life Reviews 47, arXiv:2210.12761; Definitions 1 and 4 and the
sufficient-statistics remark, referred to under Fundamentals); Friston, Rosch, Parr, Price &
Bowman 2017, *Deep temporal models and active inference* (Neuroscience & Biobehavioral Reviews
77, doi:10.1016/j.neubiorev.2017.04.009, the ascending message is a posterior); Pearl 1988,
*Probabilistic Reasoning in Intelligent Systems*, section 2.2.2 (virtual evidence); Hinton 2012,
*A practical guide to training restricted Boltzmann machines*, section 3 (probabilities in place
of binary states).*

## Why

The project page's two tiers are declared as one composite under `merge` and run as two
`subsystem` views on two clocks. Four things cross between the tiers, and the runner carries
every one of them by hand:

1. the scheduler's reading of the rover's position, computed as the argmax of the rover's
   position belief's mass over cells and fed to the mission view as a plain observation;
2. the option, chosen by the mission view and held in the rover view as a pinned control node;
3. the rover's learned field, written into the composite before each decision with `write_back`
   and read by the scheduler through `SharedParameter`;
4. the clocks: the mission view is called every twelfth tick, the rover view on every tick.
   T9's `hold` already does this under `run`, synchronously, without a hand-written loop; the
   runner cannot use `run` only because item 1 has no wire, so it re-implements the clock.

The runner's docstring says of the first: "the scheduler's cell reading is computed in the runner
from the rover view's position belief, because `couple` wires process outputs only." Asked
whether the library should carry it, the walk opened the question underneath: may one mind's
belief be another's observation, and if so in what form. The answer turned out to be a principle
the record already implied but never stated, and a reading of `merge` and `couple` that the
partition grounds. This note records both and the decisions they yield.

## What was settled before, quoted

`nesting-systems.md` principle 6: "**No telepathy, no telekinesis.** Couplings connect blanket
ports only, wires from action nodes to observation nodes. Reads: one System's posterior enters
another only through observations. Writes: nothing may rewrite another System's internal state
directly; all influence enters as evidence or dynamics through the blanket."

`control-states.md` C12: "an **action node** per control node, the dual of a modality: a child of
the control node with a kernel `P(a | u)` that is a delta unless declared otherwise (a stochastic
actuator is the dual of soft evidence, and a belief pushes through it)." And: "`ActionControl`
binding B's control node to A's was an identification of two latents, a model merge within one
mind, not a blanket coupling; the relay and `ObservationLink` are merges too."

`control-states.md` C13: "A **wire** is an identity between an out-port and an in-port of the
same type ... action node to process input, process observation to modality, or action node
straight to another mind's modality. A wire carries no kernel; anything that transforms ... is a
System with no planner run as a process. An action node with an actuator kernel is realised
**once per tick, at the node**, and every wire from it carries that value."

`action-node-modalities.md` S2: nothing is scored about the action node; the agent learns of its
realised value only through a modality that reads it.

`step-durations.md` T2: "Two tiers are two Systems; one clock per System. A tier that steps once
per option is its own System, coupled to the fast tier by ... the fast tier's filtered belief
carried in as the slow System's factor." T8: "The environment keeps no clock.
`System.elapsed(action, params)` returns the duration of a realised keyed action ... for an
outer loop to advance its own clock by."

`structure-learning.md` S8, as `System.lagged` carries it: a modality's datum is "the source's
value at the previous tick, read from the carry ... a state factor (the datum is its carried
belief's mean: a Gaussian's location, a categorical's probability vector as a soft observation)
or an action node (the level taken). The modality's own parents explain that value, so a factor
can emit what the model just believed."

`terrain-field.md` F1: "the new node is a categorical child of a continuous parent with
indicator features: `c = cell(x)`, a deterministic discretisation ... The belief over the cell is
the mass of the position belief in each cell."

`shared-parameters.md`, deferred: "parameter ports in `couple`, so a separate mind can take
another's Dirichlet as a soft observation."

In the code, the discrete filter's `_observed_log_likelihood` contracts a vector datum with the
log-likelihood: the expected log-likelihood under the vector.

## Fundamentals

**F1. The partition classifies states by dependency.** Every state of a particle is external,
sensory, active or internal. External states depend on external, sensory and active states;
sensory states on external, sensory and active states; active states on sensory, active and
internal states; internal states on sensory, active and internal states. The two excluded
directions are internal to sensory and external to active, and internal and external never
depend on each other directly. The blanket, sensory and active together, is what makes internal
and external states conditionally independent. A state's label is its role in this graph and
nothing else: whatever an external state depends on directly is a blanket state, and a blanket
state driven by internal states is active. (Friston 2019; Friston et al. 2023, equation (10). A
strange particle, Definition 4 there, additionally drops the active-to-internal dependency.)

**F2. Internal states are beliefs.**
This can be derived from the FEP: internal states are screened from external states by the blanket, 
so the most likely internal state given a blanket state and the most likely external state given 
the same blanket state are both functions of the blanket state, and when the first map is invertible 
they synchronise: the internal mode encodes the external mode through the blanket. Thus the internal 
states parameterize accurate (probabilistic) beliefs about the external ones. Friston et al. 2023
makes the same kind of point with respect to paths.

**F3. Beliefs are internal states.**
Thought-experiment: Add to a particle a set h that reads the internal states μ, putatively making
the belief-encoding states μ also active states read by h. Then either:

- **h touches nothing external.** The blanket condition is untouched: h adds no path between
  μ and η, so (μ, h) and η are still independent given the blanket, and h carries no
  information about η that μ does not. h is thus internal relative to the particle; interactions 
  among internal states are allowed. If h has a blanket of its own, μ is h's external world, and 
  the same state is internal to the particle, parameterising its belief about η, and external to h, 
  at the same time. That dual role is legitimate.
- **h additionally couples to η.** h cannot be internal, since it directly couples to external
  states. But μ stays internal, because the only path from μ to η runs through h, so μ and η are 
  independent given the enlarged blanket (b, h). h's role is the definition of an active state: 
  it depends on internal states and external states depend on it.

That proves that a belief in the FEP's sense (a sufficient statistic of external
states) cannot be an active state of the particle whose belief it is: children of this belief
are either internal or active states. The same logic shows that a belief cannot be a sensory
state. In brief: suppose h reads μ in addition to its relation to b. Either h couples to η or
not. If not, h is just more internal state, and conditional independence with respect to η
(and thus veridical representation) still hold. If so, then h is again part of the blanket.
And of course the assumption that a belief is an external state, relative to a given particle,
directly undermines the blanket construction of the premise. So we've shown that FEP-beliefs can
only be internal states.

**Adopted as an invariant beside principle 6:** *beliefs are internal (F2); what crosses a
blanket is action or sensation (F1-F3).*

## Nesting and coupling: what `merge` and `couple` mean

The two thought experiments are the two constructs.

**`merge` makes one particle.** It identifies latents across components into one model with one
blanket to the world: one set of internal states, however structured. Inside it, a component may
keep a blanket of its own. When every binding across a component's boundary is blanket-shaped, a
sibling-belief read as its sensing and an `ActionControl` as its acting, that component is
**nested**: its beliefs are its own, the rest of the particle is its external world, and its reads
are sensing inside one particle, with its own no-telepathy rule holding at its own blanket. When
a binding identifies a latent outright, `SharedParameter`, the relay, a joint `ObservationLink`,
the two components share that variable with messages both ways and no blanket between them on
it. Both are merges; nesting is a shape a merge can have, per component, and a merge can report
it. Friston's deep temporal models are nestings: the descending empirical prior is an internal
interaction, which a nest permits as a distribution through `D_deps` and `H_deps`, and the
ascending message, the lower level's posterior, is a sibling read. Jason's hierarchical model is
a nesting. The two-tier rover is nested but for one identification, which by N2's definition makes its
boundary shared rather than nested (A11): the scheduler is the rover's
own slow tier, its reads of the position are sensing, its option is `ActionControl`, and the
field is one home shared outright because the rover's counts and the scheduler's novelty are one
belief by design.

**`couple` joins particles.** Each participant is outside the other's blanket, and what crosses
is action and sensation. A belief crosses only as an action node realised from it; a parameter
crosses only as a report; nothing crosses as a belief of the receiver, only as a value, and that
value may be distribution-shaped, the sender's statistics (N8). Two agents, an agent and a
ground station, an agent and the world it runs in: coupling. A world merged in as a component,
as the composing guide's agent-and-world example does, is the agent's own model of its world
inside one particle, run forward as the composite's environment for simulation; the true world of
a run is a process under `couple`.

**The topology selects.** Is the reader inside the holder's blanket? Then nest, and read.
Outside? Then couple, and act. The `merge` question of the earlier draft is answered by this:
`merge` is not "one q" against `couple`'s "several", it is one particle against several, and a
nested component has its own q inside one particle.

## Realising a belief as a value

A kernel maps a value of its parent to a distribution over its child, the identity unless
declared (C12). It does not act on a belief. What turns a belief into a value is a realisation,
applied to the belief pushed through the kernel, and there are three:

- `sample`: a draw;
- `mode`: the argmax, or the mean of a continuous belief;
- `stats`: the belief's sufficient statistics. For a categorical belief, the probability vector,
  which is also the mean of the one-hot; for a Gaussian, its mean and covariance; for a Dirichlet,
  its concentrations.

The realisation does not depend on how the belief was formed. A policy posterior over a control
node and a filtered posterior over a state factor are realised by the same three rules through
the same kernel slot. For a control node, `sample` and `mode` are the action face's existing
`selection`; `stats` is new and emits the mixed strategy.

**What the physics says about `stats`.** On the reading in which the represented probability is
the probability over the configuration of the belief-encoding units, as in a Boltzmann machine,
a coupling between two regions is symmetric, and one-way influence exists only for a unit held
precisely. `sample` and `mode` are deltas and one-way because precise. `stats` is a vector that
asserts itself and listens to nothing, which has no realisation at the fast scale: there is no
"go 50% left" for a motor. It is allowed as a bounded efficiency, on the terms many Bayesian
frameworks use. Pearl's virtual evidence multiplies a prior by a likelihood vector; Friston's deep
models pass a posterior up; Hinton's guide communicates a unit's probability in place of its
binary state. The engine's soft observation is the expected log-likelihood under the vector,
which against the exact treatment of a datum drawn from the belief, the log of the expected
likelihood, is a lower bound (Jensen), with gap zero when the belief is a delta or the likelihood
is constant over its support, and at most the log ratio of the largest to the smallest
likelihood over that support. Over repeated reads `sample` carries the same distribution exactly,
which is how two regions of a Boltzmann machine communicate a distribution at all. Hence the
default is `sample`, as actions and observations are generated in active-inference simulations
generally; `mode` is the deterministic convenience; `stats` is the documented efficiency.

This does not depart from active inference couched as message passing; it says what that
couching is exact relative to. Inside one particle, inference is a variational scheme whose
messages are sufficient statistics, and passing them is exact relative to the variational
objective: the Jensen gap is the mean-field approximation's own looseness, not an error added on
top. The physical realisation of the same scheme by samples, Gibbs sampling against mean-field
rates, is the same dichotomy inside a particle as `sample` against `stats` across a blanket.
Across a blanket, the receiver's inference conditions on data, and `sample` is exact relative to
the true posterior while `stats` is exact relative to the mean-field bound. Both are message
passing; they are exact with respect to different targets, and N8 names which.

**The receiver's likelihood is ordinary.** Conditioning on a report of another belief does not
count that belief's prior twice. Under coupling the sender is a sensor and its prior is part of
how the datum is generated; under nesting the reader's variable is its belief about the holder's
belief, a different variable at a different level. In both the receiver's likelihood is an
ordinary likelihood on the modality, and the only remark is a modelling one: the table for "cell
reading given `visit`" should say how the rover's cell belief relates to the scheduler's variable,
and the demo's 0.95 identity is a crude answer.

**A chosen action is the degenerate case.** A one-hot over levels, a zero-variance Gaussian. So
nothing in a port's type says "realised"; every port carries a value, and `stats` is the one
realisation whose value is distribution-shaped.

## Decisions (for adjudication)

**N1. The invariant.** *Beliefs are internal; what crosses a blanket is action or sensation.*
Recorded beside principle 6, which it sharpens: "one System's posterior enters another only
through observations" names the mechanism, an action node realised from the belief into a
modality under coupling, or a sibling read inside a nest. Its ground is F1 to F3, from the
partition's dependency structure.

**N2. `merge` is one particle; nesting is a shape; `couple` is coupling.** `merge` keeps its name
and its bindings. A component of a merge is nested when every binding across its boundary is a
sibling-belief read (N3) or an `ActionControl`; `merge` can say which components are nested, and
the description document marks it. `couple` keeps C13 unchanged. The roadmap entry of the earlier
draft, whether `merge` survives, is closed by this.

**N3. The sibling-belief read: `lagged` generalised.** A modality of one component may declare
as its source a state factor of another component, through a kernel at the source (the identity,
or an indicator such as F1's `c = cell(x)`), realised by `sample`, `mode` or `stats`, default
`sample`. The datum is the source's belief in the carry, pushed through the kernel and realised,
when the reading component infers; the carry holds the source's belief as of the end of the
previous round, so S8's one-tick lag is kept, and it is the lag a reader that infers after its
source sees in any case. The modality's own parents explain the datum, as S8 says. This is
`System.lagged` with three generalisations: the source may be in another component, a kernel may
sit at the source, and the realisation is declared (today's `lagged` is the identity kernel with
`stats` for a categorical source and `mode`, the location alone, for a Gaussian one). How the
planner predicts a `stats`-realised modality is settled at the build: the information-gain
estimators score hard outcomes, which overstate what a soft datum delivers by the same Jensen
gap, so the first build scores a read exactly under `sample` and `mode` and documents the
hard-outcome estimate as an upper bound under `stats`. A read is sensing inside one particle, so
it needs no action node; the belief it reads
stays internal to the particle and external to the reader. For the rover, the scheduler's cell
modality reads the rover's position through the cell indicator.

**N4. Clocks: T9's `hold` under `run`, as built, plus one piece of bookkeeping.** Nothing new
is scheduled. `run` is synchronous: every round every mind responds, and a mind wrapped in
`hold` responds by returning its stored action and counting down until its last action's
`elapsed` duration has passed, then infers and acts again. That is "multiple clocks" with one
clock, no asynchrony and no hand-written loop, and the library has had it since T9. The runner
re-implements it only because the cell reading (item 1) has no wire, which N3 supplies. The one
addition is to `run`'s composite form: the composite's params are the single home, a view that
learns is written back each round it learns (`write_back`, as built), and the other views are
re-taken from the composite, so a `SharedParameter` needs no bookkeeping in the example.
Readings dropped between a held mind's decisions are a planning-time count under T10, not an
inference-time pooling.

**N5. Under coupling, an action node of a state factor.** C12's action node, with a state factor
as its parent, declared by the mind and realised by the same three rules through the same kernel
slot, outside the scored model (S2), held once per tick (C13). Nothing else is new. For a
belief read from outside, this is the only form.

**N6. Under coupling, a parameter crosses as a report of counts.** The holder declares an action
node of the parameter variable, realised by `stats`. What `stats` reports for a Dirichlet is the
counts since the last report, the per-call statistic that learning already separates from its
absorption, so that the reader's conjugate update adds each report to its own concentrations and
no evidence is counted twice; a report of the holder's whole concentration would carry the
holder's prior as well as its counts, and is the named variant, consumed by replacing what that
sender reported before. The reader observes the report through a likelihood over its own
parameter variable, the multinomial, whose conjugate update adds the reported counts to the
reader's concentrations; when the reader's prior is the holder's and it has no counts of its own
the result is a copy of the holder's posterior, and otherwise the two bodies of evidence combine
correctly. This is the "parameter as the parent of a likelihood" the shared-parameters amendment
deferred, pulled in for the count case; `mode` (the expected table) and `sample` (a table) of a
parameter are not count vectors, need a likelihood over tables, and stay deferred with the rest of
that item. Nothing clamps a reader's home from outside: that would write an internal state,
principle 6's telekinesis. Inside one particle, `SharedParameter` and `write_back` stay as built;
they are bookkeeping of one home between views of one particle, not a crossing, and a component
of one particle that runs on another host is still that particle, its home crossing a deployment
cut and not a blanket. The claim made in the walk's dialogue (Alex, 2026-11-06) that parameters
can be part of a blanket and so observed is restated: a parameter belief is internal, and a
parameter is observable by others only as reported through an action node of its variable, or as
a sibling read inside a nest.

**N7. Who supplies a control node.** Two questions were entangled in the earlier draft and are
separated here. This decision is only about who chooses. A mind may declare a control node as
held: another participant supplies it, the mind plans no decision for it and its action node is
dropped, and its factors read what is supplied as they read any control node. Under coupling the
supplier is a wire into the node; inside a nest it is `ActionControl`, as built. The engine
spelling is a control node because the engine's parents are factors and control nodes, never
modalities; in the partition it is sensory to the mind that holds it, since the mind does not
choose it and the outside sets it. What is supplied, a level or a belief's statistics, is N8's
question and applies to a held node as to any in-port. A nest additionally permits a descending
prior as a distribution through `D_deps` and `H_deps`, which is not a supplied control node but
an empirical prior on a latent, and is already built.

**N8. Sufficient statistics are first-class on every wire; the receiver consumes them by its
kind.** The earlier draft allowed `stats` into one kind of in-port and not another, with a
discrete-against-continuous rule. There is no principled asymmetry: a wire carries a value, and
`stats` makes that value the sender's sufficient statistics, as is standard in message-passing
algorithms. So `stats` may cross any wire, and what the receiver does with it is fixed by
the receiver's kind, each a known approximation with its bound:

- a modality consumes it as a soft observation, the expected log-likelihood (Jensen-bounded as
  above); for Gaussian statistics into a linear-Gaussian modality it is exact, the reported
  covariance adding to the modality's noise;
- a process input consumes it by marginalising its kernel over the statistics, the world
  stepping under the expected action; exact for a linear-Gaussian kernel, Jensen-bounded for a
  table;
- a held control node consumes it by averaging each reader's table over the statistics, which is
  what `BoundaryPrior` already does with a probability vector, so a soft option is built from
  parts that exist;
- a receiver that cannot marginalise, an outside hand-written process whose input is a level,
  refuses the wire naming `sample` or `mode`.

The physics paragraph above stays as the caveat: a `stats` value is a bounded efficiency wherever
it lands, and `sample` is the exact form. *Alex to confirm:* this closes the earlier draft's open
question in favour of uniformity.

**N9. The demo.** It stays a merge: the scheduler nested but for the one variable it shares, the
field, so its boundary reports as shared (A11). The runner's four pieces
become: the scheduler's cell modality reads the rover's position through the cell indicator (N3,
`mode` to reproduce the page's run, `sample` thereafter); the option is `ActionControl`, as now;
the field is `SharedParameter`, as now, written back by `run`'s composite form (N4) rather than
by the runner; the clocks are `hold` under `run` (N4, as built). The page's runs are unchanged in
substance. `python examples/survey_run.py --level 2` becomes one `run` call with the scheduler's
respond wrapped in `hold`, and the runner's own loop, schedule and write-back go.

**N10. The audit.** Every design doc, user doc and docstring that speaks of ports, beliefs
crossing, `merge` against `couple`, telepathy, soft observations, `lagged`, or parameters being
read or written is brought under N1 and N2, with hedges and epicycles removed rather than
annotated. Old decisions are not revised; their wording is. The audit runs after this note is
adopted and before the build.

## The format

The description document records a read on the modality record's `options.datum` entry, a
string naming the source for a pair and, for a `Read`, a record with the source, the kernel
(`identity`, or `indicator` with its grid) and the realisation; marks a merged
component `nested` when N2 says so; gives an action node of a state factor a record beside the
control nodes' action-node records, with its kernel and realisation; and marks a control node
another participant supplies `held`. A coupling is not part of the document and stays so; a
run's record names the wires. Details at build, under the format's own revision rule.

## Acceptance

- N3: a rover-shaped component and a scheduler-shaped component merged, the scheduler's cell
  modality reading the rover's position through the indicator: under `mode` the scheduler's
  posterior equals the runner's today; under `sample` the empirical read frequency over many
  ticks matches the cell mass; under `stats` the posterior equals the filter's soft-observation
  posterior and the measured Jensen gap lies within the stated bound. `merge` reports the
  scheduler's boundary as shared, the field being its one identification, and nested once the
  field is not shared.
- N4: `run` on the composite with the scheduler's respond in `hold` (durations twelve) and the
  rover's unwrapped; decision ticks at 0, 12, 24; the held action stands between; the field the
  composite holds after each round equals the runner's write-back today.
- N5 and N6 under `couple`: a mind with an action node of a state factor wired to another mind's
  modality, each realisation; a count report wired to a reader's multinomial modality equals the
  copy when priors match and the sum of counts when the reader has its own.
- N8: `stats` into a modality, a process input and a held control node each give the stated
  consumption, exact in the linear-Gaussian cases and within the Jensen bound for tables; an
  outside process with a level input refuses the wire naming `sample` or `mode`; a level into a
  soft modality refused; a held control node wired from nothing refused.
- N9: the demo reproduces the page's run to the recorded tolerance with the runner's four pieces
  deleted.
- The notebook gate.

## Deferred, named

- The continuous cases of N8 (Gaussian statistics into a linear-Gaussian modality or kernel),
  exact but not in the first build, which is categorical.
- A parameter as the parent of a likelihood beyond the count case.
- A random wait between decisions (T8's own deferral).
- A nest that reads both ways at once, the joint `ObservationLink` form, on two clocks; the
  one-way read is what the clocks force today.

## Amendments after the N10 audit (2026-11-06, Fable; for Alex's confirmation)

The audit of the other documents and the docstrings against this note found the following
places where the note was wrong about the code, inconsistent with itself, or silent. Each is
recorded here as a correction of wording or fact; none reopens a ruling.

**A1. `ActionControl`: which form is the nest's acting binding.** C12 calls a control-node
`ActionControl` "an identification of two latents", and the merge module says "the two nodes
become one". Both are true of the composite's node count and both stand. The binding is
nonetheless blanket-shaped at the driven component's boundary, and N2 and N7 mean the
control-node form: the one node is the driver's decision, the driven component's factors read it
as a supplied level, a decision is not a belief, and no belief of the driven component is
identified with any of the driver's. The state-driver form of `ActionControl`, the relay, binds
a state factor's belief into the driven factor's parent set a step later and is an
identification. C12's sentence is wording N10 revises; its ruling, that `ActionControl` is a
merge and not a coupling, is unchanged.

**A2. Today's `lagged`, exactly.** For a categorical source the datum is the probability vector
(`stats`); for a Gaussian source it is the location alone (`mode`); an action node may be a
source (the level taken), and the generalisation keeps that. The read is of the previous
round's belief, and N3 keeps that lag. A `(modality, source)` pair as declared today keeps its
behaviour; `sample` is the default of the new declaration form only, so no shipped model changes.
`merge` drops `System.lagged` today, which the N3 build fixes.

**A3. N4, exactly.** `hold` is built and is the clock. The rest of N4 is the build: `run`'s
composite form today neither learns nor writes back and cannot re-take a view, because each
respond closes over its view's params outside `run`; and the driver does not route a driving
component's decision into the driven view, which the survey runner does by a partner prior on
the view. Both move into `run`'s composite form. `hold`'s two restrictions stand: a mind wrapped
in it owns no world process, and none of its action nodes declares a non-identity actuator.

**A4. "held" and `hold`.** T9's "a mind held for an option" and N7's "a held control node" share a
stem and mean different things. In prose, a control node another participant supplies is
"supplied" where the two could be confused; `held` stays the mark in the format. Inside a merge,
`held` names what a driven node under `ActionControl` already is; the new piece is only the
in-port under `couple`.

**A5. N8, exactly.** The per-tick consumption of a probability vector by a control node's readers
is the planner's contraction (`take_controls`); `BoundaryPrior` is the static case, averaged
once at merge time. A reader with no table, a nonlinear transition, cannot average, so a vector
wired into a held node it reads is refused naming `sample` or `mode`. Gaussian statistics into a
nonlinear reader would go by moment matching, which has no Jensen-type bound, and are deferred.
The receiver's expected free energy under a soft supplied node inherits the convexity record of
`mixed-control-planning.md`, for which no bound is stated. A `stats` value into a process input is
the mean-field departure the physics paragraph names, and is allowed on those terms.

*2026-11-17, on A5's refusal of a probability vector into a nonlinear reader: the switching
mixture of 0.21.0 lifts it exactly. Push the belief through the transition per level of the held
node and weight the results by the vector. The linear bank's table average is the
Jensen-bounded mean-dynamics reading; it could be offered the same mixture. Not built; the
refusal stands.*

**A6. Planning over reads and reports.** The information-gain estimators score hard outcomes;
under `stats` they overstate a soft datum by the Jensen gap (N3, as amended). A count report
(N6) has no estimator entry; its information gain is novelty about the reader's variable reached
through a report. Both are settled at the build.

**A7. Labels.** This note's N1 to N10 collide with `nesting-systems.md` N1 to N3 and
`nonlinear-transitions.md` N1 to N6. Other documents cite this note's labels file-qualified,
"nesting-and-coupling N2", and this note cites the others file-qualified.

**A8. T2's wording.** T2's "the fast tier's filtered belief carried in as the slow System's
factor" is wording N3 changes: the belief enters as a modality's datum, explained by the
reader's own parents, not as a factor of the reader. The quotation under "What was settled
before" stands as the record; `step-durations.md` carries the dated pointer.

**A9. Open at the build, not needed by the demo.** Whether a nested component may carry its own
policy module (`merge` refuses one today) and its own policy precision (`merge` carries one);
the demo uses neither.

**A10. Names at the build.** `System.act` already has a `mode` keyword meaning marginal against
policy, and its `selection` takes "stochastic" and "deterministic". The realisations `sample` and
`mode` map onto those values; the build does not reuse `mode` as a keyword name.

**A11. From the N3 build and its review (2026-11-06).** A read takes the source's *posterior* of
the previous round, as today's `lagged` does under `replay`, `learn` and
`infer_states(previous=)`; under `run` the mind's carry holds that posterior beside the pushed
prior, and the driver supplies every read of a minded component from it, a component's own reads
included. A read's draw under `sample` uses its own `read_key`, never a face's `key`, which means
posterior collapse. The description records a read on the modality's `options.datum` entry (the
format paragraph above, corrected: no `source` field existed to extend). The nested report is a
tri-state `boundary` on the subsystem record: blanket (every crossing binding is a `SiblingRead`
or a control-node `ActionControl`, and the source component of a read counts as blanket-shaped
too), shared (any identification crosses, `SharedParameter` included, by N2's definition), or
none (no binding crosses; such a component is not nested, the vacuous case refused);
`BoundaryPrior` crosses nothing. So the demo's scheduler reports shared, the field being its one
identification; "a nesting with one identification" in N9 and the acceptance list meant that, and
now say it. The kernel and realisation of a read are not yet priced in the description length
(open: `description-length.md`, which prices the source as a parent choice when the format
records it). The driver finds cross reads from the composite's `lagged`, the subsystem record
being the record and not the source of truth; a `SiblingRead` of an action node is refused; the
planner's scoring of a read is as N3 says, the hard-outcome estimate under `stats` documented as
an upper bound and not otherwise built.

*2026-11-09, on A11's timing and mechanism: the fixed lag is retired and reads are read as they
stand. Under the orders a reader sees the value standing at its turn and never a value that does
not yet exist: a reader listed before its source sees the previous round's realisation (the survey
scheduler's case, which keeps the lag), one listed after sees this round's (the composite
`run` keeps the one-round lag under every order until chunk B rebuilds it over a closed System).
The datum of a read
is an emission of the source's System that the reader's modality observes (`observes`), realised
from the source's posterior and carried (`MindCarry.emitted`) where A11 has `MindCarry.posterior`
and `lagged`; `read_key`, `previous=` and `System.lagged` are retired, a sampled emission drawing
from `fold_in(step key, EMIT_TAG)`. The `Link` with a `realise` stays as sugar for the pair
(`design/engine-consolidation.md`, chunk B). The classification of a read as blanket is unchanged.*

**A12. N4 built as a particle respond, not as a change to `run`'s composite form (2026-11-06).**
The survey runner is the wired form: the true world is a hand-written process under `couple`,
and `run`'s composite form has no outside world, its world being the composite's own model run
forward. So N4's bookkeeping lands where T9's `hold` did, in a respond wrapper:
`particle(composite, params, minds)` makes a merged composite with per-view minds one participant
of a coupling. Its carry holds the inner minds' carries, the standing routed levels and the
parameter home; each round it supplies every read from the carries (N3), hands each view its
params from the home at the routed levels, calls the inner minds in the order given (a mind
listed later sees an earlier one's decision this round, which is how the option reaches the
rover in the same round), writes a learner's params back with the stale rule by construction,
and emits the composite's action nodes. `hold` wraps an inner mind as before. `run`'s composite
form stays as it is. The level-1 ablation is the same composite with a mission mind that always
returns the pinned option (`merge` refuses a `BoundaryPrior` on a node a carried target of the
rover is conditional on), and it reproduces the one-level run of the runner's old loop exactly.
The level-2 run differs from the old loop's in two things only, both by design: the read is one
round late, and absent at round 0. The world's key stream is unchanged, since the particle is one
mind of the wired form, so the two runs agree bitwise until the first option that flips. A mind
whose returned params differ outside its parameter variables is refused: only declared variables
are written back to the home.
