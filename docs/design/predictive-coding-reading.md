# The control arc read as predictive coding

*Status: a reading, not a ruling (Alex + Claude, 2026-10-11). Nothing here changes the adopted
design in `control-states.md`; it records how that design corresponds to the predictive-coding
account of motor control, the questions that correspondence raises, and the roadmap items it
points to. Entered during the Phase 5 walk of `model-description-0.3.md` (Q17, the name of the
actuator factor's role).*

## The observation

In predictive coding and active inference, a motor command is a descending prediction into a
proprioceptive modality, fulfilled by a reflex arc (Adams, Shipp & Friston 2013, *Predictions
not commands*). That is the same mechanism vision uses in the other direction: the units driven
top-down in imagination are the units driven bottom-up in perception. An agent embodied as an
integrated predictive-coding network therefore has no separate actuator factors: its actuator
factors *are* likelihoods, into interoceptive modalities, and within the core generative model
sensors and actuators are symmetric. What is not part of the network, at either periphery, is an
external factor: a transducer on the sensory side, a reflex arc or motor on the effector side.

## The correspondence with C11 and C13

The adopted structure already is this picture with the reflex arc made explicit:

- the **control node** `u` is the latent whose top-down prediction is the command;
- the **action node** `a` with its actuator kernel `P(a | u)`, realised on the process side, is
  the reflex arc: the periphery that turns the prediction into a change in the world;
- a **wire** from the action node to a modality (C13 allows it) is the proprioceptive afferent,
  and the agent's likelihood on that modality is its proprioceptive likelihood;
- under the delta kernel and an identity wire the path `u → a → o` collapses to `o = u`, which is
  the symmetric case: the factor that was called an actuator is a likelihood into a proprioceptive
  modality, one tick later.

The two readings coincide whenever the control belief is a clamp (the enumeration, sophisticated
and CEM planners) and the reflex arc is exact. They can part under a belief-valued control (C3)
on a message-passing backend (C7): predictive coding emits the prediction, a sampler emits a
draw. Which the action face emits there is the backend's question, not settled here.

## Where interoceptive observations come from

From the process side, in either of two forms the wiring already supports. A direct wire from the
action node delivers the realised action itself, the efference copy made observable, with the
one-tick lag every wire has. A body in between, a System with no planner run as a process, maps
the realised action and its own state to what a sensor reports, a joint angle after the dynamics
for instance. Ports are typed by level count or width, so the shapes are already right; a
transformed proprioception needs the process, as any transformation does under C13. The gap is on
the engine side: a modality whose parent is an action node is accepted by `infer_states`,
`update_empirical_prior` and `as_env` only (build-plan gap table). Planning through the actuator
is what the predictive-coding control story needs, so that row is the one this reading leans on.

## Inference backends

The engine's inference is already a set of local schemes of the kind the constrained Bethe
free-energy recipe classifies (Şenöz, van de Laar, Bagaev & de Vries 2021, *Entropy* 23:807,
[doi:10.3390/e23070807](https://doi.org/10.3390/e23070807)): `fpi` is mean-field variational message passing on the tabular
factors, the Kalman steps are exact belief propagation on the linear-Gaussian chain, the
amortized heads are a learned recognition model. Predictive coding is the Gaussian mean-field (or
Laplace) member of the same family with gradient steps in place of closed-form updates, so it is
a backend in exactly the sense C7 gives the epistemic-priors formulation. Its natural home is the
deferred generic layer (`infer(query, clamped)`, project-brief open question 13); before that it
can be a per-family solver for the linear-Gaussian and neural families, as `fpi` is for the
tabular ones. Predictive coding as a *learning* rule, local updates with no global backward pass
(prospective configuration, Song et al. 2024), is a separate item: an optional learner beside
optax, of interest for the hardware roadmap because every update is local.

## The recognition model is part of the specification

Whether inference method and graph layout are entangled turns on whether the recognition
structure is part of the factor graph. In the constrained-FFG formalism it is: a form or
factorisation constraint on a local marginal is an annotation on an edge or node, not a second
graph, and the algorithm follows from the annotations. The engine already works this way without
saying so: the mean-field factorisation over state factors, the amortized `RecognitionPlan`, and
the block structure of the term trees are all declarations of `q` on the model. Making the
annotation explicit per variable is a document question for a later revision, not for 0.3.

The entanglement predictive coding introduces is then a constraint on *where `q` reaches*: the
motor story needs every effector path the recognition model covers to be an observed path too.
The resolution is the periphery. Just as the generative model need not reach the photoreceptors,
the recognition model need not reach the effectors; beyond its reach a factor of the agent's body
terminates on a variable that is not observed, and that is what a generic actuator kernel is. In
the engine the recognition model stops at the control node (nothing observes an action node, and
`InferenceDetail` carries no node entry), so a non-predictive-coding agent is simply one whose
whole effector path is ballistic, and a predictive-coding agent is one whose declared recognition
reaches the proprioceptive modality and whose actuators are deltas or are themselves processes it
observes. The layering is a declaration of where `q` stops, not a new kind of factor.

## Bodies

The layering Alex asked for, a core generative model that is bidirectional under predictive
coding and peripheral transducers and effectors around it, has a first form in the engine
already: an actuator kernel is periphery declared with the mind and realised process-side. The
general form is a **body**: a mind wired to its peripheral processes and presented to a larger
wiring as one participant, whose ports are the bundle's unbound ports and whose record nests its
members' records. That is a `Wiring` as a participant of a `couple()`, nothing more; strictly the
periphery could stay environment, but a body is what one wants to name, save and reuse.

## Roadmap items this reading adds

1. Planning through the actuator (the existing gap row): the predicted proprioceptive
   observation `P(o | u)` at the decision, so a wired action node is an ordinary modality on every
   path.
2. A predictive-coding inference backend, as a constrained-BFE scheme on the generic layer, or
   as a per-family solver before that layer exists; prospective-configuration learning as an
   optional learner.
3. Explicit `q` annotations per variable in a later description revision.
4. Bodies: a `Wiring` as a participant of `couple()`.

## What this leaves alone

The 0.3 role name for the factor `P(a | u)`. Until a predictive-coding backend exists, the engine
treats that factor and a likelihood differently (one is drawn on the process side and passes no
message up; the other conditions inference), so the document should name them differently;
`actuator` or `motor` is a naming choice, and the symmetry becomes a document question only when
the backend lands.
