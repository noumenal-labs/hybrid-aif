# Shared parameters across Systems — design note

*Status: ADOPTED as design direction (Alex, 2026-10-01); not built. Implementation
waits for the close of the R-AIF arc. It amends `amortization.md` A11 refinement (ii) once
built; the current text is quoted below so the delta is explicit. S2 is resolved: gradients
from every reader are accumulated before one update, and alternation is not supported.*

## The capability

Two Systems may declare that they contain the **same mechanism**: one parameter set,
read by both, learned from both. In a physical / Bayesian-mechanical reading this is an
identity claim between parts of two models. The two subsystems are not each carrying a
copy of something similar. There is one quantity, and both are evidence about it. Under
gradient learning that means the shared parameter's gradient is the **sum** of the
gradients of every loss that reads it, the log-loss analogue of multiplying the
likelihoods of two evidence streams about one variable.

The first concrete customer is R-AIF's preference model. Their CRSPP reads the world
model's encoder, learned initial state and prior head, and its contrast loss trains all
three. Ours keeps its own initial and takes the embedding as data
(`amortization-debt-ledger.md` entry 5). The actor-entropy diagnostic §28–§29 found
this is **not** what separated the two sides' behaviour, so sharing is a modelling
capability and not a fidelity requirement.

## What the rule says today

`amortization.md` A11, refinement (ii): *"module-object sharing across entries RAISES
(the pytree-forking trap: apparent sharing silently duplicates parameters at the first
optimizer step; shared torsos are E's feature by name)."* It is enforced at three points:
the recognition slot (`inference/amortized.py`, the `shared` owner map), the scoring
slot (`objectives/amortized.py`, the `owners` map), and the CRSPP's own validation
(`reference/raif/crspp.py`).

The trap is technical, not semantic. An optimizer step maps a pytree to a new pytree
leaf by leaf, so two references to one Python object become two leaves, each receiving
only its own path's gradient, and they fork at the first update. The rule already
anticipates sharing **by name**. This note proposes delivering that now for parameters
across Systems, independently of the generic q-graph layer.

## Proposal

1. **Declared by address, never by object.** A shared parameter has exactly one
   *home*: an address in one System's `SystemParams` (the same addresses `lr_by_slot`
   and freezing already use, namespaced through `merge()`). Every other reader holds a
   reference token naming that address. Object aliasing keeps raising, with the
   message pointing to the declaration.
2. **Resolution is pure.** References are resolved to the home's arrays at the top of
   any function that reads them, before tracing reaches the forward pass. The pytree
   the optimizer sees contains the parameter once.
3. **One home, one step.** The home address owns the only optimizer state. A trainer
   that steps a shared parameter must see every loss that reads it with gradient, and
   the gradients sum. A reader that is not in the step reads the parameter under a
   stop-gradient, declared per reader (`grad=False`), which is what today's
   external-input path already does for data.
4. **Beliefs are never shared.** Per-tick carries stay per System, as they are now.
   State crosses between Systems as data, through the existing external-input parent
   kind and the `aux_beliefs` seam.
5. **Serialization and description.** A reference is encoded as its home address
   (`description/__init__.py`), never duplicated, so a saved model round-trips with the identity
   intact.

## A related obligation this does not cover

§29's defect was not about parameters. Two modules read one replay batch and disagreed
on its **segmentation**: the world model restarted its filter at `is_first`, and the
preference model carried across episode boundaries. Sharing parameters does not
protect against that. Any System that filters sequences drawn from a shared batch must
receive the batch's episode boundaries. That is the subject of a separate, smaller
follow-up: a contract that a scoring row's System running a filter over batch windows
takes `is_first`, with one test.

## Decisions for Alex

- **S1 — amend A11 (ii).** From "sharing raises" to "aliasing raises; sharing is
  declared by address." Recommendation: yes. The current rule already names
  by-address sharing as the intended feature.
- **S2 — joint step as the default.** Should a shared parameter require a joint step
  over all reading losses, with stop-gradient readers as the explicit exception?
  Recommendation: yes. Today each trainer steps a disjoint set of slots from one
  `LearnState`, which is what makes their call order safe. A shared slot would be
  stepped by two trainers. Alternating two optimizers over it is a legitimate
  algorithm, but it does not learn one quantity from two evidence streams: each
  optimizer's moments see half the gradient stream, the update is order-dependent,
  and the slot's learning rate and clip become ambiguous. **Resolved (Alex):**
  accumulate first, then update; alternation is not supported, and a second trainer
  stepping a shared slot raises. The joint form
  is gradient accumulation by address: each trainer contributes its gradient for the
  shared slot within one train step, and the home applies one update. R-AIF's own
  `model_opt` is this form, since it steps the world model and the CRSPP on their
  summed loss with one Adam.
- **S2′ — sharing as a binding in `merge()`.** `merge()` already yields one
  `SystemParams` and one `LearnState` with namespaced, disjoint addresses. A parameter
  binding (`a.x` is `b.x`), declared like a node binding, would make sharing a
  property of the composite. Within one trainer over the composite the summed gradient
  then comes for free. Recommendation: yes, as the declaration site. It does not
  cover a scoring-slot System such as the CRSPP, which `merge()` refuses, so the
  address-reference form above is still needed.
- **S3 — scope of the first build.** Recommendation: parameters across Systems
  composed by `merge()` or nested in a scoring slot. Sharing across members of a
  `vmap` population, meaning tied weights across agents, is a different question and
  stays out of scope.
- **S4 — R-AIF.** Should the reference tier adopt sharing to match their CRSPP exactly
  (ledger rows 4.1 and 4.14), once the capability exists? Recommendation: run it as an
  experiment, not as a fidelity fix. §28–§29 show it is not needed for parity.
