# Control overhaul: build plan

*Status: working plan for building `control-states.md` (as amended: C6 canonical form, C10 names
at the faces) and `model-description-0.3.md`. Branch `control-overhaul`, from f7bedde. Each phase
is reviewed adversarially before its commit; open questions are settled when their phase starts.*

## The gate every phase keeps

A baseline harness (kept outside the repo, under the worktree's ignored `runs/`) recorded at
f7bedde every public face on 15 fixtures: policies, scores and term-tree leaves, marginals, `act`,
beliefs, prediction, Dirichlet, conjugate and gradient learning, `as_env`, rollouts, replay,
`imagine`, the descriptions, and the compiled graphs of the benchmark's three functions. Values are
keyed by name, so only the harness's adapter layer changes with the API. A phase passes when every
recorded value is bitwise equal, the compiled graphs are equal with debug metadata stripped, and the
suite passes; a test migration that changes only how values are indexed is its own reviewed diff
and never changes a numeric expectation. `examples/demo.ipynb` re-executes at every phase that
touches public API.

## Phases

1. **Names at the faces, then control nodes.**
   - *1a.* Beliefs, actions, observations and `InferenceDetail` become name-keyed mappings at every
     public face (C10); `aux_beliefs` stays keyed by scoring row, its inner carry keyed by the
     nested scoring-slot System's factor names (the seam is engine-tier; only its producer is
     reference-tier);
     positions stay inside; an explicit converter on the System. Actions were keyed by control node
     (`u_<f>`) here and are keyed by action node from 4c. Policies stay the internal positional tensor.
   - *1b.* Control nodes as factor-table nodes, per the representation design of 2026-10-06 (C11:
     no static sizes, `policies(params)`; `controls` by node index like `B_deps`; node beliefs
     optional on input and always emitted, continuous nodes omitted until Phase 3; a node prior in
     params refused until Phase 2; `legal=` per state factor until Phase 2; `policy_len`
     keyword-only; `replace` with sugar kwargs or `factor_names=None` on a System with nodes
     raises; the public `planner_context` takes `horizon` and the enumerating planners follow it,
     default `policy_len`; uncontrolled factors' size-1 axis read at a batched zero, as today, so
     the benchmark graphs stay equal). Control nodes as factor-table nodes; old-style declarations translated at build time and
     appended (C6); one control table read by every consumer; enumeration from it; policies become
     one column per control node (Q5). Clamp only.
2. **Clamp and belief, priced controls, legality by node** (C3, C8; Q6–Q9, Q13, Q19's `legal=`
   spelling).
3. **Continuous control nodes, the LG `G u` term, mixed spaces, the LQ demo** (C4; Q2, Q11, Q12,
   Q14, Q15). R-AIF's continuous path stays bitwise.
   Open before the build: the Python spelling of a native continuous node and where its box
   attaches (Q2); the planner for the LQ demo (Q11); what "bounded Gaussian" means (Q12); discrete
   nodes entering `G u` (Q14); how far non-tabular shared nodes go (Q15).
4. **Composition**: same-step binding beside the relay, legality across a merge (Q10).
4a. **Audit** (C12): every design doc, user doc and docstring that conflates a control state with
   an action, listed with its sentence and its correction, and every code path that rests on the
   conflation (ports on control nodes, the action face keyed by control names, replay's action
   columns, the driver's routing, `as_env`), before any edit. Read-only pass, reported to Alex.
4c. **Action nodes and `merge()`**: the action node per control node (C12), the action face and
   traces keyed by action-node names, the purge of control-node-as-port, `couple()` renamed
   `merge()` with the Phase 4 tree kept as the merge it is.
4d. **`couple()`, coupling through blankets** (Alex, 2026-10-09; C13): participants are minds
   and processes; a `Wire` is an identity between typed ports (level count to cardinality, width
   to dimension), with no kernel — an external likelihood is a System with no planner run as a
   process; an action node is realised once per tick, at the node, and fanned out; unbound
   in-ports refused with their fix named (`fixed=` for a process input; an absent-modality
   declaration waits on the standing-mask question), unbound out-ports ignored; `couple(...)`
   returns a type-checked `Wiring`, `run` takes it and returns one record per participant; `run`
   on a merged composite stays as sugar, bitwise. The driver docstring's "the world" becomes "the
   wires".
5. **Description 0.3** (D1–D8; Q16–Q18, Q20), the agent bake-off of 0.2 against 0.3 as the reader's acceptance test (`model-description-0.3.md`, Q17), then 0.12.0 with the migration note (C9).

## Gaps the baseline exposed

| Gap | Where it is handled |
|---|---|
| `rollout()` stores observations as scalar int32, so no continuous modality | Phase 1a: a name-keyed observation carries its own dtype and shape; add the LG fixtures' rollouts as new tests (no baseline exists for them). **Done (1696506)** |
| A conjugate continuous likelihood cannot plan with `epistemic=True` (no continuous parameter-novelty estimator) | Recorded for the baseline with `epistemic` selecting only the state variables; the estimator itself is queued after the overhaul with the bounds research |
| CEM refuses a `policy_prior` | With control-node priors, which Q6 keeps out of Phase 2 (C2: the path-dynamics build): a prior over a continuous node is what CEM would read |
| Replay's learning path passes no `detail=`, so learning a switching noise through replay raises | A small fix of its own after 1a (replay is touched there; the fix changes behaviour, so it gets its own test, not a baseline) |
| `test_the_aux_belief_seam_is_general_and_reaches_score_policies` passes vacuously: the tiny R-AIF agent's `cosine_sim_to_pref_rollout` leaf is NaN with and without auxiliary beliefs, and `nan != nan` (also at f7bedde) | Cause found: the row precomputes its preferred trajectory for `policy_len` steps and reads it with `jnp.take`, which fills an out-of-range step with NaN when the scored plan is longer (`score_policies` takes any length; `imagine(horizon=)` overrides `policy_len`). R-AIF runs never hit it (agent and `imagine` share one horizon; logs finite). *Decided (Alex, 2026-10-06): adapt.* A small fix of its own after 1a: rows are told the scored plan's horizon at trace time (`policy_len` stays the planners' and `imagine`'s default); every step-indexed lookup is range-checked statically; the term name's engine registration moves to the reference tier with the row; the test scores shorter and longer plans and asserts finite and different. **Done (c802c3c)** |
| `System.names_generated`, a static flag pair recording which name tuples construction filled in, exists only so the 0.2 document's `generated_names` record stays string-equal (a declared `()` and a defaulted one are otherwise indistinguishable) | Kept through Phase 5 (Alex, 2026-10-06), so no commit knowingly carries a broken document round trip; dropped with 0.3, which does not record defaulted names. **Done (049bdb6)** |
| `examples/ensemble_leaf_variants.py` fails when run directly (`train/world.py` shape check: obs `(4, 2)`, expected `(B, T, 2)`), also at f7bedde on `main`; no test covers it | Pre-existing; a small fix of its own after 1b, with the example added to the notebook/example smoke set. **Fixed 2026-10-13: the example had kept the static-buffer sampler after the move to stream buffers and called `_control_layout` without params; `tests/test_ensemble_leaf_variants.py` runs it at its smallest settings** |
| `num_controls` asserts a size the System no longer holds (C11), so the C6 disagreement check cannot fire | Dropped (Alex, 2026-10-07; C6 amended again): a Sonnet migration after the 1b fix round removes the keyword and property everywhere (tests, examples, notebooks, the harness's fixture constructors; goldens unchanged); the 0.2 reader keeps checking documents. **Done (5942ae3)** |
| `imagine` prices a control node as the planning path does, but no case reaches it: the amortized path cannot hold a discrete control node (the GRU carry exposes no axis for it; the code's initial state is read without an action) and `imagine` rolls out only a Gaussian actor | The wrapper is unit-tested; the first reachable case is a priced continuous node, which Phase 3 brings. Discrete controls on the amortized path are a capability gap, not a theoretical one: the path was built for one customer (the R-AIF agent, continuous actions), so the carry embeds a float action and `imagine` rolls out a Gaussian actor; a categorical actor over a node's levels with the level embedded into the carry is the standard shape (Dreamer-style discrete actions). The `MLPConditional` reader is not part of the gap: it reads a level as a one-hot. To look into after the overhaul (Alex, 2026-10-08) |
| `merge()` cannot bind a native (non-sugar) control node: the `Subsystem` record and the 0.2 document name bound control nodes by owning factor, which a native node has no | Phase 5: the 0.3 document names nodes; the refusal says so until then. **Done (049bdb6; `test_a_merge_binds_native_control_nodes`)** |
| A modality whose parent is an action node (proprioception) is supported on `infer_states`, `update_empirical_prior` and `as_env` only; planning, `learn`, `replay`, `merge`, `subsystem`, `describe` and the hybrid and amortized paths refuse it | After the arc: under the delta kernel the predicted observation is `P(o \| a)` at the decision, an ordinary modality; the scan carry needs the action node's prediction per step. No customer yet (non-delta actuators). Under the predictive-coding reading (`predictive-coding-reading.md`) this is the row the motor-control story leans on: planning through the actuator is what makes a wired action node a proprioceptive modality |
| `as_env` refuses the amortized path | Out of scope here; queued (R-AIF uses its own environment) |
| Habit planner has no policy set or marginals; `System.learn` refuses on R-AIF | By design, not gaps |
| Description documents record the library version | Expected to differ from 0.12.0; document equality is a gate only until Phase 5 |
| The engine's two validators disagree on an uncontrolled transition table's layout: the plain path (`_validate_table_arity`) takes the table with or without a trailing axis of size 1, the composite path (`_validate_control_deps`) requires it | Not slated (found by the bake-off, 2026-10-13). `build` pads on the composite path, so a document is layout-free on both; a native composite declared without the axis is still refused where a plain System is not. One rule for both when the validators are next touched |
| `Env` is a class of its own, with a mind/process distinction in the wiring (C13), though the library's endpoint has every participant as a blanketed System (Alex, 2026-10-09) | Roadmap, after the arc. The recast is precise: an Env's control slot is a state factor of the process observed through a delta modality wired from the driver's action node, so B reads the action as taken by exact inference and the process infers its states as any agent does (the symmetry Alex imposed); the direct form, the in-port setting the control slot with no inference, is the same model with the exact step skipped, and whether a latent may be conditioned directly on an observation is *open* (Alex, 2026-10-10: to think about what the bedrock patterns should be); its emitted observation is an action node whose control node has a conditional prior given states (the A table read as a policy), which an ancestral respond samples (infer, push, emit); run after its drivers in the tick it reproduces the Env's one-tick lag, so the mind/process bit reduces to schedule position plus that respond function, `as_env` becomes a rewiring and `Env` the pymdp wrapper's sugar. Needs: state-conditioned control priors; a guarded zero-variance (delta) observation on the LG path; somewhere for `as_env`'s learnable-table lowering to happen. The 4d build is a strict subset, nothing to undo. Under the factor-graph policy-selection machinery the planner/process distinction goes; open there is which search algorithms stay baked in and which are machines of subSystems, with the baked-in planners as the machines' reference implementations and kept where generality costs speed (Alex, 2026-10-10) |
| The periphery is bundled with a mind in one form only, the actuator kernel; a richer body (a transducer or effector process with state) is either environment or a hand-made wiring | Roadmap (`predictive-coding-reading.md`, 2026-10-11): a **body** is a `Wiring` as a participant of `couple()`, its ports the bundle's unbound ports, its record nesting its members'. Beside it: a predictive-coding inference backend as a constrained-BFE scheme on the generic layer (or a per-family solver before it), prospective-configuration learning as an optional learner, and explicit `q` annotations per variable in a later description revision |
| An actuator kernel exists for discrete action nodes only (a `(levels, levels)` table); on a continuous node the entry is refused (`check_kernel`, "drop the entry"), so a continuous effector is always exact | Not yet slated (noted 2026-10-12). The natural form is a Gaussian kernel on the action node, `a = u + noise` or an affine map with noise, the delta being its zero-covariance case, drawn at the node as the discrete one is; it is the reflex arc of `predictive-coding-reading.md` for continuous controls and shares the guarded zero-variance Gaussian the Env-as-System row needs. The 0.3 `actuator` record admits the family without change |
| A preference target over an observation or a state (`obs_targets`, `state_targets`) is an unconditional distribution, scored as an expected log-probability or a KL against it; only the terminal target takes parents (`H_deps`) | Not yet slated (noted 2026-10-12, Alex: C may be conditional on variables as the terminal target is). The 0.3 `target` record already carries a `parents` list, so the document admits it; the engine's scoring of a conditional C is the gap, and with it a general step set for when a target applies, under which the terminal target becomes the target at the last step. The engine also accepts a `ConditionalGaussian` target whose parent axis is read off the array with no parent declared; description 0.3 refuses to write one (a document would promise what the reader rejects), and the `target` record with parents is its form once the engine reads parents for a Gaussian target. **Done** (`System(C_deps=...)`, `design/step-durations.md` T5; `tests/test_conditional_targets.py`, among them `test_a_conditional_family_without_c_deps_and_an_unconditional_one_with_it_are_refused`: a `ConditionalGaussian` target without a declared parent is now refused by the engine and written by the document with its `parents`) |
| The policy prior E is a distribution over the enumerated policy set only (`policy_prior`, an array or Categorical over `System.policies`), or a belief-conditioned policy module the habit planner acts from; a per-step prior on a control node, tabular and conditional on a state, is not scored | Not yet slated (noted 2026-10-12). Inside G it is an expected log-prior along the rollout, computed as a target under a latent child is, outside the precision as E is. The 0.3 `prior` record under a control node already writes it |
| A gradient family carries a stored `frozen` flag; a conjugate slot can be held fixed only by a call-time learning rate of zero (`lr_by_slot`) | Not yet slated (noted 2026-10-13). The 0.3 `configuration.learning.frozen` list names clamped variables of either kind; the gap is a stored marking the learn step reads for a conjugate hyperparameter, so a document's `frozen` round-trips for both |
| `policy_prior` holds two different objects: E over the policy set, and the habit planner's policy module (beliefs to an action distribution) | Not slated (noted 2026-10-13, Alex: a conscious deferral). Description 0.3 writes the module as the control node's `recognition` record, which is what it is (an amortized q over the node); the engine move puts it in `recognition` keyed by the node and leaves E alone in the slot, touching the trainers' actor key and the reference agent |
| The objective is four fields that must agree: `terms`, `epistemic`, `scoring` (both halves) and `decomposition` | Not slated (noted 2026-10-13). Description 0.3 writes one list of term records, each with its target and its own parameters, `decomposition` as sugar for the standard pair; the engine move makes that list the System's declaration and compiles the term tree from it |
| Composite membership is a stored record `merge()` builds (`Subsystem`), with refusals that name nodes a binding removed | Not slated (noted 2026-10-13). Description 0.3 derives membership from the reserved separator in names, as `build` reconstructs the record; the engine move derives it the same way at run time and drops the record and that class of refusal, which needs the engine to reserve the dot in names as the reader does |
| A System constructed without names carried a placeholder flag (`names_generated`) whose only consumer was `describe` | Phase 5 removes it (Alex, 2026-10-13): a structural `replace` on a placeholder-named System is refused with or without it, and names in a document are declared names. **Done (049bdb6)** |
| The pymdp-style view of controls (`_sugar_view`, `controls.py`: which factor each control node belongs to, `controlled` and the control wiring) is derived from the `u_<factor>` name pattern, and the amortized path and the driver read that view, so a name decides wiring there and a freely named control node is refused on the amortized path | Not slated (noted 2026-10-13, found by the Phase 5 review once the constructor stopped demanding `u_<action>` names under C11's native-declaration rule). The structural view is the control table itself: a node is factor `f`'s control when `f`'s transition reads it and no other does. Description 0.3 refuses the amortized case at `relations.controls` with an address and states the limit in the format document |
| No type distinguishes a System that chooses from one that does not; `Agent` as a subclass of `System`, or as a constructor returning one, was raised (Alex, 2026-10-13) | A question, not a gap, and not 0.3's. C13 treats mind and process as roles a System plays; a constructor named for the role that requires control nodes, targets and a planner gives the readability without a type |
