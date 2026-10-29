# Design: evidence no state can explain — a flagged posterior, not a silent one

*Status: ADJUDICATED 2026-09-18 (Alex) — Z1–Z7 as written, after Z3/Z4/Z7 were revised the same day when he questioned refusing `−inf` (PGMax accepts it); the flag threshold `log(1e-12)` accepted. **LANDED 2026-09-18:** `inference.discrete.LOG_ZERO`/`SUPPORT_FLOOR`/`unsupported`, the tempered `fpi`, `InferenceDetail.unsupported` (per factor), `infer_states(support_floor=)`, `replay(...).unsupported`, `build` refusing `NaN` and dead observation levels (`description.check_observation_support`), `description.check_joint_support`, `tests/test_zero_support.py`. *As built:* `LOG_ZERO` is `-1e4`, not `-1e30` — `exp` of anything below about `-104` is already exactly zero in float32, and at `-1e30` a finite term added to a clipped one is lost to the ulp, so every impossible level would tie; at `-1e4` the finite residuals keep `~1e-3` nats of resolution, and levels still rank by the evidence they retain. Z3's "the posterior is the prior" holds exactly when the impossibility is uniform over levels (tested); when modalities disagree on *which* levels are impossible, the answer is the prior reweighted by the finite residuals — the level with the least impossible evidence wins — still the arithmetic's own answer, still flagged. On the hybrid path the flag is read off the tabular FPI stage, before any LG mode message. **Amended as built after the adversarial review (three verified defects):** (1) the clip happens *before* normalisation in the one lowering every reader uses (`_as_log_likelihood`), and an entry impossible before normalisation stays at the clip after it — otherwise a state whose column is all `-inf` normalised to `NaN` (or, once clipped, to a uniform emission that explained anything at `1/levels`); `build` refuses such a column by parent configuration. (2) **Z1's criterion is the minimum over messages, not their sum:** a factor is unsupported when the maximum over its levels of the *minimum* over its incoming messages and the prior falls below `SUPPORT_FLOOR`. The sum grows with the number of modalities and crossed a fixed floor on a perfectly supported tick at about thirty noisy modalities — the first consumer's regime — while one impossible message fails the minimum on its own; masked-out modalities send the zero message, which no floor rejects. (3) `check_joint_support`'s budget uses exact Python integers (`np.prod` wrapped in int64 and the guard that protects the largest spaces failed exactly there). Also: `check_observation_support` lowers Dirichlet tables through the same tempered path as the FPI, so the three layers agree; `replay_document` forwards `support_floor` to `build`; padding ticks are never unsupported; and `Replay.unsupported` is a **tuple of per-factor `(T,)` arrays with `None` for continuous factors**, not one `(T, F)` array, matching `InferenceDetail`. **Attribution under mean-field, measured:** the flag reads each factor's *expected* incoming messages, so a contradiction between modalities reading factor `f` also reaches `f`'s co-parents — on the T-maze contradiction `context` is flagged beside `location`, because the cue message arrives at `context` as an expectation under the (wrong) location belief and that expectation is uniformly impossible. A flag therefore says "this factor's evidence was uniformly impossible under current beliefs about its co-parents", not "this factor is the one the contradiction is about"; exact attribution is `check_joint_support`'s. The alternative — a per-factor upper bound taking the *maximum* over co-parent configurations instead of the expectation — attributes the T-maze case exactly but can miss a contradiction that only the joint over two factors shows, so it was not adopted. **Ruled 2026-09-19 (Alex): report the true and useful thing — "in general we should only report what can be strictly justified. If only a weaker statement can be justified in some case, report that instead."** The expectation-based flag is the justified statement — *under the beliefs this sweep held about its co-parents, no level of this factor was compatible with the evidence* — and it never claims to name the contradiction's source; exact attribution is `check_joint_support`'s, and the max-bound, which would claim more than mean-field can justify in the two-factor case, stays unbuilt. His earlier ruling, which shaped Z3:
a **flagged** posterior; a fallback to the prior "shouldn't be silent anyway,
and I'm not sure it's semantically always the right thing." Nothing is built.
Companion to `hybrid-system.md` D3 (inference), `model-description.md` M5/M6
(build-time validation and domains).*

## What happens today, measured (2026-09-18)

A composer declares several modalities whose likelihoods, taken together, leave
some observation tuple with **no state that explains it** — a disjunction the
factored likelihoods cannot express ("not committable because confidence is low
*or* too few readings"), or simply a dark camera reporting zero readings. Active
inference is correct to have no answer. What it does instead is the problem:

| tables | `infer_states` on the impossible tuple |
|---|---|
| **true zeros** (`log 0 = −inf`) | every logit `−inf` → `softmax` → **`NaN` posterior**, no warning; `act` then returns `NaN` too (Z4 makes this finite and flagged) |
| **floored** (the T-maze's `1e-16`, i.e. `−36.8` nats) | a **finite, confidently wrong** posterior: the exact location channel wins outright and the contradicting cue evidence is silently discarded |

Both were reproduced on the T-maze by observing `location = CENTRE` together
with a cue reading only possible at `CUE`. The first consumer hit the second
form on a real input and went from 13 stages to 17 working around it.

## The three layers, and who pays for each

The exhaustive check — "does any observation tuple lack a consistent state?" —
is exponential in the number of modalities and of factors. It cannot be the
engine's behaviour for an unknown consumer, so the design is three layers with a
stated cost each (Alex's correction: never scope engine behaviour to one
consumer's model size).

### Z1. The definition: an unsupported tick, per factor

After the mean-field sweep, factor `f`'s *unnormalised* log posterior is the sum
of its incoming observation messages plus its log prior. The tick is
**unsupported for `f`** when the **maximum over its levels** of that quantity is
below a floor `support_floor` — no level of `f` explains what was seen, under
the current beliefs about `f`'s co-parents. With true zeros the maximum is
`−inf`; with the T-maze's floor it is about `−37`. Mean-field caveat, stated:
this is the per-factor reading of a joint impossibility, exact when one factor
is observed, a sufficient signal otherwise.

### Z2. Layer one — the flag, at inference, whenever the record is asked for

`InferenceDetail` gains `unsupported: tuple[Array | None, ...]`, one 0-d bool
per factor (`None` on paths where the notion does not apply — see Z6),
computed from the sweep's own quantities at no extra cost. `infer_states`
already returns `(beliefs, detail)`; the flag rides `detail=True`. `replay`
passes `detail=True` and surfaces `unsupported: (T, F)` beside `beliefs`, so a
scorer reading a trace sees the tick where the model had no answer. An eager
caller who wants an exception wraps the flag.

*Why not always beside the beliefs:* the faces' arity is settled (`system.md`,
`composing-systems.md`), and the record is the channel the library already has
for "what inference computed and would otherwise discard".

### Z3. The posterior is the arithmetic's own answer, flagged — never replaced

Flagged, not replaced (Alex's ruling). No rule substitutes the prior or a
uniform. What the arithmetic *gives* at an unsupported tick, once Z4 makes
exact zeros finite, is worth stating: every level of the factor collects the
same floor from the impossible evidence, so the evidence adds one constant to
the whole row and the posterior **is the prior** (or, where some modalities do
discriminate, the prior reweighted by only those). That is not a fallback; it
is what "the likelihood discriminates nothing" means under mean-field, and the
flag is what tells the consumer it happened.

### Z4. `−inf` is legal everywhere; the FPI is tempered, so nothing NaNs

Alex (2026-09-18): "putting in `−inf` is normal there (for ex PGMax) … why is it
important to set a finite floor, given these other libraries use `−inf` just
fine?" It is not. The first draft proposed refusing `−inf` at the document
boundary; that was the wrong lever. The failure is not the `−inf` in a table
but the normalisation of an all-`−inf` row, and PGMax accepts `−inf` because
its message passing is **tempered**: log values are clipped at a huge finite
negative inside the kernel — the "tempered log-space numerics" the README
already credits it for.

So: `−inf` stays legal in tables, priors and documents (the `log_probability`
and `logits` domains refuse **`NaN`** only). Inside `fpi`, every incoming log
term — observed log-likelihood messages and log priors — is clipped at one
constant, `LOG_ZERO` (large enough in magnitude that `exp` of it is exactly
zero in float32 and that a sum over every modality of a model stays finite;
PGMax's pattern). Exact zeros then behave as exact zeros wherever they
discriminate, and an all-impossible row normalises instead of producing `NaN`.
The engine does not floor anyone's *tables*; it tempers its own arithmetic. The
T-maze's `1e-16` floors become unnecessary rather than forbidden.

### Z5. Layer two — per-modality support, at build, always

Every observation level of every tabular modality has at least one state
configuration with log-likelihood above `support_floor` — linear in the tables,
run in `build`, a sentence naming the modality and the dead level. This catches
a level nothing can produce; it **does not** catch the disjunction case, and
the doc and the sentence both say so rather than imply completeness.

### Z6. Layer three — exhaustive joint support, opt-in, budgeted

`hybrid_aif.description.check_joint_support(system, params, *, max_combinations)`:
enumerates observation tuples against state configurations and reports every
tuple no configuration supports above `support_floor`. Refuses — with the count
and the budget in the sentence — when the product of observation levels times
the product of state levels exceeds `max_combinations`; never a silent sample of
the space. A diagnostic a composer runs on a candidate declaration, not a face.

The LG and mixture paths have no zero-support notion — a Gaussian likelihood is
positive everywhere; a surprising observation is a low `log_evidence`, already
returned — so Z1/Z2 flag `None` for continuous factors and Z5/Z6 cover tabular
modalities only.

### Z7. Two constants, one adjudicable

`LOG_ZERO`, the tempering clip (Z4): an engine numerics constant, not a knob —
around `−1e30`, where `exp` is exactly zero in float32 and a sum over any
realistic number of modalities cannot overflow. `SUPPORT_FLOOR`, the flag
threshold (Z1): `log(1e-12) ≈ −27.6` nats by default, overridable per call
(`infer_states(..., support_floor=)`, `replay(..., support_floor=)`,
`check_joint_support(..., support_floor=)`); below the T-maze's accidental
`1e-16` floors, far below any probability a composer meant to be possible. The
second number is the one to adjudicate.

## Acceptance

- The T-maze contradiction above flags `location` unsupported under both the
  floored and the true-zero tables; a supported observation flags nothing.
- `replay` surfaces `(T, F)` flags; a trace with one impossible tick flags
  exactly that tick.
- A true-`−inf` table gives a finite, flagged posterior equal to the prior on the
  contradicted factor, not `NaN`; `build` refuses `NaN` by address and a dead
  observation level by modality and level, and accepts `−inf`.
- `check_joint_support` finds the probe model's disjunction case when given
  two contradicting binned modalities, and refuses above its budget with the
  count.

## Addendum 2026-09-30: Z1 and Z7 as built — ADJUDICATED (Alex): the code stands

*Found by the 2026-09-30 docstring paring: two places where this record and
the code disagree. Alex's ruling: the reasons that came up in implementation
are the ones to keep, so the code's definitions are the record's from here.
Z1 and Z7 above are left as written; this addendum states what is built.*

1. **Z1's flag, as built: the minimum over terms, not the sum.** Z1 defines
   the flag on the sum of a factor's incoming observation messages plus its
   log prior, maximised over levels. The code (`inference/discrete.py`,
   `unsupported`) takes the *minimum* over terms — `jnp.min` over the stacked
   log prior and messages — before the maximum over levels, and a factor is
   unsupported when that maximum is below `support_floor`. The reason: a sum of
   many legitimately noisy modalities grows with their number and would cross
   a fixed floor on a perfectly supported tick, while one impossible message
   (`-inf` tempered to `LOG_ZERO`, or an accidental `log 1e-16`) fails the
   minimum on its own. `System.infer_states`'s docstring now points at
   `unsupported` for the definition rather than restating either reading.
2. **Z7's `LOG_ZERO`, as built: `−1e4`, not `−1e30`.** Z7 puts the constant
   around `−1e30`. The code sets `LOG_ZERO = -1.0e4`. The reason: `exp` of it
   is still exactly 0.0 in float32, and a finite term added to `−1e30` is lost
   to the ulp, so every impossible level would tie; at `−1e4` the finite
   evidence keeps about 1e-3 nats of resolution and impossible levels still
   rank by the evidence they retain.
