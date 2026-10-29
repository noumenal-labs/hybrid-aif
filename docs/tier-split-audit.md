# Tier-split audit: what in the engine is still R-AIF's

*Status: AUDIT 2026-09-20, read-only; **landed 2026-09-21** in three phases — 45d198e (the
move, T7/T8/T10), b165a0d (the engine, T1–T6) and the prose pass (T9) with the 0.9.0 bump.
A forensic record, not a design doc:
it lives beside `de-risk-report.md` and `scoping-review.md`. Decisions
T1–T10 at the end were ruled the same day (Alex; rulings noted inline).
Companion to `design/system.md`'s addendum of 2026-08-28
("package tiers"), which set the cut and which this audit finds applied
imperfectly. Trigger: `replay`'s chunk two showed the trainer's sampled filter
and the agent's mean-field filter to be two conventions over one trace
(`replay.md`, roadmap item "one scan over a record, or two"), and the question
"is `RSSMCore` a generic model type or R-AIF's?" could not be answered by
reading, because the engine's prose says one thing and the addendum another.*

## Method

Two read-only surveys on this date, one over every engine module
(64 modules outside `reference/`), one over the reference tier and its
consumers; the highest-value claims were then re-read at their cited lines.
Counts are of lines matching a replication-narrative pattern ("R-AIF",
"their …", "tiny", "addendum §", "oracle", "P1′ chunk", "divergence N",
cites of their `nets.py`/`agent.py`/`epistemics.py`) after excluding generic
uses ("reference implementation", nesting-systems' "chunk 2"). Line numbers
are as of bde8659 and will drift.

## The finding in one paragraph

The import rule holds: no engine module imports `hybrid_aif.reference`, and
`tests/test_layering.py` enforces it. The leak is elsewhere, in two forms.
**Code:** the engine carries R-AIF's episode columns as its only episode type,
registers R-AIF's CRSPP scoring row with a trainer string pointing into the
reference tier, pins its only world-model trainer to R-AIF's exact head
triple, ships their config row as a constructor (`RSSMCore.tiny`) and their
config key (`winit`) as an argument name across the neural families, and
switches two R-AIF-only actor-loss terms on by default. **Prose:** fourteen
engine modules narrate the replication in docstrings and comments, about
eight hundred lines of divergence, adjudication and measurement history that
the docstring rule (functionality, not history) already forbids; two of those
docstrings are now contradicted by the code beneath them. The reference tier,
for its part, is ten R-AIF modules sitting flat beside two shared targets,
six of them not naming their target in their first two lines.

## 1. Engine: code stragglers

| module | items | what |
|---|---|---|
| `train/replay.py` | 4 | `Episode` columns `is_success`, `has_succeeded`, `can_self_imitate`, `pref_label` with their dtype table and buffer zero-allocations; `Episode.cont = 1 − is_success` (their Mountain Car's goal predicate *is* its termination predicate); `alternate(a, b)` whose only customer is `run_mc --alternate`; `fit_length`'s padding justified by their env |
| `objectives/amortized.py` | 5 | `cosine_sim_to_pref_rollout` row registered in the engine with `trainer="reference.crspp.fit_crspp"` and block key `"crspp"`; `scoring_opt_state` switches on row *names* including that one; `_ensemble_disagreement` hard-codes their `1/sqrt(num_vars)` scaling; `_predicted_reward` and `_g_tail` refuse any head that is not their `symlog_disc` (255 bins over ±20); `MODEL_UNCERTAINTY` row kept (legitimate pessimism row) |
| `inference/rssm.py` | 4 | `RSSMCore.tiny` (every default is their `gym_mtc tiny` row); `from_declarations` requires exactly their triple (`symlog_mse` decoder, `symlog_disc` reward, `binary` cont), so `fit_world` refuses any other amortized System; `__check_init__` asserts the same triple; `winit` threaded through every part |
| `distributions/neural.py` | 4 | `winit` argument on six constructors with error strings quoting their config; `MLPPredictor`/`MLPCategorical`/`GRUCell` `from_shapes` defaults (`width=32, depth=1, hidden=32`) are the `tiny` row while `MLPConditional`'s are not (`width=64, tanh`); `_HEAD_LOSSES` positive list documented as "R-AIF's config uses exactly these"; `_linear_stack` docstring says the `winit` apparatus "is not kept" while the function takes and honours it (**stale**) |
| `train/imagine.py` | 4 | `DEFAULT_WMENT=3e-4` (their ζ) and `DEFAULT_BC_SCALE=0.5` (their `actor_refresh`) on by default, neither in DreamerV3; parameter names `wment`, `actent`, `can_self_imitate`; DreamerV3-generic defaults documented as "the oracle's resolved config"; structural errors that cite "R-AIF's RSSM" |
| `train/ensemble.py` | 3 | `learning_gate`/`NO_LEARNING` transcription of their `ig_head_random_no_learning` member dropout, documented as inert at parity; batch builder typed by `RSSMState` rather than the faces' belief tuple; error strings citing their `epistemics.py` |
| `train/world.py` | 2 | `embeddings()` whose only consumer is `reference/crspp.py`; `assemble`/`fit_world` pinned through `from_declarations` |
| `factors.py` | 1 | `Deterministic.gru(hidden=32, winit=)`: `hidden` default is their `units` |
| `inference/amortized.py` | 1 | structural rules fine; error strings quote their heads and their `initial: learned` |
| `system/core.py` | 2 | `as_env` fence names `hybrid_aif.reference.envs.mountain_car` as "the environment" (the addendum calls this deliberate; it still recommends one benchmark to every user); a hard-coded list of mounted-row names that will drift from `registry.ROWS` |
| `system/selectors.py` | 1 | error text names `hybrid_aif.reference.worlds.crspp_rssm` |

Deliberately **not** flagged, being DreamerV3 or library machinery with a
generic shape: `symlog`/`symexp`, two-hot bins and their losses, `unimix`,
straight-through one-hot, factored-categorical KL and `kl_balance`
(0.5/0.1/1.0), `GRUCell` arithmetic, `trunc_normal_avg`/`uniform_avg` as
initialisers, `lambda_return`, `perc_ema`, the slow-critic update,
`_cont_mode`/`rollout_weight`, `WorldBatch`/`world_loss`,
`ensemble_loss`, the six registry rows as concepts, `RSSMState`/`observe`/
`img_step`/`obs_step`, `Deterministic`/`DerivedInitial`/`Recognition`,
`soft_clip`, the buffers, `habit`. The optimizer constants in the three
trainers (`lr`, `eps`, `clip`) are DreamerV3's defaults and defensible as
generic; only their docstrings are the replication's.

## 2. Engine: prose

Counts of narrative lines per module: `neural.py` ≈160, `imagine.py` ≈110,
`train/replay.py` ≈90, `objectives/amortized.py` ≈70, `rssm.py` ≈55,
`system/core.py` ≈55, `ensemble.py` ≈48, `inference/amortized.py` ≈37,
`world.py` ≈26, `registry.py` ≈18, `factors.py` ≈17, `selectors.py` ≈14,
`habit.py` ≈9, `controls.py` ≈7; single lines in `validation.py`,
`objectives/hybrid.py`, `objectives/base.py`, `replay.py`, `enumerate.py`,
`train/__init__.py`. Everything else is clean (all of `distributions/` but
`neural.py`, `inference/` but the two amortized modules, `planning/` but
`habit.py`, `composition/`, `data/`, `envs/`, `learning/`, `description.py`).

Characteristic: `_linear_stack`'s ~80-line experiment write-up with a result
table and seeds (`neural.py`); `_ensemble_disagreement`'s 110-line inline
adjudication log ("42 h of compute found this"); `Episode.cont`'s "**it was
wrong here until 2026-09-07**"; `ensemble.py`'s `.. _DIVERGENCES:` section;
`imagine.py`'s bullet list of their resolved config. Two are **stale**:
`neural.py`'s "`winit` … is not kept" (it is), and `objectives/base.py:72`'s
"their λ-return applies a per-term `G_scales`" (the `g_scales` knob was
removed, as `imagine.term_value` itself records). All of this material has a
home already: `docs/experiments/raif-mountain-car/` (the ledgers, the
divergence register, `plateau-analysis.md`).

## 3. Reference tier

| module | target | class | names target in lines 1–2 |
|---|---|---|---|
| `envs/tmaze.py` | pymdp's `TMaze` | shared (10 tests, 2 notebooks, 1 example) | yes |
| `envs/mountain_car.py` | their vendored `mtc.py` (sparse reward, `easy_mode`, repeat 2, limit 100) | R-AIF spec, **shared in use** (`examples/mountain_car.py`, `test_cem.py`, the `as_env` fence) | yes |
| `worlds.py` | `tiny` RSSM, CRSPP RSSM; slot-name constants are their names | R-AIF, nothing else in it; de-facto amortized test fixture for seven engine-side tests | yes |
| `crspp.py` | their `agent.py` CRSPP loss + `nets.py` no-action scan | R-AIF | **no** |
| `revise.py` | line-for-line port of `callbacks.py::lookback_on_episode` | R-AIF | **no** |
| `seed.py` | their driver's row convention, wrapper flags, 22-episode budget; `bang_bang` is ours | R-AIF (typed `env: MountainCar`); `seed_stats` is generic | **no** |
| `eval.py` | ACR/SR/R-S with their offsets; imports only numpy | R-AIF | yes |
| `fixtures/__init__.py` | oracle-export `.npz` format + loader; `OBS_DIM=2`, `ACTION_DIM=1` | R-AIF data; loader shape is ours | partial |
| `fixtures/export_oracle_mc.py` | one-off export against their log streams | R-AIF | partial |
| `raif_agent.py` | the assembled agent, IDD Alg. 2, their optimizer groups | R-AIF | yes |
| `run_mc.py` | their training protocol as a resumable 4-seed harness + CLI (1,924 lines) | R-AIF; checkpoint/resume plumbing is generic harness code welded in | **no** |

Generic-but-misplaced candidates by the tier's own criterion ("is its shape
ours to choose?"): `seed.seed_stats`, `fixtures.load_fixture`/`save_fixture`
(an episode-fixture format over engine `EPISODE_FIELDS`, width-pinned to
Mountain Car), and `run_mc`'s checkpoint machinery. None is urgent; the
harness plumbing would be extracted when a second replication lands.

Also found: the addendum in `system.md` still describes the tier as
`worlds.py` + `envs/` only, predating everything else that landed there. And
the repo-root `/reference/` directory (untracked clones of pymdp, PGMax and
others, gitignored) shares the name with `src/hybrid_aif/reference/`; nothing
imports it, but a reader can confuse them.

## 4. The proposed layout

```
src/hybrid_aif/reference/
  __init__.py            tier charter, listing subtrees
  envs/tmaze.py          stays (pymdp target, shared)
  envs/mountain_car.py   stays (the one benchmark env; see T8)
  raif/                  everything pinned to NACLab/robust-active-inference @ 9b1864e8
    __init__.py          the target repo, commit, paper, the "their/oracle/IDD" vocabulary, once
    world.py      <- worlds.py       declared models (tiny_rssm, crspp_rssm, slot-name constants)
    agent.py      <- raif_agent.py   the assembled agent + train_step
    crspp.py      <- crspp.py        port of their agent.py loss + nets.py scan
    revise.py     <- revise.py       port of callbacks.py::lookback_on_episode
    metrics.py    <- eval.py         ACR/SR/R-S ("eval" shadows a builtin and says nothing)
    seed.py       <- seed.py
    run.py        <- run_mc.py       python -m hybrid_aif.reference.raif.run
    fixtures/     <- fixtures/       loader, four .npz + provenance, resolved config, export script
```

Naming by kind so a second replication reads the same way: `world.py`,
`agent.py`, `metrics.py`, `seed.py`, `run.py`, `fixtures/`, and `<their
file>.py` for ports. The `raif_` and `_mc` affixes become redundant inside
the subpackage.

**Mechanics, checked.** Import statements to touch: 37 in tests across ~25
files (50 if `mountain_car` moves too), 2 in `examples/`, ~20 in-tier, 0 in
notebooks (both notebook imports are `tmaze`), 1 in `test_layering.py`
(`from hybrid_aif.reference import worlds`; the rest of that test already
covers any subtree). Two engine strings: `objectives/amortized.py`'s
`trainer="reference.crspp.fit_crspp"` and `selectors.py`'s error text.
**Checkpoints survive the rename**: `state.eqx`/`buffers.eqx` are
leaf-order serialisations with no module paths; `run.json`, `checkpoint.json`
and `pending.npz` carry config fields, counters and coverage only. Only the
launch command changes. Path-by-`__file__` uses (`fixture_path`,
`export_oracle_mc.OUT`, `_code_provenance`, two parity tests) follow the
package; the fixtures directory moves with its `__init__.py`, and hatch's
`packages = ["src/hybrid_aif"]` includes the data files wherever they sit.
No entry points, no CI config, no pickles. **Latent only:** `description.py`
serialises callables by `module:qualname`; no checked-in document names a
reference-tier callable, but one saved with such a callable would break on
rename. The live run (`crspp_init_100k`, resumed from main at bde8659) has
its modules imported already and is not disturbed by a rename on disk; a
later `--resume` would use the new path. Docs that cite `reference/…` paths
in prose (`amortization.md` ×47, the three experiment ledgers, `system.md`,
`capabilities.md` ×3, the addendum, the debt ledger) are historical except
the addendum and `capabilities.md`.

## 5. Decisions pending (T1–T10)

- **T1. The episode format.** `Episode` becomes generic (`obs`, `action`,
  `reward`, continuation/terminal, `is_first`, `is_last`) with an `extras`
  pytree of task-specific columns; R-AIF's four columns, their dtype table and
  the success-as-termination `cont` move to `reference/raif/`. Alternative:
  keep as is, renamed `RAIFEpisode` in reference, with a generic `Episode`
  beside it. Recommendation: the first; it is the largest straggler and the
  one a second consumer hits first. *Ruled: yes, provided `extras` is a
  sound pattern here (see the note under T1 below).*
- **T2. The CRSPP row.** Register `cosine_sim_to_pref_rollout` from
  `reference/raif/crspp.py` (`register_row` is public, registration is
  import-time), taking `CRSPP_BLOCK` and the trainer string with it; make the
  optimizer-carry shape a property of `registry.Estimator` so no engine
  function switches on row names. Alternative: generalise to
  `cosine_sim_to_rollout` with a neutral block key. Recommendation: register
  from reference; the row's semantics are the CRSPP's. *Ruled: yes.*
- **T3. Unpin `fit_world`.** `from_declarations` and `__check_init__` accept
  any set of `PredictorHead`s, each contributing its own `nll`, instead of
  exactly their triple; `_predicted_reward`/`_g_tail` accept any head with a
  scalar `predict`. This is what makes `RSSMCore` a generic model type in
  fact, not just by the addendum's assertion. *Ruled: yes.*
- **T4. `RSSMCore.tiny` and `winit`.** `tiny` moves to `reference/raif/
  world.py` beside `tiny_rssm`; if a generic constructor is wanted it is
  `from_sizes` with sizes required and no config-row defaults. `winit` is
  renamed to a neutral initialiser selector (`init=`) across `neural.py`,
  `factors.py`, `rssm.py`; the sibling `from_shapes` defaults are aligned or
  made required so no default is silently the `tiny` row. Note `winit` is
  the parity recipe's knob (`raif_mountain_car(winit="normal")`), so the
  reference agent carries the rename. *Ruled: yes.*
- **T5. The actor trainer.** `wment` and `bc_scale` default to `0.0` (off),
  with R-AIF's values set in `reference/raif/agent.py`; rename `wment`,
  `actent`, `can_self_imitate` to descriptive names (`prior_entropy_scale`,
  `policy_entropy_scale`, `bc_weights`). This changes `imagine`'s signature:
  an engine API change. *Ruled: yes.*
- **T6. The ensemble trainer and `world.embeddings`.** `learning_gate`/
  `NO_LEARNING` either move to reference as a wrapper or stay as a neutral
  "member dropout" with a one-line docstring; the batch builder takes the
  faces' belief tuple (or `(deter, code)` arrays) rather than `RSSMState`;
  `embeddings()` moves to `reference/raif/crspp.py`. *Ruled (Alex: drop
  unless generically useful): the random gate, its floor and the config
  name go; what stays is a generic `member_weights=` argument on the
  ensemble loss, an `(M,)` per-member loss weight defaulting to ones.
  Reason: the rate grammar addresses slots, and the members live inside
  one slot, so per-member control is a real gap the grammar cannot
  reach; a per-member weight is its smallest general form (bootstrap
  masks, curricula, or R-AIF's own random gate drawn by the caller).
  Extending the grammar to sub-slot addresses is a separate, larger
  change and is not taken here. The divergence register keeps the record
  that the transcribed gate was inert at parity.*
- **T7. Error strings and the row list.** Drop the reference-module names
  from the `as_env` fence and the selectors error (the generic sentence
  before each is complete); derive the mounted-row list in `core.py` from
  `registry.ROWS`. Reword every error that quotes their config as a
  structural requirement in our vocabulary. *Ruled: yes.*
- **T8. The layout of §4**, with `envs/mountain_car.py` staying at
  `reference/envs/` as the one benchmark env (half its consumers are not
  R-AIF's), `worlds.py` disappearing rather than becoming a shim (tier
  clause 2: no deprecation burden; the seven tests rewrite their import), and
  the `system.md` addendum rewritten by dated amendment to list the subtree
  and the crspp/revise cut at its new path. *Ruled: yes.*
- **T9. Prose.** Every engine docstring and comment listed in §2 is rewritten
  to describe functionality; narrative that is not already in
  `docs/experiments/raif-mountain-car/` or the debt ledger is moved there
  before deletion, not dropped; the two stale docstrings are corrected.
  Reference-tier modules name their target in line 1, with the package
  docstring carrying the citation vocabulary once. *Ruled: yes, keeping
  generic Mountain Car commentary apart from R-AIF's; see the note below.*
- **T10. Optional hoists.** `seed_stats` and the fixture loader/saver to the
  engine's `train.replay` neighbourhood, with the Mountain Car widths passed
  in. Not required for the move; can wait for a second replication.
  *Ruled: hoist `seed_stats` in this round (generic over `Episode`, no
  width pin); the fixture loader waits, since hoisting it means
  parameterising its Mountain Car widths with no second case to check
  against.*

**Order and gates.** Everything above waits for `runs/crspp_init_100k` to
reach its 500k gate (the harness is mid-run and T1/T4/T5 touch what it
loads). One refactor round, adversarially reviewed before commit, with the
full gates and the notebook exhibit re-executed. T4 and T5 change engine
signatures, so the round is a minor version (0.9.0), not a patch. The
filtering question (sampled versus mean-field as two general shapes) is
**not** decided here: it goes to the 6.5 round on top of the tree this round
leaves, where the generic/specific line is legible.
