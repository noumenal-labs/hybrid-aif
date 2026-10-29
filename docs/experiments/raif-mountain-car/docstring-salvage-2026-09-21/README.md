# Docstring salvage, 2026-09-21 — index

The tier-split prose pass (`docs/tier-split-audit.md`, T9) removed the replication
narrative from fourteen engine modules' docstrings. It first copied every passage
here verbatim, arranged by module and symbol. The distillation pass of 2026-09-30
checked each passage against the records and moved the facts that were recorded
nowhere else. It then deleted the four content files; they are at `a0f4916`
(`git show a0f4916:<this folder>/carriers.md`, and likewise `neural.md`,
`trainers.md`, `objectives.md`). Each line below says where one section's fact
now lives. **dup** marks a section whose fact was already recorded there.
**moved** marks one this pass added. **nil** marks one with nothing to keep: a
stale claim, a mechanism the docstring still states, or text still in the code.

Key: **A** `docs/design/amortization.md` · **Add** `amortization-p0-addendum.md` ·
**DL** `amortization-debt-ledger.md` · **MC** `module-correspondence.md` (**MC+**
its 2026-09-30 addendum) · **ELT**/**ELA** the training/acting execution ledgers
(**ELA+** the acting ledger's 2026-09-30 D8 addendum) · **DR** `divergence-register.md` ·
**A+** A's 2026-09-30 addendum · **PH** the build log.

## carriers.md

- `train/replay.py::module`: placement dup A chunk 2c §(d′); flat stream dup ELT 1.3, ELA D8; alternation dup A A10 disp. 9
- `StaticBuffer`: dup A chunk 2c §(d) FLAG (7) note, ELA D8
- `StaticBuffer.ready`: dup A A10 disp. 9
- `sample`: dup Add §5, ELT 1.3
- `StreamBuffer`: padding dup A chunk 2c §(d) FLAG (7) note, ELA D8; `Uniform` term for term, one-stream argument moved ELA+
- `StreamBuffer.windows`, `StreamBuffer.ready`: moved ELA+
- `extend`: nil (stale); the ρ instance dup ELA D8
- `sample_windows`: dup ELT 1.3, ELA+
- `inference/rssm.py::module`: dup Add §2–3, MC §1.2, A A1
- `RSSMState`, `RSSMCore`, heads: dup Add §2 gap note 2, MC §1.2/§1.4
- `RSSMCore.from_declarations`: nil (superseded by T3, A "Tier-split refactor, phase two, as built")
- `RSSMCore.initial_state`: dup DR, PH; correspondence MC §1.2 `rssm/initial`
- `RSSMCore.img_step` / `obs_step`, `observe`: dup ELT 1.11, ELA D8; moved MC+
- `inference/amortized.py::module`: measurements dup `pref-rollout-probe-500k.md`, PH; "row" naming moved A+ item 1
- `EXTERNAL`: dup DL entry 5
- `RecognitionPlan`, `head_parents`: dup Add §2 gap note 2, MC §1.1
- `validate_structure`: dup A A12 item 2, A10 disp. 4
- `initial_state`, `predict` / `recognize`: dup A A11, MC §1.2; moved MC+
- `factors.py::module`, `Deterministic`, `Deterministic.gru`: dup A A10 disp. 2, A11, MC §1.2
- `DerivedInitial`, `Recognition`: dup A A11, MC §1.2
- `controls.py::module`: dup A A9; the list's current membership moved A+ item 3
- `soft_clip`: dup ELT §10 item 22, ELA D5, `actor-trace-audit.md` A5; moved MC+

## neural.md

- `module`: dup Add §2/§3, MC; bounded head dup A chunk 2d §(f) 15; LayerNorm dup A §(f) 14
- `adam`: dup A chunk 2d §(a) item 5, ELT 5.9, 6.43
- `_ACTIVATIONS`, `symlog`, `unimix_logits`: dup Add §2, A chunk 2d §(e)
- `_TRUNC_NORMAL_STD`, `trunc_normal_avg`: dup A chunk 2d §(f) 17, MC conventions 3; moved MC+
- `uniform_avg`: dup ELT §10 item 26 (26b), MC, A §(f) 17
- `MLPConditional.from_shapes`: dup DR "Closed, by round" (14–17), training ledger §10 item 26b, A chunk 2d §(e)
- `MLPConditional.diag_moments`: nil (sizing illustration; A chunk 2c §(e) item 1)
- `MLPConditional.diag_nll`: dup A chunk 2c §(e) item 1, ELT 5.5
- `_linear_stack`: dup A chunk 2d §(f) 17, DR row 17 + 26 (which supersedes the 2026-09-07 state)
- `MLPPredictor`, `PredictorHead`: dup Add §2, A chunk 2d §(e)
- `MLPCategorical`: dup MC §1.2
- `factored_categorical_kl`, `symlog_mse_nll` / `disc_bins` / `twohot`, `binary_nll`: moved MC+
- `factored_categorical_entropy`: dup Add §3, A A12 ("`imagine()` MUST sample")
- `kl_balance`: dup Add §3, A chunk 2d §(e)
- `symlog_disc_nll`: dup A A10 disp. 6, A12 item 5
- `GRUCell`: dup MC §1.2 `rssm/gru`, DR row 17 + 26
- `LearnedVector`: dup MC §1.2 `rssm/initial`; the frozen instance moved MC+
- "Left in the code": nil here. Still in code: `GRUCell.from_shapes`'s inner function `theirs`, and two "see divergence 29" error strings in `train/imagine.py`

## trainers.md

- `imagine.py` module: dup A chunk 2c §(d) FLAG (4), FLAG (6), A chunk 2d §(a) item 5, DL entry 5; the renames dup A "Tier-split refactor, phase two, as built" (T4, T5)
- `DEFAULT_LR` / `CLIP` / `EPS`, `DEFAULT_DISCOUNT` / `LAMBDA` / `SLOW_*` / `RETNORM`: dup A chunk 2d §(a) item 5, §(e), Add §5
- `SCORING_KEY_TAG`: dup A chunk 2c §(a), PH
- `MOMENT_KEYS`: dup A chunk 2d §(a″) 13; moved MC+
- `ImagineMetrics`: dup `series-sweep.md` §6.1, `plateau-analysis.md` §7 item 2, A chunk 2d §(a″) 11
- `_cont_mode`, `_slow_update`: dup A chunk 2d §(a″) 10, 13, DR
- `rollout_weight`, `lambda_return` / `perc_ema` / `init_retnorm`: dup ELT §6 (6.28), A chunk 2d §(a″) 11
- `term_value`: dup A chunk 2c §(d) FLAG (4), RESOLVED 2026-08-28 (the salvage lead said "nowhere else"; that section records it)
- `imagine`: dup Add §5, ELT §6–§7, A chunk 2d §(a) item 5
- `roll_batch`: dup A chunk 2d §(a″) 12, ELT §10 item 22, ELA D5
- `_sampled_step` / `_code_and_carry` / `_cont_head`: dup A A12 ("`imagine()` MUST sample"); error strings nil (still in code)
- `ensemble.py` module, its defaults, `ensemble_loss`: dup A chunk 2d §(a) items 4–5, ELT 5.7; `ROWS` dup A chunk 2c §(d) FLAG (4)
- `fit_ensemble`: dup DL entry 6
- `world.py` module, `DEFAULT_*`, `WorldBatch`, `world_loss`: dup Add §3, §5, A A6, chunk 2d §(a) item 5, ELT 1.7
- `fit_world`: dup DL entry 16, A A10 disp. 9
- `posterior_states`: dup A A12 ("`imagine()` MUST sample"); the `embeddings` clause nil (stale)
- `train/__init__.py`: dup A chunk 2c §(d′), A10 disp. 9, A6

## objectives.md

- `objectives/amortized.py` module: dup A A10 disp. 7, A12 item 4, `objective-terms.md`
- `features`, `_occupancy`, `_predicted_reward`, `make_transition`: dup MC §1.4; moved MC+; the stale clauses nil
- `_diag_entropy`: dup A chunk 2c §(e) item 2; jit-only discovery moved A+ item 4
- `_ensemble_disagreement`: dup ELT §10 item 20, DR, A chunk 2d §(a) items 1, 3 and §(b), PH 058
- `_g_tail`: dup A A12 item 5, Add §3
- `MODEL_UNCERTAINTY`, `objectives/base.py::term_sign`: dup A chunk 2c §(d) FLAG (4), `objective-terms.md`
- `registry.py` module, `StepContext`: dup A A10 disp. 3, A12 ("`imagine()` MUST sample")
- `Estimator`, `LearnState`, `init_learn_state`, `selectors.py::_resolve_rates` (comment): dup A A12, DL entry 6
- `ACTOR_KEY`: dup Add §3, ELT §6
- `SystemParams`: nil (the E-is-γ-independent note is still a field comment in `system/core.py`)
- `Subsystem`: nil (`subsystem()` shipped; `nesting-systems.md`)
- `System.scoring`, `external_inputs`, `_amortized_scoring`: dup A A12 item 2, DL entry 5
- `initial_priors`, `infer_states`, `planner_context`, `_amortized_context`, `update_empirical_prior`: dup MC §1.2; moved MC+
- `System.learn` (docstring): dup A A2
- `System.learn` (return-type comment): moved A+ item 2
- `System.terms`: dup A "A12 implementation note: the `terms=` key grammar"
- `System._habit_fn`, `planning/habit.py::_from_module`, `habit_action`: dup ELA D5, PH
- `planning/habit.py` module: dup A A12 item 3
- `selectors.py` module: dup DL entry 1, `system.md`
- `selectors.py::_resolve_rates` (docstring): dup `per-variable-infogain.md` (V7 specificity ladder; the salvage lead said "nowhere else")
- `validation.py::_check_pattern`: nil (the error text was made generic)
- `objectives/hybrid.py::_predict_mlp_gaussian`: dup A A10 disp. 10
- `replay.py::replay`: nil (mechanism kept; `reference/raif/run.py::collect`)
- `planning/enumerate.py::_prior_logits`: dup DL entry 12
