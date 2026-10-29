# P0 addendum — the R-AIF target, pinned

*Status: **reviewed and adjudicated** (Alex + Fable, 2026-08-06;
drafted by a P0 research agent 2026-08-04). Executes P0 of the Build
plan in `amortization.md`; §7 was rewritten at review (match paper
numbers, no pre-lowered bars) and the eleven §8 gaps carry
dispositions in `amortization.md` **A10**. Sources: arXiv:2409.14216v1 (the only
arXiv version; §/Eq. numbers below are that HTML rendering), its
appendix `docs/ImplementationDetailsDocumentation.pdf` in the official
repo (cited as **IDD §X**), and the repo
`NACLab/self-revising-active-inference` @ `main` (cited by file:line).
Published as **SR-AIF**, ICRA 2025 pp. 6510–6518
(doi 10.1109/ICRA55743.2025.11127713) — the repo README's bibtex; the
name changed from R-AIF, and **the IEEE version was not read**, so any
number below is the arXiv/repo number. Nothing in `amortization.md` is
edited by this file; §8 raises gaps for adjudication, it does not
resolve them.*

## 1. Their Mountain Car

Not gymnasium's `MountainCarContinuous-v0`. It is a vendored, modified
copy registered as `"mtc"`
(`raif/embodied/envs/__init__.py:3` → `raif/embodied/envs/customs/mtc.py`).

| Property | Value | Cite |
|---|---|---|
| Class | `MountainCarContinuousImageObservation(MountainCarContinuous)` | `mtc.py:333`, `:53` |
| Observation | dict: `image` `Box(0,255,(64,64,3),uint8)` **and** `state` `Box((2,),float32)` = (position, velocity) | `mtc.py:356-362` |
| Obs actually fed to agent | `image` only (`encoder.mlp_keys:'^$'`, `cnn_keys:'image'`) | `raif/aif/configs.yaml:117-118` |
| Frame stack | none — single frame; temporal info lives in the RSSM's GRU | IDD §A |
| Action | `Box(-1,1,(1,))`, `power=0.0015`, `max_speed=0.07`, position ∈ [-1.2, 0.6] | `mtc.py:117-126,143` |
| Reward (sparse) | `-1.0` every step; `0.0` on the terminating step | `mtc.py:191-194`; paper §4 "Environment Setup" |
| Termination | `position >= 0.45` (`easy_mode=True` default, velocity ignored) | `mtc.py:122,175-176` |
| Episode length | `TimeLimit(100)` **agent** steps with `ActionRepeat(2)` ⇒ ≤200 raw env steps | `configs.yaml:116`; `wrappers.py:27,54` |
| Success criterion | `is_success = state[0] >= 0.45` | `wrappers.py:660` |
| Cosmetic change | dark background / white car "to improve visualization for human experimenters" | IDD §G |

Consequences of `ActionRepeat(2)`: reward per agent step is `-2`, so the
floor of the episode score is `-200` (matches DAIVPG-G's `-200.0±0.0`,
Table 2), and R-AIF's `-68.3` ⇒ ≈34 agent steps ≈68 raw env steps to
the flag.

**A state-based variant already exists in their code** (`state` is in
the obs dict, merely filtered out by the encoder regex). Flipping
`encoder/decoder.mlp_keys` to `'state'` and `cnn_keys` to `'^$'` yields
a state-based MC on their exact dynamics/reward/limits. **No numbers
are published for it** — see §7 (OUR variant).

## 2. Architecture

Fork of DreamerV3's `embodied`/`ninjax` codebase; five parameter sets:
world model θ, CRSPP φ, actor ψ, critic χ, IG ensemble {ω_i} (IDD
Alg. 2).

- **World model**: RSSM (paper §3.1, Eq. 3) — GRU deterministic carry
  `deter` plus **categorical** stochastic latent: `stoch=32` variables ×
  `classes=32`, one-hot, `unimix=0.01`
  (`configs.yaml:69`; `nets.py:28-76`, `CRSPP_RSSM`). Explicitly *not*
  Gaussian latents (paper §3.1 final ¶; IDD §B).
- **Encoder / decoder**: `MultiEncoder`/`MultiDecoder`, ResNet CNN
  (`cnn: resnet`, `minres: 4`) + MLP branch; image dist `mse`, vector
  dist `symlog_mse`; `symlog_inputs: True`. Reward and continuation
  heads are MLPs over `[deter, stoch]` — reward `symlog_disc`,
  `bins: 255` over [-20,20]; cont `binary`
  (`configs.yaml:70-73`; `nets.py:279,329,568-583`).
- **Actor**: MLP over `[deter, stoch]`, `dist: normal` ⇒
  `tfd.Normal(tanh(out), std)` with `std = (maxstd-minstd)·σ(s+2)+minstd`,
  `minstd 0.1 / maxstd 1.0` (`configs.yaml:82,87`; `nets.py:573-580`).
  Matches paper §3.3's "tanh-normalized … Gaussian".
- **Critic**: MLP, `symlog_disc`, 255 bins, plus a slow copy
  (`slow_critic_fraction 0.02`, `slow_critic_update 1`) used as a
  `logprob` regularizer (`configs.yaml:83,94-96`; `agent.py:504-546`).
- **CRSPP (the prior-preference model)**: "an RSSM without action
  encoded into the transition prior" (paper §3.2) — implemented as a
  *second parameter set inside the same `CRSPP_RSSM` module*:
  `observe_no_action` / `img_step_no_action` with their own
  `img_in_2/img_out_2/obs_out_2/…_stats_2` layers and a second GRU
  `_gru_2` (`nets.py:90,151,189,220`). It has **no decoder of its own**
  (IDD §J); the world-model decoder is reused only for visualization.
- **IG ensemble**: `ig_models: 4` MLPs, ~~`layers 3, units 256`~~ **`layers
  2, units 32`**, `dist: normal`, inputs `[deter, stoch, action]`, target
  `[stoch]` (`configs.yaml:102-106`; `epistemics.py:23-61`).
  **CORRECTION, 2026-09-02.** The struck size was wrong, and it contradicted
  the *next* bullet, which is right: under `tiny` the regexes `.*layers 2` /
  `.*units 32` cover `ig_head` and `critic`, with `actor {layers 3, units
  256}` named as the exception precisely because it *is* one. The oracle's
  resolved config settles it — `ig_head: {layers: 2, units: 32}` and
  `critic: {layers: 2, units: 32}`. The build followed this bullet rather
  than the next one and shipped a critic and an ensemble ~8× too large each;
  see `amortization.md`'s chunk-2d note §(a′), divergences 6 and 7, and the
  audit rule recorded at the top of that note (**resolve a target's config by
  running it, not by reading it**). Also worth correcting here: `dist:
  normal` on `ig_head` passes no `minstd`/`maxstd`, so their member σ is the
  **constant 1.0** — divergence 15, §(f), still open.
- **Sizes**: the defaults are large (`deter 4096`, `units 1024`,
  `layers 5`, `cnn_depth 96`), but the README's MC command is
  `--configs gym_mtc tiny`, i.e. `cnn_depth 16`, `.*layers 2`,
  `.*units 32`, `mlp_units 64`, `actor {layers 3, units 256}`,
  `rssm {deter 128, stoch 32, classes 32}` (`configs.yaml:149-158`;
  README "Experiment(s)"). **Which size produced Table 2 is not stated
  anywhere** — unverified (§9).

## 3. Objective

- **World-model loss** (paper Eq. 4): KL(q(s|o) ‖ p(s|s₋,a₋))
  (complexity) + reconstruction NLL (accuracy), with KL balancing
  `rep 0.1 / dyn 0.5`, free bits `1.0`, symlog inputs (IDD §B Eq. 1;
  `configs.yaml:74-75,79`; `agent.py:212-219`).
- **EFE** (paper Eq. 7a–7d, derived in IDD §H Eq. 6–11):
  `G_τ = −H[E_Q Q(π|o,s)] + (H[E Q(s|θ,π)] − E H[Q(s|θ,π)]) − E H[Q(s|θ,π)] − r_τ − sim(s_τ, ŝ_τ)`
  i.e. policy entropy (7a), **parameter** info gain (7b), generative
  world-model state entropy (7c, sign flipped deliberately to *reward*
  state uncertainty — IDD §H after Eq. 9), and the instrumental term
  (7d) = predicted reward + cosine similarity to the CRSPP-imagined
  preferred state.
- **`sim`** is plain cosine similarity (paper §3.2, before Eq. 5). In
  code it is computed on the **concatenation of `deter` and the
  flattened one-hot `stoch`** (`agent.py:165-179`, `cosine_sim`).
- **What the critic actually scores**: `G = reward + sim − IG`
  (`agent.py:548-577`). So the critic is a value of *negative EFE*, not
  of reward. **Corrected 2026-08-28** (the draft attributed the minus
  sign to `G_scales`, `configs.yaml:112`): the scales there are all
  `1.0`, `ig: 1.0` included, and the sign is in the code —
  `neg_param_info_gain = -igfn(traj)` (`agent.py:561`). It is a
  modelling claim, not a weight, and the paper says so twice: Eq. 6
  writes the value as `Σ γ^(t−τ)(r + sim − IG)` and calls the term "the
  negative information gain `−IG_τ`", while Eq. 7b's prose has the
  policy "trained to reduce the model parameter's uncertainty … takes
  more certain actions to maintain 'homeostasis'" — exploration being
  carried by 7c/7a instead. Consequence for the port: this is a
  registered row of its own (`model_uncertainty`, sign +1), never a
  scale on an info-gain term (`amortization.md` A12 chunk-2c note,
  FLAG (4) RESOLVED).
- **Critic target** (paper Eq. 8): λ-return over `G` with
  `discount = 1 − 1/horizon = 1 − 1/333 ≈ 0.997`, `return_lambda 0.95`,
  bootstrapped by the critic at the leaf, multiplied by the predicted
  continuation `c_τ` (`configs.yaml:92-93`; `agent.py:569-576`).
- **Critic loss**: paper Eq. 9 says MSE; **the code minimizes
  `−log_prob` of the 255-bin discrete distribution plus a slow-target
  logprob regularizer** (`agent.py:528-546`). Divergence, flagged in §8.
- **Actor objective** (paper Eq. 10): maximize
  `f_χ(v|s) + ζ·H[p_θ(s|s₋,a₋)] + η·H[π_ψ(a|s)] + ln(a*_τ)` with
  `ζ = η = 3e-4` (`actent`/`wment`, `configs.yaml:98-99`). Implemented
  as: normalized advantage `(ret − base)` via percentile-EMA
  (`retnorm: perc_ema, perclo 5, perchi 95, decay 0.99, max 1.0`;
  IDD §E Eq. 4–5), `actor_grad_cont: backprop` (reparameterized, no
  REINFORCE), minus `actent·H[π]`, minus `wment·H[state prior]`, plus
  the **actor-refresh / BC term** `−log π(a_start) · can_self_imitate`
  scaled `0.5` (`configs.yaml:79`; `agent.py:444-484`).
- **IG estimator** (IDD §D Eq. 2–3): ensemble trained by Gaussian NLL
  on next `stoch`; `IG = ent(std_i({s_i})) − E_i[ent(σ_i)]`, scaled by
  `1/sqrt(stoch) = 1/sqrt(32)` (`epistemics.py:33-45`). Three details of
  that function are load-bearing and were only pinned in P1′ chunk 2d,
  when the smoke measured them: `ent` is a **per-dimension mean**
  (`.mean(-1)`, `:41-43`) and not a joint entropy; the `{s_i}` whose
  spread is taken are one **sample** per member (`dist.sample`, `:39-40`)
  and not the members' means; and `:57-61`'s
  `ig_head_random_no_learning` gates each member's loss per step (their
  value, and the oracle's, is `0.0`). Our port matches the first and
  third; on the second it **deliberately diverges** — the std of the
  member means, argued with measurements in `amortization.md`'s chunk-2d
  note per A8's rule, and named there as the first suspect if the
  replication lags §7's landmarks.

## 4. Self-revision, exactly

Two buffers, and a per-episode backward pass computing `ρ_t ∈ [-1,1]`.

- **Initialization**: `M⁺` (positive buffer) is seeded by
  `scripts/collect_prior.py` running a **pretrained scripted expert**
  (`raif/sb3/agent.py`, `ExpertAgent`), every step marked
  `can_self_imitate = True` (`run/collect_prior.py:105-109`). The CRSPP
  nets are randomly initialized like any other module — there is no
  preference-parameter prior.
- **Schedule**: `ρ` is recomputed **once per finished episode**, as a
  driver `on_episode` callback, before the episode is written to the
  buffers (`run/train.py:134-135`; IDD §C Alg. 1;
  `run/callbacks.py:31-59` `lookback_on_episode`). Per-step flags
  recorded by the env wrapper: `is_success` (goal reached this step),
  `has_succeeded` (this episode ever), plus `can_self_imitate`
  (expert-authored step) (`wrappers.py:685-722`).
- **Update rule** (code, matching IDD Alg. 1): expert flags are gated
  by episode success (`can_self_imitate *= has_succeeded[-1]`); then a
  backward carry `c ← max(α·c, is_success_t)` with `α = disc = 0.4`,
  quantized to 0 below `ε = thr = 0.001`; the positive mask is
  `clip(can_self_imitate + c, 0, 1)` **shifted one step** into
  `positive_mask`. Failure branch: `neg_t = −β^(T−1−t)` with
  `β = fail = 0.3` — so the *last* step of a failed episode is condemned
  at −1, not at −β (their `[fail]*T` cumprod, reversed, divided by
  `fail`: exponents `T−1 … 0`; corrected 2026-08-28 against
  `callbacks.py::lookback_on_episode`, which the P1′ chunk-2c port reads
  line by line — the draft said `−β^(T−t)`) — zeroed when the episode
  succeeded and on expert steps. Finally
  `pref_label = clip(positive_mask + neg, −1, 1)` with `pref_label[0] = 0`
  (`callbacks.py:34-58`; hyperparameters `configs.yaml:109`).
- **Consumption**: `pref_label` (= ρ) does two jobs — (i) it gates the
  CRSPP KL loss to *positive states only* (`max(0, sgn ρ)`, paper Eq. 5;
  `agent.py:225-226`) and (ii) it scales the contrastive
  cosine-similarity term `−ρ · sim(C_post, sg(post))` with
  `loss_scales.contrast = 1.0` (`agent.py:229-231`). `can_self_imitate`
  separately gates the actor's BC term (§3). Note the code's
  `can_self_imitate` picks up *the agent's own* near-success steps, not
  only expert steps — so the "refresh" term is genuine self-imitation.
- **Buffer routing**: episodes with `has_succeeded[-1]` go to **both**
  `M⁺` and `M`; failures only to `M` (`callbacks.py:17-29`). Training
  strictly alternates the two buffers (`i % 2`), falling back to `M`
  while `M⁺` is under one batch (`run/train.py:158-167`; IDD §I) — **per
  the paper; the code's alternation is inert and the parity recipe does
  not alternate** (FLAG 18, adjudicated 2026-09-02 — see §7's note).

## 5. Training protocol

| Knob | Value | Cite |
|---|---|---|
| Replay | uniform, `replay_size 5e5` **per buffer** (two buffers) | `configs.yaml:7-8`; `train_aif.py:69-70` |
| Batch (MC) | `batch_size 16`, `batch_length 64` ⇒ `batch_steps 1024` | `configs.yaml:120-121`; `train_aif.py:56` |
| Batch (MW / robosuite) | `6 × 20` | `configs.yaml:131-132,141-142` |
| Update ratio | `train_ratio 64.0` ⇒ `Ratio(64/1024)` = **1 gradient step per 16 agent steps** | `configs.yaml:35`; `run/train.py:108` |
| Imagination horizon | `imag_horizon 8` (MC), `4` (MW, robosuite) | `configs.yaml:122,133,143` |
| Optimizers | model `adam lr 1e-4 clip 1000`; actor & critic `adam lr 3e-5 clip 100`; IG `adam lr 1e-4` | `configs.yaml:76,84-85,102` |
| Precision | float16 compute, GPU | `configs.yaml:17` |
| Envs | 4 parallel processes | `configs.yaml:50` |
| Steps | 1M / 3M / 5M for MC / Meta-World / robosuite | paper §4 |
| Seeds | **4** uniquely seeded trials per (agent, env) | paper §4 |
| Seed/expert data | ≈1,000 steps ≈22 episodes (MC); 3,300 ≈6 eps (MW); 20,000 ≈40 eps (robosuite) | paper §4; `configs_expert.yaml:73,85,90` |

**Expert data is scripted, not human.** Experts are SB3 policies
trained on the **MDP** (state-observation) version of each task: PPO
for MC and Meta-World, SAC for robosuite (paper §4;
`raif/sb3/configs_expert.yaml:53,56,59`), checked in as
`expert_models/ppo_gym_mtc.zip`, `sac_robosuite_{Door,Lift,Wipe}.zip`.
The repo *also* ships `experimental/robosuite_keyboard.py` for
human/keyboard collection, but it is out of the documented pipeline and
the paper attributes robosuite seed data to SAC — see §9.

## 6. Evaluation and numbers

Metrics are **online training** statistics over "the last 100 recorded
agent training episodes" (paper §4), reported as mean ± sd over 4 seeds.

- **ACR** — average cumulative reward = mean episode score, raw summed
  reward (`callbacks.py:64,72-73`). No offset.
- **SR** — fraction of the last-100-episode window in which the goal
  was reached at least once (`wrappers.py:639-722`, `window=100`).
- **R-S** (relative stability) — `mean(|rolling_mean_100(C) − max(C)| /
  max(C))` over the last 100 windows, where `C` is the per-episode sum
  of `(reward + r_reward_offset)`; offset is `1.1` for MC, `3.1` for
  Meta-World, `1.0` for robosuite (`wrappers.py:724-763`;
  `configs.yaml:116,126,137`). **Lower is better**; it measures how far
  the running mean sits below the best episode ever seen. The offset
  is a metric-only rescaling and must be replicated exactly or R-S is
  not comparable.

**Baselines** (paper §4): DreamerV3; **DAIVPG-G** (their
generalization of ref [47]); **Himst-G** (ref [82], with its stacked-frame
3D-conv replaced by a state-space model). None of the three is in the
public repo.

**Table 2, the rows we need** (paper §4; ACR / R-S / SR):

| Task | R-AIF | DreamerV3 | DAIVPG-G | Himst-G |
|---|---|---|---|---|
| Mountain Car | **−68.3±1.0 / 0.1±0.0 / 1.0±0.0** | −79.7±12.4 / 0.4±0.2 / 0.9±0.1 | −200.0±0.0 / 0.7±0.5 / 0.0±0.0 | −199.9±0.2 / 0.8±0.3 / 0.0±0.0 |
| R Door Open | **425.8±10.0 / 0.1±0.0 / 1.0±0.0** | 0.0±0.0 / 0.0±0.0 / 0.0±0.0 | 0.0±0.0 / 0.0±0.0 / 0.0±0.0 | 0.4±0.4 / 0.3±0.2 / 0.0±0.0 |
| R Block Lift | **120.8±159.8 / 0.4±0.2 / 0.4±0.4** | 0.0±0.0 / 0.0±0.0 / 0.0±0.0 | 0.0±0.0 / 0.1±0.1 / 0.0±0.0 | 0.1±0.1 / 0.0±0.0 / 0.0±0.0 |
| M Door Open | **−62.5±0.6 / 0.0±0.0 / 1.0±0.0** | −69.9±2.6 / 0.0±0.0 / 1.0±0.0 | −440.0±54.5 / 0.8±0.1 / 0.2±0.2 | −481.0±0.0 / 0.7±0.2 / 0.0±0.0 |

(All 13 Meta-World rows are in the paper's Table 2; R-AIF takes SR 1.0
on 8 of 13 and the best ACR on 8 of 13. Meta-World's observed floor is
−481, ≈ `TimeLimit(160) × ActionRepeat(3)`; the exact off-by-one is not
worth chasing but must be matched empirically, not derived. Hardest
rows for everyone: M Pick Place, M Disassemble — R-AIF SR 0.1 / 0.0.)

Robosuite: sparse reward is **+1 per step in the success state, 0
otherwise** (IDD §G) and the episode does *not* end at the goal, with
`TimeLimit(500)`, `repeat 1` — so ACR ≈ number of successful steps out
of 500, and 425.8 means the Door is opened at ≈step 74 and held.

**Curves (Figure 4)** are cumulative reward (y) vs. environment steps
(x), one panel per task, with pink dashed lines for the MDP expert's
mean. I could not read numeric values off the figure (raster image);
qualitatively (paper §4 "Results"): R-AIF converges earliest on most
tasks; on MC DreamerV3 is flat early and only rises once its random
exploration banks successes; R-AIF exceeds the expert line on some
tasks (e.g. M Push, SR 0.6 vs DreamerV3 0.1); on robosuite R-AIF is the
only agent to leave zero. Endpoints = the Table 2 ACR values above.

## 7. Acceptance (adjudicated: match the paper, no pre-lowered bars)

**Ruling (Alex, 2026-08-06):** the a-priori acceptance is *match
paper numbers*. Tolerance bands are not set in advance — a shortfall
triggers discussion and possibly iterative debugging rounds, not a
pre-excused pass. Low bars discourage experimentation; the earlier
draft's engineered tolerances (ACR ≥ −110 / ≥ −90 etc.) are
superseded. Where a "planning-horizon tax" or similar argument
predicts a principled gap (search-only P1 vs. amortized value), that
prediction is something to *measure and report*, not to bake into the
bar.

**P1 — state-based MC.** Their env with obs = `state` (position,
velocity): same dynamics, sparse reward, `goal=0.45`,
`TimeLimit(100) × ActionRepeat(2)`. No expert data, no CRSPP, no
habit: CEM (A9) over a learned world model. They publish no
state-based number, so the target is generated, not invented: run
their code with
`--encoder.mlp_keys state --encoder.cnn_keys '^$' --decoder.mlp_keys state --decoder.cnn_keys '^$'`
and **match that oracle number** (SR / ACR, same seeds and steps as
the oracle run). The search-only-vs-amortized gap, if any, gets
measured against the same oracle and reported.

**The oracle record (complete, 2026-09-01; full report: the "Oracle
State Baseline — Mountain Car (P1′)" artifact).**

- **Commit** `9b1864e8fab9d1becd6dd248f2c590d06ff22848` — repo
  `raif-paper`, upstream NACLab/robust-active-inference at the initial
  fork + citation commit; no code changes from the paper release.
- **Commands** (per seed N ∈ {0,1,2,3}):
  `collect_prior.py --configs collect gym_mtc --run.steps 2000
  --expname oracle_state_mc_seedN --seed N`, then
  `train_aif.py --configs gym_mtc tiny --run.steps 2000000` with the
  four state-key regexes (`--encoder.mlp_keys state --encoder.cnn_keys
  '^$' --decoder.mlp_keys state --decoder.cnn_keys '^$'`) and
  `--jax.platform cpu` (device-only). Paper defaults otherwise,
  incl. `replay_size 5e5`. NOTE: the oracle's expert seeding is
  **2,000 steps per seed**, not the paper's ≈1,000 — our replication
  seeds from the oracle's actual episodes (the fixture export), so
  this is parity-neutral but must not be "corrected" to the paper.
- **Environment**: Apple M3 Max, CPU-only jax 0.4.26, ~50 awake-hours
  per seed; seed 3 resumed twice from on-disk checkpoints (host
  reboots), identical command, no config change.
- **Numbers** (agent steps; ×2 for env frames — counter verified):

  | seed | best 200k window | final-100 ACR | last-500k mean | SR (> −100, last 500k) |
  |---|---|---|---|---|
  | 0 | −68.8 @ 400–600k | −68.3 | −70.7 | 96.4% |
  | 1 | −68.7 @ 1800–2000k | −67.8 | −71.8 | 95.5% |
  | 2 | −69.5 @ 1600–1800k | −82.2 (mid-dip) | −72.0 | 96.9% |
  | 3 | −68.1 @ 400–600k | −67.6 | −73.2 | 94.0% |

  Expert (PPO) seed-data mean: **−76.1** — every seed's converged
  agent beats its own expert by ~7 points.
- **The P1′ target, stated window-honestly** (per this section's
  ruling): converged ACR ≈ **−68 to −69.5** judged on windowed
  statistics — best-200k window and last-500k mean — never single
  endpoint reads (seed 2's final-100 lands mid-dip while its windows
  match the others; the recurrent self-revision dips with full
  recoveries are characteristic behavior our replication should
  SHARE, not smooth away). The trajectory shape is part of the
  target: ~~off the −200 floor by ~50k~~ **off the floor within ~4k
  and at ≈−88 by 15k** (see the correction below), a universal dip
  near 200k, convergence by 400–500k.
- **CORRECTION, 2026-09-02 — the early landmark was wrong, and it was
  wrong in the *lenient* direction.** Recomputed directly from the
  oracle's own `logs/oracle_state_mc_seedN/scores.jsonl` (the
  `episode/score` rows; the `prior_episode/score` rows are the expert
  collect stream and are not the agent learning). The rest of this
  section's table reproduces **exactly** from those files — best-200k
  −68.8 @ 400–600k, −68.7 @ 1800–2000k, −69.5 @ 1600–1800k, −68.1 @
  400–600k, and the last-500k and SR columns too — so the step axis is
  the right one and only this landmark is at fault. On 2,048-step
  windows the four oracle seeds read:

      0–2k   −200 −198 −200 −192      (on the floor: one window)
      2–4k   −167 −199 −177 −190
      4–6k   −160 −134 −144 −130      (already past −150)
      8–10k  −133 −131 −111 −125
      10–12k −111 −101 −116 −101
      14–16k  −89  −98  −98 −111
      28–30k  −90  −90  −86  −86      (plateau ≈ −88)

  So the oracle is on the −200 floor for roughly its **first 2k agent
  steps**, passes −150 by ~4–6k, and plateaus near **−88** from ~15k
  until the 200k dip. "By ~50k" overstated the time to leave the floor
  by more than an order of magnitude, which matters in exactly one
  way: it makes a *smoke* look acceptable when it is well behind. Any
  ~30k smoke should be read against **≈−88**, not against "off the
  floor" — a smoke sitting at −150 has not nearly matched the oracle's
  early curve even though it has cleared the old landmark. The
  oracle's head start is not mysterious: its buffers are pre-seeded
  with 2,000 steps of expert collect data (mean −76.2), which is the
  same fixture our replication seeds from.
- **Perception is not the bottleneck on MC**: the state-based oracle
  converges to the same −68 as the pixel runs (tier1c), so P1′'s
  architecture-parity claim is tested at the paper's own score.
- **Data**: `logs/oracle_state_mc_seedN/` in the raif-paper worktree
  (`scores.jsonl`, `metrics.jsonl`, replay chunks).
- **Fixtures (A12 decision 6) — EXPORTED 2026-09-01**:
  `reference/fixtures/oracle_mc_seed{0..3}.npz` + provenance JSONs,
  written by `reference/fixtures/export_oracle_mc.py` (the complete
  per-seed collect stream from `replay/`; M⁺ reproduced by the
  recipe's routing rule). Per seed: 48 episodes, 1,903–1,911 steps,
  **mean return −76.2 / −75.8 / −76.1 / −76.2, SR 1.00, mean length
  ≈ 39.8** — matching the oracle report's expert line (−76.1), so the
  seed-quality report beside 2d's acceptance numbers is a parity check
  by construction. `tests/test_oracle_fixtures.py` keeps them loadable
  and matched to their provenance.

**P3 — their pixel MC.** Full R-AIF configuration: 64×64 image obs,
CRSPP + self-revision, ≈1,000 steps of scripted-expert seed data,
`imag_horizon 8`, `tiny` preset (confirmed empirically — see §9),
2M steps per the README recipe. Target: **their Table 2 numbers —
ACR −68.3, SR 1.0, R-S 0.1 — over 4 seeds** (their count).
Achievability is established: Alex has replicated pixel MC in the
sibling `raif-jepa` repo with exactly this recipe.

**P4 — one robosuite task: `Door` (`robosuite_Door`).** Clearest
numbers by far: R-AIF **425.8±10.0 ACR / 1.0±0.0 SR** against
0.0/0.0/0.0 for two baselines; target = those numbers at 5M steps,
3 seeds. `Block Lift` is unusable as a target (ACR sd 159.8, SR
0.4±0.4 — the mean is not a target). Fallback if the
robosuite/MuJoCo toolchain or the 5M-step × 4-camera cost blocks us:
**Meta-World `door-open-v2`**, target their −62.5±0.6 / 1.0±0.0 at
3M steps — a cheaper arm task, not a weaker one (DAIVPG-G scores
0.2 SR there), and the one matching the original MetaWorld framing.

**FLAG 18, adjudicated 2026-09-02 — the parity recipe does NOT
alternate M/M⁺ world-model batches.** §5's "strict 50/50 alternation"
describes the *paper*; the fork-commit *code* never does it: in
`raif/embodied/run/train.py:158-167` the alternation runs as `i % 2`
inside `for i in range(should_train(step))`, but `should_train =
when.Ratio(train_ratio / batch_steps) = Ratio(64/1024)` and the Driver
invokes `train_step` once per single transition (the step counter
advances by 1 between calls), so `Ratio` returns only 0 or 1, `i` is
always 0, and `next(positive_dataset)` is unreachable in every
published config — the paper's own results never sampled M⁺ for
world-model batches. Parity = match the code, not the paper text.
Concretely: `run_mc` / `raif_agent`'s default samples world-model
batches from **M only**; the `alternate()` API stays intact as the
documented paper-described option (separately adjudicated — not
deleted, not renamed); M⁺ keeps its live roles (`pref_label`/ρ CRSPP
gating, success routing) unchanged. Alternation-ON is reserved as a
**labeled A/B experiment after parity** (`run_mc --alternate`). Ledger
entry 16 records the debt and its payment.

## 8. Gap notes for the build (flagged, not resolved)

1. **Categorical latents vs A1's Gaussian head.** Their latent is
   32×32 one-hot with `unimix`; every consumer (KL, entropies, cosine
   sim) assumes it. *Proposed:* add a categorical output head as a
   second `MLPConditional` head variant for the P3 track, keeping the
   Gaussian head as A1's default.
2. **The GRU `deter` carry is part of "the state".** `sim`, all heads,
   and the actor read `[deter, stoch]`; our `Beliefs` seam has no
   deterministic component. *Proposed:* declare `deter` as a
   deterministic parent variable in the graph rather than extending
   `Beliefs`.
3. **Their epistemic term is not our infogain.** It is disagreement of
   a 4-MLP ensemble predicting `stoch`, scaled `1/sqrt(stoch)` — no
   relation to the exact per-variable infogain rows (V4).
   *Proposed:* a new `ensemble_disagreement` estimator-registry row,
   `derived=False`, sibling to the infogain rows.
4. **Preference is a similarity kernel, not a Distribution.** A5
   assumes a designation whose target is a Distribution; R-AIF's C is a
   *rolled-out preferred state trajectory* scored by cosine similarity.
   *Proposed:* let a preference designation name its own instrumental
   estimator row, with `cosine_sim_to_pref_rollout` as R-AIF's.
5. **CRSPP is an action-free twin of the transition, sharing a module.**
   Under A1 that is a second `MLPConditional` in a preference slot whose
   parents exclude the action. *Proposed:* express it as exactly that —
   no new machinery — and record that they share nothing but the class.
6. **Critic: paper says MSE, code uses 255-bin discrete log-prob +
   slow-target regularizer.** A4's `g_tail` returns
   `Interval.exact(v, derived=False)`. *Proposed:* g_tail returns the
   bin distribution's mean; note the paper/code divergence in the
   oracle notes and follow the code.
7. **Sign convention.** Their critic values *negative* EFE
   (`G = rew + sim − IG`, maximized); our G is minimized.
   *Proposed:* fix one convention doc-wide before P2 and convert at the
   registry boundary.
8. **The BC/"actor refresh" term lives inside the actor loss.** Under
   A3/A6 that is `distill()` mixed into `imagine()`, not a separate
   pass. *Proposed:* give `imagine()` an optional supervised term over a
   replay batch; decide at P2.
9. **Two buffers with strict 50/50 alternation is algorithmic, not
   incidental.** A6's trainers take "data/buffer". *Proposed:* keep
   trainers batch-level and pure; the alternation lives in the caller's
   loop (preserves à-la-carte).
10. **A1(a) has no oracle.** Their code contains only amortized
    recognition (A1(b)); structured/unscented filtering through a net
    is unvalidated against R-AIF. *Proposed:* validate A1(a) against our
    own tabular/LG exactness tests, and say so in the row's docs.
11. **Almost every leaf in their G is non-derived.** Reward head, critic,
    IG ensemble, CRSPP similarity — four `derived=False` rows. Worth
    checking that A4's propagation-AND doesn't make the resulting trees
    useless for diagnostics.

## 9. What I could not verify

- **Which model-size config produced Table 2.** The paper has no
  hyperparameter table (its own comments field says "2 tables") and
  never names `tiny/small/medium/large`. The README's MC command uses
  `tiny`. All §2 sizes are therefore config values, not confirmed
  experiment values. **Resolved empirically for MC (2026-08-06):
  Alex replicated the pixel-MC result in the sibling `raif-jepa` repo
  with `--configs gym_mtc tiny --run.steps 2000000` — `tiny` it is,
  at the README's 2M steps.** Other tasks' sizes remain unconfirmed.
- **Whether "1/3/5 million steps" are agent steps or raw env steps.**
  The logged counter increments per *driver* (agent) step
  (`run/train.py:136`), and action repeat is 2/3/1 — so the raw-env
  figure differs by that factor for MC and Meta-World. `configs.yaml:119`
  sets `run.steps: 1e6` for MC (matching the paper) while the README's
  example passes `2000000`. For MC the README recipe (2M) is the
  replication-validated one (see above). Assume agent steps; confirm
  per task on first oracle run.
- **Figure 4 numeric endpoints and convergence step counts.** The
  figure is a raster; §6 gives only the qualitative reading plus the
  Table 2 endpoints. If we need "R-AIF reaches SR 1.0 by N steps" as an
  acceptance criterion, N must come from an oracle run, not the paper.
- **Whether the reported robosuite runs used keyboard/human demos.**
  Paper §4 says SAC experts and `expert_models/sac_robosuite_Door.zip`
  is checked in, so the scripted reading is strongly supported — but
  `experimental/robosuite_keyboard.py` exists and the README documents
  it, so this is inference, not a stated fact.
- **The baselines.** DreamerV3-as-configured, DAIVPG-G and Himst-G are
  not in the public repo; their numbers cannot be reproduced from it.
- **The ICRA/IEEE version (SR-AIF, pp. 6510–6518).** Not read
  (paywalled); the rename R-AIF → SR-AIF suggests other edits. Every
  number here is arXiv v1 / repo `main`.

## Addendum (2026-09-30): rationale from the docstring paring

*Recorded nowhere but the text the 2026-09-30 paring pass cut from the
reference package's docstrings.*

- **§1, `reference/envs/mountain_car.py`: what one agent step is worth.**
  Action repeat is folded into one agent step, as their wrapper does, which is
  why their ACR floor is −200 rather than −100. `MountainCar.step` runs
  `action_repeat` raw physics steps, summing reward and breaking on
  termination. When reading scores: a non-terminating agent step is worth −2,
  an agent step whose *second* raw step reaches the flag −1, and one whose
  *first* raw step does 0.
- **§4 Consumption, `reference/raif/revise.py::positive_gate`: a hard mask, not
  a weight.** The preference model is trained to predict the positive states
  and is told nothing about the rest. That is what makes it a model of
  preferences rather than of the trajectory distribution.
