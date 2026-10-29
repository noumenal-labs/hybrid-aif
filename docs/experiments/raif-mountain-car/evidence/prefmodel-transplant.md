# CRSPP weight transplant, and what it led to (2026-09-30)

Stages: ours `causal32k_key0_s20480`, theirs `probe28k_state_mc_seed0_s20400` (fp16). Slopes are the §27 cosine-direct, ×10⁻³ per unit ε, at ε .05 / .2, key set 1 (key set 2). Starts are the 512 held starts per source.

## Layer map

| theirs | ours (`crspp_rssm` part) |
|---|---|
| `img_in_2.kernel` (1024×32), `.norm.{scale,bias}` | `gru.map.input_proj.weight` (= kernelᵀ), `gru.map.norms[0]` |
| `gru_2.kernel` (160×384), `.norm` | `gru.map.gate.weight`ᵀ, `norms[1]`; input `[deter, x]`, split reset / cand / update on both sides |
| `img_out_2`, `img_stats_2` (+bias) | `prior.trunk.layers[0]` + `norms[0]`, `layers[1]` (+bias) |
| `obs_out_2`, `obs_stats_2` (+bias) | `posterior.trunk.layers[0]` + `norms[0]`, `layers[1]` (+bias); parents `(deter, embed)` |
| WM `initial` → WM `img_out`/`img_stats` mode | ours: own `LearnedVector` → CRSPP prior mode |

The map covers 172,992 numbers, matching audit §1.2. **Our 20.5k CRSPP has no extra biases**: the post-065 `winit` fix removed them, so 0 were zeroed or dropped. The only structural differences left are the +128 own-initial and the reset convention (below).

## Forward equivalence (mode codes, step by step; their held windows, then ours)

| | deter max | post logits max | prior logits max | codes agree |
|---|---|---|---|---|
| theirs → our module, fp32 | 4.2e-7 / 6.4e-7 | 9.5e-7 / 1.2e-6 | 4.8e-7 | 100% |
| theirs → our module, vs their fp16 | 1.5e-3 / 1.3e-3 before first code flip | t0 4e-3 | – | 98.8% (fp16 argmax flips, then trajectories diverge) |
| ours → their module, fp32 | 3.0e-7 / 3.6e-7 | 9.5e-7 | 4.8e-7 | 100% |
| ours → their module, fp16 | 1.4e-3 / 1.3e-3 before first flip | 4.0e-3 | – | 99.6% |

For the theirs → ours check, our scan was patched to reset at `is_first` to their WM initial, which is their convention. For ours → theirs, their scan started from our initial with `is_first` all False. The harness scan reproduces `crspp.observe_no_action` to 0.0. The patched sampler was checked: stoch equals onehot(argmax), difference 0.

## Steps 1–3: CRSPP-only transplant (as specified)

| module / weights | starts | direct slope | cos level | deter cos(C post, agent post) | t8 mean, sd, corr, goal |
|---|---|---|---|---|---|
| ours / ours (ctrl) | ours | **+30.6/+38.2** (+30.5/+40.5) | 2.43 | **0.884** | −0.327, .002, +.07, 0 |
| ours / ours (ctrl) | theirs | +32.7/+36.3 (+26.5/+31.6) | 2.53 | 0.893 | −0.327, .002, +.12, 0 |
| ours / **theirs**, WM-init + resets | ours | +2.0/+2.1 (+2.0/+2.1) | 0.29 | **0.087** | +0.64, .08, +.04, 1.00 |
| ours / theirs, WM-init + resets | theirs | −0.0/+0.2 (−0.4/−0.3) | 0.29 | 0.085 | +0.64, .08, −.10, 1.00 |
| ours / theirs, own-init | ours / theirs | +1.6/+1.7 ; −0.6/−0.4 | 0.29 | 0.085 | +0.64, .06–.07, ~0, 1.00 |
| theirs / theirs (ctrl) | theirs | **−7.7/−12.5** (−10.1/−12.7) | 3.41 | **0.873** | −0.226, .21, +.37, .22 |
| theirs / theirs (ctrl) | ours | +1.8/+0.9 (+1.6/+1.1) | 3.41 | 0.883 | −0.251, .20, +.46, .17 |
| theirs / **ours** | theirs | +3.7/+4.1 (+3.1/+3.8) | 0.39 | **0.089** | +0.06, .23, −.03, .59 |
| theirs / ours | ours | +5.2/+6.5 (+4.9/+5.9) | 0.38 | 0.087 | +0.06, .23, +.02, .59 |

The controls reproduce prefmodel.md (+31/+38 and −8/−13). **The interpretation key cannot be applied.** A CRSPP's latent is only meaningful in the latent space it was contrasted against. In a foreign host its posterior has deter cos ≈ 0.09 with the host agent's posterior, against 0.88 native. The cosine level collapses about 8×, and both transplants read near zero. The decoded t8 values for transplanted rows pass a foreign latent through the host decoder and mean nothing.

## Supplementary 1: recipe swap, no transplant

| | own recipe | swapped recipe |
|---|---|---|
| ours: mean-field → sampled pref rollout | +30.6/+38.2 | +24.9/+31.3 (+24.6/+33.1) |
| theirs: sampled → mean-field (stoch norm 5.66 → 1.01) | −7.7/−12.5 | −7.3/−12.5 (−10.5/−13.1) |

Row 6.5 does not carry the sign.

## Supplementary 2: train in the other side's latent space, transplant back (map verified above)

Each trainer was run from scratch on the 1024 batches of exported embeddings and agent posteriors and read in the host's module with the host's recipe. Values are on host starts (other starts in brackets).

| trainer → host space | is_first resets in training scan | N=256 | 512 | 1024 | t8 sd / corr / goal at 1024 |
|---|---|---|---|---|---|
| ours → ours (ours_C) | no | +28/+35 | +30/+37 | **+29/+36** | .06 / – / 0 |
| ours → ours (X3) | no | +28.1/+35.2 | +29.8/+37.2 | **+30.0/+36.8** | .08 / +.77 / 0 |
| ours → ours (X3) | **yes** | +20.9/+26.5 | +11.9/+15.6 | **−22.5/−29.2** (−38/−45) | .48 / +.68 / .46 |
| theirs → ours (X2) | yes | +20.9/+26.4 | +11.6/+15.2 | **−20.6/−26.7** (−38/−43) | .47 / +.69 / .45 |
| ours → theirs (X1) | yes | +10.5/+10.4 | −12.9/−19.6 | **−36.7/−50.2** (−25/−34) | 1.33 / +.28 / .47 |
| ours → theirs (X3) | no | +19.4/+21.9 | +19.3/+21.9 | **+20.0/+21.6** (+20/+23) | .20 / +.53 / .03 |
| theirs → theirs (theirs_C) | yes | +6/+4 | −9/−14 | **−16/−23** | .52† / – / .33 |
| theirs → theirs, `is_first` zeroed in the CRSPP scan only | no | +18.0/+20.6 | +18.4/+21.0 | **+19.7/+21.0** (+16.4/+20.2) | .16† / – / .05 |

† is the sd of the 4-key mean (theirs_C convention); elsewhere sd is key 0.

Cross-check, our space: trained with resets and read without gives N=1024 −13.1/−17.9. Trained without and read with gives +29.4/+36.1. **The training convention sets the sign; the read convention does not.**

Sanity checks:
- X1 at N=0 equals theirs_C init (−3.3/−5.0).
- X1 losses track theirs_C within about 3% (contrast −0.1063 vs −0.1071 at 640).
- X2 losses track ours (contrast −0.0261 vs −0.0262 at 128).
- X2 and X3-with-resets agree to within 2 at every read.

## Reading

The question was weights versus recipe. Weights cannot be moved between latent spaces, and the recipe swap leaves both signs intact. The sign does not follow the trainer code (loss, optimizer, fp16 vs fp32), the init draw, the latent space, the actor, or the recipe. It follows one thing: **whether the CRSPP's training scan resets at mid-window `is_first`** (about 1.7% of rows on both sides). With resets, every cell goes negative and start-conditioned, including our trainer in our space. Without them, every cell goes positive with the start-blind fixed point, including their own code with one line changed.

Their code resets. Ours cannot: `crspp.observe_no_action(parts, key, embeds)` and `fit_crspp` take no `is_first`. So our CRSPP carries its state across episode boundaries inside a window. prefmodel-seed.md listed this as untested. I flag it as a fidelity divergence (candidate defect) and have not changed it. Which convention is *desirable* is not established here.

## Caveats

- One seed and one stage per side. Only the data_theirs cell was run for the no-reset theirs_C variant.
- X1 starts row 0 from the WM initial; their trainer starts from the carried agent latent.
- Why resets flip the sign (mechanism) was not probed.

Scripts are `transplant_*.py`. Outputs are `transplant_*.json`, `C_theirs_theirs_init_noreset.json`, and the `*.out` files. Progress is logged in `progress.log`.

## Addendum: resetting to the CRSPP's own initial (fix shape)

This uses the same setup as X3 (our trainer, our latent space and data, R.build init). The reset target is the CRSPP's own `parts.initial()` / `parts.prior.mode((deter0,))`, at row 0 and at every mid-window `is_first`; gradients pass through the initial. Reads use the same convention. Script: `transplant_x3.py` with `OWN0=1`; output: `transplant_X3_x2_ours_resets1_own0.json`.

| reset convention | N=256 direct | N=1024 direct (our starts) | N=1024 (their starts) | t8 sd / corr / goal at 1024 |
|---|---|---|---|---|
| none (control) | +28.1/+35.2 (+28.2/+37.4) | **+30.0/+36.8** (+29.4/+38.5) | +31.4/+34.8 | .083 / +.77 / 0 |
| reset to WM initial | +20.9/+26.5 (+21.3/+28.7) | **−22.5/−29.2** (−22.5/−29.6) | −38.4/−44.5 | .480 / +.68 / .46 |
| reset to own CRSPP initial | +20.9/+26.4 (+21.2/+28.5) | **−22.1/−28.2** (−21.7/−28.7) | −36.9/−43.2 | .446 / +.65 / .49 |

**The reset target does not matter; the reset does.** A fix shaped as "thread `is_first` into our scan and reset to our own initial" reproduces their behaviour.

Mid-window `is_first` rows in our training data: mean 17.1 per 1024-row batch (sd 2.6, range 9–25), a share of 1.67%. Every window also starts with `is_first` at row 0. Their data has 17.2 per batch (1.68%), and row 0 is `is_first` in only 1.6% of their windows.

The exported data files `transplant_x{1,2}_data.npz` (0.9 GB) have been deleted. Re-export with `transplant_x1_export.py` / `transplant_x2_export.py`.

## Addendum 2: reset training from a trained stage CRSPP (X4, `transplant_x4.py`)

This uses ours_C's step, with each stage's world model held fixed and its stage adam state. Training data is data_ours_long, cycled after 1024 batches. The scan resets at `is_first` to the CRSPP's own initial (my patch, not the repo's in-progress `is_first`). Reads use the same convention, on our starts.

| from stage | N=0 | 256 | 1024 | 4096 | t8 sd / corr / goal at 0 → 1024 → 4096 |
|---|---|---|---|---|---|
| 20.5k | +30.3/+38.0 (+30.2/+40.3) | +26.2/+32.9 (+26.3/+35.2) | −2.3/−3.8 (−3.1/−3.8) | **−28.6/−37.6** (−28.3/−38.4) | .002/−.03/0 → .33/+.71/.34 → 1.83/+.19/.52 |
| 16k | +41.6/+53.6 (+42.3/+53.9) | +37.0/+47.5 (+37.3/+47.4) | +7.8/+9.3 (+6.3/+7.3) | **−26.5/−37.7** (−27.8/−39.1) | .005/−.10/0 → .24/+.63/.27 → 1.96/+.15/.49 |

The decoded t8 sd above 1 at N=4096 means the rollout has left the decoder's range, so the numbers in that column are not positions.

## Addendum 3: real checkpoints, no training (X5, `transplant_x5_ckpts.py`)

Each checkpoint is read with its own actor and CRSPP, through the repo's working-tree `crspp_recognize`, on our starts. The first value passes the held windows' `is_first`; values after the semicolon omit it.

| step | fix1k (resets in training) | t8 sd/corr/goal | causal1k (no resets) | t8 sd/corr/goal |
|---|---|---|---|---|
| 20480 | +28.2/+35.1 ; +29.0/+36.0 | .022/+.23/0 | +30.3/+38.0 ; +30.6/+38.2 | .002/−.03/0 |
| 24576 | +12.7/+16.6 ; +17.7/+21.7 | .046/+.27/0 | +20.5/+24.9 ; +20.8/+25.2 | .002/+.21/0 |
| 28672 | +0.1/+1.0 ; +3.6/+4.5 | .142/+.39/0 | +15.7/+19.9 ; +16.1/+20.5 | .002/+.66/0 |

## Addendum 4: from-scratch run with the restart from step 0 (X5 harness)

Each checkpoint is read with its own actor and CRSPP on our 512 starts, through the repo's `crspp_recognize`. The first value passes `is_first`; values after the semicolon omit it. Key set 2 is in brackets. Output: `transplant_x5_fix32k.json`, `transplant_x5_causal32k.json`.

| step | fix32k (restart from 0) | t8 sd/corr/goal | causal32k (unfixed) | t8 sd/corr/goal |
|---|---|---|---|---|
| 20480 | +3.1/+7.8 (+7.6/+9.8) ; +9.7/+14.5 | .150/+.38/0 | +30.3/+38.0 (+30.2/+40.3) ; +30.6/+38.2 | .002/−.03/0 |
| 24576 | −6.8/−9.6 (−5.9/−9.1) ; −2.2/−5.5 | .294/+.58/.34 | +20.5/+24.9 (+21.6/+26.6) ; +20.8/+25.2 | .002/+.21/0 |
| 28672 | **−24.6/−34.2** (−22.7/−32.3) ; −19.0/−27.7 | .432/+.62/.51 | +15.7/+19.9 (+16.3/+22.0) ; +16.1/+20.5 | .002/+.66/0 |

Run from scratch with the restart, the stage CRSPP turns negative with a start-conditioned rollout, as theirs does. The unfixed run keeps its fixed point. `causal1k_key0_s{20480,24576,28672}` is bitwise identical to `causal32k` (same `state.eqx`/`buffers.eqx` md5), so addendum 3's unfixed column is this column.
