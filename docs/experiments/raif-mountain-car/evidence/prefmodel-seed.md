# CRSPP filter seed probe (2026-09-30)

Our code on our data at the 20.5k stage (`causal32k_key0_s20480`). The CRSPP starts from `R.build` init (seed 0) and gets 1024 updates. lr, keys and schedule are those of `ours_C.py`, and the world model, actor, critic and ensemble are held at the stage.

The one change is that `crspp.observe_no_action` is monkeypatched. Before any trace, the module attribute is replaced by a version that starts the scan from a seed `(deter0, code0)`. The seed is carried in extra embed channels: width 64 + 128 + 1024 = 1216, with the code reshaped to the CRSPP's (32, 32). An assert inside the patch checks that width. The patch was traced twice per run, once by `fit_crspp` and once by `crspp_recognize`, and the assert never fired.

Slopes are the (C) cosine-direct ×10⁻³ per unit ε, at ε .05 / .2, with key set 1 (key set 2 in brackets), on our 512 held starts with the stage actor. Script: `ours_seed_C.py`. Outputs: `Cseed_{v0,v1,v2}_ours_ours.json` and `.out`, plus `Cseed_v0_ours_ours_N256.json` (the first V0 run, to N = 256).

## Seeds

- **V0**: the CRSPP's own `parts.initial()` and `parts.prior.mode((deter0,))`, recomputed from the current CRSPP params at every step. It is passed as data, so `initial` gets no gradient; in the unpatched code it does.
- **V1**: the agent world model's posterior `(deter, stoch)` at row 0 of each window. It comes from the same `posterior_states` call that builds `feats` in training; at readout it comes from `starts`' `posterior_states` (key 6).
- **V2**: the world model's `RSSMCore.initial_state()`, constant across windows. That is `deter = LearnedVector()` with the tanh built into our `LearnedVector` (no extra tanh added), and `stoch = core.prior.mode((deter,))`, the WM prior's one-hot mode. At the stage this initial deter has cos 0.68 with the CRSPP's own initial deter.

## V0 sanity: passes

| | N = 0 | N = 256 |
|---|---|---|
| control `C_ours_ours_init` | −2.9 / −3.3 (−3.5 / −3.7) | +28.1 / +35.2 (+28.2 / +37.5) |
| V0 | −2.9 / −3.3 (−3.5 / −3.7) | +28.0 / +35.0 (+28.0 / +37.3) |

N = 0 is identical to the digit, and t8 −0.259 / sd 0.176 also matches. At N = 256 the gap is ≤ 0.2, which is attributable to detaching `initial`. V0 was then extended to N = 1024 to serve as the like-for-like control.

## Results

| seed | N | direct slope | t=8 mean | t=8 sd | corr(t8, start x) | goal share | t=0 sd / corr |
|---|---|---|---|---|---|---|---|
| V0 | 0 | −2.9 / −3.3 (−3.5 / −3.7) | −0.259 | 0.176 | +0.02 | 0.06 | 0.395 / +0.04 |
| V0 | 256 | +28.0 / +35.0 (+28.0 / +37.3) | −0.269 | 0.003 | +0.13 | 0 | 0.069 / +0.05 |
| V0 | 1024 | +29.5 / +36.6 (+29.2 / +38.6) | −0.324 | 0.045 | +0.76 | 0 | 0.165 / +0.65 |
| V1 | 0 | −1.9 / −2.2 (−2.1 / −2.2) | −0.315 | 0.169 | −0.03 | 0.02 | 0.382 / +0.04 |
| V1 | 256 | +28.3 / +35.5 (+28.4 / +37.7) | −0.273 | 0.003 | +0.16 | 0 | 0.036 / +0.10 |
| V1 | 1024 | +28.4 / +34.9 (+28.0 / +36.6) | −0.304 | 0.077 | +0.79 | 0 | 0.209 / +0.74 |
| V2 | 0 | −1.4 / −1.4 (−1.9 / −1.7) | −0.323 | 0.166 | +0.01 | 0.02 | 0.420 / +0.06 |
| V2 | 256 | +28.1 / +35.2 (+28.2 / +37.4) | −0.270 | 0.003 | +0.09 | 0 | 0.069 / +0.04 |
| V2 | 1024 | +29.3 / +36.0 (+28.9 / +37.6) | −0.316 | 0.080 | +0.76 | 0 | 0.220 / +0.68 |

The wall share is 0 everywhere. N = 64 and 512 are in the `.out` files and follow the same paths; at 512 the three seeds agree within 0.1 on the slope.

The training losses track the control's closely. For example, contrast at update 128 is −0.0262 for V0 against −0.0265 for the control.

## Reading

**The seed hypothesis is not supported.** Changing where the CRSPP's filter starts changes nothing material:

- **Slope.** All three seeds converge to the same positive direct slope, +28…+37, on both key sets. That matches the unpatched control (+29 / +36) and is nowhere near their −16…−38.
- **Fixed point.** At N = 256 all three show the same start-blind fixed point: t=8 sd 0.003, mean ≈ −0.27, corr ≈ 0.1.
- **Start-dependence at N = 1024.** By N = 1024 all three show a weak start correlation: corr +0.76…+0.79, but sd only 0.045–0.080, mean −0.30 to −0.32, and goal share 0. This appears equally under V0, so it comes from more training, not from the seed. (The unpatched control at N = 1024 has t8 sd 0.063; its correlation was not recorded.) It is still far from their 0.5–0.7 sd with goal-ward drift.
- **V1 versus V0.** At N = 1024, V1's t=0 posterior is marginally more start-correlated (+0.74 vs +0.65). That is a small difference from a single seed and is not separable from noise here.

**Their-data cell skipped.** Neither V1 nor V2 flipped the sign or broke the fixed point, so by the stated condition the "winning variant on their data" run was not done.

## Caveats and unverified points

- **V1 is offset by one step from theirs.** Their `prev_latent` is the agent posterior before row 0; V1 uses the posterior at row 0. That posterior has already seen row 0's observation, and the CRSPP then applies one GRU step before reading row 0's embed.
- **V2 is the closer analogue on our data.** On our training data every window has `is_first` at row 0, and so do the held windows. Their `obs_step_no_action` would therefore reset to the WM initial there.
- **Mid-window resets are not handled.** Their scan also resets at mid-window `is_first` (about 1.7% of rows); ours, patched or not, does not. This was not tested.
- **V0 detaches `initial`** (see Seeds). Its 1024 endpoint sd of 0.045 versus the control's 0.063 may reflect that.
- One seed, one stage. Our data only.
