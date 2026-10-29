# Their actor gradient by term (`theirs_critic/measure_grad.py`)

One real `agent.train` per batch from each snapshot, up to the actor optimizer (world-model, ensemble and retnorm updates as shipped); the actor Optimizer replaced by a capture that takes `nj.grad` w.r.t. the actor module only (fp16: loss × grad_scale, grads ÷ grad_scale, as their Optimizer does → true unscaled gradients; no clip, no Adam). Terms transcribed from `ImagActorCritic.loss`; the sum of the four term gradients reproduces the gradient of their own loss closure (rel. err column). 8 batches from the general replay (`gen`) and 8 from `positive_replay` (`pos`; their train loop alternates the two), both restricted to chunks saved before the checkpoint. ± = sd over batches.


## (A) gradient norms, source `gen`

| snapshot | ‖g‖ full | ‖g‖ return | ‖g‖ imit | ‖g‖ ent | ‖g‖ wment | ‖g_imit‖/‖g_ret‖ | cos(imit, ret) | cos(full, ret) | flag share (batch) | retnorm invscale | sum-vs-orig rel.err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 17400 | 0.349 ± 0.27 | 0.0679 ± 0.0143 | 0.371 ± 0.306 | 0.00197 ± 4.49e-05 | 4.74e-05 ± 5.51e-06 | 5.22 ± 3.71 (min 0.35, max 12.65) | -0.460 ± +0.487 | -0.172 ± +0.766 | 0.083 ± 0.056 | 1.289 | 1.7e-04 |
| fp16 19400 | 0.247 ± 0.147 | 0.0812 ± 0.0242 | 0.219 ± 0.142 | 0.00191 ± 4.49e-05 | 5.02e-05 ± 3.15e-06 | 2.86 ± 1.93 (min 0.26, max 6.65) | +0.084 ± +0.415 | +0.483 ± +0.427 | 0.076 ± 0.042 | 1.140 | 1.5e-04 |
| fp16 21400 | 0.333 ± 0.282 | 0.0871 ± 0.0135 | 0.275 ± 0.274 | 0.00175 ± 2.78e-05 | 5.21e-05 ± 6.25e-06 | 3.27 ± 3.57 (min 0.27, max 11.69) | +0.343 ± +0.320 | +0.785 ± +0.159 | 0.073 ± 0.046 | 1.878 | 1.4e-04 |
| fp16 23400 | 0.175 ± 0.0689 | 0.0659 ± 0.0133 | 0.148 ± 0.08 | 0.00181 ± 4.47e-05 | 4.91e-05 ± 4.02e-06 | 2.39 ± 1.32 (min 0.26, max 3.70) | +0.134 ± +0.250 | +0.559 ± +0.301 | 0.057 ± 0.029 | 3.060 | 1.3e-04 |
| fp32 17600 | 0.308 ± 0.278 | 0.0555 ± 0.0175 | 0.274 ± 0.269 | 0.00164 ± 3.4e-05 | 5.58e-05 ± 3.25e-06 | 4.74 ± 4.22 (min 0.37, max 13.81) | +0.398 ± +0.317 | +0.745 ± +0.162 | 0.083 ± 0.056 | 1.237 | 2.1e-07 |
| fp32 19600 | 0.212 ± 0.0648 | 0.108 ± 0.0408 | 0.219 ± 0.133 | 0.00162 ± 4.88e-05 | 6.66e-05 ± 6.73e-06 | 2.26 ± 1.29 (min 0.14, max 3.63) | -0.404 ± +0.371 | +0.030 ± +0.653 | 0.077 ± 0.042 | 1.174 | 1.3e-07 |
| fp32 21600 | 0.278 ± 0.227 | 0.0537 ± 0.0159 | 0.261 ± 0.229 | 0.00151 ± 2.92e-05 | 6.21e-05 ± 6.05e-06 | 5.83 ± 6.27 (min 0.25, max 19.52) | +0.129 ± +0.328 | +0.479 ± +0.341 | 0.072 ± 0.044 | 1.965 | 1.4e-07 |

## (B) plain step −lr·g (their lr 3e-5), actor mean on 512 held starts, ×10⁻⁹, source `gen`

⟨Δm·sgn⟩ ± sd (batches > 0) and ⟨Δ|m|⟩ = ⟨sign(m₀)Δm⟩. First-order (JVP, float32 compute). Held set `slope` = measure_slope/measure_update's (last chunks of the whole replay dir, i.e. end-of-run states); `snap` = same rule on the replay as of the checkpoint. To compare with our §23 plain-step table (lr 1e-5, ×10⁻⁹) divide by 3.

| snapshot | return ⟨Δm·sgn⟩ slope | return ⟨Δ\|m\|⟩ | return ⟨Δm·sgn⟩ snap | return agree / disagree (snap) | full ⟨Δm·sgn⟩ slope | full ⟨Δ\|m\|⟩ | full ⟨Δm·sgn⟩ snap | imit ⟨Δm·sgn⟩ snap | full takes imit's sign (snap) | FD chk full (slope) |
|---|---|---|---|---|---|---|---|---|---|---|
| fp16 17400 | +973 ± +214 (8/8) | -1466 ± +373 | +966 ± +204 (8/8) | -595 / +4734 | -3287 ± +5050 (3/8) | +4753 ± +7574 | -3135 ± +4805 (3/8) | -4091 ± +4921 | 7/8 | -3301 vs -3287 |
| fp16 19400 | +1477 ± +430 (8/8) | -3682 ± +1186 | -1206 ± +411 (0/8) | -4833 / +5902 | +2223 ± +3526 (7/8) | -5005 ± +8284 | -1629 ± +2633 (1/8) | -429 ± +2466 | 6/8 | +2225 vs +2223 |
| fp16 21400 | +3500 ± +550 (8/8) | -2565 ± +413 | +1513 ± +239 (8/8) | -1518 / +8827 | +10884 ± +9898 (8/8) | -7963 ± +7657 | +4778 ± +4319 (8/8) | +3274 ± +4260 | 7/8 | +10855 vs +10884 |
| fp16 23400 | +609 ± +124 (8/8) | -2953 ± +607 | -2245 ± +460 (0/8) | -2453 / -1784 | +842 ± +648 (8/8) | -3782 ± +2799 | -2942 ± +2038 (0/8) | -711 ± +1962 | 6/8 | +830 vs +842 |
| fp32 17600 | +1026 ± +376 (8/8) | -1414 ± +527 | +2108 ± +780 (8/8) | -92 / +7175 | +4957 ± +4961 (8/8) | -6846 ± +7522 | +10201 ± +10467 (8/8) | +8113 ± +10229 | 8/8 | +4940 vs +4957 |
| fp32 19600 | -878 ± +362 (0/8) | -2232 ± +883 | -462 ± +273 (0/8) | -1935 / +2931 | +254 ± +1111 (5/8) | +1023 ± +2995 | +157 ± +577 (5/8) | +615 ± +536 | 7/8 | +251 vs +254 |
| fp32 21600 | -375 ± +122 (0/8) | -1466 ± +504 | -1512 ± +508 (0/8) | -1263 / -2027 | -938 ± +1200 (1/8) | -4416 ± +5791 | -4094 ± +4872 (1/8) | -2594 ± +5091 | 5/8 | -930 vs -938 |

## (C) dL/dε along Toward(ε), FD at ε = 0.05, ×10⁻³ (negative = the term wants bang-bang), source `gen`

| snapshot | return | imit | ent | wment | full | batches return < 0 | batches imit < 0 | ε=0 reproduces terms (max abs) |
|---|---|---|---|---|---|---|---|---|
| fp16 17400 | -11.34 ± +2.02 | +19.48 ± +22.97 | -0.015 ± +0.003 | -0.005 ± +0.001 | +8.11 ± +23.66 | 8/8 | 1/8 | 0.0e+00 |
| fp16 19400 | +4.03 ± +10.90 | +8.62 ± +11.08 | -0.013 ± +0.003 | -0.004 ± +0.002 | +12.63 ± +11.72 | 4/8 | 2/8 | 0.0e+00 |
| fp16 21400 | +7.11 ± +5.49 | +0.37 ± +11.40 | -0.005 ± +0.002 | -0.004 ± +0.001 | +7.47 ± +12.23 | 0/8 | 3/8 | 0.0e+00 |
| fp16 23400 | +13.12 ± +5.52 | +5.69 ± +5.10 | -0.011 ± +0.002 | -0.002 ± +0.002 | +18.80 ± +7.41 | 0/8 | 0/8 | 0.0e+00 |
| fp32 17600 | -13.96 ± +3.13 | +7.90 ± +9.07 | -0.005 ± +0.003 | -0.005 ± +0.001 | -6.07 ± +8.58 | 8/8 | 2/8 | 0.0e+00 |
| fp32 19600 | +4.95 ± +7.90 | +14.36 ± +28.46 | -0.012 ± +0.002 | -0.001 ± +0.001 | +19.30 ± +28.36 | 2/8 | 2/8 | 0.0e+00 |
| fp32 21600 | +14.83 ± +7.32 | +9.51 ± +13.29 | -0.008 ± +0.002 | -0.003 ± +0.002 | +24.33 ± +12.94 | 0/8 | 1/8 | 0.0e+00 |

## (A) gradient norms, source `pos`

| snapshot | ‖g‖ full | ‖g‖ return | ‖g‖ imit | ‖g‖ ent | ‖g‖ wment | ‖g_imit‖/‖g_ret‖ | cos(imit, ret) | cos(full, ret) | flag share (batch) | retnorm invscale | sum-vs-orig rel.err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 17400 | 0.768 ± 0.303 | 0.0815 ± 0.0193 | 0.825 ± 0.33 | 0.00184 ± 3.27e-05 | 5.13e-05 ± 5.06e-06 | 10.03 ± 3.04 (min 6.96, max 14.76) | -0.694 ± +0.288 | -0.636 ± +0.336 | 0.180 ± 0.064 | 1.291 | 2.5e-05 |
| fp16 19400 | 0.34 ± 0.127 | 0.102 ± 0.0253 | 0.323 ± 0.12 | 0.00176 ± 3.43e-05 | 5.37e-05 ± 5.04e-06 | 3.28 ± 1.23 (min 1.77, max 5.50) | +0.190 ± +0.587 | +0.402 ± +0.530 | 0.128 ± 0.042 | 1.141 | 1.1e-04 |
| fp16 21400 | 0.496 ± 0.147 | 0.0821 ± 0.0201 | 0.446 ± 0.148 | 0.00163 ± 3.6e-05 | 5.07e-05 ± 2.64e-06 | 5.79 ± 2.68 (min 2.81, max 11.27) | +0.505 ± +0.291 | +0.614 ± +0.267 | 0.124 ± 0.029 | 1.876 | 4.4e-05 |
| fp16 23400 | 0.34 ± 0.095 | 0.0376 ± 0.0118 | 0.321 ± 0.0917 | 0.00161 ± 3.92e-05 | 5.17e-05 ± 4.38e-06 | 8.86 ± 1.96 (min 5.33, max 11.26) | +0.458 ± +0.417 | +0.524 ± +0.409 | 0.125 ± 0.031 | 3.050 | 2.6e-05 |
| fp32 17600 | 0.605 ± 0.315 | 0.0785 ± 0.00848 | 0.555 ± 0.309 | 0.00162 ± 2.63e-05 | 5.89e-05 ± 4.71e-06 | 7.17 ± 3.93 (min 2.63, max 15.37) | +0.593 ± +0.226 | +0.695 ± +0.165 | 0.199 ± 0.056 | 1.239 | 8.3e-08 |
| fp32 19600 | 0.492 ± 0.245 | 0.114 ± 0.0231 | 0.48 ± 0.237 | 0.00156 ± 4.6e-05 | 6.8e-05 ± 6.81e-06 | 4.68 ± 3.04 (min 1.28, max 10.53) | -0.056 ± +0.637 | +0.129 ± +0.652 | 0.160 ± 0.063 | 1.175 | 1.0e-07 |
| fp32 21600 | 0.333 ± 0.0908 | 0.052 ± 0.0189 | 0.318 ± 0.0838 | 0.00151 ± 3.59e-05 | 6.99e-05 ± 5.13e-06 | 6.72 ± 2.72 (min 4.41, max 13.20) | +0.284 ± +0.419 | +0.405 ± +0.376 | 0.122 ± 0.029 | 1.962 | 9.3e-08 |

## (B) plain step −lr·g (their lr 3e-5), actor mean on 512 held starts, ×10⁻⁹, source `pos`

⟨Δm·sgn⟩ ± sd (batches > 0) and ⟨Δ|m|⟩ = ⟨sign(m₀)Δm⟩. First-order (JVP, float32 compute). Held set `slope` = measure_slope/measure_update's (last chunks of the whole replay dir, i.e. end-of-run states); `snap` = same rule on the replay as of the checkpoint. To compare with our §23 plain-step table (lr 1e-5, ×10⁻⁹) divide by 3.

| snapshot | return ⟨Δm·sgn⟩ slope | return ⟨Δ\|m\|⟩ | return ⟨Δm·sgn⟩ snap | return agree / disagree (snap) | full ⟨Δm·sgn⟩ slope | full ⟨Δ\|m\|⟩ | full ⟨Δm·sgn⟩ snap | imit ⟨Δm·sgn⟩ snap | full takes imit's sign (snap) | FD chk full (slope) |
|---|---|---|---|---|---|---|---|---|---|---|
| fp16 17400 | +1210 ± +287 (8/8) | -1784 ± +452 | +1191 ± +278 (8/8) | -698 / +5752 | -8849 ± +6405 (1/8) | +13468 ± +10383 | -8461 ± +6102 (1/8) | -9645 ± +6272 | 7/8 | -8861 vs -8849 |
| fp16 19400 | +1972 ± +480 (8/8) | -4733 ± +1239 | -1478 ± +410 (0/8) | -6144 / +7665 | +3268 ± +4307 (6/8) | -7307 ± +10363 | -2217 ± +3287 (2/8) | -743 ± +3557 | 8/8 | +3276 vs +3268 |
| fp16 21400 | +3341 ± +822 (8/8) | -2446 ± +632 | +1455 ± +358 (8/8) | -1398 / +8339 | +12597 ± +7110 (8/8) | -9390 ± +4538 | +5571 ± +3093 (8/8) | +4122 ± +2811 | 8/8 | +12588 vs +12597 |
| fp16 23400 | +374 ± +117 (8/8) | -1665 ± +557 | -1219 ± +418 (0/8) | -1383 / -855 | +2094 ± +1975 (7/8) | -7918 ± +7430 | -5896 ± +5340 (1/8) | -4686 ± +5172 | 8/8 | +2082 vs +2094 |
| fp32 17600 | +1526 ± +165 (8/8) | -2150 ± +231 | +3111 ± +340 (8/8) | -151 / +10626 | +8995 ± +5365 (8/8) | -13056 ± +8299 | +18436 ± +11048 (8/8) | +15340 ± +11093 | 8/8 | +8986 vs +8995 |
| fp32 19600 | -907 ± +209 (0/8) | -2454 ± +526 | -426 ± +175 (0/8) | -2065 / +3349 | -876 ± +3342 (4/8) | -2054 ± +9010 | -762 ± +2060 (4/8) | -338 ± +2037 | 8/8 | -875 vs -876 |
| fp32 21600 | -310 ± +141 (0/8) | -1343 ± +626 | -1272 ± +595 (0/8) | -1113 / -1599 | -843 ± +1079 (2/8) | -4511 ± +5427 | -3940 ± +4536 (2/8) | -2678 ± +4673 | 8/8 | -837 vs -843 |

## (C) dL/dε along Toward(ε), FD at ε = 0.05, ×10⁻³ (negative = the term wants bang-bang), source `pos`

| snapshot | return | imit | ent | wment | full | batches return < 0 | batches imit < 0 | ε=0 reproduces terms (max abs) |
|---|---|---|---|---|---|---|---|---|
| fp16 17400 | -14.03 ± +3.26 | +53.89 ± +18.38 | -0.003 ± +0.002 | -0.008 ± +0.001 | +39.85 ± +20.05 | 8/8 | 0/8 | 0.0e+00 |
| fp16 19400 | -7.55 ± +8.87 | +9.99 ± +12.53 | -0.001 ± +0.002 | -0.009 ± +0.001 | +2.43 ± +13.06 | 6/8 | 1/8 | 0.0e+00 |
| fp16 21400 | +4.29 ± +7.13 | +3.84 ± +17.88 | +0.003 ± +0.002 | -0.008 ± +0.001 | +8.13 ± +18.81 | 3/8 | 3/8 | 0.0e+00 |
| fp16 23400 | +9.26 ± +5.75 | +4.50 ± +11.29 | -0.001 ± +0.002 | -0.008 ± +0.002 | +13.75 ± +12.57 | 0/8 | 2/8 | 0.0e+00 |
| fp32 17600 | -17.26 ± +3.92 | +19.48 ± +24.30 | +0.007 ± +0.003 | -0.010 ± +0.001 | +2.22 ± +24.09 | 8/8 | 1/8 | 0.0e+00 |
| fp32 19600 | +1.63 ± +12.12 | +42.20 ± +32.82 | -0.002 ± +0.003 | -0.008 ± +0.001 | +43.82 ± +31.23 | 5/8 | 1/8 | 0.0e+00 |
| fp32 21600 | +4.67 ± +6.65 | +16.76 ± +26.32 | +0.002 ± +0.001 | -0.008 ± +0.002 | +21.42 ± +26.46 | 2/8 | 2/8 | 0.0e+00 |

## Replay at snapshot time vs replay dir

- fp16 17400: 24 of 32 general-replay chunks saved before the checkpoint
- fp16 19400: 24 of 32 general-replay chunks saved before the checkpoint
- fp16 21400: 28 of 32 general-replay chunks saved before the checkpoint
- fp16 23400: 28 of 32 general-replay chunks saved before the checkpoint
- fp32 17600: 24 of 36 general-replay chunks saved before the checkpoint
- fp32 19600: 24 of 36 general-replay chunks saved before the checkpoint
- fp32 21600: 28 of 36 general-replay chunks saved before the checkpoint

## Reading (against ours, §23 / update.md)

- Norm ratio ‖g_imit‖/‖g_return‖: theirs, general batches 2.3–5.8 (batch range 0.14–19.5), positive batches 3.3–10.0; ours 1.9–26 per row (median ≈ 6; "3–30×"). Ours pairs bc with ret:cos_sim only; theirs pairs it with the whole return term. Flag share: theirs 0.06–0.08 on general batches and 0.12–0.20 on positive ones; ours (general) 0.06–0.14. Same order of weight: a little smaller on their general batches, the same or larger on their positive batches (half of their steps).
- The plain step takes imitation's sign 5–8 of 8 batches on theirs; ours 16 of 18.
- Return-only plain step (÷3 for lr parity): toward bang-bang on every batch at fp16 17.4/19.4/21.4/23.4k and fp32 17.6k on the `slope` held set; away at fp32 19.6/21.6k. On the `snap` held set it is away at 19.4k, 23.4k, 19.6k and 21.6k. In every row it lowers |m| (⟨Δ|m|⟩ < 0). Split agree/disagree: it softens the agreeing states and pulls the disagreeing ones toward the rule, as on ours.
- (C) The return term's dL/dε is negative (wants bang-bang) at 17.4k/17.6k (−11…−17 ×10⁻³, 8/8 batches) and turns positive from about 21k (+4…+15). Ours: −13…−30 on all 18 rows. The imitation term opposes bang-bang on average on every snapshot (+0.4…+54 ×10⁻³), more strongly than ours (−22…+10, positive on 7 of 18).

**Correction (follow-up):** their train loop never samples positive_replay (`positive_replay/samples` = 0 after step 1 in both runs; `should_train` yields at most 1 step, so the `i % 2` branch is unreachable). The `pos` rows above are a hypothetical, not part of their training; only `gen` is real.
