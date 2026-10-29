# Preference models, both sides: rollouts, training inputs, from-scratch CRSPP swap (2026-09-30)

Stages: ours `causal32k_key0_s20480` / `causal1k_key0_s24576`; theirs `probe28k_state_mc_seed0_s20400` / `_s24400` (fp16).
Starts: the 512 held starts of each data source from `crossdata/` (32 windows, every 4th state), read by both codes
through the bridge (the 20.5k-era held sets are used at both stages). Slopes ×10⁻³ per unit ε, ε = .05 / .2, + = wants bang-bang.

## (A) The preference rollouts and the cosine feature space

Decoded through each side's own agent decoder (their `report` does this for `openl_pref`; §13's "attractor" read did too).
Values are own-states / other-states.

| | ours 20.5k | ours 24.5k | theirs 20.4k | theirs 24.4k |
|---|---|---|---|---|
| t=8 position, mean | −0.327 / −0.327 | −0.297 / −0.297 | −0.226 / −0.251 | +0.076 / +0.018 |
| t=8 sd across starts | **0.002** | **0.001** | 0.21 / 0.20 | 0.60 / 0.55 |
| corr(t=8, start x) | 0.07 / 0.12 | ~0 | 0.37 / 0.46 | 0.43 / 0.52 |
| share x > 0 (goal side) / x < −0.9 (wall) | 0 / 0 | 0 / 0 | 0.22 / 0.17 ; 0 | 0.46 / 0.45 ; 0 |
| mean position t=0 → t=8 | −0.31 → −0.33 | ≈ flat | −0.32 → −0.23 | −0.15 → +0.08 |
| decoded v > 0 share, t=1 → t=8 | 0.84 → 1.00 (v ≈ +0.012) | same | 0.58 → 0.70 | 0.58 → 0.68 |
| cos(pref_t, pref_1) full / deter | 0.995 / – | 0.995 | 0.62 / 0.98 | 0.58 / 0.94 |

- **Ours is a fixed point. It does not depend on the start or the data.** Even the CRSPP posterior at t=0 decodes with an sd of only 0.044, against a start sd of 0.39. A *sampled* rollout of our CRSPP (their recipe) is just as start-blind: −0.316, sd 0.015. So the fixed point is in our CRSPP's weights, not in the mean-field rollout (register row 6.5).
- **Theirs is a start-dependent rollout that drifts toward the goal.** The drift grows from 20.4k to 24.4k (endpoint sd 0.2 → 0.6, goal side 20% → 46%).

**Feature space.** On both sides the cosine is taken over `[deter (128), flat stoch (32×32 = 1024)]` = 1152 numbers.

- Ours compares the agent's sampled one-hot trajectory (stoch norm 5.66) against the CRSPP's **mean-field** rollout. That rollout's stoch part is essentially **uniform** (norm 1.01; uniform gives 1.00), so the stoch contribution is a constant ≈ 0.015.
- Theirs compares sampled against sampled one-hots. Code agreement there is about 1 group in 32 (contribution ≈ 0.013).
- **On both sides the cosine lives in the 128-d deter** (≥ 97% of its value) **and so does all of its ε-response.** Stoch Δ is 0.0 ± 0.2 at every step on both sides.
- Cosine level at ε = 0: ours 0.62–0.66 (p10–p90 ≈ 0.35–0.72); theirs 0.51–0.53 at 20.4k and 0.45–0.50 at 24.4k.

Per-step Δcos(ε .2 − 0)/ε (×10⁻³, t = 1…8; SE 0.2–1.5):

- ours 20.5k: +0.7 +1.9 +3.8 +6.2 +8.7 +11.9 +15.5 **+19.4** (their states: same shape, to +17.6)
- ours 24.5k: rising to +14.1 / +10.8
- theirs 20.4k: −0.7 … −2.4 on their states; −0.4 … +2.7 on ours
- theirs 24.4k: −1.1 … **−15.3** on their states; −0.9 … −9.5 on ours

The response builds with depth on both sides. In each case it is the deter of the saturated trajectory moving toward (ours) or away from (theirs) the preferred deter.

## (B) What the CRSPP trains on (20.5k / 20.4k; both draw every batch from M — debt-ledger entry 16)

| | ours | theirs |
|---|---|---|
| replay rows / episodes | 22,214 / 378 | 22,333 / 383 |
| expert episodes (rows), all at the start | 48 (1,911) | 48 (1,897) |
| own successes / own failures | 255 / 75 | 252 / 79 |
| ρ > 0 / ρ < 0 / ρ = 0 (ρ < −10⁻³) | .176 / .243 / .581 (.020) | .173 / .309 / .518 (.021) |
| positive rows from expert / own successes | **.477 / .523** | **.477 / .523** |
| ρ⁺ mass from expert | **.814** | **.814** |
| per batch (16×64): pos / neg / pos-expert / pos-own | 183 / 253 / 89 / 94 | 170 / 317 / 76 / 94 |
| positive KL-mask share | .178 (.04–.42) | .166 (.05–.33) |

The label rule checks out on the data, and it is identical on both sides:

- **Expert episodes:** ρ = +1 on every row except the first (ρ = 0), and csi = 1 on every row.
- **Own successes:** ρ at step-from-end k = 0, 1, 2, … is 1, .4, .16, .064, …, which is the 0.4 lookback; it reaches 0 by k ≈ 8.
- **Failures:** −0.3^k, so only about 2% of rows fall below −10⁻³.

The batches as drawn match the exported training data exactly on both sides. **No asymmetry was found in how gold rows are labelled or weighted.**

One fact holds on both sides: the gated KL sits **at its free-nats floor** throughout. kl / (0.6 · positive share) = 1.000–1.002 at init, at stage, and after 1024 updates. So the masked KL carries essentially no gradient, and the **contrast term is the CRSPP's only live training signal.**

## (C) From-scratch CRSPP swap

Only the CRSPP was trained, 1024 updates on frozen labelled data. The world model, actor, critic and ensemble were held at the stage. For ours the init came from `R.build` (seed 0). For theirs it was the `_2` layers from their own step-0 checkpoint (`probe28k…_s0`). The readout is the §27 cosine-direct slope with the stage actor.

| code / data | N = 0 | 64 | 256 | 512 | 1024 | control: same, from stage CRSPP (0 → 1024) |
|---|---|---|---|---|---|---|
| ours / theirs | −1 / −1 | +8 / +9 | +27 / +30 | +28 / +31 | **+27 / +29** | +33/+36 → +29/+32 |
| ours / ours | −3 / −3 | +13 / +16 | +28 / +35 | +30 / +37 | **+29 / +36** | +31/+38 → +30/+37 |
| theirs / theirs | −3 / −5 | +1 / −1 | +6 / +4 | −9 / −14 | **−16 / −23** | −8/−13 → −20/−29 |
| theirs / ours | −2 / −4 | +6 / +6 | +18 / +22 | −1 / −3 | **−27 / −38** | +2/+1 → −26/−36 |

The second key set agrees in sign at every N ≥ 256 (e.g. theirs/ours 1024 gives −29/−38). Their gradients were finite throughout.

- **From init, ours re-forms the same fixed point on either data.** By N = 256 the t=8 endpoint is −0.25 to −0.27 with sd 0.003.
- **From init, theirs re-forms a start-dependent, goal-ward rollout.** By N = 1024 the endpoint sd is 0.52 / 0.70 (on our data the mean is +0.07).

## Reading

Both from-scratch CRSPPs start near zero. They are trained on identical labelled data whose label composition is itself identical across the sides. Ours converges positive (+27…+36) on both data sources; theirs, after a transient positive phase at 64–256 updates, converges negative (−16…−38) on both. The from-stage controls end in the same place. **So the sign is the CRSPP code, not its history.**

The mechanism visible in (A) is that our CRSPP learns a start-blind fixed point, while theirs learns a start-conditioned rollout that drifts goal-ward. Because the KL is at its floor, the contrast term is what shapes both. Candidate code differences between the two CRSPPs, none tested here:

- ours conditions its posterior on the agent embedding as data, while their C loss also reaches the shared encoder (frozen here, yet theirs still went negative, so this is not required);
- their `observe_no_action` starts from the main RSSM's carried latent and learned `initial`, where ours uses its own initial;
- ours uses a mean-field rollout (it does not explain the fixed point; see A).

## Caveats

- One seed and one stage per side.
- Their CRSPP-only training freezes their encoder, which in their real loop receives the C gradient, and uses a fixed 1e4 loss scale.
- The held starts come from the same frozen source as the training data.
- The direct slope is the λ-return split (§27). The per-step Δcos in (A) is raw and undiscounted.
- Decoding CRSPP latents through the agent decoder assumes a shared latent geometry. That holds only partly: deter cos(CRSPP post, agent post) is 0.88 ours and 0.87 / 0.77 theirs.

Files: `{ours,theirs}_{A,A2,B,C}.py`, `common_*.py`, `A_*.json`, `A2_*.json`, `B_*.json`, `C_*_{init,stage}.json`, `progress.log`.
