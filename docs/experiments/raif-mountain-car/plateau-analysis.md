# Plateau analysis — why run 4 sits at −105 from 16k to 400k while the oracle improves

**Status: build-time analysis, not adjudication.** Written 2026-09-15 against `hybrid-aif`
`859faf0` and `raif-paper` `9b1864e8` (the oracle). Every number below is read from logs that
already exist or from read-only forward passes on existing checkpoints; no training was run
and no source was changed. Rulings are proposed, not made. The scripts that produced the
tables live in the session scratch directory (`matched.py`, `perseed.py`, `actprobe.py`,
`rolloutprobe.py`, `modelprobe.py`, `mktables.py`); each is a pure reader of `runs/*` and of
`raif-paper/logs/oracle_state_mc_seed{0..3}/metrics.jsonl`.

**Data.** Ours: `runs/mc_2d` (run 4, all known divergences closed, 760k of 2M steps logged at
analysis time; `metrics.jsonl` samples one gradient step per 1,024 pooled agent steps per seed,
`scores.jsonl` has every episode). Theirs: four oracle seeds, `train/*` rows every ~3,344 steps
(each the mean over the logging interval), `episode/*` rows per episode. Checkpoints probed:
`runs/mc_2d/checkpoint` (step 775,168) and `runs/mc_2d_dropcarry/checkpoint` (run 3, step
2,000,896). The `mc_2d_igfix`/`mc_2d_igbug` directories predate the actor channel and the
current `Config` schema and could not be rebuilt for probing.

## 0. Verdict in brief

1. The gap is not a plateau. Three of our four seeds fall into deep, long **collapses** — success
   rate 0.06–0.11 and episode length 90–98 for 56k–170k steps at a time — twice each before 450k,
   while the fourth seed (seed 2) tracks or beats the oracle from 40k to 200k. The oracle has no
   window with SR < 0.5 in 16k–400k (0 of 192 seed-bins at 8k resolution); we have 67 of 192.
   The −105 average is the arithmetic of one good seed and three seeds taking turns at −190.
2. Before a seed collapses, **nothing on the imagination side distinguishes it** from seed 2:
   entropy, G and its three terms, λ-return, critic, advantage, normalizer band, BC weight,
   CRSPP losses, world-model losses and prior entropy are all within seed noise. The one leading
   symptom is real episode length creeping from 46 to 57–63 steps between 24k and 64k while SR
   stays 0.99–1.00; the oracle's length falls from 46 to 41 in the same window.
3. **The collapsed policy is a constant push-right actor and the world model knows it fails.**
   Probed at step 775k, seed 3 (mid-collapse) takes `a > 0` on 100.0% of 1,600 real steps with
   |a| = 0.998, never passes position +0.06, and its own model predicts termination within 30
   imagined steps on 2.5% of states (healthy seeds: 66–83%, matching reality within 7 points).
   This kills model exploitation as the mechanism; it is an attractor of the imagined objective
   that the actor cannot leave, not a model error the actor is cashing in.
4. Hypotheses H1 (exploration), H2 (critic), H3 (M+), H4 (the value crossing) and the one
   acting-side divergence on the record (mean-field beliefs at act time) are **killed by
   measurement**, each with the discriminating number in §5. What survives is a set of three
   recorded divergences whose consequence for this attractor is inferred, not measured: the
   world model learns ~2× faster than theirs and its prior entropy is 7–27 nats lower from 60k
   to 192k (open divergence 17/26 is the primary attribution; the signature is *measured* in
   the window); their encoder is trained
   by the ρ-signed CRSPP contrast and ours is not; and their info-gain leaf is a constant with a
   noisy action gradient where ours is zero with none.
5. The proposed experiment (§6) is a 4-seed, 100k-step run with divergences 17 and 26 closed as
   a pair, judged on five pre-stated series with pass bands taken from the oracle's own seed
   spread, and with the logging the present analysis lacked. Two further arms are ordered
   behind it.

## 1. What the gap actually is

### 1.1 Collapse census (8k bins, SR < 0.5, through 760k)

| side | seed | windows with SR < 0.5 (start–end, min SR) | shallow dips (SR < 0.9 for ≥ 16k) |
|---|---|---|---|
| ours | 0 | 64–160k (0.11), 168–184k (0.23), 192–200k (0.47), 272–392k (0.29) | 16–40k (0.62), 48–240k, 256–400k |
| ours | 1 | 80–152k (0.06), 344–352k, 360–368k, 384–416k (0.38) | 56–160k, 344–432k, 680–696k (0.85) |
| ours | 2 | none after 8k | none after 16k |
| ours | 3 | 88–144k (0.07), 264–432k (0.16); a third is in progress at 775k+ (SR 0 in the probe, −200 in the live log) | 72–448k |
| theirs | 0 | none after 8k | none |
| theirs | 1 | 688–696k (0.34) | 232–304k (0.77), 688–704k |
| theirs | 2 | none after 8k | 232–320k (0.71) |
| theirs | 3 | none after 8k | 272–320k (0.72) |

Seed-bins with SR < 0.5 in 16k–400k: **ours 67 / 192, theirs 0 / 192.** The oracle's own
"universal dip near 200k" (addendum §7) is the 232–320k event in three of its seeds, at SR
0.71–0.77; our seeds 0 and 3 enter a dip in the same window (256–272k) and go to SR 0.16–0.29.

### 1.2 Per-seed trajectories through 200k (8k bins)

Episode return, ours seeds 0–3 then theirs seeds 0–3:

| bin | o0 | o1 | o2 | o3 | t0 | t1 | t2 | t3 |
|---|---|---|---|---|---|---|---|---|
| 0–8k | -111 | -141 | -194 | -120 | -153 | -152 | -149 | -162 |
| 8–16k | -81 | -106 | -121 | -100 | -105 | -106 | -109 | -111 |
| 16–24k | -109 | -89 | -97 | -97 | -88 | -91 | -93 | -99 |
| 24–32k | -152 | -110 | -96 | -110 | -87 | -91 | -86 | -94 |
| 32–40k | -127 | -106 | -85 | -106 | -90 | -89 | -90 | -85 |
| 40–48k | -128 | -102 | -81 | -110 | -84 | -91 | -87 | -85 |
| 48–56k | -124 | -101 | -75 | -122 | -79 | -85 | -85 | -84 |
| 56–64k | -128 | -112 | -74 | -121 | -79 | -85 | -80 | -81 |
| 64–72k | -151 | -137 | -73 | -132 | -79 | -82 | -81 | -80 |
| 72–80k | -170 | -157 | -72 | -143 | -79 | -82 | -82 | -78 |
| 80–88k | -163 | -183 | -71 | -167 | -79 | -81 | -81 | -78 |
| 88–96k | -174 | -187 | -71 | -186 | -78 | -83 | -81 | -78 |
| 96–128k | -179 | -186 | -71 | -185 | -79 | -80 | -80 | -78 |
| 128–160k | -168 | -173 | -72 | -153 | -79 | -79 | -78 | -77 |
| 160–192k | -148 | -98 | -74 | -114 | -81 | -80 | -81 | -81 |
| 192–200k | -143 | -92 | -76 | -106 | -86 | -88 | -82 | -83 |

Success fraction and episode length, same layout, 16–96k (the window the question is about):

| bin | SR o0 | o1 | o2 | o3 | t0 | t1 | t2 | t3 | len o0 | o1 | o2 | o3 | t0 | t1 | t2 | t3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16–24k | 0.81 | 0.99 | 0.98 | 0.98 | 0.96 | 0.98 | 0.95 | 0.95 | 56 | 46 | 50 | 50 | 45 | 46 | 47 | 50 |
| 24–32k | 0.62 | 0.99 | 0.98 | 0.99 | 0.98 | 0.96 | 0.99 | 0.94 | 77 | 57 | 50 | 57 | 44 | 46 | 44 | 48 |
| 32–40k | 0.87 | 1.00 | 1.00 | 0.99 | 0.95 | 0.98 | 0.98 | 0.98 | 65 | 55 | 44 | 55 | 46 | 45 | 46 | 43 |
| 40–48k | 0.93 | 1.00 | 1.00 | 1.00 | 0.98 | 0.97 | 0.99 | 0.99 | 66 | 53 | 42 | 56 | 43 | 46 | 44 | 43 |
| 48–56k | 0.88 | 0.99 | 1.00 | 1.00 | 0.99 | 0.98 | 0.99 | 0.99 | 64 | 52 | 39 | 63 | 40 | 43 | 43 | 42 |
| 56–64k | 0.73 | 0.89 | 1.00 | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 | 65 | 58 | 39 | 62 | 40 | 43 | 41 | 41 |
| 64–72k | 0.48 | 0.66 | 1.00 | 0.92 | 1.00 | 0.99 | 1.00 | 0.99 | 77 | 70 | 38 | 67 | 40 | 42 | 41 | 41 |
| 72–80k | 0.32 | 0.58 | 1.00 | 0.79 | 1.00 | 1.00 | 1.00 | 1.00 | 86 | 80 | 38 | 73 | 40 | 42 | 42 | 40 |
| 80–88k | 0.35 | 0.27 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 83 | 93 | 37 | 85 | 40 | 41 | 41 | 40 |
| 88–96k | 0.23 | 0.20 | 1.00 | 0.15 | 1.00 | 1.00 | 1.00 | 1.00 | 88 | 95 | 37 | 94 | 40 | 42 | 41 | 40 |

Two readings matter. Seeds 1 and 3 hold SR 0.99–1.00 from 24k to 56k while their episodes
lengthen from 46 to 57–63 steps; the drift toward the attractor is visible in *length* thirty
to forty thousand steps before it is visible in *success*. And seed 2 reaches −71 at 80k, which
no oracle seed does before 400k: the machinery can produce the oracle's trajectory, so the
question is what makes it unstable, not what makes it slow.

The reward is −1 per raw physics step and 0 at the goal (`raif/embodied/envs/customs/mtc.py`,
"sparse reward: −1 for every step, 0 for end"; action repeat 2), so return is −2 × steps up to
the reset-row convention and carries no information beyond length. Episode length is the
whole content of the return.

## 2. Matched series, ours / theirs

Cells are `ours / theirs`, each the across-seed mean of per-seed bin means; the full 44-series
version with across-seed standard deviations is in scratch (`tables_run4.md`). Series
mapping: `policy_entropy` ↔ `train/policy_entropy_mean` (σ = exp(H − ½(1 + ln 2π)) for the
one-dimensional Normal head on both sides); `value` ↔ `train/extr_G_mean`; the three
`term.*` ↔ `train/extr_G_{rew,crspp,ig}_mean`; `ret` ↔ `train/extr_return_raw_mean`;
`critic_value` ↔ `train/extr_critic_mean`; `advantage` ↔ `train/adv_mean`; our `invscale` is
`max(1, retnorm_high − retnorm_low)` and theirs is inferred as `raw_std / normed_std`, from
which their `low` and `high` follow; `decoder` ↔ `train/state_loss_mean` (both sum over the two
state dimensions; checked); `dyn` ↔ `train/dyn_loss_mean`; `prior_entropy` ↔
`train/policy_wm_entropy_mean`. "M+ steps added" is the cumulative length of successful
agent-collected episodes through the bin end (the same rule on both sides; theirs also logs
`positive_replay/size`, which agrees once the seeded 1.85k and one reset row per episode are
added back).

### 2.1 Behaviour and the actor channel

| bin | return | length | SR | M+ steps added | σ (imag.) | adv (normed) | invscale | p95 return |
|---|---|---|---|---|---|---|---|---|
| 0–4k | -164 / -187 | 83 / 94 | 0.34 / 0.20 | 1k / 0k | 0.649 / 0.816 | 1.200 / 0.655 | 1.51 / 1.22 | 1.56 / 1.32 |
| 4–8k | -127 / -142 | 65 / 71 | 0.70 / 0.51 | 3k / 1k | 0.251 / 0.377 | 0.771 / 0.772 | 3.13 / 2.94 | 3.68 / 3.46 |
| 8–12k | -119 / -116 | 61 / 59 | 0.74 / 0.82 | 6k / 4k | 0.210 / 0.240 | 0.662 / 0.667 | 3.45 / 3.17 | 4.06 / 3.74 |
| 12–16k | -90 / -101 | 47 / 51 | 0.97 / 0.93 | 10k / 7k | 0.214 / 0.223 | 0.540 / 0.610 | 2.93 / 2.92 | 3.43 / 3.45 |
| 16–20k | -92 / -92 | 48 / 47 | 0.96 / 0.95 | 13k / 11k | 0.243 / 0.217 | 0.076 / 0.380 | 1.69 / 2.06 | 1.65 / 2.41 |
| 20–24k | -107 / -93 | 55 / 47 | 0.91 / 0.97 | 17k / 14k | 0.254 / 0.213 | -0.810 / -0.627 | 1.60 / 1.62 | 0.12 / 0.35 |
| 24–28k | -116 / -90 | 60 / 46 | 0.86 / 0.96 | 20k / 18k | 0.254 / 0.205 | -0.712 / -0.760 | 3.49 / 3.53 | -0.11 / -0.43 |
| 28–32k | -119 / -89 | 61 / 45 | 0.92 / 0.97 | 23k / 21k | 0.240 / 0.202 | -0.605 / -0.669 | 5.21 / 5.07 | -0.04 / -0.59 |
| 32–36k | -107 / -88 | 55 / 45 | 0.95 / 0.97 | 27k / 25k | 0.248 / 0.194 | -0.581 / -0.557 | 6.37 / 6.79 | -0.04 / -0.17 |
| 36–40k | -105 / -89 | 54 / 45 | 0.98 / 0.98 | 31k / 28k | 0.243 / 0.190 | -0.552 / -0.503 | 7.13 / 8.24 | -0.07 / 0.23 |
| 40–44k | -103 / -87 | 53 / 44 | 0.99 / 0.98 | 35k / 32k | 0.256 / 0.184 | -0.549 / -0.488 | 7.65 / 9.24 | -0.08 / 0.34 |
| 44–48k | -107 / -86 | 55 / 44 | 0.98 / 0.99 | 39k / 36k | 0.246 / 0.179 | -0.537 / -0.473 | 8.09 / 9.86 | -0.08 / 0.36 |
| 48–52k | -107 / -83 | 55 / 42 | 0.98 / 0.99 | 43k / 39k | 0.223 / 0.178 | -0.523 / -0.470 | 8.49 / 10.09 | -0.08 / 0.35 |
| 52–56k | -105 / -84 | 54 / 43 | 0.96 / 0.99 | 46k / 43k | 0.221 / 0.171 | -0.516 / -0.464 | 8.88 / 10.35 | -0.07 / 0.34 |
| 56–60k | -109 / -82 | 56 / 42 | 0.90 / 0.99 | 50k / 47k | 0.207 / 0.171 | -0.511 / -0.458 | 9.26 / 10.60 | -0.05 / 0.34 |
| 60–64k | -109 / -80 | 56 / 41 | 0.90 / 1.00 | 53k / 51k | 0.200 / 0.163 | -0.499 / -0.442 | 9.57 / 10.91 | -0.04 / 0.32 |
| 64–96k | -138 / -80 | 71 / 41 | 0.61 / 1.00 | 70k / 81k | 0.168 / 0.155 | -0.464 / -0.409 | 10.79 / 11.75 | 0.03 / 0.38 |
| 96–128k | -155 / -79 | 79 / 40 | 0.38 / 1.00 | 80k / 110k | 0.137 / 0.144 | -0.421 / -0.348 | 12.67 / 13.27 | -0.20 / 0.48 |
| 128–160k | -140 / -78 | 71 / 40 | 0.52 / 1.00 | 93k / 140k | 0.128 / 0.139 | -0.388 / -0.292 | 14.86 / 15.03 | -0.75 / 0.52 |
| 160–192k | -108 / -81 | 56 / 41 | 0.79 / 1.00 | 116k / 170k | 0.120 / 0.137 | -0.354 / -0.233 | 17.12 / 17.32 | -0.98 / 0.54 |
| 192–224k | -104 / -87 | 53 / 44 | 0.85 / 0.98 | 139k / 199k | 0.120 / 0.137 | -0.312 / -0.193 | 19.04 / 19.56 | -0.77 / 0.56 |
| 224–256k | -96 / -99 | 50 / 50 | 0.92 / 0.91 | 167k / 224k | 0.125 / 0.143 | -0.265 / -0.163 | 21.98 / 21.85 | -0.40 / 0.57 |
| 256–288k | -118 / -98 | 60 / 50 | 0.73 / 0.87 | 187k / 247k | 0.144 / 0.160 | -0.224 / -0.137 | 24.70 / 24.53 | -0.15 / 0.58 |
| 288–320k | -127 / -93 | 65 / 47 | 0.68 / 0.88 | 206k / 270k | 0.179 / 0.178 | -0.198 / -0.114 | 27.68 / 27.73 | -0.37 / 0.58 |
| 320–352k | -134 / -81 | 69 / 41 | 0.67 / 0.96 | 225k / 297k | 0.204 / 0.182 | -0.167 / -0.091 | 30.34 / 30.97 | -0.51 / 0.57 |
| 352–384k | -145 / -79 | 74 / 40 | 0.58 / 0.97 | 240k / 325k | 0.230 / 0.174 | -0.159 / -0.077 | 33.22 / 34.29 | -0.66 / 0.57 |
| 384–416k | -132 / -75 | 68 / 38 | 0.64 / 0.98 | 257k / 354k | 0.239 / 0.172 | -0.146 / -0.071 | 35.74 / 36.98 | -0.77 / 0.56 |
| 416–448k | -110 / -70 | 57 / 36 | 0.83 / 1.00 | 282k / 383k | 0.240 / 0.163 | -0.130 / -0.054 | 39.21 / 39.91 | -0.60 / 0.58 |
| 448–480k | -94 / -69 | 48 / 35 | 0.97 / 1.00 | 312k / 413k | 0.191 / 0.157 | -0.124 / -0.040 | 42.19 / 42.63 | -0.28 / 0.58 |
| 480–512k | -81 / -69 | 42 / 35 | 1.00 / 1.00 | 344k / 443k | 0.152 / 0.147 | -0.100 / -0.030 | 46.05 / 44.43 | 0.04 / 0.60 |
| 512–576k | -75 / -71 | 39 / 36 | 1.00 / 1.00 | 408k / 503k | 0.119 / 0.125 | -0.065 / -0.021 | 51.30 / 48.30 | 0.28 / 0.62 |
| 576–640k | -73 / -70 | 38 / 36 | 0.99 / 1.00 | 471k / 562k | 0.116 / 0.121 | -0.028 / -0.014 | 55.60 / 52.43 | 0.56 / 0.64 |
| 640–704k | -73 / -85 | 38 / 43 | 0.99 / 0.96 | 534k / 617k | 0.117 / 0.125 | -0.013 / -0.029 | 57.49 / 64.10 | 0.68 / 0.66 |
| 704–768k | -74 / -93 | 39 / 47 | 0.98 / 0.97 | 588k / 675k | 0.117 / 0.125 | -0.009 / -0.024 | 59.23 / 70.89 | 0.70 / 0.66 |

### 2.2 The value and the world model

| bin | G (per step) | G: reward | G: cosine | G: −IG | λ-return | critic | decoder loss | dyn KL | prior H |
|---|---|---|---|---|---|---|---|---|---|
| 0–4k | 0.440 / 0.248 | -0.007 / -0.003 | 0.447 / 0.194 | -0.0002 / 0.0569 | 1.77 / 0.91 | 0.00 / 0.00 | 0.0578 / 1.0473 | 1.002 / 1.000 | 110.5 / 110.6 |
| 4–8k | 0.599 / 0.566 | -0.030 / -0.025 | 0.629 / 0.532 | -0.0000 / 0.0580 | 2.41 / 2.29 | 0.02 / 0.01 | 0.0218 / 0.0416 | 1.003 / 1.000 | 110.5 / 110.7 |
| 8–12k | 0.572 / 0.528 | -0.090 / -0.066 | 0.662 / 0.536 | -0.0000 / 0.0580 | 2.32 / 2.15 | 0.04 / 0.03 | 0.0105 / 0.0220 | 1.006 / 1.001 | 110.6 / 110.6 |
| 12–16k | 0.402 / 0.444 | -0.247 / -0.152 | 0.649 / 0.538 | -0.0000 / 0.0580 | 1.66 / 1.83 | 0.06 / 0.05 | 0.0063 / 0.0142 | 1.013 / 1.002 | 110.6 / 110.5 |
| 16–20k | 0.049 / 0.216 | -0.590 / -0.381 | 0.639 / 0.540 | -0.0000 / 0.0580 | 0.31 / 0.94 | 0.08 / 0.07 | 0.0052 / 0.0115 | 1.021 / 1.005 | 110.5 / 110.4 |
| 20–24k | -0.346 / -0.288 | -0.983 / -0.876 | 0.637 / 0.530 | -0.0000 / 0.0580 | -1.22 / -1.02 | 0.09 / 0.08 | 0.0041 / 0.0097 | 1.024 / 1.013 | 110.4 / 110.2 |
| 24–28k | -0.621 / -0.706 | -1.251 / -1.271 | 0.631 / 0.507 | -0.0000 / 0.0580 | -2.37 / -2.60 | 0.07 / 0.07 | 0.0040 / 0.0085 | 1.034 / 1.022 | 110.2 / 109.9 |
| 28–32k | -0.804 / -0.876 | -1.423 / -1.424 | 0.619 / 0.489 | -0.0000 / 0.0580 | -3.09 / -3.31 | 0.05 / 0.04 | 0.0039 / 0.0080 | 1.041 / 1.029 | 109.8 / 109.7 |
| 32–36k | -0.945 / -0.944 | -1.564 / -1.478 | 0.619 / 0.476 | -0.0000 / 0.0580 | -3.68 / -3.75 | 0.02 / 0.01 | 0.0035 / 0.0076 | 1.042 / 1.037 | 109.2 / 109.4 |
| 36–40k | -1.015 / -1.036 | -1.617 / -1.556 | 0.603 / 0.461 | -0.0000 / 0.0580 | -3.95 / -4.16 | -0.02 / -0.03 | 0.0035 / 0.0071 | 1.048 / 1.041 | 108.4 / 109.0 |
| 40–44k | -1.081 / -1.134 | -1.681 / -1.642 | 0.600 / 0.450 | -0.0000 / 0.0580 | -4.27 / -4.62 | -0.07 / -0.12 | 0.0032 / 0.0066 | 1.057 / 1.051 | 107.4 / 108.4 |
| 44–48k | -1.122 / -1.181 | -1.710 / -1.690 | 0.587 / 0.451 | -0.0000 / 0.0580 | -4.47 / -4.89 | -0.13 / -0.23 | 0.0029 / 0.0059 | 1.058 / 1.055 | 106.3 / 107.7 |
| 48–52k | -1.154 / -1.205 | -1.734 / -1.717 | 0.579 / 0.455 | -0.0000 / 0.0580 | -4.66 / -5.06 | -0.21 / -0.32 | 0.0028 / 0.0059 | 1.058 / 1.059 | 104.5 / 107.3 |
| 52–56k | -1.190 / -1.224 | -1.759 / -1.735 | 0.569 / 0.453 | -0.0000 / 0.0580 | -4.88 / -5.21 | -0.30 / -0.42 | 0.0025 / 0.0056 | 1.067 / 1.063 | 102.5 / 106.8 |
| 56–60k | -1.227 / -1.243 | -1.781 / -1.753 | 0.554 / 0.451 | -0.0000 / 0.0580 | -5.13 / -5.39 | -0.40 / -0.54 | 0.0023 / 0.0052 | 1.064 / 1.064 | 100.4 / 106.1 |
| 60–64k | -1.240 / -1.244 | -1.791 / -1.760 | 0.550 / 0.457 | -0.0000 / 0.0580 | -5.29 / -5.55 | -0.51 / -0.73 | 0.0022 / 0.0049 | 1.075 / 1.074 | 98.1 / 105.2 |
| 64–96k | -1.301 / -1.253 | -1.817 / -1.773 | 0.517 / 0.462 | -0.0000 / 0.0580 | -6.12 / -6.21 | -1.12 / -1.41 | 0.0020 / 0.0037 | 1.075 / 1.083 | 88.0 / 101.7 |
| 96–128k | -1.383 / -1.236 | -1.853 / -1.772 | 0.469 / 0.478 | -0.0000 / 0.0580 | -7.61 / -7.40 | -2.30 / -2.78 | 0.0017 / 0.0023 | 1.066 / 1.106 | 72.1 / 93.8 |
| 128–160k | -1.511 / -1.226 | -1.867 / -1.773 | 0.356 / 0.488 | -0.0000 / 0.0580 | -9.27 / -8.78 | -3.50 / -4.40 | 0.0015 / 0.0016 | 1.067 / 1.123 | 60.7 / 86.3 |
| 160–192k | -1.615 / -1.202 | -1.867 / -1.767 | 0.252 / 0.507 | -0.0000 / 0.0580 | -10.88 / -10.24 | -4.81 / -6.22 | 0.0013 / 0.0013 | 1.063 / 1.128 | 53.4 / 80.3 |
| 192–224k | -1.611 / -1.196 | -1.860 / -1.767 | 0.249 / 0.513 | -0.0000 / 0.0580 | -12.14 / -11.64 | -6.21 / -7.87 | 0.0011 / 0.0012 | 1.073 / 1.130 | 49.4 / 75.6 |
| 224–256k | -1.621 / -1.203 | -1.857 / -1.769 | 0.236 / 0.508 | -0.0000 / 0.0580 | -13.80 / -12.94 | -7.96 / -9.38 | 0.0011 / 0.0011 | 1.086 / 1.133 | 47.3 / 71.6 |
| 256–288k | -1.601 / -1.231 | -1.852 / -1.774 | 0.251 / 0.485 | -0.0000 / 0.0580 | -15.60 / -14.36 | -10.05 / -11.00 | 0.0010 / 0.0011 | 1.083 / 1.136 | 47.2 / 68.4 |
| 288–320k | -1.654 / -1.280 | -1.855 / -1.778 | 0.201 / 0.440 | -0.0000 / 0.0580 | -18.10 / -15.96 | -12.61 / -12.79 | 0.0013 / 0.0011 | 1.090 / 1.140 | 49.7 / 65.9 |
| 320–352k | -1.639 / -1.296 | -1.854 / -1.779 | 0.215 / 0.425 | -0.0000 / 0.0580 | -20.45 / -17.44 | -15.37 / -14.62 | 0.0011 / 0.0009 | 1.097 / 1.138 | 54.4 / 64.6 |
| 352–384k | -1.730 / -1.335 | -1.858 / -1.776 | 0.128 / 0.383 | -0.0000 / 0.0580 | -22.50 / -18.84 | -17.18 / -16.20 | 0.0011 / 0.0009 | 1.096 / 1.142 | 58.0 / 63.1 |
| 384–416k | -1.729 / -1.380 | -1.862 / -1.773 | 0.133 / 0.335 | -0.0000 / 0.0580 | -24.31 / -20.20 | -19.07 / -17.57 | 0.0012 / 0.0009 | 1.097 / 1.140 | 62.0 / 61.8 |
| 416–448k | -1.745 / -1.345 | -1.863 / -1.772 | 0.118 / 0.368 | -0.0000 / 0.0580 | -26.09 / -20.92 | -20.98 / -18.77 | 0.0011 / 0.0009 | 1.097 / 1.139 | 65.5 / 60.8 |
| 448–480k | -1.720 / -1.326 | -1.858 / -1.767 | 0.138 / 0.383 | -0.0000 / 0.0580 | -27.40 / -21.39 | -22.17 / -19.65 | 0.0011 / 0.0009 | 1.102 / 1.136 | 66.5 / 60.0 |
| 480–512k | -1.688 / -1.298 | -1.850 / -1.764 | 0.162 / 0.408 | -0.0000 / 0.0580 | -28.64 / -21.32 | -24.02 / -20.00 | 0.0011 / 0.0009 | 1.100 / 1.134 | 66.0 / 59.4 |
| 512–576k | -1.688 / -1.395 | -1.840 / -1.754 | 0.151 / 0.301 | -0.0000 / 0.0580 | -30.90 / -22.44 | -27.58 / -21.42 | 0.0011 / 0.0009 | 1.108 / 1.133 | 64.7 / 58.0 |
| 576–640k | -1.653 / -1.412 | -1.826 / -1.750 | 0.173 / 0.280 | -0.0000 / 0.0580 | -32.22 / -24.25 | -30.65 / -23.51 | 0.0011 / 0.0009 | 1.112 / 1.132 | 60.9 / 56.5 |
| 640–704k | -1.598 / -1.463 | -1.806 / -1.755 | 0.208 / 0.234 | -0.0000 / 0.0580 | -31.86 / -36.68 | -31.12 / -34.77 | 0.0009 / 0.0009 | 1.119 / 1.133 | 56.8 / 55.7 |
| 704–768k | -1.562 / -1.519 | -1.794 / -1.767 | 0.232 / 0.190 | -0.0000 / 0.0580 | -32.69 / -43.38 | -32.13 / -41.65 | 0.0009 / 0.0009 | 1.121 / 1.131 | 54.5 / 55.8 |

### 2.3 The plateau window as ratios

| bin | decoder ours/theirs | dyn KL above the free-bits floor, ours / theirs | prior H ours − theirs | σ ours − theirs | invscale ours/theirs | critic gap ours/theirs |
|---|---|---|---|---|---|---|
| 4–8k | 0.52 | 0.003 / 0.000 | -0.2 | -0.126 | 1.06 | 1.05 |
| 8–12k | 0.48 | 0.006 / 0.001 (6.3×) | +0.0 | -0.030 | 1.09 | 1.08 |
| 12–16k | 0.44 | 0.013 / 0.002 (5.5×) | +0.1 | -0.009 | 1.01 | 0.90 |
| 16–20k | 0.45 | 0.021 / 0.005 (4.3×) | +0.1 | +0.026 | 0.82 | 0.27 |
| 20–24k | 0.42 | 0.024 / 0.013 (1.9×) | +0.2 | +0.042 | 0.98 | 1.19 |
| 24–28k | 0.47 | 0.034 / 0.022 (1.5×) | +0.3 | +0.048 | 0.99 | 0.92 |
| 28–32k | 0.49 | 0.041 / 0.029 (1.4×) | +0.1 | +0.038 | 1.03 | 0.94 |
| 32–36k | 0.46 | 0.042 / 0.037 (1.1×) | -0.1 | +0.053 | 0.94 | 0.98 |
| 36–40k | 0.49 | 0.048 / 0.041 (1.2×) | -0.6 | +0.053 | 0.87 | 0.95 |
| 40–44k | 0.48 | 0.057 / 0.051 (1.1×) | -1.0 | +0.072 | 0.83 | 0.93 |
| 44–48k | 0.49 | 0.058 / 0.055 (1.0×) | -1.5 | +0.067 | 0.82 | 0.93 |
| 48–52k | 0.47 | 0.058 / 0.059 (1.0×) | -2.8 | +0.045 | 0.84 | 0.94 |
| 52–56k | 0.46 | 0.067 / 0.063 (1.1×) | -4.2 | +0.050 | 0.86 | 0.95 |
| 56–60k | 0.44 | 0.064 / 0.064 (1.0×) | -5.7 | +0.036 | 0.87 | 0.97 |
| 60–64k | 0.45 | 0.075 / 0.074 (1.0×) | -7.1 | +0.037 | 0.88 | 0.99 |

The two columns that differ from unity by more than seed noise across the whole window are the
decoder loss (ours 0.42–0.52× theirs for 60k steps, then equal from 128k) and, from 36k on,
the prior entropy (ours lower by 7 nats at 64k, 14 at 64–96k, 22 at 96–128k, 26–27 at
128–192k; theirs never catches up until ours turns back *up* during the second collapse,
256–512k). The KL leaves the free-bits floor 4–6× earlier in ours (8–20k) and matches from 44k.
The actor's σ is 0.03–0.07 wider than theirs from 16k to 64k and 0.13 narrower at 4–8k.

### 2.4 Two closed-form identifications along the way

**Their info-gain term is a constant, to four figures, and ours is zero.** `train/extr_G_ig_mean`
is 0.0580 in every bin from 4k to 768k with across-seed sd 0.0000. Their leaf is
`(mean_dims log(std over 4 sampled members) − log 1) / √32` with every member at σ ≡ 1
(`epistemics.py:33-45`); when the members agree, the sample standard deviation of four unit
normals has `E[log s] = ½(ψ(3/2) + ln 2 − ln 4) = −0.32833`, so the leaf is −0.05804 and
`G_ig = +0.05804`. The logged value is that number: the term measures the sampling noise of
its own estimator and nothing about the ensemble. Ours is the moment-matched estimator on the
same members (adjudicated 2026-09-05) and reads −2 × 10⁻⁴ at 1k and −1.3 × 10⁻⁶ at 760k, i.e.
the four members agree to 0.4%. A constant per-step bonus shifts every 8-step λ-return by
`0.058 × Σ_{k<8} (γλ)^k = 0.058 × 6.666 = +0.387` where the rollout is not truncated, which is
why their `p95 return` sits at +0.34 through 40–64k while ours sits at −0.08 with the same
success rate (§2.1); it is also a small bias *against* terminating, since reaching the goal
forfeits the bonus, so its direction cannot explain our longer episodes. What it does have
that ours lacks is a *gradient*: the sampled `std` depends on the member means, which condition
on the action, so their actor receives a random-direction action gradient at every imagined
step. That consequence is inferred and unpriced; §5 ranks it.

**Both codes carry about one nat.** Their posterior entropy tracks their prior entropy to within
0.7 nats through 64k (110.35/110.62 at 0–8k, 105.23/106.04 at 48–64k) and `dyn_loss` sits at
the free-bits floor (1.000–1.07), so the 32 × 32 code carries at most ~1 nat per step of
observation information on their side, and 1–1.07 on ours. Localization runs through the GRU
carry integrating those nats and the actions; the decoder's 0.005 symlog MSE is consistent with
knowing everything but the initial position. This bounds what the act-time mean-field
divergence can do and is why the probe in §4.1 finds it inert.

## 3. Killed hypotheses — the discriminating number for each

**H1, exploration (σ collapses to the floor, or stays too wide).** Killed as a cause. In 16–64k
our imagined σ is 0.20–0.26 against their 0.16–0.22 (wider by 0.03–0.07, never narrower), and
the collapse itself is *confident*: σ falls to 0.12–0.15 as SR falls to 0.1 (seed 0: 0.24 at
24–32k → 0.16 at 88–96k → 0.13 at 120k). The collapsed actor at 775k pushes right on 100% of
steps with σ 0.115. Two of the three collapsing seeds (0 and 1) did carry the widest σ before
collapse (0.24–0.31 vs seed 2's 0.22–0.24), but seed 3 collapsed with seed 2's σ, so σ is a
covariate at best. Their acting-time σ (from `stats/mean_log_entropy`) is 1.05–1.15× their
imagined σ; ours is not logged (§7).

**H2, the critic damps the advantage.** Killed. The critic gap `ret − critic` is 0.90–0.99× theirs
from 24k to 64k (both critics lag by 5 units at 60k against true values near −70), and our
normalizer band is *narrower* (invscale 0.82–0.88× theirs from 36k), so our normalized
advantage is 10–20% *larger* in magnitude, not smaller. The slow critic (live since divergence
29 closed) behaves like theirs: the gap ratio never exceeds 1.08 outside the two crossing bins.

**H3, M+ fills slower.** Killed for the window. Our successful-episode steps run ahead of theirs
by 1.8k–3.4k at every bin from 4k to 64k (53.2k vs 50.6k at 64k); we fall behind only after
the collapses (70k vs 81k at 64–96k). The CRSPP series agree: our `crspp` total 0.06–0.07
against their `C_model + contrast` ≈ 0.10 − 0.03 = 0.07, and our positive-gate fraction is
0.15–0.19 throughout.

**H4, the objective handoff.** Killed. The per-step value crosses zero in the same bin on both
sides (16–20k: ours +0.049 → −0.346, theirs +0.216 → −0.288) and the normalizer band compresses
identically (3.2 → 1.7 → 4.3 vs 3.0 → 1.8 → 4.5). Seed 0's first dip does start there, but
seeds 1 and 3 hold SR 0.99 for another 40k steps and the oracle never dips at all, so the
crossing is not the trigger.

**H6a, mean-field beliefs at act time (the one acting-path divergence on the record, "stands,
untouched" in `amortization.md` §(f)).** Killed by measurement in §4.1: with identical
parameters, acting on sampled codes as their `obs_step` does gives equal or *longer* episodes
than our mean-field path at both checkpoints, even though the two paths disagree on the sign of
the actor's mean on 2–6% of steps.

**Model exploitation.** Killed by measurement in §4.2: the collapsed actor's own world model
predicts its failure (2.5% imagined termination within 30 steps against 66–83% for the
healthy seeds, whose models match reality within 7 points).

## 4. Checkpoint probes (measured, read-only)

### 4.1 The act-time belief mismatch, priced

For each seed, 32 windows of 64 replayed steps from that seed's M were run through the
training recognition path (straight-through sampled codes, `train/world.posterior_states`) and
through the collection path exactly as `run_mc.make_collect_fn` runs it (`System.infer_states`
mean-field posterior, `update_empirical_prior` on the recorded action, reset to
`episode_start_beliefs` at `is_first`), and the actor was evaluated on both feature vectors.

| checkpoint | seed | code entropy of the mean-field posterior (nats) | max-prob per variable | argmax(probs) = sampled one-hot | \|Δ deter\| / \|deter\| | actor μ sign flips | σ act / σ train |
|---|---|---|---|---|---|---|---|
| run 4, 775k | 0 | 48.9 | 0.43 | 43% | 15% | 3.4% | 0.98 |
| run 4, 775k | 1 | 52.9 | 0.41 | 41% | 15% | 4.0% | 0.96 |
| run 4, 775k | 2 | 60.7 | 0.41 | 39% | 21% | 5.7% | 0.91 |
| run 4, 775k | 3 | 53.7 | 0.40 | 40% | 8% | 0.05% | 1.01 |
| run 3, 2.0M | 0–3 | 46.7–60.2 | 0.38–0.45 | 38–44% | 10–13% | 2.1–4.2% | 0.93–0.99 |

The mismatch is real (the code features differ by 1.5× their own scale; the actor's mean
changes sign on 2–6% of steps for three of four seeds) and it is inert for behaviour: 64
paired real episodes per seed with common reset and action-noise keys give

| checkpoint | pooled SR mean-field / sampled | pooled length | per-seed paired Δlength (mean-field − sampled) ± s.e. |
|---|---|---|---|
| run 4, 775k | 0.734 / 0.738 | 53.4 / 54.1 | −0.84 ± 0.44, −0.80 ± 0.88, −1.22 ± 1.36, 0.00 (seed 3: SR 0 on both) |
| run 3, 2.0M | 1.000 / 0.996 | 35.5 / 36.0 | −0.14 ± 0.10, −0.14 ± 0.19, −1.52 ± 0.79, −0.02 ± 0.08 |

Mean-field acting is equal or slightly better on every seed. The reading in §2.4 explains why:
the probability tensor carries the same ~1 nat as the sample without the sample's 50-nat noise.
Recorded so the divergence is priced rather than argued; nothing here suggests closing it.

### 4.2 The collapsed actor against its own model

Sixteen real episodes per seed under the run's acting path; along each, the sampled posterior
was filtered and, from every fifth real state, 30 imagined steps were rolled with the actor
through `img_step` and the three heads.

| checkpoint | seed | real SR | real length | steps with a > 0 | position range | model: terminates within 30 | real: goal within 30 | model max position | real max position | model cont prob | model reward |
|---|---|---|---|---|---|---|---|---|---|---|---|
| run 4, 775k | 0 | 1.00 | 34.6 | 90% | [−0.88, 0.49] | 0.83 | 0.83 | 0.45 | 0.38 | 0.59 | −1.19 |
| run 4, 775k | 1 | 1.00 | 35.5 | 85% | [−1.02, 0.50] | 0.81 | 0.80 | 0.41 | 0.34 | 0.68 | −1.35 |
| run 4, 775k | 2 | 1.00 | 38.8 | 81% | [−1.20, 0.49] | 0.66 | 0.73 | 0.34 | 0.28 | 0.63 | −1.28 |
| run 4, 775k | **3** | **0.00** | **100.0** | **100.0%** | **[−0.63, +0.06]** | **0.025** | 0 | **−0.04** | −0.13 | **0.995** | **−1.99** |
| run 3, 2.0M | 0–3 | 1.00 | 34.3–36.4 | 63–89% | goal reached | 0.70–0.87 | 0.77–0.86 | 0.39–0.46 | 0.32–0.39 | 0.63–0.79 | −1.28 to −1.60 |

The collapsed policy is a constant `+1` push. In Mountain Car a constant push cannot climb; the
car oscillates between −0.63 and +0.06. The world model predicts exactly that (continuation
0.995, decoded maximum position −0.04, reward −2 per step) — it is not being exploited. Healthy
models are mildly optimistic (+0.06 in decoded maximum position) on both runs, equally.

The mechanism this leaves is the standard failure of an imagination actor-critic once the
policy stops succeeding: from any replayed start, every imagined rollout under a constant-push
policy returns ≈ −2 per step with no termination, so the λ-return loses its state dependence,
the critic regresses toward a state-independent constant, and the pathwise advantage gradient
that would say "push left first" vanishes. Recovery then depends on whatever shaping survives —
here the cosine term (still +0.24 for seed 3 at 760k) and the σ = 0.1 floor. Their `action_mean`
series shows the same attractor pulling at their dip (imagined mean action +0.23 at 160k rising
to +0.66 at 300k, SR falling to 0.71–0.77) and letting go; theirs is shallow and ours is deep.

## 5. Surviving hypotheses, ranked

Ranking is by weight of evidence in the 16k–64k window, then by mechanism. "Measured" means a
number from the logs or probes above; "inferred" means a reading of the two codes.

**1. H5 — the world model learns faster and sharpens earlier than theirs; open divergences 17
and 26 are the primary attribution.** *Measured:* decoder loss 0.42–0.52× theirs for 60k steps
from the first logged row (0.0218 vs 0.0416 at 4–8k), the KL above the free-bits floor 4–6×
theirs at 8–20k, the imagined prior entropy lower by 7 nats at 60–64k, 14 at 64–96k and 26–27 at
128–192k on every one of our seeds (seed 2 included), and the actor sharpening 1.5× faster in
the imitation phase (σ 0.251 vs 0.377 at 4–8k) as the downstream consequence of sharper
features. *Inferred:* that a faster and sharper model makes the attractor more likely to be
entered and harder to leave — the register's own note that 17/26 "would show up as early
world-model trajectory differences" is confirmed, and the 2026-09-07 measurement that their
initializer *loses* to ours on the open-loop probe is the same fact from the other side (their
model learns slower). Seed 2 has the same model and did not collapse, so the link is
probabilistic at best. Their side: `raif/aif/nets.py:656-690` (`Linear`, `_bias = bias and
norm == 'none'` at :664) and `:740-763` (`Initializer`, `winit: normal`, `fan: avg`). Ours:
`distributions/neural.py:689` (`_linear_stack`, `eqx.nn.Linear` defaults) and `GRUCell.from_shapes`.
Alternative attributions for the same measured signature, not separable from logs: their
`float16` compute (argued divergence) and the CRSPP gradient scope below.

**2. H6b — their encoder is trained by the ρ-signed CRSPP contrast; ours is not.** *Inferred
mechanism, no series:* their `similarity = cosine_sim(C_post, sg(post))` back-propagates
through `C_post`'s `embed` input into the shared encoder inside one `model_opt`
(`agent.py:182-183, 191, 197-198, 229-231, 241`), so the representation the actor and critic
read is shaped to separate success-like from failure-like states; ours hands the CRSPP
pre-computed embeddings with no gradient to the encoder (`reference/crspp.py:317`
`fit_crspp` over `train/world.py:359` `embeddings`; the deliberate seam of debt-ledger entry 5
and `amortization.md` §(f) "CRSPP gradient scope"). A value gradient that survives a failing
policy is exactly what the attractor lacks. This is a re-proposal of a signed-off seam and is
flagged as such: it belongs in an A/B arm, not in the parity default.

**3. H6c — the argued sampled-std info-gain leaf carries a random action gradient theirs has and
ours does not.** *Measured:* the constant (§2.4). *Inferred:* the gradient's magnitude. Estimated
at 10⁻⁴–10⁻³ per imagined step from the ensemble's action input, which is negligible against
the reward and cosine gradients where those are non-zero and comparable to them where the
landscape is flat, i.e. in the attractor. Cheap to test because the sampled formula already
exists as a port (`tests/test_oracle_transcription.py::_their_information_gain`).

**4. The cosine term's level and gradient (mean-field preferred rollout, D6-adjacent).** Ours
runs 0.10–0.15 above theirs for 60k steps (0.63 vs 0.53) and then collapses to ≈ 0 during the
dips (seed 0: 0.53 at 88k → 0.27 at 152k → −0.03 at 192k) where theirs holds 0.45–0.55; the
level is not comparable by construction (register) and the collapse *lags* SR by ~50k steps,
so it is a consequence that prolongs a dip rather than its start. Unranked as a cause; recorded.

Killed: H1, H2, H3, H4, H6a (acting path) and model exploitation, §3.

## 6. Proposed intervention and the cheapest experiment

**Intervention (arm B, first):** close divergences 17 and 26 as a pair on the six remaining
blocks — their truncated-normal `fan: avg` initializer and their no-bias-where-normed rule on
the encoder, decoder, reward and cont heads, critic and slow copy, and the GRU. The
implementation exists in the 2026-09-07 history and was reverted on a 300-step open-loop
probe; the register already rules that "a revisit belongs on a trajectory rather than a probe".

**Run:** 4 seeds × 100k pooled steps (≈ 2.8 h at 10 steps/s), parity config otherwise, plus the
logging in §7 (no semantics change). Run 4's own first 100k is the control; no second run of
the control is needed.

**Series to watch and pass bands (pre-stated; oracle bands are the 4-seed mean ± its spread).**

| # | series | run 4 (control) | oracle | pass if |
|---|---|---|---|---|
| 1 | decoder loss ratio ours/oracle, 8–64k | 0.42–0.52 | 1.0 | 0.80–1.25 in every 4k bin from 8k |
| 2 | dyn KL above the free-bits floor, 8–20k | 4.3–6.3× oracle | 1.0 | ≤ 1.5× |
| 3 | imagined prior entropy at 60–64k / 64–96k | 98.1 / 88.0 | 105.2 ± 0.7 / 101.7 ± 1.2 | ≥ 103.5 / ≥ 99 |
| 4 | actor σ at 4–8k | 0.251 | 0.377 ± 0.037 | ≥ 0.32 |
| 5 | episode length at 24–32k, per seed / pooled | 77, 57, 50, 57 / 60 | 44, 46, 44, 48 / 46 | no seed > 50; pooled ≤ 48 |
| 6 | episode length at 56–64k, pooled | 56 | 41 ± 1 | ≤ 46 |
| 7 | seeds with any 8k bin at SR < 0.5 in 16–100k | 3 of 4 | 0 of 4 | ≤ 1 of 4 |

**Decision rule.** Rows 1–4 test whether the intervention closes the *measured* world-model
signature; rows 5–7 test whether closing it removes the *behaviour*. If 1–4 pass and 5–7 pass,
H5 is confirmed and the arm becomes a parity change to adjudicate. If 1–4 pass and 5–7 fail,
the model-speed difference is closed but is not the cause, and arm A runs next. If 1–4 fail,
17/26 is not the source of the model-speed difference; the remaining attributions are `float16`
and the CRSPP gradient scope, and arm A runs next with row 1–3 still watched.

**Arm A (second):** let the ρ-signed contrast and the gated CRSPP KL back-propagate into the
encoder, as one joint world/CRSPP loss under one clip, quoting and setting aside debt-ledger
entry 5 for the arm only. Same run, same rows 5–7, plus their `train/similarity_mean` logged on
our side (0.44–0.52 on theirs through 250k). **Arm C (third):** restore their sampled-std leaf as
the info-gain estimator; watch `term.model_uncertainty` land on +0.0580 (row 0 of the
signature) and rows 5–7.

**What would falsify the whole framing:** a control-arm seed that collapses with no length drift
in 24–56k, or an arm that passes 5–7 while 1–4 are unchanged.

## 7. Defects and gaps found regardless of parity

1. **`--resume` follows the `out` stored in `run.json`, not the directory it was pointed at.**
   `run_mc.parse` (`reference/run_mc.py:1631-1652`) rebuilds `cfg = Config(**stored)` from the
   run's own `run.json`, whose `out` field is the path the run was *launched* with. All three
   renamed directories (`mc_2d_dropcarry`, `mc_2d_igfix`, `mc_2d_igbug`) store
   `"out": "runs/mc_2d"`, so `--out runs/mc_2d_dropcarry --resume` would load
   `runs/mc_2d/checkpoint` — the live run — and checkpoint back into it. The probe in §4.1
   did exactly this on its first attempt and returned run 4's numbers under run 3's name until
   `out` was overridden. Fix: `stored["out"] = args.out` before `Config(**stored)`, or stop
   persisting `out`.
2. **`bc_logprob` is the unweighted mean of `log π(a_start | s_0)` over all starts**
   (`train/imagine.py:500`), so it says nothing about the imitation term, which is
   `mean(−log π · can_self_imitate)`. It reads −3 to −5 in the plateau window and −50 to −90 at
   430k+ because old raw draws from a wide early policy are far from a saturated `tanh` mean —
   a distraction, not a signal. Their `actor_refresh_mean` is the flag-weighted quantity; ours
   should log the same (`bc_scale · mean(bc)`) beside it.
3. **Series the analysis needed and could not get:** imagined action mean and std (their
   `train/action_{mean,std}`, which showed their dip as a push-right drift), act-time policy
   entropy per episode (their `stats/mean_log_entropy`), the flag-weighted BC term, and the
   fraction of imagined steps at continuation mode 0. All are one line each in
   `ImagineMetrics`/`run_mc.train` and belong in any run that will be compared with the oracle.
4. **The epistemic term is inert in both codebases at this configuration** (§2.4): theirs is a
   constant equal to `E[log s]` of four unit normals divided by √32, ours is 10⁻⁶. Not a defect in
   our code, but a fact the sampled-std adjudication of 2026-09-05 should carry, since the
   remaining difference between the two estimators is a gradient, not a value.

## 8. What this analysis could not cover

There is no checkpoint inside 16k–64k, so the two probes in §4 read the converged regime
(775k, 2M) and the collapsed seed they caught; the attractor's *entry* was characterised from
`scores.jsonl` alone. The three surviving candidates share the property that their consequence
for the attractor cannot be read from either side's logs, which is why §6 is an experiment
rather than a ruling. The `mc_2d_igfix` and `mc_2d_igbug` directories could not be rebuilt by
the current `Config` (schema drift) and lack the actor channel, so the contrast with earlier
runs rests on `mc_2d_dropcarry` (run 3), whose first 16k steps match run 4's to within seed
noise before its normalizer fault diverges it (−130 at 16–20k against run 4's −92).
