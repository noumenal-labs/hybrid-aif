# Cross-data experiment: does their reversal follow the model code or the data?

**Answer: the code (together with each stage's parameters).** Every cell run with our code keeps a
positive slope toward bang-bang, on either data source. Every cell run with their code keeps a negative
one, on either source. This holds at N = 0 (each stage model read on the other side's states), after
N = 256 updates (two independent batch/key replicates), and after N = 1024. Training on the other side's
data never flips a sign. Over training, their code drives the cosine leaf negative on **our** data too,
and our code keeps it positive on **theirs**. So the drift also follows the code.

A correction to §26 that came out of this: at this stage **our** return also has the reward-through-cont
channel (−39/−50 ×10⁻³ at N = 0, the same size as theirs). What separates the two sides is the
**direct cosine leaf**: ours +22…+39, theirs −27…+2.

## Design as run

- Stages: ours `runs/causal32k_key0_s20480` (seed 0, step 20480, models + actor + LearnState).
  Theirs `raif-paper/logs/probe28k_state_mc_seed0_s20400` (step 20400, **fp16 as shipped**, all
  varibs including optimizer moments, retnorm and grad_scale). The replay is time-filtered by
  `tf_chunks.unique_chunks_tf`: 24 chunks, 22,333 frames. Our general buffer holds 22,214 frames.
- Frozen data, no environment interaction. Train batches are 16×64. Ours are drawn with
  `sample_windows(buffers[0][1], PRNGKey(20000+j))`. Theirs are drawn uniformly over windows of the
  time-filtered chunks, rng 1000+j, using `measure_grad`'s rule. The replicate uses +5000 on both
  batch seeds and train keys. The long run's first 256 batches are the main run's (checked
  bitwise), and its N = 256 row reproduces the main run's.
- Updates: ours call `reference.raif.agent.train_step` directly (jitted), with the world model,
  ensemble, CRSPP, actor and critic, and the learn state carried from the stage. Theirs call
  `nj.pure(agent.train)(…, create=False, modify=True)` under `jax.jit`, the real update with their
  Optimizer inside (loss scale 1e4, no overflow logged). Their train carry (last latent and action)
  is threaded across updates as their loop does, starting from `train_initial`.
- Readout: 32 held windows per data source (ours: PRNGKey(5), as in slope_probe; theirs: rng 5).
  They are recognised by each side's own world model, flattened, and every 4th state is kept, giving
  512 starts. Policy: `(1−ε)·m_actor + ε·sign(v̂)` with the actor's scale; ε ∈ {0, .05, .2}.
  Two key sets of 8 keys each; the table gives their mean, ± half their gap. Our R is slope_probe's
  λ-return averaged over the horizon; theirs is slope_leaf_probe's ret₀. Both follow the established
  per-side method.
  - Leaves are exact splits of the λ-return. The split error is below 1e-4 on ours and below 5e-3 on
    theirs (fp16 ret vs float64 recursion).
  - "Via cont" means the leaf's per-step values are held at ε = 0 and only the discount/cont changes.
    "Cosine direct" is the cosine leaf minus its via-cont part.
  - The termination figure is the share of (start, key) rollouts with min cont < .5 over the horizon.
- Validation:
  - Our readout reproduces slope_probe on off4 seed 0 at 18.4k: +29.5 / +28.0 vs +29.50 / +27.99.
  - Their N = 0 read on their data gives total −16/−24, cosine −1, reward −17. §26's 20.4k row reads
    −18 / −0 / −20.

## The 2×2 (stage actor held; ×10⁻³ per unit ε; each entry reads ε=.05 / ε=.2; sign + = wants bang-bang)

| cell | N | total | cosine leaf | cosine direct | reward via cont | terminating rollouts ε=0→.2 |
|---|---|---|---|---|---|---|
| (1) ours code / ours data | 0 | +11 / +14 | +47 / +61 | +31 / +39 | −39 / −50 | .152 → .146 |
| | 256 | +14 / +14 | +30 / +37 | +25 / +30 | −16 / −23 | .168 → .166 |
| | 256 rep | +13 / +10 | +32 / +42 | +26 / +32 | −19 / −33 | .167 → .162 |
| | 1024 | +10 / +12 | +26 / +30 | +22 / +28 | −6 / −4 | .165 → .164 |
| (2) ours code / THEIR data | 0 | +19 / +20 | +39 / +46 | +30 / +34 | −21 / −27 | .104 → .100 |
| | 256 | +11 / +17 | +29 / +35 | +23 / +29 | −18 / −19 | .104 → .102 |
| | 256 rep | +13 / +18 | +27 / +30 | +22 / +26 | −15 / −12 | .116 → .115 |
| | 1024 | +14 / +17 | +24 / +27 | +22 / +26 | −2 / −2 | .115 → .115 |
| (3) their code / their data | 0 | −16 / −24 | −1 / −0 | −9 / −13 | −16 / −26 | .065 → .063 |
| | 256 | −32 / −29 | +6 / +4 | −7 / −7 | −42 / −35 | .056 → .053 |
| | 256 rep | −33 / −49 | +6 / +8 | −6 / −10 | −41 / −60 | .062 → .057 |
| | 1024 | −42 / −53 | −7 / −21 | −16 / −26 | −20 / −11 | .086 → .083 |
| (4) their code / OUR data | 0 | −21 / −12 | +23 / +14 | +2 / +1 | −49 / −29 | .091 → .087 |
| | 256 | −31 / −44 | −2 / −1 | −14 / −17 | −31 / −45 | .140 → .137 |
| | 256 rep | −28 / −38 | +7 / +8 | −7 / −10 | −37 / −49 | .156 → .153 |
| | 1024 | −48 / −55 | −15 / −19 | −22 / −28 | −12 / −9 | .154 → .152 |

Noise: the half-gap between key sets is ±0–15 (median ~3). The per-start SE at ε = .2 is ±4–12 on the
total. The rows at N = 512 and 768 are in `tables.txt`, and they are monotone with these.

**Reading.** (2) looks like (1), and (4) looks like (3), in the sign of the total, in the sign of the
cosine-direct leaf, and in how each drifts under training. So the reversal **follows the code**. It is
already fixed in the stage parameters at N = 0: each stage model gives its own sign on the other side's
states. Up to 1024 updates of the other side's data, 16k env steps' worth at the shipped ratio, do not
move the sign.

**What does follow the data** is the level of imagined termination. Both codes predict more
terminating rollouts on our states (.14–.17) than on theirs (.05–.12). Its ε-sensitivity is the same
small negative in every cell: moving toward bang-bang slightly *delays* imagined termination
(−0.5 to −6‰ at ε = .2), on both sides and on both data.

**The cont channel is not what separates the sides.** Reward through cont is negative in all four cells
at N = 0. Under our training it decays on either data: −39 → −6 on ours, −21 → −2 on theirs by 1024.
Under theirs it persists (−9 … −45). The side-separating term is the direct cosine leaf. Ours stays
+22…+30 under training on either data. Theirs goes negative on either: −9 → −16/−26 on their data and
+2 → −22/−28 on ours.

## Secondary read (the cell's own trained actor)

This read is lower on our side, because the actor moves too:

- (1) +13/+10 at N = 256 and +2/−1 at 1024.
- (2) −4/+4 at 256 and −11/−11 at 1024.

Their side is more negative:

- (3) −22/−32 at 256 and −83/−99 at 1024.
- (4) −18/−44 at 256 and −48/−63 at 1024.

The code split survives, but on our side the trained actor ends near a flat slope, and on their data
slightly negative. The actor keeps training in every cell, so this read differs from the primary.

## Sanity: losses learned the foreign data equally (mean of the first 32 → last 32 updates)

| cell | reward NLL | cont NLL | obs/state | contrast | at N=1024 reward NLL |
|---|---|---|---|---|---|
| (1) | .747 → .547 | .026 → .021 | .0083 → .0081 | −.054 → −.052 | .286 |
| (2) | .744 → .540 | .026 → .021 | .0084 → .0076 | −.034 → −.040 | .286 |
| (3) | .742 → .540 | .036 → .031 | .0166 → .0143 | −.038 → −.044 | .281 |
| (4) | .745 → .548 | .033 → .027 | .0090 → .0085 | −.056 → −.058 | .298 |

The reward head's NLL falls about as much in every cell. Beyond 256 updates it is overfitting the
frozen ~22k frames: 256 updates are ~12 passes, 1024 are ~47.

## Mapping notes (the bridge)

- **Row convention.** Both sources are checked identical (`convention_check.py`).
  - Row t pairs obs_t with the action a_t drawn at it; reward[t] is a_{t−1}'s reward (−2 per step,
    −1 or 0 on the goal row).
  - The terminal row has action 0 and the next row is `is_first`.
  - Regressing Δv_{t+1} on the actions gives 0.0029 for a_t and −0.0001 for a_{t−1} on both sources.
  - Our `to_world_batch` shifts to a_{t−1}, and their `observe` builds `prev_actions` internally from
    the same a_t column. So the bridge is a rename and needs no shift.
- **Ours → theirs.** `state = obs`, `action`, `reward`, `is_first`, `is_last`,
  `is_terminal = cont < .5` (ours `cont = 1 − is_success`), plus `can_self_imitate`, `pref_label`,
  `is_success`, `has_succeeded` copied. `cont` is refilled by their `preprocess`, and
  `unrelated_action` is zero-filled by their `populate_data`. `image` is absent (their encoder is
  state-only).
- **Theirs → ours.** `obs = state`, `cont = 1 − is_terminal`, and the same flags go into extras.
  `is_first[:, 0]` is forced True, as our `sample_windows` does. Only 2% of their windows start on an
  episode boundary; in their own code the latent carried over from the previous batch is used instead,
  and cells 3/4 keep that quirk. Our windows always start `is_first`.
- **Nothing had to be approximated.** Both sides record `can_self_imitate` (the imitation mask) and
  `pref_label` (ρ) by the same transcription-tested rule, so they are used as recorded. The shares
  differ a little by data: csi 0.106 vs 0.093, ρ < 0 on 0.25 vs 0.31 of rows.

## Caveats

- **Code includes history.** "Code" here is inseparable from each stage's 20k steps of own-data
  history, because every cell starts from its stage's parameters and optimizer state. What is shown
  is that the other side's data, 256–1024 updates of it, neither flips the sign nor stops each code's
  own drift. A from-scratch swap is not what was run.
- Their models ran in fp16 as shipped; the fp32 snapshot was not used.
- The learn state and optimizer moments came from each stage.
- The held states are drawn from the same frozen source as the training batches, so they are fixed,
  not held out.
- One seed and one stage per side.

Files: `cell_ours.py`, `cell_theirs.py`, `export_*.py`, `convention_check.py`, `table.py`,
`tables.txt` (every read, including N = 128/512/768 and the trained-actor rows), `cell_*.json`, and
`progress.log`.
