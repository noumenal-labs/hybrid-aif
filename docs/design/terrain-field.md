# A categorical field over a continuous state

Status: design, 2026-11-05. Decided in dialogue; the showcase rebuild (a mapping rover with solar
charging) is its first customer.

## What is wanted

A tabular factor whose parent is the cell a continuous factor falls in. The customer is a
terrain field: one Dirichlet row per cell of a grid over the rover's field, so the terrain under
the rover is drawn from the row of the cell it stands in, the rows are learned by the conjugate
count update as the rover sees terrain, and the expected information gain about the rows (the
novelty term) is the epistemic drive that makes mapping a planning objective. The same parent
serves known tables too: an energy band whose drain and charge depend on where the rover is.

In factor-graph terms the new node is a categorical child of a continuous parent with indicator
features: `c = cell(x)`, a deterministic discretisation. It is the conjugate special case of the
discrete-given-continuous family; multinomial logistic regression (MNLR, de-risk report section
1) is the smooth-basis sibling, needs the Polya-Gamma augmentation, and cannot express a patch of
one terrain in a field of another without being handed indicator features. The indicator node is
built first and is not subsumed by MNLR; MNLR reuses its plumbing later.

## The messages

**Down, from the position to the cell (F1).** The belief over the cell is the mass of the
position belief in each cell. For a mixture it is the weight-sum over components of each
component's mass. The mass of a Gaussian component in a rectangular cell is taken as the product
of its two marginal masses (differences of the normal CDF at the cell edges), the correlation
dropped; the first and last cell on each axis absorb the tails. This is a smooth function of the
component means, so the velocity planner's gradient sees the field through it, and an uncertain
rover writes fractional counts across the cells it might be in, a bimodal one to two places. No
bilinear read, no spatial prior: cells are independent Dirichlets, and smoothness, if ever
wanted, enters as a basis change (piecewise-linear nodes), not a coupling prior. The same indicator serves as the kernel at the source of a sibling-belief read, through which the slow tier's cell modality reads the rover's position (`nesting-and-coupling.md` N3).

**Up, from the cell to the position (F2, deferred).** The terrain seen under the rover is a
likelihood over position that is piecewise constant across the field: the field's probability of
that terrain at each cell. Multiplying a mixture by it reweights the components by the value at
each mean. Localisation by map is a bonus the messages give for free, never something a run
depends on; it is not built in the first round.

## Where it plugs in

The engine reads a tabular factor's parent beliefs in three places, and each takes a probability
vector (a `Categorical` or a raw vector): the transition prior (`_categorical_step` in
objectives/hybrid.py), the count statistics (`ConditionalDirichlet.stats` called from
system/faces/learn.py), and the transition novelty leaf (`transition_novelty_leaf` in
objectives/discrete.py, through `qs_prev`). Nothing today refuses a continuous factor in a tabular
factor's `B_deps`; it would fail on `.probs` downstream. The build:

- **Declaration.** `System(cells={"x": Grid(low, high, shape, axes)})`: the continuous factor's
  name, the grid's extent over the named axes of its state, and the number of cells per axis.
  A tabular factor may name that factor in `B_deps` only when a grid is declared for it; without
  one the refusal says so. The table's parent axis for that parent has one entry per cell, cells
  flattened row-major (`prod(shape)`), so every arity, learning and pricing rule sees one parent
  axis as it does for any categorical parent.
- **The read.** One function, `cell_mass(belief, grid) -> (cells,)`, over a `Gaussian` or
  `MixtureGaussian`; it is what the three sites receive as that parent's belief, wrapped as a
  `Categorical`. The sites learn nothing else: a cell parent is a categorical parent whose
  probability vector is computed rather than stored.
- **Order within a tick.** Categorical factors step first and read their parents' pre-transition
  beliefs; a mixture factor reads its discrete parents after they step. With the field as the
  terrain's parent and the terrain as the position's switching parent, this gives
  `terrain_t ~ field[cell(x_{t-1})]`: the terrain is read where the rover was one tick ago, and
  the dynamics then use it. The lag is one step of motion and is documented, not corrected.
  Learning uses the same pairing: the terrain posterior at `t` is counted into the cells of the
  position posterior at `t-1`.
- **Validation.** `validate_structure` keeps its rule that a continuous factor's parents are
  categorical; it gains the rule that a categorical factor's continuous parent must have a grid.
  The arity check counts the cell axis as one parent axis.
- **Novelty.** `transition:<field>` selects it as for any learnable transition; `qs_prev` for a
  cell parent is the cell mass of the predictive position belief at that step of the plan, so a
  plan's novelty is the expected information about the rows the rover will stand on.
- **Description length.** A cell parent's axis has no live-level mask (every cell is live); a
  learnable field is priced as rows times the tabular family cost plus the Dirichlet terms, as any
  table with a prior.
- **Faces.** `infer_states`, `update_empirical_prior`, `infer_policies`, `act` and `learn` all go
  through the three sites; `compiled()` needs nothing new. The fused tabular-only objective path
  is not reached by a system with a continuous factor.

## What the planner sees

The mass of a sharp belief deep inside one cell is flat in the mean: the normal CDF has
saturated at both edges, so the novelty term has an exactly zero gradient there, and a gradient
descent that starts from the zero plan stays put even when moving would earn several times the
information (a measured case: 0.29 against 0.05 nats, gradient 0.0 at the zero plan and about
0.7 at a plan of 0.3). Two consequences of the lag add to this: the last action of a plan moves
no read, so it has no novelty gradient, and with `policy_len=1` the term does not depend on the
action at all. The field's epistemic drive therefore needs `policy_len >= 2` and a planner that
does not start from the zero plan; the mixed and gradient planners' warm starts (`starts >= 2`,
a spread over the box) are that, and the showcase uses them. The predictive belief also widens
with process noise along a plan, so later steps see a smoother landscape than the first. No
tempering of the read is built; it is recorded here as the reason the planner settings are not
free.

## The conditional image preference (a second, smaller build)

The rover's preference for bright images is conditional on its energy band, so that mapping is not
coloured by a standing taste for light. Today a `ConditionalGaussian` target may condition on
control nodes only; a state parent is refused because the target is then a mixture at each step.
For the utility term (the expected log-density of the predicted observation under the target),
the parent-belief-weighted average over the levels is closed form and cheap, and that is what the
build adds: a Gaussian target with a categorical state parent, scored as the expected utility
over the parent's belief, on linear-Gaussian and neural modalities. The risk form (a divergence
to a mixture target) stays refused with the sentence it has now.

Each level is scored relative to its own peak, `E_q[log N_l(o)] - log N_l(m_l)`, which drops the
level's log-normaliser: with different scales per level the normaliser would differ by thousands
of nats and pay the agent for being in the level with the smallest scale, so a level's scale sets
the strength of its pull and never a bonus for the level itself (the unconditional utility keeps
its normaliser, a constant over plans). A target conditional on control nodes alone keeps each level's
normaliser (the durations convention, where a selected level's tightness is a bonus); averaging
over a state parent's levels drops them. The asymmetry is intended.

## What the showcase declares (level 1)

- `x`: position and velocity, a mode-aligned mixture switched by terrain; `v`: the velocity
  command (Box).
- `terrain`: categorical, parent `x` through a grid; its transition is the learnable field
  (a `ConditionalDirichlet` of shape `(3, cells)`), its initial prior uniform.
- `energy`: four bands, parents `(energy, x)` through the same grid; a known table built from the
  world's slope and illumination maps (drain on inclines, charge in light); a state target that
  prefers the full band.
- `fix`: a satellite position fix whose noise map is built from the height map's sky openness.
- `img`: the forward camera, a neural likelihood over `(x, terrain)`: ground texture marks the
  terrain, brightness is the illumination at the position (a smooth field the decoder learns);
  a Gaussian image target for bright images, conditional on `energy`.
- Objective terms: novelty on the field, the uncertainty term on `x`, the energy target, the
  conditional image utility. No exposure node.

Level 2 is the slow tier below.

## The slow tier (level 2)

Four 200-tick runs of level 1 showed the predicted stall: coverage plateaus at about a fifth of
the field by tick 150 and the rover never reaches the ice or the far rough ground, because novelty
is local (a flat gradient beyond the cells the plan can reach) while a mapping survey is a global
task. The slow tier is a survey scheduler with both drives: an epistemic drive over the field it
shares with the rover, and a pragmatic drive over energy. It is composed with the rover by `merge` as the rover's own slow tier, a nested component in the sense of `nesting-and-coupling.md` N2 with one identification (the field), and its slower clock is T9's `hold` under `run` (N4).

**Factors.** `visit`: the cell the rover stands on, a categorical over the grid's cells, observed through a `cell_obs` modality that reads the rover's position belief through the cell indicator, a sibling-belief read (`nesting-and-coupling.md` N3) realised by `mode` (the cell of largest mass) to reproduce the page's run and by `sample` thereafter; this is the coverage reading from the fast tier. Its transition is
conditioned on `(visit, option)`: under "survey sector k" a uniform distribution over sector k's
cells whatever the previous cell, under "charge" the identity, the rover stays where it is. That
last clause matters: a first draft gave charge a uniform distribution over the current sector,
which promised the same information as surveying it at no energy cost, and the scheduler charged
at every decision. `energy`: the four bands, a known table conditioned on `(energy, visit, option)`
built from the world's maps so that distance enters as energy cost: the expected drain from the
visited cell to sector k's centre plus about twelve ticks of surveying, in bands; under charge the
band rises with a probability set by the field-wide mean illumination, since the scheduler does
not know where light falls. `map`: the terrain at the visited cell, parent `visit`, whose table is
the parameter variable `field`, identified at `merge` with the rover's
(`docs/design/shared-parameters.md`, the 2026-11-05 amendment). This is the nest's one identification (`nesting-and-coupling.md` N2): the field is one home, because the rover's counts and the scheduler's novelty are one belief by design. The scheduler has no observation
of `map`'s child: it never counts into the field, and its view is run with the field's rate at
zero. Its novelty term (`transition:map`) is the expected information about the field from the
cells its plan visits, and that expectation is honoured by the rover counting on the fast clock.
This is the one place a reader could be tempted to add a second count update; do not, it would
double the evidence.

**Observations and options.** `cell_obs` and `gauge` (the energy band). Options: survey sector 1
to 4, charge; a slow step of twelve fast ticks. Objective: novelty on `map`, the energy target.
The energy target's logits are `(-12, -4, -0.3, 0)` on both tiers: a survey of a far sector must
be affordable at the full band, and a steeper ladder (`-12, -6, -2, 0`, the first draft) priced
every survey above its novelty, so the scheduler never left its first sector.

**The pooled reading.** A survey step pools about twelve fast readings of distinct cells, so the
scheduler declares `readings=(("map", "u_option"),)` with twelve readings under each survey
level and one under charge, where the rover is stationary and its readings repeat one cell
(`docs/design/step-durations.md`, T10). Without it a survey was credited one reading against its
whole energy cost and the scheduler never left its first sector; with it the two-level run
leaves at the fourth decision, surveys all four sectors and covers more than twice the field the
one-level ablation does by tick 400. With twelve readings under charge as well, the scheduler
charged at the full band at most of its charges, since a stationary step promised a survey's
information. The pooling is a planning-time count under T10, not an inference-time pooling of the readings dropped between a held mind's decisions (`nesting-and-coupling.md` N4).

**Downward.** The option reaches the rover as a held control node, supplied inside the nest by `ActionControl` (`nesting-and-coupling.md` N7), and conditions
two targets: a position target at the chosen sector's centre under a survey option (none under
charge), and the image preference on `(energy, option)`, so that charge pulls the rover to light
at every band while a survey only does so when the band is low.

**Lag and horizon.** The map is read where the survey stands one slow step earlier, as the terrain
is read where the rover stood one fast tick earlier (the order-within-a-tick rule above), so the
slow planner needs `policy_len >= 2` for the same reason the fast one does.

**How the runner drives the tiers.** The composite is one `merge` of the scheduler and the rover,
run as two `subsystem` views. Today the runner drives this by hand: the mission view decides every twelfth tick, the rover view runs between decisions with the option pinned on its control node, the rover's learned field is written back to the composite before each decision and the mission view is taken again from it, the scheduler never learns, and the scheduler's cell reading is computed in the runner from the rover view's position belief. The reading is not a wire under `couple`: the scheduler is the rover's own slow tier, inside one particle, so the reading is a sibling-belief read (`nesting-and-coupling.md` N3). At the build (N3, N4), the run is to be one `run` call on the composite with the scheduler wrapped in `hold` (built, and the clock); the write-back of the learned field, the re-taking of the mission view and the routing of the option into the rover view move into `run`'s composite form (N4, N9), and the runner's own loop, schedule and write-back go.

**The ablation.** The page shows the one-level run beside the two-level one: the same composite
with the option pinned to a single survey sector, which is level 1 as run.

## Out of scope here

MNLR; the upward message F2; a spatial prior over the field; a learnable energy table; cross-
factor continuous parents.

## The survey after 0.21.0 (2026-11-17)

**Plan learning on the rover.** `plan_learning` on the shipped rover is +0.15 coverage alone and
removes the scheduler's early lead, so the two-tier story becomes energy management (measured
2026-11-17, record kept outside the repo). The wall belongs in the vehicle step (a clip,
propagated unscented), after which the Gaussian-mixture preference is a constraint no longer
needed but still the carrier of the page's G bracket. Open; 0.22.0 shipped without deciding it.

**Multi-rover, step one (examples only, no library change).** Two rovers as two readers of one
`field`: `survey_scheduler.scheduler(..., rovers=2)` (per-rover factors, nodes and modalities
under `_1`/`_2`, one `field`, one `energy_table` counted by both, the durations on the first
option node), `survey_world.world_system(vehicles=2)` (two vehicles side by side, no
rover-to-rover physics), and `survey_run.settings_multi` with three arms: the two rovers merged
without a scheduler (stepped as one group), under one scheduler with joint options
(`JOINT_HORIZON` 3, 15,625 sequences, the field learned within the plan; the rovers stepped one
by one under `"sequential"`, each driven by its own option node and emitting its own cell
reading), and the union of two one-rover runs from mirrored starts. Five seeds by
`--replicas 5`, 800 ticks, coverage mean ± sd:

| Arm | 200 | 400 | 600 | 800 |
|-----|-----|-----|-----|-----|
| Two rovers, one scheduler (two-tier) | 0.543±0.026 | 0.735±0.026 | 0.791±0.030 | 0.818±0.030 |
| Two rovers, no scheduler | 0.457±0.013 | 0.703±0.043 | 0.835±0.050 | 0.866±0.058 |
| Union of two independent rovers | 0.377±0.031 | 0.603±0.028 | 0.738±0.026 | 0.805±0.050 |
| One rover | 0.259 | 0.452 | 0.579 | 0.684±0.049 |
| One rover, scheduler | 0.341 | 0.478 | 0.588 | 0.657±0.019 |

**The two-tier row is not what the default program now runs.** It was measured with the two
rovers stepped one by one under `"sequential"`, since `run` refused the group then: rover 2 read
the field rover 1 had just counted into. Since 2026-11-18 (`batched-run.md`, "R12 amended")
`run_multi` groups both arms by default, which makes the two rovers simultaneous among
themselves, both reading the field as the group's turn began. The default program will not
reproduce the two-tier numbers; `run_multi(..., group=False)` restores the stepping they were
measured under. The arm without a scheduler was a group already.

Sharing the field beats pooling two independent maps at every mark; the scheduler leads every
arm to tick 400 and ends level with the union, behind the shared map without it.

*Amended 2026-11-21 (Alex): the union now pools the two maps' counts per cell.* The union row
above counted a cell as mapped where either rover's own map was confident of it, which credits
the two rovers' maps with their contradictions as well as their agreements. `union_rows` now sums
the two fields' counts per cell under one prior (`field_counts`, read off the published uint8
means and counts), the map one shared field would hold from the same readings, and counts a cell
as mapped where that pooled map's top probability exceeds the threshold, as seen where the pooled
counts reach one. What the comparison then isolates is the gain from sharing the map *during* the
survey, not that plus the gain from pooling afterwards. Same five seeds, the pooled union:
0.373±0.033 | 0.596±0.029 | 0.733±0.027 | 0.797±0.049 (against 0.377 | 0.603 | 0.738 | 0.805
under the either-map reading; the pooling resolves a few cells the two maps read differently).
The page's replay seed is unchanged.

**Tuned on the way.** The first build lost late on every seed (0.798±0.058), both rovers held to
the sector above the charger, because the survey option predicted its readings as a Gaussian of
sd 0.66 about the sector's centre, so once the centres were read no option saw a sector's unread
edge cells. The readings are now uniform over the sector (the form "The slow tier" gives above),
and the travel to a sector is measured to its nearest cell, not its centre, since a 24-tick
option could not reach the far sector's centre (25 ticks at cruise). What remains of the pull:
six cells of that sector straddle a terrain boundary and their readings alternate (ice itself is
the easiest class to recognise, 98 percent of ticks), and the charger is next door.

**Spread and reproducibility.** One seed says nothing at this spread, and the same seed
unbatched is a different draw (float rounding of the batched program grown over 800 closed-loop
ticks: 0.508 against 0.625 for the one-rover scheduler; three further unbatched seeds fall within
0.04 of their replicas at every mark; `batched-run.md`, "Replica reproducibility, measured").

**Cost.** Wall: 894 s and 71 GB peak for five two-tier two-rover replicas with their traces,
659 s and 8.7 GB for five one-level, 396 s for the two single-seed two-tier runs.

**Fixture.** The one-rover declarations are as before; the two-level case of
`tests/data/survey_run.npz` is re-recorded (a different first option from the start cell), the
one-level case bitwise unchanged.

**Next.** `run` grouping copies driven by different option nodes, which landed 2026-11-18 as
"R12 amended"; a collision servo. Nested rounds were under review and are shelved
(`nested-rounds.md`, 2026-11-18).

## The charger, rethought (2026-11-18)

**Diagnosis**, on the five-seed runs of the table above. In the two-tier two-rover arm 79 charge
options over five seeds and two rovers; 73 started more than 2.0 from the charging place and
gained -0.02 of a level (the whole 24-tick option was the drive there, at 0.15 a tick), the six
that started near it gained +1.0. Each rover spent 24 percent of its ticks under charge, mostly
driving; end levels 1.3 to 2.9 of 4. The scheduler's energy transition was
`energy[next, band, option]`, with no dependence on where the option starts, learned at rate 1
from its band beliefs: it learned a charge as +0.06 of a level, the average of fruitless drives
and real charges, and could not price a far survey's travel either. Hence the loop "one far
survey, drive home, charge, repeat". The rovers without a scheduler never went to the charging
place (5 to 10 percent of their ticks within 1.0 of it): they parked about half their ticks in
cells of yield 0.3 to 0.4 (+0.015 of a level a tick) and ended at 3.4 to 3.95. In the world,
parked, `net_rate` is +0.005 a tick at the median yield (0.22), +0.039 at 0.7 and +0.057 at the
charging place (0.96); cruising at the median yield, -0.017. The sectors' yields (mean/max): 0.41/0.96
(sector 1, the charging place's), 0.18/0.29, 0.24/0.64, 0.26/0.77.

**Two designs.** A, the soft charge: under charge the rover's target leaves the position free
and holds the speed at rest (A': a looser rest, speed sd 0.5, so that it may creep to light within
its plan); the scheduler's visit table under charge is the identity (the rover stays on its cell)
and its charge readings one (the rover stands on one cell, as "The pooled reading" above has it);
the energy prior prices a charge parked at the field's median yield, a yield the scheduler states
since it does not know where the rover will stop. The bet: the image preference on
`(energy, option)` and the energy preference over the gauge draw a stopped rover to the light near
it, as they do for the rover alone. B, the energy table's cell dependence: the energy transition
reads `(energy, visit, option)`, the parameter variable `energy_table` is `[next, band, cell,
option]` (8 by 8 by 256 by 5), its prior per band, cell and option from the geometry: a survey
drives its travel to the sector's nearest cell at the cruising speed (0.17) and the rest of the
option at the measured survey speed (0.13, the mean true speed of the ticks a survey spent inside
its sector), both at the median yield, so a far survey drains up to 0.05 of a level more than one
from inside; a charge (in the form it is given) drives its trip to the charging place at 0.15 and
parks the rest on the charging place's yield, so a charge from a cell more than an option's trip
away is priced at -0.37 and one from the charging place's cell at +1.37. Learned at rate 1 as
before, the counts landing on the visited cells' rows: within a mission they barely move a row,
and the table's value is the travel cost its prior declares. The page reads the table per band and
option: the decision records hold it at the believed cell (the two rovers' visit beliefs averaged
in the joint record), the run's end the cell-marginal, and the counts summed over the cells.

**Measured**, seed 0, 800 ticks, unbatched, the two rovers grouped (coverage at 200/400/600/800;
per rover: share of ticks under charge, level gained per charge option, yield of the cell it
charged on, share of ticks within 1.0 of the charging place, min band, end level, stall ticks):

| Variant | Coverage | Under charge | Gain/option | Yield | Near site | Min band | End level | Stalls |
|---|---|---|---|---|---|---|---|---|
| Before (charge to the site) | 0.500/0.699/0.836/0.898 | 0.24, 0.24 | -0.12, +0.36 | 0.77, 0.79 | 0.27, 0.23 | 1, 0 | 1.50, 1.46 | 0, 21 |
| A (soft charge, parked) | 0.531/0.645/0.641/0.645 | 0.67, 0.42 | +0.03, -0.05 | 0.19, 0.09 | 0.05, 0.03 | 1, 0 | 1.43, 0.12 | 0, 446 |
| A' (soft charge, rest sd 0.5) | 0.535/0.664/0.656/0.656 | 0.55, 0.64 | -0.05, +0.02 | 0.14, 0.20 | 0.05, 0.03 | 0, 0 | 0.09, 0.67 | 179, 16 |
| A + B | 0.531/0.645/0.641/0.715 | 0.55, 0.36 | +0.02, -0.06 | 0.18, 0.09 | 0.05, 0.03 | 0, 0 | 0.40, 0.12 | 26, 446 |
| A' + B | 0.535/0.664/0.738/0.801 | 0.55, 0.52 | -0.04, +0.06 | 0.15, 0.31 | 0.05, 0.07 | 0, 0 | 0.17, 0.17 | 177, 108 |
| A' + B, charge priced at the best yield within a cell (0.38) | 0.535/0.766/0.809/0.793 | 0.34, 0.45 | -0.04, +0.19 | 0.22, 0.32 | 0.05, 0.05 | 0, 0 | 0.20, 1.81 | 420, 16 |
| Site charge + B, charge prior parked on the site's yield from every cell | 0.500/0.695/0.812/0.844 | 0.30, 0.30 | +0.16, +0.26 | 0.80, 0.85 | 0.32, 0.31 | 1, 2 | 1.59, 1.85 | 0, 0 |
| **Site charge + B, the trip priced (kept)** | 0.551/0.715/0.836/0.875 | 0.27, 0.27 | +0.59, +0.50 | 0.90, 0.86 | 0.35, 0.31 | 1, 2 | 1.91, 1.89 | 0, 0 |

**The soft charge lost.** A stopped rover stays where the option found it: it charged on cells of
yield 0.1 to 0.3, gained nothing an option, and every soft variant stalled one rover or both for
hundreds of ticks; neither the image preference nor the energy preference moved it to light within
its 12-tick plan, at rest or creeping. The rover alone finds light because nothing holds it still
and its plans keep it moving through the field; a held "stop here" removes exactly that. The
diagnosis's arithmetic (parking in a sector's light earns more than the trip) holds only where the
rover parks in the light, which the soft option does not bring about. A second prior for the
blind charge (the mean over cells of the best yield within one cell, 0.38) changed the timing,
not the outcome. The soft form is not kept as a keyword (its visit table, readings and prior
differ from the site form's in four places); its numbers are recorded here.

**Kept: the site charge with B**, the one variant without a stall, best at 800 among those. B
with the trip priced is what the diagnosis asked for: the scheduler now sees that a charge from
afar is the drive, so it charges from nearby. The charge options starting more than 2.0 from the
charging place fall from 92 to 51 percent, and the mean gain of a charge option rises from +0.06
to +0.44 of a level (five seeds, two rovers: 79 options either way; the far ones still gain -0.02,
the near ones +0.91). With the charge priced flat at the site's yield (the first B) the scheduler
still drove to the site from afar.

**Five seeds** by `--replicas 5`, 800 ticks, the rovers grouped (the default program; the
two-tier two-rover row above was measured with the rovers stepped one by one, so the change of
that row is the charger and the grouping together; the one-rover row is the charger alone):

| Arm | 200 | 400 | 600 | 800 |
|-----|-----|-----|-----|-----|
| Two rovers, one scheduler (two-tier), before (sequential) | 0.543±0.026 | 0.735±0.026 | 0.791±0.030 | 0.818±0.030 |
| Two rovers, one scheduler (two-tier), now (grouped) | 0.577±0.031 | 0.780±0.046 | 0.839±0.047 | 0.859±0.054 |
| Two rovers, no scheduler (unchanged) | 0.457±0.013 | 0.703±0.043 | 0.835±0.050 | 0.866±0.058 |
| Union of two independent rovers (unchanged) | 0.377±0.031 | 0.603±0.028 | 0.738±0.026 | 0.805±0.050 |
| One rover, scheduler, before | 0.341 | 0.478 | 0.588 | 0.657±0.019 |
| One rover, scheduler, now | 0.338±0.033 | 0.468±0.030 | 0.618±0.021 | 0.706±0.025 |

Under charge 24 percent of the two rovers' ticks and 29 percent of the one rover's; end levels
1.19 to 3.12 (two rovers) and 1.29 to 2.06 (one); one stall in ten rovers (38 ticks, seed 1,
rover 2), none for the one rover (11 stall ticks before). The scheduler now leads every arm to
tick 600 and ends level with the shared map without it (0.859 against 0.866, within a seed's
spread). Wall: 855 s and 68 GB peak for the five two-rover replicas, 714 s and 33 GB for the one.

**Still open.** Whether grouping alone moves the two-tier row (the "before" charger was not
re-run grouped over five seeds; on seed 0 grouped it reached 0.898 at 800 with a 21-tick stall,
the kept form 0.875 without); a charge that finds light near the rover would want a target over
the light itself (a yield map the rover reads), not only a rest; the survey speed is a measured
constant of the old charger's runs.

**Superseded in part, the same day: one charge option per sunny region** (below). What stays
true of the kept design: the energy table's cell axis and its geometric prior, the trip priced,
the site charge's target at rest. What changes: one charging place becomes one per sunny region,
and the two-rover scheduler plans with the factorised planner.

### One charge option per sunny region

**The rule.** A grid cell is sufficiently sunny when its mean solar yield (`cell_yields`) is at
least `SUNNY_YIELD` = 0.5; a sunny region is a 4-connected patch of such cells of at least
`MIN_REGION_CELLS` = 4 (`sunny_regions`); each region's charging place is its sunniest cell
(`charge_site_cells`, `charge_sites`), in order of that yield, so site 1 is the old
`charge_site()`. One charge option per site after the four surveys: `N_CHARGE` = 3, `N_OPTIONS`
= 7, the charge levels `CHARGE_LEVELS` (`is_charge`). Each charge option's target is its own
site at rest; its visit column is the path to its site then the site; its energy prior is its
own trip at the charge speed, then parked on its own site's yield (`charge_travel` is
`(cells, N_CHARGE)`). This is the declared-model form of the showcase's "charge on the nearest
sunny ground" (48a8a63), which patched the target from Python each tick.

**The regions found** on the 16 by 16 field:

| Site | Cell | Position | Yield | Region cells | Where |
|---|---|---|---|---|---|
| 1 | 86 | (1.06, 0.44) | 0.96 | 20 | the old charging place's cluster, sector 1 |
| 2 | 1 | (-0.81, -1.44) | 0.84 | 5 | bottom-left patch |
| 3 | 189 | (3.31, 3.06) | 0.77 | 17 | top-right cluster, sector 4 |

**The planner.** At seven options the two-rover scheduler's joint enumeration is 7^6 = 117,649
sequences (15,625 at five): a seed-0 run held 19 GB resident (5.2 GB at five options) and two
replicas 52 to 58 GB, so five replicas no longer fit a 128 GB host. The two-rover scheduler now
declares the factorised planner (`factorised-planning.md`): one posterior per rover over its own
7^3 = 343 sequences, three Gauss–Seidel sweeps, 3 · 2 · 343 = 2,058 sequence evaluations a round.
The one-rover scheduler keeps the enumeration planner (one node: the two agree). Measured, seed 0,
800 ticks, unbatched, two rovers grouped:

| Planner (seven options) | Coverage 200/400/600/800 | s/tick (compiled rounds) | Peak resident |
|---|---|---|---|
| Enumeration, 117,649 joint sequences | 0.559/0.750/0.848/0.887 | 0.29 (0.22) | 19.1 GB |
| Factorised, 2 · 343 sequences, 3 sweeps | 0.559/0.750/0.848/0.887 | 0.33 (0.24) | 3.3 GB |

Memory is below the five-option figure (5.2 GB) and time is within a seventh of the enumeration's.
The seed-0 runs are decision for decision the same (34 decisions; the scheduler's posteriors are
near one-hot at γ = 16). The first measurement of the factorised run (1.63 s/tick) was a batching
bug in the planner, the other nodes' marginals read once per sequence rather than once per step,
fixed in 8a63c8f (`factorised-planning.md`, "As built").

**Five seeds** by `--replicas 5`, 800 ticks, the rovers grouped; the two-rover replicas now fit
one batch (47.9 GB peak resident, 2,445 s wall before the planner fix; the enumeration
planner's five ran as batches of 2, 2 and 1, 52 to 58 GB each for two). Kept design: the single
charging place, enumeration.

| Arm | 200 | 400 | 600 | 800 |
|-----|-----|-----|-----|-----|
| Two rovers, kept design | 0.577±0.031 | 0.780±0.046 | 0.839±0.047 | 0.859±0.054 |
| Two rovers, per region, enumeration | 0.576±0.032 | 0.776±0.037 | 0.839±0.026 | 0.863±0.016 |
| Two rovers, per region, factorised (now) | 0.579±0.034 | 0.764±0.048 | 0.832±0.033 | 0.863±0.008 |
| One rover, kept design | 0.338±0.033 | 0.468±0.030 | 0.618±0.021 | 0.706±0.025 |
| One rover, per region (now) | 0.317±0.030 | 0.464±0.073 | 0.614±0.048 | 0.713±0.030 |

The charge behaviour over the five seeds (all charge options; "far": starting more than 2.0
from the chosen site):

| Arm | Charges | Far share | Gain all / far / near | Yield charged on | Under charge | End level | Stall ticks | Site 1 / 2 / 3 |
|---|---|---|---|---|---|---|---|---|
| Two rovers, kept | 79 | 0.51 | +0.44 / -0.02 / +0.91 | 0.86 | 0.24 | 2.11 (1.19-3.12) | 38 | 79 / - / - |
| Two rovers, per region, enumeration | 97 | 0.58 | +0.33 / -0.11 / +0.93 | 0.74 | 0.30 | 1.89 (0.95-2.50) | 0 | 47 / 2 / 48 |
| Two rovers, per region, factorised | 95 | 0.56 | +0.34 / -0.08 / +0.87 | 0.75 | 0.29 | 1.70 (0.95-2.38) | 30 | 43 / 4 / 48 |
| One rover, kept | 47 | 0.53 | +0.48 / -0.02 / +1.05 | 0.87 | 0.29 | 1.79 (1.29-2.06) | 0 | 47 / - / - |
| One rover, per region | 54 | 0.52 | +0.36 / -0.02 / +0.77 | 0.79 | 0.33 | 1.73 (1.05-2.63) | 0 | 30 / 5 / 19 |

**Read, honestly.** Coverage is a tie: every row is within a seed's spread of the kept design's at
every mark, the per-region design neither better nor worse at 800 (0.863 against 0.859 for two
rovers, 0.713 against 0.706 for one). What it changes is where the rovers charge: site 3, in
sector 4, takes half the two-rover charges (mostly from sectors 3 and 4), so a rover in the far
half of the field no longer drives to sector 1 to charge; site 2, a five-cell patch, is seldom
chosen. The far share of charges did not fall (0.51 to 0.58) and the gain per charge fell (+0.44
to +0.33), since site 3 yields 0.77 against site 1's 0.96. The two planners agree on seeds 1 and
2 decision for decision and part at near-ties on 0, 3 and 4 (rover 2's split 0.48/0.52 between
sectors 2 and 3 in the enumeration planner is 0/1 in the factorised one; the last sweep's movement
there 0.03, three sweeps not quite settled); the factorised five lose the per-region
enumeration's zero stalls (18 ticks, seed 0 rover 2; 12 ticks, seed 3 rover 1), about the kept
design's 38. Both stalls have one shape: a survey chosen at band 2 (level 1.07 and 1.19, the charge
0.37 and 2.4 nats behind it) drained the rover to band 1 on the far side of the field, and the
charge that followed, 2.9 and 2.8 from its site, stalled en route (13 and 11 ticks) and on arrival
before it took, where the enumeration's rover on the same seed had charged earlier (seed 0) or
entered the same survey-then-charge pair at band 3 (seed 3, level 1.78). The factorised five's
seed sd at 800 (0.008) is the smallest of the three, on five seeds.

**Density of sites, measured in scratch (the tree's rule unchanged).** The scratch rule replaces
`charge_site_cells` by greedy suppression: of the sunny cells (yield >= 0.5) in order of yield, a
cell is a site when no site already taken lies within `SITE_RADIUS` cells (Euclidean, cell units).
Radius 2 gives 7 sites (11 options): the three regions' sites plus cells 55 (0.73) and 119 (0.62)
and 100 (0.51) in region 1 and cell 143 (0.65) in region 3. Radius 1 gives 21 sites (25 options,
15,625 sequences a rover a sweep), runnable once the factorised planner scored each node as a
prefix tree (8a63c8f's successor commit; under the flat per-node scan the run made no progress in
1 h 45 at 36 GB). Two rovers, five seeds by `--replicas 5`, 800 ticks, factorised planner:

| Sites | Coverage 200 | 400 | 600 | 800 | Stall ticks | Charges | Far share | Gain all / far / near | Yield charged on | Under charge | Sites used (chosen) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Regions, 3 | 0.579±0.034 | 0.764±0.048 | 0.832±0.033 | 0.863±0.008 | 30 | 95 | 0.56 | +0.34 / -0.08 / +0.87 | 0.75 | 0.29 | 1: 43, 2: 4, 3: 48 |
| Radius 2, 7 | 0.561±0.020 | 0.752±0.046 | 0.819±0.035 | 0.847±0.038 | 0 | 166 | 0.70 | +0.12 / -0.07 / +0.59 | 0.58 | 0.51 | 1: 21, 2: 0, 3: 17, 4: 32, 5: 56, 6: 36, 7: 4 |
| Radius 1, 21 | 0.551±0.043 | 0.722±0.060 | 0.777±0.068 | 0.811±0.047 | 0 | 195 | 0.82 | +0.06 / -0.03 / +0.45 | 0.53 | 0.59 | 1: 4, 2: 36, 3: 14, 4: 2, 5: 1, 6: 2, 7: 1, 8: 3, 9: 0, 10: 4, 11: 3, 12: 17, 13: 42, 14: 8, 15: 18, 16: 6, 17: 0, 18: 24, 19: 0, 20: 9, 21: 1 |

Seed 0 alone, unbatched: 7 sites 0.41 s/tick (compiled 0.33), 5.1 GB resident, coverage 0.867
(3 sites: 0.33 (0.24), 3.3 GB, 0.887; 21 sites, prefix-tree planner: 0.29 (0.22), 7.9 GB, 0.805).
The five-replica batch peaked at 56.9 GB resident, 1,395 s wall (3 sites: 47.9 GB; 21 sites:
65.9 GB, 936 s wall).

Read: seven sites do not beat three. Coverage is lower at every mark, by less than the seed spread
(0.847 ± 0.038 against 0.863 ± 0.008 at 800), so it is a tie at best. The extra sites change the
charging, not the survey: the rovers charge almost twice as often (166 against 95), half their
ticks under a charge option, more of them from far (0.70), on poorer ground (yield 0.58 against
0.75), for a third of the gain a charge (+0.12 against +0.34). The stalls go (0 against 30 ticks;
the end level rises to 2.04 from 1.70), bought with the time spent charging. The new sites 5 and 6
take more charges than the regions' own; site 2 is never chosen. Over 3 / 7 / 21 sites the trend
is one way: coverage at 800 falls 0.863 / 0.847 / 0.811, the 21-site loss (0.052) larger than its
own seed sd (0.047), while charges rise 95 / 166 / 195, the far share 0.56 / 0.70 / 0.82, the time
under charge 0.29 / 0.51 / 0.59 and the gain a charge falls +0.34 / +0.12 / +0.06 (far charges net
-0.03), the 21-site rovers favouring the sites on the regions' inner edges (13, 2, 18, 15, 12) over the
regions' sunniest (1: 4, 5: 1). The tree keeps the regions rule.

**Fixture.** The one-rover declarations are as before and the one-level case of
`tests/data/survey_run.npz` is bitwise unchanged (all 177 arrays against HEAD's); the two-level
case is re-recorded with the per-region design: seven options, so 49 recorded sequences a
decision where there were 25, and a different first option from the start cell (the surveys no
longer tie: sector 1, from inside it, where a tie picked sector 2 before).
