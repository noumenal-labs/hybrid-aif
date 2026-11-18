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
