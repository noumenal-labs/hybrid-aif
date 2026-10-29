# The gradient planner's step in a flat direction, and the iterate it returns

Status: SETTLED (Alex, 2026-10-22).

## Why

The gradient planner descends `G` over a continuous plan with Adam: each coordinate's mean
gradient divided by the root of its mean square plus an absolute guard of 1e-8, times a step
size set by the box width, decayed linearly to zero. The normalisation makes every coordinate move
about one step size per iteration whatever its gradient's magnitude, which is the point of Adam on
a well-scaled problem. In a direction where the gradient is rounding noise of the evaluation of
`G`, the same division produces a full-size step, so the descent wanders there, and which way it
wanders depends on which compiled program produced the noise. The planner returned the last
iterate, so a descent that wanders can end above its warm start. Both were seen on 2026-10-21: a
downstream agent's hundred-step descents ended above the warm start's `G` in every configuration
measured (the turn rate at a straight start plan, reachable only through a noise map's yaw axis,
has a gradient at rounding level), and two library builds that changed only the summation order
of the gradient moved that agent's plans and the sensor-detour demo's pinned score by more than
the demo's tolerance. The score of every plan was bitwise unchanged throughout.

## Decisions

**F1 — The guard is the dtype's precision, not a constant.** The denominator of Adam's update
is floored: `scale_t = max(sqrt(v_hat), floor_t) + 1e-8` with `floor_t = sqrt(machine epsilon of
the plan's dtype) · RMS(grad_t)` over all plan coordinates at that step, so a coordinate whose
gradient is below the precision of a float objective takes a step smaller in proportion to how far
below it is, and one with a real gradient keeps Adam's per-coordinate adaptivity. The floor is a
maximum rather than an added term (built 2026-10-22): an added term of 3.5e-4 of the RMS moves
every iterate by about 1e-3 in float32 even where no coordinate is near the floor, where the
maximum leaves the update bitwise the previous rule wherever the floor does not bind; the 1e-8
stays as the absolute term so an exactly zero gradient gives an exactly zero step. In float32 the
floor is about 3.5e-4 of the gradient's RMS. Nothing is fitted: the downstream agent and the
detour demo are acceptance, not calibration. The floor binds while a dominant coordinate carries a
real gradient; once the dominant coordinates converge the gradient's RMS shrinks with them and a
flat coordinate is floored no longer, so the floor shrinks the wandering (the detour spread halves)
and F2 is what bounds the result. The plain `optimizer="gradient"` step is untouched.

**F2 — The planner returns the best iterate it scored.** The descent's carry keeps the plan with
the lowest `G` seen, including the warm start and the final plan, and that plan is the result;
`trace` keeps every step's `G` as before. A result is then never worse than its warm start,
whatever the floor does.

*Note, 2026-10-22.* The result also carries `gradient_trace`, the root mean square of the gradient
over all plan coordinates before each step (the quantity the floor is defined against), and
`best_step`, the index of the iterate returned: `t` for the plan held before step `t` and `steps`
for the final plan.

**F3 — No setting.** Both are the planner's behaviour. The score of any plan is unchanged; plans
found in flat directions move, and a pinned plan or score from an unconverged descent is
re-recorded with its tolerance measured.

## Acceptance

- *Unchanged where it does not apply.* On models whose plan coordinates all carry gradients well
  above the floor (the lq_control example, the small_lg and payoff fixtures, the nonlinear
  transition tests) the descent's iterates are bitwise those of the previous rule, and so is the
  returned plan when the last iterate is the best.
- *Never above the warm start.* On every gradient-planner fixture in the test suite, the
  sensor-detour demo and the downstream agent's step-0 model, the returned `G` is at most the
  warm start's, up to the rounding between the term tree's total (which `G` reports) and the
  scalar score the descent minimises (a few float32 spacings: the downstream model returns its
  warm start with `G` 170.993683 against 170.993668). A descent whose final plan scores a
  non-finite `G` returns that plan and score, so a NaN descent still surfaces as NaN.
- *The flat direction.* On the downstream agent's step-0 model the turn-rate coordinates' first
  step at the straight start plan is 6e-4 to 9.7e-4 rad/s against 0.032 to 0.045 before (about 45
  times smaller; it is not zero: the step is `m_hat / (floor + 1e-8)`). The floored descent's last
  iterate still wanders above the warm start (171.069, against 171.085 before and 170.994 for the
  warm start); the returned plan is the warm start, by F2. The two downstream tests read a
  recorded model through an environment variable and skip in the default suite, so the suite does
  not run the motivating fixture.
- *Reproducibility.* The sensor-detour perturbed-start spread of `G` over a dozen starts is about
  half what it was (4.8e-4 against 9.6e-4 over seeds 1-12; over 24 seeds 4.8e-4 against 1.0e-3,
  and 6.0e-4 under fast-math), the plan still moves with the compiled program's summation order
  (for all 24 seeds, by a median 1.4e-4 against 1.7e-4), and under x64, where the plan stays
  float32, the spread does not improve (4.5e-4 against 4.3e-4). The float64 floor (1.5e-8 of the
  RMS) is not reached: the control box is cast to float32 (`controls.py`).
- *Quality against the previous rule.* Over a battery of fixtures and seeded warm starts the
  returned `G` is better than the previous rule's by more than 1e-5 in most cases that differ
  and worse by more than 1e-5 in a few unconverged ones (over 203 cases: better in 64, worse in
  6). The detour agent from the box midpoint is one: 400 steps end at -3.187774 against -3.187862
  (worse by 8.8e-5), and 10 000 steps at -3.556134 against -3.556169 (3.5e-5).
