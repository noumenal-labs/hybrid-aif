# Changelog

Releases before 0.18.0 predate this repository. Their dates and summaries are
recorded here; the per-release capability and migration notes are in
`docs/capabilities.md`.

| version | date | summary |
|---|---|---|
| 0.23.1 | 2026-11-28 | `run` with `plan()` works again on a model with a modality over an action node (refused since 0.22.0); under `run` such a modality is absent in the first round, where it was drawn at level 0 |
| 0.23.0 | 2026-11-27 | Float precision follows the model: the package computes in the dtype of the arrays it is given, using the wider dtype for mixed models, and casts nothing to 32-bit on its own, so a 64-bit model runs in 64-bit under JAX's 64-bit mode (with that mode off, results are as in 0.22.0 except that a custom step's half-precision carry stays 32-bit); `cast` and `SystemParams.cast` move a model, or one part of it, to float32 or float64; a neural network computes in the dtype of its weights (its `dtype` property), with `dtype=` on the network constructors for the dtype the weights are drawn in; `run` settles the dtype of each value it carries before its loop |
| 0.22.0 | 2026-11-21 | `run` takes `batch=` (replicas by one program, per-replica keys, params and seeds, `record=` and `chunk=`, passes, spread over the devices), grouped leaves (a tuple key in `steps`, their partners' ports paired), one compiled round under `lax.scan` from the reset with `Trajectory.faults` and `.timing`, async under a batch, `rollout` retired; the `factorised` planner (one policy posterior per control node by mean-field sweeps, each node a prefix tree); the survey demo's second version on the project page, two rovers reading one field under a scheduler with a charge option per sunny region |
| 0.21.0 | 2026-11-16 | Switching nonlinear dynamics (a `NonlinearGaussianTransition` over discrete state parents, carried as a mixture), parameters learned within a plan (`plan_learning`: working Dirichlet counts along the rollout on both planning paths), and learnable process noise (`noise_prior` on both Gaussian transitions, a noise as a parameter variable shared by `R` and `Q`) |
| 0.20.0 | 2026-11-15 | One engine for composed Systems (`couple`, `merge` and `run` over a closed System), nodes named by kind, a neural likelihood's update in state space, state- and control-dependent transition noise, the API reference |
| 0.19.0 | 2026-11-04 | Recognition beside analytic factors, a neural likelihood inverted by linearisation, a levelled noise map, more of `merge` and `subsystem`, and a slow-tier mind that holds its action for an option's duration |
| 0.18.1 | 2026-10-31 | Two planner tests tolerate the rounding of a jax point release |
| 0.18.0 | 2026-10-22 | Planning improvements, including a planner for mixed control spaces |
| 0.17.0 | 2026-10-21 | A shared innovation noise across modes, fused leaves over a dimension selection, and the posterior chain with noise maps at 0.14.0 speed |
| 0.16.0 | 2026-10-21 | A modality over an action node in planning, presence per level of a control node, and a faster evaluation of the linear-Gaussian epistemic leaves |
| 0.15.0 | 2026-10-20 | Closed-loop scoring, per-decomposition epistemic leaves, conditional per-step targets, rounds of different durations |
| 0.14.0 | 2026-10-18 | Growing-mixture structure learning, `System.carry`, MNIW novelty, selectable dimensions, custom nonlinear Gaussian transitions |
| 0.13.0 | 2026-10-15 | Planning that expects to observe; the description length |
| 0.12.0 | 2026-10-13 | One namespace of nodes and modalities, with the migration note |
| 0.11.0 | 2026-10-05 | One control read by several factors |
| 0.10.0 | 2026-10-02 | Selection-relative information gain |
| 0.9.0 | 2026-09-22 | Two belief rollouts; the tier split done properly |
| 0.8.0 | 2026-09-20 | Describe, replay, constrain |
| 0.7.0 | 2026-08-28 | Scoring slot, `terms=`, policy prior, `Interval.derived`, habit planner, five amortized rows; reference tier; composition subpackage |
| 0.6.0 | 2026-08-18 | Continuous observation preferences: `obs_targets` on linear-Gaussian modalities, exact for Gaussian targets, with per-mode banks for switching models |
| 0.5.0 | 2026-08-18 | Nesting: one type, one builder, one view, one driver protocol with three tick modes |
| 0.4.0 | 2026-08-03 | Per-variable epistemic API: unified selector, wildcards, observation variables, the `infogain` term, the estimator registry |
| 0.3.0 | 2026-07-27 | Parameter learning through the faces |
| 0.2.0 | 2026-07-27 | The hybrid `System` milestone |
| 0.1.0 | 2026-07-21 | First milestone |
