# Changelog

Releases before 0.18.0 predate this repository. Their dates and summaries are
recorded here; the per-release capability and migration notes are in
`docs/capabilities.md`.

| version | date | summary |
|---|---|---|
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
