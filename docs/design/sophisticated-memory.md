# The sophisticated planner's memory, and legality below the root

Status: SETTLED (Alex, 2026-10-23).

## Why

The sophisticated planner evaluates the exact recursion of Friston et al. 2021 breadth first,
one level per horizon step inside one jitted program, and materialises the whole deepest level in
one vmap: `(A·m)^(H−1)·A` children for `A` first-step actions and `m` branched outcomes. A
measurement on 2026-10-23 found memory linear in that count (about 100 to 290 bytes per deepest
child beyond the belief, mostly unfused step-term intermediates), 576 MB at eight actions, three
outcomes and horizon five, 59 GB with a second branched modality. Earlier levels are freed as the
next is built; nothing is recomputed. Separately, `legal=` masks only the root: the tree expands
and scores illegal actions at every deeper level, and their values enter the deeper softmaxes.

## Decisions

**SM1 — Chunked subtrees, exact up to rounding.** Below a split depth `d` (default 1, directly
under the root), each node's subtree is evaluated with `jax.lax.map(..., batch_size=k)` instead
of the whole level in one vmap; within a subtree the evaluation is the existing one. The result
equals whole-level evaluation up to float rounding, a few ulps, the order of eager against `jit`
(a batched contraction can accumulate in an order that depends on the batch size); it is often
identical, and only speed and memory change. (Amended 2026-10-23: the decision first read
"bit-identical"; a 2026-10-23 prototype on identity-like tables had measured 0.)

**SM2 — An opt-in memory budget, never automatic.** The planner evaluates the whole deepest
level, as before, unless told otherwise. `make_sophisticated_planner(split_depth=, batch_size=)`
does exactly what it says (`split_depth` is 1 when only `batch_size` is given).
`make_sophisticated_planner(memory_budget=)` (bytes) chooses them: at trace time it estimates the
deepest level's memory from declared sizes (node count times, per child, six beliefs plus the
measured per-node overhead at the top of its range, so the estimate errs high). If the whole
level fits the budget it is evaluated whole; if not, the planner chunks with the largest batch
that fits. With no `split_depth` given, the split deepens from 1 until one subtree fits (and,
where none fits at any depth, takes batch 1 at the depth whose estimate is least, the budget
being a best effort); a given `split_depth` is kept, and a given `batch_size` overrides the
budget's batch. The batch is the largest divisor of the node count that fits, so `lax.map` has
no remainder to evaluate beside its batches. Everything is computed from static shapes at trace
time: no recompilation, and no read of the device. `branching_tree_memory(system, params,
beliefs, memory_budget=None)` returns the estimated whole-level bytes and, given a budget, the
depth, batch and estimate it would choose, evaluating nothing; a caller runs it once at setup
to see the tree's memory and pins the result with `make_sophisticated_planner`. Because results
agree to rounding at every setting, the per-node overhead is an engineering constant, not a
tuned behaviour.

*Note (2026-10-23).* An automatic default from the device's reported free memory (GPU) was built
and removed before release: it made the batch, and so the last bits of the results, depend on
what else was allocated when the program was traced.

**SM3 — Legality at every depth.** A per-step `legal=` mask (`(policy_len, levels)` per control
node), refused by the sophisticated planner today, masks illegal actions out of the softmax at
each depth: their `G` is replaced by `+∞` before `σ(−γG)`, so they carry no weight in `V`. A
root-only mask applies at the root, as now. The padded tree still expands illegal actions (static
shapes), so memory is unchanged; values no longer count them.

## Acceptance

- `G`, `q_pi` and selected action equal, to float rounding, between whole-level evaluation and chunked
  evaluation at several `(split_depth, batch_size)` pairs, on the T-maze with one and with two
  `branch_on` modalities and at horizons 1–4, and under `jit`.
- A tree whose whole-level evaluation exceeds a given small `memory_budget` runs under it, and its
  compiled temp memory (`memory_analysis()`) is below the budget.
- The choice: with no option the program equals today's; with a budget the batch chosen is the
  largest divisor that fits, the split deepening as needed; forced options override it;
  `branching_tree_memory` reports the same numbers and evaluates nothing.
- SM3: with every action legal the result is bit-identical to today's; with a per-step mask an
  illegal action's subtree contributes exactly zero weight at that depth (compare against a model
  where the action is removed from the level set at that depth, or against a hand computation on
  a horizon-2 tree); a mask that leaves no legal action at a depth is refused with a sentence.
- No retrace across calls that change only values or the `legal=` mask's values.

## Build notes (2026-10-23)

Built as decided: `planning/tree.py` (`evaluate_branching_tree(legal=, split_depth=,
batch_size=)`, `branching_tree_bytes`, `chunk_choice`, `BRANCHING_BELIEF_FACTOR = 6`,
`BRANCHING_BYTES_PER_NODE = 290`), `planning/sophisticated.py` (`make_sophisticated_planner(
memory_budget=, split_depth=, batch_size=)`, `chunk_plan`, `branching_tree_memory`),
`planning/enumerate.py` (`legal_steps`), `planning/base.py` (`PlannerContext.legal_steps`, the
`legal_depth` mark of `register_planner`), `system/faces/plan.py` (a keyed `legal=` becomes the
per-step mask for a planner marked `legal_depth`; the root row also forms the root indicator; a
mask on the first step alone reaches the planner as one row, and the tree then takes no mask).
Tests: `tests/test_sophisticated_memory.py`.

**Estimate against measurement** (CPU, float32, compiled temp memory of the jitted
`infer_policies`, whole level; the estimate is `branching_tree_bytes`):

| model | estimate | measured | ratio |
|---|---|---|---|
| T-maze, two modalities, H=4 | 209 MB | 56 MB | 3.7 |
| random tables, H=5 | 137 MB | 27 MB | 5.1 |
| A=8, O=3, H=5, one modality | 1,402 MB | 576 MB | 2.4 |
| A=5, O=3, H=5, two modalities | 9,425 MB | 2,959 MB | 3.2 |
| Gaussian state D=16, H=4 | 71 MB | 54 MB | 1.32 |
| Gaussian state D=32, H=4 | 266 MB | 214 MB | 1.25 |
| Gaussian state D=64, H=3 | 87 MB | 57 MB | 1.53 |

A Gaussian belief's per-child overhead measured 3.9 to 4.9 beliefs, which the first constant
(a fixed 290 bytes per child) understated about fourfold; the factor of 6 covers it with a
fifth to spare, and overestimates categorical beliefs, whose overhead is the fixed part.

**Budgets met** (compiled temp memory of the planner with `memory_budget=`):

| case | budget | split depth, batch | temp |
|---|---|---|---|
| A=8, O=3, H=5, one modality (whole: 576 MB) | 64 MB | 1, 1 | 22 MB |
| | 128 MB | 1, 2 | 45 MB |
| | 256 MB | 1, 4 | 89 MB |
| A=5, O=3, H=5, two modalities (whole: 2,959 MB) | 1,024 MB | 1, 3 | 249 MB |
| | 256 MB | 1, 1 | 78 MB |
| | 64 MB | 2, 9 | 16 MB |
| | 16 MB | 2, 3 | 5 MB |
| Gaussian D=16, 32; H=4 | whole/2, /4, /8 | 1, 4 / 1, 2 / 1, 1 | all under budget |

Batches are divisors of the node count (24 nodes at depth 1 in the first case, 45 in the
T-maze with two modalities): with a batch of 23 of 45, `lax.map` evaluated the remainder of 22
beside its two batches and held 50.6 MB against 27.3 MB at 22.

**Deepening is not monotone.** One subtree shrinks as the split deepens, but the nodes held at
the split grow, so a deeper split can hold more (Gaussian D=64, H=3: 5.3 MB at depth 1, 8.5 MB at
depth 2, batch 1). Where no depth fits, the planner takes batch 1 at the depth whose estimate is
least, not the deepest.

**Rounding, SM1.** On the T-maze at horizons 3 and 4 chunked and whole-level `G` differ in the
last bits (at most 4 float32 ulps, 3.8e-6 on `G` near 11; `q_pi` logits up to 6e-5 absolute,
after the softmax's scale of 16 and normalisation); at horizons 1 and 2, and on identity-like
tables, they are identical. Beliefs from a batch of one node differ from the whole level's at
1e-23: XLA's CPU contraction of a batched `einsum` accumulates in an order that depends on the
batch size, so the difference is in the transition, not in the fold. The whole-level program
already differs between eager and `jit` by the same amount (1.9e-6). In float64, with the
fixture cast to float64, chunked and whole-level agree to at most 2 ulps (3.6e-15); an earlier
reading of 1.5e-8 was a float32 leak in the T-maze fixture, whose tables stay float32 under x64.
The selected action and `argmax q_pi` were the same in every case tried. Unmasked, with no
budget, the jaxpr and the eager result equal the previous code's exactly.

A traced `legal=` mask is not checked (as before); one leaving no legal action at a depth gives
a `NaN` `q_pi`.
