# Experiments

Records of what we *observed* while chasing an external target, one folder per
experiment. This tier exists because `docs/design` holds what we **decided**
and why — adjudications that should stay small, stable and readable — and a
replication generates a great deal of material that is none of those things:
divergence audits, per-key disposition tables, oracle baselines, measurements
that settled an argument once and will never be read again.

Adopted 2026-09-07 (Alex): "not in design docs, and probably way too detailed
for a history entry."

## What goes here

Anything whose subject is **one target**: the divergence register and its
ledgers, config dispositions, the oracle's own numbers and how it was produced,
measurements that decided a specific question, and the experiment's own notes on
method. It is append-mostly; a superseded entry is marked superseded rather than
deleted, for completeness.

## What does not

- **Adjudications about the library.** Those live in `docs/design`. See the
  graduation rule below.
- **Code.** The replication's code is a source tier: `src/hybrid_aif/reference/`
  (importable, tested, and imported by nothing in the engine — see
  `tests/test_layering.py`). Fixtures and vendored external configs stay there
  too, because tests import them. Prose about the experiment comes here.
- **Run artifacts.** Training outputs live in `runs/` and are not tracked.

## The graduation rule

One-way, parallel to the reference tier's import rule: an observation about 
*this target* lives here; when it becomes a rule about *the library*, it 
graduates to `docs/design` as an adjudication and this tier points at it. 
So a divergence is recorded here, but "a term's sign is registered once
and for all" is a design decision and lives there.

Nothing in `docs/design` should need to read this tier to be understood.

## Experiments

- **`raif-mountain-car/`** — replicating R-AIF's state-based Mountain Car
  results with architecture parity (project goal 2). Target repo `raif-paper`
  at fork commit `9b1864e8fab9d1becd6dd248f2c590d06ff22848`; doctrine is *the
  oracle is the code*, and where their paper and their code disagree, the code
  wins. Closed 2026-10-10 (entry 113): eight seeds a side match the oracle,
  `acceptance-report.md`. Start at `divergence-register.md`; the code is
  `src/hybrid_aif/reference/raif/`.
- **`description-bakeoff/`** — the acceptance test of the description 0.3 reader
  (`design/model-description-0.3.md`, Q17): four models in prose, each written by Sonnet
  and Haiku agents with the format document as their only reference, in 0.2 and in both
  0.3 spellings; every document written, the reader's verdicts, and a reading test. Start
  at `README.md`.
