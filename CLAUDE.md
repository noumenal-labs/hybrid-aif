I am very causal in my interactions with coding agents and don't make many demands. 
One thing I do ask is that by default I sign off on all git operations that change state (e.g. `git log` is OK without my agreement but not `git push`).

Check the README for the project description.
Thanks!

## Project conventions

- Layout: `src/hybrid_aif/` (src layout), tests in `tests/`.
- Environment: `uv venv && uv pip install -e ".[dev]"`; run tests with `.venv/bin/python -m pytest`.
- Style: ruff (lint + format), line length 100. Pure-functional JAX discipline: no side effects in jitted code, static shapes, `equinox.Module` for parameterized components.
- Design docs are authoritative: `docs/project-brief.md` (architecture, workstreams), `docs/de-risk-report.md` (algorithm choices and their citations). Consult before structural decisions.
- Subagents: never commit; leave the working tree for the main session to review, test, and stage.
- `examples/demo.ipynb` is a living exhibit and an **adoption gate**: re-execute it (nbconvert) whenever a change touches public API, and treat execution errors as regressions. (Adopted 2026-07-23 after an API rewrite broke the notebook while all tests stayed green.)