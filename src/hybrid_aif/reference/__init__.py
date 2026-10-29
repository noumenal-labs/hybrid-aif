"""Second tier: reference content, below the engine.

1. **Pinned to external targets.** Everything here reproduces something outside the
   library (a paper's architecture, a benchmark environment, an oracle's numbers).
2. **May churn without an engine API break.**
3. **Never imported by the engine.** Modules under ``hybrid_aif`` outside this package must
   not import ``hybrid_aif.reference``; ``tests/test_layering.py`` enforces it.

Env and trainer abstractions typed over declarations are engine; concrete envs and trainers
pinned to a target live here. ``envs/`` holds targets shared by more than one agent; each
single-target replication has its own subtree (``raif/``).

Design: ``docs/design/system.md`` (tiers).
"""

#  Importing the tier registers the objective rows its subtrees define, so a System or a
#  description that names one builds after `import hybrid_aif.reference`.
from hybrid_aif.reference import raif as _raif  # noqa: E402, F401
