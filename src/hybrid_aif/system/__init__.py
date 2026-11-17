"""System: the unit of composition and of model specification.

``from hybrid_aif.system import System, SystemParams`` is the import path. The package is split
across:

- :mod:`~hybrid_aif.system.core` — :class:`System`, :class:`SystemParams`, :class:`LearnState`,
  :class:`Subsystem`, :class:`Ports`, and the faces;
- :mod:`~hybrid_aif.system.faces` — the engines behind the inference, learning and planning faces,
  one function per face, each taking the System as its first argument;
- :mod:`~hybrid_aif.system.selectors` — the key grammar and its resolution against a System;
- :mod:`~hybrid_aif.system.validation` — declaration-time validation.

``learn``'s per-slot sufficient statistics live in :mod:`hybrid_aif.learning.conjugate_stats`.

Design: docs/design/system.md (the ``system`` package).
"""

from hybrid_aif.system.core import (
    ACTOR_KEY,
    UNSET,
    AmortizedScoring,
    Belief,
    LearnState,
    PartParams,
    Ports,
    Subsystem,
    System,
    SystemParams,
    WorkingCarry,
    _carry_prev,
)
from hybrid_aif.system.selectors import (
    _KEY_KINDS,
    _is_pattern,
    _param_blocks,
    _slot_learnable,
)

__all__ = [
    "ACTOR_KEY",
    "UNSET",
    "AmortizedScoring",
    "Belief",
    "LearnState",
    "PartParams",
    "Ports",
    "Subsystem",
    "System",
    "SystemParams",
    "WorkingCarry",
]

#   The private names below ``__all__`` are re-exported because
#   ``hybrid_aif.composition.merge`` reads the key grammar (``_KEY_KINDS``,
#   ``_is_pattern``) to namespace a composite's keys, and the declaration
#   predicates and ``_carry_prev`` are reached by name from the tests.
__all__ += ["_KEY_KINDS", "_carry_prev", "_is_pattern", "_param_blocks", "_slot_learnable"]
