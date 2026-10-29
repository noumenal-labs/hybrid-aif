"""Compositional active inference: Systems merged into one, Systems coupled by wires, and the
clock that runs them.

- :mod:`~hybrid_aif.composition.merge` — :func:`merge` makes several Systems one namespaced
  composite, identifying latents across them (bindings: control-node identification or relay,
  a modality's parents, target emission, boundary prior); :func:`subsystem` derives one mind's
  System from the composite.
- :mod:`~hybrid_aif.composition.couple` — :func:`couple` joins minds and processes through
  their blankets: a :class:`Wire` from an action node to a process input or a mind's modality,
  or from a process observation to a mind's modality, typed and checked into a :class:`Wiring`.
- :mod:`~hybrid_aif.composition.driver` — :func:`run` closes the loop under a declared tick
  order, over a :class:`Wiring` (one :class:`MindRecord` or :class:`ProcessRecord` per
  participant) or over a merged composite, each mind acting through its own view (a
  :class:`Trajectory`).

Design: docs/design/nesting-systems.md.
"""

from hybrid_aif.composition.couple import Wire, Wiring, couple
from hybrid_aif.composition.driver import (
    MindCarry,
    MindRecord,
    ProcessRecord,
    Trajectory,
    agent_respond,
    ancestral_respond,
    run,
)
from hybrid_aif.composition.merge import (
    ActionControl,
    BoundaryPrior,
    ObservationLink,
    TargetEmission,
    merge,
    namespaced_key,
    resolved_epistemic,
    subsystem,
    subsystem_factors,
)

__all__ = [
    "ActionControl",
    "BoundaryPrior",
    "MindCarry",
    "MindRecord",
    "ObservationLink",
    "ProcessRecord",
    "TargetEmission",
    "Trajectory",
    "Wire",
    "Wiring",
    "agent_respond",
    "ancestral_respond",
    "couple",
    "merge",
    "namespaced_key",
    "resolved_epistemic",
    "run",
    "subsystem",
    "subsystem_factors",
]
