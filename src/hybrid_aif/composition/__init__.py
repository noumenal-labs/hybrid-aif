"""Compositional active inference: Systems merged into one particle, Systems coupled through their
ports, and ``run``, which steps the closed System either makes.

- :mod:`~hybrid_aif.composition.merge`: :func:`merge` makes several Systems one particle: a
  namespaced composite whose components share latents through bindings, of three kinds: a
  :class:`Link` (a driver's action node into a control node, a relayed state, a modality's
  parents, a belief read, a terminal target), an :class:`Identify` (two parameter variables made
  one, or an out-port joined to an in-port) and a :class:`Clamp` (a control node held at a
  constant); a component bound only by blanket-shaped bindings, belief reads and action-node
  links, is nested; :func:`subsystem` derives one mind's System from the composite and
  :func:`write_back` returns a view's learned parameter variables to it.
- :mod:`~hybrid_aif.composition.joint`: :func:`joint_planner` builds the one planning problem of
  a merged or coupled System, or of one of its components, as a plain ``(System, SystemParams)``:
  each control node driven by an action node inside the node is collapsed into the driver's, and
  the leaves' configuration, habit, precision, durations and scoring are composed, what cannot be
  joined refused.
- :mod:`~hybrid_aif.composition.couple`: :func:`couple` joins components through their ports
  into one ``(System, SystemParams)``: an :class:`Identify` of an out-port (an action node or an
  emission) with an in-port (a modality or a control node) and a :class:`Clamp`, each component
  kept as a leaf with its own configuration.
- :mod:`~hybrid_aif.composition.runner`: :func:`run` steps a closed System (a merge or a coupling
  whose every modality is fed by an identification) for a number of rounds and records a
  :class:`Trajectory`. Each leaf, or a non-leaf node planned jointly, takes a :data:`Step`:
  :func:`plan` for a planning mind (its carry a :class:`MindCarry`), :func:`sample` for a
  process, or a function of the same shape. :func:`with_mirror` couples a System to its own
  mirror, which closes it. A node whose view declares durations decides once per option,
  ``clocks=`` converting a duration into rounds.
- :mod:`~hybrid_aif.composition.black_box`: :func:`black_box` declares an external process (an
  environment, a hand-written world) by its ports and their types alone, a component
  :func:`couple` binds like any other and :func:`run` steps with a custom step;
  :func:`env_step` is that step for a positional environment.
- :mod:`~hybrid_aif.composition.component_mirror`: :func:`~hybrid_aif.system.mirror.mirror` with
  ``per_component=True``, the merge of one mirror per component of a merged or coupled System,
  a state factor of another component that a transition or a modality reads fed to the reader by
  an identity emission of its owner.

Design: docs/design/nesting-systems.md; docs/design/couple-and-run.md; nesting-and-coupling.md
N1, N2.
"""

from hybrid_aif.composition.black_box import black_box, env_step, is_black_box
from hybrid_aif.composition.couple import couple, with_mirror
from hybrid_aif.composition.joint import joint_planner
from hybrid_aif.composition.merge import (
    Clamp,
    Identify,
    Link,
    driven_nodes,
    merge,
    namespaced_key,
    nested,
    resolved_epistemic,
    subsystem,
    subsystem_states,
    write_back,
)
from hybrid_aif.composition.runner import (
    Maybe,
    MindCarry,
    Step,
    Trace,
    Trajectory,
    plan,
    run,
    sample,
)

__all__ = [
    "Clamp",
    "Identify",
    "Link",
    "Maybe",
    "MindCarry",
    "Step",
    "Trace",
    "Trajectory",
    "black_box",
    "couple",
    "driven_nodes",
    "env_step",
    "is_black_box",
    "joint_planner",
    "merge",
    "namespaced_key",
    "nested",
    "plan",
    "resolved_epistemic",
    "run",
    "sample",
    "subsystem",
    "subsystem_states",
    "with_mirror",
    "write_back",
]
