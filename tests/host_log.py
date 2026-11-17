"""A step's view of a run, logged on the host.

``run`` steps every round after the first inside one compiled scan (design batched-run.md R1, R8),
so a test step that appends what it is handed to a Python list would append tracers. ``logged``
appends through ``jax.debug.callback`` (ordered), which runs on the host each time the step runs,
on round 0's concrete values and inside the scan alike, and hands the callback NumPy arrays.
"""

import jax


def logged(log: list, value) -> None:
    """Append ``value`` (any pytree of arrays) to ``log`` each time the calling step runs."""
    jax.debug.callback(lambda v: log.append(v), value, ordered=True)


def noted(log: list, item) -> None:
    """Append ``item`` (any Python value, kept as it is) to ``log`` each time the calling step
    runs."""
    jax.debug.callback(lambda: log.append(item), ordered=True)
