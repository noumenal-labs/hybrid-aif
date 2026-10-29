"""Which modalities are observed at each plan step, and under which levels of a control node.

A :class:`Presence` holds a ``(horizon, M)`` bool mask over modalities, and for a modality whose
availability depends on a discrete control node, the availability per level of that node. The
planning closures resolve it at a step to one ``(M,)`` bool vector (:func:`presence_at`); the
closed-loop pass resolves it over the whole horizon from the recorded levels
(:func:`presence_from_levels`). ``None`` stands for every modality present throughout.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hybrid_aif.controls import ControlLayout, is_belief, node_entry
from hybrid_aif.interval import Interval
from hybrid_aif.objectives.base import TermTree


class LevelEntry(NamedTuple):
    """Modality ``modality`` (column) is present at a step exactly where ``levels`` (``(n,)``
    bool, one per level of control node ``node``, a column of the control table) is ``True``
    at the step's level of that node. ``modality_name`` and ``node_name`` label refusals."""

    modality: int
    node: int
    levels: Array
    modality_name: str = ""
    node_name: str = ""


class Presence(NamedTuple):
    """``mask`` is the ``(horizon, M)`` bool availability per step (a per-level modality's
    column is ignored); ``by_level`` lists the modalities read at the step's level of a node;
    ``absent`` is the set of modalities known when the presence was declared to be absent at
    every step (``None``: read it off a concrete ``mask``, see :func:`static_absent`)."""

    mask: Array
    by_level: tuple[LevelEntry, ...] = ()
    absent: frozenset[int] | None = None


def static_absent(presence: Presence | None) -> frozenset[int]:
    """The modalities (columns) absent at every plan step whatever the controls do: those
    declared absent as a Python ``False`` or a concrete all-``False`` array with no per-level
    entry (``presence.absent``), or, where that was not recorded, the concrete all-``False``
    columns of the mask. A traced availability has none, and a modality in the set takes no part
    in any per-step computation that its zeroed leaves would discard."""
    if presence is None:
        return frozenset()
    if presence.absent is not None:
        return presence.absent
    if isinstance(presence.mask, jax.core.Tracer):
        return frozenset()
    never = ~np.asarray(presence.mask, bool).any(axis=0)
    by_level = {entry.modality for entry in presence.by_level}
    return frozenset(int(m) for m in np.flatnonzero(never) if int(m) not in by_level)


def presence_at(
    presence: Presence, action: object, t: Array, layout: ControlLayout | None
) -> Array:
    """The ``(M,)`` bool availability at step ``t``: ``mask[t]`` with each per-level modality
    read from its node's level in ``action``. A node whose entry at this step is a probability
    vector over its levels raises."""
    row = presence.mask[t]
    for entry in presence.by_level:
        index = node_entry(action, entry.node, layout)
        if is_belief(index):
            raise ValueError(
                f"present[{entry.modality_name!r}] is read at each step's level of "
                f"{entry.node_name!r}, which at this step is a probability vector over its "
                "levels; a per-level availability needs a clamped level, as a float mask is "
                "refused (planning-with-observations.md D4). Plan with clamped steps, or pass a "
                "per-step bool array"
            )
        row = row.at[entry.modality].set(entry.levels[jnp.asarray(index, jnp.int32)])
    return row


def presence_from_levels(presence: Presence, levels: Array) -> Array:
    """The ``(H, M)`` bool availability over a horizon, each per-level modality read at the
    recorded ``levels`` (``(H, num_nodes)``, the step's level of every discrete node)."""
    mask = presence.mask
    for entry in presence.by_level:
        mask = mask.at[:, entry.modality].set(entry.levels[levels[:, entry.node]])
    return mask


def mask_absent(tree: TermTree, present_t: Array, modality_names: tuple[str, ...]) -> TermTree:
    """``tree`` with every leaf of a modality absent at this step zeroed: the leaves keyed by
    the modality's name (utility, risk, ambiguity, its channel's information gain) and by its
    likelihood slot (``likelihood:<name>``, a parameter's novelty), by a select, so a leaf that
    is not finite where the modality is read is an exact zero where it is absent. ``present_t``
    is the step's ``(M,)`` bool vector."""
    columns = {name: m for m, name in enumerate(modality_names)}
    columns.update({f"likelihood:{name}": m for m, name in enumerate(modality_names)})
    out: TermTree = {}
    for term, leaves in tree.items():
        masked = {}
        for key, iv in leaves.items():
            if key in columns:
                keep = present_t[columns[key]]
                iv = Interval(
                    jnp.where(keep, iv.lower, jnp.zeros_like(iv.lower)),
                    jnp.where(keep, iv.upper, jnp.zeros_like(iv.upper)),
                    iv.derived,
                )
            masked[key] = iv
        out[term] = masked
    return out
