"""Carrying learned posteriors across a rebuild of a model's params.

An agent that rebuilds its :class:`~hybrid_aif.system.SystemParams` every step (re-expressing
targets, noise maps or a linearisation) loses what learning put into them unless it copies
the posteriors across by hand. :func:`carry_learned` does the copy: it returns the freshly
built params with every learned posterior block taken from the previous params. It refuses,
rather than copy or drop silently, a block whose family, shape or dtype differs between the
two, a selected slot that holds a posterior on one side only, and a selection that would
split a growable factor's blocks, which growth and reduction keep in step.

A *block* is one posterior that :meth:`~hybrid_aif.system.System.learn` updates, addressed as
its rates are (``"likelihood:<modality>"``, ``"transition:<factor>"``,
``"initial_state_prior:<factor>"``, and ``"likelihood:<modality>:noise"`` for a
linear-Gaussian observation's noise): a Dirichlet or conditional Dirichlet, a matrix-normal
inverse-Wishart emission (one joint block, head and noise), a linear-Gaussian observation's
learnable noise, a trainable neural conditional. A block is carried whole, its anchors and
growth propensity included. Everything else comes from the fresh params: point values, and
a separate ``MatrixNormal`` head, which ``learn`` does not update (parameter-learning.md D8).

The carry is exact when what was learned means the same thing in the rebuilt params, as
counts over the same levels do. A posterior over a parameter the rebuild re-expresses (a head
in a frame that moves with the agent) is carried as it stands, in the old expression, and is
then wrong: keep such a parameter in a frame the rebuild does not move.

Design: docs/design/parameter-learning.md D9
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import equinox as eqx
import jax
import numpy as np

from hybrid_aif.distributions import ConditionalDirichlet, Dirichlet
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianObservationMNIW

_SLOTS = ("likelihood", "transition", "initial_state_prior")


def learnable_blocks(decl) -> tuple[str, ...]:
    """The posterior blocks ``learn`` updates in a declaration: ``("",)`` for a whole-slot
    posterior, ``("noise",)`` for a linear-Gaussian observation's learnable noise, ``()``
    otherwise."""
    if isinstance(decl, LinearGaussianObservationMNIW):
        return ("",)
    if isinstance(decl, LinearGaussianObservation):
        return ("noise",) if decl.learnable_noise else ()
    if isinstance(decl, Dirichlet | ConditionalDirichlet):
        return ("",)
    if getattr(decl, "learnable", False) is True:  # a trainable neural conditional
        return ("",)
    return ()


def _part(decl, block: str):
    return decl if block == "" else getattr(decl, block)


def _same_form(key: str, fresh, learned) -> None:
    if type(fresh) is not type(learned):
        raise ValueError(
            f"carry: {key} is a {type(fresh).__name__} in the fresh params and a "
            f"{type(learned).__name__} in the learned ones; a posterior carries into its own "
            "family only"
        )
    fresh_leaves, fresh_tree = jax.tree_util.tree_flatten(fresh)
    learned_leaves, learned_tree = jax.tree_util.tree_flatten(learned)

    def form(leaves):
        return [
            (np.shape(leaf), leaf.dtype if hasattr(leaf, "dtype") else np.result_type(leaf))
            for leaf in leaves
        ]

    if fresh_tree != learned_tree or form(fresh_leaves) != form(learned_leaves):
        raise ValueError(
            f"carry: {key} differs in form between the learned params and the fresh ones (its "
            "fields, their shapes or dtypes, or its static settings); a posterior carries only "
            "into a declaration of the same form"
        )


def _selected(key: str, keys: Iterable[str]) -> list[str]:
    return [k for k in keys if k == key or k.startswith(f"{key}:")]


def carry_learned(system, fresh, learned, slots: Iterable[str] | None = None, growable=()):
    """``fresh`` (positional params) with the learned posterior blocks taken from ``learned``:
    every block when ``slots`` is ``None``, else the blocks the keys name (``"<slot>:<name>"``
    names all of a declaration's blocks, ``"<slot>:<name>:noise"`` one; no wildcards).
    ``growable`` is the growable factors' indices, whose blocks move together."""
    if isinstance(slots, str):
        raise TypeError(f"carry: slots is a collection of keys, got the string {slots!r}")
    names = {
        "likelihood": tuple(system.modality_names),
        "transition": tuple(system._facs),
        "initial_state_prior": tuple(system._facs),
    }
    held: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    blocks: dict[str, tuple[str, int, str]] = {}
    for slot in _SLOTS:
        for i, name in enumerate(names[slot]):
            new, old = getattr(fresh, slot)[i], getattr(learned, slot)[i]
            on_fresh, on_learned = learnable_blocks(new), learnable_blocks(old)
            held[f"{slot}:{name}"] = (on_fresh, on_learned)
            for block in dict.fromkeys(on_fresh + on_learned):
                blocks[f"{slot}:{name}" + (f":{block}" if block else "")] = (slot, i, block)
    if slots is None:
        chosen = list(blocks)
    else:
        chosen = []
        for key in slots:
            if not isinstance(key, str) or len(key.split(":")) < 2 or "*" in key:
                raise ValueError(
                    f"carry: {key!r} is not a block key; name a declaration "
                    "('<slot>:<name>') or one of its blocks ('<slot>:<name>:noise'), without "
                    "wildcards"
                )
            matched = _selected(key, blocks)
            if not matched:
                raise ValueError(
                    f"carry: {key!r} names no learned posterior; the posterior blocks are "
                    f"{sorted(blocks)}"
                )
            chosen.extend(k for k in matched if k not in chosen)
        for f in growable:
            factor = names["transition"][f]
            group = [f"transition:{factor}", f"initial_state_prior:{factor}"] + [
                f"likelihood:{o}"
                for m, o in enumerate(names["likelihood"])
                if f in system.A_deps[m]
            ]
            members = [k for key in group for k in _selected(key, blocks)]
            picked = [k for k in members if k in chosen]
            if picked and len(picked) < len(members):
                raise ValueError(
                    f"carry: {factor!r} is growable, and growth and reduction keep its blocks "
                    f"in step; carry all of {members} or none, not {picked}"
                )
    containers = {slot: list(getattr(fresh, slot)) for slot in _SLOTS}
    for key in chosen:
        slot, i, block = blocks[key]
        on_fresh, on_learned = held[f"{slot}:{names[slot][i]}"]
        if on_fresh != on_learned:
            raise ValueError(
                f"carry: {slot}:{names[slot][i]} holds posteriors {on_learned or 'none'} in the "
                f"learned params and {on_fresh or 'none'} in the fresh ones; a rebuild that "
                "changes what is learned would drop or invent a posterior"
            )
        new, old = containers[slot][i], getattr(learned, slot)[i]
        _same_form(key, _part(new, block), _part(old, block))
        if block == "":
            containers[slot][i] = old
        else:
            containers[slot][i] = eqx.tree_at(
                lambda d, b=block: getattr(d, b), new, _part(old, block)
            )
    return dataclasses.replace(fresh, **{slot: tuple(containers[slot]) for slot in _SLOTS})
