"""Objective term trees: the metrics-bearing currency of the objective layer.

A term tree is a pytree ``term_name -> block_name -> Interval`` with per-policy
batch axes; it is what ``infer_policies`` returns as metrics. Leaves keep each
quantity's natural sign, and :func:`total` folds them into G (minimized) using each
term's registered sign; an unregistered term raises. Every :data:`INFOGAIN` leaf is
an information gain (higher is better, G subtracts it), a mutual information >= 0 up to
its interval's bounds. Block keys under ``infogain`` are a bare name for a state leaf
(the observing channel; the code variable on the amortized path) and slot-qualified
(``"likelihood:vision"``) for a parameter leaf. A disabled block is omitted from the
tree, never zeroed.

Design: docs/design/per-variable-infogain.md V3, V11
"""

import jax.tree_util as jtu

from hybrid_aif.interval import Interval

TermTree = dict[str, dict[str, Interval]]

#: The single epistemic term name (V3); every estimator's leaves land here.
INFOGAIN = "infogain"

_TERM_SIGNS: dict[str, float] = {
    "utility": -1.0,
    INFOGAIN: -1.0,
    "risk": 1.0,
    "ambiguity": 1.0,
}


def register_term(name: str, sign: float) -> None:
    """Register a term name with its sign as a contribution to G (+1 cost, -1 value)."""
    if sign not in (1.0, -1.0):
        raise ValueError(f"sign must be +1.0 or -1.0, got {sign}")
    _TERM_SIGNS[name] = sign


def term_sign(name: str) -> float:
    """A registered term's sign as a contribution to G (+1 cost, −1 value).

    Raises :class:`KeyError` for an unregistered term.
    """
    if name not in _TERM_SIGNS:
        raise KeyError(f"unregistered term {name!r}; use register_term(name, sign)")
    return _TERM_SIGNS[name]


def total(tree: TermTree, exclude: tuple[str, ...] = ()) -> Interval:
    """Fold a term tree into a single G Interval (expected free energy, minimized).

    Raises :class:`ValueError` when no leaves remain (e.g. a habit planner's ``{}``).
    """
    parts = []
    for term, blocks in tree.items():
        if term in exclude:
            continue
        if term not in _TERM_SIGNS:
            raise KeyError(f"unregistered term {term!r}; use register_term(name, sign)")
        parts.extend(_TERM_SIGNS[term] * iv for iv in blocks.values())
    if not parts:
        raise ValueError("term tree has no included leaves")
    out = parts[0]
    for iv in parts[1:]:
        out = out + iv
    return out


def merge(*trees: TermTree) -> TermTree:
    """Merge term trees block-wise, dropping empty terms.

    Trees sharing a term have their blocks merged; later trees win on a repeated
    ``(term, block)`` pair.
    """
    out: TermTree = {}
    for tree in trees:
        for term, blocks in tree.items():
            if blocks:
                out.setdefault(term, {}).update(blocks)
    return out


def map_leaves(fn, tree: TermTree) -> TermTree:
    """Apply ``fn`` to every Interval leaf, preserving the term/block structure."""
    return jtu.tree_map(fn, tree, is_leaf=lambda x: isinstance(x, Interval))
