"""Which declarations learn, and how: the predicates every learning path and selector reads."""

from hybrid_aif.distributions import Learnable
from hybrid_aif.factors import GradientDeclaration, LGObservation


def gradient_learnable(decl) -> bool:
    """Whether a declaration learns by gradient rather than by conjugacy.

    True for a non-frozen member of :data:`~hybrid_aif.factors.GradientDeclaration`; these are the
    slots that need a :class:`~hybrid_aif.system.LearnState` entry.
    """
    return isinstance(decl, GradientDeclaration) and decl.learnable


def slot_learnable(decl) -> bool:
    """Whether a slot is learnable, by conjugate update or by gradient.

    True for a declaration that is itself a parameter posterior (tabular families), an LG head whose
    ``noise`` field holds one, or a non-frozen neural conditional. A learnable slot is
    rate-addressable and appears in the epistemic namespace.
    """
    return (
        isinstance(decl, Learnable)
        or (isinstance(decl, LGObservation) and (decl.learnable_noise or decl.learnable_head))
        or gradient_learnable(decl)
    )
