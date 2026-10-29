"""The planner seam: context, protocol, and registry.

A System resolves its planner by name through the registry here, so it holds
only a string and stays a serializable pytree. A :class:`PlannerContext` is what
it offers a planner for one call: jit-traceable closures over the params plus
static action-space structure. It is built per call and never stored.

Design: docs/design/system.md "Planning"
"""

from collections.abc import Callable
from typing import Any, Literal, NamedTuple, Protocol, TypeAlias

from jax import Array

from hybrid_aif.controls import FLOAT, LEVEL, ControlLayout, ControlTable
from hybrid_aif.distributions import Categorical
from hybrid_aif.objectives.base import TermTree

Beliefs: TypeAlias = Any
"""Opaque belief pytree threaded through a planner's scan.

Planners never introspect beliefs; the System's closures define their meaning.
"""

TransitionFn = Callable[[Beliefs, Array], Beliefs]
StepTermsFn = Callable[[Beliefs, Array, Array], TermTree]
TerminalTermsFn = Callable[[Beliefs], TermTree]
BranchFn = Callable[[Beliefs], tuple[Array, Beliefs]]
UseBound = Literal["midpoint", "lower", "upper"]


class PlannerContext(NamedTuple):
    """Everything a System offers a planner for a single ``infer_policies`` call.

    ``transition_fn(qs, action) -> qs`` pushes beliefs through one control step;
    ``step_terms_fn(qs, action, t) -> TermTree`` scores post-transition beliefs;
    ``terminal_terms_fn(qs_final) -> TermTree`` (or ``None``) scores the horizon
    end. The posterior is ``q_pi ∝ softmax(-gamma * G + policy_prior)``.

    ``action`` is an int32 ``(K,)`` vector, one level per control node in node
    order (``controls`` is the System's :class:`~hybrid_aif.controls.ControlTable`,
    ``levels`` each node's level count), unless ``continuous_controls`` is set:
    then every node is continuous, the step is the float ``(action_dim,)`` flat vector, its
    box ``continuous_controls.bounds``, and every level count is ``1``. ``kinds`` gives each
    node's kind and ``sizes`` its level count (a discrete node) or width (a continuous one);
    ``boxes`` holds each continuous node's :class:`~hybrid_aif.controls.Box` (``None`` for a
    discrete node). A space mixing discrete and continuous nodes has no flat form: the planner
    ``"mixed"`` searches it and the other built-in planners refuse it (:func:`refuse_mixed`).
    ``policy_len`` is the horizon the
    planner plans to. ``num_controls`` and ``controlled`` read the same space in the form
    :func:`enumerate_policies` takes, so ``enumerate_policies(ctx.num_controls,
    ctx.policy_len, ctx.controlled)`` is the grid. ``key`` is ``None`` except for planners
    registered with ``needs_key=True``. ``habit_fn(beliefs, key) -> control step`` is
    set exactly when ``policy_prior`` is a policy module. ``branch_fn(beliefs) ->
    (p, beliefs_after)`` is set exactly when the System declares ``branch_on``:
    ``p`` has shape ``(n,)`` over the joint outcomes and ``beliefs_after`` a
    leading ``(n,)`` axis. ``legal_steps`` is set when the caller passed a per-step ``legal=`` to
    a planner registered with ``legal_depth=True``: a bool ``(L, A)`` over the ``A``
    first-step actions (``enumerate_policies(num_controls, 1, controlled)[:, 0]``), true where
    that action is legal at that depth; ``L`` is ``policy_len`` for a mask given per step and 1
    for a mask on the first step alone.
    """

    transition_fn: TransitionFn
    step_terms_fn: StepTermsFn
    terminal_terms_fn: TerminalTermsFn | None
    controls: ControlTable
    levels: tuple[int, ...]
    policy_len: int
    gamma: float
    policy_prior: Array | Categorical | object | None
    init_beliefs: Beliefs
    continuous_controls: ControlLayout | None = None
    key: Array | None = None
    habit_fn: Callable[[Beliefs, Array], Array] | None = None
    branch_fn: BranchFn | None = None
    kinds: tuple[str, ...] = ()
    sizes: tuple[int, ...] = ()
    boxes: tuple[object, ...] = ()
    legal_steps: Array | None = None

    @property
    def num_controls(self) -> tuple[int, ...]:
        """Each policy column's level count (one column per control node)."""
        return self.levels

    @property
    def controlled(self) -> tuple[bool, ...]:
        """Every policy column is planned over."""
        return (True,) * len(self.levels)


def refuse_mixed(ctx: PlannerContext, planner: str) -> None:
    """Raise, with a sentence, when ``ctx``'s control space mixes discrete and continuous
    nodes: the built-in planners other than ``"mixed"`` search one kind of space. The sentence
    from the ``"gradient"`` and ``"cem"`` planners names ``"mixed"``."""
    if LEVEL in ctx.kinds and FLOAT in ctx.kinds:
        names = ctx.controls.control_names
        discrete = [n for n, k in zip(names, ctx.kinds, strict=True) if k == LEVEL]
        continuous = [n for n, k in zip(names, ctx.kinds, strict=True) if k == FLOAT]
        lead = (
            f"the {planner!r} planner was asked to search a space mixing discrete control nodes "
            f"{discrete} and continuous ones {continuous}, "
        )
        if planner in ("gradient", "cem"):
            raise NotImplementedError(
                lead + 'and it searches one kind of space: plan it with the "mixed" planner, '
                "which enumerates the discrete nodes' policies and descends the continuous "
                "controls of each, or score a mixed plan with System.score_policies"
            )
        raise NotImplementedError(
            lead + 'and it does not search their product; the "mixed" planner does. Or score a '
            "mixed plan with System.score_policies, or plan each part with a System of its own"
        )


class Planner(Protocol):
    """Policy inference over a :class:`PlannerContext`.

    Returns ``(q_pi, tree, policies)``: the policy posterior, the term tree
    (per-policy leaves aligned with ``q_pi``), and the planner's own policy set,
    int32 ``(num_policies, policy_len, K)`` (one column per control node) or, over continuous
    actions, float ``(num_policies, policy_len, action_dim)``. Callers sample
    from these policies. ``metrics=False`` returns ``None`` for the tree and must
    leave ``q_pi`` unchanged up to float reassociation. A planner that builds no
    tree returns ``{}`` under ``metrics=True``.
    """

    def __call__(
        self, ctx: PlannerContext, use_bound: UseBound = "midpoint", metrics: bool = True
    ) -> tuple[Categorical, TermTree | None, Array]: ...


_PLANNERS: dict[str, Planner] = {}


def register_planner(
    name: str, *, needs_key: bool = False, branching: bool = False, legal_depth: bool = False
) -> Callable[[Planner], Planner]:
    """Decorator registering a planner under ``name`` for resolution by Systems.

    ``needs_key=True`` marks a stochastic planner, which reads
    :attr:`PlannerContext.key`. ``branching=True`` marks a closed-loop planner that scores each
    step on beliefs updated by the outcomes of earlier steps; it may be declared with a System's
    ``branch_on`` modalities, and when they are declared its context's step and terminal terms
    score a state target by its cross-entropy rather than its KL (design:
    per-variable-infogain.md V11 decisions 6 and 8). The marks are set on the function object,
    so they travel with a planner passed as a callable: ``System.infer_policies`` / ``act`` then
    require a ``plan_key`` for a stochastic planner and build the branching step terms for a
    branching one. ``legal_depth=True`` marks a planner that reads
    :attr:`PlannerContext.legal_steps`, masking the illegal actions at every depth of its plan;
    a ``legal=`` mask given per step is then handed to it, and the System applies only the
    first step's row to the policies the planner returns.
    """
    stochastic, branches, per_depth = needs_key, branching, legal_depth

    def register(fn: Planner) -> Planner:
        fn.needs_key = stochastic  # type: ignore[attr-defined]
        fn.branching = branches  # type: ignore[attr-defined]
        fn.legal_depth = per_depth  # type: ignore[attr-defined]
        _PLANNERS[name] = fn
        return fn

    return register


def planner_needs_key(planner: Planner) -> bool:
    """Whether ``planner`` declared itself stochastic (see :func:`register_planner`)."""
    return bool(getattr(planner, "needs_key", False))


def planner_branches(planner: Planner | str) -> bool:
    """Whether ``planner`` (a planner, or a registered name) declared itself branching (see
    :func:`register_planner`); an unregistered name is not."""
    if isinstance(planner, str):
        planner = _PLANNERS.get(planner)
    return bool(getattr(planner, "branching", False))


def planner_legal_depth(planner: Planner) -> bool:
    """Whether ``planner`` declared that it masks illegal actions at every depth (see
    :func:`register_planner`)."""
    return bool(getattr(planner, "legal_depth", False))


def get_planner(name: str) -> Planner:
    """Resolve a registered planner by name; unknown names fail loudly."""
    if name not in _PLANNERS:
        raise KeyError(
            f"unknown planner {name!r}; registered planners: {sorted(_PLANNERS)}. "
            "Use register_planner(name) to add one."
        )
    return _PLANNERS[name]
