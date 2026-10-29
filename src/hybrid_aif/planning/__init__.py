"""Planning backends: enumeration, CEM, gradient, mixed, habit and sophisticated.

Planners are resolved through the registry in :mod:`hybrid_aif.planning.base`
(Systems own a planner *name*, never a planner); importing this package
registers the built-in ``"enumeration"``, ``"cem"``, ``"gradient"``, ``"mixed"``, ``"habit"``
and ``"sophisticated"`` backends.
Enumeration serves the discrete level-index action space, CEM and the gradient planner the
continuous one (``docs/design/amortization.md`` A9; the gradient planner descends G to the mode
of the policy posterior), ``mixed`` a space of discrete and continuous nodes together (it
enumerates the discrete policies and descends the continuous controls of each), and ``habit``
acts from the policy prior E without ranking anything by G — the three settings of A3's
search-habit dial.
``sophisticated`` scores an action by what the agent would do after each outcome
of the System's ``branch_on`` modalities.
"""

from hybrid_aif.planning.base import (
    Planner,
    PlannerContext,
    get_planner,
    planner_branches,
    planner_legal_depth,
    planner_needs_key,
    register_planner,
)
from hybrid_aif.planning.cem import (
    CEMResult,
    cem_plan,
    cem_planner,
    make_cem_planner,
    mean_action,
)
from hybrid_aif.planning.enumerate import (
    action_marginals,
    enumerate_policies,
    enumeration_planner,
    evaluate_policies,
    policy_posterior,
    sample_action,
)
from hybrid_aif.planning.gradient import (
    GradientResult,
    PlanPosterior,
    gradient_plan,
    gradient_planner,
    make_gradient_planner,
)
from hybrid_aif.planning.habit import habit_action, habit_planner
from hybrid_aif.planning.mixed import (
    MixedResult,
    control_steps,
    make_mixed_planner,
    mixed_plan,
    mixed_planner,
)
from hybrid_aif.planning.sophisticated import (
    BranchingTreeMemory,
    branching_tree_memory,
    make_sophisticated_planner,
    sophisticated_plan,
    sophisticated_planner,
)
from hybrid_aif.planning.tree import (
    evaluate_branching_tree,
    evaluate_policy_tree,
    evaluate_policy_tree_total,
    expand_level,
)

__all__ = [
    "BranchingTreeMemory",
    "CEMResult",
    "GradientResult",
    "MixedResult",
    "PlanPosterior",
    "Planner",
    "PlannerContext",
    "action_marginals",
    "cem_plan",
    "cem_planner",
    "control_steps",
    "enumerate_policies",
    "enumeration_planner",
    "branching_tree_memory",
    "evaluate_branching_tree",
    "evaluate_policies",
    "evaluate_policy_tree",
    "evaluate_policy_tree_total",
    "expand_level",
    "get_planner",
    "gradient_plan",
    "gradient_planner",
    "habit_action",
    "habit_planner",
    "make_cem_planner",
    "make_gradient_planner",
    "make_mixed_planner",
    "make_sophisticated_planner",
    "mean_action",
    "mixed_plan",
    "mixed_planner",
    "planner_branches",
    "planner_legal_depth",
    "planner_needs_key",
    "policy_posterior",
    "register_planner",
    "sample_action",
    "sophisticated_plan",
    "sophisticated_planner",
]
