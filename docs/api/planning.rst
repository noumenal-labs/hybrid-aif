Planning
========

A System holds the name of a planner, never a planner. Planners are resolved through a registry,
and the built-in ones are enumeration over a policy grid, the factorised planner (one policy
posterior per discrete control node, updated by mean-field sweeps), the cross-entropy method,
gradient descent, a mixed discrete and continuous search, a habit policy, and the sophisticated
planner that branches on observations. Each has a ``..._plan`` function, a ``..._planner`` object that fits the
registry and a ``make_...`` constructor that binds options.

The registry
------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.planning.Planner
   ~hybrid_aif.planning.PlannerContext
   ~hybrid_aif.planning.register_planner
   ~hybrid_aif.planning.get_planner
   ~hybrid_aif.planning.planner_needs_key
   ~hybrid_aif.planning.planner_branches
   ~hybrid_aif.planning.planner_legal_depth
   ~hybrid_aif.planning.planner_takes_init

Enumeration and policy scoring
------------------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.planning.enumeration_planner
   ~hybrid_aif.planning.enumerate_policies
   ~hybrid_aif.planning.evaluate_policies
   ~hybrid_aif.planning.evaluate_policy_tree
   ~hybrid_aif.planning.evaluate_policy_tree_total
   ~hybrid_aif.planning.expand_level
   ~hybrid_aif.planning.policy_posterior
   ~hybrid_aif.planning.action_marginals
   ~hybrid_aif.planning.sample_action
   ~hybrid_aif.planning.control_steps

Factorised
----------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.planning.factorised_planner
   ~hybrid_aif.planning.make_factorised_planner
   ~hybrid_aif.planning.factorised_plan
   ~hybrid_aif.planning.FactorisedPolicies

Cross-entropy and gradient
--------------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.planning.cem_planner
   ~hybrid_aif.planning.make_cem_planner
   ~hybrid_aif.planning.cem_plan
   ~hybrid_aif.planning.CEMResult
   ~hybrid_aif.planning.gradient_planner
   ~hybrid_aif.planning.make_gradient_planner
   ~hybrid_aif.planning.gradient_plan
   ~hybrid_aif.planning.GradientResult
   ~hybrid_aif.planning.PlanPosterior
   ~hybrid_aif.planning.mean_action

Mixed, habit and sophisticated
------------------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.planning.mixed_planner
   ~hybrid_aif.planning.make_mixed_planner
   ~hybrid_aif.planning.mixed_plan
   ~hybrid_aif.planning.MixedResult
   ~hybrid_aif.planning.habit_planner
   ~hybrid_aif.planning.habit_action
   ~hybrid_aif.planning.sophisticated_planner
   ~hybrid_aif.planning.make_sophisticated_planner
   ~hybrid_aif.planning.sophisticated_plan
   ~hybrid_aif.planning.evaluate_branching_tree
   ~hybrid_aif.planning.branching_tree_memory
   ~hybrid_aif.planning.BranchingTreeMemory

