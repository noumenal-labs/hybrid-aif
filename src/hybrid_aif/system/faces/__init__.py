"""Each module implements one face of ``System``: it reads the System's declarations and
parameters and assembles calls into the engine packages (``hybrid_aif.inference``,
``hybrid_aif.learning``, ``hybrid_aif.planning``, ``hybrid_aif.objectives``), which hold the
algorithms and never import a System. The public methods on ``System`` keep the signatures and
docstrings and delegate to the functions here; the private engines that other modules reach by
name on a System keep a one-line delegate there.

- :mod:`~hybrid_aif.system.faces.infer` — ``infer_states``, ``predict_beliefs``,
  ``update_empirical_prior``;
- :mod:`~hybrid_aif.system.faces.learn` — ``learn``, ``carry``, ``reduce``, ``grow``,
  ``init_learn_state``;
- :mod:`~hybrid_aif.system.faces.plan` — ``planner_context``, ``infer_policies``,
  ``score_policies``, ``act``.
"""
