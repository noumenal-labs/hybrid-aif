Distributions
=============

Beliefs, priors and parameter posteriors are distribution objects: immutable ``equinox`` modules
that hold their parameters as arrays. The conditional families give a child distribution for every
configuration of its parents, either by table or by a small network. Exponential-family members
expose their natural and mean parameters through one shared contract.

Base types
----------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.Distribution
   ~hybrid_aif.ExponentialFamily
   ~hybrid_aif.Learnable
   ~hybrid_aif.Conditional

Distributions over states
-------------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.Categorical
   ~hybrid_aif.JointCategorical
   ~hybrid_aif.Gaussian
   ~hybrid_aif.MixtureGaussian
   ~hybrid_aif.TruncatedGaussian
   ~hybrid_aif.Dirichlet

Conditional families
--------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.ConditionalCategorical
   ~hybrid_aif.ConditionalGaussian
   ~hybrid_aif.ConditionalDirichlet
   ~hybrid_aif.TabularObservation
   ~hybrid_aif.TabularTransition
   ~hybrid_aif.TabularPrior
   ~hybrid_aif.MLPCategorical
   ~hybrid_aif.MLPConditional
   ~hybrid_aif.MLPPredictor
   ~hybrid_aif.PredictorHead
   ~hybrid_aif.LearnedVector

Parameter priors and further families
-------------------------------------

These are exported by ``hybrid_aif.distributions`` only.

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.distributions.MatrixNormal
   ~hybrid_aif.distributions.InverseGamma
   ~hybrid_aif.distributions.InverseWishart
   ~hybrid_aif.distributions.LowRankMixture
   ~hybrid_aif.distributions.GRUCell

