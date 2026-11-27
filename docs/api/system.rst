System and SystemParams
=======================

A ``System`` is the unit of model specification: one declaration of the state factors, the
observation modalities, the control nodes and the objective. ``SystemParams`` is the numeric state
that goes with it: the transition, likelihood and prior of each variable, the preferences, and the
parameters of anything learned. Everything else in the library takes or returns a System. A System
may be discrete, continuous or mixed. The entries a ``SystemParams`` holds are described under
:doc:`components`.

The declaration
---------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.System
   ~hybrid_aif.SystemParams

Float precision
---------------

A model computes in the float dtype of the arrays it is given, and arrays of different dtypes
compute in the wider one where they meet. ``cast`` moves a ``SystemParams``, or any tree of
arrays, to float32 or float64 (the latter under JAX's 64-bit mode); ``SystemParams.cast`` is the
same call on the params.

A model whose parts are in different float dtypes (a mixed model, which needs JAX's 64-bit mode)
costs more than the same model in one dtype. It computes in the wider dtype from where its parts
meet. ``run`` finds the dtype of each value it carries by tracing its first round before the
loop, twice or more on every call, where a model in one dtype usually needs no such trace. In a
hand-written loop the world's state changes dtype at its first step, so ``env.step`` compiles
twice. Casting the model to one dtype avoids all three.

A neural network (``hybrid_aif.distributions.neural``) is the exception: it computes in the
dtype of its weights, its ``dtype`` property, and casts its inputs to that dtype, so a wider
input is rounded to the network's precision and its outputs are in that dtype whatever feeds
it. ``dtype=`` on a network's constructor only chooses the dtype the weights are drawn in; to
change a network's precision afterwards, ``cast`` it, alone or as one part of a model
(``params.cast(jnp.float64, where=lambda p: p.recognition["encoder"])``).

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.cast

Structure of a built System
---------------------------

These are carried on a System and read back from it, more often than they are written.

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ~hybrid_aif.system.LearnState
   ~hybrid_aif.system.Ports
   ~hybrid_aif.system.Subsystem
   ~hybrid_aif.system.PartParams

