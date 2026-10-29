"""The ``Learnable`` protocol: what a parameter-distribution declaration provides.

A slot holds either a point value or a distribution over its parameters;
``isinstance(decl, Learnable)`` is the capability test, resolved at trace time. Whether a call
advances the posterior, and how fast, is a per-call rate on ``System.learn``, never a field here.

- ``expected_log_params()``: ``E[log params]``, what inference averages over (FPI's digamma
  table); distinct from ``ConditionalCategorical.expected_log(parent_beliefs)``.
- ``point()``: the posterior-mean declaration; its return type is documented per class.
- ``stats(...)`` + ``absorb(stats, lr, decay)``: one tick's sufficient statistic, then its
  (optionally decayed) absorption; ``stats`` signatures are per family.
- ``novelty()``: parameter-info-gain weights (exact ``spm_wnorm``, better-when-larger).
- ``sample(key)``: a parameter draw from the posterior, for ``as_env(params, param_key=...)``.

Design: docs/design/parameter-learning.md D3 (the protocol)
"""

from typing import Protocol, runtime_checkable

from jax import Array


@runtime_checkable
class Learnable(Protocol):
    """Structural capability protocol (see module docstring), satisfied by method presence.

    New conjugate families join by implementing the members; there is no registry and no flag.
    """

    def expected_log_params(self) -> Array: ...

    def point(self): ...  # per-slot return type; documented on each class

    def stats(self, *args, **kwargs) -> Array: ...  # per-family signature

    def absorb(self, stats: Array, lr: float | Array = 1.0, decay: float | Array = 1.0): ...

    def novelty(self) -> Array: ...

    def sample(self, key: Array): ...
