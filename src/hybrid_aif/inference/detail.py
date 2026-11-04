"""Structured record of what one ``infer_states`` sweep computed internally.

``infer_states`` always returns ``(beliefs, detail)``; ``detail=False`` puts ``None`` in the
second slot, and the beliefs are bitwise identical either way. The record surfaces quantities
inference would otherwise discard, such as a switching LG modality's grown mixture before
``pad_mixture`` loses the mode pairing.

Design: docs/design/parameter-learning.md O3
"""

from collections.abc import Mapping

import equinox as eqx
from jax import Array

#: A per-node field: a mapping keyed by name at the faces, a tuple in node order inside.
PerNode = Mapping[str, object] | tuple


class InferenceDetail(eqx.Module):
    """Per-modality internals of one sweep; ``None`` where inapplicable.

    The first three fields have one entry per modality, ``unsupported`` one per factor. As
    :meth:`System.infer_states <hybrid_aif.system.System.infer_states>` returns it (and
    :meth:`~hybrid_aif.system.System.learn` takes it) each field is a mapping keyed by modality or
    factor name; inside, the same record holds tuples in declaration order
    (``System.from_positional("detail", ...)`` / ``to_positional`` convert). Structure is static
    for a fixed model, so the object is jit-safe.
    """

    pre_reduce: PerNode  # of MixtureGaussian | None
    #   Switching LG modality: the grown K·M mixture BEFORE pad/reduce —
    #   components pair (mode config × prior component), the mode-conditional
    #   posterior the reduced belief no longer carries.
    mode_log_evidence: PerNode  # of Array | None
    #   Switching LG: per-mode-config integrated log-evidence (the boundary
    #   message read pre-reduce). Mode-aligned mixture LG: the per-component
    #   observation log-likelihoods that formed the mode message.
    log_evidence: PerNode  # of Array | None
    #   Any LG modality: integrated log p(y) under the beliefs the tick holds before this
    #   modality's message (after the tabular stage) — ``condition``'s second return; for a
    #   switching or mode-aligned mixture channel, the total over mode configurations under
    #   that mode belief.
    unsupported: PerNode = ()  # of Array | None
    #   Per FACTOR: a 0-d bool, True when no level was compatible with every present
    #   tabular modality and the prior (``inference.discrete.unsupported``; Z1/Z2). It does
    #   not attribute the contradiction: under mean-field, co-parents' flags fire too
    #   (exact attribution: ``description.check_joint_support``). ``None`` for a continuous
    #   factor. Read off the tabular FPI stage, before any LG mode message.
    free_energy: Array | None = None
    #   The tick's variational free energy, a scalar: the mean-field posteriors' KL from their
    #   priors less the expected tabular log-likelihoods, less every LG or neural modality's
    #   integrated log-evidence (a neural modality's read linearised at the belief the factor holds
    #   when the modality is reached, so the bound is on the linearised model: the
    #   ``approximate`` status of the planner's intervals),
    #   plus ``- log Z + E_q[log m]`` for each recognition message ``m`` and ``- E_q[log p(o | s)]``
    #   under the final belief for each modality a recognition entry covers. ``-free_energy``
    #   bounds the tick's log evidence from below and equals it where every modality is LG or
    #   one categorical factor carries every tabular one
    #   (``inference.discrete.mean_field_free_energy``). ``None`` on the amortized path.
    recognition_log_evidence: Mapping[str, Array] = eqx.field(default_factory=dict)
    #   Analytic path with recognition entries: per entry, keyed ``"recognition:<variable>"``, the
    #   log normaliser ``log Z`` of its message applied to the belief the tick holds before it
    #   (``condition``'s log evidence for a Gaussian or mixture factor, ``log sum_k q_k r_k`` for a
    #   categorical one); 0.0 for an entry absent at the tick. It is not a modality evidence term:
    #   it enters ``free_energy`` through ``- log Z + E_q[log m]``.
    covered_terms: Mapping[str, Array] = eqx.field(default_factory=dict)
    #   Analytic path with recognition entries: per modality an entry covers, keyed by modality
    #   name at the faces and inside alike, the ``- E_q[log p(o | s)]`` term it adds to
    #   ``free_energy`` under the tick's final belief (a linear-Gaussian, emission or tabular
    #   modality in closed form, a neural one to second order at each component's mean); 0.0
    #   for a covered modality absent at the tick. An uncovered modality has no entry. Each is a
    #   cost (the negative of an expected log-likelihood), so it adds to ``free_energy`` where an
    #   evidence term subtracts.
