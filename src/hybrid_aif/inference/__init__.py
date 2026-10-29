"""Inference algorithms over distribution objects.

Discrete mean-field FPI; linear-Gaussian conditioning and mixture rollouts.
"""

from hybrid_aif.inference.detail import InferenceDetail
from hybrid_aif.inference.discrete import fpi
from hybrid_aif.inference.linear_gaussian import condition, predict
from hybrid_aif.inference.mixture_filter import condition_mixture
from hybrid_aif.inference.mixture_rollout import propagate, rollout_gpb1, rollout_reduced

__all__ = [
    "InferenceDetail",
    "condition",
    "condition_mixture",
    "fpi",
    "predict",
    "propagate",
    "rollout_gpb1",
    "rollout_reduced",
]
