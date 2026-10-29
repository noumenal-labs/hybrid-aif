"""The ``infer_states`` detail channel (design: parameter-learning.md O3).

``infer_states`` always returns ``(beliefs, detail)``; ``detail=True`` opts
into an :class:`InferenceDetail` carrying what the sweep computed and would
otherwise discard — above all the pre-reduction K·M mixture on a switching
modality, whose (mode × component) pairing ``pad_mixture`` destroys and the
LG sufficient statistic (D8) needs. Contract points covered:

- beliefs are bitwise identical with the flag on and off (the detail is a
  record, never a code path);
- the switching modality's ``pre_reduce`` really is the grown K·M mixture,
  its ``mode_log_evidence`` matches the brute-force per-mode evidence, and
  ``log_evidence`` is their total under the mode prior;
- a plain (non-switching) LG modality reports its integrated evidence and
  nothing else;
- the discrete path returns an all-``None`` detail (uniform contract, no
  tabular internals to report yet).
"""

import jax.numpy as jnp
import numpy as np

from hybrid_aif.distributions import Categorical, ConditionalCategorical, Gaussian
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianTransition
from hybrid_aif.inference import InferenceDetail
from hybrid_aif.inference.detail import InferenceDetail as _same
from hybrid_aif.system import System, SystemParams
from tests.test_switching_observation import (
    MODE_PRIOR,
    X_PRIOR,
    brute_force,
    make_switch_params,
    make_switch_system,
)


def _bitwise_equal_beliefs(a, b) -> bool:
    import jax

    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return all(
        np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)
        for x, y in zip(la, lb, strict=True)
    )


def test_export_is_single_class():
    assert InferenceDetail is _same


def test_beliefs_bitwise_identical_with_and_without_detail():
    system, params = make_switch_system(), make_switch_params()
    y = system.from_positional("observations", (jnp.asarray([0.8]),))
    priors = system.from_positional("beliefs", (MODE_PRIOR, X_PRIOR))
    off, det_off = system.infer_states(params, y, priors=priors)
    on, det_on = system.infer_states(params, y, priors=priors, detail=True)
    assert det_off is None
    assert isinstance(det_on, InferenceDetail)
    assert _bitwise_equal_beliefs(off, on)


def test_switching_detail_contents_match_brute_force():
    system, params = make_switch_system(), make_switch_params()
    y_val = 0.8
    _, det = system.infer_states(
        params,
        system.from_positional("observations", (jnp.asarray([y_val]),)),
        priors=system.from_positional("beliefs", (MODE_PRIOR, X_PRIOR)),
        detail=True,
    )
    a = params.likelihood["y"]
    *_, per_mode_ev_ref = brute_force(
        X_PRIOR,
        np.asarray(MODE_PRIOR.probs),
        np.broadcast_to(np.asarray(a.H), (2, 1, 1)),
        np.broadcast_to(np.asarray(a.c), (2, 1)),
        np.asarray(a.R_chol),
        y_val,
    )
    grown = det.pre_reduce["y"]
    assert grown is not None and grown.num_components == X_PRIOR.num_components * 2
    np.testing.assert_allclose(
        np.exp(np.asarray(det.mode_log_evidence["y"])), per_mode_ev_ref, rtol=1e-5
    )
    total = np.log(np.dot(np.asarray(MODE_PRIOR.probs), per_mode_ev_ref))
    np.testing.assert_allclose(np.asarray(det.log_evidence["y"]), total, rtol=1e-5)


def test_plain_lg_detail_reports_log_evidence_only():
    system = System(
        A_deps=((0,),),
        B_deps=((),),
        controlled=(False,),
        policy_len=1,
    )
    params = SystemParams(
        likelihood={
            "modality_0": LinearGaussianObservation(
                H=jnp.asarray([[1.0]]), c=jnp.zeros(1), noise=jnp.asarray([[0.5]])
            ),
        },
        transition={
            "factor_0": LinearGaussianTransition(
                F=jnp.asarray([[1.0]]),
                innovations=__import__(
                    "hybrid_aif.distributions", fromlist=["ConditionalGaussian"]
                ).ConditionalGaussian(
                    loc=jnp.zeros((1, 1)), chol=jnp.asarray([[[0.3]]]), parent_ndims=1
                ),
            ),
        },
        initial_state_prior={"factor_0": Gaussian(loc=jnp.zeros(1), chol=jnp.asarray([[1.0]]))},
    )
    _, det = system.infer_states(params, {"modality_0": jnp.asarray([0.4])}, detail=True)
    assert det.pre_reduce == {"modality_0": None}
    assert det.mode_log_evidence == {"modality_0": None}
    # exact scalar: y ~ N(0, prior_var + R^2) at y=0.4
    var = 1.0 + 0.25
    ref = -0.5 * (0.4**2 / var + np.log(2 * np.pi * var))
    np.testing.assert_allclose(np.asarray(det.log_evidence["modality_0"]), ref, rtol=1e-5)


def test_discrete_path_detail_is_all_none_entries():
    system = System(
        A_deps=((0,),),
        B_deps=((),),
        controlled=(True,),
        policy_len=1,
    )
    table = jnp.log(jnp.asarray([[0.8, 0.3], [0.2, 0.7]]))
    params = SystemParams(
        likelihood={"modality_0": ConditionalCategorical(log_table=table, parent_ndims=1)},
        transition={
            "factor_0": ConditionalCategorical(
                log_table=jnp.log(jnp.full((2, 2, 2), 0.5)), parent_ndims=2
            ),
        },
        initial_state_prior={"factor_0": Categorical.from_probs(jnp.asarray([0.5, 0.5]))},
    )
    obs = {"modality_0": jnp.asarray(0, jnp.int32)}
    beliefs, det = system.infer_states(params, obs, detail=True)
    assert isinstance(det, InferenceDetail)
    assert det.pre_reduce == {"modality_0": None}
    assert det.mode_log_evidence == {"modality_0": None}
    assert det.log_evidence == {"modality_0": None}
    _, det_off = system.infer_states(params, obs)
    assert det_off is None
