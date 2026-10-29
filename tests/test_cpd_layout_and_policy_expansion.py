"""Acceptance-fix 5: layout-safe CPD constructor.

``ConditionalCategorical.from_parent_leading`` takes the natural
``(*parent_shape, num_levels)`` layout and stores the child-first table —
removing the silent-wrongness ``log_table=W.T`` transpose trap. Pure sugar:
round-trips against the manual transpose bitwise.

"""

import jax.numpy as jnp
import numpy as np

from hybrid_aif.distributions import Categorical, ConditionalCategorical
from tests.test_hybrid_switching_system import ACTION_MODE_LOGW

# ------------------------------------------------------------ from_parent_leading


def test_single_parent_round_trip_vs_manual_transpose():
    got = ConditionalCategorical.from_parent_leading(ACTION_MODE_LOGW, parent_ndims=1)
    manual = ConditionalCategorical(log_table=ACTION_MODE_LOGW.T, parent_ndims=1)
    assert jnp.array_equal(got.log_table, manual.log_table)
    assert got.parent_shape == (3,) and got.num_levels == 2
    for u in range(3):
        assert jnp.array_equal(
            got.condition_on((u,)).log_probs, manual.condition_on((u,)).log_probs
        )


def test_multi_parent_with_batch_axes():
    rng = np.random.default_rng(0)
    natural = jnp.asarray(rng.standard_normal((4, 3, 2, 5)))  # batch (4,), parents (3, 2), levels 5
    got = ConditionalCategorical.from_parent_leading(natural, parent_ndims=2)
    manual = ConditionalCategorical(log_table=jnp.moveaxis(natural, -1, -3), parent_ndims=2)
    assert jnp.array_equal(got.log_table, manual.log_table)
    assert got.batch_shape == (4,) and got.parent_shape == (3, 2) and got.num_levels == 5
    # semantics: the child distribution at (parent_0=1, parent_1=0) is natural[:, 1, 0, :]
    assert jnp.array_equal(got.condition_on((1, 0)).logits, natural[:, 1, 0, :])


def test_average_agrees_with_manual_layout():
    natural = jnp.log(jnp.asarray([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]]))
    got = ConditionalCategorical.from_parent_leading(natural, parent_ndims=1)
    manual = ConditionalCategorical(log_table=natural.T, parent_ndims=1)
    belief = (Categorical(logits=jnp.asarray([0.3, -0.2, 1.1])),)
    np.testing.assert_allclose(
        np.asarray(got.average(belief).probs), np.asarray(manual.average(belief).probs), atol=1e-7
    )
