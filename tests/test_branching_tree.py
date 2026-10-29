"""The sophisticated planner on the level-wise expansion (design: system.md addendum, E1, E6).

The planner's recursive form — ``values``/``expected`` descending from the root,
one ``vmap`` over actions per node — is kept here as the reference, and the
level-wise fold in ``planning.tree`` is held to it on the T-maze at every
horizon, with and without a branch closure, on either bound; and the count of
transitions it evaluates at the deepest level is pinned to ``(A·n)^(H−1)·A``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.distributions import Categorical
from hybrid_aif.objectives.base import total
from hybrid_aif.planning.enumerate import enumerate_policies
from hybrid_aif.planning.sophisticated import sophisticated_planner
from tests.test_policy_tree import _counting_transition
from tests.test_sophisticated import _replanned
from tests.test_system import _initial_beliefs, tmaze_system

TOL = 1e-6


def _recursive_sophisticated(ctx, use_bound="midpoint", metrics=True):
    """The sophisticated planner before the tree fold: recursion from the root (the reference)."""
    actions = enumerate_policies(ctx.num_controls, 1, ctx.controlled)[:, 0, :]  # (A, K)
    horizon = ctx.policy_len

    def point(tree) -> jax.Array:
        g = total(tree)
        return {"midpoint": g.midpoint, "lower": g.lower, "upper": g.upper}[use_bound]

    def values(qs, t: int) -> jax.Array:
        def one(action):
            qs_next = ctx.transition_fn(qs, action)
            tree = ctx.step_terms_fn(qs_next, action, jnp.int32(t))
            g = point(tree) if tree else jnp.zeros(())
            if t + 1 < horizon:
                if ctx.branch_fn is None:
                    g = g + expected(qs_next, t + 1)
                else:
                    p, after = ctx.branch_fn(qs_next)
                    g = g + jnp.sum(p * jax.vmap(lambda q: expected(q, t + 1))(after))
            elif ctx.terminal_terms_fn is not None:
                g = g + point(ctx.terminal_terms_fn(qs_next))
            return g

        return jax.vmap(one)(actions)

    def expected(qs, t: int) -> jax.Array:
        G = values(qs, t)
        return jnp.sum(jax.nn.softmax(-ctx.gamma * G) * G)

    G = values(ctx.init_beliefs, 0)
    q_pi = Categorical(logits=-ctx.gamma * G)
    return q_pi, ({} if metrics else None), actions[:, None, :]


def _soph(policy_len, epistemic, branch_on):
    system, params = tmaze_system(policy_len=policy_len, epistemic=epistemic)
    kw = {"branch_on": branch_on} if branch_on else {}
    soph = _replanned(system, planner="sophisticated", **kw)
    return soph, params, _initial_beliefs(soph, params)


def _assert_parity(soph, params, beliefs, **kw):
    q_pi, tree = soph.infer_policies(params, beliefs, **kw)
    ref_q_pi, ref_tree = soph.infer_policies(
        params, beliefs, planner=_recursive_sophisticated, **kw
    )
    np.testing.assert_allclose(
        np.asarray(q_pi.probs), np.asarray(ref_q_pi.probs), rtol=TOL, atol=TOL
    )
    # the root G, recovered from the logits up to the constant a Categorical drops
    g = -np.asarray(q_pi.logits) / soph.gamma
    ref_g = -np.asarray(ref_q_pi.logits) / soph.gamma
    np.testing.assert_allclose(g - g.mean(), ref_g - ref_g.mean(), rtol=TOL, atol=TOL)
    assert tree == ref_tree


@pytest.mark.parametrize("branch_on", [("cue",), ("cue", "reward"), None])
@pytest.mark.parametrize("epistemic", [True, False])
@pytest.mark.parametrize("policy_len", [1, 2, 3, 4])
def test_tmaze_parity(policy_len, epistemic, branch_on):
    soph, params, beliefs = _soph(policy_len, epistemic, branch_on)
    _assert_parity(soph, params, beliefs)


@pytest.mark.parametrize("use_bound", ["lower", "upper"])
def test_bound_parity(use_bound):
    soph, params, beliefs = _soph(3, True, ("cue",))
    _assert_parity(soph, params, beliefs, use_bound=use_bound)
    _assert_parity(soph, params, beliefs, use_bound=use_bound, metrics=False)


@pytest.mark.parametrize("policy_len", [2, 3, 4])
def test_deepest_level_holds_an_times_n_to_the_h_minus_one_times_a_beliefs(policy_len):
    """Transitions evaluated per level: ``(A·n)^t·A`` at depth ``t``, so the deepest
    level — and the whole evaluation's peak — is ``(A·n)^(H−1)·A``, with ``n = 3``
    cue outcomes and ``A = 5``; without a branch closure ``n = 1``."""
    for branch_on, n in ((("cue",), 3), (None, 1)):
        soph, params, beliefs = _soph(policy_len, True, branch_on)
        ctx = soph.planner_context(params, beliefs)
        sizes: list[int] = []
        ctx = ctx._replace(transition_fn=_counting_transition(ctx.transition_fn, sizes))
        sophisticated_planner(ctx)
        A = 5
        assert sizes == [(A * n) ** t * A for t in range(policy_len)]
        assert sizes[-1] == (A * n) ** (policy_len - 1) * A
