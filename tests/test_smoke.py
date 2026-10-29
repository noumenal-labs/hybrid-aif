"""Scaffolding smoke tests: the environment is sound and the parity oracle is importable."""

import jax
import jax.numpy as jnp
import pytest

import hybrid_aif


def test_version():
    assert hybrid_aif.__version__


def test_jax_jit_works():
    @jax.jit
    def f(x):
        return jnp.sum(x**2)

    assert f(jnp.arange(3.0)) == 5.0


def test_equinox_importable():
    import equinox as eqx

    class M(eqx.Module):
        w: jax.Array

    m = M(w=jnp.ones(2))
    assert jax.tree_util.tree_leaves(m)[0].shape == (2,)


def test_parity_oracle_importable():
    """pymdp (JAX) is the reference oracle for discrete-node parity tests."""
    pymdp = pytest.importorskip("pymdp")
    from pymdp.agent import Agent  # noqa: F401

    assert pymdp is not None
