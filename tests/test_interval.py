import jax.numpy as jnp

from hybrid_aif import Interval


def test_exact_has_zero_width():
    iv = Interval.exact(jnp.asarray(3.0))
    assert iv.width == 0.0
    assert iv.midpoint == 3.0


def test_addition_adds_bounds():
    a = Interval(jnp.asarray(1.0), jnp.asarray(2.0))
    b = Interval(jnp.asarray(10.0), jnp.asarray(20.0))
    c = a + b
    assert (c.lower, c.upper) == (11.0, 22.0)


def test_negation_flips_bounds():
    a = Interval(jnp.asarray(1.0), jnp.asarray(2.0))
    n = -a
    assert (n.lower, n.upper) == (-2.0, -1.0)


def test_subtraction():
    a = Interval(jnp.asarray(1.0), jnp.asarray(2.0))
    b = Interval(jnp.asarray(0.5), jnp.asarray(1.0))
    c = a - b
    assert (c.lower, c.upper) == (0.0, 1.5)


def test_negative_scalar_multiplication_flips():
    a = Interval(jnp.asarray(1.0), jnp.asarray(2.0))
    c = -2.0 * a
    assert (c.lower, c.upper) == (-4.0, -2.0)


def test_scalar_addition_shifts():
    a = Interval(jnp.asarray(1.0), jnp.asarray(2.0))
    c = a + 1.0
    assert (c.lower, c.upper) == (2.0, 3.0)


def test_clip_lower_clamps_both_bounds():
    a = Interval(jnp.asarray(-1.0), jnp.asarray(0.5))
    c = a.clip_lower(0.0)
    assert (c.lower, c.upper) == (0.0, 0.5)
    c2 = a.clip_lower(1.0)
    assert (c2.lower, c2.upper) == (1.0, 1.0)
    untouched = a.clip_lower(-2.0)
    assert (untouched.lower, untouched.upper) == (-1.0, 0.5)


# --------------------------------------------------- the warrant axis (A4)


def test_derived_defaults_true_and_exact_can_opt_out():
    """``amortization.md`` A4: width (point vs spread) and warrant (is the enclosure
    guaranteed?) are orthogonal. Every pre-amortization site is a genuine
    derivation, so the flag defaults ``True`` and the unusual thing is marked at its
    call site; ``Interval.exact(v, derived=False)`` is the amortized leaf — a point
    estimate with no warrant."""
    assert Interval.exact(jnp.asarray(1.0)).derived is True
    assert Interval(jnp.asarray(0.0), jnp.asarray(1.0)).derived is True
    net = Interval.exact(jnp.asarray(1.0), derived=False)
    assert net.derived is False and float(net.width) == 0.0


def test_derived_propagates_as_an_and_through_every_combinator():
    """A value is derived iff every input to it is — which is why one amortized leaf
    honestly poisons the G it is summed into, rather than that being bookkeeping."""
    warranted = Interval(jnp.asarray(1.0), jnp.asarray(2.0))
    net = Interval.exact(jnp.asarray(3.0), derived=False)
    assert (warranted + net).derived is False
    assert (net + warranted).derived is False
    assert (warranted - net).derived is False
    assert (warranted + warranted).derived is True
    assert (-net).derived is False  # negation is not laundering
    assert (-2.0 * net).derived is False
    assert (net * 2.0).derived is False
    assert net.clip_lower(0.0).derived is False
    assert (net + 1.0).derived is False  # a bare array is not a claim either way
    assert (warranted + 1.0).derived is True


def test_derived_is_static_so_it_is_pytree_structure():
    """Static, not data: the flag cannot be traced or branched on inside jit, and a
    ``lax.scan`` carry whose warrant changed between its initial value and its first
    step is an error rather than a silent relabelling."""
    import jax

    leaves, treedef = jax.tree_util.tree_flatten(Interval.exact(jnp.asarray(1.0), derived=False))
    assert len(leaves) == 2  # lower and upper only
    assert jax.tree_util.tree_unflatten(treedef, leaves).derived is False
    assert (
        jax.eval_shape(lambda x: Interval.exact(x, derived=False), jnp.zeros(())).derived is False
    )
