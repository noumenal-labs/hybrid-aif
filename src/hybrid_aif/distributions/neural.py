"""Neural conditional families: a parametrized function in a declaration slot.

- :class:`MLPConditional`: an MLP over the concatenated parents with a diagonal-Gaussian head;
- :class:`MLPPredictor`: the trunk alone (parents → a raw output vector), with no density;
- :class:`PredictorHead`: a trunk plus the name of the loss that reads its output;
- :class:`MLPCategorical`: a trunk with a factored, unimix-mixed categorical head;
- :class:`GRUCell`: a deterministic carry update, declared through
  :class:`~hybrid_aif.factors.Deterministic`;
- :class:`LearnedVector`: a trainable vector behind an activation (a parametrized constant);
- the world-model loss pieces (:func:`kl_balance`, :func:`symlog_mse_nll`,
  :func:`symlog_disc_nll`, :func:`binary_nll`).

Everything here is role-free (no class knows which slot holds it) and unbatched: ``parents`` is
a tuple of unbatched vectors concatenated in declaration order, ``parent_dims`` their widths; a
discrete parent arrives as its one-hot or probability vector, a continuous action as its raw
vector. Batch by ``vmap``. ``frozen`` is a static field: a frozen declaration has no trainable
marking, allocates no optimizer state and comes back from ``learn`` bitwise.

**Gaussian head** (:class:`MLPConditional`). The final layer emits ``2·child_dim`` numbers,
mean then scale pre-activation, read one of two ways (:attr:`MLPConditional.head`):

- ``"softplus"`` (default): mean raw, scale ``min_scale + softplus(raw)``; the floor keeps the
  log-density finite;
- ``"bounded"``: mean ``tanh(raw)``, scale ``minstd + (maxstd − minstd)·sigmoid(raw + 2)``;
  ``minstd == maxstd`` is a fixed-σ head whose scale parameters get zero gradient.

**LayerNorm.** A normed hidden layer runs ``linear → norm → activation`` (eps 1e-3) and carries
no bias; the head layer is never normed and keeps its bias.

Design: docs/design/amortization.md A1, A2, A9, A10
"""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from hybrid_aif.distributions.categorical import Categorical
from hybrid_aif.distributions.gaussian import Gaussian

#: The default optimizer, at unit learning rate: ``learn`` scales the updates by the slot's
#: resolved rate, which equals ``optax.adam(lr)`` and keeps the rate traceable.
OPTIMIZER = optax.adam(1.0)

#: ``optax.adam``'s own ``eps``; :data:`OPTIMIZER` runs at this value.
DEFAULT_EPS = 1e-8


def adam(eps: float = DEFAULT_EPS) -> optax.GradientTransformation:
    """:data:`OPTIMIZER` with an explicit ``eps``: a per-slot adam at unit rate.

    ``eps`` enters adam's update, not its ``init``, so an optimizer state built by
    :meth:`~hybrid_aif.system.System.init_learn_state` (with :data:`OPTIMIZER`) is compatible with
    any ``eps``.

    Design: docs/design/amortization.md chunk 2d note §(a) item 5
    """
    return optax.adam(1.0, eps=float(eps))


_LOG_2PI = float(jnp.log(2.0 * jnp.pi))

_ACTIVATIONS = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "silu": jax.nn.silu,  # DreamerV3's default activation
}


def _inverse_softplus(y: float) -> float:
    return float(jnp.log(jnp.expm1(jnp.asarray(y))))


#: The standard deviation of a unit normal truncated to [-2, 2]. Dividing a truncated
#: draw by it makes the weights' std come out at ``sqrt(scale / denom)`` exactly rather
#: than 0.8796× it.
_TRUNC_NORMAL_STD = 0.87962566103423978


def trunc_normal_avg(key: Array, shape: tuple[int, ...], fan_in: int, fan_out: int) -> Array:
    """Truncated-normal weights with variance scaled by the average fan, at scale 1.0.

    ``std = sqrt(1 / mean(fan_in, fan_out)) / 0.87962566…`` times a unit normal truncated to
    ``[-2, 2]`` (DreamerV3's ``winit: normal, fan: avg``). Returns a float32 array of ``shape``;
    biases are the caller's. The fans are arguments so a caller can draw half of a fused layer at
    that half's own fan.
    """
    denom = (float(fan_in) + float(fan_out)) / 2.0
    std = (1.0 / denom) ** 0.5 / _TRUNC_NORMAL_STD
    return std * jax.random.truncated_normal(key, -2.0, 2.0, shape, jnp.float32)


def uniform_avg(key: Array, shape: tuple[int, ...], fan_in: int, fan_out: int) -> Array:
    """Uniform weights with variance scaled by the average fan, at scale 1.0.

    ``U(−limit, limit)`` with ``limit = sqrt(3 / mean(fan_in, fan_out))``: the second moment of
    :func:`trunc_normal_avg` in a uniform shape (DreamerV3's ``winit: uniform``). Returns a float32
    array of ``shape``. Under ``init="trunc_normal_avg"`` a stack's read-out layer draws this.
    """
    denom = (float(fan_in) + float(fan_out)) / 2.0
    limit = (3.0 / denom) ** 0.5
    return jax.random.uniform(key, shape, jnp.float32, -limit, limit)


class MLPConditional(eqx.Module):
    """MLP over concatenated parents → diagonal-Gaussian child (see module docstring).

    The family contract is :meth:`log_prob`, :meth:`sample` and :meth:`moments`, all through
    :meth:`condition_on`, the plain :class:`~hybrid_aif.distributions.Gaussian` the parents select.
    :meth:`stats` (one tick's gradient of the NLL) and :meth:`absorb` (the optax step) learn it. It
    is not a :class:`~hybrid_aif.distributions.Learnable` (there is no posterior over the weights):
    ``isinstance(decl, Learnable)`` is ``False``, so the conjugate paths leave it alone.

    Construct with :meth:`from_shapes`; the field constructor stays the dataclass default so
    ``dataclasses.replace`` and ``eqx.tree_at`` work. Freeze with
    ``dataclasses.replace(decl, frozen=True)``.
    """

    layers: tuple[eqx.nn.Linear, ...]  # the trainable leaves live here
    parent_dims: tuple[int, ...] = eqx.field(static=True)  # width of each parent vector
    child_dim: int = eqx.field(static=True)
    norms: tuple[eqx.nn.LayerNorm, ...] = ()  # one per hidden layer, or () for no norm
    activation: str = eqx.field(static=True, default="tanh")
    min_scale: float = eqx.field(static=True, default=1e-3)  # softplus head's additive floor
    frozen: bool = eqx.field(static=True, default=False)  # the trainable marking
    head: str = eqx.field(static=True, default="softplus")  # or "bounded" — see module docstring
    minstd: float = eqx.field(static=True, default=0.1)  # bounded head only
    maxstd: float = eqx.field(static=True, default=1.0)  # bounded head only

    def __check_init__(self) -> None:
        if not self.parent_dims or any(d <= 0 for d in self.parent_dims):
            raise ValueError(
                f"parent_dims must be a non-empty tuple of positive widths, got "
                f"{self.parent_dims!r} (one entry per declared parent; a discrete parent's "
                "width is its level count — the one-hot convention)"
            )
        if self.child_dim <= 0:
            raise ValueError(f"child_dim must be positive, got {self.child_dim}")
        if self.activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown activation {self.activation!r}; known activations are "
                f"{sorted(_ACTIVATIONS)}"
            )
        if self.min_scale <= 0.0:
            raise ValueError(
                f"min_scale must be strictly positive, got {self.min_scale} (it is the floor "
                "that keeps the conditional log-density finite; see the module docstring)"
            )
        if not self.layers:
            raise ValueError("layers must be non-empty")
        if self.layers[0].in_features != self.parent_dim:
            raise ValueError(
                f"first layer takes {self.layers[0].in_features} inputs but the parents "
                f"concatenate to {self.parent_dim} (parent_dims={self.parent_dims})"
            )
        if self.layers[-1].out_features != 2 * self.child_dim:
            raise ValueError(
                f"final layer emits {self.layers[-1].out_features} outputs but a Gaussian "
                f"head over a {self.child_dim}-dimensional child needs {2 * self.child_dim} "
                "(mean, then pre-softplus scale)"
            )
        for a, b in zip(self.layers[:-1], self.layers[1:], strict=True):
            if a.out_features != b.in_features:
                raise ValueError(
                    f"layer widths do not chain: {a.out_features} out, {b.in_features} in"
                )
        if self.head not in ("softplus", "bounded"):
            raise ValueError(
                f"unknown head {self.head!r}; the built heads are ['bounded', 'softplus'] "
                "(the module docstring's two scale parameterizations)"
            )
        if self.head == "bounded" and not 0.0 < self.minstd <= self.maxstd:
            raise ValueError(
                f"a bounded head needs 0 < minstd <= maxstd, got ({self.minstd}, {self.maxstd}) "
                "— equality is the fixed-sigma case, not an error"
            )
        if self.norms and len(self.norms) != len(self.layers) - 1:
            raise ValueError(
                f"got {len(self.norms)} norm(s) for {len(self.layers) - 1} hidden layer(s) "
                "(the head layer is never normed; "
                "pass () for an unnormalized net)"
            )

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_shapes(
        cls,
        parent_dims: tuple[int, ...],
        child_dim: int,
        *,
        key: Array,
        width: int = 64,
        depth: int = 2,
        activation: str = "tanh",
        min_scale: float = 1e-3,
        init_scale: float = 1.0,
        frozen: bool = False,
        norm: bool = False,
        head: str = "softplus",
        minstd: float = 0.1,
        maxstd: float = 1.0,
        init: str = "default",
    ) -> "MLPConditional":
        """A randomly-initialized net: ``depth`` hidden layers of ``width`` units.

        ``init_scale`` centres a softplus head's initial standard deviation (the scale-half bias)
        and must exceed ``min_scale``; a bounded head ignores it. Options, all off by default:

        - ``norm=True``: a LayerNorm between each hidden linear and its activation, hidden biases
          dropped;
        - ``head="bounded"`` with ``minstd``/``maxstd`` (``minstd == maxstd`` is the fixed-σ head);
        - ``init="trunc_normal_avg"``: :func:`trunc_normal_avg` on the hidden layers,
          :func:`uniform_avg` on the fused head layer (each half at its own ``(in, child_dim)``
          fan), every bias zero. ``init="default"`` is ``eqx.nn.Linear``'s initializer throughout.
        """
        parent_dims = tuple(int(d) for d in parent_dims)
        child_dim = int(child_dim)
        if depth < 0:
            raise ValueError(f"depth must be non-negative, got {depth}")
        if init not in ("default", "trunc_normal_avg"):
            raise ValueError(
                f"unknown init {init!r}; the initializers are ['default', 'trunc_normal_avg'] "
                "(eqx.nn.Linear's own, or a truncated normal scaled by the fan average, "
                "biasless under a norm)"
            )
        sizes = [sum(parent_dims), *([width] * depth), 2 * child_dim]
        keys = jax.random.split(key, len(sizes) - 1)
        layers = [
            #  a normed hidden layer carries no bias (module docstring);
            #  the head layer (i == depth) is never normed and always keeps its bias
            eqx.nn.Linear(a, b, use_bias=not (norm and i < depth), key=k)
            for i, (a, b, k) in enumerate(zip(sizes[:-1], sizes[1:], list(keys), strict=True))
        ]
        if init == "trunc_normal_avg":
            wkeys = jax.random.split(jax.random.fold_in(key, 0x17), len(layers) + 1)
            for i, layer in enumerate(layers):
                fan_in = layer.in_features
                if i == len(layers) - 1:
                    #  the fused head is two read-outs (mean, scale): each half is drawn
                    #  uniform at its own fan (fan_out = child_dim)
                    weight = jnp.concatenate(
                        [
                            uniform_avg(wkeys[i], (child_dim, fan_in), fan_in, child_dim),
                            uniform_avg(wkeys[-1], (child_dim, fan_in), fan_in, child_dim),
                        ],
                        axis=0,
                    )
                else:
                    out = layer.out_features
                    weight = trunc_normal_avg(wkeys[i], (out, fan_in), fan_in, out)
                layer = eqx.tree_at(lambda m: m.weight, layer, weight)
                if layer.bias is not None:  # every bias starts at zero
                    layer = eqx.tree_at(lambda m: m.bias, layer, jnp.zeros_like(layer.bias))
                layers[i] = layer
        if head == "softplus":
            if init_scale <= min_scale:
                raise ValueError(
                    f"init_scale {init_scale} must exceed min_scale {min_scale} (the head's "
                    "scale is min_scale + softplus(·), so it cannot start at or below the floor)"
                )
            last = layers[-1]
            bias = last.bias.at[child_dim:].set(_inverse_softplus(init_scale - min_scale))
            layers[-1] = eqx.tree_at(lambda m: m.bias, last, bias)
        return cls(
            layers=tuple(layers),
            parent_dims=parent_dims,
            child_dim=child_dim,
            norms=(tuple(eqx.nn.LayerNorm(w, eps=1e-3) for w in sizes[1:-1]) if norm else ()),
            activation=activation,
            min_scale=min_scale,
            frozen=frozen,
            head=head,
            minstd=float(minstd),
            maxstd=float(maxstd),
        )

    # -------------------------------------------------------------------- structure

    @property
    def parent_dim(self) -> int:
        """Width of the concatenated parent vector."""
        return sum(self.parent_dims)

    @property
    def learnable(self) -> bool:
        """Capability: a non-frozen net's weights are a trainable block."""
        return not self.frozen

    # -------------------------------------------------------------- family contract

    def _concat(self, parents: tuple[Array, ...]) -> Array:
        if len(parents) != len(self.parent_dims):
            raise ValueError(
                f"got {len(parents)} parent value(s) for {len(self.parent_dims)} declared "
                f"parent(s) (parent_dims={self.parent_dims})"
            )
        cast = [jnp.asarray(p) for p in parents]
        for i, (p, d) in enumerate(zip(cast, self.parent_dims, strict=True)):
            if p.shape != (d,):
                raise ValueError(
                    f"parent {i} has shape {p.shape}, expected ({d},) — parents are "
                    "unbatched vectors (a discrete parent as a one-hot / probability "
                    "vector); batch with vmap"
                )
        return jnp.concatenate(cast, axis=-1)

    def _head(self, parents: tuple[Array, ...]) -> tuple[Array, Array]:
        """``(mean, scale)``: the Gaussian head's two halves, per :attr:`head`."""
        h = self._concat(parents)
        act = _ACTIVATIONS[self.activation]
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(h)
            if self.norms:  # linear -> norm -> act (module docstring)
                h = self.norms[i](h)
            h = act(h)
        out = self.layers[-1](h)
        mean, raw = out[: self.child_dim], out[self.child_dim :]
        if self.head == "bounded":
            #  bounded head: Normal(tanh(out), lo + (hi-lo)*sigmoid(raw+2))
            lo, hi = self.minstd, self.maxstd
            return jnp.tanh(mean), lo + (hi - lo) * jax.nn.sigmoid(raw + 2.0)
        return mean, self.min_scale + jax.nn.softplus(raw)

    def condition_on(self, parents: tuple[Array, ...]) -> Gaussian:
        """The conditional as a plain :class:`Gaussian` with a diagonal Cholesky."""
        mean, scale = self._head(parents)
        return Gaussian(loc=mean, chol=jnp.diag(scale))

    def log_prob(self, child: Array, parents: tuple[Array, ...]) -> Array:
        """``log p(child | parents)`` (scalar), computed on the diagonal (:meth:`diag_nll`)."""
        return -self.diag_nll(child, parents)

    def sample(self, key: Array, parents: tuple[Array, ...]) -> Array:
        """One ancestral draw of the child given its parents, shape ``(child_dim,)``."""
        return self.condition_on(parents).sample(key)

    def moments(self, parents: tuple[Array, ...]) -> tuple[Array, Array]:
        """``(mean, cov)`` given the parents; ``cov`` is the full, diagonal-valued
        ``(child_dim, child_dim)`` matrix.
        """
        mean, scale = self._head(parents)
        return mean, jnp.diag(scale**2)

    def diag_moments(self, parents: tuple[Array, ...]) -> tuple[Array, Array]:
        """``(mean, scale)``: the head's diagonal view, with no covariance built."""
        return self._head(parents)

    def diag_nll(self, child: Array, parents: tuple[Array, ...]) -> Array:
        """Negative log-density on the diagonal: ``0.5·Σ((y−μ)/σ)² + Σ log σ + 0.5·d·log 2π``.

        Exactly ``−log_prob`` for this head, without building or solving a ``(d, d)`` Cholesky.

        Design: docs/design/amortization.md chunk 2c note §(e) item 1
        """
        mean, scale = self._head(parents)
        z = (jnp.asarray(child) - mean) / scale
        return (
            0.5 * jnp.sum(z**2, axis=-1)
            + jnp.sum(jnp.log(scale), axis=-1)
            + 0.5 * self.child_dim * _LOG_2PI
        )

    # ------------------------------------------------------------- gradient learning

    def nll(self, child: Array, parents: tuple[Array, ...]) -> Array:
        """Negative conditional log-density: the gradient loss on one evidence pair."""
        return self.diag_nll(child, parents)

    def stats(self, child: Array, parents: tuple[Array, ...]) -> "MLPConditional":
        """The per-tick statistic ``∇ nll(child, parents)``, as a same-structured pytree.

        ``None`` sits at every non-trainable position. Raises ``ValueError`` on a frozen
        declaration.
        """
        if self.frozen:
            raise ValueError(
                "stats() on a frozen MLPConditional: a frozen declaration has no trainable "
                "parameters, so there is no gradient statistic to take (design: "
                "amortization.md A1 — frozen ⇔ no trainable marking). Unfreeze with "
                "dataclasses.replace(decl, frozen=False)."
            )
        return eqx.filter_grad(lambda m: m.nll(child, parents))(self)

    def init_opt_state(self):
        """Fresh optimizer state for this declaration — one entry of ``LearnState``."""
        return OPTIMIZER.init(eqx.filter(self, eqx.is_inexact_array))

    def absorb(
        self,
        stats: "MLPConditional",
        opt_state,
        lr: float | Array = 1.0,
        decay: float | Array = 1.0,
    ) -> tuple["MLPConditional", object]:
        """One optax step: returns ``(declaration, opt_state)``.

        The optimizer state is passed in and out. ``lr`` scales the step and may be traced;
        ``decay`` other than ``1.0`` raises ``NotImplementedError``.
        """
        if not (isinstance(decay, int | float) and float(decay) == 1.0):
            raise NotImplementedError(
                "decay != 1.0 on an MLPConditional slot: the conjugate families' leaky-count "
                "decay (α ← decay·α + (1−decay)·anchor) has no settled meaning for network "
                "weights, and weight decay toward an anchor is a different regularizer with "
                "a different justification, so none is supplied here (design: "
                "docs/design/amortization.md A2, which fixes the rate grammar but not decay "
                "semantics for neural slots). Leave decay at 1.0 on this slot."
            )
        params = eqx.filter(self, eqx.is_inexact_array)
        updates, opt_state = OPTIMIZER.update(stats, opt_state, params)
        updates = jax.tree_util.tree_map(lambda u: lr * u, updates)
        return eqx.apply_updates(self, updates), opt_state


# ==================================================================== symlog / symexp


def symlog(x: Array) -> Array:
    """``sign(x)·log(1+|x|)``, DreamerV3's symmetric log squash, inverted by :func:`symexp`."""
    x = jnp.asarray(x)
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: Array) -> Array:
    """``sign(x)·(exp|x|−1)`` — the exact inverse of :func:`symlog`."""
    x = jnp.asarray(x)
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


# ================================================== the optax step, module-generic
#
# The shared optax step for modules whose gradient arrives from their consumer.
# Optimizer state is passed in and out; nothing is stashed on a module.


def init_opt_state(module, optimizer: optax.GradientTransformation = OPTIMIZER):
    """Fresh optimizer state for any module's trainable leaves — one ``LearnState`` entry."""
    return optimizer.init(eqx.filter(module, eqx.is_inexact_array))


def apply_gradient(
    module,
    grads,
    opt_state,
    *,
    lr: float | Array = 1.0,
    decay: float | Array = 1.0,
    optimizer: optax.GradientTransformation = OPTIMIZER,
):
    """One optax step on ``module`` from a same-structured gradient pytree.

    Returns ``(module, opt_state)``. ``lr`` scales the update and may be traced; ``decay != 1.0``
    raises ``NotImplementedError``, as in :meth:`MLPConditional.absorb`.
    """
    if not (isinstance(decay, int | float) and float(decay) == 1.0):
        raise NotImplementedError(
            f"decay={decay!r} on a gradient-trained module: the conjugate families' leaky-count "
            "decay has no settled meaning for network weights, so none is supplied "
            "(design: docs/design/amortization.md A2). Leave decay at 1.0."
        )
    params = eqx.filter(module, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    updates = jax.tree_util.tree_map(lambda u: lr * u, updates)
    return eqx.apply_updates(module, updates), opt_state


# ============================================================== the shared MLP trunk


def _concat_parents(parents: tuple[Array, ...], parent_dims: tuple[int, ...]) -> Array:
    """The parent convention: one unbatched vector per declared parent, concatenated."""
    if len(parents) != len(parent_dims):
        raise ValueError(
            f"got {len(parents)} parent value(s) for {len(parent_dims)} declared parent(s) "
            f"(parent_dims={parent_dims})"
        )
    cast = [jnp.asarray(p) for p in parents]
    for i, (p, d) in enumerate(zip(cast, parent_dims, strict=True)):
        if p.shape != (d,):
            raise ValueError(
                f"parent {i} has shape {p.shape}, expected ({d},) — parents are unbatched "
                "vectors (a categorical parent as its flattened one-hot); batch with vmap"
            )
    return jnp.concatenate(cast, axis=-1)


def _linear_stack(
    sizes: list[int],
    key: Array,
    *,
    out_scale: float,
    init: str = "default",
    hidden: int | None = None,
    norm: bool = False,
) -> tuple[eqx.nn.Linear, ...]:
    """``len(sizes) − 1`` linears chained ``sizes[0] → sizes[1] → … → sizes[-1]``.

    The first ``hidden`` layers (default: all but the last) are hidden layers, which the caller
    follows with a norm and an activation; the rest is the read-out. ``init="default"`` is
    ``eqx.nn.Linear``'s initializer, biases everywhere. ``init="trunc_normal_avg"`` draws
    :func:`trunc_normal_avg` on hidden layers (biasless when ``norm``) and :func:`uniform_avg` on
    the read-out, every kept bias starting at zero. ``out_scale`` multiplies the read-out weights
    and zeroes its bias (``0.0`` starts at the constant zero output). ``norm`` only places biases;
    the caller builds the LayerNorms. Raises ``ValueError`` on an unknown ``init``.
    """
    if init not in ("default", "trunc_normal_avg"):
        raise ValueError(
            f"unknown init {init!r}; the initializers are ['default', 'trunc_normal_avg'] "
            "(eqx.nn.Linear's own, or a truncated normal scaled by the fan average, "
            "biasless under a norm)"
        )
    keys = jax.random.split(key, len(sizes) - 1)
    n_hidden = len(sizes) - 2 if hidden is None else int(hidden)
    layers = []
    for i, (a, b, k) in enumerate(zip(sizes[:-1], sizes[1:], list(keys), strict=True)):
        if init == "default":
            layers.append(eqx.nn.Linear(a, b, key=k))
            continue
        is_hidden = i < n_hidden  # hidden: truncated normal, and biasless under a norm
        layer = eqx.nn.Linear(a, b, use_bias=not (is_hidden and norm), key=k)
        draw = trunc_normal_avg if is_hidden else uniform_avg
        layer = eqx.tree_at(lambda m: m.weight, layer, draw(k, (b, a), a, b))
        if layer.bias is not None:  # a kept bias starts at zero
            layer = eqx.tree_at(lambda m: m.bias, layer, jnp.zeros_like(layer.bias))
        layers.append(layer)
    if out_scale != 1.0:
        last = layers[-1]
        layers[-1] = eqx.tree_at(lambda m: m.weight, last, out_scale * last.weight)
        if last.bias is not None:
            layers[-1] = eqx.tree_at(lambda m: m.bias, layers[-1], jnp.zeros_like(last.bias))
    return tuple(layers)


class MLPPredictor(eqx.Module):
    """MLP over concatenated parents → a raw output vector, with no density of its own.

    The trunk the other classes here are built from; paired with a loss it is a head without a
    density family (:class:`PredictorHead`). ``depth`` hidden layers of ``width`` units, each
    optionally layer-normed (eps 1e-3) then activated, and a final linear to ``out_dim``.
    ``out_activation=True`` norms and activates the final layer too, making the trunk an encoder.
    ``symlog_inputs`` applies :func:`symlog` to the concatenated parents before the first layer.
    ``frozen`` is the trainable marking.
    """

    layers: tuple[eqx.nn.Linear, ...]
    norms: tuple[eqx.nn.LayerNorm, ...]  # one per activated layer, or () for no norm
    parent_dims: tuple[int, ...] = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    activation: str = eqx.field(static=True, default="silu")
    out_activation: bool = eqx.field(static=True, default=False)
    frozen: bool = eqx.field(static=True, default=False)
    symlog_inputs: bool = eqx.field(static=True, default=False)

    def __check_init__(self) -> None:
        if not self.parent_dims or any(d <= 0 for d in self.parent_dims):
            raise ValueError(
                f"parent_dims must be a non-empty tuple of positive widths, got "
                f"{self.parent_dims!r} (one entry per declared parent)"
            )
        if self.out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {self.out_dim}")
        if self.activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown activation {self.activation!r}; known activations are "
                f"{sorted(_ACTIVATIONS)}"
            )
        if not self.layers:
            raise ValueError("layers must be non-empty")
        if self.layers[0].in_features != sum(self.parent_dims):
            raise ValueError(
                f"first layer takes {self.layers[0].in_features} inputs but the parents "
                f"concatenate to {sum(self.parent_dims)} (parent_dims={self.parent_dims})"
            )
        if self.layers[-1].out_features != self.out_dim:
            raise ValueError(
                f"final layer emits {self.layers[-1].out_features} outputs, not out_dim="
                f"{self.out_dim}"
            )
        if self.norms and len(self.norms) != self._num_activated:
            raise ValueError(
                f"got {len(self.norms)} norm(s) for {self._num_activated} activated layer(s) "
                "(pass () for an unnormalized trunk)"
            )

    @classmethod
    def from_shapes(
        cls,
        parent_dims: tuple[int, ...],
        out_dim: int,
        *,
        key: Array,
        width: int,
        depth: int = 2,
        activation: str = "silu",
        norm: bool = True,
        out_activation: bool = False,
        out_scale: float = 1.0,
        frozen: bool = False,
        symlog_inputs: bool = False,
        init: str = "default",
    ) -> "MLPPredictor":
        """A randomly-initialized trunk: ``depth`` hidden layers of ``width`` units.

        ``width`` is required. ``norm=True`` puts a LayerNorm (eps 1e-3) before each activation.
        ``out_scale`` scales the final layer's weights and zeroes its bias (``0.0`` starts at the
        constant zero prediction). ``init`` is as in :func:`_linear_stack`.
        """
        parent_dims = tuple(int(d) for d in parent_dims)
        out_dim = int(out_dim)
        if depth < 0:
            raise ValueError(f"depth must be non-negative, got {depth}")
        sizes = [sum(parent_dims), *([width] * depth), out_dim]
        activated = len(sizes) - 1 if out_activation else len(sizes) - 2
        layers = _linear_stack(
            sizes, key, out_scale=out_scale, init=init, hidden=activated, norm=norm
        )
        norms = (
            tuple(eqx.nn.LayerNorm(layer.out_features, eps=1e-3) for layer in layers[:activated])
            if norm
            else ()
        )
        return cls(
            layers=layers,
            norms=norms,
            parent_dims=parent_dims,
            out_dim=out_dim,
            activation=activation,
            out_activation=out_activation,
            frozen=frozen,
            symlog_inputs=symlog_inputs,
        )

    @property
    def _num_activated(self) -> int:
        return len(self.layers) if self.out_activation else len(self.layers) - 1

    @property
    def parent_dim(self) -> int:
        """Width of the concatenated parent vector."""
        return sum(self.parent_dims)

    @property
    def learnable(self) -> bool:
        """Capability: a non-frozen trunk's weights are a trainable block."""
        return not self.frozen

    def __call__(self, parents: tuple[Array, ...]) -> Array:
        """The raw output vector, shape ``(out_dim,)``."""
        h = _concat_parents(parents, self.parent_dims)
        if self.symlog_inputs:
            h = symlog(h)
        act = _ACTIVATIONS[self.activation]
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < self._num_activated:
                if self.norms:
                    h = self.norms[i](h)
                h = act(h)
        return h

    def init_opt_state(self):
        """Fresh optimizer state — one entry of a ``LearnState``."""
        return init_opt_state(self)

    def absorb(self, stats, opt_state, lr: float | Array = 1.0, decay: float | Array = 1.0):
        """One optax step from a gradient pytree: ``(trunk, opt_state)``."""
        return apply_gradient(self, stats, opt_state, lr=lr, decay=decay)


# ==================================================== the factored categorical head


def unimix_logits(logits: Array, unimix: float) -> Array:
    """Mix ``unimix`` of the uniform into a categorical's probabilities, returned as logits.

    ``probs ← (1−u)·softmax(logits) + u/num_classes``: no class probability reaches zero, so
    ``log_prob`` of any one-hot is finite. ``unimix`` must lie in ``[0, 1)``; ``0.0`` passes the
    logits through. The last axis is the class axis.
    """
    if not 0.0 <= unimix < 1.0:
        raise ValueError(f"unimix must lie in [0, 1), got {unimix}")
    if unimix == 0.0:
        return jnp.asarray(logits)
    probs = jax.nn.softmax(jnp.asarray(logits), axis=-1)
    uniform = 1.0 / logits.shape[-1]
    return jnp.log((1.0 - unimix) * probs + unimix * uniform)


def straight_through_onehot(key: Array, logits: Array) -> Array:
    """Sample a one-hot per variable, with the probabilities' gradient attached.

    ``sample + probs − sg(probs)``: the forward value is the hard draw, the backward value
    ``∂probs`` (DreamerV3's ``OneHotDist.sample``). ``logits`` may carry leading batch axes; the
    last axis is the class axis.
    """
    draw = jax.random.categorical(key, logits, axis=-1)
    hard = jax.nn.one_hot(draw, logits.shape[-1], dtype=logits.dtype)
    probs = jax.nn.softmax(logits, axis=-1)
    return jax.lax.stop_gradient(hard) + (probs - jax.lax.stop_gradient(probs))


def onehot_mode(logits: Array) -> Array:
    """The argmax one-hot per variable, with the probabilities' gradient attached."""
    hard = jax.nn.one_hot(jnp.argmax(logits, axis=-1), logits.shape[-1], dtype=logits.dtype)
    probs = jax.nn.softmax(logits, axis=-1)
    return jax.lax.stop_gradient(hard) + (probs - jax.lax.stop_gradient(probs))


class MLPCategorical(eqx.Module):
    """MLP over concatenated parents → a factored categorical child.

    ``num_vars`` independent categoricals of ``num_classes`` classes each, unimix-mixed and sampled
    as a straight-through one-hot. The child value is the one-hot code, shape ``(num_vars,
    num_classes)``; :attr:`flat_dim` is its width as a downstream parent. There is no ``moments``:
    :meth:`probs` is this family's moment view.
    """

    trunk: MLPPredictor
    num_vars: int = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)
    unimix: float = eqx.field(static=True, default=0.01)

    def __check_init__(self) -> None:
        if self.num_vars <= 0 or self.num_classes <= 0:
            raise ValueError(
                f"num_vars and num_classes must be positive, got {self.num_vars} and "
                f"{self.num_classes}"
            )
        if self.trunk.out_dim != self.num_vars * self.num_classes:
            raise ValueError(
                f"the trunk emits {self.trunk.out_dim} numbers but a factored categorical over "
                f"{self.num_vars} variable(s) of {self.num_classes} class(es) needs "
                f"{self.num_vars * self.num_classes} (one logit per (variable, class))"
            )
        if not 0.0 <= self.unimix < 1.0:
            raise ValueError(f"unimix must lie in [0, 1), got {self.unimix}")

    @classmethod
    def from_shapes(
        cls,
        parent_dims: tuple[int, ...],
        num_vars: int,
        num_classes: int,
        *,
        key: Array,
        width: int,
        depth: int = 1,
        activation: str = "silu",
        norm: bool = True,
        unimix: float = 0.01,
        frozen: bool = False,
        init: str = "default",
    ) -> "MLPCategorical":
        """A randomly-initialized head: ``depth`` normed, activated hidden layers of ``width`` units
        (required), then a linear to the ``num_vars·num_classes`` logits. ``init`` is as in
        :meth:`MLPPredictor.from_shapes`.
        """
        trunk = MLPPredictor.from_shapes(
            parent_dims,
            int(num_vars) * int(num_classes),
            key=key,
            width=width,
            depth=depth,
            activation=activation,
            norm=norm,
            frozen=frozen,
            init=init,
        )
        return cls(
            trunk=trunk, num_vars=int(num_vars), num_classes=int(num_classes), unimix=float(unimix)
        )

    # -------------------------------------------------------------------- structure

    @property
    def parent_dims(self) -> tuple[int, ...]:
        return self.trunk.parent_dims

    @property
    def parent_dim(self) -> int:
        return self.trunk.parent_dim

    @property
    def flat_dim(self) -> int:
        """Width of the flattened one-hot code — what it contributes as a parent."""
        return self.num_vars * self.num_classes

    @property
    def frozen(self) -> bool:
        return self.trunk.frozen

    @property
    def learnable(self) -> bool:
        return self.trunk.learnable

    # ---------------------------------------------------------------- the contract

    def logits(self, parents: tuple[Array, ...]) -> Array:
        """Unimix-mixed logits, shape ``(num_vars, num_classes)``."""
        raw = self.trunk(parents).reshape(self.num_vars, self.num_classes)
        return unimix_logits(raw, self.unimix)

    def condition_on(self, parents: tuple[Array, ...]) -> Categorical:
        """The conditional as a :class:`Categorical` whose batch axis is the variable
        axis, so entropies and KLs come out per variable.
        """
        return Categorical(logits=self.logits(parents))

    def probs(self, parents: tuple[Array, ...]) -> Array:
        """The ``(num_vars, num_classes)`` probability tensor: this family's moment view."""
        return jax.nn.softmax(self.logits(parents), axis=-1)

    def _as_code(self, child: Array) -> Array:
        child = jnp.asarray(child)
        if child.shape == (self.num_vars, self.num_classes):
            return child
        if child.shape == (self.flat_dim,):
            return child.reshape(self.num_vars, self.num_classes)
        raise ValueError(
            f"child has shape {child.shape}; a factored-categorical child is the one-hot code, "
            f"either ({self.num_vars}, {self.num_classes}) or its flattening ({self.flat_dim},)"
        )

    def log_prob(self, child: Array, parents: tuple[Array, ...]) -> Array:
        """``log q(child | parents)`` for a one-hot code, summed over variables.

        ``Σ_v Σ_c child[v,c]·log p[v,c]``: differentiable through a straight-through sample, and
        finite for every code (:func:`unimix_logits`).
        """
        code = self._as_code(child)
        return jnp.sum(code * jax.nn.log_softmax(self.logits(parents), axis=-1))

    def sample(self, key: Array, parents: tuple[Array, ...]) -> Array:
        """One straight-through one-hot draw, shape ``(num_vars, num_classes)``."""
        return straight_through_onehot(key, self.logits(parents))

    def mode(self, parents: tuple[Array, ...]) -> Array:
        """The argmax code, same shape as :meth:`sample`."""
        return onehot_mode(self.logits(parents))

    # ---------------------------------------------------------------- the optax step

    def init_opt_state(self):
        return init_opt_state(self)

    def absorb(self, stats, opt_state, lr: float | Array = 1.0, decay: float | Array = 1.0):
        return apply_gradient(self, stats, opt_state, lr=lr, decay=decay)


# =================================================================== the loss pieces


def factored_categorical_kl(post_logits: Array, prior_logits: Array) -> Array:
    """``KL(q ‖ p)`` between two factored categoricals, summed over variables.

    Leading batch axes are preserved. Returns a plain array (a loss term), not an
    :class:`~hybrid_aif.interval.Interval`.
    """
    lq = jax.nn.log_softmax(jnp.asarray(post_logits), axis=-1)
    lp = jax.nn.log_softmax(jnp.asarray(prior_logits), axis=-1)
    return jnp.sum(jnp.exp(lq) * (lq - lp), axis=(-2, -1))


def factored_categorical_entropy(logits: Array) -> Array:
    """``H[q]`` of a factored categorical, summed over variables.

    Read off the logits, not a sampled code (whose entropy is zero); leading batch axes are
    preserved. ``train.imagine``'s ``prior_entropy_scale`` term reads it.

    Design: docs/design/amortization-p0-addendum.md §3
    """
    lq = jax.nn.log_softmax(jnp.asarray(logits), axis=-1)
    return -jnp.sum(jnp.exp(lq) * lq, axis=(-2, -1))


class KLBalance(NamedTuple):
    """The complexity term of the world-model loss, decomposed.

    ``total = dyn_scale·dyn + rep_scale·rep``; ``dyn``/``rep`` are the two
    stop-gradient directions *after* the free-bits floor and ``dyn_raw``/
    ``rep_raw`` the same KLs before it — reported so a caller can see whether
    the floor is currently binding (``dyn > dyn_raw`` ⇔ clamped).
    """

    total: Array
    dyn: Array
    rep: Array
    dyn_raw: Array
    rep_raw: Array


def kl_balance(
    post_logits: Array,
    prior_logits: Array,
    *,
    dyn_scale: float = 0.5,
    rep_scale: float = 0.1,
    free_nats: float = 1.0,
) -> KLBalance:
    """KL balancing with free bits: the complexity term of a world-model loss.

    - ``dyn = KL(sg(q) ‖ p)`` at ``dyn_scale`` trains the prior toward the posterior;
    - ``rep = KL(q ‖ sg(p))`` at ``rep_scale`` trains the posterior toward the prior.

    ``free_nats`` floors each direction after the sum over variables, per element (batch by
    ``vmap`` and average afterwards); below the floor the term is constant with zero gradient.
    Raises ``ValueError`` on a negative floor.

    Design: docs/design/amortization-p0-addendum.md §3
    """
    if free_nats < 0.0:
        raise ValueError(f"free_nats must be non-negative, got {free_nats}")
    sg = jax.lax.stop_gradient
    dyn_raw = factored_categorical_kl(sg(post_logits), prior_logits)
    rep_raw = factored_categorical_kl(post_logits, sg(prior_logits))
    dyn = jnp.maximum(dyn_raw, free_nats)
    rep = jnp.maximum(rep_raw, free_nats)
    return KLBalance(
        total=dyn_scale * dyn + rep_scale * rep,
        dyn=dyn,
        rep=rep,
        dyn_raw=dyn_raw,
        rep_raw=rep_raw,
    )


def symlog_mse_nll(prediction: Array, target: Array) -> Array:
    """The ``symlog_mse`` loss: ``Σ_i (pred_i − symlog(target)_i)²``.

    The prediction lives in symlog space; read it back in the target's units with :func:`symexp`.
    """
    return jnp.sum((jnp.asarray(prediction) - symlog(target)) ** 2, axis=-1)


def disc_bins(num_bins: int = 255, low: float = -20.0, high: float = 20.0) -> Array:
    """The ``symlog_disc`` head's bin centres: ``num_bins`` evenly spaced over ``[low, high]``."""
    if num_bins < 2:
        raise ValueError(f"num_bins must be at least 2, got {num_bins}")
    if not high > low:
        raise ValueError(f"need low < high, got low={low}, high={high}")
    return jnp.linspace(low, high, num_bins)


def twohot(target: Array, bins: Array) -> Array:
    """Two-hot encoding of a scalar target over ``bins`` (DreamerV3's ``DiscDist``).

    Mass on the two bracketing bins, split linearly, so the encoding's expectation is the target
    (clipped at the ends). Vectorize over a batch with ``vmap``.
    """
    target, bins = jnp.asarray(target), jnp.asarray(bins)
    if target.shape != ():
        raise ValueError(
            f"twohot takes a scalar target, got shape {target.shape} — vmap over batch axes"
        )
    n = bins.shape[0]
    below = jnp.clip(jnp.sum((bins <= target).astype(jnp.int32)) - 1, 0, n - 1)
    above = jnp.clip(n - jnp.sum((bins > target).astype(jnp.int32)), 0, n - 1)
    equal = below == above
    d_below = jnp.where(equal, 1.0, jnp.abs(bins[below] - target))
    d_above = jnp.where(equal, 1.0, jnp.abs(bins[above] - target))
    total = d_below + d_above
    return (
        jax.nn.one_hot(below, n) * (d_above / total)[..., None]
        + jax.nn.one_hot(above, n) * (d_below / total)[..., None]
    )


def symlog_disc_nll(logits: Array, target: Array, bins: Array) -> Array:
    """The ``symlog_disc`` NLL: cross-entropy against the two-hot of ``symlog(target)``.

    The loss of a binned-scalar head: a reward head, or a critic trained on a return target.

    Design: docs/design/amortization.md A10 disposition 6, A12 item 5
    """
    return -jnp.sum(twohot(symlog(target), bins) * jax.nn.log_softmax(logits, axis=-1), axis=-1)


def symlog_disc_mean(logits: Array, bins: Array) -> Array:
    """Decode a ``symlog_disc`` head: ``symexp`` of the expected bin."""
    return symexp(jnp.sum(jax.nn.softmax(logits, axis=-1) * bins, axis=-1))


def binary_nll(logit: Array, target: Array) -> Array:
    """The ``binary`` head's loss: the Bernoulli NLL ``softplus(logit) − target·logit``.

    Stable for large logits; ``target`` may be a soft value in [0, 1].
    """
    logit, target = jnp.asarray(logit), jnp.asarray(target)
    return jax.nn.softplus(logit) - target * logit


# ======================================================== trunk + loss: a declaration

#: The output-head losses :class:`PredictorHead` can name, with the width its
#: trunk must emit given the width of the value it predicts (``None`` = the value's
#: own width; ``symlog_disc`` emits one logit per bin, ``binary`` a single logit).
_HEAD_LOSSES: dict[str, int | None] = {"symlog_mse": None, "symlog_disc": -1, "binary": 1}


class PredictorHead(eqx.Module):
    """An :class:`MLPPredictor` plus the name of the loss that reads its output.

    What a ``likelihood`` slot holds for a head without a density family: ``symlog_mse``
    reconstructs a vector, ``symlog_disc`` predicts a scalar through ``num_bins`` two-hot bins over
    ``[bin_low, bin_high]`` in symlog space, ``binary`` predicts a Bernoulli from one logit.
    :meth:`nll` is the training loss and :meth:`predict` the decoded prediction, in the value's own
    units. Not a Distribution or a :class:`~hybrid_aif.distributions.Conditional`, and role-free.
    The three loss names are a closed list; an unknown name raises.

    Design: docs/design/amortization-p0-addendum.md §2
    """

    trunk: MLPPredictor
    loss: str = eqx.field(static=True)
    num_bins: int = eqx.field(static=True, default=255)  # symlog_disc only
    bin_low: float = eqx.field(static=True, default=-20.0)
    bin_high: float = eqx.field(static=True, default=20.0)

    def __check_init__(self) -> None:
        if self.loss not in _HEAD_LOSSES:
            raise ValueError(
                f"unknown head loss {self.loss!r}; the built losses are "
                f"{sorted(_HEAD_LOSSES)} (a symlog_mse vector head, a symlog_disc binned scalar "
                "head, a binary Bernoulli head)"
            )
        if self.loss == "symlog_disc":
            if self.num_bins < 2:
                raise ValueError(f"symlog_disc needs at least 2 bins, got {self.num_bins}")
            if not self.bin_high > self.bin_low:
                raise ValueError(
                    f"symlog_disc bins need bin_low < bin_high, got {self.bin_low} and "
                    f"{self.bin_high}"
                )
            if self.trunk.out_dim != self.num_bins:
                raise ValueError(
                    f"a symlog_disc head emits one logit per bin, so its trunk must emit "
                    f"num_bins={self.num_bins} numbers, not {self.trunk.out_dim}"
                )
        elif self.loss == "binary" and self.trunk.out_dim != 1:
            raise ValueError(
                f"a binary head is one logit, so its trunk must emit 1 number, not "
                f"{self.trunk.out_dim}"
            )

    @classmethod
    def from_shapes(
        cls,
        parent_dims: tuple[int, ...],
        *,
        loss: str,
        key: Array,
        out_dim: int | None = None,
        num_bins: int = 255,
        bin_low: float = -20.0,
        bin_high: float = 20.0,
        width: int,
        depth: int = 2,
        activation: str = "silu",
        norm: bool = True,
        out_scale: float = 1.0,
        frozen: bool = False,
        symlog_inputs: bool = False,
        init: str = "default",
    ) -> "PredictorHead":
        """A randomly-initialized head. ``out_dim`` is required for ``symlog_mse`` and rejected by
        the other two losses, which fix the trunk's width (one logit per bin; one logit). ``width``
        is required; the trunk arguments are :meth:`MLPPredictor.from_shapes`'s.
        """
        if loss not in _HEAD_LOSSES:
            raise ValueError(
                f"unknown head loss {loss!r}; the built losses are {sorted(_HEAD_LOSSES)}"
            )
        if loss == "symlog_mse":
            if out_dim is None:
                raise ValueError(
                    "a symlog_mse head predicts a vector, so out_dim (its width) is required"
                )
            trunk_out = int(out_dim)
        else:
            if out_dim is not None:
                raise ValueError(
                    f"a {loss} head's width is fixed by the loss ("
                    f"{'one logit per bin' if loss == 'symlog_disc' else 'one logit'}), so "
                    "out_dim must be left unset"
                )
            trunk_out = int(num_bins) if loss == "symlog_disc" else 1
        return cls(
            trunk=MLPPredictor.from_shapes(
                parent_dims,
                trunk_out,
                key=key,
                width=width,
                depth=depth,
                activation=activation,
                norm=norm,
                out_scale=out_scale,
                frozen=frozen,
                symlog_inputs=symlog_inputs,
                init=init,
            ),
            loss=loss,
            num_bins=int(num_bins),
            bin_low=float(bin_low),
            bin_high=float(bin_high),
        )

    # -------------------------------------------------------------------- structure

    @property
    def parent_dims(self) -> tuple[int, ...]:
        return self.trunk.parent_dims

    @property
    def parent_dim(self) -> int:
        return self.trunk.parent_dim

    @property
    def out_dim(self) -> int:
        """Width of the trunk's raw output (logits, for the two classifier losses)."""
        return self.trunk.out_dim

    @property
    def target_dim(self) -> int:
        """Width of the value this head predicts: the vector's, or 1 for a scalar."""
        return self.trunk.out_dim if self.loss == "symlog_mse" else 1

    @property
    def frozen(self) -> bool:
        return self.trunk.frozen

    @property
    def learnable(self) -> bool:
        return self.trunk.learnable

    @property
    def bins(self) -> Array:
        """A ``symlog_disc`` head's bin centres (a constant; XLA folds it)."""
        if self.loss != "symlog_disc":
            raise ValueError(f"a {self.loss} head has no bins")
        return disc_bins(self.num_bins, self.bin_low, self.bin_high)

    # ---------------------------------------------------------------- the contract

    def __call__(self, parents: tuple[Array, ...]) -> Array:
        """The trunk's raw output — logits for the classifier losses."""
        return self.trunk(parents)

    def nll(self, target: Array, parents: tuple[Array, ...]) -> Array:
        """This head's loss on one target: the training half of the contract."""
        out = self.trunk(parents)
        if self.loss == "symlog_mse":
            return symlog_mse_nll(out, target)
        if self.loss == "symlog_disc":
            return symlog_disc_nll(out, target, self.bins)
        return binary_nll(out[0], target)

    def predict(self, parents: tuple[Array, ...]) -> Array:
        """The decoded prediction, in the predicted value's own units.

        ``symexp`` of a symlog_mse output, the expected bin of a symlog_disc head, ``σ(logit)`` for
        a binary head.
        """
        out = self.trunk(parents)
        if self.loss == "symlog_mse":
            return symexp(out)
        if self.loss == "symlog_disc":
            return symlog_disc_mean(out, self.bins)
        return jax.nn.sigmoid(out[0])

    # ---------------------------------------------------------------- the optax step

    def init_opt_state(self):
        return init_opt_state(self)

    def absorb(self, stats, opt_state, lr: float | Array = 1.0, decay: float | Array = 1.0):
        return apply_gradient(self, stats, opt_state, lr=lr, decay=decay)


# =========================================================== the deterministic map


class GRUCell(eqx.Module):
    """A GRU carry update ``(deter, *inputs) -> deter'``: a pure map, not a conditional.

    Declared through :class:`~hybrid_aif.factors.Deterministic`. DreamerV3's GRU variant: project
    the non-carry inputs to ``hidden`` units, normalize, activate; concatenate the carry; one linear
    to ``3·deter`` (normalized) giving ``(reset, candidate, update)``; then

        ``deter' = u·tanh(σ(reset)·candidate) + (1−u)·deter``,  ``u = σ(update − 1)``

    ``inputs`` are unbatched vectors of widths ``input_dims``, in declaration order; the carry is
    passed separately.
    """

    input_proj: eqx.nn.Linear  # non-carry inputs -> hidden
    gate: eqx.nn.Linear  # [deter, hidden] -> 3*deter
    norms: tuple[eqx.nn.LayerNorm, ...]  # (input, gate) norms, or () for none
    deter_dim: int = eqx.field(static=True)
    input_dims: tuple[int, ...] = eqx.field(static=True)
    activation: str = eqx.field(static=True, default="silu")

    def __check_init__(self) -> None:
        if self.deter_dim <= 0:
            raise ValueError(f"deter_dim must be positive, got {self.deter_dim}")
        if not self.input_dims or any(d <= 0 for d in self.input_dims):
            raise ValueError(
                f"input_dims must be a non-empty tuple of positive widths, got "
                f"{self.input_dims!r} (the non-carry inputs, e.g. (flat code, action))"
            )
        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {self.activation!r}")
        if self.input_proj.in_features != sum(self.input_dims):
            raise ValueError(
                f"input projection takes {self.input_proj.in_features} inputs but input_dims "
                f"{self.input_dims} concatenate to {sum(self.input_dims)}"
            )
        if self.gate.in_features != self.deter_dim + self.input_proj.out_features:
            raise ValueError(
                f"gate linear takes {self.gate.in_features} inputs but [deter, hidden] is "
                f"{self.deter_dim} + {self.input_proj.out_features}"
            )
        if self.gate.out_features != 3 * self.deter_dim:
            raise ValueError(
                f"gate linear emits {self.gate.out_features} outputs; a GRU over a "
                f"{self.deter_dim}-unit carry needs {3 * self.deter_dim} (reset, candidate, update)"
            )
        if self.norms and len(self.norms) != 2:
            raise ValueError(f"norms must be () or (input_norm, gate_norm), got {self.norms!r}")

    @classmethod
    def from_shapes(
        cls,
        deter_dim: int,
        input_dims: tuple[int, ...],
        *,
        key: Array,
        hidden: int,
        activation: str = "silu",
        norm: bool = True,
        init: str = "default",
    ) -> "GRUCell":
        """A randomly-initialized cell; ``hidden`` (required) is the width the
        non-carry inputs are projected to.

        ``norm=True`` puts a LayerNorm (eps 1e-3) after each of the two linears. ``init="default"``
        keeps both biases and ``eqx.nn.Linear``'s draw; ``init="trunc_normal_avg"`` draws
        :func:`trunc_normal_avg` weights on both linears, drops their biases when normed and starts
        any kept bias at zero.
        """
        if init not in ("default", "trunc_normal_avg"):
            raise ValueError(
                f"unknown init {init!r}; the initializers are ['default', 'trunc_normal_avg']"
            )
        deter_dim, hidden = int(deter_dim), int(hidden)
        input_dims = tuple(int(d) for d in input_dims)
        k_in, k_gate = jax.random.split(key)
        n_in, n_gate = sum(input_dims), deter_dim + hidden
        use_bias = init == "default" or not norm
        input_proj = eqx.nn.Linear(n_in, hidden, use_bias=use_bias, key=k_in)
        gate = eqx.nn.Linear(n_gate, 3 * deter_dim, use_bias=use_bias, key=k_gate)
        if init == "trunc_normal_avg":

            def reinit(layer: eqx.nn.Linear, k: Array) -> eqx.nn.Linear:
                fan_in, fan_out = layer.in_features, layer.out_features
                weight = trunc_normal_avg(k, (fan_out, fan_in), fan_in, fan_out)
                layer = eqx.tree_at(lambda m: m.weight, layer, weight)
                if layer.bias is not None:  # a kept bias starts at zero
                    layer = eqx.tree_at(lambda m: m.bias, layer, jnp.zeros_like(layer.bias))
                return layer

            input_proj, gate = reinit(input_proj, k_in), reinit(gate, k_gate)
        return cls(
            input_proj=input_proj,
            gate=gate,
            norms=(
                (eqx.nn.LayerNorm(hidden, eps=1e-3), eqx.nn.LayerNorm(3 * deter_dim, eps=1e-3))
                if norm
                else ()
            ),
            deter_dim=deter_dim,
            input_dims=input_dims,
            activation=activation,
        )

    def __call__(self, deter: Array, *inputs: Array) -> Array:
        """The next carry, shape ``(deter_dim,)``. Unbatched; batch with ``vmap``."""
        deter = jnp.asarray(deter)
        if deter.shape != (self.deter_dim,):
            raise ValueError(
                f"deter has shape {deter.shape}, expected ({self.deter_dim},) — the carry is an "
                "unbatched vector; batch with vmap"
            )
        x = self.input_proj(_concat_parents(inputs, self.input_dims))
        if self.norms:
            x = self.norms[0](x)
        x = _ACTIVATIONS[self.activation](x)
        g = self.gate(jnp.concatenate([deter, x], axis=-1))
        if self.norms:
            g = self.norms[1](g)
        reset, candidate, update = jnp.split(g, 3, axis=-1)
        candidate = jnp.tanh(jax.nn.sigmoid(reset) * candidate)
        update = jax.nn.sigmoid(update - 1.0)
        return update * candidate + (1.0 - update) * deter


# ======================================================== a parametrized constant


class LearnedVector(eqx.Module):
    """A trainable vector behind an activation: a parametrized constant, with no parents.

    The canonical use is an episode-start carry: zeros through ``tanh``, so a fresh one starts at
    the zero carry and stays in [-1, 1]. Role-free; its gradient arrives from whatever consumes its
    value. Trainable, it is DreamerV3's ``initial: 'learned'``; frozen, ``initial: 'zeros'``.
    """

    value: Array  # the pre-activation parameter
    activation: str = eqx.field(static=True, default="tanh")
    frozen: bool = eqx.field(static=True, default=False)

    def __check_init__(self) -> None:
        if self.activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown activation {self.activation!r}; known activations are "
                f"{sorted(_ACTIVATIONS)}"
            )
        if jnp.ndim(self.value) != 1:
            raise ValueError(
                f"LearnedVector holds one unbatched vector, got shape {jnp.shape(self.value)}"
            )

    @classmethod
    def zeros(cls, dim: int, *, activation: str = "tanh", frozen: bool = False) -> "LearnedVector":
        """The zero pre-activation — hence, under ``tanh``, the zero output."""
        return cls(value=jnp.zeros((int(dim),), jnp.float32), activation=activation, frozen=frozen)

    @property
    def dim(self) -> int:
        return int(jnp.shape(self.value)[0])

    @property
    def learnable(self) -> bool:
        """Capability: a non-frozen vector is a trainable block."""
        return not self.frozen

    def __call__(self) -> Array:
        """The value, activated — shape ``(dim,)``. No parents: it is a constant."""
        return _ACTIVATIONS[self.activation](self.value)

    def init_opt_state(self):
        return init_opt_state(self)

    def absorb(self, stats, opt_state, lr: float | Array = 1.0, decay: float | Array = 1.0):
        return apply_gradient(self, stats, opt_state, lr=lr, decay=decay)
