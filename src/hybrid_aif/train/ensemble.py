"""``fit_ensemble``: Gaussian NLL for an ensemble of predictive heads.

The trainer behind the two disagreement rows (``ensemble_disagreement`` and
``model_uncertainty``): each ``MLPConditional`` member is fit to a code factor's
successor at the parents the row evaluates it at::

    L = Σ_i −log N( code_{t+1} | mu_i(carry_t, code_t, a_t), sigma_i )

One adam over all members (:data:`DEFAULT_EPS`, :data:`DEFAULT_CLIP`); the loss is
a sum over members. The posterior states arrive as data; the world model is not
differentiated here.

Design: docs/design/amortization.md A6, A12 chunk 2d §(b)
"""

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from hybrid_aif.distributions import MLPConditional
from hybrid_aif.distributions.neural import adam, apply_gradient
from hybrid_aif.system import LearnState, System, SystemParams

#: The ensemble optimizer's defaults: adam at ``lr 1e-4`` with ``eps 1e-5`` (not
#: optax's ``1e-8``) and a global-norm ``clip 100`` — the info-gain optimizer group.
DEFAULT_LR = 1e-4
DEFAULT_EPS = 1e-5
DEFAULT_CLIP = 100.0


#: The two rows this trainer serves (one estimator under opposite signs); the loss is
#: the same for either.
ROWS: tuple[str, ...] = ("ensemble_disagreement", "model_uncertainty")

#: The default row: the one the ``epistemic`` selector reaches.
ROW = "ensemble_disagreement"


def state_key(row: str = ROW) -> str:
    """A row's ``LearnState`` / ``lr_by_slot`` address: ``scoring:<row>``."""
    return f"scoring:{row}"


class EnsembleLosses(NamedTuple):
    """The ensemble's loss: the sum over members, and each member's own.

    ``per_member`` is the unscaled NLL; ``weights`` the ``(M,)`` vector each
    member's loss was folded with (ones unless ``member_weights`` was given).
    """

    total: Array
    per_member: Array
    weights: Array = ()  # (M,) the per-member loss weights the total was folded with


class EnsembleBatch(NamedTuple):
    """The flat ``(N, ...)`` batch the members are pointwise functions of.

    ``deter`` ``(N, deter_dim)``, ``code`` ``(N, flat_code)`` (the current code),
    ``action`` ``(N, action_dim)``, and ``target`` ``(N, flat_code)`` (the next
    code). A ``(B, T)`` batch becomes ``B·(T−1)`` examples (:func:`transitions`).
    """

    deter: Array
    code: Array
    action: Array
    target: Array

    def check(self, members: Sequence[MLPConditional]) -> None:
        """Validate against the members' declared parent and child widths."""
        if not members:
            raise ValueError("fit_ensemble: the ensemble is empty")
        widths = (self.deter.shape[-1], self.code.shape[-1], self.action.shape[-1])
        wrong = {i: m.parent_dims for i, m in enumerate(members) if m.parent_dims != widths}
        if wrong:
            raise ValueError(
                f"ensemble member(s) {wrong} declare those parent widths but this batch "
                f"supplies {widths} — [carry, flat code, action]"
            )
        bad_child = {i: m.child_dim for i, m in enumerate(members) if m.child_dim != widths[1]}
        if bad_child:
            raise ValueError(
                f"ensemble member(s) {bad_child} predict that many numbers but the target is "
                f"the next code, flattened: {widths[1]}"
            )
        n = self.deter.shape[0]
        shapes = {name: getattr(self, name).shape for name in EnsembleBatch._fields}
        if any(shape[0] != n or len(shape) != 2 for shape in shapes.values()):
            raise ValueError(f"EnsembleBatch wants flat (N, dim) columns, got {shapes}")


def transitions(deter: Array, code: Array, actions: Array) -> EnsembleBatch:
    """A batch of posterior states as the ensemble's ``(s_t, a_t) -> s_{t+1}`` examples.

    ``deter`` and ``code`` have leaves ``(B, T, ...)``; ``actions`` is
    ``(B, T, action_dim)`` in the replay convention (``actions[t]`` is ``a_{t−1}``),
    so the action moving ``s_t`` to ``s_{t+1}`` is read from slot ``t+1``.
    """
    deter, code = jnp.asarray(deter), jnp.asarray(code)
    b, t = deter.shape[0], deter.shape[1]
    if t < 2:
        raise ValueError(f"transitions: needs at least 2 steps per sequence, got T={t}")
    flat = code.reshape(b, t, -1)
    return EnsembleBatch(
        deter=deter[:, :-1].reshape(b * (t - 1), -1),
        code=flat[:, :-1].reshape(b * (t - 1), -1),
        action=jnp.asarray(actions)[:, 1:].reshape(b * (t - 1), -1),
        target=flat[:, 1:].reshape(b * (t - 1), -1),
    )


def ensemble_loss(
    members: Sequence[MLPConditional],
    batch: EnsembleBatch,
    *,
    member_weights: Array | None = None,
) -> tuple[Array, EnsembleLosses]:
    """``(total, EnsembleLosses)``: the summed member NLL, plus the per-member vector.

    Pure and differentiable in ``members``; each member's NLL is
    :meth:`~hybrid_aif.distributions.MLPConditional.diag_nll` (the diagonal form,
    which keeps gradients finite under jit). ``member_weights`` is an ``(M,)``
    weight per member (default ones; a zero removes that member's gradient); any
    other shape raises ``ValueError``. The reduction is a sum, which matters where
    adam is not scale-free (``eps`` and the clip).
    """
    batch.check(members)

    def one(member: MLPConditional) -> Array:
        return jnp.mean(
            jax.vmap(lambda d, c, a, y: member.diag_nll(y, (d, c, a)))(
                batch.deter, batch.code, batch.action, batch.target
            )
        )

    per_member = jnp.stack([one(member) for member in members])
    weights = (
        jnp.ones((len(members),), jnp.float32)
        if member_weights is None
        else jnp.asarray(member_weights, jnp.float32)
    )
    if weights.shape != (len(members),):
        raise ValueError(
            f"member_weights must be one weight per member, ({len(members)},); got {weights.shape}"
        )
    total = jnp.sum(weights * per_member)
    return total, EnsembleLosses(total=total, per_member=per_member, weights=weights)


def fit_ensemble(
    system: System,
    params: SystemParams,
    learn_state: LearnState | None,
    batch: EnsembleBatch,
    *,
    row: str | None = None,
    member_weights: Array | None = None,
    lr: float | Array = DEFAULT_LR,
    lr_by_slot: Mapping[str, float | Array] | None = None,
    clip: float | None = DEFAULT_CLIP,
    adam_eps: float = DEFAULT_EPS,
) -> tuple[SystemParams, LearnState, EnsembleLosses]:
    """One gradient step of ensemble learning: ``(params, learn_state, losses)``.

    Pure, jit-able, resumable; steps only its row's ``scoring:<row>`` address and
    threads every other ``LearnState`` entry through. ``row`` names which of
    :data:`ROWS` to step; ``None`` resolves the one ``SystemParams`` carries and
    raises if it carries both. ``member_weights`` goes to :func:`ensemble_loss`.
    ``clip`` is a global-norm clip over all members (``None`` disables it);
    ``adam_eps`` builds the optimizer. ``lr_by_slot`` is the rate grammar, a
    static ``0.0`` skipping the update at trace time.

    Raises ``ValueError`` when ``learn_state`` is ``None`` or lacks this row's
    optimizer entry (never re-seeded), and when ``row`` cannot be resolved.
    """
    row = _resolve_row(params, row)
    #  `address`: the LearnState / lr_by_slot slot name.
    address = state_key(row)
    members = tuple(params.scoring[row])
    optimizer = adam(adam_eps)  # per-slot adam with this trainer's eps, not optax's default
    opt = _resolve(learn_state, members, address, optimizer)
    rates = system._resolve_rates(lr_by_slot, "lr_by_slot", params)
    rate = lr if rates is None else rates.get(address, lr)

    grads, out = eqx.filter_grad(
        lambda ms: ensemble_loss(ms, batch, member_weights=member_weights), has_aux=True
    )(members)
    if clip is not None:
        clipper = optax.clip_by_global_norm(clip)
        grads, _ = clipper.update(grads, clipper.init(grads))
    if isinstance(rate, int | float) and float(rate) == 0.0:
        return params, LearnState(opt_states=opt, moments=_moments(learn_state)), out

    updated, opt[address] = apply_gradient(
        members, grads, opt[address], lr=rate, optimizer=optimizer
    )
    return (
        system._write_slots(params, {address: updated}),
        LearnState(opt_states=opt, moments=_moments(learn_state)),
        out,
    )


def _resolve_row(params: SystemParams, row: str | None) -> str:
    """Which of :data:`ROWS` to step: the named one, or the one that is there."""
    if row is not None:
        if row not in ROWS:
            raise ValueError(f"fit_ensemble: row must be one of {list(ROWS)}, got {row!r}")
        if row not in params.scoring:
            raise ValueError(
                f"fit_ensemble: SystemParams.scoring carries no {row!r} entry (it holds "
                f"{sorted(params.scoring)})"
            )
        return row
    present = [name for name in ROWS if name in params.scoring]
    if not present:
        raise ValueError(
            "fit_ensemble: SystemParams.scoring carries neither of the ensemble rows "
            f"{list(ROWS)} (it holds {sorted(params.scoring)}); the ensemble's parameters live "
            "in the scoring slot under its estimator's own name (design: amortization.md A12 "
            "item 1)"
        )
    if len(present) > 1:
        raise ValueError(
            f"fit_ensemble: this SystemParams carries both ensemble rows {present}, which are "
            "two independent ensembles (one estimator, opposite signs — curiosity under "
            "'infogain', pessimism under 'model_uncertainty'), so which to step is not "
            "inferable. Name it: fit_ensemble(..., row='model_uncertainty'), once per row."
        )
    return present[0]


def _resolve(
    learn_state: LearnState | None, members: tuple, key: str, optimizer
) -> dict[str, object]:
    if learn_state is None:
        raise ValueError(
            "fit_ensemble needs its optimizer carry: pass "
            "learn_state=system.init_learn_state(params) on the first call and thread the "
            "returned LearnState through thereafter. None does not silently re-initialize "
            "adam (design: amortization.md A2's explicit-init rule)."
        )
    opt = dict(learn_state.opt_states)
    if key not in opt:
        raise ValueError(
            f"fit_ensemble: learn_state.opt_states is missing {key!r} (present: {sorted(opt)}). "
            "system.init_learn_state(params) seeds it for every mounted ensemble row, so this "
            "LearnState was built by hand or lost the entry between trainers. A missing state "
            "is not re-seeded here: restarting adam mid-run silently "
            "resets its moments every step."
        )
    return opt


def _moments(learn_state: LearnState | None) -> dict[str, object]:
    return {} if learn_state is None else dict(learn_state.moments)
