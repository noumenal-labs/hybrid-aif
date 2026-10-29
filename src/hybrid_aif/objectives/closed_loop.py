"""Closed-loop scoring of Gaussian targets: the linear-quadratic future self.

Under ``System(feedback="linear_quadratic")`` a plan is scored as the policy that follows it and
corrects deviations by linear feedback, ``u_τ = ū_τ + K_τ (x̂_τ − m̄_τ)``, with the gains of the
linear-quadratic regulator. The state at each step is then distributed ``N(m̄_τ, S_τ)``,
``S_τ = Σ_τ + C_τ``: ``m̄_τ`` the predictive mean, ``Σ_τ`` the epistemic chain's posterior
covariance, ``C_τ`` the spread the corrections leave on the estimate,

    C_{τ+1} = (F + G K)_τ C_τ (F + G K)_τᵀ + M_{τ+1},    M = Σ⁻ − Σ,    C_0 = 0,

and every covered Gaussian target is scored by its cross-entropy on it. The gains come from the
backward Riccati pass over ``W_τ = Tᵀ Λ T`` summed over the targets at ``τ``, with ``R_τ`` the
precision of the Gaussian targets on the continuous control nodes (zero without one, solved
through the pseudo-inverse).

The planning scan is per step and the gains need the whole trajectory, so the pass runs in two
parts. :func:`closed_loop` wraps the planning closures: the carry gains a :class:`Record` the
transition fills with each step's linearisation ``(F_τ, G_τ)`` and the three covariances of the
covered factors (predictive, before and after the step's observations), and the terminal terms
run the Riccati pass over the record, adding to each covered leaf its correction
``½ tr(Λ T (S_τ − Σ_pred,τ) Tᵀ)`` (the step terms score the open-loop cross-entropy on the
predictive belief) and to each control target's leaf its correction cost ``½ tr(R K_τ C_τ K_τᵀ)``.
The leaves keep their keys and planners sum them as they sum every leaf.

Design: docs/design/closed-loop-planning.md
"""

from collections.abc import Callable, Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import block_diag, cho_solve

from hybrid_aif.controls import (
    FLOAT,
    LEVEL,
    ControlSpace,
    ControlTable,
    ControlTarget,
    is_belief,
    node_entry,
)
from hybrid_aif.controls import lg_read as _lg_read
from hybrid_aif.distributions import (
    ConditionalGaussian,
    Gaussian,
    MLPConditional,
    TruncatedGaussian,
)
from hybrid_aif.distributions.gaussian import sigma_points
from hybrid_aif.factors import (
    LGObservation,
    LinearGaussianTransition,
    NonlinearGaussianTransition,
    lg_point,
)
from hybrid_aif.interval import Interval
from hybrid_aif.objectives.base import TermTree
from hybrid_aif.objectives.hybrid import (
    GAUSSIAN,
    UNSCENTED_KAPPA,
    BeliefPair,
    _lg_inputs,
    _nl_inputs,
)
from hybrid_aif.objectives.presence import Presence, presence_from_levels

#: The values ``System.feedback`` takes besides ``None``.
FEEDBACK = ("linear_quadratic",)


class _Target(NamedTuple):
    """A covered target: its leaf's term and key, the factor it reads, ``T`` (``None``: the
    whole factor) and ``Λ``, the availability column of the channel that carries it, whether it
    scores the last step only, the leaf's sign (``−1`` for a utility, a negated cost), and
    ``parents``, the control node columns a conditional target reads, under which ``Λ`` is
    per level, ``(*levels, d, d)``, and the pass takes each step's level's."""

    term: str
    key: str
    factor: int
    T: Array | None
    precision: Array
    column: int | None
    terminal: bool
    sign: float = 1.0
    parents: tuple[int, ...] = ()


class _Cost(NamedTuple):
    """A control target's precision ``R`` on its node's block of the joint input."""

    term: str
    key: str
    span: tuple[int, int]
    precision: Array
    terminal: bool


class Coverage(NamedTuple):
    """What the pass reads, fixed per context: the covered factors and their blocks of the
    joint state, the continuous control nodes and their blocks of the joint input, the
    targets and the control costs."""

    factors: tuple[int, ...]
    spans: tuple[tuple[int, int], ...]
    nodes: tuple[int, ...]
    node_spans: tuple[tuple[int, int], ...]
    targets: tuple[_Target, ...]
    costs: tuple[_Cost, ...]

    @property
    def dim(self) -> int:
        return self.spans[-1][1] if self.spans else 0

    @property
    def du(self) -> int:
        return self.node_spans[-1][1] if self.node_spans else 0


class Record(NamedTuple):
    """The per-step data the pass reads, one row per plan step, written by the transition:
    ``F`` and ``G`` the step's linearisation on the joint state and input, ``predictive``,
    ``prior`` and ``posterior`` the covered factors' covariances after the step (predict only,
    before and after its observations), ``levels`` the step's level of each discrete control
    node (``0`` for a continuous one), which a conditional target's precision is taken at, and
    ``t`` the next row."""

    t: Array
    F: Array
    G: Array
    predictive: Array
    prior: Array
    posterior: Array
    levels: Array


def _precision(target) -> Array:
    """The precision of a Gaussian (or, on a control node, a truncated Gaussian's underlying
    Gaussian, whose quadratic its log-density is on the box); of a conditional Gaussian, one
    per level, with its parent axes leading."""
    if isinstance(target, TruncatedGaussian):
        return jnp.diag(1.0 / jnp.asarray(target.scale) ** 2)
    chol = target.stacked_chol() if isinstance(target, ConditionalGaussian) else target.chol
    eye = jnp.broadcast_to(jnp.eye(target.dim, dtype=chol.dtype), chol.shape)
    return cho_solve((chol, True), eye)


def coverage(
    *,
    families: tuple[str, ...],
    B: tuple,
    A: tuple,
    A_deps: tuple[tuple[int, ...], ...],
    state_targets: Mapping[str, object],
    terminal_targets: Mapping[str, object],
    obs_targets: Mapping[str, object],
    control_targets: tuple[ControlTarget, ...],
    decomposition: str,
    factor_names: tuple[str, ...],
    modality_names: tuple[str, ...],
    space: ControlSpace,
    c_deps: Mapping[str, tuple[tuple[str, int], ...]] | None = None,
) -> Coverage:
    """The targets ``feedback="linear_quadratic"`` scores in closed loop, with its refusals.

    Covered: a Gaussian state or terminal target on a Gaussian-carried factor, and a Gaussian
    observation target on a linear-Gaussian channel observing one Gaussian-carried factor. A
    target on a mixture-carried factor (or a channel observing one), a mixture target, and a
    covered factor with a neural transition raise :class:`ValueError`. Categorical targets are
    not covered and keep their scoring. A per-step target conditional on control nodes
    (``c_deps``, a :class:`~hybrid_aif.distributions.ConditionalGaussian`) is covered with a
    precision per level, taken at each step's level."""
    facs, mods = factor_names, modality_names
    where = 'System(feedback="linear_quadratic")'
    targets: list[_Target] = []
    covered: set[int] = set()
    c_deps = dict(c_deps or {})

    def parents_of(name: str) -> tuple[int, ...]:
        return tuple(k for _, k in c_deps.get(name, ()))

    def check_factor(f: int, what: str) -> None:
        if families[f] == GAUSSIAN:
            if isinstance(B[f], MLPConditional):
                raise ValueError(
                    f"{where}: {what} sits on factor {facs[f]!r}, whose transition is neural "
                    "(MLPConditional); the closed-loop pass needs the step's linearisation of the "
                    "dynamics, which a linear-Gaussian or NonlinearGaussianTransition gives. "
                    "Declare one of those, or plan with feedback=None"
                )
            return
        raise ValueError(
            f"{where}: {what} sits on factor {facs[f]!r}, which is carried as a mixture; a "
            "closed-loop score needs a Riccati pass per component, which is not built (design: "
            "closed-loop-planning.md L4). Drop the target, or plan with feedback=None"
        )

    def check_gaussian(target, what: str, name: str = "") -> None:
        if name in c_deps and isinstance(target, ConditionalGaussian):
            return
        if not isinstance(target, Gaussian):
            raise ValueError(
                f"{where}: {what} is a {type(target).__name__}; the closed-loop pass weighs a "
                "target by its quadratic cost, which a Gaussian target has. Declare a Gaussian "
                "target, or plan with feedback=None"
            )

    for f, name in enumerate(facs[: len(families)]):
        scopes = (("state_targets", state_targets), ("terminal_targets", terminal_targets))
        for scope, given in scopes:
            if name not in given or families[f] == "categorical":
                continue
            what = f"{scope}[{name!r}]"
            check_factor(f, what)
            check_gaussian(given[name], what, "" if scope == "terminal_targets" else name)
            terminal = scope == "terminal_targets"
            targets.append(
                _Target(
                    term="terminal_risk" if terminal else "risk",
                    key=name,
                    factor=f,
                    T=None,
                    precision=_precision(given[name]),
                    column=None,
                    terminal=terminal,
                    parents=() if terminal else parents_of(name),
                )
            )
            covered.add(f)
    obs_term = "utility" if decomposition == "utility_infogain" else "risk"
    for m, decl in enumerate(A):
        if mods[m] not in obs_targets or not isinstance(decl, LGObservation):
            continue
        what = f"obs_targets[{mods[m]!r}]"
        f = A_deps[m][0]
        # a Gaussian emission over categorical parents, or a channel reading an action node
        if f >= len(families) or families[f] == "categorical":
            continue
        check_factor(f, what)
        if len(A_deps[m]) > 1:
            raise ValueError(
                f"{where}: {what} sits on a channel that selects its head by an action node; the "
                "closed-loop pass would take the head at each step's level, which is not built. "
                "Drop the target, or plan with feedback=None"
            )
        check_gaussian(obs_targets[mods[m]], what, mods[m])
        targets.append(
            _Target(
                term=obs_term,
                key=mods[m],
                factor=f,
                T=lg_point(decl).H,
                precision=_precision(obs_targets[mods[m]]),
                column=m,
                terminal=False,
                sign=-1.0 if obs_term == "utility" else 1.0,
                parents=parents_of(mods[m]),
            )
        )
        covered.add(f)

    factors = tuple(sorted(covered))
    spans, at = [], 0
    for f in factors:
        d = B[f].dim if isinstance(B[f], NonlinearGaussianTransition) else B[f].innovations.dim
        spans.append((at, at + d))
        at += d
    nodes = tuple(k for k, kind in enumerate(space.kinds) if kind == FLOAT)
    node_spans, at = [], 0
    for k in nodes:
        node_spans.append((at, at + space.sizes[k]))
        at += space.sizes[k]
    costs = []
    for ct in control_targets:
        if ct.target is None or ct.column not in nodes:
            continue
        lo_hi = node_spans[nodes.index(ct.column)]
        costs.append(
            _Cost(
                term="terminal_risk" if ct.terminal else "risk",
                key=ct.name,
                span=lo_hi,
                precision=_precision(ct.target),
                terminal=ct.terminal,
            )
        )
    return Coverage(factors, tuple(spans), nodes, tuple(node_spans), tuple(targets), tuple(costs))


# --------------------------------------------------------------------- linearisation


def _nonlinear_mean(B_f: NonlinearGaussianTransition, loc: Array, chol: Array, inputs) -> Array:
    """The propagated mean of ``f(x, controls)`` (before the innovation's offset), as
    :func:`~hybrid_aif.objectives.hybrid._nonlinear_moments` computes it: ``f`` at the joint
    mean under ``first_order``, the sigma points' weighted mean under ``unscented``."""
    d = loc.shape[-1]
    spans, at = [], d
    for _, mean, _ in inputs.beliefs:
        spans.append((at, at + mean.shape[-1]))
        at += mean.shape[-1]
    loc_z = jnp.concatenate([loc, *(mean for _, mean, _ in inputs.beliefs)], axis=-1)

    def g(z: Array) -> Array:
        controls = list(inputs.controls)
        for (pos, _, _), (lo, hi) in zip(inputs.beliefs, spans, strict=True):
            controls[pos] = z[lo:hi]
        return B_f.f(z[:d], tuple(controls))

    if B_f.propagation == "first_order":
        return g(loc_z)
    roots = [chol, *(root for _, _, root in inputs.beliefs)]
    root_z = block_diag(*roots) if len(roots) > 1 else chol
    points, weights = sigma_points(loc_z, root_z, kappa=UNSCENTED_KAPPA)
    return weights @ jax.vmap(g)(points)


def _nonlinear_jacobians(
    B_f: NonlinearGaussianTransition, q: Gaussian, inputs, positions: dict[int, int]
) -> tuple[Array, dict[int, Array]]:
    """``F`` and each continuous node's ``G`` block for a nonlinear factor: the Jacobians of
    its propagated mean in the belief's mean and in the control values (a clamp's value, a
    belief's mean). Under ``first_order`` they are ``f``'s Jacobians at the mean; under
    ``unscented``, the sigma-point averages of ``f``'s Jacobians, the statistical
    linearisation. ``positions`` maps each continuous node to its position among the factor's
    control parents."""
    believed = {pos: i for i, (pos, _, _) in enumerate(inputs.beliefs)}
    values = {
        k: inputs.beliefs[believed[pos]][1] if pos in believed else inputs.controls[pos]
        for k, pos in positions.items()
    }

    def mean_of(loc: Array, vals: dict[int, Array]) -> Array:
        controls = list(inputs.controls)
        beliefs = list(inputs.beliefs)
        for k, pos in positions.items():
            if pos in believed:
                i = believed[pos]
                beliefs[i] = (pos, vals[k], beliefs[i][2])
            else:
                controls[pos] = vals[k]
        return _nonlinear_mean(
            B_f, loc, q.chol, inputs._replace(controls=tuple(controls), beliefs=tuple(beliefs))
        )

    F, Gs = jax.jacfwd(mean_of, argnums=(0, 1))(q.loc, values)
    return F, Gs


def make_linearise(
    cov: Coverage,
    B: tuple,
    table: ControlTable,
    space: ControlSpace,
    dtype,
) -> Callable[[tuple, object], tuple[Array, Array]]:
    """``(qs, action) -> (F, G)``: the step's linearisation on the joint state of the covered
    factors and the joint input of the continuous control nodes, read at the pre-step
    predictive beliefs. A linear-Gaussian factor gives its ``F`` and ``G`` at the step's
    discrete levels; a nonlinear one, :func:`_nonlinear_jacobians`."""
    kinds, layout = space.kinds, space.layout
    d, du = cov.dim, cov.du
    reads = {
        f: _lg_read(table, kinds, f)
        for f in cov.factors
        if isinstance(B[f], LinearGaussianTransition)
    }

    def linearise(qs: tuple, action) -> tuple[Array, Array]:
        Fs, G = [], jnp.zeros((d, du), dtype)
        for f, (lo, hi) in zip(cov.factors, cov.spans, strict=True):
            q = qs[f].predictive if isinstance(qs[f], BeliefPair) else qs[f]
            entries = {k: node_entry(action, k, layout) for k, _ in table.control_parents[f]}
            B_f = B[f]
            if isinstance(B_f, NonlinearGaussianTransition):
                parents = table.control_parents[f]
                inputs = _nl_inputs(B_f, parents, kinds, entries)
                positions = {
                    k: pos for pos, (k, _) in enumerate(parents) if kinds and kinds[k] != LEVEL
                }
                F_f, G_f = _nonlinear_jacobians(B_f, q, inputs, positions)
                blocks = G_f
            else:
                inputs = _lg_inputs(B_f, reads[f], entries)
                F_f, G_all = inputs.F, inputs.G
                if inputs.trivial:
                    if F_f.ndim > 2:
                        F_f = F_f[..., 0, :, :]
                    if G_all is not None and G_all.ndim > 2:
                        G_all = G_all[..., 0, :, :]
                blocks, at = {}, 0
                for k in reads[f].floats:
                    blocks[k] = G_all[:, at : at + space.sizes[k]]
                    at += space.sizes[k]
            Fs.append(jnp.asarray(F_f, dtype))
            for k, block in blocks.items():
                a, b = cov.node_spans[cov.nodes.index(k)]
                G = G.at[lo:hi, a:b].add(jnp.asarray(block, dtype).reshape(hi - lo, b - a))
        return block_diag(*Fs), G

    return linearise


# ------------------------------------------------------------------------ the record


def _covariances(qs: tuple, cov: Coverage) -> tuple[Array, Array, Array]:
    """The covered factors' predictive, epistemic and conditioned covariances, block-diagonal
    on the joint state."""
    pairs = [qs[f] for f in cov.factors]
    return tuple(
        block_diag(*(getattr(p, member).cov for p in pairs))
        for member in ("predictive", "epistemic", "conditioned")
    )


def empty_record(cov: Coverage, horizon: int, num_nodes: int, dtype) -> Record:
    """A record of ``horizon`` zero rows, its next row the first."""
    d, du = cov.dim, cov.du
    zeros = jnp.zeros((horizon, d, d), dtype)
    return Record(
        t=jnp.asarray(0, jnp.int32),
        F=zeros,
        G=jnp.zeros((horizon, d, du), dtype),
        predictive=zeros,
        prior=zeros,
        posterior=zeros,
        levels=jnp.zeros((horizon, num_nodes), jnp.int32),
    )


# ---------------------------------------------------------------------- the pass


def _embed(cov: Coverage, target: _Target, levels: Array) -> tuple[Array, tuple[int, int]]:
    """``Tᵀ Λ_τ T`` of a target on its factor's block at each step (``(H, d, d)``), ``Λ_τ`` the
    level's precision under ``levels`` (``(H, K)``) for a conditional target, and the block."""
    span = cov.spans[cov.factors.index(target.factor)]
    T, precision = target.T, target.precision
    if target.parents:
        precision = precision[tuple(levels[:, k] for k in target.parents)]
    else:
        precision = jnp.broadcast_to(precision, (levels.shape[0], *precision.shape))
    weight = precision if T is None else jnp.einsum("ji,tjk,kl->til", T, precision, T)
    return weight, span


def _weights(
    cov: Coverage, horizon: int, present: Array | None, levels: Array, dtype
) -> tuple[Array, Array]:
    """``W_τ`` (``(H, d, d)``, the state-cost weights at the state after step ``τ``) and
    ``R_τ`` (``(H, du, du)``) summed over the targets and control costs that score step ``τ``;
    an observation target counts where its channel is present, a conditional target with its
    level's precision at each step (``levels``)."""
    d, du = cov.dim, cov.du
    steps = jnp.arange(horizon)
    W = jnp.zeros((horizon, d, d), dtype)
    for target in cov.targets:
        weight, (lo, hi) = _embed(cov, target, levels)
        at = jnp.ones((horizon,), dtype)
        if target.terminal:
            at = (steps == horizon - 1).astype(dtype)
        elif target.column is not None and present is not None:
            at = present[:, target.column].astype(dtype)
        W = W.at[:, lo:hi, lo:hi].add(at[:, None, None] * weight)
    R = jnp.zeros((horizon, du, du), dtype)
    for cost in cov.costs:
        a, b = cost.span
        at = (steps == horizon - 1).astype(dtype) if cost.terminal else jnp.ones((horizon,), dtype)
        R = R.at[:, a:b, a:b].add(at[:, None, None] * cost.precision[None])
    return W, R


def riccati_gains(F: Array, G: Array, W: Array, R: Array) -> Array:
    """The finite-horizon regulator's gains ``K_τ`` (``(H, du, d)``) for ``s_{τ+1} = F_τ s_τ +
    G_τ u_τ`` with cost ``½ sᵀ W_τ s`` on the state after step ``τ`` and ``½ uᵀ R_τ u`` on the
    step's input::

        P_H = W_{H−1}
        K_τ = −(R_τ + G_τᵀ P_{τ+1} G_τ)⁺ G_τᵀ P_{τ+1} F_τ
        P_τ = W_{τ−1} + F_τᵀ P_{τ+1} (F_τ + G_τ K_τ)          (W_{−1} = 0)
    """
    horizon, d, du = G.shape
    if du == 0:
        return jnp.zeros((horizon, 0, d), F.dtype)
    W_before = jnp.concatenate([jnp.zeros_like(W[:1]), W[:-1]], axis=0)

    def back(P, xs):
        F_t, G_t, R_t, W_t = xs
        PG = P @ G_t
        K = -jnp.linalg.pinv(R_t + G_t.T @ PG, hermitian=True) @ (PG.T @ F_t)
        P_prev = W_t + F_t.T @ P @ (F_t + G_t @ K)
        return 0.5 * (P_prev + P_prev.T), K

    _, K = jax.lax.scan(back, W[-1], (F, G, R, W_before), reverse=True)
    return K


def closed_loop_terms(record: Record, cov: Coverage, present: Presence | None) -> TermTree:
    """The corrections the pass adds to the covered leaves, from a filled ``record``.

    The gains are held fixed under differentiation (``stop_gradient``): they minimise the
    closed-loop cost, so its derivative in the plan through them is zero (the envelope
    theorem) and the gradient of the sum of the corrected leaves is exact without
    differentiating the pseudo-inverse."""
    horizon = record.F.shape[0]
    mask = None if present is None else presence_from_levels(present, record.levels)
    W, R = _weights(cov, horizon, mask, record.levels, record.F.dtype)
    K = jax.lax.stop_gradient(riccati_gains(record.F, record.G, W, R))
    d = cov.dim

    def forward(C, xs):
        F_t, G_t, K_t, prior, post = xs
        KC = K_t @ C @ K_t.T
        A = F_t + G_t @ K_t
        C_next = A @ C @ A.T + (prior - post)
        C_next = 0.5 * (C_next + C_next.T)
        return C_next, (post + C_next, KC)

    _, (S, KCK) = jax.lax.scan(
        forward,
        jnp.zeros((d, d), record.F.dtype),
        (record.F, record.G, K, record.prior, record.posterior),
    )
    spread = S - record.predictive  # (H, d, d): what the corrections change on the state
    steps = jnp.arange(horizon)
    tree: dict[str, dict[str, Interval]] = {}

    def add(term: str, key: str, value: Array) -> None:
        leaves = tree.setdefault(term, {})
        iv = Interval.exact(value)
        leaves[key] = leaves[key] + iv if key in leaves else iv

    for target in cov.targets:
        weight, (lo, hi) = _embed(cov, target, record.levels)
        per_step = 0.5 * jnp.einsum("tij,tji->t", weight, spread[:, lo:hi, lo:hi])
        if target.terminal:
            per_step = jnp.where(steps == horizon - 1, per_step, 0.0)
        elif target.column is not None and mask is not None:
            per_step = per_step * mask[:, target.column].astype(per_step.dtype)
        add(target.term, target.key, target.sign * jnp.sum(per_step))
    for cost in cov.costs:
        a, b = cost.span
        per_step = 0.5 * jnp.einsum("ij,tji->t", cost.precision, KCK[:, a:b, a:b])
        if cost.terminal:
            per_step = jnp.where(steps == horizon - 1, per_step, 0.0)
        add(cost.term, cost.key, jnp.sum(per_step))
    return tree


def _merge(base: TermTree, extra: TermTree) -> TermTree:
    out = {term: dict(leaves) for term, leaves in base.items()}
    for term, leaves in extra.items():
        slot = out.setdefault(term, {})
        for key, iv in leaves.items():
            slot[key] = slot[key] + iv if key in slot else iv
    return out


def closed_loop(
    transition_fn,
    step_terms_fn,
    terminal_terms_fn,
    init_beliefs,
    *,
    cov: Coverage,
    B: tuple,
    table: ControlTable,
    space: ControlSpace,
    horizon: int,
    present: Presence | None,
    beliefs_of: Callable[[object], tuple],
):
    """Wrap the planning closures so the carry is ``(inner, record)``: the transition fills
    the :class:`Record`, the step terms read ``inner``, and the terminal terms add
    :func:`closed_loop_terms` to the inner terminal tree. ``beliefs_of`` reads the belief
    tuple from an inner carry."""
    first = beliefs_of(init_beliefs)[cov.factors[0]]
    dtype = (first.predictive if isinstance(first, BeliefPair) else first).loc.dtype
    linearise = make_linearise(cov, B, table, space, dtype)
    kinds, layout = space.kinds, space.layout
    by_target = {p for target in cov.targets for p in target.parents}
    by_presence = {} if present is None else {e.node: e for e in present.by_level}
    read = sorted(by_target | set(by_presence))

    def step_levels(action) -> Array:
        """The level at the step of each discrete node a conditional target or a per-level
        availability reads (``(K,)``, ``0`` elsewhere)."""
        levels = jnp.zeros((len(kinds),), jnp.int32)
        for k in read:
            entry = node_entry(action, k, layout)
            if is_belief(entry):
                if k in by_target:
                    what = "a conditional target's precision"
                    node = f"control node column {k}"
                else:
                    e = by_presence[k]
                    what = f"present[{e.modality_name!r}]"
                    node = f"control node {e.node_name!r}"
                raise ValueError(
                    f"{what} is taken at {node}'s level, and this step carries a probability "
                    "vector over its levels; the closed-loop pass reads a clamped level"
                )
            levels = levels.at[k].set(jnp.asarray(entry, jnp.int32))
        return levels

    def transition(carry, action):
        inner, rec = carry
        F, G = linearise(beliefs_of(inner), action)
        nxt = transition_fn(inner, action)
        pred, prior, post = _covariances(beliefs_of(nxt), cov)
        t = rec.t
        rec = Record(
            t=t + 1,
            F=rec.F.at[t].set(F),
            G=rec.G.at[t].set(G),
            predictive=rec.predictive.at[t].set(pred),
            prior=rec.prior.at[t].set(prior),
            posterior=rec.posterior.at[t].set(post),
            levels=rec.levels.at[t].set(step_levels(action)),
        )
        return nxt, rec

    def step_terms(carry, action, t, *rest):
        return step_terms_fn(carry[0], action, t, *rest)

    def terminal_terms(carry):
        inner, rec = carry
        base = {} if terminal_terms_fn is None else terminal_terms_fn(inner)
        return _merge(base, closed_loop_terms(rec, cov, present))

    record = empty_record(cov, horizon, len(kinds), dtype)
    return transition, step_terms, terminal_terms, (init_beliefs, record)


__all__ = [
    "FEEDBACK",
    "Coverage",
    "Record",
    "closed_loop",
    "closed_loop_terms",
    "coverage",
    "riccati_gains",
]
