"""Description length: what a model costs, in nats, as a chain of codes from equilibrium.

A model's length is a sum of links (``docs/design/description-length.md`` D1): a **graph** code
over its canonical structure (how many factors, modalities and control nodes, which family
each record takes from the fixed menu, how many levels or what width each variable has, which
parents each record reads; D2), each family's **parametric complexity** at a resolution ``n``
(Rissanen's ``k/2 · log(n/2π)`` plus the log of the room the family's base admits, closed form
for a table as the simplex's Fisher-Rao volume; D3), the **prior** link (the KL of the declared
prior from the family's base: the Jeffreys prior for a table, and for a Gaussian family, whose
flat base is improper, a reference prior of unit width the caller may override; D4), and the
**posterior** link where a posterior is given beside its prior (D5). Given a replay of a trace,
the length is the model's and the trace's together: the model part plus the data part (D6),
the replay's free energy summed over the trace, each address's parameters paid for once.

The measure is computed on the declaration ``build`` lowers a document to, so names, order,
placement and spelling cannot move it. ``structure`` is the graph and family links, functions of
the structure and ``n`` alone; ``parameters`` is the prior and posterior links, functions of the
arrays. Every link is reported per address with an ``exact`` flag, and a link the model gives
no closed form for is ``None``, unscored rather than zero. Nats throughout; :attr:`bits` is a
view. Eager NumPy, not for a jitted body.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, NamedTuple

import equinox as eqx
import jax
import numpy as np
from scipy.special import digamma, gammaln, multigammaln

from hybrid_aif import description as _description
from hybrid_aif import distributions as _dist
from hybrid_aif import factors as _factors
from hybrid_aif.system import System, SystemParams

_LOG_2PI = math.log(2.0 * math.pi)
_LOG_2 = math.log(2.0)
#: Rissanen's normaliser for the universal integer code, ``Σ_n 2^{-log₂* n}``.
RISSANEN_C0 = 2.865064

#: The families a record under each generative label may take: the menu the graph code prices
#: a family from, one uniform code per label.
MENUS = {
    label: tuple(name for name, fam in _description.FAMILIES.items() if label in fam.labels)
    for label in ("initial", "transition", "likelihood")
}


# ================================================================== the codes (D2)


def log_star(count: int) -> float:
    """The universal integer code for ``count ≥ 1`` in nats: Rissanen's ``log* n`` (the iterated
    binary logarithm's positive terms plus ``log₂ c₀``), computed in bits and converted."""
    if count < 1:
        raise ValueError(f"log_star codes a count of at least one, got {count}")
    bits = math.log2(RISSANEN_C0)
    term = math.log2(count)
    while term > 0.0:
        bits += term
        term = math.log2(term)
    return bits * _LOG_2


def count_code(count: int, cap: int | None = None) -> float:
    """A count's code: ``log cap`` under a declared cap (the room declared, whichever count is
    live), else ``log* count``. A count that may be zero is coded as ``count + 1``."""
    if cap is not None:
        return math.log(cap)
    return log_star(count)


def parent_set_code(size: int, candidates: int) -> float:
    """``log*(|parents| + 1) + log C(P, |parents|)``: which of ``candidates`` a record reads."""
    return (
        log_star(size + 1)
        + math.lgamma(candidates + 1)
        - math.lgamma(size + 1)
        - math.lgamma(candidates - size + 1)
    )


def family_code(label: str) -> float:
    """``log |menu|``: one family from the menu a record under ``label`` may take."""
    return math.log(len(MENUS[label]))


# ============================================================ the family costs (D3, D4)


def resolution_term(k: int, n: float) -> float:
    """``k/2 · log(n/2π)``: Rissanen's count of distributions a ``k``-parameter family tells
    apart at resolution ``n``."""
    return 0.5 * k * math.log(n / (2.0 * math.pi))


def simplex_log_volume(levels: int) -> float:
    """``log ∫ √det I`` over the simplex of ``levels`` outcomes: ``log(π^{L/2} / Γ(L/2))``, the
    Jeffreys Dirichlet's normaliser."""
    return 0.5 * levels * math.log(math.pi) - math.lgamma(0.5 * levels)


def tabular_family_cost(levels: int, n: float) -> float:
    """One categorical over ``levels`` outcomes at resolution ``n``: ``(L−1)/2 · log(n/2π)`` plus
    the simplex's log Fisher-Rao volume."""
    return resolution_term(levels - 1, n) + simplex_log_volume(levels)


def dirichlet_prior_link(concentration: np.ndarray) -> float:
    """The prior link of a table declared under ``Dir(α)`` along the last axis, summed over the
    rows: ``KL(Dir(α) ‖ Dir(½, …, ½))``, what the declared prior holds away from the family's
    Jeffreys base. A sharper prior costs more; the family link prices the room (D1, D4)."""
    a = np.asarray(concentration, np.float64)
    a0 = a.sum(axis=-1)
    half = np.full(a.shape[-1], 0.5)
    log_b = gammaln(a).sum(axis=-1) - gammaln(a0)
    log_b_half = gammaln(half).sum() - gammaln(half.sum())
    kl = log_b_half - log_b + ((a - half) * (digamma(a) - digamma(a0)[..., None])).sum(-1)
    return float(kl.sum())


def dirichlet_fisher_rao_entropy(concentration: np.ndarray) -> float:
    """``h(p) = −∫ p log(p / √det I)`` for ``Dir(α)`` along the last axis: the Dirichlet's entropy
    plus ``−½ Σ_i E[log θ_i]``, the prior's entropy measured in Fisher-Rao volume (D1)."""
    a = np.asarray(concentration, np.float64)
    a0 = a.sum(axis=-1)
    levels = a.shape[-1]
    log_b = gammaln(a).sum(axis=-1) - gammaln(a0)
    entropy = log_b + (a0 - levels) * digamma(a0) - ((a - 1.0) * digamma(a)).sum(-1)
    e_log_theta = (digamma(a) - digamma(a0)[..., None]).sum(-1)
    return float((entropy - 0.5 * e_log_theta).sum())


def _log_det(spd: np.ndarray) -> float:
    return float(np.linalg.slogdet(spd)[1])


def _inverse_wishart_moments(psi: np.ndarray, nu: float) -> tuple[float, float]:
    """``(H(IW), E[log|R|])`` for ``R ~ IW(Ψ, ν)``: the entropy is the log normaliser plus
    ``(ν + d + 1)/2 · E[log|R|] + νd/2``, and
    ``E[log|R|] = log|Ψ| − d log 2 − Σ_i ψ((ν − i)/2)``. The closed form is the entropy consistent
    with the density; it differs from ``scipy.stats.invwishart.entropy`` by ``(d + 1) log 2`` for
    ``d ≥ 2``."""
    psi = np.asarray(psi, np.float64)
    d = psi.shape[-1]
    e_log_det = _log_det(psi) - d * _LOG_2 - sum(digamma(0.5 * (nu - i)) for i in range(d))
    log_z = multigammaln(0.5 * nu, d) + 0.5 * nu * d * _LOG_2 - 0.5 * nu * _log_det(psi)
    return float(log_z + 0.5 * (nu + d + 1.0) * e_log_det + 0.5 * nu * d), float(e_log_det)


def inverse_wishart_fisher_rao_entropy(psi: np.ndarray, nu: float) -> float:
    """``h(p)`` for ``R ~ IW(Ψ, ν)`` as the covariance of a ``d``-dimensional Gaussian with a
    known mean: the inverse-Wishart's entropy plus ``E[log √det I]``, the Fisher-Rao density of
    the covariance family being ``2^{−d/2} |R|^{−(d+1)/2}``."""
    d = np.asarray(psi).shape[-1]
    entropy, e_log_det = _inverse_wishart_moments(psi, nu)
    return entropy - 0.5 * (d + 1.0) * e_log_det - 0.5 * d * _LOG_2


def inverse_gamma_fisher_rao_entropy(alpha: np.ndarray, beta: np.ndarray) -> float:
    """``h(p)`` for independent ``σ²_i ~ IG(α_i, β_i)``, each the variance of a one-dimensional
    Gaussian with a known mean (Fisher-Rao density ``2^{−1/2} / σ²``), summed over the
    dimensions."""
    a, b = np.asarray(alpha, np.float64), np.asarray(beta, np.float64)
    entropy = a + np.log(b) + gammaln(a) - (1.0 + a) * digamma(a)
    e_log_var = np.log(b) - digamma(a)
    return float((entropy - e_log_var - 0.5 * _LOG_2).sum())


def gaussian_extent(mean_scale: np.ndarray, psi: np.ndarray, nu: float) -> float:
    """``h(p)`` for a normal-inverse-Wishart prior over a Gaussian's mean and covariance:
    ``R ~ IW(Ψ, ν)``, ``μ | R ~ N(m, R · v)`` with ``v = mean_scale`` (a ``1×1`` column
    covariance), the Gaussian family's Fisher-Rao density being ``2^{−d/2} |R|^{−(d+2)/2}``
    (D4)."""
    psi = np.asarray(psi, np.float64)
    d = psi.shape[-1]
    v = float(np.asarray(mean_scale, np.float64).reshape(()))
    h_iw, e_log_det = _inverse_wishart_moments(psi, nu)
    h_mean_given_r = 0.5 * d * (_LOG_2PI + 1.0) + 0.5 * d * math.log(v) + 0.5 * e_log_det
    return float(h_iw + h_mean_given_r - 0.5 * (d + 2.0) * e_log_det - 0.5 * d * _LOG_2)


def kl_inverse_wishart(psi1, nu1, psi0, nu0) -> float:
    """``KL(IW(Ψ₁, ν₁) ‖ IW(Ψ₀, ν₀))``: ``log Z₀ − log Z₁ + (ν₀ − ν₁)/2 · E₁[log|R|] − ν₁ d/2
    + ν₁/2 · tr(Ψ₀ Ψ₁⁻¹)``."""
    psi1, psi0 = np.asarray(psi1, np.float64), np.asarray(psi0, np.float64)
    d = psi1.shape[-1]
    _, e_log_det = _inverse_wishart_moments(psi1, nu1)

    def log_z(psi, nu):
        return multigammaln(0.5 * nu, d) + 0.5 * nu * d * _LOG_2 - 0.5 * nu * _log_det(psi)

    return float(
        log_z(psi0, nu0)
        - log_z(psi1, nu1)
        + 0.5 * (nu0 - nu1) * e_log_det
        - 0.5 * nu1 * d
        + 0.5 * nu1 * np.trace(np.linalg.solve(psi1, psi0))
    )


def kl_matrix_normal_inverse_wishart(m1, v_inv1, psi1, nu1, m0, v_inv0, psi0, nu0) -> float:
    """``KL`` between two matrix-normal inverse-Wishart priors over a ``d × p`` head and its
    noise: the inverse-Wishart KL plus the expected KL between the conditional matrix normals,
    ``½ [d · tr(V₀⁻¹ V₁) − dp + ν₁ · tr(V₀⁻¹ Δᵀ Ψ₁⁻¹ Δ) + d log|V₀| − d log|V₁|]`` with
    ``Δ = M₁ − M₀`` and ``E₁[R⁻¹] = ν₁ Ψ₁⁻¹``."""
    m1, m0 = np.asarray(m1, np.float64), np.asarray(m0, np.float64)
    v_inv1, v_inv0 = np.asarray(v_inv1, np.float64), np.asarray(v_inv0, np.float64)
    psi1 = np.asarray(psi1, np.float64)
    d, p = m1.shape[-2], m1.shape[-1]
    delta = m1 - m0
    v1 = np.linalg.inv(v_inv1)
    quad = np.trace(v_inv0 @ delta.T @ np.linalg.solve(psi1, delta))
    conditional = 0.5 * (
        d * np.trace(v_inv0 @ v1) - d * p + nu1 * quad - d * _log_det(v_inv0) + d * _log_det(v_inv1)
    )
    return kl_inverse_wishart(psi1, nu1, psi0, nu0) + float(conditional)


def kl_inverse_gamma(alpha1, beta1, alpha0, beta0) -> float:
    """``KL(IG(α₁, β₁) ‖ IG(α₀, β₀))`` per dimension, summed."""
    a1, b1 = np.asarray(alpha1, np.float64), np.asarray(beta1, np.float64)
    a0, b0 = np.asarray(alpha0, np.float64), np.asarray(beta0, np.float64)
    kl = (
        (a1 - a0) * digamma(a1)
        - gammaln(a1)
        + gammaln(a0)
        + a0 * (np.log(b1) - np.log(b0))
        + a1 * (b0 - b1) / b1
    )
    return float(kl.sum())


def reference_noise(d: int, scale: float) -> tuple[np.ndarray, float]:
    """The reference inverse-Wishart over a ``d``-dimensional noise covariance at ``scale``:
    ``Ψ₀ = scale² I``, ``ν₀ = d + 2``, the vaguest proper prior whose mean is ``scale² I``."""
    return scale * scale * np.eye(d), float(d + 2)


def reference_inverse_gamma(scale: float) -> tuple[float, float]:
    """The reference inverse-Gamma over one variance at ``scale``: the ``d = 1`` reference
    inverse-Wishart, ``IG(3/2, scale²/2)``."""
    return 1.5, 0.5 * scale * scale


def reference_head(d: int, p: int, scale: float):
    """The reference matrix-normal inverse-Wishart over a ``d × p`` head and its noise: a zero
    head with unit column covariance over the reference noise."""
    psi0, nu0 = reference_noise(d, scale)
    return np.zeros((d, p)), np.eye(p), psi0, nu0


# ================================================================== the report


class Links(NamedTuple):
    """One address's links in nats: ``graph`` (always exact), ``family`` at resolution ``n``,
    ``prior`` and ``posterior`` (``None`` where the model gives nothing to score); ``exact`` is
    whether every scored link is the exact quantity rather than a bound; ``note`` names the
    family and the prior the links were read from."""

    graph: float
    family: float | None
    prior: float | None
    posterior: float | None
    exact: bool
    n: float | None
    note: str

    @property
    def structure(self) -> float:
        return self.graph + (self.family or 0.0)

    @property
    def parameters(self) -> float:
        return (self.prior or 0.0) + (self.posterior or 0.0)


class DescriptionLength(NamedTuple):
    """A model's length, or a model's and a trace's together, in nats: ``structure`` (graph +
    family) and ``parameters`` (prior + posterior), the model part; ``data``, the data part
    (the trace's free energy summed along a replay; ``None`` for the model alone); ``total``,
    the model part plus the data part where there is one. ``per_address`` holds the links
    behind the model part, ``exact`` whether every address's are exact (the data part's own
    exactness is the replay's), ``unscored`` the
    ``(address, link)`` pairs reported ``None``. ``learned`` names the addresses a replay
    learned, whose family links the data part pays for instead (D6)."""

    structure: float
    parameters: float
    total: float
    per_address: Mapping[str, Links]
    exact: bool
    unscored: tuple[tuple[str, str], ...]
    data: float | None = None
    learned: tuple[str, ...] = ()

    @property
    def model(self) -> float:
        """The model part: ``structure + parameters``."""
        return self.structure + self.parameters

    @property
    def bits(self) -> tuple[float, float, float]:
        """``(structure, parameters, total)`` in bits; ``total`` includes the data part where
        there is one."""
        return (self.structure / _LOG_2, self.parameters / _LOG_2, self.total / _LOG_2)


# ================================================================== the walk


def _leaf_count(module) -> int:
    return int(sum(np.size(leaf) for leaf in jax.tree_util.tree_leaves(module)))


def _map_count(module) -> int:
    """The point parameters a user-written map holds: the size of each of its floating-point
    leaves (a Python float counts one), its integer and static fields being structure."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(module, eqx.is_inexact_array_like))
    return int(sum(np.size(leaf) for leaf in leaves))


def _child_mass(decl) -> np.ndarray | None:
    """Per child level, the largest weight any record entry gives it: a concentration for a
    Dirichlet, a probability for a table; ``None`` for a non-categorical record."""
    if isinstance(decl, _dist.ConditionalDirichlet):
        a = np.asarray(decl.concentration, np.float64)
        return np.moveaxis(a, 0, -1).reshape(-1, a.shape[0]).max(axis=0)
    if isinstance(decl, _dist.ConditionalCategorical):
        t = np.asarray(decl.log_table, np.float64)
        return np.exp(np.moveaxis(t, 0, -1).reshape(-1, t.shape[0]).max(axis=0))
    if isinstance(decl, _dist.Dirichlet):
        a = np.asarray(decl.concentration, np.float64)
        return a.reshape(-1, a.shape[-1]).max(axis=0)
    if isinstance(decl, _dist.Categorical):
        p = np.asarray(decl.probs, np.float64)
        return p.reshape(-1, p.shape[-1]).max(axis=0)
    return None


def _live_levels(initial, transition) -> np.ndarray | None:
    """Which levels of a categorical state are live: a level no record can put mass on (zero
    weight at the first step and zero probability or concentration as a transition's child) is
    dead room a cap declared (``structure-learning.md`` S3, S6). ``None`` for a state whose
    records are not categorical tables."""
    first, later = _child_mass(initial), _child_mass(transition)
    if first is None and later is None:
        return None
    live = np.zeros(first.shape if first is not None else later.shape, bool)
    for mass in (first, later):
        if mass is not None:
            live |= mass > 0.0
    return live


def _live_modes(masks, batch_shape) -> np.ndarray:
    """``(modes,)`` bool over a per-mode family's flattened parent configurations: a mode is live
    when every state parent's level in it is live. A dead level's mode is room, not a member of
    the family, so it takes no family or prior link, as a dead level's table rows take none."""
    keep = np.ones(batch_shape, bool)
    if len(masks) == len(batch_shape):
        for axis, mask in enumerate(masks):
            if mask is not None:
                shape = [1] * len(batch_shape)
                shape[axis] = -1
                keep = keep & np.reshape(mask, shape)
    return keep.reshape(-1)


def _lg_counts(decl) -> tuple[int, int]:
    """A linear-Gaussian observation's parameter counts: its head's and its noise's."""
    head = np.asarray(decl._head_value)
    d = head.shape[-2]
    noise = decl.noise
    if isinstance(noise, _dist.InverseGamma):
        k_noise = _field_count(noise.alpha)
    elif isinstance(noise, _dist.InverseWishart):
        k_noise = _batch(noise.Psi, 2) * (d * (d + 1) // 2)
    elif decl.noise_map:  # a point map: one factor per grid node
        k_noise = _batch(noise.values, 2) * (d * (d + 1) // 2)
    else:
        k_noise = _batch(noise, 2) * (d * (d + 1) // 2)
    return _field_count(head), k_noise


def _field_count(array) -> int:
    """The parameters one field holds: its size, which counts a per-mode field once per mode and
    a shared one once."""
    return int(np.size(array))


def _batch(array, event_ndims: int) -> int:
    """How many modes a field carries: the product of its leading axes."""
    return int(np.prod(np.shape(array)[:-event_ndims])) if np.ndim(array) > event_ndims else 1


def _gaussian_k(d: int) -> int:
    return d + d * (d + 1) // 2


def _size_code(domain: Mapping[str, Any]) -> float:
    """A variable's size from its record: ``log*`` of its levels, its width, and of each
    axis of a factored code."""
    if "levels" in domain:
        code = log_star(int(domain["levels"]))
        for axis in domain.get("shape", ()):
            code += log_star(int(axis))
        return code
    return log_star(int(domain["dim"]))


class _Walk:
    """One declaration's links, address by address. ``woken`` maps a factor to the dead levels a
    replay woke: a likelihood over the factor prices their heads and columns as live, since a
    woken level starts from what the declaration holds there."""

    def __init__(self, system, params, n, prior, scale: float, woken=None):
        self.system, self.params, self.prior = system, params, prior
        self.n, self.scale = n, scale
        table = system._table
        self.states = tuple(table.state_names)
        self.controls = tuple(table.control_names)
        self.mods = tuple(system.modality_names)
        self.out: dict[str, Links] = {}
        self.live = {
            f: _live_levels(params.initial_state_prior[f], params.transition[f])
            for f in range(len(self.states))
        }
        self.used = dict(self.live)
        for f, mask in (woken or {}).items():
            if self.live.get(f) is not None:
                self.used[f] = self.live[f] | mask

    def resolution(self, address: str) -> float | None:
        if isinstance(self.n, Mapping):
            value = self.n.get(address)
            return None if value is None else float(value)
        return None if self.n is None else float(self.n)

    def axis_masks(self, deps: tuple[int, ...], used: bool = False) -> tuple:
        """Per parent axis, the live levels of a state parent (with ``used``, the levels a replay
        woke too); ``None`` leaves the axis whole."""
        live = self.used if used else self.live
        return tuple(live.get(g) if g < len(self.states) else None for g in deps)

    def run(self) -> DescriptionLength:
        f_count, m_count, u_count = len(self.states), len(self.mods), len(self.controls)
        self.out["graph"] = Links(
            graph=log_star(f_count) + log_star(m_count + 1) + log_star(u_count + 1),
            family=None,
            prior=None,
            posterior=None,
            exact=True,
            n=None,
            note=f"{f_count} state factors, {m_count} modalities, {u_count} control nodes",
        )
        space = self.system._space(self.params) if self.controls else None
        for k, u in enumerate(self.controls):
            continuous = space.kinds[k] == _description.FLOAT
            self.out[f"control:{u}"] = Links(
                graph=log_star(int(space.sizes[k])),
                family=None,
                prior=None,
                posterior=None,
                exact=True,
                n=None,
                note="a continuous control node" if continuous else "a control node",
            )
        d_deps = self.system.D_deps
        for f, s in enumerate(self.states):
            deps_d = d_deps[f] if d_deps is not None else ()
            prior_d = self.prior.initial_state_prior[f] if self.prior is not None else None
            self.address(
                f"initial_state_prior:{s}",
                "initial",
                self.params.initial_state_prior[f],
                prior_d,
                (len(deps_d), f_count),
                self.axis_masks(deps_d),
                self.live[f],
                size_code=None,
            )
            deps = self.system.B_deps[f]
            prior_b = self.prior.transition[f] if self.prior is not None else None
            live = self.live[f]
            if live is not None and not live.all():
                size_code = count_code(int(live.sum()), cap=live.size)
            else:
                size_code = _size_code(_description._state_domain(self.params, f, s, ""))
            self.address(
                f"transition:{s}",
                "transition",
                self.params.transition[f],
                prior_b,
                (len(deps), f_count + u_count),
                self.axis_masks(deps),
                live,
                size_code=size_code,
            )
        for m, o in enumerate(self.mods):
            deps = self.system.A_deps[m]
            decl = self.params.likelihood[m]
            prior_a = self.prior.likelihood[m] if self.prior is not None else None
            self.address(
                f"likelihood:{o}",
                "likelihood",
                decl,
                prior_a,
                (len(deps), f_count + u_count),
                self.axis_masks(deps, used=True),
                None,
                size_code=_size_code(_description._observation_domain(decl, f"likelihood:{o}")),
                mode_masks=tuple(
                    mask
                    for g, mask in zip(deps, self.axis_masks(deps, used=True), strict=True)
                    if not (g < f_count and self.live[g] is None)
                ),
            )
        ordered = [self.out[key] for key in sorted(self.out)]
        structure = math.fsum(link.structure for link in ordered)
        parameters = math.fsum(link.parameters for link in ordered)
        unscored = tuple(
            (addr, name)
            for addr, link in self.out.items()
            for name in ("family", "prior", "posterior")
            if getattr(link, name) is None and not (addr == "graph" or addr.startswith("control:"))
        )
        return DescriptionLength(
            structure=structure,
            parameters=parameters,
            total=structure + parameters,
            per_address=dict(self.out),
            exact=all(link.exact for link in self.out.values()),
            unscored=unscored,
        )

    def address(
        self, key, label, decl, prior_decl, parents, masks, live, size_code, mode_masks=None
    ) -> None:
        """One record's links: ``parents`` is ``(how many it reads, how many it could read)``;
        ``mode_masks`` are the live levels of the parents a per-mode family is batched over
        (its discrete parents, a continuous one being the regressor)."""
        n = self.resolution(key)
        graph = family_code(label) + parent_set_code(*parents)
        if size_code is not None:
            graph += size_code
        family = prior = posterior = None
        exact = True
        note = type(decl).__name__
        if isinstance(decl, _dist.ConditionalCategorical | _dist.ConditionalDirichlet):
            table = _live_table(
                np.asarray(
                    decl.concentration
                    if isinstance(decl, _dist.ConditionalDirichlet)
                    else decl.log_table
                ),
                live,
                masks,
            )
            rows, levels = int(np.prod(table.shape[1:])), table.shape[0]
            if n is not None:
                family = rows * tabular_family_cost(levels, n)
            if isinstance(decl, _dist.ConditionalDirichlet):
                conc = np.moveaxis(table, 0, -1).reshape(-1, levels)
                prior = dirichlet_prior_link(conc)
                note += " under a Dirichlet prior"
                if isinstance(prior_decl, _dist.ConditionalDirichlet):
                    before = _live_table(np.asarray(prior_decl.concentration), live, masks)
                    posterior = _dirichlet_kl(conc, np.moveaxis(before, 0, -1).reshape(-1, levels))
            else:
                exact = False
                note += " (a point table: priced by its family alone)"
        elif isinstance(decl, _dist.Categorical | _dist.Dirichlet):
            arr = np.asarray(
                decl.concentration if isinstance(decl, _dist.Dirichlet) else decl.logits
            )
            levels = arr.shape[-1]
            keep = live if live is not None else np.ones(levels, bool)
            rows = int(np.prod(arr.shape[:-1])) if arr.ndim > 1 else 1
            if n is not None:
                family = rows * tabular_family_cost(int(keep.sum()), n)
            if isinstance(decl, _dist.Dirichlet):
                conc = arr.reshape(-1, levels)[:, keep]
                prior = dirichlet_prior_link(conc)
                note += " under a Dirichlet prior"
                if isinstance(prior_decl, _dist.Dirichlet):
                    before = np.asarray(prior_decl.concentration).reshape(-1, levels)[:, keep]
                    posterior = _dirichlet_kl(conc, before)
            else:
                exact = False
                note += " (a point distribution: priced by its family alone)"
        elif isinstance(decl, _dist.Gaussian):
            d = decl.loc.shape[-1]
            if n is not None:
                family = resolution_term(_gaussian_k(d), n)
            exact = False
            note += " (fixed; no prior over its parameters, so its extent is unscored)"
        elif isinstance(decl, _dist.MixtureGaussian):
            k_mix, d = decl.means.shape[-2], decl.means.shape[-1]
            if n is not None:
                family = resolution_term(k_mix - 1 + k_mix * _gaussian_k(d), n)
            exact = False
            note += " (singular; k/2 is an upper bound on the learning coefficient)"
        elif isinstance(decl, _factors.LinearGaussianTransition):
            d = decl.F.shape[-1]
            innovations = decl.innovations
            k = (
                _field_count(decl.F)
                + _field_count(innovations.loc)
                + _batch(innovations.chol, 2) * (d * (d + 1) // 2)
            )
            if getattr(decl, "G", None) is not None:
                k += _field_count(decl.G)
            if n is not None:
                family = resolution_term(k, n)
            exact = False
            note += " (a regression family: fixed, its extent unscored)"
        elif isinstance(decl, _factors.NonlinearGaussianTransition):
            d = decl.dim
            innovations = decl.innovations
            k = (
                _map_count(decl.f)
                + _field_count(innovations.loc)
                + _batch(innovations.chol, 2) * (d * (d + 1) // 2)
            )
            if n is not None:
                family = resolution_term(k, n)
            exact = False
            note += f" (a point map of {_map_count(decl.f)} parameters: fixed, its extent unscored)"
        elif isinstance(decl, _factors.LinearGaussianObservationMNIW):
            d, p = decl.M.shape[-2], decl.M.shape[-1]
            batch = decl.M.shape[:-2]
            keep = _live_modes(masks if mode_masks is None else mode_masks, batch)
            modes = int(keep.sum())
            k = d * p + d * (d + 1) // 2
            m0, v_inv0, psi0, nu0 = reference_head(d, p, self.scale)
            if n is not None:
                family = modes * resolution_term(k, n)
                if p == 1:
                    family += modes * gaussian_extent(1.0, psi0, nu0)
            prior = sum(
                kl_matrix_normal_inverse_wishart(m, v_inv, psi, float(nu), m0, v_inv0, psi0, nu0)
                for m, v_inv, psi, nu in zip(
                    np.asarray(decl.M).reshape(-1, d, p)[keep],
                    np.broadcast_to(np.asarray(decl.V_inv), (*batch, p, p)).reshape(-1, p, p)[keep],
                    np.broadcast_to(np.asarray(decl.Psi), (*batch, d, d)).reshape(-1, d, d)[keep],
                    np.broadcast_to(np.asarray(decl.nu), batch).reshape(-1)[keep],
                    strict=True,
                )
            )
            exact = p == 1
            note += self._base_note(p == 1)
        elif isinstance(decl, _factors.LinearGaussianObservation):
            noise = decl.noise
            d = np.asarray(decl._head_value).shape[-2]
            k_head, k_noise = _lg_counts(decl)
            if n is not None:
                family = resolution_term(k_head + k_noise, n)
            exact = False  # the head's room is unscored: a point, or a matrix normal whose row
            # covariance is external, so no closed form over it alone
            if decl.learnable_noise:
                if isinstance(noise, _dist.InverseGamma):
                    a0, b0 = reference_inverse_gamma(self.scale)
                    if n is not None:
                        family += _field_count(noise.alpha) * inverse_gamma_fisher_rao_entropy(
                            a0, b0
                        )
                    prior = kl_inverse_gamma(noise.alpha, noise.beta, a0, b0)
                else:
                    psi0, nu0 = reference_noise(d, self.scale)
                    modes = _batch(noise.Psi, 2)
                    if n is not None:
                        family += modes * inverse_wishart_fisher_rao_entropy(psi0, nu0)
                    prior = sum(
                        kl_inverse_wishart(psi, float(nu), psi0, nu0)
                        for psi, nu in zip(
                            np.asarray(noise.Psi).reshape(-1, d, d),
                            np.asarray(noise.nu).reshape(-1),
                            strict=True,
                        )
                    )
                head_note = (
                    "an independent matrix-normal head (its prior unscored)"
                    if decl.learnable_head
                    else "a point head (priced by its count alone)"
                )
                note += f" with {head_note} and a noise prior" + self._base_note(True)
            elif decl.learnable_head:
                note += " (an independent matrix-normal head: its prior is unscored)"
            else:
                note += " (a regression family, fixed: its extent is unscored)"
        elif isinstance(decl, _factors.DerivedInitial):
            family = 0.0
            note += " (nothing to count)"
        else:
            k = _leaf_count(decl)
            if n is not None:
                family = resolution_term(k, n)
            exact = False
            note += f" ({k} weights; k/2 is an upper bound on the learning coefficient)"
        self.out[key] = Links(graph, family, prior, posterior, exact, n, note)

    def _base_note(self, extent_scored: bool) -> str:
        base = f"KL from the reference prior at scale {self.scale:g}"
        if extent_scored:
            return f" ({base}; the room the reference admits in closed form)"
        return f" ({base}; a regression family, so the room the reference admits is unscored)"


def _live_table(table: np.ndarray, live, masks) -> np.ndarray:
    """A child-first table cut to its live child levels and its parents' live levels."""
    if live is not None:
        table = np.compress(live, table, axis=0)
    for axis, mask in enumerate(masks, start=1):
        if mask is not None and axis < table.ndim:
            table = np.compress(mask, table, axis=axis)
    return table


def _dirichlet_kl(after: np.ndarray, before: np.ndarray) -> float:
    """``KL(Dir(after) ‖ Dir(before))`` along the last axis, summed over the rows."""
    a, b = np.asarray(after, np.float64), np.asarray(before, np.float64)
    a0 = a.sum(-1)
    log_b = gammaln(a).sum(-1) - gammaln(a0)
    log_b_before = gammaln(b).sum(-1) - gammaln(b.sum(-1))
    kl = log_b_before - log_b + ((a - b) * (digamma(a) - digamma(a0)[..., None])).sum(-1)
    return float(kl.sum())


def resolution_from_evidence(evidence: Mapping[str, Any]) -> dict[str, float]:
    """The per-address resolution ``description_length`` takes as ``n``, read off
    ``System.log_evidence(prior, posterior)``: each scored slot's absorbed count (D1). A slot
    with no closed form or no positive count is left out, so its family link is unscored."""
    return {
        address: float(entry.count)
        for address, entry in evidence.items()
        if entry.count is not None and float(entry.count) > 0.0
    }


def _trace_resolution(system: System, valid: np.ndarray) -> dict[str, float]:
    """Each address's ``n`` along a trace (D1): the data its family is judged against there. A
    likelihood sees every valid tick (a lagged one every tick but an episode's first, which has
    no datum), a transition every valid tick but an episode's first, an initial prior one draw
    per episode (a row of the trace's leading axes with a valid tick). An address with no data
    along the trace is left out (its family link unscored)."""
    ticks = float(valid.sum())
    episodes = float(np.sum(np.any(valid, axis=-1)))
    states = tuple(system._table.state_names)
    lagged = {modality for modality, _ in system.lagged}
    n = {
        **{
            f"likelihood:{o}": ticks - (episodes if o in lagged else 0.0)
            for o in system.modality_names
        },
        **{f"transition:{s}": ticks - episodes for s in states},
        **{f"initial_state_prior:{s}": episodes for s in states},
    }
    return {address: count for address, count in n.items() if count > 0.0}


def _changed(start, end) -> bool:
    """Whether any leaf of ``end`` (with any leading axes) differs from ``start``'s, compared
    in ``end``'s dtype so that a cast alone is no change."""
    pairs = zip(jax.tree_util.tree_leaves(start), jax.tree_util.tree_leaves(end), strict=True)
    for a, b in pairs:
        b = np.asarray(b)
        if np.any(b != np.asarray(a).astype(b.dtype)):
            return True
    return False


def _unpaid_family(start, end, family: float | None, n: float | None) -> float | None:
    """What of an address's family link a learning replay did not pay for: nothing where every
    part of the declaration learned, and for a linear-Gaussian observation whose head and noise
    are separate parts, the resolution of the part that did not (``k/2 · log(n/2π)`` is linear
    in ``k``, so the parts' terms add)."""
    if (
        family is None
        or n is None
        or not isinstance(start, _factors.LinearGaussianObservation)
        or isinstance(start, _factors.LinearGaussianObservationMNIW)
    ):
        return None
    head_moved = _changed(start._head_value, end._head_value)
    noise_moved = _changed(start.noise, end.noise)
    k_head, _ = _lg_counts(start)
    if noise_moved and not head_moved:
        return resolution_term(k_head, n)
    if head_moved and not noise_moved:
        return family - resolution_term(k_head, n)
    return None


def description_length(
    model: Mapping[str, Any] | System,
    n: float | Mapping[str, float] | None = None,
    *,
    replay=None,
    params: SystemParams | None = None,
    arrays: Mapping[str, np.ndarray] | None = None,
    prior: SystemParams | None = None,
    reference_scale: float = 1.0,
) -> DescriptionLength:
    """A model's description length (D1–D4), or with ``replay`` the description length of the
    model and a trace together, its model part plus its data part (D1, D6). ``model`` is a
    description document (with its ``arrays`` file when its clamps are refs) or a
    :class:`System` with its ``params``.

    ``n`` is the resolution every family is priced at: one number for every candidate in a
    comparison, or a mapping from address to that address's own resolution, an address missing
    from it reported with its family link unscored; ``None`` leaves every family link
    unscored. ``prior`` is the params the model's posteriors were learned from, which fills the
    posterior link; without it a learned family's arrays are the prior link alone, the model at
    one moment. ``reference_scale`` is the width of the reference prior a location-scale
    family's prior link is measured from (D4): the one convention a Gaussian family needs, in
    the units the observations are declared in.

    ``replay`` is a :class:`~hybrid_aif.replay.Replay` of a trace from ``params`` (the model as
    declared). The data part is its free energy summed over the valid ticks, and each
    parameter is paid for once. A part of the declaration the replay learned (its parameters
    changed along it) is paid for in the free energy, each tick scored under what the ticks
    before it taught, and takes no family link; every other part, a point table or a slot the
    replay left as it found it, keeps its family link. An initial prior is read at the first
    tick only, before anything is learned, so it keeps its family link. A dead level the replay
    woke is priced in the likelihoods over its factor as a live one, its declared head or column
    being where the woken level started. ``n`` then defaults to each address's count along the
    trace (D1): a likelihood at the valid ticks, a transition at the valid ticks after each
    episode's first, an initial prior at the episodes; a modality absent at some ticks
    (``present``) is counted at all of them, so pass ``n`` for such a trace. A replay with
    leading axes (episodes under ``vmap``) is summed over them, and refused when it learned:
    each episode learns from the starting parameters and would pay for them again. ``prior`` is
    refused beside a replay: a learning replay's free energy already holds what the posterior
    link would price.

    ``exact`` and :attr:`DescriptionLength.bits`' first two entries speak for the model part;
    the data part is the replay's free energy, the exact evidence where the tick's posterior is
    exact and a bound elsewhere (``Replay``).
    """
    if isinstance(model, System):
        if params is None:
            raise TypeError("description_length(system, n, params=...) needs the params")
        system = model
    else:
        if params is not None:
            raise TypeError(
                "description_length(document, n) takes no params; the document holds them"
            )
        system, params = _description.build(model, arrays=arrays)
    if isinstance(reference_scale, bool) or not float(reference_scale) > 0.0:
        raise ValueError(
            "reference_scale is the reference prior's width and must be positive, got "
            f"{reference_scale!r}"
        )
    data, moved, woken = None, {}, {}
    if replay is not None:
        if prior is not None:
            raise ValueError(
                "description_length: prior= prices a posterior's link, and beside a replay the "
                "replay's free energy pays for what its learning put in; pass one or the other"
            )
        if replay.free_energy is None:
            raise ValueError("description_length: the replay carries no free energy")
        energy = np.asarray(replay.free_energy, np.float64)
        valid = np.broadcast_to(np.asarray(replay.valid, bool), energy.shape)
        if replay.params is not None:
            before = system._params_in(params, "description_length")
            after = system._params_in(replay.params, "description_length")
            states = tuple(system._table.state_names)
            slots = [
                (f"likelihood:{o}", before.likelihood[m], after.likelihood[m])
                for m, o in enumerate(system.modality_names)
            ] + [
                (f"transition:{s}", before.transition[f], after.transition[f])
                for f, s in enumerate(states)
            ]
            moved = {key: (a, b) for key, a, b in slots if _changed(a, b)}
            if moved and energy.ndim > 1:
                raise ValueError(
                    "description_length: a learning replay under leading axes learns each "
                    "episode from the starting parameters and would pay for them once per "
                    "episode; price each episode's replay on its own, or replay the episodes as "
                    "one stream where the model is meant to carry what it learned across them"
                )
            if energy.ndim == 1:
                for f in range(len(states)):
                    start = _live_levels(before.initial_state_prior[f], before.transition[f])
                    end = _live_levels(after.initial_state_prior[f], after.transition[f])
                    if start is not None and end is not None and np.any(end & ~start):
                        woken[f] = end & ~start
        data = float(np.sum(np.where(valid, energy, 0.0)))
        if n is None:
            n = _trace_resolution(system, valid)
    if n is not None:
        values = n.values() if isinstance(n, Mapping) else (n,)
        for value in values:
            if isinstance(value, bool | str) or not float(value) > 0.0:
                raise ValueError(
                    f"n is the resolution a family is priced at and must be positive, got {n!r}"
                )
    params = system._params_in(params, "description_length")
    if prior is not None:
        prior = system._params_in(prior, "description_length")
    out = _Walk(system, params, n, prior, float(reference_scale), woken=woken).run()
    if data is None:
        return out
    links = dict(out.per_address)
    for key, (start, end) in moved.items():
        link = links[key]
        kept = _unpaid_family(start, end, link.family, link.n)
        note = link.note + (
            " (its family link paid by the replay's free energy)"
            if kept is None
            else " (the part the replay learned paid by its free energy)"
        )
        links[key] = link._replace(family=kept, n=link.n if kept is not None else None, note=note)
    learned = tuple(moved)
    ordered = [links[key] for key in sorted(links)]
    structure = math.fsum(link.structure for link in ordered)
    parameters = math.fsum(link.parameters for link in ordered)
    return DescriptionLength(
        structure=structure,
        parameters=parameters,
        total=structure + parameters + data,
        per_address=links,
        exact=out.exact,
        unscored=tuple(u for u in out.unscored if not (u[1] == "family" and u[0] in learned)),
        data=data,
        learned=learned,
    )
