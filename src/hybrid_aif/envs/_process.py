"""The positional generative processes behind :class:`~hybrid_aif.envs.GenerativeEnv` (internal).

:class:`DiscreteProcess` is the environment defined by its own discrete generative model
(pymdp's ``PymdpEnv``).

Params hold per-modality likelihood tensors ``A[m]`` shaped
``(num_obs_m, *[num_levels_f for f in A_deps[m]])``, per-factor transition
tensors ``B[f]`` shaped ``(num_levels_f, *[num_levels_g for g in B_deps[f]],
num_actions_f)``, and initial priors ``D[f]`` — the same tensor conventions as
``fpi``'s model representation. States and observations are drawn from
:class:`Categorical` heads built by indexing these tensors: the env is just
another system's generative model.

:class:`HybridProcess` is the hybrid sibling (design doc "as_env"):
its params hold the *declaration objects* of a hybrid System directly, and
sampling is ancestral per family in declaration order — categorical draws
first, then ``x' = F x + b_j + eps`` with ``j`` the discrete parents' sampled
values *this step*. Exact, no mixtures on the env side: the env holds a point
state (int32 scalars for categorical factors, float vectors for continuous).

Both speak positionally: observations, states and actions in node order, and
:class:`~hybrid_aif.envs.GenerativeEnvParams` with tuple containers (any object with
``likelihood`` / ``transition`` / ``initial_state_prior`` tuples will do). They are not exported;
the public face is :class:`~hybrid_aif.envs.GenerativeEnv`, which converts by name.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.controls import (
    LEVEL,
    ControlLayout,
    ControlStep,
    ControlTable,
    lg_read,
    node_entry,
    parent_levels,
)
from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalDirichlet,
    Dirichlet,
    Gaussian,
    MLPConditional,
)
from hybrid_aif.envs.base import Env, Obs, State
from hybrid_aif.factors import NonlinearGaussianTransition


def _reject_param_distribution(decl, slot: str, index: int) -> None:
    """Raise for a declaration that is a distribution over parameters.

    Use ``System.as_env(params, param_key=...)``, which lowers every learnable
    slot to its posterior mean (or one sampled table) first.
    """
    if isinstance(decl, Dirichlet | ConditionalDirichlet):
        raise ValueError(
            f"{slot} {index} is a {type(decl).__name__} — a distribution over parameters, "
            "not a sampleable declaration; lower it first via "
            "System.as_env(params, param_key=...)"
        )


def _null_levels(control_deps, num_factors: int) -> Array:
    """Level zeros: one per action node under a control table, else one per factor."""
    width = len(control_deps.controls) if isinstance(control_deps, ControlTable) else num_factors
    return jnp.zeros((width,), jnp.int32)


def _draw(key: Array, probs: Array) -> Array:
    return Categorical.from_probs(probs).sample(key).astype(jnp.int32)


class DiscreteProcess(Env):
    """Discrete POMDP generative process; dependency structure is static."""

    A_deps: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    B_deps: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    D_deps: tuple[tuple[int, ...], ...] | None = eqx.field(static=True, default=None)
    B_control_deps: "tuple[tuple[int, ...], ...] | ControlTable | None" = eqx.field(
        static=True, default=None
    )
    #   which action indexes each transition's control axes: a System's
    #   :class:`~hybrid_aif.controls.ControlTable` (the action one level per action node, read
    #   where its control node is a parent), or a per-factor declaration (``None`` = each
    #   factor's own action on its one axis)
    sensed: tuple[int, ...] = eqx.field(static=True, default=())
    #   the action columns a modality observes: ``A_deps`` index ``num_factors + i`` reads the
    #   action's column ``sensed[i]`` (the realised action of the step; level 0 at reset)

    @property
    def num_factors(self) -> int:
        return len(self.B_deps)

    def null_action(self) -> Array:
        """The no-op action: level zeros, one per action node under a control table."""
        return _null_levels(self.B_control_deps, self.num_factors)

    def _with_action(self, state: State, action: Array) -> State:
        """``state`` followed by the observed action columns."""
        if not self.sensed:
            return state
        return (*state, *(action[..., k] for k in self.sensed))

    @property
    def num_modalities(self) -> int:
        return len(self.A_deps)

    def reset(self, key: Array, params: Any, state: State | None = None) -> tuple[Obs, State]:
        """Ancestrally sample an initial state from ``D`` and observations of it.

        Conditional priors (``D_deps``) index their child-first table by the
        parents' already-sampled levels — a *joint* draw, not a product of
        marginals. ``state`` overrides the draw with a known point start — one
        int32 scalar per factor — so the agent's initial prior (the *belief*
        ``D``) and the process's true starting point can differ; observations
        are still sampled from the given state.
        """
        d_deps = self.D_deps if self.D_deps is not None else ((),) * self.num_factors
        if state is None:
            keys = jax.random.split(key, self.num_factors + 1)
            drawn: list[Array] = []
            for k, d, deps in zip(list(keys[1:]), params.initial_state_prior, d_deps, strict=True):
                probs = d[(slice(None), *(drawn[g] for g in deps))] if deps else d
                drawn.append(_draw(k, probs))
            state = tuple(drawn)
            null = self.null_action()
            return self._observe(keys[0], self._with_action(state, null), params), state
        if len(state) != self.num_factors:
            raise ValueError(f"state override has {len(state)} factors, env has {self.num_factors}")
        state = tuple(jnp.asarray(s, jnp.int32) for s in state)
        return self._observe(key, self._with_action(state, self.null_action()), params), state

    def step(
        self,
        key: Array,
        state: State,
        action: Array,
        params: Any,
        factors: tuple[bool, ...] | None = None,
    ) -> tuple[Obs, State]:
        """Sample the next state under ``action`` and an observation of it.

        ``factors`` masks the step: a factor whose entry is ``False`` keeps its value, and every
        transition reads the previous state, so a masked step is one sweep's worth of the
        factors it names (the driver's sequential and async orders step each component's
        factors at that component's turn). ``None`` steps every factor.
        """
        k_obs, k_state = jax.random.split(key)
        keys = jax.random.split(k_state, self.num_factors)
        cdeps = self.B_control_deps
        mask = (True,) * self.num_factors if factors is None else tuple(factors)
        state_next = tuple(
            _draw(
                k,
                b[
                    (
                        ...,
                        *parent_levels(tuple(state[g] for g in deps), action, f, cdeps, b.ndim - 1),
                    )
                ],
            )
            if mask[f]
            else state[f]
            for f, (k, b, deps) in enumerate(
                zip(list(keys), params.transition, self.B_deps, strict=True)
            )
        )
        return self._observe(k_obs, self._with_action(state_next, action), params), state_next

    def _observe(self, key: Array, state: State, params: Any) -> Obs:
        keys = jax.random.split(key, self.num_modalities)
        return tuple(
            _draw(k, a[(..., *[state[g] for g in deps])])
            for k, a, deps in zip(list(keys), params.likelihood, self.A_deps, strict=True)
        )


class HybridProcess(Env):
    """Hybrid generative process: ancestral sampling per family in declaration order.

    Point state per factor: categorical factors are int32 scalar indices,
    continuous factors float vectors (a MixtureGaussian prior contributes an
    ancestral component draw at reset; thereafter the state is a point).
    Observations mirror the evidence-tuple typing: int32 index per categorical
    modality, float array per LG modality.

    A :class:`~hybrid_aif.distributions.MLPConditional` declaration samples
    through the net, with the slot's parent order; ``as_env(params, param_key=...)``
    randomizes nothing on a neural slot. ``kinds``/``sizes`` give each action node's kind and
    width (its control node's): a discrete node's entry is a level, a continuous node's a float
    vector. With ``controls`` (every node continuous) ``action`` is the flat float vector; over a
    mixed space it is a :class:`~hybrid_aif.controls.ControlStep`, one entry per action node. A
    linear-Gaussian factor steps ``x' = F_j x + G_j u + b_j + ε``, its discrete parents (state
    factors and discrete control nodes) selecting ``j``; a nonlinear one ``x' = f(x, controls) +
    ε_j``, ``f`` given each control node's entry in declared order (a discrete node's level, a
    continuous node's value) and the discrete nodes selecting ``j``.
    """

    A_deps: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    B_deps: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    controlled: tuple[bool, ...] = eqx.field(static=True)
    D_deps: tuple[tuple[int, ...], ...] | None = eqx.field(static=True, default=None)
    controls: ControlLayout | None = eqx.field(static=True, default=None)
    B_control_deps: "tuple[tuple[int, ...], ...] | ControlTable | None" = eqx.field(
        static=True, default=None
    )
    #   as on :class:`DiscreteProcess`
    kinds: tuple[str, ...] = eqx.field(static=True, default=())
    sizes: tuple[int, ...] = eqx.field(static=True, default=())
    sensed: tuple[int, ...] = eqx.field(static=True, default=())
    #   the action columns a modality observes, as on :class:`DiscreteProcess`: ``A_deps`` index
    #   ``num_factors + i`` reads the level of the action's column ``sensed[i]``

    @property
    def num_factors(self) -> int:
        return len(self.B_deps)

    def _with_action(self, state: State, action: Array) -> State:
        """``state`` followed by the observed action columns' levels."""
        if not self.sensed:
            return state
        return (
            *state,
            *(jnp.asarray(node_entry(action, k, self.controls), jnp.int32) for k in self.sensed),
        )

    def null_action(self) -> Array:
        """The no-op action: level zeros, the flat float zeros under ``controls``, or a mixed
        space's entries of each (level 0, the zero vector)."""
        if self.controls is not None:
            return jnp.zeros((self.controls.action_dim,), jnp.float32)
        if self.kinds and any(k != LEVEL for k in self.kinds):
            return ControlStep(
                entries=tuple(
                    jnp.int32(0) if k == LEVEL else jnp.zeros((n,), jnp.float32)
                    for k, n in zip(self.kinds, self.sizes, strict=True)
                )
            )
        return _null_levels(self.B_control_deps, self.num_factors)

    @property
    def num_modalities(self) -> int:
        return len(self.A_deps)

    @property
    def _table(self) -> ControlTable:
        return self.B_control_deps

    def _kind(self, k: int) -> str:
        return self.kinds[k] if self.kinds else LEVEL

    def _size(self, k: int) -> int:
        return self.sizes[k]

    def reset(self, key: Array, params: Any, state: State | None = None) -> tuple[Obs, State]:
        """Sample an initial state from ``D`` (default) and observations of it.

        ``state`` overrides the draw with a known point start (int32 scalar per
        categorical factor, float vector per continuous factor), so the agent's
        prior ``D`` and the true start can differ; observations are still
        sampled from the given state.
        """
        for f, d in enumerate(params.initial_state_prior):
            _reject_param_distribution(d, "initial_state_prior factor", f)
        d_deps = self.D_deps if self.D_deps is not None else ((),) * self.num_factors
        if state is None:
            keys = jax.random.split(key, self.num_factors + 1)
            drawn: list[Array] = []
            for k, d, deps in zip(list(keys[1:]), params.initial_state_prior, d_deps, strict=True):
                if isinstance(d, ConditionalCategorical):
                    drawn.append(d.condition_on(tuple(drawn[g] for g in deps)).sample(k))
                else:
                    drawn.append(d.sample(k))
            state = tuple(drawn)
            null = self.null_action()
            return self._observe(keys[0], self._with_action(state, null), params), state
        if len(state) != self.num_factors:
            raise ValueError(f"state override has {len(state)} factors, env has {self.num_factors}")
        state = tuple(
            jnp.asarray(s, jnp.int32) if isinstance(d, Categorical) else jnp.asarray(s)
            for s, d in zip(state, params.initial_state_prior, strict=True)
        )
        observed = self._with_action(state, self.null_action())
        return self._observe(key, observed, params), state

    def step(
        self,
        key: Array,
        state: State,
        action: Array,
        params: Any,
        factors: tuple[bool, ...] | None = None,
    ) -> tuple[Obs, State]:
        """Sample the next state under ``action`` and an observation of it; ``factors`` masks
        the step as on :meth:`DiscreteProcess.step`, a kept factor being read as kept by any
        factor that reads this step's values."""
        k_obs, k_state = jax.random.split(key)
        keys = jax.random.split(k_state, self.num_factors)
        mask = (True,) * self.num_factors if factors is None else tuple(factors)
        next_state: list[Array] = []
        for f, (k, B_f, deps) in enumerate(
            zip(list(keys), params.transition, self.B_deps, strict=True)
        ):
            _reject_param_distribution(B_f, "transition factor", f)
            if not mask[f]:
                next_state.append(jnp.asarray(state[f]))
            elif isinstance(B_f, ConditionalCategorical):
                idx = parent_levels(
                    tuple(state[g] for g in deps), action, f, self.B_control_deps, B_f.parent_ndims
                )
                next_state.append(B_f.condition_on(idx).sample(k))
            elif isinstance(B_f, MLPConditional):
                # A1's contract member (ii): ancestral sampling through the net.
                # Parent order is the transition slot's — own previous value, then its
                # control nodes (a discrete node's one-hot, a continuous node's value) —
                # the same convention System.learn feeds the gradient and the planner
                # feeds prediction.
                parents_mlp = (jnp.asarray(state[f]),)
                for c, _ in self._table.control_parents[f]:
                    value = node_entry(action, c, self.controls)
                    parents_mlp += (
                        jax.nn.one_hot(value, self._size(c))
                        if self._kind(c) == LEVEL
                        else jnp.asarray(value),
                    )
                next_state.append(B_f.sample(k, parents_mlp))
            elif isinstance(B_f, NonlinearGaussianTransition):
                controls, levels = [], []
                for c, _ in self._table.control_parents[f]:
                    value = node_entry(action, c, self.controls)
                    if self._kind(c) == LEVEL:
                        value = jnp.asarray(value, jnp.int32)
                        levels.append(value)
                    else:
                        value = jnp.asarray(value, jnp.float32)
                    controls.append(value)
                innov = B_f.innovations.condition_on(tuple(levels) or (0,))
                next_state.append(B_f.f(jnp.asarray(state[f]), tuple(controls)) + innov.sample(k))
            else:
                # Discrete parents are read from *this step's* sampled values
                # (ancestral in declaration order), matching the planner's
                # within-step semantics; a discrete control node's level is a mode index.
                read = lg_read(self._table, self.kinds, f)
                parents: tuple[Array | int, ...] = tuple(
                    next_state[i] if what == "state" else node_entry(action, i, self.controls)
                    for what, i in read.axes
                ) or (0,)  # trivial (1,) parent axis
                innov = B_f.innovations.condition_on(parents)
                F, G = B_f.F, B_f.G
                index = tuple(jnp.asarray(p, jnp.int32) for p in parents)
                if F.ndim > 2:  # per-mode F_j, conditioned exactly as the innovations
                    F = F[index]
                x = (F @ state[f][..., None])[..., 0] + innov.sample(k)
                if read.floats:
                    if G.ndim > 2:
                        G = G[index]
                    u = jnp.concatenate(
                        [jnp.asarray(node_entry(action, c, self.controls)) for c in read.floats],
                        axis=-1,
                    )
                    x = x + (G @ u[..., None])[..., 0]
                next_state.append(x)
        observed = self._with_action(tuple(next_state), action)
        return self._observe(k_obs, observed, params), tuple(next_state)

    def _observe(self, key: Array, state: State, params: Any) -> Obs:
        keys = jax.random.split(key, self.num_modalities)
        obs: list[Array] = []
        for m, (k, A_m, deps) in enumerate(
            zip(list(keys), params.likelihood, self.A_deps, strict=True)
        ):
            _reject_param_distribution(A_m, "likelihood modality", m)
            if isinstance(A_m, ConditionalCategorical):
                obs.append(A_m.condition_on(tuple(state[g] for g in deps)).sample(k))
            elif isinstance(A_m, MLPConditional):
                # One parent per A_deps factor, in order: a continuous factor's
                # value, a categorical factor's one-hot (A1's parent convention).
                parents_mlp = tuple(
                    jax.nn.one_hot(state[g], d)
                    if jnp.issubdtype(jnp.asarray(state[g]).dtype, jnp.integer)
                    else jnp.asarray(state[g])
                    for g, d in zip(deps, A_m.parent_dims, strict=True)
                )
                obs.append(A_m.sample(k, parents_mlp))
            else:
                # Switching observation: the categorical parents named after the
                # observed factor select (H_j, c_j, R_j) from any per-mode array,
                # exactly as step() indexes a per-mode F_j by its discrete parents.
                H, c = A_m.H, A_m.c
                if H.shape[-1] == 0:
                    # a Gaussian emission over categorical parents alone: every parent selects
                    # the offset and the noise, and there is no state to map
                    R = A_m.R_chol
                    parents = tuple(jnp.asarray(state[g], jnp.int32) for g in deps)
                    loc = c[parents] if c.ndim > 1 else c
                    if R.ndim > 2:
                        R = R[parents]
                    obs.append(Gaussian(loc=loc, chol=R).sample(k))
                    continue
                # a noise map is read at the true state the observation is drawn from
                R = A_m.R_chol_at(jnp.asarray(state[deps[0]])) if A_m.noise_map else A_m.R_chol
                parents = tuple(jnp.asarray(state[g], jnp.int32) for g in deps[1:])
                if parents:
                    if H.ndim > 2:
                        H = H[parents]
                    if c.ndim > 1:
                        c = c[parents]
                    if R.ndim > 2:
                        R = R[parents]
                loc = (H @ state[deps[0]][..., None])[..., 0] + c
                obs.append(Gaussian(loc=loc, chol=R).sample(k))
        return tuple(obs)
