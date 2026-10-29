"""``lax.scan`` rollout driver connecting an :class:`Env` to a policy callable.

The policy side is a plain function ``policy_fn(key, history, t) -> action``
over a fixed-window :class:`History` (static shapes throughout, so the whole
driver jits). Slots left of the first real observation carry ``-1`` sentinels
with ``valid == False``; the window is right-aligned and slides right, pymdp's
``rollout.py`` (pymdp/envs/rollout.py) masked-buffer scheme. ``rollout`` vmaps one single-agent scan
across ``batch_size`` agents, one row of ``params`` per agent.

Observations, states and actions keep whatever structure the env gives them: the
name-keyed mappings of :meth:`System.as_env <hybrid_aif.system.System.as_env>`
(each modality buffered with its own dtype and shape, so continuous observations
roll out too), or a positional env's tuples and ``(num_factors,)`` action.
"""

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.envs.base import Env, Obs

if TYPE_CHECKING:
    from hybrid_aif.system import System


class History(eqx.Module):
    """Right-aligned fixed window; ``valid[i]`` marks slots holding real data.

    ``obs`` and ``actions`` have the env's structure with a leading ``(window_len,)``
    axis on every leaf, each in the leaf's own dtype. ``actions[i]`` is the action
    that produced ``obs[i]``. Invalid slots, and the reset slot's action, hold ``-1``.
    """

    obs: Any  # per modality (window_len, *shape), -1 sentinel
    actions: Any  # per action leaf (window_len, *shape), -1 sentinel
    valid: Array  # (window_len,) bool

    @property
    def window_len(self) -> int:
        return self.valid.shape[-1]


#: ``policy_fn(key, history, t) -> action``, in the env's action form: keyed by the action
#: nodes' names (``System.names.actions``) for a :class:`~hybrid_aif.envs.GenerativeEnv`, the
#: positional array for a positional env.
PolicyFn = Callable[[Array, History, Array], Any]


def _shift(buf: Array, new: Array) -> Array:
    return jnp.concatenate([buf[1:], new[None]])


def _check_action_form(action: Any, null_action: Any) -> None:
    """The policy's action must have the env's action form (that of ``env.null_action()``)."""
    if isinstance(action, Mapping) != isinstance(null_action, Mapping):
        want = (
            "a mapping keyed by action-node name"
            if isinstance(null_action, Mapping)
            else "a positional action array"
        )
        got = "a mapping" if isinstance(action, Mapping) else "a positional array"
        raise TypeError(
            f"rollout: the policy returned {got}, but this env takes {want} (the form of its "
            "null_action()); convert with system.to_positional('actions', ...) / "
            "system.from_positional('actions', ...)"
        )


def _append(hist: History, obs: Obs, action: Any) -> History:
    return History(
        obs=jax.tree_util.tree_map(_shift, hist.obs, obs),
        actions=jax.tree_util.tree_map(_shift, hist.actions, action),
        valid=_shift(hist.valid, jnp.asarray(True)),
    )


def _sentinel(leaf: Array, window_len: int) -> Array:
    leaf = jnp.asarray(leaf)
    return jnp.full((window_len, *leaf.shape), -1, leaf.dtype)


def _init_history(obs: Obs, window_len: int, null_action: Any) -> History:
    return History(
        obs=jax.tree_util.tree_map(lambda o: _sentinel(o, window_len).at[-1].set(o), obs),
        actions=jax.tree_util.tree_map(lambda a: _sentinel(a, window_len), null_action),
        valid=jnp.zeros((window_len,), bool).at[-1].set(True),
    )


class Trajectory(eqx.Module):
    """Scan-produced traces with leading ``(batch, time)`` axes.

    ``observations``/``states`` span ``num_steps + 1`` times (reset included);
    ``actions`` spans ``num_steps``; ``history`` snapshots the window each
    action was chosen from (``num_steps`` decision points).
    """

    observations: Any
    states: Any
    actions: Any
    history: History


def rollout(
    key: Array,
    env: Env,
    params: Any,
    policy_fn: PolicyFn,
    num_steps: int,
    batch_size: int,
    window_len: int = 8,
) -> Trajectory:
    """Roll ``env`` against ``policy_fn`` for ``num_steps``, vmapped over agents.

    ``params`` leaves must carry a leading ``(batch_size,)`` axis (see
    :func:`hybrid_aif.envs.generative.broadcast_params` for shared parameters).
    """
    return _rollout_batched(key, env, params, policy_fn, num_steps, batch_size, window_len)


@eqx.filter_jit
def _rollout_batched(key, env, params, policy_fn, num_steps, batch_size, window_len):
    """:func:`rollout`'s compiled body: ``env``, ``policy_fn`` and the sizes are static
    (pass the same function object to share the compiled program)."""
    keys = jax.random.split(key, batch_size)

    def one(k: Array, p: Any) -> Trajectory:
        return _rollout_single(k, env, p, policy_fn, num_steps, window_len)

    return jax.vmap(one)(keys, params)


def _rollout_single(
    key: Array,
    env: Env,
    params: Any,
    policy_fn: PolicyFn,
    num_steps: int,
    window_len: int,
) -> Trajectory:
    key_reset, key_scan = jax.random.split(key)
    obs0, state0 = env.reset(key_reset, params)
    null_action = env.null_action()
    hist0 = _init_history(obs0, window_len, null_action)

    def step(carry, xs):
        state, hist = carry
        key_t, t = xs
        k_pol, k_env = jax.random.split(key_t)
        action = policy_fn(k_pol, hist, t)
        _check_action_form(action, null_action)
        obs, state_next = env.step(k_env, state, action, params)
        return (state_next, _append(hist, obs, action)), (obs, state_next, action, hist)

    xs = (jax.random.split(key_scan, num_steps), jnp.arange(num_steps))
    _, (obs_seq, state_seq, actions, hists) = jax.lax.scan(step, (state0, hist0), xs)

    def prepend(first: Array, seq: Array) -> Array:
        return jnp.concatenate([first[None], seq])

    return Trajectory(
        observations=jax.tree_util.tree_map(prepend, obs0, obs_seq),
        states=jax.tree_util.tree_map(prepend, state0, state_seq),
        actions=actions,
        history=hists,
    )


class _RandomPolicy(eqx.Module):
    """Uniform-random action per factor. A Module rather than a closure so that equal
    ``num_actions`` compare and hash equal: :func:`rollout` takes ``policy_fn`` as static,
    and two calls to :func:`random_policy` then share one compiled program. ``keys`` names
    the controlled factors' draws ``(action node, factor)`` for a name-keyed action, else
    ``None``."""

    num_actions: tuple[int, ...] = eqx.field(static=True)
    keys: tuple[tuple[str, int], ...] | None = eqx.field(static=True, default=None)

    def __call__(self, key: Array, history: History, t: Array) -> Any:
        del history, t
        keys = jax.random.split(key, len(self.num_actions))
        draws = [
            jax.random.randint(k, (), 0, n)
            for k, n in zip(list(keys), self.num_actions, strict=True)
        ]
        action = jnp.stack(draws).astype(jnp.int32)
        if self.keys is None:
            return action
        return {name: action[f] for name, f in self.keys}


def random_policy(system_or_levels: "System | tuple[int, ...]", params=None) -> PolicyFn:
    """Uniform-random action.

    Given a sugar-shaped :class:`~hybrid_aif.system.System` and its ``params`` (the level counts
    live with the tables), returns the action keyed by the action nodes' names
    (``System.names.actions``) that its :meth:`~hybrid_aif.system.System.as_env` takes, one
    uniform level per action node. Given a tuple of levels per factor, returns the positional
    ``(num_factors,)`` action a positional env takes. The draws are the same either way: one per
    state factor, a factor owning no control node drawing its single no-op level.
    """
    from hybrid_aif.system import System

    num_actions = system_or_levels
    if not isinstance(num_actions, System):
        return _RandomPolicy(tuple(int(n) for n in num_actions))
    if params is None:
        raise TypeError(
            "random_policy(system, params): a System's control levels are read off its tables, "
            "so pass its params too"
        )
    params = num_actions._params_in(params, "random_policy")
    names = num_actions._names(params)
    if any(k != "level" for k in names.kinds):
        raise ValueError(
            "random_policy draws control levels, and this System has continuous control nodes, "
            "which have none; write a policy_fn that samples their boxes in params.control"
        )
    return _RandomPolicy(
        tuple(int(n) for n in num_actions._num_controls(params, "random_policy")),
        tuple(zip(names.actions, num_actions._sugar("random_policy").owners, strict=True)),
    )
