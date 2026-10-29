"""A recurrent state-space world model (RSSM, DreamerV3's architecture), assembled from parts.

    ``deter'`` = GRU(``deter``, flat ``stoch``, ``u``)          the carry node
    ``p(stoch' | deter')``      factored categorical              the *prior*
    ``q(stoch | deter, e(o))``  factored categorical              the *posterior*
    ``e(o)``                    MLP over ``symlog(o)``            the encoder
    ``ô, r̂, ĉ`` from ``[deter, flat stoch]``                      the three heads

Every part is a family from :mod:`hybrid_aif.distributions.neural` or a
:class:`~hybrid_aif.factors.Deterministic` node, and lives in a ``SystemParams`` slot; the
System's faces run them through :mod:`hybrid_aif.inference.amortized`. :class:`RSSMCore` is the
assembly the sequence trainer (:mod:`hybrid_aif.train.world`) reads out of the slots
(:meth:`RSSMCore.from_declarations`), differentiates, and writes back by slot key. Inference is
the recognition net: there is no filter. Unbatched, static-shaped, explicit-carry: one sequence
per :meth:`RSSMCore.observe` call, batched by ``vmap``.

Design: docs/design/amortization.md A1(b), A11
"""

import dataclasses
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.distributions.neural import (
    LearnedVector,
    MLPCategorical,
    MLPPredictor,
    PredictorHead,
    straight_through_onehot,
)
from hybrid_aif.factors import Deterministic
from hybrid_aif.inference import amortized as _amortized

#: The assembly's parts in canonical order: the field order, hence the order the world-model
#: loss's global-norm clip sees.
PART_FIELDS: tuple[str, ...] = (
    "gru",
    "prior",
    "posterior",
    "encoder",
    "decoder",
    "reward",
    "cont",
    "initial",
)


class RSSMSlots(NamedTuple):
    """Each part's slot-qualified address in the System that declared it.

    One entry per :data:`PART_FIELDS` field, in the ``LearnState`` / ``lr_by_slot`` / ``epistemic``
    key space, resolved from the System's own names (so a ``merge()`` composite yields namespaced
    addresses).
    """

    gru: str
    prior: str
    posterior: str
    encoder: str
    decoder: str
    reward: str
    cont: str
    initial: str


class RSSMState(eqx.Module):
    """The RSSM's state: a deterministic carry and a categorical one-hot code.

    Consumers read ``[deter, flat stoch]`` (:attr:`features`). Unbatched; a stacked-over-time state
    is a pytree of ``(T, …)`` leaves, its methods reached through ``vmap``.
    """

    deter: Array  # (deter_dim,)
    stoch: Array  # (num_vars, num_classes) one-hot code

    @property
    def features(self) -> Array:
        """``concat([deter, flatten(stoch)])`` — the feature vector every head reads."""
        return jnp.concatenate([self.deter, self.stoch.reshape(-1)], axis=-1)


def _head_roles(
    A: tuple,
    A_deps: tuple[tuple[int, ...], ...],
    modality_names: tuple[str, ...],
    state_modality: int,
    roles: dict[str, str] | None,
) -> dict[str, int]:
    """Which declared modality plays decoder, reward and continuation.

    The decoder is the modality the encoder reads. ``roles`` names the other two
    (``{"reward": ..., "cont": ...}``); without it they are found by the names
    ``"reward"``/``"cont"``, else by shape (the single-logit binary head is the continuation, the
    remaining scalar head the reward), and anything else asks for ``roles=``.
    """
    names = {name: m for m, name in enumerate(modality_names)}
    others = [m for m in range(len(A)) if m != state_modality]
    if roles is not None:
        missing = sorted({"reward", "cont"} - set(roles))
        unknown = sorted(set(roles) - {"reward", "cont", "decoder"})
        if missing or unknown:
            raise ValueError(
                f"roles= names 'reward' and 'cont' (and optionally 'decoder'); "
                f"missing {missing}, unknown {unknown}"
            )
        bad = {role: name for role, name in roles.items() if name not in names}
        if bad:
            raise ValueError(
                f"roles= names modalities that are not declared: {bad}; declared {list(names)}"
            )
        if "decoder" in roles and names[roles["decoder"]] != state_modality:
            raise ValueError(
                f"roles['decoder'] is {roles['decoder']!r} but the encoder reads "
                f"{modality_names[state_modality]!r}; the decoder reconstructs what the encoder "
                "reads"
            )
        picked = {
            "decoder": state_modality,
            "reward": names[roles["reward"]],
            "cont": names[roles["cont"]],
        }
    elif "reward" in names and "cont" in names and set(others) == {names["reward"], names["cont"]}:
        picked = {"decoder": state_modality, "reward": names["reward"], "cont": names["cont"]}
    else:
        binary = [m for m in others if A[m].loss == "binary" and A[m].target_dim == 1]
        scalar = [m for m in others if m not in binary and A[m].target_dim == 1]
        if len(others) != 2 or len(binary) != 1 or len(scalar) != 1:
            declared = {modality_names[m]: (A[m].loss, A[m].target_dim) for m in range(len(A))}
            raise ValueError(
                "the RSSM assembly is a decoder (the modality the encoder reads), one scalar "
                "reward head and one single-logit continuation head; this System declares "
                f"{declared}. Name the roles: from_declarations(..., "
                "roles={'reward': ..., 'cont': ...})"
            )
        picked = {"decoder": state_modality, "reward": scalar[0], "cont": binary[0]}
    return picked


class RSSMCore(eqx.Module):
    """An RSSM world model: GRU carry, categorical prior/posterior, three heads.

    Build it with the field constructor or from a declared System (:meth:`from_declarations`).
    :meth:`__check_init__` checks that widths agree at every seam.
    """

    gru: Deterministic  # (deter, flat stoch, u) -> deter'
    prior: MLPCategorical  # p(stoch' | deter')
    posterior: MLPCategorical  # q(stoch | deter, embed)
    encoder: MLPPredictor  # obs -> embed (symlog_inputs)
    decoder: PredictorHead  # features -> observation (symlog_mse)
    reward: PredictorHead  # features -> reward (symlog_disc, num_bins logits)
    cont: PredictorHead  # features -> continuation (binary, 1 logit)
    initial: LearnedVector  # the learned episode-start deter
    obs_dim: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)

    def __check_init__(self) -> None:
        deter, flat = self.deter_dim, self.prior.flat_dim
        if self.posterior.num_vars != self.prior.num_vars or (
            self.posterior.num_classes != self.prior.num_classes
        ):
            raise ValueError(
                f"prior and posterior must be the same factored categorical: prior is "
                f"{self.prior.num_vars}x{self.prior.num_classes}, posterior is "
                f"{self.posterior.num_vars}x{self.posterior.num_classes} — the KL that trains "
                "them is between these two"
            )
        expected = {
            "gru inputs": (tuple(self.gru.map.input_dims), (flat, self.action_dim)),
            "prior parents": (self.prior.parent_dims, (deter,)),
            "posterior parents": (self.posterior.parent_dims, (deter, self.embed_dim)),
            "encoder parents": (self.encoder.parent_dims, (self.obs_dim,)),
            "decoder parents": (self.decoder.parent_dims, (deter, flat)),
            "reward parents": (self.reward.parent_dims, (deter, flat)),
            "cont parents": (self.cont.parent_dims, (deter, flat)),
        }
        bad = {k: v for k, v in expected.items() if v[0] != v[1]}
        if bad:
            raise ValueError(
                "RSSM parts do not chain: "
                + "; ".join(f"{k} are {got} but should be {want}" for k, (got, want) in bad.items())
                + f" (deter={deter}, flat stoch={flat}, embed={self.embed_dim}, "
                f"obs={self.obs_dim}, action={self.action_dim})"
            )
        scalar = {name: getattr(self, name).target_dim for name in ("reward", "cont")}
        if any(dim != 1 for dim in scalar.values()):
            raise ValueError(
                f"the reward and continuation heads predict one number each; got target widths "
                f"{scalar}"
            )
        if self.decoder.out_dim != self.obs_dim:
            raise ValueError(
                f"decoder emits {self.decoder.out_dim} numbers for a {self.obs_dim}-dimensional "
                "observation"
            )
        if self.initial.dim != deter:
            raise ValueError(
                f"the learned initial carry is {self.initial.dim}-dimensional but deter is {deter}"
            )

    # ------------------------------------------------------------------ construction

    # -------------------------------------------------------------------- structure

    @property
    def deter_dim(self) -> int:
        return self.gru.map.deter_dim

    @property
    def embed_dim(self) -> int:
        return self.encoder.out_dim

    @property
    def num_vars(self) -> int:
        return self.prior.num_vars

    @property
    def num_classes(self) -> int:
        return self.prior.num_classes

    @property
    def num_bins(self) -> int:
        """The reward head's bin count (declared by the head, read here for the loss)."""
        return self.reward.num_bins

    @property
    def bins(self) -> Array:
        """The reward head's bin centres (a constant; XLA folds it)."""
        return self.reward.bins

    def parts(self) -> dict[str, object]:
        """The assembly's parts, keyed by field name, in :data:`PART_FIELDS` order."""
        return {name: getattr(self, name) for name in PART_FIELDS}

    def replace_parts(self, updated: dict[str, object]) -> "RSSMCore":
        """A copy with the named parts replaced (keys as in :data:`PART_FIELDS`)."""
        unknown = sorted(set(updated) - set(PART_FIELDS))
        if unknown:
            raise ValueError(f"unknown RSSM part(s) {unknown}; the parts are {list(PART_FIELDS)}")
        return dataclasses.replace(self, **updated)

    # ------------------------------------------------------ assembly from slots

    @classmethod
    def from_declarations(
        cls,
        structure: _amortized.AmortizedStructure,
        A: tuple,
        B: tuple,
        D: tuple,
        recognition: dict,
        A_deps: tuple[tuple[int, ...], ...],
        modality_names: tuple[str, ...],
        factor_names: tuple[str, ...],
        roles: dict[str, str] | None = None,
    ) -> tuple["RSSMCore", RSSMSlots]:
        """Read this assembly out of a System's slots: ``(core, slot addresses)``.

        What :func:`hybrid_aif.train.world.fit_world` calls before writing the parts back by the
        returned addresses. The assembly has one shape: one carry, one code, an encoder over one
        observation modality, and a decoder, a scalar reward head and a single-logit continuation
        head over ``[deter, flat stoch]``; anything else raises. ``roles`` is as in
        :func:`_head_roles`.
        """
        fams = structure.families
        carries = [f for f, fam in enumerate(fams) if fam == _amortized.CARRY]
        codes = [f for f, fam in enumerate(fams) if fam == _amortized.CODE]
        if len(carries) != 1 or len(codes) != 1:
            raise ValueError(
                f"the RSSM assembly is one carry node and one code factor; this System declares "
                f"carries {[factor_names[f] for f in carries]} and codes "
                f"{[factor_names[f] for f in codes]}. Several world models in one System is a "
                "nesting question (merge()), not an assembly one."
            )
        carry, code = carries[0], codes[0]
        posterior = next((p for p in structure.plan if p.factor == code), None)
        auxiliary = [p for p in structure.plan if p.factor is None]
        if posterior is None or len(auxiliary) != 1 or len(structure.plan) != 2:
            raise ValueError(
                f"the RSSM assembly's recognition side is exactly two q-side factors — the "
                f"amortized posterior over {factor_names[code]!r} and the observation encoding "
                f"it reads — got {[p.variable for p in structure.plan]}"
            )
        embed = auxiliary[0]
        if posterior.parents != (
            (_amortized.STATE, carry),
            (_amortized.RECOGNITION, embed.variable),
        ):
            raise ValueError(
                f"the RSSM's posterior conditions on (the carry, the observation encoding) in "
                f"that order — q(code | carry, e(o)); recognition:{posterior.variable} "
                f"declares parents {list(posterior.parents)}"
            )
        if len(embed.parents) != 1 or embed.parents[0][0] != _amortized.OBS:
            raise ValueError(
                f"the RSSM's encoder reads exactly one observation; recognition:"
                f"{embed.variable} declares parents {list(embed.parents)}"
            )
        state_modality = embed.parents[0][1]
        heads = _head_roles(A, A_deps, modality_names, state_modality, roles)
        for name, m in heads.items():
            if A_deps[m] != (carry, code):
                raise ValueError(
                    f"the RSSM's {name} head reads [deter, flat stoch] — A_deps "
                    f"({factor_names[carry]!r}, {factor_names[code]!r}) — but modality "
                    f"{modality_names[m]!r} declares {[factor_names[g] for g in A_deps[m]]}"
                )
        gru_inputs = tuple(B[carry].map.input_dims)
        if len(gru_inputs) != 2:
            raise ValueError(
                f"the RSSM's carry map takes (flat stoch, action); factor "
                f"{factor_names[carry]!r} declares inputs {gru_inputs}"
            )
        core = cls(
            gru=B[carry],
            prior=B[code],
            posterior=recognition[posterior.variable],
            encoder=recognition[embed.variable],
            decoder=A[heads["decoder"]],
            reward=A[heads["reward"]],
            cont=A[heads["cont"]],
            initial=D[carry],
            obs_dim=A[heads["decoder"]].target_dim,
            action_dim=gru_inputs[-1],
        )
        slots = RSSMSlots(
            gru=f"transition:{factor_names[carry]}",
            prior=f"transition:{factor_names[code]}",
            posterior=f"recognition:{posterior.variable}",
            encoder=f"recognition:{embed.variable}",
            decoder=f"likelihood:{modality_names[heads['decoder']]}",
            reward=f"likelihood:{modality_names[heads['reward']]}",
            cont=f"likelihood:{modality_names[heads['cont']]}",
            initial=f"initial_state_prior:{factor_names[carry]}",
        )
        return core, slots

    # ------------------------------------------------------------------- the steps

    def initial_state(self) -> RSSMState:
        """The episode-start state: the learned carry, and the prior's mode at it.

        The initial ``deter`` is the ``initial_state_prior`` slot's
        :class:`~hybrid_aif.distributions.neural.LearnedVector`; the same numbers as
        ``System.initial_priors`` on the declared form.
        """
        deter = self.initial()
        return RSSMState(deter=deter, stoch=self.prior.mode((deter,)))

    def img_step(self, key: Array, state: RSSMState, action: Array) -> tuple[RSSMState, Array]:
        """One imagined step: carry forward, then sample the prior.

        Returns ``(state', prior_logits)``; no observation is read.
        """
        action = jnp.asarray(action)
        if action.shape != (self.action_dim,):
            raise ValueError(
                f"action has shape {action.shape}, expected ({self.action_dim},) — an unbatched "
                "continuous action vector (controls.py's flat layout); batch with vmap"
            )
        deter = self.gru(state.deter, state.stoch.reshape(-1), action)
        logits = self.prior.logits((deter,))
        # sample from the logits we already have, not through `prior.sample`,
        # which would evaluate the head a second time inside the scan
        return RSSMState(deter=deter, stoch=straight_through_onehot(key, logits)), logits

    def obs_step(
        self, key: Array, state: RSSMState, action: Array, obs: Array
    ) -> tuple[RSSMState, Array, Array]:
        """One observed step: amortized recognition.

        Returns ``(state', posterior_logits, prior_logits)``; the state carries the posterior's
        code. ``action`` is ``a_{t−1}``, the action that produced ``obs``.
        """
        obs = jnp.asarray(obs)
        if obs.shape != (self.obs_dim,):
            raise ValueError(
                f"obs has shape {obs.shape}, expected ({self.obs_dim},) — one unbatched "
                "observation vector; batch with vmap"
            )
        embed = self.encoder((obs,))  # the encoder symlogs its own input (symlog_inputs)
        return self.embed_step(key, state, action, embed)

    def embed_step(
        self, key: Array, state: RSSMState, action: Array, embed: Array
    ) -> tuple[RSSMState, Array, Array]:
        """:meth:`obs_step` from an already-encoded observation: the recurrent part."""
        action = jnp.asarray(action)
        if action.shape != (self.action_dim,):
            raise ValueError(f"action has shape {action.shape}, expected ({self.action_dim},)")
        deter = self.gru(state.deter, state.stoch.reshape(-1), action)
        prior_logits = self.prior.logits((deter,))
        post_logits = self.posterior.logits((deter, embed))
        stoch = straight_through_onehot(key, post_logits)  # one head evaluation, not two
        return RSSMState(deter=deter, stoch=stoch), post_logits, prior_logits

    def observe(
        self,
        key: Array,
        actions: Array,
        observations: Array,
        state: RSSMState | None = None,
        is_first: Array | None = None,
    ) -> tuple[RSSMState, Array, Array]:
        """Filter one sequence: the encoder over every step, then ``lax.scan`` of
        :meth:`embed_step`.

        ``actions`` is ``(T, action_dim)`` and ``observations`` ``(T, obs_dim)``, ``actions[t]``
        being the action that produced ``observations[t]`` (the ``a_{t−1}`` convention). Returns
        ``(states, posterior_logits, prior_logits)`` stacked over time, ``states`` the posterior
        states. ``state`` defaults to :meth:`initial_state`.

        ``is_first``, optional ``(T,)`` bool, marks episode starts inside the sequence: there the
        carry restarts from :meth:`initial_state` and the incoming action is zeroed, so spliced
        episodes are each filtered as if alone. Omitted, the sequence is one episode starting at
        ``state``.
        """
        actions, observations = jnp.asarray(actions), jnp.asarray(observations)
        if actions.ndim != 2 or observations.ndim != 2:
            raise ValueError(
                f"observe takes ONE sequence: actions (T, {self.action_dim}) and observations "
                f"(T, {self.obs_dim}), got {actions.shape} and {observations.shape} — batch with "
                "vmap"
            )
        if actions.shape[0] != observations.shape[0]:
            raise ValueError(
                f"actions and observations disagree on T: {actions.shape[0]} vs "
                f"{observations.shape[0]}"
            )
        if state is None:
            state = self.initial_state()
        keys = jax.random.split(key, observations.shape[0])
        #  the encoder does not depend on the carry: one batched call, not T sequential ones
        embeds = jax.vmap(lambda obs: self.encoder((obs,)))(observations)

        if is_first is None:

            def step(carry: RSSMState, xs):
                k, action, embed = xs
                new, post_logits, prior_logits = self.embed_step(k, carry, action, embed)
                return new, (new, post_logits, prior_logits)

            xs = (keys, actions, embeds)
        else:
            is_first = jnp.asarray(is_first)
            if is_first.shape != (observations.shape[0],):
                raise ValueError(
                    f"is_first has shape {is_first.shape}, expected "
                    f"({observations.shape[0]},) — one flag per step of THIS sequence; batch "
                    "with vmap"
                )
            #  hoisted: the reset target is a read of the learned initial carry and the
            #  prior's mode at it, the same at every step, so it is not scanned over
            restart = self.initial_state()

            def step(carry: RSSMState, xs):
                k, action, embed, first = xs
                carry = RSSMState(
                    deter=jnp.where(first, restart.deter, carry.deter),
                    stoch=jnp.where(first, restart.stoch, carry.stoch),
                )
                action = jnp.where(first, jnp.zeros_like(action), action)
                new, post_logits, prior_logits = self.embed_step(k, carry, action, embed)
                return new, (new, post_logits, prior_logits)

            xs = (keys, actions, embeds, is_first.astype(jnp.bool_))

        _, (states, post_logits, prior_logits) = jax.lax.scan(step, state, xs)
        return states, post_logits, prior_logits

    def open_loop(self, key: Array, state: RSSMState, actions: Array) -> tuple[RSSMState, Array]:
        """Prior-only rollout from ``state``: ``lax.scan`` of :meth:`img_step`.

        Returns ``(states, prior_logits)`` stacked over the ``(K, action_dim)`` action sequence. No
        actor and no value estimate (that is :func:`hybrid_aif.train.imagine.imagine`).
        """
        actions = jnp.asarray(actions)
        if actions.ndim != 2 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"open_loop takes ONE action sequence (K, {self.action_dim}), got {actions.shape}"
            )
        keys = jax.random.split(key, actions.shape[0])

        def step(carry: RSSMState, xs):
            k, action = xs
            new, logits = self.img_step(k, carry, action)
            return new, (new, logits)

        _, (states, logits) = jax.lax.scan(step, state, (keys, actions))
        return states, logits

    # -------------------------------------------------------------------- the heads

    def _features(self, state: RSSMState) -> tuple[Array, Array]:
        """The heads' parent tuple: ``(deter, flat stoch)``."""
        return (state.deter, state.stoch.reshape(-1))

    def decode_symlog(self, state: RSSMState) -> Array:
        """The decoder's raw prediction — in **symlog** space, where its loss lives."""
        return self.decoder(self._features(state))

    def decode(self, state: RSSMState) -> Array:
        """The predicted observation in its own units: ``symexp`` of the decoder."""
        return self.decoder.predict(self._features(state))

    def reward_logits(self, state: RSSMState) -> Array:
        """The reward head's ``num_bins`` logits (a ``symlog_disc`` head)."""
        return self.reward(self._features(state))

    def predict_reward(self, state: RSSMState) -> Array:
        """The reward head's decoded expectation, in reward units."""
        return self.reward.predict(self._features(state))

    def cont_logit(self, state: RSSMState) -> Array:
        """The continuation head's single logit (a ``binary`` head)."""
        return self.cont(self._features(state))[0]

    def predict_cont(self, state: RSSMState) -> Array:
        """``P(episode continues)`` — the continuation head's probability."""
        return self.cont.predict(self._features(state))
