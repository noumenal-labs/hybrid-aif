"""Reference world models as declared Systems: R-AIF's ``tiny`` RSSM and its CRSPP.

The placement of :func:`tiny_core`'s parts into slots::

    factors        deter  = the Deterministic GRU carry   (controlled: the action)
                   stoch  = the 32x32 categorical code    (parent: deter, same step)
    transition     transition:deter  GRU(deter, flat stoch_{t-1}, action)
                   transition:stoch  p(stoch | deter_t)
    initial priors initial_state_prior:deter  LearnedVector
                   initial_state_prior:stoch  DerivedInitial()
    modalities     likelihood:state   symlog_mse over [deter, flat stoch]
                   likelihood:reward  symlog_disc, 255 bins over [-20, 20]
                   likelihood:cont    binary (continuation, not termination)
    recognition    recognition:embed  MLP over the state observation (symlog inputs)
                   recognition:stoch  q(stoch | deter_t, embed)

``deter`` is declared first because ``stoch``'s transition parent is the carry at the same
step. The G side is not part of the world model; the agent mounts it
(:func:`hybrid_aif.reference.raif.agent.raif_mountain_car`).

Design: ``docs/design/amortization-p0-addendum.md`` §§1–3.
"""

import dataclasses

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.controls import Box, control_name
from hybrid_aif.distributions import MLPCategorical
from hybrid_aif.distributions.neural import LearnedVector, MLPPredictor, PredictorHead
from hybrid_aif.factors import DerivedInitial, Deterministic, Recognition
from hybrid_aif.inference.rssm import RSSMCore
from hybrid_aif.reference.raif import crspp as _crspp  # noqa: F401  (registers the CRSPP row)
from hybrid_aif.system import System, SystemParams

#: Their ``winit`` values mapped onto the engine's initializer names.
_INIT = {"default": "default", "normal": "trunc_normal_avg"}


def init_of(winit: str) -> str:
    """The engine's ``init=`` for their ``winit`` (``"default"`` | ``"normal"``)."""
    if winit not in _INIT:
        raise ValueError(f"unknown winit {winit!r}; theirs are {sorted(_INIT)}")
    return _INIT[winit]


CARRY_NAME = "deter"
CODE_NAME = "stoch"
STATE_MODALITY = "state"
REWARD_MODALITY = "reward"
CONT_MODALITY = "cont"
EMBED_VARIABLE = "embed"


def tiny_core(
    obs_dim: int,
    action_dim: int,
    *,
    key: Array,
    deter: int = 128,
    num_vars: int = 32,
    num_classes: int = 32,
    units: int = 32,
    mlp_units: int = 64,
    mlp_layers: int = 2,
    embed: int = 64,
    unimix: float = 0.01,
    num_bins: int = 255,
    bin_low: float = -20.0,
    bin_high: float = 20.0,
    activation: str = "silu",
    norm: bool = True,
    winit: str = "default",
) -> RSSMCore:
    """R-AIF's ``tiny`` world model as an :class:`RSSMCore`, state-based.

    ``rssm {deter 128, stoch 32, classes 32}``, ``units 32`` inside the RSSM and on the
    reward/continuation heads, ``mlp_units 64`` / 2 layers on the encoder and decoder,
    ``unimix 0.01``, a 255-bin ``symlog_disc`` reward over [-20, 20]. ``winit`` reaches every
    block: ``"normal"`` is their ``winit: normal, fan: avg`` with no bias where a layer is
    normed; ``"default"`` is eqx's initializer with every bias kept.

    Design: ``docs/experiments/raif-mountain-car/divergence-register.md`` (divergences 17, 26).
    """
    keys = jax.random.split(key, 7)
    flat = int(num_vars) * int(num_classes)
    return RSSMCore(
        gru=Deterministic.gru(
            deter,
            (flat, action_dim),
            key=keys[0],
            hidden=units,
            activation=activation,
            norm=norm,
            init=init_of(winit),
        ),
        prior=MLPCategorical.from_shapes(
            (deter,),
            num_vars,
            num_classes,
            key=keys[1],
            width=units,
            depth=1,
            activation=activation,
            norm=norm,
            unimix=unimix,
            init=init_of(winit),
        ),
        posterior=MLPCategorical.from_shapes(
            (deter, embed),
            num_vars,
            num_classes,
            key=keys[2],
            width=units,
            depth=1,
            activation=activation,
            norm=norm,
            unimix=unimix,
            init=init_of(winit),
        ),
        # their MLP encoder branch: the last normed, activated layer is the embedding,
        # hence out_activation and depth = mlp_layers - 1
        encoder=MLPPredictor.from_shapes(
            (obs_dim,),
            embed,
            key=keys[3],
            width=mlp_units,
            depth=max(mlp_layers - 1, 0),
            activation=activation,
            norm=norm,
            out_activation=True,
            symlog_inputs=True,
            init=init_of(winit),
        ),
        decoder=PredictorHead.from_shapes(
            (deter, flat),
            loss="symlog_mse",
            out_dim=obs_dim,
            key=keys[4],
            width=mlp_units,
            depth=mlp_layers,
            activation=activation,
            norm=norm,
            init=init_of(winit),
        ),
        reward=PredictorHead.from_shapes(
            (deter, flat),
            loss="symlog_disc",
            num_bins=num_bins,
            bin_low=bin_low,
            bin_high=bin_high,
            key=keys[5],
            width=units,
            depth=2,
            activation=activation,
            norm=norm,
            out_scale=0.0,  # their reward outscale 0.0
            init=init_of(winit),
        ),
        cont=PredictorHead.from_shapes(
            (deter, flat),
            loss="binary",
            key=keys[6],
            width=units,
            depth=2,
            activation=activation,
            norm=norm,
            init=init_of(winit),
        ),
        initial=LearnedVector.zeros(deter),
        obs_dim=int(obs_dim),
        action_dim=int(action_dim),
    )


def tiny_rssm(
    obs_dim: int,
    action_dim: int,
    *,
    key: Array,
    action_low: float = -1.0,
    action_high: float = 1.0,
    policy_len: int = 8,
    learned_initial: bool = True,
    **tiny: object,
) -> tuple[System, SystemParams]:
    """R-AIF's ``tiny`` world model as a ``(System, SystemParams)`` pair.

    ``obs_dim``/``action_dim`` are the environment's (Mountain Car: 2 and 1). ``key`` and the
    extra keyword arguments (``deter``, ``num_vars``, ``units``, ``unimix``, …) go to
    :func:`tiny_core`. ``learned_initial=False`` freezes the initial carry at zero (their
    ``initial: 'zeros'``). ``policy_len`` is their ``imag_horizon``: the explicit rollout
    steps before the ``g_tail`` critic closes the horizon. The carry reads a continuous control
    node, its box ``[action_low, action_high]`` in ``params.control``; its action node emits the
    value to the env. The G side is
    mounted on the returned pair by the agent.
    """
    core = tiny_core(obs_dim, action_dim, key=key, **tiny)  # type: ignore[arg-type]
    system = System(
        A_deps=((0, 1), (0, 1), (0, 1)),  # every head reads [deter, flat stoch]
        B_deps=((1,), (0,)),  # deter <- stoch_{t-1} (+ action); stoch <- deter_t
        controlled=(True, False),  # the action enters the carry, their img_step
        policy_len=policy_len,
        planner="cem",  # continuous actions (A9); "habit" once an actor is declared
        epistemic=False,  # no epistemic row over these beliefs yet (2b)
        factor_names=(CARRY_NAME, CODE_NAME),
        modality_names=(STATE_MODALITY, REWARD_MODALITY, CONT_MODALITY),
        recognition=(
            Recognition(variable=EMBED_VARIABLE, parents=(f"obs:{STATE_MODALITY}",)),
            Recognition(variable=CODE_NAME, parents=(CARRY_NAME, f"recognition:{EMBED_VARIABLE}")),
        ),
    )
    # their `initial: 'zeros'` is the same declaration, marked untrainable
    initial = core.initial if learned_initial else dataclasses.replace(core.initial, frozen=True)
    params = SystemParams(
        transition={CARRY_NAME: core.gru, CODE_NAME: core.prior},
        initial_state_prior={CARRY_NAME: initial, CODE_NAME: DerivedInitial()},
        likelihood={
            STATE_MODALITY: core.decoder,
            REWARD_MODALITY: core.reward,
            CONT_MODALITY: core.cont,
        },
        recognition={EMBED_VARIABLE: core.encoder, CODE_NAME: core.posterior},
        control={
            control_name(CARRY_NAME): Box(
                jnp.full((action_dim,), action_low, jnp.float32),
                jnp.full((action_dim,), action_high, jnp.float32),
            )
        },
    )
    system._validate(params)
    return system, params


def crspp_rssm(
    *,
    key: Array,
    policy_len: int = 8,
    deter: int = 128,
    num_vars: int = 32,
    num_classes: int = 32,
    units: int = 32,
    embed: int = 64,
    unimix: float = 0.01,
    activation: str = "silu",
    norm: bool = True,
    winit: str = "default",
) -> tuple[System, SystemParams]:
    """R-AIF's CRSPP prior-preference model: ``tiny_rssm`` without the action or the heads.

    A separate parameter set with its own GRU and prior and no modality, so it is only rolled
    forward, never observed. Its ``recognition:stoch`` conditions on
    ``(deter, external:embed)``: ``embed``'s width is declared in ``System.external_inputs``
    and must match the agent's encoder (``tiny_rssm``'s ``embed``, 64 by default); its value
    is supplied per call by :func:`hybrid_aif.reference.raif.crspp.crspp_recognize`. The
    ``cosine_sim_to_pref_rollout`` row owns it through ``System.scoring`` and
    ``SystemParams.scoring``. ``policy_len`` is carried but unused: the row rolls the agent's
    horizon.

    Design: ``docs/design/amortization-debt-ledger.md`` entry 5.
    """
    keys = jax.random.split(key, 3)
    flat = int(num_vars) * int(num_classes)
    system = System(
        A_deps=(),  # no decoder of its own
        B_deps=((1,), (0,)),  # deter <- stoch_{t-1} (no action); stoch <- deter_t
        controlled=(False, False),  # the whole point: no action in the transition prior
        policy_len=policy_len,
        epistemic=False,
        factor_names=(CARRY_NAME, CODE_NAME),
        modality_names=(),
        external_inputs=((EMBED_VARIABLE, int(embed)),),
        recognition=(
            Recognition(variable=CODE_NAME, parents=(CARRY_NAME, f"external:{EMBED_VARIABLE}")),
        ),
    )
    params = SystemParams(
        transition=(
            Deterministic.gru(
                deter,
                (flat,),
                key=keys[0],
                hidden=units,
                activation=activation,
                norm=norm,
                init=init_of(winit),
            ),
            MLPCategorical.from_shapes(
                (deter,),
                num_vars,
                num_classes,
                key=keys[1],
                width=units,
                depth=1,
                activation=activation,
                norm=norm,
                unimix=unimix,
                init=init_of(winit),
            ),
        ),
        initial_state_prior=(LearnedVector.zeros(deter), DerivedInitial()),
        recognition={
            CODE_NAME: MLPCategorical.from_shapes(
                (deter, int(embed)),
                num_vars,
                num_classes,
                key=keys[2],
                width=units,
                depth=1,
                activation=activation,
                norm=norm,
                unimix=unimix,
                init=init_of(winit),
            )
        },
    )
    params = system.from_positional("params", params)
    system._validate(params)
    return system, params
