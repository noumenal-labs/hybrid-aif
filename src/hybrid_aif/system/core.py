"""The System itself: its declarations, its params, and its faces.

A System holds static generative-model structure; every array lives in a separate
:class:`SystemParams` pytree. The faces are ``as_env`` (forward: ancestral sampling over the same
tensors), ``infer_states`` / ``infer_policies`` / ``act`` (backward), ``learn``, and
``update_empirical_prior`` for the tick-to-tick belief carry. Methods are pure; belief state lives
in the caller's carry. All-tabular declarations take the compiled discrete path; any LG, Gaussian or
mixture declaration switches every face to the hybrid path (:mod:`hybrid_aif.objectives.hybrid`);
neural state-space declarations take the amortized path. Selector resolution and validation live in
the sibling modules, with a thin delegating method kept on ``System`` for each; the engines behind
the inference, learning and planning faces are the functions of :mod:`hybrid_aif.system.faces`.

The face names (``infer_states``, ``infer_policies``, ``update_empirical_prior``,
``sample_action``), the A/B/C/D/E roles, ``A_deps``/``B_deps`` (pymdp's
``A_dependencies``/``B_dependencies``) and the split of static structure from a parameter pytree
follow pymdp's JAX ``Agent`` (pymdp/agent.py); the letters are SPM's MDP fields
(``spm_MDP_VB_XXX``: A, B, C, D, E, H), which pymdp adopted.

Design: docs/design/system.md; see docs/composing-systems.md.
"""

import dataclasses
import functools
from collections.abc import Callable, Iterable, Mapping
from typing import Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.controls import (
    FLOAT,
    LEVEL,
    Box,
    ControlLayout,
    ControlSpace,
    ControlTable,
    ControlTarget,
    SensedNodes,
    SugarView,
    action_node_beliefs,
    check_kernel,
    control_space,
    control_table,
    effective_control_deps,
    target_on_box,
    validate_control_deps,
)
from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalDirichlet,
    Distribution,
    Gaussian,
    Learnable,
    MixtureGaussian,
    MLPConditional,
)
from hybrid_aif.distributions.neural import LearnedVector, MLPCategorical, PredictorHead
from hybrid_aif.envs._process import DiscreteProcess, HybridProcess
from hybrid_aif.envs.generative import GenerativeEnv, GenerativeEnvParams
from hybrid_aif.factors import (
    DerivedInitial,
    Deterministic,
    LinearGaussianObservation,
    LinearGaussianTransition,
    NonlinearGaussianTransition,
    Recognition,
    TabularObservation,
    TabularPrior,
    TabularTransition,
)
from hybrid_aif.inference import amortized as _amortized
from hybrid_aif.inference.detail import InferenceDetail
from hybrid_aif.inference.discrete import SUPPORT_FLOOR
from hybrid_aif.learning import conjugate_stats
from hybrid_aif.learning import evidence as _evidence
from hybrid_aif.objectives import TermTree
from hybrid_aif.objectives import amortized as _amortized_terms
from hybrid_aif.objectives import closed_loop as _closed_loop
from hybrid_aif.objectives import hybrid as _hybrid
from hybrid_aif.objectives.discrete import resolve_conditional_targets
from hybrid_aif.objectives.presence import Presence
from hybrid_aif.planning.base import (
    Planner,
    PlannerContext,
    UseBound,
    planner_branches,
)
from hybrid_aif.planning.enumerate import (
    action_marginals,
    enumerate_policies,
)
from hybrid_aif.system import keyed, selectors, validation
from hybrid_aif.system.selectors import _KEY_KINDS, _EpistemicVar

Belief = Categorical | Gaussian | MixtureGaussian


class _Derived(tuple):
    """A sugar value read back off a System: ``dataclasses.replace`` passes it to the
    constructor, which recognises and ignores it."""

    __slots__ = ()


class _Unreadable(_Derived):
    """A sugar property with nothing to read back: any use raises ``message``. It still passes
    through ``dataclasses.replace`` (as every :class:`_Derived` value does)."""

    def __new__(cls, message: str) -> "_Unreadable":
        out = super().__new__(cls, ())
        out.message = message
        return out

    def _refuse(self, *_args, **_kwargs):
        raise AttributeError(self.message)

    __getitem__ = __iter__ = __len__ = __bool__ = __contains__ = _refuse
    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = __add__ = _refuse
    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"<unreadable: {self.message}>"


#: The :class:`LearnState` address of a **policy module** in ``policy_prior``: a bare key,
#: since ``policy_prior`` is not one of the five parameter slots. It is not rate-addressable;
#: the actor's trainer takes its optimizer as an argument.
ACTOR_KEY = "policy_prior"
#: The ``LearnState.moments`` entry counting ``learn`` calls for the reduction schedule.
REDUCTION_KEY = "reduction"


def _strong(tree):
    """``tree`` with every weak-typed array leaf made strong-typed (same dtype, same values).

    A leaf built from a Python scalar (``jnp.full(shape, 2.0)``) is weak-typed, and a compiled
    face traces it differently from the strong-typed leaf :meth:`System.learn` returns, so the
    first call would compile twice. Non-array leaves pass through.
    """

    def weak(x):
        return isinstance(x, jax.Array) and getattr(x, "weak_type", False)

    if not any(weak(x) for x in jax.tree_util.tree_leaves(tree)):
        return tree
    return jax.tree_util.tree_map(
        lambda x: jax.lax.convert_element_type(x, x.dtype) if weak(x) else x, tree
    )


def _precision_in(value):
    """A policy precision as a strong-typed floating array, or ``None``."""
    if value is None:
        return None
    arr = jnp.asarray(value)
    if not jnp.issubdtype(arr.dtype, jnp.floating):
        arr = arr.astype(jnp.result_type(float))
    return _strong(arr)


class SystemParams(eqx.Module):
    """All numeric state of a System: generative tensors, designations and loop factors.

    Declaration types carry the family: tabular declarations give the discrete System; LG
    declarations (:mod:`hybrid_aif.factors`) and Gaussian/mixture priors give the hybrid one. A
    ``likelihood`` / ``transition`` / ``initial_state_prior`` entry is a point declaration, a
    distribution over its parameters (``ConditionalDirichlet``, ``Dirichlet``) — learnable by
    :meth:`System.learn` and samplable by ``as_env(param_key=...)`` — or a parametrized function
    (:class:`~hybrid_aif.distributions.MLPConditional`), learnable by gradient.

    ``recognition`` holds the amortized inference modules, keyed by the variable each infers
    (structure on ``System.recognition``); ``scoring`` holds the parameters of the
    estimator-registry rows that carry their own, keyed by row name (structure on ``System.terms`` /
    ``System.scoring``). Both join the slot-qualified key space (``recognition:stoch``,
    ``scoring:g_tail``) and namespace under ``merge()``.

    ``control`` holds the box of each continuous control node, keyed by the node's name
    (``params.control["u_x"]``): a :class:`~hybrid_aif.controls.Box` of per-dimension bounds,
    the node's support. A control node with an entry is continuous.

    ``actuator`` holds the kernel ``P(a | u)`` of each action node that is not a delta, keyed
    by the action node's name (``params.actuator["a_x"]``): a ``(levels, levels)`` probability
    table whose column ``u`` is the realised action's distribution given the decision ``u``.

    ``durations`` holds a round's duration per level of one discrete control node, keyed by the
    node's name (``params.durations["u_option"]``): a positive float ``(levels,)`` array, or a
    scalar for every level, in the unit the model's preferences are per. A round steps every
    factor once, so a round has one duration and at most one node carries it; the engine checks
    positivity only. The objective never reads it; :meth:`System.elapsed` returns the realised
    action's duration for an outer loop's clock (1 when none is declared).
    Design: docs/design/step-durations.md T1, T4, T8.

    ``likelihood`` is keyed by modality name, ``transition`` and ``initial_state_prior`` by factor
    name (``params.likelihood["position"]``); every face takes and returns them so, and a
    positional (tuple) container raises, naming ``System.from_positional("params", params)``.
    Inside, the System reads a positional view (tuples in node order), built by name lookup.

    ``precision`` is the policy precision γ, an optional scalar array: when present the planners
    read it in place of ``System.gamma`` (the default, a static field), and because it is a
    parameter leaf it flows traced through every face, so changing its value recompiles nothing.

    The fields fall in three groups: the declared model (``transition``, ``likelihood``,
    ``initial_state_prior``), the closed-loop factors (``recognition``, ``policy_prior``), and
    scoring material (the three target dicts, ``scoring`` and ``precision``).
    ``tests/test_scoring_slot.py`` asserts the partition.

    Design: docs/design/system.md (SystemParams groups); docs/design/parameter-learning.md D3;
    docs/design/amortization.md A1 A2 A11 A12.
    """

    transition: Mapping[
        str,
        TabularTransition
        | LinearGaussianTransition
        | NonlinearGaussianTransition
        | MLPConditional
        | Deterministic
        | MLPCategorical,
    ] = eqx.field(converter=_strong)
    #   AIF B, by factor name; trailing parent axis = control (tables) / conditioned
    #   innovations (LG) / a one-hot-or-continuous action parent (neural, deterministic). A
    #   Deterministic member is a *carry* node (its own previous value is the map's carry, not
    #   a parent) and an MLPCategorical member a factored-categorical code whose parents are
    #   read within the step — the neural state-space pair
    initial_state_prior: Mapping[
        str, TabularPrior | Gaussian | MixtureGaussian | LearnedVector | DerivedInitial
    ] = eqx.field(converter=_strong)
    #   AIF D, by factor name; a LearnedVector is a carry node's episode-start *value*, a
    #   DerivedInitial says "read it off my own transition at t=0"
    likelihood: Mapping[
        str, TabularObservation | LinearGaussianObservation | MLPConditional | PredictorHead
    ] = eqx.field(default_factory=dict, converter=_strong)
    #   AIF A, by modality name; child-first (num_obs, *dep_sizes) tables (point or
    #   Dirichlet-over-table), LG (H, c, R_chol) heads, a neural conditional, or a
    #   PredictorHead (trunk + named loss: symlog_mse/symlog_disc/binary); default {}: an
    #   open-loop model observes nothing
    recognition: dict[str, object] = eqx.field(default_factory=dict, converter=_strong)
    #   ``{"stoch": posterior_net, "embed": encoder}``
    obs_targets: dict[str, Distribution] = eqx.field(
        default_factory=dict, converter=_strong
    )  # standing C designations, keyed by modality: a Categorical over a tabular modality's
    #   levels, or a Gaussian/MixtureGaussian over an LG modality's observation space
    #   (covariance as tolerance). A neural modality's raises. A ConditionalCategorical or
    #   ConditionalGaussian entry is a target conditional on the parents ``System.C_deps``
    #   names under its key, one distribution per parent configuration
    state_targets: dict[str, Distribution] = eqx.field(
        default_factory=dict, converter=_strong
    )  # standing risk-over-states designations; a conditional entry as in ``obs_targets``, and
    #   a Categorical under a discrete control node's name its fixed cost per level
    terminal_targets: dict[str, Distribution | ConditionalCategorical] = eqx.field(
        default_factory=dict, converter=_strong
    )  # H of Friston et al. 2023 and SPM's ``spm_MDP_VB_XXX``, scored at horizon end;
    #   a ConditionalCategorical entry is a *conditional* terminal target and needs a
    #   ``System.H_deps`` entry naming its parent
    #   factor(s) (see :meth:`System.resolve_terminal_targets`)
    scoring: dict[str, object] = eqx.field(default_factory=dict, converter=_strong)
    #   ``{"g_tail": (critic, slow), "ensemble_disagreement": (m0..m3)}``; a row that owns a
    #   declared model holds that model's params
    # The prior over policies (AIF's E): a log-space array or Categorical over an enumerated
    # policy set, or a policy module (beliefs -> action distribution) that the "habit" planner
    # acts from. q_pi ~ softmax(-gamma*G + log E), so E is gamma-independent: habits belong
    # here, costs that must trade against utilities belong in targets (inside G).
    policy_prior: Array | Categorical | object | None = eqx.field(default=None, converter=_strong)
    control: dict[str, Box] = eqx.field(default_factory=dict, converter=_strong)
    #   the box of each continuous control node, keyed by the node's name: a Box of its
    #   per-dimension bounds, the node's support. A node with an entry is continuous (its
    #   readers read a float vector of the box's width); a node without one is discrete.
    #   Design: docs/design/control-states.md C4
    actuator: dict[str, Array] = eqx.field(default_factory=dict, converter=_strong)
    #   the kernel P(a | u) of each action node that is not a delta, keyed by the action node's
    #   name: a (levels, levels) probability table, column u the distribution over the realised
    #   action given the decision u (a discrete control node only). An action node without an
    #   entry realises its control node's decision exactly. The kernel is process-side: the
    #   environment's actuator draws the realised action from it (System.as_env); the control
    #   node stays clamped to the decision. Design: docs/design/control-states.md C12
    durations: dict[str, Array] = eqx.field(default_factory=dict, converter=_strong)
    #   a round's duration per level of one discrete control node, keyed by the node's name:
    #   a positive (levels,) array or a scalar for every level. The objective never reads it.
    #   Design: docs/design/step-durations.md T1
    precision: Array | None = eqx.field(default=None, converter=_precision_in)
    #   the policy precision gamma as a scalar array, traced through every face: q_pi ~
    #   softmax(-precision * G + log E). ``None`` reads ``System.gamma``. Design:
    #   docs/design/recompilation.md R5

    def __check_init__(self) -> None:
        if self.precision is not None and jnp.ndim(self.precision) != 0:
            raise ValueError(
                f"SystemParams.precision is the policy precision, a scalar; got shape "
                f"{tuple(jnp.shape(self.precision))}"
            )

    @property
    def A(self):
        """AIF-letter alias for ``likelihood``."""
        return self.likelihood

    @property
    def B(self):
        """AIF-letter alias for ``transition``."""
        return self.transition

    @property
    def D(self):
        """AIF-letter alias for ``initial_state_prior``."""
        return self.initial_state_prior

    @property
    def E(self):
        """AIF-letter alias for ``policy_prior`` (read-only; the field is the writable name)."""
        return self.policy_prior


class LearnState(eqx.Module):
    """Optimizer state for an agent's gradient-learnable parts: ``learn``'s explicit carry.

    Returned by :meth:`System.learn` and the objective-side trainers and handed back on the next
    call; never held in a module attribute. A sibling of :class:`SystemParams`, not a field of it.

    ``opt_states`` is keyed by slot-qualified address (``"transition:x"``, the ``lr_by_slot`` key
    space), plus ``scoring:<row>`` for each mounted row with a trainer
    (:attr:`objectives.registry.Estimator.trainer`) and :data:`ACTOR_KEY` for a policy module in
    ``policy_prior``. ``moments`` holds non-optimizer running statistics (the actor's
    return-normalization EMAs, the slow critic's update counter), keyed by trainer name. Both are
    seeded by :meth:`System.init_learn_state`, so the carry's structure is fixed from the first
    call; each trainer steps its own entries, passes the rest through unchanged, and raises when an
    entry it owns is missing. Dict keys are static under jit.

    Design: docs/design/amortization.md A2 A6 and chunk 2c §(b).
    """

    opt_states: dict[str, object] = eqx.field(default_factory=dict)
    moments: dict[str, object] = eqx.field(default_factory=dict)


def _carry_prev(transition_fn, step_terms_fn, terminal_terms_fn, init_beliefs):
    """Wrap planning closures so the scan carry is ``(qs, qs_prev)``.

    Used when a step term reads the previous step's beliefs (a learnable transition's novelty, or a
    mounted row that declares ``reads_prev``); every other configuration keeps the bare belief
    tuple.
    """

    def transition(carry, action):
        qs, _ = carry
        return transition_fn(qs, action), qs

    def step_terms(carry, action, t):
        qs_next, qs_prev = carry
        return step_terms_fn(qs_next, action, t, qs_prev)

    terminal = None if terminal_terms_fn is None else (lambda carry: terminal_terms_fn(carry[0]))
    return transition, step_terms, terminal, (init_beliefs, init_beliefs)


class AmortizedScoring(NamedTuple):
    """The amortized path's resolved scoring closures (:meth:`System._amortized_scoring`).

    ``step_terms_fn(qs, action, t, qs_prev=None)`` and ``terminal_terms_fn(qs)`` (or ``None``);
    ``needs_prev`` says whether the previous beliefs must be supplied, ``controls`` is the flat
    continuous-action layout when one is declared, and ``structure`` the resolved amortized
    declaration.
    """

    structure: "_amortized.AmortizedStructure"
    step_terms_fn: object
    terminal_terms_fn: object | None
    needs_prev: bool
    controls: ControlLayout | None


class Subsystem(NamedTuple):
    """One component's identity inside a merged composite, carried on ``System.subsystems``.

    Built by :func:`hybrid_aif.composition.merge.merge`. Metadata — no face computes from it
    — recording which edges crossed a boundary, which the flat graph does not show;
    :func:`~hybrid_aif.composition.merge.subsystem` derives a per-mind view from it, and a
    mapping keyed by a node a binding removed is refused with a sentence naming what it became.

    ``factors`` and ``modalities`` are composite state-factor/modality indices in the
    subsystem's declaration order (its block is contiguous, so local index ``i`` is
    ``factors[i]``). The control nodes, by composite name, as this merge left them:

    - ``controls`` — the control nodes this subsystem still chooses (the composite plans them);
    - ``actions`` — ``(control node, action node)`` for every control node this record names;
    - ``relayed`` — ``(node, state)``: a control node identified with another subsystem's state,
      which its reader reads a step later (an ``ActionControl`` from a state);
    - ``terminated`` — control nodes a ``BoundaryPrior`` replaced with a prior;
    - ``identified`` — ``(node, driver)``: a control node identified with another subsystem's
      control node ``driver`` (an ``ActionControl`` from a control node, read at the same step),
      so every factor that read ``node`` reads ``driver``;
    - ``free_modalities`` / ``bound_modalities`` — modalities reading only this subsystem's factors,
      versus ``(modality, foreign parents)`` for those an observation binding pointed across the
      boundary;
    - ``target_emissions`` — ``(factor, parents)`` whose ``H_deps`` parents lie outside this
      subsystem.

    A record :func:`~hybrid_aif.description.build` reconstructs from a model description carries
    only what the document shows: the composite's names, its factors, modalities and the control
    nodes still present. ``relayed``, ``terminated`` and ``identified`` are empty there, and
    ``actions`` names the present control nodes only, since a node a binding removed is absent
    from the document (design: model-description-0.3.md Q18.2).

    Design: docs/design/nesting-systems.md; docs/design/control-states.md C12.
    """

    factors: tuple[int, ...]
    modalities: tuple[int, ...]
    controls: tuple[str, ...] = ()
    actions: tuple[tuple[str, str], ...] = ()
    relayed: tuple[tuple[str, str], ...] = ()
    terminated: tuple[str, ...] = ()
    identified: tuple[tuple[str, str], ...] = ()
    free_modalities: tuple[int, ...] = ()
    bound_modalities: tuple[tuple[int, tuple[int, ...]], ...] = ()
    target_emissions: tuple[tuple[int, tuple[int, ...]], ...] = ()


class System(eqx.Module):
    """Static structure of a System, discrete or hybrid (see module docstring).

    Every field is static; the field table with defaults and meanings is
    ``docs/composing-systems.md``. The factor table holds every node: state factors first, then
    the **control nodes** the System chooses (``controls``, by node index). ``B_deps``,
    ``factor_names``, ``D_deps`` and ``H_deps`` run over all nodes; ``A_deps`` names state
    factors. A transition reads its state parents from the previous step and its control parents
    at the step the control is chosen; a control node's levels are its readers' table axes.
    ``epistemic`` names variables: naming a state factor mounts the infogain of every modality
    that observes it, so switching one factor off means naming the others.

    The pymdp-style keywords ``controlled`` and ``B_control_deps`` are construction sugar,
    translated once into control nodes: each controlled factor ``f`` (in factor order) gets a
    node ``u_<f>``, appended after the declared factors, and the node is the last transition
    parent of ``f`` (under ``B_control_deps``, of every factor listing ``f``, at the listed
    position), so every table keeps its layout. A System built from the sugar equals the one that
    declares those nodes natively. The sugar is readable back (``controlled``,
    ``B_control_deps``) on a sugar-shaped System. A discrete node's level count is its readers'
    parent axis (:meth:`policies`); a node is continuous when ``SystemParams.control`` holds its
    box, and its readers then read a float vector (a neural transition's input, a linear-Gaussian
    transition's ``G u`` term).
    Design: docs/design/control-states.md C1, C4, C6, C11.
    """

    B_deps: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    A_deps: tuple[tuple[int, ...], ...] = eqx.field(static=True, default=())
    planner: str = eqx.field(static=True, default="enumeration")
    gamma: float = eqx.field(static=True, default=16.0)
    num_iters: int = eqx.field(static=True, default=16)
    decomposition: Literal["utility_infogain", "risk_ambiguity"] = eqx.field(
        static=True, default="utility_infogain"
    )
    epistemic: bool | tuple[str, ...] = eqx.field(static=True, default=True)
    #   the per-variable epistemic selector: ``True`` mounts every eligible variable,
    #   ``False``/``()`` none, a tuple names the variables that carry an epistemic leaf: an
    #   ``infogain`` leaf under ``utility_infogain``, an ``ambiguity`` leaf under
    #   ``risk_ambiguity`` (risk is never touched by the selector).
    #   The grammar, shared with ``learn``'s rate keys (:mod:`hybrid_aif.system.selectors`):
    #
    #     <state>                     a state variable, by bare factor name ("pose_cell")
    #     <state>:<dim>               one dimension of a continuous state factor ("pose:b1"),
    #                                 named by ``dims`` or ``dim0``, ``dim1``, … by position
    #     obs:<modality>              an observation variable ("obs:cue")
    #     <slot>:<name>               a declaration's parameter variables as a group,
    #                                 slot one of likelihood/transition/initial_state_prior
    #     <slot>:<name>:<block>       one independent posterior block ("likelihood:pose:noise")
    #
    #   Any segment may be ``*``, matching that whole segment (no partial globs): ``("*",)``
    #   is every state variable, ``("*:*",)`` every parameter variable, ``("obs:*",)`` every
    #   observation variable, ``("pose:*",)`` every dimension of ``pose`` (the whole factor). A
    #   wildcarded first segment ranges over the parameter slots only. Selection is a union. A
    #   wildcard never validates less than its literal part; only a class empty through its
    #   wildcarded segment resolves silently to nothing.
    #   A state name mounts every channel that observes it, and a channel's leaf is its
    #   information about the *selected* parents, the unselected ones marginalised (its
    #   ambiguity likewise); ``obs:<modality>`` scores the channel over all its parents.
    #   Dimension keys on one factor select one sub-block: on a Gaussian-carried factor each
    #   channel's leaf is its information about those dimensions, the others marginalised, and
    #   under ``risk_ambiguity`` its ambiguity is ``E H[y | x_S]``; a sub-block of a
    #   mixture-carried factor raises. Selecting every dimension selects the factor. A
    #   continuous factor no channel observes has no information to gain: named literally it
    #   raises, and the drive toward a sharper belief about it is the ``uncertainty`` row.
    mixture_k: int | None = eqx.field(static=True, default=None)  # shared Runnalls budget
    mixture_schedule: Literal["runnalls", "mode_aligned"] = eqx.field(
        static=True, default="runnalls"
    )  # grow->reduce compression; mode_aligned keeps slots mode-owned (needed to filter)
    D_deps: tuple[tuple[int, ...], ...] | None = eqx.field(static=True, default=None)
    #   hierarchical conditional priors: parent factors of each factor's initial prior
    #   (None = all unconditional). Ancestral: parents must be earlier, tabular families.
    H_deps: tuple[tuple[int, ...], ...] | None = eqx.field(static=True, default=None)
    #   parent factors of each factor's ``terminal_targets`` entry (``None`` = all
    #   unconditional; see :meth:`resolve_terminal_targets`). As ``D_deps``, except parents
    #   need not be declared earlier and a factor may not parent its own terminal target.
    C_deps: tuple[tuple[str, tuple[str, ...]], ...] = eqx.field(static=True, default=())
    #   the parents of each conditional per-step target, **by name**: ``{"lane": ("u_option",),
    #   "cue": ("context",)}`` at construction, keyed by the targeted node (a state factor's
    #   ``state_targets`` entry, a modality's ``obs_targets`` entry), each parent a discrete
    #   control node or a categorical state factor; stored as ``(key, parents)`` pairs sorted by
    #   key because static fields must hash (``dict(system.C_deps)`` reads it). The target under
    #   a key is a ``ConditionalCategorical`` (child axis first, then one axis per parent in
    #   this order) or a ``ConditionalGaussian`` (``loc`` ``(*levels, d)``, ``chol``
    #   ``(*levels, d, d)``; control parents only). At each step of a plan the target is
    #   resolved under its parents, a clamped control node's level selecting its own
    #   distribution and a state parent averaging mean-field, then scored as an unconditional
    #   target is. Design: step-durations.md T5.
    modality_names: tuple[str, ...] = eqx.field(static=True, default=None)
    #   one name per modality, in node order; ``None`` at construction fills in ``modality_{m}``
    factor_names: tuple[str, ...] = eqx.field(static=True, default=None)
    #   one name per node (state factors, then control nodes), in node order; ``None`` at
    #   construction fills in ``factor_{i}`` (and ``u_factor_{i}`` for the sugar's nodes). The two
    #   tuples are the single source of names and order: keys in SystemParams, targets and rate
    #   maps refer to them and are validated at every face. No node is named after a reserved
    #   first segment of the key grammar (``obs`` or a slot word)
    dims: tuple[tuple[str, tuple[str, ...]], ...] = eqx.field(static=True, default=())
    #   names for the dimensions of continuous state factors: ``{factor: (name, ...)}`` at
    #   construction (``{"pose": ("x", "y", "yaw", "b1", "b2")}``), stored as ``(factor, names)``
    #   pairs in factor order because static fields must hash (``dict(system.dims)`` reads it).
    #   ``epistemic`` selects a dimension as ``<factor>:<name>``; a factor declared without names
    #   has ``dim0``, ``dim1``, … by position, which change meaning when the state is reordered.
    #   Names are unique within a factor, one per dimension, single key segments (no ``:``,
    #   ``.``, ``*`` or brackets), and ``dimN`` may name position ``N`` only
    recognition: tuple[Recognition, ...] | None = eqx.field(static=True, default=None)
    #   the structural half of the recognition slot: which variables an amortized module
    #   infers and what each conditions on (:class:`~hybrid_aif.factors.Recognition`);
    #   ``None``/``()`` = none. The modules live in ``SystemParams.recognition``.
    terms: tuple[str, ...] = eqx.field(static=True, default=())
    #   the row selector: which estimator-registry rows G carries beyond the ``infogain`` rows
    #   ``epistemic`` selects (:data:`~hybrid_aif.objectives.registry.ROWS`). Never inferred
    #   from ``SystemParams.scoring``, so the term-tree shape is a function of the System. The
    #   grammar is the row name, optionally applied to an ``epistemic`` variable key:
    #
    #     <row>                       a row with no target ("g_tail", or a row scoring
    #                                 the whole rollout against a model of its own)
    #     <row>:<slot>:<name>         a row on a declared kernel ("occupancy:transition:stoch")
    #     <row>:obs:<modality>        a row on an observation channel
    #                                 ("predicted_reward:obs:reward")
    #     <row>:<state>               a row on a state variable ("uncertainty:pose")
    #     <row>:<state>:<dim>         a row on dimensions of a continuous state variable
    #                                 ("uncertainty:pose:b1"; keys on one factor are one
    #                                 sub-block, as under ``epistemic``)
    #
    #   The row segment is always literal; the variable part may wildcard where the row
    #   declares a target class (``"occupancy:transition:*"``). Selection is a union. A row
    #   mounts on a target only if the registry has an estimator for that variable's family.
    scoring: tuple[tuple[str, object], ...] = eqx.field(static=True, default=())
    #   the structural half of the scoring slot, keyed by row name (parameters live in
    #   ``SystemParams.scoring``): e.g. a row's preference model that is itself a declared
    #   System. A tuple of pairs because static fields must hash; ``dict(system.scoring)``
    #   reads it. ``terms`` does the mounting: an entry whose row it does not mount raises.
    external_inputs: tuple[tuple[str, int], ...] = eqx.field(static=True, default=())
    #   ``(name, width)`` vectors a q-side factor may condition on that this System does not
    #   declare, spelled ``"external:<name>"`` in a Recognition parent set and supplied per
    #   call (``infer_states(externals={...})``); the width lets ``parent_dims`` validate
    #   statically
    subsystems: tuple[tuple[str, Subsystem], ...] | None = eqx.field(static=True, default=None)
    #   ``(name, Subsystem)`` pairs recording which factors, modalities and control nodes came
    #   from which component of a :func:`hybrid_aif.composition.merge.merge` call (``None``
    #   for a hand-declared System); ``dict(system.subsystems)`` reads it. No face computes from it;
    #   a refusal reads it to name what a node a binding removed became.
    branch_on: tuple[str, ...] = eqx.field(static=True, default=())
    #   the observation modalities, by name, whose outcomes the ``"sophisticated"`` planner
    #   branches on (:mod:`hybrid_aif.planning.sophisticated`): each categorical, over
    #   categorical parents only. Empty for every other planner, which never reads it.
    planning_belief: Literal["prior", "posterior"] = eqx.field(static=True, default="prior")
    #   which belief a plan's information-gain leaves read on a Gaussian-carried factor:
    #   ``"prior"`` the predictive belief (predict only); ``"posterior"`` a second chain that
    #   conditions on the observations expected along the plan, so a linear-Gaussian channel's
    #   leaf is the gain of its observation given the plan's earlier ones. Every other leaf
    #   reads the predictive belief, a Gaussian target under ``feedback`` excepted (design:
    #   planning-with-observations.md D1, D2; closed-loop-planning.md).
    feedback: Literal["linear_quadratic"] | None = eqx.field(static=True, default=None)
    #   how a plan's Gaussian targets are scored: ``None`` open loop, on the predictive belief;
    #   ``"linear_quadratic"`` closed loop, as the policy that follows the plan and corrects
    #   deviations from it by the linear-quadratic regulator's feedback on the estimate. Each
    #   Gaussian state, terminal and linear-Gaussian observation target on a Gaussian-carried
    #   factor is then scored by its cross-entropy on the closed-loop state distribution (the
    #   epistemic chain's posterior plus the estimate spread the corrections leave), and a
    #   Gaussian target on a continuous control node also charges the corrections. Requires
    #   ``planning_belief="posterior"`` (design: closed-loop-planning.md)
    lagged: tuple[tuple[str, str], ...] = eqx.field(static=True, default=())
    #   ``(modality, source)`` pairs: the modality's datum is the source's value at the previous
    #   tick, read from the carry at the filter step rather than supplied; the source is a state
    #   factor (the datum is its carried belief's mean: a Gaussian's location, a categorical's
    #   probability vector as a soft observation) or an action node (the level taken). The
    #   modality's own parents explain that value, so a factor can emit what the model just
    #   believed. An explicit observation for the modality wins over the carry; without either
    #   (the first tick) the modality is absent. Design: structure-learning.md S8
    reduction_period: int | None = eqx.field(static=True, default=None)
    #   the schedule of Bayesian model reduction on the growable factors: every
    #   ``reduction_period``-th call of ``learn`` runs :meth:`reduce` on each of them, the
    #   counter carried in ``LearnState.moments``; ``None`` leaves reduction to explicit calls
    #   (design: structure-learning.md S4)
    reduction_pairs: int = eqx.field(static=True, default=8)
    #   how many candidate pairs a reduction tests, the best-scoring by mutual expected
    #   log-likelihood, greedily
    controls: tuple[int, ...] = eqx.field(static=True, default=())
    #   the control nodes, by node index, ascending: the nodes the System sets by decision
    #   rather than by inference. They follow every state factor; each is read by at least one
    #   transition and has no parents, initial prior or terminal target of its own. A policy
    #   carries one column per control node, in this order
    action_names: tuple[str, ...] = eqx.field(static=True, default=None)
    #   one action node per control node, in ``controls`` order: the actuator channel, a child of
    #   its control node (kernel ``P(a | u)``, a delta unless ``SystemParams.actuator`` declares
    #   one). Declared with the control nodes, as ``modality_names`` with ``A_deps``, paired with
    #   them by position, both names free. The sugar's node for factor ``f`` is ``u_<f>`` with
    #   action ``a_<f>``. Action
    #   nodes follow the control nodes in node order, so ``A_deps`` names action node ``k`` as
    #   ``len(B_deps) + k``
    policy_len: int = eqx.field(static=True, kw_only=True)
    controlled: dataclasses.InitVar[tuple[bool, ...] | None] = eqx.field(default=None, kw_only=True)
    #   sugar: per state factor, whether it gets a control node ``u_<f>``
    B_control_deps: dataclasses.InitVar[tuple[tuple[int, ...], ...] | None] = eqx.field(
        default=None, kw_only=True
    )
    #   sugar: per state factor, the controlled factors whose nodes slice its transition, in axis
    #   order (``None``: each controlled factor reads its own node alone). Design:
    #   docs/design/control-states.md (Timing; C6)

    def __post_init__(self, controlled, B_control_deps) -> None:
        """Translate the control sugar into control nodes, and fill in generated names.

        The sugar keywords are translated once (see the class docstring): the stored fields then
        hold the node form, and the sugar is gone. Values that a ``dataclasses.replace`` passes
        back from the derived properties are recognised and ignored; sugar given explicitly to a
        System that already has control nodes raises (construct afresh from the sugar, or edit
        the nodes). Names: ``None`` fills in the generated names (``factor_{i}`` per state
        factor, ``u_<f>`` per sugar node), so the name tuples are always present, the single
        source of names and node order."""
        sugar = {"controlled": controlled, "B_control_deps": B_control_deps}
        given = {k: v for k, v in sugar.items() if v is not None and not isinstance(v, _Derived)}
        if given:
            if self.controls:
                raise ValueError(
                    f"{sorted(given)} declare controls the pymdp way, and this System already "
                    f"has control nodes {[self.factor_names[c] for c in self.controls]} "
                    "(controls=...): give one declaration or the other. A dataclasses.replace "
                    "with sugar keywords on a System that has control nodes cannot re-translate "
                    "them: construct the System afresh from the sugar, or edit B_deps/controls"
                )
            self._translate_sugar(given)
        elif not self.controls and type(controlled) is _Derived and any(controlled):
            raise ValueError(
                "controlled= was read back off a System with control nodes and passed to a "
                "System declaring none (controls=()): the node-form B_deps and factor_names "
                "already hold the control nodes, so pass controls= with them, or declare the "
                "sugar afresh with controlled="
            )
        elif self.factor_names is None and self.controls:
            raise ValueError(
                "factor_names=None on a System with control nodes "
                f"(controls={self.controls}): regenerating names would rename its control "
                "nodes; name every node, or construct the System afresh from the sugar"
            )
        else:
            self._derive_control_names()
        if self.factor_names is None:
            self.factor_names = keyed.generated_names("factor", len(self.B_deps))
        if self.modality_names is None:
            self.modality_names = keyed.generated_names("modality", len(self.A_deps))
        self.dims = self._normalized_dims()
        self.C_deps = self._normalized_C_deps()

    def _normalized_C_deps(self) -> tuple:
        """``C_deps`` as ``(key, parents)`` pairs of tuples sorted by key. A value of the wrong
        shape is passed through for :meth:`_check_C_deps` to refuse."""
        deps = self.C_deps
        pairs = list(deps.items()) if isinstance(deps, Mapping) else deps
        try:
            pairs = [
                (key, (parents,) if isinstance(parents, str) else tuple(parents))
                for key, parents in pairs
            ]
        except (TypeError, ValueError):
            return deps
        return tuple(sorted(pairs, key=lambda pair: str(pair[0])))

    def _normalized_dims(self) -> tuple:
        """``dims`` as ``(factor, names)`` pairs of tuples in factor order (a factor that is not
        a node sorts last, for :meth:`_check_dims` to name). A value of the wrong shape is passed
        through for :meth:`_check_dims` to refuse."""
        dims = self.dims
        pairs = list(dims.items()) if isinstance(dims, Mapping) else dims
        try:
            pairs = [
                (factor, names if isinstance(names, str) else tuple(names))
                for factor, names in pairs
            ]
        except (TypeError, ValueError):
            return dims
        order = {name: i for i, name in enumerate(self.factor_names or ())}
        return tuple(sorted(pairs, key=lambda pair: order.get(pair[0], len(order))))

    def _derive_control_names(self) -> None:
        """A native declaration's action nodes: ``action_names`` is required with control
        nodes."""
        if not self.controls:
            if self.action_names not in (None, ()):
                raise ValueError(
                    f"action_names={self.action_names!r} declares action nodes for a System "
                    "with no control nodes: an action node is the child of a control node, so "
                    "declare the control nodes (controls=...) or drop action_names"
                )
            self.action_names = ()
            return
        if self.action_names is None:
            names = self.factor_names
            raise ValueError(
                f"control node(s) {[names[c] for c in self.controls if c < len(names)]} are "
                "declared without action nodes: pass action_names=(...), one action name per "
                "control node in controls order"
            )
        if not isinstance(self.action_names, tuple) or not all(
            isinstance(a, str) for a in self.action_names
        ):
            raise TypeError(
                f"action_names must be a tuple of str, got {type(self.action_names).__name__}"
            )

    def _translate_sugar(self, given: Mapping[str, object]) -> None:
        """Rewrite ``self`` from the pymdp-style declaration ``given`` to the node form."""
        n = len(self.B_deps)
        controlled = given.get("controlled")
        if controlled is None:
            raise ValueError(
                f"{sorted(given)} without controlled: the pymdp-style control declaration "
                "names the controlled factors with controlled= (one entry per factor); a "
                "controlled factor's level count is its transition table's trailing axis"
            )
        generated = self.factor_names is None
        names = keyed.generated_names("factor", n) if generated else tuple(self.factor_names)
        if len(names) != n:
            raise ValueError(
                f"factor_names has {len(names)} name(s) for the {n} factor(s) B_deps declares: "
                "with the pymdp-style control keywords, name each declared factor (the control "
                "nodes they create are named u_<factor>)"
            )
        if not isinstance(controlled, tuple | list) or len(controlled) != n:
            raise ValueError(f"controlled must have one entry per factor ({n}), got {controlled!r}")
        controlled = tuple(bool(c) for c in controlled)
        validate_control_deps(given.get("B_control_deps"), controlled, names)
        owners = tuple(f for f, c in enumerate(controlled) if c)
        node = {f: n + k for k, f in enumerate(owners)}
        deps = effective_control_deps(given.get("B_control_deps"), controlled)
        unread = [names[c] for c in owners if not any(c in d for d in deps)]
        if unread:
            raise ValueError(
                f"factor(s) {unread} are declared controlled but no factor reads their control "
                f"(B_control_deps={given.get('B_control_deps')!r}): list each in the entry of a "
                "factor it moves, or declare it uncontrolled"
            )
        self.B_deps = (
            *((*tuple(self.B_deps[f]), *(node[c] for c in deps[f])) for f in range(n)),
            *((),) * len(owners),
        )
        created = [keyed.control_name(names[f]) for f in owners]
        clash = [c for c in created if c in names]
        if clash:
            raise ValueError(
                f"factor_names declares {clash}, the name(s) the control sugar gives the "
                "control node(s) it creates (u_<factor>); rename the state factor(s), or "
                "declare the control nodes natively to choose their names"
            )
        self.factor_names = (*names, *created)
        actions = tuple(keyed.action_name(names[f]) for f in owners)
        if self.action_names is not None and tuple(self.action_names) != actions:
            raise ValueError(
                f"action_names={self.action_names!r} with the pymdp-style control keywords: the "
                f"sugar names each controlled factor's action node a_<factor> ({list(actions)}); "
                "declare the control nodes natively to choose the action names"
            )
        self.action_names = actions
        for field in ("D_deps", "H_deps"):
            value = getattr(self, field)
            if value is not None and len(value) == n:
                setattr(self, field, (*tuple(value), *((),) * len(owners)))
        self.controls = tuple(node[f] for f in owners)

    def __check_init__(self) -> None:
        """Validate the name counts, the control nodes, ``epistemic`` and ``branch_on``."""
        for what, names, count in (
            ("factor_names", self.factor_names, len(self.B_deps)),
            ("modality_names", self.modality_names, len(self.A_deps)),
        ):
            if not isinstance(names, tuple) or not all(isinstance(n, str) for n in names):
                raise TypeError(f"{what} must be a tuple of str, got {type(names).__name__}")
            if len(names) != count:
                fix = (
                    "construct the System afresh with one name per node (its state factors, "
                    "then its control nodes)"
                    if what == "factor_names" and self.controls
                    else f"pass {what}=None to regenerate them, or give new names"
                )
                raise ValueError(
                    f"{what} has {len(names)} name(s) for {count} "
                    f"{'node' if what == 'factor_names' else 'modality'}(s): one name per node "
                    f"is required; {fix}"
                )
        if len(set(self.factor_names)) != len(self.factor_names):
            dup = sorted({n for n in self.factor_names if self.factor_names.count(n) > 1})
            raise ValueError(f"factor_names repeats {dup}: every node needs its own name")
        reserved = sorted(set(self.factor_names) & set(_KEY_KINDS))
        if reserved:
            raise ValueError(
                f"factor_names {reserved} are reserved: a key's first segment is one of "
                f"{list(_KEY_KINDS)} or a factor's name ('pose:b1', a dimension of 'pose'), so a "
                "factor named after one would be ambiguous; rename it"
            )
        self._check_dims()
        shared = sorted(set(self.factor_names) & set(self.modality_names))
        if shared:
            raise ValueError(
                f"{shared} name both a node and a modality: a System has one namespace of "
                "variables, so name the modality apart (for a sensor reading the node, "
                f"{shared[0]!r}_obs)"
            )
        self._check_action_names()
        last = len(self.B_deps) + len(self.controls)
        beyond = sorted({g for deps in self.A_deps for g in deps if not 0 <= g < last})
        if beyond:
            raise ValueError(
                f"A_deps names node(s) {beyond}, which are not nodes: a modality reads state "
                f"factors (0..{len(self.B_deps) - len(self.controls) - 1}) or action nodes "
                f"({len(self.B_deps)}..{last - 1})"
            )
        if not isinstance(self.epistemic, bool | tuple):
            raise ValueError(
                f"epistemic must be True, False, or a tuple of variable names, got "
                f"{self.epistemic!r}"
            )
        if not isinstance(self.branch_on, tuple) or not all(
            isinstance(n, str) for n in self.branch_on
        ):
            raise ValueError(f"branch_on must be a tuple of modality names, got {self.branch_on!r}")
        if self.branch_on and not planner_branches(self.planner):
            raise ValueError(
                f"branch_on={self.branch_on!r} names modalities to branch on, which only a "
                f'branching planner does (the "sophisticated" one, or one registered with '
                f"branching=True); this System plans with {self.planner!r}"
            )
        if self.planning_belief not in ("prior", "posterior"):
            raise ValueError(
                f'planning_belief must be "prior" or "posterior", got {self.planning_belief!r}'
            )
        if self.planning_belief == "posterior" and self.branch_on:
            raise ValueError(
                'planning_belief="posterior" conditions the plan on expected observations, and '
                "branch_on already branches the plan on their outcomes; declare one of the two"
            )
        if self.feedback is not None and self.feedback not in _closed_loop.FEEDBACK:
            raise ValueError(
                f"feedback must be None or one of {list(_closed_loop.FEEDBACK)}, got "
                f"{self.feedback!r}"
            )
        if self.feedback is not None and self.planning_belief != "posterior":
            raise ValueError(
                f"feedback={self.feedback!r} scores targets on the closed-loop state distribution, "
                "whose estimate spread the epistemic chain gives, and planning_belief="
                f"{self.planning_belief!r} carries none (the score would silently be the "
                'open-loop cross-entropy); set planning_belief="posterior"'
            )
        self._table  # noqa: B018  (validates the control nodes)
        self._check_lagged()
        self._check_C_deps()

    def _check_dims(self) -> None:
        """``dims``: pairs of a node name and a tuple of str, one entry per factor, each list of
        names well formed (:func:`~hybrid_aif.system.validation._check_dim_names`). The count
        against the factor's dimension is checked with the params."""
        dims = self.dims
        if not isinstance(dims, tuple) or not all(
            isinstance(entry, tuple)
            and len(entry) == 2
            and isinstance(entry[0], str)
            and isinstance(entry[1], tuple)
            and all(isinstance(n, str) for n in entry[1])
            for entry in dims
        ):
            raise TypeError(
                f"dims must map factor names to tuples of dimension names, got {dims!r}"
            )
        factors = [factor for factor, _ in dims]
        repeated = sorted({f for f in factors if factors.count(f) > 1})
        if repeated:
            raise ValueError(f"dims names {repeated} twice: one tuple of names per factor")
        unknown = sorted(set(factors) - set(self.factor_names))
        if unknown:
            raise ValueError(
                f"dims names unknown factor(s) {unknown}; the factors are "
                f"{sorted(self.factor_names)}"
            )
        for factor, names in dims:
            validation._check_dim_names(factor, names)

    def _check_C_deps(self) -> None:
        """``C_deps``: pairs of a targeted node's name and a tuple of parent names; a key is a
        state factor or a modality, a parent a control node or a state factor other than the key
        (whether a control parent is discrete, and a target's family and shape, are checked with
        the params)."""
        deps = self.C_deps
        if not isinstance(deps, tuple) or not all(
            isinstance(entry, tuple)
            and len(entry) == 2
            and isinstance(entry[0], str)
            and isinstance(entry[1], tuple)
            and all(isinstance(n, str) for n in entry[1])
            for entry in deps
        ):
            raise TypeError(
                f"C_deps must map a targeted node's name to a tuple of parent names, got {deps!r}"
            )
        table = self._table
        keys = [key for key, _ in deps]
        repeated = sorted({k for k in keys if keys.count(k) > 1})
        if repeated:
            raise ValueError(f"C_deps names {repeated} twice: one tuple of parents per target")
        targetable = set(table.state_names) | set(self.modality_names)
        for key, parents in deps:
            if key in table.control_names:
                raise ValueError(
                    f"C_deps names the control node {key!r}; a control node's own target is its "
                    "fixed cost per level (a Categorical in state_targets) and takes no parents"
                )
            if key not in targetable:
                raise ValueError(
                    f"C_deps names {key!r}, which is neither a state factor nor a modality; a "
                    "conditional target sits under the state factor (state_targets) or modality "
                    f"(obs_targets) it scores, one of {sorted(targetable)}"
                )
            if not parents:
                raise ValueError(f"C_deps[{key!r}] names no parent; drop the entry or name one")
            dup = sorted({p for p in parents if parents.count(p) > 1})
            if dup:
                raise ValueError(f"C_deps[{key!r}] names {dup} twice")
            for p in parents:
                if p == key:
                    raise ValueError(
                        f"C_deps[{key!r}] names {key!r} itself; a target conditioned on its own "
                        "belief is a moving target, not a preference: condition it on a control "
                        "node or another state factor"
                    )
                if p in self.modality_names:
                    raise ValueError(
                        f"C_deps[{key!r}] names the modality {p!r} as a parent; a target's "
                        "parents are discrete control nodes or state factors, read at the same "
                        "step"
                    )
                if p not in table.state_names and p not in table.control_names:
                    raise ValueError(
                        f"C_deps[{key!r}] names {p!r}, which is not a node; a target's parents "
                        "are discrete control nodes or state factors, one of "
                        f"{sorted((*table.state_names, *table.control_names))}"
                    )

    def _effective_C_deps(self) -> dict[str, tuple[tuple[str, int], ...]]:
        """``C_deps`` as the planning closures read it: each key's parents as ``("control",
        column)`` or ``("state", factor index)``, in declared order."""
        table = self._table
        return {
            key: tuple(
                ("control", table.control_names.index(p))
                if p in table.control_names
                else ("state", table.state_names.index(p))
                for p in parents
            )
            for key, parents in self.C_deps
        }

    def _check_action_names(self) -> None:
        """One action node per control node, paired by position with ``controls`` and named apart
        from every other node. A native declaration chooses both names freely; the sugar names
        the pair ``u_<f>`` and ``a_<f>`` for a state factor ``f``."""
        actions = self.action_names
        if not isinstance(actions, tuple) or len(actions) != len(self.controls):
            raise ValueError(
                f"action_names has {len(actions or ())} name(s) for {len(self.controls)} control "
                "node(s): declare one action name per control node, in controls order"
            )
        if len(set(actions)) != len(actions):
            dup = sorted({a for a in actions if actions.count(a) > 1})
            raise ValueError(f"action_names repeats {dup}: every action node needs its own name")
        clash = sorted(set(actions) & (set(self.factor_names) | set(self.modality_names)))
        if clash:
            raise ValueError(
                f"action_names {clash} are also factor or modality names: every node needs its "
                "own name"
            )
        empty = [a for a in actions if not a.rpartition(keyed.NAMESPACE_SEPARATOR)[2]]
        if empty:
            raise ValueError(f"action_names {empty} have an empty name: name every action node")

    @property
    def _table(self) -> ControlTable:
        """The control table every consumer reads (derived, cached on the declaration)."""
        return control_table(
            self.B_deps,
            self.A_deps,
            self.factor_names,
            self.controls,
            self.D_deps,
            self.H_deps,
        )

    def _sugar(self, what: str) -> SugarView:
        """The :class:`SugarView`, or a sentence naming the face ``what`` for a System whose
        nodes are not sugar-shaped."""
        sugar = self._table.sugar
        if sugar is None:
            raise NotImplementedError(
                f"{what}: this System's control nodes {list(self._table.control_names)} are not "
                "in the pymdp-style shape (each named u_<factor> after its own factor, after the "
                f"state parents of every transition that reads it), and {what} reads control "
                "nodes in that shape only on this System's declarations (tabular transitions "
                "read any control node)"
            )
        return sugar

    # ------------------------------------------------- the engine's view of the nodes

    @property
    def _B(self) -> tuple[tuple[int, ...], ...]:
        """Each state factor's state parents (control parents left out)."""
        return self._table.B_deps

    @property
    def _facs(self) -> tuple[str, ...]:
        """The state factors' names."""
        return self._table.state_names

    def _ctl(self, face: str) -> tuple[bool, ...]:
        """Per state factor, whether it owns a control node (the sugar view; ``face`` names the
        public face a refusal speaks for)."""
        return self._sugar(face).controlled

    def _cdeps(self, face: str) -> tuple[tuple[int, ...], ...] | None:
        """Per state factor, the owners of its control parents (``None``: default wiring)."""
        return self._sugar(face).B_control_deps

    def _space(self, params: SystemParams) -> ControlSpace:
        """Each control node's kind and size as ``params`` declare them
        (:func:`~hybrid_aif.controls.control_space`): a node with a box in ``params.control`` is
        continuous, its readers' widths checked against it; readers that disagree raise."""
        params = self._p(params)
        for f, parents in enumerate(self._B):  # switching dynamics, before any layout reading
            _hybrid.refuse_switching_nonlinear(params.transition[f], parents, self._facs, f)
        return control_space(self._table, params.transition, params.control, self._facs)

    def _control_layout(self, params: SystemParams) -> ControlLayout | None:
        """The flat action-vector layout of an all-continuous space, else ``None``."""
        return self._space(params).layout

    def action_dim(self, params: SystemParams) -> int | None:
        """Width of the flat continuous action vector, or ``None`` unless every control node is
        continuous (its box in ``params.control``)."""
        layout = self._control_layout(params)
        return None if layout is None else layout.action_dim

    def _control_levels(self, params: SystemParams) -> tuple[int, ...]:
        """Each control node's level count, read off its readers' tables (1 for a continuous
        node). Readers that disagree raise."""
        return self._space(params).levels

    def _num_controls(self, params: SystemParams, face: str) -> tuple[int, ...]:
        """The level count per state factor read off the tables: its own node's levels, 1 when
        it owns none."""
        sugar = self._sugar(face)
        levels = self._control_levels(params)
        by_owner = dict(zip(sugar.owners, levels, strict=True))
        return tuple(by_owner.get(f, 1) for f in range(self._table.num_states))

    def _actuator(self, params: SystemParams, face: str) -> dict[str, Array]:
        """``params.actuator`` checked: keyed by action node, each entry a ``(levels, levels)``
        probability table over a discrete node's levels, columns summing to one."""
        params = self._p(params)
        if not params.actuator:
            return {}
        names = self._names(params)
        keyed.check_mapping(
            names, "actions", params.actuator, face, label="params.actuator", partial=True
        )
        return {
            a: check_kernel(
                params.actuator[a], f"{face}: params.actuator[{a!r}]", names.kind(k), names.sizes[k]
            )
            for k, a in enumerate(names.actions)
            if a in params.actuator
        }

    def _durations(self, params: SystemParams, face: str) -> dict[str, Array]:
        """``params.durations`` checked: at most one entry, naming a discrete control node, each
        a positive scalar or a ``(levels,)`` array; returned as ``(levels,)`` arrays of the given
        float dtype (a float one, for an integer entry). Positivity is checked on concrete values
        (a traced entry is read as given)."""
        params = self._p(params)
        if not params.durations:
            return {}
        controls = self._table.control_names
        space = self._space(params)
        if len(params.durations) > 1:
            raise ValueError(
                f"{face}: params.durations names {sorted(params.durations)}: a round steps every "
                "factor once, so it has one clock; declare durations on one control node "
                "(design: step-durations.md T1)"
            )
        out = {}
        for node, value in params.durations.items():
            if node not in controls:
                raise ValueError(
                    f"{face}: params.durations names {node!r}, which is not a control node; "
                    f"the control nodes are {list(controls)}"
                )
            k = controls.index(node)
            if space.kinds[k] != LEVEL:
                raise ValueError(
                    f"{face}: params.durations[{node!r}] is on a continuous control node; a "
                    "duration is one number per level of a discrete one"
                )
            levels = space.sizes[k]
            arr = jnp.asarray(value)
            if not jnp.issubdtype(arr.dtype, jnp.floating):
                arr = arr.astype(jnp.result_type(float))
            if arr.ndim == 0:
                arr = jnp.broadcast_to(arr, (levels,))
            if arr.shape != (levels,):
                raise ValueError(
                    f"{face}: params.durations[{node!r}] has shape {tuple(arr.shape)}; control "
                    f"node {node!r} has {levels} level(s), so give {levels} number(s) or one "
                    "for every level"
                )
            if not isinstance(arr, jax.core.Tracer) and not bool(jnp.all(arr > 0)):
                raise ValueError(
                    f"{face}: params.durations[{node!r}] = {arr.tolist()} has a "
                    "non-positive entry; a duration is a positive number per level"
                )
            out[node] = arr
        return out

    def elapsed(self, action: Mapping[str, Array], params: SystemParams) -> Array:
        """The duration of a realised round: the entry of ``params.durations`` at the level of
        the realised action of the node that carries it (1.0 when no durations are declared).

        ``action`` is keyed by action node, as the environment returns it and :meth:`act` emits
        it; under a stochastic actuator it is the realised level, not the decided one, that is
        looked up. Array in, array out: works under ``jit``. Design: step-durations.md T8.
        """
        params = self._params_in(params, "elapsed")
        durations = self._durations(params, "elapsed")
        if not durations:
            return jnp.asarray(1.0, jnp.result_type(float))
        ((node, table),) = durations.items()
        a = self.action_names[self._table.control_names.index(node)]
        if a not in action:
            raise ValueError(
                f"elapsed: the action has no entry for {a!r}, the action node of {node!r}, which "
                f"carries the durations; the action is keyed by {list(self.action_names)}"
            )
        return table[jnp.asarray(action[a], jnp.int32)]

    def _modality_types(self, params: SystemParams) -> dict[str, tuple[str, int]]:
        """Each modality's port type, by name: ``(LEVEL, levels)`` for a tabular likelihood,
        ``(FLOAT, dimension)`` for a linear-Gaussian or neural one, read off its declaration."""
        params = self._p(params)
        return {
            m: _observation_type(decl, f"params.likelihood[{m!r}]")
            for m, decl in zip(self.modality_names, params.likelihood, strict=True)
        }

    # ------------------------------------------------------------ names at the faces

    def _names(self, params: SystemParams | None = None) -> keyed.FaceNames:
        """The names the conversions read; with ``params``, each control node's kind and size
        too (a continuous node's box lives in ``params.control``)."""
        table = self._table
        if table.sugar is not None:
            owners = table.sugar.owners
        else:  # each node at its first reader's column, when no two nodes share one
            owners = tuple(readers[0][0] for readers in table.readers)
            owners = owners if len(set(owners)) == len(owners) else None
        gone = self._bound_away()
        if params is None or not table.controls:
            return keyed.FaceNames(
                table.state_names,
                self.modality_names,
                owners,
                None,
                table.control_names,
                gone=gone,
                actions=self.action_names,
                sensed=self._sensed,
            )
        space = self._space(params)
        return keyed.FaceNames(
            table.state_names,
            self.modality_names,
            owners,
            None if space.layout is None else space.layout.structure(),
            table.control_names,
            space.kinds,
            space.sizes,
            gone,
            self.action_names,
            self._sensed,
        )

    def _bound_away(self, prefix: str = "") -> tuple[tuple[str, str], ...]:
        """Each control node, and its action node, a ``merge()`` binding removed, by name, with
        the rest of the sentence that refuses a mapping keyed by it (what the node became), read
        off ``subsystems``. ``prefix`` qualifies every name, for a merge that nests this one."""
        if not self.subsystems:
            return ()
        return _bound_away(self.subsystems, self._table.control_names, self.action_names, prefix)

    @property
    def names(self) -> keyed.Names:
        """A read-only view of the System's names, every tuple in node-index order.

        ``names.states`` are the state factors' names (the belief keys), ``names.controls`` the
        control nodes', both from ``factor_names``; ``names.modalities`` is ``modality_names``;
        ``names.actions`` the action nodes' (the action keys), one per control node in the same
        order, from ``action_names``. Node-index order is the order of every positional
        structure (tables, policy columns, what the converters produce); iterating a mapping
        gives no such guarantee, so take order from ``names``.
        """
        return self._names().public

    def _node_priors(self, params: SystemParams) -> dict[str, Categorical]:
        """Each discrete control node's prior, by name: uniform over its levels (C2's
        independence at every step)."""
        names = self._names(params)
        if not names.node_beliefs:
            return {}
        levels = self._control_levels(params)
        return {
            n: Categorical(logits=jnp.zeros((levels[k],)))
            for k, n in enumerate(names.controls)
            if n in names.node_beliefs
        }

    def _beliefs_in(
        self, params: SystemParams, mapping: Mapping[str, Belief], face: str
    ) -> tuple[Belief, ...]:
        """A face's beliefs as the state factors' positional tuple, each control node's entry
        (optional) checked: a Categorical over that node's levels."""
        names = self._names(params)
        out = keyed.beliefs_in(names, mapping, face)
        given = keyed.node_beliefs_in(names, mapping)
        if given:
            levels = dict(zip(names.controls, self._control_levels(params), strict=True))
            for name, q in given.items():
                size = q.probs.shape[-1] if isinstance(q, Categorical) else None
                if size != levels[name]:
                    got = (
                        f"a Categorical over {size} level(s)"
                        if isinstance(q, Categorical)
                        else f"a {type(q).__name__}"
                    )
                    raise ValueError(
                        f"{face}: beliefs[{name!r}] is the belief over control node {name!r}, a "
                        f"Categorical over its {levels[name]} level(s); got {got}"
                    )
        return out

    def _beliefs_out(
        self,
        params: SystemParams,
        beliefs,
        given: Mapping[str, Belief] | None = None,
        sensed: Mapping[str, Belief] | None = None,
    ) -> dict[str, Belief]:
        """The positional state beliefs keyed, with every discrete control node's entry (the
        one in ``given`` when it carries it, else the node's prior), then the observed action
        nodes' entries ``sensed`` when given."""
        names = self._names(params)
        nodes = self._node_priors(params)
        if given is not None:
            nodes.update(keyed.node_beliefs_in(names, given))
        if sensed:
            nodes.update(sensed)
        return keyed.beliefs_out(names, beliefs, nodes)

    # ------------------------------------------------- action nodes a modality observes

    @property
    def _sensed_columns(self) -> tuple[int, ...]:
        """The action columns some modality's ``A_deps`` names (action node ``k`` is node
        ``len(B_deps) + k``), ascending."""
        return _sensed_columns(self.A_deps, len(self.B_deps))

    @property
    def _sensed(self) -> tuple[str, ...]:
        """The names of the action nodes some modality observes."""
        return tuple(self.action_names[k] for k in self._sensed_columns)

    def _refuse_sensed(self, face: str) -> None:
        """Raise on a face that does not read a modality on an action node."""
        if not self._sensed_columns:
            return
        n = len(self.B_deps)
        mods = [
            self.modality_names[m]
            for m, deps in enumerate(self.A_deps)
            if max(deps, default=-1) >= n
        ]
        raise NotImplementedError(
            f"{face}: modality(ies) {mods} observe action node(s) {list(self._sensed)}, which "
            f"{face} does not read; drop the action node from their A_deps to use {face}"
        )

    def _check_sensed_kinds(self, params: SystemParams, face: str) -> None:
        """Raise when a modality observes the action node of a continuous control node."""
        if not self._sensed_columns:
            return
        kinds = self._space(params).kinds
        continuous = [self.action_names[k] for k in self._sensed_columns if kinds[k] != LEVEL]
        if continuous:
            raise NotImplementedError(
                f"{face}: a modality observes action node(s) {continuous} of a continuous "
                "control node, whose values have no levels for a tabular likelihood to read; "
                "drop the action node from its A_deps"
            )

    def _sensed_priors(
        self, params: SystemParams, given: Mapping[str, Belief] | None, face: str
    ) -> tuple[Categorical, ...]:
        """Each observed action node's prior, in column order: the entry ``given`` carries,
        else uniform over its control node's levels."""
        self._check_sensed_kinds(params, face)
        levels = self._control_levels(params)
        out = []
        for k, name in zip(self._sensed_columns, self._sensed, strict=True):
            q = None if given is None else given.get(name)
            if q is None:
                out.append(Categorical(logits=jnp.zeros((levels[k],))))
                continue
            if not isinstance(q, Categorical) or q.probs.shape[-1] != levels[k]:
                raise ValueError(
                    f"{face}: beliefs[{name!r}] is the belief over action node {name!r}, a "
                    f"Categorical over its {levels[k]} level(s)"
                )
            out.append(q)
        return tuple(out)

    @property
    def _engine_A_deps(self) -> tuple[tuple[int, ...], ...]:
        """``A_deps`` at the engine's positions: an observed action node follows the state
        factors, in :attr:`_sensed_columns` order."""
        n, s = len(self.B_deps), self._table.num_states
        at = {n + k: s + i for i, k in enumerate(self._sensed_columns)}
        return tuple(tuple(at.get(g, g) for g in deps) for deps in self.A_deps)

    def _sensed_nodes(self, params: SystemParams, face: str) -> SensedNodes | None:
        """The observed action nodes as the planning closures read them, or ``None`` when no
        modality observes one."""
        if not self._sensed_columns:
            return None
        self._check_sensed_kinds(params, face)
        space = self._space(params)
        kernels = self._actuator(params, face)
        columns = self._sensed_columns
        return SensedNodes(
            offset=self._table.num_states,
            columns=columns,
            names=self._sensed,
            levels=tuple(space.levels[k] for k in columns),
            kernels={
                k: kernels[self.action_names[k]] for k in columns if self.action_names[k] in kernels
            },
            layout=space.layout,
        )

    def _sensed_predictions(
        self, params: SystemParams, action: Array, face: str = "update_empirical_prior"
    ) -> dict[str, Categorical]:
        """The agent's prediction of each observed action node under the positional
        ``action`` (one level per control node): its kernel's column at the decision, the
        one-hot of the decision under the delta kernel."""
        sensed = self._sensed_nodes(params, face)
        if sensed is None:
            return {}
        return {
            name: Categorical.from_probs(probs)
            for name, probs in zip(sensed.names, action_node_beliefs(action, sensed), strict=True)
        }

    def beliefs_from_states(self, params: SystemParams, states) -> dict[str, object]:
        """An amortized model's filter states as the beliefs the faces take, keyed by factor.

        ``states`` is an :class:`~hybrid_aif.inference.rssm.RSSMState` (e.g.
        :func:`~hybrid_aif.train.world.posterior_states`): the carry's vector and the code's
        tensor, by factor name, leading batch and time axes passing through (flatten them for
        :func:`~hybrid_aif.train.imagine.imagine`'s ``(N, ...)`` start beliefs).
        """
        from hybrid_aif.train.world import belief_tuple

        params = self._params_in(params, "beliefs_from_states")
        positional = belief_tuple(self._amortized_structure(params, "beliefs_from_states"), states)
        return self._beliefs_out(params, positional)

    def _p(self, params: "SystemParams"):
        """``params`` with positional per-variable containers, the form the internals index.
        Idempotent (positional containers pass through)."""
        return keyed.params_positional(self._names(), params)

    def _params_in(self, params: "SystemParams", face: str):
        """A face's params: keyed containers required (positional ones raise, naming the
        converter), returned as the positional view."""
        return keyed.params_in(self._names(), params, face)

    def _params_out(self, params: "SystemParams"):
        """``params`` with keyed per-variable containers, the form the faces return."""
        return keyed.params_keyed(self._names(), params)

    def to_positional(
        self,
        kind: keyed.Kind,
        keyed_value: Mapping[str, object] | InferenceDetail | SystemParams,
        params: SystemParams | None = None,
    ) -> "tuple | Array | InferenceDetail | SystemParams":
        """A name-keyed value as the positional form the internals use.

        ``kind`` and what comes back: ``"beliefs"`` → a tuple of the state factors' beliefs, in
        factor order (a control node's entry is left out); ``"observations"`` → a tuple in
        modality order (a complete ``present`` mapping converts as observations); ``"actions"``
        (keyed by action node) → an int32 ``(..., K)`` level array, one column per action node
        in node order, the
        flat float ``(..., action_dim)`` vector when every node is continuous, or a
        :class:`~hybrid_aif.controls.ControlStep` (one entry per node) over a mixed space;
        ``"detail"`` → an :class:`~hybrid_aif.inference.InferenceDetail` with tuple fields;
        ``"params"`` → a :class:`SystemParams` (or environment params) with tuple containers.
        Missing or unknown keys raise. ``params`` (keyed) is needed to convert a continuous
        node's value, whose box lives in ``params.control``. See :mod:`hybrid_aif.system.keyed`.
        """
        return keyed.to_positional(self._face_names(params), kind, keyed_value)

    def from_positional(
        self, kind: keyed.Kind, positional, params: SystemParams | None = None
    ) -> "dict[str, object] | InferenceDetail | SystemParams":
        """Positional data as the name-keyed form every face takes (inverse of
        :meth:`to_positional`).

        ``kind`` and what comes back: ``"beliefs"`` / ``"observations"`` (a tuple in node order)
        → a dict by state factor / modality name; ``"actions"`` (an ``(..., K)`` level array, one
        column per action node, the flat continuous vector, or a mixed space's
        :class:`~hybrid_aif.controls.ControlStep`) → a dict by action-node name; ``"detail"``
        (an :class:`~hybrid_aif.inference.InferenceDetail` with tuple fields) → the same record
        with dict fields; ``"params"`` (a :class:`SystemParams` or environment params with tuple
        containers) → the same object with dict containers. ``params`` (keyed) is needed to
        convert a continuous node's value, whose box lives in ``params.control``.
        """
        return keyed.from_positional(self._face_names(params), kind, positional)

    def _face_names(self, params: SystemParams | None) -> keyed.FaceNames:
        """:meth:`_names` for a converter: ``params`` arrive keyed."""
        if params is None:
            return self._names()
        return self._names(self._params_in(params, "the converter"))

    # ------------------------------------------------------------ family dispatch

    def _is_hybrid(self, params: SystemParams) -> bool:
        """Whether the declarations leave the compiled tabular path.

        Fully tabular models, conditional and learnable tabular forms included, stay discrete. Any
        other declaration type routes to the hybrid classifiers, whose errors name what is
        unsupported.
        """
        params = self._p(params)
        return (
            any(not isinstance(a, TabularObservation) for a in params.likelihood)
            or any(not isinstance(b, TabularTransition) for b in params.transition)
            or any(not isinstance(d, TabularPrior) for d in params.initial_state_prior)
        )

    def _effective_D_deps(self) -> tuple[tuple[int, ...], ...]:
        """The state factors' ``D_deps`` entries (all empty when undeclared)."""
        deps = self._table.D_deps
        return deps if deps is not None else ((),) * len(self._B)

    def _effective_H_deps(self) -> tuple[tuple[int, ...], ...]:
        """The state factors' ``H_deps`` entries (all empty when undeclared)."""
        deps = self._table.H_deps
        return deps if deps is not None else ((),) * len(self._B)

    def resolve_terminal_targets(
        self, params: SystemParams, final_beliefs: Mapping[str, Belief]
    ) -> dict[str, Distribution]:
        """The effective terminal designations at ``final_beliefs``, with ``H_deps`` resolved.

        A terminal target is ``h``, the prior over intended final states, reciprocal to the initial
        prior ``d``. An ``H_deps`` entry makes it a
        :class:`~hybrid_aif.distributions.ConditionalCategorical` over the designated factor,
        conditional on parent factors (under ``merge`` possibly a higher System's). It resolves
        mean-field, as ``D_deps`` does in :meth:`initial_priors` — ``Σ_parents h(· | parents)
        q(parents)`` — with ``q(parents)`` read from the final predicted beliefs of the rollout
        being scored, so a parent the plan can steer makes the target plan-dependent. Learnable and
        continuous conditional targets raise.

        Design: docs/design/nesting-systems.md N1.
        """
        params = self._params_in(params, "resolve_terminal_targets")
        final = self._beliefs_in(params, final_beliefs, "resolve_terminal_targets")
        return resolve_conditional_targets(
            params.terminal_targets,
            self._effective_H_deps(),
            final,
            self._facs,
        )

    def initial_priors(self, params: SystemParams) -> dict[str, Belief]:
        """Ancestral mean-field marginals of the initial priors, ``D_deps`` resolved, by factor.

        A conditional prior is averaged over its parents' resolved marginals in declaration order,
        unconditional priors pass through, and a learnable prior contributes its posterior-mean
        ``point()``. ``infer_states`` uses this when ``priors=None``; planning faces take it in
        place of a raw conditional ``D``. On the amortized path it returns the episode-start values:
        each carry's learned initial value, and each code's transition read at it.
        """
        params = self._params_in(params, "initial_priors")
        return self._beliefs_out(params, self._initial_priors(params, "initial_priors"))

    def _initial_priors(self, params: SystemParams, face: str = "this face") -> tuple[Belief, ...]:
        """:meth:`initial_priors` over the positional tuple."""
        params = self._p(params)
        if self._is_amortized(params):
            structure = self._amortized_structure(params, face)
            return _amortized.initial_state(
                params.transition, params.initial_state_prior, self._B, structure.families
            )
        deps_all = self._effective_D_deps()
        out: list[Belief] = []
        for d, deps in zip(params.initial_state_prior, deps_all, strict=True):
            if isinstance(d, Learnable):
                d = d.point()
            if isinstance(d, ConditionalCategorical):
                out.append(d.average(tuple(out[g] for g in deps)))
            else:
                out.append(d)
        return tuple(out)

    def _families(self, params: SystemParams) -> tuple[str, ...]:
        params = self._p(params)
        return selectors._families(self, params)

    # -------------------------------------------------------------- forward face

    def as_env(
        self, params: SystemParams, param_key: Array | None = None
    ) -> tuple[GenerativeEnv, GenerativeEnvParams]:
        """The same tensors run forward: ``(env, env_params)``.

        ``env`` is a :class:`~hybrid_aif.envs.GenerativeEnv` (observations by modality name,
        states by factor name, actions by action-node name) over the discrete process
        (probability tables) for an all-tabular System, the hybrid one (the declarations
        themselves) otherwise; ``env_params`` keys its containers by name and carries
        ``params.actuator``, from which the env realises each action it is given.
        Learnable slots are lowered to their posterior mean, or with ``param_key`` to one
        ``sample`` each, so the env is a draw from the model's own parameter prior. Raises on the
        amortized path.
        """
        params = self._params_in(params, "as_env")
        if self._is_amortized(params):
            self._amortized_structure(params, "as_env")
            raise NotImplementedError(
                "as_env on an amortized (neural state-space) System: running these declarations "
                "forward as an environment means sampling the code prior and decoding its heads "
                "— which is imagination, not an environment (nothing in it emits a reward the "
                "agent has not predicted, so it cannot score the agent, only agree with it — and "
                "every as_env consumer, from test oracles to the nesting driver's ground truth, "
                "assumes scoring). The rollout face for it is "
                "hybrid_aif.train.imagine.imagine over predict_beliefs (design: "
                "amortization.md A6/A11); predict_beliefs itself is the open-loop "
                "prediction, run against a concrete environment of the caller's. "
                "The one coherent unfence, recorded for whoever wants "
                "it: a FROZEN world model stipulated as a synthetic "
                "environment is just a declared generative process, and could be admitted "
                "here on that stipulation — the fence holds while the "
                "model is learning."
            )
        lowered = self._lower_learnable(params, param_key)
        if self._is_hybrid(params):
            self._validate(params, "as_env")
            space = self._space(params)
            env = HybridProcess(
                A_deps=self._engine_A_deps,
                B_deps=self._B,
                controlled=tuple(bool(p) for p in self._table.control_parents),
                D_deps=self._effective_D_deps(),
                controls=None if space.layout is None else space.layout.structure(),
                B_control_deps=self._table,
                kinds=space.kinds,
                sizes=space.sizes,
                sensed=self._sensed_columns,
            )
            return GenerativeEnv(env, self._names(params)), self._params_out(
                GenerativeEnvParams(
                    likelihood=lowered.likelihood,
                    transition=lowered.transition,
                    initial_state_prior=lowered.initial_state_prior,
                    actuator=self._actuator(params, "as_env"),
                )
            )
        validation._validate_table_arity(self, params, self._facs)
        if not self._table.default_wiring:
            self._validate(params, "as_env")
        n, s = len(self.B_deps), self._table.num_states
        at = {n + k: s + i for i, k in enumerate(self._sensed_columns)}
        env = DiscreteProcess(
            A_deps=tuple(tuple(at.get(g, g) for g in deps) for deps in self.A_deps),
            B_deps=self._B,
            D_deps=self._effective_D_deps(),
            B_control_deps=self._table,
            sensed=self._sensed_columns,
        )
        return GenerativeEnv(env, self._names(params)), self._params_out(
            GenerativeEnvParams(
                likelihood=tuple(jnp.exp(a.log_probs) for a in lowered.likelihood),
                transition=tuple(jnp.exp(b.log_probs) for b in lowered.transition),
                initial_state_prior=tuple(
                    d.probs if isinstance(d, Categorical) else jnp.exp(d.log_probs)
                    for d in lowered.initial_state_prior
                ),
                actuator=self._actuator(params, "as_env"),
            )
        )

    def _lower_learnable(self, params: SystemParams, param_key: Array | None) -> SystemParams:
        params = self._p(params)
        return conjugate_stats._lower_learnable(params, param_key)

    # ------------------------------------------------------------- backward faces

    def infer_states(
        self,
        params: SystemParams,
        observations: Mapping[str, Array | None],
        priors: Mapping[str, Belief] | None = None,
        detail: bool = False,
        externals: Mapping[str, Array] | None = None,
        present: Mapping[str, bool | Array] | None = None,
        support_floor: float = SUPPORT_FLOOR,
        key: Array | None = None,
        previous: Mapping[str, Belief | Array] | None = None,
    ) -> tuple[dict[str, Belief], InferenceDetail | None]:
        """Mean-field state posteriors for one tick's observations: ``(beliefs, detail)``.

        ``observations`` is keyed by modality name, one entry per modality: an ``int32`` level (or
        one-hot) for a categorical modality, a float array for an LG one. ``priors`` and the
        returned beliefs are keyed by factor name; ``priors=None`` uses :meth:`initial_priors`.
        A discrete control node's entry is optional in ``priors`` and always returned: no
        inference runs on it, so it passes through (its uniform prior when absent). An action
        node a modality observes (``A_deps`` naming it) is inferred like a state factor on the
        tabular path: its entry in ``priors`` is its prior (uniform when absent), and its
        posterior is returned under its name.
        With ``detail=True`` (a static bool) the second slot is an
        :class:`~hybrid_aif.inference.InferenceDetail` (pre-reduction switching mixtures, per-mode
        and total log-evidence keyed by modality name; ``unsupported`` keyed by factor name), else
        ``None``; the beliefs are identical either way.

        ``present`` says which modalities are observed this tick, a flag per modality keyed by
        name: a modality it leaves out is present, ``None`` means all are, and an unknown key
        raises. An absent modality is not clamped: its value is ignored and its factors keep what
        the other modalities leave them; all absent is a predict-only tick. A traced flag
        computes the update and selects against it (its ``detail`` entries report the
        hypothetical update, ``log_evidence`` zeroed); a Python ``False`` skips it statically. On
        the amortized path the modalities the recognition modules read are present or absent
        together: static flags splitting them raise, traced ones reduce with ``all`` (absent).

        ``detail.unsupported`` flags each categorical factor none of whose levels explains this
        tick's tabular evidence, as :func:`hybrid_aif.inference.discrete.unsupported` defines it
        against ``support_floor``; the posterior is left as computed.

        Hybrid Systems run one structured sweep in declaration order: the discrete FPI over the
        categorical sub-model; exact LG ``condition`` per LG modality, sequential in modality order;
        exact Gaussian-sum conditioning for a mixture-carried factor, with a mode-parent evidence
        message under ``mixture_schedule="mode_aligned"``; and grow-then-reduce conditioning for a
        switching LG modality (categorical mode parents after the observed factor in ``A_deps``),
        under either schedule. A modality that observes a mixture-carried factor directly (one
        whose *transition* switches) raises under ``"runnalls"`` and needs ``"mode_aligned"``;
        non-LG likelihoods on continuous factors raise.

        On the amortized path this runs the recognition modules in order
        (:func:`hybrid_aif.inference.amortized.recognize`): the value ``None`` is allowed for a
        modality no module reads (its key is still present), ``externals`` supplies
        ``"external:<name>"`` parents, and ``detail`` is ``None``.

        With ``key`` the posterior is collapsed onto one draw, as a belief of its own family
        (one-particle filtering; amortized carries pass through; on a predict-only tick it is
        the prior that collapses). See docs/composing-systems.md.

        ``previous`` is the carry a lagged modality (``System.lagged``) reads its datum from:
        the previous tick's beliefs keyed by factor name and the previous action keyed by
        action name, any entry optional. A lagged modality whose observation is ``None`` takes
        its datum from there; with no entry for its source it is absent.
        """
        return _infer_face.infer_states(
            self,
            params,
            observations,
            priors,
            detail,
            externals,
            present,
            support_floor,
            key,
            previous,
        )

    def _infer_states(
        self,
        params: SystemParams,
        observations: tuple[Array | None, ...],
        priors: tuple[Belief, ...] | None = None,
        detail: bool = False,
        externals: Mapping[str, Array] | None = None,
        present: tuple[bool | Array, ...] | Array | None = None,
        support_floor: float = SUPPORT_FLOOR,
        key: Array | None = None,
        face: str = "this face",
        previous: tuple | None = None,
    ) -> tuple[tuple[Belief, ...], InferenceDetail | None]:
        """The engine: :func:`hybrid_aif.system.faces.infer._infer_states`."""
        return _infer_face._infer_states(
            self,
            params,
            observations,
            priors,
            detail,
            externals,
            present,
            support_floor,
            key,
            face,
            previous,
        )

    @property
    def _lagged_sources(self) -> dict[int, tuple[str, int]]:
        """Each lagged modality's index to ``("factor", f)`` or ``("action", k)``."""
        out: dict[int, tuple[str, int]] = {}
        mods, facs = self.modality_names, self._facs[: len(self.B_deps)]
        actions = tuple(self.action_names or ())
        for modality, source in self.lagged:
            m = mods.index(modality)
            if source in facs:
                out[m] = ("factor", facs.index(source))
            else:
                out[m] = ("action", actions.index(source))
        return out

    def _check_lagged(self) -> None:
        if not self.lagged:
            return
        mods, facs = self.modality_names, self._facs[: len(self.B_deps)]
        actions = tuple(self.action_names or ())
        seen = set()
        for pair in self.lagged:
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise ValueError(f"lagged holds (modality, source) pairs of names, got {pair!r}")
            modality, source = pair
            if modality not in mods:
                raise ValueError(
                    f"lagged names the modality {modality!r}, which is not one of {list(mods)}"
                )
            if modality in seen:
                raise ValueError(f"lagged names the modality {modality!r} twice")
            seen.add(modality)
            if source not in facs and source not in actions:
                raise ValueError(
                    f"lagged: modality {modality!r} reads {source!r} at the previous tick, which "
                    f"is neither a state factor {list(facs)} nor an action node {list(actions)}"
                )

    def _with_lagged(self, observations: Mapping[str, Array | None]) -> Mapping:
        """``observations`` with a ``None`` entry for every lagged modality it leaves out: a
        lagged modality's datum comes from the carry, so its key is optional."""
        if not self.lagged or not isinstance(observations, Mapping):
            return observations
        return {**{o: None for o, _ in self.lagged}, **dict(observations)}

    def _fill_lagged(
        self,
        observations: tuple[Array | None, ...],
        present: tuple[bool | Array, ...] | None,
        prev_beliefs: tuple[Belief | None, ...] | None,
        prev_action: Array | None,
        first: Array | bool = False,
    ) -> tuple[tuple[Array | None, ...], tuple[bool | Array, ...] | None]:
        """The engine: :func:`hybrid_aif.system.faces.infer._fill_lagged`."""
        return _infer_face._fill_lagged(
            self,
            observations,
            present,
            prev_beliefs,
            prev_action,
            first,
        )

    @staticmethod
    def _present(
        present: tuple[bool | Array, ...] | Array | None, n_mods: int
    ) -> tuple[bool | Array, ...] | None:
        """The engine: :func:`hybrid_aif.system.faces.infer._present`."""
        return _infer_face._present(present, n_mods)

    # -------------------------------------------------------------- learning face

    def learn(
        self,
        params: SystemParams,
        observations: Mapping[str, Array],
        beliefs: Mapping[str, Belief],
        prev_beliefs: Mapping[str, Belief] | None = None,
        action: Mapping[str, Array] | None = None,
        lr: float | Array = 1.0,
        decay: float | Array = 1.0,
        lr_by_slot: Mapping[str, float | Array] | None = None,
        decay_by_slot: Mapping[str, float | Array] | None = None,
        detail: InferenceDetail | None = None,
        learn_state: "LearnState | None" = None,
        prior: SystemParams | None = None,
    ) -> "SystemParams | tuple[SystemParams, LearnState]":
        """One tick's parameter update: every learnable slot absorbs this tick's statistic.

        A slot holding a distribution over its parameters absorbs its conjugate sufficient
        statistic; point-valued slots are not learnable. A non-frozen
        :class:`~hybrid_aif.distributions.MLPConditional` takes an optax step on its negative
        log-density at point evidence from the beliefs (posterior means, probability vectors for
        discrete variables, a one-hot action). A transition net's parents are its factor's previous
        value then the action one-hot if controlled; a likelihood net's are the ``A_deps`` factors
        in order.

        ``observations`` is keyed by modality name, ``beliefs`` and ``prev_beliefs`` by factor
        name, ``action`` by action-node name (as :meth:`act` returns it). ``action`` is
        **a_{t-1}**, the action that produced this tick's observations; each control node is
        clamped to its value. ``prev_beliefs`` and
        ``action`` are required exactly when a learnable transition has a nonzero-or-traced rate
        (``action`` only if that factor is controlled). ``detail`` is the keyed
        :class:`~hybrid_aif.inference.InferenceDetail` ``infer_states(detail=True)`` returned.

        Returns bare params for a conjugate-only model, ``(params, learn_state)`` when the model has
        a gradient slot or ``learn_state`` is passed. Start the carry with
        ``learn_state=system.init_learn_state(params)``; ``None`` with gradient slots present
        raises. Raises on the amortized path (train that with
        :func:`hybrid_aif.train.world.fit_world`).

        **Rates.** ``lr`` applies to every learnable slot; ``lr_by_slot`` overrides it by
        slot-qualified key (``"likelihood:<modality>"``, ``"transition:<factor>"``,
        ``"initial_state_prior:<factor>"``); ``decay``/``decay_by_slot`` use the same keys. A third
        segment addresses one independent posterior block (``"likelihood:pose:noise"`` /
        ``"…:head"``; the joint ``LinearGaussianObservationMNIW`` has none, and its sub-block keys
        raise). Any segment may be
        ``*``; the most specific match (most literal segments) wins, and equal-specificity matches
        on one variable raise::

            decay_by_slot={"*:*": 0.999, "likelihood:pose": 0.97}
            lr_by_slot={"likelihood:pose": 0.0, "likelihood:pose:noise": 1.0}  # head frozen

        A static Python ``0.0`` skips the slot at trace time, decay included; a traced rate
        multiplies the statistic (``jnp.where(t < 1000, 1.0, 0.0)``). ``jnp.float32(0.0)`` is
        traced, and a frozen slot that should still drift under decay needs a traced zero. State and
        ``obs:`` keys are not rate-addressable and raise.

        **Decay** is ``α ← decay·α + (1−decay)·anchor + lr·stats``, anchored at the declaration's
        own anchor (its prior, a uniform table, or an array it was built with); ``decay == 1.0``
        is exact conjugacy and reads no anchor; any other decay, traced included, raises on a
        declaration without one and on a neural slot.

        **Timing.** The transition statistic is the product of filtered marginals (an
        approximation). At tick 0 pass a dummy ``prev_beliefs`` with a traced zero rate; gate
        initial-prior rates to tick 0, since they accumulate once per episode. ``learn`` is
        unbatched; lift it with ``vmap``.

        **Growth.** A growable factor (a memoryless factor whose weights carry a growth
        propensity) runs its growth step first: the tick's data scored under every level, the
        first dead level woken when the best live level explains them worse than the prior does
        by the propensity's margin, the woken level taking the datum hard and seeded from it by
        the ordinary update. Every ``reduction_period``-th call then reduces each growable factor
        (:meth:`reduce`), which needs ``prior``, the model's prior params (the statistics a merge
        pools are the posterior's beyond them, and a killed level returns to them), and a
        ``learn_state`` carrying the counter (:meth:`init_learn_state`). Both move with the
        weights: a tick where no modality over the factor is present grows nothing, and the
        weights' rate (``transition:<factor>``) gates growth and reduction, a static zero
        freezing the factor's structure and a traced zero skipping it at run time.

        **Lagged modalities** (``System.lagged``) read their datum from ``prev_beliefs`` and
        ``action`` and may be left out of ``observations``; at tick 0, where ``prev_beliefs`` is
        a dummy, freeze their likelihood slots and the weights with a traced zero rate as for a
        transition.

        Design: docs/design/parameter-learning.md D1 D3 D4 D5 D7 D8;
        docs/design/per-variable-infogain.md V1 V7; docs/design/amortization.md A2;
        docs/design/structure-learning.md S3 S4.
        """
        return _learn_face.learn(
            self,
            params,
            observations,
            beliefs,
            prev_beliefs,
            action,
            lr,
            decay,
            lr_by_slot,
            decay_by_slot,
            detail,
            learn_state,
            prior,
        )

    def _learn(
        self,
        params: SystemParams,
        observations: tuple[Array, ...],
        beliefs: tuple[Belief, ...],
        prev_beliefs: tuple[Belief, ...] | None = None,
        action: Array | None = None,
        lr: float | Array = 1.0,
        decay: float | Array = 1.0,
        lr_by_slot: Mapping[str, float | Array] | None = None,
        decay_by_slot: Mapping[str, float | Array] | None = None,
        detail: InferenceDetail | None = None,
        learn_state: "LearnState | None" = None,
        face: str = "this face",
        present: tuple[bool | Array, ...] | None = None,
        prior: SystemParams | None = None,
        grow: bool = True,
    ) -> "SystemParams | tuple[SystemParams, LearnState]":
        """The engine: :func:`hybrid_aif.system.faces.learn._learn`."""
        return _learn_face._learn(
            self,
            params,
            observations,
            beliefs,
            prev_beliefs,
            action,
            lr,
            decay,
            lr_by_slot,
            decay_by_slot,
            detail,
            learn_state,
            face,
            present,
            prior,
            grow,
        )

    def carry(
        self,
        fresh: SystemParams,
        learned: SystemParams,
        slots: Iterable[str] | None = None,
    ) -> SystemParams:
        """``fresh`` with the posteriors learning put into ``learned`` carried over: for an
        agent that rebuilds its params every step and must keep what it has learned. A block is
        one posterior :meth:`learn` updates (a Dirichlet or conditional Dirichlet, a
        matrix-normal inverse-Wishart emission, a linear-Gaussian observation's learnable
        noise, a trainable neural conditional), carried whole, its anchors and growth
        propensity included. Every block is carried when ``slots`` is ``None``, else the ones
        the keys name: ``"likelihood:<modality>"``, ``"transition:<factor>"``,
        ``"initial_state_prior:<factor>"`` or ``"likelihood:<modality>:noise"``, without
        wildcards (under ``jit``, pass ``slots`` as a static argument). Everything else comes
        from ``fresh``: point values, and a separate ``MatrixNormal`` head, which ``learn`` does
        not update. A block whose family, shape or dtype differs between the two raises, as do
        a selected slot that holds a posterior on one side only, a key that names no posterior,
        and a selection that would split a growable factor's blocks, which growth and reduction
        keep in step. Trainable slots that ``learn`` does not train (a recognition model, a
        scoring row, a policy module) are not carried; a gradient slot's optimizer state is the
        ``LearnState`` the agent already carries.

        The carry is exact when what was learned means the same thing in the rebuilt params, as
        counts over the same levels do. A posterior over a parameter the rebuild re-expresses
        (a head in a frame that moves with the agent) is carried in its old expression, and is
        then wrong.

        Design: docs/design/parameter-learning.md D9
        """
        return _learn_face.carry(self, fresh, learned, slots)

    def reduce(
        self,
        params: SystemParams,
        prior: SystemParams,
        factor: str | None = None,
        n_pair: int | None = None,
    ) -> SystemParams:
        """Bayesian model reduction on a growable factor, or on every one when ``factor`` is
        ``None``: two live levels are merged when the evidence of their pooled statistics, in
        every slot the factor is an axis of, is no lower than that of the separate levels. The
        merged level keeps its own prior once and both levels' data; the other returns to dead,
        its weight zero and its emissions at their prior. Candidate pairs are the ``n_pair``
        (default ``reduction_pairs``) best by mutual expected log-likelihood, tested greedily.
        ``prior`` is the params the posterior was learned from, which the evidence is relative
        to. ``learn`` runs this every ``reduction_period``-th call.

        Design: docs/design/structure-learning.md S4
        """
        return _learn_face.reduce(self, params, prior, factor, n_pair)

    def grow(
        self,
        params: SystemParams,
        observations: Mapping[str, Array],
        factor: str,
        rounds: int = 1,
        lr: float | Array = 1.0,
    ) -> SystemParams:
        """The batch form of growth on a growable factor: ``observations`` holds a window of
        ``N`` data, ``(N, …)`` per modality over the factor alone. Each of ``rounds`` rounds
        scores every datum under every level, takes the worst-explained one and wakes the first
        dead level for it while it falls below the growth threshold, seeding the level from that
        datum; the window is then absorbed at rate ``lr`` under the responsibilities the woken
        levels leave, into the factor's weights and its modalities. ``learn`` is this with one
        datum and one round.

        Design: docs/design/structure-learning.md S3
        """
        return _learn_face.grow(self, params, observations, factor, rounds, lr)

    def _gradient_slots(
        self, params: SystemParams
    ) -> tuple[tuple[str, str, int, MLPConditional], ...]:
        """The engine: :func:`hybrid_aif.system.faces.learn._gradient_slots`."""
        return _learn_face._gradient_slots(self, params)

    def init_learn_state(self, params: SystemParams) -> LearnState:
        """A fresh :class:`LearnState` for every gradient-learnable part of this agent.

        One ``opt_states`` entry per gradient-learnable slot regardless of rate, plus
        ``scoring:<row>`` for each mounted row with a trainer and :data:`ACTOR_KEY` for a policy
        module in ``policy_prior``. A scoring entry's shape is the row's own (a row that owns a
        declared System carries a nested ``LearnState`` over its slots). ``moments`` is seeded with
        every running statistic a mounted row's trainer carries
        (:func:`~hybrid_aif.objectives.amortized.scoring_moments`), so the carry's structure is
        complete at init.
        """
        return _learn_face.init_learn_state(self, params)

    def _habit_module(self, params: SystemParams):
        """``policy_prior`` if it holds a policy module, else ``None``."""
        params = self._p(params)
        module = params.policy_prior
        if module is None or hasattr(module, "shape") or isinstance(module, Categorical):
            return None
        return module

    @staticmethod
    def _resolve_learn_state(
        learn_state: "LearnState | None", grad_slots: tuple[tuple[str, str, int, object], ...]
    ) -> dict[str, object]:
        """The engine: :func:`hybrid_aif.system.faces.learn._resolve_learn_state`."""
        return _learn_face._resolve_learn_state(learn_state, grad_slots)

    def log_evidence(
        self, prior: SystemParams, posterior: SystemParams
    ) -> dict[str, _evidence.Evidence]:
        """Every slot's parameter evidence: the closed-form log marginal likelihood of the
        statistics ``posterior`` absorbed from ``prior``, keyed as :meth:`learn` keys its rates
        (the per-tick data evidence of a filtering step is ``InferenceDetail.log_evidence``, a
        different quantity). A slot with no closed form is unscored (``nats`` ``None``,
        ``exact`` ``False``); a conjugate slot that absorbed no mass is unscored with ``exact``
        ``True`` and its ``count``. Eager, not for a jitted body. See
        :func:`hybrid_aif.learning.evidence.log_evidence`."""
        return _evidence.evidence_by_slot(self, prior, posterior)

    def policies(self, params: SystemParams) -> Array:
        """The enumerated policy set, int32 ``(P, policy_len, K)``: one column per control node,
        in node order, over each node's levels (read off the tables at trace time), ordered as
        pymdp orders its policies (time-major, the last column at the last step fastest).

        Discrete control nodes only: raises under continuous ones. Under the sophisticated
        planner it is the first step's control assignments.
        """
        params = self._params_in(params, "policies")
        if FLOAT in self._space(params).kinds:
            raise ValueError(
                "System.policies(params) enumerates a discrete policy grid, and this System has "
                "continuous control nodes: there is no finite policy set to enumerate. Plan with "
                'the "cem" or "gradient" planner and read its plan off the planner\'s return '
                "(design: amortization.md A9)."
            )
        levels = self._control_levels(params)
        every = (True,) * len(levels)
        if self.planner == "sophisticated":
            # Its plan is a first step and what to do after each outcome: the
            # policies it returns are the first step's actions.
            return enumerate_policies(levels, 1, every)
        return enumerate_policies(levels, self.policy_len, every)

    def _slot_declarations(self, params: SystemParams) -> dict[str, tuple[str, int | str, object]]:
        params = self._p(params)
        return selectors._slot_declarations(self, params)

    def _write_slots(self, params: SystemParams, updated: Mapping[str, object]) -> SystemParams:
        return selectors._write_slots(self, params, updated)

    def _is_amortized(self, params: SystemParams) -> bool:
        """Whether this System is declared on the amortized (neural state-space) path.

        Any amortized declaration family (a ``Deterministic`` carry, a categorical code, a predictor
        head, a learned or derived initial value) or a recognition entry routes every face there,
        and that path's validator refuses a declaration it cannot express in full.
        """
        params = self._p(params)
        return (
            bool(self.recognition)
            or bool(params.recognition)
            or any(isinstance(b, Deterministic | MLPCategorical) for b in params.transition)
            or any(isinstance(a, PredictorHead) for a in params.likelihood)
            or any(
                isinstance(d, LearnedVector | DerivedInitial) for d in params.initial_state_prior
            )
        )

    def _amortized_structure(
        self, params: SystemParams, face: str = "this face"
    ) -> "_amortized.AmortizedStructure":
        """Validate and resolve the amortized declaration. Static, per call."""
        params, _ = self._control_targets(self._p(params))  # scored around the step terms
        mods, facs = (self.modality_names, self._facs)
        validation._refuse_shared_controls(self, facs, "on the amortized path", face)
        return _amortized.validate_structure(
            self.recognition,
            params.recognition,
            params.likelihood,
            params.transition,
            params.initial_state_prior,
            self.A_deps,
            self._B,
            self._ctl(face),
            self._num_controls(params, face),
            params.obs_targets,
            params.state_targets,
            params.terminal_targets,
            mods,
            facs,
            self._control_layout(params),
            self.external_inputs,
        )

    @staticmethod
    def _check_slot_key(
        key: str,
        what: str,
        table: Mapping[str, tuple[str, int, object]],
        learnable: bool = True,
    ) -> None:
        validation._check_slot_key(key, what, table, learnable)

    def _resolve_rates(
        self, mapping: Mapping[str, float | Array] | None, what: str, params: SystemParams
    ) -> dict[str, float | Array] | None:
        params = self._p(params)
        return selectors._resolve_rates(self, mapping, what, params)

    def _epistemic_variables(self, params: SystemParams) -> dict[str, _EpistemicVar]:
        params = self._p(params)
        return selectors._epistemic_variables(self, params)

    def _check_epistemic_key(
        self, key: str, params: SystemParams, variables: Mapping[str, _EpistemicVar]
    ) -> None:
        params = self._p(params)
        validation._check_epistemic_key(self, key, params, variables)

    def _check_pattern(
        self, pattern: str, what: str, params: SystemParams, *, epistemic: bool
    ) -> None:
        params = self._p(params)
        validation._check_pattern(self, pattern, what, params, epistemic=epistemic)

    def _expand_epistemic(
        self, pattern: str, params: SystemParams, variables: Mapping[str, _EpistemicVar]
    ) -> tuple[str, ...]:
        params = self._p(params)
        return selectors._expand_epistemic(self, pattern, params, variables)

    def _row_mounts(self, params: SystemParams) -> tuple["_amortized_terms.Mount", ...]:
        params = self._p(params)
        return selectors._row_mounts(self, params)

    def _declared_structures(self) -> dict[str, object]:
        return selectors._declared_structures(self)

    def _amortized_epistemic(
        self, params: SystemParams, structure: "_amortized.AmortizedStructure"
    ) -> tuple[int, ...]:
        params = self._p(params)
        return selectors._amortized_epistemic(self, params, structure)

    def _term_targets(
        self, params: SystemParams, planning: bool = True
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        params = self._p(params)
        return selectors._term_targets(self, params, planning)

    def _channel_selections(self, params: SystemParams) -> tuple[tuple[bool, ...] | None, ...]:
        params = self._p(params)
        return selectors._channel_selections(self, params)

    def _dim_selections(self, params: SystemParams) -> tuple[tuple[int, ...] | None, ...]:
        params = self._p(params)
        return selectors._dim_selections(self, params)

    def _resolve_param_novelty(
        self,
        sel: bool | tuple[str, ...],
        params: SystemParams,
        planning: bool,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        params = self._p(params)
        return selectors._resolve_param_novelty(self, sel, params, planning)

    def _eligible_state_blocks(
        self, params: SystemParams
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        params = self._p(params)
        return selectors._eligible_state_blocks(self, params)

    def _validate(self, params: SystemParams, face: str = "this face") -> None:
        params = self._p(params)
        validation._validate(self, params, face)

    def _validate_H_deps(self, params: SystemParams, facs: tuple[str, ...]) -> None:
        params = self._p(params)
        validation._validate_H_deps(self, params, facs)

    @staticmethod
    def _factor_levels(params: SystemParams, f: int) -> int | None:
        return validation._factor_levels(params, f)

    def planner_context(
        self,
        params: SystemParams,
        beliefs: Mapping[str, Belief],
        key: Array | None = None,
        aux_beliefs: Mapping[str, object] | None = None,
        scoring_key: Array | None = None,
        horizon: int | None = None,
        present: Mapping[str, bool | Array] | None = None,
    ) -> PlannerContext:
        """Build the per-call context a planner runs against (never stored).

        ``beliefs`` is keyed by factor name and must be marginals: resolve a conditional or
        learnable ``D`` with :meth:`initial_priors` first (a control node's entry is ignored).
        The context itself is positional (``init_beliefs`` in factor order, actions and policies
        with one column per control node), as planners index it. ``horizon`` is the plan length
        the planner plans to and the closures are built for (``None``: ``policy_len``). ``key``
        is the plan key a stochastic planner (CEM) reads; ``scoring_key`` is the scoring rows'
        own key, under which a row that rolls a model of its own takes the sampled rollout (only
        :meth:`score_policies` passes one). ``aux_beliefs`` is keyed by scoring-row name; each
        value is the belief carry of that row's own model, keyed by the nested System's factor
        names (amortized path only).

        All-tabular Systems get the compiled prob-space closures
        (:mod:`hybrid_aif.objectives.compiled`, beliefs as probability vectors); any other
        declaration gets the family-dispatched closures of :mod:`hybrid_aif.objectives.hybrid`
        (beliefs as distributions, mixtures padded to ``mixture_k``); the amortized path is
        :meth:`_amortized_context`. :meth:`_term_targets` decides which variables carry which
        epistemic leaf. When a step term reads the previous step's beliefs, the closures carry
        ``(qs, qs_prev)`` pairs (:func:`_carry_prev`). Tables are rebuilt per call, so a context
        cached across learning ticks goes stale.

        ``present`` says which modalities the plan observes, keyed by name: a bool for the whole
        plan, a ``(horizon,)`` bool array, or a one-key mapping from a discrete control node's
        name to a bool per level, read at each step's level of the node; a modality left out is
        present. An absent modality's leaves are zero at that step on every path.
        """
        return _plan_face.planner_context(
            self,
            params,
            beliefs,
            key,
            aux_beliefs,
            scoring_key,
            horizon,
            present,
        )

    def _aux_in(
        self, aux_beliefs: Mapping[str, object] | None, face: str
    ) -> dict[str, object] | None:
        """The engine: :func:`hybrid_aif.system.faces.plan._aux_in`."""
        return _plan_face._aux_in(self, aux_beliefs, face)

    def _planner_context(
        self,
        params: SystemParams,
        beliefs: tuple[Belief, ...],
        key: Array | None = None,
        aux_beliefs: Mapping[str, object] | None = None,
        scoring_key: Array | None = None,
        horizon: int | None = None,
        face: str = "this face",
        present: Mapping[str, bool | Array] | None = None,
        branching: bool = False,
    ) -> PlannerContext:
        """The engine: :func:`hybrid_aif.system.faces.plan._planner_context`."""
        return _plan_face._planner_context(
            self,
            params,
            beliefs,
            key,
            aux_beliefs,
            scoring_key,
            horizon,
            face,
            present,
            branching,
        )

    def _control_targets(
        self, params: SystemParams
    ) -> tuple[SystemParams, tuple[ControlTarget, ...]]:
        """``params`` without its control-node targets, and those targets as
        :class:`~hybrid_aif.controls.ControlTarget` entries (``params`` itself when there are
        none)."""
        controls = self._table.control_names
        state = [k for k in params.state_targets if k in controls]
        terminal = [k for k in params.terminal_targets if k in controls]
        if not state and not terminal:
            return params, ()
        space = self._space(params)

        def control_target(name: str, target, is_terminal: bool) -> ControlTarget:
            k = controls.index(name)
            if space.kinds[k] == LEVEL:
                return ControlTarget(
                    column=k,
                    name=name,
                    log_target=jax.nn.log_softmax(target.logits, axis=-1),
                    terminal=is_terminal,
                )
            span = None if space.layout is None else space.layout.node_spans[k]
            target = target_on_box(target, space.boxes[k], f"the target on {name!r}")
            return ControlTarget(
                column=k, name=name, log_target=None, terminal=is_terminal, target=target, span=span
            )

        control_targets = tuple(
            control_target(name, targets[name], is_terminal)
            for targets, names, is_terminal in (
                (params.state_targets, state, False),
                (params.terminal_targets, terminal, True),
            )
            for name in names
        )
        stripped = dataclasses.replace(
            params,
            state_targets={k: v for k, v in params.state_targets.items() if k not in controls},
            terminal_targets={
                k: v for k, v in params.terminal_targets.items() if k not in controls
            },
        )
        return stripped, control_targets

    def _validate_node_targets(self, params: SystemParams, face: str = "this face") -> None:
        """The engine: :func:`hybrid_aif.system.faces.plan._validate_node_targets`."""
        return _plan_face._validate_node_targets(self, params, face)

    def _present_in(
        self,
        present: Mapping[str, bool | Array | Mapping[str, Array]] | None,
        horizon: int,
        face: str,
        params: SystemParams,
    ) -> Presence | None:
        """The engine: :func:`hybrid_aif.system.faces.plan._present_in`."""
        return _plan_face._present_in(self, present, horizon, face, params)

    def _amortized_scoring(
        self,
        params: SystemParams,
        aux_beliefs: Mapping[str, object] | None = None,
        key: Array | None = None,
        horizon: int | None = None,
        face: str = "this face",
    ) -> "AmortizedScoring":
        """The engine: :func:`hybrid_aif.system.faces.plan._amortized_scoring`."""
        return _plan_face._amortized_scoring(self, params, aux_beliefs, key, horizon, face)

    def _legal_logits(
        self, params: SystemParams, legal, policies: Array, face: str = "this face"
    ) -> Array | None:
        """The engine: :func:`hybrid_aif.system.faces.plan._legal_logits`."""
        return _plan_face._legal_logits(self, params, legal, policies, face)

    def _plan(
        self,
        params: SystemParams,
        beliefs: tuple[Belief, ...],
        planner: Planner | None,
        use_bound: UseBound,
        metrics: bool,
        plan_key: Array | None = None,
        legal: "Mapping[str, Array] | Array | None" = None,
        face: str = "this face",
        present: Mapping[str, bool | Array] | None = None,
    ) -> tuple[Categorical, TermTree | None, Array]:
        """The engine: :func:`hybrid_aif.system.faces.plan._plan`."""
        return _plan_face._plan(
            self,
            params,
            beliefs,
            planner,
            use_bound,
            metrics,
            plan_key,
            legal,
            face,
            present,
        )

    def infer_policies(
        self,
        params: SystemParams,
        beliefs: Mapping[str, Belief],
        planner: Planner | None = None,
        use_bound: UseBound = "midpoint",
        metrics: bool = True,
        plan_key: Array | None = None,
        legal: "Mapping[str, Array] | Array | None" = None,
        present: Mapping[str, bool | Array] | None = None,
    ) -> tuple[Categorical, TermTree | None]:
        """Policy posterior and the EFE term tree: ``(q_pi, tree)``.

        ``planner`` is a callable, or ``None`` for the System's registry planner. ``metrics=False``
        skips the tree (``None``); the posterior is the same up to float reassociation. A planner
        that selects without building a tree (``"habit"``) returns ``{}``; :meth:`score_policies`
        gives the tree G would assign its plan. ``plan_key`` is required by, and only by, a
        stochastic planner (``"cem"``). ``legal`` is the hard constraint of :meth:`act`, keyed by
        control node or one bool per policy. ``beliefs`` is keyed by factor name. The planner
        must return clamped (integer) policies: a belief-valued policy set raises. ``present``
        says which modalities the plan expects to observe at each step, keyed by name: a bool
        for the whole plan, a per-step bool array (``(policy_len,)``, or ``(horizon,)`` under
        ``horizon=``), or a one-key mapping from a discrete
        control node's name to a bool per level (``{"answer": {"u_option": (False, True)}}``),
        read at each step's level of the node; a modality left out is present throughout. An
        absent modality's leaves are zero at that step, and under
        ``planning_belief="posterior"`` the epistemic chain is not conditioned on it there.
        """
        return _plan_face.infer_policies(
            self,
            params,
            beliefs,
            planner,
            use_bound,
            metrics,
            plan_key,
            legal,
            present,
        )

    def score_policies(
        self,
        params: SystemParams,
        beliefs: Mapping[str, Belief],
        policies: "Array | Mapping[str, Array | Categorical]",
        aux_beliefs: Mapping[str, object] | None = None,
        key: Array | None = None,
    ) -> TermTree:
        """The EFE term tree for a given policy set, without a planner.

        ``policies`` is int32 ``(P, H, K)``, one column per control node in node order (as
        :meth:`policies` gives them), or one ``(H, K)`` policy whose per-policy axis is then
        squeezed from every leaf. Over an all-continuous action space it takes float
        ``(P, H, action_dim)`` or ``(H, action_dim)``. Fold the tree with
        :func:`~hybrid_aif.objectives.base.total` for per-policy G. Scoring a habit planner's plan
        here reads the G it would get (:mod:`hybrid_aif.planning.habit`).

        A policy may instead be a mapping keyed by control-node name, one entry per node. A
        discrete node's entry is its levels as an int ``(P, H)`` (or ``(H,)``) array, a clamp, or
        a :class:`~hybrid_aif.distributions.Categorical` over its levels with batch shape
        ``(P, H)`` (or ``(H,)``), a belief. A continuous node's entry is its values as a float
        ``(P, H, d)`` (or ``(H, d)``) array, a clamp, or a
        :class:`~hybrid_aif.distributions.Gaussian` /
        :class:`~hybrid_aif.distributions.TruncatedGaussian` over them with batch shape
        ``(P, H)`` (or ``(H,)``), a belief. A mapping is how a space mixing discrete and
        continuous nodes is scored. The type of each entry selects, statically, how its node is
        read: a level indexes the tables, a belief mixes them, a continuous value enters a
        linear-Gaussian factor's ``G u`` as an offset and a Gaussian belief over it pushes through
        exactly (a truncated one by its closed moments), a nonlinear factor's map reads a value as
        an input and a belief through its moments, and every leaf is computed under the result.
        The tree is then the objective of committing to the *randomised* policy that draws each
        node's level from its belief, step by step: a term linear in the control
        (utility, ambiguity, the predicted state) is the mixture of its clamped values, a
        nonlinear one (risk, info gain) is not. Under a belief, transition novelty is the
        mean-field mixture (the novelty weights averaged over ``q(u)``, contracted with the mixed
        prediction) and occupancy is the entropy of the mixed transition, ``H[Σ_u q(u) B_u]``.
        That is a different quantity from the expected
        objective under a policy posterior ``q_pi`` over clamped policies, which is the
        ``q_pi``-weighted ``total`` of the clamped tree. A one-hot belief scores as its clamp;
        an all-clamp mapping is the positional policy. A belief on a discrete node is read by
        categorical (tabular) transitions; a continuous factor reading a belief-valued discrete
        node raises, as does a neural transition reading a belief-valued continuous one.

        A ``state_targets`` entry on a control node scores it: a ``risk`` leaf keyed by the
        node's name, ``−log C(u)`` under a clamp and ``KL(q(u) ‖ C)`` under a belief, at every
        step; a ``terminal_targets`` entry scores the last step as ``terminal_risk``.

        ``beliefs`` is keyed by factor name. In either form the plan's length ``H`` need not be
        ``policy_len``: the scoring rows are built for the plan's own horizon.
        ``aux_beliefs`` passes to :meth:`planner_context`. ``key`` is the scoring rows' key: rows
        that roll a model of their own take the sampled rollout under it, the expected one without.
        """
        return _plan_face.score_policies(self, params, beliefs, policies, aux_beliefs, key)

    def _check_plan(self, plan: Array, what: str, params: SystemParams) -> Array:
        return validation._check_plan(self, plan, what, params)

    def act(
        self,
        key: Array,
        params: SystemParams,
        beliefs: Mapping[str, Belief],
        selection: Literal["stochastic", "deterministic"] = "stochastic",
        alpha: float | None = None,
        mode: Literal["marginal", "policy"] | None = None,
        planner: Planner | None = None,
        use_bound: UseBound = "midpoint",
        metrics: bool = True,
        plan_key: Array | None = None,
        legal: "Mapping[str, Array] | Array | None" = None,
    ) -> tuple[dict[str, Array], Categorical, TermTree | None]:
        """Infer policies and emit an action: ``(action, q_pi, metrics)``.

        ``beliefs`` is keyed by factor name; the action by action-node name, one entry per
        action node, the intended value of its control node (a level, or a continuous node's
        float value): a draw (or the mode) from each control node's posterior, sampled over the
        policy set the planner returned. ``key`` selects the action,
        ``plan_key`` drives a stochastic planner; ``metrics`` as in :meth:`infer_policies`.
        ``mode=None`` takes the first-step action marginals for discrete controls and the drawn
        policy's first step for continuous ones, where ``mode="marginal"`` raises. A planner that
        emits a posterior over a continuous plan (the gradient planner under
        ``posterior="laplace"``, a truncated Gaussian on the box) has the action drawn from that
        posterior's first step under ``selection="stochastic"``, and its mode under
        ``"deterministic"``. ``legal`` is this
        tick's hard constraint: a mapping keyed by control-node name, each entry a boolean mask
        over that node's levels (``(levels,)`` for the first step, ``(policy_len, levels)`` per
        step; a node left out is unrestricted, an unknown key raises), or one bool per policy of
        the planner's set. An illegal level gets exactly zero mass in ``q_pi`` and the marginals.
        It is separate from ``policy_prior``, the soft prior. The planner must return clamped
        policies (integer levels, or a mixed space's :class:`~hybrid_aif.controls.ControlStep`
        of clamps, from which the action is drawn as one policy's first step): a belief-valued
        policy set raises.

        Design: docs/design/action-legality.md.
        """
        return _plan_face.act(
            self,
            key,
            params,
            beliefs,
            selection,
            alpha,
            mode,
            planner,
            use_bound,
            metrics,
            plan_key,
            legal,
        )

    def action_marginals(
        self, params: SystemParams, q_pi: Categorical, policies: Array | None = None
    ) -> dict[str, Categorical]:
        """First-step marginal posterior over each control node's levels, by control-node name:
        the posterior an action is drawn from.

        ``policies`` is the positional policy set ``q_pi`` is over (``None``: :meth:`policies`);
        the level counts come from ``params``' tables. Discrete controls only.
        """
        params = self._params_in(params, "action_marginals")
        names = self._names(params)
        if FLOAT in names.kinds:
            raise ValueError(
                "action_marginals is a distribution over control levels, and this System has "
                "continuous control nodes, which have none"
            )
        marginals = action_marginals(
            q_pi,
            self.policies(self._params_out(params)) if policies is None else policies,
            self._control_levels(params),
        )
        return dict(zip(names.controls, marginals, strict=True))

    def predict_beliefs(
        self,
        params: SystemParams,
        beliefs: Mapping[str, Belief],
        policy: Array,
        key: Array | None = None,
    ) -> list[dict[str, Belief]]:
        """Open-loop prediction: the beliefs, keyed by factor name, after each step of one policy.

        ``policy`` stays positional: int32 ``(horizon, K)``, one column per control node, or float
        ``(horizon, action_dim)`` over an all-continuous action space; step ``t`` of the result
        follows ``policy[t]``, through the transition :meth:`update_empirical_prior` uses (no
        observations enter); an action node a modality observes carries its prediction at each
        step, as :meth:`update_empirical_prior` gives it. ``key`` selects the sampled rollout,
        split once per step; ``None`` is the expected one.
        """
        return _infer_face.predict_beliefs(self, params, beliefs, policy, key)

    def update_empirical_prior(
        self,
        params: SystemParams,
        beliefs: Mapping[str, Belief],
        action: Mapping[str, Array],
        key: Array | None = None,
    ) -> dict[str, Belief]:
        """Next tick's state priors: ``beliefs`` pushed through ``action``, keyed by factor name.

        ``action`` is keyed by action-node name as :meth:`act` returns it (``{}`` when nothing is
        controlled): a level per discrete node, a float vector per continuous one, the two kinds
        side by side over a mixed space. Each control node is clamped to its action's value at
        this step (an efference copy): each discrete node's returned entry is its prior for the
        next step, uniform over its levels. An action node a modality observes gets an entry
        too, the agent's prediction of its action: the column of ``params.actuator`` at the
        clamped level, the one-hot of that level under the delta kernel. A continuous node enters
        a linear-Gaussian factor as the ``G u`` offset, a neural one as part of its input, a
        nonlinear one as an entry of its map's controls. Hybrid
        factors go through their own transitions (exact Gaussian ``predict``; grow then
        ``reduce(mixture_k)`` for mode-coupled factors), the closure the planner uses for one step.
        On the amortized path each carry advances, then each code is read off its transition at this
        step's parents.

        Without ``key`` the result is the expected belief. With one it is the sampled belief: one
        draw per factor as a belief of its family (:mod:`hybrid_aif.inference.sample`). That is a
        legal carry but a point mass: a collapsed Gaussian or mixture cannot be conditioned and has
        no finite entropy. See docs/composing-systems.md.
        """
        return _infer_face.update_empirical_prior(self, params, beliefs, action, key)

    def _update_empirical_prior(
        self,
        params: SystemParams,
        beliefs: tuple[Belief, ...],
        action: Array | None,
        key: Array | None = None,
        face: str = "this face",
    ) -> tuple[Belief, ...]:
        """The engine: :func:`hybrid_aif.system.faces.infer._update_empirical_prior`."""
        return _infer_face._update_empirical_prior(self, params, beliefs, action, key, face)


def _derived(name: str, read: Callable[[SugarView], object]) -> property:
    def get(self: System):
        sugar = self._table.sugar
        if sugar is None:
            return _Unreadable(
                f"System.{name} reads back the pymdp-style control declaration, and this "
                f"System's control nodes {list(self._table.control_names)} are not in that shape "
                "(each named u_<factor> after its own factor, after the state parents of every "
                "transition that reads it); read System.names.controls and B_deps instead"
            )
        value = read(sugar)
        return None if value is None else _Derived(value)

    return property(get)


#  The control sugar read back off the node form (defined after the class: the names are also
#  the constructor's InitVar keywords, whose dataclass default must stay ``None``). Where there
#  is nothing to read back, the value raises on use with a sentence (and still passes through
#  ``dataclasses.replace``).
System.controlled = _derived("controlled", lambda sugar: sugar.controlled)  # type: ignore[assignment]
System.B_control_deps = _derived(  # type: ignore[assignment]
    "B_control_deps", lambda sugar: sugar.B_control_deps
)


@functools.cache
def _bound_away(
    subsystems: tuple, control_names: tuple[str, ...], action_names: tuple[str, ...], prefix: str
) -> tuple[tuple[str, str], ...]:
    """Each control node, and its action node, a ``merge()`` binding removed, by name, with
    the rest of the sentence that refuses a mapping keyed by it (what the node became), read
    off ``subsystems``. ``prefix`` qualifies every name, for a merge that nests this one."""
    live = set(control_names)
    live_action = dict(zip(control_names, action_names, strict=True))

    def q(name: str) -> str:
        return f"{prefix}{name}"

    action_of: dict[str, str] = {}
    relayed: dict[str, str] = {}
    ended: dict[str, None] = {}
    same: dict[str, str] = {}
    for _, sub in subsystems:
        action_of.update(dict(sub.actions))
        relayed.update(dict(sub.relayed))
        ended.update(dict.fromkeys(sub.terminated))
        same.update(dict(sub.identified))
    action_of.update(live_action)
    out: dict[str, str] = {}

    def gone(node: str, control: str, action: str | None = None) -> None:
        out.setdefault(q(node), control)
        if node in action_of and action is not None:
            out.setdefault(q(action_of[node]), action)

    for node, driver in same.items():
        seen = {node}
        while driver in same and driver not in seen:
            seen.add(driver)
            driver = same[driver]
        if node in live:
            continue
        if driver in live:
            gone(
                node,
                f"a control node merge() identified with {q(driver)!r} (a same-step "
                f"ActionControl): key it by {q(driver)!r}, the node it became, whose choice "
                "it constrains",
                f"the action node of a control node merge() identified with {q(driver)!r}: "
                f"key it by {q(live_action[driver])!r}",
            )
        elif driver in ended:
            gone(
                node,
                f"a control node merge() made one with {q(driver)!r}, which a BoundaryPrior "
                f"terminated, so no node {q(node)!r} exists here: drop the key",
                f"the action node of a control node merge() made one with {q(driver)!r}, "
                "which a BoundaryPrior terminated: drop the key",
            )
        else:
            gone(
                node,
                f"a control node merge() identified with {q(driver)!r} (a same-step "
                f"ActionControl), which another subsystem chooses: key it by {q(driver)!r} "
                "in that subsystem's view",
                f"the action node of a control node merge() identified with {q(driver)!r}, "
                "which another subsystem chooses: key the action in that subsystem's view",
            )
    for node, state in relayed.items():
        if node in live:
            continue
        gone(
            node,
            f"a control node merge() identified with the state {q(state)!r}, which its "
            f"reader reads a step later, so no node {q(node)!r} exists here: drop the key, "
            f"or constrain the choice that moves {q(state)!r}",
            f"the action node of a control node merge() identified with the state "
            f"{q(state)!r}: drop the key",
        )
    for node in ended:
        if node in live:
            continue
        gone(
            node,
            "a control node merge() terminated with a BoundaryPrior, so no node "
            f"{q(node)!r} exists here: drop the key",
            "the action node of a control node merge() terminated with a BoundaryPrior: "
            "drop the key",
        )
    return tuple(out.items())


@functools.cache
def _sensed_columns(A_deps: tuple[tuple[int, ...], ...], n: int) -> tuple[int, ...]:
    """The action columns ``A_deps`` names, node ``n + k`` being action node ``k``."""
    return tuple(sorted({g - n for deps in A_deps for g in deps if g >= n}))


def _observation_type(decl: object, what: str) -> tuple[str, int]:
    """An observation's port type from its likelihood: ``(LEVEL, levels)`` for a table (a
    tabular declaration, or a probability table with the child axis first), ``(FLOAT,
    dimension)`` for a linear-Gaussian or neural conditional."""
    if isinstance(decl, ConditionalCategorical | ConditionalDirichlet):
        return LEVEL, int(decl.num_levels)
    if isinstance(decl, MLPConditional):
        return FLOAT, int(decl.child_dim)
    if hasattr(decl, "obs_dim"):
        return FLOAT, int(decl.obs_dim)
    if hasattr(decl, "shape") and getattr(decl, "ndim", 0) >= 1:
        return LEVEL, int(decl.shape[0])
    raise NotImplementedError(
        f"{what} is a {type(decl).__name__}, which gives its observation no level count or "
        "dimension; declare a tabular, MLP or linear-Gaussian likelihood to type the port (an "
        "amortized mind cannot be coupled)"
    )


# The face modules import this one, so they load after every name above exists.
from hybrid_aif.system.faces import infer as _infer_face  # noqa: E402
from hybrid_aif.system.faces import learn as _learn_face  # noqa: E402
from hybrid_aif.system.faces import plan as _plan_face  # noqa: E402
