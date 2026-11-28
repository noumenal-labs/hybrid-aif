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
from typing import TYPE_CHECKING, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hybrid_aif.cells import Grid
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
from hybrid_aif.debug import eager_face as _eager_face
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
from hybrid_aif.dtypes import as_float, table_dtype, tree_float_dtype, widened
from hybrid_aif.dtypes import cast as _cast
from hybrid_aif.envs._process import DiscreteProcess, HybridProcess
from hybrid_aif.envs.generative import CONTROL_TAG, GenerativeEnv, GenerativeEnvParams
from hybrid_aif.factors import (
    DerivedInitial,
    Deterministic,
    LinearGaussianObservation,
    LinearGaussianTransition,
    NoiseHome,
    NonlinearGaussianTransition,
    Parameter,
    ParameterHome,
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
from hybrid_aif.learning.slots import slot_learnable as _slot_learnable
from hybrid_aif.objectives import TermTree
from hybrid_aif.objectives import amortized as _amortized_terms
from hybrid_aif.objectives import closed_loop as _closed_loop
from hybrid_aif.objectives import hybrid as _hybrid
from hybrid_aif.objectives import pooling as _pooling
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
from hybrid_aif.reads import BLACK_BOX, REALISATIONS, Emission
from hybrid_aif.system import emissions as _emissions
from hybrid_aif.system import keyed, selectors, validation
from hybrid_aif.system.selectors import _KEY_KINDS, _EpistemicVar

if TYPE_CHECKING:
    from hybrid_aif.system.compiled import CompiledSystem

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


class _Lowered(tuple):
    """A canonical ``branch_on`` or ``plan_learning`` value lowered from ``sophisticated=``: a
    ``dataclasses.replace`` passes it back to the constructor, which recognises it as the lowering
    (not a second declaration) and lowers ``sophisticated`` afresh."""

    __slots__ = ()


#: The first segments of a ``sophisticated`` key that carry a table's counts within the plan
#: (``plan_learning``'s grammar); ``obs`` branches on a modality (``branch_on``).
_SOPHISTICATED_TABLES = ("transition", "likelihood", "parameter")


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


def _refuse_aliases(values: Mapping[str, object]) -> None:
    """Refuse one learnable declaration object held under two names of ``transition``,
    ``likelihood`` and ``parameters`` taken together.

    A pytree map rebuilds each occurrence, so two names holding one object would fork at the first
    update, each learning from its own evidence only. Two readers of one table name it, with
    ``Parameter(name)`` and an entry in ``SystemParams.parameters``."""
    seen: dict[int, str] = {}
    for container in ("transition", "likelihood", "emission", "parameters"):
        tree = values.get(container)
        if not isinstance(tree, Mapping):
            continue  # a positional view holds a variable's table at every reader, by design
        for key, decl in tree.items():
            if isinstance(decl, Parameter) or not (
                isinstance(decl, Learnable) or _slot_learnable(decl)
            ):
                continue
            label = f"{container}[{key!r}]"
            if id(decl) in seen:
                raise ValueError(
                    f"{seen[id(decl)]} and {label} hold one declaration object; sharing by "
                    "object aliasing would fork the parameter at the first update. Declare a "
                    "named parameter variable (SystemParams.parameters) and give each reader "
                    "Parameter(name)"
                )
            seen[id(decl)] = label
    for container, field in (("transition", "noise_prior"), ("likelihood", "noise")):
        tree = values.get(container)
        if not isinstance(tree, Mapping):
            continue
        for key, decl in tree.items():
            prior = getattr(decl, field, None)
            if not isinstance(prior, NoiseHome):
                continue
            label = f"{container}[{key!r}].{field}"
            if id(prior) in seen:
                raise ValueError(
                    f"{seen[id(prior)]} and {label} hold one noise prior object; sharing by "
                    "object aliasing would fork the noise at the first update. Declare a named "
                    "parameter variable (SystemParams.parameters) and give each reader "
                    "Parameter(name) in that field"
                )
            seen[id(prior)] = label


def _precision_in(value):
    """A policy precision as a strong-typed floating array, or ``None``."""
    if value is None:
        return None
    arr = jnp.asarray(value)
    if not jnp.issubdtype(arr.dtype, jnp.floating):
        arr = arr.astype(jnp.result_type(float))
    return _strong(arr)


class _Unset:
    """The value of a planner field a composite cannot state: its leaves differ on it."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (_Unset, ())


UNSET = _Unset()
#: The sentinel a composite's planner field holds where its leaves differ (``policy_len``,
#: ``planner``, ``gamma``, ``num_iters``, ``decomposition``, ``planning_belief`` or ``feedback``):
#: a face that needs the field refuses, as every planning face refuses a coupling. Each leaf's own
#: value is in ``Subsystem.config``.


class PartParams(eqx.Module):
    """One leaf's own planner parameters inside a composite, ``SystemParams.parts[leaf]``.

    A leaf's view takes them from here, so two leaves' critics never answer to one name and each
    keeps its own clock.
    """

    #: The leaf's habit: the prior over its own policy set (as ``SystemParams.policy_prior``).
    policy_prior: Array | Categorical | object | None = eqx.field(default=None, converter=_strong)
    #: The leaf's policy precision (as ``SystemParams.precision``).
    precision: Array | None = eqx.field(default=None, converter=_precision_in)
    #: The leaf's durations, keyed by its control node's namespaced name (as
    #: ``SystemParams.durations``).
    durations: dict[str, Array] = eqx.field(default_factory=dict, converter=_strong)
    #: The parameters of the leaf's scoring estimators, keyed by term name (as
    #: ``SystemParams.scoring``).
    scoring: dict[str, object] = eqx.field(default_factory=dict, converter=_strong)


class SystemParams(eqx.Module):
    """All numeric state of a System: generative tensors, designations and the closed-loop factors.

    A :class:`System` holds the static structure; its ``SystemParams`` holds every array, as a
    pytree passed to each *face* (a method of :class:`System` that takes the params, such as
    ``infer_states``, ``act`` or ``learn``) and returned by those that update it.

    **Keying.** ``likelihood`` is keyed by modality name, ``transition`` and
    ``initial_state_prior`` by state factor name (``params.likelihood["position"]``); every face
    takes and returns them so, and a positional (tuple) container raises, naming
    ``System.from_positional("params", params)``. Inside, the System reads a positional view
    (tuples in node order), built by name lookup. ``recognition`` and ``scoring`` join the
    slot-qualified key space (``recognition:stoch``, ``scoring:g_tail``) and namespace under
    ``merge()``.

    **Families.** Declaration types carry the family: tabular declarations give a discrete System;
    linear-Gaussian declarations (:mod:`hybrid_aif.factors`) and Gaussian or mixture priors give a
    continuous or mixed one. A ``likelihood``, ``transition`` or ``initial_state_prior`` entry is a
    point declaration, a distribution over its parameters (``ConditionalDirichlet``,
    ``Dirichlet``), learnable by :meth:`System.learn` and samplable by ``as_env(param_key=...)``,
    or a parametrized function (:class:`~hybrid_aif.distributions.MLPConditional`), learnable by
    gradient.

    **Groups.** The fields fall in three groups, plus the per-leaf record of a composite:

    - the declared model: ``transition``, ``likelihood``, ``initial_state_prior``, ``control``,
      ``actuator``, ``durations``, ``parameters``, ``references``, ``readings``, ``emission``;
    - the closed-loop factors (the two factors of the loop that return the distribution the loop
      acts on): ``recognition`` and ``policy_prior``;
    - scoring material: ``obs_targets``, ``state_targets``, ``terminal_targets``, ``scoring``
      and ``precision``;
    - ``parts``, each leaf's own planner parameters when the System is a splice of leaves.

    ``tests/test_scoring_slot.py`` asserts the partition. The AIF letters ``A``, ``B``, ``D`` and
    ``E`` are read-only aliases of ``likelihood``, ``transition``, ``initial_state_prior`` and
    ``policy_prior``. A table of the fields is in ``docs/composing-systems.md``.

    Design: docs/design/system.md (SystemParams groups); docs/design/parameter-learning.md D3;
    docs/design/amortization.md A1 A2 A11 A12.
    """

    #: AIF's B: the transition of each state factor, keyed by state factor name. The trailing
    #: parent axis is the control (tables), the conditioned innovations (linear-Gaussian), or a
    #: one-hot or continuous action parent (neural, deterministic). A ``Deterministic`` entry is a
    #: *carry* node (its own previous value is the map's carry, not a parent); an
    #: ``MLPCategorical`` entry is a factored-categorical code whose parents are read within the
    #: step (the two make the neural state-space pair). A ``Parameter(name)`` entry reads the table
    #: held under ``name`` in ``parameters``.
    transition: Mapping[
        str,
        TabularTransition
        | LinearGaussianTransition
        | NonlinearGaussianTransition
        | MLPConditional
        | Deterministic
        | MLPCategorical
        | Parameter,
    ] = eqx.field(converter=_strong)
    #: AIF's D: the initial prior of each state factor, keyed by state factor name. A
    #: ``LearnedVector`` is a carry node's episode-start *value*; a ``DerivedInitial`` says "read
    #: it off my own transition at t=0".
    initial_state_prior: Mapping[
        str, TabularPrior | Gaussian | MixtureGaussian | LearnedVector | DerivedInitial
    ] = eqx.field(converter=_strong)
    #: AIF's A: the likelihood of each modality, keyed by modality name. An entry is a child-first
    #: ``(num_obs, *dep_sizes)`` table (a point table or a Dirichlet over one), a linear-Gaussian
    #: ``(H, c, R_chol)`` head, a neural conditional, a ``PredictorHead`` (a trunk and a named loss:
    #: ``symlog_mse``, ``symlog_disc`` or ``binary``), or a ``Parameter(name)`` reading the table
    #: held under ``name`` in ``parameters``. Default ``{}``: an open-loop model observes nothing.
    likelihood: Mapping[
        str,
        TabularObservation | LinearGaussianObservation | MLPConditional | PredictorHead | Parameter,
    ] = eqx.field(default_factory=dict, converter=_strong)
    #: The amortized inference modules, keyed by the variable each infers, for example
    #: ``{"stoch": posterior_net, "embed": encoder}``. Their structure (what each module infers and
    #: conditions on) is :attr:`System.recognition`.
    recognition: dict[str, object] = eqx.field(default_factory=dict, converter=_strong)
    #: Standing preferences over observations (AIF's C), keyed by modality name: a ``Categorical``
    #: over a tabular modality's levels, or a ``Gaussian`` or ``MixtureGaussian`` over a
    #: linear-Gaussian modality's observation space (the covariance is the tolerance). An entry for
    #: a neural modality raises. A ``ConditionalCategorical`` or ``ConditionalGaussian`` entry is a
    #: target conditional on the parents :attr:`System.C_deps` names under its key, one
    #: distribution per parent configuration.
    obs_targets: dict[str, Distribution] = eqx.field(default_factory=dict, converter=_strong)
    #: Standing preferences over states, scored as risk, keyed by state factor name. A conditional
    #: entry is as in ``obs_targets``; a ``Categorical`` under a discrete control node's name is
    #: that node's fixed cost per level.
    state_targets: dict[str, Distribution] = eqx.field(default_factory=dict, converter=_strong)
    #: Terminal preferences, scored at the end of the horizon (the H of Friston et al. 2023 and of
    #: SPM's ``spm_MDP_VB_XXX``), keyed by state factor name. A ``ConditionalCategorical`` entry is
    #: a *conditional* terminal target and needs a :attr:`System.H_deps` entry naming its parent
    #: state factors (see :meth:`System.resolve_terminal_targets`).
    terminal_targets: dict[str, Distribution | ConditionalCategorical] = eqx.field(
        default_factory=dict, converter=_strong
    )
    #: The parameters of the objective terms that carry their own (entries of the estimator
    #: registry), keyed by term name, for example ``{"g_tail": (critic, slow),
    #: "ensemble_disagreement": (m0, m1, m2, m3)}``; a term that owns a declared model holds that
    #: model's params. The structure is :attr:`System.terms` and :attr:`System.scoring`.
    scoring: dict[str, object] = eqx.field(default_factory=dict, converter=_strong)
    #: The prior over policies (AIF's E, read-only alias ``E``): a log-space array or a
    #: ``Categorical`` over an enumerated policy set, or a policy module (beliefs to an action
    #: distribution) that the ``"habit"`` planner acts from. Since q_pi ~ softmax(-gamma G + log E),
    #: E does not depend on gamma: habits belong here; costs that must trade against utilities
    #: belong in the targets (inside G).
    policy_prior: Array | Categorical | object | None = eqx.field(default=None, converter=_strong)
    #: The box of each continuous control node, keyed by the node's name
    #: (``params.control["u_x"]``): a :class:`~hybrid_aif.controls.Box` of per-dimension bounds,
    #: the node's support, which the CEM and gradient planners search inside. A node with an entry
    #: is continuous (its readers read a float vector of the box's width); a node without one is
    #: discrete. Design: docs/design/control-states.md C4.
    control: dict[str, Box] = eqx.field(default_factory=dict, converter=_strong)
    #: The kernel P(a | u) of each action node that is not a delta, keyed by the action node's name
    #: (``params.actuator["a_x"]``): a ``(levels, levels)`` probability table whose column ``u`` is
    #: the distribution of the realised action given the decision ``u`` (discrete control nodes
    #: only). An action node without an entry realises its control node's decision exactly. The
    #: action node is realised once per tick, by a draw from the kernel (``sample``;
    #: ``System.act``'s ``selection`` gives the other realisation, ``mode``, for the decision
    #: itself), by ``System.as_env`` when the environment is stepped on its own and by the driver
    #: under ``run`` and ``couple``; the control node stays clamped to the decision. Design:
    #: docs/design/control-states.md C12, C13; nesting-and-coupling.md N3, N5.
    actuator: dict[str, Array] = eqx.field(default_factory=dict, converter=_strong)
    #: A round's duration per level of one discrete control node, keyed by the node's name
    #: (``params.durations["u_option"]``): a positive float ``(levels,)`` array, or a scalar for
    #: every level, in the unit the model's preferences are per. A round steps every state factor
    #: once, so a round has one duration and at most one node carries it; the engine checks
    #: positivity only. The objective never reads it; :meth:`System.elapsed` returns the realised
    #: action's duration for an outer loop's clock (1 when none is declared). Design:
    #: docs/design/step-durations.md T1, T4, T8.
    durations: dict[str, Array] = eqx.field(default_factory=dict, converter=_strong)
    #: The policy precision gamma, an optional scalar array: q_pi ~ softmax(-precision G + log E).
    #: When present the planners read it in place of :attr:`System.gamma` (the static default,
    #: read when this is ``None``); being a parameter leaf it flows traced through every face, so
    #: changing its value recompiles nothing. Design: docs/design/recompilation.md R5.
    precision: Array | None = eqx.field(default=None, converter=_precision_in)
    #: The reading counts of :attr:`System.readings`, keyed by the named state factor or modality:
    #: one positive count per level of its control node, the number of fast readings one slow
    #: reading pools under that level. Read only by the novelty term of that table. Design:
    #: step-durations.md T10.
    readings: dict[str, Array] = eqx.field(default_factory=dict, converter=_strong)
    #: Named parameter variables, keyed by name: a tabular declaration (a table, or a Dirichlet over
    #: one) that several ``transition`` or ``likelihood`` entries read through ``Parameter(name)``;
    #: an entry that is itself a table is that state factor's or modality's own variable. Every
    #: reader of one name reads one table; :meth:`System.learn` sums the readers' count statistics
    #: and updates the named table once, under the rate key ``parameter:<name>``. The faces return
    #: the references, never copies of the table. A variable may instead hold a noise prior (an
    #: ``InverseGamma`` or ``InverseWishart``), read by one field of a declaration: a Gaussian
    #: transition's ``noise_prior`` or a ``LinearGaussianObservation``'s ``noise`` holding
    #: ``Parameter(name)``; its readers' residual statistics sum into one update the same way.
    #: Design: docs/design/shared-parameters.md; process-noise-learning.md D4.
    parameters: dict[str, ParameterHome] = eqx.field(default_factory=dict, converter=_strong)
    #: The positional view's record of which reader positions hold a resolved reference, as
    #: ``(slot, index, parameter name)`` triples; empty in the keyed form every face takes and
    #: returns, where the entries are the ``Parameter`` tokens themselves.
    references: tuple[tuple[str, int, str], ...] = eqx.field(default=(), static=True)
    #: The positional view's record of which readers hold a resolved noise variable in one field
    #: (a transition's ``noise_prior``, a likelihood's ``noise``), as ``(slot, index, parameter
    #: name)`` triples; empty in the keyed form, where the field holds the ``Parameter`` token.
    noise_references: tuple[tuple[str, int, str], ...] = eqx.field(default=(), static=True)
    #: The kernel of each emission (an output the System draws from its state factors,
    #: :attr:`System.emissions`), keyed by emission name: a likelihood of any family (a table, a
    #: linear-Gaussian or neural conditional, a learnable form, or a ``Parameter`` reference to a
    #: named parameter variable), over the emission's parents in the order
    #: :attr:`System.emissions` lists them. The emitted port's type is read off the kernel, as a
    #: modality's is. No inference, planning or learning face reads it. Design:
    #: nesting-and-coupling.md N5.
    emission: dict[str, object] = eqx.field(default_factory=dict, converter=_strong)
    #: The leaves' own planner parameters in a composite, keyed by leaf name
    #: (:class:`~hybrid_aif.system.PartParams`: habit, precision, durations and scoring
    #: parameters); empty for a System that is not a splice of leaves. A leaf's entry holds its
    #: own; a composite node holds none. Design: engine-consolidation.md.
    parts: dict[str, PartParams] = eqx.field(default_factory=dict)

    def __init__(self, *args, **kwargs):
        fields = [f for f in dataclasses.fields(self) if f.init]
        if len(args) > len(fields):
            raise TypeError(f"SystemParams takes at most {len(fields)} positional arguments")
        values = dict(zip((f.name for f in fields), args, strict=False))
        for key, value in kwargs.items():
            if key in values or key not in {f.name for f in fields}:
                raise TypeError(f"SystemParams got an unexpected or repeated argument {key!r}")
            values[key] = value
        _refuse_aliases(values)
        for f in fields:  # the field converters run after this, as they do for a generated init
            if f.name in values:
                value = values[f.name]
            elif f.default is not dataclasses.MISSING:
                value = f.default
            elif f.default_factory is not dataclasses.MISSING:
                value = f.default_factory()
            else:
                raise TypeError(f"SystemParams is missing the argument {f.name!r}")
            setattr(self, f.name, value)

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

    def cast(self, dtype: object, where: Callable | None = None) -> "SystemParams":
        """These params with every float array in ``dtype`` (float32 or float64), or with only
        the part ``where`` picks cast (``where=lambda p: p.likelihood["camera"]``):
        :func:`hybrid_aif.dtypes.cast` on the params."""
        return _cast(self, dtype, where)


class LearnState(eqx.Module):
    """Optimizer state for an agent's gradient-learnable parts: ``learn``'s explicit carry.

    Returned by :meth:`System.learn` and the objective-side trainers and handed back on the next
    call; never held in a module attribute. A sibling of :class:`SystemParams`, not a field of it.

    Both fields are seeded by :meth:`System.init_learn_state`, so the carry's structure is fixed
    from the first call; each trainer steps its own entries, passes the rest through unchanged,
    and raises when an entry it owns is missing. Dict keys are static under jit.

    Design: docs/design/amortization.md A2 A6 and chunk 2c §(b).
    """

    #: Optimizer states, keyed by slot-qualified address (``"transition:x"``, the ``lr_by_slot``
    #: key space), plus ``scoring:<term>`` for each mounted term with a trainer
    #: (:attr:`objectives.registry.Estimator.trainer`) and :data:`ACTOR_KEY` for a policy module
    #: in ``policy_prior``.
    opt_states: dict[str, object] = eqx.field(default_factory=dict)
    #: Running statistics that are not optimizer state (the actor's return-normalization moving
    #: averages, the slow critic's update counter, the model-reduction counter), keyed by trainer
    #: name.
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


WorkingCarry = _hybrid.WorkingCarry  # defined beside PlanLearning; exported here


def _carry_learning(transition_fn, step_terms_fn, terminal_terms_fn, init_beliefs, counts):
    """Wrap the hybrid planning closures so the scan carry is a :class:`WorkingCarry` seeded with
    ``counts``: ``transition_fn(qs, action, working) -> (qs, working)`` and
    ``step_terms_fn(qs, action, t, qs_prev, working=)`` are the learning closures of
    :func:`~hybrid_aif.objectives.hybrid.make_transition` and
    :func:`~hybrid_aif.objectives.hybrid.make_step_terms`. The layer sits inside
    :func:`_carry_prev`, whose ``qs_prev`` is then the previous :class:`WorkingCarry`; every
    planner treats the carry as opaque."""

    def transition(carry, action):
        qs, after = transition_fn(carry.beliefs, action, carry.counts)
        return WorkingCarry(qs, after, carry.counts)

    def step_terms(carry, action, t, qs_prev=None):
        prev = None if qs_prev is None else qs_prev.beliefs
        return step_terms_fn(carry.beliefs, action, t, prev, working=carry.before)

    terminal = (
        None if terminal_terms_fn is None else (lambda carry: terminal_terms_fn(carry.beliefs))
    )
    return transition, step_terms, terminal, WorkingCarry(init_beliefs, counts, counts)


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


#: The planner fields a composite states only where its leaves agree (:data:`UNSET` elsewhere).
_PLANNER_FIELDS = (
    "policy_len",
    "planner",
    "gamma",
    "num_iters",
    "decomposition",
    "planning_belief",
    "feedback",
)


class Subsystem(NamedTuple):
    """One component's identity inside a merged composite, carried on ``System.subsystems``.

    Built by :func:`hybrid_aif.composition.merge.merge`. Metadata (no face computes from it)
    recording which edges crossed a boundary, which the flat graph does not show;
    :func:`~hybrid_aif.composition.merge.subsystem` derives a per-mind view from it, and a
    mapping keyed by a node a binding removed is refused with a sentence naming what it became.
    Indices are composite state factor and modality indices; the control node fields name nodes
    by composite name, as the merge left them.

    A record :func:`~hybrid_aif.description.build` reconstructs from a model description carries
    only what the document shows: the composite's names, its state factors, modalities and the
    control nodes still present. ``relayed``, ``terminated`` and ``identified`` are empty there and
    ``boundary`` is derived from the edges the document shows between components (a parent, a
    parameter variable or a read across them), ``reads`` is the document's own, and ``actions``
    names the present control nodes only, since a node a binding removed is absent from the
    document (design: model-description-0.3.md Q18.2).

    Design: docs/design/nesting-systems.md; docs/design/control-states.md C12;
    nesting-and-coupling.md N2, N3.
    """

    #: The composite state factor indices of the subsystem, in its declaration order (its block is
    #: contiguous, so local index ``i`` is ``states[i]``).
    states: tuple[int, ...]
    #: The composite modality indices of the subsystem, in its declaration order.
    modalities: tuple[int, ...]
    #: The control nodes this subsystem still chooses (the composite plans them).
    controls: tuple[str, ...] = ()
    #: ``(control node, action node)`` for every control node this record names (a driven node's
    #: is the one it had before it was driven, which is no action node of the System).
    actions: tuple[tuple[str, str], ...] = ()
    #: Pairs ``(node, state)``: a control node identified with another subsystem's state, which
    #: its reader reads a step later (a ``Link`` from a state).
    relayed: tuple[tuple[str, str], ...] = ()
    #: Control nodes a ``Clamp`` replaced with a prior.
    terminated: tuple[str, ...] = ()
    #: Pairs ``(node, driver)``: a driven control node and the control node ``driver`` whose
    #: action node drives it (an ``Identify``, or a ``Link`` from the action node, read at the
    #: same step); a joint planner makes the two one node, so every state factor that read
    #: ``node`` reads ``driver``.
    identified: tuple[tuple[str, str], ...] = ()
    #: Modalities reading only this subsystem's state factors.
    free_modalities: tuple[int, ...] = ()
    #: ``(modality, foreign parents)`` for the modalities an observation binding pointed across
    #: the boundary.
    bound_modalities: tuple[tuple[int, tuple[int, ...]], ...] = ()
    #: ``(state factor, parents)`` for terminal targets whose ``H_deps`` parents lie outside this
    #: subsystem.
    target_emissions: tuple[tuple[int, tuple[int, ...]], ...] = ()
    #: ``(modality, source, kernel, realise)``, composite names, for every modality of this
    #: subsystem that reads a state factor of another component (a belief ``Link``): the view
    #: leaves the modality an ordinary observation, and the driver supplies it each round from
    #: the source component's carry.
    reads: tuple[tuple[str, str, "str | Grid", str], ...] = ()
    #: What binds this subsystem to the others: ``"none"`` (no binding crosses its boundary),
    #: ``"blanket"`` (every binding across it is a belief read or a ``Link`` from an action node
    #: into a control node), ``"shared"`` (some binding identifies a latent outright: a relay, a
    #: structural ``Link``, a terminal ``Link``, an ``Identify``), or ``"coupled"`` (a top-level
    #: component of a :func:`~hybrid_aif.composition.couple.couple`, joined to the others by
    #: identified ports alone). :attr:`nested` is the ``"blanket"`` case: the subsystem keeps a
    #: blanket of its own inside the particle.
    boundary: str = "none"
    #: The composite names of the component's own emissions, its outputs drawn from its state
    #: factors (:attr:`System.emissions`).
    emissions: tuple[str, ...] = ()
    #: ``(field, value)`` pairs, the leaf's own planner configuration (``policy_len``,
    #: ``planner``, ``gamma``, ``num_iters``, ``decomposition``, ``planning_belief``,
    #: ``feedback``, ``mixture_schedule``, ``mixture_k``, ``epistemic``, ``terms`` and
    #: ``scoring``, the static declaration), with names qualified as the composite's; empty for a
    #: node that is itself a composite, whose leaves each hold their own. The leaf's habit,
    #: precision, durations and scoring parameters are ``SystemParams.parts[<leaf>]``.
    config: tuple[tuple[str, object], ...] = ()
    #: The composite names of the amortized recognition variables the leaf owns
    #: (``System.recognition``).
    recognition: tuple[str, ...] = ()

    @property
    def nested(self) -> bool:
        """Whether every binding across this component's boundary is a belief read or a
        ``Link`` from an action node into a control node, so the component keeps a blanket of its
        own inside the particle (and at least one binding crosses it)."""
        return self.boundary == "blanket"


class Ports(NamedTuple):
    """A System's ports and their types (:meth:`System.ports`): each maps a port's name to
    ``(LEVEL, levels)`` or ``(FLOAT, dimension)``."""

    #: The in-ports: each modality and each control node, except the slots ``System.observes``
    #: names (see :meth:`System.ports`).
    ins: dict[str, tuple[str, int]]
    #: The out-ports: each action node (typed as its control node) and each emission (typed by
    #: its kernel).
    outs: dict[str, tuple[str, int]]


class _SystemMeta(type(eqx.Module)):
    """The System's metaclass: a call that names the nodes with one combined keyword is refused
    with a sentence naming the keywords that declare the nodes' names."""

    def __call__(cls, *args, **kwargs):
        if "factor_names" in kwargs:
            raise TypeError(
                "System() got the keyword factor_names, which is not a System keyword: the "
                "nodes' names are declared as state_names= (the state factors', in order) and "
                "control_names= (the control nodes', in controls order), beside modality_names=. "
                "A factor_names tuple splits in two: its first len(B_deps) - len(controls) "
                "entries are state_names, and the control nodes' names after them are "
                "control_names"
            )
        return super().__call__(*args, **kwargs)


class System(eqx.Module, metaclass=_SystemMeta):
    """Static structure of a System: discrete, continuous or mixed.

    Every field is static; a table with field defaults and meanings is in
    ``docs/composing-systems.md``. A System's variables are its *nodes*: the state factors first,
    then the **control nodes**, which the System sets by decision (``controls`` gives their node
    indices). ``state_names`` names the state factors and ``control_names`` the control nodes;
    :attr:`node_names` is the two in node order. ``B_deps``, ``D_deps`` and ``H_deps`` have one
    entry per node;
    ``A_deps`` has one entry per modality, listing the state factors it observes. A transition
    reads its state parents from the previous step and its control parents at the step the control
    is chosen; a control node's levels are its readers' table axes. ``epistemic`` names variables:
    naming a state factor mounts the infogain of every modality that observes it, so switching one
    state factor off means naming the others.

    The pymdp-style keywords ``controlled`` and ``B_control_deps`` are construction sugar,
    translated once into control nodes: each controlled state factor ``f`` (in node order) gets a
    node ``u_<f>``, appended after the declared state factors, and the node is the last transition
    parent of ``f`` (under ``B_control_deps``, of every state factor listing ``f``, at the listed
    position), so every table keeps its layout. A System built from the sugar equals the one that
    declares those nodes natively. The sugar is readable back (``controlled``,
    ``B_control_deps``) on a sugar-shaped System. A discrete node's level count is its readers'
    parent axis (:meth:`policies`); a node is continuous when ``SystemParams.control`` holds its
    box, and its readers then read a float vector (a neural transition's input, a linear-Gaussian
    transition's ``G u`` term).

    The fields fall in these groups (each field's entry below gives its meaning):

    - the graph: ``B_deps``, ``A_deps``, ``D_deps``, ``H_deps``, ``C_deps``, ``cells``,
      ``controls``, ``action_names``, ``emissions``, ``observes``, ``black_box``, and the sugar
      ``controlled`` and ``B_control_deps``;
    - names: ``state_names``, ``control_names``, ``modality_names``, ``dims``;
    - planning and the objective: ``policy_len``, ``planner``, ``gamma``, ``decomposition``,
      ``epistemic``, ``terms``, ``scoring``, ``branch_on``, ``planning_belief``, ``feedback``,
      and ``sophisticated`` (the alternative syntax for ``branch_on`` and ``plan_learning``);
    - inference: ``num_iters``, ``mixture_k``, ``mixture_schedule``, ``plan_learning``,
      ``recognition``, ``external_inputs``;
    - learning: ``readings``, ``reduction_period``, ``reduction_pairs``;
    - provenance: ``subsystems``.

    A field written **by name** at construction (a dict) is stored as a tuple of pairs, because
    static fields must hash; ``dict(system.<field>)`` reads it back.

    Design: docs/design/control-states.md C1, C4, C6, C11.
    """

    #: One entry per node: the node indices each state factor's transition conditions on (its
    #: state parents, read from the previous step, and its control parents). The length fixes the
    #: node count. A continuous state factor may name only earlier categorical state factors (never
    #: itself), since state factors are computed within a step in declaration order.
    B_deps: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    #: One entry per modality: the state factors it observes, by node index; the length fixes the
    #: modality count. Action node ``k`` is named as ``len(B_deps) + k``.
    A_deps: tuple[tuple[int, ...], ...] = eqx.field(static=True, default=())
    #: The planner's registry name: ``"enumeration"``, ``"factorised"``, ``"cem"``,
    #: ``"gradient"``, ``"mixed"``, ``"habit"`` or ``"sophisticated"``. Resolved at call time; a
    #: face's ``planner=`` argument overrides it.
    planner: str = eqx.field(static=True, default="enumeration")
    #: The policy precision in q_pi ~ softmax(-gamma G + log E), read when
    #: ``SystemParams.precision`` is ``None``.
    gamma: float = eqx.field(static=True, default=16.0)
    #: The number of mean-field fixed-point sweeps in ``infer_states``.
    num_iters: int = eqx.field(static=True, default=16)
    #: Which decomposition of the expected free energy G the term tree carries:
    #: ``"utility_infogain"`` or ``"risk_ambiguity"``. The ``epistemic`` selector mounts each one's
    #: own epistemic leaf (``infogain`` or ``ambiguity``) and never touches risk.
    decomposition: Literal["utility_infogain", "risk_ambiguity"] = eqx.field(
        static=True, default="utility_infogain"
    )
    #: The per-variable epistemic selector. ``True`` mounts every eligible variable,
    #: ``False`` or ``()`` none, and a tuple names the variables that carry an epistemic leaf: an
    #: ``infogain`` leaf under ``utility_infogain``, an ``ambiguity`` leaf under ``risk_ambiguity``
    #: (risk is never touched by the selector). The grammar, shared with ``learn``'s rate keys
    #: (:mod:`hybrid_aif.system.selectors`):
    #:
    #: .. code-block:: text
    #:
    #:    <state>                a state variable, by bare state factor name ("pose_cell")
    #:    <state>:<dim>          one dimension of a continuous state factor ("pose:b1"),
    #:                           named by ``dims`` or dim0, dim1, ... by position
    #:    obs:<modality>         an observation variable ("obs:cue")
    #:    <slot>:<name>          a declaration's parameter variables as a group, the slot
    #:                           one of likelihood, transition, initial_state_prior
    #:    <slot>:<name>:<block>  one independent posterior block ("likelihood:pose:noise")
    #:
    #: Any segment may be ``*``, matching that whole segment (no partial globs): ``("*",)`` is
    #: every state variable, ``("*:*",)`` every parameter variable, ``("obs:*",)`` every
    #: observation variable, ``("pose:*",)`` every dimension of ``pose`` (the whole state factor).
    #: A wildcarded first segment ranges over the parameter slots only. Selection is a union. A
    #: wildcard never validates less than its literal part; only a class empty through its
    #: wildcarded segment resolves silently to nothing.
    #:
    #: A state name mounts every modality that observes it, and a modality's leaf is its
    #: information about the *selected* parents, the unselected ones marginalised (its ambiguity
    #: likewise); ``obs:<modality>`` scores the modality over all its parents. Dimension keys on
    #: one state factor select one sub-block: on a Gaussian-carried state factor each modality's
    #: leaf is its information about those dimensions, the others marginalised, and under
    #: ``risk_ambiguity`` its ambiguity is ``E H[y | x_S]``; a sub-block of a mixture-carried state
    #: factor raises. Selecting every dimension selects the state factor. A continuous state
    #: factor no modality observes has no information to gain: named literally it raises, and the
    #: drive toward a sharper belief about it is the ``uncertainty`` term.
    epistemic: bool | tuple[str, ...] = eqx.field(static=True, default=True)
    #: The shared Runnalls budget: the number of components every mixture-carried belief keeps.
    #: Required as soon as any state factor carries a ``MixtureGaussian``.
    mixture_k: int | None = eqx.field(static=True, default=None)
    #: The mixture compression (the reduce half of grow and reduce): ``"runnalls"``, or
    #: ``"mode_aligned"``, which keeps each component owned by one mode (needed to filter a state
    #: factor whose transition switches).
    mixture_schedule: Literal["runnalls", "mode_aligned"] = eqx.field(
        static=True, default="runnalls"
    )
    #: The learnable tables a plan reads as learned within the plan: carried through the planning
    #: scan as working counts, seeded from the posterior the plan starts from and absorbing each
    #: imagined step's expected count statistic (the model's joint over child and parents), so a
    #: later read of a row predicts through the counts the plan has imagined and its novelty
    #: declines (the Pólya urn). A tuple of keys in the grammar of ``learn``'s rates:
    #: ``transition:<factor>``,
    #: ``likelihood:<modality>`` (a ``ConditionalDirichlet``, directly or as the reader of a
    #: parameter variable) and ``parameter:<name>`` (every reader of the variable); or ``True``,
    #: every learnable table. Readers of one parameter variable learned within the plan share one
    #: working table. ``()`` (the default) reads every table frozen at its posterior mean within a
    #: plan, pymdp's convention. An inference setting, as ``mixture_schedule`` is; carried on the
    #: hybrid and the fused all-tabular paths (on the fused path a ``branch_on`` branch absorbs
    #: its own outcome), refused on the amortized path. Design:
    #: docs/design/parameters-within-the-plan.md D1, D2, A3.
    plan_learning: bool | tuple[str, ...] = eqx.field(static=True, default=())
    #: Hierarchical conditional priors: the parent state factors of each state factor's initial
    #: prior (``None``: all unconditional). Ancestral: parents must be earlier, of tabular
    #: families.
    D_deps: tuple[tuple[int, ...], ...] | None = eqx.field(static=True, default=None)
    #: The parent state factors of each state factor's ``terminal_targets`` entry (``None``: all
    #: unconditional; see :meth:`resolve_terminal_targets`). As ``D_deps``, except that parents
    #: need not be declared earlier and a state factor may not parent its own terminal target.
    H_deps: tuple[tuple[int, ...], ...] | None = eqx.field(static=True, default=None)
    #: The parents of each conditional per-step target, **by name**: ``{"lane": ("u_option",),
    #: "cue": ("context",)}`` at construction, keyed by the targeted node (a state factor's
    #: ``state_targets`` entry, a modality's ``obs_targets`` entry), each parent a discrete control
    #: node or a categorical state factor; stored as ``(key, parents)`` pairs sorted by key. The
    #: target under a key is a ``ConditionalCategorical`` (child axis first, then one axis per
    #: parent in this order) or a ``ConditionalGaussian`` (``loc`` ``(*levels, d)``, ``chol``
    #: ``(*levels, d, d)``; control parents only). At each step of a plan the target is resolved
    #: under its parents, a clamped control node's level selecting its own distribution and a
    #: state parent averaging mean-field, then scored as an unconditional target is. Design:
    #: step-durations.md T5.
    C_deps: tuple[tuple[str, tuple[str, ...]], ...] = eqx.field(static=True, default=())
    #: The grid of each continuous state factor that a tabular state factor reads as a parent,
    #: **by name**: ``{"x": Grid(...)}`` at construction, stored as ``(state factor, Grid)`` pairs
    #: in node order. A tabular state factor naming such a state factor in ``B_deps`` has a table
    #: parent axis of ``Grid.n_cells`` entries, the cell the parent's belief falls in, flattened in
    #: C order. Design: terrain-field.md.
    cells: tuple[tuple[str, Grid], ...] = eqx.field(static=True, default=())
    #: One name per modality, in modality order; ``None`` at construction fills in
    #: ``modality_{m}``. These are the keys of ``SystemParams.likelihood``, ``obs_targets`` and the
    #: term tree.
    modality_names: tuple[str, ...] = eqx.field(static=True, default=None)
    #: One name per state factor, in node order; ``None`` at construction fills in ``factor_{i}``
    #: for state factor ``i``. With ``control_names`` and ``modality_names``, the single source of
    #: names and order: keys in ``SystemParams``, targets and rate maps refer to them and are
    #: validated at every face. No node is named after a reserved first segment of the key
    #: grammar (``obs`` or a slot word).
    state_names: tuple[str, ...] = eqx.field(static=True, default=None)
    #: One name per control node, in ``controls`` order; ``None`` at construction fills in the
    #: sugar's ``u_<f>`` per controlled state factor ``f`` (``u_factor_{i}`` under generated state
    #: names) and is refused on a System that declares its control nodes natively
    #: (``controls=``), whose nodes need names of their own. Under the sugar, a given tuple must
    #: equal the generated one.
    control_names: tuple[str, ...] = eqx.field(static=True, default=None)
    #: Names for the dimensions of continuous state factors: ``{factor: (name, ...)}`` at
    #: construction (``{"pose": ("x", "y", "yaw", "b1", "b2")}``), stored as pairs in node order.
    #: ``epistemic`` selects a dimension as ``<factor>:<name>``; a state factor declared without
    #: names has ``dim0``, ``dim1``, ... by position, which change meaning when the state is
    #: reordered. Names are unique within a state factor, one per dimension, single key segments
    #: (no ``:``, ``.``, ``*`` or brackets), and ``dimN`` may name position ``N`` only.
    dims: tuple[tuple[str, tuple[str, ...]], ...] = eqx.field(static=True, default=())
    #: The structural half of the recognition slot: which variables an amortized module infers and
    #: what each conditions on (:class:`~hybrid_aif.factors.Recognition`); ``None`` or ``()``:
    #: none. The modules live in ``SystemParams.recognition``.
    recognition: tuple[Recognition, ...] | None = eqx.field(static=True, default=None)
    #: The term selector: which estimator-registry terms G carries beyond the ``infogain`` terms
    #: ``epistemic`` selects (:data:`~hybrid_aif.objectives.registry.ROWS`). Never inferred from
    #: ``SystemParams.scoring``, so the term tree's shape is a function of the System. The grammar
    #: is the term name, optionally applied to an ``epistemic`` variable key:
    #:
    #: .. code-block:: text
    #:
    #:    <term>                   a term with no target ("g_tail", or a term scoring the
    #:                             whole rollout against a model of its own)
    #:    <term>:<slot>:<name>     a term on a declared kernel ("occupancy:transition:stoch")
    #:    <term>:obs:<modality>    a term on an observation modality
    #:                             ("predicted_reward:obs:reward")
    #:    <term>:<state>           a term on a state variable ("uncertainty:pose")
    #:    <term>:<state>:<dim>     a term on dimensions of a continuous state variable
    #:                             ("uncertainty:pose:b1"; keys on one state factor are one
    #:                             sub-block, as under epistemic)
    #:
    #: The term segment is always literal; the variable part may wildcard where the term declares
    #: a target class (``"occupancy:transition:*"``). Selection is a union. A term mounts on a
    #: target only if the registry has an estimator for that variable's family.
    terms: tuple[str, ...] = eqx.field(static=True, default=())
    #: The structural half of the scoring slot, keyed by term name (the parameters live in
    #: ``SystemParams.scoring``): for example a term's preference model that is itself a declared
    #: System; ``dict(system.scoring)`` reads it. ``terms`` does the mounting: an entry whose term
    #: it does not mount raises.
    scoring: tuple[tuple[str, object], ...] = eqx.field(static=True, default=())
    #: ``(name, width)`` vectors that a recognition module may condition on but this System does
    #: not declare, spelled ``"external:<name>"`` in a ``Recognition`` parent set and supplied per
    #: call (``infer_states(externals={...})``); the width lets ``parent_dims`` validate
    #: statically.
    external_inputs: tuple[tuple[str, int], ...] = eqx.field(static=True, default=())
    #: ``(name, Subsystem)`` pairs recording which state factors, modalities and control nodes came
    #: from which component of a :func:`~hybrid_aif.composition.merge.merge` call (``None`` for a
    #: hand-declared System); ``dict(system.subsystems)`` reads it. No face computes from it; a
    #: refusal reads it to name what a node a binding removed became.
    subsystems: tuple[tuple[str, Subsystem], ...] | None = eqx.field(static=True, default=None)
    #: The observation modalities, by name, whose outcomes the ``"sophisticated"`` planner branches
    #: on (:mod:`hybrid_aif.planning.sophisticated`): each categorical, over categorical parents
    #: only. Empty for every other planner, which never reads it.
    branch_on: tuple[str, ...] = eqx.field(static=True, default=())
    #: The alternative syntax for the two halves of sophisticated planning, in ``epistemic``'s
    #: selector grammar: ``"obs:<modality>"`` branches the plan on that modality's outcomes (the
    #: state half, ``branch_on``), and ``"transition:<factor>"``, ``"likelihood:<modality>"`` and
    #: ``"parameter:<name>"`` carry that table's counts within the plan (the parameter half,
    #: ``plan_learning``). ``sophisticated=("obs:fix", "transition:terrain")`` is exactly
    #: ``branch_on=("fix",), plan_learning=("transition:terrain",)``. The two fields are the
    #: canonical form: construction lowers this tuple into them, every face reads them, and a
    #: description writes them alone; this field keeps what was written. A tuple of literal keys
    #: (``plan_learning=True`` carries every table); declaring it beside a ``branch_on`` or
    #: ``plan_learning`` of one's own is refused. ``None`` (the default): not used. Design:
    #: docs/design/parameters-within-the-plan.md A1.
    sophisticated: tuple[str, ...] | None = eqx.field(static=True, default=None)
    #: Which belief a plan's information-gain leaves read on a Gaussian-carried state factor:
    #: ``"prior"``, the predictive belief (predict only); ``"posterior"``, a second chain that
    #: conditions on the observations expected along the plan, so a linear-Gaussian modality's
    #: leaf is the gain of its observation given the plan's earlier ones. Every other leaf reads
    #: the predictive belief, except a Gaussian target under ``feedback``. Design:
    #: planning-with-observations.md D1, D2; closed-loop-planning.md.
    planning_belief: Literal["prior", "posterior"] = eqx.field(static=True, default="prior")
    #: How a plan's Gaussian targets are scored. ``None``: open loop, on the predictive belief.
    #: ``"linear_quadratic"``: closed loop, as the policy that follows the plan and corrects
    #: deviations from it by the linear-quadratic regulator's feedback on the estimate. Each
    #: Gaussian state, terminal and linear-Gaussian observation target on a Gaussian-carried state
    #: factor is then scored by its cross-entropy on the closed-loop state distribution (the
    #: epistemic chain's posterior plus the estimate spread the corrections leave), and a Gaussian
    #: target on a continuous control node also charges the corrections. Requires
    #: ``planning_belief="posterior"``. Design: closed-loop-planning.md.
    feedback: Literal["linear_quadratic"] | None = eqx.field(static=True, default=None)
    #: Which variable each slot observes: ``(slot, variable)`` pairs, given as ``{"cell_obs":
    #: "cell"}`` at construction. A slot is a modality or a control node; a variable is an action
    #: node or an emission of this System (the variables it generates), and the slot's port type
    #: must equal the variable's. A control node named as a slot is *driven*: the variable's
    #: realised value is its level, its ``action_names`` entry is ``None``, and no planner chooses
    #: it. A slot the map does not name observes the variable of its own name, so it is an in-port
    #: of the System (:meth:`ports`).
    #:
    #: The datum of an observed variable is an observation like any other: the caller supplies
    #: it, keyed by the variable (:meth:`infer_states` fans it out to every slot observing it),
    #: and ``replay`` and ``run`` realise the emissions after a round and feed them back as the
    #: next round's observation. An own-belief read: ``emissions={"cell": Emission(("x",),
    #: realise="mode", kernel=GRID)}, observes={"cell_obs": "cell"}``. The variable set (state
    #: factors, control nodes, action nodes, emissions and the slots ``observes`` does not name)
    #: is derived from the declarations, never stored. Design: engine-consolidation.md;
    #: structure-learning.md S8; nesting-and-coupling.md N3.
    observes: tuple[tuple[str, str], ...] = eqx.field(static=True, default=())
    #: The outputs the System draws from its state factors, **by name**: ``{"fix": ("x",),
    #: "img": Emission(("x", "terrain"), realise="mode")}`` at construction (a tuple of parents is
    #: a draw), stored as ``(name, Emission)`` pairs in the order given (the order fixes the key
    #: each emission draws with). An emission is the action node of a state factor: its parents are
    #: state factors and control nodes (a control parent reads the node's realised level), its
    #: kernel is ``SystemParams.emission[name]`` or the static identity or ``Grid`` kernel its
    #: declaration names, and its port type is read off that kernel.
    #:
    #: An emission has no node index: nothing in the model reads or scores it, so no ``A_deps``,
    #: target, ``C_deps``, ``H_deps``, ``epistemic`` key, ``branch_on`` or ``readings`` entry may
    #: name it. A modality may observe it (``observes``), which is the only way the model reads
    #: it. Design: nesting-and-coupling.md N5.
    emissions: tuple[tuple[str, Emission], ...] = eqx.field(static=True, default=())
    #: ``(state factor or modality, control node)`` pairs: the named learnable table's one reading
    #: at a step pools ``SystemParams.readings[name][level]`` fast readings, the level being the
    #: control node's at that step, and its novelty term is scaled by that count. A table not
    #: named pools one. Design: step-durations.md T10.
    readings: tuple[tuple[str, str], ...] = eqx.field(static=True, default=())
    #: The schedule of Bayesian model reduction on the growable state factors: every
    #: ``reduction_period``-th call of ``learn`` runs :meth:`reduce` on each of them, the counter
    #: carried in ``LearnState.moments``; ``None`` leaves reduction to explicit calls. Design:
    #: structure-learning.md S4.
    reduction_period: int | None = eqx.field(static=True, default=None)
    #: How many candidate pairs a reduction tests, chosen greedily as the best-scoring by mutual
    #: expected log-likelihood.
    reduction_pairs: int = eqx.field(static=True, default=8)
    #: The control nodes, by node index, ascending: the nodes the System sets by decision rather
    #: than by inference. They follow every state factor; each is read by a transition or sized
    #: by another declaration that names it (a target, a modality, durations). A control node has
    #: no parents, initial prior or terminal target of its own. A policy carries one column per
    #: control node, in this order.
    controls: tuple[int, ...] = eqx.field(static=True, default=())
    #: One action node per control node, in ``controls`` order: the node through which the
    #: decision leaves the System, a child of its control node (kernel P(a | u), a delta unless
    #: ``SystemParams.actuator`` declares one). Declared with the control nodes, as
    #: ``modality_names`` with ``A_deps``, paired with them by position, both names free; the
    #: sugar's pair for state factor ``f`` is ``u_<f>`` and ``a_<f>``. The entry is ``None`` for a
    #: driven control node (an input, driven from outside through ``observes``: nothing chooses
    #: it, so nothing realises it). Action nodes follow the control nodes in node order, so
    #: ``A_deps`` names action node ``k`` as ``len(B_deps) + k``.
    action_names: tuple[str, ...] = eqx.field(static=True, default=None)
    #: The ports of a black box (an external process declared by its ports alone), **by name**:
    #: ``{"v": (FLOAT, 2), "gauge": (LEVEL, 4)}`` at construction, stored as ``(port, (kind,
    #: size))`` pairs in the order given. An in-port is a control node no declaration reads and no
    #: action node realises (a link into it drives it); an out-port is an emission with no parents
    #: whose kernel is ``"black_box"``. Nothing between them is stated, so no face that needs the
    #: inside (inference, planning, learning, sampling, ``as_env``, ``mirror``, ``joint_planner``)
    #: runs on a System that holds one, and its step under ``run`` is a custom one. A splice
    #: carries each component's entries under its prefix. Built by
    #: :func:`~hybrid_aif.composition.black_box.black_box`; docs/composing-systems.md, "External
    #: processes".
    black_box: tuple[tuple[str, tuple[str, int]], ...] = eqx.field(static=True, default=())
    #: The planning horizon: the number of steps in a policy (for the enumeration planner, the
    #: policy length).
    policy_len: int = eqx.field(static=True, kw_only=True)
    #: Sugar: per state factor, whether it gets a control node ``u_<f>`` (with action node
    #: ``a_<f>``).
    controlled: dataclasses.InitVar[tuple[bool, ...] | None] = eqx.field(default=None, kw_only=True)
    #: Sugar: per state factor, the controlled state factors whose nodes slice its transition, in
    #: axis order (``None``: each controlled state factor reads its own node alone). Design:
    #: docs/design/control-states.md (Timing; C6).
    B_control_deps: dataclasses.InitVar[tuple[tuple[int, ...], ...] | None] = eqx.field(
        default=None, kw_only=True
    )

    def __post_init__(self, controlled, B_control_deps) -> None:
        """Translate the control sugar into control nodes, and fill in generated names.

        The sugar keywords are translated once (see the class docstring): the stored fields then
        hold the node form, and the sugar is gone. Values that a ``dataclasses.replace`` passes
        back from the derived properties are recognised and ignored; sugar given explicitly to a
        System that already has control nodes raises (construct afresh from the sugar, or edit
        the nodes). Names: ``state_names=None`` fills in ``factor_{i}`` per state factor and
        ``control_names=None`` the sugar's ``u_<f>`` per sugar node, so the name tuples are always
        present, the single source of names and node order."""
        sugar = {"controlled": controlled, "B_control_deps": B_control_deps}
        given = {k: v for k, v in sugar.items() if v is not None and not isinstance(v, _Derived)}
        if given:
            if self.controls:
                named = self.control_names or self.controls
                raise ValueError(
                    f"{sorted(given)} declare controls the pymdp way, and this System already "
                    f"has control nodes {list(named)} "
                    "(controls=...): give one declaration or the other. A dataclasses.replace "
                    "with sugar keywords on a System that has control nodes cannot re-translate "
                    "them: construct the System afresh from the sugar, or edit B_deps/controls"
                )
            self._translate_sugar(given)
        elif not self.controls and type(controlled) is _Derived and any(controlled):
            raise ValueError(
                "controlled= was read back off a System with control nodes and passed to a "
                "System declaring none (controls=()): the node-form B_deps and control_names "
                "already hold the control nodes, so pass controls= with them, or declare the "
                "sugar afresh with controlled="
            )
        elif self.control_names is None and self.controls:
            raise ValueError(
                "control_names=None on a System with control nodes "
                f"(controls={self.controls}): a natively declared control node has no generated "
                "name; pass control_names=(...), one name per control node in controls order, "
                "or construct the System afresh from the sugar"
            )
        else:
            self._derive_control_names()
        if self.state_names is None:
            self.state_names = keyed.generated_names(
                "factor", len(self.B_deps) - len(self.controls)
            )
        if self.control_names is None:
            self.control_names = ()
        if self.modality_names is None:
            self.modality_names = keyed.generated_names("modality", len(self.A_deps))
        self.dims = self._normalized_dims()
        self.C_deps = self._normalized_C_deps()
        self.cells = self._normalized_cells()
        self.emissions = self._normalized_emissions()
        self.observes = self._normalized_observes()
        self.black_box = self._normalized_black_box()
        self._lower_sophisticated()

    def _lower_sophisticated(self) -> None:
        """Lower ``sophisticated=`` into the canonical ``branch_on`` and ``plan_learning``.

        A canonical value lowered here is marked (:class:`_Lowered`), so a ``dataclasses.replace``
        that passes it back, with ``sophisticated`` unchanged or replaced, lowers afresh; a value
        of the caller's own beside ``sophisticated`` is the second syntax, refused."""
        soph = self.sophisticated
        if soph is None:
            for name in ("branch_on", "plan_learning"):
                value = getattr(self, name)
                if isinstance(value, _Lowered):
                    setattr(self, name, tuple(value))
            return
        if isinstance(soph, bool):
            raise ValueError(
                f"sophisticated={soph!r}: sophisticated takes a tuple of keys "
                "('obs:<modality>' to branch on, 'transition:<factor>', 'likelihood:<modality>' or "
                "'parameter:<name>' to carry a table's counts); to carry every learnable table "
                "within the plan declare plan_learning=True instead"
            )
        if not isinstance(soph, tuple) or not all(isinstance(k, str) for k in soph):
            raise ValueError(
                f"sophisticated must be a tuple of keys ('obs:<modality>', "
                f"'transition:<factor>', 'likelihood:<modality>', 'parameter:<name>'), got "
                f"{soph!r}; write sophisticated=(...) with one key per entry"
            )
        mine = [
            name
            for name in ("branch_on", "plan_learning")
            if not isinstance(getattr(self, name), _Lowered)
            and getattr(self, name) not in ((), False)
        ]
        if mine:
            raise ValueError(
                f"sophisticated={soph!r} and {' and '.join(f'{n}=' for n in mine)} both declare "
                "sophisticated planning: declare one syntax, either sophisticated=... or "
                "branch_on=/plan_learning=... (a dataclasses.replace of branch_on or plan_learning "
                "on a System declared through sophisticated= passes sophisticated=None with it)"
            )
        branch, tables = [], []
        for key in soph:
            kind, _, name = key.partition(":")
            if "*" in key:
                raise ValueError(
                    f"sophisticated: {key!r} is a pattern; sophisticated names its modalities and "
                    "tables by literal keys (to carry every learnable table within the plan, "
                    "declare plan_learning=True instead)"
                )
            if kind == "obs" and name and ":" not in name:
                branch.append(name)
            elif kind in _SOPHISTICATED_TABLES and name:
                tables.append(key)
            else:
                raise ValueError(
                    f"sophisticated: unknown key {key!r}; a key is 'obs:<modality>' (branch the "
                    "plan on its outcomes) or 'transition:<factor>', 'likelihood:<modality>' or "
                    "'parameter:<name>' (carry that table's counts within the plan)"
                )
        self.branch_on = _Lowered(branch)
        self.plan_learning = _Lowered(tables)

    @property
    def node_names(self) -> tuple[str, ...]:
        """Every node's name in node order: ``state_names``, then ``control_names``. The
        index-based fields (``B_deps``, ``A_deps``, ``D_deps``, ``H_deps``, ``controls``) index
        into this tuple."""
        return (*(self.state_names or ()), *(self.control_names or ()))

    def _normalized_black_box(self) -> tuple:
        """``black_box`` as ``(port, (kind, size))`` pairs (a mapping, or pairs), the in-ports in
        control-node order and then the out-ports in emission order, whatever order they were
        given in, so every path that builds one System builds it alike; a name that is neither
        sorts last, and a value of another shape is passed through, for
        :meth:`_check_black_box` to refuse."""
        given = self.black_box
        if not given:
            return ()
        items = tuple(given.items()) if isinstance(given, Mapping) else given
        try:
            pairs = [
                (name, (kind, int(size) if isinstance(size, np.integer) else size))
                for name, (kind, size) in items
            ]
        except (TypeError, ValueError):
            return given
        names = self.node_names
        nodes = [names[c] for c in self.controls if 0 <= c < len(names)]
        try:
            emitted = [e for e, _ in self.emissions]
        except (TypeError, ValueError):
            emitted = []
        order = {name: i for i, name in enumerate([*nodes, *emitted])}
        return tuple(sorted(pairs, key=lambda pair: order.get(pair[0], len(order))))

    def _normalized_observes(self) -> tuple:
        """``observes`` as ``(slot, variable)`` pairs in the order given (a mapping, or pairs); a
        value of another shape is passed through for :meth:`_check_observes` to refuse."""
        given = self.observes
        if not given:
            return ()
        if isinstance(given, Mapping):
            return tuple(given.items())
        if isinstance(given, tuple | list):
            return tuple(tuple(item) if isinstance(item, list) else item for item in given)
        return given

    def _normalized_emissions(self) -> tuple:
        """``emissions`` as ``(name, Emission)`` pairs in the order given. A mapping, or pairs, of
        a name to a tuple of parents (a draw) or an :class:`~hybrid_aif.reads.Emission`; a value
        of another shape is passed through for :meth:`_check_emissions` to refuse."""
        given = self.emissions
        if not given:
            return ()
        if isinstance(given, Mapping):
            items = tuple(given.items())
        elif isinstance(given, tuple | list):
            items = tuple(given)
        else:
            return given
        out = []
        for item in items:
            if not (isinstance(item, tuple | list) and len(item) == 2):
                return given
            name, decl = item
            if isinstance(decl, Emission):
                decl = Emission(
                    tuple(decl.parents) if isinstance(decl.parents, list) else decl.parents,
                    decl.realise,
                    decl.kernel,
                )
            elif isinstance(decl, tuple | list):
                decl = Emission(tuple(decl))
            out.append((name, decl))
        return tuple(out)

    def _normalized_cells(self) -> tuple:
        """``cells`` as ``(factor, Grid)`` pairs in factor order (a name that is not a node sorts
        last, for :meth:`_check_cells` to name). A value of the wrong shape is passed through
        for :meth:`_check_cells` to refuse."""
        cells = self.cells
        if cells is None:
            return ()
        pairs = list(cells.items()) if isinstance(cells, Mapping) else cells
        try:
            pairs = [(name, grid) for name, grid in pairs]
        except (TypeError, ValueError):
            return cells
        order = {name: i for i, name in enumerate(self.node_names)}
        return tuple(sorted(pairs, key=lambda pair: order.get(pair[0], len(order))))

    def _check_cells(self) -> None:
        """``cells``: pairs of a state factor's name and a :class:`~hybrid_aif.cells.Grid`, one
        per factor. Whether the factor is continuous, and which tabular factors read it, is
        checked with the params."""
        cells = self.cells
        if not isinstance(cells, tuple) or not all(
            isinstance(entry, tuple)
            and len(entry) == 2
            and isinstance(entry[0], str)
            and isinstance(entry[1], Grid)
            for entry in cells
        ):
            raise TypeError(
                f"cells must map state factor names to Grid declarations, got {cells!r}"
            )
        names = [name for name, _ in cells]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(f"cells names {repeated} twice: one Grid per factor")
        states = self.state_names
        unknown = sorted(set(names) - set(states))
        if unknown:
            raise ValueError(
                f"cells names {unknown}, which are not state factors; the state factors are "
                f"{sorted(states)}"
            )

    @property
    def _cell_grids(self) -> dict[int, Grid]:
        """The declared grids by state factor index."""
        index = {name: i for i, name in enumerate(self.node_names)}
        return {index[name]: grid for name, grid in self.cells}

    def _cell_parents(self) -> dict[int, Grid]:
        """The grids of the factors some tabular factor reads as a parent, by factor index: the
        factors whose belief a transition site reads as a probability vector over cells."""
        grids = self._cell_grids
        read = {g for deps in self._B for g in deps}
        return {g: grid for g, grid in grids.items() if g in read}

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
        order = {name: i for i, name in enumerate(self.node_names)}
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
            raise ValueError(
                f"control node(s) {list(self.control_names)} are "
                "declared without action nodes: pass action_names=(...), one action name per "
                "control node in controls order"
            )
        if not isinstance(self.action_names, tuple) or not all(
            a is None or isinstance(a, str) for a in self.action_names
        ):
            raise TypeError(
                "action_names must be a tuple of str (None for a control node that is an input, "
                f"driven from outside), got {self.action_names!r}"
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
        generated = self.state_names is None
        names = keyed.generated_names("factor", n) if generated else tuple(self.state_names)
        if len(names) != n:
            raise ValueError(
                f"state_names has {len(names)} name(s) for the {n} state factor(s) B_deps "
                "declares: with the pymdp-style control keywords, name each declared state factor "
                "(the control nodes they create are named u_<state factor>)"
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
                f"state_names declares {clash}, the name(s) the control sugar gives the "
                "control node(s) it creates (u_<factor>); rename the state factor(s), or "
                "declare the control nodes natively to choose their names"
            )
        if self.control_names is not None and tuple(self.control_names) != tuple(created):
            raise ValueError(
                f"control_names={self.control_names!r} with the pymdp-style control keywords: "
                f"the sugar names each controlled state factor's control node u_<factor> "
                f"({created}); declare the control nodes natively to choose their names"
            )
        self.state_names = names
        self.control_names = tuple(created)
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
        n_controls = len(self.controls)
        for what, names, count, kind in (
            ("state_names", self.state_names, len(self.B_deps) - n_controls, "state factor"),
            ("control_names", self.control_names, n_controls, "control node"),
            ("modality_names", self.modality_names, len(self.A_deps), "modality"),
        ):
            if not isinstance(names, tuple) or not all(isinstance(n, str) for n in names):
                raise TypeError(f"{what} must be a tuple of str, got {type(names).__name__}")
            if len(names) != count:
                fix = (
                    "one name per control node, in controls order"
                    if what == "control_names"
                    else f"pass {what}=None to regenerate them, or give new names"
                )
                raise ValueError(
                    f"{what} has {len(names)} name(s) for {count} {kind}(s) (B_deps declares "
                    f"{len(self.B_deps)} node(s), {n_controls} of them control nodes): one name "
                    f"per {kind} is required; {fix}"
                    if what != "modality_names"
                    else f"{what} has {len(names)} name(s) for {count} {kind}(s): one name per "
                    f"{kind} is required; {fix}"
                )
        nodes = self.node_names
        if len(set(nodes)) != len(nodes):
            dup = sorted({n for n in nodes if nodes.count(n) > 1})
            raise ValueError(
                f"state_names and control_names repeat {dup}: every node needs its own name"
            )
        reserved = sorted(set(nodes) & set(_KEY_KINDS))
        if reserved:
            raise ValueError(
                f"node names {reserved} are reserved: a key's first segment is one of "
                f"{list(_KEY_KINDS)} or a state factor's name ('pose:b1', a dimension of 'pose'), "
                "so a node named after one would be ambiguous; rename it"
            )
        self._check_dims()
        shared = sorted(set(nodes) & set(self.modality_names))
        if shared:
            raise ValueError(
                f"{shared} name both a node and a modality: a System has one namespace of "
                "variables, so name the modality apart (for a sensor reading the node, "
                f"{shared[0]!r}_obs)"
            )
        self._check_action_names()
        self._check_emissions()
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
        if not (
            isinstance(self.plan_learning, bool)
            or (
                isinstance(self.plan_learning, tuple)
                and all(isinstance(k, str) for k in self.plan_learning)
            )
        ):
            raise ValueError(
                "plan_learning must be True, False or a tuple of keys "
                "('transition:<factor>', 'likelihood:<modality>', 'parameter:<name>'), got "
                f"{self.plan_learning!r}"
            )
        if not isinstance(self.branch_on, tuple) or not all(
            isinstance(n, str) for n in self.branch_on
        ):
            raise ValueError(f"branch_on must be a tuple of modality names, got {self.branch_on!r}")
        for name in self.branch_on:
            self._refuse_emission(name, self._branch_on_label)
        if self.sophisticated is not None:
            unknown = [n for n in self.branch_on if n not in self.modality_names]
            if unknown:
                raise ValueError(
                    f"sophisticated: {[f'obs:{n}' for n in unknown]} name(s) no modality; an "
                    f"'obs:<modality>' key names one of the modalities {list(self.modality_names)}"
                )
        if self.branch_on and self.planner is not UNSET and not planner_branches(self.planner):
            raise ValueError(
                f"{self._branch_on_label}={self.branch_on!r} names modalities to branch on, "
                "which only a "
                f'branching planner does (the "sophisticated" one, or one registered with '
                f"branching=True); this System plans with {self.planner!r}"
            )
        if self.planning_belief is not UNSET and self.planning_belief not in ("prior", "posterior"):
            raise ValueError(
                f'planning_belief must be "prior" or "posterior", got {self.planning_belief!r}'
            )
        if self.planning_belief == "posterior" and self.branch_on:
            raise ValueError(
                'planning_belief="posterior" conditions the plan on expected observations, and '
                "branch_on already branches the plan on their outcomes; declare one of the two"
            )
        if (
            self.feedback is not None
            and self.feedback is not UNSET
            and self.feedback not in _closed_loop.FEEDBACK
        ):
            raise ValueError(
                f"feedback must be None or one of {list(_closed_loop.FEEDBACK)}, got "
                f"{self.feedback!r}"
            )
        if (
            self.feedback is not None
            and self.feedback is not UNSET
            and self.planning_belief is not UNSET
            and self.planning_belief != "posterior"
        ):
            raise ValueError(
                f"feedback={self.feedback!r} scores targets on the closed-loop state distribution, "
                "whose estimate spread the epistemic chain gives, and planning_belief="
                f"{self.planning_belief!r} carries none (the score would silently be the "
                'open-loop cross-entropy); set planning_belief="posterior"'
            )
        self._table  # noqa: B018  (validates the control nodes)
        self._check_observes()
        self._check_readings()
        self._check_C_deps()
        self._check_cells()
        self._check_black_box()

    def _check_black_box(self) -> None:
        """``black_box``: each entry a port and its type, ``(LEVEL, levels)`` or ``(FLOAT,
        width)`` with a positive size; an in-port a control node with no action node that no
        transition, modality, emission, target or reading names, an out-port an emission whose
        kernel is ``"black_box"``; and every such emission declared here. A System with no
        components that declares one is a black box: it has no state factors or modalities, and
        its every control node and emission is a port."""
        entries = self.black_box
        if not isinstance(entries, tuple) or not all(
            isinstance(e, tuple)
            and len(e) == 2
            and isinstance(e[0], str)
            and isinstance(e[1], tuple)
            and len(e[1]) == 2
            for e in entries
        ):
            raise TypeError(
                "black_box maps each port of a black box to its type, (LEVEL, levels) or (FLOAT, "
                f"width), got {entries!r}"
            )
        names = [name for name, _ in entries]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(f"black_box names {repeated} twice: one type per port")
        for name, (kind, size) in entries:
            if kind not in (LEVEL, FLOAT) or isinstance(size, bool) or not isinstance(size, int):
                raise ValueError(
                    f"black_box[{name!r}] is ({kind!r}, {size!r}); a port's type is (LEVEL, "
                    "levels) or (FLOAT, width), with an integer size"
                )
            if size < 1:
                raise ValueError(
                    f"black_box[{name!r}] has size {size}; a port has at least one level or one "
                    "dimension"
                )
        table = self._table
        emissions = dict(self.emissions)
        boxed = {n for n, e in emissions.items() if e.kernel == BLACK_BOX}
        undeclared = sorted(boxed - set(names))
        if undeclared:
            raise ValueError(
                f"emission(s) {undeclared} have the kernel {BLACK_BOX!r}, an out-port of a black "
                "box, and black_box gives them no type; declare each one's type there"
            )
        readers = {
            *(table.control_names[k] for k, rs in enumerate(table.readers) if rs),
            *(p for e in emissions.values() for p in e.parents),
            *(p for _, parents in self.C_deps for p in parents),
            *(node for _, node in self.readings),
        }
        for name in names:
            if name in boxed:
                continue
            if name not in table.control_names:
                raise ValueError(
                    f"black_box names {name!r}, which is neither a control node (an in-port) nor "
                    f"an emission with the kernel {BLACK_BOX!r} (an out-port) of this System"
                )
            k = table.control_names.index(name)
            if self.action_names[k] is not None or name in readers:
                raise ValueError(
                    f"black_box names the control node {name!r} as an in-port, and "
                    + (
                        f"it has the action node {self.action_names[k]!r}"
                        if self.action_names[k] is not None
                        else "a declaration of the System reads it"
                    )
                    + ": an in-port of a black box is read by nothing the System states and "
                    "realised by nothing, so declare it with no action node and no reader"
                )
        if self.subsystems is None and entries:
            stated = [
                *table.state_names,
                *self.modality_names,
                *(u for u in table.control_names if u not in names),
                *(e for e in emissions if e not in boxed),
            ]
            if stated:
                raise ValueError(
                    f"this System declares the black-box ports {names} and also {stated}: a "
                    "black box states nothing but its ports, so declare those in a System of "
                    "their own and couple the two"
                )

    @property
    def _black_boxes(self) -> tuple[str, ...]:
        """The black boxes this System holds, by component name: ``("",)`` for a black box
        itself, and for a composite each leaf that owns a port of ``black_box``."""
        if not self.black_box:
            return ()
        if not self.subsystems:
            return ("",)
        ports = [name for name, _ in self.black_box]
        sep = keyed.NAMESPACE_SEPARATOR
        return tuple(
            name
            for name, sub in self.subsystems
            if sub.config and any(p.startswith(f"{name}{sep}") for p in ports)
        )

    @property
    def _only_ports(self) -> bool:
        """Whether this System states nothing but black-box ports: no state factor, no modality,
        and every control node and emission a port of ``black_box`` (a black box, a view of one,
        or a coupling or merge of black boxes alone)."""
        if not self.black_box:
            return False
        boxed = dict(self.black_box)
        table = self._table
        return (
            not table.state_names
            and not self.modality_names
            and all(u in boxed for u in table.control_names)
            and all(e in boxed for e, _ in self.emissions)
        )

    def _refuse_black_box(self, face: str) -> None:
        """Refuse ``face`` on a System that is or holds a black box, whose inside it would need."""
        boxes = self._black_boxes
        if not boxes:
            return
        which = "is a black box" if boxes == ("",) else f"holds the black box(es) {list(boxes)}"
        raise ValueError(
            f"{face}: this System {which}, an external process declared by its ports alone, "
            f"and {face} needs what lies between the ports, which is not stated. Under run a "
            "black box is stepped by a custom step of its own; where a face needs the inside, "
            "declare the process as a System of factors and kernels in its place"
        )

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
        unknown = sorted(set(factors) - set(self.node_names))
        if unknown:
            raise ValueError(
                f"dims names unknown factor(s) {unknown}; the factors are {sorted(self.node_names)}"
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
            self._refuse_emission(key, "C_deps")
            for p in parents:
                self._refuse_emission(p, f"C_deps[{key!r}]")
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

    def _plannable(self, face: str, params: "SystemParams | None" = None) -> None:
        """Raise unless this System states one planning problem. A coupling is run leaf by leaf and
        is planned as one System only through :func:`~hybrid_aif.composition.joint.joint_planner`:
        a leaf keeps its own habit, precision, durations or scoring in ``params.parts``, or the
        leaves differ on a planner field (:data:`UNSET`), or several top-level components are
        coupled. No planner chooses a control node that has no action node (an input, driven
        from outside)."""
        self._refuse_black_box(face)
        coupled = sum(sub.boundary == "coupled" for _, sub in self.subsystems or ()) > 1
        unset = [f for f in _PLANNER_FIELDS if getattr(self, f) is UNSET]
        if unset or coupled or (params is not None and params.parts):
            raise ValueError(
                f"{face}: this System is a coupling (or a merge whose leaves keep their own "
                "planner settings), and a coupling is run leaf by leaf: it is not planned as "
                "one System. "
                "Plan one leaf's view, subsystem(system, params, leaf), or plan the composite as "
                "one mind with joint_planner(system, params, node)"
            )
        driven = [
            self.node_names[c]
            for c, a in zip(self.controls, self.action_names or (), strict=True)
            if a is None
        ]
        if driven:
            raise ValueError(
                f"{face}: control node(s) {driven} are driven (no action node chooses them), so "
                "no planner chooses them; slice them with subsystem(..., inputs={variable: "
                "level}) and plan the view, or collapse each into its driver's node with "
                "joint_planner(system, params, node)"
            )

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
        named = [a for a in actions if a is not None]
        if len(set(named)) != len(named):
            dup = sorted({a for a in named if named.count(a) > 1})
            raise ValueError(f"action_names repeats {dup}: every action node needs its own name")
        if not all(a is None or isinstance(a, str) for a in actions):
            raise TypeError(
                f"action_names must be a tuple of str or None, got {self.action_names!r}"
            )
        driven = {slot for slot, _ in self.observes if isinstance(slot, str)}
        for c, a in zip(self.controls, actions, strict=True):
            node = self.node_names[c]
            if node in driven and a is not None:
                raise ValueError(
                    f"control node {node!r} is a slot in observes, driven by a variable, and its "
                    f"action node {a!r} is named: a driven node's action_names entry is None "
                    "(nothing chooses it, so nothing realises it)"
                )
        clash = sorted(set(named) & (set(self.node_names) | set(self.modality_names)))
        if clash:
            raise ValueError(
                f"action_names {clash} are also factor or modality names: every node needs its "
                "own name"
            )
        empty = [a for a in named if not a.rpartition(keyed.NAMESPACE_SEPARATOR)[2]]
        if empty:
            raise ValueError(f"action_names {empty} have an empty name: name every action node")

    def _check_emissions(self) -> None:
        """``emissions``: pairs of a name and an :class:`~hybrid_aif.reads.Emission`, each name
        unique in the System's one namespace, each parent a state factor or a control node, each
        realisation a declared one, each static kernel a :class:`~hybrid_aif.cells.Grid` over one
        state factor. The kernels in the params are checked with them."""
        emissions = self.emissions
        if not isinstance(emissions, tuple) or not all(
            isinstance(entry, tuple)
            and len(entry) == 2
            and isinstance(entry[0], str)
            and isinstance(entry[1], Emission)
            and isinstance(entry[1].parents, tuple)
            and all(isinstance(p, str) for p in entry[1].parents)
            for entry in emissions
        ):
            raise TypeError(
                "emissions must map an emission's name to a tuple of parent names or an Emission "
                f"(parents, realise, kernel), got {emissions!r}"
            )
        if not emissions:
            return
        names = [name for name, _ in emissions]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(f"emissions repeats {repeated}: every emission needs its own name")
        table = self._table
        taken = {
            "a state factor or control node": set(self.node_names),
            "a modality": set(self.modality_names),
            "an action node": set(self.action_names or ()),
        }
        for name in names:
            for what, used in taken.items():
                if name in used:
                    raise ValueError(
                        f"the emission {name!r} is also {what}'s name: a System has one "
                        "namespace of variables, so name the emission apart"
                    )
            if not name.rpartition(keyed.NAMESPACE_SEPARATOR)[2]:
                raise ValueError(f"the emission {name!r} has an empty name: name every emission")
        for name, emit in emissions:
            if emit.realise not in REALISATIONS:
                raise ValueError(
                    f"emission {name!r}: realise must be one of {list(REALISATIONS)}, got "
                    f"{emit.realise!r}"
                )
            if emit.kernel == BLACK_BOX:
                if emit.parents:
                    raise ValueError(
                        f"emission {name!r} is an out-port of a black box (kernel "
                        f"{BLACK_BOX!r}) and names parents {list(emit.parents)}: nothing inside a "
                        "black box is stated, so its out-ports have none; declare the kernel, or "
                        "drop the parents"
                    )
                continue
            if emit.kernel is not None and not (
                emit.kernel == "identity" or isinstance(emit.kernel, Grid)
            ):
                raise ValueError(
                    f"emission {name!r}: a static kernel is 'identity' or a Grid ({BLACK_BOX!r} "
                    f"marks an out-port of a black box), got {emit.kernel!r}; kernel=None means "
                    "the kernel is in params.emission, and any other kernel is declared there"
                )
            if emit.kernel is not None and len(emit.parents) != 1:
                raise ValueError(
                    f"emission {name!r}: a static kernel (the identity or an indicator) reads "
                    f"one state factor, and the emission names {len(emit.parents)} parents"
                )
            if not emit.parents:
                raise ValueError(
                    f"emission {name!r} names no parent: an emission is drawn from state "
                    "factors (and control nodes), so name at least one"
                )
            dup = sorted({p for p in emit.parents if emit.parents.count(p) > 1})
            if dup:
                raise ValueError(f"emission {name!r} names {dup} twice among its parents")
            for p in emit.parents:
                if p in table.state_names or p in table.control_names:
                    if emit.kernel is not None and p not in table.state_names:
                        raise ValueError(
                            f"emission {name!r}: a static kernel reads a state factor, and "
                            f"{p!r} is a control node"
                        )
                    continue
                if p in names:
                    why = "an emission, an output that nothing reads"
                elif p in self.modality_names:
                    why = "a modality, an observation the System receives"
                elif p in (self.action_names or ()):
                    why = (
                        "an action node; an emission reads the control node "
                        f"{table.control_names[self.action_names.index(p)]!r} for the level taken"
                    )
                else:
                    why = "not a node of this System"
                raise ValueError(
                    f"emission {name!r} names {p!r} as a parent, which is {why}; its parents are "
                    f"state factors {list(table.state_names)} or control nodes "
                    f"{list(table.control_names)}"
                )

    @property
    def _branch_on_label(self) -> str:
        """``branch_on`` as a refusal names it: with where it was declared, when through
        ``sophisticated=``."""
        if self.sophisticated is None:
            return "branch_on"
        return "branch_on (declared through sophisticated=)"

    @property
    def _plan_learning_label(self) -> str:
        """``plan_learning`` as a refusal names it (see :attr:`_branch_on_label`)."""
        if self.sophisticated is None:
            return "plan_learning"
        return "plan_learning (declared through sophisticated=)"

    def _refuse_emission(self, name: str, where: str) -> None:
        """Refuse ``name`` at ``where`` when it is an emission: nothing in the model reads or
        scores an output."""
        if name in dict(self.emissions):
            raise ValueError(
                f"{where} names {name!r}, an emission: an emission is an output of the System, "
                "and nothing in the model reads or scores it"
            )

    @property
    def _table(self) -> ControlTable:
        """The control table every consumer reads (derived, cached on the declaration)."""
        return control_table(
            self.B_deps,
            self.A_deps,
            self.node_names,
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
        mods = self.modality_names
        targets = []
        for key, parents in self.C_deps:
            target = (params.obs_targets if key in mods else params.state_targets).get(key)
            if target is not None:
                targets.append((key, parents, target))
        return control_space(
            self._table,
            params.transition,
            params.control,
            self._facs,
            targets=targets,
            observations=tuple(zip(mods, self.A_deps, params.likelihood, strict=True)),
            durations={
                **{n: v for part in params.parts.values() for n, v in part.durations.items()},
                **params.durations,  # the composite's own value is authoritative
            },
            emissions=self._emission_readers(params),
            declared={
                name: kind_size
                for name, kind_size in self.black_box
                if name in self._table.control_names
            },
        )

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
            try:
                concrete = np.asarray(value)  # raises on a tracer
            except jax.errors.TracerArrayConversionError:
                concrete = None
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
            if concrete is not None and not bool(np.all(concrete > 0)):
                raise ValueError(
                    f"{face}: params.durations[{node!r}] = {concrete.tolist()} has a "
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
        self._plannable("elapsed", params)
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

    # ----------------------------------------------------------------------- emissions

    @property
    def _emission_names(self) -> tuple[str, ...]:
        """The emissions' names, in declaration order."""
        return tuple(name for name, _ in self.emissions)

    def _kernelled(self, params: SystemParams) -> tuple[tuple[str, Emission], ...]:
        """The emissions whose kernel is in ``params.emission`` (declared with ``kernel=None``),
        in declaration order: every emission but the static ones (``kernel="identity"``, or a
        Grid indicator)."""
        return tuple((name, emit) for name, emit in self.emissions if emit.kernel is None)

    def _emission_decls(self, params: SystemParams) -> tuple:
        """The kernels of :meth:`_kernelled` in declaration order, a ``Parameter`` reference
        replaced by the parameter variable it names; a key that names no emission, a static
        kernel that also has one in the params, and an emission with no kernel anywhere raise."""
        kernels = params.emission
        names = self._emission_names
        unknown = sorted(set(kernels) - set(names))
        if unknown:
            raise ValueError(
                f"params.emission holds kernels for {unknown}, which are not emissions of this "
                f"System; it declares {list(names)}"
            )
        static = sorted(n for n, e in self.emissions if e.kernel is not None and n in kernels)
        if static:
            raise ValueError(
                f"params.emission holds a kernel for {static}, whose declaration names a static "
                "kernel: an emission has one kernel, the declared one or the one in the params"
            )
        missing = [n for n, _ in self._kernelled(params) if n not in kernels]
        if missing:
            raise ValueError(
                f"params.emission holds kernels for {sorted(kernels)} and the System declares "
                f"the emissions {missing} with kernel=None: give one kernel per emission, keyed "
                "by its name, or declare kernel='identity' (or a Grid)"
            )
        out = []
        for name, _ in self._kernelled(params):
            decl = kernels[name]
            if isinstance(decl, Parameter):
                decl = keyed._home(params, decl.name, "emission", name)
            out.append(decl)
        return tuple(out)

    def _emission_readers(self, params: SystemParams) -> tuple:
        """``(name, parents, kernel)`` per emission that reads a control node, the parents as
        node indices with a control node ``k`` written ``len(B_deps) + k``, the convention of a
        modality that observes an action node: what sizes a control node only an emission
        reads."""
        if not set(params.emission) <= set(self._emission_names):
            return ()
        table = self._table
        n = len(self.B_deps)
        out = []
        for (name, emit), decl in zip(
            self._kernelled(params), self._emission_decls(params), strict=True
        ):
            if not any(p in table.control_names for p in emit.parents):
                continue
            deps = tuple(
                n + table.control_names.index(p)
                if p in table.control_names
                else table.state_names.index(p)
                for p in emit.parents
            )
            out.append((name, deps, decl))
        return tuple(out)

    @property
    def _emission_columns(self) -> tuple[int, ...]:
        """The control nodes (by position among them) some emission reads, ascending."""
        table = self._table
        return tuple(
            sorted(
                {
                    table.control_names.index(p)
                    for _, emit in self.emissions
                    for p in emit.parents
                    if p in table.control_names
                }
            )
        )

    def _emission_engine_deps(self, params: SystemParams) -> tuple[tuple[int, ...], ...]:
        """Each kernelled emission's parents at the forward process's positions: a state factor
        at its index, a control node after the state factors in :attr:`_emission_columns`
        order."""
        table = self._table
        s = table.num_states
        at = {k: s + i for i, k in enumerate(self._emission_columns)}
        return tuple(
            tuple(
                at[table.control_names.index(p)]
                if p in table.control_names
                else table.state_names.index(p)
                for p in emit.parents
            )
            for _, emit in self._kernelled(params)
        )

    def _static_type(self, params: SystemParams, emit: Emission, name: str) -> tuple[str, int]:
        """The port type of a static-kernel emission: a Grid's is ``(LEVEL, cells)``, and the
        identity's is its parent's, ``(LEVEL, levels)`` of a categorical factor and ``(FLOAT,
        width)`` of a continuous one; an out-port of a black box's is its declared type."""
        if emit.kernel == BLACK_BOX:
            return dict(self.black_box)[name]
        if isinstance(emit.kernel, Grid):
            return LEVEL, emit.kernel.n_cells
        (parent,) = emit.parents
        if parent in self._table.control_names:
            raise ValueError(
                f"emission {name!r} is the identity of the control node {parent!r}: its action "
                "node is that port, and an emission is the identity of a state factor"
            )
        f = self._table.state_names.index(parent)
        if self._families(params)[f] == _hybrid.CATEGORICAL:
            return LEVEL, int(self._factor_levels(params, f))
        return FLOAT, int(selectors._factor_dim(params, f))

    def _emission_types(self, params: SystemParams) -> dict[str, tuple[str, int]]:
        """Each emission's port type, by name: read off its kernel as a modality's is, or the
        static kernel's (:meth:`_static_type`)."""
        params = self._p(params)
        kernelled = {
            name: _observation_type(decl, f"params.emission[{name!r}]")
            for (name, _), decl in zip(
                self._kernelled(params), self._emission_decls(params), strict=True
            )
        }
        return {
            name: kernelled[name] if name in kernelled else self._static_type(params, emit, name)
            for name, emit in self.emissions
        }

    def ports(self, params: SystemParams) -> "Ports":
        """The System's ports with their types, by name.

        ``ins`` are the in-ports: each modality, typed by its likelihood, and each control node
        under its own name (a node the System chooses is listed as well, since a link into it
        makes it driven), typed ``(LEVEL, levels)`` or ``(FLOAT, width)``, except the slots
        ``System.observes`` names: a slot that observes a variable of the System is no in-port,
        since the System feeds it. A control node nothing drives is listed even where its own
        planner chooses it, since it can be driven, so empty ``ins`` is not what closedness means:
        that is a property of the modalities alone. ``outs`` are the out-ports: each action node,
        typed as its control node, and each emission, typed by its kernel (the identity's is its
        parent's, a Grid's is a level per cell). A type is ``(LEVEL, levels)`` for a table and
        ``(FLOAT, dimension)`` for a linear-Gaussian or neural kernel. There is no node index for a
        port: a port is its name.
        """
        params = self._params_in(params, "ports")
        ins = dict(self._modality_types(params))
        outs: dict[str, tuple[str, int]] = {}
        if self._table.controls:
            space = self._space(params)
            for k, (u, a) in enumerate(
                zip(self._table.control_names, self.action_names, strict=True)
            ):
                ins[u] = (space.kinds[k], int(space.sizes[k]))
                if a is not None:
                    outs[a] = (space.kinds[k], int(space.sizes[k]))
        outs.update(self._emission_types(params))
        fed = set(dict(self.observes))
        return Ports({slot: t for slot, t in ins.items() if slot not in fed}, outs)

    # ------------------------------------------------------------ names at the faces

    def _names(self, params: SystemParams | None = None) -> keyed.FaceNames:
        """The names the conversions read; with ``params``, each control node's kind and size
        too (a continuous node's box lives in ``params.control``)."""
        table = self._table
        if table.sugar is not None:
            owners = table.sugar.owners
        elif all(table.readers):  # each node at its first reader's column, if no two share one
            owners = tuple(readers[0][0] for readers in table.readers)
            owners = owners if len(set(owners)) == len(owners) else None
        else:  # a node no factor reads has no column
            owners = None
        gone = self._bound_away()
        generated = (*(a for a in self.action_names or () if a is not None), *self._emission_names)
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
                emissions=self._emission_names,
                observes=self.observes,
                generated=generated,
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
            self._emission_names,
            self.observes,
            generated,
        )

    def _bound_away(self, prefix: str = "") -> tuple[tuple[str, str], ...]:
        """Each control node, and its action node, a ``merge()`` binding removed, by name, with
        the rest of the sentence that refuses a mapping keyed by it (what the node became), read
        off ``subsystems``. ``prefix`` qualifies every name, for a merge that takes this one as a
        component."""
        if not self.subsystems:
            return ()
        return _bound_away(self.subsystems, self._table.control_names, self.action_names, prefix)

    @property
    def names(self) -> keyed.Names:
        """A read-only view of the System's names, every tuple in node-index order.

        ``names.states`` are the state factors' names (the belief keys, ``state_names``),
        ``names.controls`` the control nodes' (``control_names``); ``names.modalities`` is
        ``modality_names``;
        ``names.actions`` the action nodes' (the action keys), one per control node in the same
        order, from ``action_names``. Node-index order is the order of every positional
        structure (tables, policy columns, what the converters produce); iterating a mapping
        gives no such guarantee, so take order from ``names``.
        """
        return self._names().public

    def _node_priors(self, params: SystemParams) -> dict[str, Categorical]:
        """Each discrete control node's prior, by name: uniform over its levels (C2's
        independence at every step), in :func:`table_dtype` of ``params``."""
        names = self._names(params)
        if not names.node_beliefs:
            return {}
        levels = self._control_levels(params)
        dtype = table_dtype(params)
        return {
            n: Categorical(logits=jnp.zeros((levels[k],), dtype))
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
        self,
        params: SystemParams,
        given: Mapping[str, Belief] | None,
        face: str,
    ) -> tuple[Categorical, ...]:
        """Each observed action node's prior, in column order: the entry ``given`` carries, else
        uniform over its control node's levels (in :func:`table_dtype` of ``params``)."""
        self._check_sensed_kinds(params, face)
        levels = self._control_levels(params)
        dtype = table_dtype(params)
        out = []
        for k, name in zip(self._sensed_columns, self._sensed, strict=True):
            q = None if given is None else given.get(name)
            if q is None:
                out.append(Categorical(logits=jnp.zeros((levels[k],), dtype)))
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
            dtype=table_dtype(params),
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
        """An amortized model's filter states as the beliefs the faces take, keyed by state factor.

        ``states`` is an :class:`~hybrid_aif.inference.rssm.RSSMState` (e.g.
        :func:`~hybrid_aif.train.world.posterior_states`): the carry's vector and the code's
        tensor, by state factor name, leading batch and time axes passing through (flatten them for
        :func:`~hybrid_aif.train.imagine.imagine`'s ``(N, ...)`` start beliefs).
        """
        from hybrid_aif.train.world import belief_tuple

        self._refuse_black_box("beliefs_from_states")
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
        state factor order (a control node's entry is left out); ``"observations"`` → a tuple in
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
        convert a continuous node's value, whose box lives in ``params.control``. For params
        with parameter variables, the positional form holds a variable's table at every reader's
        position; a reader position that differs from the variable's entry raises, and the edit
        is made at ``parameters[<name>]`` in the keyed form. That check compares concrete arrays
        only: under ``jit`` the arrays are traced and it cannot run, so an edit made at a reader
        position is not detected there.
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
        :class:`~hybrid_aif.distributions.ConditionalCategorical` over the designated state factor,
        conditional on parent state factors (under ``merge`` possibly a higher System's). It
        resolves mean-field, as ``D_deps`` does in :meth:`initial_priors` — ``Σ_parents h(· |
        parents) q(parents)`` — with ``q(parents)`` read from the final predicted beliefs of the
        rollout being scored, so a parent the plan can steer makes the target plan-dependent.
        Learnable and continuous conditional targets raise.

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

    def compiled(self, *, watch: bool = True) -> "CompiledSystem":
        """This System's faces compiled once, behind input canonicalisation.

        Returns a :class:`~hybrid_aif.system.compiled.CompiledSystem` with the same face methods
        as the System. ``watch=False`` turns off the retrace warning.
        """
        from hybrid_aif.system.compiled import CompiledSystem

        return CompiledSystem(self, watch=watch)

    def initial_priors(self, params: SystemParams) -> dict[str, Belief]:
        """Ancestral mean-field marginals of the initial priors, ``D_deps`` resolved, by state
        factor.

        A conditional prior is averaged over its parents' resolved marginals in declaration order,
        unconditional priors pass through, and a learnable prior contributes its posterior-mean
        ``point()``. ``infer_states`` uses this when ``priors=None``; planning faces take it in
        place of a raw conditional ``D``. On the amortized path it returns the episode-start values:
        each carry's learned initial value, and each code's transition read at it.

        An action node a modality observes gets an entry too, uniform over its levels (no action
        precedes the first step), so the mapping has the entries :meth:`infer_states` and
        :meth:`update_empirical_prior` return.
        """
        self._refuse_black_box("initial_priors")
        params = self._params_in(params, "initial_priors")
        sensed = None
        if self._sensed_columns:
            priors = self._sensed_priors(params, None, "initial_priors")
            sensed = dict(zip(self._sensed, priors, strict=True))
        return self._beliefs_out(
            params, self._initial_priors(params, "initial_priors"), sensed=sensed
        )

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
        #  in the dtype the first tick's posterior takes (the widest of the model's tables and
        #  initial priors), so a filter over a model of mixed dtypes keeps one dtype from the start
        dtype = jnp.promote_types(table_dtype(params), tree_float_dtype(out))
        return tuple(widened(b, dtype) for b in out)

    def _families(self, params: SystemParams) -> tuple[str, ...]:
        params = self._p(params)
        return selectors._families(self, params)

    # -------------------------------------------------------------- forward face

    def as_env(
        self, params: SystemParams, param_key: Array | None = None
    ) -> tuple[GenerativeEnv, GenerativeEnvParams]:
        """The same tensors run forward: ``(env, env_params)``.

        ``env`` is a :class:`~hybrid_aif.envs.GenerativeEnv` (observations by modality name,
        states by state factor name, actions by action-node name) over the discrete process
        (probability tables) for an all-tabular System, the hybrid one (the declarations
        themselves) otherwise; ``env_params`` keys its containers by name and carries
        ``params.actuator``, from which the env realises each action it is given.
        Learnable slots are lowered to their posterior mean, or with ``param_key`` to one
        ``sample`` each, so the env is a draw from the model's own parameter prior. Raises on the
        amortized path. It is :func:`~hybrid_aif.system.mirror.mirror` (each modality an
        emission), lowered, run forward by the positional process, with the env's faces keyed by
        this System's names; a System that declares emissions has no such mirror and is refused.
        """
        self._refuse_black_box("as_env")
        from hybrid_aif.system.mirror import mirror

        ms, mp = mirror(self, params, lower=True, param_key=param_key, face="as_env")
        process, pp = ms._forward(mp)
        return GenerativeEnv(process, self._names(params)), self._params_out(
            dataclasses.replace(pp, actuator=self._actuator(params, "as_env"))
        )

    def _forward(self, params: SystemParams) -> tuple:
        """This System run forward as a positional process: ``(process, process params)``.

        ``params`` are keyed and lowered (no learnable declaration, no parameter reference). The
        process is the discrete one, over probability tables, when every transition, every
        initial prior and every emission is tabular, else the hybrid one over the declarations
        themselves. Its observation side is the emissions, in declaration order, a control parent
        read as a sensed action column; its transitions, priors and wiring are the System's own.
        """
        p = self._p(params)
        decls = self._emission_decls(p)
        for what, entries in (
            ("emission", decls),
            ("transition", p.transition),
            ("initial_state_prior", p.initial_state_prior),
        ):
            for decl in entries:
                if _slot_learnable(decl) and not isinstance(decl, MLPConditional):
                    raise ValueError(
                        f"a forward process takes point declarations, and a {what} is a "
                        f"{type(decl).__name__}: lower the learnable declarations first "
                        "(mirror(..., lower=True))"
                    )
        table = self._table
        if (
            any(not isinstance(a, ConditionalCategorical) for a in decls)
            or any(not isinstance(b, TabularTransition) for b in p.transition)
            or any(not isinstance(d, TabularPrior) for d in p.initial_state_prior)
        ):
            space = self._space(p)
            process = HybridProcess(
                A_deps=self._emission_engine_deps(p),
                B_deps=self._B,
                controlled=tuple(bool(c) for c in table.control_parents),
                D_deps=self._effective_D_deps(),
                controls=None if space.layout is None else space.layout.structure(),
                B_control_deps=table,
                kinds=space.kinds,
                sizes=space.sizes,
                sensed=self._emission_columns,
            )
            return process, GenerativeEnvParams(
                likelihood=decls,
                transition=p.transition,
                initial_state_prior=p.initial_state_prior,
            )
        process = DiscreteProcess(
            A_deps=self._emission_engine_deps(p),
            B_deps=self._B,
            D_deps=self._effective_D_deps(),
            B_control_deps=table,
            sensed=self._emission_columns,
        )
        return process, GenerativeEnvParams(
            likelihood=tuple(jnp.exp(a.log_probs) for a in decls),
            transition=tuple(jnp.exp(b.log_probs) for b in p.transition),
            initial_state_prior=tuple(
                d.probs if isinstance(d, Categorical) else jnp.exp(d.log_probs)
                for d in p.initial_state_prior
            ),
        )

    def _forward_env(self, params: SystemParams) -> tuple[GenerativeEnv, GenerativeEnvParams]:
        """This System's forward process behind name-keyed faces, observations being the
        emissions: ``(env, env_params)`` over the learnable declarations at their posterior
        mean, parameter variables dissolved into the tables their readers hold."""
        params = self._params_in(params, "sample_step")
        lowered = self._lower_learnable(params, None)
        lowered = dataclasses.replace(lowered, parameters={}, references=(), noise_references=())
        process, pp = self._forward(lowered)
        names = self._names(params)._replace(
            modalities=tuple(name for name, _ in self._kernelled(lowered)),
            #  the action keys of the forward process: a driven node, having no action node, is
            #  keyed by its control node
            actions=tuple(
                u if a is None else a
                for u, a in zip(self._table.control_names, self.action_names, strict=True)
            ),
            sensed=(),
            observes=(),
            generated=(),
        )
        return GenerativeEnv(process, names), keyed.params_keyed(names, pp)

    def _free_controls(
        self, key: Array, params: SystemParams, driven: Iterable[str]
    ) -> dict[str, Array]:
        """A value for every control node not in ``driven``, drawn from
        ``fold_in(key, CONTROL_TAG)``. Without a habit a discrete node is uniform over its levels
        and a continuous one uniform within its box. With ``policy_prior`` (a distribution over
        this System's enumerated first-step level tuples) one tuple is drawn and the free nodes
        take its entries; a habit that is a policy module, that meets continuous nodes or that
        does not match the enumerated set is refused."""
        table = self._table
        free = [k for k, u in enumerate(table.control_names) if u not in set(driven)]
        if not free:
            return {}
        space = self._space(params)
        keys = jax.random.split(jax.random.fold_in(key, CONTROL_TAG), len(table.control_names))
        habit = params.policy_prior
        tuple_ = None
        if habit is not None and any(space.kinds[k] == LEVEL for k in free):
            if not (isinstance(habit, Categorical) or hasattr(habit, "shape")):
                raise NotImplementedError(
                    "a free control node's habit is a policy module, which reads beliefs a world "
                    "does not hold; drive the node, or declare policy_prior as a distribution "
                    "over the enumerated first-step level tuples"
                )
            if FLOAT in space.kinds:
                raise NotImplementedError(
                    "a free control node's habit is a distribution over enumerated level "
                    "tuples, and this System has continuous control nodes, which have no "
                    "enumerated set; drive the discrete nodes or drop the habit"
                )
            logits = habit.logits if isinstance(habit, Categorical) else habit
            first = self.policies(params)[:, 0, :]
            if jnp.shape(logits) != (first.shape[0],):
                raise ValueError(
                    f"a free control node's habit has shape {tuple(jnp.shape(logits))} and this "
                    f"System enumerates {first.shape[0]} first-step level tuples; declare one "
                    "entry per tuple"
                )
            tuple_ = first[jax.random.categorical(keys[0], logits)]
        out = {}
        for k in free:
            if space.kinds[k] == FLOAT:
                box = space.boxes[k]
                dtype = jnp.result_type(box.low, box.high)  # drawn in the box's dtype
                unit = jax.random.uniform(keys[k], (space.sizes[k],), dtype)
                out[table.control_names[k]] = box.low + (box.high - box.low) * unit
            elif tuple_ is not None:
                out[table.control_names[k]] = tuple_[k].astype(jnp.int32)
            else:
                levels = space.sizes[k]
                out[table.control_names[k]] = jax.random.categorical(
                    keys[k], jnp.log(jnp.full((levels,), 1.0 / levels))
                ).astype(jnp.int32)
        return out

    def _unsampled(
        self, key: Array, params: SystemParams, state: Mapping[str, Array], controls: Mapping
    ) -> dict[str, object]:
        """The emissions the forward process does not draw: those realised by ``mode`` or
        ``stats``, and the static-kernel ones (the identity and indicator kernels), from
        ``state`` and the control nodes' values."""
        left = [n for n, e in self.emissions if e.realise != "sample" or e.kernel is not None]
        if not left:
            return {}
        values = {**state, **controls}
        emitted = self.emit(jax.random.fold_in(key, _emissions.PARENT_TAG), params, values)
        return {n: emitted[n] for n in left}

    def sample_initial(
        self,
        key: Array,
        params: SystemParams,
        state: Mapping[str, Array] | None = None,
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        """Draw an initial state (or take ``state``, one point per state factor) and the emissions
        of it: ``(emitted, state)``, both keyed by name. The forward process's reset, over the
        emissions in place of modalities, each emission realised by its declared rule; a control
        node an emission reads is at level 0 (the zero vector when continuous)."""
        self._refuse_black_box("sample_initial")
        env, ep = self._forward_env(params)
        emitted, state = env.reset(key, ep, state)
        space = self._space(self._params_in(params, "sample_initial"))
        null = {
            u: jnp.zeros((), jnp.int32) if space.kinds[k] == LEVEL else jnp.zeros(space.sizes[k])
            for k, u in enumerate(self._table.control_names)
        }
        return {**emitted, **self._unsampled(key, params, state, null)}, state

    def sample_step(
        self,
        key: Array,
        params: SystemParams,
        state: Mapping[str, Array],
        inputs: Mapping[str, Array] | None = None,
        states: Iterable[str] | None = None,
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        """One round forward from ``state``: ``(emitted, next state)``, keyed by name.

        ``inputs`` gives the value of each control node an input drives (a level, or a float
        vector); a control node it does not name is free and is drawn first with
        ``fold_in(key, CONTROL_TAG)`` (:meth:`_free_controls`), so the key stream of the
        process itself is the one it has when every node is driven. ``states`` names the state
        factors to step; the others keep their values. Every emission is realised by its declared
        rule. For ``mirror(s, p)`` of a System ``s`` with no actuator kernel and every control node
        driven, the emissions and the state equal ``s.as_env(p).step`` under the same key.
        """
        self._refuse_black_box("sample_step")
        inputs = dict(inputs or {})
        unknown = sorted(set(inputs) - set(self._table.control_names))
        if unknown:
            raise ValueError(
                f"sample_step: inputs names {unknown}, which are not control nodes; the control "
                f"nodes are {list(self._table.control_names)}"
            )
        env, ep = self._forward_env(params)
        controls = self._sample_controls(key, params, inputs)
        action = {
            u if a is None else a: controls[u]
            for u, a in zip(self._table.control_names, self.action_names, strict=True)
        }
        emitted, nxt = env.step(key, state, action, ep, states)
        return {**emitted, **self._unsampled(key, params, nxt, controls)}, nxt

    def _sample_controls(
        self, key: Array, params: SystemParams, inputs: Mapping[str, Array]
    ) -> dict[str, Array]:
        """The value of every control node of one sampled round: ``inputs``, a node whose driver
        is an action node of this System taking that action node's level, and each remaining node
        drawn freely (:meth:`_free_controls`). A driven node that ``inputs`` does not supply and no
        action node of this System drives is refused."""
        table = self._table
        observed = dict(self.observes)
        pairs = list(zip(table.control_names, self.action_names, strict=True))
        action_node = {a: u for u, a in pairs if a}
        internal = {
            u: action_node[observed[u]]
            for u, a in zip(table.control_names, self.action_names, strict=True)
            if a is None and u not in inputs and observed.get(u) in action_node
        }
        lone = [
            u
            for u, a in zip(table.control_names, self.action_names, strict=True)
            if a is None and u not in inputs and u not in internal
        ]
        if lone:
            raise ValueError(
                f"sample_step: control node {lone[0]!r} is driven (it has no action node) and "
                "inputs does not supply it; pass its value in inputs, keyed by its control node"
            )
        controls = {**inputs, **self._free_controls(key, params, {**inputs, **internal})}
        controls.update({u: controls[d] for u, d in internal.items()})
        return controls

    def _emission_observer(self, kernelled: tuple, decls: tuple) -> tuple:
        """A positional process whose observation side is the ``kernelled`` emissions (pairs of
        a name and its declaration, with their kernels ``decls``) over the distinct parents in
        first-use order: ``(process, process params, parents)``. Observing it at a tuple of
        parent values draws every emission with ``split(key, E)[e]``."""
        parents = tuple(dict.fromkeys(p for _, e in kernelled for p in e.parents))
        deps = tuple(tuple(parents.index(p) for p in e.parents) for _, e in kernelled)
        if all(isinstance(a, ConditionalCategorical) for a in decls):
            process = DiscreteProcess(A_deps=deps, B_deps=())
            likelihood = tuple(jnp.exp(a.log_probs) for a in decls)
        else:
            process = HybridProcess(A_deps=deps, B_deps=(), controlled=())
            likelihood = decls
        return process, GenerativeEnvParams(likelihood, (), ()), parents

    def emit(
        self, key: Array, params: SystemParams, values: Mapping[str, object]
    ) -> dict[str, object]:
        """Realise every emission from ``values``, keyed by emission name in declaration order.

        ``values`` holds each parent of an emission by name: a state factor as a point (a level,
        or a float vector: a world's state) or as a belief (an agent's posterior), a control node
        as its realised level. Each emission is realised by its own rule (``Emission.realise``)
        with ``jax.random.split(key, E)[e]`` for emission ``e`` of the ``E`` emissions that have
        a kernel in ``params.emission`` (the others draw from ``split(key, E')[e]`` over all
        ``E'``). From points, ``sample`` conditions the kernel on the parents and draws (the
        draw the forward process makes of an observation), ``mode`` is the most probable level or
        the mean, ``stats`` the probability vector or the ``(mean, covariance)``. From beliefs,
        ``sample`` draws the parents and then the kernel (an exact predictive draw), and
        ``mode`` and ``stats`` are those of the predictive distribution
        (:mod:`hybrid_aif.system.emissions`). Learnable kernels act at their posterior mean. A
        static kernel realises its parent directly: the identity gives the state factor's level,
        probability vector or location, and a :class:`~hybrid_aif.cells.Grid` gives the cell a
        point falls in, or from a belief the cell index by ``sample`` and ``mode`` and the cell
        mass by ``stats``. Nothing in inference or planning reads an emission; a modality
        observes one through ``System.observes``, its datum then supplied to the faces.
        """
        self._refuse_black_box("emit")
        params = self._params_in(params, "emit")
        if self.emissions:
            self._validate(params, "emit")
        return self._emit(key, params, values)

    def _emit(
        self, key: Array, params: SystemParams, values: Mapping[str, object]
    ) -> dict[str, object]:
        """:meth:`emit` over the positional view of the params, unvalidated: what a compiled
        loop or a scan calls with params it has already checked."""
        params = self._p(params)
        if not self.emissions:
            return {}
        names = self._emission_names
        parents = tuple(dict.fromkeys(p for _, e in self.emissions for p in e.parents))
        absent = [p for p in parents if p not in values]
        if absent:
            raise ValueError(
                f"emit: values has no entry for {absent}, which the emissions read; give each "
                f"parent of {list(names)} by name ({list(parents)})"
            )
        kernelled = self._kernelled(params)
        out: dict[str, object] = {}
        if kernelled:
            lowered = self._lower_learnable(params, None)
            decls = tuple(lowered.emission[n] for n, _ in kernelled)
            if any(e.realise == "sample" for _, e in kernelled):
                parent_keys = jax.random.split(
                    jax.random.fold_in(key, _emissions.PARENT_TAG), len(parents)
                )
                points = tuple(
                    _emissions.point_of(k, values[p])
                    for k, p in zip(parent_keys, parents, strict=True)
                )
                process, pp, _ = self._emission_observer(kernelled, decls)
                drawn = process._observe(key, points, pp)
            for i, ((name, emit), decl) in enumerate(zip(kernelled, decls, strict=True)):
                if emit.realise == "sample":
                    out[name] = drawn[i]
                else:
                    out[name] = _emissions.predictive(
                        decl, tuple(values[p] for p in emit.parents), emit.realise, name
                    )
        static = [(i, n, e) for i, (n, e) in enumerate(self.emissions) if n not in out]
        if static:
            keys = None if key is None else jax.random.split(key, len(self.emissions))
            types = None
            for i, name, emit in static:
                (parent,) = emit.parents
                if isinstance(emit.kernel, Grid):
                    out[name] = _emissions.indicator(
                        values[parent],
                        emit.kernel,
                        emit.realise,
                        None if keys is None else keys[i],
                        name,
                    )
                    continue
                if emit.realise == "stats" and not _emissions.is_belief(values[parent]):
                    types = types or self._emission_types(params)
                out[name] = _emissions.identity(
                    values[parent],
                    emit.realise,
                    None if keys is None else keys[i],
                    name,
                    None if types is None else types[name][1],
                )
        return {name: out[name] for name in names}

    def _lower_learnable(self, params: SystemParams, param_key: Array | None) -> SystemParams:
        """``params`` (positional view) with every learnable declaration at its point form or one
        draw; the emissions' kernels join the lowering after the likelihood, a parameter variable
        keeping one draw for all its readers, and come back keyed by name with the references
        dissolved."""
        params = self._p(params)
        if not params.emission:
            return conjugate_stats._lower_learnable(params, param_key)
        names = tuple(name for name, _ in self._kernelled(params))
        decls = self._emission_decls(params)
        read = tuple(
            ("emission", i, params.emission[name].name)
            for i, name in enumerate(names)
            if isinstance(params.emission[name], Parameter)
        )
        staged = dataclasses.replace(
            params, emission=decls, references=tuple(params.references) + read
        )
        lowered = conjugate_stats._lower_learnable(staged, param_key)
        return dataclasses.replace(
            lowered,
            emission=dict(zip(names, lowered.emission, strict=True)),
            references=tuple(r for r in lowered.references if r[0] != "emission"),
        )

    # ------------------------------------------------------------- backward faces

    @_eager_face
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
    ) -> tuple[dict[str, Belief], InferenceDetail | None]:
        """Mean-field state posteriors for one tick's observations: ``(beliefs, detail)``.

        ``observations`` is keyed by modality name, one entry per modality: an ``int32`` level, a
        one-hot or a probability vector over the levels (a soft observation: the expected
        log-likelihood under the vector) for a categorical modality, a float array for an LG
        one. Design: nesting-and-coupling.md N8. ``priors`` and the
        returned beliefs are keyed by state factor name; ``priors=None`` uses
        :meth:`initial_priors`. A discrete control node's entry is optional in ``priors`` and always
        returned: no inference runs on it, so it passes through (its uniform prior when absent). An
        action node a modality observes (``A_deps`` naming it) is inferred like a state factor on
        the tabular path: its entry in ``priors`` is its prior (uniform when absent), and its
        posterior is returned under its name.
        With ``detail=True`` (a static bool) the second slot is an
        :class:`~hybrid_aif.inference.InferenceDetail` (pre-reduction switching mixtures, per-mode
        and total log-evidence keyed by modality name; ``unsupported`` keyed by state factor name),
        else ``None``; the beliefs are identical either way.

        ``present`` says which modalities are observed this tick, a flag per modality keyed by
        name: a modality it leaves out is present, ``None`` means all are, and an unknown key
        raises. An absent modality is not clamped: its value is ignored and its state factors keep
        what the other modalities leave them; all absent is a predict-only tick. A traced flag
        computes the update and selects against it (its ``detail`` entries report the
        hypothetical update, ``log_evidence`` zeroed); a Python ``False`` skips it statically. On
        the amortized path the modalities the recognition modules read are present or absent
        together: static flags splitting them raise, traced ones reduce with ``all`` (absent).

        ``detail.unsupported`` flags each categorical state factor none of whose levels explains
        this tick's tabular evidence, as :func:`hybrid_aif.inference.discrete.unsupported` defines
        it against ``support_floor``; the posterior is left as computed.

        Hybrid Systems run one structured sweep in declaration order: the discrete FPI over the
        categorical sub-model; exact LG ``condition`` per LG modality, sequential in modality order;
        exact Gaussian-sum conditioning for a mixture-carried state factor, with a mode-parent
        evidence message under ``mixture_schedule="mode_aligned"``; and grow-then-reduce
        conditioning for a switching LG modality (categorical mode parents after the observed state
        factor in ``A_deps``), under either schedule. A modality that observes a mixture-carried
        state factor directly (one whose *transition* switches) raises under ``"runnalls"`` and
        needs ``"mode_aligned"``; non-LG likelihoods on continuous state factors raise.

        On the amortized path this runs the recognition modules in order
        (:func:`hybrid_aif.inference.amortized.recognize`): the value ``None`` is allowed for a
        modality no module reads (its key is still present), ``externals`` supplies
        ``"external:<name>"`` parents, and ``detail`` is ``None``.

        With ``key`` the posterior is collapsed onto one draw, as a belief of its own family
        (one-particle filtering; amortized carries pass through; on a predict-only tick it is
        the prior that collapses). See docs/composing-systems.md.

        ``observations`` and ``present`` are keyed by **variable**: a modality that
        ``System.observes`` names takes the datum of the variable it observes (an emission or an
        action node of this System), and one datum feeds every slot observing the variable; a key
        naming a slot the map renames is refused, naming the variable. The datum of an observed
        variable is optional: a slot whose variable is left out, or ``None``, is absent that tick
        (the first tick of an own-belief read has none). Design: nesting-and-coupling.md N3.
        """
        self._refuse_black_box("infer_states")
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
        )

    def _check_observes(self) -> None:
        """``observes``: pairs of a slot (a modality or a control node, named at most once) and
        a variable the System generates (an action node or an emission). A control node in
        ``observes`` is driven: the variable's realised value is its level, and it has no action
        node of its own. The port types are checked with the params."""
        observes = self.observes
        if not isinstance(observes, tuple) or not all(
            isinstance(entry, tuple) and len(entry) == 2 and all(isinstance(x, str) for x in entry)
            for entry in observes
        ):
            raise TypeError(
                f"observes maps a modality's name to the variable it "
                f"observes, a mapping of two names or (slot, variable) pairs, got {observes!r}"
            )
        if not observes:
            return
        mods, nodes = self.modality_names, self._table.control_names
        generated = {*(a for a in self.action_names or () if a is not None), *self._emission_names}
        seen: set[str] = set()
        for slot, variable in observes:
            if slot not in mods and slot not in nodes:
                raise ValueError(
                    f"observes names the slot {slot!r}, which is neither a modality {list(mods)} "
                    f"nor a control node {list(nodes)}"
                )
            if slot in seen:
                raise ValueError(
                    f"observes names the slot {slot!r} twice: a slot observes one variable"
                )
            seen.add(slot)
            if variable not in generated:
                raise ValueError(
                    f"observes: {slot!r} observes {variable!r}, which this System does not "
                    f"generate; the variables it generates are its action nodes "
                    f"{[a for a in self.action_names or () if a is not None]} and its emissions "
                    f"{list(self._emission_names)}"
                )

    @property
    def _observed(self) -> dict[str, str]:
        """``observes`` as a mapping: the variable of each slot the map names."""
        return dict(self.observes)

    @property
    def _modality_variables(self) -> tuple[str, ...]:
        """The variable each modality observes, in modality order: the one ``observes`` names,
        else the modality's own name."""
        observed = self._observed
        return tuple(observed.get(m, m) for m in self.modality_names)

    def _check_readings(self) -> None:
        if not self.readings:
            return
        table = self._table
        facs, mods, nodes = table.state_names, self.modality_names, table.control_names
        seen = set()
        for pair in self.readings:
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise ValueError(
                    "readings holds (factor or modality, control node) pairs of names, "
                    f"got {pair!r}"
                )
            name, node = pair
            self._refuse_emission(name, "readings")
            if name in facs and name in mods:
                raise ValueError(
                    f"readings names {name!r}, which is both a factor and a modality; rename one"
                )
            if name not in facs and name not in mods:
                raise ValueError(
                    f"readings names {name!r}, which is neither a state factor {list(facs)} nor "
                    f"a modality {list(mods)}"
                )
            if name in seen:
                raise ValueError(f"readings names {name!r} twice")
            seen.add(name)
            if node not in nodes:
                raise ValueError(
                    f"readings: {name!r} pools its reading over {node!r}, which is not a control "
                    f"node of this System ({list(nodes)}); a reading pools over the System's own "
                    "node"
                )

    def _pooled(self, params: SystemParams) -> "_pooling.Pooled | None":
        """The declared reading counts as the planner's per-step scales, or ``None``."""
        if not self.readings:
            return None
        params = self._p(params)
        nodes = self._table.control_names
        mods = self.modality_names
        entries = tuple(
            _pooling.PooledEntry(
                f"likelihood:{name}" if name in mods else f"transition:{name}",
                nodes.index(node),
                as_float(params.readings[name]),  # a float keeps its dtype
            )
            for name, node in self.readings
        )
        return _pooling.Pooled(entries)

    @staticmethod
    def _present(
        present: tuple[bool | Array, ...] | Array | None, n_mods: int
    ) -> tuple[bool | Array, ...] | None:
        """The engine: :func:`hybrid_aif.system.faces.infer._present`."""
        return _infer_face._present(present, n_mods)

    # -------------------------------------------------------------- learning face

    @_eager_face
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
        priors: Mapping[str, Belief] | None = None,
        externals: Mapping[str, Array] | None = None,
    ) -> "SystemParams | tuple[SystemParams, LearnState]":
        """One tick's parameter update: every learnable slot absorbs this tick's statistic.

        A slot holding a distribution over its parameters absorbs its conjugate sufficient
        statistic; point-valued slots are not learnable. A non-frozen
        :class:`~hybrid_aif.distributions.MLPConditional` takes an optax step on its negative
        log-density at point evidence from the beliefs (posterior means, probability vectors for
        discrete variables, a one-hot action). A transition net's parents are its state factor's
        previous value then the action one-hot if controlled; a likelihood net's are the ``A_deps``
        state factors in order.

        A non-frozen :class:`~hybrid_aif.distributions.MLPConditional` likelihood takes the
        gradient of the free energy ``infer_states(detail=True)`` reports for the tick: the
        inference pass is re-run from ``priors``, the prior beliefs ``infer_states`` was given
        (keyed by state factor name), with every other input held fixed, and differentiated in the
        decoder. Without ``priors`` they are the prediction from ``prev_beliefs`` and ``action``
        when ``prev_beliefs`` is given, else the initial priors. ``externals`` is the mapping
        ``infer_states`` took for a recognition entry's external parents. A modality a
        recognition entry covers contributes its expected negative log-likelihood under the final
        belief, as ``infer_states`` records it.

        A parameter variable several state factors read (``SystemParams.parameters``) absorbs the
        sum of its readers' statistics once, at the rate ``lr_by_slot["parameter:<name>"]`` (default
        ``lr``). A reader's own key (``"transition:<factor>"``, ``"likelihood:<modality>"``) takes
        only a static ``0.0``, which keeps that reader's statistic out of the variable; any other
        value raises, naming the variable's key. A reader with no observed child counts its
        predictive belief like any state factor does, which is why this exclusion exists: a
        scheduler that reads a table it never observes would otherwise add its own predictions to
        the counts. A literal key outranks a pattern, for readers as for rates: a pattern
        (``"transition:*"``, ``"*:*"``) at ``0.0`` excludes the referencing readers it matches
        only while the variable's own rate is not named literally, so ``{"*:*": 0.0,
        "parameter:T": 1.0}`` learns ``T`` from all its readers and ``{"*:*": 0.0}`` freezes
        everything; a literal ``0.0`` on a reader excludes it either way. A nonzero value from a
        pattern with a literal slot raises on a referencing reader, and a fully wildcarded
        nonzero pattern reaches the variable through ``parameter:*`` instead.

        ``observations`` is keyed by modality name, ``beliefs`` and ``prev_beliefs`` by state factor
        name, ``action`` by action-node name (as :meth:`act` returns it). ``action`` is
        **a_{t-1}**, the action that produced this tick's observations; each control node is
        clamped to its value. ``prev_beliefs`` and
        ``action`` are required exactly when a learnable transition has a nonzero-or-traced rate
        (``action`` only if that state factor is controlled). ``detail`` is the keyed
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

        **Growth.** A growable state factor (a memoryless state factor whose weights carry a growth
        propensity) runs its growth step first: the tick's data scored under every level, the
        first dead level woken when the best live level explains them worse than the prior does
        by the propensity's margin, the woken level taking the datum hard and seeded from it by
        the ordinary update. Every ``reduction_period``-th call then reduces each growable state
        factor (:meth:`reduce`), which needs ``prior``, the model's prior params (the statistics a
        merge pools are the posterior's beyond them, and a killed level returns to them), and a
        ``learn_state`` carrying the counter (:meth:`init_learn_state`). Both move with the
        weights: a tick where no modality over the state factor is present grows nothing, and the
        weights' rate (``transition:<factor>``) gates growth and reduction, a static zero
        freezing the state factor's structure and a traced zero skipping it at run time.

        ``observations`` is keyed by variable, as :meth:`infer_states` takes it: the datum of an
        emission or action node a modality observes (``System.observes``) is the one the caller
        supplied to ``infer_states``, and a slot whose datum is absent teaches its likelihood
        slot nothing that tick. Design: nesting-and-coupling.md N3.

        Design: docs/design/parameter-learning.md D1 D3 D4 D5 D7 D8;
        docs/design/per-variable-infogain.md V1 V7; docs/design/amortization.md A2;
        docs/design/structure-learning.md S3 S4.
        """
        self._refuse_black_box("learn")
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
            priors,
            externals,
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
        priors: tuple[Belief, ...] | None = None,
        externals: Mapping[str, Array] | None = None,
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
            priors,
            externals,
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
        ``"initial_state_prior:<factor>"``, ``"parameter:<name>"`` (a named parameter variable,
        the one block of every state factor and modality that reads it) or
        ``"likelihood:<modality>:noise"``, without wildcards (under ``jit``, pass ``slots`` as a
        static argument). Everything else comes from ``fresh``: point values, and a separate
        ``MatrixNormal`` head, which ``learn`` does not update. A block whose family, shape or dtype
        differs between the two raises, as do a selected slot that holds a posterior on one side
        only, a key that names no posterior, and a selection that would split a growable state
        factor's blocks, which growth and reduction keep in step. Trainable slots that ``learn``
        does not train (a recognition model, a scoring row, a policy module) are not carried; a
        gradient slot's optimizer state is the ``LearnState`` the agent already carries.

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
        state: str | None = None,
        n_pair: int | None = None,
    ) -> SystemParams:
        """Bayesian model reduction on the growable state factor ``state`` names, or on every one
        when ``state`` is ``None``: two live levels are merged when the evidence of their pooled
        statistics, in every slot the state factor is an axis of, is no lower than that of the
        separate levels. The merged level keeps its own prior once and both levels' data; the other
        returns to dead, its weight zero and its emissions at their prior. Candidate pairs are the
        ``n_pair`` (default ``reduction_pairs``) best by mutual expected log-likelihood, tested
        greedily. ``prior`` is the params the posterior was learned from, which the evidence is
        relative to. ``learn`` runs this every ``reduction_period``-th call.

        Design: docs/design/structure-learning.md S4
        """
        return _learn_face.reduce(self, params, prior, state, n_pair)

    def grow(
        self,
        params: SystemParams,
        observations: Mapping[str, Array],
        state: str,
        rounds: int = 1,
        lr: float | Array = 1.0,
    ) -> SystemParams:
        """The batch form of growth on the growable state factor ``state`` names: ``observations``
        holds a window of ``N`` data, ``(N, …)`` per modality over the state factor alone. Each of
        ``rounds`` rounds scores every datum under every level, takes the worst-explained one and
        wakes the first dead level for it while it falls below the growth threshold, seeding the
        level from that datum; the window is then absorbed at rate ``lr`` under the
        responsibilities the woken levels leave, into the state factor's weights and its
        modalities. ``learn`` is this with one datum and one round.

        Design: docs/design/structure-learning.md S3
        """
        return _learn_face.grow(self, params, observations, state, rounds, lr)

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
        self._refuse_black_box("policies")
        self._plannable("policies", params)
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

        True iff some transition is a ``Deterministic`` carry or an ``MLPCategorical`` code, the two
        families the analytic engine cannot hold as a belief. A predictor head, a learned or derived
        initial value and a recognition entry do not route by themselves; the amortized validator
        refuses a declaration it cannot express in full and the analytic validator refuses a head or
        a neural initial prior (design: recognition-beside-analytic.md R1).
        """
        params = self._p(params)
        return any(isinstance(b, Deterministic | MLPCategorical) for b in params.transition)

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

    def _plan_learning(self, params: SystemParams) -> tuple[tuple[str, tuple], ...]:
        """The engine: :func:`hybrid_aif.system.selectors._plan_learning`."""
        params = self._p(params)
        return selectors._plan_learning(self, params)

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

    def _validate(
        self, params: SystemParams, face: str = "this face", declared: bool = False
    ) -> None:
        params = self._p(params)
        try:
            validation._validate(self, params, face, declared)
        except ValueError as exc:
            note = validation.reference_note(self, params, str(exc))
            if note:
                raise ValueError(f"{exc}{note}") from exc
            raise

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

        ``beliefs`` is keyed by state factor name and must be marginals: resolve a conditional or
        learnable ``D`` with :meth:`initial_priors` first (a control node's entry is ignored).
        The context itself is positional (``init_beliefs`` in state factor order, actions and
        policies with one column per control node), as planners index it. ``horizon`` is the plan
        length the planner plans to and the closures are built for (``None``: ``policy_len``).
        ``key`` is the plan key a stochastic planner (CEM) reads; ``scoring_key`` is the scoring
        rows' own key, under which a row that rolls a model of its own takes the sampled rollout
        (only :meth:`score_policies` passes one). ``aux_beliefs`` is keyed by scoring-row name; each
        value is the belief carry of that row's own model, keyed by the nested System's state factor
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
        self._refuse_black_box("planner_context")
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
        self._plannable(face, params)
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

    @_eager_face
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
        return_policies: bool = False,
    ):
        """Policy posterior and the EFE term tree: ``(q_pi, tree)``, or ``(q_pi, tree,
        policies)`` with ``return_policies``, the planner's clamped policy set as a
        :class:`~hybrid_aif.controls.ControlStep` with one entry per control node and leading
        axes ``(P, policy_len)``, in the order ``q_pi`` and the tree's leaves follow.

        ``planner`` is a callable, or ``None`` for the System's registry planner. ``metrics=False``
        skips the tree (``None``); the posterior is the same up to float reassociation. A planner
        that selects without building a tree (``"habit"``) returns ``{}``; :meth:`score_policies`
        gives the tree G would assign its plan. ``plan_key`` is required by, and only by, a
        stochastic planner (``"cem"``). ``legal`` is the hard constraint of :meth:`act`, keyed by
        control node or one bool per policy. ``beliefs`` is keyed by state factor name. The planner
        must return clamped (integer) policies: a belief-valued policy set raises. ``present``
        says which modalities the plan expects to observe at each step, keyed by name: a bool
        for the whole plan, a per-step bool array (``(policy_len,)``, or ``(horizon,)`` under
        ``horizon=``), or a one-key mapping from a discrete
        control node's name to a bool per level (``{"answer": {"u_option": (False, True)}}``),
        read at each step's level of the node; a modality left out is present throughout. An
        absent modality's leaves are zero at that step, and under
        ``planning_belief="posterior"`` the epistemic chain is not conditioned on it there.
        """
        self._refuse_black_box("infer_policies")
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
            return_policies,
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
        linear-Gaussian state factor's ``G u`` as an offset and a Gaussian belief over it pushes
        through exactly (a truncated one by its closed moments), a nonlinear state factor's map
        reads a value as an input and a belief through its moments, and every leaf is computed under
        the result. The tree is then the objective of committing to the *randomised* policy that
        draws each node's level from its belief, step by step: a term linear in the control
        (utility, ambiguity, the predicted state) is the mixture of its clamped values, a
        nonlinear one (risk, info gain) is not. Under a belief, transition novelty is the
        mean-field mixture (the novelty weights averaged over ``q(u)``, contracted with the mixed
        prediction) and occupancy is the entropy of the mixed transition, ``H[Σ_u q(u) B_u]``.
        That is a different quantity from the expected
        objective under a policy posterior ``q_pi`` over clamped policies, which is the
        ``q_pi``-weighted ``total`` of the clamped tree. A one-hot belief scores as its clamp;
        an all-clamp mapping is the positional policy. A belief on a discrete node is read by
        categorical (tabular) transitions; a continuous state factor reading a belief-valued
        discrete node raises, as does a neural transition reading a belief-valued continuous one.

        A ``state_targets`` entry on a control node scores it: a ``risk`` leaf keyed by the
        node's name, ``−log C(u)`` under a clamp and ``KL(q(u) ‖ C)`` under a belief, at every
        step; a ``terminal_targets`` entry scores the last step as ``terminal_risk``.

        ``beliefs`` is keyed by state factor name. In either form the plan's length ``H`` need not
        be ``policy_len``: the scoring rows are built for the plan's own horizon.
        ``aux_beliefs`` passes to :meth:`planner_context`. ``key`` is the scoring rows' key: rows
        that roll a model of their own take the sampled rollout under it, the expected one without.
        """
        self._refuse_black_box("score_policies")
        return _plan_face.score_policies(self, params, beliefs, policies, aux_beliefs, key)

    def _check_plan(self, plan: Array, what: str, params: SystemParams) -> Array:
        return validation._check_plan(self, plan, what, params)

    @_eager_face
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

        ``beliefs`` is keyed by state factor name; the action by action-node name, one entry per
        action node, the intended value of its control node (a level, or a continuous node's
        float value): a draw (or the mode) from each control node's posterior, sampled over the
        policy set the planner returned. ``key`` selects the action,
        ``plan_key`` drives a stochastic planner; ``metrics`` as in :meth:`infer_policies`.
        ``mode=None`` takes the first-step action marginals for discrete controls and the drawn
        policy's first step for continuous ones, where ``mode="marginal"`` raises. A planner that
        emits a posterior over a continuous plan (the gradient planner under
        ``posterior="laplace"``, a truncated Gaussian on the box) has the action drawn from that
        posterior's first step under ``selection="stochastic"``, and its mode under
        ``"deterministic"``. ``selection`` is the action's realisation: ``"stochastic"``
        corresponds to the realisation ``sample`` (a draw) and ``"deterministic"`` to ``mode``
        (the argmax, or the mean of a continuous belief); a mixed strategy (``stats``) is not
        emitted here. Design: nesting-and-coupling.md N3. ``legal`` is this
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
        self._refuse_black_box("act")
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
        self._refuse_black_box("action_marginals")
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
        """Open-loop prediction: the beliefs, keyed by state factor name, after each step of one
        policy.

        ``policy`` stays positional: int32 ``(horizon, K)``, one column per control node, or float
        ``(horizon, action_dim)`` over an all-continuous action space; step ``t`` of the result
        follows ``policy[t]``, through the transition :meth:`update_empirical_prior` uses (no
        observations enter); an action node a modality observes carries its prediction at each
        step, as :meth:`update_empirical_prior` gives it. ``key`` selects the sampled rollout,
        split once per step; ``None`` is the expected one.
        """
        self._refuse_black_box("predict_beliefs")
        return _infer_face.predict_beliefs(self, params, beliefs, policy, key)

    @_eager_face
    def update_empirical_prior(
        self,
        params: SystemParams,
        beliefs: Mapping[str, Belief],
        action: Mapping[str, Array],
        key: Array | None = None,
    ) -> dict[str, Belief]:
        """Next tick's state priors: ``beliefs`` pushed through ``action``, keyed by state factor
        name.

        ``action`` is keyed by action-node name as :meth:`act` returns it (``{}`` when nothing is
        controlled): a level per discrete node, a float vector per continuous one, the two kinds
        side by side over a mixed space. Each control node is clamped to its action's value at
        this step (an efference copy): each discrete node's returned entry is its prior for the
        next step, uniform over its levels. An action node a modality observes gets an entry
        too, the agent's prediction of its action: the column of ``params.actuator`` at the
        clamped level, the one-hot of that level under the delta kernel. A continuous node enters
        a linear-Gaussian state factor as the ``G u`` offset, a neural one as part of its input, a
        nonlinear one as an entry of its map's controls. Hybrid
        state factors go through their own transitions (exact Gaussian ``predict``; grow then
        ``reduce(mixture_k)`` for mode-coupled state factors), the closure the planner uses for one
        step. On the amortized path each carry advances, then each code is read off its transition
        at this step's parents.

        Without ``key`` the result is the expected belief. With one it is the sampled belief: one
        draw per state factor as a belief of its family (:mod:`hybrid_aif.inference.sample`). That
        is a legal carry but a point mass: a collapsed Gaussian or mixture cannot be conditioned and
        has no finite entropy. See docs/composing-systems.md.
        """
        self._refuse_black_box("update_empirical_prior")
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
    off ``subsystems``. ``prefix`` qualifies every name, for a merge that takes this one as a
    component."""
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
                f"Link): key it by {q(driver)!r}, the node it became, whose choice "
                "it constrains",
                f"the action node of a control node merge() identified with {q(driver)!r}: "
                f"key it by {q(live_action[driver])!r}",
            )
        elif driver in ended:
            gone(
                node,
                f"a control node merge() made one with {q(driver)!r}, which a Clamp "
                f"terminated, so no node {q(node)!r} exists here: drop the key",
                f"the action node of a control node merge() made one with {q(driver)!r}, "
                "which a Clamp terminated: drop the key",
            )
        else:
            gone(
                node,
                f"a control node merge() identified with {q(driver)!r} (a same-step "
                f"Link), which another subsystem chooses: key it by {q(driver)!r} "
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
            "a control node merge() terminated with a Clamp, so no node "
            f"{q(node)!r} exists here: drop the key",
            "the action node of a control node merge() terminated with a Clamp: drop the key",
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
