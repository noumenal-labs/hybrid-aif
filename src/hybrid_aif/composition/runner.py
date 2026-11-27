"""``run`` over a closed System: every leaf stepped, one value per variable per round.

A **closed System** is a merge or a coupling in which every modality is fed by an identification
(``System.observes``): nothing is observed from outside. It is run by naming a step for each of
its **stepped nodes**, the leaves (components that are not themselves composed) or a non-leaf
node planned jointly as one mind (:func:`~hybrid_aif.composition.joint.joint_planner`); the keys
of ``steps`` form an antichain that covers every leaf. A step is the function
``step(system, key, obs, carry, params) -> (out, carry, params)``:

- ``system`` and ``params`` are its view (the leaf's :func:`~hybrid_aif.composition.merge.subsystem`
  view, or the joint view) and the view's params, taken from the parameter home each round;
- ``obs`` maps each of its modality slots and each control node a variable drives, by the slot's
  own name, to the value of the variable that slot observes; ``obs`` is ``None`` on the step's
  first call, its reset. A slot whose variable may have no value yet at the step's turn (a
  variable the reset left without a value, read before it is realised in round 0: at the round's
  boundary, or before its owner's turn) is handed a :class:`Maybe` ``(value, present)`` every
  round, ``value`` zeros of the variable's form and ``present`` false in round 0, the value and
  true after; which slots these are is a static fact of the structure pass, and every other slot
  is handed its value. ``plan()`` reads an absent value as an absent modality;
- ``out`` maps its out-ports to values: the action nodes of the nodes it chooses (each an
  intended level, or a float vector for a continuous node) and its emissions;
- the params returned are the very object handed in when the step did not learn; a learner's
  returned parameter variables are written back to the home
  (:func:`~hybrid_aif.composition.merge.write_back`).

:func:`plan` builds the step of a planning mind, :func:`sample` the step of a process (a world
sampled forward from its declarations); a custom step is any function of that shape.

**Variables.** A System's variables that a step generates are its action nodes and its emissions;
each has one value per round. An action node is realised from its node's ``params.actuator``
kernel (a draw at the intended level) when the node decides, and stands until realised again; an
emission is realised by the step that owns it, from the point or the belief that step carries. A
modality, or a control node, that ``System.observes`` names reads the variable it observes as it
stands at its reader's turn, and never a value that does not yet exist.

**Timing.** A round is one tick under every order, and every stepped node steps once. A stepped
node whose control node is identified with another stepped node's action node steps after that
action is realised in the round, and the slot carries the realised value. ``"simultaneous"``
(default): the nodes none of whose control nodes is driven by an action node step on the round's
boundary (phase one), their action nodes realised together; the others step after, in an order
that respects the rule (phase two). A node of phase two reads a control node driven by another
node's action as realised this round, and every other slot, an emission-driven control node
included, at the boundary. ``"sequential"``: the listing orders the sources, the nodes none of
whose control nodes an action drives, which step in listing order, each reading what stands and
realising its action nodes as it chooses; each driven node steps as soon as the last of its
drivers has realised, before the sweep returns to nodes made ready earlier (a chain follows its
source, a join waits for its last driver and steps there, nodes made ready by the same realisation
keep listing order), so a driven node's own place in the listing does not matter. ``"async"``:
sequential with a fresh random listing of the sources each round (reproducible from the run key:
``jax.random.permutation(fold_in(fold_in(key, N), t), sources)`` at round ``t``), one total
order per round, every node reading what
stands at its turn and each driven node placed by the same derived rule; there is no boundary
read. Sequential is a systematic scan, async a random scan,
simultaneous a synchronous update. A cycle of action-to-control identifications is refused.

**Write-backs.** Under ``"simultaneous"`` every node's view in a round is taken from the parameter
home as the round began, and the params each learner returns are applied at the round's boundary,
after its last realisation and before the next round's reads. Two nodes of one round that wrote
back one parameter variable are combined only where it is Dirichlet counts: the round-start
concentration plus the sum of their deltas, the posterior given both nodes' data, and a sum that
is not positive is refused as ``write_back`` refuses one; for any other form two ``plan()`` steps
whose ``learn`` both reach the variable are refused before the first round, and two other steps
that both wrote it back in one round are recorded as a fault of that round (the variable keeps
its round-start value), as is a sum that is not positive. Under
``"sequential"`` and ``"async"`` a write-back is live: the next view is taken from the home as it
then stands.

**Clocks.** A stepped node whose view declares durations (``params.durations``, one control node
of the node carrying a duration per level) decides once per option. On a decision round it steps
and ``run`` reads the duration of the level it chose (the entry of ``params.durations`` at the
intended level of the node that carries them),
times the node's entry of ``clocks`` (the rounds one unit of duration lasts, 1.0 by default); the
product must be a whole number of rounds of at least one, and the node does not step again until
that many rounds, the decision round included, have passed. On a held round there is no view, no
step call, no write-back and no realisation: the node's action nodes and emissions keep their
values, and a variable it drives keeps its level. The first round always decides. The count is
``run``'s own and is not in the step's carry, so ``carries`` seeds a step and never a count. A node
with durations refuses a non-identity actuator on its action nodes, and ``Trajectory.decided``
records which rounds decided.

**Keys.** The run key splits into ``(key, k_reset)``; stepped node ``j`` (in the order of the
System's components, a joint node at the place of its node) calls its reset step with ``k_reset``
for ``j = 0`` and ``fold_in(k_reset, j)`` after. Each round ``(key, k_tick) = split(key)`` and
``k_tick`` splits into ``2N + 1`` keys for ``N`` stepped nodes: node ``j`` steps with key ``j``;
its action nodes are realised with key ``N + j``, except the nodes of phase one under
``"simultaneous"``, realised together with the last key. Realising with key ``k`` draws action
node ``i`` of the ``A`` action nodes of the System (in order, the action nodes of driven control
nodes excluded) from its kernel's column at the intended level with key ``i`` of
``split(fold_in(k, ACTUATOR_TAG), A)``. An emission is realised with the owner's step key folded
by ``EMIT_TAG``.

**The compiled round.** A run is the structure pass, the reset, and one compiled ``lax.scan`` of
every round from round 0. The structure pass reads structure, not values, on the host before
any step is called: the antichain, the order (under async every distinct derived order), the
clock table (the rounds an option lasts at every level of each durations table, each
validated), and the refusals above. The reset runs on concrete values (unbatched); the state it
leaves is then settled into the form of every round's state: a variable it left without a value
holds zeros of its form with a presence flag (the :class:`Maybe` a step is handed), and a
:class:`MindCarry` its every field, the ones round 0 sets as zeros its ``started`` flag guards
(a factor carried as a mixture is padded to ``mixture_k`` components with zero-weight ones at
the reset); any other carry must already have the form round 0 returns, refused naming the step
otherwise. The forms are read off the structure (``plan()``'s out-ports and carry from the
System's declarations, every other step's from what its reset returned) and checked on the
scan's own trace; a ``plan(trace=True)`` node's trace takes the form round 0 gives it,
evaluated abstractly on the state so settled (the planner traced once, its trace reused by the
scan's compile); where a form is not derivable (a custom step that leaves an out-port without a
value at its reset), or the check finds one that does not hold, round 0 of the state from the
reset is evaluated abstractly (nothing run) and the state settled from it, one state every way
(``Trajectory.timing["settled"]``). Every round is the same body
traced once and scanned, so a step is a traced function: its outputs JAX arrays of one shape
from round to round, its carry and its params of the structure it was handed, and nothing
host-side (``np.asarray``, ``float()``, a Python branch on a value) on what it is handed, which
is refused naming the step. The view of a sliced step is taken every round with its level a
traced index (one System serves every level; a level handed as a :class:`Maybe` is a
probability vector, its one-hot or, absent, uniform); a clocked node's decide-or-hold is a
``lax.cond``, its count an int32 array; write-backs are applied by arithmetic on the values. A
check of a value inside the scan cannot raise: it is recorded in ``Trajectory.faults`` (a
write-back that was not applied, a level out of range, a step that changed its params outside
its variables), round 0 included; the reset's checks refuse.

**Replicas.** ``run(..., batch=B)`` runs ``B`` replicas of the closed System by one program:
replica ``b`` under ``fold_in(key, b)`` (or ``key[b]`` for a stack of ``B`` keys), and inside it
every key derived as above, so a replica is the unbatched run under its key. The reset is a
vmapped call over the replicas and the scan a vmapped scan; a step never sees the replica axis
(it is handed one replica's view, observations, carry and key). Where several local devices
exist (accelerators, or a CPU host given ``--xla_force_host_platform_device_count``) the
vmapped scan is a ``shard_map`` over them: each device runs the whole scan over its share of
the pass's replicas, with no collective and no rendezvous between rounds, a pass filled with
copies of its last replica to a multiple of the devices it uses (the fewest replicas per device,
then the fewest devices at that load; ``Trajectory.timing["devices"]``). On a CPU host with one
device a batched run is no faster than its replicas one after another. Replicas hold
their options independently: a clocked node whose options last different rounds at different
levels has a per-replica decide-or-hold, which vmap runs as both branches and a select. Under
async each replica draws its own order and a switch over the distinct derived orders picks it,
which vmap runs as every branch and a select: the cost of a batched async round is the number
of distinct orders times the step work (two sources, two orders), and more than four sources
(24 orders) are refused under ``batch=``.

**Groups.** A key of ``steps`` may be a tuple of node names, ``steps={("agent.rover1",
"agent.rover2"): step}``: the named nodes are copies of one node (their views equal up to their
names) and are stepped by one vmapped call of the step over their stacked views' params,
observations, carries and keys, each copy handed its own under the first copy's names. The
copies are simultaneous among themselves: each reads the round-start home (under sequential and
async, the home as it stood when the group's turn began) and their write-backs are combined as
simultaneous ones are (Dirichlet counts summed). A group is one node to the order; its record is
each node's own, under its name.

Design: docs/design/engine-consolidation.md; nesting-and-coupling.md N3, N4, N5; control-states.md
C13; batched-run.md R1–R14 (the compiled round, replicas, groups; Phase C: the run settled from
the reset, independent per-device scans).
"""

import dataclasses
import fnmatch
import functools
import itertools
import math
import numbers
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hybrid_aif.composition.joint import joint_planner
from hybrid_aif.composition.merge import (
    SEPARATOR,
    MixtureRefusal,
    _inputs_of,
    _subsystem_meta,
    _write_back,
    driven_nodes,
    subsystem,
)
from hybrid_aif.controls import FLOAT, LEVEL, ControlStep
from hybrid_aif.distributions import Categorical, ConditionalDirichlet, Dirichlet
from hybrid_aif.dtypes import as_float, as_floats, table_dtype
from hybrid_aif.envs.generative import ACTUATOR_TAG
from hybrid_aif.objectives import hybrid as _hybrid
from hybrid_aif.planning.base import Planner, get_planner, planner_takes_init
from hybrid_aif.planning.gradient import PlanPosterior
from hybrid_aif.reads import BLACK_BOX, EMIT_TAG
from hybrid_aif.system import Belief, LearnState, System, SystemParams
from hybrid_aif.system.faces import plan as _plan_face

__all__ = ["Maybe", "Step", "Trace", "Trajectory", "plan", "run", "sample"]

#: ``step(system, key, obs, carry, params) -> (out, carry, params)``: see the module docstring.
Step = Callable[
    [System, Array, "Mapping[str, Array | None] | None", Any, SystemParams],
    tuple[Mapping[str, Array], Any, SystemParams],
]

TickOrder = Literal["simultaneous", "sequential", "async"]


class MindCarry(NamedTuple):
    """One planning mind's internal state between rounds, the only place its beliefs live: the
    carry of :func:`plan`.

    Carries are never merged, so two minds share no leaf.

    Design: nesting-and-coupling.md N1.
    """

    #: The empirical prior the next round filters from: the last posterior pushed through its
    #: transition under the chosen action; after the reset, the declared initial priors, a
    #: factor carried as a mixture padded to ``mixture_k`` components with zero-weight ones
    #: (``None`` in a carry not yet reset: the declared initial priors are used).
    beliefs: Mapping[str, Belief] | None = None
    #: The carry of ``System.learn`` once the mind learns online.
    learn_state: LearnState | None = None
    #: The posterior of the last round, the ``prev_beliefs`` that ``System.learn`` takes
    #: (``None`` before the first round, or zeros in ``run``'s settled carry, ``started`` false).
    previous: Mapping[str, Belief] | None = None
    #: The action the last round chose, the ``action`` that ``System.learn`` takes (``None``
    #: before the first round, or zeros in ``run``'s settled carry).
    action: Mapping[str, Array] | None = None
    #: The last round's :class:`Trace` under ``plan(trace=True)``.
    trace: object | None = None
    #: Under ``plan(warm_start=True)``, the continuous controls of the plan the last round acted
    #: on, ``(policy_len, action_dim)``: the policy of highest ``q_pi`` in the planner's set (the
    #: gradient and mixed planners put all of ``q_pi`` on it). ``None`` before the first round
    #: and without ``warm_start``.
    plan: Array | None = None
    #: Whether a round has been planned (a bool array, ``False`` after the reset): ``run`` hands
    #: round 0 a carry of the form of every later round, its ``previous``, ``action``, ``trace``
    #: and ``plan`` zeros, and the round learns only where this holds.
    started: Array | None = None


class Trace(NamedTuple):
    """What a planning step kept of one round's plan (``plan(trace=True)``, in
    ``MindCarry.trace``)."""

    #: The posterior over the policy set.
    q_pi: Categorical
    #: The term tree of the objective (``System.act``'s metrics).
    tree: Any
    #: The enumerated policy set ``q_pi`` is over (``System.policies``); ``None`` unless the plan
    #: is over an enumerated set.
    policies: Array | None
    #: The policy set as the planner returned it, in the order of ``q_pi`` and the tree's
    #: leaves: every start's final plan ``(starts, policy_len, action_dim)`` for the gradient
    #: planner, a :class:`~hybrid_aif.controls.ControlStep` for the mixed planner, integer levels
    #: for an enumerating one.
    plans: Any = None
    #: The round's :class:`~hybrid_aif.inference.InferenceDetail` (``System.infer_states`` with
    #: ``detail=True``: the log-evidence per modality, the free energy); ``None`` on the
    #: amortized path.
    detail: Any = None
    #: The view's parameter variables (``SystemParams.parameters``) the round planned with: after
    #: the round's learning under ``learn``, else as handed in.
    parameters: Mapping[str, Any] | None = None


# ------------------------------------------------------------------------------------ plan


def _learn_arguments(learn) -> tuple | None:
    """``plan(learn=...)`` as the keywords of ``System.learn``, a hashable tuple of items."""
    if learn is None or learn is False:
        return None
    if learn is True:
        return ()
    if isinstance(learn, Mapping):
        return tuple(sorted(learn.items()))
    if isinstance(learn, Sequence) and not isinstance(learn, str):
        return (("lr", 0.0), ("lr_by_slot", tuple(sorted((k, 1.0) for k in learn))))
    raise TypeError(
        f"plan(learn={learn!r}): learn is None, True (every learnable slot), a mapping of "
        "System.learn keywords, or a sequence of slot keys"
    )


def _learn_keywords(items: tuple) -> dict[str, Any]:
    out = dict(items)
    if isinstance(out.get("lr_by_slot"), tuple):
        out["lr_by_slot"] = dict(out["lr_by_slot"])
    return out


def _shifted(plan: Array) -> Array:
    """A plan moved one step on: its steps after the first, the last step repeated."""
    return jnp.concatenate([plan[1:], plan[-1:]], axis=0)


def _first_plan(policy_len: int, layout, sizes, kinds, boxes) -> Array:
    """The warm start of :func:`plan`'s first round: zeros ``(policy_len, width)``, the width
    that of the continuous controls concatenated in node order (``layout``, the flat layout of
    an all-continuous space, or the float nodes among ``sizes`` and ``kinds``), in the dtype of
    the continuous nodes' ``boxes`` (``None`` for a discrete node), the dtype a plan over them
    takes. One rule for the planner's context (:func:`_warm`) and for the state ``run`` settles
    from the reset (:func:`_settle_static`), which reads the same control space."""
    if layout is not None:
        width = int(layout.action_dim)
    else:
        width = int(sum(size for size, kind in zip(sizes, kinds, strict=True) if kind == FLOAT))
    if layout is not None and layout.low is not None:
        bounds = [layout.low, layout.high]
    else:
        bounds = [b for box in boxes if box is not None for b in (box.low, box.high)]
    return jnp.zeros((policy_len, width), as_floats(*bounds)[0].dtype if bounds else float)


def _warm(planner, previous):
    """``planner`` handed the warm start of :func:`plan` (``warm_start=True``) as its ``init``:
    ``previous`` (the continuous plan the last round acted on) shifted one step on, or zeros
    ``(policy_len, action_dim)`` in the first round; the planner's marks are carried over."""

    def warm(ctx, use_bound="midpoint", metrics=True):
        if previous is None:
            init = _first_plan(
                ctx.policy_len, ctx.continuous_controls, ctx.sizes, ctx.kinds, ctx.boxes
            )
        else:
            init = _shifted(previous)
        return planner(ctx, use_bound=use_bound, metrics=metrics, init=init)

    for mark in ("needs_key", "branching", "legal_depth", "takes_init"):
        if hasattr(planner, mark):
            setattr(warm, mark, getattr(planner, mark))
    return warm


def _acted_plan(q_pi: Categorical, policies) -> Array:
    """The continuous controls ``(policy_len, action_dim)`` of the policy of highest ``q_pi`` in
    a planner's set: a float array, a :class:`~hybrid_aif.planning.gradient.PlanPosterior`'s
    plans, or a :class:`~hybrid_aif.controls.ControlStep` (its float entries concatenated)."""
    best = jnp.argmax(q_pi.probs)
    if isinstance(policies, PlanPosterior):
        return policies.mode[best]
    if isinstance(policies, ControlStep):
        blocks = [e[best] for e in policies.entries if jnp.issubdtype(e.dtype, jnp.floating)]
        return jnp.concatenate(blocks, axis=-1)
    return policies[best]


def _check_warm_start(planner, system: System) -> None:
    """Refuse ``plan(warm_start=True)`` for a planner that takes no ``init``."""
    resolved = get_planner(system.planner) if planner is None else planner
    if not planner_takes_init(resolved):
        name = system.planner if planner is None else getattr(planner, "__name__", repr(planner))
        raise ValueError(
            f"plan(warm_start=True) hands the planner the plan it last acted on, shifted one "
            f"step on, as its init, and the planner {name!r} takes no init: plan with the "
            '"gradient" or "mixed" planner (make_gradient_planner, make_mixed_planner), or a '
            "planner that accepts init= and carries the mark takes_init = True"
        )


@eqx.filter_jit
def _plan_tick(system, params, planner, kwargs, trace, learn, warm_start, key, seen, carry):
    """One compiled round: infer, learn, act, push, emit; ``params`` is an argument, so equal
    views share one program. ``seen`` is keyed by variable where the view observes its own."""
    present = {m: False for m, v in seen.items() if v is None}
    present.update({m: v.present for m, v in seen.items() if isinstance(v, Maybe)})
    seen = {m: v.value if isinstance(v, Maybe) else v for m, v in seen.items()}
    beliefs, detail = system.infer_states(
        params, seen, priors=carry.beliefs, present=present or None, detail=trace
    )
    learned, learn_state = None, carry.learn_state
    if learn is not None and carry.previous is not None:
        extra = {} if carry.beliefs is None else {"priors": carry.beliefs}
        out = system.learn(
            params,
            seen,
            beliefs,
            prev_beliefs=carry.previous,
            action=carry.action,
            learn_state=carry.learn_state,
            **extra,
            **_learn_keywords(learn),
        )
        learned, learn_state = out if isinstance(out, tuple) else (out, carry.learn_state)
        if carry.started is not None:
            # a settled carry (run's round 0 onward): no previous round before the first
            learned = _select(carry.started, learned, params)
            learn_state = _select(carry.started, learn_state, carry.learn_state)
    held = params if learned is None else learned
    if warm_start:
        planner = _warm(get_planner(system.planner) if planner is None else planner, carry.plan)
    action, q_pi, tree, policies = _plan_face.act_and_policies(
        system, key, held, beliefs, planner=planner, metrics=trace, **dict(kwargs)
    )
    pushed = system.update_empirical_prior(held, beliefs, action)
    emitted = {}
    if system.emissions:
        names = system._names()
        values = {
            **beliefs,
            **{
                u: action[a]
                for u, a in zip(names.controls, names.actions, strict=True)
                if a in action
            },
        }
        emitted = system._emit(jax.random.fold_in(key, EMIT_TAG), held, values)
    stored = carry._replace(
        beliefs=pushed,
        learn_state=learn_state,
        previous=beliefs,
        action=action,
        trace=None,
        plan=_acted_plan(q_pi, policies) if warm_start else carry.plan,
        started=jnp.asarray(True),
    )
    planned = (q_pi, tree, policies, detail, held.parameters) if trace else None
    return action, emitted, stored, learned, planned


class _Plan:
    """The step :func:`plan` builds: a callable that also says what ``run`` needs of it. ``sliced``
    marks a step that is handed a view sliced at its driven control nodes; a wrapper around it
    copies the mark with ``functools.wraps``."""

    def __init__(self, planner, learn, trace, replace, act_kwargs, warm_start=False):
        self.sliced = True
        self.planner = planner
        self.learn = _learn_arguments(learn)
        self.trace = bool(trace)
        self.warm_start = bool(warm_start)
        if self.warm_start and planner is not None and not planner_takes_init(planner):
            _check_warm_start(planner, None)
        self.replace = dict(replace or {})
        self.act_kwargs = tuple(sorted(act_kwargs.items()))
        self._replaced: dict[int, tuple[System, System]] = {}
        self._policies: dict[int, tuple[System, Any]] = {}

    def _view(self, system: System) -> System:
        if not self.replace:
            return system
        held = self._replaced.get(id(system))
        if held is None or held[0] is not system:
            held = (system, dataclasses.replace(system, **self.replace))
            self._replaced[id(system)] = held
        return held[1]

    def _enumerated(self, system: System, params: SystemParams):
        """The enumerated policy set of the view, computed once per view on concrete params
        (``None`` unless the plan is over an enumerated set); inside ``run``'s compiled round it
        is computed on the trace and not kept, so no traced value outlives its trace."""
        held = self._policies.get(id(system))
        if held is None or held[0] is not system:
            policies = None
            if self.planner is None and system.planner == "enumeration":
                kinds = system._space(system._params_in(params, "plan")).kinds
                policies = None if FLOAT in kinds else system.policies(params)
            if isinstance(policies, jax.core.Tracer):
                return policies
            held = (system, policies)
            self._policies[id(system)] = held
        return held[1]

    def __call__(self, system, key, obs, carry, params):
        system = self._view(system)
        driven = [
            u
            for u, a in zip(system._table.control_names, system.action_names, strict=True)
            if a is None
        ]
        if driven:
            raise ValueError(
                f"plan(): the view has driven control node(s) {driven}, which a planner is handed "
                "sliced at the driver's level. A step that wraps plan() must carry its "
                "`sliced = True` mark: decorate the wrapper with functools.wraps(plan(...)) or "
                "set wrapper.sliced = True"
            )
        carry = MindCarry() if carry is None else carry
        if obs is None:
            legal = dict(self.act_kwargs).get("legal")
            if legal is not None:
                _check_legal(system, params, legal, "plan")
            if self.warm_start:
                _check_warm_start(self.planner, system)
            if self.learn is not None and carry.learn_state is None:
                carry = carry._replace(learn_state=system.init_learn_state(params))
            beliefs = system.initial_priors(params) if carry.beliefs is None else carry.beliefs
            carry = carry._replace(
                beliefs=_canonical(system, params, beliefs), started=jnp.asarray(False)
            )
            return {}, carry, params
        renamed = dict(system.observes)
        seen = {renamed.get(slot, slot): value for slot, value in obs.items()}
        action, emitted, stored, learned, planned = _plan_tick(
            system,
            params,
            self.planner,
            self.act_kwargs,
            self.trace,
            self.learn,
            self.warm_start,
            key,
            seen,
            carry._replace(trace=None),
        )
        if planned is not None:
            q_pi, tree, plans, detail, parameters = planned
            stored = stored._replace(
                trace=Trace(
                    q_pi,
                    tree,
                    self._enumerated(system, params),
                    plans=plans,
                    detail=detail,
                    parameters=parameters,
                )
            )
        return {**action, **emitted}, stored, params if learned is None else learned


def _settled_mind(carry, like):
    """A planning mind's carry from the reset in the form of every round's (``like``, round 0's
    carry as an abstract value): each field the reset leaves ``None`` that a round fills
    (``previous``, ``action``, ``trace``, ``plan``) as zeros of its form, which ``started``
    (false) guards. Read off the carry, so a wrapper of :func:`plan` that hands the carry
    through settles as ``plan()`` does (design batched-run.md, Phase C)."""
    if not isinstance(carry, MindCarry) or not isinstance(like, MindCarry):
        return carry
    filled = {
        field: _zeros_like(getattr(like, field))
        for field in MindCarry._fields
        if getattr(carry, field) is None and getattr(like, field) is not None
    }
    return carry._replace(**filled)


def _canonical(system: System, params: SystemParams, beliefs: Mapping[str, Belief]) -> dict:
    """``beliefs`` in the form a planning mind carries from round to round: each state factor
    carried as a mixture (a mode-coupled continuous factor) brought to exactly ``mixture_k``
    components, padded with zero-weight ones (``pad_mixture``), so the carry has one form from
    the reset on; inference, the propagation and the reduction are invariant to the padding."""
    if not system.mixture_k or system._is_amortized(system._p(params)):
        return dict(beliefs)
    held = system._params_in(params, "plan")
    families = system._families(held)
    names = system._table.state_names
    out = dict(beliefs)
    for name, family in zip(names, families, strict=True):
        if family == _hybrid.MIXTURE and name in out:
            out[name] = _hybrid.as_mixture(out[name], system.mixture_k)
    return out


def _zeros_like(tree):
    """``tree`` with every array (or abstract array) leaf as zeros of its shape and dtype."""

    def leaf(x):
        if isinstance(x, jax.ShapeDtypeStruct) or eqx.is_array(x):
            return jnp.zeros(x.shape, x.dtype)
        return x

    return jax.tree_util.tree_map(leaf, tree)


class Maybe(eqx.Module):
    """A value handed to a step for a slot whose variable may have no value yet at the step's
    turn: ``value`` (zeros of the variable's form where it has none) and ``present`` (a bool).

    It is handed only for a variable that the reset left without a value (an action node or
    emission of a node that has not yet stepped) and that the step reads before it is realised
    in round 0 (at the round's boundary, or before its owner's turn), a static fact of the
    structure pass: the slot is then handed a ``Maybe`` every round, ``present`` false in round
    0 and true after. Every other slot is handed its value. ``plan()`` reads an absent value as
    an absent modality; a custom step reads ``obs[slot].value`` where ``obs[slot].present``.
    """

    value: Array
    present: Array


def plan(
    planner: Planner | None = None,
    learn=None,
    trace: bool = False,
    replace: Mapping[str, Any] | None = None,
    warm_start: bool = False,
    **act_kwargs,
) -> Step:
    """The step of a planning mind: infer, learn, act, push the prior, emit.

    Built once, compiled per view (equal views share one program, ``params`` being an argument).
    Each round, on its view: infer the posterior from the carried prior (a modality whose value is
    ``None`` is absent that round; a modality that observes an emission or action node of the view
    takes the datum of that variable); learn when ``learn`` is set and a previous round exists
    (``System.learn`` with the previous posterior and action); choose an action
    (``System.act(key, params, beliefs, planner=planner, **act_kwargs)``); push the posterior
    through the transition under the action as the next prior; realise every emission of the view
    from the posterior (the step's key folded with ``EMIT_TAG``). It returns the action nodes and
    the emissions, and the params it was given, or the learned ones. The carry is a
    :class:`MindCarry`, ``None`` meaning a fresh one; the step's
    first call (``obs is None``) returns nothing and leaves the declared initial priors as the
    first round's prior.

    ``learn`` is ``None`` (nothing is learned), ``True`` (every learnable slot), a mapping of
    ``System.learn`` keywords, or a sequence of slot keys (those slots learn at rate 1, the rest at
    0). ``trace=True`` keeps each round's :class:`Trace` in ``carry.trace``: the policy
    posterior, the term tree, the planner's policy set, the inference detail and the parameter
    variables the round planned with. ``warm_start=True`` makes the plan receding-horizon: the
    carry keeps the continuous controls of the plan each round acted on (``MindCarry.plan``),
    and the next round's planner starts from that plan shifted one step on (its steps after the
    first, the last step repeated), passed as its ``init``; the first round starts from zeros.
    The shift is by one of the node's own steps, so a node on a clock or holding an option
    shifts once per decision. It needs a planner that takes an ``init`` (the gradient and mixed
    planners, :func:`~hybrid_aif.planning.planner_takes_init`) and refuses any other.
    ``replace`` sets
    fields of the view (``{"policy_len": 2}``) before it plans; on a non-leaf node it is also
    where the joint view's configuration is given (as ``joint_planner`` takes it). ``act_kwargs``
    pass to ``System.act`` (``selection=``, ``legal=``, ...); a ``legal`` is checked on the first
    call. A view with a driven continuous control node is refused, since a planner holds a driven
    input at one level: use a custom step.
    """
    return _Plan(planner, learn, trace, replace, act_kwargs, warm_start)


# ---------------------------------------------------------------------------------- sample


@eqx.filter_jit
def _sample_tick(system, params, key, state, inputs):
    emitted, state = system.sample_step(key, params, state, inputs)
    controls = system._sample_controls(key, params, inputs)
    names = system._table.control_names
    actions = {
        a: controls[u] for u, a in zip(names, system.action_names, strict=True) if a is not None
    }
    return emitted, state, actions


class _Sample:
    """The step :func:`sample` builds."""

    def __call__(self, system, key, obs, carry, params):
        if obs is None:
            emitted, state = system.sample_initial(key, params, state=carry)
            return emitted, state, params
        nodes = set(system._table.control_names)
        unset = [k for k, v in obs.items() if k in nodes and isinstance(v, Maybe)]
        if unset:
            raise ValueError(
                f"sample(): control node {unset[0]!r} is driven by a variable that has no value "
                "at the reset and is read before it is realised in round 0, so the process would "
                "step with no value for it; have the driver return that variable at its reset "
                "(obs is None), drive the node from an action node (its owner steps before the "
                'process in every round), or run under "sequential" with the driver listed first'
            )
        inputs = {k: v for k, v in obs.items() if k in nodes and v is not None}
        emitted, state, actions = _sample_tick(system, params, key, carry, inputs)
        return {**emitted, **actions}, state, params


def sample() -> Step:
    """The step of a process: its System sampled forward from its declarations.

    The first call (the reset) draws the initial state, or takes the partial point state the
    carry was seeded with through ``run(carries=...)``, and returns the emissions of that state.
    Each round calls :meth:`System.sample_step` on the unsliced view with the control nodes that
    other nodes drive taken from ``obs`` (a driven node's value as the driver realised it),
    returning every emission realised by its declared rule from the stepped state and the action
    node of each control node that no variable drives, drawn from its habit or uniformly. The
    carry is the point state, keyed by state factor name.
    """
    return _Sample()


# -------------------------------------------------------------------------- the record


class Trajectory(NamedTuple):
    """What :func:`run` recorded: a tree with one node per component of the System.

    The root holds ``params``; every node holds its ``parts`` (its components, by their name
    relative to the node) and the records below, each a mapping keyed by name relative to the
    node, a non-leaf node's mappings being the prefixed union of its parts'. The records are
    stacked over rounds: ``received``, ``present`` and ``carries`` have ``ticks + 1`` entries,
    ``emitted``, ``intended`` and ``decided`` have ``ticks``.

    ``trajectory["agent.rover"]`` walks the tree by dotted path; a name that is no node is
    refused, naming the nodes. A node that is planned jointly is a leaf of the tree, and the
    nodes of a group each have their own record.

    Under ``run(..., batch=B)`` every array leads with an axis of the ``B`` replicas, before the
    rounds; ``carries`` is a tuple over rounds whose carries' arrays lead with it. A field that
    ``run(record=...)`` did not keep is ``None``.
    """

    #: The parameter home after the last round (held at the root).
    params: SystemParams | None
    #: The node's components, each a ``Trajectory``, keyed by name relative to the node.
    parts: dict[str, "Trajectory"]
    #: The value each modality slot, and each control node a variable drives, held for its step,
    #: ``ticks + 1`` entries: entry ``t`` is what the step was handed at round ``t``, the last what
    #: the slots hold after the last round. An absent value is zeros. On a round the node did not
    #: step (see ``decided``) it holds what the node's in-ports held.
    received: dict[str, Array]
    #: Whether each entry of ``received`` held a value, ``ticks + 1`` entries.
    present: dict[str, Array]
    #: The realised value of each action node and emission of the node, ``ticks`` entries. On a
    #: round the node did not step it holds the node's standing values.
    emitted: dict[str, Array]
    #: Each action node as its step chose it, ``ticks`` entries: equal to ``emitted`` where no
    #: kernel in ``params.actuator`` moved the draw. On a round the node did not step it holds the
    #: node's held action.
    intended: dict[str, Array]
    #: The out-ports that held a value entering round 0, as the reset call left them.
    start: dict[str, Array]
    #: A stepped node's carry after each round (``ticks + 1`` entries, the first after the reset
    #: call, settled into every round's form: a ``plan()`` carry with ``started`` false) when
    #: ``record_carries`` asked for them, else ``None``. For a node that is not itself
    #: stepped, a mapping from each stepped node beneath it, by relative name. Each entry is the
    #: object the step returned, kept as it is and not copied, so a step whose carries are
    #: recorded returns a new carry each round rather than mutating the one it was handed.
    carries: Any
    #: A stepped node's carry after the last round; for a node that is not itself stepped, a
    #: mapping from each stepped node beneath it, by relative name.
    carry: Any
    #: A bool per round, whether the node stepped: ``False`` on a round that a node declaring
    #: durations spent inside its option, ``True`` every round of any other node.
    decided: Any
    #: At the root, the checks of values that failed inside the compiled rounds, each a bool per
    #: round (``ticks`` entries) keyed by what failed: a parameter variable whose write-back was
    #: not applied (a concentration that is not positive, two writers of a form that is not
    #: counts; the variable kept its round-start value), an out-port given a level out of range,
    #: or ``"<node>:params"`` for a step that changed its params outside its parameter variables
    #: and its own tables; every round is traced, round 0 included. Empty when nothing was
    #: checked; ``None`` below the root.
    faults: Any = None
    #: At the root, the host's wall time of the run's parts, in seconds, and how it ran:
    #: ``compile`` (tracing and compiling the scan of rounds), ``scan`` (executing it, compile
    #: excluded) over ``scanned`` rounds, ``reset`` (the reset calls) and ``settle`` (bringing
    #: the state from the reset into every round's form), ``settled`` (how: ``"static"``, the
    #: forms read off the structure; ``"static+trace"``, the same with each ``plan(trace=True)``
    #: node's trace read off an abstract round 0 of that state; ``"traced"``, round 0 of the
    #: state from the reset evaluated abstractly; ``None`` for a run of no rounds), ``chunks``,
    #: ``passes``, ``replicas_per_pass``, and ``devices``
    #: with ``replicas_per_device`` (see :func:`run`).
    #: ``None`` below the root.
    timing: Any = None

    def __getitem__(self, path):
        if not isinstance(path, str):
            return tuple.__getitem__(self, path)
        node, rest = self, path
        while rest:
            names = rest.split(SEPARATOR)
            for i in range(1, len(names) + 1):
                head = SEPARATOR.join(names[:i])
                if head in node.parts:
                    node, rest = node.parts[head], SEPARATOR.join(names[i:])
                    break
            else:
                raise KeyError(
                    f"{path!r} is no node of this trajectory at {rest!r}; "
                    f"the nodes here are {sorted(node.parts)}"
                )
        return node


# --------------------------------------------------------------------------- the nodes


def _reset_keys(key: Array, count: int) -> list[Array]:
    """The keys of the reset calls: node ``j`` takes ``key`` for ``j = 0``, ``fold_in(key, j)``
    after."""
    return [key if j == 0 else jax.random.fold_in(key, j) for j in range(count)]


def _round_keys(key: Array, count: int) -> list[Array]:
    """The keys of one round for ``count`` stepped nodes: node ``j`` steps with key ``j`` and
    realises with key ``count + j``; key ``2 * count`` realises phase one under simultaneous."""
    return list(jax.random.split(key, 2 * count + 1))


@dataclasses.dataclass
class _Node:
    """A stepped node and what ``run`` needs to route to it."""

    name: str
    index: int
    step: Callable
    joint: bool
    reads: dict[str, str]  # slot -> the variable it observes (modalities and driven nodes)
    driven: dict[str, str]  # the control nodes a variable outside the node drives
    inputs: tuple[str, ...]  # the variables a sliced view takes a level of, in order
    chosen: tuple[str, ...]  # the action nodes of the nodes it chooses
    emissions: tuple[str, ...]
    phase_two: bool = False

    @property
    def outs(self) -> tuple[str, ...]:
        return (*self.chosen, *self.emissions)


def _check_order(tick_order: str) -> None:
    if tick_order not in ("simultaneous", "sequential", "async"):
        raise ValueError(
            f"tick_order must be 'simultaneous', 'sequential', or 'async', got {tick_order!r}. "
            "A round is one tick under every order and every stepped node steps once. "
            "simultaneous = the nodes no variable drives step on the round's boundary, then the "
            "driven nodes step on what was just realised; sequential = the nodes no action drives "
            "step in the order of the System's components and each driven node steps as soon "
            "as the last node that drives it has realised, before the sweep returns to nodes made "
            "ready earlier, each reading what stands; async = sequential with a "
            "RANDOM order of the nodes no action drives each round (reproducible from the run "
            "key), driven nodes placed by the same rule, each reading what stands"
        )


def _structure(system: System, params: SystemParams, steps: Mapping[str, Callable]) -> list[_Node]:
    """The stepped nodes of ``system`` in component order, checked: ``steps`` an antichain over
    the System's components that covers every leaf, every modality fed, a joint node holding no
    sampled leaf."""
    if system.subsystems is None:
        raise ValueError(
            "run: this System has no components (it is not a merge or a coupling), so there is "
            "no leaf to give a step; build the System with couple() or merge(), or with_mirror()"
        )
    if not isinstance(steps, Mapping) or not steps:
        raise TypeError(
            "run: steps is a mapping from a leaf (or a non-leaf node planned jointly) to its "
            "step: plan(), sample() or a function step(system, key, obs, carry, params)"
        )
    table = dict(system.subsystems)
    order = list(table)
    leaves = [n for n, sub in table.items() if sub.config]
    unknown = sorted(set(steps) - set(table))
    if unknown:
        raise ValueError(
            f"run: steps names {unknown}, which are no component of this System; its leaves are "
            f"{leaves} and its nodes {sorted(set(order) - set(leaves))}"
        )
    for name, step in steps.items():
        if not callable(step):
            raise TypeError(
                f"run: steps[{name!r}] is a {type(step).__name__}; give plan(), sample() or a "
                "function step(system, key, obs, carry, params)"
            )

    def beneath(node: str) -> list[str]:
        return [n for n in order if n.startswith(f"{node}{SEPARATOR}")]

    for name in steps:
        for below in beneath(name):
            if below not in steps:
                continue
            if isinstance(steps[below], _Sample):
                raise ValueError(
                    f"run: steps names {name!r}, planned as one mind, and {below!r} beneath it "
                    "with sample(): the joint view would take the process's tables as part of its "
                    f"model; step the other leaves of {name!r} instead, and give {below!r} its "
                    "own sample()"
                )
            raise ValueError(
                f"run: steps names {name!r} and {below!r} beneath it; the keys of steps are an "
                f"antichain, so step {name!r} as one node or its parts one by one, not both"
            )
    covered = {
        leaf
        for leaf in leaves
        if any(leaf == k or leaf.startswith(f"{k}{SEPARATOR}") for k in steps)
    }
    missing = [leaf for leaf in leaves if leaf not in covered]
    if missing:
        raise ValueError(
            f"run: steps has no step for the leaf {missing[0]!r}"
            + (f" (nor for {missing[1:]})" if missing[1:] else "")
            + "; every leaf of a closed System is stepped, by its own key or by the key of a "
            "node it lies under"
        )
    observes = dict(system.observes)
    open_slots = [m for m in system.modality_names if m not in observes]
    if open_slots:
        raise ValueError(
            f"run: modality {open_slots[0]!r} is observed by no identification"
            + (f" (nor are {open_slots[1:]})" if open_slots[1:] else "")
            + ", so the System is open; feed it with Identify(<out-port>, "
            f"{open_slots[0]!r}) (an action node or an emission of another component), or "
            "couple the System to its mirror with with_mirror()"
        )
    names = system._names()
    action_of = dict(zip(names.controls, names.actions, strict=True))
    orphans = [u for u, a in action_of.items() if a is None and u not in observes]
    boxed = [u for u in orphans if u in dict(system.black_box)]
    if boxed:
        raise ValueError(
            f"run: {boxed[0]!r} is an in-port of a black box and no variable drives it, so its "
            "step would get no value; identify an action node (or emission) of another component "
            f"with it, Identify(<out-port>, {boxed[0]!r})"
        )
    if orphans:
        raise ValueError(
            f"run: control node {orphans[0]!r} has no action node and no variable drives it, so no "
            "step would give it a value; identify an action node (or emission) of another "
            "component with it, hold it with a Clamp, or, for a model's own internal "
            "identification, close the System with with_mirror()"
        )
    nodes: list[_Node] = []
    for index, name in enumerate(n for n in order if n in steps):
        meta = _subsystem_meta(system, name)
        mods = [system.modality_names[i] for i in meta.modalities]
        chosen = tuple(action_of[u] for u in meta.controls if action_of.get(u) is not None)
        emissions = tuple(meta.emissions)
        inside = set(chosen) | set(emissions)
        joint = not table[name].config
        own = {slot: v for slot, v in driven_nodes(system, name).items() if v in chosen}
        if own and not joint:
            slot = next(iter(own))
            raise ValueError(
                f"run: node {name!r} has its control node {slot!r} driven by its own action node "
                f"{own[slot]!r}, a cycle of length one; drive the node from another component"
            )
        driven = {
            slot: variable
            for slot, variable in driven_nodes(system, name).items()
            if variable not in inside
        }
        reads = {m: observes[m] for m in mods if m in observes}
        reads.update(driven)
        valid = _inputs_of(system, name)
        inputs = tuple(v for v in valid if v not in inside)
        nodes.append(
            _Node(
                name,
                index,
                steps[name],
                joint,
                reads,
                driven,
                inputs,
                chosen,
                emissions,
            )
        )
    return nodes


def _check_black_boxes(system: System, nodes: list[_Node]) -> None:
    """Refuse a step that needs the inside of a black box: ``plan()`` or ``sample()`` on one, a
    step marked ``sliced`` (its view would be cut at a driven level), and a node planned jointly
    with a black box beneath it."""
    boxes = system._black_boxes
    for node in nodes:
        held = [b for b in boxes if b == node.name or b.startswith(f"{node.name}{SEPARATOR}")]
        if not held:
            continue
        if node.joint:
            raise ValueError(
                f"run: steps names {node.name!r}, planned as one mind, and the black box "
                f"{held[0]!r} lies beneath it, whose inside a joint planner would need as part of "
                f"its model; step {held[0]!r} on its own with a custom step, and the other leaves "
                f"of {node.name!r} with theirs"
            )
        if isinstance(node.step, _Plan | _Sample):
            kind = "plan()" if isinstance(node.step, _Plan) else "sample()"
            raise ValueError(
                f"run: steps[{node.name!r}] is {kind}, and {node.name!r} is a black box, declared "
                f"by its ports alone: {kind} would {'plan with' if kind == 'plan()' else 'sample'} "
                "a model it does not state. Give it a custom step(system, key, obs, carry, "
                "params) that returns its out-ports (env_step for a positional environment), or "
                "declare the process as a System of factors and kernels (a world run forward as "
                "a model, stepped by sample())"
            )
        if getattr(node.step, "sliced", False):
            raise ValueError(
                f"run: steps[{node.name!r}] is marked sliced, and {node.name!r} is a black box: "
                "a sliced view cuts a driven node out of the tables that read it, and a black box "
                "states none; drop the mark, and the step reads its in-ports from obs"
            )


def _check_timing(
    nodes: list[_Node],
) -> tuple[list[_Node], Callable[[Sequence[_Node]], list[_Node]]]:
    """Mark phase two (a driven control node whose driver is another node's action node) and
    return ``(phase two in an order that respects the rule, derive)``; refuse a cycle.
    ``derive(sources)`` is the order of one sweep of ``"sequential"`` or ``"async"``: the nodes no
    action drives (the sources) in the order given, and each driven node as soon as the last of
    its drivers has realised, before the sweep returns to nodes made ready earlier; the nodes one
    realisation makes ready keep listing order."""
    owner = {a: n for n in nodes for a in n.chosen}
    drivers: dict[str, set[str]] = {}
    for node in nodes:
        for variable in node.driven.values():
            if variable in owner:
                drivers.setdefault(node.name, set()).add(owner[variable].name)
    by_name = {n.name: n for n in nodes}
    for name, owners in drivers.items():
        by_name[name].phase_two = True
        if name in owners:
            raise ValueError(
                f"run: node {name!r} has a control node driven by its own action node; an "
                "identification of a node's action node with its own control node is a cycle"
            )
    placed: list[_Node] = []
    waiting = [n for n in nodes if n.phase_two]
    done = {n.name for n in nodes if not n.phase_two}
    while waiting:
        ready = next((n for n in waiting if all(o in done for o in drivers.get(n.name, ()))), None)
        if ready is None:
            raise ValueError(
                f"run: the identifications of action nodes with control nodes between "
                f"{sorted(n.name for n in waiting)} form a cycle, so no node can step after the "
                "one that drives it; break the cycle"
            )
        waiting.remove(ready)
        placed.append(ready)
        done.add(ready.name)

    def derive(sources: Sequence[_Node]) -> list[_Node]:
        """The order of one sweep: ``sources`` as given, and after each node the nodes it makes
        ready (the driven nodes whose last driver it was), in listing order, each followed by the
        nodes it makes ready in turn before the next of its siblings, depth first."""
        derived: list[_Node] = []
        done: set[str] = set()
        held = [n for n in nodes if n.phase_two]

        def step(node: _Node) -> None:
            derived.append(node)
            done.add(node.name)
            ready = [n for n in held if all(o in done for o in drivers[n.name])]
            for n in ready:
                held.remove(n)
            for n in ready:
                step(n)

        for source in sources:
            step(source)
        return derived

    return placed, derive


# ---------------------------------------------------------------------------- the views


_JOINT_CONFIG = (
    "policy_len",
    "planner",
    "gamma",
    "num_iters",
    "decomposition",
    "planning_belief",
    "feedback",
    "mixture_schedule",
    "mixture_k",
    "epistemic",
    "terms",
)


def _joint_config(step: Callable) -> dict[str, Any]:
    """The joint view's configuration a step's ``replace`` gives (``plan(replace=...)``)."""
    return {k: v for k, v in getattr(step, "replace", {}).items() if k in _JOINT_CONFIG}


def _take_view(system: System, node: "_Node", home: SystemParams, levels: tuple):
    """The view of a stepped node taken from ``home``: sliced at ``levels`` (pairs of a variable
    and its level, an ``int``, a traced integer scalar, a probability vector, or ``None`` for the
    uniform average) for a step marked ``sliced``, unsliced otherwise.

    A view is taken every turn from the home as it stands. Inside the scanned round that is a
    cost of the trace alone: the view's System is the same for every level (its tables differ
    only by the gathers at the level, a traced index), so one System serves every level and no
    view is cached (design batched-run.md R4)."""
    inputs = dict(levels) if levels else None
    if node.joint:
        return joint_planner(system, home, node.name, inputs=inputs, **_joint_config(node.step))
    return subsystem(system, home, node.name, inputs=inputs)


def _reset_view(system: System, node: "_Node", home: SystemParams):
    """The view of a node's reset call: averaged uniformly over its driven inputs, or, where
    ``subsystem`` refuses that average with a :class:`MixtureRefusal` (a Gaussian target or a
    linear-Gaussian transition conditional on a driven node would be a mixture), sliced at the
    first level of each input. A reset call reads no table, so the level is immaterial."""
    try:
        return _take_view(system, node, home, _uniform(node))
    except MixtureRefusal:
        levels = _uniform(node)
        if not levels:
            raise
        return _take_view(system, node, home, tuple((v, 0) for v, _ in levels))


def _uniform(node: "_Node") -> tuple:
    """The input levels of a node's first view (nothing realised yet: each averaged uniformly)."""
    if not getattr(node.step, "sliced", False):
        return ()
    return tuple((v, None) for v in node.inputs)


def _level_of(value, size: int | None = None, dtype=None) -> "int | Array | None":
    """The input a sliced view takes for a variable's value: an ``int`` for a concrete level, the
    traced integer scalar itself inside the scanned round, the vector itself otherwise, ``None``
    before the variable has a value. A :class:`Maybe` (a variable the reset left without a
    value, read before it is realised in round 0) is a probability vector over the ``size``
    levels every round: the level's one-hot where it is present, uniform where it is not (the
    average the view takes of a variable with no value), in ``dtype`` (that of the tables the
    view averages; the default float dtype when ``None``) or the vector's own."""
    if value is None:
        return None
    if isinstance(value, Maybe):
        level = jnp.asarray(value.value)
        probs = (
            jax.nn.one_hot(level, size, dtype=dtype or float)
            if level.ndim == 0
            else as_float(level)
        )
        return jnp.where(value.present, probs, jnp.full((size,), 1.0 / size, probs.dtype))
    array = jnp.asarray(value)
    if array.ndim == 0 and jnp.issubdtype(array.dtype, jnp.integer):
        return array if isinstance(array, jax.core.Tracer) else int(array)
    return array


def _null(kind_size: tuple[str, int], dtype=float) -> Array:
    """The value of no action: level 0 (int32), or a float vector of zeros in ``dtype``."""
    kind, size = kind_size
    if kind == LEVEL:
        return jnp.zeros((), jnp.int32)
    return jnp.zeros((size,), dtype)


# ---------------------------------------------------------------------------- the clocks


class _Clock(NamedTuple):
    """A clocked node's hold schedule, read and validated in the structure pass: the action node
    whose intended level selects the option, the rounds an option lasts at each of its levels,
    and, where every level lasts the same number of rounds, that number (the schedule is then a
    function of the round alone: the node decides on the rounds that are multiples of it)."""

    action: str
    rounds: tuple[int, ...]
    lockstep: int | None


def _check_clocks(
    clocks: Mapping[str, float] | None,
    names: set[str],
    clocked: set[str],
    boxes: Sequence[str] = (),
):
    """The unit conversion of each clock, a positive finite number of rounds per unit of
    duration, defaulting to 1.0 for a node that declares durations; a key that is no stepped node
    or a node that declares no durations is refused."""
    if clocks is not None and not isinstance(clocks, Mapping):
        raise TypeError(
            f"run: clocks is a {type(clocks).__name__}; pass a mapping from a stepped node to the "
            "rounds one unit of duration lasts"
        )
    clocks = dict(clocks or {})
    unknown = sorted(set(clocks) - names)
    if unknown:
        raise ValueError(
            f"run: clocks names {unknown}, which are no stepped node; the stepped nodes are "
            f"{sorted(names)}"
        )
    idle = sorted(set(clocks) - clocked)
    boxed = [n for n in idle if n in boxes]
    if boxed:
        raise ValueError(
            f"run: clocks names {boxed}, a black box, which declares no durations and so holds no "
            "option to convert; a black box paces itself: hold its out-ports in its own step "
            "(return the standing values from its carry), and drop the entry"
        )
    if idle:
        raise ValueError(
            f"run: clocks names {idle}, whose params declare no durations, so there is no option "
            "to convert; declare durations for the node's control node, or drop the entry"
        )
    return {n: _rounds_per_unit(clocks.get(n, 1.0), f"run: clocks[{n!r}]") for n in clocked}


def _clock_of(system: System, node: "_Node", home: SystemParams, scale: float) -> _Clock:
    """The hold schedule of a clocked node: the duration of each level of the control node that
    carries ``params.durations`` (read off the node's reset view, as its every view declares
    it), times ``scale`` rounds per unit, each a whole number of rounds of at least one, refused
    otherwise with ``hold``'s arithmetic and sentences (design batched-run.md R2, R5)."""
    view, vparams = _reset_view(system, node, home)
    ((control, table),) = view._durations(view._params_in(vparams, "run"), "run").items()
    action = view.action_names[view._table.control_names.index(control)]
    what = f"run: the duration of control node {control!r} of {node.name!r}"
    clock = f"its clock of {scale} rounds per unit"
    rounds = tuple(
        _option_rounds(float(unit), scale, what, clock, level=level)
        for level, unit in enumerate(np.asarray(table).tolist())
    )
    return _Clock(action, rounds, rounds[0] if len(set(rounds)) == 1 else None)


def _declared_durations(params: SystemParams, name: str) -> dict:
    """The durations a view of the stepped node ``name`` declares, keyed by control node, read
    from the home with no view built (a view at no realised level may not exist): with
    ``subsystem``'s precedence, the System's own ``params.durations`` where it holds any (its
    entries on nodes at or beneath ``name``), else the ``parts`` of the leaves at or beneath it."""

    def beneath(key: str) -> bool:
        return key == name or key.startswith(f"{name}{SEPARATOR}")

    if params.durations:
        return {node: d for node, d in params.durations.items() if beneath(node)}
    out: dict = {}
    for leaf, part in (params.parts or {}).items():
        if beneath(leaf):
            out.update(part.durations or {})
    return out


# ---------------------------------------------------------------------------- the round


class _State(NamedTuple):
    """What a round hands the next: the run key, the value of every variable, each stepped
    node's carry, the parameter home, each clocked node's option in force (the rounds left to
    hold after this one, an int32 array, and the intended action it chose), the key the async
    order is drawn from, and, for each variable the reset left without a value, whether it has
    one (a bool array). Straight from the reset a variable with no value is ``None``; the state
    settled from the reset (:func:`_settle_static`, :func:`_settle`) holds zeros of its form
    there, its flag false."""

    key: Array
    values: dict
    carries: dict
    home: SystemParams
    held: dict
    sweep: Array
    present: dict


def _traced(x) -> bool:
    return isinstance(x, jax.core.Tracer)


def _select(flag, new, old):
    """``new`` where ``flag`` holds, else ``old``, leaf by leaf; a concrete ``flag`` picks one
    side in Python, so a concrete round computes nothing it discards."""
    if not _traced(flag):
        return new if bool(flag) else old
    return jax.tree_util.tree_map(lambda a, b: jnp.where(flag, a, b), new, old)


def _like(value, reference, what: str, settling: bool = False):
    """``value`` in the dtype of ``reference`` (the value it replaces in the scan's state), which
    it must match in shape: a port, a carry and the home keep their structure from round to
    round, since they are the state of the scan over rounds (design batched-run.md R3, R7). A
    float is only widened into the state, which :func:`_settle` settles at the widest float dtype
    round 0 returns (a state settled without a trace that a round would widen is settled again
    by :func:`_settle`): a float array wider than the state is refused (kept as it is while
    ``settling``, in round 0 evaluated outside the scan, whose dtypes settle the state), a Python
    number takes the state's dtype."""
    if reference is None or not hasattr(reference, "dtype"):
        return value
    array = jnp.asarray(value)
    if array.shape != jnp.shape(reference):
        raise ValueError(
            f"run: {what} has shape {tuple(array.shape)} this round and "
            f"{tuple(jnp.shape(reference))} the round before; a value keeps its shape from round "
            "to round (each round is one step of a scan), so return a value of the same shape "
            "every round"
        )
    if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
        return array
    dtype = jnp.result_type(reference)
    if (
        jnp.issubdtype(array.dtype, jnp.floating)
        and jnp.issubdtype(dtype, jnp.floating)
        and not _weak(array)
        and jnp.finfo(array.dtype).bits > jnp.finfo(dtype).bits
    ):
        if settling:
            return array
        raise ValueError(
            f"run: {what} is {array.dtype.name} this round and {jnp.dtype(dtype).name} the "
            "round before; a value keeps its dtype from round to round (each round is one step "
            "of a scan), and a model, a rate or an input in a wider float dtype than the rest "
            "can first widen it in a later round, so give them one float dtype "
            "(cast(params, dtype) casts the model)"
        )
    return jax.lax.convert_element_type(array, dtype)


def _weak(x) -> bool:
    """Whether ``x`` is weakly typed (a Python number made an array, which takes the dtype of the
    arrays beside it)."""
    return bool(getattr(getattr(x, "aval", x), "weak_type", False))


def _widened(x, t):
    """A reset's leaf ``x`` in the dtype it keeps from round to round, given ``t`` (the leaf
    round 0 returns, abstract): a float array at the wider of its dtype and ``t``'s, a Python
    number or weakly typed float at ``t``'s; anything else as it is."""
    dtype = getattr(t, "dtype", None)
    if dtype is None or not jnp.issubdtype(dtype, jnp.floating):
        return x
    if not (isinstance(x, float | np.ndarray | np.generic) or eqx.is_array(x)):
        return x
    array = jnp.asarray(x)
    if not jnp.issubdtype(array.dtype, jnp.floating):
        return x
    if _weak(array):
        return jnp.asarray(array, dtype)
    return jnp.asarray(array, jnp.promote_types(array.dtype, dtype))


def _like_tree(tree, reference, what: str):
    """``tree`` with each array leaf in the dtype of ``reference``'s (see :func:`_like`), the two
    of one structure."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    ref_leaves, ref_def = jax.tree_util.tree_flatten(reference)
    if treedef != ref_def:
        raise ValueError(
            f"run: {what} changed structure between rounds ({ref_def} to {treedef}); each round "
            "is one step of a scan, whose state keeps one structure: return a carry of the "
            "structure it was handed (fields filled from the first round on), and the params "
            "handed in or learned ones of the same structure"
        )
    paths = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(tree)[0]]
    return treedef.unflatten(
        [
            _like(x, r, f"{what} at {path}" if path else what)
            if hasattr(r, "dtype") and hasattr(x, "dtype")
            else x
            for x, r, path in zip(leaves, ref_leaves, paths, strict=True)
        ]
    )


def _strong(tree):
    """``tree`` with Python and NumPy numbers made JAX arrays and every weakly typed array made
    strongly typed: the scan's state starts in the dtypes its body returns."""

    def leaf(x):
        if isinstance(x, bool | int | float | complex | np.ndarray | np.generic):
            x = jnp.asarray(x)
        if isinstance(x, Array) and not jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            return jnp.asarray(x, x.dtype)
        return x

    return jax.tree_util.tree_map(leaf, tree)


_HOST_ERRORS = (
    jax.errors.ConcretizationTypeError,
    jax.errors.TracerArrayConversionError,
    jax.errors.TracerBoolConversionError,
    jax.errors.TracerIntegerConversionError,
)


class _Round:
    """One round's working values: the variables, the carries and the home as they stand, the
    values at the round's boundary, the write-backs held for it, each node's record of the round
    and the faults it raised."""

    def __init__(self, state: _State, t, scanned: bool = False):
        self.scanned = scanned
        self.values = dict(state.values)
        self.boundary = dict(state.values)
        #: whether each variable the reset left without a value has one: a flag of the state
        #: (traced), or ``True`` once realised this round
        self.present = dict(state.present)
        self.boundary_present = dict(state.present)
        self.carries = dict(state.carries)
        self.home = state.home
        self.round_home = state.home
        self.held = dict(state.held)
        self.t = t
        self.pending: list = []
        self.received: dict = {}
        self.intended: dict = {}
        self.decided: dict = {}
        self.faults: dict = {}

    def flag(self, bad, label: str, sentence) -> None:
        """A check of a value: on a concrete value (the reset) a failure is refused with
        ``sentence``; on a traced one (every round) it is recorded under ``label`` in
        ``Trajectory.faults`` and the round goes on (design batched-run.md R6, R7)."""
        _flagged(self.faults, bad, label, sentence)


def _flagged(faults: dict | None, bad, label: str, sentence) -> None:
    """``_Round.flag`` over a faults mapping; ``None`` refuses a traced failure too."""
    if not _traced(bad):
        if bool(bad):
            raise ValueError(sentence() if callable(sentence) else sentence)
        return
    if faults is None:
        raise ValueError(sentence() if callable(sentence) else sentence)
    faults[label] = faults[label] | bad if label in faults else jnp.asarray(bad)


class _Engine:
    """The round of ``run``: a function ``round(state, t) -> (state, record)`` of the structure
    pass's static facts (the stepped nodes, the groups, the order, the clocks, the action nodes
    and their kernels), scanned over every round (under ``batch=``, vmapped over the replicas),
    and evaluated abstractly on the state straight from the reset (round 0) where the state's
    form is settled by a trace (:func:`_settle`)."""

    def __init__(
        self, system, nodes, second, derive, tick_order, kernels, types, clocks, groups=None
    ):
        self.system = system
        self.nodes = nodes
        self.first = [n for n in nodes if not n.phase_two]
        self.second = second
        self.derive = derive
        self.tick_order = tick_order
        self.count = len(nodes)
        self.action_nodes = [a for a in system.action_names if a is not None]
        self.index = {a: i for i, a in enumerate(self.action_nodes)}
        self.kernels = kernels
        self.types = types
        self.clocks = clocks
        #: each group (design batched-run.md R12) by its first member: ``(members, swaps)``,
        #: ``swaps[i]`` renaming member ``i``'s names to the first member's and back
        self.groups: dict[str, tuple[list[_Node], list[Callable]]] = dict(groups or {})
        self.member_of = {m.name: lead for lead, (ms, _) in self.groups.items() for m in ms}
        #: the units of the order: each node, or a group as one node
        self.units = self._units(self.first)
        self.sequential = list(derive(self.first))
        if tick_order == "sequential":
            self._check_contiguous(self.sequential, "the sequential order")
        if tick_order == "simultaneous":
            self._check_contiguous(self.second, "the order of the driven nodes")
        #: whether any turn traced so far wrote back (a fact of the trace, read after the run:
        #: a run in which no step wrote back returns the starting home itself)
        self.wrote = False
        #: set while the scan of a statically settled state is traced: a round's variable or
        #: carry of another form than the state's raises :class:`_Unformed` (:func:`_exact`)
        #: where it would be converted
        self.strict = False
        self.orders, self.permutations, self.order_of = [], None, None

    def _units(self, nodes: Sequence[_Node]) -> list[list[_Node]]:
        """``nodes`` as the units of an order: a group's members at its first member's place,
        every other node on its own."""
        units = []
        for node in nodes:
            lead = self.member_of.get(node.name)
            if lead is None:
                units.append([node])
            elif lead == node.name:
                units.append(list(self.groups[lead][0]))
        return units

    def _check_contiguous(self, order: Sequence[_Node], what: str) -> None:
        """Refuse a group whose members do not step one after another in ``order``."""
        names = [n.name for n in order]
        for members, _ in self.groups.values():
            at = sorted(names.index(m.name) for m in members if m.name in names)
            if at and at[-1] - at[0] + 1 != len(at):
                group = tuple(m.name for m in members)
                between = [names[i] for i in range(at[0], at[-1]) if names[i] not in group]
                raise ValueError(
                    f"run: the group {group} is not contiguous in {what} ({names}): "
                    f"{between} steps between its members, a node one of them drives or that "
                    "drives one of them; a group steps as one node, its members simultaneous "
                    "among themselves, so its members must step one after another: step them "
                    'ungrouped, or run the System under "simultaneous"'
                )

    def async_orders(self, batch: int | None) -> None:
        """Under ``"async"``: every distinct derived order of a permutation of the units of the
        sources (a group is one unit), the permutations, and the index of each permutation's
        derived order. Under ``batch=`` each replica draws its own order and the switch over
        the orders runs every branch (vmap lowers it to a select), so more than
        :data:`_ASYNC_SOURCES` source units are refused (design batched-run.md R9)."""
        if self.tick_order != "async":
            return
        units = self.units
        if batch is not None and len(units) > _ASYNC_SOURCES:
            raise ValueError(
                f"run: under batch= the async order is drawn for each replica and dispatched by "
                "a switch over every distinct derived order, which vmap evaluates in full each "
                f"round (the cost is the number of orders times the step work); this System has "
                f"{len(units)} sources, up to {math.factorial(len(units))} orders, above the "
                f"ceiling of {_ASYNC_SOURCES} sources ({math.factorial(_ASYNC_SOURCES)} orders): "
                'run it under "sequential" or "simultaneous", group identical sources into '
                "one node (steps={(a, b): step}), or run the replicas one call each without "
                "batch="
            )
        orders: list[list[_Node]] = []
        seen: dict[tuple, int] = {}
        permutations, order_of = [], []
        for perm in itertools.permutations(range(len(units))):
            derived = self.derive([n for i in perm for n in units[i]])
            self._check_contiguous(derived, "an async order")
            names = tuple(n.name for n in derived)
            if names not in seen:
                seen[names] = len(orders)
                orders.append(derived)
            permutations.append(perm)
            order_of.append(seen[names])
        self.orders = orders
        self.permutations = np.asarray(permutations, np.int32)
        self.order_of = np.asarray(order_of, np.int32)

    # --------------------------------------------------------------------- reads

    def stands(self, node: _Node, variable: str) -> bool:
        """Whether ``node`` reads ``variable`` as it stands now rather than at the round's
        boundary. Under ``"sequential"`` and ``"async"`` every read stands; under
        ``"simultaneous"`` only a control node driven by an action node does, reading the value
        realised this round."""
        return self.tick_order != "simultaneous" or (
            variable in self.index and variable in node.driven.values()
        )

    def value(self, node: _Node, variable: str, now, at_boundary) -> Any:
        """``variable`` as ``node`` reads it, from ``now`` or ``at_boundary`` (each a pair of the
        values and the presence flags): its value, a :class:`Maybe` where whether it has one is a
        flag of the state (a variable the reset left without a value, not yet realised this
        round), or ``None`` straight from the reset."""
        values, present = now if self.stands(node, variable) else at_boundary
        value, flag = values.get(variable), present.get(variable, True)
        return value if flag is True or value is None else Maybe(value, flag)

    def read(self, node: _Node, now, at_boundary) -> dict:
        return {slot: self.value(node, v, now, at_boundary) for slot, v in node.reads.items()}

    # --------------------------------------------------------------------- the round

    def round(self, state: _State, t, scanned: bool = False):
        """Round ``t``: ``(state, record)``. ``t`` is the Python int 0 for the abstract round 0
        that settles the state (``scanned`` false: a carry may still change form), a traced one
        in the scan (``scanned``)."""
        key, k_tick = jax.random.split(state.key)
        keys = _round_keys(k_tick, self.count)
        state = state._replace(key=key)
        if self.tick_order == "simultaneous":
            return self._sweep(state, keys, t, None, scanned)
        if self.tick_order == "sequential":
            return self._sweep(state, keys, t, self.sequential, scanned)
        perm = jax.random.permutation(jax.random.fold_in(state.sweep, t), len(self.units))
        if not _traced(perm):
            match = np.all(self.permutations == np.asarray(perm), axis=1)
            order = self.orders[int(self.order_of[np.argmax(match)])]
            return self._sweep(state, keys, t, order, scanned)
        match = jnp.all(jnp.asarray(self.permutations) == perm, axis=1)
        which = jnp.asarray(self.order_of)[jnp.argmax(match)]
        branches = [
            functools.partial(self._sweep, t=t, order=order, scanned=scanned)
            for order in self.orders
        ]
        if len(branches) == 1:
            return branches[0](state, keys)
        return jax.lax.switch(which, _alike(branches, state, keys), state, keys)

    def _turns(self, order: Sequence[_Node]) -> list[list[_Node]]:
        """``order`` as its turns: a group's members together at its first member's place."""
        turns = []
        for node in order:
            lead = self.member_of.get(node.name)
            if lead is None:
                turns.append([node])
            elif lead == node.name:
                turns.append(list(self.groups[lead][0]))
        return turns

    def _sweep(self, state: _State, keys, t, order, scanned=False):
        """One round under ``order`` (``None``: simultaneous)."""
        r = _Round(state, t, scanned)
        count = self.count
        if order is None:
            chosen: dict[str, Array] = {}
            flags: dict[str, Any] = {}
            for turn in self._turns(self.first):
                for picked, decided in self._turn_of(r, turn, keys):
                    chosen.update(picked)
                    flags.update(dict.fromkeys(picked, decided))
            members = [a for n in self.first for a in n.chosen]
            self._realise(r, keys[2 * count], chosen, flags, members)
            for turn in self._turns(self.second):
                results = self._turn_of(r, turn, keys)
                for node, (picked, decided) in zip(turn, results, strict=True):
                    self._realise(
                        r,
                        keys[count + node.index],
                        picked,
                        dict.fromkeys(picked, decided),
                        node.chosen,
                    )
            if r.pending:
                r.home = _at_the_boundary(self.system, r.round_home, r.pending, t, r.flag)
        else:
            for turn in self._turns(order):
                results = self._turn_of(r, turn, keys)
                for node, (picked, decided) in zip(turn, results, strict=True):
                    self._realise(
                        r,
                        keys[count + node.index],
                        picked,
                        dict.fromkeys(picked, decided),
                        node.chosen,
                    )
        received = {
            n: {s: x.value if isinstance(x, Maybe) else x for s, x in seen.items()}
            for n, seen in r.received.items()
        }
        present = {
            n: {
                s: x.present if isinstance(x, Maybe) else jnp.asarray(x is not None)
                for s, x in seen.items()
            }
            for n, seen in r.received.items()
        }
        record = dict(
            received=received,
            present=present,
            intended=r.intended,
            decided=r.decided,
            emitted={n.name: {v: r.values[v] for v in n.outs} for n in self.nodes},
            carries=dict(r.carries),
            faults=r.faults,
        )
        flags = {v: jnp.asarray(r.present[v]) for v in state.present}
        return _State(state.key, r.values, r.carries, r.home, r.held, state.sweep, flags), record

    def _turn_of(self, r: _Round, turn: list[_Node], keys) -> list:
        if len(turn) == 1 and turn[0].name not in self.member_of:
            return [self._decide(r, turn[0], keys[turn[0].index])]
        return self._decide_group(r, turn, keys)

    def _realise(self, r: _Round, k, chosen, flags, members) -> None:
        """Realise the intended action nodes ``chosen`` of ``members`` from their kernels with
        key ``k``; an action node whose node held (its flag false) keeps its realised value."""
        draws = [
            (self.index[a], self.kernels[a], chosen[a])
            for a in members
            if a in self.kernels and a in chosen
        ]
        drawn = _actuate(k, len(self.action_nodes), draws)
        for a in members:
            if a in chosen:
                new = drawn.get(self.index[a], chosen[a])
                r.values[a] = (
                    _select(flags[a], new, r.values[a]) if r.values[a] is not None else new
                )
                # realised, or held after an earlier decision (round 0 always decides)
                r.present[a] = True

    def _base(self, r: _Round) -> SystemParams:
        """The home a turn's view is taken from: the round's start under ``"simultaneous"``, the
        home as it stands otherwise."""
        return r.round_home if self.tick_order == "simultaneous" else r.home

    def _prepare(self, r: _Round, node: _Node, base):
        """What ``node`` reads this turn (recorded), the levels its sliced view is cut at, its
        view and the observations its step is handed."""
        now, at_boundary = (r.values, r.present), (r.boundary, r.boundary_present)
        seen = self.read(node, now, at_boundary)
        r.received[node.name] = seen
        levels: tuple = ()
        if getattr(node.step, "sliced", False):
            dtype = table_dtype(base)
            levels = tuple(
                (v, _level_of(self.value(node, v, now, at_boundary), self.types[v][1], dtype))
                for v in node.inputs
            )
        view, vparams = _take_view(self.system, node, base, levels)
        cut = {s for s, v in node.driven.items() if v in {w for w, _ in levels}}
        obs = {s: x for s, x in seen.items() if s not in cut}
        return view, vparams, levels, obs

    def _decide(self, r: _Round, node: _Node, k):
        """``node``'s turn: stepped on what it reads, or, a clocked node inside its option,
        holding (its out-ports keep their values, its count goes one round down); its return
        checked, its emissions stored and its write-back applied (live) or held for the boundary
        (simultaneous). Returns the intended action nodes and whether the node decided."""
        base = self._base(r)
        view, vparams, levels, obs = self._prepare(r, node, base)
        identity: dict[str, bool] = {}
        out, stepped, learned, held, decided = self._turn(
            node,
            view,
            r.t,
            r.scanned,
            identity,
            k,
            obs,
            r.carries[node.name],
            vparams,
            {v: r.values[v] for v in node.outs},
            r.held.get(node.name),
        )
        if identity.get("params"):
            learned = vparams  # both branches hand back the params they were given
        chosen = self._finish(r, node, out, stepped, learned, held, decided, view, vparams, levels)
        return chosen, decided

    def _turn(self, node, view, t, scanned, identity, k, obs, carry, vparams, standing, held):
        """The step of ``node`` called on its view, or, a clocked node inside its option, its
        standing values: ``(out, carry, params, held, decided)``, ``held`` the option in force
        after the turn (``None`` for a node with no clock). A function of its arrays alone, so
        a group vmaps it over its members."""

        def act(k, obs, carry, vparams):
            out, stepped, learned = self._call(node, view, k, obs, carry, vparams)
            if not isinstance(learned, SystemParams):
                raise ValueError(
                    f"run: the step of {node.name!r} returned {type(learned).__name__} as its "
                    "params; a step returns the SystemParams it was given, or learned ones"
                )
            strict = scanned and self.strict
            out = _check_out(
                node,
                out,
                self.types,
                self.system,
                False,
                None if strict else standing,
                settling=not scanned,
            )
            identity["params"] = learned is vparams
            if strict:
                ports = [k for k in out if k in standing]
                _exact({k: out[k] for k in ports}, {k: standing[k] for k in ports})
                _exact(_strong(stepped), carry)
            if scanned:
                stepped = _like_tree(_strong(stepped), carry, f"the carry of {node.name!r}")
                if learned is not vparams:
                    learned = _like_tree(learned, vparams, f"the params of {node.name!r}")
            return out, stepped, learned

        clock = self.clocks.get(node.name)
        if clock is None:
            out, stepped, learned = act(k, obs, carry, vparams)
            return out, stepped, learned, None, True
        remaining, action = held
        if isinstance(t, int) and t == 0:
            decided = True  # the first round always decides (its count is traced under batch=)
        else:
            decided = (t % clock.lockstep == 0) if clock.lockstep else (remaining == 0)
        if not _traced(decided):
            decided = bool(decided)

        def hold(k, obs, carry, vparams):
            return {**{e: standing[e] for e in node.emissions}, **action}, carry, vparams

        if decided is True:
            out, stepped, learned = act(k, obs, carry, vparams)
        elif decided is False:
            out, stepped, learned = hold(k, obs, carry, vparams)
        else:
            try:
                out, stepped, learned = jax.lax.cond(decided, act, hold, k, obs, carry, vparams)
            except TypeError as error:
                raise ValueError(
                    f"run: the clocked node {node.name!r} decides or holds by a traced flag, and "
                    "its step returns a carry of another structure or shape than it was handed, "
                    "so the two branches differ; return a carry of the form it was handed every "
                    "round, from the reset on"
                ) from error
        chosen = {a: out[a] for a in node.chosen}
        level = jnp.asarray(chosen[clock.action], jnp.int32)
        rounds = jnp.asarray(clock.rounds, jnp.int32)[level]
        held = (
            _select(decided, rounds - 1, remaining - 1).astype(jnp.int32),
            _select(decided, chosen, action),
        )
        return out, stepped, learned, held, decided

    def _finish(
        self, r, node, out, stepped, learned, held, decided, view, vparams, levels, pending=None
    ):
        """A turn's return taken into the round: its levels' ranges checked, its carry, option
        and emissions stored, its intended actions and decision recorded and its write-back
        applied or held (``pending``: for the end of a group's turn). Returns the intended
        action nodes."""
        _check_ranges(node, out, self.types, self.system, r.flag)
        r.carries[node.name] = stepped
        if held is not None:
            r.held[node.name] = held
        for e in node.emissions:
            r.values[e] = out[e]
            r.present[e] = True
        chosen = {a: out[a] for a in node.chosen}
        r.intended[node.name] = chosen
        r.decided[node.name] = decided
        if learned is not vparams:
            self._write(r, node, view, vparams, learned, levels, self._base(r), pending)
        return chosen

    def _decide_group(self, r: _Round, members: list[_Node], keys) -> list:
        """A group's turn (design batched-run.md R12): its members' steps called by one vmapped
        call of the step over their stacked keys, observations, carries, views' params and
        options, each member renamed to the first's names (the members' views are one view up
        to their names, checked in the structure pass). The members read the home as the turn
        began and their write-backs are summed at its end (Dirichlet counts; at the round's
        boundary under ``"simultaneous"``), as simultaneous nodes' are. Returns each member's
        intended action nodes and decision."""
        lead = members[0]
        swaps = self.groups[lead.name][1]
        base = self._base(r)
        prepared = [self._prepare(r, m, base) for m in members]
        rows = [
            swaps[i](
                (
                    keys[m.index],
                    prepared[i][3],
                    r.carries[m.name],
                    prepared[i][1],
                    {v: r.values[v] for v in m.outs},
                    r.held.get(m.name),
                )
            )
            for i, m in enumerate(members)
        ]
        stacked = _stack_members(rows, members, r.t)
        view = prepared[0][0]
        identity: dict[str, bool] = {}

        def turn(row):
            return self._turn(lead, view, r.t, r.scanned, identity, *row)

        outs = eqx.filter_vmap(turn)(stacked)
        pending: list | None = None if self.tick_order == "simultaneous" else []
        results = []
        for i, m in enumerate(members):
            out, stepped, learned, held, decided = swaps[i](_row_of(outs, i))
            _, vparams, levels, _ = prepared[i]
            if identity.get("params"):
                learned = vparams
            chosen = self._finish(
                r, m, out, stepped, learned, held, decided, prepared[i][0], vparams, levels, pending
            )
            results.append((chosen, decided))
        if pending:
            r.home = _at_the_boundary(self.system, base, pending, r.t, r.flag)
        return results

    def _call(self, node: _Node, view, k, obs, carry, vparams):
        """The step called; on traced values (the scan, or any round under ``batch=``) a step
        that calls host-side code is refused naming it (design batched-run.md R8)."""
        try:
            return node.step(view, k, obs, carry, vparams)
        except _HOST_ERRORS as error:
            if not _traced(k):
                raise
            raise ValueError(
                f"run: the step of {node.name!r} calls host-side code on a traced value "
                f"({type(error).__name__}); run steps every round inside one compiled scan "
                "(and every call under batch=, the reset included, vmapped over the replicas), "
                "so a step is a traced function whose outputs are JAX arrays: "
                "write it with jax.numpy and lax control flow (no np.asarray, float(), int(), "
                "bool() or a Python branch on a value), and keep host-side work such as timing "
                "or logging outside run (record what a step sees in its carry, "
                "record_carries=True)"
            ) from error
        except (TypeError, AttributeError) as error:
            unset = [s for s, v in (obs or {}).items() if isinstance(v, Maybe)]
            if not unset or isinstance(node.step, _Plan | _Sample):
                raise
            raise ValueError(
                f"run: the step of {node.name!r} failed ({type(error).__name__}: {error}) and "
                f"was handed obs[{unset[0]!r}] as a Maybe(value, present): its variable has no "
                "value at the reset and the step reads it before it is realised in round 0, so "
                "every round hands it the value with a flag (present false in round 0); read "
                f"obs[{unset[0]!r}].value where obs[{unset[0]!r}].present (jnp.where), or have "
                "the variable's owner return it at its reset (obs is None)"
            ) from error

    def _write(self, r: _Round, node, view, vparams, learned, levels, base, pending=None) -> None:
        """A learner's write-back: its own learned tables and its parameter variables, applied to
        the home now (sequential, async), held for the boundary (simultaneous), or held for the
        end of its group's turn (``pending``)."""
        self.wrote = True
        learned, written = _own_slots(view, node, vparams, learned, levels, base, r.flag)
        _check_parameters_only(view, node.name, vparams, learned, r.flag)
        new = _write_back(self.system, base, node.name, vparams, learned, r.flag)
        if self.tick_order == "simultaneous":
            r.pending.append((node.name, new, written))
        elif pending is not None:
            pending.append((node.name, new, written))
        else:
            r.home = new if not written else dataclasses.replace(new, **written(new))


def _alike(branches: list, state, keys) -> list:
    """The async round's branches (one per derived order) made to return records of one
    structure, for a switch whose index is traced: every branch records every fault label
    (``False`` where it raised none), and, straight from the reset (the settling pass, where a
    variable may still be ``None``), a slot one order reads before its variable has a value and
    another after is recorded as zeros where absent."""
    shapes = [eqx.filter_eval_shape(b, state, keys)[1] for b in branches]
    received: dict[str, dict[str, Any]] = {}
    for shape in shapes:
        for node, slots in shape["received"].items():
            for slot, value in slots.items():
                held = received.setdefault(node, {})
                if held.get(slot) is None:
                    held[slot] = value
    labels = list(dict.fromkeys(label for shape in shapes for label in shape["faults"]))

    def alike(branch):
        def call(state, keys):
            out, record = branch(state, keys)
            seen = {}
            for node, slots in received.items():
                seen[node] = {}
                for slot, like in slots.items():
                    value = record["received"][node].get(slot)
                    if value is None and like is not None:
                        value = jnp.zeros(like.shape, like.dtype)
                    seen[node][slot] = value
            faults = {
                label: jnp.asarray(record["faults"].get(label, False), bool) for label in labels
            }
            return out, {**record, "received": seen, "faults": faults}

        return call

    return [alike(b) for b in branches]


#: The most source units ``"async"`` takes under ``batch=`` (design batched-run.md R9): each
#: replica's order is a switch over the distinct derived orders, all of which vmap runs.
_ASYNC_SOURCES = 4


def _stack_members(rows: list, members: list[_Node], t):
    """The rows of a group's members, each renamed to the first member's names, stacked on a
    leading axis: their arrays stacked, their other leaves equal. Refused, naming the members,
    where the rows differ in structure (a value one member has and another has not yet)."""
    parts = [eqx.partition(row, eqx.is_array) for row in rows]
    defs = [jax.tree_util.tree_structure(d) for d, _ in parts]
    statics = [s for _, s in parts]
    for i, m in enumerate(members[1:], start=1):
        same = defs[i] == defs[0] and _same_static(statics[i], statics[0])
        same = same and all(
            jnp.shape(a) == jnp.shape(b) and jnp.result_type(a) == jnp.result_type(b)
            for a, b in zip(
                jax.tree_util.tree_leaves(parts[i][0]),
                jax.tree_util.tree_leaves(parts[0][0]),
                strict=True,
            )
        )
        if not same:
            raise ValueError(
                f"run: the group {tuple(n.name for n in members)} steps its members by one "
                f"vmapped call, and at round {t} what {m.name!r} is handed (its observations, "
                f"carry, view's params or option) differs in structure or shape from what "
                f"{members[0].name!r} is handed; a group holds copies of one node that read "
                "values of one form: give every member the same kind of input (a value present "
                "for all or none) and carries of one structure, or step them ungrouped"
            )
    dynamic = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[d for d, _ in parts])
    return eqx.combine(dynamic, statics[0])


def _row_of(tree, i: int):
    """Entry ``i`` of a tree stacked on a leading axis (its other leaves as they are)."""
    return jax.tree_util.tree_map(lambda x: x[i] if eqx.is_array(x) else x, tree)


def _swapper(a: str, b: str, keep: frozenset, pairs: dict | None = None) -> Callable:
    """A renaming of every name in a tree from component ``a``'s to ``b``'s and back (the swap
    of the two prefixes, an involution), the names in ``keep`` (the parameter variables both
    views share) left as they are and the names in ``pairs`` (the ports of the members'
    partners, paired by ``_partner_pairs``; closed under inversion) taken to their pair before
    any prefix is read: dict keys, string leaves and the strings in a dataclass's or a module's
    fields, a ``<head>:<name>`` key by its name. Arrays are untouched."""
    pa, pb = f"{a}{SEPARATOR}", f"{b}{SEPARATOR}"
    pairs = pairs or {}

    def name(s: str) -> str:
        if s in pairs:
            return pairs[s]
        if s in keep:
            return s
        if s == a:
            return b
        if s == b:
            return a
        if s.startswith(pa):
            return pb + s[len(pa) :]
        if s.startswith(pb):
            return pa + s[len(pb) :]
        return s

    def string(s: str) -> str:
        head, colon, rest = s.rpartition(":")
        return f"{head}{colon}{name(rest)}" if colon else name(s)

    def go(x):
        if isinstance(x, str):
            return string(x)
        if x is None or isinstance(
            x, jax.core.Tracer | Array | np.ndarray | np.generic | numbers.Number | type
        ):
            return x
        if isinstance(x, tuple) and hasattr(x, "_fields"):
            return type(x)._make(go(v) for v in x)
        if isinstance(x, tuple | list):
            return type(x)(go(v) for v in x)
        if isinstance(x, dict):
            return type(x)((go(k), go(v)) for k, v in x.items())
        if isinstance(x, frozenset | set):
            return type(x)(go(v) for v in x)
        if dataclasses.is_dataclass(x) and hasattr(x, "__dict__"):
            new = object.__new__(type(x))
            for k, v in vars(x).items():
                object.__setattr__(new, k, go(v))
            return new
        return x

    return go


def _same_tree(a, b) -> bool:
    """Whether two trees have one structure (static leaves included) and arrays of one shape and
    dtype: the test ``plan()``'s compiled tick makes to share a program."""
    da, sa = eqx.partition(a, eqx.is_array)
    db, sb = eqx.partition(b, eqx.is_array)
    if jax.tree_util.tree_structure(da) != jax.tree_util.tree_structure(db):
        return False
    if jax.tree_util.tree_structure(sa) != jax.tree_util.tree_structure(sb):
        return False
    return all(
        jnp.shape(x) == jnp.shape(y) and jnp.result_type(x) == jnp.result_type(y)
        for x, y in zip(jax.tree_util.tree_leaves(da), jax.tree_util.tree_leaves(db), strict=True)
    )


def _equal(x, y) -> bool:
    """Whether two static declarations are equal (``False`` where comparing them is not a
    truth value, as for arrays)."""
    try:
        return bool(x == y)
    except (TypeError, ValueError):
        return False


def _partner_pairs(names, a: str, b: str, lead_view, view, keep: frozenset) -> dict:
    """The ports of member ``b``'s partners paired with those of the first member ``a``'s (design
    batched-run.md R12, amended 2026-11-18), so that copies of one node wired to different ports
    of their partners are one view up to the swap: a driven control node's driver by the
    member's own port (the ``identified`` pair whose own node is the swap of the first's), an
    emission by its declaration (the member's one emission equal to the first's under the swap,
    its parents, realisation and kernel). Each pair ``x <-> y`` comes with the pair of their
    prefix swaps, so the swap stays an involution. Refused, naming the port: a driven node the
    member does not have, an emission whose declaration two of the first's share, a name paired
    with two, and a pair holding a parameter variable both views share. Returns the pairs,
    closed under inversion."""
    prefix = _swapper(a, b, keep)
    pairs: dict[str, str] = {}

    def pair(x: str, y: str, port: str) -> None:
        if prefix(x) == y:
            return
        for u, v in ((x, y), (prefix(x), prefix(y))):
            for s, w in ((u, v), (v, u)):
                if s in keep or pairs.get(s, w) != w:
                    clash = (
                        "a parameter variable both views share"
                        if s in keep
                        else f"paired with {pairs[s]!r} already"
                    )
                    raise ValueError(
                        f"run: the group {names} renames {b!r} to {a!r} by its partners' ports "
                        f"as well as its name, and at {port!r} the pairing is inconsistent: "
                        f"{s!r} would be paired with {w!r}, and it is {clash}; a group holds "
                        "copies of one node, each wired to its own port of a partner: wire the "
                        "copies to distinct ports, or step them ungrouped"
                    )
                pairs[s] = w

    theirs = dict(view.subsystems)
    for component, sub in lead_view.subsystems:
        other = theirs.get(prefix(component))
        if other is None:
            continue  # the views differ in structure: refused by the caller
        for own, partner in sub.identified:
            found = [p for o, p in other.identified if o == prefix(own)]
            if len(found) != 1:
                raise ValueError(
                    f"run: the group {names} steps copies of one node, and {a!r}'s control node "
                    f"{own!r} is driven by {partner!r} while {b!r} has no control node "
                    f"{prefix(own)!r} driven by another node; a group's members are driven at "
                    "the same own ports: group only copies wired alike, or step them ungrouped"
                )
            pair(partner, found[0], prefix(own))
    swap = _swapper(a, b, keep, pairs)
    mine = {e: swap(d) for e, d in lead_view.emissions}
    for e, declared in mine.items():
        target = swap(e)
        if _equal(declared, dict(view.emissions).get(target)):
            continue
        found = [m for m, d in view.emissions if _equal(declared, d)]
        alike = [f for f, d in mine.items() if f != e and _equal(declared, d)]
        if len(found) > 1 or alike:
            raise ValueError(
                f"run: the group {names} pairs each emission of {a!r} with the emission of "
                f"{b!r} declared alike, and the emission {e!r} is ambiguous (declared alike "
                f"by {[e, *alike]!r} of {a!r} and {found!r} of {b!r}); give the copies' "
                "emissions distinct declarations, or step them ungrouped"
            )
        if found:
            pair(e, found[0], e)
    return pairs


def _check_groups(system, params, nodes, groups, schedule, tick_order):
    """The groups of ``steps`` (tuple keys, design batched-run.md R12), checked in the structure
    pass: members all driven by another node's action or none, all clocked on one schedule or
    none, and views equal up to their names and their partners' ports (the System and the
    params' structure, the test ``plan()`` makes to share one compiled program, after the swap
    of the members' prefixes and of the ports ``_partner_pairs`` pairs: the drivers of their
    control nodes and the names of their emissions); two ``plan()`` learners of a variable that
    is not counts within one group are refused under every order, since a group's members are
    simultaneous among themselves. Returns ``{first member: (members, swaps)}``, ``swaps[i]``
    the swap the runtime renames member ``i``'s inputs and returns by, partners' ports
    included."""
    by_name = {n.name: n for n in nodes}
    out = {}
    for group in groups:
        members = [by_name[name] for name in group]
        names = tuple(group)
        if len({m.phase_two for m in members}) > 1:
            raise ValueError(
                f"run: the group {names} mixes nodes a node's action drives with nodes none "
                "drives; a group steps as one node at one turn of the round, so its members are "
                "all driven or none: group only copies wired alike"
            )
        clocked = [m.name for m in members if m.name in schedule]
        if clocked and len(clocked) != len(members):
            raise ValueError(
                f"run: the group {names} mixes clocked nodes {clocked} with unclocked ones; a "
                "group decides as one node, so its members all declare durations or none do: "
                "group the clocked copies apart from the others"
            )
        if clocked and len({schedule[c].rounds for c in clocked}) > 1:
            raise ValueError(
                f"run: the group {names} holds clocked nodes whose options last different "
                f"rounds ({ {c: schedule[c].rounds for c in clocked} }); a group's members are "
                "copies of one node: give them one durations table and one clock, or step them "
                "ungrouped"
            )
        views = [_reset_view(system, m, params) for m in members]
        lead_view, lead_params = views[0]
        swaps: list[Callable] = [lambda x: x]
        for m, (view, vparams) in zip(members[1:], views[1:], strict=True):
            keep = frozenset(lead_params.parameters) & frozenset(vparams.parameters)
            pairs = _partner_pairs(names, members[0].name, m.name, lead_view, view, keep)
            swap = _swapper(members[0].name, m.name, keep, pairs)
            structure = jax.tree_util.tree_structure
            same = (
                structure(swap(lead_view)) == structure(view)
                and structure(swap(view)) == structure(lead_view)
                and _same_tree(swap(vparams), lead_params)
            )
            if not same:
                raise ValueError(
                    f"run: the group {names} steps its members by one vmapped call of one step, "
                    f"and the views of {members[0].name!r} and {m.name!r} differ in static "
                    "structure beyond their names and their partners' ports (their Systems, or "
                    "their params' forms and shapes, the test plan() makes to share one compiled "
                    "program); a group holds copies of one node: group only components built "
                    "alike, or step them ungrouped"
                )
            swaps.append(swap)
        if tick_order != "simultaneous":
            _check_shared_learning(system, members, params)
        out[members[0].name] = (members, swaps)
    return out


#: The fields of the record ``record=`` can name (``present`` goes with ``received``; ``decided``
#: and ``faults`` are always kept).
_RECORDABLE = ("received", "emitted", "intended", "carries")


def _record_fields(record, record_carries: bool) -> frozenset:
    """The fields of the record stacked over rounds: ``record`` (default ``received``,
    ``emitted`` and ``intended``) and ``carries`` under ``record_carries``."""
    if record is None:
        fields = {"received", "emitted", "intended"}
    else:
        if isinstance(record, str) or not isinstance(record, Collection):
            raise TypeError(
                f"run: record={record!r}; pass a collection of the fields to stack over rounds, "
                f"from {list(_RECORDABLE)} (decided and faults are always kept)"
            )
        unknown = sorted(set(record) - set(_RECORDABLE))
        if unknown:
            raise ValueError(
                f"run: record names {unknown}, which are no field of the record it stacks; "
                f"name fields from {list(_RECORDABLE)} (decided and faults are always kept)"
            )
        fields = set(record)
    if record_carries:
        fields.add("carries")
    return frozenset(fields)


def _kept(record: dict, fields: frozenset) -> dict:
    """A round's record as the scan stacks it: the fields ``record=`` keeps, a carry's arrays,
    each node's decision an array."""
    out = {**record, "decided": {k: jnp.asarray(v) for k, v in record["decided"].items()}}
    out["carries"] = eqx.filter(record["carries"], eqx.is_array) if "carries" in fields else None
    for field in ("received", "emitted", "intended"):
        if field not in fields:
            out[field] = None
    if "received" not in fields:
        out["present"] = None
    return out


def _scan_program(engine: _Engine, static, fields: frozenset, batched: bool, last=None, mesh=None):
    """The compiled scan of the rounds over the array part of the state (its other leaves,
    ``static``, the same every round): ``program(dynamic, rounds) -> (dynamic, records)``,
    ``rounds`` the indices of the rounds it runs. Under ``batch=`` it is vmapped over the
    replicas (the leading axis of ``dynamic``), and with a ``mesh`` of several devices it is a
    ``shard_map`` of that vmapped scan: each device runs the whole scan over its share of the
    replicas to the end, with no collective and no rendezvous between rounds (nothing crosses
    replicas). ``last``: the run's round count, past which a round of a padded last chunk leaves
    the state as it was."""

    def body(dynamic, t):
        state = eqx.combine(dynamic, static)
        state, record = engine.round(state, t, scanned=True)
        out, rest = eqx.partition(state, eqx.is_array)
        if engine.strict:
            _exact((out.values, out.carries), (dynamic.values, dynamic.carries))
            _exact(rest.carries, static.carries)
        _check_static(rest, static)
        out = _like_tree(out, dynamic, "the round's state")
        if last is not None:
            live = t < last
            out = jax.tree_util.tree_map(lambda new, old: jnp.where(live, new, old), out, dynamic)
        return out, _kept(record, fields)

    def scan(dynamic, rounds):
        return jax.lax.scan(body, dynamic, rounds)

    if not batched:
        return jax.jit(scan)
    replicas = jax.vmap(scan, in_axes=(0, None))
    if mesh is None:
        return jax.jit(replicas)
    spec = jax.sharding.PartitionSpec
    return jax.jit(
        jax.shard_map(
            replicas,
            mesh=mesh,
            in_specs=(spec("replicas"), spec()),
            out_specs=spec("replicas"),
            check_vma=False,
        )
    )


def _same_static(a, b) -> bool:
    da, db = jax.tree_util.tree_flatten(a), jax.tree_util.tree_flatten(b)
    return da[1] == db[1] and all(x is y or x == y for x, y in zip(da[0], db[0], strict=True))


def _check_static(rest, static) -> None:
    """Refuse a round whose state changed in a leaf that is no array (a step's carry holding a
    Python value that changed, or that became an array)."""
    for name in static.carries:
        a, b = static.carries[name], rest.carries[name]
        if not _same_static(a, b):
            raise ValueError(
                f"run: the carry of {name!r} changed a value that is not an array between rounds "
                f"({a!r} to {b!r}); each round is one step of a scan, whose state is arrays: "
                "hold a count or a flag in the carry as a JAX array (jnp.asarray(0)), and keep "
                "every other value of the carry the same from round to round"
            )
    if not _same_static(rest, static):
        raise ValueError(
            "run: the round changed a value of its state that is not an array (the parameter "
            "home's or a held option's); a step returns the params it was given or learned ones "
            "of the same form"
        )


# ---------------------------------------------------------------------------------- run


def _run(
    system: System,
    params: SystemParams,
    ticks: int,
    key: Array,
    steps: Mapping[Any, Callable],
    clocks: Mapping[str, float] | None = None,
    tick_order: TickOrder = "simultaneous",
    carries: Mapping[str, Any] | None = None,
    record_carries: bool = False,
    batch: int | None = None,
    record: Collection[str] | None = None,
    chunk: int | None = None,
    replicas_per_pass: int | None = None,
    per_replica: Collection[str] = (),
) -> Trajectory:
    """:func:`~hybrid_aif.composition.runner.run`, the engine: the structure pass, then per pass
    of replicas the reset, the rounds on concrete values while the state settles, and the
    compiled scan of the rounds after them in chunks."""
    _check_order(tick_order)
    if not isinstance(system, System):
        raise TypeError(f"run: the first argument is a {type(system).__name__}, not a System")
    if not isinstance(params, SystemParams):
        raise TypeError(f"run: params is a {type(params).__name__}, not SystemParams")
    if isinstance(ticks, bool) or not isinstance(ticks, int | np.integer) or ticks < 0:
        raise ValueError(f"run: ticks={ticks!r} is not a non-negative integer")
    ticks = int(ticks)
    batch = _count(batch, "batch", "the number of replicas run by one program")
    chunk = _count(chunk, "chunk", "the rounds of one compiled chunk of the scan")
    per_pass = _count(
        replicas_per_pass, "replicas_per_pass", "the most replicas one pass of batch= holds"
    )
    if batch is None and (per_pass is not None or per_replica):
        given = "replicas_per_pass" if per_pass is not None else "per_replica"
        raise ValueError(
            f"run: {given}= is about the replicas of batch=, and batch is None; pass batch=B, "
            f"or drop {given}"
        )
    fields = _record_fields(record, record_carries)
    carries = dict(carries or {})
    shared_params = _check_per_replica(per_replica, params, carries, batch)
    flat, groups = _flat_steps(steps)
    # ---- the structure pass: everything that reads structure, not values (R2), on the
    # params of one replica
    system._params_in(shared_params, "run")
    nodes = _structure(system, shared_params, flat)
    names = {n.name for n in nodes}
    unknown = sorted(set(carries) - names)
    if unknown:
        raise ValueError(
            f"run: carries names {unknown}, which are no stepped node; the stepped nodes "
            f"are {sorted(names)}"
        )
    second, derive = _check_timing(nodes)
    _check_black_boxes(system, nodes)
    for node in nodes:
        if getattr(node.step, "sliced", False):
            floats = [s for s in node.driven if _kind(system, shared_params, s) == FLOAT]
            if floats:
                raise NotImplementedError(
                    f"run: node {node.name!r} is stepped by plan() and its control node "
                    f"{floats[0]!r} is driven by a continuous variable; a planner holds a driven "
                    "input at one level, so give the node a custom step"
                )
    kernels = system._actuator(shared_params, "run")
    ports = system.ports(shared_params)
    types = dict(ports.outs)
    declared_by = {n.name: _declared_durations(shared_params, n.name) for n in nodes}
    clocked = {name for name, declared in declared_by.items() if declared}
    scales = _check_clocks(clocks, names, clocked, system._black_boxes)
    for node in nodes:
        if node.name in clocked:
            declared = next(iter(declared_by[node.name]))
            if declared in node.driven:
                raise ValueError(
                    f"run: node {node.name!r} declares durations on its control node {declared!r}, "
                    "which another node's variable drives, so the node chooses no option and has "
                    "nothing to hold; drop the durations from this node's params (a mirror "
                    "carries none), or declare them on the node that chooses the option"
                )
    for node in nodes:
        if node.name not in clocked:
            continue
        for a in node.chosen:
            if a in kernels and not _is_identity(kernels[a]):
                raise ValueError(
                    f"run: node {node.name!r} declares durations and action node {a!r} has a "
                    "non-identity actuator in params.actuator; a decision held for its duration "
                    "is realised once, so the kernel's draw would stand for the whole option and "
                    "the duration is read off the intended level; give the node an identity "
                    "actuator, or drop its durations"
                )
    schedule = {
        n.name: _clock_of(system, n, shared_params, scales[n.name])
        for n in nodes
        if n.name in clocked
    }
    if tick_order == "simultaneous":
        _check_shared_learning(system, nodes, shared_params)
    grouped = _check_groups(system, shared_params, nodes, groups, schedule, tick_order)
    engine = _Engine(system, nodes, second, derive, tick_order, kernels, types, schedule, grouped)
    engine.async_orders(batch)

    # ---- the replicas, in passes; each pass the reset, the state settled from it and the
    # compiled scan of every round in chunks (R1, R11), its replicas spread over the devices
    if batch is None:
        simulation = _Simulation(engine, ticks, chunk, fields, False, shared_params)
        out = simulation.run(key, carries, params)
        size, passes, devices, per_device = None, 1, 1, None
    else:
        keys = _replica_keys(key, batch)
        size = batch if per_pass is None else min(per_pass, batch)
        passes = -(-batch // size)
        devices, per_device = _devices_for(size)
        width = devices * per_device  # a pass's replicas, filled to a multiple of the devices
        mesh = _mesh(devices)
        simulation = _Simulation(engine, ticks, chunk, fields, True, shared_params, mesh)
        params_axis = eqx.if_array(0) if "params" in per_replica else None
        seed_axes = {n: (eqx.if_array(0) if n in per_replica else None) for n in carries}
        outs, counts = [], []
        for p in range(passes):
            # each pass is filled to its width with copies of its last replica, so every pass
            # runs one program and every device as many replicas; the copies are dropped
            real = np.arange(p * size, min((p + 1) * size, batch))
            index = np.concatenate([real, np.full(width - len(real), real[-1])])
            simulation.passes = p
            outs.append(
                simulation.run(
                    keys[index],
                    {n: _rows(s, index) if n in per_replica else s for n, s in carries.items()},
                    _rows(params, index) if "params" in per_replica else params,
                    seed_axes,
                    params_axis,
                )
            )
            counts.append(len(real))
        out = _joined(outs, counts)
    state, start, reset_carries, rest, final = out
    if not engine.wrote:
        state = state._replace(home=params)  # no step wrote back: the home is the one handed in
    timing = dict(
        simulation.timing,
        passes=passes,
        replicas_per_pass=size,
        devices=devices,
        replicas_per_device=per_device,
    )
    return _assemble_record(
        system,
        shared_params,
        state,
        nodes,
        ticks,
        types,
        ports,
        start,
        reset_carries,
        rest,
        final,
        fields,
        batch,
    )._replace(timing=timing)


def _count(value, what: str, meaning: str) -> int | None:
    """``value`` as a positive count, or ``None``; refused otherwise naming ``what``."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise ValueError(f"run: {what}={value!r} is not a positive integer ({meaning}) or None")
    return int(value)


def _flat_steps(steps) -> tuple[dict, list[tuple[str, ...]]]:
    """``steps`` with each group (a tuple of node names, design batched-run.md R12) spread over
    its members, and the groups."""
    if not isinstance(steps, Mapping):
        return steps, []
    flat, groups, owner = {}, [], {}
    for k, step in steps.items():
        members = (k,) if isinstance(k, str) else k
        if not isinstance(k, str):
            if (
                not isinstance(k, tuple)
                or len(k) < 2
                or not all(isinstance(m, str) for m in k)
                or len(set(k)) != len(k)
            ):
                raise TypeError(
                    f"run: steps has the key {k!r}; a key is a node's name, or a tuple of two or "
                    "more distinct node names stepped as one group by one vmapped call of the step"
                )
            groups.append(k)
        for m in members:
            if m in owner:
                raise ValueError(
                    f"run: steps names {m!r} twice (under {owner[m]!r} and {k!r}); a node is "
                    "stepped by one step, alone or in one group"
                )
            owner[m] = k
            flat[m] = step
    return flat, groups


def _leading(tree, batch: int) -> bool:
    """Whether every array leaf of ``tree`` has a leading axis of ``batch`` entries."""
    leaves = [x for x in jax.tree_util.tree_leaves(tree) if eqx.is_array(x)]
    return all(jnp.ndim(x) >= 1 and jnp.shape(x)[0] == batch for x in leaves)


def _rows(tree, index):
    """``tree`` at the replicas ``index`` (its arrays' leading axis); other leaves as they are."""
    return jax.tree_util.tree_map(lambda x: x[index] if eqx.is_array(x) else x, tree)


def _check_per_replica(per_replica, params, carries, batch) -> SystemParams:
    """Check ``per_replica`` (which inputs carry a leading replica axis: ``"params"`` and stepped
    nodes' seeds in ``carries``) and return the params of one replica, which the structure pass
    reads; per-replica params must agree across the replicas in what is structure (the
    durations and the actuator kernels)."""
    if isinstance(per_replica, str) or not isinstance(per_replica, Collection):
        raise TypeError(
            f"run: per_replica={per_replica!r}; pass a collection naming the inputs that carry "
            'a leading replica axis: "params" and stepped nodes whose seed in carries does'
        )
    per_replica = set(per_replica)
    unknown = sorted(per_replica - {"params"} - set(carries))
    if unknown:
        raise ValueError(
            f'run: per_replica names {unknown}, which is neither "params" nor a node seeded in '
            "carries; name the inputs that carry a leading replica axis"
        )
    for name in sorted(per_replica):
        tree = params if name == "params" else carries[name]
        if not _leading(tree, batch):
            what = "params" if name == "params" else f"the seed of {name!r} in carries"
            raise ValueError(
                f"run: per_replica names {name!r}, and an array of {what} has no leading axis "
                f"of batch={batch} entries; give every array of it a leading replica axis "
                "(envs.generative.broadcast_params(tree, batch) for the shared ones, then "
                "replace those that differ), or drop it from per_replica to share it"
            )
    if "params" not in per_replica:
        return params
    for field in ("durations", "actuator"):
        tables = [getattr(params, field)] + [
            getattr(part, field) for part in (params.parts or {}).values()
        ]
        for leaf in jax.tree_util.tree_leaves(tables):
            if eqx.is_array(leaf) and not bool(jnp.all(leaf == leaf[:1])):
                raise ValueError(
                    f"run: params.{field} differs across the replicas of batch=; the "
                    f"{field} are structure the structure pass reads once (the clock table, "
                    "the actuator's draws), so share them: give every replica the same values"
                )
    return _rows(params, 0)


def _replica_keys(key, batch: int) -> Array:
    """The replicas' run keys (design batched-run.md R10): replica ``b`` runs under
    ``fold_in(key, b)``, or under ``key[b]`` where ``key`` is a stack of ``batch`` keys."""
    key = jnp.asarray(key)
    typed = jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)
    single = key.ndim == (0 if typed else 1)
    if single:
        return jax.vmap(lambda b: jax.random.fold_in(key, b))(jnp.arange(batch))
    if key.ndim != (1 if typed else 2) or key.shape[0] != batch:
        raise ValueError(
            f"run: key has shape {tuple(key.shape)}; under batch={batch} pass one key (replica b "
            f"runs under fold_in(key, b)) or a stack of {batch} keys, one per replica"
        )
    return key


def _joined(outs: list, counts: list[int]):
    """The passes' results, each cut to its ``counts`` real replicas (the copies that filled it
    dropped), joined on the replica axis."""
    parts = [eqx.partition(o, eqx.is_array) for o in outs]
    dynamic = jax.tree_util.tree_map(
        lambda *xs: jnp.concatenate([x[:n] for x, n in zip(xs, counts, strict=True)]),
        *[d for d, _ in parts],
    )
    return eqx.combine(dynamic, parts[0][1])


def _devices_for(size: int) -> tuple[int, int]:
    """``(devices, replicas per device)`` for a pass of ``size`` replicas over the local devices
    (several accelerators, or a CPU host given ``--xla_force_host_platform_device_count``): the
    fewest replicas per device the devices allow, then the fewest devices that hold the pass at
    that load, so a pass is filled with at most ``devices - 1`` copies (none where the count
    divides it): ten replicas on sixteen devices run on ten, sixteen on eight run two each,
    thirteen on eight run two each on seven, one copy added."""
    count = len(jax.local_devices())
    per_device = -(-size // max(count, 1))
    return -(-size // per_device), per_device


def _mesh(devices: int):
    """A one-axis mesh (``"replicas"``) over the first ``devices`` local devices, or ``None``
    for one."""
    if devices < 2:
        return None
    return jax.sharding.Mesh(np.asarray(jax.local_devices()[:devices]), ("replicas",))


def _placed(dynamic, mesh):
    """The scan's state with its replica axis split over ``mesh``'s devices (as it is without
    one)."""
    if mesh is None:
        return dynamic
    spec = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("replicas"))
    return jax.device_put(dynamic, spec)


def _reset(engine: _Engine, key, seeds, home) -> _State:
    """The reset (section 4.4): each node's first call, with no observation, in node order with
    the reset keys; the state entering round 0, a variable the reset gave no value ``None``."""
    system, nodes = engine.system, engine.nodes
    values: dict[str, Array | None] = {v: None for n in nodes for v in n.outs}
    live = {n.name: seeds.get(n.name) for n in nodes}
    key, k_reset = jax.random.split(key)
    for node, k in zip(nodes, _reset_keys(k_reset, engine.count), strict=True):
        view, vparams = _reset_view(system, node, home)
        out, live[node.name], _ = engine._call(node, view, k, None, live[node.name], vparams)
        out = _check_out(node, out, engine.types, system, reset=True)
        values.update(out)
    held = {
        n.name: (jnp.zeros((), jnp.int32), {a: _null(engine.types[a]) for a in n.chosen})
        for n in nodes
        if n.name in engine.clocks
    }
    return _State(key, values, live, home, held, _sweep_key(key, engine.count), {})


def _settle(engine: _Engine, state: _State, lead: tuple) -> _State:
    """The state straight from the reset brought to the form of every round's state, so that
    round 0 is the scan's first round (design batched-run.md, Phase C), by a trace: the path
    taken where :func:`_settling` finds a form the structure does not give (a statically
    settled state, :func:`_settle_static`, is the same state). Round 0 is evaluated
    abstractly (``eval_shape``, nothing runs, the planner's search traced), each variable the
    reset left without a value
    takes zeros of the form round 0 gives it and a presence flag, false (``lead``: the replica
    axis of a pass, or ``()``), and each step's carry the form round 0 returns: a
    :class:`MindCarry` (``plan()``'s) by filling what round 0 sets with zeros its ``started``
    flag guards, any other carry as it is (Python numbers made arrays), refused naming the step
    where its form differs from round 0's. A float the reset set (a variable's value, a carry's
    leaf, a parameter of the home, a held option) settles at the wider of its dtype and the one
    round 0 returns for it, so a round only ever widens a float into the state (:func:`_like`):
    where that widens a leaf, and once in any case under JAX's 64-bit mode (a step that learns
    from the second round on first shows a wider table on the settled state), round 0 is
    evaluated again on the settled state, until it returns no float wider than the state it is
    handed."""
    reset = state
    round0 = functools.partial(engine.round, t=0)
    if lead:
        round0 = eqx.filter_vmap(round0)
    like = eqx.filter_eval_shape(round0, _strong(state))[0]
    values = {
        v: _zeros_like(like.values[v]) if x is None else _widened(x, like.values[v])
        for v, x in state.values.items()
    }
    present = {v: jnp.zeros(lead, bool) for v, x in state.values.items() if x is None}
    carries, before = {}, {}
    for node in engine.nodes:
        target = like.carries[node.name]
        carry = _settled_mind(_strong(state.carries[node.name]), target)
        before[node.name] = carry
        carries[node.name] = _formed(carry, target, node.name)
    home = _widened_tree(state.home, like.home)
    held = _widened_tree(state.held, like.held)
    state = _strong(
        state._replace(values=values, carries=carries, present=present, home=home, held=held)
    )
    set_values = [v for v, x in reset.values.items() if x is not None]
    was = _strong(([reset.values[v] for v in set_values], before, reset.home, reset.held))
    now = ([state.values[v] for v in set_values], state.carries, state.home, state.held)
    if not (jax.config.jax_enable_x64 or _float_widened(was, now)):
        return state  # one float dtype and nothing widened: the state is settled
    for _ in range(len(jax.tree_util.tree_leaves(state)) + 1):
        like = eqx.filter_eval_shape(round0, state)[0]
        parts = ("values", "carries", "home", "held")
        wider = _strong(
            state._replace(**{f: _widened_tree(getattr(state, f), getattr(like, f)) for f in parts})
        )
        if not _float_widened(state, wider):
            break
        state = wider
    return state


def _widened_tree(tree, like):
    """``tree`` with each leaf :func:`_widened` against its leaf in ``like``; ``tree`` as it is
    where the two differ in structure (the scan refuses that, naming what changed)."""
    none = lambda x: x is None  # noqa: E731
    if jax.tree_util.tree_structure(tree, is_leaf=none) != jax.tree_util.tree_structure(
        like, is_leaf=none
    ):
        return tree
    return jax.tree_util.tree_map(_widened, tree, like, is_leaf=none)


def _float_widened(was, now) -> bool:
    """Whether a float leaf of ``was`` is a float of another dtype in ``now`` (the two of one
    structure)."""

    def floating(x):
        return hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)

    return any(
        floating(a) and floating(b) and a.dtype != b.dtype
        for a, b in zip(jax.tree_util.tree_leaves(was), jax.tree_util.tree_leaves(now), strict=True)
    )


def _formed(carry, target, name: str):
    """``carry`` (a reset's) in the dtypes of ``target`` (round 0's carry, abstract), refused
    where the two differ in structure, a shape, or a leaf that is no array."""
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    targets, target_def = jax.tree_util.tree_flatten(target)
    problem = None
    if treedef != target_def:
        problem = f"its structure is {treedef} at the reset and {target_def} at round 0"
    else:
        paths = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(carry)[0]]
        out = []
        for x, t, path in zip(leaves, targets, paths, strict=True):
            if isinstance(t, jax.ShapeDtypeStruct):
                if not _arrayish(x) or jnp.shape(x) != t.shape:
                    problem = (
                        f"at {path or 'its root'} it is {_described(x)} at the reset and an "
                        f"array of shape {t.shape} at round 0"
                    )
                    break
                typed = jax.dtypes.issubdtype(t.dtype, jax.dtypes.prng_key)
                floats = jnp.issubdtype(jnp.result_type(x), jnp.floating) and jnp.issubdtype(
                    t.dtype, jnp.floating
                )
                if typed:
                    out.append(x)
                elif floats:
                    out.append(_widened(x, t))
                else:
                    out.append(jnp.asarray(x, t.dtype))
            elif x is t or _same_static(x, t):
                out.append(x)
            else:
                problem = f"at {path or 'its root'} it is {x!r} at the reset and {t!r} at round 0"
                break
        else:
            return treedef.unflatten(out)
    raise ValueError(
        f"run: the step of {name!r} returns at its reset (obs is None) a carry of another form "
        f"than at round 0 ({problem}); every round, round 0 included, is one step of a scan "
        "from the reset, whose state keeps one form, so return from the reset a carry of the "
        "structure, shapes and dtypes every round returns (zeros where round 0 sets a value, "
        "with a flag that says whether it is set)"
    )


def _plan_of(step) -> "_Plan | None":
    """The :func:`plan` step a node is stepped by: the step itself, or the one a wrapper made
    with ``functools.wraps`` wraps (its ``__wrapped__`` chain); ``None`` for any other step."""
    for _ in range(16):
        if isinstance(step, _Plan) or step is None:
            break
        step = getattr(step, "__wrapped__", None)
    return step if isinstance(step, _Plan) else None


#: How ``run`` settles the state from the reset (``Trajectory.timing["settled"]``): every form
#: read off the structure; the same with the :class:`Trace` of each ``plan(trace=True)`` node
#: read off an abstract round 0 of that settled state; round 0 of the state straight from the
#: reset evaluated abstractly.
STATIC, STATIC_TRACE, TRACED = "static", "static+trace", "traced"


def _settling(engine: _Engine, state: _State) -> str:
    """How the state straight from the reset is settled (:data:`STATIC` and the rest). The
    forms follow from the structure where every node is stepped by :func:`plan` (or a wrapper of
    it) and keeps a :class:`MindCarry`, or by a step that gave each of its out-ports a value at
    its reset and whose carry is no :class:`MindCarry` the reset left unfilled
    (:func:`_settle_static`). A ``plan(trace=True)`` node's :class:`Trace` has the form of its
    planner's search, which only a trace gives: it is read off round 0 evaluated abstractly on
    the state settled from the structure, whose trace of the planner the scan's compile reuses
    (``jit`` caches it by those forms and that wrapping), so the search is traced once. A custom
    step that leaves an out-port without a value at its reset gives it a form of its own in
    round 0: the state is settled by round 0 of the state from the reset (:func:`_settle`)."""
    how = STATIC
    for node in engine.nodes:
        carry = state.carries[node.name]
        planned = _plan_of(node.step)
        if planned is not None:
            if not isinstance(carry, MindCarry):
                return TRACED
            if planned.trace:
                how = STATIC_TRACE
        elif any(state.values[v] is None for v in node.outs):
            return TRACED
        elif isinstance(carry, MindCarry) and (carry.previous is None or carry.action is None):
            return TRACED
    return how


def _absent(engine: _Engine, variable: str, dtype) -> Array:
    """Zeros of the form a :func:`plan` step gives an out-port it returns nothing for at its
    reset: an action node a level (int32) or a continuous node's float vector (:func:`_null`),
    an emission its realisation, a level's probability vector under ``realise="stats"`` and
    otherwise a level or a float vector of the port's width; a float in ``dtype`` (the model's
    tables')."""
    kind, size = engine.types[variable]
    emission = dict(engine.system.emissions).get(variable)
    if kind == LEVEL and emission is not None and emission.realise == "stats":
        return jnp.zeros((size,), dtype)
    return _null(engine.types[variable], dtype)


def _settle_static(engine: _Engine, state: _State, lead: tuple, one: SystemParams) -> _State:
    """The state straight from the reset brought to the form of every round's state without a
    trace, where :func:`_settling` allows: the state :func:`_settle` gives, each form read off
    the structure (a ``plan(trace=True)`` node's ``trace`` left ``None``, for
    :func:`_with_traces`). A variable the reset left without a value (an out-port of a :func:`plan`
    node) takes zeros of :func:`_absent`'s form and a presence flag, false (``lead``: the
    replica axis of a pass, or ``()``); a plan node's :class:`MindCarry` takes ``previous`` of
    the form of its ``beliefs`` (the posterior a round stores has the form of the canonical
    prior it filters from, :func:`_canonical`), ``action`` one :func:`_null` per action node it
    chooses, and under ``warm_start`` ``plan`` as :func:`_first_plan` of its view's control space
    (``one``: the params of one replica, read for structure), all zeros that ``started`` (false)
    guards; any other carry is kept as it is (Python numbers made arrays). The scan's trace
    checks every form (:class:`_Unformed`), and where one does not hold the state is settled by
    the trace after all."""
    state = _strong(state)
    dtype = table_dtype(one)  # a float the reset left unset, in the dtype of the model's tables
    values, present = {}, {}
    for v, x in state.values.items():
        if x is None:
            form = _absent(engine, v, dtype)
            values[v] = jnp.zeros(lead + form.shape, form.dtype)
            present[v] = jnp.zeros(lead, bool)
        else:
            values[v] = x
    carries = dict(state.carries)
    for node in engine.nodes:
        planned = _plan_of(node.step)
        if planned is None:
            continue
        carry, filled = carries[node.name], {}
        if carry.previous is None:
            filled["previous"] = _zeros_like(carry.beliefs)
        if carry.action is None:
            nulls = {a: _null(engine.types[a], dtype) for a in node.chosen}
            filled["action"] = {a: jnp.zeros(lead + x.shape, x.dtype) for a, x in nulls.items()}
        if planned.warm_start and carry.plan is None:
            view, vparams = _reset_view(engine.system, node, one)
            view = planned._view(view)
            space = view._space(view._params_in(vparams, "plan"))
            first = _first_plan(
                view.policy_len, space.layout, space.sizes, space.kinds, space.boxes
            )
            filled["plan"] = jnp.zeros(lead + first.shape, first.dtype)
        carries[node.name] = _strong(carry._replace(**filled))
    return state._replace(values=values, carries=carries, present=present)


def _with_traces(engine: _Engine, state: _State, lead: tuple, mesh=None) -> _State:
    """``state`` (settled from the structure) with the ``trace`` of each ``plan(trace=True)``
    node as zeros of the :class:`Trace` round 0 gives it, round 0 evaluated abstractly on
    ``state`` itself and wrapped as the scan wraps its round (vmapped over the replicas, a
    ``shard_map`` over ``mesh``): the planner is then traced on the forms and in the context
    the scan's compile traces it in, and the compile reuses that trace (see
    :func:`_settling`)."""
    dynamic, static = eqx.partition(state, eqx.is_array)
    held = {}

    def round0(dynamic):
        out, _ = engine.round(eqx.combine(dynamic, static), 0)
        arrays, held["rest"] = eqx.partition(out, eqx.is_array)
        return arrays

    if lead:
        round0 = jax.vmap(round0)
    if mesh is not None:
        spec = jax.sharding.PartitionSpec("replicas")
        round0 = jax.shard_map(round0, mesh=mesh, in_specs=(spec,), out_specs=spec, check_vma=False)
    like = eqx.combine(jax.eval_shape(round0, dynamic), held["rest"])
    carries = dict(state.carries)
    for node in engine.nodes:
        planned = _plan_of(node.step)
        carry, target = carries[node.name], like.carries[node.name]
        if planned is not None and planned.trace and carry.trace is None:
            if isinstance(target, MindCarry) and target.trace is not None:
                carries[node.name] = carry._replace(trace=_zeros_like(target.trace))
    return state._replace(carries=carries)


class _Unformed(Exception):  # noqa: N818 - a signal of the settling, not an error a caller sees
    """Raised on the trace of the scan of a statically settled state (:func:`_settle_static`)
    where a round returns a variable or a carry of another form than the state was settled to:
    the static rule did not hold for this run, and ``run`` settles the state by the trace of
    round 0 (:func:`_settle`) and compiles again, so both paths give one state, and a step whose
    carry differs from round 0's is refused by :func:`_formed`'s sentence on either."""


def _exact(tree, reference) -> None:
    """Raise :class:`_Unformed` where ``tree`` differs from ``reference`` in structure, in an
    array leaf's shape or dtype, or in a leaf that is no array."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    refs, ref_def = jax.tree_util.tree_flatten(reference)
    if treedef != ref_def:
        raise _Unformed
    for x, r in zip(leaves, refs, strict=True):
        if eqx.is_array(x) or eqx.is_array(r):
            same = eqx.is_array(x) and eqx.is_array(r)
            if not (same and x.shape == r.shape and x.dtype == r.dtype):
                raise _Unformed
        elif not _same_static(x, r):
            raise _Unformed


def _arrayish(x) -> bool:
    return eqx.is_array(x) or isinstance(x, bool | int | float | complex | np.ndarray | np.generic)


def _described(x) -> str:
    if x is None:
        return "None"
    if _arrayish(x):
        return f"an array of shape {tuple(jnp.shape(x))}"
    return f"a {type(x).__name__}"


class _Simulation:
    """The replicas of one run, pass by pass: the reset, the state settled from it into the form
    of every round (statically, :func:`_settle_static`, where :func:`_settling` allows, else by
    a trace of round 0, :func:`_settle`), and the compiled scan of every round from round 0, a
    host loop over chunks of ``chunk`` rounds (one program re-run). Under ``batch=`` every call
    is vmapped over the pass's replicas: the reset as a vmapped call, the scan as a vmapped
    scan. The program compiled in the first pass serves every pass after it, and the first
    pass's way of settling every pass after it. ``one``: the params of one replica, read for
    structure."""

    def __init__(self, engine: _Engine, ticks: int, chunk, fields, batched: bool, one, mesh=None):
        self.engine = engine
        self.one = one
        self.mesh = mesh
        self.ticks = ticks
        self.chunk = chunk
        self.fields = fields
        self.batched = batched
        self.compiled = None
        self.static = None
        self.passes = 0
        self.timing: dict[str, Any] = {
            "compile": 0.0,
            "scan": 0.0,
            "scanned": 0,
            "reset": 0.0,
            "settle": 0.0,
            "settled": None,
            "chunks": [],
        }

    def run(self, key, seeds, home, seed_axes=None, home_axis=None):
        engine = self.engine
        begun = time.perf_counter()
        if self.batched:
            state = eqx.filter_vmap(
                functools.partial(_reset, engine), in_axes=(0, seed_axes, home_axis)
            )(key, seeds, home)
        else:
            state = _reset(engine, key, seeds, home)
        start = {
            n.name: {v: state.values[v] for v in n.outs if state.values[v] is not None}
            for n in engine.nodes
        }
        self.timing["reset"] += time.perf_counter() - begun
        rest, reset_carries = None, dict(state.carries)
        if self.ticks:
            state, reset_carries, rest = self._settled_scan(state, jnp.shape(key)[:1])
        now = (state.values, state.present)
        final = {n.name: engine.read(n, now, now) for n in engine.nodes}
        return state, start, reset_carries, rest, final

    def _settled_scan(self, reset: _State, replicas: tuple):
        """The state from the reset settled and scanned: ``(state, settled carries, records)``.
        Settled as :func:`_settling` says (``timing["settled"]``): from the structure, its
        traces read off an abstract round 0 of the settled state, or by the abstract round 0 of
        the state from the reset; where the scan's trace finds a form the structure did not give
        (:class:`_Unformed`), or refuses, settled by the last and compiled again, so a run
        refuses with the sentence round 0's trace refuses with (a step changing its params'
        structure, a carry of another form than round 0's) whichever way it was settled.
        ``replicas``: the pass's replica axis under ``batch=``, else ``()``."""
        engine = self.engine
        lead = replicas if self.batched else ()
        how = TRACED if self.timing["settled"] == TRACED else _settling(engine, reset)
        begun = time.perf_counter()
        try:
            if how == TRACED:
                state = _settle(engine, reset, lead)
            else:
                state = _settle_static(engine, reset, lead, self.one)
                if how == STATIC_TRACE:
                    state = _with_traces(engine, state, lead, self.mesh)
            self.timing["settle"] += time.perf_counter() - begun
            begun = None
            out = self._scan(state, strict=how != TRACED)
        except (_Unformed, ValueError, TypeError):
            if how == TRACED:
                raise
            if begun is not None:  # refused while the round 0 of the traces was evaluated
                self.timing["settle"] += time.perf_counter() - begun
            self.compiled = None
            how = TRACED
            begun = time.perf_counter()
            state = _settle(engine, reset, lead)
            self.timing["settle"] += time.perf_counter() - begun
            out = self._scan(state)
        self.timing["settled"] = how
        return out[0], dict(state.carries), out[1]

    def _scan(self, state, strict: bool = False):
        """Every round from round 0, scanned in chunks: ``(state, records)``; ``strict``: the
        state was settled statically, and the trace of the program checks its forms
        (:class:`_Unformed`)."""
        dynamic, static = eqx.partition(state, eqx.is_array)
        ticks = self.ticks
        length = ticks if self.chunk is None else min(self.chunk, ticks)
        padded = -(-ticks // length) * length
        if self.compiled is None or not _same_static(static, self.static):
            last = ticks if padded != ticks else None
            program = _scan_program(self.engine, static, self.fields, self.batched, last, self.mesh)
            rounds = jnp.arange(0, length, dtype=jnp.int32)
            begun = time.perf_counter()
            self.engine.strict = strict
            try:
                self.compiled = program.lower(_placed(dynamic, self.mesh), rounds).compile()
            finally:
                self.engine.strict = False
                self.timing["compile"] += time.perf_counter() - begun
            self.static = static
        dynamic = _placed(dynamic, self.mesh)
        records = []
        for s in range(0, padded, length):
            rounds = jnp.arange(s, s + length, dtype=jnp.int32)
            begun = time.perf_counter()
            dynamic, record = jax.block_until_ready(self.compiled(dynamic, rounds))
            seconds = time.perf_counter() - begun
            records.append(record)
            self.timing["scan"] += seconds
            self.timing["chunks"].append(
                {
                    "pass": self.passes,
                    "start": s,
                    "rounds": min(length, ticks - s),
                    "seconds": seconds,
                }
            )
        if self.passes == 0:
            self.timing["scanned"] += ticks
        axis = 1 if self.batched else 0
        if len(records) == 1 and padded == ticks:
            return eqx.combine(dynamic, static), records[0]
        joined = jax.tree_util.tree_map(
            lambda *xs: jax.lax.slice_in_dim(jnp.concatenate(xs, axis), 0, ticks, axis=axis),
            *records,
        )
        return eqx.combine(dynamic, static), joined


def _sweep_key(key: Array, count: int) -> Array:
    """The key the async order is drawn from: the run key folded by the node count (so the
    per-round key stream is untouched); round ``t`` permutes the sources with
    ``jax.random.permutation(fold_in(sweep, t), sources)``."""
    return jax.random.fold_in(key, count)


# --------------------------------------------------------------------------- write-backs


def _unchanged(a, b):
    """Whether two trees hold the same values: ``True`` where every leaf is the same object or
    equal on concrete values, ``False`` where the structure or a concrete leaf differs, else a
    traced flag."""
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    if ta != tb or len(la) != len(lb):
        return False
    flags = []
    for x, y in zip(la, lb, strict=True):
        if x is y:
            continue
        try:
            same = jnp.array_equal(x, y, equal_nan=True)
        except TypeError:
            same = x == y
        if not _traced(same):
            if not bool(same):
                return False
            continue
        flags.append(same)
    if not flags:
        return True
    return functools.reduce(jnp.logical_and, flags)


def _own_slots(view, node, given, learned, wanted, home, flag=None):
    """The tables a view learned that are its own, split off for the home.

    Returns ``(learned', write)``: ``learned'`` is ``learned`` with each own table (a likelihood,
    transition or initial prior that no parameter variable fills) put back to ``given``'s, so that
    what remains differing is the parameter variables, and ``write`` maps the home to the fields
    that hold the learned tables under their namespaced keys, or is ``None`` when none changed. A
    learned table whose shape differs from the home's (the view was sliced at a driven variable)
    is refused, naming the table and the variable; inside the scanned round (traced values) the
    refusal is handed to ``flag`` and the table is not written."""
    filled = {(slot, i) for slot, i, _ in given.references}
    changed: dict[str, dict[str, Any]] = {}
    keep = {}
    for field, names in (
        ("likelihood", view.modality_names),
        ("transition", view._table.state_names),
        ("initial_state_prior", view._table.state_names),
    ):
        before, after = getattr(given, field), getattr(learned, field)
        pairs = {}
        restore = False
        for i, name in enumerate(names):
            if (field, i) in filled:
                continue
            same = _unchanged(before[name], after[name])
            if same is True:
                continue
            home_entry = getattr(home, field)[name]
            if not _same_shape(home_entry, before[name]):
                sliced = [v for v, level in wanted] or ["a driven variable"]
                _flagged_or_raise(
                    flag,
                    jnp.logical_not(same) if _traced(same) else True,
                    f"{node.name}:{field}[{name}]",
                    f"run: the step of {node.name!r} learned {field}[{name!r}], whose shape "
                    f"differs from the home's because the view is sliced at {sliced}; learn it "
                    "through a declared parameter variable, or leave that slot out of learn",
                )
                restore = True
                continue
            pairs[name] = after[name]
        if pairs or restore:
            keep[field] = {
                **after,
                **{n: before[n] for n in pairs},
                **{n: before[n] for n in before if restore and n not in pairs},
            }
        if pairs:
            changed[field] = pairs
    if not keep:
        return learned, None
    restored = dataclasses.replace(learned, **keep)
    if not changed:
        return restored, None

    def write(home):
        return {field: {**getattr(home, field), **pairs} for field, pairs in changed.items()}

    return restored, write


def _same_shape(a, b) -> bool:
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    return ta == tb and all(jnp.shape(x) == jnp.shape(y) for x, y in zip(la, lb, strict=True))


def _kind(system: System, params: SystemParams, control: str) -> str:
    names = system._names(system._params_in(params, "run"))
    return names.kind(names.controls.index(control))


def _learned_variables(step: Callable, variables) -> set[str]:
    """The parameter variables among ``variables`` that ``step`` is known to learn: a ``plan()``
    step given ``learn`` learns a variable unless its rates leave it at a static zero
    (``learn=True`` learns every learnable slot, a sequence of slot keys only the slots named, a
    mapping the slots its ``lr`` and ``lr_by_slot`` reach). Any other step is not known in advance
    and is checked at the round's boundary."""
    if not isinstance(step, _Plan) or step.learn is None:
        return set()
    keywords = _learn_keywords(step.learn)
    lr = keywords.get("lr", 1.0)
    rates = dict(keywords.get("lr_by_slot") or {})
    out = set()
    for variable in variables:
        key = f"parameter:{variable}"
        if key in rates:
            rate = rates[key]
        else:
            matched = [r for k, r in rates.items() if "*" in k and fnmatch.fnmatchcase(key, k)]
            rate = matched[0] if matched else lr
        if not (isinstance(rate, int | float) and rate == 0):
            out.add(variable)
    return out


def _conjugate(declaration) -> bool:
    """Whether two write-backs of a variable of this form combine exactly by summing their
    deltas: Dirichlet concentrations, which are counts."""
    return isinstance(declaration, Dirichlet | ConditionalDirichlet)


def _check_shared_learning(system: System, nodes: list[_Node], home: SystemParams) -> None:
    """Under ``"simultaneous"``: refuse, before the first round, two ``plan()`` steps whose
    ``learn`` both reach one parameter variable whose form is not conjugate counts
    (``_at_the_boundary`` could not combine their write-backs). Other steps are checked there."""
    learners: dict[str, list[str]] = {}
    for node in nodes:
        variables = _reset_view(system, node, home)[1].parameters
        for variable in _learned_variables(node.step, variables):
            learners.setdefault(variable, []).append(node.name)
    for variable, names in learners.items():
        declaration = home.parameters[variable]
        if len(names) > 1 and not _conjugate(declaration):
            raise ValueError(
                f"run: parameter variable {variable!r} ({type(declaration).__name__}) is learned "
                f"by both {names[0]!r} and {names[1]!r} (their plan(learn=...)) under "
                '"simultaneous", where every write-back of a round lands at its boundary and two '
                "write-backs of one variable combine only as counts (Dirichlet concentrations, "
                'their deltas summed); run them under "sequential" or "async", or give one of them '
                f"a learn= that leaves 'parameter:{variable}' at 0"
            )


def _at_the_boundary(
    system: System, start: SystemParams, pending: list, round_, flag=None
) -> SystemParams:
    """The home after a ``"simultaneous"`` round: ``start`` (the home as the round began) with
    every write-back of the round applied. A variable one node wrote takes its value; one that
    several wrote takes the start's concentration plus the sum of their deltas, the posterior
    given every node's data (Dirichlet counts only, refused for any other form, and refused, as
    ``write_back`` refuses one write-back, where the sum is not positive); each node's own learned
    tables are written under their names.

    Which nodes wrote a variable is read off its values. On concrete values (round 0) a refusal
    raises; inside the scanned round each refusal is handed to ``flag`` (a recorded fault) and the
    variable keeps its round-start value (design batched-run.md R6)."""
    held = system._params_in(start, "run")
    homes = dict(held.parameters)
    writers: dict[str, list[tuple[str, Any, Any]]] = {}
    for name, new, _ in pending:
        for variable, declaration in system._params_in(new, "run").parameters.items():
            same = _unchanged(declaration, homes[variable])
            if same is not True:
                wrote = jnp.logical_not(same) if _traced(same) else True
                writers.setdefault(variable, []).append((name, declaration, wrote))
    for variable, entries in writers.items():
        homes[variable] = _combined(variable, homes[variable], entries, round_, flag)
    out = dataclasses.replace(held, parameters=homes)
    for slot, i, variable in held.references:
        container = list(getattr(out, slot))
        container[i] = homes[variable]
        out = dataclasses.replace(out, **{slot: tuple(container)})
    out = system._params_out(out)
    for _, _, written in pending:
        if written:
            out = dataclasses.replace(out, **written(out))
    return out


def _combined(variable: str, begun, entries: list, round_, flag):
    """One variable after the boundary: ``entries`` its writers ``(node, declaration, wrote)``,
    ``wrote`` a concrete ``True`` or a traced flag."""
    names = [name for name, _, _ in entries]
    sentence_two = (
        f"run: parameter variable {variable!r} ({type(begun).__name__}) was written back "
        f"by both {names[0]!r} and {names[-1]!r} in round {round_} under "
        '"simultaneous", where every write-back of a round lands at its boundary and two '
        "write-backs of one variable combine only as counts (Dirichlet concentrations, "
        'their deltas summed); run them under "sequential" or "async", or let one node '
        "learn it"
    )
    sentence_sum = (
        f"write_back: parameter variable {variable!r} has a concentration that is not "
        f"positive after the views of {names} learned, their deltas summed at the "
        f"boundary of round {round_} (a decay or rate that overshoots the anchor); it is "
        "not a Dirichlet and is not written"
    )
    traced = any(_traced(w) for _, _, w in entries)
    if not traced:
        if len(entries) == 1:
            return entries[0][1]
        if not _conjugate(begun):
            raise ValueError(sentence_two)
        total = begun.concentration + sum(
            d.concentration - begun.concentration for _, d, _ in entries
        )
        if not bool(jnp.all(total > 0.0)):
            raise ValueError(sentence_sum)
        return eqx.tree_at(lambda d: d.concentration, begun, total)
    wrote = [jnp.asarray(w) for _, _, w in entries]
    n = sum(w.astype(jnp.int32) for w in wrote)
    single = begun
    for (_, declaration, _), w in zip(entries, wrote, strict=True):
        single = _select(w, declaration, single)
    if len(entries) == 1:
        return single
    if not _conjugate(begun):
        _flagged_or_raise(flag, n > 1, variable, sentence_two)
        return _select(n > 1, begun, single)
    total = begun.concentration + sum(
        jnp.where(w, d.concentration - begun.concentration, 0.0)
        for (_, d, _), w in zip(entries, wrote, strict=True)
    )
    positive = jnp.all(total > 0.0)
    _flagged_or_raise(flag, (n > 1) & ~positive, variable, sentence_sum)
    summed = eqx.tree_at(lambda d: d.concentration, begun, total)
    return _select(n > 1, _select(positive, summed, begun), single)


def _flagged_or_raise(flag, bad, label: str, sentence: str) -> None:
    if flag is None:
        _flagged(None, bad, label, sentence)
    else:
        flag(bad, label, sentence)


# ------------------------------------------------------------------------- the outputs


def _check_out(
    node: _Node,
    out: Mapping[str, Array],
    types,
    system: System,
    reset: bool,
    standing=None,
    settling: bool = False,
) -> dict[str, Array]:
    """A step's return checked: a mapping keyed by out-ports of the node, each action node one
    level (or a float vector of its width), every emission present from the first round. Shapes
    and dtypes are checked as the step returns them (inside the scan, once, on the trace); a
    level's range is checked here on a concrete value and by :func:`_check_ranges` on a traced
    one. ``standing`` holds the values the ports hold, whose shapes and dtypes the round's values
    keep (the scan's state); ``settling``: the round is evaluated outside the scan, and a value
    wider than its standing one is kept (:func:`_like`)."""
    if not isinstance(out, Mapping):
        raise TypeError(
            f"run: the step of {node.name!r} returned a {type(out).__name__}; a step returns a "
            f"mapping from its out-ports {list(node.outs)} to values"
        )
    unknown = [k for k in out if k not in node.outs]
    if unknown:
        raise ValueError(
            f"run: the step of {node.name!r} returned {unknown}, which are no out-port of the "
            f"node; its out-ports are {list(node.outs)}"
            + (
                " (a control node is keyed by its action node)"
                if any(k in system._table.control_names for k in unknown)
                else ""
            )
        )
    if not reset:
        missing = [k for k in node.outs if k not in out]
        if missing:
            emission = [k for k in missing if k in node.emissions]
            what = (
                f"emission {emission[0]!r}, which its owner must realise every round (a plan() "
                "step does; a custom step calls view.emit(...))"
                if emission
                else f"action node {missing[0]!r}, which no step realises, so nothing would "
                "ever emit it"
            )
            raise ValueError(
                f"run: the step of {node.name!r} returned no value for {what}; "
                f"it returned {sorted(out)}"
            )
    checked = {}
    for k, value in out.items():
        checked[k] = _checked_value(node, k, value, types, system)
        if standing is not None and standing.get(k) is not None:
            checked[k] = _like(
                checked[k],
                standing[k],
                f"{k!r}, returned by the step of {node.name!r}",
                settling,
            )
    return checked


def _checked_value(node: _Node, k: str, value, types, system: System) -> Array:
    kind, size = types[k]
    if k in node.emissions and dict(system.emissions)[k].kernel == BLACK_BOX:
        return _box_value(node.name, k, value, kind, size)
    if not isinstance(value, Array | np.ndarray | np.generic | numbers.Number):
        raise TypeError(
            f"run: the step of {node.name!r} returned {k!r} as a {type(value).__name__}; a step "
            "returns JAX arrays (a level, or a vector of the port's width)"
        )
    level = jnp.asarray(value)
    if k in node.emissions:
        soft = dict(system.emissions)[k].realise == "stats"
        floating = jnp.issubdtype(level.dtype, jnp.floating)
        if kind == LEVEL and floating and level.shape == (size,):
            return level  # a probability vector over the port's levels
        if kind == LEVEL and not (level.shape == () and jnp.issubdtype(level.dtype, jnp.integer)):
            raise ValueError(
                f"run: the step of {node.name!r} returned {k!r} of dtype {level.dtype} and "
                f"shape {tuple(level.shape)}; the port has {size} levels, so emit one integer "
                f"level, or a probability vector of shape ({size},)"
            )
        if kind == LEVEL and not _traced(level) and not 0 <= int(level) < size:
            raise ValueError(
                f"run: the step of {node.name!r} returned {k!r} = {int(level)}; the port has "
                f"{size} levels, so emit a level in [0, {size})"
            )
        if kind == FLOAT and not soft and level.shape != (size,):
            raise ValueError(
                f"run: the step of {node.name!r} returned {k!r} of shape "
                f"{tuple(level.shape)}; the port is a float vector of width {size}, shape "
                f"({size},)"
            )
        return level
    if kind == FLOAT:
        if level.shape != (size,) or not jnp.issubdtype(level.dtype, jnp.floating):
            raise ValueError(
                f"run: the step of {node.name!r} returned {k!r} of dtype {level.dtype} and "
                f"shape {tuple(level.shape)}; a continuous action node takes a float vector "
                f"of its width, shape ({size},)"
            )
        return level
    if level.shape != () or not jnp.issubdtype(level.dtype, jnp.integer):
        raise ValueError(
            f"run: the step of {node.name!r} returned {k!r} of dtype {level.dtype} and shape "
            f"{tuple(level.shape)}; an action node takes one integer level, shape ()"
        )
    if not _traced(level) and not 0 <= int(level) < size:
        raise ValueError(
            f"run: the step of {node.name!r} returned {k!r} = {int(level)}; the action node "
            f"has {size} levels, so return a level in [0, {size})"
        )
    return level.astype(jnp.int32)


def _check_ranges(node: _Node, out: Mapping[str, Array], types, system: System, flag) -> None:
    """The range of each traced level a step returned, a recorded fault where it is out of
    range (a level of a concrete round was refused by :func:`_check_out`); design batched-run.md
    R7."""
    for k, value in out.items():
        kind, size = types[k]
        if kind != LEVEL or not _traced(value) or jnp.shape(value) != ():
            continue
        bad = (value < 0) | (value >= size)
        flag(bad, k, f"run: the step of {node.name!r} returned {k!r} out of [0, {size})")


def _box_value(name: str, port: str, value, kind: str, size: int) -> Array:
    """A black box's out-port value checked against the port's declared type and stored in one
    dtype: a level port takes one integer level in range (any integer type, stored as int32), a
    float port a vector of its width (an integer one converted, a scalar taken for width 1, stored
    in the library's float dtype). Anything else is refused naming the port. Shapes and dtypes are
    checked on the trace; a level's range on a concrete value (a traced one by
    :func:`_check_ranges`)."""
    if not isinstance(value, Array | np.ndarray | np.generic | numbers.Number):
        raise ValueError(
            f"run: the step of {name!r} returned {port!r} as a {type(value).__name__}; the port is "
            "a black box's out-port, so return a JAX array (a level, or a vector of the port's "
            "width)"
        )
    dtype = value.dtype if hasattr(value, "dtype") else np.asarray(value).dtype
    got = f"{dtype} of shape {tuple(np.shape(value))}"
    numeric = np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)
    if not numeric:
        what = (
            f"out-port of {size} levels, so return one integer level in [0, {size})"
            if kind == LEVEL
            else f"float out-port of width {size}, so return a vector of shape ({size},) of numbers"
        )
        raise ValueError(
            f"run: the step of {name!r} returned {port!r} as {got}; the port is a black box's "
            + what
        )
    arr = jnp.asarray(value)
    if kind == LEVEL:
        ok = arr.shape == () and jnp.issubdtype(arr.dtype, jnp.integer)
        if not ok:
            raise ValueError(
                f"run: the step of {name!r} returned {port!r} as {got}; the port is a black box's "
                f"out-port of {size} levels, so return one integer level in [0, {size})"
            )
        if not _traced(arr) and not 0 <= int(arr) < size:
            raise ValueError(
                f"run: the step of {name!r} returned {port!r} = {int(arr)}; the port is a black "
                f"box's out-port of {size} levels, so return one integer level in [0, {size})"
            )
        return arr.astype(jnp.int32)
    if arr.shape == () and size == 1:
        arr = arr.reshape(1)
    if arr.shape != (size,):
        raise ValueError(
            f"run: the step of {name!r} returned {port!r} as {got}; the port is a black box's "
            f"float out-port of width {size}, so return a vector of shape ({size},) of numbers"
            + (" (or one number)" if size == 1 else "")
        )
    return as_float(arr)  # a float keeps its dtype, an integer takes the default float dtype


# ----------------------------------------------------------------------- the trajectory


def _series(body, null: Array, batch=None, body_present=None, tail=None) -> tuple[Array, Array]:
    """A record over rounds: ``body``, stacked over the rounds (``None`` for a run of no rounds),
    then, for what the slots received, the ``tail`` entry ``(value, present)``, what they hold
    after the last round (``None`` where nothing has a value: zeros of the record's form, or of
    ``null``); with the flags of which entries held a value (``body_present``, true where not
    given). Under ``batch=`` every value has a leading replica axis and ``body`` its rounds
    second."""
    lead = () if batch is None else (batch,)
    axis = len(lead)
    value, flag = (None, False) if tail is None else tail
    if body is None:
        like = null if value is None else jnp.asarray(value)[(0,) * axis]
        body = jnp.zeros(lead + (0, *jnp.shape(like)), jnp.result_type(like))
        body_present = jnp.zeros(lead + (0,), bool)
    if body_present is None:
        body_present = jnp.ones(lead + (body.shape[axis],), bool)
    if tail is None:
        return body, body_present
    if value is None:
        value, flag = jnp.zeros(lead + body.shape[axis + 1 :], body.dtype), False
    stacked = jnp.concatenate([body, jnp.expand_dims(jnp.asarray(value, body.dtype), axis)], axis)
    flags = jnp.broadcast_to(jnp.asarray(flag, bool), lead)[..., None]
    return stacked, jnp.concatenate([jnp.asarray(body_present, bool), flags], -1)


def _unstacked(stacked, static, rows: int, axis: int = 0) -> list:
    """The ``rows`` entries of a tree stacked over rounds (on ``axis``), each recombined with
    ``static`` (the leaves that are no arrays)."""
    leaves, treedef = jax.tree_util.tree_flatten(stacked)
    columns = [list(jnp.unstack(leaf, axis=axis)) if rows else [] for leaf in leaves]
    return [eqx.combine(treedef.unflatten([c[t] for c in columns]), static) for t in range(rows)]


def _assemble_record(
    system,
    params,
    state,
    nodes,
    ticks,
    types,
    ports,
    start,
    reset_carries,
    rest,
    final,
    fields,
    batch=None,
) -> Trajectory:
    table = dict(system.subsystems)
    order = list(table)
    modality_types = system._modality_types(system._params_in(params, "run"))
    control_types = {u: t for u, t in ports.ins.items() if u in system._table.control_names}
    records: dict[str, Trajectory] = {}
    lead = () if batch is None else (batch,)

    def slot_null(node: _Node, slot: str) -> Array:
        t = modality_types.get(slot) or control_types.get(slot) or types[node.reads[slot]]
        return _null(t)

    def body(field: str, name: str, key: str):
        return None if rest is None else rest[field][name][key]

    for node in nodes:
        rec: dict[str, Array] | None = None
        pres: dict[str, Array] | None = None
        if "received" in fields:
            rec, pres = {}, {}
            for slot in node.reads:
                last = final[node.name][slot]
                if isinstance(last, Maybe):
                    tail = (last.value, last.present)
                else:  # nothing has a value after a run of no rounds: (None, False)
                    tail = (last, last is not None)
                rec[slot], pres[slot] = _series(
                    body("received", node.name, slot),
                    slot_null(node, slot),
                    batch,
                    body("present", node.name, slot),
                    tail,
                )
        em: dict[str, Array] | None = None
        if "emitted" in fields:
            em = {
                v: _series(body("emitted", node.name, v), _null(types[v]), batch)[0]
                for v in node.outs
            }
        it: dict[str, Array] | None = None
        if "intended" in fields:
            it = {
                a: _series(body("intended", node.name, a), _null(types[a]), batch)[0]
                for a in node.chosen
            }
        decided, _ = _series(
            None if rest is None else rest["decided"][node.name], jnp.zeros((), bool), batch
        )
        log = None
        if "carries" in fields:
            log = [reset_carries[node.name]]
            if rest is not None:
                static = eqx.filter(state.carries[node.name], eqx.is_array, inverse=True)
                log.extend(_unstacked(rest["carries"][node.name], static, ticks, len(lead)))
            log = tuple(log)
        records[node.name] = Trajectory(
            None,
            {},
            _relative(rec, node.name),
            _relative(pres, node.name),
            _relative(em, node.name),
            _relative(it, node.name),
            _relative(start[node.name], node.name),
            log,
            state.carries[node.name],
            decided.astype(bool),
        )
    # the tree above the stepped nodes: one node per component
    parents = {
        name: max((p for p in order if name.startswith(f"{p}{SEPARATOR}")), key=len, default=None)
        for name in order
    }

    def union(name: str, kids: dict[str, Trajectory]) -> Trajectory:
        """The node ``name`` over its parts ``kids`` (by full name): the prefixed union of
        their records."""
        prefix = f"{name}{SEPARATOR}" if name else ""
        parts = {k.removeprefix(prefix): v for k, v in kids.items()}

        def merged(field: str) -> dict | None:
            out = {}
            for rel, part in parts.items():
                value = getattr(part, field)
                if value is None:
                    return None
                out.update({f"{rel}{SEPARATOR}{k}": x for k, x in value.items()})
            return out

        def collected(field: str, keep_none: bool = False) -> dict | None:
            out = {}
            for full, part in kids.items():
                rel, value = full.removeprefix(prefix), getattr(part, field)
                if value is None and not (keep_none and full in records):
                    continue
                if full in records:
                    out[rel] = value
                else:
                    out.update({f"{rel}{SEPARATOR}{k}": x for k, x in value.items()})
            return out or None

        return Trajectory(
            None,
            parts,
            merged("received"),
            merged("present"),
            merged("emitted"),
            merged("intended"),
            merged("start"),
            collected("carries"),
            collected("carry", keep_none=True),
            collected("decided"),
        )

    def build(name: str) -> Trajectory | None:
        if name in records:
            return records[name]
        kids = {k: build(k) for k in order if parents[k] == name}
        kids = {k: v for k, v in kids.items() if v is not None}
        return union(name, kids) if kids else None

    top = {k: build(k) for k in order if parents[k] is None}
    root = union("", {k: v for k, v in top.items() if v is not None})
    return root._replace(params=state.home, faults=_faults(rest))


def _faults(rest) -> dict:
    """The record's faults, a bool per round (and replica) under each label raised in any
    round."""
    if rest is None:
        return {}
    return {label: jnp.asarray(flags, bool) for label, flags in rest["faults"].items()}


def _relative(values: Mapping[str, Any] | None, node: str) -> dict[str, Any] | None:
    if values is None:
        return None
    prefix = f"{node}{SEPARATOR}"
    return {k.removeprefix(prefix): v for k, v in values.items()}


def run(
    system: System,
    params: SystemParams,
    ticks: int,
    key: Array,
    steps: Mapping[str, Callable],
    clocks: Mapping[str, float] | None = None,
    tick_order: TickOrder = "simultaneous",
    carries: Mapping[str, Any] | None = None,
    record_carries: bool = False,
    *,
    batch: int | None = None,
    record: Collection[str] | None = None,
    chunk: int | None = None,
    replicas_per_pass: int | None = None,
    per_replica: Collection[str] = (),
) -> Trajectory:
    """Step a closed System for ``ticks`` rounds and record a :class:`Trajectory`.

    ``system`` is a merge or a coupling whose every modality is fed by an identification (a
    modality no identification feeds is refused naming it, and the sentence names the fix);
    ``params`` is its parameter home. ``steps`` maps each stepped node to its step: a leaf, or a
    non-leaf node that is then planned as one mind (``joint_planner`` of the node); the keys are
    an antichain over the System's components that covers every leaf. The module docstring
    states the step's contract, the rounds, the order of the nodes and the keys.

    ``tick_order`` is ``"simultaneous"``, ``"sequential"`` or ``"async"``. ``carries`` seeds the
    carry of the named stepped nodes (``None``, the default, lets each step make its own; a
    seeded carry of a ``sample()`` node is a partial point state); ``record_carries=True`` keeps
    every round's carry in ``Trajectory.carries``. ``clocks`` maps a stepped node whose view
    declares durations to the rounds one unit of duration lasts (1.0 for a node it omits), a
    positive finite number. A step that learns has its returned parameter variables written back
    to the home each round (the next view is taken from the new home), and ``Trajectory.params``
    is the home after the last round.

    Every round, round 0 included, is one compiled ``lax.scan`` of the round body from the reset
    (module docstring, *The compiled round*): a step is a traced function whose outputs keep one
    shape and dtype from round to round and whose carry keeps the form it has from the reset on
    (a ``plan()`` carry is filled at the reset; a custom step returns from its reset the carry
    form every round returns, refused naming it otherwise), and a step that calls host-side code
    on a traced value is refused naming it (log from inside a step with ``jax.debug.callback``,
    or record what it sees in its carry). A slot whose variable the reset left without a value
    and that the step reads before it is realised in round 0 is handed a :class:`Maybe`
    ``(value, present)`` every round (a custom step reads ``obs[slot].value`` where
    ``obs[slot].present``; ``plan()`` reads an absent value as an absent modality). A check of a
    value inside the scan is recorded in ``Trajectory.faults`` and the run goes on, round 0
    included; the reset's checks refuse.

    ``batch=B`` runs ``B`` replicas by one program (module docstring, *Replicas*): replica ``b``
    is the unbatched run under ``fold_in(key, b)``, or under ``key[b]`` when ``key`` is a stack
    of ``B`` keys. Every array of the record gains a leading replica axis (``decided``,
    ``present``, ``faults``, the final ``carry`` and ``params`` included; ``carries`` stays a
    tuple over rounds, each carry's arrays leading with the replicas); ``batch=1`` keeps an axis
    of length one, and ``batch=None`` gives the unbatched record. Inputs are shared unless
    ``per_replica`` names them: ``"params"`` (every array with a leading axis of ``B``,
    ``envs.generative.broadcast_params`` then the leaves that differ replaced; the durations and
    actuator kernels, which are structure, must agree across the replicas) and a stepped node
    whose seed in ``carries`` leads with ``B``. ``replicas_per_pass`` runs the replicas in host
    passes of at most that many, one compiled program for all, bounding working memory, which is
    linear in the replicas a pass holds; ``None`` is one pass. Under async each replica's order
    is a switch over the distinct derived orders that vmap evaluates in full (cost: the number of
    orders times the step work), refused above four sources.

    *Devices.* A pass's replicas run in parallel across the local devices: each device runs the
    whole scan over its share (a ``shard_map``, no rendezvous between rounds), the fewest
    replicas per device the devices allow and the fewest devices at that load, a pass filled
    with copies of its last replica to that multiple and the copies dropped (ten replicas on
    sixteen devices run on ten; thirteen on eight run two each on seven).
    ``Trajectory.timing`` records ``devices`` and ``replicas_per_device``. On a CPU host there
    is one device unless ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` is set before
    jax is imported, and a batched run on one device is no faster than its replicas one after
    another (a survey rover's step is compute-bound; ``batch=10`` on one device took 8.9 times
    one replica's tick). Measured on a 16-core CPU host, the one-level survey: 101 replica-ticks
    per second for ``batch=16`` over 16 host devices against 8.7 unbatched (11.7 times), and the
    two-tier survey's ten 800-round replicas in 154 s against 1099 s one after another
    (design batched-run.md, Phase C).

    ``record`` names the fields stacked over rounds (``"received"`` with its ``present``,
    ``"emitted"``, ``"intended"``, ``"carries"``; default all but ``"carries"``, which
    ``record_carries=True`` adds; ``decided`` and ``faults`` are always kept); a field not
    recorded is ``None``. ``chunk`` runs the scan as a host loop over compiled chunks of that
    many rounds (one program re-run; a last chunk shorter than ``chunk`` runs padded rounds that
    change nothing and are dropped), ``None`` as one scan. ``Trajectory.timing`` holds the host's
    wall time: ``compile`` (the scan program, once), ``scan`` (its execution over every chunk
    and pass) over ``scanned`` rounds, ``reset`` (the reset calls), ``settle`` (bringing the
    state from the reset into every round's form) and ``settled`` (``"static"`` where the forms
    were read off the structure; ``"static+trace"`` where a ``plan(trace=True)`` node's trace
    was then read off an abstract round 0 of that state, whose trace of the planner the compile
    reuses; ``"traced"`` where round 0 of the state from the reset was evaluated abstractly: a
    custom step leaving an out-port without a value at its reset, or a carry whose form the
    structure did not give), ``chunks`` (each chunk's ``pass``, ``start``,
    ``rounds`` and ``seconds``), ``passes``, ``replicas_per_pass``, ``devices`` and
    ``replicas_per_device``.

    A key of ``steps`` may be a tuple of nodes, a group stepped by one vmapped call (module
    docstring, *Groups*); refused, naming the fix: views that differ beyond their names in
    static structure, a group mixing nodes another node's action drives with nodes none drives
    or clocked with unclocked nodes (or clocked on different durations), a node named twice,
    and a group whose members do not step one after another in the sequential order or in an
    async order.

    Refused, each naming the fix: an open System; ``steps`` that is not an antichain covering
    every leaf, that names no component, or that names a non-leaf node and a ``sample()`` leaf
    beneath it; a step that omits an emission or an action node of its node on a round; a cycle
    of identifications of action nodes with control nodes; ``clocks`` or ``carries`` naming no
    stepped node, or ``clocks`` naming a node that declares no durations; a duration whose
    product with its clock is under one round or not a whole number of rounds, at any level of
    its durations table; an action node of a node with durations under a non-identity actuator;
    a step that calls host-side code on a traced value, or whose carry, params or out-ports change
    structure or shape between rounds. A black box
    (:func:`~hybrid_aif.composition.black_box.black_box`) is stepped by a custom step; ``plan()``
    or ``sample()`` on one, a step marked ``sliced`` on one, a node planned jointly with one
    beneath it and an in-port of one that nothing drives are refused.
    """
    return _run(
        system,
        params,
        ticks,
        key,
        steps,
        clocks,
        tick_order,
        carries,
        record_carries,
        batch=batch,
        record=record,
        chunk=chunk,
        replicas_per_pass=replicas_per_pass,
        per_replica=per_replica,
    )


# ----------------------------------------------------------------------- shared checks


def _check_legal(system: System, params: SystemParams, legal, who: str) -> None:
    """Refuse a ``legal`` mask that masks every level, on the concrete mask: inside the compiled
    tick the mask is traced and cannot be refused there."""
    if system._control_layout(system._params_in(params, who)) is None and not isinstance(
        legal, jax.core.Tracer
    ):
        system._legal_logits(system._params_in(params, who), legal, system.policies(params), who)


def _is_identity(kernel) -> bool:
    """Whether an action kernel is the identity table (square, equal to the identity)."""
    table = np.asarray(kernel)
    return (
        table.ndim == 2
        and table.shape[0] == table.shape[1]
        and bool(np.allclose(table, np.eye(table.shape[0])))
    )


def _actuate(k: Array, n: int, draws: list[tuple[int, Array, Array]]) -> dict[int, Array]:
    """The actuator's draws at the node: for each ``(i, kernel, intended)``, a level drawn from
    the kernel's column at the intended level with key ``i`` of
    ``split(fold_in(k, ACTUATOR_TAG), n)``, keyed by ``i``. No draw, no key."""
    if not draws:
        return {}
    keys = jax.random.split(jax.random.fold_in(k, ACTUATOR_TAG), n)
    out = {}
    for i, kernel, level in draws:
        log_p = jnp.log(as_float(kernel))
        column = jnp.moveaxis(jnp.take(log_p, jnp.asarray(level, jnp.int32), axis=-1), 0, -1)
        out[i] = jax.random.categorical(keys[i], column).astype(jnp.int32)
    return out


def _own_fields(system: System, view_params: SystemParams) -> dict[str, object]:
    """The fields of positional view params that no parameter variable fills, by name (a
    transition by its factor, a likelihood by its modality): everything but ``parameters``,
    ``references`` and the reader positions the references refill."""
    refilled = {(slot, i) for slot, i, _ in view_params.references}
    named = {"transition": system._table.state_names, "likelihood": system.modality_names}
    out: dict[str, object] = {}
    for field in dataclasses.fields(view_params):
        name = field.name
        value = getattr(view_params, name)
        if name in ("parameters", "references"):
            continue
        if name in named:
            for i, entry in enumerate(value):
                if (name, i) not in refilled:
                    out[f"{name}[{named[name][i]!r}]"] = entry
        else:
            out[name] = value
    return out


def _check_parameters_only(
    system: System, name: str, given: SystemParams, learned: SystemParams, flag=None
) -> None:
    """Refuse learned params that differ from the given ones outside the parameter variables.
    Leaves are compared by identity first and by value only where the identity changed; inside
    the scanned round (traced values) a difference is handed to ``flag``, a recorded fault, and
    the home is written from the parameter variables alone."""
    before = _own_fields(system, system._params_in(given, "run"))
    after = _own_fields(system, system._params_in(learned, "run"))
    differ = [] if before.keys() == after.keys() else ["the set of fields"]
    traced = []
    for field in before.keys() & after.keys():
        same = _unchanged(before[field], after[field])
        if same is False:
            differ.append(field)
        elif same is not True:
            traced.append((field, same))
    sentence = (
        f"the step of component {name!r} returned params that differ from the ones it was "
        f"given in {sorted(differ or [f for f, _ in traced])}; on a composite only declared "
        "parameter variables are written back to the home, so a table that is learned must be "
        "declared as a Parameter variable"
    )
    if differ:
        raise ValueError(sentence)
    if traced:
        bad = functools.reduce(jnp.logical_or, [jnp.logical_not(same) for _, same in traced])
        _flagged_or_raise(flag, bad, f"{name}:params", sentence)


def _rounds_per_unit(value, what: str) -> float:
    """``value`` as the rounds one unit of duration lasts: a positive finite real number or a
    scalar array, or a refusal naming ``what``."""
    scalar = isinstance(value, numbers.Real) or (
        isinstance(value, np.ndarray | Array) and value.ndim == 0
    )
    if isinstance(value, bool | np.bool_) or not scalar:
        raise ValueError(
            f"{what}={value!r} is not a number; pass a positive real "
            "number or a scalar array, the rounds one unit of duration lasts"
        )
    value = float(value)
    if not value > 0 or not math.isfinite(value):
        raise ValueError(
            f"{what}={value!r} is not positive and finite; pass the "
            "number of rounds one unit of duration lasts"
        )
    return value


def _option_rounds(unit: float, scale: float, what: str, clock: str, level=None) -> int:
    """The rounds an option of duration ``unit`` lasts at ``scale`` rounds per unit, which must be
    a whole number of at least one; ``what`` names the duration, ``clock`` the scale and
    ``level`` the level (``None``: the chosen one) in the refusals."""
    value = unit * scale
    rounds = round(value)
    at = "the chosen level" if level is None else f"level {level}"
    if value < 1 - 1e-5:
        raise ValueError(
            f"{what} at {at} is {unit:g}, which times {clock} is shorter than one round"
        )
    if abs(value - rounds) > 1e-5 * max(1.0, abs(value)):
        raise ValueError(
            f"{what} at {at} is {unit:g}, which times {clock} is not a whole number of rounds"
        )
    return rounds
