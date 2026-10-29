"""Every refusal ``build`` and ``describe`` can say, each fired by one malformed input.

The reader's table: a base document, one edit that makes one fault, and the sentence ``build``
refuses it with (one sentence, naming the address and what to write instead). The writer's
table: a System or params ``describe`` cannot write, and its sentence. Design:
docs/design/model-description-0.3.md; the format: docs/description-format.md.
"""

from __future__ import annotations

import copy
import dataclasses
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hybrid_aif as H
from hybrid_aif.description import DescriptionError, build, describe
from hybrid_aif.distributions import InverseGamma
from hybrid_aif.factors import LinearGaussianObservationMNIW
from hybrid_aif.reference.raif.agent import raif_mountain_car
from tests.test_description import _examples, _lg_model
from tests.test_system import tmaze_system

KEY = jax.random.PRNGKey(0)


def _doc(system, params, mode="inline"):
    document, arrays = describe(system, params, arrays=mode)
    return json.loads(json.dumps(document)), arrays


@pytest.fixture(scope="module")
def bases():
    examples = _examples()
    sensor = H.LinearGaussianObservation(
        H=jnp.asarray([[1.0]]),
        c=jnp.zeros(1),
        noise=InverseGamma(alpha=jnp.asarray([3.0]), beta=jnp.asarray([0.5])),
    )
    mniw = LinearGaussianObservationMNIW.from_head_noise(
        jnp.asarray([[1.5, 0.3]]), jnp.asarray([[0.4]]), strength=2.0
    )
    agent = raif_mountain_car(KEY)
    drawn = build(examples["amortized"], key=KEY)
    return {
        "X3D": _doc(*drawn),
        "T": _doc(*tmaze_system()),
        "X1": (examples["tabular"], {}),
        "X2": (examples["hybrid"], {}),
        "X3": (examples["amortized"], {}),
        "X4": (examples["composite"], {}),
        "IG": _doc(*_lg_model(sensor)),
        "MN": _doc(*_lg_model(mniw)),
        "RA": _doc(agent.system, agent.params, "file"),
    }


def at(document, path):
    node = document
    for key in path:
        node = node[key]
    return node


def put(*path_value):
    *path, value = path_value

    def edit(document):
        at(document, path[:-1])[path[-1]] = copy.deepcopy(value)

    return edit


def drop(*path):
    def edit(document):
        del at(document, path[:-1])[path[-1]]

    return edit


def both(*edits):
    def edit(document):
        for e in edits:
            e(document)

    return edit


def rename_var(old, new):
    """Rename a variable everywhere it is named in the document's text."""

    def edit(document):
        text = json.dumps(document).replace(f'"{old}"', f'"{new}"')
        document.clear()
        document.update(json.loads(text))

    return edit


F = "factors"
V = "variables"
CL = "clamps"
CONF = "configuration"
OBJ = "objective"
REL = "relations"
TR = ("factors", "location", "transition")
X_TR = ("factors", "x", "transition")
X_LIK = ("factors", "y", "likelihood")


def _g_tail(document):
    return next(t for t in document[OBJ]["terms"] if t["term"] == "g_tail")


def _mu(document):
    return next(t for t in document[OBJ]["terms"] if t["term"] == "model_uncertainty")


def _clone(old, new):
    """A copy of variable ``old`` (its record and its clamp) under the name ``new``."""

    def edit(document):
        document[V][new] = copy.deepcopy(document[V][old])
        document[CL][new] = copy.deepcopy(document[CL][old])

    return edit


def _forget(*names):
    def edit(document):
        for name in names:
            del document[V][name], document[CL][name]

    return edit


def _new(name, record, clamp):
    """A fresh clamped parameter variable."""
    return both(put(V, name, record), put(CL, name, clamp))


def _crspp(document):
    return next(t for t in document[OBJ]["terms"] if "model" in t)


CASES = [
    # ------------------------------------------------------------------ sections
    ("no variables", "T", drop(V), r"the document has no 'variables' section"),
    ("variables not an object", "T", put(V, []), r"variables must be an object, got list"),
    ("no configuration", "T", drop(CONF), r"the document has no 'configuration' section"),
    (
        "model with an unknown key",
        "RA",
        lambda d: _crspp(d)["model"].update(extra={}),
        r"a term's model has key\(s\) \['extra'\]",
    ),
    # ------------------------------------------------------------------ variables
    (
        "variable name with a bracket",
        "T",
        put(V, "odd[1]", {"domain": "real", "dim": 1}),
        r"variables\['odd\[1\]'\]: a variable name is non-empty and contains no bracket",
    ),
    (
        "leaf with an unknown key",
        "T",
        put(V, "location", "colour", "red"),
        r"variables\['location'\] has key\(s\) \['colour'\]",
    ),
    (
        "unknown domain",
        "T",
        put(V, "location", "domain", "boolean"),
        r"unknown domain 'boolean'; the domains are",
    ),
    (
        "levels not positive",
        "T",
        put(V, "location", "levels", 0),
        r"variables\['location'\].levels must be a positive integer, got 0",
    ),
    (
        "levels on a real domain",
        "T",
        put(V, "location", {"domain": "real", "levels": 2}),
        r"levels sizes a categorical domain",
    ),
    (
        "dim beside shape",
        "T",
        put(V, "D_location", {"domain": "real", "dim": 2, "shape": [5]}),
        r"dim sizes a per-step real domain, and a parameter's size is its shape",
    ),
    (
        "bounds without dim",
        "T",
        put(V, "D_location", {"domain": "real", "low": 0, "shape": [5]}),
        r"low and high bound a real domain with its dim",
    ),
    (
        "shape not a list of sizes",
        "T",
        put(V, "D_location", "shape", [5, -1]),
        r"variables\['D_location'\].shape must be a list of sizes",
    ),
    (
        "record not a record",
        "T",
        put(V, "location", "five"),
        r"variables\['location'\] must be a variable record",
    ),
    # ------------------------------------------------------------------ factors
    (
        "factors for an undeclared child",
        "T",
        put(F, "ghost", {}),
        r"factors\['ghost'\]: no variable of that name; declare it under variables",
    ),
    (
        "factor block not an object",
        "T",
        put(F, "location", []),
        r"factors\['location'\] must be an object keyed by label",
    ),
    ("record not an object", "T", put(*TR, []), r"must be a factor record \(an object\)"),
    (
        "record with an unknown key",
        "T",
        put(*TR, "weight", 1.0),
        r"has key\(s\) \['weight'\]; a factor record has",
    ),
    (
        "generic steps unknown",
        "T",
        both(
            put(F, "location", "moves", {"family": "ConditionalCategorical", "steps": "often"}),
            drop(*TR),
        ),
        r"steps must be one of \['0', '1\+', 'all'\], got 'often'",
    ),
    (
        "parents not a list",
        "T",
        put(*TR, "parents", "location"),
        r"parents must be a list of variable names",
    ),
    (
        "parent not a name",
        "T",
        put(*TR, "parents", ["location", 3]),
        r"a parent is a variable name, got 3",
    ),
    (
        "unknown parent",
        "T",
        put(*TR, "parents", ["locaton", "u_location"]),
        r"names 'locaton', which is not a variable; the variables are \['location'",
    ),
    (
        "parent twice",
        "T",
        put(*TR, "parents", ["location", "location"]),
        r"parents names 'location' twice",
    ),
    (
        "parameters not an object",
        "T",
        put(*TR, "parameters", "B_location"),
        r"parameters must be an object",
    ),
    (
        "initial reads t-1",
        "T",
        put(F, "context", "initial", "parents", ["location[t-1]"]),
        r"a record under 'initial' reads its parents at the first step; write location",
    ),
    (
        "likelihood reads t-1",
        "T",
        put(F, "cue", "likelihood", "parents", ["location[t-1]", "context"]),
        r"a record under 'likelihood' reads its parents at the child's step; write location",
    ),
    (
        "actuator reads t-1",
        "X1",
        put(F, "a_location", "actuator", "parents", ["u_location[t-1]"]),
        r"a record under 'actuator' reads its parents at the child's step; write u_location",
    ),
    (
        "recognition reads t-1",
        "X3",
        put(F, "embed", "recognition", "parents", ["state[t-1]"]),
        r"a record under 'recognition' reads its parents at the child's step; write state",
    ),
    (
        "terminal reads t-1",
        "T",
        put(F, "location", "terminal", {"family": "Categorical", "parents": ["context[t-1]"]}),
        r"a record under 'terminal' reads its parents at the plan's end; write context",
    ),
    (
        "prior family not a prior",
        "X1",
        put(F, "A_cue", "prior", "family", "Categorical"),
        r"a Categorical record is not written under the label 'prior'; a parameter's prior is one",
    ),
    (
        "prior with parents",
        "X1",
        put(F, "A_cue", "prior", "parents", ["cue"]),
        r"a prior on a parameter takes no parents",
    ),
    # ------------------------------------------------------------------ relations
    (
        "relations unknown key",
        "T",
        put(REL, "bound", []),
        r"relations has key\(s\) \['bound'\]; the relations are",
    ),
    (
        "controls not an object",
        "T",
        put(REL, "controls", ["u_location"]),
        r"relations.controls maps each control node to its action node",
    ),
    (
        "relations names an unknown variable",
        "T",
        put(REL, "observed", ["cue", "smell"]),
        r"relations names 'smell', which is not a variable",
    ),
    (
        "a variable in two relations",
        "T",
        put(REL, "observed", ["cue", "reward", "location_obs", "u_location"]),
        r"relations names 'u_location' as a control node in relations.controls and as observed",
    ),
    (
        "controls pairs a latent",
        "T",
        put(REL, "controls", {"u_location": "context"}),
        r"relations.controls pairs 'u_location' with 'context'.*carries a record under "
        r"'initial'; pair",
    ),
    (
        "external input not real",
        "X3",
        both(put(V, "ext", {"domain": "categorical", "levels": 2}), put(REL, "external", ["ext"])),
        r"variables\['ext'\]: an external input is a vector, domain 'real' with dim",
    ),
    (
        "external input read by a transition",
        "X3",
        both(
            put(V, "ext", {"domain": "real", "dim": 2}),
            put(REL, "external", ["ext"]),
            lambda d: d[F]["deter"]["transition"]["parents"].append("ext"),
        ),
        r"reads the external input\(s\) \['ext'\]; an external input is read by recognition",
    ),
    (
        "record under an external input",
        "X3",
        both(
            put(V, "ext", {"domain": "real", "dim": 2}),
            put(REL, "external", ["ext"]),
            put(F, "ext", {"start": {"family": "Gaussian", "steps": "0"}}),
        ),
        r"'ext' is an external input, data the System receives; it takes no record",
    ),
    # ------------------------------------------------------------------ configuration
    (
        "configuration unknown sub-block",
        "T",
        put(CONF, "acting", {}),
        r"configuration has sub-block\(s\) \['acting'\]",
    ),
    (
        "configuration sub-block not an object",
        "T",
        put(CONF, "inference", 3),
        r"configuration.inference must be an object",
    ),
    (
        "configuration unknown field",
        "T",
        put(CONF, "planning", "candidates", 64),
        r"configuration.planning has field\(s\) \['candidates'\].*call-time arguments",
    ),
    (
        "horizon not an integer",
        "T",
        put(CONF, "planning", "horizon", True),
        r"configuration.planning.horizon must be a positive integer \(the plan length\), got True",
    ),
    (
        "iterations zero",
        "T",
        put(CONF, "inference", {"iterations": 0}),
        r"configuration.inference.iterations must be a positive integer",
    ),
    (
        "mixture budget negative",
        "T",
        put(CONF, "inference", {"mixture_k": -3}),
        r"configuration.inference.mixture_k must be a positive integer",
    ),
    (
        "planner not registered",
        "T",
        put(CONF, "planning", "planner", "nope"),
        r"planner names 'nope'; the registered planners are \['cem', 'enumeration'",
    ),
    (
        "gamma not a number",
        "T",
        put(CONF, "planning", "gamma", "high"),
        r"configuration.planning.gamma is the policy precision, a number; got 'high'",
    ),
    (
        "branch_on names no observation",
        "T",
        both(
            put(CONF, "planning", "planner", "sophisticated"),
            put(CONF, "planning", "branch_on", ["nope"]),
        ),
        r"branch_on lists observed variables .*got \['nope'\]",
    ),
    (
        "branch_on without a branching planner",
        "T",
        put(CONF, "planning", "branch_on", ["cue"]),
        r"which only a branching planner .* branches on, and the planner is 'enumeration'",
    ),
    (
        "mixture schedule unknown",
        "T",
        put(CONF, "inference", {"mixture_schedule": "greedy"}),
        r"mixture_schedule is one of \['runnalls', 'mode_aligned'\], got 'greedy'",
    ),
    (
        "frozen not a list",
        "T",
        put(CONF, "learning", {"frozen": "B_location"}),
        r"configuration.learning.frozen must be a list of variable names",
    ),
    (
        "frozen names no variable",
        "T",
        put(CONF, "learning", {"frozen": ["ghost"]}),
        r"configuration.learning.frozen names 'ghost', which is not a variable",
    ),
    (
        "frozen names a clamped table",
        "T",
        put(CONF, "learning", {"frozen": ["B_location"]}),
        r"names 'B_location', which no trainer moves",
    ),
    # ------------------------------------------------------------------ objective records
    ("objective not an object", "T", put(OBJ, []), r"objective must be an object"),
    (
        "objective unknown key",
        "T",
        put(OBJ, "weights", {}),
        r"objective has key\(s\) \['weights'\]; its keys are decomposition and terms",
    ),
    ("terms not a list", "T", put(OBJ, {"terms": {}}), r"objective.terms must be a list"),
    (
        "term without a name",
        "T",
        put(OBJ, {"terms": [{"on": "cue"}]}),
        r"objective.terms\[0\] must be a term record with a 'term' name",
    ),
    (
        "term with an unknown key",
        "T",
        put(OBJ, {"terms": [{"term": "utility", "weight": 2}]}),
        r"objective.terms\[0\] has key\(s\) \['weight'\]",
    ),
    (
        "term on no variable",
        "T",
        put(OBJ, {"terms": [{"term": "infogain", "on": "ghost"}]}),
        r"objective.terms\[0\].on names 'ghost', which is not a variable",
    ),
    (
        "term parameter no variable",
        "X3",
        lambda d: _g_tail(d)["parameters"].update(critic="ghost"),
        r"parameters names 'ghost', which is not a variable",
    ),
    (
        "decomposition unknown",
        "T",
        put(OBJ, {"decomposition": "free_energy"}),
        r"objective.decomposition is 'utility_infogain' or 'risk_ambiguity', got 'free_energy'",
    ),
    (
        "terminal_risk listed",
        "T",
        put(OBJ, {"terms": [{"term": "terminal_risk"}]}),
        r"terminal_risk is scored by a terminal record",
    ),
    (
        "two infogain terms, one without on",
        "T",
        put(OBJ, {"terms": [{"term": "infogain"}, {"term": "infogain", "on": "location"}]}),
        r"an infogain term without on selects every eligible variable",
    ),
    (
        "infogain owns no model",
        "T",
        put(OBJ, {"terms": [{"term": "infogain", "model": {}}]}),
        r"an infogain term owns no model",
    ),
    (
        "infogain options without parameters",
        "T",
        put(OBJ, {"terms": [{"term": "infogain", "options": {}}]}),
        r"options describe a term's parameters",
    ),
    (
        "infogain parameters no estimator takes",
        "X3",
        both(
            _clone("g_tail_critic", "critic2"),
            lambda d: d[OBJ]["terms"].append(
                {"term": "infogain", "parameters": {"critic": "critic2"}}
            ),
        ),
        r"no infogain estimator takes the parameters \['critic'\]",
    ),
    (
        "infogain on a control node",
        "T",
        put(OBJ, {"terms": [{"term": "infogain", "on": "u_location"}]}),
        r"infogain on 'u_location', which is not a latent, an observation or a parameter",
    ),
    (
        "row with no target given on",
        "X3",
        lambda d: _g_tail(d).update(on="deter"),
        r"the term 'g_tail' has no target, so it takes no on",
    ),
    (
        "slot row on an observation",
        "T",
        put(OBJ, {"terms": [{"term": "occupancy", "on": "cue"}]}),
        r"the term 'occupancy' reads a latent's transition, and 'cue' has none",
    ),
    (
        "observation row on a latent",
        "X3",
        lambda d: d[OBJ]["terms"][0].update(on="deter"),
        r"the term 'predicted_reward' reads an observation; 'deter' is not one",
    ),
    (
        "latent row on an observation",
        "RA",
        lambda d: _mu(d).update(on="reward"),
        r"the term 'model_uncertainty' reads a latent; 'reward' is not one",
    ),
    (
        "term options keyed apart from parameters",
        "X3",
        lambda d: _g_tail(d).update(options={}),
        r"options gives each parameter's family, configuration and layout, keyed as its parameters",
    ),
    (
        "term parameters unknown",
        "X3",
        lambda d: _g_tail(d).update(
            parameters={"value": "g_tail_critic"},
            options={"value": _g_tail(d)["options"]["critic"]},
        ),
        r"the term 'g_tail' takes the parameter\(s\) \['critic', 'slow'\]",
    ),
    (
        "ensemble not a list",
        "RA",
        both(
            lambda d: _mu(d)["parameters"].update(ensemble="model_uncertainty_ensemble_0"),
            _forget(*(f"model_uncertainty_ensemble_{i}" for i in (1, 2, 3))),
        ),
        r"parameters.ensemble is a list of variables",
    ),
    (
        "ensemble options of the wrong count",
        "RA",
        lambda d: _mu(d)["options"].update(ensemble=[_mu(d)["options"]["ensemble"]]),
        r"options.ensemble gives one record per variable",
    ),
    (
        "a later parameter without the earlier",
        "RA",
        both(
            lambda d: (_g_tail(d)["parameters"].pop("critic"), _g_tail(d)["options"].pop("critic")),
            _forget("g_tail_critic"),
        ),
        r"names its parameters in order \['critic', 'slow'\]",
    ),
    (
        "model on a row without one",
        "X3",
        lambda d: _g_tail(d).update(model={}),
        r"the term 'g_tail' owns no model",
    ),
    (
        "model not a document",
        "RA",
        lambda d: _crspp(d).update(model=[]),
        r"model is a nested document",
    ),
    (
        "a second model",
        "RA",
        lambda d: d[OBJ]["terms"].append(copy.deepcopy(_crspp(d))),
        r"a second 'cosine_sim_to_pref_rollout' term gives its model",
    ),
    (
        "parameters on a row without any",
        "T",
        put(OBJ, {"terms": [{"term": "occupancy", "on": "location", "parameters": {}}]}),
        r"the term 'occupancy' takes no parameters",
    ),
    (
        "a second set of parameters",
        "X3",
        both(
            _clone("g_tail_critic", "critic2"),
            lambda d: d[OBJ]["terms"].append(
                {**copy.deepcopy(_g_tail(d)), "parameters": {"critic": "critic2"}}
            ),
        ),
        r"a second 'g_tail' term gives its parameters",
    ),
    (
        "term parameter of no term family",
        "X3",
        lambda d: _g_tail(d)["options"]["critic"].update(family="Categorical"),
        r"a term parameter's options give its family, one of",
    ),
    (
        "term parameter option unknown",
        "X3",
        lambda d: _g_tail(d)["options"]["critic"].update(depth=2),
        r"PredictorHead has no option\(s\) \['depth'\]",
    ),
    (
        "term parameter layout missing",
        "X3",
        lambda d: _g_tail(d)["options"]["critic"].pop("parent_dims"),
        r"a term parameter has no wiring, so its options give its layout; write \['parent_dims'\]",
    ),
]


CASES += [
    # ------------------------------------------------------------------ kinds and roles
    (
        "per-step variable with a parameter record",
        "T",
        put(V, "context", {"domain": "real", "shape": [2]}),
        r"'context' is per-step \(used at .*\), so its record is",
    ),
    (
        "parameter under another component",
        "X4",
        both(
            put(F, "agent.seen", "likelihood", "parameters", "log_table", "world.D_location"),
            put(F, "world.location", "initial", "parameters", "logits", "agent.A_seen"),
        ),
        r"names 'agent.A_seen', a parameter of the component 'agent', and its child belongs to "
        r"'world'",
    ),
    (
        "bounds on a latent",
        "X2",
        put(V, "x", "low", 0),
        r"bounds \(low, high\) give a continuous control node's box, and 'x' is not a control",
    ),
    (
        "continuous control node without a box",
        "X3",
        put(V, "u_deter", {"domain": "real", "dim": 1}),
        r"a continuous control node's domain carries its box: give low and high",
    ),
    (
        "box with low above high",
        "X3",
        put(V, "u_deter", "low", 2.0),
        r"variables\['u_deter'\]: its bounds are not a box",
    ),
    (
        "target under an external input",
        "X3",
        both(
            put(V, "ext", {"domain": "real", "dim": 2}),
            put(REL, "external", ["ext"]),
            put(F, "ext", {"target": {"family": "Gaussian"}}),
        ),
        r"a target sits under a latent, an observation or a control node, and 'ext' is none",
    ),
    (
        "terminal under an observation",
        "T",
        put(
            F,
            "reward",
            "terminal",
            copy.deepcopy({"family": "Categorical", "parameters": {"logits": "C_reward"}}),
        ),
        r"a terminal sits under a latent or a control node, and 'reward' is none of these; write"
        r" it",
    ),
    (
        "terminal under a control node with parents",
        "T",
        put(
            F,
            "u_location",
            {"terminal": {"family": "ConditionalCategorical", "parents": ["location"]}},
        ),
        r"a terminal under the control node 'u_location' is unconditional; drop its parents",
    ),
    (
        "terminal reads an observation",
        "T",
        put(F, "location", "terminal", {"family": "ConditionalCategorical", "parents": ["cue"]}),
        r"a terminal's parents are latents, and 'cue' is not one",
    ),
    (
        "conditional Gaussian terminal",
        "X2",
        put(F, "x", "terminal", {"family": "Gaussian", "parents": ["mode"]}),
        r"a terminal with parents is a table over them \(ConditionalCategorical\), and the engine "
        r"reads no conditional Gaussian",
    ),
    (
        "policy prior with parents",
        "T",
        put(F, "u_location", {"policy_prior": {"family": "Categorical", "parents": ["location"]}}),
        r"the policy prior takes no parents",
    ),
    (
        "policy prior under a latent",
        "T",
        put(F, "location", "policy_prior", {"family": "Categorical"}),
        r"the policy prior sits under the first control node",
    ),
    (
        "recognition under an observation",
        "X3",
        put(F, "reward", "recognition", {"family": "MLPPredictor", "parents": ["deter"]}),
        r"a recognition record sits under a latent or an auxiliary variable, and 'reward' is"
        r" observed",
    ),
    (
        "action node with a generic first-step record",
        "X1",
        both(
            drop(F, "a_location", "actuator"),
            put(
                F,
                "a_location",
                "slip",
                {
                    "family": "ConditionalCategorical",
                    "steps": "0",
                    "parents": ["u_location[t]"],
                    "parameters": {"log_table": "K_a_location"},
                },
            ),
        ),
        r"'a_location' is an action node, whose one record is its actuator",
    ),
    (
        "actuator reads a latent",
        "X1",
        put(F, "a_location", "actuator", "parents", ["location"]),
        r"an actuator reads its control node at the same step: write parents \['u_location'\]",
    ),
    (
        "observation with an initial",
        "T",
        put(F, "cue", "initial", {"family": "Categorical"}),
        r"'cue' is observed, and an observation takes a likelihood \(every step\) and a target",
    ),
    (
        "likelihood reads a control node",
        "T",
        put(F, "cue", "likelihood", "parents", ["u_location", "context"]),
        r"a likelihood reads latents or action nodes at the observation's step; \['u_location'\]",
    ),
    (
        "likelihood under a latent",
        "T",
        put(F, "context", "likelihood", {"family": "ConditionalCategorical"}),
        r"a likelihood sits under an observed variable or an action node, and 'context' is a"
        r" latent",
    ),
    (
        "a latent without a transition",
        "T",
        both(drop(F, "context", "transition"), _forget("B_context")),
        r"the latent 'context' has no transition record; every latent has an initial and a"
        r" transition",
    ),
    (
        "a latent without an initial",
        "T",
        both(drop(F, "context", "initial"), _forget("D_context")),
        r"the latent 'context' has no initial record",
    ),
    (
        "initial reads an observation",
        "T",
        put(
            F,
            "context",
            "initial",
            {
                "family": "ConditionalCategorical",
                "parents": ["cue"],
                "parameters": {"log_table": "D_context"},
            },
        ),
        r"an initial record's parents are latents, and 'cue' is not one",
    ),
    (
        "conditional Gaussian initial",
        "X2",
        put(F, "x", "initial", "parents", ["mode"]),
        r"an initial record with parents is a table over them \(ConditionalCategorical\), and the "
        r"engine reads no conditional MixtureGaussian prior",
    ),
    (
        "control node read by no transition",
        "T",
        put(*TR, "parents", ["location"]),
        r"the control node 'u_location' is read by no transition",
    ),
    (
        "transition reads the action node",
        "T",
        put(*TR, "parents", ["location", "a_location"]),
        r"'a_location' is the action node of 'u_location'; a transition reads the control node, "
        r"so write 'u_location' in its place",
    ),
    (
        "recognition reads a control node",
        "X3",
        put(F, "embed", "recognition", "parents", ["u_deter"]),
        r"a recognition record reads latents .*; 'u_deter' cannot be read",
    ),
    (
        "recognition records in a cycle",
        "X3",
        both(
            put(V, "code", {"domain": "real", "dim": 3}),
            lambda d: d[F].update(
                code={
                    "recognition": {
                        **copy.deepcopy(d[F]["embed"]["recognition"]),
                        "parents": ["embed"],
                    }
                }
            ),
            put(F, "embed", "recognition", "parents", ["code"]),
        ),
        r"the recognition records of \['embed', 'code'\] read one another in a cycle \(embed -> "
        r"code -> embed\)",
    ),
    # ------------------------------------------------------------------ transitions
    (
        "transition reads its own child at t (table)",
        "T",
        put(*TR, "parents", ["location[t]", "u_location"]),
        r"a transition reads its own child at the step before, never at the child's step; write "
        r"location\[t-1\]",
    ),
    (
        "transition reads its own child at t (linear-Gaussian)",
        "X2",
        put(*X_TR, "parents", ["x[t]", "mode[t]"]),
        r"a transition reads its own child at the step before.*write x\[t-1\]",
    ),
    (
        "code names itself",
        "X3",
        put(F, "stoch", "transition", "parents", ["stoch", "deter[t]"]),
        r"a code reads its own past through its other parents; it cannot name itself",
    ),
    (
        "control node read at t",
        "T",
        put(*TR, "parents", ["location", "u_location[t]"]),
        r"a transition reads the control node 'u_location' at the step it is chosen",
    ),
    (
        "MLPConditional transition reads a latent",
        "X2",
        both(
            put(
                *X_TR,
                {
                    "family": "MLPConditional",
                    "parents": ["x", "mode"],
                    "parameters": {"layers": "x_transition_F"},
                },
            ),
            _forget("x_transition_innovations_loc", "x_transition_innovations_chol"),
        ),
        r"an MLPConditional transition reads its own previous value and control nodes only",
    ),
    (
        "transition reads an observation",
        "T",
        put(*TR, "parents", ["location", "u_location", "cue"]),
        r"a transition reads latents and control nodes; 'cue' is neither",
    ),
    (
        "self-reader without its own value first",
        "X2",
        put(*X_TR, "parents", ["mode[t]", "x"]),
        r"list 'x' first among its parents",
    ),
    (
        "linear-Gaussian reads its mode a step back",
        "X2",
        put(*X_TR, "parents", ["x", "mode"]),
        r"reads the latent 'mode' at the child step: write mode\[t\]",
    ),
    (
        "same-step misorder",
        "X2",
        lambda d: d.update(variables={"x": d[V].pop("x"), **d[V]}),
        r"so 'mode' must be declared before 'x'",
    ),
    # ------------------------------------------------------------------ clamps
    ("clamp for no variable", "T", put(CL, "ghost", 1.0), r"clamps\['ghost'\]: no variable of"),
    (
        "clamp on a per-step variable",
        "T",
        put(CL, "location", 1.0),
        r"clamps\['location'\]: 'location' is per-step, and a clamp gives a parameter's value",
    ),
    (
        "one record naming a variable twice",
        "X2",
        both(put(F, "x", "target", "parameters", "chol", "x_target_loc"), _forget("x_target_chol")),
        r"names 'x_target_loc' twice; a variable fills one field",
    ),
    (
        "two records naming one variable",
        "T",
        both(put(F, "cue", "likelihood", "parameters", "log_table", "A_reward"), _forget("A_cue")),
        r"names 'A_reward', which factors\['reward'\]\['likelihood'\] names already",
    ),
    (
        "probabilities with a negative entry",
        "T",
        both(
            put(V, "B_context", "domain", "simplex"),
            put(CL, "B_context", [[[1.5], [0.0]], [[-0.5], [1.0]]]),
        ),
        r"clamps\['B_context'\] must be probabilities \(domain 'simplex': entries >= 0 and each "
        r"distribution sums to one\); normalise it",
    ),
    (
        "probabilities not summing to one",
        "T",
        both(
            put(V, "B_context", "domain", "simplex"),
            put(CL, "B_context", [[[0.6], [0.0]], [[0.6], [1.0]]]),
        ),
        r"clamps\['B_context'\] must be probabilities \(domain 'simplex'",
    ),
    (
        "an actuator kernel declared logits",
        "T",
        both(
            put(V, "K_a_location", {"domain": "logits", "shape": [5, 5]}),
            put(CL, "K_a_location", 0.0),
            put(
                F,
                "a_location",
                {
                    "actuator": {
                        "family": "ConditionalCategorical",
                        "parents": ["u_location"],
                        "parameters": {"log_table": "K_a_location"},
                    }
                },
            ),
        ),
        r"an actuator kernel is a probability table, so 'K_a_location' holds normalised "
        r"log-probabilities, domain 'log_simplex' \(or probabilities, domain 'simplex'\)",
    ),
    (
        "ref without a key",
        "T",
        put(CL, "B_context", {"ref": "B_context"}),
        r"a ref is written \{'ref': '<file>#<key>'\}, got 'B_context'",
    ),
    (
        "init on a table",
        "T",
        put(CL, "B_context", {"init": {}}),
        r"an init recipe draws a network's weights or a learned vector, not this parameter",
    ),
    (
        "clamp not numbers",
        "T",
        put(CL, "B_context", "abc"),
        r"the clamp is not a number or a nested list of numbers",
    ),
    (
        "clamp shape disagrees",
        "T",
        put(CL, "D_location", [0.0, 0.0]),
        r"clamps\['D_location'\]: the clamp has shape \[2\] and the variable declares \[5\]",
    ),
    (
        "parameter record without a shape",
        "T",
        put(V, "D_location", {"domain": "logits"}),
        r"a parameter variable's record gives its shape",
    ),
    # ------------------------------------------------------------------ priors
    (
        "prior missing a hyperparameter",
        "X1",
        both(put(F, "A_cue", "prior", "parameters", {}), _forget("A_cue_concentration")),
        r"a Dirichlet prior needs 'concentration'",
    ),
    (
        "hyperparameter with a prior of its own",
        "X1",
        both(
            drop(CL, "A_cue_concentration"),
            _new("h", {"domain": "positive", "shape": [2, 2]}, 1.0),
            put(
                F,
                "A_cue_concentration",
                {"prior": {"family": "Dirichlet", "parameters": {"concentration": "h"}}},
            ),
        ),
        r"the hyperparameter 'A_cue_concentration' has a prior of its own",
    ),
    (
        "prior with an unknown parameter",
        "X1",
        both(
            _new("h", {"domain": "positive", "shape": [2]}, 1.0),
            put(F, "A_cue", "prior", "parameters", "scale", "h"),
        ),
        r"Dirichlet has no parameter\(s\) \['scale'\]; its parameters are",
    ),
    (
        "prior shape disagrees",
        "X1",
        put(V, "A_cue", "shape", [2, 3]),
        r"the prior's concentration has shape \[2, 2\] and the variable 'A_cue' declares \[2, 3\]",
    ),
    (
        "network weights with a prior",
        "X3",
        both(
            drop(CL, "embed_recognition_layers"),
            _new("h", {"domain": "positive", "shape": [2]}, 1.0),
            put(
                F,
                "embed_recognition_layers",
                {"prior": {"family": "Dirichlet", "parameters": {"concentration": "h"}}},
            ),
        ),
        r"a network's weights are clamped values a trainer moves, and take no prior record",
    ),
    (
        "actuator kernel learned",
        "X1",
        both(
            drop(CL, "K_a_location"),
            put(
                F,
                "K_a_location",
                {
                    "prior": {
                        "family": "Dirichlet",
                        "parameters": {"concentration": "A_cue_concentration"},
                    }
                },
            ),
            put(F, "A_cue", "prior", "parameters", "concentration", "K2"),
            put(V, "K2", {"domain": "positive", "shape": [2, 2]}),
            put(CL, "K2", 1.0),
        ),
        r"an actuator kernel is clamped; the engine does not learn it",
    ),
    (
        "actuator kernel as logits",
        "X1",
        put(V, "K_a_location", "domain", "logits"),
        r"an actuator kernel is a probability table, so 'K_a_location' holds normalised",
    ),
    (
        "clamped field with a prior",
        "X2",
        both(
            drop(CL, "x_target_loc"),
            _new("h", {"domain": "positive", "shape": [1]}, 1.0),
            put(
                F,
                "x_target_loc",
                {"prior": {"family": "Dirichlet", "parameters": {"concentration": "h"}}},
            ),
        ),
        r"the Gaussian field 'loc' is clamped; it takes no prior",
    ),
    (
        "linear-Gaussian transition learned",
        "X2",
        both(
            drop(CL, "x_transition_F"),
            _new("h", {"domain": "positive", "shape": [2, 1, 1]}, 1.0),
            put(
                F,
                "x_transition_F",
                {"prior": {"family": "Dirichlet", "parameters": {"concentration": "h"}}},
            ),
        ),
        r"a linear-Gaussian transition is clamped",
    ),
    (
        "row covariance on an inverse-gamma noise",
        "IG",
        both(
            drop(CL, "y_likelihood_head"),
            put(
                F,
                "y_likelihood_head",
                {
                    "prior": {
                        "family": "MatrixNormal",
                        "parameters": {"M": "hm", "V_inv": "hv", "row_cov": "y_likelihood_noise"},
                    }
                },
            ),
            put(V, "hm", {"domain": "real", "shape": [1, 2]}),
            put(CL, "hm", [[1.0, 0.0]]),
            put(V, "hv", {"domain": "spd", "shape": [2, 2]}),
            put(CL, "hv", [[1.0, 0.0], [0.0, 1.0]]),
        ),
        r"row_cov names 'y_likelihood_noise', whose prior is InverseGamma",
    ),
    (
        "row covariance naming another variable",
        "MN",
        put(F, "y_likelihood_head", "prior", "parameters", "row_cov", "y_likelihood_head_M"),
        r"row_cov names 'y_likelihood_head_M', and a head's row covariance is its likelihood's"
        r" noise",
    ),
    (
        "noise prior point disagrees with the variable",
        "IG",
        put(V, "y_likelihood_noise", "shape", [2, 2]),
        r"the noise prior's point has shape \[1, 1\] and 'y_likelihood_noise' declares \[2, 2\]",
    ),
]


BARE = {"domain": "real"}
LINEAR = {"weight": [[1.0, 1.0]]}

CASES += [
    # ------------------------------------------------------------------ records against families
    (
        "record missing a parameter field",
        "T",
        both(put(*TR, "parameters", {}), _forget("B_location")),
        r"ConditionalCategorical needs the parameter\(s\) \['log_table'\]",
    ),
    (
        "nested record incomplete",
        "X2",
        both(
            put(*X_TR, "parameters", "innovations", {"loc": "x_transition_innovations_loc"}),
            _forget("x_transition_innovations_chol"),
        ),
        r"parameters.innovations is a nested record naming \['loc', 'chol'\]",
    ),
    (
        "parameter field not a name",
        "X2",
        both(put(*X_TR, "parameters", "F", 3), _forget("x_transition_F")),
        r"parameters.F must name a variable",
    ),
    (
        "Gaussian over a categorical variable",
        "X2",
        both(lambda d: d[F]["mode"].update(target=d[F]["x"].pop("target"))),
        r"a Gaussian is over a real variable, and 'mode' is categorical",
    ),
    (
        "table reads a real variable",
        "X2",
        put(F, "mode", "transition", "parents", ["u_mode", "x"]),
        r"a table reads and emits categorical variables, and 'x' is not one",
    ),
    (
        "Categorical of the wrong length",
        "T",
        both(put(V, "D_location", "shape", [4]), put(CL, "D_location", 0.0)),
        r"'D_location' has shape \[4\], and a Categorical over 'location' has one entry per level",
    ),
    (
        "policy prior over a continuous control node",
        "X3",
        both(
            _new("E", {"domain": "logits", "shape": [4]}, 0.0),
            put(
                F,
                "u_deter",
                {"policy_prior": {"family": "Categorical", "parameters": {"logits": "E"}}},
            ),
        ),
        r"a policy prior is a distribution over the enumerated policy set, and a continuous",
    ),
    (
        "Dirichlet as a generic transition",
        "T",
        both(
            drop(*TR),
            put(
                F,
                "location",
                "moves",
                {
                    "family": "Dirichlet",
                    "parents": ["location[t-1]", "u_location[t-1]"],
                    "steps": "1+",
                    "parameters": {"concentration": "B_location"},
                },
            ),
        ),
        r"a Dirichlet record is a prior under a parameter",
    ),
    (
        "hyperparameter not a name",
        "X1",
        both(
            put(F, "A_cue", "prior", "parameters", "concentration", 3),
            _forget("A_cue_concentration"),
        ),
        r"parameters.concentration must name a variable",
    ),
    # ------------------------------------------------------------------ sizes against the variables
    (
        "initial mixture of another dim",
        "X2",
        put(V, "x", "dim", 3),
        r"'x_initial_means' has shape \[2, 1\], and over 'x' \(dim 3\) its shape is \[2, 3\]",
    ),
    (
        "target of another dim",
        "X2",
        both(put(V, "x_target_loc", "shape", [3]), put(CL, "x_target_loc", 3.0)),
        r"'x_target_loc' has shape \[3\], and over 'x' \(dim 1\) its shape is \[1\]",
    ),
    (
        "transition innovations of another dim",
        "X2",
        both(
            put(V, "x_transition_innovations_loc", "shape", [2, 3]),
            put(CL, "x_transition_innovations_loc", 0.0),
        ),
        r"'x_transition_innovations_loc' has shape \[2, 3\], and over 'x' \(dim 1\) its shape is "
        r"\[2, 1\] \(one leading axis per discrete parent, in order: \{'mode': 2\}\)",
    ),
    (
        "observation head of another width",
        "X2",
        both(put(V, "y_likelihood_head", "shape", [1, 3]), put(CL, "y_likelihood_head", 0.5)),
        r"'y_likelihood_head' has shape \[1, 3\], and over 'y' \(dim 1\) its last axes are \[1,"
        r" 2\]",
    ),
    (
        "observation of another dim",
        "X2",
        put(V, "y", "dim", 4),
        r"'y_likelihood_head' has shape \[1, 2\], and over 'y' \(dim 4\) its last axes are \[4,"
        r" 2\]",
    ),
    (
        "observation noise of another dim",
        "X2",
        both(
            put(V, "y_likelihood_noise", "shape", [2, 2]),
            put(CL, "y_likelihood_noise", [[0.5, 0.0], [0.0, 0.5]]),
        ),
        r"'y_likelihood_noise' has shape \[2, 2\], and over 'y' \(dim 1\) its last axes are \[1,"
        r" 1\]",
    ),
    (
        "joint noise of another dim",
        "MN",
        put(V, "y", "dim", 2),
        r"'y_likelihood_head' has shape \[1, 2\], and over 'y' \(dim 2\) its last axes are \[2,"
        r" 2\]",
    ),
    (
        "binned reward of dim 7",
        "X3",
        put(V, "reward", "dim", 7),
        r"a symlog_disc head predicts one number, and 'reward' has dim 7",
    ),
    # ------------------------------------------------------------------ domains
    (
        "cholesky not a matrix",
        "X2",
        both(put(V, "x_target_chol", "shape", [1]), put(CL, "x_target_chol", [0.8])),
        r"clamps\['x_target_chol'\] must be a matrix \(domain 'cholesky'\), got shape \[1\]",
    ),
    (
        "cholesky diagonal not positive",
        "X2",
        put(CL, "x_target_chol", [[-0.8]]),
        r"must have a strictly positive diagonal \(domain 'cholesky'\)",
    ),
    (
        "spd not symmetric",
        "MN",
        put(CL, "y_likelihood_head_V_inv", [[2.0, 1.0], [0.0, 2.0]]),
        r"must be symmetric positive definite \(domain 'spd'\)",
    ),
    (
        "module check fails",
        "X3D",
        put(V, "embed", "dim", 7),
        r"factors\['embed'\]\['recognition'\] \(MLPPredictor\): final layer emits 6 outputs, not "
        r"out_dim=7",
    ),
    # ------------------------------------------------------------------ networks
    (
        "linear-Gaussian map with another mode axis",
        "X2",
        both(put(V, "x_transition_F", "shape", [3, 1, 1]), put(CL, "x_transition_F", 0.9)),
        r"'x_transition_F' has shape \[3, 1, 1\], and over 'x' \(dim 1\) its shape is \[2, 1, 1\] "
        r"\(one leading axis per discrete parent, in order: \{'mode': 2\}\)",
    ),
    (
        "learned vector of another width than its carry",
        "X3",
        both(put(V, "deter_initial_value", "shape", [4]), put(CL, "deter_initial_value", 0.0)),
        r"'deter_initial_value' has shape \[4\], and over 'deter' \(dim 8\) its shape is \[8\]",
    ),
    (
        "freely named control node on the amortized path",
        "X3",
        rename_var("u_deter", "go"),
        r"relations.controls pairs 'go' with 'a_deter', and on the amortized path .* write "
        r"\{'u_deter': 'a_deter'\}",
    ),
    (
        "freely named action node on the amortized path",
        "X3",
        rename_var("a_deter", "move"),
        r"relations.controls pairs 'u_deter' with 'move'.*write \{'u_deter': 'a_deter'\}",
    ),
    (
        "control node named after a latent that does not read it",
        "X3",
        both(rename_var("u_deter", "u_stoch"), rename_var("a_deter", "a_stoch")),
        r"relations.controls pairs 'u_stoch' with 'a_stoch'.*write \{'u_deter': 'a_deter'\}",
    ),
    (
        "init clamp with another key",
        "X3",
        put(CL, "embed_recognition_layers", {"init": {}, "x": 1}),
        r"an init clamp is \{'init': \{...\}\}",
    ),
    (
        "init recipe missing its width",
        "X3",
        put(CL, "stoch_transition_trunk", {"init": {}}),
        r"an init recipe for MLPCategorical needs 'width'",
    ),
    (
        "init recipe missing its hidden width",
        "X3",
        put(CL, "deter_transition_map", {"init": {}}),
        r"an init recipe for Deterministic needs 'hidden'",
    ),
    (
        "init recipe that does not draw",
        "X3",
        put(F, "embed", "recognition", "options", "activation", "bogus"),
        r"the init recipe does not draw a network: unknown activation 'bogus'",
    ),
    (
        "tree list of another length",
        "X3D",
        lambda d: d[CL]["embed_recognition_layers"]["layers"].pop(),
        r"clamps\['embed_recognition_layers'\].layers: the clamp's tree differs from the"
        r" variable's",
    ),
    (
        "network tree without layers",
        "X3",
        put(CL, "embed_recognition_layers", {"weights": [[1.0]]}),
        r"a network's tree is \{'layers': \[...\], 'norms': \[...\]\}",
    ),
    (
        "layers that do not chain",
        "X3",
        put(CL, "embed_recognition_layers", {"layers": [LINEAR, {"weight": [[1.0, 1.0, 1.0]]}]}),
        r"layer 1 takes 3 inputs and layer 0 emits 1; each layer reads the one before",
    ),
    (
        "norm option against the tree",
        "X3",
        put(CL, "embed_recognition_layers", {"layers": [LINEAR]}),
        r"options.norm is True and the network's tree carries no norms",
    ),
    (
        "MLPCategorical over a real variable",
        "X3",
        put(
            F,
            "embed",
            "recognition",
            {
                "family": "MLPCategorical",
                "parents": ["state"],
                "parameters": {"trunk": "embed_recognition_layers"},
            },
        ),
        r"an MLPCategorical emits a code, and 'embed' is not categorical",
    ),
    (
        "MLPCategorical init with an activated output",
        "X3",
        put(F, "stoch", "transition", "options", "out_activation", True),
        r"the init recipe of an MLPCategorical draws a trunk with out_activation and symlog_inputs",
    ),
    (
        "PredictorHead without a loss",
        "X3",
        drop(F, "reward", "likelihood", "options", "loss"),
        r"a PredictorHead needs options.loss",
    ),
    (
        "PredictorHead init with an activated output",
        "X3",
        put(F, "reward", "likelihood", "options", "out_activation", True),
        r"the init recipe of a PredictorHead draws a trunk with out_activation off",
    ),
    (
        "named-function map with parameters",
        "X3",
        put(F, "deter", "transition", "options", "map", {"impl": "numpy:tanh"}),
        r"a Deterministic over a named function takes no parameters",
    ),
    (
        "map of an unknown kind",
        "X3",
        put(F, "deter", "transition", "options", "map", "LSTM"),
        r"options.map is 'GRUCell' .* or \{'impl': 'package.module:name'\}, got 'LSTM'",
    ),
    (
        "GRU map without its variable",
        "X3",
        both(put(F, "deter", "transition", "parameters", {}), _forget("deter_transition_map")),
        r"a GRUCell map names its tree variable under parameters.map",
    ),
    (
        "GRU tree malformed",
        "X3",
        put(CL, "deter_transition_map", {"gate": LINEAR}),
        r"a GRUCell's tree is \{'input_proj': ..., 'gate': ..., 'norms': \[...\]\}",
    ),
    (
        "GRU norm option against the tree",
        "X3",
        put(CL, "deter_transition_map", {"input_proj": LINEAR, "gate": LINEAR}),
        r"options.norm disagrees with the map's norms",
    ),
    (
        "learned vector with a prior",
        "X3",
        both(
            drop(CL, "deter_initial_value"),
            _new("h", {"domain": "positive", "shape": [8]}, 1.0),
            put(
                F,
                "deter_initial_value",
                {"prior": {"family": "Dirichlet", "parameters": {"concentration": "h"}}},
            ),
        ),
        r"a learned vector is clamped and trained",
    ),
    (
        "an initial under a control node",
        "T",
        put(
            F,
            "u_location",
            {"initial": {"family": "Categorical", "parameters": {"logits": "D_location"}}},
        ),
        r"'u_location' is a control node, chosen rather than generated",
    ),
    (
        "two infogain terms giving one estimator's parameters",
        "RA",
        both(
            _clone("model_uncertainty_ensemble_0", "e1"),
            _clone("model_uncertainty_ensemble_0", "e2"),
            lambda d: d[OBJ]["terms"].extend(
                {
                    "term": "infogain",
                    "on": "stoch",
                    "parameters": {"ensemble": [e]},
                    "options": {"ensemble": _mu(d)["options"]["ensemble"]},
                }
                for e in ("e1", "e2")
            ),
        ),
        r"a second infogain term gives ensemble_disagreement's parameters",
    ),
    (
        "named function that does not import",
        "X3",
        both(
            put(F, "deter", "transition", "options", "map", {"impl": "numpy:no_such_function"}),
            put(F, "deter", "transition", "parameters", {}),
            _forget("deter_transition_map"),
        ),
        r"impl 'numpy:no_such_function' could not be imported",
    ),
    (
        "named function not module and name",
        "X3",
        both(
            put(F, "deter", "transition", "options", "map", {"impl": "numpy.tanh"}),
            put(F, "deter", "transition", "parameters", {}),
            _forget("deter_transition_map"),
        ),
        r"impl 'numpy.tanh' is not 'package.module:name'",
    ),
]


def _ids(cases):
    return [c[0] for c in cases]


@pytest.mark.parametrize("name, base, edit, sentence", CASES, ids=_ids(CASES))
def test_build_refuses(bases, name, base, edit, sentence):
    document, arrays = bases[base]
    document = copy.deepcopy(document)
    edit(document)
    with pytest.raises(DescriptionError, match=sentence):
        build(document, arrays=arrays, key=KEY)


# ================================================================ faults listed together


def _faults(document, arrays=None):
    with pytest.raises(DescriptionError) as info:
        build(document, arrays=arrays, key=KEY)
    return str(info.value)


def test_independent_clamp_values_failing_are_listed_in_one_refusal(bases):
    document, arrays = bases["T"]
    document = copy.deepcopy(document)
    document[CL]["B_context"] = 0.0
    document[CL]["A_cue"] = 0.0
    message = _faults(document, arrays)
    lines = message.splitlines()
    assert lines[0] == "the document has 2 faults:"
    assert lines[1].startswith("- clamps['B_context'] must be normalised log-probabilities")
    assert lines[2].startswith("- clamps['A_cue'] must be normalised log-probabilities")


def test_one_fault_is_its_own_sentence(bases):
    document, arrays = bases["T"]
    document = copy.deepcopy(document)
    document[CL]["A_cue"] = 0.0
    assert _faults(document, arrays).startswith("clamps['A_cue'] must be")


def test_each_phase_lists_its_own_records_and_an_earlier_phase_hides_a_later_one(bases):
    document, _ = bases["T"]
    doc = copy.deepcopy(document)
    doc[V]["location"]["levels"] = 0
    doc[V]["cue"]["domain"] = "nope"
    doc[F]["location"]["transition"]["family"] = "Nope"  # a factors fault, not reached
    message = _faults(doc)
    assert message.startswith("the document has 2 faults:")
    assert "variables['location'].levels" in message and "variables['cue']" in message
    assert "Nope" not in message
    doc = copy.deepcopy(document)
    doc[F]["location"]["transition"]["family"] = "Nope"
    doc[F]["cue"]["likelihood"]["family"] = "Nope"
    doc[F]["ghost"] = {}
    message = _faults(doc)
    assert message.startswith("the document has 3 faults:")
    assert message.count("unknown family 'Nope'") == 2 and "factors['ghost']" in message
    doc = copy.deepcopy(document)
    doc[CL]["ghost"] = 1.0
    doc[CL]["location"] = 1.0
    message = _faults(doc)
    assert message.startswith("the document has 2 faults:")
    assert "clamps['ghost']" in message and "clamps['location']" in message


# ======================================================================= the writer


def _tmaze():
    return tmaze_system()


def _with(params_or_system, **fields):
    return dataclasses.replace(params_or_system, **fields)


def _one_state(transition, initial, **system):
    """One latent ``x`` reading its own previous value, no control node, no observation."""
    s = H.System(B_deps=((0,),), policy_len=1, factor_names=("x",), **system)
    p = H.SystemParams(transition={"x": transition}, initial_state_prior={"x": initial})
    return s, p


def _table(n=2):
    return H.ConditionalCategorical(jnp.log(jnp.full((n, n), 1.0 / n)), 1)


def _rssm():
    agent = raif_mountain_car(KEY)
    return agent.system, agent.params


def _named_d():
    """A latent ``x`` beside a latent named ``D_x``: the letter name is taken."""
    s = H.System(B_deps=((0,), (1,)), policy_len=1, factor_names=("x", "D_x"))
    tables = {n: _table() for n in ("x", "D_x")}
    return s, H.SystemParams(
        transition=tables, initial_state_prior={n: H.Categorical(jnp.zeros(2)) for n in tables}
    )


def _tiny():
    from hybrid_aif.reference.raif.world import tiny_rssm

    return tiny_rssm(2, 1, key=KEY)


def _layer_norm_eps():
    s, p = _tiny()
    embed = p.recognition["embed"]
    norms = tuple(eqx.nn.LayerNorm(n.shape, eps=1e-5) for n in embed.norms)
    return s, _with(
        p, recognition={**p.recognition, "embed": eqx.tree_at(lambda m: m.norms, embed, norms)}
    )


def _deter(map_):
    s, p = _tiny()
    return s, _with(p, transition={**p.transition, "deter": map_})


def _gaussian_target():
    from tests.test_description import _lg_model

    sensor = H.LinearGaussianObservation(
        H=jnp.asarray([[1.0]]), c=jnp.zeros(1), noise=jnp.asarray([[0.5]])
    )
    s, p = _lg_model(sensor)
    target = H.ConditionalGaussian(jnp.zeros((2, 1)), jnp.ones((2, 1, 1)), 1)
    return s, _with(p, state_targets={"x": target})


def _batched_likelihood():
    s, p = _tmaze()
    cue = p.likelihood["cue"]
    batched = H.ConditionalCategorical(cue.log_table, cue.parent_ndims - 1)
    return s, _with(p, likelihood={**p.likelihood, "cue": batched})


def _scoring(rows, scoring, owned=()):
    s, p = _rssm()
    return _with(s, terms=rows, scoring=owned), _with(p, scoring=scoring)


def _raif_entry(row):
    return _rssm()[1].scoring[row]


WRITER_CASES = [
    (
        "a placement mode unknown",
        lambda: (*_tmaze(), {"arrays": "sidecar"}),
        ValueError,
        r"arrays= must be 'auto', 'inline' or 'file', got 'sidecar'",
    ),
    (
        "a System the validator refuses",
        lambda: (_with(_tmaze()[0], epistemic=("likelihood:zz",)), _tmaze()[1], {}),
        DescriptionError,
        r"^the System is invalid, so it has no document: ",
    ),
    (
        "a default name already taken",
        lambda: (*_named_d(), {}),
        DescriptionError,
        r"the default name 'D_x' is already a variable of this document",
    ),
    (
        "a layer norm off the families' recipe",
        lambda: (*_layer_norm_eps(), {}),
        DescriptionError,
        r"recognition:embed/norms/0: a LayerNorm with eps=1e-05 and weight=True",
    ),
    (
        "a conditional target with no parents declared",
        lambda: (*_gaussian_target(), {}),
        ValueError,
        r"state_targets\['x'\] is a ConditionalGaussian.*no C_deps entry names them",
    ),
    (
        "a table with batch axes",
        lambda: (*_batched_likelihood(), {}),
        DescriptionError,
        r"likelihood:cue: its parent_ndims does not follow from its arrays' shapes",
    ),
    (
        "a deterministic map of another module",
        lambda: (*_deter(H.Deterministic(map=eqx.nn.Linear(3, 3, key=KEY))), {}),
        DescriptionError,
        r"transition:deter/map: a Deterministic map of type Linear has no document",
    ),
    (
        "a frozen named function",
        lambda: (*_deter(H.Deterministic(map=np.tanh, frozen=True)), {}),
        DescriptionError,
        r"transition:deter: a frozen Deterministic over a named function",
    ),
    (
        "a bracket in a name",
        lambda: (
            H.System(B_deps=((0,),), policy_len=1, factor_names=("x[0]",)),
            H.SystemParams(
                transition={"x[0]": _table()},
                initial_state_prior={"x[0]": H.Categorical(jnp.zeros(2))},
            ),
            {},
        ),
        DescriptionError,
        r"a state is named 'x\[0\]', and a variable name may not contain a bracket",
    ),
    (
        "a policy prior with no control node",
        lambda: (
            _one_state(_table(), H.Categorical(jnp.zeros(2)))[0],
            _with(
                _one_state(_table(), H.Categorical(jnp.zeros(2)))[1],
                policy_prior=H.Categorical(jnp.zeros(1)),
            ),
            {},
        ),
        DescriptionError,
        r"policy_prior is declared on a System with no control node",
    ),
    (
        "estimator parameters with no infogain term",
        lambda: (*_scoring((), {"ensemble_disagreement": _raif_entry("model_uncertainty")}), {}),
        DescriptionError,
        r"scoring\['ensemble_disagreement'\] holds parameters no term reads",
    ),
    (
        "a scoring entry no term mounts",
        lambda: (
            _with(_rssm()[0], terms=tuple(t for t in _rssm()[0].terms if t != "g_tail")),
            _rssm()[1],
            {},
        ),
        DescriptionError,
        r"SystemParams.scoring holds \['g_tail'\], which no term",
    ),
    (
        "a term parameter off the term families",
        lambda: (*_scoring(("g_tail",), {"g_tail": H.Categorical(jnp.zeros(2))}), {}),
        DescriptionError,
        r"a term parameter of type Categorical has no document form",
    ),
    (
        "an ensemble that is not a tuple",
        lambda: (
            *_scoring(
                ("model_uncertainty:stoch",),
                {"model_uncertainty": _raif_entry("model_uncertainty")[0]},
            ),
            {},
        ),
        DescriptionError,
        r"'model_uncertainty' holds its 'ensemble' as a tuple, got MLPConditional",
    ),
    (
        "more items than parameter names",
        lambda: (
            *_scoring(("g_tail",), {"g_tail": (*_raif_entry("g_tail"), _raif_entry("g_tail")[0])}),
            {},
        ),
        DescriptionError,
        r"'g_tail' holds 3 items but names \['critic', 'slow'\]",
    ),
]


@pytest.mark.parametrize("name, factory, error, sentence", WRITER_CASES, ids=_ids(WRITER_CASES))
def test_describe_refuses(name, factory, error, sentence):
    system, params, kwargs = factory()
    with pytest.raises(error, match=sentence):
        describe(system, params, **kwargs)


def test_describe_refuses_a_parametric_row_that_declares_no_parameter_names():
    """A row registered without ``parameters`` cannot have its parameters written."""
    from hybrid_aif.objectives import registry

    row = dataclasses.replace(registry.ROWS["g_tail"], name="g_tail_unnamed", parameters=())
    registry.ROWS[row.name] = row
    try:
        system, params = _scoring((row.name,), {row.name: _raif_entry("g_tail")})
        with pytest.raises(
            DescriptionError, match=r"the 'g_tail_unnamed' estimator declares no parameter names"
        ):
            describe(system, params)
    finally:
        del registry.ROWS[row.name]
