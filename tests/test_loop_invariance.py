"""A tick loop compiles once: the faces, wrapped once in ``eqx.filter_jit`` and run for twenty
ticks starting from ``initial_priors`` with only values changing, trace once each, and the
carried structures (priors, parameters) come back as they went in.

Design: docs/design/recompilation.md R1, R2, R5.
"""

import collections
import dataclasses
import importlib.util
import pathlib

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from hybrid_aif import Categorical, ConditionalDirichlet, MixtureGaussian
from hybrid_aif.objectives.hybrid import pad_mixture
from hybrid_aif.reference.raif.world import tiny_rssm
from tests.test_action_node_modalities import payoff_model_by_node, scalar_model
from tests.test_closed_loop import small_lg
from tests.test_conditional_targets import C_ASK
from tests.test_system import tmaze_system

TICKS = 20
_EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"


def _example(name: str):
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def signature(tree):
    """The pytree structure and each leaf's ``(shape, dtype, weak_type)``."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return treedef, tuple(
        (jnp.shape(x), str(jnp.result_type(x)), bool(getattr(x, "weak_type", False)))
        for x in leaves
    )


class Counter:
    """``eqx.filter_jit`` wrappers that count how often each wrapped function is traced."""

    def __init__(self):
        self.counts = collections.Counter()

    def __call__(self, name, fn):
        def inner(*args, **kwargs):
            self.counts[name] += 1
            return fn(*args, **kwargs)

        return eqx.filter_jit(inner)


def run_loop(
    system, params, *, clamp=None, learnable=False, act=True, plan_key=False, pad_mixtures=False
):
    """Twenty ticks of ``infer_states`` → ``act`` → ``update_empirical_prior`` (and ``learn``
    from the second tick when ``learnable``), from ``initial_priors``. Returns the trace counts,
    the first and second ticks' priors, and the parameters in and out of the first ``learn``."""
    count = Counter()
    infer = count("infer_states", lambda s, p, o, pr: s.infer_states(p, o, pr)[0])
    push = count("update_empirical_prior", lambda s, p, b, a: s.update_empirical_prior(p, b, a))
    choose = count(
        "act", lambda s, k, p, b, pk: s.act(k, p, b, plan_key=pk, selection="stochastic")[0]
    )
    learn = count("learn", lambda s, p, o, b, pb, a: s.learn(p, o, b, prev_beliefs=pb, action=a))
    key = jax.random.PRNGKey(3)
    key, k0 = jax.random.split(key)
    if act:
        env, env_params = system.as_env(params)
        step = count("env.step", lambda e, k, st, a, ep: e.step(k, st, a, ep))
        obs, state = env.reset(k0, env_params)
    priors = system.initial_priors(params)
    if clamp:
        priors = {
            **priors,
            **{n: Categorical.from_probs(jax.nn.one_hot(v, k)) for n, (v, k) in clamp.items()},
        }
    if pad_mixtures:
        priors = {
            n: pad_mixture(q, system.mixture_k) if isinstance(q, MixtureGaussian) else q
            for n, q in priors.items()
        }
    first = priors
    pushed = None
    learned = None
    prev_beliefs = prev_action = None
    for _ in range(TICKS):
        if not act:
            key, ko = jax.random.split(key)
            obs = {
                m: jax.random.normal(ko, (2,)) if m not in ("reward", "cont") else jnp.zeros(())
                for m in system.names.modalities
            }
            action = {n: jnp.zeros((1,)) for n in system.names.actions}
        beliefs = infer(system, params, obs, priors)
        if act:
            ka, ke, key = jax.random.split(key, 3)
            action = choose(system, ka, params, beliefs, ka if plan_key else None)
        if learnable and prev_beliefs is not None:
            before = params
            params = learn(system, params, obs, beliefs, prev_beliefs, prev_action)
            learned = learned or (before, params)
        priors = push(system, params, beliefs, action)
        pushed = pushed or priors
        prev_beliefs, prev_action = beliefs, action
        if act:
            obs, state = step(env, ke, state, action, env_params)
    return count.counts, first, pushed, learned


def _tmaze():
    return tmaze_system(policy_len=2), {}


def _learnable_tmaze():
    system, params = tmaze_system(policy_len=2)
    cue = params.likelihood["cue"]
    concentration = jnp.exp(cue.log_table) * 4.0 + 0.5
    likelihood = {
        **params.likelihood,
        "cue": ConditionalDirichlet(concentration=concentration, parent_ndims=cue.parent_ndims),
    }
    return (system, dataclasses.replace(params, likelihood=likelihood)), {"learnable": True}


def _scalar_lg():
    return scalar_model(), {"clamp": {"a": (0, 3)}}


def _payoff_mixed():
    system, params = payoff_model_by_node(C_ASK, 3)
    return (dataclasses.replace(system, planner="mixed"), params), {"clamp": {"option": (2, 3)}}


def _small_lg_gradient():
    return small_lg(), {}


def _sophisticated_tmaze():
    system, params = tmaze_system(policy_len=2)
    system = dataclasses.replace(system, planner="sophisticated", branch_on=("cue",))
    return (system, params), {}


def _switching():
    return _example("switching_filter").make_agent(epistemic=True), {"pad_mixtures": True}


def _tiny_rssm():
    return tiny_rssm(2, 1, key=jax.random.PRNGKey(0), policy_len=2), {"act": False}


MODELS = {
    "tmaze": _tmaze,
    "learnable_tmaze": _learnable_tmaze,
    "scalar_lg": _scalar_lg,
    "payoff_mixed": _payoff_mixed,
    "small_lg_gradient": _small_lg_gradient,
    "sophisticated_tmaze": _sophisticated_tmaze,
    "switching_mixture": _switching,
    "tiny_rssm": _tiny_rssm,
}


@pytest.fixture(scope="module", params=sorted(MODELS))
def loop(request):
    (system, params), options = MODELS[request.param]()
    return run_loop(system, params, **options)


def test_each_face_traces_once_over_the_loop(loop):
    counts = loop[0]
    assert counts, "the loop ran no face"
    assert set(counts.values()) == {1}, dict(counts)


def test_initial_priors_have_the_structure_of_the_pushed_priors(loop):
    _, first, pushed, _ = loop
    assert signature(first) == signature(pushed)


def test_learn_returns_the_parameters_structure_it_was_given():
    (system, params), options = _learnable_tmaze()
    before, after = run_loop(system, params, **options)[3]
    assert signature(before) == signature(after)


def test_weak_typed_parameters_enter_the_faces_strong():
    """A parameter leaf built from a Python scalar is weak-typed; ``SystemParams`` holds it
    strong, as ``learn`` returns it, so the first call does not retrace."""
    system, params = tmaze_system(policy_len=2)
    weak = jnp.full((4, 3), 2.0)
    assert weak.weak_type
    held = dataclasses.replace(
        params, likelihood={**params.likelihood, "cue": ConditionalDirichlet(weak, 1)}
    )
    leaf = held.likelihood["cue"].concentration
    assert not leaf.weak_type
    assert leaf.dtype == weak.dtype
    assert jnp.array_equal(leaf, weak)


# ------------------------------------------------------- schedules and observation forms


def test_replay_compiles_once_over_a_schedule_of_python_float_rates():
    from hybrid_aif.replay import _replay_core, replay
    from tests.test_learn import _sense_system

    system, params = _sense_system()
    observations = {m: jnp.zeros((6,), jnp.int32) for m in system.names.modalities}
    actions = {n: jnp.zeros((6,), jnp.int32) for n in system.names.actions}
    before = _replay_core._cached._cache_size()
    for lr in (1.0, 0.5, 0.25):
        replay(system, params, observations, actions, learn=True, lr=lr)
    assert _replay_core._cached._cache_size() - before == 1


def test_replay_keeps_a_python_zero_rate_static():
    from hybrid_aif.replay import _traced_rate

    assert _traced_rate(0.0) == 0.0 and isinstance(_traced_rate(0.0), float)
    assert isinstance(_traced_rate(0.5), jax.Array) and _traced_rate(0.5).dtype == jnp.float32
    array = jnp.float32(0.5)
    assert _traced_rate(array) is array


def test_the_mountain_car_episode_compiles_once_over_a_decaying_rate(monkeypatch):
    mountain_car = _example("mountain_car")
    made = []
    real = mountain_car.make_episode

    def spy(*args, **kwargs):
        made.append(real(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(mountain_car, "make_episode", spy)
    tiny = dataclasses.replace(mountain_car.MountainCar(), time_limit=6)
    monkeypatch.setattr(mountain_car, "MountainCar", lambda: tiny)
    argv = "x --episodes 3 --horizon 3 --width 8 --depth 1 --candidates 4 --iters 1"
    monkeypatch.setattr("sys.argv", argv.split())
    mountain_car.main()
    assert made[0]._cached._cache_size() == 1


def _lg_tick(observation):
    system, params = scalar_model()
    return system.infer_states(params, observation, system.initial_priors(params))


@pytest.mark.parametrize(
    ("observation", "seen"),
    [
        ([0.1], "Python list"),
        (0.1, "Python float"),
        (jnp.asarray(0.1), r"shape \(\)"),
        (jnp.asarray([[0.1], [0.2]]), r"shape \(2, 1\)"),
    ],
)
def test_a_linear_gaussian_observation_of_the_wrong_form_is_refused_with_a_sentence(
    observation, seen
):
    with pytest.raises(ValueError, match=rf"observation 'y'.*shape \(1,\).*{seen}"):
        _lg_tick({"y": observation})


def test_a_categorical_observation_given_as_a_list_is_refused_with_a_sentence():
    system, params = tmaze_system(policy_len=2)
    env, env_params = system.as_env(params)
    obs, _ = env.reset(jax.random.PRNGKey(0), env_params)
    with pytest.raises(ValueError, match="observation 'cue' is a Python list"):
        system.infer_states(params, {**obs, "cue": [0]})
    system.infer_states(params, obs)  # the array form still runs


def test_a_mixture_prior_below_the_budget_costs_one_extra_trace_unless_padded():
    """``initial_priors`` returns the declared mixture, whose component count is below
    ``mixture_k`` while every later tick's priors carry ``mixture_k``: the first tick traces the
    smaller structure. Padding the mixture to ``mixture_k`` first (as the other models'
    loops do) makes the loop one trace."""
    (system, params), _ = _switching()
    counts, first, pushed, _ = run_loop(system, params)
    assert counts["infer_states"] == 2
    assert signature(first)[0] == signature(pushed)[0]
    assert signature(first) != signature(pushed)
    padded, *_ = run_loop(system, params, pad_mixtures=True)
    assert padded["infer_states"] == 1
