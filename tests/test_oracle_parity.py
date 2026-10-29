"""Parity against the oracle's **resolved** configuration, field by field.

Design: ``docs/design/amortization.md``'s "A12 implementation note: chunk 2d",
whose §(e) records the audit this file is the executable half of. The ground truth
is one file — the oracle run's own **resolved** config, with every default already
substituted and dumped by their own driver — and the reason it is the ground truth
rather than ``configs.yaml`` is the reason this file exists: ``configs.yaml`` is an
inheritance chain whose leaves a reader resolves by hand, and five of chunk 2d's
divergences were hand-resolutions that went wrong (a critic and an ensemble built
~8× too large each, an ``eps`` left at optax's default, a missing clip, and four
environments that were only ever one).

It is **vendored into this repo** at
``reference/raif/fixtures/oracle_state_mc.config.yaml`` with a provenance sidecar, so a
claim about "their value" is checkable against a versioned file rather than against
an external checkout that exists on one machine.

So the numbers below are **transcribed from that file** and asserted against ours.
A test that reimplements our own expression can only confirm that we agree with
ourselves — that was divergence 1's whole failure mode — so nothing here computes
anything from our constants: each assertion has a literal on the right-hand side and
the config line it came from in a comment. The last section then closes the one gap
that leaves (a literal typed wrong is a literal no test catches) by *parsing* the
vendored dump and comparing it against the same constants.

Three sections:

``the resolved config``   sizes, outscales, the four optimizer groups, the protocol
``the ground truth``      the vendored dump: its own values, and §(e)'s coverage
"""

import json
import pathlib
import re

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from hybrid_aif.distributions import MLPConditional
from hybrid_aif.distributions.neural import (
    OPTIMIZER,
    PredictorHead,
    adam,
    apply_gradient,
    init_opt_state,
)
from hybrid_aif.reference.raif import agent as raif_agent
from hybrid_aif.reference.raif import run as run_mc
from hybrid_aif.reference.raif.agent import (
    ACTOR_LAYERS,
    ACTOR_UNITS,
    CRITIC_LAYERS,
    CRITIC_UNITS,
    ENSEMBLE_LAYERS,
    ENSEMBLE_UNITS,
    raif_mountain_car,
)
from hybrid_aif.reference.raif.world import tiny_rssm

#  `hybrid_aif.train` re-exports the `imagine` FUNCTION, so the module is imported by
#  its names (that package's own import note). Each name is aliased to the config
#  block it is being checked against, which is the only reason a test file wants six
#  aliases: the left-hand side should read like the config's key.
from hybrid_aif.train.ensemble import DEFAULT_CLIP as IG_OPT_CLIP
from hybrid_aif.train.ensemble import DEFAULT_EPS as IG_OPT_EPS
from hybrid_aif.train.ensemble import DEFAULT_LR as IG_OPT_LR
from hybrid_aif.train.ensemble import fit_ensemble
from hybrid_aif.train.imagine import DEFAULT_CLIP as ACTOR_OPT_CLIP
from hybrid_aif.train.imagine import (
    DEFAULT_DISCOUNT,
    DEFAULT_LAMBDA,
    DEFAULT_RETNORM,
    DEFAULT_SLOW_FRACTION,
    DEFAULT_SLOW_PERIOD,
    DEFAULT_SLOWREG,
    imagine,
)
from hybrid_aif.train.imagine import DEFAULT_EPS as ACTOR_OPT_EPS
from hybrid_aif.train.imagine import DEFAULT_LR as ACTOR_OPT_LR
from hybrid_aif.train.world import DEFAULT_CLIP as MODEL_OPT_CLIP
from hybrid_aif.train.world import DEFAULT_EPS as MODEL_OPT_EPS
from hybrid_aif.train.world import DEFAULT_LR as MODEL_OPT_LR
from hybrid_aif.train.world import assemble

KEY = jax.random.PRNGKey(0)


@pytest.fixture(scope="module")
def agent():
    """The parity agent at its declared sizes — no ``head_units`` override."""
    return raif_mountain_car(KEY)


def _widths(layers) -> list[tuple[int, int]]:
    return [tuple(le.weight.shape) for le in layers]


# ======================================================== the resolved config: sizes


def test_the_three_head_sizes_are_three_different_numbers(agent):
    """``actor 3×256``, ``critic 2×32``, ``ig_head 2×32`` — one block each, not one number.

    Divergences 6 and 7 (closed 2026-09-02). Addendum §2 read the ``actor`` block's
    ``layers 3, units 256`` and generalized it to the critic and the ensemble, which
    the resolved config contradicts::

        actor:   {layers: 3, units: 256}
        critic:  {layers: 2, units:  32}
        ig_head: {layers: 2, units:  32}

    Asserted on the *built modules* rather than on the constants, because the bug was
    in the wiring (one pair of arguments reaching three constructors) and not in a
    number: a test on ``ACTOR_UNITS`` alone would still have passed while the critic
    was 256 wide.
    """
    critic, slow = agent.params.scoring["g_tail"]
    #  `layers: N` counts N hidden layers, so a stack of N+1 Linears
    assert _widths(agent.params.policy_prior.layers) == [
        (256, 1152),
        (256, 256),
        (256, 256),
        (2, 256),  # a Gaussian over a 1-d action: (mean, scale)
    ]
    for head in (critic, slow):  # the slow copy is the critic's twin, by construction
        assert _widths(head.trunk.layers) == [(32, 1152), (32, 32), (255, 32)]
    members = agent.params.scoring["model_uncertainty"]
    assert len(members) == 4  # ig_models: 4
    for member in members:
        assert _widths(member.layers) == [(32, 1153), (32, 32), (2048, 32)]


def test_the_head_units_override_still_moves_all_three_together(agent):
    """``head_units``/``head_layers`` are the one knob a smoke wants, and it is not parity."""
    small = raif_mountain_car(KEY, head_units=8, head_layers=1)
    critic, _ = small.params.scoring["g_tail"]
    assert _widths(small.params.policy_prior.layers) == [(8, 1152), (2, 8)]
    assert _widths(critic.trunk.layers) == [(8, 1152), (255, 8)]
    assert _widths(small.params.scoring["model_uncertainty"][0].layers) == [(8, 1153), (2048, 8)]
    #  and the parity build is NOT that: the override defaults to None, not to a size
    assert _widths(agent.params.policy_prior.layers)[0] == (256, 1152)


def test_the_critic_and_reward_heads_are_zero_initialized_and_the_others_are_not(agent):
    """``outscale: 0.0`` on ``critic`` and ``reward_head``, ``1.0`` everywhere else.

    Their zero-init (DreamerV3's) starts a head at the constant-zero prediction so
    early training moves it rather than un-learning noise. ``outscale: 1.0`` is
    exactly "leave the final layer at its random init", which is why the actor and
    the ensemble members — ``MLPConditional``, which has no ``out_scale`` argument at
    all — already match: 1.0 is the *absence* of the knob. FLAGGED in the 2d note
    rather than papered over, because a config that ever asked for ``0.0`` on the
    actor could not be expressed without adding one.
    """
    critic, slow = agent.params.scoring["g_tail"]
    core, _ = assemble(agent.system, agent.params)
    for head, name in ((critic, "critic"), (slow, "slow"), (core.reward, "reward_head")):
        assert float(jnp.max(jnp.abs(head.trunk.layers[-1].weight))) == 0.0, name
        assert float(jnp.max(jnp.abs(head.trunk.layers[-1].bias))) == 0.0, name
    #  cont_head outscale 1.0, decoder outscale 1.0 — random init, so NOT zero
    for head, name in ((core.cont, "cont_head"), (core.decoder, "decoder")):
        assert float(jnp.max(jnp.abs(head.trunk.layers[-1].weight))) > 0.0, name
    #  actor outscale 1.0, ig_head outscale 1.0
    assert float(jnp.max(jnp.abs(agent.params.policy_prior.layers[-1].weight))) > 0.0
    for member in agent.params.scoring["model_uncertainty"]:
        assert float(jnp.max(jnp.abs(member.layers[-1].weight))) > 0.0


def test_the_zero_outscale_head_predicts_zero_before_training():
    """What ``outscale: 0.0`` is *for*, not just what it looks like."""
    head = PredictorHead.from_shapes(
        (4,), loss="symlog_disc", key=KEY, width=32, depth=2, out_scale=0.0
    )
    #  a uniform-logit symlog_disc head decodes to the mean bin centre, i.e. 0 up to
    #  the float32 error of summing 255 symmetric bins
    assert float(head.predict((jnp.ones((4,)),))) == pytest.approx(0.0, abs=1e-5)


def test_the_world_model_sizes_are_the_tiny_blocks(agent):
    """``rssm`` / ``encoder`` / ``decoder`` / ``reward_head`` / ``cont_head``, resolved.

    ``rssm: {classes: 32, deter: 128, stoch: 32, units: 32}``,
    ``encoder``/``decoder`` ``{mlp_layers: 2, mlp_units: 64}``,
    ``reward_head``/``cont_head`` ``{layers: 2, units: 32}``, reward
    ``{dist: symlog_disc, bins: 255}``.
    """
    core, _ = assemble(agent.system, agent.params)
    assert core.deter_dim == 128  # rssm.deter
    assert (core.num_vars, core.num_classes) == (32, 32)  # rssm.stoch, rssm.classes
    assert core.num_bins == 255  # reward_head.bins
    assert _widths(core.reward.trunk.layers) == [(32, 1152), (32, 32), (255, 32)]
    assert _widths(core.cont.trunk.layers) == [(32, 1152), (32, 32), (1, 32)]
    assert _widths(core.decoder.trunk.layers) == [(64, 1152), (64, 64), (2, 64)]
    #  encoder: mlp_layers 2 normed+activated layers of 64, the last BEING the embed
    assert core.embed_dim == 64


# =================================================== the resolved config: optimizers


def _measured_eps(optimizer: optax.GradientTransformation, probe: float = 1e-6) -> float:
    """The ``eps`` an optax adam is *actually running*, recovered from one step.

    Adam's first step on a scalar parameter with gradient ``g`` is
    ``-g / (|g| + eps)`` (the bias corrections cancel exactly at ``count = 1``), so
    one step at unit rate identifies ``eps = g·(1 − |u|) / |u|``. Measured rather than
    read off a constructor argument because "which number is in the config" and
    "which number the optimizer uses" are the two things that came apart, and only
    the second one trains an agent. ``probe`` is chosen between the two candidate
    values (1e-8, 1e-5) so that ``|u|`` is far from both 0 and 1 in float32.
    """
    param = jnp.asarray(0.0)
    state = optimizer.init(param)
    updates, _ = optimizer.update(jnp.asarray(probe), state, param)
    u = abs(float(updates))
    return probe * (1.0 - u) / u


def test_the_eps_probe_recovers_a_known_eps():
    """The measurement above, checked against optax's own argument. Calibration first."""
    for eps in (1e-8, 1e-6, 1e-5, 1e-4):
        assert _measured_eps(optax.adam(1.0, eps=eps)) == pytest.approx(eps, rel=2e-3), eps


def test_the_four_optimizer_groups_carry_their_resolved_eps_and_clip():
    """The oracle's four optimizer lines, transcribed::

        model_opt:  {clip: 1000.0, eps: 1e-08, lr: 0.0001, opt: adam, wd: 0.0}
        actor_opt:  {clip:  100.0, eps: 1e-05, lr: 3e-05,  opt: adam, wd: 0.0}
        critic_opt: {clip:  100.0, eps: 1e-05, lr: 3e-05,  opt: adam, wd: 0.0}
        ig_opt:     {clip:  100.0, eps: 1e-05, lr: 0.0001, opt: adam, wd: 0.0}

    Divergence 4 (closed 2026-09-02): ``eps`` is a **group** setting and the groups
    disagree — 1e-8 on the world model, 1e-5 on all three amortized heads — and we
    ran optax's 1e-8 everywhere, with no clip at all on the ensemble.

    ``model_opt``'s line is asserted here **because it already matched**. It is the
    one group that needs no per-slot optimizer (its ``eps`` is optax's default, so
    ``OPTIMIZER`` is already right), which makes it exactly the kind of agreement
    that a later "let's tidy the shared optimizer" would break silently.
    """
    #  the amortized groups: theirs, and NOT optax's default
    assert (ACTOR_OPT_EPS, ACTOR_OPT_CLIP, ACTOR_OPT_LR) == (1e-5, 100.0, 3e-5)
    assert (IG_OPT_EPS, IG_OPT_CLIP, IG_OPT_LR) == (1e-5, 100.0, 1e-4)
    #  the world-model group: theirs, and it happens to BE optax's default
    assert (MODEL_OPT_EPS, MODEL_OPT_CLIP, MODEL_OPT_LR) == (1e-8, 1000.0, 1e-4)
    #  and the shared engine optimizer, which `absorb` (hence fit_world) runs, is
    #  measured to be at that value — the assertion that stops a drift in either
    #  direction: a changed OPTIMIZER, or a changed model_opt reading
    assert _measured_eps(OPTIMIZER) == pytest.approx(MODEL_OPT_EPS, rel=2e-3)
    assert _measured_eps(adam()) == pytest.approx(1e-8, rel=2e-3)
    assert _measured_eps(adam(ACTOR_OPT_EPS)) == pytest.approx(1e-5, rel=2e-3)


def test_the_reference_recipe_states_its_own_eps_and_clip():
    """The parity recipe says what it runs, rather than inheriting a default.

    The resolved config's rule, applied to
    the optimizer line: a number the reader can check against the config beats a
    default the reader has to go and look up.
    """
    assert raif_agent.ACTOR_EPS == 1e-5  # actor_opt.eps == critic_opt.eps
    assert raif_agent.ENSEMBLE_EPS == 1e-5  # ig_opt.eps
    assert raif_agent.ENSEMBLE_CLIP == 100.0  # ig_opt.clip


def test_a_slot_optimizers_eps_actually_reaches_the_update():
    """``adam_eps`` is wired to the step, not merely accepted and dropped.

    The failure this catches is the one a constants test cannot: a keyword that is
    threaded as far as a docstring. Two identical steps at two ``eps`` values must
    land in different places.
    """
    module = MLPConditional.from_shapes((3,), 2, key=KEY, width=4, depth=1)
    grads = jax.tree_util.tree_map(
        lambda x: jnp.full_like(x, 1e-6), eqx.filter(module, eqx.is_inexact_array)
    )
    out = {}
    for eps in (1e-8, 1e-5):
        optimizer = adam(eps)
        stepped, _ = apply_gradient(
            module, grads, init_opt_state(module, optimizer), lr=1.0, optimizer=optimizer
        )
        out[eps] = np.asarray(stepped.layers[0].weight)
    assert not np.allclose(out[1e-8], out[1e-5])
    #  and the direction is the one adam's algebra predicts: a larger eps damps
    delta = {eps: np.abs(w - np.asarray(module.layers[0].weight)) for eps, w in out.items()}
    assert delta[1e-5].max() < delta[1e-8].max()


def test_the_ensemble_trainer_clips_by_default_now():
    """``ig_opt``'s ``clip: 100.0``: the default, not an opt-in (divergence 5, closed).

    The claim this replaces was a *false* one in ``train/ensemble.py``'s docstring
    ("their ``ig_opt`` declares none"), which is why the number is pinned rather
    than left to the recipe: a default that reads as deliberate is what stopped the
    original reader from checking.
    """
    import inspect

    signature = inspect.signature(fit_ensemble)
    assert signature.parameters["clip"].default == 100.0
    assert signature.parameters["adam_eps"].default == 1e-5
    assert inspect.signature(imagine).parameters["adam_eps"].default == 1e-5
    assert inspect.signature(imagine).parameters["clip"].default == 100.0


# ================================================= the resolved config: the protocol


def test_the_training_protocol_numbers_are_theirs():
    """``batch_size 16``, ``batch_length 64``, ``imag_horizon 8``, ``train_ratio 64``.

    ``train_ratio 64`` over ``batch_steps`` (``batch_size × batch_length`` = 1024) is
    one gradient step per 16 agent steps, which is
    :data:`~hybrid_aif.reference.raif.agent.TRAIN_EVERY`.
    """
    assert raif_agent.BATCH_SIZE == 16  # batch_size
    assert raif_agent.BATCH_LENGTH == 64  # batch_length
    assert raif_agent.IMAG_HORIZON == 8  # imag_horizon
    assert raif_agent.TRAIN_EVERY == 16  # train_ratio 64 / batch_steps 1024
    assert raif_agent.ENSEMBLE_SIZE == 4  # ig_models
    assert raif_agent.BC_SCALE == 0.5  # loss_scales.actor_refresh
    assert raif_agent.ACTENT == 3e-4  # actent
    assert raif_agent.WMENT == 3e-4  # wment
    assert DEFAULT_LAMBDA == 0.95  # return_lambda
    assert DEFAULT_DISCOUNT == 1.0 - 1.0 / 333.0  # horizon: 333
    assert DEFAULT_SLOW_FRACTION == 0.02  # slow_critic_fraction
    assert DEFAULT_SLOWREG == 1.0  # loss_scales.slowreg
    assert DEFAULT_RETNORM == (5.0, 95.0, 0.99, 1.0)  # retnorm perclo/perchi/decay/max


def test_the_harness_runs_their_env_count():
    """``envs: {amount: 4}`` — divergence 8, closed 2026-09-02.

    Four environment lanes per seed, and their ``Ratio(64/1024)`` counting the pool:
    ``train_every // lanes`` steps in each lane per gradient step. The arithmetic that
    must not move is the second assertion — one gradient step per 16 **pooled** agent
    steps, whatever the lane count.
    """
    assert run_mc.LANES == 4  # envs.amount
    assert run_mc.Config(out="x").lanes == 4
    assert run_mc.CHUNK_STEPS == 1024  # batch_steps, and 64 gradient steps
    assert run_mc.CHUNK_STEPS // run_mc.TRAIN_EVERY == 64
    assert run_mc.Config(out="x").time_limit == 100  # wrapper.length
    #  replay_size 5e5 STEPS per buffer, verbatim: StreamBuffer counts steps and their
    #  stride-1 item capacity retains the same number of distinct steps (FLAG 7, closed)
    assert run_mc.CAPACITY == 500_000
    #  and one update sees batch_steps REAL steps — no padding in a spliced stream
    assert raif_agent.BATCH_SIZE * raif_agent.BATCH_LENGTH == 1024


def test_the_env_is_their_wrapped_mountain_car():
    """``wrapper: {length: 100, repeat: 2, r_reward_offset: 1.1}`` (addendum §1)."""
    from hybrid_aif.reference.envs.mountain_car import MountainCar
    from hybrid_aif.reference.raif.metrics import MC_OFFSET

    env = MountainCar()
    assert env.time_limit == 100  # wrapper.length
    assert env.action_repeat == 2  # wrapper.repeat
    assert MC_OFFSET == 1.1  # wrapper.r_reward_offset


def test_the_declared_system_is_their_tiny_rssm():
    """``rssm: {initial: learned, unimix: 0.01}``, on both categorical heads.

    ``unimix`` is declared on the *nets* rather than applied at a call site, so it is
    read off the prior and the posterior, both — theirs mixes in the uniform inside
    ``Dist``, which every consumer of those logits then sees.
    """
    system, params = tiny_rssm(2, 1, key=KEY)
    core, _ = assemble(system, params)
    assert core.prior.unimix == 0.01  # rssm.unimix
    assert core.posterior.unimix == 0.01  # rssm.unimix
    assert not core.initial.frozen  # rssm.initial: learned


# ================================ the ground truth itself, and the audit's coverage


def _resolved_config():
    """The vendored resolved config, parsed. ``pytest.importorskip`` on the parser.

    The config the oracle *ran* is vendored at
    ``reference/raif/fixtures/oracle_state_mc.config.yaml`` (with a provenance sidecar)
    so that every claim in this file and in ``amortization.md``'s §(e) can be
    checked against a file **in this repo** rather than against an external
    checkout that only exists on one machine. ``yaml`` is not a runtime dependency
    of the library, so the two tests that read it skip rather than fail where it is
    absent; the assertions above stand on their own transcribed literals either way.
    """
    yaml = pytest.importorskip("yaml", reason="pyyaml parses the vendored oracle config")
    import hybrid_aif.reference.raif.fixtures as fx

    path = pathlib.Path(fx.__file__).parent / "oracle_state_mc.config.yaml"
    return yaml.safe_load(path.read_text())


def _num(value) -> float:
    """A config scalar as a float — because ``1e-05`` in their dump is a **string**.

    PyYAML implements YAML **1.1**, whose float resolver requires a ``.`` or a signed
    exponent, so their driver's ``eps: 1e-05`` / ``lr: 3e-05`` / ``eps: 1e-08`` all
    parse as `str` while ``clip: 100.0`` parses as `float`. Worth a named helper
    rather than a sprinkling of `float()`: a test that compared ``'1e-05' == 1e-05``
    would fail confusingly, and — the sharper point — an *audit* that eyeballed the
    same lines would not notice that the three most consequential numbers in the file
    are the ones a naive parse gets wrong.
    """
    return float(value)


def test_the_transcribed_literals_are_the_vendored_configs_own_values():
    """Every number this file asserts, re-read **from the config** rather than typed.

    The point of the whole 2026-09-02 audit was that a hand-resolved default is a
    guess, and the tests above are hand-transcribed literals — which is the right
    form for a test (a reader can check them against the config by eye) but leaves
    exactly one gap: a literal typed wrong is a literal no test catches. So this
    closes the loop by parsing the vendored dump and comparing it against the same
    constants, and it is the assertion that would have failed in August.
    """
    cfg = _resolved_config()
    assert cfg["critic"]["layers"] == CRITIC_LAYERS
    assert cfg["critic"]["units"] == CRITIC_UNITS
    assert cfg["critic"]["outscale"] == 0.0
    assert cfg["critic"]["bins"] == 255
    assert cfg["critic"]["dist"] == "symlog_disc"
    assert cfg["ig_head"]["layers"] == ENSEMBLE_LAYERS
    assert cfg["ig_head"]["units"] == ENSEMBLE_UNITS
    assert cfg["ig_head"]["outscale"] == 1.0
    assert cfg["actor"]["layers"] == ACTOR_LAYERS
    assert cfg["actor"]["units"] == ACTOR_UNITS
    assert cfg["actor"]["outscale"] == 1.0
    assert cfg["reward_head"]["outscale"] == 0.0
    assert cfg["cont_head"]["outscale"] == 1.0
    assert cfg["ig_models"] == raif_agent.ENSEMBLE_SIZE
    #  their per-member random no-learning gate is 0.0 in the resolved config, i.e. inert;
    #  the engine carries a per-member loss weight instead of the gate (tier audit T6)
    assert _num(cfg["ig_head_random_no_learning"]) == 0.0
    assert cfg["envs"]["amount"] == run_mc.LANES
    #  the four optimizer groups, whole
    assert (
        _num(cfg["actor_opt"]["eps"]),
        _num(cfg["actor_opt"]["clip"]),
        _num(cfg["actor_opt"]["lr"]),
    ) == (
        ACTOR_OPT_EPS,
        ACTOR_OPT_CLIP,
        ACTOR_OPT_LR,
    )
    assert (
        _num(cfg["critic_opt"]["eps"]),
        _num(cfg["critic_opt"]["clip"]),
        _num(cfg["critic_opt"]["lr"]),
    ) == (
        ACTOR_OPT_EPS,
        ACTOR_OPT_CLIP,
        ACTOR_OPT_LR,
    )
    assert (_num(cfg["ig_opt"]["eps"]), _num(cfg["ig_opt"]["clip"]), _num(cfg["ig_opt"]["lr"])) == (
        IG_OPT_EPS,
        IG_OPT_CLIP,
        IG_OPT_LR,
    )
    assert (
        _num(cfg["model_opt"]["eps"]),
        _num(cfg["model_opt"]["clip"]),
        _num(cfg["model_opt"]["lr"]),
    ) == (
        MODEL_OPT_EPS,
        MODEL_OPT_CLIP,
        MODEL_OPT_LR,
    )
    #  the protocol and the objective's scalars
    assert cfg["batch_size"] == raif_agent.BATCH_SIZE
    assert cfg["batch_length"] == raif_agent.BATCH_LENGTH
    assert cfg["imag_horizon"] == raif_agent.IMAG_HORIZON
    assert cfg["run"]["train_ratio"] / (cfg["batch_size"] * cfg["batch_length"]) == 1 / (
        raif_agent.TRAIN_EVERY
    )
    assert _num(cfg["actent"]) == raif_agent.ACTENT
    assert _num(cfg["wment"]) == raif_agent.WMENT
    assert cfg["return_lambda"] == DEFAULT_LAMBDA
    assert 1.0 - 1.0 / cfg["horizon"] == DEFAULT_DISCOUNT
    assert cfg["slow_critic_fraction"] == DEFAULT_SLOW_FRACTION
    assert cfg["slow_critic_update"] == DEFAULT_SLOW_PERIOD
    assert cfg["loss_scales"]["slowreg"] == DEFAULT_SLOWREG
    assert cfg["loss_scales"]["actor_refresh"] == raif_agent.BC_SCALE
    assert cfg["loss_scales"]["dyn"] == 0.5 and cfg["loss_scales"]["rep"] == 0.1
    assert cfg["dyn_loss"]["free"] == 1.0 and cfg["rep_loss"]["free"] == 1.0
    assert (
        cfg["retnorm"]["perclo"],
        cfg["retnorm"]["perchi"],
        cfg["retnorm"]["decay"],
        cfg["retnorm"]["max"],
    ) == DEFAULT_RETNORM
    assert cfg["rssm"]["deter"] == 128
    assert cfg["rssm"]["stoch"] == 32 and cfg["rssm"]["classes"] == 32
    assert cfg["rssm"]["units"] == 32 and cfg["rssm"]["unimix"] == 0.01
    assert cfg["encoder"]["mlp_units"] == 64 and cfg["decoder"]["mlp_units"] == 64
    assert cfg["wrapper"]["length"] == 100 and cfg["wrapper"]["repeat"] == 2
    assert {k: _num(v) for k, v in cfg["self_imitation"].items()} == {
        "disc": 0.4,
        "fail": 0.3,
        "thr": 0.001,
    }
    #  and the two fields we deliberately do NOT match, so the divergences cannot
    #  quietly become non-divergences under a re-vendored config
    assert cfg["jax"]["precision"] == "float16", "divergence 9 (argued) — we run float32"
    assert cfg["run"]["steps"] == 2_000_000, "divergence 19 (open) — DEFAULT_STEPS is 1M"


def test_the_audit_table_dispositions_every_field_of_that_config():
    """``amortization.md`` §(e) must mention every key — the audit's own coverage claim.

    §(e) says the config surface is "exhausted, not sampled", and that is a claim a
    test can check: walk the vendored dump and require each top-level key, and each
    one-level leaf, to appear somewhere in the table. It is what stops the table
    from silently falling behind a re-vendored config — the failure mode of every
    hand-maintained compatibility matrix.
    """
    cfg = _resolved_config()
    doc = (pathlib.Path(__file__).resolve().parents[1] / "docs/design/amortization.md").read_text()
    start = doc.index("### (e) The complete resolved-config diff")
    table = doc[start : doc.index("### (f) OPEN divergences")]

    missing = []
    for key, value in sorted(cfg.items()):
        if not re.search(rf"`{re.escape(key)}[.`{{ ]", table):
            missing.append(key)
        if not isinstance(value, dict):
            continue
        for sub, leaf in sorted(value.items()):
            if isinstance(leaf, dict):
                continue  # a two-level subtree: its parent's row carries the reason
            pattern = (
                rf"{re.escape(key)}\.(\{{[^}}]*\b{re.escape(sub)}\b[^}}]*\}}"
                rf"|{re.escape(sub)}\b)"
            )
            if not re.search(pattern, table):
                missing.append(f"{key}.{sub}")
    assert not missing, (
        f"{len(missing)} resolved-config field(s) are not dispositioned in "
        f"amortization.md §(e): {missing}"
    )


def test_the_vendored_config_is_the_one_the_oracle_ran():
    """Its provenance sidecar, and the cross-seed claim §(e) makes.

    The fixtures' convention (A12 item 6): a vendored artefact carries the commit it
    came from, so "their value" is checkable rather than asserted. The
    ``identical_across_seeds`` flag is §(e)'s own cross-check — all four oracle dumps
    agree apart from ``expname``, ``logdir`` and ``seed`` — which is why one table
    covers the whole oracle record.
    """
    import hybrid_aif.reference.raif.fixtures as fx

    directory = pathlib.Path(fx.__file__).parent
    provenance = json.loads((directory / "oracle_state_mc.config.provenance.json").read_text())
    assert len(provenance["commit_sha"]) == 40
    assert provenance["identical_across_seeds"] is True
    assert "raif-paper" in provenance["repo"]
    #  the same commit the episode fixtures came from, so the seed data and the
    #  configuration describe one run of one revision
    episodes = json.loads((directory / "oracle_mc_seed0.provenance.json").read_text())
    assert provenance["commit_sha"] == episodes["commit_sha"]
    raw = (directory / "oracle_state_mc.config.yaml").read_text().splitlines()
    assert len(raw) == provenance["n_lines"]


#: Their per-optimizer parameter counts, read from `train/params_agent/*` in
#: `raif-paper/logs/oracle_state_mc_seed0/metrics.jsonl`. A count is the sharpest
#: architectural check available: any divergence in a layer's width, depth, or
#: bias changes it, and no amount of matching formulas can hide it.
THEIR_PARAMETERS = {"model": 513_314, "actor": 428_034, "critic": 46_431, "ensemble": 422_528}

#: The CRSPP keeps its own learned initial state where theirs shares the world
#: model's, because A11 refinement (ii) forbids sharing a module object across
#: Systems. `deter` numbers, and the only accepted count divergence.
CRSPP_OWN_INITIAL = 128


def _parameters(tree) -> int:
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
    return int(sum(x.size for x in leaves))


def test_every_optimizers_parameter_count_matches_the_oracles():
    """Their counts, part for part — the check that catches an architectural drift.

    Found the 480-parameter one it was written for: `crspp_rssm` took no ``winit``,
    so the CRSPP kept four biases their normed layers drop *and* drew its six weight
    matrices from a different distribution. The bias half showed up here; the
    initializer half costs no parameters and needed a trace audit to see, which is
    why this test asserts the counts and :func:`test_the_crspp_is_initialized_like_the_world_model`
    asserts the draw.
    """
    agent = raif_mountain_car(jax.random.PRNGKey(0))
    params = agent.params
    world = sum(
        _parameters(getattr(params, field))
        for field in ("transition", "recognition", "initial_state_prior", "likelihood")
    )
    critic, slow = params.scoring["g_tail"]
    got = {
        "model": world + _parameters(agent.crspp_params) - CRSPP_OWN_INITIAL,
        "actor": _parameters(params.policy_prior),
        "critic": _parameters(critic),
        "ensemble": _parameters(params.scoring["model_uncertainty"]),
    }
    assert got == THEIR_PARAMETERS
    assert _parameters(slow) == _parameters(critic)  # the slow copy is the critic's twin


def test_the_crspp_is_initialized_like_the_world_model():
    """``winit`` reaches the CRSPP's own GRU, prior and posterior, not just the agent's.

    Their ``CRSPP_RSSM`` carries ``winit: normal`` exactly as their world model does.
    Ours threaded it into ``tiny_rssm`` and stopped there, so the CRSPP drew from
    ``eqx``'s default — 2.4x narrower on its GRU input projection, 2.3x wider on its
    stats layers — and kept four biases their normed layers drop. The bias half is
    480 parameters; the draw half costs nothing and reaches G through the
    ``cosine_sim_to_pref_rollout`` row.
    """
    normal = raif_mountain_car(jax.random.PRNGKey(0), winit="normal")
    default = raif_mountain_car(jax.random.PRNGKey(0), winit="default")
    #  the four biases their normed layers drop: img_in_2 32, gru_2 384, img_out_2 32,
    #  obs_out_2 32 — the measured 480 of the 608-parameter gap against their model_opt
    assert _parameters(default.crspp_params) - _parameters(normal.crspp_params) == 480

    #  and the draw itself differs, which no parameter count can show
    def widths(agent):
        return [
            float(jnp.std(x))
            for x in jax.tree_util.tree_leaves(eqx.filter(agent.crspp_params, eqx.is_inexact_array))
            if x.ndim == 2
        ]

    assert widths(normal) != widths(default)


# ============================================ the module tree, leaf by leaf (2026-09-16)

#: **Their** leaf modules, enumerated *mechanically* from the oracle's own checkpoint
#: (``raif-paper/logs/oracle_state_mc_seed0/checkpoint.ckpt``, step 2,000,000 — a plain
#: pickle of ``{parameter path: array}``), not from our design docs and not from a
#: hand-resolution of their config. Every previous architectural defect in this
#: replication entered through a prose specification derived by hand from their module
#: tree; this table is that tree, and the audit that built it is
#: ``docs/experiments/raif-mountain-car/module-correspondence.md``.
#:
#: Each row is ``(their leaf, our leaf paths, parameter count)``, where a "leaf" is one
#: of their ``Linear``/``Norm`` blocks — the unit their ``self.get(...)`` names — and
#: the count is the block's whole cost (kernel, bias if their ``Linear`` keeps one, and
#: the LayerNorm's scale and bias if the block is normed). Our paths are ``keystr``
#: paths into the built agent's pytrees, one group per tree in :func:`_leaf_arrays`.
#:
#: ``None`` on their side is the one accepted structural divergence: the CRSPP's own
#: learned initial carry (:data:`CRSPP_OWN_INITIAL`).
ORACLE_LEAVES: tuple[tuple[str | None, tuple[str, ...], int], ...] = (
    # ------------------------------------------------------------------- encoder
    (
        "enc/mlp/h0",
        (
            "world.recognition['embed'].layers[0].weight",
            "world.recognition['embed'].norms[0].weight",
            "world.recognition['embed'].norms[0].bias",
        ),
        256,
    ),
    (
        "enc/mlp/h1",
        (
            "world.recognition['embed'].layers[1].weight",
            "world.recognition['embed'].norms[1].weight",
            "world.recognition['embed'].norms[1].bias",
        ),
        4_224,
    ),
    # ---------------------------------------------------------------------- RSSM
    ("rssm/initial", ("world.initial_state_prior['deter'].value",), 128),
    (
        "rssm/img_in",
        (
            "world.transition['deter'].map.input_proj.weight",
            "world.transition['deter'].map.norms[0].weight",
            "world.transition['deter'].map.norms[0].bias",
        ),
        32_864,
    ),
    (
        "rssm/gru",
        (
            "world.transition['deter'].map.gate.weight",
            "world.transition['deter'].map.norms[1].weight",
            "world.transition['deter'].map.norms[1].bias",
        ),
        62_208,
    ),
    (
        "rssm/img_out",
        (
            "world.transition['stoch'].trunk.layers[0].weight",
            "world.transition['stoch'].trunk.norms[0].weight",
            "world.transition['stoch'].trunk.norms[0].bias",
        ),
        4_160,
    ),
    (
        "rssm/img_stats",
        (
            "world.transition['stoch'].trunk.layers[1].weight",
            "world.transition['stoch'].trunk.layers[1].bias",
        ),
        33_792,
    ),
    (
        "rssm/obs_out",
        (
            "world.recognition['stoch'].trunk.layers[0].weight",
            "world.recognition['stoch'].trunk.norms[0].weight",
            "world.recognition['stoch'].trunk.norms[0].bias",
        ),
        6_208,
    ),
    (
        "rssm/obs_stats",
        (
            "world.recognition['stoch'].trunk.layers[1].weight",
            "world.recognition['stoch'].trunk.layers[1].bias",
        ),
        33_792,
    ),
    # --------------------------------------------------------------------- CRSPP
    (
        "rssm/img_in_2",
        (
            "crspp.transition['deter'].map.input_proj.weight",
            "crspp.transition['deter'].map.norms[0].weight",
            "crspp.transition['deter'].map.norms[0].bias",
        ),
        32_832,
    ),
    (
        "rssm/gru_2",
        (
            "crspp.transition['deter'].map.gate.weight",
            "crspp.transition['deter'].map.norms[1].weight",
            "crspp.transition['deter'].map.norms[1].bias",
        ),
        62_208,
    ),
    (
        "rssm/img_out_2",
        (
            "crspp.transition['stoch'].trunk.layers[0].weight",
            "crspp.transition['stoch'].trunk.norms[0].weight",
            "crspp.transition['stoch'].trunk.norms[0].bias",
        ),
        4_160,
    ),
    (
        "rssm/img_stats_2",
        (
            "crspp.transition['stoch'].trunk.layers[1].weight",
            "crspp.transition['stoch'].trunk.layers[1].bias",
        ),
        33_792,
    ),
    (
        "rssm/obs_out_2",
        (
            "crspp.recognition['stoch'].trunk.layers[0].weight",
            "crspp.recognition['stoch'].trunk.norms[0].weight",
            "crspp.recognition['stoch'].trunk.norms[0].bias",
        ),
        6_208,
    ),
    (
        "rssm/obs_stats_2",
        (
            "crspp.recognition['stoch'].trunk.layers[1].weight",
            "crspp.recognition['stoch'].trunk.layers[1].bias",
        ),
        33_792,
    ),
    (None, ("crspp.initial_state_prior['deter'].value",), 0),  # ours only: CRSPP_OWN_INITIAL
    # --------------------------------------------------------------------- heads
    (
        "dec/mlp/h0",
        (
            "world.likelihood['state'].trunk.layers[0].weight",
            "world.likelihood['state'].trunk.norms[0].weight",
            "world.likelihood['state'].trunk.norms[0].bias",
        ),
        73_856,
    ),
    (
        "dec/mlp/h1",
        (
            "world.likelihood['state'].trunk.layers[1].weight",
            "world.likelihood['state'].trunk.norms[1].weight",
            "world.likelihood['state'].trunk.norms[1].bias",
        ),
        4_224,
    ),
    (
        "dec/mlp/dist_state/out",
        (
            "world.likelihood['state'].trunk.layers[2].weight",
            "world.likelihood['state'].trunk.layers[2].bias",
        ),
        130,
    ),
    (
        "rew/h0",
        (
            "world.likelihood['reward'].trunk.layers[0].weight",
            "world.likelihood['reward'].trunk.norms[0].weight",
            "world.likelihood['reward'].trunk.norms[0].bias",
        ),
        36_928,
    ),
    (
        "rew/h1",
        (
            "world.likelihood['reward'].trunk.layers[1].weight",
            "world.likelihood['reward'].trunk.norms[1].weight",
            "world.likelihood['reward'].trunk.norms[1].bias",
        ),
        1_088,
    ),
    (
        "rew/dist_out/out",
        (
            "world.likelihood['reward'].trunk.layers[2].weight",
            "world.likelihood['reward'].trunk.layers[2].bias",
        ),
        8_415,
    ),
    (
        "cont/h0",
        (
            "world.likelihood['cont'].trunk.layers[0].weight",
            "world.likelihood['cont'].trunk.norms[0].weight",
            "world.likelihood['cont'].trunk.norms[0].bias",
        ),
        36_928,
    ),
    (
        "cont/h1",
        (
            "world.likelihood['cont'].trunk.layers[1].weight",
            "world.likelihood['cont'].trunk.norms[1].weight",
            "world.likelihood['cont'].trunk.norms[1].bias",
        ),
        1_088,
    ),
    (
        "cont/dist_out/out",
        (
            "world.likelihood['cont'].trunk.layers[2].weight",
            "world.likelihood['cont'].trunk.layers[2].bias",
        ),
        33,
    ),
    # --------------------------------------------------------------------- actor
    (
        "ac/actor/h0",
        ("actor.layers[0].weight", "actor.norms[0].weight", "actor.norms[0].bias"),
        295_424,
    ),
    (
        "ac/actor/h1",
        ("actor.layers[1].weight", "actor.norms[1].weight", "actor.norms[1].bias"),
        66_048,
    ),
    (
        "ac/actor/h2",
        ("actor.layers[2].weight", "actor.norms[2].weight", "actor.norms[2].bias"),
        66_048,
    ),
    #  their `dist_out/out` and `dist_out/std` are two linears; ours is one fused
    #  layer of 2*action_dim, split by MLPConditional._head. Same count, same draw.
    ("ac/actor/dist_out", ("actor.layers[3].weight", "actor.layers[3].bias"), 514),
    # ------------------------------------------------------- critic + slow copy
    (
        "critic/net/h0",
        (
            "critic.trunk.layers[0].weight",
            "critic.trunk.norms[0].weight",
            "critic.trunk.norms[0].bias",
        ),
        36_928,
    ),
    (
        "critic/net/h1",
        (
            "critic.trunk.layers[1].weight",
            "critic.trunk.norms[1].weight",
            "critic.trunk.norms[1].bias",
        ),
        1_088,
    ),
    (
        "critic/net/dist_out/out",
        ("critic.trunk.layers[2].weight", "critic.trunk.layers[2].bias"),
        8_415,
    ),
    (
        "critic/slow/h0",
        ("slow.trunk.layers[0].weight", "slow.trunk.norms[0].weight", "slow.trunk.norms[0].bias"),
        36_928,
    ),
    (
        "critic/slow/h1",
        ("slow.trunk.layers[1].weight", "slow.trunk.norms[1].weight", "slow.trunk.norms[1].bias"),
        1_088,
    ),
    (
        "critic/slow/dist_out/out",
        ("slow.trunk.layers[2].weight", "slow.trunk.layers[2].bias"),
        8_415,
    ),
) + tuple(
    row
    for i in range(4)
    for row in (
        (
            f"epis/ig{i}/h0",
            (f"ig{i}.layers[0].weight", f"ig{i}.norms[0].weight", f"ig{i}.norms[0].bias"),
            36_960,
        ),
        (
            f"epis/ig{i}/h1",
            (f"ig{i}.layers[1].weight", f"ig{i}.norms[1].weight", f"ig{i}.norms[1].bias"),
            1_088,
        ),
        #  theirs is `dist_out/out` + `dist_out/std`, 33,792 each; ours is one fused
        #  2*1024 layer. The std half is dead on BOTH sides — see the sigma test.
        (f"epis/ig{i}/dist_out", (f"ig{i}.layers[2].weight", f"ig{i}.layers[2].bias"), 67_584),
    )
)

#: Their ``Initializer`` (``nets.py:740-786``) per weight leaf, as
#: ``(our path, fan_in, fan_out, dist, outscale)``. ``dist`` is ``"normal"`` for the
#: hidden ``h{i}`` blocks and the RSSM's normed linears (their ``winit: normal`` reaches
#: them through ``**self._kw``), ``"uniform"`` for every ``Dist`` output linear and for
#: ``…_stats`` (built with no ``**kw`` at all — divergence 26b), and ``"zeros"`` where
#: ``outscale: 0.0``. Both draws realize the same std ``sqrt(outscale / mean(fans))``;
#: the *support* separates them, which is what the initializer-law test below keys on.
ORACLE_DRAWS: tuple[tuple[str, int, int, str, float], ...] = (
    ("world.recognition['embed'].layers[0].weight", 2, 64, "normal", 1.0),
    ("world.recognition['embed'].layers[1].weight", 64, 64, "normal", 1.0),
    ("world.transition['deter'].map.input_proj.weight", 1025, 32, "normal", 1.0),
    ("world.transition['deter'].map.gate.weight", 160, 384, "normal", 1.0),
    ("world.transition['stoch'].trunk.layers[0].weight", 128, 32, "normal", 1.0),
    ("world.transition['stoch'].trunk.layers[1].weight", 32, 1024, "uniform", 1.0),
    ("world.recognition['stoch'].trunk.layers[0].weight", 192, 32, "normal", 1.0),
    ("world.recognition['stoch'].trunk.layers[1].weight", 32, 1024, "uniform", 1.0),
    ("world.likelihood['state'].trunk.layers[0].weight", 1152, 64, "normal", 1.0),
    ("world.likelihood['state'].trunk.layers[1].weight", 64, 64, "normal", 1.0),
    ("world.likelihood['state'].trunk.layers[2].weight", 64, 2, "uniform", 1.0),
    ("world.likelihood['reward'].trunk.layers[0].weight", 1152, 32, "normal", 1.0),
    ("world.likelihood['reward'].trunk.layers[1].weight", 32, 32, "normal", 1.0),
    ("world.likelihood['reward'].trunk.layers[2].weight", 32, 255, "zeros", 0.0),
    ("world.likelihood['cont'].trunk.layers[0].weight", 1152, 32, "normal", 1.0),
    ("world.likelihood['cont'].trunk.layers[1].weight", 32, 32, "normal", 1.0),
    ("world.likelihood['cont'].trunk.layers[2].weight", 32, 1, "uniform", 1.0),
    ("crspp.transition['deter'].map.input_proj.weight", 1024, 32, "normal", 1.0),
    ("crspp.transition['deter'].map.gate.weight", 160, 384, "normal", 1.0),
    ("crspp.transition['stoch'].trunk.layers[0].weight", 128, 32, "normal", 1.0),
    ("crspp.transition['stoch'].trunk.layers[1].weight", 32, 1024, "uniform", 1.0),
    ("crspp.recognition['stoch'].trunk.layers[0].weight", 192, 32, "normal", 1.0),
    ("crspp.recognition['stoch'].trunk.layers[1].weight", 32, 1024, "uniform", 1.0),
    ("actor.layers[0].weight", 1152, 256, "normal", 1.0),
    ("actor.layers[1].weight", 256, 256, "normal", 1.0),
    ("actor.layers[2].weight", 256, 256, "normal", 1.0),
    #  the fused head, half by half: each half is drawn at ITS OWN fan_out (1), not at
    #  the fused layer's 2 — which is what makes ours their two separate linears
    ("actor.layers[3].weight[:1]", 256, 1, "uniform", 1.0),
    ("actor.layers[3].weight[1:]", 256, 1, "uniform", 1.0),
    ("critic.trunk.layers[0].weight", 1152, 32, "normal", 1.0),
    ("critic.trunk.layers[1].weight", 32, 32, "normal", 1.0),
    ("critic.trunk.layers[2].weight", 32, 255, "zeros", 0.0),
    ("slow.trunk.layers[0].weight", 1152, 32, "normal", 1.0),
    ("slow.trunk.layers[1].weight", 32, 32, "normal", 1.0),
    ("slow.trunk.layers[2].weight", 32, 255, "zeros", 0.0),
) + tuple(
    row
    for i in range(4)
    for row in (
        (f"ig{i}.layers[0].weight", 1153, 32, "normal", 1.0),
        (f"ig{i}.layers[1].weight", 32, 32, "normal", 1.0),
        (f"ig{i}.layers[2].weight[:1024]", 32, 1024, "uniform", 1.0),
        (f"ig{i}.layers[2].weight[1024:]", 32, 1024, "uniform", 1.0),
    )
)

#: The divisor in their ``Initializer``'s normal branch (``nets.py:761``): the std of a
#: unit normal truncated to [-2, 2], so the drawn weights' std is ``sqrt(scale/denom)``
#: exactly and their support is ``2/0.8796 = 2.274`` of it.
TRUNC_NORMAL_STD = 0.87962566103423978


def _leaf_arrays(agent) -> dict[str, np.ndarray]:
    """Every inexact array of the built agent, keyed ``<group><keystr path>``."""
    import dataclasses

    trees = {
        #  the control box is the action's support, not a parameter of any of their modules
        "world": dataclasses.replace(agent.params, scoring={}, policy_prior=None, control={}),
        "crspp": agent.crspp_params,
        "actor": agent.params.policy_prior,
        "critic": agent.params.scoring["g_tail"][0],
        "slow": agent.params.scoring["g_tail"][1],
        **{f"ig{i}": m for i, m in enumerate(agent.params.scoring["model_uncertainty"])},
    }
    out = {}
    for group, tree in trees.items():
        flat = jax.tree_util.tree_flatten_with_path(eqx.filter(tree, eqx.is_inexact_array))[0]
        for path, x in flat:
            out[group + jax.tree_util.keystr(path)] = np.asarray(x)
    return out


def _slice_of(arrays: dict[str, np.ndarray], path: str) -> np.ndarray:
    """``path`` with an optional trailing ``[:n]``/``[n:]`` naming a fused head's half."""
    if path.endswith("]") and "weight[" in path:
        base, spec = path.split("weight[")
        arr = arrays[base + "weight"]
        spec = spec[:-1]
        return arr[: int(spec[1:])] if spec.startswith(":") else arr[int(spec[:-1]) :]
    return arrays[path]


def test_the_module_tree_is_theirs_leaf_for_leaf(agent):
    """Their tree, block for block, against ours — the standing structural reference.

    The counts on the right are **their** blocks' costs, read off their own checkpoint
    rather than computed from our sizes, so this fails on any drift in a width, a
    depth, a bias, or a norm. The one accepted divergence is the CRSPP's own learned
    initial, which is the row whose oracle side is ``None``.

    Coverage is asserted both ways: every leaf of the built agent belongs to exactly
    one row, so a *new* parameter anywhere fails here too.
    """
    arrays = _leaf_arrays(agent)
    seen: set[str] = set()
    for their, ours, count in ORACLE_LEAVES:
        got = 0
        for path in ours:
            assert path not in seen, f"{path} claimed by two rows"
            assert path in arrays, f"{path} is not a leaf of the built agent"
            seen.add(path)
            got += arrays[path].size
        expect = count if their is not None else CRSPP_OWN_INITIAL
        assert got == expect, f"{their or '(ours only)'}: theirs {expect}, ours {got}"
    assert seen == set(arrays), f"unmapped leaves: {sorted(set(arrays) - seen)}"

    #  and the two ends of the reconciliation: their four optimizer totals, and the sum
    theirs = sum(c for their, _, c in ORACLE_LEAVES if their is not None)
    assert theirs == sum(THEIR_PARAMETERS.values()) + THEIR_PARAMETERS["critic"]  # + slow
    assert sum(x.size for x in arrays.values()) == theirs + CRSPP_OWN_INITIAL


def test_every_weight_leaf_obeys_their_initializer_law(agent):
    """Their ``Initializer``'s std **and** its support, leaf by leaf.

    The second moment alone cannot tell their ``normal`` branch from their ``uniform``
    one — that is divergence 26b's whole content — so this keys on the support too: a
    uniform draw cannot exceed ``sqrt(3)·std_law`` and a truncated normal's reaches
    ``2·std_law/0.8796``. The 2026-09-16 audit's finding that the CRSPP's six weight
    matrices were on eqx's ``U(+-1/sqrt(fan_in))`` would have failed four of these rows
    on the std and all six on the support.
    """
    arrays = _leaf_arrays(agent)
    covered = {p.split("weight[")[0] + "weight" if "weight[" in p else p for p, *_ in ORACLE_DRAWS}
    assert covered == {p for p in arrays if p.endswith("weight") and arrays[p].ndim == 2}
    for path, fan_in, fan_out, dist, outscale in ORACLE_DRAWS:
        arr = _slice_of(arrays, path)
        std, support = float(arr.std()), float(np.abs(arr).max())
        if dist == "zeros":
            assert support == 0.0, f"{path}: outscale 0.0 must draw exactly zeros"
            continue
        law = float(np.sqrt(outscale / ((fan_in + fan_out) / 2)))
        assert 0.85 <= std / law <= 1.15, f"{path}: std {std:.5f} against law {law:.5f}"
        if dist == "uniform":
            assert support <= np.sqrt(3) * law * 1.001, f"{path}: outside U(+-sqrt(3)*std)"
            assert support >= np.sqrt(3) * law * 0.90, f"{path}: too narrow for a uniform"
        else:
            hi = 2 * law / TRUNC_NORMAL_STD
            assert support <= hi * 1.001, f"{path}: outside their +-2 sigma truncation"
            assert support > np.sqrt(3) * law * 0.92, f"{path}: no tail — a uniform draw?"


def test_every_bias_their_linear_keeps_starts_at_zero(agent):
    """Their kept biases are ``jnp.zeros`` (``nets.py:679``), and the normed ones absent.

    The presence half is already pinned by :func:`test_the_module_tree_is_theirs_leaf_for_leaf`
    (a bias their ``Linear`` drops is 32-384 parameters ours would carry); this pins the
    *value*, which no count can see, and the LayerNorm pair's ``(ones, zeros)`` with it.
    """
    arrays = _leaf_arrays(agent)
    for path, arr in arrays.items():
        if path.endswith(".bias") and ".norms[" not in path:
            assert np.all(arr == 0.0), f"{path} does not start at zero"
        if ".norms[" in path:
            want = 1.0 if path.endswith(".weight") else 0.0
            assert np.all(arr == want), f"{path} does not start at {want}"


def test_the_ensemble_heads_sigma_is_the_constant_one_with_no_gradient(agent):
    """Their ``ig_head`` declares no ``minstd``/``maxstd``, so ``Dist``'s (1.0, 1.0)
    defaults make ``std = (1-1)*sigma(raw+2) + 1`` identically 1.0 — divergence 15.

    Half of each member's output layer (33,792 of its 105,632 parameters) is therefore
    dead on **both** sides. The count test above cannot see that, and a future change to
    :data:`~hybrid_aif.reference.raif.agent.ENSEMBLE_STD` that separated the two bounds
    would silently wake 135,168 parameters across the ensemble.
    """
    member = agent.params.scoring["model_uncertainty"][0]
    parents = (
        jnp.ones(member.parent_dims[0]) * 0.3,
        jnp.zeros(member.parent_dims[1]).at[::32].set(1.0),
        jnp.array([0.5]),
    )
    _, scale = member.diag_moments(parents)
    assert jnp.all(scale == 1.0)

    grad = eqx.filter_grad(lambda m: jnp.sum(m.diag_moments(parents)[1]))(member)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grad, eqx.is_inexact_array))
    assert all(bool(jnp.all(g == 0.0)) for g in leaves)
