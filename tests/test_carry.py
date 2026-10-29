"""``System.carry``: a rebuilt model's params keep the posteriors learning put into the old ones
(``docs/design/parameter-learning.md`` D9)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hybrid_aif as H
from hybrid_aif.distributions import InverseGamma
from hybrid_aif.learning.carry import learnable_blocks
from hybrid_aif.replay import replay
from hybrid_aif.system import System, SystemParams
from tests.test_description_length import _tabular_system, _total_trace


def _same(a, b) -> bool:
    leaves_a, leaves_b = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(leaves_a) == len(leaves_b) and all(
        np.array_equal(np.asarray(x), np.asarray(y))
        for x, y in zip(leaves_a, leaves_b, strict=True)
    )


def test_a_learned_table_is_carried_and_a_point_table_rebuilt():
    system, params = _tabular_system(learned=("o0",))
    learned = _total_trace(system, params, learn=True).params
    assert not _same(learned.likelihood["o0"], params.likelihood["o0"])  # it learned
    rebuilt = H.ConditionalCategorical(
        log_table=params.transition["s0"].log_table + 0.0, parent_ndims=2
    )
    fresh = dataclasses.replace(params, transition={"s0": rebuilt})
    out = system.carry(fresh, learned)
    assert _same(out.likelihood["o0"], learned.likelihood["o0"])
    assert out.transition["s0"] is rebuilt
    assert _same(jax.jit(system.carry)(fresh, learned), out)


def _lg_system():
    system = System(
        A_deps=((0,),),
        B_deps=((),),
        controlled=(False,),
        policy_len=1,
        modality_names=("y",),
        factor_names=("x",),
        epistemic=False,
    )

    def params(slope):
        return SystemParams(
            likelihood={
                "y": H.LinearGaussianObservation(
                    H=jnp.asarray([[slope]]),
                    c=jnp.zeros(1),
                    noise=InverseGamma.from_R_chol(jnp.full((1, 1), 0.6), strength=2.0),
                )
            },
            transition={
                "x": H.LinearGaussianTransition(
                    F=jnp.asarray([[1.0]]),
                    innovations=H.ConditionalGaussian(
                        loc=jnp.zeros((1, 1)), chol=jnp.asarray([[[0.3]]]), parent_ndims=1
                    ),
                )
            },
            initial_state_prior={
                "x": H.Gaussian(loc=jnp.asarray([0.2]), chol=jnp.asarray([[1.0]]))
            },
        )

    return system, params


def test_a_learned_noise_is_carried_beside_a_rebuilt_point_head():
    """The noise posterior comes from the learned params, the re-expressed point head from the
    fresh ones; the noise is its own block, and a key for the absent head posterior raises."""
    system, params = _lg_system()
    y = jnp.asarray(np.random.default_rng(0).normal(size=(20, 1)), jnp.float32)
    learned = replay(system, params(1.0), {"y": y}, {}, learn=True, metrics=False).params
    fresh = params(2.0)
    for slots in (None, ("likelihood:y",), ("likelihood:y:noise",)):
        out = system.carry(fresh, learned, slots)
        assert float(out.likelihood["y"].H[0, 0]) == 2.0
        assert _same(out.likelihood["y"].noise, learned.likelihood["y"].noise)
    with pytest.raises(ValueError, match="names no learned posterior"):
        system.carry(fresh, learned, ("likelihood:y:head",))


def test_a_joint_emission_is_one_block():
    k = 3
    system = System(
        A_deps=((0,),),
        B_deps=((),),
        controlled=(False,),
        policy_len=1,
        modality_names=("y",),
        factor_names=("z",),
        epistemic=False,
    )
    conc = jnp.ones(k)
    params = SystemParams(
        likelihood={
            "y": H.gaussian_emission(
                jnp.zeros((k, 2)), jnp.broadcast_to(jnp.eye(2), (k, 2, 2)), strength=0.5
            )
        },
        transition={"z": H.ConditionalDirichlet(concentration=conc[:, None], parent_ndims=1)},
        initial_state_prior={"z": H.Dirichlet(concentration=conc)},
    )
    y = jnp.asarray(np.random.default_rng(1).normal(size=(12, 2)), jnp.float32)
    learned = replay(system, params, {"y": y}, {}, learn=True, metrics=False).params
    out = system.carry(params, learned)
    assert _same(out.likelihood["y"], learned.likelihood["y"])
    assert _same(out.transition["z"], learned.transition["z"])
    with pytest.raises(ValueError, match="names no learned posterior"):
        system.carry(params, learned, ("likelihood:y:noise",))


def test_a_rebuild_that_changes_what_is_learned_is_refused():
    system, params = _tabular_system(learned=("o0",))
    learned = _total_trace(system, params, learn=True).params
    point = H.ConditionalCategorical(
        log_table=jnp.log(params.likelihood["o0"].mean()), parent_ndims=1
    )
    with pytest.raises(ValueError, match="drop or invent a posterior"):
        system.carry(dataclasses.replace(params, likelihood={"o0": point}), learned)
    plain = dataclasses.replace(
        learned,
        initial_state_prior={"s0": H.Dirichlet(concentration=jnp.ones(3))},
    )
    growable = dataclasses.replace(
        params,
        initial_state_prior={"s0": H.Dirichlet(concentration=jnp.ones(3), growth=jnp.asarray(1.0))},
    )
    with pytest.raises(ValueError, match="differs in form"):
        system.carry(growable, plain, ("initial_state_prior:s0",))


def test_a_trainable_neural_conditional_is_a_block_and_a_frozen_one_is_not():
    net = H.MLPConditional.from_shapes((1, 1), 1, key=jax.random.PRNGKey(0), depth=0)
    assert learnable_blocks(net) == ("",)
    assert learnable_blocks(dataclasses.replace(net, frozen=True)) == ()


def test_a_separate_matrix_normal_head_is_not_learned_so_it_is_rebuilt():
    """``learn`` does not update a separate ``MatrixNormal`` head (D8), so the carry takes it
    from the fresh params beside the carried noise."""
    from hybrid_aif.distributions.matrix_normal import MatrixNormal

    system, params = _lg_system()

    def with_head(slope):
        decl = params(1.0).likelihood["y"]
        head = MatrixNormal.from_head(jnp.asarray([[slope, 0.0]]), strength=2.0)
        return dataclasses.replace(
            params(1.0),
            likelihood={"y": H.LinearGaussianObservation(head=head, noise=decl.noise)},
        )

    start = with_head(1.0)
    beliefs, _ = system.infer_states(start, {"y": jnp.asarray([0.4])}, None)
    learned = system.learn(
        start,
        {"y": jnp.asarray([0.4])},
        beliefs,
        prev_beliefs=beliefs,
        lr_by_slot={"likelihood:y:head": 0.0},
    )
    out = system.carry(with_head(2.0), learned)
    assert float(out.likelihood["y"].H[0, 0]) == 2.0
    assert _same(out.likelihood["y"].noise, learned.likelihood["y"].noise)


def test_a_growable_factors_blocks_move_together():
    """Growth and reduction keep a growable factor's weights, initial prior and emissions in
    step, so a selection that splits them is refused; all of them, or none, is fine."""
    k = 4
    system = System(
        A_deps=((0,),),
        B_deps=((),),
        controlled=(False,),
        policy_len=1,
        modality_names=("y",),
        factor_names=("z",),
        epistemic=False,
    )
    weights = jnp.zeros(k).at[0].set(1.0)
    params = SystemParams(
        likelihood={
            "y": H.gaussian_emission(
                jnp.zeros((k, 2)), jnp.broadcast_to(jnp.eye(2), (k, 2, 2)), strength=0.1
            )
        },
        transition={
            "z": H.ConditionalDirichlet(
                concentration=weights[:, None], parent_ndims=1, growth=jnp.asarray(1.0)
            )
        },
        initial_state_prior={"z": H.Dirichlet(concentration=weights, growth=jnp.asarray(1.0))},
    )
    centres = np.asarray([[0.0, 0.0], [9.0, 9.0], [-9.0, 9.0]])
    rng = np.random.default_rng(3)
    y = jnp.asarray(centres[rng.integers(0, 3, 30)] + 0.3 * rng.normal(size=(30, 2)), jnp.float32)
    learned = replay(system, params, {"y": y}, {}, learn=True, metrics=False).params
    with pytest.raises(ValueError, match="carry all of"):
        system.carry(params, learned, ("likelihood:y",))
    whole = system.carry(params, learned, ("likelihood:y", "transition:z", "initial_state_prior:z"))
    assert _same(whole, system.carry(params, learned))


def test_the_selector_takes_block_keys_only():
    system, params = _tabular_system(learned=("o0",))
    learned = _total_trace(system, params, learn=True).params
    with pytest.raises(TypeError, match="collection of keys"):
        system.carry(params, learned, "likelihood:o0")
    for key in ("likelihood:*", "likelihood"):
        with pytest.raises(ValueError, match="not a block key"):
            system.carry(params, learned, (key,))
    with pytest.raises(ValueError, match="names no learned posterior"):
        system.carry(params, learned, ("likelihood:o",))
    out = jax.jit(system.carry, static_argnums=2)(params, learned, ("likelihood:o0",))
    assert _same(out.likelihood["o0"], learned.likelihood["o0"])
