"""Compose a small System from scratch: "which object is meant?"

A tutorial for the *composition* path — what you write down to get a new
generative model, as opposed to running one the library already declares.
It is the worked example behind `docs/composing-systems.md` and `docs/agent-composition-probe.md`.

The model
---------
An operator refers to one of two objects ("the one near the bin", "the big
one") and the agent must work out which. Two noisy features are binned into
tabular modalities; a third modality is a supervisor's *answer*, available
only after the agent spends an action asking for it.

    factors     target  {cand0, cand1, ambiguous}   uncontrolled, persists
                mode    {idle, asked}               moved by u_mode
    modalities  distance {near, mid, far}   observes target      (learnable)
                size     {small, mid, big}  observes target      (learnable)
                answer   {none, says0, says1} observes (target, mode)
    actions     {pick0, pick1, ask} -- only `ask` puts mode into `asked`
    preference  obs_targets["answer"] favours a resolved answer over "none"

Five sections, each printing what it demonstrates:

    1. the canonical loop over five observations -- posterior and action
    2. a conjugate likelihood update from a supervisor's label, and the
       posterior flip it produces on the same observation
    3. patching a live declaration with `dataclasses.replace` (add a modality)
    4. an ill-formed patch, and the error it actually raises

Shape rules this file exists to demonstrate (see capabilities.md):

  * every transition table carries a trailing control axis, size 1 when the
    factor is uncontrolled -- `parent_ndims == len(B_deps[f]) + 1` always;
  * `parent_ndims` counts parent *axes*, it is not their sizes;
  * tabular tables are child-first, `(num_levels, *parent_shape)`.

Run: .venv/bin/python examples/compose_target_factor.py
"""

import dataclasses

import jax
import jax.numpy as jnp

import hybrid_aif as H
from hybrid_aif.system import System, SystemParams

T, M, L = 3, 2, 3  # targets, modes, feature bins


def logt(a):
    """Log of a probability table, floored so a structural zero stays finite."""
    return jnp.log(jnp.asarray(a) + 1e-9)


# --- transitions ----------------------------------------------------------
# Child-first, with a trailing control axis on BOTH factors: (next, prev, u).
# `target` is uncontrolled, so its control axis has size 1 -- it is not
# optional. `make_transition` slices axis -1 by the action for every factor.
B_target = logt(jnp.eye(T))[:, :, None]  # (next=3, prev=3, u=1)
B_mode = jnp.stack(
    [
        logt(jnp.array([[1.0, 1.0], [0.0, 0.0]])),  # u=pick0 -> idle
        logt(jnp.array([[1.0, 1.0], [0.0, 0.0]])),  # u=pick1 -> idle
        logt(jnp.array([[0.0, 0.0], [1.0, 1.0]])),  # u=ask   -> asked
    ],
    axis=-1,
)  # (next=2, prev=2, u=3)

# --- likelihoods ----------------------------------------------------------
# Child-first again: (level, *parents). The two feature channels are declared
# as ConditionalDirichlet, which is what makes them *learnable* -- a
# point-valued ConditionalCategorical is incapable of absorbing a statistic,
# not merely frozen.
A_dist = jnp.array([[0.7, 0.1, 0.2], [0.2, 0.2, 0.4], [0.1, 0.7, 0.4]])  # (bin, target)
A_size = jnp.array([[0.6, 0.2, 0.3], [0.3, 0.3, 0.4], [0.1, 0.5, 0.3]])
A_answer = jnp.stack(
    [
        logt(jnp.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),  # idle: "none"
        logt(jnp.array([[0.05, 0.05, 0.1], [0.9, 0.05, 0.45], [0.05, 0.9, 0.45]])),  # asked
    ],
    axis=-1,
)  # (answer=3, target=3, mode=2)

system = System(
    B_deps=((0,), (1,)),  # each factor's transition conditions on itself
    controlled=(False, True),
    policy_len=2,
    A_deps=((0,), (0,), (0, 1)),  # distance/size see target; answer sees both
    planner="enumeration",
    gamma=4.0,
    factor_names=("target", "mode"),
    modality_names=("distance", "size", "answer"),
    epistemic=("target",),  # mount an infogain leaf per channel observing `target`
)

# Each slot is keyed by name: transitions and initial priors by factor, likelihoods by modality.
params = SystemParams(
    transition={
        "target": H.ConditionalCategorical(B_target, 2),  # len(B_deps[0]) + 1 == 2
        "mode": H.ConditionalCategorical(B_mode, 2),
    },
    initial_state_prior={
        "target": H.Categorical(jnp.zeros(T)),  # flat over targets
        "mode": H.Categorical(logt(jnp.array([1.0, 0.0]))),  # starts idle
    },
    likelihood={
        "distance": H.ConditionalDirichlet(A_dist * 4.0 + 0.5, 1),  # learnable
        "size": H.ConditionalDirichlet(A_size * 4.0 + 0.5, 1),  # learnable
        "answer": H.ConditionalCategorical(A_answer, 2),  # fixed
    },
    # A preference is a designation: a target distribution for a named block.
    # "Resolving the target" = strongly preferring an informative answer.
    obs_targets={"answer": H.Categorical(jnp.array([-3.0, 2.0, 2.0]))},
)


def obs(*levels, names=system.modality_names):
    """One tick's evidence, keyed by modality: an int32 level index per tabular modality."""
    return {n: jnp.asarray(i, jnp.int32) for n, i in zip(names, levels, strict=True)}


def q_target(beliefs):
    return jnp.exp(beliefs["target"].log_probs).round(3)


ACTIONS = ("pick0", "pick1", "ask")


def main() -> None:
    key = jax.random.PRNGKey(0)

    # === 1. the canonical loop ===========================================
    # infer_states -> (beliefs, detail); act -> (action, q_pi, metrics). Observations are keyed
    # by modality, beliefs by factor, the action by action node ("a_mode": the action of `u_mode`).
    print("1. five ticks of the canonical loop")
    priors = None
    tree = None
    for t, o in enumerate([(0, 0, 0), (0, 1, 0), (0, 0, 0), (1, 0, 0), (0, 0, 0)]):
        beliefs, _detail = system.infer_states(params, obs(*o), priors)
        action, _q_pi, tree = system.act(key, params, beliefs, selection="deterministic")
        print(f"   t={t}  q(target)={q_target(beliefs)}  action={ACTIONS[int(action['a_mode'])]}")
        priors = system.update_empirical_prior(params, beliefs, action)
    # The term tree is {row: {leaf_key: Interval}}, one value per policy.
    # infogain leaves are keyed by MODALITY, not by the state variable named
    # in `epistemic=`: `("target",)` mounts one per channel that observes it.
    print("   term tree:", {row: sorted(leaves) for row, leaves in tree.items()})

    # === 2. conjugate learning from a supervisor's label =================
    # The supervisor says it was candidate 1. That is a claim about the
    # *state*, so it enters as a belief: `learn` absorbs one tick's
    # sufficient statistic into every learnable slot, which here moves the
    # Dirichlet counts of the two feature likelihoods.
    print("\n2. a labelled answer moves the likelihood")
    ambiguous = obs(0, 0, 0)  # the reading that currently looks like cand0
    before, _ = system.infer_states(params, ambiguous, None)
    label = {
        "target": H.Categorical(logt(jnp.array([0.0, 1.0, 0.0]))),  # target = cand1
        "mode": H.Categorical(logt(jnp.array([1.0, 0.0]))),  # mode = idle
    }
    taught = params
    for _ in range(8):  # eight times the supervisor corrects the same reading
        taught = system.learn(taught, ambiguous, label, lr=1.0)
    after, _ = system.infer_states(taught, ambiguous, None)
    print(f"   before: q(target)={q_target(before)}")
    print(f"   after:  q(target)={q_target(after)}   <- same observation, moved likelihood")

    # === 3. patch a live declaration =====================================
    # A System is all-static, so it is hashable and `dataclasses.replace`
    # patches it. Adding a modality is a paired edit: the structure on
    # `System`, the values on `SystemParams`.
    print("\n3. patching the declaration: add a 'colour' modality")
    patched = dataclasses.replace(
        system,
        A_deps=system.A_deps + ((0,),),
        modality_names=system.modality_names + ("colour",),
    )
    patched_params = dataclasses.replace(
        params,
        likelihood={
            **params.likelihood,
            "colour": H.ConditionalCategorical(
                logt(jnp.array([[0.8, 0.1, 0.5], [0.2, 0.9, 0.5]])), 1
            ),
        },
    )
    beliefs, _ = patched.infer_states(
        patched_params, obs(0, 0, 0, 1, names=patched.modality_names), None
    )
    print(f"   4 modalities: q(target)={q_target(beliefs)}")

    # === 4. an ill-formed patch ==========================================
    # Half a patch: the System grew a modality, the params did not. This is
    # the first thing a program-generated patch gets wrong, and it
    # escapes validation.
    print("\n4. the same patch, applied to only one half")
    try:
        patched.infer_states(params, obs(0, 0, 0, 1, names=patched.modality_names), None)
    except Exception as exc:  # noqa: BLE001
        print(f"   raised: {type(exc).__name__}: {exc}")
        print("   what a validator should have said: System declares 4 modalities")
        print("   ('distance', 'size', 'answer', 'colour') but SystemParams.likelihood")
        print("   has 3 entries; add a declaration for 'colour' or drop it from A_deps.")


if __name__ == "__main__":
    main()
