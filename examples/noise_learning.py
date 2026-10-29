"""Per-cell sensor-noise learning: the notebook's Section 7 exhibit.

The coverage story in miniature (design: ``parameter-learning.md`` D8): a
robot's position fix is sharp in some cells of its map and loose in others.
A *cell* modality reports which cell it is in (noisily); a *pose* modality
observes a latent 1-D position whose noise level depends on the cell. The
per-cell noise sits in the likelihood slot as an :class:`InverseGamma`
posterior — the polymorphic-slot rule: a slot holds a point value or the
distribution over it, and the distributional form is what makes it
learnable — and ``System.learn`` sharpens it online from the agent's own
filtered beliefs, with the credit assignment done by the pre-reduction
responsibilities that ``infer_states(detail=True)`` surfaces.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hybrid_aif.distributions import (
    Categorical,
    ConditionalCategorical,
    ConditionalGaussian,
    InverseGamma,
    MixtureGaussian,
)
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianTransition
from hybrid_aif.system import System, SystemParams

TRUE_SIGMA = (0.3, 0.9)  # cell 0: sharp fixes; cell 1: loose fixes
CELL_PRIOR = Categorical.from_probs(jnp.asarray([0.6, 0.4]))
X_PRIOR = MixtureGaussian(
    log_weights=jnp.zeros(1), means=jnp.asarray([[0.0]]), chols=jnp.asarray([[[0.35]]])
)


def make_system() -> System:
    return System(
        A_deps=((0,), (1, 0)),  # cell reports the mode; pose observes x, switched by it
        B_deps=((), ()),
        controlled=(False, False),
        policy_len=1,
        mixture_k=4,
        mixture_schedule="runnalls",
        modality_names=("cell", "pose"),
        factor_names=("mode", "x"),
    )


def make_params(noise) -> SystemParams:
    """``noise`` is a per-cell ``R_chol`` array (the true world) or an
    ``InverseGamma`` posterior over it (the learning agent)."""
    return SystemParams(
        likelihood={
            "cell": ConditionalCategorical(  # the cell report: right 95% of the time
                log_table=jnp.log(jnp.asarray([[0.95, 0.05], [0.05, 0.95]])), parent_ndims=1
            ),
            "pose": LinearGaussianObservation(H=jnp.asarray([[1.0]]), c=jnp.zeros(1), noise=noise),
        },
        transition={
            "mode": ConditionalCategorical(
                log_table=jnp.log(jnp.asarray([[0.5], [0.5]])), parent_ndims=1
            ),
            "x": LinearGaussianTransition(
                F=jnp.asarray([[1.0]]),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 1)), chol=jnp.asarray([[[0.05]]]), parent_ndims=1
                ),
            ),
        },
        initial_state_prior={"mode": CELL_PRIOR, "x": X_PRIOR},
    )


class LearningRun(NamedTuple):
    sigma: jnp.ndarray  # (n_ticks, 2) — the running per-cell noise estimate
    responsibility: jnp.ndarray  # (n_ticks,) — q(cell 0) at each tick


def run(key, n_ticks: int = 3000, r0: float = 0.6, decay: float | None = 0.999) -> LearningRun:
    """Learn both cells' noise online from the System's own forward samples.

    The agent starts ignorant (both cells at ``r0``) and never sees the true
    cell — only the 95%-reliable report — so every update is weighted by the
    posterior responsibility read from the pre-reduction mixture (O3).
    ``decay=None`` runs the pure conjugate accumulation; a value < 1 is the
    anchored-forgetting knob (D7), which washes out the early
    biased-estimate phase (see ``test_lg_noise_learning`` for the measured
    convergence baselines).
    """
    system = make_system()
    true = make_params(jnp.asarray([[[TRUE_SIGMA[0]]], [[TRUE_SIGMA[1]]]]))
    agent = make_params(
        InverseGamma.from_R_chol(jnp.full((2, 1, 1), r0), strength=2.0, anchor=True)
    )
    env, env_params = system.as_env(true)
    obs = jax.vmap(lambda k: env.reset(k, env_params)[0])(jax.random.split(key, n_ticks))
    priors = {"mode": CELL_PRIOR, "x": X_PRIOR}
    kw = {} if decay is None else {"decay_by_slot": {"likelihood:pose": decay}}

    def tick(params, obs_t):
        beliefs, det = system.infer_states(params, obs_t, priors=priors, detail=True)
        params = system.learn(params, obs_t, beliefs, detail=det, **kw)
        ig = params.likelihood["pose"].noise
        return params, (jnp.sqrt(ig.beta / ig.alpha)[:, 0], beliefs["mode"].probs[0])

    _, (sigma, resp) = jax.jit(lambda p: jax.lax.scan(tick, p, obs))(agent)
    return LearningRun(sigma=sigma, responsibility=resp)


def recover_table(key, n_ticks: int = 400):
    """The discrete twin, three lines of story: draw the TRUE observation
    table from the agent's own Dirichlet prior (``as_env(param_key=...)``),
    generate data through it, and learn it back. Returns the running
    max-abs-error between the learned mean table and the drawn truth."""
    from hybrid_aif.distributions import ConditionalDirichlet

    system = System(
        A_deps=((0,),),
        B_deps=((0,),),
        controlled=(False,),
        policy_len=1,
        modality_names=("sense",),
        factor_names=("s",),
    )
    prior = ConditionalDirichlet.from_table(
        jnp.asarray([[0.7, 0.3], [0.3, 0.7]]), parent_ndims=1, strength=2.0
    )
    params = SystemParams(
        likelihood={"sense": prior},
        transition={
            "s": ConditionalCategorical(log_table=jnp.log(jnp.full((2, 2, 1), 0.5)), parent_ndims=2)
        },
        initial_state_prior={"s": Categorical.from_probs(jnp.asarray([0.5, 0.5]))},
    )
    k_draw, k_data = jax.random.split(key)
    env, env_params = system.as_env(params, param_key=k_draw)  # the drawn TRUE table
    true_table = env_params.likelihood["sense"]  # discrete env params hold the raw table
    obs = jax.vmap(lambda k: env.reset(k, env_params)[0])(jax.random.split(k_data, n_ticks))

    def tick(p, obs_t):
        beliefs, _ = system.infer_states(p, obs_t)
        p = system.learn(p, obs_t, beliefs)
        err = jnp.max(jnp.abs(jnp.exp(p.likelihood["sense"].point().log_probs) - true_table))
        return p, err

    _, err = jax.jit(lambda p: jax.lax.scan(tick, p, obs))(params)
    return err, true_table


if __name__ == "__main__":
    out = run(jax.random.PRNGKey(7))
    print("final sigma:", out.sigma[-1], "true:", TRUE_SIGMA)
    err, _ = recover_table(jax.random.PRNGKey(0))
    print("table recovery max-err:", float(err[-1]))
