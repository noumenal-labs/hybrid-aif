# Hybrid-AIF: Hybrid continuous-discrete active inference & RL

A software package (in active development, and in many respects experimental; see `docs/capabilities.md`) enabling production-grade, potentially large-scale implementations of active inference (AIF), with efficient parallel policy search across multiple agents using JAX, both continuous and discrete observations and state factors including mixtures, various options for approximate inference (variational message-passing schemes, MCMC, etc.), and flexibility in specifying models and objective functions.

Status: see `docs/capabilities.md` (maintained per release) and
`docs/project-brief.md`. Supporting docs in `/docs`:

- `capabilities.md` — what ships, release by release, with each release's migration note (start here to use it)
- `composing-systems.md` — the `System` and its fields, the term tree, merging Systems
- `description-format.md` — the model description, a versioned JSON document of a model
- `project-brief.md` — the design overview for team circulation (start here for the design)
- `scoping-review.md` — vetting of this plan against the reference repos and papers (frozen snapshot)
- `de-risk-report.md` — resolution of the three highest-uncertainty design items (MNLR inference, mixture propagation, model representation)
- `design/system.md`, `design/conditional-families.md` — adopted architecture decisions

## Install

```bash
uv venv && uv pip install -e ".[dev]"   # or: pip install -e ".[dev]"
.venv/bin/python -m pytest              # 2,500+ tests
```

Requires jax>=0.11, equinox>=0.13.8, numpy>=2.0 (older jax minors break
mixture KL paths; the floors are load-bearing).

## Seed idea

`pymdp` is great for fast active inference on discrete spaces; this project explores the complementary hybrid continuous-discrete (mixture-model) setting, which, to our knowledge, existing tools do not yet cover. Mixture/switching structure matters here because the epistemic term of expected free energy is provably constant in linear-Gaussian systems (Koudahl et al. 2021), so hybrid models are the minimal class in which continuous-state epistemic AIF does anything at all. No existing work propagates mixture beliefs through policy rollouts with a principled reduction schedule — the ingredients exist (tracking, information theory), the composition is open. Given the relative ease of agent-assisted coding, this seems like a good time to build the stack.

The framework should also be expressive enough to *build* modern model-based agents inside it — e.g. R-AIF-style learned preferences, or JEPA-style latent-space objectives (which amount to moving prediction-error terms from observation space to latent space — something AIF in principle supports, plus anti-collapse machinery).

## Use of software agents

Most of this code was written by Claude, through Claude Code under the direction of the maintainers: we set the goals and made the design decisions, and agents proposed, implemented, tested and reviewed (using a Fable agent for most top-level design collaboration). The design records in `docs/design/` show how each decision was reached. This project is in part an exercise in learning how useful research software can be built leveraging coding agents, exploiting the efficiency that they afford without compromising on scientific or academic integrity (the jury is still out on the extent to which this is possible).

## Design principles

- **Modular** - can take whatever else we use in our stack that's relevant as input and produce outputs consumable by various environments including the real world

- **Multi-agent capability**: `vmap` not only across policies for policy search, but across agents. Homogeneous-shape agent batches first (the `pymdp`-jax pattern), built in at ground level; heterogeneous shapes via padding later.

- **Flexibility** - Rich options for specifying agents and policy objectives:
    - Hierarchical priors on any slow lower-level parameter ("parametric depth" — the $\gamma$/precision mechanism of [Deeply Felt Affect](https://direct.mit.edu/neco/article/33/2/398/95642/Deeply-Felt-Affect-The-Emergence-of-Valence-in) is one instance)
    - Enable or disable epistemic and prior/KL terms per variable — at mean-field-block granularity (the additive per-variable decomposition exists exactly where the posterior factorizes; structured posteriors get documented approximations)
    - Preferences as flexibly specifiable as any distribution: any node can carry a "phenotypic prior" / homeostatic target; online preference learning as an option
    - Hierarchical latent variable models
    - Any variable can be made conditional/time-dependent
    - Ablations recover standard model-based RL: EFE ⊃ KL-control ⊃ expected utility

- **Model representation** - A typed **node menu** (Categorical, Linear-Gaussian, GMM, MNLR, custom) composed in hierarchical templates, rather than a general factor-graph language. Source of truth: typed Python spec (pydantic) carrying distribution type, mean-field block id, inference method, and wiring-by-name; exported/imported as a versioned JSON schema so external tools can author models without touching Python. Custom functions by registry reference. Time via 2-slice (`lag`) tags, unrolled by the compiler.
    - An overarching `System` class defines both agents and envs — an env *is* a `System` (anything on either side of a Markov blanket), nestable/composable; the outermost `System` boundary interfaces with external environments (the real world, simulators, other software)

- **Backend/inference** - Should enable the usual flavors of VI but leave user free to use other techniques including efficient sampling — [NumPyro](https://github.com/pyro-ppl/numpyro) as a sampling backend where it fits (NUTS on exact likelihoods; note NumPyro provides no model representation of its own).
    - Closed-form updates on genuine conjugate pairs (Categorical–Dirichlet, Linear-Gaussian–NIW, GMM via indicator CAVI, MNLR via Pólya-Gamma-augmented CAVI — a pure `tanh`+Gaussian update, no sampling); never *assume* conjugacy in the architecture
    - Mixture beliefs propagate with static component counts: exact grow, then Runnalls/IMM reduction; mixture-entropy EFE terms via closed-form differentiable bounds (see `docs/de-risk-report.md`)
    - Allow custom functions to be used for any map (e.g. likelihood, transition), with quadrature/RTS-style Gaussian message rules for deterministic nonlinear nodes (Petersen et al. 2018; transitions so far: `NonlinearGaussianTransition`, first-order or unscented, since 0.14)
    - Interacting state factors supported (as in `pymdp`-jax `B_dependencies`), not just per-factor dynamics

- **Metrics** - Quantities like expected free energy as well as its risk/epistemic breakdown, per-variable/per-policy, available on demand — the objective layer returns structured term trees, with estimator bounds surfaced (e.g. entropy estimates as lower/upper pairs)

- **Deployment-aware** - The three loops (state inference / policy search / learning) exchange only serializable sufficient statistics — belief snapshots, parameter pytrees, policy posteriors, the versioned model spec — so every deployment cut is available without API change: fully fused in one jit'd program (the default in simulation), or split across compute tiers (edge / near-edge / cloud), where more capable tiers can absorb lower loops subject to control-loop deadlines. End-to-end gradients through planning require colocation. Expectations route through pluggable sampler backends (default PRNG Monte Carlo), so external samplers — including energy-efficient thermodynamic hardware — slot in behind one protocol without touching the engine. Teacher–student distillation (heavy planner → light amortized policy) is the supported path for moving capability down-tier; see `docs/project-brief.md` for the tier mapping.

## Scope

### Must include

- Mixture model support (GMM and MNLR nodes), including mixture-belief rollouts for planning
- Core active inference functions (ablated -> model-based RL)
- Continuous and discrete nodes
- On-demand EFE/VFE metrics with term breakdowns

### Out of scope

- Fully general message-passing / factor graph engine — i.e., closed-form inference for arbitrary factor pairings. The node menu is deliberately closed; escape hatches are custom update functions and the sampling backend.
- Giving arbitrary pretrained ML models *inference semantics* inside the engine (inverting, training, or message-passing through them). Frozen models are supported as **transducers**: sensory boundary components declared in the model spec (registry reference + declared output shape + declared uncertainty model), executed by the `System` runtime at the tick boundary — e.g. a frozen V-JEPA-class encoder feeding a latent pseudo-observation modality, or a frozen predictor wrapped as a custom transition map. A small blessed adapter set ships with the library; the adapter contract is library-owned (the format-registry pattern applied to perception).

### v0.1 target

Categorical node at parity with `pymdp`-jax on T-maze (their test suite as oracle) + Linear-Gaussian node on a mountain-car-class task + one GMM demo; single explicit planner; metrics tree exposed.

## Stack choices (tentative)

Equinox for the module system (per `pymdp`-jax precedent), optax for non-conjugate learning, NumPyro for sampling. Data pipeline behind a format registry (HDF5 and Lance backends, per `stable-worldmodel`'s design).

## Reference designs

Reference repos made available in `/reference` (note: not committed on remote; clone if needed)

- `pymdp` (JAX): efficient multi-agent policy search in JAX for discrete models. We build on its skeleton (Agent-as-pytree, rollout/scan machinery, planner hook) and extend the model-specification layer to continuous and hybrid nodes. [repo](https://github.com/infer-actively/pymdp.git)
- `stable-worldmodel`: reference for the data pipeline (format registry, Lance schema, replay buffer, goal relabeling) and solver protocol layering; not for vectorization. [repo](https://github.com/galilai-group/stable-worldmodel.git)
- `pgmax`: pattern reference for discrete compute kernels (tempered log-space numerics, scan discipline); dense batched einsum is preferred where model shapes are homogeneous. Dormant; reference only. [repo](https://github.com/google-deepmind/PGMax.git)

## Bibliography

References, with links:

- [arXiv:2201.03904](https://arxiv.org/abs/2201.03904) - PyMDP library spec
- [arXiv:2202.04110](https://arxiv.org/abs/2202.04110) - `pgmax` library whitepaper
- [arXiv:2409.14216](https://arxiv.org/abs/2409.14216) - R-AIF: Practical implementation of active inference with continuous variables and learned preferences
- [arXiv:2603.19312](https://arxiv.org/abs/2603.19312) - LeWorldModel: compact JEPA (latent-space-prediction objective) world modeling
- [arXiv:2602.08968](https://arxiv.org/abs/2602.08968) - `stable-worldmodel` reference doc
- [doi:10.1080/17588928.2015.1020053](https://doi.org/10.1080/17588928.2015.1020053) - Classic theory paper on epistemic value in active inference
- [doi:10.3390/e23070807](https://doi.org/10.3390/e23070807) - comprehensive approach to deriving message passing algorithms from constraints on factor graphs
- [doi:10.1162/neco_a_01341](https://doi.org/10.1162/neco_a_01341) - affective inference (contains the $\gamma$ proposal)
- [arXiv:1903.09136](https://arxiv.org/abs/1903.09136) - Gaussian approximations for deterministic non-linear transformations (useful for using arbitrary functions in factor graphs); its forward rule is `NonlinearGaussianTransition`'s unscented option, its backward (smoothing) rule the reference for a continuous smoother
- [arXiv:2607.13612](https://arxiv.org/abs/2607.13612) - SIGReg-as-VFE: interprets the SIGReg regularizer as variational free energy — proves JEPA objectives are exact active-inference free energies iff the anti-collapse regulariser is SIGReg, and identifies the state-epistemic term as the one EFE quantity no JEPA world model computes (theory only; empirical side open)
- [arXiv:2312.07547](https://arxiv.org/abs/2312.07547) - Active Inference and Intentional Behaviour (Friston et al., incl. A. Kiefer): the probabilistic grounding of terminal/intended-state priors — `h` as a prior over final states, reciprocal to `d`; capital-`H` as the derived inductive policy prior (backward reachability through a precision-thresholded `B`, `P(u) = σ(−G − H)`); explicitly charters conditioning `h` on a supraordinate hierarchical level and learning it via Dirichlet counts — the design basis for `H_deps` target emission in nesting (see `docs/design/nesting-systems.md` addendum)
- [doi:10.1038/s41467-024-49711-1](https://doi.org/10.1038/s41467-024-49711-1) - Maximum occupancy principle (Ramírez-Ruiz et al., *Nat Commun* 2024): behaviour as maximization of discounted future action-state *path entropy*, `V(s) = E[Σ γ^t (α H(A|s_t) + β H(S'|s_t,a_t))]`, with the axiomatic result that weighted path entropy is the only occupancy measure consistent with per-step additivity, positivity and smoothness; positions itself against FEP and empowerment (both argued to collapse toward deterministic policies). Its `β` term is the `occupancy` row of our term lattice — and is exactly R-AIF's sign-flipped state-entropy term `wment` (see `docs/design/objective-terms.md`)

- [arXiv:2006.04120](https://arxiv.org/abs/2006.04120) - Sophisticated Inference (Friston, Da Costa, Hafner, Hesp & Parr; *Neural Computation* 33(3):713–763, 2021; doi:10.1162/neco_a_01351): planning as a recursion over what the agent will *observe* — the value of an action is its own expected free energy plus the expectation, over the outcomes it may produce, of the softmax-weighted value of the next decision — so information is worth what it lets the agent do next, not only what it reduces. The `"sophisticated"` planner (`planning/sophisticated.py`, over `System.branch_on` modalities) is this recursion, exact and unpruned; the paper's belief-based pruning and pymdp's budgeted tree are the road on from it
- [doi:10.1016/j.biopsycho.2024.108891](https://doi.org/10.1016/j.biopsycho.2024.108891) - Supervised structure learning (Friston, Da Costa, Tschantz, Kiefer, Salvatori, Neacsu, Koudahl, Heins, Sajid, Markovic, Parr, Verbelen & Buckley; *Biological Psychology* 193:108891, 2024): structure learning of discrete generative models by Bayesian model selection under expected-free-energy priors, in a POMDP generalised with *paths* as latent variables — a control variable read as "a path that may or may not be controllable", and a policy as "a combination of paths". The grounding of `docs/design/control-states.md` (control states as ordinary nodes; path dynamics as a control node's transition prior)
- [arXiv:2505.24784](https://arxiv.org/abs/2505.24784) (not in `/papers`; the PDF is 36 MB) - AXIOM (Heins, Van de Maele, Tschantz et al., VERSES, 2025): object-centric world models that grow — every module a conjugate mixture whose components are added when a datum is explained by none of them better than by an empty one (Algorithm 1) and merged by Bayesian model reduction when the merged model's evidence is no lower (Algorithm 2); planning by sampled policies under utility and Dirichlet-parameter information gain. The source of `docs/design/structure-learning.md`: its growth and merge rules are stated there for any categorical factor with conjugate emissions, its recurrent mixture is the customer for a modality that reads a previous-tick belief, and its pixel layer is a separate roadmap row. Academic-license code, not read; ideas only
- [arXiv:2212.01508](https://arxiv.org/abs/2212.01508) - Space is a latent sequence (Raju, Guntupalli, Zhou, Lázaro-Gredilla & George, 2022): clone-structured cognitive graphs — a discrete state cloned so that the same observation can mean different places in a sequence, learned by Baum–Welch and read out by Viterbi over the clone graph — as a unified account of hippocampal representations. On the roadmap after structure learning: a clone is a level, and the tools are the sequence-learning pair over it

Additional key references (Pólya-Gamma, mixture reduction, entropy bounds, hybrid-AIF prior art) are collected at the end of `docs/de-risk-report.md`.

## Acknowledgements

hybrid-aif's discrete engine owes its design to [pymdp](https://github.com/infer-actively/pymdp) (Heins, Millidge, Demekas, Klein, Friston, Couzin & Tschantz, *JOSS* 7(73):4098, 2022, doi:10.21105/joss.04098), and especially to its JAX rewrite by Conor Heins, Alexander Tschantz, Tim Verbelen, Dimitrije Markovic and the contributors listed in its README. pymdp is MIT-licensed, © 2019 Conor Heins and Alec Tschantz. Several pieces are adapted from it directly: the exact Dirichlet novelty weights (`maths._exact_wnorm`), the factorised fixed-point iteration (`algos.run_factorized_fpi`), policy enumeration (`control.construct_policies`) and the classic T-maze tensors (`envs.TMaze`). Others are reimplemented from its specification with pymdp's own functions as our parity oracle: expected states and observations, utility, state information gain and the A/B parameter information gain (`control.py`); the policy posterior, action marginals and action and policy sampling with its `alpha` precision; and the Dirichlet count updates (`learning.py`). We also took its architecture: the agent as a pytree with static configuration and vmapped agents, `A_dependencies`/`B_dependencies` wiring and the A/B/C/D/E roles, the `infer_states`/`infer_policies`/`update_empirical_prior` faces, the scan-over-horizon and vmap-over-policies evaluation, the functional `Env`, and the masked fixed-window rollout history. We are grateful to its authors for building it, by hand and in the open.

Much of that design, and pymdp's, descends from the active-inference routines of [SPM](https://www.fil.ion.ucl.ac.uk/spm/) (Statistical Parametric Mapping; Tierney et al., *SPM 25: open source neuroimaging analysis software*, *JOSS* 10(110):8103, 2025, doi:10.21105/joss.08103), written chiefly by Karl Friston in its DEM toolbox. We follow its `spm_MDP_VB_X`, `spm_MDP_VB_XX` and `spm_MDP_VB_XXX` for the exact Dirichlet novelty weights (`spm_wnorm`) and their masked, product-of-marginals average, the risk-plus-ambiguity form of expected free energy, the `alpha`-sharpened action marginal, the sophisticated recursion (`spm_forwards`) and priors over final states (`H`), and its `spm_MDP_VB_sleep` and `spm_MDP_log_evidence` for the idea of Bayesian model reduction over Dirichlet counts.
