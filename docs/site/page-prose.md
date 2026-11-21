---
title: hybrid-aif
description: hybrid-aif is a JAX library for active inference over models that mix discrete and continuous state, as well as closed-form updates and amortized inference. This demo page replays precomputed runs of two variants of a solar mapping rover model.
---

<!--
The text of the project page. Edit here, then run `python docs/site/build_page.py` from the
repository root; it regenerates index.html. Do not edit index.html by hand.

Conventions:

- Front matter: `title` is the browser tab title, `description` the search-engine summary.
- The `#` title line and everything up to the first `##` heading is the page header; the title
  is the `title-image` block (the logo in static/, with its alt text). The marker
  `tldr` comment opens the boxed TLDR; the header's paragraphs and list after it go in the box.
- `## Heading {#id}` starts a section; the id is used by the page's script and the URL. Keep it.
  `### Heading` is a subheading inside a section.
- Every paragraph inside a section is a lead paragraph (the page's body text style). Paragraphs in
  the header and footer are plain. A line `{.name}` directly above a paragraph gives it the class
  `name` instead (the header uses `{.sub}` and `{.byline}`).
- A `block: NAME` comment on a line of its own stands for a piece of HTML kept in
  page-blocks.html under that name (viewer panels, controls, figures, code listings, the page
  head and the noscript fallback). Inside a paragraph it is pasted inline (the version number,
  the dynamics rule the script fills in).
- Lines starting with `>` right after a block are that block's caption.
- The `footer` comment ends the sections; the paragraph after it is the page footer.
- Inline: **strong**, *emphasis*, `code`, [link text](url). Type the typographic apostrophe and
  other special characters directly; the build writes them as HTML entities. A plain `'` stays
  plain. `\[` and `\]` are literal brackets, for brackets around a link.
- Line breaks inside a paragraph do not matter; a blank line ends it.
- Per-version text: a line `<!-- version: v1 -->` opens a span shown only while the page shows
  version v1 of the replay (v2 likewise), and a line `<!-- /version -->` closes it. Everything
  outside a span is shared by both versions. A span stays inside one section: put no `#` or `##`
  heading, no `tldr`/`footer` marker and not the demo's viewer block inside it, and close it
  before the next section. A `###` inside a span keeps its contents entry in both versions.
  Without JavaScript every span shows.
-->

# <!-- block: title-image -->

{.sub}
Continuous-discrete-amortized active inference in JAX

{.byline}
Noumenal Labs · <!-- block: version --> · MIT · \[[github](https://github.com/noumenal-labs/hybrid-aif)\]

<!-- block: links -->

<!-- tldr -->

`hybrid-aif` is a Python/JAX library for production-grade active inference and probabilistic
machine learning. It supports inference, learning, and planning on models with heterogeneous 
pieces: continuous, discrete, and mixture factors as well as amortized functions (e.g. neural 
nets). Models and objectives can be specified flexibly and come with detailed model comparison 
and logging tools.

{.demo}
**Demo:** [Solar rover](#demo)

**Core features:**

- **Factor-graph primitives beyond the exponential family.** Core distributions like conditional 
  categoricals and Gaussians, as well as various types of mixture model (e.g. Gaussian mixtures, 
  switching linear dynamical systems, and mixture dynamics), but also user-defined nonlinear and
  deterministic functions and amortized neural factors.

- **Hybrid inference engine.** Generative models are compiled and differentiable using JAX, 
  and closed-form conjugate message-passing can run alongside gradient- and sample-based 
  inference and learning in the same model.

- **Flexible generative model declaration.** Control states and parameters are first-class 
  variables in a factor graph, and can thus be conditioned on other variables or used as 
  targets for preferences or terminal-state constraints. Latent control states and realized
  actions are distinct variables in the model, linked via actuator kernels.
  
- **Compositionality.** Agents, environments, and their interactions are described as nested 
  physical systems using a single underlying `System` primitive, so hierarchical and nested 
  systems are expressed in the same language as coupled agent-environment interactions.

- **Fine-grained objective specification.** The objective for planning is standard expected free 
  energy by default, but information gain and parameter novelty terms can be ablated per-variable, 
  and a handful of term primitives can be used to express AIF-adjacent objectives such as 
  maximum-occupancy and others (on the roadmap: empowerment).

- **Runtime enabling parallelism and scheduling.** Simulations can be duplicated along a batch 
  dimension. Layers in hierarchical and other multi-component systems can run at different 
  relative speeds. User-defined update schedules with sensible defaults allow both synchronous
  (Jacobi) and fixed-schedule (Gibbs) or random (Glauber-style) sequential dynamics.
  
- **Model description and description length.** Each model can be expressed as a document, which
  can be assigned a Shannon description length (~= free energy), on its own or with a data-
  likelihood component given some data, so that models can be compared by the minimum description 
  length principle. This facilitates using coding agents to search over models.

- **Detailed logging.** Term-by-term, per-factor, per-policy objective function records 
  can be returned at each step, or turned off for a slight gain in speed. 

### Related work

`hybrid-aif` owes its discrete core to [pymdp](https://github.com/infer-actively/pymdp), the reference implementation of active inference for discrete state spaces in JAX, which in turn ports many functions from Karl Friston's [SPM](https://www.fil.ion.ucl.ac.uk/spm/software/) software. We aim to extend the core idea—of posterior inference over policies using an expected-free-energy objective—natively to continuous and mixed states, to learned and amortized components, and to nested and hierarchical systems. 

At the same time, we are inspired by the elegance of factor-graph formulations of probabilistic inference as in [RxInfer.jl](https://rxinfer.com/), a reactive message-passing engine for factor graphs in Julia, whose authors have shown that expected-free-energy-based planning can be accomplished purely via message passing. `hybrid-aif`'s design philosophy treats the active inference API as an abstraction over a more generic factor-graph layer, in which algorithms for explicit counterfactual policy search (enumeration, sophisticated inference) are optional planners.

Our development of a new software package was based on the need both to develop active inference models in a research
setting and to embed them in larger machine-learning systems. A single canonical model description facilitates the
port from research into production code, and also potentially allows one to use alternate backends for models 
specified in this common language (in particular, `pymdp` and `RxInfer` backend support via a thin translation layer
are on the roadmap).

## Solar rover demo {#demo}

### Overview

In this demo, a solar-powered rover builds a map of a 2.5D environment (a field with continuously 
varying height), classifying terrain into three kinds (firm, rough, and ice). The rover's battery
charges in sunlight, which depends on which slopes are shaded. For navigation it relies on a forward 
camera and a GPS-like satellite signal whose quality varies over the field. The demo shows two 
variants: one with just the rover and one a hierarchical model whose second layer is a discrete 
mission planner that provides options to map a given coarse-grained sector, or to charge the battery
by heading to one of three locations with optimal sunlight.

<!-- version: v1 -->

The page replays one run of each setting (scheduler + rover, and rover alone),
seed 4 of 5:
`python examples/survey_run.py --ticks 800 --replicas 5 --settings two-level --out FIVE` and the
same with `--settings one-level`, then
`python examples/survey_run.py --assemble --version v1 --seeds-dir FIVE --out OUT`, then
`python examples/survey_trim.py OUT/bundle_v1.json docs/site --name HYBRID_AIF_BUNDLE_V1 --file bundle_v1.js`;
`docs/site/README.md` explains how the page's data is rebuilt from it.

<!-- /version -->

<!-- version: v2 -->

The page replays one run of each setting (two rovers coordinated via a scheduler, or via a shared map),
seed 3 of 5:
`python examples/survey_run.py --ticks 800 --replicas 5 --settings two-rover-two-level --out FIVE`
and the same with `--settings two-rover-one-level`, `--settings one-level` and
`--settings one-level-second-start` (the two one-rover runs pooled in the comparison), then
`python examples/survey_run.py --assemble --version v2 --seeds-dir FIVE --out OUT`, then
`python examples/survey_trim.py OUT/bundle_v2.json docs/site --name HYBRID_AIF_BUNDLE_V2 --file bundle_v2.js`;
`docs/site/README.md` explains how the page's data is rebuilt from it.

<!-- /version -->

<!-- block: hero-viewer -->

<!-- version: v1 -->

> **Left:** Contour map, with rover position/heading beliefs, satellite fixes, and patches colored 
> by accumulated certainty about terrain class. The chosen plan's path is drawn in red, alternate 
> plans thinner in the text color. Plans are rolled forward by the same nonlinear functions the 
> generative process uses (see [World](#world) below).

> **Right:** the model, with the table
> `rover.field` read by `rover.terrain` in the fast tier and by `mission.map` in the slow one. This
> Dirichlet table is one learned belief shared by the two tiers. The belief over position is a Gaussian, 
> drawn as its 2σ ellipse with a minimum size so it stays visible; the camera and the fix together keep 
> it small. `energy_table` is the mission's energy-band transition model, learned online.

<!-- /version -->

<!-- version: v2 -->

> **Left:** Contour map, with each rover's position/heading beliefs, satellite fixes and true path, 
> and patches colored by the shared map's accumulated certainty about terrain class. A rover's chosen 
> plan is drawn bold in its color, alternate plans thinner. Plans are rolled forward by the same nonlinear 
> functions the generative process uses (see [World](#world) below). Suns mark charging sites.

> **Right:** the model: the mission tier in the middle, with the rover models on each side. The Dirichlet
> table `mission.field` is a single map read by both rovers' `terrain` factors and by `map_1` and `map_2` 
> in the mission tier. The mission level has one option node per rover, `u_option_1` and `u_option_2`. The 
> belief over position is a Gaussian, drawn as its 2σ ellipse with a minimum size so it stays visible;
> the camera and the fix together keep it small. `energy_table` is the mission's energy-band transition 
> model, learned online from both rovers.

<!-- /version -->

### Model \[[code](https://github.com/noumenal-labs/hybrid-aif/blob/main/examples/survey_run.py)\]

#### Rover tier \[[code](https://github.com/noumenal-labs/hybrid-aif/blob/main/examples/survey_model.py)\]

<!-- version: v1 -->

- **Sensation.** The vehicle's position is inferred via Bayesian fusion of the satellite `fix` with the 
camera image (`img`), whose neural likelihood is inverted during inference. The camera also informs a heading 
estimate, and the rover's beliefs about terrain classes. The rover senses its battery level via a 
continuous gauge.

- **Action.** The rover chooses continuous two-dimensional (target speed, turn rate) commands `v`.

- **Inference and learning.** Continuous state appears in the generative model as a six-dimensional 
`x` (2D position, 2D heading vector, speed, energy). The terrain map is a table of Dirichlet counts 
`rover.field`, whose cells are updated in proportion to the rover's discretized position belief, according 
to the inferred `terrain` classes.

- **Objectives.** The rover has a continuous preference for its energy `gauge` to be more full. It
also has a preference to stay within the square region of the map, implemented via a Gaussian
mixture over the GPS modality. When driven by the mission planner (see below), the rover has either 
a broad preference to be in a given quadrant of the map for surveying, or a precise one to drive to a 
good charging location and wait.

  Exploration is driven primarily by novelty on the Dirichlet counts, and also by infogain about
  position from GPS, which drives the rover to find spots good for localizing. The camera infogain 
  is left out: in trials the pose information in its pixels outweighed the rest of the objective 
  and drove the rover to the walls, which are the simulation's most visually distinguishable landmarks.

- **Planning.** Planning is via gradient descent on the rover level's expected free energy, from
four starts and over a horizon of 12 discrete time-steps.

<!-- /version -->

<!-- version: v2 -->

- **Sensation.** A vehicle's position is inferred via Bayesian fusion of the satellite `fix` with its 
camera image (`img`), whose neural likelihood is inverted during inference. The camera also informs a heading 
estimate, and the rover's beliefs about terrain classes. The rover senses its battery level via a 
continuous gauge.

- **Action.** The rover chooses continuous two-dimensional (target speed, turn rate) commands `v`.

- **Inference and learning.** Continuous state appears in the generative model as a six-dimensional 
`x` (2D position, 2D heading vector, speed, energy). The terrain map is a table of Dirichlet counts 
`mission.field` shared across both rovers, whose cells are updated in proportion to each rover's 
discretized position belief, according to the inferred `terrain` classes.

- **Objectives.** Each rover has a continuous preference for its energy `gauge` to be more full. It
also has a preference to stay within the square region of the map, implemented via a Gaussian
mixture over the GPS modality. When driven by the mission planner (see below), the rover has either 
a broad preference to be in a given quadrant of the map for surveying, or a precise one to drive to a 
good charging location and wait.

  Exploration is driven primarily by novelty on the Dirichlet counts, and also by infogain about
  position from GPS, which drives the rover to find spots good for localizing. The camera infogain 
  is left out: in trials the pose information in its pixels outweighed the rest of the objective 
  and drove the rover to the walls, which are the simulation's most visually distinguishable landmarks.

- **Planning.** Planning is via gradient descent on the rover level's expected free energy, from
four starts and over a horizon of 12 discrete time-steps, separately for each rover.

<!-- /version -->

#### Mission tier \[[code](https://github.com/noumenal-labs/hybrid-aif/blob/main/examples/survey_scheduler.py)\]

<!-- version: v1 -->

- **Sensation.** The mission tier observes a discretized (8-level) `gauge` reading of the rover's 
`energy`, as well as the rover's believed position, mapped to a 16 x 16 cell grid over the landscape.

- **Action.** Options `u_option` are to survey one of the quadrants of the map, or to head for 
a good charging location. The mission planner's actions accomplish this by setting the rover 
tier's state preferences (effectively, location and speed) for the next 24 ticks at its timescale.

- **Inference and learning.** The rover's discrete location `visit` is inferred from an
observation `cell_obs` that maps the rover's believed location to a grid. The Dirichlet table 
filled out by the rover's survey is a parameter shared by the higher-level model, which then 
informs option selection. The mission tier also learns a transition model of the rover's energy 
conditional on which option is pursued, starting from weak priors grounded in the rover's 
machinery.

- **Objectives.** Option choice is driven by expected novelty (parameter infogain) about the 
map cells, aggregated over the number of distinct cells the rover will visit under the option 
(measured empirically), and by a terminal-state prior that discourages ending an option with 
an empty battery. There's no infogain about location or battery level at this tier.

- **Planning.** Planning at the mission tier is "vanilla" expected-free-energy based policy 
search with the terms just described, with a horizon of 4 options (= 96 lower-tier steps).

<!-- /version -->

<!-- version: v2 -->

- **Sensation.** For each rover, the mission tier observes a discretized (8-level) `gauge` reading of 
its `energy`, as well as its believed position, mapped to a 16 x 16 cell grid over the landscape.

- **Action.** Options `u_option` (per rover) are to survey one of the quadrants of the map, or to 
head for a good charging location. The mission planner's actions accomplish this by setting the rovers' 
state preferences (effectively, location and speed) for the next 24 ticks at its timescale.

- **Inference and learning.** The rovers' discrete locations `visit_1` and `visit_2` are inferred 
from `cell_obs` modalities that map a rover's believed location to a grid. The Dirichlet table filled 
out by the rovers' survey is a parameter shared by the higher-level model, which then informs option 
selection. The mission tier also learns a transition model of each rover's energy conditional on which 
option is pursued, from both rovers' readings, starting from weak priors grounded in the rover's 
machinery.

- **Objectives.** Option choice is driven by expected novelty (parameter infogain) about the 
map cells, aggregated over the number of distinct cells a rover will visit under the option 
(measured empirically), and by a terminal-state prior that discourages ending an option with 
an empty battery. Planning is "sophisticated" in that the map is learned within one rollout:
imagined cell readings raise working Dirichlet counts, so later readings of that cell earn
less novelty. There's no infogain about location or battery level at this tier.

- **Planning.** The mission tier plans with a factorised expected-free-energy search: one 
policy posterior per rover over that rover's option sequences, each updated against the other 
rover's per-step option marginals by three mean-field iterations, with a horizon of 3 options 
(= 72 lower-tier steps). Each round thus scores 2 × 343 sequences three times, rather than 
the 117,649 joint sequences of the product space. The rovers are coupled only through the 
shared map, which the plan learns as it goes, so cells one rover's plan would read lose novelty 
for the other, and indirectly through the shared energy table.


<!-- /version -->

### World \[[code](https://github.com/noumenal-labs/hybrid-aif/blob/main/examples/survey_world.py)\]

<!-- version: v1 -->

The generative process of the rover's environment steps the rover's state forward from its current state and command, via the following functions:

<!-- /version -->

<!-- version: v2 -->

The world has no rover-to-rover physics. The generative process steps each rover's state forward from its current state and command via the following functions:

<!-- /version -->

The command is a drive $d \in [0, 1]$ (ratio of target to top speed) and a
turn $u \in [-1, 1]$. A stall gate $g = \sigma\big((\ell - \ell_{\mathrm{stall}}) / w\big)$ damps the drive
and the turn as energy reserves begin to run low, with current energy $\ell$, soft stall threshold $\ell_{\mathrm{stall}}$, 
and spread $w$. This is near one above the stall level and near zero near empty. Then
$$\mathbf{p}_{t+1} = \mathbf{p}_t + s_t \, (\cos\theta_t, \sin\theta_t), \qquad
\theta_{t+1} = \theta_t + \tau_{\max} \, g \, u, \qquad
s_{t+1} = \kappa \, s_t + a \, g \, d,$$

with position $\mathbf{p}$, heading angle $\theta$, forward speed $s$, battery level $\ell$ (0 to 4),
max turn rate $\tau_{\max}$ = 20 degrees per tick, $\kappa = 0.5$ speed damping and $a$ speed gain.
Under a constant command the forward speed settles toward $s_{\mathrm{top}} \, g \, d$ with $s_{\mathrm{top}} = a / (1 - \kappa)$. 
The energy level is updated as:

$$\ell_{t+1} = \ell_t + r \, y(\mathbf{p}_t)\big[\mu + (1 - \mu)\, e^{-(s_t / s_{\mathrm{park}})^2}\big]
- \big(\beta + (\delta + \delta_{\uparrow} \, c^{+}_t)\, s_t\big),$$

The charge is the solar yield $y$ of the ground under the rover (sun-facing slopes yield several
times what level ground does, and shade almost nothing), times a rate $r = 0.07$ per tick, collected
in full at rest and only a tenth of it ($\mu = 0.1$) once the speed passes the parking speed
$s_{\mathrm{park}} = 0.05$. The drain has three parts: $\beta = 0.01$ per tick whatever the
rover does, $\delta = 0.045$ per unit of speed, and $\delta_{\uparrow} = 1.5\,\delta$ per unit
of speed per unit of uphill climb $c^{+}_t$, the slope along the heading smoothed so it is
about zero going down or across and about the slope going up. The level is capped at 4. 
Gaussian noise of a fixed sd per coordinate is added, and the world clips the position to the field. 
Walls prevent the rover from leaving the field. The rover's model doesn't know this but its GPS
preference keeps it inside the walls. The rover does model stalls due to battery depletion.

## Performance comparison {#timescales .sub}

<!-- version: v1 -->

Both runs compared here use the same rover, model and world. In the rover-only condition there is no 
preferred sector. Coverage is defined as the fraction of cells whose maximum posterior class probability
is above 0.6. After a small early lead for the mission-assisted run, both runs end at similar coverage 
in the run shown (seed 4). The mission tier decides to charge about 1/3 of the time, and spends most 
time surveying sector 3. Over five seeds the mission planner ends at 0.657 against 0.670 alone, having led 
at tick 400 on every seed and on the mean at every checkpoint before 800.

<!-- /version -->

<!-- version: v2 -->

This demo has three arms. **Two rovers, one map, a mission tier over both:** two rovers count
into one map, and one mission tier chooses an option for each rover every 24 ticks. **Two rovers,
one map, no mission tier:** the same two rovers, each driven by its own objective, count into one 
shared map, with no scheduler. **Two rovers alone, the maps pooled afterwards:** two one-rover runs, 
one from each start, each with its own map; a cell counts as mapped where either rover's own map is 
confident of it.

The two rovers read one map: the parameter variable `mission.field` is read by both rovers'
terrain factors and by the scheduler's two map factors. They start at different positions mirrored 
through the field's centre. The rovers do not observe each other, and the world has no collision 
between them. The scheduler has two option nodes, `u_option_1` and `u_option_2`, one per rover, 
each with seven options: a survey of each of the four sectors, and a charge at each of three sites, 
one per sunny region. It plans with the factorised planner: one posterior per rover over that rover's 
343 sequences of three options, each updated against the other rover's option marginals by three 
mean-field sweeps. One energy table, next band × band × cell × option, is shared by the two rovers.

<!-- block: seed-table -->

<!-- block: comparison-figure -->

<!-- /version -->

<!-- block: timeline-figure -->
> Top to bottom: the option in force (a tick above the band marks each decision, every 24
> ticks), the energy level against its eight bands, and coverage over the run, solid for the selected
> setting and dashed for the other.

<!-- block: decision-figure -->

## The camera \[[code](https://github.com/noumenal-labs/hybrid-aif/blob/main/examples/survey_camera.py)\] {#camera .sub}

<!-- version: v1 -->

The rover's forward camera is a 32 by 32 grayscale image along the current actual heading direction.
The panels show it at 6x scale. The rover's model includes a neural likelihood for the camera image,
a 4-layer MLP that predicts the picture from the rover's (position, heading) pose and the inferred 
ground type. Inference inverts the likelihood given the empirical prior over state, linearizing around 
the belief's mean.

<!-- /version -->

<!-- version: v2 -->

Each rover's forward camera is a 32 by 32 grayscale image along that rover's current actual heading.
The panels show it at 4x scale. The rover's model includes a neural likelihood for the camera image,
a 4-layer MLP that predicts the picture from the rover's (position, heading) pose and the inferred 
ground type. Inference inverts the likelihood given the empirical prior over state, linearizing around 
the belief's mean.

<!-- /version -->

<!-- block: camera-figure -->
> **Observed image:** what the camera sees now. **Decoder prediction:** the image the model expects at 
> the current belief. **Terrain readout:** the rover's belief over the three kinds of ground, with the 
> true class marked with a dot. Brightness of a ground point depends on how much sun reaches it, taking
> slopes into account. Terrain classes differ by ground texture, and the ground under the rover is visible; 
> the ground beyond the field's walls is dark, and far ground fades toward a gray haze.

## Term-by-term objectives {#objective .sub}

The objective is the sum of the risk and information gain terms described earlier, and every score is 
reported as an interval: the bar spans the lower and upper bound and the mark is the midpoint, which 
the planner ranks on. Exact terms are points. A term over a mixture carries a bracket from closed-form 
entropy bounds; in this demo that is the wall preference on the fix. The rover's terms are marked 
approximate because its plans use a first-order approximation of beliefs pushed through the vehicle's 
nonlinear motion. Below are the rover's gradient planner and the mission tier's exact enumeration of 
option sequences.

<!-- block: objective-viewer -->

## Description length {#mdl .sub}

What each part of the model costs to write down, in nats, grouped by tier. The description length is
the model's own complexity measure, computed from the same declaration that runs the agent. A
component declared as a fixed table, density or network is priced by its family alone and flagged
approximate.

<!-- block: mdl-table -->

## Install and quickstart {#start}

<!-- block: install-code -->

**Quickstart: Mountain Car.** Mountain Car in its linear regime: a cart climbing to a hilltop
it cannot rest on, with three thrust levels, as one declaration. The linear-Gaussian dynamics and
the position sensor are the model, the hilltop is a target on the state, and the objective prices
the reading's information gain and the uncertainty a plan injects. The same declaration run forward (`as_env`) is the
world, so the loop below needs nothing but the package: each tick the agent filters the reading,
plans over the 81 four-tick thrust sequences, acts and predicts its next prior. It reaches the
top in about two seconds, printing the lines under the code. The same agent, with plots and the
objective's term tree, opens the [demo notebook](https://github.com/noumenal-labs/hybrid-aif/blob/main/examples/demo.ipynb).

<!-- block: scheduler-code -->

<!-- block: run-code -->

<!-- footer -->

Noumenal Labs, 2026. Most of `hybrid-aif`'s code was written with Claude under the direction of the maintainers; 
the design records in [docs/design/](https://github.com/noumenal-labs/hybrid-aif/tree/main/docs/design) show how 
each decision was reached. Licensed MIT.
