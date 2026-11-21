# The project page

A static page that replays precomputed runs of the survey rover showcase: a mapping rover with
solar charging, a fast rover tier and a slow survey scheduler that read one learned terrain
table; in its second version, two such rovers reading one field under one scheduler. GitHub Pages serves `docs/`; `docs/index.html` redirects here.

Files:

- `index.html`, `style.css`, `app.js`: the page and its plain-JS viewer (no build step, no
  dependencies); `nav.js` switches the page's panes.
- `bundle_v1.js`, `bundle_v2.js`: the replay data of the demo's current version (v2, two rovers)
  and of its earlier one (v1, one rover), each one JavaScript file defining
  `window.HYBRID_AIF_BUNDLE_V2` / `window.HYBRID_AIF_BUNDLE_V1`. The page reads as one demo: the
  only place it names a version is the switch in the sticky demo bar, to the right of the demo's
  title ("version:", options "v2 (latest)", the default, and "v1 (one rover)"; part of the
  `demo-bar` block, so it is present on the demo pane and its child panes). Choosing one loads its file
  on demand, once, by a script tag; the first paint waits for it, and the status line names the
  option while it loads ("Loading the replay data: v1 (one rover)…"). The text
  `app.js` writes into the page body names no version (`meta.version` and `meta.version_label`
  are not shown).
- `static/`: rendered views for readers without JavaScript.

## Layout

A fixed sidebar (a top bar with a menu button below 900 px wide) holds a logo (the
`sidebar-logo` block), the table of contents and the small print (version, licence, links); beside
it the page shows one pane at a time. The Intro pane is everything in `page-prose.md` before the
first `##`, headed by the logo (the `title-image` block); each `## Heading {#id}` section is a
pane with that id, and its `###` subheadings are listed under it in the contents. A section
written `## Heading {#id .sub}` is a child pane, listed under the nearest section before it
without `.sub`; while a child is open its parent's entry is marked open. The demo (the section that
places the `hero-viewer` block) and its children share a bar that sticks to the top of the column:
the demo's heading and the playback controls (the `demo-bar` block). The URL hash selects the pane: `#id` opens a pane, the id of a subheading opens its pane
and scrolls to it, and an empty or unknown hash opens `#intro`; a link in the prose to `#id`
therefore switches panes. The last pane, Documentation (`#documentation`), is added by the build
from the `documentation` block in `page-blocks.html` until the markdown has a section with that id.
`nav.js` shows the pane and fires `hybrid-aif:pane`, on which `app.js` redraws the viewer at the
pane's size.

## Editing the text

The page's prose lives in `page-prose.md`; `index.html` is generated from it. Edit the markdown,
then run, from the repository root (or from `docs/site` as `python build_page.py`):

```bash
.venv/bin/python docs/site/build_page.py          # regenerate index.html
.venv/bin/python docs/site/build_page.py --check  # exit 1 if index.html is out of date
.venv/bin/python docs/site/build_page.py --prose COPY.md --out OUT.html  # build a draft copy
```

The conventions (section ids, lead paragraphs, captions, placeholders) are listed in a comment at
the top of `page-prose.md`; the build's docstring has the full list. Sections are the panes and
their headings the contents entries, so adding, renaming or reordering a `##` section changes the
sidebar too; `{#id .sub}` (id and class in either order) nests a section under the one before
it, and any other class is a build error. A `###` subheading gets an id from its words, or the one
written as `### Title {#id}`.
Text that differs between the page's versions goes in a version span: a line
`<!-- version: v1 -->` opens a span shown only while the page shows v1 (likewise `v2`), and a line
`<!-- /version -->` closes it; everything outside a span is shared. The build writes the span as
`<div class="ver" data-version="v1">`, and `app.js` sets `hidden` on every span of the other
version, on load (before the bundle arrives) and on every version switch; without JavaScript
(the noscript reader) every span shows. A span stays inside one section: it may not hold a `#` or
`##` heading, the `tldr` or `footer` marker or the demo's `hero-viewer` block, and must be closed
before the next section (the build stops with an error otherwise). A `###` inside a span keeps
its contents entry under both versions. A block's caption can carry spans of its own: a marker
line between groups of `> ` lines opens one, and it must close before the caption ends, so the
viewer block stays shared while a caption paragraph is written per version. The sidebar and pane switching are unaffected: a section's
heading stays outside its spans, so its pane and contents entry exist under both versions. The
Performance comparison pane is the example: its original text is a `v1` span, the two-rover
facts, the five-seed table and the comparison figure a `v2` span, and the timeline and the
decision chart, after both, are shared.
A heading ending in `\[[code](URL)\]` (before any `{...}`) gets a small `[code]` link after its
title, left out of the contents and of its id.
The header's `links` block and the `version` block are shown in the sidebar's small print, and the
TLDR is plain prose. What is not in the markdown lives in `page-blocks.html` as named
blocks, kept verbatim, which the markdown refers to with `<!-- block: NAME -->` placeholders: the
page head (title and description come from the markdown's front matter), the header links, the
viewer panels and their controls and figures, the code listings, the description-length table, the
noscript fallback with its image alt texts, and the script tag. The version number in the byline
is the `version` block. Text the viewer writes at run time (status line, decision and objective
notes and captions, the table footnote) is in `app.js`. The build needs only the standard library and is deterministic; with the sources
as committed it reproduces `index.html` byte for byte.

To regenerate the data (`examples/survey_run.py`'s docstring has the forms): each version takes
five seeds of every arm by one batched run per setting (`--replicas 5`, seeds 0 to 4) into one
directory, FIVE; `--assemble --version` then writes the version's bundle from those part files
(`--seeds-dir`). The replay is a replica from the same batches, of the seed `replay_seed` picks:
among the seeds where every arm's coverage at tick 800 is within one sample sd of that arm's mean,
the seed at which the scheduled arm's (two-level, two-rover-two-level) coverage at 800 is highest
(seed 4 for v1 and seed 3 for v2 in the bundles as committed). No separate replay runs:

```bash
# the five seeds (v1: two-level, one-level; v2: two-rover-two-level, two-rover-one-level, and
# one-level-second-start, which with one-level makes the union arm)
python examples/survey_run.py --out FIVE --ticks 800 --settings two-level --replicas 5
python examples/survey_run.py --out FIVE --ticks 800 --settings one-level --replicas 5
python examples/survey_run.py --out FIVE --ticks 800 --settings two-rover-two-level --replicas 5
python examples/survey_run.py --out FIVE --ticks 800 --settings two-rover-one-level --replicas 5
python examples/survey_run.py --out FIVE --ticks 800 --settings one-level-second-start --replicas 5
# each version: assembled (the replay the rule's seed's replicas) and trimmed
python examples/survey_run.py --out OUT --assemble --version v1 --seeds-dir FIVE
python examples/survey_trim.py OUT/bundle_v1.json docs/site --name HYBRID_AIF_BUNDLE_V1 --file bundle_v1.js
python examples/survey_run.py --out OUT --assemble --version v2 --seeds-dir FIVE
python examples/survey_trim.py OUT/bundle_v2.json docs/site --name HYBRID_AIF_BUNDLE_V2 --file bundle_v2.js
```

`--replay-from unbatched` replays unbatched runs of the rule's seed instead (`--seed K`, one
process per setting, part files in `--out`).

The first call in a directory trains and caches the camera's decoder there. Settings can be run
in parallel, one process each.
`--seed` sets the run key (default 0); `--camera none` and `--no-fix` are the page's two
ablations, and `--camera look` has the rover observe the recogniser's reading instead of the
image. `--level 2` is accepted and is the default (the two settings are the only grid).

Bundle schema notes. Top level: `meta`, `graph`, `settings`, `runs` (v1: `two-level`,
`one-level`; v2: `two-rover-two-level`, `two-rover-one-level`, `two-rover-union`), `mdl`,
`description`. A setting or run marked `coverage_only` (the union arm) is not replayable: the
setting select leaves it out and it appears in the timeline's coverage and the comparison figure
only.

- `meta.version` (`"v1"`, `"v2"`) and `meta.version_label` (a sentence naming the version's
  arms).
- `meta.replay`: `seed` (the seed every setting of the version is replayed at), `rule` (a
  sentence, lower case and without a full stop, the page writes after "The replay is seed K: ":
  how that seed was picked), `rule_arm` (the scheduled arm the rule ranks by), `mark` (the tick
  it reads, 800), `from` (`"replica"`: the seed's replicas from the five-seed batches; or
  `"unbatched"`) and `batch` (`seeds`, the batch's seeds; `replicas`, their number; `seed_arg`,
  the `--seed` the batch was run with).
- `meta.comparison`: `marks` (200, 400, 600, 800), `note` (a sentence on what the numbers are),
  `arms` (in the version's order: `id` (a run's id), `label`, `seeds`, `coverage` (per seed, the
  coverage at the marks), `mean` and `sd` (the sample sd) at the marks). The page draws each arm's
  replayed coverage with its seeds' mean in a band of one sd (the comparison figure) and the
  numbers as a table (`seed-table`), with the replay's seed and rule under it.
- `meta.starts` (v2): the rovers' starts, `[[x, y], [x, y]]`.

- `meta.grid`: the rover's cell grid (`shape` 16 x 16, `low`, `high`); `meta.field` is the field
  extent `[[x0, x1], [y0, y1]]`; `meta.start` the start position.
- `meta.world`: 61 x 61 maps over the field (`xs`, `ys`), each indexed `[ix][iy]`, x first:
  `height`, `light` (illumination), `openness`, `sd_fix` (the satellite fix's noise sd, larger
  where terrain blocks the sky), `slope`, `terrain` (class index per `meta.names.terrain`) and,
  in bundles with a solar battery, `sun` (the solar yield, where parking charges; the viewer's
  illumination layer shows it when present).
- `meta.seed`: the run key every setting of the bundle was run with (null when they differ).
- `meta.names`: `terrain` (firm, rough, ice) and `energy` (the band names, empty to full: eight
  bands of half a unit, `empty`, `0.5-1`, ..., `3-3.5`, `full`; four, empty, low, medium, full,
  in bundles with bands of a unit). The viewer draws as many bands as this list names.
- `meta.tiers`: `slow_ticks` (24), `options` (seven: survey sector 1 to 4, then charge at site 1
  to 3; the page's short labels are S1-S4 and C1-C3, and an option is a charge when its index is
  at least the number of `zones`), `zones`,
  `zone_split` (the field's quadrants are split at these x and y), `sector_centres`,
  `charge_site` (the charging place the charge option drives to and parks at, in bundles whose
  charge option has one fixed place), `charge_sites` (the charging places, one per charge option,
  each a sunny region's sunniest cell; the viewer draws them as stars, C1-C3, in the two-tier
  settings only), `sunny_yield`, `charge_form` (`"per-option"`: a charge option's place is
  `charge_sites[option - len(zones)]`; with it, or `"nearest"`, the viewer never falls back to
  `charge_site`), `charge_to` (a sentence on the per-tick field),
  `survey_readings` and `charge_readings` (the cells a survey's and a charge's pooled reading
  counts), `horizon` (the options the scheduler looks ahead; the viewer reads two when it is
  absent), `terminal_penalty` (the terminal target's penalty on the empty band; absent in bundles
  whose scheduler scored a per-step target over the bands, which the viewer then calls the energy
  risk), `energy_model` and `terminal` (sentences: the learned energy table and its prior, the
  terminal target and how its penalty is set), `cell_reading`, `cell_acc` and `cell_error_sd`
  (the scheduler's cell reading and its local error model), `measured` (the scheduler's constants measured on
  earlier runs: `cruise_speed`, `survey_spread`, `charge_speed`, `survey_readings`,
  `charge_readings`, and a `note`), `one_level` (a sentence: the
  one-level run is the rover alone, with no scheduler and no option). Bundles whose one-level run
  pinned an option carry `pinned_option` instead; the viewer reads either.
- `meta.image`: `side` (32), `layout` (row 0 is the top of the picture), `view` (what the camera
  renders: the scene ahead of the rover along its heading; a comparison camera fixed along +y
  carries `camera_heading_degrees`), `camera_mode`, `heading_rule` (the rover's heading,
  `true_heading` per tick, is part of its state and turns only by the turn command),
  `field_of_view_degrees`, `tempering` (`alpha` and a `note`: the image's evidence is tempered for
  the decoder's own error, its noise sd the calibrated sd over sqrt(alpha)) and `camera_report`
  (the decoder's noise calibration and the tempering's measurement). `meta.map_layout` describes
  `map` and `counts`.
- `meta.wall_preference`: `offset` and `note`: the rover's wall preference leaf (`utility` on the
  fix) and each rover plan's `G` are recorded less `offset`, the preference's constant value over
  a plan inside the field. `meta.position_risk`: `note`: the rover's risk leaf on its position
  and each plan's `G` are also recorded less a constant per tick, the smallest value of that leaf
  over the tick's plans (each tick's `offsets.position_risk`; `offsets.wall` is the wall's).
- `meta.dynamics`: `kind` ("vehicle"), `dt`, `gain`, `keep`, `turn_max_degrees`, `command_low`
  and `command_high` (the command's box: drive from 0 to 1, a target speed as a share of the top
  speed, and turn from -1 to 1), `state_sd`, `state`, `top_speed`, `cruise_speed` (the measured
  typical speed the scheduler's visit table is built at), `terrain_names` and `rule`, a sentence giving the
  vehicle's integration rule (position along the heading at the speed, a turn of at most
  `turn_max_degrees`, in place too, the speed moving toward the target). Bundles with a solar
  battery also carry `stall` (`level`, `width`: the drive delivers the share 1 / (1 + exp(-(level
  - stall.level) / stall.width)) of the command) and `energy` (`max`, the drain and charge
  constants and their `rule`; `bands`, `band_width`, `gauge_acc` and `band_meter`, the band
  meter's error model); the energy level is the state's last coordinate. The page rolls each plan of
  (drive, turn) forward with it from the believed heading and speed (and, with a stall, the drive
  share at the believed level, held over the plan), and while playing moves the rover between
  ticks (unless the reader asks for reduced motion).
- `meta.sign_convention`: every leaf is added to G with its sign applied, so an epistemic leaf
  appears negative.
- `graph`: `tiers` (factors, modalities, controls per tier: `mission` and `rover`; v2 `mission`,
  `rover1` and `rover2`, the mission's nodes per rover suffixed `_1`, `_2`, and the field the
  parameter `mission.field`, read by both rovers' terrain and both map factors), `edges` (`from`, `to`, `kind`:
  transition, likelihood, target, parameter) and `parameters` (`[{name: "rover.field", readers:
  [...]}]`). The viewer places the nodes by hand (the model is fixed) and draws the parameter
  variable once, with an edge to each reader.
- Per tick of a run: `t`, `v`, `true_x` (position and velocity), `true_heading`, `true_terrain`,
  `recogniser` (the frozen recogniser's class probabilities for the image), `look` (the class it
  reported, when the rover observes it: `--camera look`; such a run has no `recon`),
  `true_energy` (`band`, `level`), `stalled` (with a solar battery: the level below
  `meta.dynamics.stall.level`; else the empty band), `drive_share` (the share of the command the
  drive delivers), `fix`, `gauge` (the gauge's reading: of the level, a number, with a solar
  battery; of the band, an index, before), `img` and `recon` (32 x 32 uint8,
  observed and the decoder's prediction at the belief), `belief` (`components` with `w`, `mu`,
  `cov`; `terrain`; `energy` (band probabilities; with a solar battery from the Gaussian belief
  over the level, whose mean and sd are `energy_level` and `energy_sd`); `velocity`; `mean`; `heading` and `heading_sd`, the believed heading
  and its sd; `speed` and `speed_sd`), `true_speed`, `camera` (`log_evidence`, `scale_mean`, the
  decoder's noise at the belief), `policies` (the rover tier's four warm
  starts: `id`, `start`, `v_plan`, `terms`, `G`, `q_pi`), `chosen` (the index of the acted plan), `approximate`, `camera`,
  `free_energy`, `seconds` (the wall time of the rover's `plan()` round, timed whole), `offsets` (`wall`, `position_risk`: the constants the plans' records are
  shown less), `map` (the field's posterior mean, three uint8 per cell, cells row-major with x
  first), `counts` (observations absorbed per cell), `coverage` (fraction of cells whose top
  probability exceeds 0.6), `seen` (fraction with at least one count), `option` (the option in
  force; absent in the rover-alone run), `charge_to` (`[x, y]`, the charging place of the option in
  force when it is a charge option, else null; two-tier runs only; the viewer draws a line from
  the rover to it).
- Per tick of a multi-rover run (v2), the run's `ticks` are the shared rows: `t`, `map`, `counts`,
  `coverage`, `seen`, `seconds`, and in the two-tier run `options` (`[o1, o2]`, each rover's
  option in force) and `slow`; each rover's rows, with every per-tick field above (its `option`
  and `charge_to` included), are `rovers.rover1` and `rovers.rover2`. The run also carries
  `starts` and `grouped` (the rovers stepped as one group), and the two-tier run `energy_table`.
  The union arm `two-rover-union` is `{ticks: [{t, coverage, seen}], coverage_only: true,
  union_of, starts}`.
- Per run, `timing` (`infer`, the belief update, and `plan`, the plan's descent: medians over
  `reps` compiled calls of a timing harness on the rover's own declaration at the run's last round,
  apart from the run, since a compiled round is not timed by parts) and `planner` (`steps`,
  `starts`). The rover's per-tick records are read after the run from what `plan(trace=True)`
  kept in its carry (`rover_record`).
- Per tick of the two-level run, `slow`: `decision` (true every 24th tick), and on a decision
  `option`, `cell` (the scheduler's cell reading: the cell holding most of the rover's position belief), `efe_by_option` (G of the best sequence per first option), `q_by_option`,
  `policies` (the best option sequences, trimmed to keep at least the best one per first option: `options`, `terms` with the `infogain`
  `transition:mission.map` leaf and the `terminal_risk` `mission.energy` leaf, the terminal
  target scored at the plan's end, or in earlier bundles the `risk` `mission.energy` leaf, `G`,
  `q_pi`), `chosen`, `belief` (`energy`, `visit`), `energy_table` (the scheduler's learned energy
  table the decision planned with: `change`, the expected change of band over one option,
  `[band][option]`; `probs`, the next band's probabilities, `[band][option][next]`; `counts`, the
  band changes each case has counted beyond its prior, `[band][option]`; the viewer shows it under
  the decision chart) and `seconds`. The one-level run has no `slow`. The two-level run also
  carries `energy_table` and `energy_counts` (the table at the run's end).
- Per decision of the two rovers' scheduler (v2, the factorised planner), `slow`: `decision`,
  `options` (the option chosen per rover), `cells` (per rover the cell reading), `chosen` (per
  rover the sequence its posterior puts highest: `options`, `terms` (both rovers' leaves, `_1` and
  `_2`, the sequence scored against the other rover's option marginals), `G`, `q_pi`),
  `efe_by_option` (`[rover][option]`, the best G of that rover's sequences starting with the
  option), `q_by_option` (`[rover][option]`, the posterior's first-step marginal), `policies`
  (`[343, 343]`, the number of sequences per rover's posterior: no per-sequence list is kept),
  `movement` (the largest change of a marginal in the last mean-field sweep), `belief` (per rover
  `energy`, `visit`), `energy_table` (one table, `[band][option]`, at the rovers' believed cells)
  and `seconds`. The decision chart shows per rover the chosen sequence's terms and the G per
  first option with its probability, the band belief and the movement; the objective chart's
  mission tier one row per rover's chosen sequence.
- `mdl.components`: one entry per priced component (`name`, `kind`, `nats`, `exact`, `entries`),
  including `parameter:rover.field` and `parameter:mission.energy_table`, each priced once; their
  readers' `transition` entries carry structure only (`entries` 0). The page groups them by tier from the name's prefix.

The viewer opens on version v2 and its two-tier run unless the URL names a version and a
setting; in bundles without a two-tier run it opens on the first replayable setting. With two
rovers every per-rover panel shows both at once, each rover in its own colour and numbered: the
field (track, fix, belief, heading, plans with the chosen one bold), the HUD (a labelled group per
rover), the camera (the two feeds side by side), the timeline (an option strip and an energy curve
per rover), the decision chart (a block per rover) and the rover objective (a block per rover).
The legends keep the rovers in a group of their own, apart from the options. The field's legend
says that the crosses are the satellite fix's raw readings, one each tick, and, when the rover
reads the camera's image, that the position belief is tighter than their scatter because the
camera localises the rover too.

Testing: open `index.html` from a local server (or from disk); the URL hash
`#demo&version=v1&t=150&setting=one-level&tier=rover&view=terms&layer=fix&theme=dark` opens the
viewer's pane and sets the initial view (without a pane id, settings open the viewer's pane).
`version=v1|v2` picks the bundle; the key=value settings are read wherever they sit in the hash
(`#version=v1`, `#demo&version=v1`, `#demo&setting=one-level&version=v1`), keys and the version's
value case-insensitively (`V1` and `1` name v1), and the version is applied before the first
bundle loads. A setting that exists in one version only implies that version when the hash names
none (`setting=two-level` or `one-level` opens v1; `two-rover-two-level` or `two-rover-one-level`
opens v2; the map is `SETTING_VERSION` in `app.js`); a named version wins over a setting of the
other. Changing the hash on the open page (a `hashchange`, which reloads nothing) applies its
version and setting too. `window.__app.set({...})` drives the viewer from a script (it returns a
promise, which waits for a version's bundle when `version` changes it), and `window.__app.state()`
reports the version, setting and tick.

The viewer also reads fields that the example's bundles do not hold: `meta.tiers.zone_edges`
(sectors' edges per axis, else it draws the quadrants from `zone_split`), `meta.trim` (a
thinned replay: kept ticks, heavy fields and a play stride) and a camera fixed along +y (`meta.image.camera_heading_degrees`).

Images: after `survey_trim.py`, each tick's `img`, `recon` and `map` is one base64 string of the
bytes in the layout `meta.image.layout` and `meta.map_layout` describe (`meta.image.encoding` and
`meta.map_encoding` say so); before trimming they are lists. The viewer accepts both forms.
The noscript fallback's `static/hero.png` (868 x 744: the field, the HUD down to its coverage row with both rovers' groups, and the model graph),
`static/timeline.png` (868 x 486) and `static/objective.png` (868 x 716: the scheduler's decision
chart) are screenshots of v2 at tick 150 of `two-rover-two-level`, each element shot at a rendered
width of 868 px from a headless Chrome load of `index.html#demo&version=v2&t=150` (and
`#timescales&...`). `static/rover-graph.png` and `static/rover-map.gif`, used by the repository's
README, show the one-rover two-level run of an earlier bundle.
