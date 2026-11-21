(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var B = null, S = { version: null, setting: null, rover: "rover1", t: 0, frac: 0, playing: false, timer: null, raf: null, showAll: false, layer: "terrain" };
  var $ = function (id) { return document.getElementById(id); };
  var TOP_N = 20, CACHE = {}, WIN = null, BG = null, tipEl;

  function el(tag, attrs, parent, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function fmt(x, d) { return (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d == null ? 2 : d); }
  function pretty(s) { return String(s).replace(/_/g, " "); }
  function short(n) { var i = n.indexOf("."); return i < 0 ? n : n.slice(i + 1); }
  function esc(s) { return String(s).replace(/[&<>]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]; }); }
  function names(k) { return (B.meta.names && B.meta.names[k]) || []; }
  function nm(k, i) { return names(k)[i] != null ? names(k)[i] : String(i); }
  function argmax(a) { var b = 0; for (var i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function placeLabels(svg, labels, x, gap) {
    // end labels [{y, value, text, color}] at a chart's right edge, kept in the order of their
    // values (a higher value's label never below a lower one's) and at least ``gap`` apart: sorted
    // by value, then each run of labels that would overlap is spread evenly about its mean ideal
    // position, merging runs until none overlap
    gap = gap || 11;
    var ls = labels.slice().sort(function (a, b) { return b.value - a.value || a.y - b.y; }), runs = [];
    ls.forEach(function (l) {
      runs.push({ items: [l], top: l.y });
      while (runs.length > 1) {
        var a = runs[runs.length - 2], b = runs[runs.length - 1];
        if (a.top + a.items.length * gap <= b.top) break;
        var items = a.items.concat(b.items), sum = 0;
        items.forEach(function (it, i) { sum += it.y - i * gap; });
        runs.splice(runs.length - 2, 2, { items: items, top: sum / items.length });
      }
    });
    runs.forEach(function (r) {
      r.items.forEach(function (l, i) { el("text", { x: x, y: r.top + i * gap, "font-size": 10.5, fill: l.color, "font-weight": 600 }, svg, l.text); });
    });
  }
  function hexrgb(h) { h = h.replace("#", ""); return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]; }

  // ---- loading: after first paint, one version's bundle by script tag (works over http and
  // file://), each version's file loaded once, on demand ----
  var VERSIONS = {
    v2: { file: "bundle_v2.js", name: "HYBRID_AIF_BUNDLE_V2", label: "v2 (latest)" },
    v1: { file: "bundle_v1.js", name: "HYBRID_AIF_BUNDLE_V1", label: "v1 (one rover)" }
  };
  var DEFAULT_VERSION = "v2", LOADING = {};
  function load(v) {
    var spec = VERSIONS[v];
    if (LOADING[v]) return LOADING[v];
    return (LOADING[v] = new Promise(function (res, rej) {
      if (window[spec.name]) return res(window[spec.name]);
      var s = document.createElement("script");
      s.src = spec.file;
      s.onload = function () { window[spec.name] ? res(window[spec.name]) : rej(new Error("the data file did not define the replay data")); };
      s.onerror = function () { delete LOADING[v]; rej(new Error("the data file did not load")); };
      document.head.appendChild(s);
    }));
  }

  // A run's `ticks` are its shared rows (the field's map, counts and coverage, the scheduler's
  // record). A one-rover run's ticks are also the rover's rows; a multi-rover run keeps each
  // rover's rows under `rovers` (rover1, rover2), and `rows(rk)` reads them (the selected rover's
  // by default). A run marked coverage_only (the union arm) is not replayable.
  function run(id) { return B.runs[id || S.setting]; }
  function tick() { return run().ticks[S.t]; }
  function multi(id) { return !!run(id).rovers; }
  function roverKeys(id) { return multi(id) ? Object.keys(run(id).rovers) : [null]; }
  function rows(rk, id) { var r = run(id); return r.rovers ? r.rovers[rk || S.rover] : r.ticks; }
  function rtick(rk) { return rows(rk)[S.t]; }
  function roverNo(rk) { return +String(rk || S.rover).replace(/\D/g, "") || 1; }
  function roverName(rk) { return "rover " + roverNo(rk); }
  function roverColor(rk) { return css("--r" + roverNo(rk)) || css("--fg"); }
  function replayable(id) { return !!B.runs[id] && !B.runs[id].coverage_only; }
  function settingRec(id) { for (var i = 0; i < B.settings.length; i++) if (B.settings[i].id === (id || S.setting)) return B.settings[i]; return null; }
  function twoLevel() { var r = settingRec(); return !!(r && r.tiers === 2); }
  function cache(id) {
    // per run: the index of the latest decision at or before each tick, and per rover the
    // segments of the option in force
    id = id || S.setting;
    if (CACHE[id]) return CACHE[id];
    var T = B.runs[id].ticks, last = [], li = -1, segs = {};
    T.forEach(function (t, i) {
      if (t.slow && t.slow.decision) li = i;
      last.push(li);
    });
    roverKeys(id).forEach(function (rk) {
      var R = rows(rk, id), sg = [];
      R.forEach(function (t, i) { if (!sg.length || sg[sg.length - 1].option !== t.option) sg.push({ option: t.option, t0: i }); });
      sg.forEach(function (s, i) { s.t1 = i + 1 < sg.length ? sg[i + 1].t0 : R.length; });
      segs[rk] = sg;
    });
    return (CACHE[id] = { last: last, segs: segs });
  }
  function decTick() { var i = cache().last[S.t]; return i < 0 ? null : run().ticks[i]; }

  // ---- names and colours ----
  var TERM_VAR = { risk: "--c1", terminal_risk: "--c1", uncertainty: "--c2", ambiguity: "--c5", infogain: "--c3", utility: "--c4" };
  function termColor(t) { return css(TERM_VAR[t] || "--muted"); }
  function optColor(i) {
    // the page's option colours, and for options beyond them (a field of many sectors) a hue each
    var c = css("--o" + i);
    return c || "hsl(" + Math.round((i * 137.5) % 360) + ",55%,55%)";
  }
  function nOpts() { return (B.meta.tiers && B.meta.tiers.options || []).length || 5; }
  function heavy(key, at, rk) {
    // the latest tick at or before ``at`` that carries ``key``: a trimmed bundle keeps the heavy
    // fields (pictures, map, plans) on a subset of ticks (meta.trim), every other field on all.
    // The map is the shared field's; pictures and plans are the rover's (rk, else the selected)
    var T = key === "map" ? run().ticks : rows(rk), i = at == null ? S.t : at;
    while (i > 0 && T[i][key] == null) i--;
    return T[i];
  }
  function optLabel(i) { return (B.meta.tiers.options || [])[i] || "option " + i; }
  function hasOpt(t) { return t.option != null; }
  function pinnedOpt() { return B.meta.tiers.pinned_option; }  // set by bundles whose one-level run pins an option
  // the scheduler's horizon in options (meta.tiers.horizon; bundles without it looked two ahead)
  function horizon() { return (B.meta.tiers && B.meta.tiers.horizon) || 2; }
  // the energy bands the bundle names (four in bundles whose bands are a unit wide), and the level
  // of a full battery they divide
  function nBands() { return names("energy").length || 4; }
  function energyMax() { return (B.meta.dynamics && B.meta.dynamics.energy && B.meta.dynamics.energy.max) || 4; }
  function sunSw(color) {  // the legend's sun: the charging place's marker at legend size
    var rays = "";
    for (var a = 0; a < 8; a++) { var an = a * Math.PI / 4; rays += "<line x1='" + (7 + 4.6 * Math.cos(an)).toFixed(1) + "' y1='" + (6 + 4.6 * Math.sin(an)).toFixed(1) + "' x2='" + (7 + 6.6 * Math.cos(an)).toFixed(1) + "' y2='" + (6 + 6.6 * Math.sin(an)).toFixed(1) + "'/>"; }
    return "<svg class='lsw' style='width:14px' viewBox='0 0 14 12' fill='none' stroke='" + color + "' stroke-width='1.5'><circle cx='7' cy='6' r='3.1'/>" + rays + "</svg>";
  }
  // the field legend's glyphs, one per colour given (a rover's colour each, or the page's one)
  function lineSw(colors, dash, w) {
    var n = colors.length, seg = 26 / n, s = "";
    colors.forEach(function (c, k) { s += "<line x1='" + (1 + k * seg).toFixed(1) + "' x2='" + (1 + (k + 1) * seg - (n > 1 ? 2 : 0)).toFixed(1) + "' y1='5' y2='5' stroke='" + c + "' stroke-width='" + (w || 2.4) + "' stroke-linecap='round'" + (dash ? " stroke-dasharray='" + dash + "'" : "") + "/>"; });
    return "<svg class='lsw' viewBox='0 0 28 10' fill='none'>" + s + "</svg>";
  }
  function xSw(colors) {
    var s = "";
    colors.forEach(function (c, k) { var x = 5 + k * 11; s += "<path d='M" + (x - 3.5) + ",1.5 L" + (x + 3.5) + ",8.5 M" + (x - 3.5) + ",8.5 L" + (x + 3.5) + ",1.5' stroke='" + c + "' stroke-width='2.4' stroke-linecap='round'/>"; });
    var w = 11 * colors.length - 1;
    return "<svg class='lsw' style='width:" + w + "px' viewBox='0 0 " + w + " 10' fill='none'>" + s + "</svg>";
  }
  function circSw(colors) {
    var s = "";
    colors.forEach(function (c, k) { var x = 5 + k * 11; s += "<circle cx='" + x + "' cy='5' r='3.6' fill='" + c + "' fill-opacity='.35' stroke='" + c + "' stroke-width='1.5'/>"; });
    var w = 11 * colors.length - 1;
    return "<svg class='lsw' style='width:" + w + "px' viewBox='0 0 " + w + " 10'>" + s + "</svg>";
  }
  function countWord(n) { return ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"][n - 1] || String(n); }
  function horizonWord() { return ["one", "two", "three", "four", "five", "six"][horizon() - 1] || String(horizon()); }
  // the scheduler's energy term: a target scored once at the plan's end, or at every step
  function terminalEnergy() { return !!(B.meta.tiers && B.meta.tiers.terminal_penalty != null); }
  function energyTerm() { return terminalEnergy() ? "energy at the plan's end" : "energy risk"; }
  function aloneText() { return "\u2013"; }  // the option row's value when the rover runs alone
  // the options are the surveys, one per sector (meta.tiers.zones), then the charges, one per
  // charging place: an option is a charge when its index is at least the number of sectors
  function nZones() { return (B.meta.tiers.zones || []).length || 4; }
  function isCharge(i) { return i != null && i >= nZones(); }
  function optShort(i) { return isCharge(i) ? "C" + (i - nZones() + 1) : "S" + (i + 1); }
  function sectorOf(opt) { return opt != null && opt >= 0 && opt < nZones() ? opt : -1; }
  function chargeSites() {
    // the charging places the charge options drive to: one per charge option (charge_sites), else
    // the one fixed place, unless the charge option has no fixed place
    var tr = B.meta.tiers;
    if (tr.charge_sites) return tr.charge_sites;
    return tr.charge_form === "nearest" || tr.charge_form === "per-option" || !tr.charge_site ? [] : [tr.charge_site];
  }
  function terrColor(c) { return css(["--mf", "--mr", "--mi"][c]); }
  function terrBg(c) { return css(["--tf", "--tr", "--ti"][c]); }

  // ---- the model graph: both tiers, the parameter variable between them ----
  // One rover (v1): the mission tier above the rover tier (POS1). Two rovers (v2): the mission
  // tier in the middle column, rover 1's tier to its left and rover 2's, mirrored, to its right;
  // the mission's nodes per rover (_1, _2) on that rover's side; the two parameter variables
  // (the energy table and the field) drawn once, in the middle, with an edge to each reader.
  var POS, GW, GH, MULTI_GRAPH = false;
  function pos2() {
    var L = {
      "mission.u_option_1": [295, 62], "mission.visit_1": [250, 180], "mission.energy_1": [340, 180],
      "mission.cell_obs_1": [250, 290], "mission.gauge_1": [340, 290], "mission.map_1": [325, 400],
      "rover1.v": [135, 62], "rover1.fix": [50, 130], "rover1.x": [135, 180], "rover1.gauge": [50, 260],
      "rover1.terrain": [135, 400], "rover1.img": [50, 400], "rover1.look": [50, 400]
    }, P = { "mission.energy_table": [380, 118], "mission.field": [380, 470] };
    Object.keys(L).forEach(function (n) { P[n] = L[n]; P[n.replace(/_1$/, "_2").replace(/^rover1\./, "rover2.")] = [760 - L[n][0], L[n][1]]; });
    return P;
  }
  function pbox(name) {  // a parameter box's half width and half height
    if (MULTI_GRAPH) return /energy_table$/.test(name) ? [46, 17] : [52, 24];
    return /energy_table$/.test(name) ? [40, 20] : [52, 24];
  }
  // the energy table's box sits at the left of the energy node, clear of the option edge's lane
  var POS1 = {
    "mission.u_option": [180, 52], "mission.energy": [180, 130], "mission.visit": [320, 130], "mission.map": [460, 130],
    "mission.reading": [420, 52],
    "mission.gauge": [180, 208], "mission.cell_obs": [320, 208],
    "rover.field": [460, 282], "mission.energy_table": [84, 130],
    "rover.v": [160, 440], "rover.x": [320, 440], "rover.energy": [190, 366], "rover.terrain": [460, 440],
    "rover.fix": [180, 528], "rover.gauge": [320, 528], "rover.img": [460, 528], "rover.look": [460, 528]
  };
  // x sits directly below cell_obs, so the belief read runs straight up between them; the two
  // nodes on that line carry their family label beside them, clear of the arrow and of the edge
  // from x down to the gauge: [dx, dy, anchor] from the node's centre
  var SUB = { "rover.x": [10, -32, "start"], "mission.cell_obs": [32, 4, "start"] };
  var BEND = { "rover.x>rover.energy": 0, "rover.terrain>rover.energy": 0, "rover.x>rover.img": 0, "rover.terrain>rover.img": 0, "rover.energy>rover.img": 8 };
  var FAM = { categorical: "categorical", gaussian_mixture: "Gauss. mixture", linear_gaussian: "lin. Gaussian", neural: "neural (MLP)" };
  var KINDS = {
    transition: { dash: "", color: "--fg", w: 1.3, label: "transition" },
    likelihood: { dash: "5 3", color: "--c1", w: 1.3, label: "likelihood" },
    target: { dash: "2 3", color: "--c2", w: 2, label: "preference (target)" },
    parameter: { dash: "", color: "--c3", w: 3, label: "parameter variable read" }
  };
  // the scheduler's cell reading: a read of the rover's position belief across the tiers
  var BELIEF = { dash: "4 3", color: "--muted", w: 1.4, label: "belief read (the cell of the rover's position belief)" };
  function drawGraph() {
    var svg = $("graph"); svg.innerHTML = "";
    MULTI_GRAPH = B.graph.tiers.some(function (t) { return /^rover\d+$/.test(t.name); });
    POS = MULTI_GRAPH ? pos2() : POS1; GW = MULTI_GRAPH ? 760 : 520; GH = MULTI_GRAPH ? 540 : 600;
    var W = GW, H = GH;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H); svg.style.aspectRatio = W + "/" + H;
    svg.setAttribute("aria-label", MULTI_GRAPH ? "Graphical model of the mission tier and the two rovers, with the shared field and energy table" : "Graphical model of the two tiers with the shared terrain table");
    var defs = el("defs", {}, svg);
    Object.keys(KINDS).forEach(function (k) {
      [["ar-", "auto"], ["as-", "auto-start-reverse"]].forEach(function (p) {
        var m = el("marker", { id: p[0] + k, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: p[1] }, defs);
        el("path", { d: "M0,0 L10,5 L0,10 z", fill: css(KINDS[k].color) }, m);
      });
    });
    var mb = el("marker", { id: "ar-belief", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto" }, defs);
    el("path", { d: "M0,0 L10,5 L0,10 z", fill: css(BELIEF.color) }, mb);
    // the scheduler tier (its frame, nodes, edges and the reads that cross into it) is one group,
    // drawn first and greyed by updateGraph in a setting where the scheduler does not run
    var slow = el("g", { id: "graph-slow" }, svg);
    var frames = MULTI_GRAPH
      ? [[208, 12, 344, 516, "mission tier (slow): one decision every " + B.meta.tiers.slow_ticks + " ticks", slow, true],
         [8, 12, 192, 516, "rover 1 (fast)", svg], [560, 12, 192, 516, "rover 2 (fast)", svg]]
      : [[8, 12, 504, 232, "mission tier (slow): one decision every " + B.meta.tiers.slow_ticks + " ticks", slow, true], [8, 316, 504, 268, "rover tier (fast): one step every tick", svg]];
    frames.forEach(function (f) {
      el("rect", { x: f[0], y: f[1], width: f[2], height: f[3], rx: 8, fill: "none", stroke: css("--line"), "stroke-width": 1.5 }, f[5]);
      var ft = el("text", { x: f[0] + 12, y: f[1] + 18, "font-size": 11.5, "font-weight": 600, class: "muted" }, f[5], f[4]);
      // the mission frame's note when the scheduler does not run: on the title line (one rover),
      // on a line of its own under it (two rovers, a narrower frame)
      if (f[6] && !MULTI_GRAPH) el("tspan", { class: "idle" }, ft, "");
      else if (f[6]) el("text", { x: f[0] + 12, y: f[1] + 32, "font-size": 10.5, class: "muted idle" }, f[5], "");
      else if (MULTI_GRAPH) el("text", { x: f[0] + 12, y: f[1] + 32, "font-size": 10.5, class: "muted" }, f[5], "one step every tick");
    });
    var info = {};
    B.graph.tiers.forEach(function (t) {
      t.factors.forEach(function (n) { info[n.name] = { kind: "factor", nd: n }; });
      t.modalities.forEach(function (n) { info[n.name] = { kind: "modality", nd: n }; });
      t.controls.forEach(function (n) { info[n.name] = { kind: "control", nd: n }; });
    });
    (B.graph.parameters || []).forEach(function (p) { info[p.name] = { kind: "parameter", nd: p }; });
    // a node belongs to the scheduler's group when it is the mission's, unless it is a parameter
    // that a rover reads too (two rovers share the field with or without the scheduler)
    function slowNode(n) {
      if (!/^mission\./.test(n)) return false;
      var i = info[n];
      return !(i && i.kind === "parameter" && (i.nd.readers || []).some(function (r) { return !/^mission\./.test(r); }));
    }
    function inSlow(e) { return slowNode(e.from) || slowNode(e.to); }
    // edges (reciprocal transitions drawn once, with a head at each end)
    var edges = B.graph.edges, done = {};
    function pathFor(a, b, bend, r1, r2) {
      var dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy) || 1, ux = dx / len, uy = dy / len;
      var x1 = a[0] + ux * r1, y1 = a[1] + uy * r1, x2 = b[0] - ux * r2, y2 = b[1] - uy * r2;
      var cx = (x1 + x2) / 2 - uy * bend, cy = (y1 + y2) / 2 + ux * bend;
      return "M" + x1 + "," + y1 + " Q" + cx + "," + cy + " " + x2 + "," + y2;
    }
    // the distance from a node's centre to its outline toward the point `to`
    function rad(n, to) {
      var k = info[n] && info[n].kind;
      if (k === "parameter" && (MULTI_GRAPH || /energy_table$/.test(n)) && to) {  // a box: its outline toward `to`
        var pc = POS[n], pb = pbox(n), pdx = to[0] - pc[0], pdy = to[1] - pc[1], plen = Math.hypot(pdx, pdy) || 1;
        return Math.min(pb[0] / Math.max(Math.abs(pdx) / plen, 1e-6), pb[1] / Math.max(Math.abs(pdy) / plen, 1e-6)) + 6;
      }
      if (k === "parameter") return 34;
      if (k !== "control" || !to) return 27;
      var c = POS[n], dx = to[0] - c[0], dy = to[1] - c[1], len = Math.hypot(dx, dy) || 1;
      return Math.min(38 / Math.max(Math.abs(dx) / len, 1e-6), 15 / Math.max(Math.abs(dy) / len, 1e-6)) + 2;
    }
    edges.forEach(function (e, i) {
      var a = POS[e.from], b = POS[e.to]; if (!a || !b || e.from === e.to) return;
      var key = e.from + ">" + e.to, rev = e.to + ">" + e.from, st = KINDS[e.kind] || KINDS.transition;
      var attrs = { fill: "none", stroke: css(st.color), "stroke-width": st.w, "stroke-dasharray": st.dash, "marker-end": "url(#ar-" + (KINDS[e.kind] ? e.kind : "transition") + ")" };
      var d, layer = inSlow(e) ? slow : svg;
      if (/^mission\.u_option$/.test(e.from) && /^rover\./.test(e.to)) {
        // the option edge between tiers runs down the left margin; to x it turns right above the
        // v pill and enters x from its upper left
        var ty = e.to === "rover.x" ? b[1] - 44 : 528, lane = e.to === "rover.x" ? 24 : 36;
        if (e.to === "rover.img") ty = b[1] + 14;
        d = "M" + (a[0] - 24) + "," + a[1] + " C" + (lane - 10) + "," + a[1] + " " + lane + "," + (a[1] + 20) + " " + lane + "," + (a[1] + 90) + " L" + lane + "," + (e.to === "rover.x" ? ty - 14 : ty + 40) +
          (e.to === "rover.x" ? " C" + lane + "," + ty + " " + (lane + 20) + "," + ty + " " + (lane + 40) + "," + ty + " L" + (b[0] - 80) + "," + ty + " C" + (b[0] - 44) + "," + ty + " " + (b[0] - 30) + "," + (ty + 10) + " " + (b[0] - 21) + "," + (b[1] - 21) : " C" + lane + "," + (ty + 62) + " " + (b[0] - 120) + "," + (ty + 62) + " " + (b[0] - 18) + "," + (ty + 20));
        el("path", Object.assign({ d: d }, attrs), layer);
        return;
      }
      if (done[rev] && e.kind === "transition") return;
      var recip = e.kind === "transition" && edges.some(function (o) { return o.from + ">" + o.to === rev && o.kind === "transition"; });
      done[key] = true;
      var bend = BEND[key] != null ? BEND[key] : 0;
      if (recip) { bend = (key === "rover.x>rover.terrain" || key === "rover.terrain>rover.x") ? 78 : 20; attrs["marker-start"] = "url(#as-transition)"; }
      var r1 = rad(e.from, b), r2 = rad(e.to, a) + 3;
      if (recip && bend > 40) {
        // x <-> terrain passes over the energy node: an arc above the row
        var xa = POS[key === "rover.x>rover.terrain" ? e.from : e.to], xb = POS[key === "rover.x>rover.terrain" ? e.to : e.from];
        d = "M" + (xa[0] + 8) + "," + (xa[1] - 26) + " C" + (xa[0] + 40) + "," + (xa[1] - 82) + " " + (xb[0] - 40) + "," + (xb[1] - 82) + " " + (xb[0] - 8) + "," + (xb[1] - 28);
        if (key === "rover.terrain>rover.x") return;
      } else d = pathFor(a, b, bend, r1, r2);
      el("path", Object.assign({ d: d }, attrs), layer);
    });
    var beliefRead = info["rover.x"] && info["mission.cell_obs"];
    if (MULTI_GRAPH) {
      // each rover's position belief read by its own cell observation, once per decision
      [1, 2].forEach(function (k) {
        var a = POS["rover" + k + ".x"], b = POS["mission.cell_obs_" + k];
        if (!a || !b || !info["rover" + k + ".x"] || !info["mission.cell_obs_" + k]) return;
        beliefRead = true;
        el("path", { d: pathFor(a, b, 0, 27, 30), fill: "none", stroke: css(BELIEF.color), "stroke-width": BELIEF.w, "stroke-dasharray": BELIEF.dash, "marker-end": "url(#ar-belief)" }, slow);
      });
    } else if (beliefRead) {
      // straight up from the top of x to the bottom of cell_obs, its note left of it, inside the
      // rover's frame above the option edge (the gap between the frames holds the field's note)
      var bx = POS["rover.x"], bc = POS["mission.cell_obs"];
      el("path", { d: "M" + bx[0] + "," + (bx[1] - 27) + " L" + bc[0] + "," + (bc[1] + 30), fill: "none", stroke: css(BELIEF.color), "stroke-width": BELIEF.w, "stroke-dasharray": BELIEF.dash, "marker-end": "url(#ar-belief)" }, slow);
      el("text", { x: bc[0] - 8, y: 352, "text-anchor": "end", "font-size": 10.5, class: "muted" }, slow, "belief read,");
      el("text", { x: bc[0] - 8, y: 365, "text-anchor": "end", "font-size": 10.5, class: "muted" }, slow, "once per decision");
    }
    Object.keys(POS).forEach(function (name) {
      var p = POS[name], i = info[name]; if (!i) return;
      var g = el("g", { "data-node": name }, slowNode(name) ? slow : svg), label = short(name), nd = i.nd, fs = label.length > 9 ? 11 : 12.5;
      var fill = css("--bg"), stroke = css("--fg");
      // a state factor and an observation are circles (plain and blue); a control is an orange pill
      if (i.kind === "factor") el("circle", { cx: p[0], cy: p[1], r: 26, fill: fill, stroke: stroke, "stroke-width": 1.6 }, g);
      else if (i.kind === "modality") {
        el("circle", { cx: p[0], cy: p[1], r: 26, fill: fill, stroke: "none" }, g);
        el("circle", { cx: p[0], cy: p[1], r: 26, fill: css("--c1"), "fill-opacity": 0.16, stroke: css("--c1"), "stroke-width": 1.6 }, g);
      }
      else if (i.kind === "parameter") {
        var pb = pbox(name);  // the energy table's name is the longer
        el("rect", { x: p[0] - pb[0], y: p[1] - pb[1], width: 2 * pb[0], height: 2 * pb[1], rx: 10, fill: css("--c3"), "fill-opacity": 0.16, stroke: css("--c3"), "stroke-width": 2.6 }, g);
      } else el("rect", { x: p[0] - 38, y: p[1] - 15, width: 76, height: 30, rx: 15, fill: fill, stroke: css("--c2"), "stroke-width": 1.8 }, g);
      var plabel = /energy_table$/.test(name) ? short(name) : name;
      el("text", { x: p[0], y: p[1] + 4, "text-anchor": "middle", "font-size": i.kind === "parameter" && (MULTI_GRAPH || /energy_table$/.test(name)) ? 11.5 : fs, "font-weight": i.kind === "parameter" ? 700 : 400 }, g, i.kind === "parameter" ? plabel : label);
      if (MULTI_GRAPH && i.kind === "parameter") {
        // the notes go in the legend, but for the field's size, under its box
        if (!/energy_table$/.test(name)) {
          var gs2 = B.meta.grid && B.meta.grid.shape ? B.meta.grid.shape[0] * B.meta.grid.shape[1] : 256;
          el("text", { x: p[0], y: p[1] + 40, "text-anchor": "middle", "font-size": 10.5, class: "muted" }, g, "one Dirichlet table,");
          el("text", { x: p[0], y: p[1] + 53, "text-anchor": "middle", "font-size": 10.5, class: "muted" }, g, "3 kinds × " + gs2.toLocaleString("en-US") + " cells");
        }
      } else if (MULTI_GRAPH && i.kind === "control") {
        // under the pill, centred: the mission's two pills sit side by side
        var ct2 = el("text", { x: p[0], y: p[1] + 30, "text-anchor": "middle", "font-size": 10.5, class: "muted" }, g, (nd.kind === "level" ? nd.levels + " levels" : "continuous") + "  ");
        el("tspan", { class: "now" }, ct2, "");
      } else if (i.kind === "parameter" && /energy_table$/.test(name)) {
        // under the box, centred
        el("text", { x: p[0], y: p[1] + 35, "text-anchor": "middle", "font-size": 10.5, class: "muted" }, g, "one Dirichlet per");
        el("text", { x: p[0], y: p[1] + 48, "text-anchor": "middle", "font-size": 10.5, class: "muted" }, g, "band, cell and option");
      } else if (i.kind === "parameter") {
        el("text", { x: p[0] - 58, y: p[1] - 2, "text-anchor": "end", "font-size": 10.5, class: "muted" }, g, "one Dirichlet table,");
        var gs = B.meta.grid && B.meta.grid.shape ? B.meta.grid.shape[0] * B.meta.grid.shape[1] : 256;
        el("text", { x: p[0] - 58, y: p[1] + 11, "text-anchor": "end", "font-size": 10.5, class: "muted" }, g, "3 kinds × " + gs.toLocaleString("en-US") + " cells");
      } else if (i.kind === "control") {
        // beside the pill, clear of the edges that leave it downward
        var ct = el("text", { x: p[0] + 44, y: p[1] + 19, "text-anchor": "start", "font-size": 10.5, class: "muted" }, g, (nd.kind === "level" ? nd.levels + " levels" : "continuous") + "  ");
        el("tspan", { class: "now" }, ct, "");
      } else {
        // a factor's family is its own; a modality's is its likelihood's, not the reading's
        var sub = FAM[nd.family] || pretty(nd.family || "");
        if (i.kind === "factor") sub += nd.levels ? " · " + nd.levels : " · dim " + nd.dim;
        var at = SUB[name] || [0, 40, "middle"];
        el("text", { x: p[0] + at[0], y: p[1] + at[1], "text-anchor": at[2], "font-size": MULTI_GRAPH && /^mission\./.test(name) ? 9 : 10.5, class: "muted" }, g, MULTI_GRAPH && /^mission\./.test(name) ? sub.replace("categorical", "cat.") : sub);
      }
    });
    var lg = "", present = {};
    edges.forEach(function (e) { present[e.kind] = true; });
    Object.keys(KINDS).forEach(function (k) {
      if (!present[k]) return;
      var st = KINDS[k];
      lg += "<span><svg class='lsw' viewBox='0 0 28 10'><line x1='1' x2='27' y1='5' y2='5' stroke='" + css(st.color) + "' stroke-width='" + st.w + "' stroke-dasharray='" + st.dash + "'/></svg>" + esc(st.label) + "</span>";
    });
    function pill(fillVar, strokeVar) {
      return "<svg class='lsw' viewBox='0 0 28 12'><rect x='1' y='1' width='26' height='10' rx='5' fill='" + (fillVar ? css(fillVar) : "none") + "' fill-opacity='" + (fillVar ? 0.16 : 1) + "' stroke='" + css(strokeVar) + "' stroke-width='1.4'/></svg>";
    }
    function dot(fillVar, strokeVar) {
      return "<svg class='lsw' viewBox='0 0 28 12'><circle cx='14' cy='6' r='5' fill='" + (fillVar ? css(fillVar) : "none") + "' fill-opacity='" + (fillVar ? 0.16 : 1) + "' stroke='" + css(strokeVar) + "' stroke-width='1.4'/></svg>";
    }
    if (beliefRead) lg += "<span><svg class='lsw' viewBox='0 0 28 10'><line x1='1' x2='27' y1='5' y2='5' stroke='" + css(BELIEF.color) + "' stroke-width='" + BELIEF.w + "' stroke-dasharray='" + BELIEF.dash + "'/></svg>" + esc(BELIEF.label) + "</span>";
    lg += "<span>" + dot(null, "--fg") + "state factor</span><span>" + dot("--c1", "--c1") + "observation (its likelihood's family)</span><span>" + pill(null, "--c2") + "control</span>";
    if (MULTI_GRAPH) lg += "<span style='flex-basis:100%'>" + pill("--c3", "--c3") + "parameter variables, each drawn once: energy_table (one Dirichlet per band, cell and option; read by both rovers' energy) and field (read by both rovers' terrain and both map factors)</span>";
    $("graphlegend").innerHTML = lg;
    updateGraph();
  }
  function updateGraph() {
    var slow = $("graph-slow"), idle = slow && slow.querySelector(".idle");
    if (slow) slow.classList.toggle("inactive", !twoLevel());
    if (idle) idle.textContent = twoLevel() ? "" : MULTI_GRAPH ? "(not run in this setting)" : " (not run in this setting)";
    document.querySelectorAll("#graph g[data-node]").forEach(function (g) {
      var n = g.getAttribute("data-node"), shape = g.firstElementChild, now = g.querySelector(".now");
      var m = /^mission\.u_option(?:_(\d+))?$/.exec(n);
      if (m) {
        // the option node of a rover (u_option_k) shows that rover's option in force
        var rk = m[1] && multi() ? "rover" + m[1] : null, t = rk ? rtick(rk) : rtick();
        if (hasOpt(t)) { shape.setAttribute("fill", optColor(t.option)); shape.setAttribute("fill-opacity", 0.4); }
        else { shape.setAttribute("fill", css("--bg")); shape.setAttribute("fill-opacity", 1); }
        if (now) now.textContent = hasOpt(t) ? "now: " + optShort(t.option) : MULTI_GRAPH ? "" : "not run in this setting";
      }
    });
  }

  // ---- the field canvas ----
  function computeWindow() {
    var f = B.meta.field, pad = 0.04 * Math.max(f[0][1] - f[0][0], f[1][1] - f[1][0]);
    return { x0: f[0][0] - pad, x1: f[0][1] + pad, y0: f[1][0] - pad, y1: f[1][1] + pad };
  }
  function setupCanvas(c) {
    var dpr = window.devicePixelRatio || 1, w = c.clientWidth || 520;
    var sc = w / (WIN.x1 - WIN.x0), h = Math.round((WIN.y1 - WIN.y0) * sc);
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    var g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    return {
      g: g, w: w, h: h, sc: sc, dpr: dpr,
      X: function (x) { return (x - WIN.x0) * sc; },
      Y: function (y) { return h - (y - WIN.y0) * sc; }
    };
  }
  function mapAt(m, ix, iy) { return m[ix][iy]; }
  function contours(g, m, W) {
    // marching squares on the height map, levels every 0.2
    var xs = W.xs, ys = W.ys, H = W.height, nx = xs.length, ny = ys.length, hmax = 0;
    H.forEach(function (row) { row.forEach(function (v) { hmax = Math.max(hmax, v); }); });
    var SEG = [[], [[3, 0]], [[0, 1]], [[3, 1]], [[1, 2]], [[3, 0], [1, 2]], [[0, 2]], [[3, 2]], [[2, 3]], [[0, 2]], [[0, 1], [2, 3]], [[1, 2]], [[1, 3]], [[0, 1]], [[3, 0]], []];
    g.beginPath();
    for (var lv = 0.2; lv < Math.max(1.5, hmax); lv += 0.2) {
      for (var i = 0; i < nx - 1; i++) for (var j = 0; j < ny - 1; j++) {
        var c = [H[i][j], H[i + 1][j], H[i + 1][j + 1], H[i][j + 1]], px = [xs[i], xs[i + 1], xs[i + 1], xs[i]], py = [ys[j], ys[j], ys[j + 1], ys[j + 1]];
        var idx = (c[0] > lv ? 1 : 0) | (c[1] > lv ? 2 : 0) | (c[2] > lv ? 4 : 0) | (c[3] > lv ? 8 : 0);
        var segs = SEG[idx]; if (!segs.length) continue;
        var pt = function (e) {
          var a = [0, 1, 2, 3][e], b = (e + 1) % 4, f = (lv - c[a]) / (c[b] - c[a]);
          return [m.X(px[a] + f * (px[b] - px[a])), m.Y(py[a] + f * (py[b] - py[a]))];
        };
        segs.forEach(function (s) { var p = pt(s[0]), q = pt(s[1]); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); });
      }
    }
    g.stroke();
  }
  function background(m) {
    // the true ground, an optional wash, and the height contours, cached per size, layer and theme
    var key = [m.w, m.h, S.layer, css("--tf"), css("--bg")].join("|");
    if (BG && BG.key === key) return BG.canvas;
    var W = B.meta.world, nx = W.xs.length, ny = W.ys.length, oc = document.createElement("canvas");
    oc.width = Math.round(m.w * m.dpr); oc.height = Math.round(m.h * m.dpr);
    var g = oc.getContext("2d"); g.setTransform(m.dpr, 0, 0, m.dpr, 0, 0);
    g.fillStyle = css("--panel"); g.fillRect(0, 0, m.w, m.h);
    var f = B.meta.field, step = W.xs[1] - W.xs[0], stepy = W.ys[1] - W.ys[0];
    var rgb = [0, 1, 2].map(function (c) { return hexrgb(terrBg(c)); });
    var lmin = Infinity, lmax = -Infinity, smin = Infinity, smax = -Infinity;
    var LT = W.sun || W.light;  // the illumination layer shows the solar yield where the bundle has it
    for (var i = 0; i < nx; i++) for (var j = 0; j < ny; j++) {
      lmin = Math.min(lmin, LT[i][j]); lmax = Math.max(lmax, LT[i][j]);
      smin = Math.min(smin, W.sd_fix[i][j]); smax = Math.max(smax, W.sd_fix[i][j]);
    }
    var wash = hexrgb(css("--washfix").slice(0, 7));
    for (i = 0; i < nx; i++) for (j = 0; j < ny; j++) {
      var x0 = m.X(W.xs[i] - step / 2), x1 = m.X(W.xs[i] + step / 2), y1 = m.Y(W.ys[j] - stepy / 2), y0 = m.Y(W.ys[j] + stepy / 2);
      var c = rgb[W.terrain[i][j]];
      g.fillStyle = "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
      g.fillRect(x0, y0, x1 - x0 + 0.6, y1 - y0 + 0.6);
      if (S.layer === "light") {
        var n = (LT[i][j] - lmin) / (lmax - lmin);
        g.fillStyle = "rgba(0,0,0," + (0.55 * (1 - n)).toFixed(3) + ")"; g.fillRect(x0, y0, x1 - x0 + 0.6, y1 - y0 + 0.6);
        g.fillStyle = "rgba(255,236,150," + (0.5 * n).toFixed(3) + ")"; g.fillRect(x0, y0, x1 - x0 + 0.6, y1 - y0 + 0.6);
      } else if (S.layer === "fix") {
        var q = (W.sd_fix[i][j] - smin) / (smax - smin);
        g.fillStyle = "rgba(" + wash.join(",") + "," + (0.7 * q).toFixed(3) + ")"; g.fillRect(x0, y0, x1 - x0 + 0.6, y1 - y0 + 0.6);
      }
    }
    g.strokeStyle = css("--contour"); g.lineWidth = 0.9; contours(g, m, W);
    g.strokeStyle = css("--muted"); g.lineWidth = 1.2; g.strokeRect(m.X(f[0][0]), m.Y(f[1][1]), (f[0][1] - f[0][0]) * m.sc, (f[1][1] - f[1][0]) * m.sc);
    BG = { key: key, canvas: oc };
    return oc;
  }
  function unpackMap(t) {
    if (t.map && t.map._u8) return t.map._u8;
    var s = t.map, out;
    if (typeof s === "string") { var b = atob(s); out = new Uint8Array(b.length); for (var i = 0; i < b.length; i++) out[i] = b.charCodeAt(i); }
    else { out = new Uint8Array(s.length * 3); s.forEach(function (r, i) { out[3 * i] = r[0]; out[3 * i + 1] = r[1]; out[3 * i + 2] = r[2]; }); }
    t._u8 = out; return out;
  }
  function drawMap(m, t) {
    var g = m.g, gr = B.meta.grid, nx = gr.shape[0], ny = gr.shape[1], dx = (gr.high[0] - gr.low[0]) / nx, dy = (gr.high[1] - gr.low[1]) / ny;
    t = t.map != null ? t : heavy("map");
    var u = unpackMap(t);
    for (var ix = 0; ix < nx; ix++) for (var iy = 0; iy < ny; iy++) {
      var c = ix * ny + iy;
      if (t.counts && t.counts[c] < 0.02) continue;
      var p = [u[3 * c] / 255, u[3 * c + 1] / 255, u[3 * c + 2] / 255], k = argmax(p), a = clamp((p[k] - 0.34) / 0.66, 0, 1);
      if (a <= 0.02) continue;
      g.globalAlpha = 0.12 + 0.7 * a; g.fillStyle = terrColor(k);
      var x0 = m.X(gr.low[0] + ix * dx), x1 = m.X(gr.low[0] + (ix + 1) * dx), y1 = m.Y(gr.low[1] + iy * dy), y0 = m.Y(gr.low[1] + (iy + 1) * dy);
      g.fillRect(x0, y0, x1 - x0 + 0.5, y1 - y0 + 0.5);
    }
    g.globalAlpha = 1;
    // cell lines, faint
    g.strokeStyle = css("--cellline"); g.lineWidth = 0.5; g.beginPath();
    for (ix = 0; ix <= nx; ix++) { g.moveTo(m.X(gr.low[0] + ix * dx), m.Y(gr.low[1])); g.lineTo(m.X(gr.low[0] + ix * dx), m.Y(gr.high[1])); }
    for (iy = 0; iy <= ny; iy++) { g.moveTo(m.X(gr.low[0]), m.Y(gr.low[1] + iy * dy)); g.lineTo(m.X(gr.high[0]), m.Y(gr.low[1] + iy * dy)); }
    g.stroke();
  }
  function drawSectors(m) {
    var g = m.g, f = B.meta.field, sp = B.meta.tiers.zone_split;
    // the sectors' edges per axis (meta.tiers.zone_edges), else the quadrants split at zone_split
    var ze = B.meta.tiers.zone_edges || [[f[0][0], sp[0], f[0][1]], [f[1][0], sp[1], f[1][1]]];
    var span = function (e) { var o = []; for (var i = 0; i + 1 < e.length; i++) o.push([e[i], e[i + 1]]); return o; };
    var xr = span(ze[0]), yr = span(ze[1]), nzx = xr.length, nz = nzx * yr.length;
    // the sector each rover surveys (two rovers on one sector: the second outline inset)
    var inset = {};
    roverKeys().forEach(function (rk) {
      var t = rtick(rk), k = hasOpt(t) ? sectorOf(t.option) : -1;
      if (k < 0) return;
      var r = [xr[k % nzx], yr[Math.floor(k / nzx)]], d = 1.5 + 4 * (inset[k] || 0);
      inset[k] = (inset[k] || 0) + 1;
      g.fillStyle = optColor(t.option); g.globalAlpha = 0.2 / inset[k]; g.fillRect(m.X(r[0][0]), m.Y(r[1][1]), (r[0][1] - r[0][0]) * m.sc, (r[1][1] - r[1][0]) * m.sc); g.globalAlpha = 1;
      g.strokeStyle = multi() ? roverColor(rk) : optColor(t.option); g.lineWidth = 3;
      g.strokeRect(m.X(r[0][0]) + d, m.Y(r[1][1]) + d, (r[0][1] - r[0][0]) * m.sc - 2 * d, (r[1][1] - r[1][0]) * m.sc - 2 * d);
    });
    g.strokeStyle = css("--fg"); g.globalAlpha = 0.7; g.lineWidth = 1.4; g.setLineDash([6, 4]); g.beginPath();
    ze[0].slice(1, -1).forEach(function (x) { g.moveTo(m.X(x), m.Y(f[1][0])); g.lineTo(m.X(x), m.Y(f[1][1])); });
    ze[1].slice(1, -1).forEach(function (y) { g.moveTo(m.X(f[0][0]), m.Y(y)); g.lineTo(m.X(f[0][1]), m.Y(y)); });
    g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
    // the charging places, one per charge option, as stars, in a two-tier setting only; a place a
    // rover's charge option in force drives to (its charge_to) is filled in that option's colour
    if (twoLevel()) {
      var sites = chargeSites(), extra = [];
      roverKeys().forEach(function (rk) {
        var t = rtick(rk);
        if (t.charge_to && !sites.some(function (s) { return Math.abs(s[0] - t.charge_to[0]) < 1e-6 && Math.abs(s[1] - t.charge_to[1]) < 1e-6; })) extra.push(t.charge_to);
      });
      sites.concat(extra).forEach(function (cs, si) {
        var cx = m.X(cs[0]), cyy = m.Y(cs[1]), on = null;
        roverKeys().forEach(function (rk) { var t = rtick(rk); if (t.charge_to && Math.abs(t.charge_to[0] - cs[0]) < 1e-6 && Math.abs(t.charge_to[1] - cs[1]) < 1e-6) on = t; });
        // the sun: a ring and eight rays (in the option's colour, the ring filled, while a charge
        // option drives to the place)
        g.beginPath(); g.arc(cx, cyy, 7, 0, 2 * Math.PI);
        if (on) { g.fillStyle = optColor(on.option); g.globalAlpha = 0.85; g.fill(); g.globalAlpha = 1; }
        g.strokeStyle = on ? optColor(on.option) : css("--sun"); g.lineWidth = on ? 2.5 : 1.8; g.stroke();
        for (var a = 0; a < 8; a++) { var an = a * Math.PI / 4; g.beginPath(); g.moveTo(cx + 9 * Math.cos(an), cyy + 9 * Math.sin(an)); g.lineTo(cx + 13 * Math.cos(an), cyy + 13 * Math.sin(an)); g.stroke(); }
        if (si < sites.length && sites.length > 1) {
          var txt = "C" + (si + 1); g.font = "600 10.5px system-ui, sans-serif"; g.textAlign = "left";
          g.lineWidth = 3; g.strokeStyle = css("--panel"); g.strokeText(txt, cx + 15, cyy + 4); g.fillStyle = css("--fg"); g.fillText(txt, cx + 15, cyy + 4);
        }
      });
    }
    g.font = "600 " + (nz > 4 ? 11 : 12) + "px system-ui, sans-serif"; g.textAlign = "left";
    for (var z = 0; z < nz; z++) {
      var rx = xr[z % nzx][0], ry = yr[Math.floor(z / nzx)][1];
      var tx = m.X(rx) + 6, ty = m.Y(ry) + 16, txt = (nz > 4 ? "S" : "sector ") + (z + 1);
      g.lineWidth = 3; g.strokeStyle = css("--panel"); g.strokeText(txt, tx, ty); g.fillStyle = css("--fg"); g.fillText(txt, tx, ty);
    }
  }
  function ellipse(m, comp, color, alpha) {
    var g = m.g, a = comp.cov[0][0], b = comp.cov[0][1], d = comp.cov[1][1];
    var tr = a + d, det = a * d - b * b, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    var l1 = tr / 2 + disc, l2 = Math.max(tr / 2 - disc, 1e-10), th = 0.5 * Math.atan2(2 * b, a - d);
    g.beginPath(); g.ellipse(m.X(comp.mu[0]), m.Y(comp.mu[1]), Math.max(4, 2 * Math.sqrt(l1) * m.sc), Math.max(4, 2 * Math.sqrt(l2) * m.sc), -th, 0, 2 * Math.PI);
    g.globalAlpha = alpha * 0.35; g.fillStyle = color; g.fill();
    g.globalAlpha = Math.min(1, alpha * 1.2 + 0.2); g.strokeStyle = color; g.lineWidth = 1.5; g.stroke();
    g.globalAlpha = 1;
  }
  function planPath(t, p) {
    var dy = B.meta.dynamics, gain = dy.gain, keep = dy.keep, dt = dy.dt;
    if (dy.kind === "vehicle" && t.belief.heading != null) {
      // the vehicle's rule: move along the heading at the speed, then turn, and move the speed
      // by the command (the bundle's command box, [-1, 1] on both when it states none)
      var th = t.belief.heading, sp = t.belief.speed || 0, turn = dy.turn_max_degrees * Math.PI / 180;
      var lo = dy.command_low || [-1, -1], hi = dy.command_high || [1, 1];
      var px = p.start[0], py = p.start[1], path = [[px, py]];
      // near empty the drive delivers a share of the command (the bundle's stall); the share at
      // the believed level is held over the plan (bundles with no stall: one)
      var st = dy.stall, lv = t.belief.energy_level;
      var share = st && lv != null ? 1 / (1 + Math.exp(-(lv - st.level) / st.width)) : 1;
      p.v_plan.forEach(function (a) {
        px += dt * sp * Math.cos(th); py += dt * sp * Math.sin(th);
        th += dt * turn * share * clamp(a[1], lo[1], hi[1]); sp = keep * sp + gain * share * clamp(a[0], lo[0], hi[0]);
        path.push([px, py]);
      });
      return path;
    }
    var terr = argmax(t.belief.terrain), v = (t.belief.velocity || [0, 0]).slice(), x = p.start[0], y = p.start[1], pts = [[x, y]];
    p.v_plan.forEach(function (a) {
      x += dt * v[0]; y += dt * v[1];
      v = [keep[terr] * v[0] + gain[terr] * a[0], keep[terr] * v[1] + gain[terr] * a[1]];
      pts.push([x, y]);
    });
    return pts;
  }
  function drawPlans(m, t, color) {
    // the rover's plans, the chosen one bold: in the page's plan colour (one rover), or each
    // rover's plans in its own colour (``color``, two rovers)
    var g = m.g;
    t.policies.forEach(function (p, i) {
      var ch = i === t.chosen, pts = planPath(t, p), cc = color || css("--plan");
      g.beginPath(); pts.forEach(function (q, k) { k ? g.lineTo(m.X(q[0]), m.Y(q[1])) : g.moveTo(m.X(q[0]), m.Y(q[1])); });
      g.globalAlpha = ch ? 1 : color ? 0.55 : 0.6; g.strokeStyle = ch ? cc : color || css("--fg"); g.lineWidth = ch ? 4 : color ? 1.5 : 1.8; g.lineCap = "round"; g.lineJoin = "round"; g.stroke();
      if (ch) { var e = pts[pts.length - 1]; g.beginPath(); g.arc(m.X(e[0]), m.Y(e[1]), 4, 0, 2 * Math.PI); g.fillStyle = cc; g.fill(); }
      g.globalAlpha = 1;
    });
  }
  function replaySettings() { return B.settings.filter(function (s) { return replayable(s.id); }); }
  function defaultSetting() {
    // the two-tier run when the bundle has one, else the first replayable setting; the hash's
    // setting= overrides
    var rs = replaySettings(), tiered = rs.filter(function (s) { return s.tiers === 2; })[0];
    return tiered ? tiered.id : rs[0].id;
  }
  function hasHeading() { return rows()[0].true_heading != null; }  // the world's heading per tick
  function stride() {
    // ticks (array entries) advanced per step while playing: the bundle's play stride over the
    // ticks it keeps (meta.trim), one otherwise
    var tr = B.meta.trim;
    return tr ? Math.max(1, Math.round((tr.play_stride || 1) / (tr.keep_every || 1))) : 1;
  }
  function interp(r, i, f) {
    // the true state at tick i plus a fraction f of the way to the next tick played (stride()
    // ticks on): position and velocity, and as a fifth entry the heading, turned along the shorter
    // arc (when the bundle has one)
    var a = r[i], b = i + stride() < r.length && f > 0 ? r[i + stride()] : a;
    var out = a.true_x.map(function (v, k) { return v + f * (b.true_x[k] - v); });
    if (a.true_heading != null) {
      var d = b.true_heading - a.true_heading; d = Math.atan2(Math.sin(d), Math.cos(d));
      out.push(a.true_heading + f * d);
    }
    return out;
  }
  function drawRover(m, xs, color, label) {
    var g = m.g, x = m.X(xs[0]), y = m.Y(xs[1]), vx = xs[2], vy = xs[3], sp = Math.hypot(vx, vy);
    g.save(); g.translate(x, y);
    // the arrow points along the heading when the bundle records one, else along the velocity
    if (xs.length > 4) g.rotate(-xs[4]);
    else if (sp > 0.03) { var ang = Math.atan2(-vy, vx); g.rotate(ang); }
    g.beginPath(); g.moveTo(13, 0); g.lineTo(-7, -7); g.lineTo(-3, 0); g.lineTo(-7, 7); g.closePath();
    g.fillStyle = color || css("--fg"); g.fill(); g.strokeStyle = css("--bg"); g.lineWidth = 2; g.stroke();
    g.restore();
    if (label) {  // the rover's number, beside it
      g.font = "700 11px system-ui, sans-serif"; g.textAlign = "left";
      g.lineWidth = 3; g.strokeStyle = css("--panel"); g.strokeText(label, x + 10, y - 9); g.fillStyle = color || css("--fg"); g.fillText(label, x + 10, y - 9);
    }
  }
  function drawBeliefHeading(m, b, color) {
    // the believed heading: a line from the believed position, with a wedge of two sd either side
    var g = m.g, x = m.X(b.mean[0]), y = m.Y(b.mean[1]), r = 26, w = Math.min(2 * b.heading_sd, Math.PI);
    g.save(); g.globalAlpha = 0.25; g.fillStyle = color || css("--ell"); g.beginPath(); g.moveTo(x, y);
    g.arc(x, y, r, -b.heading - w, -b.heading + w); g.closePath(); g.fill();
    g.globalAlpha = 1; g.strokeStyle = color || css("--ell"); g.lineWidth = 2; g.beginPath(); g.moveTo(x, y);
    g.lineTo(x + r * Math.cos(-b.heading), y + r * Math.sin(-b.heading)); g.stroke(); g.restore();
  }
  function drawField(lite) {
    // lite: a frame between ticks while playing (the rovers move on; panels below stay). Each
    // rover: its true track (dotted), its fix, its plans (the chosen one bold), its arrow, its
    // position belief and believed heading; one rover is drawn in the page's colours, two each in
    // its own
    var c = $("field"), m = setupCanvas(c), g = m.g, sh = tick(), keys = roverKeys(), two = multi(), f = B.meta.field;
    g.drawImage(background(m), 0, 0, m.w, m.h);
    drawMap(m, sh); drawSectors(m);
    keys.forEach(function (rk) {
      var r = rows(rk), t = r[S.t], xs = interp(r, S.t, S.frac), col = two ? roverColor(rk) : null;
      g.save(); g.setLineDash([2, 5]); g.lineCap = "round"; g.globalAlpha = 0.7; g.strokeStyle = col || css("--fg"); g.lineWidth = two ? 1.6 : 1.4; g.beginPath();
      for (var i = 0; i <= S.t; i++) { var p = r[i].true_x; i ? g.lineTo(m.X(p[0]), m.Y(p[1])) : g.moveTo(m.X(p[0]), m.Y(p[1])); }
      if (S.frac > 0) g.lineTo(m.X(xs[0]), m.Y(xs[1]));
      g.stroke(); g.restore();
      // a rover charging: a line from it to the charging place its option drives to
      if (t.charge_to && twoLevel()) {
        g.save(); g.setLineDash([5, 4]); g.globalAlpha = 0.9; g.strokeStyle = optColor(t.option); g.lineWidth = 2; g.beginPath();
        g.moveTo(m.X(xs[0]), m.Y(xs[1])); g.lineTo(m.X(t.charge_to[0]), m.Y(t.charge_to[1])); g.stroke(); g.restore();
      }
      g.save(); g.beginPath(); g.rect(m.X(f[0][0]), m.Y(f[1][1]), (f[0][1] - f[0][0]) * m.sc, (f[1][1] - f[1][0]) * m.sc); g.clip();
      if (t.policies) drawPlans(m, t, col);  // a trimmed bundle keeps the plans on a subset of ticks
      g.restore();
      if (t.fix) {
        var fx = m.X(t.fix[0]), fy = m.Y(t.fix[1]);
        g.strokeStyle = col || css("--c2"); g.lineWidth = 2.8; g.lineCap = "round"; g.beginPath();
        g.moveTo(fx - 4.5, fy - 4.5); g.lineTo(fx + 4.5, fy + 4.5); g.moveTo(fx - 4.5, fy + 4.5); g.lineTo(fx + 4.5, fy - 4.5); g.stroke(); g.lineCap = "butt";
      }
      drawRover(m, xs, col, two ? String(roverNo(rk)) : null);
      t.belief.components.forEach(function (cm) { ellipse(m, cm, col || css("--ell"), cm.w); });
      if (t.belief.heading != null && t.belief.heading_sd != null) drawBeliefHeading(m, t.belief, col);
    });
    if (lite) return;
    drawHud();
    drawCamera();
  }
  function energyBar(e) {
    var nb = nBands(), band = e.band, bar = "<span class='hbar' title='energy level " + e.level.toFixed(2) + " of " + energyMax() + "'>";
    for (var b = 0; b < nb; b++) bar += "<i style='width:" + (100 / nb) + "%;background:" + (b <= band ? css("--c3") : "transparent") + ";opacity:" + (b < band ? 0.55 : 1) + "'></i>";
    return bar + "</span>";
  }
  function roverMarks(colors) {
    // the field's marks drawn for a rover (its plans, position belief, fixes and true path), one
    // glyph per colour given: in the rover's own colour under its name (two rovers), or the
    // page's colours in the field's legend (one rover)
    return "<span>" + lineSw(colors.plan) + "plans</span>" +
      "<span>" + circSw(colors.ell) + "position belief, 2&sigma;</span>" +
      "<span>" + xSw(colors.fix) + "satellite fix</span>" +
      "<span>" + lineSw(colors.path, "1.5 3.5", 1.8) + "true path</span>";
  }
  function hudRover(t, rk) {
    // one rover's option in force, energy and terrain, as HUD rows; with two rovers the rows
    // open with the rover's marks on the field, in its colour, and run energy, option, terrain
    var e = t.true_energy, terr = t.belief.terrain, best = argmax(terr);
    var pinned = !twoLevel() && pinnedOpt() != null;
    var opt = hasOpt(t) ? optLabel(t.option) + (pinned ? " (pinned)" : "") : aloneText();
    var option = "<div><span class='hl'>option in force</span><b>" + esc(opt) + "</b></div>",
      energy = "<div><span class='hl'>energy</span>" + energyBar(e) + " <b>" + esc(nm("energy", e.band)) + "</b> " + e.level.toFixed(2) + (t.belief.energy_level != null ? ", believed " + t.belief.energy_level.toFixed(2) + " &plusmn; " + (2 * t.belief.energy_sd).toFixed(2) : "") + (t.stalled ? " <span class='approx'>stalled</span>" : "") + "</div>",
      terrain = "<div><span class='hl'>terrain under the rover</span>true <b>" + esc(nm("terrain", t.true_terrain)) + "</b>, believed " + esc(nm("terrain", best)) + " " + terr[best].toFixed(2) + "</div>";
    if (rk == null) return option + energy + terrain;
    var c = [roverColor(rk)];
    return "<div class='legend rmarks'>" + roverMarks({ plan: c, ell: c, fix: c, path: c }) + "</div>" + energy + option + terrain;
  }
  function drawHud() {
    // the tick; per rover (two rovers: a labelled group each) its option, energy and terrain; the
    // coverage of the (shared) field
    var sh = tick(), two = multi(), t0 = rtick(roverKeys()[0]);
    var h = "<div><span class='hl'>tick</span><b>" + sh.t + "</b></div>";
    if (two) roverKeys().forEach(function (rk) {
      h += "<div class='hrover' style='border-left-color:" + roverColor(rk) + "'><div class='hrname' style='color:" + roverColor(rk) + "'>" + esc(roverName(rk)) + "</div>" + hudRover(rtick(rk), rk) + "</div>";
    });
    else h += hudRover(t0);
    $("hud").innerHTML = h + "<div><span class='hl'>coverage" + (two ? " (shared field)" : "") + "</span><b>" + sh.coverage.toFixed(2) + "</b> of cells mapped, <b>" + sh.seen.toFixed(2) + "</b> seen</div>";
    var sw = function (c) { return "<i style='background:" + c + "'></i>"; };
    // the field's legend: the map's colours and the charge target; with one rover its marks too,
    // in the page's colours (with two rovers each rover's marks sit under its name above)
    $("fieldlegend").innerHTML =
      "<span>learned map: " + [0, 1, 2].map(function (c) { return sw(terrColor(c)) + esc(nm("terrain", c)); }).join(" ") + "</span>" +
      "<span>true ground: " + [0, 1, 2].map(function (c) { return sw(terrBg(c)) + esc(nm("terrain", c)); }).join(" ") + "</span>" +
      (two ? "" : roverMarks({ plan: [css("--plan")], ell: [css("--ell")], fix: [css("--c2")], path: [css("--fg")] })) +
      (twoLevel() ? "<span>" + sunSw(css("--sun")) + "Charge option target</span>" : "") +
      (S.layer === "light" ? "<span>" + (B.meta.world.sun ? "brighter: more solar yield (where parking charges)" : "brighter: more sunlight") + "</span>" : S.layer === "fix" ? "<span>purple: noisier fix (terrain blocks the sky)</span>" : "");
  }

  // ---- the camera ----
  function bytes(x, n) {
    if (typeof x !== "string") { var o = new Uint8Array(n * n); for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) o[i * n + j] = x[i][j]; return o; }
    var s = atob(x), out = new Uint8Array(s.length);
    for (var k = 0; k < s.length; k++) out[k] = s.charCodeAt(k);
    return out;
  }
  function paint(c, data, n) {
    var g = c.getContext("2d"), im = g.createImageData(n, n);
    for (var i = 0; i < n * n; i++) { im.data[4 * i] = im.data[4 * i + 1] = im.data[4 * i + 2] = data[i]; im.data[4 * i + 3] = 255; }
    g.putImageData(im, 0, 0);
  }
  function hexRgb(c) {
    var m = /^#([0-9a-f]{6})$/i.exec(c); if (!m) return [31, 95, 191];
    var v = parseInt(m[1], 16); return [v >> 16 & 255, v >> 8 & 255, v & 255];
  }
  function heatColor(p) {
    // a sequential scale on the page's accent: near white at 0, the accent at 0.5, dark at 1
    var a = hexRgb(css("--c1")), lo = a.map(function (v) { return v + (255 - v) * 0.9; }), hi = a.map(function (v) { return v * 0.35; });
    var q = clamp(p, 0, 1), f = q < 0.5 ? q * 2 : (q - 0.5) * 2, x = q < 0.5 ? lo : a, y = q < 0.5 ? a : hi;
    return "rgb(" + x.map(function (v, k) { return Math.round(v + (y[k] - v) * f); }).join(",") + ")";
  }
  function drawCamera() {
    // one rover: its observed image, the decoder's prediction and the terrain readout; two rovers:
    // the same per rover, side by side (rover 1 | rover 2), each with its own numbers
    var host = $("camgrid"), two = multi(), mode = two ? "two" : "one";
    if (host.getAttribute("data-mode") !== mode) {
      host.innerHTML = ""; host.setAttribute("data-mode", mode); host.classList.toggle("camtwo", two);
      if (two) roverKeys().forEach(function (rk) {
        var g = document.createElement("div"); g.className = "camrover"; g.style.borderTopColor = roverColor(rk);
        g.innerHTML = "<div class='camhead' style='color:" + roverColor(rk) + "'>" + esc(roverName(rk)) + "</div><div class='camcells' id='camcells_" + roverNo(rk) + "'></div>";
        host.appendChild(g);
      });
    }
    roverKeys().forEach(function (rk) { cameraCells(two ? $("camcells_" + roverNo(rk)) : host, rk ? "_" + roverNo(rk) : "", rk); });
  }
  function cameraCells(host, sx, rk) {
    var n = (B.meta.image && B.meta.image.side) || 48, t = rtick(rk);
    t = t.img != null ? t : heavy("img", null, rk);
    if (!host.children.length) {
      [["observed image", "obs"], ["decoder prediction at the belief", "rec"], ["terrain readout", "read"]].forEach(function (p, i) {
        var d = document.createElement("div"); d.className = "cam";
        // a fixed sky band is marked only for a camera whose top quarter is always sky
        var sky = i === 0 && !hasHeading() ? "<span class='skylab'>sky</span>" : "";
        d.innerHTML = (i < 2 ? "<div class='camc'><canvas id='cam" + i + sx + "' width='" + n + "' height='" + n + "' role='img' aria-label='" + p[0] + (rk ? ", " + roverName(rk) : "") + "'></canvas>" + sky + "</div>" : "<div class='readout' id='cam2" + sx + "' role='img' aria-label='" + p[0] + "'></div>") +
          "<span class='cl'>" + p[0] + "</span><span class='cv' id='camv" + i + sx + "'></span>";
        host.appendChild(d);
      });
    }
    paint($("cam0" + sx), bytes(t.img, n), n);
    // a model that observes the recogniser's reading has no decoder: its panel stays blank
    if (t.recon != null || !t.recogniser) paint($("cam1" + sx), bytes(t.recon || t.img, n), n);
    else $("cam1" + sx).getContext("2d").clearRect(0, 0, n, n);
    // the belief over the terrain classes as a heat strip, one cell per class in the model's order,
    // shaded by probability; a black dot marks the true class
    var h = "<div class='heat'>";
    t.belief.terrain.forEach(function (p, i) {
      var isTrue = i === t.true_terrain, fill = heatColor(p), ink = p > 0.45 ? "#fff" : "#1d1f21";
      h += "<div class='hc' title='" + esc(nm("terrain", i)) + ": " + p.toFixed(2) + (isTrue ? ", the true class" : "") + "'>" +
        "<span class='hcell' style='background:" + fill + ";color:" + ink + "'>" + (isTrue ? "<i class='hdot'></i>" : "") +
        "<span class='hp'>" + p.toFixed(2) + "</span></span><span class='rn" + (isTrue ? " true" : "") + "'>" + esc(nm("terrain", i)) + "</span></div>";
    });
    h += "</div><div class='hkey'><i class='hdot'></i>true class; darker: more probable</div>";
    if (t.recogniser && t.look != null) {
      // the recogniser's reading, shown only when the model observes it in place of the picture
      var rk2 = argmax(t.recogniser);
      h += "<div class='rn'>recogniser: " + esc(nm("terrain", t.look != null ? t.look : rk2)) + " " + t.recogniser[rk2].toFixed(2) + "</div>";
    }
    $("cam2" + sx).innerHTML = h;
    var cam = t.camera || {};
    $("camv1" + sx).textContent = (t.recon == null && t.recogniser ? "no decoder: the model observes the recogniser's reading" : "") +
      (cam.log_evidence != null ? "log evidence " + cam.log_evidence.toFixed(0) : "") +
      (cam.scale_mean != null ? ", noise sd " + cam.scale_mean.toFixed(2) : "");
    var fixed = B.meta.image && B.meta.image.camera_heading_degrees != null;
    if (t.true_heading == null || fixed) $("camv0" + sx).textContent = "looking along +y";
    else {
      // the rover's heading, in degrees counter-clockwise from +x
      var deg = Math.round(t.true_heading * 180 / Math.PI);
      $("camv0" + sx).textContent = "heading " + (deg < 0 ? deg + 360 : deg) + "° from +x";
    }
  }

  // ---- timeline ----
  function reserve(host) {
    // keep a rebuilt chart at least as tall as it was, so the pane never shrinks under the scroll
    // position while the replay plays (the browser would clamp the scroll to the top)
    var h = Math.max(host.offsetHeight, parseFloat(host.style.minHeight) || 0);
    if (h) host.style.minHeight = h + "px";
    host.innerHTML = "";
  }
  function drawTimeline() {
    // top to bottom: per rover the option in force; per rover the energy level against the bands;
    // the coverage of the field over the run, solid for the selected setting and dashed for each
    // other run of the version (the coverage-only union arm included)
    var host = $("timeline"), T = run().ticks, n = T.length, C = cache(), keys = roverKeys(), two = multi();
    var others = Object.keys(B.runs).filter(function (k) { return k !== S.setting; });
    var W = Math.max(300, host.clientWidth - 16), L = W < 480 ? 46 : 64, R = 40, sx = function (i) { return L + i / n * (W - L - R); };
    reserve(host);
    var nb = nBands(), yS = 14, hS = 26, gS = 6, yS1 = yS + keys.length * (hS + gS) - gS;
    var hE = two ? Math.max(60, 10 * nb) : Math.max(76, 14 * nb), yE0 = yS1 + 42, gE = 30;
    var yEk = function (k) { return yE0 + k * (hE + gE); };
    var yC = yEk(keys.length - 1) + hE + 36, hC = 100, H = yC + hC + 28;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img", "aria-label": "Timeline of the option in force, the energy level and coverage" }, host);
    svg.style.height = "auto";
    keys.forEach(function (rk, ki) {
      var y0 = yS + ki * (hS + gS);
      var la = el("text", { x: 4, y: y0 + 17, "font-size": 10.5, class: "muted" }, svg, two ? roverName(rk) : "option");
      if (two) { la.removeAttribute("class"); la.setAttribute("fill", roverColor(rk)); la.setAttribute("font-weight", 600); }
      C.segs[rk].forEach(function (s) {
        var x0 = sx(s.t0), x1 = sx(s.t1);
        if (s.option == null) {
          el("rect", { x: x0, y: y0, width: x1 - x0, height: hS, fill: "none", stroke: css("--line"), "stroke-width": 1, "stroke-dasharray": "4 3" }, svg);
          el("text", { x: (x0 + x1) / 2, y: y0 + 17, "text-anchor": "middle", "font-size": 10.5, class: "muted" }, svg, "no option: the rover runs alone, with no scheduler");
          return;
        }
        el("rect", { x: x0, y: y0, width: x1 - x0, height: hS, fill: optColor(s.option), "fill-opacity": 0.75, stroke: css("--panel"), "stroke-width": 1 }, svg);
        var txt = (!twoLevel() && s.option === pinnedOpt()) ? "pinned: " + optLabel(s.option) : (x1 - x0 > 16 ? optShort(s.option) : "");
        if (txt) el("title", {}, el("text", { x: (x0 + x1) / 2, y: y0 + 17, "text-anchor": "middle", "font-size": 10.5, fill: css("--fg") }, svg, txt), optLabel(s.option));
      });
    });
    if (twoLevel()) T.forEach(function (t, i) { if (t.slow && t.slow.decision) el("line", { x1: sx(i), x2: sx(i), y1: yS - 8, y2: yS, stroke: css("--fg"), "stroke-width": 1.2 }, svg); });
    el("text", { x: L, y: yC - 6, "font-size": 10.5, class: "muted" }, svg, "coverage: fraction of cells mapped" + (two ? " (the shared field)" : ""));
    // energy, one panel per rover
    var emax = energyMax(), bw = emax / nb;
    keys.forEach(function (rk, ki) {
      var yE = yEk(ki), ey = function (v) { return yE + hE - (v / emax) * hE; }, Rr = rows(rk);
      el("text", { x: L, y: yE - 9, "font-size": 10.5, class: "muted" }, svg, (two ? roverName(rk) + ": " : "") + "energy level (true), against the " + countWord(nb) + " bands");
      for (var b = 0; b <= nb; b++) el("line", { x1: L, x2: W - R, y1: ey(b * bw), y2: ey(b * bw), stroke: css("--line"), "stroke-width": 0.8 }, svg);
      for (b = 0; b < nb; b++) if (!two || b % 2 === 0 || nb <= 4) el("text", { x: L - 4, y: ey((b + 0.5) * bw) + 3, "text-anchor": "end", "font-size": 9.5, class: "muted" }, svg, nm("energy", b));
      var d = "";
      Rr.forEach(function (t, i) { d += (i ? "L" : "M") + sx(i + 0.5).toFixed(1) + "," + ey(t.true_energy.level).toFixed(1); });
      el("path", { d: d, fill: "none", stroke: two ? roverColor(rk) : css("--c3"), "stroke-width": 2 }, svg);
      el("circle", { cx: sx(S.t + 0.5), cy: ey(Rr[S.t].true_energy.level), r: 3.5, fill: two ? roverColor(rk) : css("--c3"), stroke: css("--bg"), "stroke-width": 1.5 }, svg);
    });
    // coverage
    var top = 0; [S.setting].concat(others).forEach(function (k) { B.runs[k].ticks.forEach(function (t) { top = Math.max(top, t.coverage); }); });
    var cmax = Math.max(0.5, Math.ceil(top * 4) / 4), cy = function (v) { return yC + hC - clamp(v / cmax, 0, 1) * hC; }, ticks = [];
    for (var cv = 0; cv <= cmax + 1e-9; cv += 0.25) ticks.push(cv);
    ticks.forEach(function (v) {
      el("line", { x1: L, x2: W - R, y1: cy(v), y2: cy(v), stroke: css("--line"), "stroke-width": 0.8 }, svg);
      el("text", { x: L - 4, y: cy(v) + 3, "text-anchor": "end", "font-size": 9.5, class: "muted" }, svg, v.toFixed(2));
    });
    var ends = [];
    function curve(ts, color, dash, w) {
      var s = ""; ts.forEach(function (t, i) { s += (i ? "L" : "M") + sx(i + 0.5).toFixed(1) + "," + cy(t.coverage).toFixed(1); });
      el("path", { d: s, fill: "none", stroke: color, "stroke-width": w, "stroke-dasharray": dash }, svg);
      // the end value, placed with the other curves' (placeLabels: in value order, apart)
      var e = ts[ts.length - 1];
      ends.push({ y: cy(e.coverage) + 3.5, value: e.coverage, text: e.coverage.toFixed(2), color: color });
    }
    var DASH = ["5 4", "2 3", "8 3 2 3"], OCOL = ["--muted", "--c4", "--c5"];
    others.forEach(function (k, i) { curve(B.runs[k].ticks, css(OCOL[i % 3]), DASH[i % 3], 2); });
    curve(T, css("--accent"), "", 2.4);
    placeLabels(svg, ends, W - R + 4);
    var every = [100, 200, 500, 1000, 2000, 5000].filter(function (v) { return n / v <= 10; })[0] || 10000;
    for (var k = 0; k <= n; k += every) {
      el("line", { x1: sx(k), x2: sx(k), y1: yC + hC, y2: yC + hC + 4, stroke: css("--muted") }, svg);
      var lab = k < n ? T[k].t : T[n - 1].t + 1;  // the recorded tick (a trimmed bundle keeps a subset)
      el("text", { x: sx(k), y: yC + hC + 16, "text-anchor": "middle", "font-size": 10, class: "muted" }, svg, k === n ? "tick " + lab : lab);
    }
    // cursor
    var cx = sx(S.t + 0.5);
    el("line", { x1: cx, x2: cx, y1: yS - 8, y2: yC + hC, stroke: css("--fg"), "stroke-width": 1.4 }, svg);
    el("circle", { cx: cx, cy: cy(T[S.t].coverage), r: 3.5, fill: css("--accent"), stroke: css("--bg"), "stroke-width": 1.5 }, svg);
    function pick(ev) {
      var r = svg.getBoundingClientRect(), x = (ev.clientX - r.left) * (W / r.width);
      S.t = clamp(Math.round((x - L) / (W - L - R) * n - 0.5), 0, n - 1); render();
    }
    svg.addEventListener("pointerdown", function (e) { svg.setPointerCapture(e.pointerId); pick(e); svg._d = true; });
    svg.addEventListener("pointermove", function (e) { if (svg._d) pick(e); });
    svg.addEventListener("pointerup", function () { svg._d = false; });
    svg.addEventListener("pointercancel", function () { svg._d = false; });
    // the legend in groups: the options (the strips' colours), the coverage curves, and with two
    // rovers the rovers (their energy curves), so a rover never reads as an option
    var rs = settingRec(), tl = "", used = function (o) { return keys.some(function (rk) { return C.segs[rk].some(function (s) { return s.option === o; }); }); };
    var opts = "";
    for (var o = 0; o < nOpts(); o++) if (used(o) || twoLevel()) opts += "<span><i style='background:" + optColor(o) + "'></i>" + esc(optShort(o)) + " " + esc(optLabel(o)) + "</span>";
    if (opts) tl += "<span class='lgroup'><span class='lghead'>options" + (two ? " (strip colours, both rovers)" : "") + "</span>" + opts + "</span>";
    tl += "<span class='lgroup'><span class='lghead'>coverage</span>";
    tl += "<span><svg class='lsw' viewBox='0 0 28 10'><line x1='1' x2='27' y1='5' y2='5' stroke='" + css("--accent") + "' stroke-width='2.4'/></svg>coverage, " + esc(rs.label.toLowerCase()) + "</span>";
    others.forEach(function (k, i) {
      var ro = settingRec(k), lb = ro ? ro.label : k;
      tl += "<span><svg class='lsw' viewBox='0 0 28 10'><line x1='1' x2='27' y1='5' y2='5' stroke='" + css(OCOL[i % 3]) + "' stroke-width='2' stroke-dasharray='" + DASH[i % 3] + "'/></svg>coverage, " + esc(lb.toLowerCase()) + "</span>";
    });
    tl += "</span>";
    tl += two
      ? "<span class='lgroup'><span class='lghead'>rovers (row labels and energy curves)</span>" + keys.map(function (rk) { return "<span><svg class='lsw' viewBox='0 0 28 10'><line x1='1' x2='27' y1='5' y2='5' stroke='" + roverColor(rk) + "' stroke-width='2'/></svg>" + esc(roverName(rk)) + "</span>"; }).join("") + "</span>"
      : "<span><svg class='lsw' viewBox='0 0 28 10'><line x1='1' x2='27' y1='5' y2='5' stroke='" + css("--c3") + "' stroke-width='2'/></svg>energy level</span>";
    $("tlegend").innerHTML = tl;
  }

  // ---- the scheduler's decision: G by first option, split into novelty and its energy term ----
  var DECR = null;
  function decRange() {
    if (DECR) return DECR;
    // over the version's two-tier run: one rover's G per first option and its kept sequences'
    // terms; two rovers' G per first option of each rover and their chosen sequences' terms
    var lo = 0, hi = 0, id = Object.keys(B.runs).filter(function (k) { var r = settingRec(k); return r && r.tiers === 2 && replayable(k); })[0];
    var grow = function (g) { if (isFinite(g)) { lo = Math.min(lo, g); hi = Math.max(hi, g); } };
    if (id) B.runs[id].ticks.forEach(function (t) {
      if (!t.slow || !t.slow.decision) return;
      if (Array.isArray(t.slow.options)) {
        t.slow.efe_by_option.forEach(function (row) { row.forEach(grow); });
        t.slow.chosen.forEach(function (p) { var s = leafSums(p); grow(s.novelty); grow(s.risk); });
        return;
      }
      t.slow.efe_by_option.forEach(grow);
      t.slow.policies.forEach(function (p) { var s = leafSums(p); lo = Math.min(lo, s.novelty); hi = Math.max(hi, s.risk); });
    });
    return (DECR = [lo, hi]);
  }
  function leafSums(p) {
    var nov = 0, risk = 0;
    Object.keys(p.terms).forEach(function (tm) {
      Object.keys(p.terms[tm]).forEach(function (lf) {
        var v = (p.terms[tm][lf][0] + p.terms[tm][lf][1]) / 2;
        if (tm === "infogain") nov += v; else risk += v;
      });
    });
    return { novelty: nov, risk: risk };
  }
  function showDecision(on) {
    // the decision figure, with the heading set directly over it and that heading's contents
    // entry, is shown only in a setting that has a scheduler (or pins an option)
    var fig = $("decfig"); if (!fig) return;
    fig.hidden = !on;
    var hd = fig.previousElementSibling;
    if (!hd || !/^H[3-6]$/.test(hd.tagName)) return;
    hd.hidden = !on;
    if (hd.id) document.querySelectorAll("a[href='#" + hd.id + "']").forEach(function (a) {
      if (a.parentNode.tagName === "LI") a.parentNode.hidden = !on;
    });
  }
  function drawDecision() {
    var host = $("decision"), note = $("decnote"), cap = $("deccap"), lg = $("declegend");
    showDecision(twoLevel() || pinnedOpt() != null);
    if (!twoLevel() && pinnedOpt() == null) return;
    reserve(host); lg.innerHTML = "";
    if (!twoLevel()) {
      note.textContent = "In this setting the option is pinned to " + optLabel(pinnedOpt()) + ", so the scheduler does not run and there is no decision to show.";
      cap.textContent = ""; host.hidden = true; return;
    }
    host.hidden = false;
    var d = decTick();
    if (multi()) { drawDecisionJoint(d); return; }
    var sgn ="Sign convention: every term enters G with its sign applied, so novelty appears negative (lower G is better) and the " + energyTerm() + " positive.";
    var chosen = d.slow.option;
    note.textContent = "The scheduler scores every sequence of " + horizonWord() + " options; each bar below is the best sequence that starts with that option, split into its two objective terms (novelty of the map, which is the information the plan expects to gain about the table, and " + (terminalEnergy() ? "the energy at the plan's end, scored once against a target that penalises only the empty band" : "energy risk") + "), with a mark at their sum, G.";
    var rg = decRange(), lo = rg[0], hi = rg[1], pad = (hi - lo) * 0.05; lo -= pad; hi += pad;
    var no = nOpts(), Wd = Math.max(300, host.clientWidth - 16), Lx = Wd < 520 ? 96 : 168, Rx = 70, rowH = 34, H = 10 + no * rowH + 30;
    var sx = function (v) { return Lx + (v - lo) / (hi - lo) * (Wd - Lx - Rx); };
    var svg = el("svg", { viewBox: "0 0 " + Wd + " " + H, width: Wd, height: H, role: "img", "aria-label": "Expected free energy by first option, split into novelty and " + energyTerm() }, host);
    svg.style.height = "auto";
    var step = Math.pow(10, Math.floor(Math.log10((hi - lo) / 4))); step *= (hi - lo) / step / 4 > 4 ? 5 : (hi - lo) / step / 4 > 2 ? 2 : 1;
    for (var v = Math.ceil(lo / step) * step; v <= hi; v += step) {
      el("line", { x1: sx(v), x2: sx(v), y1: 6, y2: H - 24, stroke: css("--line"), "stroke-width": v === 0 ? 1.6 : 0.8 }, svg);
      el("text", { x: sx(v), y: H - 8, "text-anchor": "middle", "font-size": 10.5, class: "muted" }, svg, Math.round(v * 100) / 100);
    }
    var best = {};
    d.slow.policies.forEach(function (p) {
      var k = p.options[0], m = (p.G[0] + p.G[1]) / 2;
      if (!best[k] || m < best[k].m) best[k] = { p: p, m: m };
    });
    var missing = false;
    for (var k = 0; k < no; k++) {
      var y = 10 + k * rowH, ch = k === chosen, G = d.slow.efe_by_option[k], b = best[k];
      if (ch) el("rect", { x: 1, y: y, width: Wd - 2, height: rowH - 2, rx: 3, fill: "none", stroke: css("--accent"), "stroke-width": 1.5 }, svg);
      var name = optShort(k);
      el("title", {}, el("text", { x: 8, y: y + rowH / 2 + 0, "font-size": 11.5, "font-weight": ch ? 700 : 400 }, svg, (ch ? "▸ " : "") + name), optLabel(k));
      if (b) {
        var s = leafSums(b.p), then = b.p.options.slice(1).map(optShort).join(", ");
        el("text", { x: 8, y: y + rowH / 2 + 12, "font-size": 9.5, class: "muted" }, svg, "then " + then);
        el("rect", { x: Math.min(sx(s.novelty), sx(0)), y: y + 6, width: Math.abs(sx(0) - sx(s.novelty)), height: rowH - 16, fill: termColor("infogain"), "fill-opacity": 0.8 }, svg);
        el("rect", { x: Math.min(sx(s.risk), sx(0)), y: y + 6, width: Math.max(1, Math.abs(sx(s.risk) - sx(0))), height: rowH - 16, fill: termColor("risk"), "fill-opacity": 0.85 }, svg);
      } else {
        missing = true;
        el("text", { x: 8, y: y + rowH / 2 + 12, "font-size": 9.5, class: "muted" }, svg, "terms not kept");
        el("rect", { x: Math.min(sx(G), sx(0)), y: y + 6, width: Math.abs(sx(0) - sx(G)), height: rowH - 16, fill: css("--muted"), "fill-opacity": 0.3, stroke: css("--muted"), "stroke-dasharray": "3 2" }, svg);
      }
      el("path", { d: "M" + sx(G) + "," + (y + 3) + " l5,11 l-5,11 l-5,-11 z", fill: css("--fg"), stroke: css("--bg"), "stroke-width": 1 }, svg);
      el("text", { x: Wd - Rx + 8, y: y + rowH / 2 + 4, "font-size": 11, "font-family": "ui-monospace,Menlo,monospace" }, svg, "G " + fmt(G, 1).replace("+", ""));
    }
    lg.innerHTML = "<span><i style='background:" + termColor("infogain") + "'></i>novelty of the map</span><span><i style='background:" + termColor("risk") + "'></i>" + energyTerm() + "</span><span>&#9670; G, the sum</span><span>outlined: the option chosen</span>";
    cap.textContent = sgn + " Sequences are scored over the " + horizonWord() + " options of the plan; the chart's axis is fixed over the run so decisions can be compared." + " Each bar is the best sequence that starts with that option.";
    drawEnergyTable(d);
  }
  // the joint decision of two rovers' scheduler (the factorised planner): per rover, the best G
  // per first option of its own sequences with that option's probability, the sequence its
  // posterior puts highest with its terms, and its energy band belief; the last sweep's movement
  function drawDecisionJoint(d) {
    var host = $("decision"), note = $("decnote"), cap = $("deccap"), lg = $("declegend");
    var keys = roverKeys(), sl = d && d.slow, H0 = sl && sl.policies ? sl.policies : [];
    note.textContent = "The scheduler decides for both rovers at once with the factorised planner: one posterior per rover over its own sequences of " + horizonWord() + " options" + (H0.length ? " (" + H0.join(" and ") + " sequences)" : "") + ", each updated against the other rover's option marginals by mean-field sweeps. Per rover, the first row is the sequence its posterior puts highest, split into its two objective terms (novelty of the map and " + (terminalEnergy() ? "the energy at the plan's end" : "energy risk") + ") with a mark at their sum, G; the rows under it are the best G of the rover's sequences that start with each option, with that option's probability.";
    if (!sl) { host.textContent = "No decision yet."; cap.textContent = ""; return; }
    var rg = decRange(), lo = rg[0], hi = rg[1], pad = (hi - lo) * 0.05; lo -= pad; hi += pad;
    var no = nOpts(), Wd = Math.max(300, host.clientWidth - 16), Lx = Wd < 520 ? 96 : 168, Rx = 116;
    var sx = function (v) { return Lx + (v - lo) / (hi - lo) * (Wd - Lx - Rx); };
    var step = Math.pow(10, Math.floor(Math.log10((hi - lo) / 4))); step *= (hi - lo) / step / 4 > 4 ? 5 : (hi - lo) / step / 4 > 2 ? 2 : 1;
    keys.forEach(function (rk, k) {
      var ch = sl.chosen[k], opt = sl.options[k], G = sl.efe_by_option[k], q = sl.q_by_option[k];
      var box = document.createElement("div"); box.className = "decrover"; host.appendChild(box);
      var hd = document.createElement("div"); hd.className = "dechead";
      hd.innerHTML = "<b style='color:" + roverColor(rk) + "'>" + esc(roverName(rk)) + "</b>: option chosen <b title='" + esc(optLabel(opt)) + "'>" + esc(optShort(opt)) + "</b>; cell read " + (sl.cells && sl.cells[k] != null ? sl.cells[k] : "none");
      box.appendChild(hd);
      var rowH = 26, chH = 36, H = 8 + chH + no * rowH + 28;
      var svg = el("svg", { viewBox: "0 0 " + Wd + " " + H, width: Wd, height: H, role: "img", "aria-label": roverName(rk) + ": expected free energy by first option, and the chosen sequence's terms" }, box);
      svg.style.height = "auto";
      for (var v = Math.ceil(lo / step) * step; v <= hi; v += step) {
        el("line", { x1: sx(v), x2: sx(v), y1: 6, y2: H - 24, stroke: css("--line"), "stroke-width": v === 0 ? 1.6 : 0.8 }, svg);
        el("text", { x: sx(v), y: H - 8, "text-anchor": "middle", "font-size": 10.5, class: "muted" }, svg, Math.round(v * 100) / 100);
      }
      // the chosen sequence: its terms stacked from zero and G
      var y = 8, s = leafSums(ch), gm = (ch.G[0] + ch.G[1]) / 2;
      el("rect", { x: 1, y: y, width: Wd - 2, height: chH - 4, rx: 3, fill: "none", stroke: css("--accent"), "stroke-width": 1.5 }, svg);
      el("text", { x: 8, y: y + 14, "font-size": 11.5, "font-weight": 700 }, svg, ch.options.map(optShort).join(" → "));
      el("text", { x: 8, y: y + 27, "font-size": 9.5, class: "muted" }, svg, "highest posterior");
      el("rect", { x: Math.min(sx(s.novelty), sx(0)), y: y + 6, width: Math.abs(sx(0) - sx(s.novelty)), height: chH - 16, fill: termColor("infogain"), "fill-opacity": 0.8 }, svg);
      el("rect", { x: Math.min(sx(s.risk), sx(0)), y: y + 6, width: Math.max(1, Math.abs(sx(s.risk) - sx(0))), height: chH - 16, fill: termColor("risk"), "fill-opacity": 0.85 }, svg);
      el("path", { d: "M" + sx(gm) + "," + (y + 4) + " l5,12 l-5,12 l-5,-12 z", fill: css("--fg"), stroke: css("--bg"), "stroke-width": 1 }, svg);
      el("text", { x: Wd - Rx + 8, y: y + 15, "font-size": 11, "font-family": "ui-monospace,Menlo,monospace" }, svg, "G " + fmt(gm, 1).replace("+", ""));
      el("text", { x: Wd - Rx + 8, y: y + 28, "font-size": 10, class: "muted", "font-family": "ui-monospace,Menlo,monospace" }, svg, "q " + ch.q_pi.toFixed(2));
      // per first option: the best G of the rover's sequences starting with it
      for (var o = 0; o < no; o++) {
        var yo = 8 + chH + o * rowH, g = G[o], isCh = o === opt;
        if (isCh) el("rect", { x: 1, y: yo, width: Wd - 2, height: rowH - 2, rx: 3, fill: "none", stroke: css("--accent"), "stroke-width": 1.2, "stroke-dasharray": "3 2" }, svg);
        el("title", {}, el("text", { x: 8, y: yo + rowH / 2 + 3, "font-size": 11, "font-weight": isCh ? 700 : 400 }, svg, (isCh ? "▸ " : "") + optShort(o)), optLabel(o));
        if (isFinite(g)) {
          el("rect", { x: Math.min(sx(g), sx(0)), y: yo + 7, width: Math.abs(sx(0) - sx(g)), height: rowH - 16, fill: css("--muted"), "fill-opacity": 0.3, stroke: css("--muted"), "stroke-dasharray": "3 2" }, svg);
          el("path", { d: "M" + sx(g) + "," + (yo + 3) + " l4,9 l-4,9 l-4,-9 z", fill: css("--fg"), stroke: css("--bg"), "stroke-width": 1 }, svg);
        }
        el("text", { x: Wd - Rx + 8, y: yo + rowH / 2 + 3, "font-size": 10.5, "font-family": "ui-monospace,Menlo,monospace" }, svg, (isFinite(g) ? "G " + fmt(g, 1).replace("+", "") : "G –") + "  q " + (q[o] < 0.005 && q[o] > 0 ? "<.01" : q[o].toFixed(2)));
      }
      // the energy band belief, as a heat strip over the bands
      var be = sl.belief && sl.belief[k] && sl.belief[k].energy;
      if (be) {
        var h = "<div class='bandbel'><span class='bl'>energy band belief</span><div class='heat'>";
        be.forEach(function (p, b) {
          h += "<div class='hc' title='" + esc(nm("energy", b)) + ": " + p.toFixed(3) + "'><span class='hcell' style='background:" + heatColor(p) + ";color:" + (p > 0.45 ? "#fff" : "#1d1f21") + "'><span class='hp'>" + (p >= 0.995 ? "1" : p.toFixed(2).replace(/^0/, "")) + "</span></span><span class='rn'>" + esc(nm("energy", b)) + "</span></div>";
        });
        box.insertAdjacentHTML("beforeend", h + "</div></div>");
      }
    });
    if (sl.movement != null) host.insertAdjacentHTML("beforeend", "<p class='decmove'><span class='mono'>" + sl.movement.toExponential(2) + "</span> <span class='muted'>movement: the largest change of a marginal in the last mean-field sweep</span></p>");
    lg.innerHTML = "<span><i style='background:" + termColor("infogain") + "'></i>novelty of the map</span><span><i style='background:" + termColor("risk") + "'></i>" + energyTerm() + "</span><span>&#9670; G</span><span><i style='background:" + css("--muted") + ";opacity:.4'></i>best G per first option (its terms are not kept)</span><span>outlined: the chosen sequence and the option chosen</span>";
    cap.textContent = "Sign convention: every term enters G with its sign applied, so novelty appears negative (lower G is better) and the " + energyTerm() + " positive. A rover's sequence is scored against the other rover's option marginals, so its record carries both rovers' leaves (the map and energy leaves _1 and _2). The record keeps no list of sequences: the terms of the highest-posterior sequence only, and per first option the best G. The chart's axis is fixed over the run so decisions can be compared.";
    drawEnergyTable(d);
  }
  // the scheduler's learned energy table at a decision: the expected change of band over one
  // option, per band and option, and the band changes each case has counted (slow.energy_table)
  function drawEnergyTable(d) {
    var box = $("decenergy");
    if (!box) {
      box = document.createElement("div"); box.id = "decenergy"; box.className = "decenergy";
      var after = $("deccap"); after.parentNode.insertBefore(box, after.nextSibling);
    }
    var et = d.slow.energy_table;
    if (!et) { box.hidden = true; return; }
    box.hidden = false;
    // the band the scheduler believes each rover is in (one rover: belief.energy; two: belief[k])
    var no = nOpts(), joint = Array.isArray(d.slow.belief), heres = {}, rks = roverKeys();
    (joint ? d.slow.belief : [d.slow.belief]).forEach(function (bl, k) {
      if (!bl || !bl.energy) return;
      var b = argmax(bl.energy); (heres[b] = heres[b] || []).push(rks[k]);
    });
    // the caption is a paragraph above the table, not a <caption>: a caption is as wide as its
    // table, and the table's width would follow its digits from one decision to the next
    var h = "<p class='etcap'>The scheduler's energy model at this decision, learned from its band readings: the expected change of band over one option (in bands), from each band, and in brackets the band changes the case has counted so far." + (joint ? " One table, shared by the rovers, at their believed cells. The last column names the rovers believed to be in each band." : " The band it believes it is in is marked.") + " Options: S1&ndash;S" + nZones() + " survey sectors 1&ndash;" + nZones() + (no > nZones() ? ", C1&ndash;C" + (no - nZones()) + " charge at sites 1&ndash;" + (no - nZones()) : "") + ".</p><table" + (joint ? " class='joint'" : "") + "><thead><tr><th>from</th>";
    for (var k = 0; k < no; k++) h += "<th>" + esc(optShort(k)) + "</th>";
    if (joint) h += "<th class='rovcol'>rovers in the band</th>";
    h += "</tr></thead><tbody>";
    for (var b = 0; b < et.change.length; b++) {
      var here = heres[b], mk = here && !joint ? "▸ " : "";
      h += "<tr" + (here ? " class='here'" : "") + "><th>" + mk + esc(nm("energy", b)) + "</th>";
      for (k = 0; k < no; k++) {
        var c = Math.round(et.change[b][k] * 100) / 100, n = et.counts[b][k];
        h += "<td>" + (c >= 0 ? "+" : "&minus;") + Math.abs(c).toFixed(2) + " <span class='muted'>(" + (Math.round(n * 10) / 10) + ")</span></td>";
      }
      if (joint) h += "<td class='rovcol'>" + (here || []).map(function (rk) { return "<span class='rovtag' style='color:" + roverColor(rk) + "'>" + esc(roverName(rk)) + "</span>"; }).join(" ") + "</td>";
      h += "</tr>";
    }
    box.innerHTML = h + "</tbody></table>";
  }

  // ---- objectives as intervals ----
  // one chart per tier, the rover's first; the mission tier's appears only in a two-level setting
  var OBJ_TIERS = ["rover", "mission"], RANGES = {};
  function tierTitle(tier) {
    // the tier keys are the bundle's own tier names (graph.tiers[].name)
    var n = tier;
    // the lower tier is one rover (v1, "Rover tier") or is factorised into two (v2, "Rover 1", "Rover 2")
    if (tier === "rover" && multi()) { var r = roverName(); return r.charAt(0).toUpperCase() + r.slice(1); }
    return n.charAt(0).toUpperCase() + n.slice(1) + " tier";
  }
  function leafRows(p) {
    var out = [];
    Object.keys(p.terms).forEach(function (tm) {
      Object.keys(p.terms[tm]).forEach(function (lf) { var iv = p.terms[tm][lf]; out.push([tm, lf, iv[0], iv[1]]); });
    });
    return out;
  }
  function termSum(p, tm) {
    var lo = 0, hi = 0, d = p.terms[tm] || {};
    Object.keys(d).forEach(function (k) { lo += d[k][0]; hi += d[k][1]; });
    return [lo, hi];
  }
  function approx(tier, tm, lf, rk) { var a = tier === "rover" && heavy("policies", null, rk).approximate; return !!(a && a[tm] && a[tm][lf]); }
  function anyApprox(tier, rk) { var a = tier === "rover" && heavy("policies", null, rk).approximate; if (!a) return false; for (var tm in a) for (var lf in a[tm]) if (a[tm][lf]) return true; return false; }
  function showTip(ev, it) {
    var h = "<b>" + esc(it.tipName || it.name) + "</b> &nbsp; " + (it.qLabel || "posterior weight qπ") + "=" + it.q.toFixed(3) + "<br>G " + (Math.abs(it.G[1] - it.G[0]) < 1e-9 ? "= " + it.G[0].toFixed(2) + " (point)" : "[" + it.G[0].toFixed(2) + ", " + it.G[1].toFixed(2) + "]") + (it.tipExtra ? "<br>" + esc(it.tipExtra) : "") + "<table>", ap = false;
    leafRows(it.p).forEach(function (r) {
      var a = approx(it.tier, r[0], r[1], it.rk); ap = ap || a;
      h += "<tr><td style='color:" + termColor(r[0]) + "'>" + r[0] + "</td><td>" + esc(r[1]) + (a ? " <span class='ap'>≈</span>" : "") + "</td><td>" + (Math.abs(r[2] - r[3]) < 1e-9 ? r[2].toFixed(2) : "[" + r[2].toFixed(2) + ", " + r[3].toFixed(2) + "]") + "</td></tr>";
    });
    tipEl.innerHTML = h + "</table>" + (ap ? "<span class='ap'>≈ approximate: the plan's beliefs are pushed through the vehicle's nonlinear motion to first order</span>" : ""); tipEl.hidden = false;
    var box = ev.currentTarget.getBoundingClientRect(), mouse = ev.clientX != null && ev.type !== "focus";
    var x = mouse ? ev.clientX + 14 : box.left + 40, y = mouse ? ev.clientY + 14 : box.bottom;
    x = Math.min(x, window.innerWidth - tipEl.offsetWidth - 8); y = Math.min(y, window.innerHeight - tipEl.offsetHeight - 8);
    tipEl.style.left = Math.max(4, x) + "px"; tipEl.style.top = Math.max(4, y) + "px";
  }
  function startName(id) {
    // start 0 is the plan acted on the tick before, shifted one tick on; the others are the
    // planner's spread points over the command box
    return id === 0 ? "carried plan" : "spread " + id;
  }
  function seqName(p) { return p.options.map(optShort).join(" → "); }
  function seqLong(p) { return p.options.map(optLabel).join(" → "); }
  function policyGamma() {
    // the scheduler's policy precision: q_pi is a softmax of -gamma * G over its sequences
    var l = B.description && B.description.components && B.description.components.leaves;
    var c = l && l.mission && l.mission.configuration;
    return c && c.gamma ? c.gamma : 16;
  }
  function objItems(tier, at, agg) {
    // the plans (rover) or kept option sequences (mission) in force at tick index ``at``; with
    // ``agg`` "action" the mission's rows are one per first option, each the marginal over every
    // sequence that starts with it: G_u = -(1/gamma) log sum exp(-gamma G), read from the record
    // as G_s + (log q_s - log q_u) / gamma for a kept sequence s starting with u, where q_u is
    // that first option's probability summed over every sequence the scheduler scored
    if (tier === "rover") {
      var t = heavy("policies", at);
      return {
        terms: Object.keys(t.policies[0].terms), decided: null,
        items: t.policies.map(function (p, i) { return { tier: tier, name: startName(p.id), tipName: startName(p.id) + " (start " + p.id + ")", sub: "q=" + p.q_pi.toFixed(2), G: p.G, q: p.q_pi, p: p, chosen: i === t.chosen, rk: S.rover }; })
      };
    }
    var di = cache().last[at], d = di < 0 ? null : run().ticks[di];
    if (d && d.slow && Array.isArray(d.slow.chosen)) {
      // two rovers: the record keeps no list of sequences, only each rover's highest-posterior
      // sequence with its terms; one row per rover, each outlined (each is its rover's choice)
      var ks = roverKeys();
      return {
        terms: Object.keys(d.slow.chosen[0].terms), decided: d.t, joint: true,
        items: d.slow.chosen.map(function (p, k) {
          return { tier: tier, name: roverName(ks[k]) + ": " + seqName(p), tipName: roverName(ks[k]) + ", highest-posterior sequence " + seqLong(p), qLabel: "its posterior probability", sub: "q=" + p.q_pi.toFixed(2), G: p.G, q: p.q_pi, p: p, chosen: true };
        })
      };
    }
    if (!d || !d.slow || !d.slow.policies) return { terms: [], items: [], decided: null };
    var terms = Object.keys(d.slow.policies[0].terms);
    if (agg === "action") {
      var rows = [], gam = policyGamma(), chosenSeq = d.slow.policies[d.slow.chosen];
      for (var o = 0; o < nOpts(); o++) {
        var best = null, ref = null;
        d.slow.policies.forEach(function (p) {
          if (p.options[0] !== o) return;
          if (!best || p.G[0] + p.G[1] < best.G[0] + best.G[1]) best = p;
          if (p.q_pi > 0 && (!ref || p.q_pi > ref.q_pi)) ref = p;
        });
        if (!best) continue;
        if (chosenSeq && chosenSeq.options[0] === o && chosenSeq.q_pi > 0) ref = chosenSeq;
        var qo = d.slow.q_by_option ? d.slow.q_by_option[o] : best.q_pi, gu, bound = !(ref && qo > 0);
        if (!bound) gu = (ref.G[0] + ref.G[1]) / 2 + (Math.log(ref.q_pi) - Math.log(qo)) / gam;
        else {
          // q below the record's float range (stored as 0): the log-sum-exp over the kept
          // sequences with this first option alone, an upper bound on the marginal
          var m = Infinity, acc = 0;
          d.slow.policies.forEach(function (p) { if (p.options[0] === o) m = Math.min(m, (p.G[0] + p.G[1]) / 2); });
          d.slow.policies.forEach(function (p) { if (p.options[0] === o) acc += Math.exp(-gam * ((p.G[0] + p.G[1]) / 2 - m)); });
          gu = m - Math.log(acc) / gam;
        }
        var bg = (best.G[0] + best.G[1]) / 2;
        rows.push({
          tier: tier, name: optShort(o),
          tipName: "first action " + optLabel(o) + ": marginal over every sequence starting with it",
          tipExtra: (bound ? "q is below the record's float range, so G is an upper bound: the log-sum-exp over the kept sequences with this first action. " : "") + "Terms below: best sequence " + seqLong(best) + ", G = " + bg.toFixed(2),
          qLabel: "probability of this first action", sub: bound ? "q<1e-38, G upper bound" : "q=" + qo.toFixed(2), G: [gu, gu], q: qo, p: best, bound: bound, chosen: o === d.slow.option
        });
      }
      rows.sort(function (a, b) { return a.G[0] - b.G[0]; });
      return { terms: terms, decided: d.t, items: rows, grouped: true, bounded: rows.some(function (r) { return r.bound; }) };
    }
    var items = d.slow.policies.map(function (p, i) {
      return { tier: tier, name: seqName(p), tipName: seqLong(p), sub: "q=" + p.q_pi.toFixed(2), G: p.G, q: p.q_pi, p: p, chosen: i === d.slow.chosen };
    });
    items.sort(function (a, b) { return (a.G[0] + a.G[1]) - (b.G[0] + b.G[1]); });
    items.forEach(function (it, i) { it.rank = i + 1; });
    return { terms: terms, decided: d.t, items: items };
  }
  function gRange(items) {
    // the span of every G interval's endpoints
    var lo = Infinity, hi = -Infinity;
    items.forEach(function (it) { lo = Math.min(lo, it.G[0]); hi = Math.max(hi, it.G[1]); });
    return [lo, hi];
  }
  function stackRange(items, terms) {
    var lo = 0, hi = 0;
    items.forEach(function (it) {
      var neg = 0, pos = 0;
      terms.forEach(function (tm) {
        if (!it.p.terms[tm]) return;
        var s = termSum(it.p, tm), md = (s[0] + s[1]) / 2, hw = (s[1] - s[0]) / 2, c;
        if (md < 0) { neg += md; c = neg - md / 2; } else { pos += md; c = pos - md / 2; }
        lo = Math.min(lo, neg, c - hw); hi = Math.max(hi, pos, c + hw);
      });
      lo = Math.min(lo, it.G[0]); hi = Math.max(hi, it.G[1]);
    });
    return [lo, hi];
  }
  function objRanges(tier) {
    // each view's value axis for this tier over the whole run of the setting: every G interval
    // (view "total") and every stacked term with its bracket and the G mark (view "terms"), over
    // every tick that carries plans or a decision, under either aggregation, zero included, with
    // a 3% margin each side
    var key = S.setting + "|" + tier + (tier === "rover" && multi() ? "|" + S.rover : "");
    if (RANGES[key]) return RANGES[key];
    var T = tier === "rover" ? rows() : run().ticks, tot = [[], []], stk = [[], []];
    T.forEach(function (t, i) {
      var has = tier === "rover" ? t.policies != null : !!(t.slow && t.slow.decision && t.slow.policies);
      if (!has) return;
      (tier === "mission" && !multi() ? ["policy", "action"] : ["policy"]).forEach(function (agg) {
        var o = objItems(tier, i, agg), a = gRange(o.items), b = stackRange(o.items, o.terms);
        tot[0].push(a[0]); tot[1].push(a[1]); stk[0].push(b[0]); stk[1].push(b[1]);
      });
    });
    // the axis spans the 1st to the 99th percentile of the ticks' extremes, so that a few ticks'
    // spikes (the rover's position risk when its target jumps) do not squash every other tick;
    // values beyond it are drawn at the edge with a marker
    function pct(vals, q) {
      if (!vals.length) return 0;
      var v = vals.slice().sort(function (a, b) { return a - b; });
      return v[Math.min(v.length - 1, Math.max(0, Math.round(q * (v.length - 1))))];
    }
    function span(r) {
      var lo = Math.min(0, pct(r[0], 0.01)), hi = Math.max(0, pct(r[1], 0.99)), m = (hi - lo) * 0.03 || 1;
      return [lo - m, hi + m];
    }
    return (RANGES[key] = { total: span(tot), terms: span(stk) });
  }
  function drawObjective() {
    var mode = $("objmode").value, two = twoLevel(), agg = two && !multi() ? $("objagg").value : "policy";
    $("objaggwrap").hidden = !two || multi();
    $("objnote").textContent = "Detail shows each policy's total G or its split by term. " + (two && !multi() ? "Aggregation shows a row per policy (for the scheduler an option sequence, for the rover a plan) or, for the scheduler, a row per first action. " : "") + "Each chart's value axis is fixed over the whole run of this setting (the 1st to the 99th percentile of the ticks' extremes; a few spiking ticks fall beyond it and are marked at the edge), so moving the tick changes only the bars and marks. Hover or focus a row for its per-term, per-leaf values; the chosen one is highlighted. Lower G is better. " + (B.meta.sign_convention ? "Sign convention: " + B.meta.sign_convention + "." : "") +
      (mode === "terms" ? " Each G is split into its objective terms, stacked from zero: negative terms to the left, positive to the right; a bracket marks each term's interval." : "");
    OBJ_TIERS.forEach(function (tier) {
      var on = tier === "rover" || two;
      $("objtier-" + tier).hidden = !on;
      if (!on) return;
      if (tier !== "rover" || !multi()) { drawObjTier(tier, mode, agg, ""); return; }
      // two rovers: a block per rover, the second a copy of the first section with its ids
      // suffixed; the rover is the context the block's plans are read in
      var keep = S.rover;
      roverKeys().forEach(function (rk, k) {
        S.rover = rk;
        drawObjTier(tier, mode, agg, k ? roverSection(rk) : "");
      });
      S.rover = keep;
    });
    var extra = document.querySelectorAll(".objtier.objcopy");
    extra.forEach(function (e) { e.hidden = !multi(); });
  }
  function roverSection(rk) {
    // the rover objective section for a further rover: the first rover's, copied once
    var sfx = "-" + rk, id = "objtier-rover" + sfx;
    if (!$(id)) {
      var base = $("objtier-rover"), c = base.cloneNode(true);
      c.id = id; c.classList.add("objcopy");
      c.querySelectorAll("[id]").forEach(function (e) { e.id = e.id + sfx; });
      var after = base; while (after.nextElementSibling && after.nextElementSibling.classList.contains("objcopy")) after = after.nextElementSibling;
      after.parentNode.insertBefore(c, after.nextSibling);
    }
    $(id).hidden = false;
    return sfx;
  }
  function drawObjTier(tier, mode, agg, sfx) {
    var host = $("objective-" + tier + sfx), mission = tier === "mission", grouped = mission && agg === "action";
    var o = objItems(tier, S.t, mission ? agg : "policy"), items = o.items, terms = o.terms;
    $("objhead-" + tier + sfx).textContent = tierTitle(tier);
    $("objnote-" + tier + sfx).textContent = mission && o.joint
      ? "The two rovers' scheduler plans with the factorised planner, whose record keeps no per-sequence policy list: per rover it keeps the sequence that rover's posterior puts highest, with its terms. The rows are those two sequences, each outlined as its rover's choice; each G is a point (the sequences are enumerated exactly), drawn as a tick. A rover's sequence is scored against the other rover's option marginals, so its terms carry both rovers' leaves. The scheduler's G per first option is in the decision chart under Performance comparison."
      : mission
      ? "The scheduler's policies are option sequences, enumerated exactly, so each G is a point (lower equals upper) and is drawn as a tick, not a bar. Its objective terms are the novelty of the map and the " + energyTerm() + "." +
        (grouped ? " Per first action, each row is the marginal over every sequence that starts with that option, G_u = -(1/\u03b3) log \u03a3 exp(-\u03b3 G) at the policy precision \u03b3 = " + policyGamma() + ", so that q_u is proportional to exp(-\u03b3 G_u); it is read from the record as G_s + (log q_s - log q_u)/\u03b3 for a kept sequence s." + (o.bounded ? " Where q_u is below the record's float range (stored as 0, shown as q<1e-38), the row's G is the log-sum-exp over the kept sequences with that first action alone, an upper bound on G_u." : "") + (mode === "terms" ? " Terms do not marginalise additively, so the term split shown is that of the best kept sequence with that first action; the \u25b2 mark is the marginal G_u." : "") : "")
      : "The gradient planner descends from four starts: one carries the plan acted on last tick, shifted one tick on, and three are well-spread plans over the command box. Descent runs from each and the best is acted on; the bars are the four plans, the acted one outlined; the one bracketed term is the wall preference on the fix, a mixture, bounded in closed form." + (agg === "action" ? " The rover's commands are continuous, so it has no aggregation per first action; its rows stay one per plan." : "") + (anyApprox(tier, S.rover) ? " The rover's terms are approximate (≈ in the tooltip): the plan's beliefs are pushed through the vehicle's nonlinear motion to first order." : "");
    $("objcap-" + tier + sfx).textContent = mission
      ? (o.decided == null ? "" : o.joint ? "Expected free energy of each rover's highest-posterior option sequence at the most recent decision (tick " + o.decided + ")." : grouped ? "Marginal expected free energy of each first action over every option sequence starting with it, at the most recent decision (tick " + o.decided + "), lowest first; q is that first action's probability, summed over every sequence the scheduler scored." : "Expected free energy of the kept option sequences (the " + items.length + " the page keeps: the twelve best, and the best for each first action) at the most recent decision (tick " + o.decided + "), best first.")
      : (multi() ? roverName() + ": e" : "E") + "xpected free energy G of the four plans at " + (heavy("policies").t === rtick().t ? "the current tick" : "tick " + heavy("policies").t + ", the latest the replay keeps plans for") + ".";
    if (mission) {
      var btn = $("objall"); btn.hidden = grouped || !(items.length > TOP_N);
      btn.textContent = S.showAll ? "Show top " + TOP_N : "Show all " + items.length + " kept";
      if (!grouped && !S.showAll && items.length > TOP_N) {
        var top = items.slice(0, TOP_N), ch = items.filter(function (i) { return i.chosen; })[0];
        if (ch && top.indexOf(ch) < 0) top.push(ch);
        items = top;
      }
    }
    reserve(host);
    var lg = $("objlegend-" + tier + sfx); lg.innerHTML = "";
    if (!items.length) { host.textContent = "No decision yet."; return; }
    var rg = objRanges(tier)[mode === "terms" ? "terms" : "total"], lo = rg[0], hi = rg[1];
    var W = Math.max(300, host.clientWidth - 16), L = mission ? (W < 520 ? 124 : 190) : 104, R = 14, rowH = mode === "total" ? 36 : 58, top0 = 8, H = top0 + items.length * rowH + 34;
    var sxRaw = function (v) { return L + (v - lo) / (hi - lo) * (W - L - R); };
    var sx = function (v) { return Math.max(L, Math.min(W - R, sxRaw(v))); }, clipped = false;
    function offAxis(g, v, y) {
      // a value beyond the axis: a small arrow at the edge it lies past
      if (v >= lo && v <= hi) return;
      clipped = true;
      var x = v > hi ? W - R + 2 : L - 2, d = v > hi ? 1 : -1;
      el("path", { d: "M" + x + "," + (y - 5) + " l" + (6 * d) + ",5 l" + (-6 * d) + ",5 z", fill: css("--fg") }, g);
    }
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img", "aria-label": tierTitle(tier) + ": expected free energy per " + (grouped ? "first action" : mission ? "option sequence" : "plan") }, host);
    svg.style.height = "auto";
    var step = Math.pow(10, Math.floor(Math.log10((hi - lo) / 5))); step *= (hi - lo) / step / 5 > 3 ? 5 : (hi - lo) / step / 5 > 1.5 ? 2 : 1;
    for (var v = Math.ceil(lo / step) * step; v <= hi; v += step) {
      el("line", { x1: sx(v), x2: sx(v), y1: top0, y2: H - 26, stroke: css("--line"), "stroke-width": Math.abs(v) < 1e-9 ? 1.6 : 0.8 }, svg);
      el("text", { x: sx(v), y: H - 10, "text-anchor": "middle", "font-size": 11, class: "muted" }, svg, (Math.round(v * 100) / 100) + "");
    }
    function mark(g, iv, y, h, color, op, strong) {
      var w = sx(iv[1]) - sx(iv[0]);
      if (w < 1.5) {
        el("line", { x1: sx(iv[0]), x2: sx(iv[0]), y1: y, y2: y + h, stroke: color, "stroke-opacity": op, "stroke-width": strong ? 4 : 3, "stroke-linecap": "round" }, g);
      } else {
        el("rect", { x: sx(iv[0]), y: y, width: w, height: h, rx: 2, fill: color, "fill-opacity": op }, g);
        var m = (iv[0] + iv[1]) / 2; el("line", { x1: sx(m), x2: sx(m), y1: y - 2, y2: y + h + 2, stroke: css("--fg"), "stroke-width": 2 }, g);
      }
    }
    items.forEach(function (it, i) {
      var y = top0 + i * rowH, ch = it.chosen, p = it.p;
      var g = el("g", { class: "row", tabindex: 0, "aria-label": it.name + ", G " + it.G[0].toFixed(2) + (Math.abs(it.G[0] - it.G[1]) < 1e-9 ? "" : " to " + it.G[1].toFixed(2)) + (ch ? ", chosen" : "") }, svg);
      el("rect", { class: "rowbg", x: 0, y: y, width: W, height: rowH, fill: "transparent" }, g);
      if (ch) el("rect", { x: 1, y: y + 1, width: W - 2, height: rowH - 2, fill: "none", stroke: css("--accent"), "stroke-width": 1.5, rx: 3 }, g);
      var label = (ch ? "▸ " : "") + (mission && !grouped && it.rank ? "#" + it.rank + " " : "") + it.name;
      var maxc = Math.floor((L - 12) / 6.2);
      el("text", { x: 8, y: y + rowH / 2 - 1, "font-size": 11.5, "font-weight": ch ? 700 : 400 }, g, label.length > maxc ? label.slice(0, maxc - 1) + "…" : label);
      el("text", { x: 8, y: y + rowH / 2 + 14, "font-size": 9.5, class: "muted" }, g, it.sub);
      if (mode === "total") {
        mark(g, it.G, y + 8, rowH - 16, css(ch ? "--accent" : "--muted"), ch ? 0.9 : 0.55, ch);
        offAxis(g, it.G[1] > hi ? it.G[1] : it.G[0], y + rowH / 2);
      } else {
        var ts = terms.filter(function (tm) { return p.terms[tm]; }), neg = 0, pos = 0, by = y + 18, bhh = 20;
        ts.forEach(function (tm) {
          var s = termSum(p, tm), md = (s[0] + s[1]) / 2, from, to;
          if (md < 0) { from = neg; neg += md; to = neg; } else { from = pos; pos += md; to = pos; }
          var x0 = Math.min(sx(from), sx(to)), w = Math.abs(sx(to) - sx(from)), cx = (sx(from) + sx(to)) / 2, hw = (sx(s[1]) - sx(s[0])) / 2;
          if (w > 0.3) el("rect", { x: x0, y: by, width: w, height: bhh, fill: termColor(tm), "fill-opacity": 0.8, stroke: css("--panel"), "stroke-width": 0.6 }, g);
          if (hw >= 1.2) {
            el("path", { d: "M" + (cx - hw) + "," + (by - 7) + "v5 M" + (cx + hw) + "," + (by - 7) + "v5 M" + (cx - hw) + "," + (by - 4.5) + "H" + (cx + hw), fill: "none", stroke: css("--fg"), "stroke-opacity": 0.75, "stroke-width": 1.4 }, g);
          }
        });
        var mg = (it.G[0] + it.G[1]) / 2; el("path", { d: "M" + sx(mg) + "," + (by + bhh + 3) + " l-4,6 h8 z", fill: css("--fg") }, g);
        offAxis(g, Math.max(pos, mg) > hi ? Math.max(pos, mg) : Math.min(neg, mg), by + bhh / 2);
      }
      var show = function (e) { showTip(e, it); }, hide = function () { tipEl.hidden = true; };
      g.addEventListener("mousemove", show); g.addEventListener("mouseleave", hide);
      g.addEventListener("focus", show); g.addEventListener("blur", hide);
    });
    if (mode === "terms") {
      terms.forEach(function (tm) {
        var lv = {};
        items.forEach(function (it) { Object.keys(it.p.terms[tm] || {}).forEach(function (k) { lv[k] = 1; }); });
        lg.innerHTML += "<span><i style='background:" + termColor(tm) + "'></i>" + esc(tm) + " (" + esc(Object.keys(lv).join(", ")) + ")</span>";
      });
      lg.innerHTML += "<span>bracket: term interval</span><span>&#9650; " + (grouped ? "marginal G per first action" : "total G midpoint") + "</span>";
    }
    else if (mission) lg.innerHTML = "<span><i style='background:" + css("--muted") + "'></i>" + (grouped ? "marginal G per first action (point)" : "G per policy (point)") + "</span><span><i style='background:" + css("--accent") + "'></i>chosen</span>";
    else lg.innerHTML = "<span><i style='background:" + css("--muted") + "'></i>interval</span><span>| midpoint</span><span><i style='background:" + css("--accent") + "'></i>chosen</span>";
    if (mode === "terms" && anyApprox(tier)) lg.innerHTML += "<span class='approx'>≈ approximate leaves (hover)</span>";
    if (clipped) lg.innerHTML += "<span>&#9656; beyond the axis (a spike; hover for the value)</span>";
  }

  // ---- description length ----
  function nats(v) { return v < 1000 ? v.toFixed(2) : Math.round(v).toLocaleString("en-US"); }
  function drawMdl() {
    var tb = document.querySelector("#mdltable tbody");
    tb.innerHTML = "";
    if (!B.mdl) { $("mdlfoot").textContent = "This replay carries no description-length table."; return; }
    var C = B.mdl.components, mx = 0, tot = 0, exact = 0;
    C.forEach(function (c) { if (c.nats != null) { mx = Math.max(mx, Math.log10(c.nats + 1)); tot += c.nats; if (c.exact) exact += c.nats; } });
    // the tiers are the graph's (mission and rover; mission, rover1 and rover2)
    var TIERS = B.graph.tiers.map(function (t) { return t.name; });
    function tierOf(c) { var m = /(?:^|:)([a-z]+\d*)\./.exec(c.name); return m && TIERS.indexOf(m[1]) >= 0 ? m[1] : null; }
    function tierHead(n) { var m = /^([a-z]+)(\d*)$/.exec(n); return m[1].charAt(0).toUpperCase() + m[1].slice(1) + (m[2] ? " " + m[2] : " tier"); }
    function comp(c, note) {
      var tr = document.createElement("tr"), un = c.nats == null;
      var a = document.createElement("td"); a.textContent = c.name.replace(/^[a-z_]+:/, "").replace(/^[a-z]+\d*\./, ""); a.title = c.name; tr.appendChild(a);
      var k = document.createElement("td"); k.className = "kind"; k.textContent = pretty(c.kind); tr.appendChild(k);
      var pc = document.createElement("td"); pc.className = "num muted"; pc.textContent = c.entries != null ? c.entries.toLocaleString() : ""; tr.appendChild(pc);
      var n = document.createElement("td"); n.className = "num" + (un ? " unscored" : ""); n.textContent = un ? "unscored" : nats(c.nats); tr.appendChild(n);
      var bc = document.createElement("td"); bc.className = "mdlbarcell";
      if (!un) { var bar = document.createElement("span"); bar.className = "mdlbar"; bar.style.width = (100 * Math.log10(c.nats + 1) / (mx || 1)).toFixed(1) + "%"; bar.title = "log10(nats + 1) = " + Math.log10(c.nats + 1).toFixed(2); bc.appendChild(bar); }
      tr.appendChild(bc);
      var e = document.createElement("td"); e.textContent = (un ? "unscored" : (c.exact ? "exact" : "approximate")) + (note ? "; " + note : ""); if (!un && !c.exact) e.className = "approx"; tr.appendChild(e);
      tb.appendChild(tr);
    }
    function paramOf(reader) {
      var ps = B.graph.parameters || [], p = ps.filter(function (q) { return q.readers.indexOf(reader) >= 0; })[0];
      return (p || ps[0] || {}).name;
    }
    function group(title, cs) {
      var h = document.createElement("tr"); h.className = "grp"; h.innerHTML = "<td colspan='6'>" + esc(title) + "</td>"; tb.appendChild(h);
      var sub = 0, ex = 0;
      cs.forEach(function (c) {
        var ref = c.exact && c.entries === 0 && /^transition:/.test(c.name);
        comp(c, ref ? "structure only, the table is " + paramOf(c.name.replace(/^transition:/, "")) : "");
        if (c.nats != null) { sub += c.nats; if (c.exact) ex += c.nats; }
      });
      var s = document.createElement("tr"); s.className = "grp light";
      s.innerHTML = "<td colspan='3'>" + (/tier/i.test(title) ? "Tier subtotal" : "Subtotal") + "</td><td class='num'>" + nats(sub) + "</td><td></td><td>" + nats(ex) + " of it exact</td>";
      tb.appendChild(s);
    }
    var isParam = function (c) { return c.kind === "parameter"; };
    var readers = function (c) { var nm2 = c.name.replace(/^parameter:/, ""), p = (B.graph.parameters || []).filter(function (q) { return q.name === nm2; })[0]; return p ? p.readers.length : 0; };
    TIERS.forEach(function (tn) { group(tierHead(tn), C.filter(function (c) { return tierOf(c) === tn && !isParam(c); })); });
    var h = document.createElement("tr"); h.className = "grp"; h.innerHTML = "<td colspan='6'>The composite</td>"; tb.appendChild(h);
    C.filter(function (c) { return c.kind === "graph"; }).forEach(function (c) { comp(c, TIERS.length === 2 ? "both tiers" : "all " + countWord(TIERS.length) + " tiers"); });
    C.filter(isParam).forEach(function (c) { var n = readers(c); comp(c, "priced once; " + (n === 1 ? "its reader pays" : "its " + countWord(n) + " readers pay") + " structure only"); });
    var tr = document.createElement("tr"); tr.className = "grp total";
    tr.innerHTML = "<td colspan='3'>total over scored components</td><td class='num'>" + nats(tot) + "</td><td></td><td>" + nats(exact) + " of it exact</td>";
    tb.appendChild(tr);
    var big = C.slice().sort(function (a, b) { return b.nats - a.nats; })[0];
    var nRef = C.filter(function (c) { return c.exact && c.entries === 0 && /^transition:/.test(c.name); }).length, nPar = C.filter(isParam).length;
    var nEm = C.filter(function (c) { return c.exact && /^emission:/.test(c.name); }).length;
    function countWord(n) { return ["no", "one", "two", "three", "four", "five"][n] || String(n); }
    // the camera's likelihood network, one per rover
    var nets = C.filter(function (c) { return c.kind === big.kind && c.entries === big.entries && /img$/.test(c.name); }).length;
    var bigName = big.name.replace(/^[a-z_]+:/, ""); var cams = C.filter(function (c) { return /^likelihood:.*img$/.test(c.name); }).length;
    $("mdlfoot").textContent = (!/img$/.test(big.name) ? "The largest component is " + bigName + " (" + big.entries.toLocaleString() + " entries, " + Math.round(100 * big.nats / tot) + "% of the total)." + (cams > 1 ? " The rovers' " + countWord(cams) + " camera likelihood networks come next, each" : " The camera's likelihood network comes next,") + " priced by its family alone and so flagged approximate." : nets > 1 ? "Most of the total is " + countWord(nets) + " components: the rovers' camera likelihood networks (" + big.entries.toLocaleString() + " weights each, each priced by its family alone and so flagged approximate)." : "Most of the total is one component: the camera's likelihood network (" + big.entries.toLocaleString() + " weights, priced by its family alone and so flagged approximate).") + " The exact entries are the graph, the controls, the " + countWord(nRef) + " structure-only readers and the " + (nPar === 1 ? "parameter variable" : countWord(nPar) + " parameter variables") + (nEm ? " and the " + (nEm === 1 ? "cell-reading emission" : countWord(nEm) + " emissions") : "") + ", which is " + (100 * exact / tot).toFixed(1) + "% of the total. Bars are proportional to log10(nats + 1)." + (C.some(function (c) { return /energy_table$/.test(c.name); }) ? " The scheduler's energy table is priced far above its size: its prior puts 0.001 pseudo-counts on the band moves it holds impossible, a strong claim against the uniform base, and the description length charges for every such entry." : "");
  }

  // ---- the arms compared: the replay's coverage per run and the five-seed numbers ----
  // Every arm of the version (meta.comparison.arms): its replayed run's coverage over the ticks,
  // solid, and the batched seeds' mean at the marks as a dashed line in a band of one sd either
  // side; below, the seeds' numbers as a table, and the replay's seed and rule (meta.replay)
  var ARMCOL = ["--accent", "--c2", "--c4", "--c5", "--c3"];
  function arms() { return (B.meta.comparison && B.meta.comparison.arms) || []; }
  function armColor(i) { return css(ARMCOL[i % ARMCOL.length]); }
  function drawComparison() {
    var host = $("comparison"), cmp = B.meta.comparison;
    if (!host) return;
    reserve(host);
    var lg = $("complegend"), cap = $("compcap");
    if (!cmp) { host.textContent = "This replay carries no comparison of seeds."; if (lg) lg.innerHTML = ""; drawSeedTable(); return; }
    var A = arms(), marks = cmp.marks, nT = B.meta.ticks || marks[marks.length - 1];
    var W = Math.max(300, host.clientWidth - 16), L = 46, R = 46, top = 12, hC = Math.max(180, Math.min(280, W * 0.42)), H = top + hC + 34;
    var sx = function (t) { return L + t / nT * (W - L - R); };
    var vmax = 0;
    A.forEach(function (a) {
      a.mean.forEach(function (m, j) { vmax = Math.max(vmax, m + (a.sd[j] || 0)); });
      if (B.runs[a.id]) B.runs[a.id].ticks.forEach(function (t) { vmax = Math.max(vmax, t.coverage); });
    });
    var cmax = Math.min(1, Math.max(0.5, Math.ceil(vmax * 4) / 4)), cy = function (v) { return top + hC - clamp(v / cmax, 0, 1) * hC; };
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img", "aria-label": "Coverage over the run per arm: the replay and the five-seed mean with its sd" }, host);
    svg.style.height = "auto";
    for (var v = 0; v <= cmax + 1e-9; v += 0.25) {
      el("line", { x1: L, x2: W - R, y1: cy(v), y2: cy(v), stroke: css("--line"), "stroke-width": 0.8 }, svg);
      el("text", { x: L - 4, y: cy(v) + 3, "text-anchor": "end", "font-size": 9.5, class: "muted" }, svg, v.toFixed(2));
    }
    [0].concat(marks).forEach(function (t) {
      el("line", { x1: sx(t), x2: sx(t), y1: top + hC, y2: top + hC + 4, stroke: css("--muted") }, svg);
      el("text", { x: sx(t), y: top + hC + 16, "text-anchor": "middle", "font-size": 10, class: "muted" }, svg, t === marks[marks.length - 1] ? "tick " + t : t);
    });
    el("text", { x: L, y: top - 2, "font-size": 10.5, class: "muted" }, svg, "coverage: fraction of cells mapped");
    var ends = [];
    A.forEach(function (a, i) {
      var col = armColor(i), up = "", dn = "", mn = "";
      // the seeds' band and mean at the marks
      marks.forEach(function (t, j) {
        up += (j ? "L" : "M") + sx(t).toFixed(1) + "," + cy(a.mean[j] + a.sd[j]).toFixed(1);
        mn += (j ? "L" : "M") + sx(t).toFixed(1) + "," + cy(a.mean[j]).toFixed(1);
      });
      for (var j = marks.length - 1; j >= 0; j--) dn += "L" + sx(marks[j]).toFixed(1) + "," + cy(a.mean[j] - a.sd[j]).toFixed(1);
      el("path", { d: up + dn + "z", fill: col, "fill-opacity": 0.13, stroke: "none" }, svg);
      el("path", { d: mn, fill: "none", stroke: col, "stroke-width": 1.6, "stroke-dasharray": "5 4" }, svg);
      marks.forEach(function (t, j) { el("circle", { cx: sx(t), cy: cy(a.mean[j]), r: 3, fill: css("--bg"), stroke: col, "stroke-width": 1.6 }, svg); });
      // the replay's curve
      var r = B.runs[a.id];
      if (!r) return;
      var s = "";
      r.ticks.forEach(function (t, k) { s += (k ? "L" : "M") + sx(t.t + 0.5).toFixed(1) + "," + cy(t.coverage).toFixed(1); });
      el("path", { d: s, fill: "none", stroke: col, "stroke-width": a.id === S.setting ? 2.6 : 1.8 }, svg);
      var e = r.ticks[r.ticks.length - 1];
      ends.push({ y: cy(e.coverage) + 3.5, value: e.coverage, text: e.coverage.toFixed(2), color: col });
    });
    placeLabels(svg, ends, W - R + 4);
    // the tick shown, moved by render
    var cur = el("line", { id: "compcursor", x1: 0, x2: 0, y1: top, y2: top + hC, stroke: css("--fg"), "stroke-width": 1, "stroke-opacity": 0.5 }, svg);
    cur._sx = sx; updateComparisonCursor();
    if (lg) {
      lg.innerHTML = A.map(function (a, i) { return "<span><i style='background:" + armColor(i) + "'></i>" + esc(a.label) + "</span>"; }).join("") +
        "<span style='flex-basis:100%'><svg class='lsw' viewBox='0 0 28 10'><line x1='1' x2='27' y1='5' y2='5' stroke='" + css("--fg") + "' stroke-width='2'/></svg>the replay (seed " + (B.meta.replay ? B.meta.replay.seed : B.meta.seed) + "; the selected setting's thicker) &nbsp; " +
        "<svg class='lsw' viewBox='0 0 28 10'><rect x='1' y='1' width='26' height='8' fill='" + css("--fg") + "' fill-opacity='.13'/><line x1='1' x2='27' y1='5' y2='5' stroke='" + css("--fg") + "' stroke-width='1.6' stroke-dasharray='5 4'/></svg>mean of " + countWord(A[0] ? A[0].seeds.length : 0) + " seeds at ticks " + marks.join(", ") + ", in a band of &plusmn;1 sd</span>";
    }
    if (cap) cap.textContent = "";  // meta.version_label names the demo's version: the switch's, not the page body's
    drawSeedTable();
  }
  function updateComparisonCursor() {
    var cur = $("compcursor"); if (!cur || !cur._sx) return;
    var t = rtick().t, x = cur._sx(t + 0.5);
    cur.setAttribute("x1", x); cur.setAttribute("x2", x);
  }
  function replayRule(rp) {
    // the replay's rule as the data states it, without the raw run ids it names in brackets (the
    // arm is named by its label: "the scheduled arm"), and without version words
    var r = String(rp.rule || "");
    Object.keys(B.runs).concat(rp.rule_arm ? [rp.rule_arm] : []).forEach(function (id) { r = r.split(" (" + id + ")").join(""); });
    return r.replace(/ of the version\b/g, "");
  }
  function drawSeedTable() {
    var tb = $("seedtable"), foot = $("seedfoot"), cmp = B.meta.comparison;
    if (!tb) return;
    if (!cmp) { tb.innerHTML = ""; if (foot) foot.textContent = ""; return; }
    var A = arms(), seeds = A[0] ? A[0].seeds : [];
    var h = "<caption>Coverage at ticks " + cmp.marks.join(", ") + ": mean &plusmn; sd over " + countWord(seeds.length) + " seeds (" + seeds[0] + "&ndash;" + seeds[seeds.length - 1] + ") per arm</caption><thead><tr><th>Arm</th>";
    cmp.marks.forEach(function (m) { h += "<th class='num'>" + m + "</th>"; });
    h += "</tr></thead><tbody>";
    A.forEach(function (a, i) {
      h += "<tr><td><i class='armsw' style='background:" + armColor(i) + "'></i>" + esc(a.label) + "</td>";
      a.mean.forEach(function (m, j) { h += "<td class='num'>" + m.toFixed(3) + " <span class='muted'>&plusmn; " + a.sd[j].toFixed(3) + "</span></td>"; });
      h += "</tr>";
    });
    tb.innerHTML = h + "</tbody>";
    var rp = B.meta.replay;
    if (foot) foot.textContent = "Coverage: the fraction of cells whose top probability exceeds 0.6. The seeds are batched replicas (seeds " + seeds.join(", ") + "); the sd is the sample sd. " +
      (rp ? "The replay is seed " + rp.seed + ": " + replayRule(rp) + ". " + (rp.from === "replica" ? "It is that seed's replica from the same batch, so its curves are the seed's numbers here." : "A batched replica is not the unbatched run of its seed, so the replay's curve need not match its seed's numbers here.") : "");
  }

  // ---- state / wiring ----
  function idxOf(t) {
    // the index of a recorded tick: the latest kept tick at or before t (a trimmed bundle keeps a
    // subset of the ticks, meta.trim); the index itself otherwise
    var T = run().ticks, n = T.length;
    if (!B.meta.trim) return clamp(Math.round(t), 0, n - 1);
    var i = 0; while (i + 1 < n && T[i + 1].t <= t) i++;
    return i;
  }
  function render() {
    $("tick").value = S.t; $("tickout").textContent = tick().t;
    drawField(); updateGraph(); drawTimeline(); drawDecision(); drawObjective(); updateComparisonCursor();
  }
  var TICK_MS = 120, MOTION = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;
  function smooth() {
    // between ticks the rover is drawn moving only when the bundle's motion is bounded (it records
    // the world's heading) and the reader has not asked for reduced motion
    return hasHeading() && !(MOTION && MOTION.matches);
  }
  function setPlaying(on) {
    S.playing = on; $("play").textContent = on ? "Pause" : "Play"; $("play").setAttribute("aria-label", on ? "Pause" : "Play");
    clearInterval(S.timer); if (S.raf) cancelAnimationFrame(S.raf); S.raf = null;
    if (S.frac) { S.frac = 0; if (B) drawField(true); }
    if (!on || !B) return;
    if (!smooth()) { S.timer = setInterval(function () { S.t = (S.t + stride()) % run().ticks.length; render(); }, TICK_MS); return; }
    var last = null;
    function frame(now) {
      if (!S.playing) return;
      S.frac += last == null ? 0 : (now - last) / TICK_MS; last = now;
      if (S.frac >= 1) {
        S.t = (S.t + stride() * Math.floor(S.frac)) % run().ticks.length; S.frac -= Math.floor(S.frac);
        if (S.t + stride() >= run().ticks.length) S.frac = 0;  // no next tick to move toward
        render();
      } else drawField(true);
      S.raf = requestAnimationFrame(frame);
    }
    S.raf = requestAnimationFrame(frame);
  }
  function fullRedraw() { BG = null; drawGraph(); render(); }
  function hashParams() {
    // the hash's key=value settings, wherever they sit among its tokens (a pane id first or not),
    // keys lower-cased; the version's value too ("V1", "1" and "v1" all name v1)
    var o = {};
    location.hash.replace(/^#/, "").split(/[&;?]/).forEach(function (kv) {
      var i = kv.indexOf("="); if (i <= 0) return;
      var k = kv.slice(0, i).trim().toLowerCase(), v = kv.slice(i + 1).trim();
      try { v = decodeURIComponent(v); } catch (e) { /* keep it raw */ }
      o[k] = v;
    });
    return o;
  }
  // the settings each version holds, so that a hash naming a setting of one version only opens
  // that version (setting=two-level opens v1)
  var SETTING_VERSION = { "two-level": "v1", "one-level": "v1", "two-rover-two-level": "v2", "two-rover-one-level": "v2", "two-rover-union": "v2" };
  function hashVersion(hp) {
    var v = String(hp.version || "").toLowerCase();
    if (/^\d+$/.test(v)) v = "v" + v;
    if (VERSIONS[v]) return v;
    return SETTING_VERSION[hp.setting] || null;
  }
  function showVersionSpans(v) {
    // prose written for one version only (page-prose.md's version spans): shown under it alone
    document.documentElement.setAttribute("data-version", v);
    document.querySelectorAll(".ver[data-version]").forEach(function (e) { e.hidden = e.getAttribute("data-version") !== v; });
  }
  function setStatus(text) {
    var st = $("status"); if (!st) return;
    st.hidden = !text; st.textContent = text || "";
  }

  // ---- versions: each a bundle of its own, loaded when chosen ----
  function useBundle(b, o) {
    // make bundle b the page's data: drop what was cached from the version before, fill the
    // setting select with this version's replayable runs and the rover select when it has two,
    // and draw everything at the setting, rover and tick asked for (o), else the defaults
    o = o || {};
    B = b; WIN = computeWindow(); CACHE = {}; DECR = null; RANGES = {}; BG = null;
    var dyn = $("dynrule"); if (dyn) dyn.textContent = B.meta.dynamics.rule;  // the caption may omit the rule
    if (B.meta.library_version) document.querySelectorAll("#version, .version").forEach(function (vs) { vs.textContent = "v" + B.meta.library_version; });
    var sel = $("setting"); sel.innerHTML = "";
    replaySettings().forEach(function (s) { var op = document.createElement("option"); op.value = s.id; op.textContent = s.label + (s.tiers === 2 ? " (two levels)" : " (one level)"); sel.appendChild(op); });
    S.setting = o.setting && replayable(o.setting) ? o.setting : defaultSetting(); sel.value = S.setting;
    S.rover = "rover1";
    var n = run().ticks.length; $("tick").max = n - 1;
    S.t = o.t != null ? idxOf(+o.t || 0) : Math.min(S.t, n - 1); S.frac = 0;
    dropReservations(); drawMdl(); fullRedraw(); drawComparison();
  }
  function dropReservations() {
    // the charts' height reservations (reserve) hold only within one version and setting: another
    // bundle or setting draws charts of other heights, so an old reservation would leave a gap
    ["timeline", "decision", "objective-rover", "objective-rover-rover2", "objective-mission", "comparison"].forEach(function (id) { var h = $(id); if (h) { h.style.minHeight = ""; h.innerHTML = ""; } });
  }
  function setVersion(v, o) {
    // load version v's bundle (once) and show it; the status line says so while it loads
    if (!VERSIONS[v]) v = DEFAULT_VERSION;
    var was = S.playing; setPlaying(false);
    S.version = v; showVersionSpans(v);
    var vs = $("bundlever"); if (vs) { vs.value = v; vs.disabled = true; }
    if (!window[VERSIONS[v].name]) setStatus("Loading the replay data: " + VERSIONS[v].label + "…");
    return load(v).then(function (b) {
      if (S.version !== v) return;  // another version was chosen meanwhile
      setStatus(""); if (vs) vs.disabled = false;
      useBundle(b, o);
      if (was) setPlaying(true);
    }, function (e) {
      if (vs) vs.disabled = false;
      setStatus("Could not load the replay data: " + e.message);
      throw e;
    });
  }

  function wire(hp) {
    // the controls' listeners, bound once
    if (hp.theme) document.documentElement.setAttribute("data-theme", hp.theme);
    var sel = $("setting");
    if (hp.view === "terms") $("objmode").value = "terms";
    var agg = hp.agg || { first: "action", sequence: "policy" }[hp.group];
    if (agg === "action" || agg === "policy") $("objagg").value = agg;
    if (hp.layer && /^(terrain|light|fix)$/.test(hp.layer)) { S.layer = hp.layer; document.querySelector("input[name=layer][value=" + hp.layer + "]").checked = true; }
    sel.addEventListener("change", function () { S.setting = sel.value; S.t = $("restart").checked ? 0 : Math.min(S.t, run().ticks.length - 1); $("tick").max = run().ticks.length - 1; dropReservations(); fullRedraw(); drawComparison(); });
    var vsel = $("bundlever");
    if (vsel) vsel.addEventListener("change", function () { setVersion(vsel.value, { t: $("restart").checked ? 0 : null }).catch(function () {}); });
    $("tick").addEventListener("input", function (e) { S.t = +e.target.value; S.frac = 0; render(); });
    $("play").addEventListener("click", function () { setPlaying(!S.playing); });
    $("objmode").addEventListener("change", drawObjective);
    $("objagg").addEventListener("change", drawObjective);
    $("objall").addEventListener("click", function () { S.showAll = !S.showAll; drawObjective(); });
    document.querySelectorAll("input[name=layer]").forEach(function (r) { r.addEventListener("change", function () { S.layer = r.value; BG = null; drawField(); }); });
    var rt; window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(function () { if (!B) return; BG = null; render(); drawComparison(); }, 100); });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () { if (!B) return; drawMdl(); fullRedraw(); drawComparison(); });
  }

  function init(hp) {
    tipEl = $("tip");
    // the demo plays itself when the reader arrives from elsewhere on the site (not when moving
    // between the demo's own panes, and not on a link that names a tick), and pauses on leaving
    var demoBar = document.getElementById("demobar");
    var demoPanes = demoBar ? (demoBar.getAttribute("data-panes") || "").split(/\s+/) : [];
    var currentPane = document.querySelector(".pane.is-current");
    var inDemo = !!currentPane && demoPanes.indexOf(currentPane.id) >= 0;
    if (inDemo && hp.t == null && !(MOTION && MOTION.matches)) setPlaying(true);
    // nav.js shows one pane at a time; a hidden pane has no size, so redraw when one is shown
    document.addEventListener("hybrid-aif:pane", function (e) {
      var now = demoPanes.indexOf(e.detail.id) >= 0;
      if (now && !inDemo && !S.playing && !(MOTION && MOTION.matches)) setPlaying(true);
      if (!now && S.playing) setPlaying(false);
      inDemo = now;
      if (!B) return;
      // a chart drawn while its pane was hidden has a fallback width and a wrong height: drop it
      // and its reservation, so the first draw in view measures an empty box
      dropReservations();
      BG = null; render(); drawComparison();
    });
    // a hash changed on the open page (edited by hand, or a link to another version) reloads
    // nothing: the version and setting it names are applied here
    window.addEventListener("hashchange", function () {
      var hp2 = hashParams(), v2 = hashVersion(hp2);
      if (v2 && v2 !== S.version) setVersion(v2, { setting: hp2.setting, t: hp2.t != null ? hp2.t : null }).catch(function () {});
      else if (B && hp2.setting && replayable(hp2.setting) && hp2.setting !== S.setting) apply({ setting: hp2.setting, t: hp2.t }, false);
    });
    window.__app = {
      set: function (o) {
        // a version change loads its bundle first: the call returns a promise either way
        if (o.version && o.version !== S.version) return setVersion(o.version, o).then(function () { apply(o, true); });
        apply(o, false);
        return Promise.resolve();
      },
      state: function () { return { version: S.version, t: S.t, frac: S.frac, setting: S.setting, playing: S.playing }; }
    };
    function apply(o, fresh) {
      if (o.setting && replayable(o.setting) && (!fresh || o.setting !== S.setting)) { S.setting = o.setting; $("setting").value = o.setting; $("tick").max = run().ticks.length - 1; BG = null; drawGraph(); drawComparison(); }
      if (o.view) $("objmode").value = o.view;
      if (o.agg || o.group) $("objagg").value = o.agg || { first: "action", sequence: "policy" }[o.group] || o.group;
      if (o.all != null) S.showAll = o.all;
      if (o.layer) { S.layer = o.layer; BG = null; document.querySelector("input[name=layer][value=" + o.layer + "]").checked = true; }
      if (o.t != null) { S.t = idxOf(o.t); S.frac = 0; }
      if (o.frac != null) S.frac = clamp(+o.frac, 0, 0.999);
      render();
    }
  }
  function start() {
    var hp = hashParams();
    wire(hp); init(hp);
    var v = hashVersion(hp) || DEFAULT_VERSION;
    setVersion(v, { setting: hp.setting, t: hp.t != null ? hp.t : 0 }).then(function () {
      window.__ready = true;
      document.dispatchEvent(new Event("hybrid-aif:ready"));
    }).catch(function (e) {
      setStatus("");
      var d = document.createElement("p"); d.className = "err"; d.textContent = "Could not load the replay data: " + e.message;
      document.querySelector("main").prepend(d);
    });
  }
  // lazy: let the page paint first, then pull the (large) bundle
  requestAnimationFrame(function () { setTimeout(start, 0); });
})();
