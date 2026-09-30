// reifyscene local app. One selection shared by the graph, the timeline and the frame view.
// No framework; Cytoscape.js for the graph. All data comes from the FastAPI backend (/api/...).

const PALETTE = ["#0e7c8b", "#c2573a", "#6b5ca5", "#3f8f4f", "#b58a1b", "#2f6fb0",
                 "#a8487a", "#5f7d1e", "#8a6245", "#4a8a8a", "#9a3f3f", "#556070"];
const PRED_COLORS = { near: "#7a8a8f", above: "#2f6fb0", below: "#6b5ca5", on: "#c2573a", in: "#b58a1b",
                      between: "#3f8f4f", closest: "#a8487a", farthest: "#8a6245" };
const PREDS = Object.keys(PRED_COLORS);
const $ = (id) => document.getElementById(id);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v);
  }
  for (const c of kids) if (c != null) e.append(c);
  return e;
};
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `${r.status} ${r.statusText}`);
  return r.json();
};

const S = {
  runs: [], run: null, graph: null, frames: [], fi: 0,
  view: "photo", flipped: false, selected: null, highlight: null,
  edits: {}, activeEdit: null, colorOf: {}, cy: null, playing: null,
  edgeScope: "frame", preds: new Set(["near", "above", "below", "on", "in", "between"]),
  selRel: null,   // index into S.graph.edges of the selected relation
};

// -- runs ------------------------------------------------------------------------------------
async function init() {
  S.runs = await api("/api/runs");
  const sel = $("run-select");
  sel.innerHTML = "";
  for (const r of S.runs) sel.append(el("option", { value: r.name, text: r.name.replace(/^loop_/, "") }));
  sel.addEventListener("change", () => loadRun(sel.value));
  const want = new URLSearchParams(location.search).get("run");
  const first = S.runs.find((r) => r.name === want) || S.runs.find((r) => r.name === "loop_lab-coat") || S.runs[0];
  if (!first) { $("run-meta").textContent = "no runs found"; return; }
  sel.value = first.name;
  await loadRun(first.name);
  setInterval(pollQueue, 3000);
}

async function loadRun(name) {
  S.run = name; S.selected = null; S.selRel = null; S.highlight = null; S.activeEdit = null; S.view = "photo"; S.flipped = false;
  S.graph = await api(`/api/runs/${name}/graph`);
  S.frames = S.graph.frames.length ? S.graph.frames
    : [...new Set(S.graph.entities.flatMap((e) => e.observed))].sort((a, b) => a - b);
  S.fi = 0;
  const cats = Object.keys(S.graph.categories);
  S.colorOf = Object.fromEntries(cats.map((c, i) => [c, PALETTE[i % PALETTE.length]]));
  $("run-meta").textContent = `${S.graph.entities.length} entities · ${S.frames.length} frames · scene: ${S.graph.checkpoint}`;
  const rm = S.graph.relations_meta;
  S.present = Object.fromEntries(S.graph.entities.map((e) => [e.node_id, new Set(e.observed)]));
  $("rel-meta").textContent = rm ? `Relations: ${rm.source}, scale from ${rm.metric_scale?.source?.replace("_", " ") || "–"}` +
    (rm.gravity_aligned ? "" : " · no gravity: vertical relations not computed") : "";
  const slider = $("frame-slider");
  slider.max = Math.max(0, S.frames.length - 1); slider.value = 0;
  fillQueryOptions(); drawPredChips(); drawGraph(); drawLegend(); drawTimeline(); drawSelection(); drawEdits(); paint();
}

// -- graph -----------------------------------------------------------------------------------
const short = (id) => id.split(":").pop();
function drawGraph() {
  const els = S.graph.entities.map((e) => ({
    data: { id: e.node_id, label: `${e.category} ${short(e.node_id)}`, color: S.colorOf[e.category],
            size: 14 + 14 * Math.sqrt(e.num_observed / Math.max(1, S.frames.length)) },
  }));
  const known = new Set(S.graph.entities.map((e) => e.node_id));
  S.graph.edges.forEach((e, i) => {
    if (!known.has(e.src) || !known.has(e.dst)) return;
    const also = e.attrs?.and;
    els.push({ data: { id: `e${i}`, source: e.src, target: e.dst, pred: e.type, color: PRED_COLORS[e.type] || "#888",
      label: e.type === "between" && also ? `between (& ${short(also)})` : e.type,
      directed: e.type !== "near" ? 1 : 0, ivs: e.attrs?.intervals || [] } });
  });
  if (S.cy) S.cy.destroy();
  const ink = getComputedStyle(document.documentElement).getPropertyValue("--ink").trim();
  const line = getComputedStyle(document.documentElement).getPropertyValue("--line").trim();
  S.cy = cytoscape({
    container: $("cy"), elements: els,
    style: [
      { selector: "node", style: { "background-color": "data(color)", width: "data(size)", height: "data(size)",
        label: "data(label)", color: ink, "font-family": "IBM Plex Mono, monospace", "font-size": 10,
        "text-valign": "bottom", "text-margin-y": 4 } },
      { selector: "edge", style: { width: 1.5, "line-color": "data(color)", "curve-style": "bezier", opacity: 0.55,
        "font-size": 8, color: ink, "text-rotation": "autorotate", "target-arrow-shape": "none" } },
      // labels only on the selected entity's edges: on every edge they bury the graph
      { selector: "edge.lab", style: { label: "data(label)", width: 2.5, opacity: 1 } },
      { selector: "edge[directed = 1]", style: { "target-arrow-shape": "triangle", "target-arrow-color": "data(color)",
        "arrow-scale": 0.8 } },
      { selector: "edge.off", style: { display: "none" } },
      { selector: "node.sel", style: { "border-width": 3, "border-color": ink } },
      { selector: ".faded", style: { opacity: 0.18 } },
      // not observed on this frame: a ghost, not gone -- an occlusion gap does not end an entity
      { selector: "edge.relsel", style: { width: 4, opacity: 1, label: "data(label)", "font-size": 10,
        "z-index": 10 } },
      { selector: "edge.reldim", style: { opacity: 0.12 } },
      { selector: "node.relsub", style: { "border-width": 4, "border-color": "#ff4040" } },
      { selector: "node.relobj", style: { "border-width": 4, "border-color": "#f0aa1e" } },
      { selector: "node.relthird", style: { "border-width": 4, "border-color": "#aa6ef0" } },
      { selector: "node.ghost", style: { opacity: 0.25, "border-width": 1.5, "border-style": "dashed",
        "border-color": ink, "background-opacity": 0.35 } },
    ],
    layout: { name: "preset" },
  });
  S.cy.on("tap", "node", (ev) => select(ev.target.id()));
  S.cy.on("tap", "edge", (ev) => selectRelation(+ev.target.id().slice(1)));
  // Entities on a circle, grouped by category: `near` holds between most pairs in these scenes,
  // and a force layout packs a dense graph into overlapping nodes. On a circle every entity
  // keeps its own place and relations read as chords. Without relations, a grid.
  const byCategory = (a, b) => a.data("label").localeCompare(b.data("label"), undefined, { numeric: true });
  (S.cy.edges().length
    ? S.cy.nodes().layout({ name: "circle", sort: byCategory, padding: 30, avoidOverlap: true, spacingFactor: 1.05 })
    : S.cy.nodes().layout({ name: "grid", padding: 24, avoidOverlapPadding: 18, sort: byCategory })).run();
  S.cy.fit(undefined, 40);
  updateEdges();
  S.cy.on("tap", (ev) => { if (ev.target === S.cy) { S.highlight = null; applyHighlight(); } });
}
function edgeActive(ivs, f) { return ivs.some(([a, b]) => a <= f && f <= b); }
function updateEdges() {
  if (!S.cy) return;
  const f = S.frames[S.fi];
  const perFrame = S.edgeScope === "frame";
  S.cy.nodes().forEach((n) => n.toggleClass("ghost", perFrame && !S.present[n.id()]?.has(f)));
  S.cy.edges().forEach((ed) => {
    const on = S.preds.has(ed.data("pred")) && (S.edgeScope === "clip" || edgeActive(ed.data("ivs"), f));
    ed.toggleClass("off", !on);
    ed.toggleClass("lab", !!S.selected && (ed.data("source") === S.selected || ed.data("target") === S.selected));
    const chosen = S.selRel !== null && ed.id() === `e${S.selRel}`;
    ed.toggleClass("relsel", chosen);
    ed.toggleClass("reldim", S.selRel !== null && !chosen);
    if (chosen) ed.removeClass("off");              // the chosen relation stays visible
  });
  const rel = S.selRel !== null ? S.graph.edges[S.selRel] : null;
  S.cy.nodes().forEach((n) => {
    n.toggleClass("relsub", !!rel && n.id() === rel.src);
    n.toggleClass("relobj", !!rel && n.id() === rel.dst);
    n.toggleClass("relthird", !!rel && n.id() === rel.attrs?.and);
  });
  const shown = S.cy.edges().filter((ed) => !ed.hasClass("off")).length;
  const seen = S.cy.nodes().filter((n) => !n.hasClass("ghost")).length;
  $("graph-note").textContent = !S.graph.relations_available
    ? (S.graph.relations_meta ? "no relations found" : "relations not computed for this run")
    : perFrame ? `frame ${f} · ${seen} seen · ${shown} relations` : `whole clip · ${shown} relations`;
}
function drawPredChips() {
  const box = $("pred-chips"); box.innerHTML = "";
  const present = new Set(S.graph.edges.map((e) => e.type));
  for (const p of PREDS) {
    if (!present.has(p)) continue;
    const n = S.graph.edges.filter((e) => e.type === p).length;
    const i = el("i"); i.style.background = PRED_COLORS[p];
    const b = el("button", { type: "button", "aria-pressed": String(S.preds.has(p)), title: `show ${p} relations` }, i, `${p} ${n}`);
    b.addEventListener("click", () => { S.preds.has(p) ? S.preds.delete(p) : S.preds.add(p);
      b.setAttribute("aria-pressed", String(S.preds.has(p))); updateEdges(); drawRelTimeline(); });
    box.append(b);
  }
}
document.querySelectorAll("#edge-scope button").forEach((b) => b.addEventListener("click", () => {
  S.edgeScope = b.dataset.scope;
  document.querySelectorAll("#edge-scope button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  updateEdges();
}));
const nameOf = (id) => { const e = entity(id); return e ? `${e.category} ${short(id)}` : short(id); };
// Phrased from the selected entity's side, as the query does: each vertical fact is stored once
// ("B above A"), so with A selected it reads "A below B"; near is symmetric.
function sentence(e, from) {
  const also = e.attrs?.and;
  if (e.type === "between") return `${nameOf(e.src)} between ${nameOf(e.dst)} & ${nameOf(also)}`;
  if (e.type === "closest" || e.type === "farthest") return `${nameOf(e.src)} ${e.type} ${entity(e.src)?.category || ""} to ${nameOf(e.dst)}`;
  if (from === e.dst && (e.type === "above" || e.type === "below"))
    return `${nameOf(e.dst)} ${e.type === "above" ? "below" : "above"} ${nameOf(e.src)}`;
  if (from === e.dst && e.type === "near") return `${nameOf(e.dst)} near ${nameOf(e.src)}`;
  return `${nameOf(e.src)} ${e.type} ${nameOf(e.dst)}`;
}
// -- relation selection: its entities outlined in role colours through the clip -----------------
function selectRelation(i) {
  const e = S.graph.edges[i];
  if (!e) return;
  S.selRel = i;
  S.highlight = [e.src, e.dst, e.attrs?.and].filter(Boolean);
  const ivs = e.attrs?.intervals || [];
  const f = S.frames[S.fi];
  if (ivs.length && !edgeActive(ivs, f)) {     // jump to the nearest interval's start
    const starts = ivs.map(([a]) => a);
    const near = starts.reduce((b, a) => (Math.abs(a - f) < Math.abs(b - f) ? a : b), starts[0]);
    S.fi = nearestIndex(near);
  }
  applyHighlight(); drawSelection(); drawRelTimeline(); paint();
}
function clearRelation() { S.selRel = null; S.highlight = null; applyHighlight(); drawSelection(); drawRelTimeline(); paint(); }
function relFrames(e) { return new Set((e.attrs?.intervals || []).flatMap(([a, b]) => Array.from({ length: b - a + 1 }, (_, k) => a + k))); }

function drawRelTimeline() {
  const box = $("rel-timeline"); box.innerHTML = "";
  if (!S.frames.length) return;
  const focus = S.selected || (S.selRel !== null ? S.graph.edges[S.selRel].src : null);
  if (!focus) return;
  const rels = S.graph.edges.map((e, i) => [e, i]).filter(([e, i]) => (S.preds.has(e.type) || i === S.selRel) &&
    [e.src, e.dst, e.attrs?.and].includes(focus));
  box.append(el("h3", { text: rels.length ? `Relations of ${nameOf(focus)} (${rels.length}) · click one to follow it`
    : S.graph.relations_available ? `No shown relations for ${nameOf(focus)}` : "" }));
  const f0 = S.frames[0], span = S.frames[S.frames.length - 1] - f0 + 1, f = S.frames[S.fi];
  for (const [e, idx] of rels.slice(0, 40)) {
    const ivs = e.attrs?.intervals || [];
    const track = el("div", { class: "track" });
    for (const [a, b] of ivs) { const d = el("div", { class: "iv" });
      d.style.left = `${(100 * (a - f0)) / span}%`; d.style.width = `${(100 * (b - a + 1)) / span}%`;
      d.style.background = PRED_COLORS[e.type]; track.append(d); }
    const row = el("div", { class: `rel-row${edgeActive(ivs, f) ? " active" : ""}${idx === S.selRel ? " chosen" : ""}`,
      title: `${e.type} · intervals ${ivs.map(([a, b]) => `${a}–${b}`).join(", ")}`, role: "button", tabindex: "0",
      onclick: () => selectRelation(idx) },
      el("span", { class: "sent", text: sentence(e, focus) }), track);
    box.append(row);
  }
}

function drawLegend() {
  const lg = $("legend"); lg.innerHTML = "";
  for (const [c, n] of Object.entries(S.graph.categories)) {
    const i = el("i"); i.style.background = S.colorOf[c];
    lg.append(el("span", { title: `select ${c}`, onclick: () => runQuery("select", { category: c }) }, i, `${c} ${n}`));
  }
}
function applyHighlight() {
  const hl = S.highlight ? new Set(S.highlight) : null;
  S.cy.nodes().forEach((n) => n.toggleClass("faded", !!hl && !hl.has(n.id())));
  S.cy.nodes().forEach((n) => n.toggleClass("sel", n.id() === S.selected));
  document.querySelectorAll(".timeline [data-node]").forEach((d) =>
    d.classList.toggle("faded", !!hl && !hl.has(d.dataset.node)));
  document.querySelectorAll(".timeline .name").forEach((d) => d.classList.toggle("sel", d.dataset.node === S.selected));
}

// -- selection --------------------------------------------------------------------------------
function select(id) {
  S.selected = id;
  S.selRel = null;
  updateEdges();
  $("e-remove").disabled = !id;
  applyHighlight(); drawSelection(); drawRelTimeline(); paint();
}
function entity(id) { return S.graph.entities.find((e) => e.node_id === id); }
function drawSelection() {
  const box = $("selection"); box.innerHTML = "";
  if (S.selRel !== null) {
    const r = S.graph.edges[S.selRel];
    const dl = el("dl");
    const key = (c, t) => { const i = el("i"); i.style.cssText = `display:inline-block;width:10px;height:10px;border:2px solid ${c};margin-right:5px`; return el("span", {}, i, t); };
    for (const [k, v] of [["relation", sentence(r)], ["frames", (r.attrs?.intervals || []).map(([a, b]) => `${a}–${b}`).join(", ")],
      ["holds", `${r.attrs?.num_frames ?? "–"} of ${r.attrs?.frames_both_present ?? "–"} frames both are seen`]]) dl.append(el("dt", { text: k }), el("dd", { text: v }));
    const keys = el("dd", {}, key("#ff4040", nameOf(r.src)), " ", key("#f0aa1e", nameOf(r.dst)));
    if (r.attrs?.and) keys.append(" ", key("#aa6ef0", nameOf(r.attrs.and)));
    dl.append(el("dt", { text: "outlined" }), keys);
    box.append(dl, el("button", { type: "button", class: "ghost", text: "clear relation (Esc)", onclick: clearRelation }));
    return;
  }
  const e = S.selected && entity(S.selected);
  if (!e) { box.append(el("p", { class: "meta", text: "Click an entity in the graph or the timeline." })); return; }
  const gaps = e.gaps.length ? e.gaps.map((g) => `${g.start}–${g.end}${g.kind === "dropout" ? " (dropout)" : ""}`).join(", ") : "none";
  const dl = el("dl");
  for (const [k, v] of [["entity", e.node_id], ["category", e.category + (e.abstained ? " (abstained)" : "")],
    ["caption", e.caption || "–"], ["present", `frames ${e.first}–${e.last} · ${e.num_observed} observed`],
    ["gaps", gaps]]) dl.append(el("dt", { text: k }), el("dd", { text: v }));
  box.append(dl);
}

// -- timeline ---------------------------------------------------------------------------------
function drawTimeline() {
  const tl = $("timeline"); tl.innerHTML = "";
  if (!S.frames.length) return;
  const f0 = S.frames[0], f1 = S.frames[S.frames.length - 1], span = f1 - f0 + 1;
  const x = (f) => `${(100 * (f - f0)) / span}%`, w = (n) => `${(100 * n) / span}%`;
  const ents = [...S.graph.entities].sort((a, b) => a.category.localeCompare(b.category) || a.node_id.localeCompare(b.node_id));
  for (const e of ents) {
    const name = el("div", { class: "name", "data-node": e.node_id, title: e.node_id, text: `${e.category} ${short(e.node_id)}`,
      onclick: () => select(e.node_id) });
    const track = el("div", { class: "track", "data-node": e.node_id });
    let run = null;
    const flush = (end) => { if (run !== null) { const s = el("div", { class: "seg-on" });
      s.style.left = x(run); s.style.width = w(end - run + 1); s.style.background = S.colorOf[e.category]; track.append(s); } };
    let prev = null;
    for (const f of e.observed) { if (prev === null || f !== prev + 1) { flush(prev); run = f; } prev = f; }
    flush(prev);
    for (const g of e.gaps) {
      const s = el("div", { class: g.kind === "dropout" ? "seg-drop" : "seg-gap", title: `${g.kind} ${g.start}–${g.end}` });
      s.style.left = x(g.start); s.style.width = w(g.end - g.start + 1); track.append(s);
    }
    track.addEventListener("click", (ev) => {
      const r = track.getBoundingClientRect();
      const f = Math.round(f0 + ((ev.clientX - r.left) / r.width) * span - 0.5);
      setFrame(nearestIndex(f)); select(e.node_id);
    });
    tl.append(name, track);
  }
  const key = el("div", { class: "timeline-key" });
  const k = (cls, t) => { const i = el("i"); i.style.background = `var(${cls})`; return el("span", {}, i, t); };
  key.append(el("span", { text: "bars: observed" }), k("--gap-occ", "occlusion candidate (3+ frames)"), k("--gap-drop", "dropout"));
  tl.append(key);
  const cursor = el("div", { class: "cursor", id: "tl-cursor" });
  tl.append(cursor);
  placeCursor();
  applyHighlight();
}
function placeCursor() {
  const c = $("tl-cursor"); const tl = $("timeline");
  if (!c || !S.frames.length) return;
  const tracks = tl.querySelectorAll(".track"); if (!tracks.length) return;
  const f0 = S.frames[0], span = S.frames[S.frames.length - 1] - f0 + 1;
  const tlr = tl.getBoundingClientRect(), tr = tracks[0].getBoundingClientRect();
  const last = tracks[tracks.length - 1].getBoundingClientRect();
  const f = S.frames[S.fi];
  c.style.left = `${tr.left - tlr.left + (tr.width * (f - f0 + 0.5)) / span}px`;
  c.style.top = `${tr.top - tlr.top}px`;          // over the bars only, not the key below them
  c.style.height = `${last.bottom - tr.top}px`;
}
const nearestIndex = (f) => S.frames.reduce((best, v, i) => (Math.abs(v - f) < Math.abs(S.frames[best] - f) ? i : best), 0);

// -- frame view -------------------------------------------------------------------------------
const VIEWS = [["photo", "photo"], ["render", "render"], ["before", "before"], ["after", "after"], ["took", "took"]];
function drawViews() {
  const seg = $("view-seg"); seg.innerHTML = "";
  const edit = S.activeEdit && S.edits[S.run]?.find((x) => x.id === S.activeEdit);
  for (const [v, lab] of VIEWS) {
    const needsEdit = ["before", "after", "took"].includes(v);
    const b = el("button", { type: "button", text: needsEdit && edit ? `${lab}` : lab,
      "aria-pressed": String(S.view === v && !S.flipped), onclick: () => { S.view = v; S.flipped = false; paint(); } });
    if (needsEdit && !(edit && edit.status === "done")) b.disabled = true;
    seg.append(b);
  }
}
function frameUrl(view, f) {
  if (["before", "after", "took"].includes(view)) return `/api/edits/${S.activeEdit}/${view}/${f}.jpg`;
  if (view === "render") return `/api/runs/${S.run}/render/${f}.jpg`;
  return `/api/runs/${S.run}/frames/${f}.jpg`;
}
function flipTarget(view) { return { after: "before", took: "before", before: "after", render: "photo", photo: "render" }[view]; }
function paint() {
  if (!S.frames.length) return;
  const f = S.frames[S.fi];
  const view = S.flipped ? flipTarget(S.view) : S.view;
  $("frame-label").textContent = f;
  $("frame-slider").value = S.fi;
  const img = $("frame-img"), sp = $("stage-spinner");
  const url = frameUrl(view, f);
  if (img.dataset.src !== url) {
    sp.hidden = view === "photo"; img.dataset.src = url;
    img.onload = () => { sp.hidden = true; }; img.onerror = () => { sp.hidden = true; };
    img.src = url;
  }
  const out = $("outline-img");
  const rel = S.selRel !== null ? S.graph.edges[S.selRel] : null;
  const nodes = rel ? [rel.src, rel.dst, rel.attrs?.and].filter(Boolean) : S.selected ? [S.selected] : [];
  if (nodes.length) { out.hidden = false; out.onerror = () => { out.hidden = true; };
    out.src = `/api/runs/${S.run}/outline/${f}.png?` + nodes.map((n) => `node=${encodeURIComponent(n)}`).join("&"); }
  else out.hidden = true;
  let sel = S.selected ? ` · ${entity(S.selected)?.category} ${short(S.selected)} outlined` : "";
  if (rel) sel = ` · ${sentence(rel)}${edgeActive(rel.attrs?.intervals || [], f) ? "" : " (not at this frame)"}`;
  $("stage-tag").textContent = `${view}${sel}`;
  drawViews(); placeCursor(); updateEdges();
  if (S.selected || S.selRel !== null) drawRelTimeline();
}
function setFrame(i) { S.fi = Math.max(0, Math.min(S.frames.length - 1, i)); paint(); }

// -- query ------------------------------------------------------------------------------------
function fillQueryOptions() {
  const cat = $("q-category"); cat.innerHTML = "";
  for (const c of Object.keys(S.graph.categories)) cat.append(el("option", { value: c, text: c }));
  const ent = $("q-entity"); ent.innerHTML = "";
  for (const e of S.graph.entities) ent.append(el("option", { value: e.node_id, text: `${e.category} ${short(e.node_id)}` }));
  syncQueryInputs();
}
function syncQueryInputs() {
  const t = $("q-template").value;
  $("q-category").hidden = !["select", "count"].includes(t);
  $("q-entity").hidden = !["when", "relate"].includes(t);
  $("q-predicate").hidden = t !== "relate";
  if (["when", "relate"].includes(t) && S.selected) $("q-entity").value = S.selected;
}
async function runQuery(template, args) {
  const ans = $("q-answer");
  try {
    const res = await api(`/api/runs/${S.run}/query`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ template, args }) });
    ans.textContent = describe(res);
    if (res.entities) { S.highlight = res.entities.map((e) => e.node_id); applyHighlight(); }
    if (res.entity) {
      const partners = (res.relations || []).flatMap((r) => [r.src, r.dst, r.and]).filter(Boolean);
      S.highlight = [res.entity, ...partners]; select(res.entity);
    }
  } catch (err) { ans.textContent = `Could not answer: ${err.message}`; }
}
function describe(res) {
  const a = res.args || {};
  const at = a.frame != null && a.frame !== "" ? ` at frame ${a.frame}` : "";
  if (res.template === "select") return `${res.count} × ${a.category}${at}\n` +
    res.entities.map((e) => `  ${e.node_id}  frames ${e.first}–${e.last}${e.gaps.length ? `  (${e.gaps.length} gaps)` : ""}`).join("\n");
  if (res.template === "count") return `${res.count} × ${a.category}${at}`;
  if (res.template === "when") { const p = res.presence;
    return `${res.entity}\npresent frames ${p.first}–${p.last}, ${p.num_observed} observed\n` +
      (p.gaps.length ? p.gaps.map((g) => `  gap ${g.start}–${g.end}: ${g.kind}`).join("\n") : "  no gaps"); }
  if (res.template === "relate") {
    if (res.available === false) return `Not available: ${res.reason}`;
    if (!res.count) return `No ${a.predicate && a.predicate !== "any" ? a.predicate + " " : ""}relations for ${nameOf(res.entity)}${at}`;
    return `${res.count} relation${res.count > 1 ? "s" : ""}${at}\n` +
      res.relations.map((r) => `  ${r.sentence}   frames ${r.intervals.map(([x, y]) => `${x}–${y}`).join(", ")}`).join("\n");
  }
  return JSON.stringify(res, null, 2);
}
$("q-template").addEventListener("change", syncQueryInputs);
$("query-form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const t = $("q-template").value;
  const args = {};
  if (["select", "count"].includes(t)) args.category = $("q-category").value;
  if (["when", "relate"].includes(t)) args.entity = $("q-entity").value;
  if (t === "relate") args.predicate = $("q-predicate").value;
  if ($("q-at-frame").checked && ["select", "count", "relate"].includes(t)) args.frame = S.frames[S.fi];
  runQuery(t, args);
});

// -- edits ------------------------------------------------------------------------------------
$("e-remove").addEventListener("click", async () => {
  if (!S.selected) return;
  const st = $("e-status"); st.className = "edit-status";
  const body = { op: "remove", entity: S.selected, slackness: parseFloat($("e-slack").value), filtered: $("e-filter").checked };
  try {
    const job = await api(`/api/runs/${S.run}/edits`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body) });
    const rec = { id: job.id, node: S.selected, status: job.status, params: body, started: Date.now() };
    (S.edits[S.run] ||= []).unshift(rec);
    drawEdits(); pollEdit(rec);
  } catch (err) { st.className = "edit-status fail"; st.textContent = `Could not start: ${err.message}`; }
});
async function pollEdit(rec) {
  const st = $("e-status");
  const tick = async () => {
    try {
      const j = await api(`/api/edits/${rec.id}`);
      rec.status = j.status;
      const secs = Math.round((Date.now() - rec.started) / 1000);
      if (j.status === "done") {
        rec.result = j.result;
        const r = j.result;
        st.className = "edit-status";
        st.textContent = `Removed ${short(rec.node)}: ${r.removed_fg.toLocaleString()} + ${r.removed_bg.toLocaleString()} Gaussians ` +
          `(${(100 * r.assigned_fraction).toFixed(2)}% of the scene)` +
          (r.filter_dropped ? `, filter dropped ${r.filter_dropped.outlier} isolated + ${r.filter_dropped.oversized} oversized` : "") +
          ` · ${r.seconds}s` + (r.renderable === "unassignable" ? " · UNASSIGNABLE: no Gaussian's evidence favours it" : "");
        S.activeEdit = rec.id; S.view = "after"; S.flipped = false; drawEdits(); paint(); return;
      }
      if (j.status === "failed") { st.className = "edit-status fail"; st.textContent = `Edit failed: ${j.error}`; drawEdits(); return; }
      st.textContent = `${j.status === "queued" ? "Queued" : "Removing"} ${short(rec.node)}… ${secs}s` +
        (secs > 8 && j.status === "running" ? " (the first edit on a run computes its evidence; later ones take seconds)" : "");
      setTimeout(tick, 1000);
    } catch (err) { st.className = "edit-status fail"; st.textContent = err.message; }
  };
  tick();
}
function drawEdits() {
  const ul = $("e-list"); ul.innerHTML = "";
  for (const rec of S.edits[S.run] || []) {
    const e = entity(rec.node);
    const li = el("li", { class: rec.id === S.activeEdit ? "active" : "",
      text: `remove ${e ? e.category : ""} ${short(rec.node)} · slack ${rec.params.slackness}${rec.params.filtered ? " · filtered" : ""} · ${rec.status}`,
      onclick: () => { if (rec.status === "done") { S.activeEdit = rec.id; S.view = "after"; S.flipped = false; drawEdits(); paint(); } } });
    ul.append(li);
  }
}
async function pollQueue() {
  try { const q = await api("/api/queue");
    $("queue-state").textContent = q.running.length || q.queued.length ? `GPU: ${q.running.length} running, ${q.queued.length} queued` : "GPU idle";
  } catch { $("queue-state").textContent = "server unreachable"; }
}

// -- controls ---------------------------------------------------------------------------------
$("frame-slider").addEventListener("input", (ev) => setFrame(+ev.target.value));
$("prev").addEventListener("click", () => setFrame(S.fi - 1));
$("next").addEventListener("click", () => setFrame(S.fi + 1));
$("play").addEventListener("click", () => {
  if (S.playing) { clearInterval(S.playing); S.playing = null; $("play").textContent = "▶ play"; return; }
  $("play").textContent = "❚❚ pause";
  S.playing = setInterval(() => {
    if (S.selRel !== null) {                     // loop the relation's own frames
      const fs = relFrames(S.graph.edges[S.selRel]);
      for (let k = 1; k <= S.frames.length; k++) {
        const i = (S.fi + k) % S.frames.length;
        if (fs.has(S.frames[i])) { setFrame(i); return; }
      }
    }
    setFrame(S.fi + 1 >= S.frames.length ? 0 : S.fi + 1);
  }, 160);
});
$("stage").addEventListener("click", () => { S.flipped = !S.flipped; paint(); });
document.addEventListener("keydown", (ev) => {
  if (ev.target.closest?.("input, select, textarea")) return;
  if (ev.key === "ArrowLeft") setFrame(S.fi - 1);
  else if (ev.key === "ArrowRight") setFrame(S.fi + 1);
  else if (ev.key === " ") { ev.preventDefault(); S.flipped = !S.flipped; paint(); }
  else if (ev.key === "Escape" && S.selRel !== null) clearRelation();
});
window.addEventListener("resize", () => { placeCursor(); if (S.cy) { S.cy.resize(); S.cy.fit(undefined, 40); } });
init().catch((err) => { $("run-meta").textContent = `Could not load: ${err.message}`; });
