// reifyscene local app. One selection shared by the graph, the timeline and the frame view.
// No framework; Cytoscape.js for the graph. All data comes from the FastAPI backend (/api/...).

const PALETTE = ["#0e7c8b", "#c2573a", "#6b5ca5", "#3f8f4f", "#b58a1b", "#2f6fb0",
                 "#a8487a", "#5f7d1e", "#8a6245", "#4a8a8a", "#9a3f3f", "#556070"];
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
  S.run = name; S.selected = null; S.highlight = null; S.activeEdit = null; S.view = "photo"; S.flipped = false;
  S.graph = await api(`/api/runs/${name}/graph`);
  S.frames = S.graph.frames.length ? S.graph.frames
    : [...new Set(S.graph.entities.flatMap((e) => e.observed))].sort((a, b) => a - b);
  S.fi = 0;
  const cats = Object.keys(S.graph.categories);
  S.colorOf = Object.fromEntries(cats.map((c, i) => [c, PALETTE[i % PALETTE.length]]));
  $("run-meta").textContent = `${S.graph.entities.length} entities · ${S.frames.length} frames · scene: ${S.graph.checkpoint}`;
  $("graph-note").textContent = S.graph.relations_available ? "" : "relations not computed yet";
  const slider = $("frame-slider");
  slider.max = Math.max(0, S.frames.length - 1); slider.value = 0;
  fillQueryOptions(); drawGraph(); drawLegend(); drawTimeline(); drawSelection(); drawEdits(); paint();
}

// -- graph -----------------------------------------------------------------------------------
const short = (id) => id.split(":").pop();
function drawGraph() {
  const els = S.graph.entities.map((e) => ({
    data: { id: e.node_id, label: `${e.category} ${short(e.node_id)}`, color: S.colorOf[e.category],
            size: 18 + 22 * Math.sqrt(e.num_observed / Math.max(1, S.frames.length)) },
  }));
  for (const e of S.graph.edges) els.push({ data: { id: `${e.src}->${e.dst}:${e.type}`, source: e.src, target: e.dst, label: e.type } });
  if (S.cy) S.cy.destroy();
  const ink = getComputedStyle(document.documentElement).getPropertyValue("--ink").trim();
  const line = getComputedStyle(document.documentElement).getPropertyValue("--line").trim();
  S.cy = cytoscape({
    container: $("cy"), elements: els,
    style: [
      { selector: "node", style: { "background-color": "data(color)", width: "data(size)", height: "data(size)",
        label: "data(label)", color: ink, "font-family": "IBM Plex Mono, monospace", "font-size": 10,
        "text-valign": "bottom", "text-margin-y": 4 } },
      { selector: "edge", style: { width: 1.5, "line-color": line, "curve-style": "bezier", label: "data(label)",
        "font-size": 8, color: ink } },
      { selector: "node.sel", style: { "border-width": 3, "border-color": ink } },
      { selector: ".faded", style: { opacity: 0.18 } },
    ],
    // Without relation edges a force layout has nothing to arrange by and collapses into a
    // line; group by category on a grid until relations exist.
    layout: S.graph.edges.length
      ? { name: "cose", animate: false, nodeRepulsion: 9000, idealEdgeLength: 60, padding: 20 }
      : { name: "grid", padding: 24, avoidOverlapPadding: 18,
          sort: (a, b) => a.data("label").localeCompare(b.data("label")) },
  });
  S.cy.on("tap", "node", (ev) => select(ev.target.id()));
  S.cy.on("tap", (ev) => { if (ev.target === S.cy) { S.highlight = null; applyHighlight(); } });
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
  $("e-remove").disabled = !id;
  applyHighlight(); drawSelection(); paint();
}
function entity(id) { return S.graph.entities.find((e) => e.node_id === id); }
function drawSelection() {
  const box = $("selection"); box.innerHTML = "";
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
  if (S.selected) { out.hidden = false; out.onerror = () => { out.hidden = true; };
    out.src = `/api/runs/${S.run}/outline/${f}.png?node=${encodeURIComponent(S.selected)}`; }
  else out.hidden = true;
  const sel = S.selected ? ` · ${entity(S.selected)?.category} ${short(S.selected)} outlined` : "";
  $("stage-tag").textContent = `${view}${sel}`;
  drawViews(); placeCursor();
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
    if (res.entity) { S.highlight = [res.entity]; select(res.entity); }
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
  if (res.template === "relate") return res.available === false ? `Not available: ${res.reason}` : JSON.stringify(res, null, 2);
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
  if ($("q-at-frame").checked && ["select", "count"].includes(t)) args.frame = S.frames[S.fi];
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
  S.playing = setInterval(() => setFrame(S.fi + 1 >= S.frames.length ? 0 : S.fi + 1), 160);
});
$("stage").addEventListener("click", () => { S.flipped = !S.flipped; paint(); });
document.addEventListener("keydown", (ev) => {
  if (ev.target.closest("input, select, textarea")) return;
  if (ev.key === "ArrowLeft") setFrame(S.fi - 1);
  else if (ev.key === "ArrowRight") setFrame(S.fi + 1);
  else if (ev.key === " ") { ev.preventDefault(); S.flipped = !S.flipped; paint(); }
});
window.addEventListener("resize", placeCursor);
init().catch((err) => { $("run-meta").textContent = `Could not load: ${err.message}`; });
