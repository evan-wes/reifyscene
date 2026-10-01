# A local web app for querying, browsing and editing a reconstructed scene

**Status:** BUILT (2026-09-29) — the app runs on all ten loop runs with VLA-3D relations over
time, relation selection and a free-viewpoint 3D view; the five-beat demo is §1 "As built"
**Date:** 2026-09-29
**Serves:** the thin loop of `docs/design/2026-07-09-queryable-graph-definition-spike-plan.md`
(step 3: select → edit → render) and G1/G2 of
`docs/design/2026-09-20-conceptgraphs-verdict-and-graph-plan.md`
**Depends on:** presence gaps, relations, the structured query and the `remove` command (§4,
steps 1–2) — the app is a surface over those, not a second implementation of them

---

## 0. Why

The project's claim is *interpret, query, edit, re-render*. Everything that demonstrates it today
runs from a terminal and ends in files: JSON indexes, image strips, one-off review pages. That can
be checked but not demoed. The app puts the loop in one browser window: ask about the scene, see
the entities and how their relations change over time, remove one, compare before and after.

It is **local**. It runs on the machine that holds the runs and the GPU, because an edit
re-renders the scene. Nothing is published (the standing decision: the repo stays private and
nothing public ships until the queryable-graph gap is closed). A shareable snapshot is a later,
separate mode (§7).

---

## 1. What a demo looks like

1. Open the app, pick a run (`lab-coat`). The graph appears: five people, a few objects, the
   relations between them.
2. Type `select person`. The people light up in the graph and are outlined in the frame view.
3. Scrub the timeline. A person walks behind another; their presence bar shows the gap, and their
   `near` edge disappears across it and comes back.
4. Click one person, press **Remove**. After a few seconds, the frame view shows before and after;
   click to flip. The "took" view shows exactly which Gaussians the removal selected.
5. `count person` now answers from the graph as it was; the edit is a view on the scene, not a
   change to the pipeline's artifacts.

### As built: the demo in five beats (`lab-coat`)

Run `reifyscene serve ../runs`, open http://127.0.0.1:8765, pick **lab-coat**. Each beat is
one claim of *interpret, query, edit, re-render*, shown and not told.

1. **The scene is a graph over time.** Frame 0: the photo beside the graph, three people joined
   by `near`. Press **play**. The graph follows the frame: edges come and go, entities not seen at
   that moment fade. *Claim: the graph is per moment, not one summary of the clip.*
2. **An entity is a thing in the video.** Click **person 2** in the graph. Its outline follows it
   through the clip; its presence bar and its relations appear under the timeline. *Claim: a
   graph node is grounded in pixels at every frame.*
3. **A relation has a lifetime.** Click the row **person 2 is near person 3**. Both are
   outlined (red, amber), and the interval 0–29 is marked. Play loops just those frames. Step to
   30: both are still in view, but the edge is gone. They stepped apart, and the graph says so.
   *Claim: relations are facts with time intervals (VLA-3D's predicates, computed per frame), not
   labels on a clip.*
4. **Ask it.** In Query: `relate` person 3, then tick *at the current frame* and ask again at
   frame 10 and at frame 35. `when` 2 gives the presence span. *Claim: the answers come from
   the graph, with no language model in the loop.*
5. **Edit and re-render.** Select person 2, **Remove**. In about a second, **after** shows the
   clip without them; **took** shows the Gaussians the removal selected. Switch to **3D**, drag
   a few degrees, then click to flip between the edited and original scene from the new viewpoint.
   *Claim: the edit is on the 3D scene, not on the pixels of one frame.*

**Keep off-camera (known and tracked, not hidden if asked):**
- Past ~15° of orbit the view smears: one moving camera never saw that side. The tag says so.
- The removal leaves a hole and some residue where the person stood; inpainting is §7.
- `lab-coat` also carries spurious proposals (cup 0, bird 4, handbag 6: pieces of the fence and
  trees). Selecting them is an honest demo of G3's open problem, not of the loop.
- The moving people are softer than the background: that is the motion model, a separate track.

**Backup runs.** `miami-surf` (two people and `near` over 60+ frames) and `crossing` (`on`
relations among street objects) are the next cleanest. Avoid `blackswan`, `gold-fish` and
`motocross-bumps` for relations: they have no person, so metric scale falls back to a guess.

---

## 2. Shape

```
reifyscene serve ../runs            (one process, one port, started for a demo)
 ├─ FastAPI backend   reifyscene/api/web/
 │    reads runs  →  GraphArtifact, presence, relations        (library calls, §4)
 │    queries     →  reifyscene.query.SceneQuery               (the same code as the CLI)
 │    edits       →  reifyscene.edit.remove                    (FlashSplat + floater filter)
 │    serves      →  photos, renders, masks, edit results
 └─ single-page front end   reifyscene/api/web/static/
      plain JS, no build step; Cytoscape.js and nothing else from a CDN
```

**No logic lives in the web layer.** Every endpoint is a thin wrapper over a function the CLI also
calls, so the app cannot disagree with the pipeline and the functions stay testable without a
browser.

---

## 3. The interface

Three panels on one screen, sharing one selection: click an entity anywhere and it is selected
everywhere.

| panel | shows | interactions |
|---|---|---|
| **Scene graph** | entities as nodes (coloured by category, size by observation count); relation edges (`near`, `above`/`below`, `on`); camera and background nodes hidden by default | click to select; filter by category; edges shown for the current frame or the whole clip |
| **Frame view + timeline** | the photo or the render at the current frame, the selected entity's mask outlined; under it, one presence bar per entity with its gaps, and each relation of the selected entity as an interval | scrub frames; toggle photo / render; click a bar to select that entity |
| **Query + edit** | a query box with the structured templates and their results (entity ids, labels, evidence); the edit controls and the edit's before / after / took views | run a query; highlight its results; **Remove** the selection; flip before/after; choose slackness |

Visual language: the review pages' (IBM Plex, the teal accent, the flip-to-compare stage), so the
app and the evidence pages read as one project. Both themes.

---

## 4. The backend

### Endpoints

| method | path | returns |
|---|---|---|
| GET | `/api/runs` | runs available under the root: name, frame count, entity count, whether a checkpoint exists |
| GET | `/api/runs/{run}/graph` | nodes (id, type, label, category, presence span and gaps, extent) and relation edges with their validity intervals |
| GET | `/api/runs/{run}/frames/{i}.jpg` | the training photo for frame *i* |
| GET | `/api/runs/{run}/render/{i}.jpg` | the trained scene rendered at frame *i*'s camera (cached) |
| GET | `/api/runs/{run}/masks/{i}.png` | the id map for frame *i* (entity outlines are drawn client-side) |
| POST | `/api/runs/{run}/query` | `{template, args}` → matching entity ids with evidence |
| POST | `/api/runs/{run}/edits` | `{op: "remove", entity, slackness}` → an edit id; runs in the background |
| GET | `/api/edits/{id}` | status, then the result: removed Gaussian counts, the gate's measures, frame URLs for before / after / took |

### Edits

- **The first edit on a run** computes FlashSplat's evidence matrix (one backward per training
  frame, under a minute on the 5k-step scenes). It is cached per checkpoint, so every later edit
  on that run is an assignment, a filter and a few renders: seconds.
- **One edit at a time on the GPU.** Edits queue; the UI shows the queue. The GPU power cap is
  checked at startup (the server refuses to start above 400 W, as `gpu_guard.sh` does) and the
  process holds one CUDA context for its lifetime.
- **Results are written under `<run>/edits/<id>/`** (edited checkpoint optional, renders, the
  measures), never into the pipeline's own stage directories. An edit is a view on a run.
- **Which checkpoint:** the run's `render/` by default; a run-level override (e.g. the
  8-views-per-step retrain) is a query parameter.

### Queries (the structured rung — no language model)

| template | args | answers |
|---|---|---|
| `select` | category, optional frame | the entities of that category (present at that frame) |
| `relate` | entity, predicate, optional frame | the entities in that relation with it, and when |
| `when` | entity | its presence span and gaps |
| `count` | category, optional frame | how many |

Natural language (the contract's rungs 1–2) comes after the loop works; the query box is built so a
parser can later produce the same `{template, args}` it sends today.

---

## 5. Process and dependencies

- **New optional dependency group `web`:** `fastapi`, `uvicorn`. Not in the core install; added to
  the environment lock.
- **`reifyscene serve <runs-root> [--port 8765]`** checks the port is free first (other sessions
  run dev servers on this machine) and says so rather than picking another silently. Started for a
  demo, stopped after; nothing runs in the background by default.
- **Memory:** the server runs inside a capped systemd scope like every other job here; one loaded
  checkpoint plus its evidence matrix is well under 4 GB of host RAM on these scenes.
- **Local only:** binds `127.0.0.1`. No auth, because nothing is exposed.

---

## 6. Order of work

Each step is checked before the next.

| | build | done when |
|---|---|---|
| **W0** | presence gaps + relation computer in the graph stage (G1's content) | on `lab-coat`, an entity's edges vanish across its own presence gaps; relations are emitted only as the world frame allows |
| **W1** | `SceneQuery` (the four templates) and `reifyscene.edit.remove`, with a CLI for each | `reifyscene query` and `reifyscene edit` run the loop on `lab-coat` with no hand-edited files; the manifest reports `renderable` grounded with each entity's assigned fraction |
| **W2** | the backend (§4) over W0–W1 | every endpoint tested against a fixture run, without a browser |
| **W3** | the front end, `lab-coat` | the §1 demo runs start to finish in a browser |
| **W4** | all ten loop runs | every run opens, queries and edits; failures are shown, not hidden |

W0 and W1 are the graph plan's G1/G2 slice, needed with or without an app; W2–W4 are the app.

**As built (phase A).** The order changed: the app was stood up first, on the graph as it is, and
W0 (relations) follows -- adopted from VLA-3D (MIT; the research verdict in the graph contract
§3.2), not hand-written rules. Built: presence with labelled gaps, `SceneQuery` (select / count /
when; `relate` reports that relations are not computed yet), `remove` via FlashSplat + floater
filter, `reifyscene query` / `edit` / `serve`, the backend and the three-panel front end.
`serve` serves each run's `render` checkpoint (or one named with `--checkpoint`); the
8-views-per-step recipe it once preferred as a retrain (`render_q3b_batch8_nodecay`) has been
the render stage's default since `91e3aaa`. On all ten runs: the graph loads, queries answer, frames render, and a removal
completes in about a second including the evidence matrix; the server holds all ten scenes in
~2.6 GB host / ~1.8 GB GPU memory. Not yet: the pipeline manifest still reports `renderable:
absent` (the edit's own result reports the assigned fraction); that moves when `remove` is wired
into the pipeline rather than run beside it.

---

## 7. Later, not now

- **A shareable snapshot:** the same front end baked into a self-contained page per run, with
  queries live in the browser and every entity's removal precomputed. Private, like the review
  pages. Built once the local app is right.
- **Natural-language queries** (the contract's rungs 1–2).
- **`move` and `recolor`**, and an edit history that stacks.
- **Re-perceive and diff** (the spike's verification step), shown as a graph diff panel.
- **Inpainting** what a removal uncovers (§4.8 of the identity scope).

---

## 8. Open

1. **Which checkpoint the app treats as a run's scene.** The pipeline's `render/` is still the
   5k-step, 1-view-per-step training on the existing loop runs; the 8-views-per-step retrains live
   beside it. Proposed: the app prefers a sharper checkpoint when one exists, and says which it
   loaded.
2. **Relation thresholds** (`near` distance, `on` contact tolerance) are guesses until W0 measures
   them against what a person would say on `lab-coat`.
3. **Render cache size.** Rendering every frame of every run on first open is minutes of GPU; the
   proposal is lazy rendering with a per-run cache.
