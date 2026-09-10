# Spike plan (v2) — grounding the definition of a queryable scene graph for room4d

**Status:** PROPOSED v2 (supersedes v1; v1 reasoning preserved in §10)
**Inception:** 2026-07-09 · **Revised:** 2026-07-13
**Branch:** `spike/queryable-graph-definition`
**Refines:** §5a (graph), §5b (query), §5e (round-trip eval), P3–P4 of
`docs/design/2026-06-24-full-pipeline-architecture-neurosymbolic-queryable-4d.md`.

---

## 0. What this spike is (and the reframe from v1)

Block-2 (the back-half P3–P7) is **paused**. Two adversarial review panels reshaped this spike:

- **Panel 1 (on the P3 graph design):** the current graph keys node identity on
  `track:{stream}:{track_id}` — a *grid-seed index* that cannot survive a re-render, so the
  eval's diff join key is noise; point-tracks are not entities; presence-from-visibility
  conflates "removed" with "occluded."
- **Panel 2 (on v1 of this plan):** v1's instrument (a null-edit re-render correspondence
  residual) was **confounded** (SAM2 re-partitions the render, real↔render embedding domain gap,
  own-renderer near-tautology, no ground truth, no chance baseline) **and overreached** — a
  node-only appearance measurement claiming to define a whole queryable graph (edges, relations,
  presence, queries all unmeasured).

**Decision (the two-part split).** Separate the two jobs the v1 plan tangled together:

- **Part 1 — THIS DOC — Definition + human audit.** Build a queryable scene graph on a *single
  video stream* whose **nodes are real physical entities** (not tracked pixels) and whose **edges
  are real physical relationships**, with per-entity **position and motion over time**. Validate it
  by **human audit across many DAVIS sequences** — visualize the graph and its temporal evolution,
  and answer basic conceptual questions. Terminal deliverable: a **grounded, human-validated
  definition**.
- **Part 2 — DEFERRED (separate plan, later).** Novel-view evaluation and **two-graph comparison**
  (the round-trip correspondence floor / discriminability measurement). *All* re-render,
  cross-input, and Level-2 machinery — and every Panel-2 confound — lives here.

**The load-bearing insight.** A graph that is **human-understandable** (real entities + real
relations) is the *precondition* for being **discriminative** in the back-half: a meaningful scene
change changes such a graph in a way a person would recognize. So the human audit is not a soft
check — it is the **acceptance test for discriminability**. Get "means something" right first;
measure "diffs cleanly across a generated answer" in Part 2.

---

## 1. The question and the terminal deliverable

**Question.** *What is a queryable scene graph for a single video stream, such that (a) it means
something to a person — nodes are real entities, edges are real physical relations, each node
carries its position and motion over time — and (b) it is therefore discriminative when diffed in
the back-half?*

**Terminal deliverable.** `docs/design/2026-07-09-queryable-scene-graph-definition.md` — the
grounded contract: the **node model**, the **attribute schema**, the **edge/relation vocabulary**,
the **query surface**, and — per Panel 2 — a **per-axis grounded / partially-grounded / ungrounded
verdict** with the evidence that earns it, plus the honest boundary of what a single monocular
stream can support.

---

## 2. The corrected graph model (what we build and audit)

**Node = one persistent physical entity.** Identity = the reid-canonical tracklet spanning every
frame the entity appears. **One node per entity** — never per-frame, never a point-track.

```
EntityNode( id = canonical entity )
  identity   : category, category_topk, caption, semantic embedding      # static
  presence   : [t_start, t_end] + observed frame set                     # temporal extent
  position(t): world-space trajectory — one 3D point per observed frame  # temporal attribute
  motion     : velocity(t), speed, path length, is_moving                # derived from position(t)
  support    : the member point-track ids + per-frame uncertainty        # provenance
```

**How `position(t)` is built (the key correction over v1).** A segmentation mask is a *single-frame
observation*, not the entity's position. Instead:

1. Assign point-tracks to entities by **mask membership** (a point-track belongs to the entity whose
   mask its 2D location falls inside, per frame).
2. The entity's position at frame *t* = a **robust aggregate (median) of its member points' 3D
   positions at t**, carried across the whole presence interval → a trajectory.
3. Robustness + uncertainty come for free from the aggregation (many points vs one); low-support
   entities are flagged and may fall back to a **mask-centroid lift** as a secondary source.

This **redeems the point-tracks**: the "meaningless pixel tracks" become the *support* that gives a
real entity its motion, rather than being nodes themselves.

**World frame is required for motion.** Per-entity motion over time must live in a **common world
frame** — otherwise DAVIS's moving cameras contaminate it. So we run room4d's **geometry stage
(VGGT: pose + depth)** per sequence and lift member points to world space. (Relations *between two
co-present entities* are camera-invariant, so those don't strictly need pose — but the single-entity
trajectory does.)

**Edges = physical relations between entity nodes, each with a validity time-interval.**
Recomputed per frame from the trajectories + masks, so edges appear and disappear over time =
**the graph's evolution.** Relation vocabulary (rung-1 set):
- **spatial:** `near`, `above` / `below`, `left-of` / `right-of`, `in-front-of` / `behind` (depth)
- **support / contact:** `on-top-of`, `supported-by`
- **containment / overlap:** `contains`, `overlaps`

**Presence = existence, handled honestly.** Presence spans first-to-last observation; interior gaps
are flagged as **occlusion candidates**, *not* logged as the entity ceasing to exist (Panel 1's
visibility≠existence fix). Full object-permanence across long occlusions is out of scope here
(design §6, future).

---

## 3. Building blocks — exists vs. build

| Element | Status | Source |
|---|---|---|
| Frame ingest (`--frames-dir`) | exists | `ingest` |
| Pose + depth (world frame) | exists | `geometry` (VGGT) |
| Instance masks + persistent ids | exists | `segment` (SAM2) **or** DAVIS GT masks |
| Canonical entity identity | exists | `reid` (DINOv2) |
| Human-readable label + caption + embedding | exists | `semantics` (SigLIP2 / COCO floor) |
| Point-tracks + per-frame lifted 3D | exists | `tracking` (`lifted_xyz`) |
| **Per-entity trajectory aggregator** | **BUILD** | mask-membership → median world trajectory + uncertainty (fills the `object_lifter.lift_to_3d` stub) |
| **Relation computer** | **BUILD** | spatial / support / containment, with time-intervals |
| **Entity-graph assembler** | **BUILD** | one node/entity + attributes + edges |
| **Audit viewer** | **BUILD** | self-contained HTML: playback + graph overlay + time scrub + query panel |
| **Conceptual-query executor** | **BUILD** | count / select / relate / motion queries over the graph |

All BUILD items are **throwaway spike code** in `../spike_graph_definition/` — they import room4d
and invoke stage outputs, but add **no** pipeline wiring.

---

## 4. Validation — the human audit (the acceptance test)

**The viewer** (self-contained HTML, one per sequence): plays the frames with entity labels + edges
overlaid on masks; a **time scrubber** to watch the graph evolve; a **conceptual-question panel**
answered *from the graph*.

**Conceptual questions = executed queries** (this is what earns the word "queryable" — Panel 2
demanded at least one real query per family):
- **count / select:** "how many entities?", "what categories are present?"
- **relate:** "what is the person near / on / holding?", "what is on top of the table?"
- **motion:** "which entities are moving?", "who approached whom?"

Fixed templated panel first; a freeform **NL→query** box (small LLM parser, VisProg-style per §5b)
is a stretch rung.

**Acceptance criteria (per axis, honest — each gets grounded / partial / ungrounded + evidence):**
- **Nodes are real entities:** a human agrees nodes ↔ the real objects/people, **one node per
  entity**, stable identity through the clip.
- **Edges are real relations:** asserted relations are physically true; false-positive / missing
  rate is acceptable by a fixed rubric.
- **Queryable:** the conceptual questions return human-correct answers.

**Controls (Panel fixes that still apply to Part 1):**
- **GT DAVIS masks first, then SAM2** — isolates the *graph definition* from *segmentation error*.
  A good definition on GT masks that degrades on SAM2 is a *perception* finding, not a definition
  failure.
- **Report per-entity trajectory uncertainty**; flag low-support entities rather than trusting them.
- **Multiple sequences with contrasting motion/content** (DAVIS breadth) — **no conclusion from one
  clip** (Panel 2's N=1 objection).
- **A short written audit rubric** so "a human agrees" is a checklist, not a vibe.

---

## 5. Protocol — staged, start tiny

- **Rung 0 (thin vertical):** ONE DAVIS sequence, **GT masks**, geometry, per-entity trajectories,
  spatial edges, a minimal viewer. Goal: **see a real scene graph and decide if it means
  something.** Go/no-go on the model before generalizing.
- **Rung 1:** add semantic labels + reid identity + the full relation set + the conceptual-query
  panel; audit on **~3–5 diverse sequences** (GT masks).
- **Rung 2:** swap GT masks → **SAM2**; measure how perception noise degrades the graph; widen
  sequences.
- **Rung 3 (scale):** batch over many DAVIS sequences; a gallery for audit-at-scale; freeform NL
  queries as a stretch.

Each rung is human-audited; findings feed the definition doc.

---

## 6. Scope boundary — explicitly NOT in Part 1

Deferred to **Part 2** (the eval-floor spike, planned after the definition lands): re-render,
novel-view synthesis, re-perception, **two-graph diff / correspondence floor**, Level-2 world-model
proxies, discriminability *measurement*. Part 1 **defines and human-validates** the graph as a
meaningful, queryable object; Part 2 **measures** whether it diffs discriminatively across a
generated answer.

---

## 7. Deliverables and homes

- **Spike code + heavy outputs** (per-sequence graphs, geometry, viewers): `../spike_graph_definition/`
  — throwaway sibling, **not** git-tracked, **not** wired into the pipeline.
- **Committed (this branch):** this plan; the definition doc on completion; and a few exported
  example graphs / viewer screenshots to illustrate the definition.

---

## 8. Cost, logistics, risks

- **GPU:** `geometry` (VGGT) per sequence is the main cost; segment/reid/semantics are moderate.
  Start with a handful of sequences. Prompt the power-limit / temp-watchdog check before GPU rungs
  (standing rule).
- **DAVIS:** `../data/Davis2017/DAVIS-2017-trainval-480p` (JPEG frames + `Annotations` GT masks);
  ingest via `--frames-dir`.
- **Risks:**
  - VGGT may fail on fast-motion DAVIS clips → pick sequences it handles, flag the ones it doesn't.
  - Low-support entities → noisy trajectory → report uncertainty, fall back to mask-centroid.
  - Relation thresholds (near / on distances) may need per-scene tuning → report sensitivity, prefer
    scale-relative thresholds.
  - "Human audit" is subjective → fixed rubric + multiple sequences.

---

## 9. Open decisions (proposed defaults)

1. **Relation set (rung 1):** `{near, above/below, left/right-of, in-front/behind, on-top-of,
   supported-by, overlaps, contains}`. *Proposed; open.*
2. **Trajectory frame:** world-space via VGGT pose (for motion); camera-space per-frame for
   co-present relations. *Proposed; open.*
3. **Query panel:** fixed conceptual templates first, freeform NL→query stretch. *Proposed; open.*
4. **Viewer:** self-contained HTML per sequence + a gallery. *Proposed; open.*
5. **Audit rubric:** a short per-sequence checklist for nodes / edges / queries — drafted with
   rung 0. *Proposed; open.*

---

## 10. Appendix — v1 → v2 (why it reshaped)

v1 pre-registered a single instrument: the **null-edit round-trip residual** (perceive → render a
near-identity answer → re-perceive → diff), intended both to measure the eval's correspondence floor
*and* to ground the graph definition. The two panels showed one instrument could do neither cleanly:

- **It couldn't cleanly measure the floor.** Same-trajectory rung-1 is rigged toward a trivial pass
  and sits inside the own-renderer near-tautology (§5e); the headline "spurious diff" is dominated by
  SAM2 re-partitioning the render and the real↔render embedding domain gap; there was no ground-truth
  or chance baseline, and the "delta of 1" reference was asserted, never computed.
- **It couldn't ground the definition.** A node-appearance measurement says nothing about edges,
  relations, presence semantics, or an actual executed query — yet the deliverable claimed to define
  a *queryable* graph. Presence-from-visibility re-imported the exact conflation the spike existed to
  kill.

**v2 resolves this by splitting the aims** (§0): Part 1 grounds the *definition* by **building the
graph and auditing it against human judgment on real, varied footage** — a construction+audit task
that needs no re-render and therefore inherits none of those confounds — while the *discriminability
measurement* moves to a dedicated Part 2 done rigorously (novel-view, partition-controlled,
GT-anchored, chance-baselined) where it belongs. The corrected **entity-with-trajectory node model**
(§2) replaces v1's implicit point-track / single-mask-lift model.
