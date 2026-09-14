# Back-half ladders + the queryable-graph contract

**Status:** PROPOSED
**Date:** 2026-09-12
**Branch:** `spike/queryable-graph-definition`
**Grounding survey:** `docs/research/2026-09-12-third-party-components-for-the-back-half.md`
**Refines:** §4A / §4A.1 (degradable ladders) and §5a–§5e of
`docs/design/2026-06-24-full-pipeline-architecture-neurosymbolic-queryable-4d.md`
**Fulfils:** step 2 of `docs/design/2026-07-09-queryable-graph-definition-spike-plan.md` — the
definition contract — folded together with the ladder plan, because the contract *is* the fixed
interface the ladders fill.

---

## 0. What this is

The front half of the pipeline is built as **degradable ladders**: each stage is a fixed artifact
contract plus a best→floor stack of backends, probed top-down at stage start, with the chosen rung
stamped into a capability manifest (§4A). The back half — relations, grounding, query, edit — has
no such treatment. It has stubs and one landed function.

This document gives the back half the §4A treatment, and states the contract those ladders fill.
It follows a survey of what is publishable and released (grounding survey, above), whose conclusion
was: **adopt per-layer, reject whole-pipeline frameworks, and the gap we actually occupy is
explicit temporal entities with typed relations grounded in an editable 4D reconstruction.**

**The governing lesson, inherited from §4A.1:** the ladder's fixed-contract guarantee is about the
**output schema, not the input dependency set**. Stepping down a rung can pull in upstream inputs
the rung above never required — that is what forced the geometry pose/depth split and the
`dyn_mask` stage. Every ladder below is therefore stated with its *input divergence*, not just its
backends.

---

## 1. The abstention defect — small, isolated, first

Independent of every ladder choice below, the graph currently asserts categories it has no
warrant for.

On `davis_lab-coat_full` the `semantics` top rung ran (`siglip2`, `open_vocab`, 8/8 named) against
a **COCO-80 candidate vocabulary** (`siglip_semantics.py:36`), producing `cup`, `bird`, `handbag`
for a scene of people in lab coats behind a fence. Top-1 cosine sits at 0.085–0.117 for every
track with a near-flat top-5 — the signature of a model with no right answer available.

**The discriminative signal is the top-1/top-2 margin, not the absolute score.** The two real
people carry margins 0.045 and 0.033; every junk label is ≤ 0.0105. `handbag` carries the highest
absolute score in the set (0.1170) and is junk — so `min_score`, the abstention hook that already
exists in `classify()`, is the wrong instrument. Full evidence table in the survey, §3.

Three defects, all isolated:

1. **`classify()` cannot abstain on margin.** It takes `min_score` (default `-inf`, never fires).
   Add `min_margin`, config-driven, defaulting to a value in the separating band.
2. **An abstention is overwritten downstream by something worse.** `pipeline.py:996` falls back to
   the detector's COCO `concept` when semantics says `unknown`. SAM2 is class-agnostic and emits
   `concept=''`, so the object category becomes the **empty string**, not `unknown`.
3. **Nothing records that abstention happened.** `SemanticRecord` carries `score` and
   `category_topk` but no margin and no explicit abstention reason, so a downstream consumer cannot
   distinguish "genuinely unknown" from "never asked".

Blast radius: `siglip_semantics.py`, `pipeline.py`, `semantics_artifact.py`, plus tests. No ladder
change, no stage reordering. `SemanticRecord.category` already defaults to `unknown` and `objects`
already branches on it, so the schema absorbs this without a migration.

**Ships as its own PR ahead of the work below.**

---

## 2. The back-half ladders

### 2.1 `semantics` (re-cut) — the vocabulary is a rung axis

| Rung | Backend | Fidelity |
|---|---|---|
| 1 | Qwen2.5-VL caption → open-set label (ConceptGraphs-style) | open-set |
| 2 | SigLIP2 + **scene-appropriate vocabulary** | open-vocab |
| 3 | SigLIP2 + COCO-80 — *today* | closed-vocab, mislabelled off-domain |
| 4 | detector `concept` string | closed-vocab |
| 5 | `unknown` | empty |

**Input divergence.** Rung 1 adds an inference-time LVLM — a model or a service the rungs below
never needed, and the first back-half dependency that can fail for reasons outside the machine.
Rung 2 needs its candidate list from somewhere; if the list is LVLM-proposed, it needs a **pass
over frames before per-track classification**, which is new intra-stage ordering rather than a new
stage. A fixed per-dataset vocabulary avoids that and is the cheaper first move.

**Note the ordering trap:** rung 2 is a *config change* to a rung that already exists and already
runs. It should be tried before rung 1 is built.

### 2.2 `graph · extents` (new capability) — the prerequisite the relations need

Not previously identified. Four of VLA-3D's eight predicates need more than a centroid — `in` and
`on` strictly require extents, `above` and `below` are unreliable without them — and we compute
only centroid trajectories today. The precise per-predicate requirement is in §3.2.

| Rung | Backend | Fidelity |
|---|---|---|
| 1 | oriented 3D box from the entity's **assigned Gaussian set** | oriented, dense |
| 2 | axis-aligned box from lifted track-support points | axis-aligned, sparse |
| 3 | centroid + isotropic radius from mask area × median depth | coarse |
| 4 | centroid only | relations degrade to the centroid-sufficient set |

**Rung 1 reuses step 1's bridge**: the entity→Gaussian assignment already gives a per-entity point
cloud, and extents follow from it directly. That is a second use for a function landed for editing,
which is the strongest argument that the bridge was the right thing to build first.

**Input divergence — and it is severe.** Rung 1 requires the **trained** Gaussian cloud, which
lives in the `render` stage, *downstream* of `graph`. Rung 2 requires only `tracking`, which is
upstream. So the extent ladder **spans two pipeline positions**, exactly as the depth ladder does
(§4A.1). Resolution follows the same precedent: rungs 2–4 serve at graph position; rung 1 is a
**render-time** enrichment that revises extents after training, at its own position. The fourth
invariant governs same-position rungs; cross-position rungs live where their inputs exist.

### 2.3 `graph · relations` (new) — adopt the vocabulary

| Rung | Backend | Fidelity |
|---|---|---|
| 1 | LVLM-derived relations over entity crops (ConceptGraphs-style) | open-set, unverifiable |
| 2 | **geometric predicates over 3D extents + trajectories** (VLA-3D's eight) | typed, checkable |
| 3 | 2D mask overlap + depth ordering | view-dependent, weak |
| 4 | no edges | graceful-empty |

**Rung 2 is the intended operating point,** not a fallback. It is checkable against ground truth,
needs no external service, and is the rung the step-4 audit can actually grade. Rung 1 sits above
it in fidelity but *below* it in verifiability, which is unusual for a ladder and should be
recorded as such.

**Input divergence.** Rung 2 needs §2.2's extents. Rung 3 needs no 3D at all — it is the rung that
still works when geometry has degraded to identity poses, and it is the reason the relations ladder
does not simply bottom out at "no edges".

### 2.4 `graph · renderable` — keep ours, and name the inversion

| Rung | Backend | Fidelity |
|---|---|---|
| 1 | Gaussian Grouping identity encodings (Apache-2.0) | learned, static-only |
| 2 | LangSplat / OpenGaussian language features | learned, static-only, similarity not membership |
| 3 | **mask-majority vote over the trained cloud** — *ours, landed* | geometric, dynamic-capable |
| 4 | mask-centroid neighbourhood | coarse |
| 5 | no handle — query-only graph, no editing | empty |

**Input divergence, and a dependency inversion.** Rungs 1–2 **retrain the splat** with an extra
per-Gaussian channel. Today `graph` *reads* the trained cloud; those rungs would make `render`
depend on `segment` identity, inverting the edge. They are also **static-scene methods** — our
clips are dynamic by construction. Rung 3 is therefore not a compromise: it is the only rung on
this ladder that works on the thing we actually build.

### 2.5 `query` (new stage; four stubs exist) — adopt the design

| Rung | Backend | Fidelity |
|---|---|---|
| 1 | LLM + **Cypher over a graph DB** (arXiv 2510.16643) | freeform NL |
| 2 | LLM → fixed program templates (VisProg-style) | freeform NL, bounded programs |
| 3 | **fixed conceptual templates, no LLM** | count / select / relate / motion |
| 4 | exact-id lookup | trivial |

**Input divergence.** Rung 1 needs the graph exported to a graph database. Choose an **embedded**
engine (Kùzu — in-process, no server) rather than a server (Neo4j), or the floor stops being
dependency-free and the whole ladder inherits a service dependency. Rungs 1–2 need an LLM at
inference; rung 3 does not, which is what keeps the acceptance criteria in the spike plan §10
runnable with no external service.

### 2.6 `compile` / edit (new) — **the ladder we deliberately cap**

| Rung | Backend | Status |
|---|---|---|
| — | diffusion inpainting after removal (Mono4DEditor-style) | **refused — see below** |
| 1 | **Gaussian transform ops on the entity set** — `remove`, `move` | intended top |
| 2 | report-only (the plan, unexecuted) | graceful |

Every other ladder in the pipeline wants to climb. This one must not. Lowering A is **faithful by
construction**: a `remove` leaves a real hole with no plausible fill, and that hole is the honest
output — it is precisely why Level 2 (a generative backbone) exists in the architecture as a
separate level rather than a better rung. A diffusion rung would make the render *look* better
while making it *mean* less.

**Recorded as a decision, not a gap.**

---

## 3. The contract the ladders fill

### 3.1 Node

**One node per persistent physical entity.** Identity is the reid-canonical tracklet. Never
per-frame, never a point track. Node ids embed the stream (`human:{stream}:{id}`,
`object:{stream}:{id}`, `background:{stream}`), which makes **cross-view association an explicit
non-claim** of this contract rather than an implicit gap.

| Block | Field | Status |
|---|---|---|
| **identity** | `category`, `category_topk`, `score`, **`margin`**, **`abstained`**, `caption`, embedding | category shipped; margin/abstained = §1; caption = semantics rung 1 |
| **presence** | `[t_start, t_end]`, observed frames, **gap intervals flagged occlusion-candidate** | span shipped; gap flagging new |
| **position(t)** | world-space trajectory, one point per observed frame | shipped |
| **extent(t)** | oriented or axis-aligned 3D extent | **new — §2.2** |
| **motion** | `velocity(t)`, `speed`, `path_length`, `is_moving` | static/moving shipped; rest new |
| **support** | member track ids, per-frame uncertainty, `mean_support_ratio` | shipped |
| **renderable** | Gaussian index set over the **trained** cloud | assignment landed; storage open, §4 |

**An entity that cannot be named does not become unnamed-but-labelled.** `category` is `unknown`
with `abstained=true` and the margin recorded. Consumers must treat `unknown` as a value, not a
missing field.

### 3.2 Relations

**Adopted from VLA-3D, view-independent, eight predicates:**
`above` · `below` · `closest` · `farthest` · `between` · `near` · `in` · `on`

Each edge carries a **validity interval** — a list of frame ranges over which the predicate holds —
recomputed per frame, so edges appear and disappear as the scene evolves. An edge asserted for the
whole clip and an edge asserted for six frames are the same type with different intervals, never
different types.

**What each predicate needs** (this is the precise version of "relations need extents"):

| Predicate | Sufficient input | Notes |
|---|---|---|
| `closest`, `farthest`, `between` | centroids | class-scoped: "closest *chair* to X" |
| `near` | centroids + **scale-relative** threshold | threshold relative to median scene depth, never metres — our scale is arbitrary up to the Sim(3) re-anchor |
| `above`, `below` | centroids computable, **extents strongly preferred** | centroid-only is defensible but weak; report which was used |
| `in`, `on` | **extents required** | `on` = above ∧ in contact along the gravity axis; not computable from centroids |

Apply VLA-3D's **overlap filter**: drop a relation when target and anchor extents significantly
overlap or one encloses the other.

**Gravity axis.** `above` / `below` / `on` are defined against the gravity axis established by the
ground-plane canonicalizer, not the camera's up. Where no canonicalizer has run, these three
predicates are **not emitted** — they are not silently computed in camera coordinates.

**View-dependent relations are not scene-graph content.** `left-of`, `right-of`, `in-front-of`,
`behind` are resolved at **query time** against a named reference frame (a camera at a stated
timestamp), because a camera-relative relation flips meaning whenever the camera moves — and our
camera always moves; that translation is the only reason reconstruction works at all.

**Relations to `background`** are admitted but flagged weak: the background is one undifferentiated
node owning every track no entity claimed (72% of Gaussians on lab-coat — fence, grass, trees), so
"person `on` background" is true and nearly uninformative. This is the §1 semantics limitation
resurfacing in the edge schema, and the contract names it rather than hiding it.

### 3.3 Presence

Presence spans first to last observation. **Interior gaps are flagged occlusion candidates, not the
entity ceasing to exist.** Object permanence across long occlusions is out of scope, and a gap
carries no claim about where the entity was.

A relation's validity interval is intersected with both endpoints' presence: **an edge is never
asserted over frames where either entity is unobserved.** Absence of an edge during a gap is not
evidence the relation was false.

### 3.4 Renderable handle

Entity id → Gaussian index set over the **trained** cloud (not the seed — training densifies and
prunes, so a seed index means nothing at render time).

Reported per entity: assigned count, assigned fraction, and an explicit **unassignable** flag for
entities too small or too occluded to earn a majority. On lab-coat: two human nodes at 25,636 and
22,518 of 315,882 Gaussians, with 72% claimed by no entity.

**Op-support matrix**, carried from the start so the IR does not fracture silently:

| Op | Lowering A (3DGS, ours) | report-only |
|---|---|---|
| `remove(entity)` | honored — leaves a real hole | planned, not executed |
| `move(entity, delta)` | honored for static transform; **dynamic motion bases are the open risk** | planned |
| `recolor` | not implemented | planned |

---

## 4. Open decisions

1. **Where the entity→Gaussian map is stored.** It belongs with the trained cloud in the `render`
   artifact, which would need `objects` and `graph` added to its dependencies. Deferred until the
   thin loop shows what it reads. **Blocking a known bug:** `render/camera_paths.py:120` reads
   `scene_artifact.dynamic_entities`, a field that has never existed — `follow_subject` would raise
   today. Repair it with whatever schema this settles.
2. **Spark-DSG as export target.** Fit unverified: its dynamic layer means agent trajectories, not
   time-valid relations. Check before adopting; our structure stays the floor either way.
3. **The `near` threshold constant**, and its sensitivity across clips.
4. **Embedded graph engine choice** (Kùzu assumed) — confirm it survives the relation schema.
5. **Audit rubric** — drafted with this contract, fixed before the step-4 audit.

---

## 5. Sequencing

1. **Abstention PR** (§1) — isolated, ships first.
2. **Vocabulary experiment** — semantics rung 2 as a config change on existing runs. Cheap, and it
   tells us how much of the perception problem is vocabulary before anything is built.
3. **Extents** (§2.2) rungs 2–4 at graph position — the relation prerequisite.
4. **Relations** (§2.3) rung 2 against this contract.
5. **Thin loop** — spike-plan step 3, unchanged.
6. **Audit** — spike-plan step 4; findings revise this contract, and the revised contract is the
   terminal deliverable.

Steps 1–2 are independent of everything else and can proceed while this document is still under
review.
