# Stage audit — what every stage owes the scene graph, and the end-to-end plan

**Status:** PROPOSED
**Date:** 2026-09-13
**Evidence run:** `davis_lab-coat_full` (47 frames, all 15 stages, VGGT-Long top rung)
**Supersedes in part:** `docs/design/2026-09-12-back-half-ladders-and-graph-contract.md` (see §5)
**Grounding survey:** `docs/research/2026-09-12-third-party-components-for-the-back-half.md`
**Graph plan + reference verdict:** `docs/design/2026-09-20-conceptgraphs-verdict-and-graph-plan.md` — what ConceptGraphs taught us, what we
adopt from it, and the ordered G1-G4 slice for the graph.
**Direction beyond this plan:** `docs/design/2026-09-19-roadmap-inhouse-components-and-modalities.md` — in-house components,
audio as a second modality, and the grounding law they all depend on. This document's work
items remain the near-term path; that one says where they lead.

---

## 0. What this is, and the change of posture

Every stage in this pipeline is built to satisfy **its own** contract. None is built to satisfy
the **graph's**. This document audits all fifteen stages against a single question — *what must the
graph be able to assert, and does this stage deliver it?* — and reports what it found on the one
clip that has run every stage.

**The posture changes with it.** §4A's degradable ladders optimize for *no combination of backends
fails to produce an artifact set*. That guarantee is intact and it is not what is wrong. What is
wrong is that **the top rung produces a graph that does not mean anything**, and no manifest says
so. Chasing completeness across rung combinations is therefore deprioritized. The goal is now **one
configuration that delivers the full vision end to end** — interpret, query, edit, re-render,
verify — on real footage. Degradability returns once that path exists and is worth protecting.

Concretely: the ladders in the superseded document are retained as *design notes about what a rung
would be*, not as work items. **§7 records the standing decision this implies** — fusion is the
target, the cascade is a measured phase with an exit condition. Work items below target the **golden path** — the top rung of every
stage — and nothing else.

---

## 1. The unifying defect: absolute constants on scale-free quantities

Three stages fail in exactly the same way, and it is not degradation.

| Stage | The constant | The quantity it is applied to |
|---|---|---|
| `semantics` | a COCO-80 candidate list | an **open**-vocabulary model's similarity space |
| `tracking` | `static_threshold_world: 0.05` | world displacement on a **monocular** reconstruction, whose scale is arbitrary up to the Sim(3) re-anchor |
| *(ours, PR #5)* | `min_margin = 0.02` | a top1−top2 cosine gap that **shrinks as the candidate list grows** |

Each stage runs its best backend, reports success, and emits something the graph cannot use. The
lesson generalizes: **any threshold in this pipeline must be relative to a quantity the pipeline
itself measures**, or it is a constant tuned to one clip wearing the costume of a default.

---

## 2. The audit

### 2.1 Capability manifests — 7 of 15 stages stamp none

| Stage | Manifest | Backend / fidelity |
|---|---|---|
| ingest, sync | **missing** | — |
| dyn_mask | yes | `all-static` — correct, a no-op under VGGT (§4A.1) |
| geometry, pose_refine | yes | `vggt-long` / `joint_reconstruction_longseq` |
| segment | yes | `sam2` / `class_agnostic_propagation` |
| reid | yes | `dinov2` / `dino_features` |
| semantics | yes | `siglip2` / `open_vocab` |
| **objects** | **missing** | — |
| humans | yes | `sapiens` / `sapiens_dense` |
| **tracking** | **missing** | — |
| **graph** | **missing** | — |
| optimization, scene, render | **missing** | — |

The three stages the graph depends on most — `tracking`, `objects`, `graph` — are precisely the
three with no manifest. §4A's promise that "degradation is observable, not silent" does not hold
where the graph is built.

**And the manifest is the wrong shape even where it exists.** It records *which backend ran*. It
cannot record *what the graph can therefore assert*. `semantics` stamping `open_vocab` while
emitting `cup` for a person is a green manifest over a worthless output. **Proposed: a second axis**
— each stage declares, per graph-relevant capability, whether it is `grounded`, `degraded` or
`absent`, so "all green, graph is garbage" stops being representable.

### 2.2 Dense 3D already exists for every entity, and the graph discards it

**Revised 2026-09-14.** The first draft named Sapiens per-person depth as the route. There is a
better one: `pose_refine/pointmaps/` holds **70 MB** of per-frame dense 3D — a world-space point per
pixel, **already in the canonical ground-plane frame**, needing no scale alignment and covering
objects as well as people.

Intersecting those pointmaps with the masks the `segment` stage already produces, measured on this
clip:

| entity | positions today | frames with 3D | 3D points per frame |
|---|---:|---:|---:|
| human A | 5 | **47** (all) | ~11,200 |
| human B | **0** | **47** (all) | ~14,400 |
| handbag | **0** | **47** (all) | ~1,000 |
| small tall object | 2 | **7** (its full presence) | ~1,100 |

Every entity gains **full presence coverage**, with centroid *and* extent, from data already on
disk. No new model, no new dependency, no per-crop scale reconciliation.

**The consequence for §2.4:** the point-track starvation does not need to be *fixed* to get
positions — it needs to be **bypassed**. Tracks stay the right source for the jobs in §2.7; they are
the wrong source for position and always were.

Sapiens `dense/` (94 files, 89 MB, depth usable 94/94) remains a **secondary refinement** for people
specifically — higher resolution inside the person crop — but it is per-crop and would need scale
alignment into the world frame, so it is not the first move. `graph/builders.py` currently reduces
all of it to `mean_visible_keypoints`, `has_densepose` and `has_smpl`: two booleans and a float.

**Care required.** Mask edges straddle depth discontinuities, so a naive centroid drags toward
background. This needs a robust estimator — median plus depth-consistency trimming — not a mean.
Absolute magnitudes are meaningless on their own: monocular scale is arbitrary up to the Sim(3), so
only ratios are interpretable.

### 2.3 Motion labels and entity assignment contradict each other

Cross-referencing `static_labels` against the graph's support assignment:

| motion label | → entity | → ambiguous | → background |
|---|---:|---:|---:|
| **moving** | 1 | 4 | **30** |
| static | 8 | 1 | 91 |
| uncertain | 0 | 0 | 9 |

Thirty tracks the pipeline calls **moving** are filed under **background**, and **8 of the 9** tracks
supporting entities are labelled **static**. Entity trajectories are built almost entirely from
points the pipeline itself believes are not moving.

43% of those thirty sit high in the frame, where the treeline is — so some fraction is plausibly
real foliage motion rather than misclassification. **Separating the two needs a controlled check**
(GT masks, or a static-camera clip); it is a work item, not a conclusion.

Related: track **visibility is 35.9%**, and `lifted_valid` matches it exactly — lifting succeeds
wherever a point is visible, so visibility is the binding constraint on 3D evidence, not lifting.

### 2.4 Seeding and assignment — why entities have no positions

Two independent defects, quantified on this clip (144 tracks: 130 never touched an entity, 5 touched
and were rejected, 9 survived).

**Seeding.** Points are seeded on a uniform 12×12 grid **on frame 0 only** — `grid_size: 12` is
literally the 144. Clips shorter than one window (48 frames) are seeded exactly once, ever; longer
clips re-seed at window boundaries, which is a *side effect of windowing*, not a mechanism for new
arrivals. Consequences no amount of grid density repairs:
- an entity appearing after frame 0 is never tracked at all;
- at frame 0 the two people occupy **3.6% and 3.2%** of the frame (they walk toward the camera and
  end at 15.9% / 17.4%), so ~5 seed points each — before any filtering.

**`tracking` declares `segment` as a dependency and never receives it** (`api/stages.py:138` passes
only `ingest` and `pose_refine`). This is the only genuine declared-but-unpassed dependency in the
pipeline — all fifteen were checked. The wiring for mask-seeded tracking is half-built and inert.

**Assignment.** `assign_track_support` requires a track to be inside an entity's mask on ≥50% of
"the frames they share", where *shared* counts every frame in which the entity has a mask and the
track is visible **anywhere in the image** (`graph/track_support.py:94`). So the denominator is the
entity's whole co-visible lifetime, not the track's dwell on it. A point that correctly follows a
person for 60 frames of a 230-frame presence scores 0.26 and is discarded.

**This actively punishes success upstream.** A `reid` merge across an occlusion gap *lengthens* an
entity's presence, inflating the denominator for every track, lowering every ratio. Better re-id
destroys more 3D support. The rule conflates *which entity is this track on* (disambiguation) with
*how long did it stay* (quality); it only needs to answer the first. **Proposed:** score over the
frames the track was inside **any** entity, and keep the inside-count as the separate dwell signal.

### 2.5 Information that dies at the graph boundary

All of this exists, is correct, and does not reach the graph:

| What | Where it lives | What the graph gets |
|---|---|---|
| open-vocab category for **humans** | `SemanticsArtifact` | nothing — `nodes.py:15` hardcodes `label="human"` |
| `SemanticsArtifact` itself | its own stage | **never passed** — not a declared dependency of `graph`; category arrives second-hand via `objects` |
| `semantic_topk` | `ObjectState.attributes`, present on **196/196** | dropped — `_entities_by_stream` builds a fresh dict |
| semantic embeddings | `semantics/*.npz` | **0/196** on objects; never joined |
| Sapiens `dense` (depth/normals) | `humans/dense/`, 89 MB | `has_densepose: False` |
| gravity axis | `geometry.metadata['world_frame']`, `ground_plane_applied=True` | **not recorded** — nothing downstream can tell whether `above`/`below` are meaningful |

These are the cheapest repairs in the audit: the data is already computed and already correct.

### 2.6 Stage-specific findings

- **`segment` — instance separation is a `segment` defect, not `reid`.** SAM2 produced **8 tracks**
  for a scene with three people plus objects; `reid` performed **no merges** (`merge_map` empty).
  The third person was never a separate instance to begin with. Any fix belongs in `segment`.
- **`humans` — the manifest cannot express the stage's own shape.** §4A defines two *independent*
  sub-ladders (dense: Sapiens→Keypoint-RCNN→bbox · SMPL: GVHMR→4D-Humans→Multi-HMR→skip), and
  `CapabilityManifest` has one `backend_used` and one `fidelity`. Result: `smpl/` is **empty (0
  files)**, `densepose/` empty, Sapiens `seg` empty on **81 of 94** instances — and the manifest
  reads `sapiens_dense`, true only of the one axis that worked. A stage with independent capability
  axes needs a manifest entry per axis.
- **`geometry` / `pose_refine` — healthy, and better than assumed.** VGGT-Long ran as the joint top
  rung; the ground-plane canonicaliser ran and **succeeded** (32,908 inliers of 72,000,
  `world_frame.type = ground_plane_z_up`). The gravity axis is real. Earlier doubt about whether
  `above`/`below` were computable was misplaced — they are; the axis just is not recorded downstream.
- **`tracking` (bug) — `track_artifact.py:333` double-rotates.** It calls the unconditional
  `_points_to_z_up` while `_debug_basis_matrix()` returns identity once the canonicaliser has run.
  The graph reads raw `lifted_xyz` and is therefore **correct**; this accessor, used by debug/QA
  exports, is the one that is wrong.
- **`ingest` / `sync` — no graph-relevant defect found**, beyond stamping no manifest.
- **`dyn_mask` — correctly a no-op** under the VGGT joint rung, per §4A.1.

### 2.7 What point tracks are actually for — and why the order forecloses it

**Added 2026-09-14**, correcting §2.4's framing that tracking is simply broken.

Tracks do four jobs. The pipeline asks them for the one they are worst at and none of the three they
are uniquely good at.

| Job | Who does it today | State |
|---|---|---|
| **position** | point tracks — the only source | badly; superseded by §2.2's pointmaps |
| **motion / static-dynamic** | point tracks | contradicts the assignment (§2.3) |
| **identity continuity through occlusion** | `reid`, by DINOv2 appearance | **tracks not used at all** |
| **objectness — catching mask false positives/negatives** | nothing | **not implemented** |

A point that physically persists through an occlusion and lands inside a different tracklet
afterwards is *geometric proof* they are the same entity — evidence appearance cannot supply, and
precisely what is needed on a clip chosen because three similar-looking people are a re-id stress
test. A region whose points scatter across incompatible depths and motions is evidence it is not one
object; a mask that blinks out while points continue coherently is evidence segmentation dropped a
frame.

**The stage order forecloses the identity use.** `reid` declares only `(ingest, segment)` and runs
*before* `tracking`, so identity is decided before point tracks exist.

**And the reordering is legal.** §4A.1's fourth invariant reads "the ladder gates whether each
upstream stage does real work — **it** never reorders": the *ladder* must not reorder the pipeline,
i.e. order must not be a function of which rung runs. It does not freeze the order against redesign.
Verified against the real registry: `tracking` declares `(ingest, pose_refine, segment)`, so moving
it to immediately after `segment` is dependency-legal **with no other change**, and its only
consumer (`graph`) stays downstream. `reid` can then declare `tracking` and merge on geometric
continuity as well as appearance.

§4A's own rationale asked only for tracking *after `segment`* ("so its dynamic split is seeded by
real foreground masks"). Placing it five stages later, after `reid`/`semantics`/`objects`/`humans`,
buys nothing and is what costs us track-informed identity.

**Cost to weigh:** `STAGE_ORDER` defines what counts as upstream for resume, and every existing
run was produced under the old order. A migration decision, not a blocker.

**The cycle this does not resolve.** Using tracks to catch mask errors needs tracks *before*
`segment`, and tracks want masks to be seeded well. Genuinely circular. The repo already has the
shape of the answer: `dyn_mask` is a deliberately coarse, geometry-free pass placed before
`geometry` purely to break the SfM-needs-a-motion-mask cycle. The same pattern applies — a coarse
grid pass before `segment` for objectness, a mask-seeded pass after it for identity.

### 2.8 The fusion stage is a scaffold with no solver

`reifyscene/optimize/` carries a full factor-graph layout — `reprojection`, `depth_consistency`,
`epipolar`, `smoothness`, `track_consistency`, `graph_factors`, `human_factors`, `object_factors`,
`priors`, `time_alignment` — and a three-stage schedule that runs on every clip.

It optimizes nothing. From the lab-coat artifact's own `pending_todos`: *"Connect stage variables to
a real optimizer backend."*, *"Add depth and epipolar consistency terms."*, *"Add object and
scene-graph consistency factors."* Its reported residuals are **observation counts** —
`track_consistency: 6768` is exactly 47 frames × 144 tracks. `human_trajectories: 0`,
`object_trajectories: 0`.

**And its position is wrong for what fusion would be for.** `optimization` runs *after* `graph`, so
even fully built it would refine cameras and trajectories *given* an entity inventory. It could
never revise the inventory. That is the structural reason a `segment` error is permanent.

**Parallelism is not the prize.** The true dependency DAG is **13 deep against 15 stages** — only
`(reid, tracking)` and `(semantics, humans)` can overlap. Extraction is genuinely sequential because
nearly everything needs poses. What a joint solve buys is not speed; it is not having to break the
cycles by hand.

---

## 3. The golden path

The end-to-end vision, and what each step needs to stop being blocked:

| Step | Blocked on | From |
|---|---|---|
| **Interpret** — entities with identity, presence, position, extent | positions + extents from pointmaps ∩ masks; mask-seeded tracking for *identity*; semantics reaching the graph | §2.2, §2.7, §2.5 |
| **Relate** — typed relations with validity intervals | extents (which §2.2 also supplies); scale-relative thresholds; gravity axis recorded | §1, §2.5 |
| **Query** — language in, entities out | the graph being worth querying; a typed schema | §2.5 |
| **Edit** — `remove`, `move` | entity→Gaussian assignment **wired into the pipeline** (landed as a library, zero callers) | — |
| **Re-render** | already works | — |
| **Verify** — re-perceive and diff | the judge reusing the *same* resolved vocabulary, or it measures the vocabulary | §1 |

Nothing on that list is a research problem. Every one is a wiring or threshold problem in a stage
that already runs.

---

## 4. Reference comparison — run the published tools on our data

The survey found released packages that do parts of this. Following the precedent already
established for Shape of Motion (`scripts/setup_reference_tool.sh`, isolated conda env, pinned
commit, reproducible patches with sentinel checks, lockfiles in `scripts/locks/`, safety guards that
refuse to target the `reifyscene` or `base` env, vendored into gitignored `third_party/`), each
reference below gets the same treatment.

**Why this matters more under the new posture:** if a published tool produces a better graph on our
footage than we do, that is the fastest possible signal about which of our stages is the real
bottleneck — and it is evidence no amount of internal auditing produces.

| Reference | What it comparably does | Input we must supply | Licence |
|---|---|---|---|
| **Feature4X** | monocular video → 4D feature field → segmentation, language edit ("delete the car"), 4D VQA, novel view | **the raw clip** — same input as us | check at vendor time |
| **ConceptGraphs** | posed RGB-D → open-vocab 3D scene graph with captions + LLM-derived relations | our VGGT poses + depth, exported to its RGB-D convention | released |
| **Gaussian Grouping** | SAM-supervised per-Gaussian identity → object removal / recomposition | posed frames; it trains its own splat | Apache-2.0 |

Secondary, if the first three pay off: **DynamicGSG** (dynamic scene-graph updates on a Gaussian
map, IROS 2025, released), **4D LangSplat** (time-sensitive open-vocab queries, CVPR 2025,
released), and **Heracles** — the reference implementation for graph-DB + Cypher querying, which is
**GPL-3.0**; it stays in gitignored `third_party/` and is never linked or vendored into this
Apache-2.0 repo.

**What to measure.** Not PSNR — these are graph questions:
- entities recovered vs. a human count on the same clip (our lab-coat: 2 human nodes for 3 people);
- labels a person agrees with (ours: 2 of 8);
- fraction of entities carrying a usable 3D position (ours: 4 of 8 have **zero**);
- whether `remove(entity)` leaves the entity's region and nothing else.

`ConceptGraphs` is the sharpest single comparison, because it consumes exactly what our front half
already produces and outputs exactly what our `graph` stage outputs. **Feature4X is the sharpest
end-to-end one**, because it takes the same raw clip and claims the same four verbs.

---

## 5. What this supersedes

From `2026-09-12-back-half-ladders-and-graph-contract.md`:

- **§0 and §2 (the five back-half ladders) are demoted** to design notes. Rung enumeration is not a
  work item under the new posture.
- **§2.2's cross-position extents ladder is withdrawn.** Positions and extents come from the
  geometry pointmaps already available at `graph` position (§2.2 of this document); the backward-edge
  problem it tried to solve does not arise.
- **§3 (the contract) stands**, with three corrections: `position(t)` is *not* shipped as "one point
  per observed frame"; per-frame uncertainty is *not* shipped; `category` is absent on human nodes.
- **§1 (the abstention defect) shipped** as PR #5, with the threshold caveat of §1 above outstanding.

From `2026-09-12-third-party-components-for-the-back-half.md`, corrections are tracked separately;
the adoption verdicts stand.

---

## 6. Work items, in order

1. **Entity positions and extents from pointmaps ∩ masks** (§2.2) — all eight entities, data already
   on disk, robust estimator required.
2. **Move `tracking` after `segment`, seed query points from masks, give `reid` the tracks** (§2.4,
   §2.7). For identity and objectness — *not* position, which item 1 supersedes.
3. **Fix the support denominator** (§2.4) so re-id success stops destroying 3D support.
4. **Stop dropping data at the graph boundary** (§2.5) — pass `SemanticsArtifact` to `graph`, carry
   `semantic_topk` and embeddings, label human nodes, record the gravity axis.
5. **Make thresholds scale-relative** (§1) — `static_threshold_world`, and revisit `min_margin`.
6. **Manifests for `tracking`, `objects`, `graph`** plus the second axis (§2.1), and per-axis entries
   for `humans`.
7. **Wire `entity_gaussians` into the pipeline** — it has zero callers today.
8. **Reference comparison** (§4), starting with ConceptGraphs on our own geometry.

Items 1–4 decide whether the graph means anything. Item 8 runs in parallel and should start early:
it is the only item that produces an outside opinion, and it feeds §7's exit condition.

---

## 7. Standing decision — fusion is the target, the cascade is a measured phase

**Decided 2026-09-14.** Recorded here so it is not relitigated each time a stage misbehaves.

### The target is fusion

The project's claim is a **correct, enumerable inventory** grounded in something you can re-render.
Not a better reconstruction — Feature4X already has that, from the same monocular input, and has no
entities at all. The inventory is the differentiator, and it is the one thing that can be *plainly
wrong* rather than gracefully vague.

A cascade structurally cannot guarantee it. `segment` commits to an entity count from 2D appearance
before any 3D evidence exists, and every later stage inherits that commitment. This clip shows both
halves: the count is wrong, **and** the evidence to correct it is sitting in the same run —
pointmaps (§2.2), motion coherence (§2.3), track continuity (§2.7). Reordering does not fix it,
because whichever stage runs first commits first (§2.7's cycle). The architecture already agrees: the
factor graph exists and was never connected (§2.8).

### The cascade is the next phase, and it has a job

Not convenience — evidence:

1. **We do not yet know how much of the error is structural.** The audit found mostly *wiring*.
   Those are cascade-compatible fixes, and the one clearly structural failure (the entity count) is
   measured on a single clip where the third person is marginal anyway (7 of 47 frames, 0.7% of
   frame). One clip is not a mandate to rebuild.
2. **Fusion needs uncertainty and we have none.** Every `GraphNode.confidence` is hardcoded `1.0`
   (`nodes.py:11,15,19,23`). A solve weights evidence; ours cannot say which evidence is weak.
   Producing calibrated uncertainty *is* the cascade work.
3. **You cannot fuse your way out of a bad question.** A joint solve over a `semantics` stage that
   was asked a closed-vocabulary question yields a confidently fused wrong answer.

### Four rules that make cascade work count as fusion groundwork

1. **Every stage emits evidence with uncertainty, not a committed answer.** PR #5 is the template: a
   stage saying "I am not sure, and here is the margin" instead of asserting `cup`.
2. **No stage destroys information it merely cannot use** (§2.5).
3. **Thresholds are relative to quantities the pipeline measures**, never absolute constants (§1).
4. **The graph's inputs stay observations, not conclusions**, so a future solve has something to
   weigh.

Follow these and the cascade work *is* the fusion substrate. Ignore them and it is a rewrite later.

### The exit condition, stated so it can fire

After items 1–4 land, run the inventory against clips with a known answer. **If the inventory is
still wrong on a clip where the correcting evidence demonstrably exists elsewhere in the run, that is
the fusion trigger**: `optimization` moves ahead of `graph` and gets a real solver backend. If the
inventory comes out right, the cascade was sufficient and fusion was premature.

Item 8 feeds the same decision from outside: if ConceptGraphs builds a better inventory **from our
own geometry**, the limit is our architecture rather than our reconstruction.
