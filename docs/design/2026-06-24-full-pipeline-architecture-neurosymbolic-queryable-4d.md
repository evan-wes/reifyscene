# Full-pipeline architecture — neurosymbolic queryable 4D scene graph + query-conditioned render-as-answer

**Date:** 2026-06-24
**Status:** DRAFT — for review (not yet eng-reviewed; not yet built)
**Framing source:** the 2026-06-21 reframe (room4d = research/learning vehicle) + the lit-review
`docs/research/2026-06-21-neurosymbolic-queryable-4d-scene-graphs-and-query-conditioned-generation.md`.
**Supersedes:** the renderer-productionization design as a standalone effort
(`a private design note`) — folded in here as F0.

---

## 0. Decisions locked this session

1. **room4d is a research/learning vehicle, not a product.** Optimize for depth / correctness /
   reproducibility / a novel contribution. The "complete pipeline" is the apparatus.
2. **Build the complete pipeline; re-cut the *back-half* contracts; keep the proven front-half and
   the stage/artifact/resume machinery.** Strangler-fig against new contracts, reuse all proven pieces.
3. **Contribution target: a graph→geometry *compiler* + a graph-space *eval*.** Structured as a
   classic compiler — a shared front-end + IR (`query → queryable 4D graph → render-plan IR`) and
   per-target back-end lowering. **The compiler seam *is* the contribution.**
4. **Render-as-answer, own-renderer first (Level 1).** v1's answer is a novel view/video rendered
   from the *reconstructed* 4D Gaussian scene via room4d's own proven trainers, along a
   query-generated camera path. A **frozen-backbone world-model adapter (Level 2)** is a
   designed-but-unbuilt second lowering target behind the same IR.
5. **Eval = round-trip (cycle) consistency in scene-graph space.** `generate answer → re-perceive
   with room4d's own stack → diff the graph`. The perception→graph stack is **dual-use: builder
   *and* judge**.
6. **Doc altitude:** lock the full 11-stage map + contracts at a high level (proven front-half
   mostly as-is); go deep only on the redesigned back-half (graph / query / compiler / render /
   eval).
7. **No-ship / aim-high framing (eng-review recalibration, 2026-06-24).** room4d is a personal
   learning project — **nothing to ship, no plan to publish** unless something genuinely novel
   surfaces. So the plan is *not* optimized to cut scope or hit a milestone; it sequences so
   **validation compounds**, treats hard risks as **experiment/learning checkpoints (not gates)**,
   and treats the stub stages as **explicit build-out scope**. The build plan (§7) extends all the
   way to the ambitious end state, with the far phases deliberately coarse.
8. **Foundation-complete gate before the back-half (2026-06-25).** The entire front-half is built to
   a real, no-stub state and run **end-to-end on multi-stream rig data** — **GATE G1** — before any
   back-half (graph/query/compiler/eval) work begins. **Real temporal sync** is built and **validated
   on the fixed rig** (must recover the known ~0 offset) even though synced rigs don't strictly need
   it — it proves the sync impl before unsynced streams depend on it. The real multi-stream joint
   optimizer is the **calibration arc's M2/M3** (`a private note`), not abstract "learning
   value": multi-stream high-fidelity reconstruction lowers the eval's noise floor (§5e/§8), so it is
   foundation, de-risked first.
9. **Front-half perception redesign — foundation-model stack + degradable ladders (2026-06-26).**
   Two adversarial deep-research passes (`a private note`) verified a modern
   stack that **replaces the hand-tuned 2D associator front-half**. This *evolves* §0.2/§0.6 ("keep
   the proven front-half as-is"): **perception is now explicit redesign scope**; the rest of the
   front-half spine (geometry / pose_refine / sync / optimization) is still kept + built-out. Core
   principle: **decouple identity from motion** — instance perception (SAM 3 mask-propagation →
   DINOv3 re-ID → open-vocab semantics) owns identity; CoTracker point-tracks own motion. Every
   perception stage is a fixed **artifact contract** filled by a **fallback ladder** (foundation
   model → lighter model → today's backend → graceful-empty), so a missing/ungated model degrades
   *fidelity*, never *completeness*; the ladder floor is today's pipeline (strict superset). Runs
   **staged** (load→infer→release) within 32 GB. Deep design in §4A.

---

## 1. Purpose

room4d's north star artifact is a **promptable, context-aware novel video**: type *"orbit the
person in the red jacket between t=2s and t=5s"* or *"remove the red car"* and get a rendered
free-viewpoint answer. The *purpose* of building it is to **understand the literature, reproduce
what's proven, and push the frontier** — not to ship.

This document re-derives the pipeline so that the genuine research frontier (a queryable 4D scene
graph whose query *answer* is a generated novel view/video) is first-class, and so that room4d's
proven reconstruction/render backbone is reused rather than rebuilt.

## 2. Contribution thesis — the wedge

The lit-review's finding: the field splits into mature halves that **do not meet**. Recent 4D
*generation* conditions on **geometry** (CAT4D / GEN3C / VerseCrafter / LiveWorld / Vista4D);
recent *reasoning* outputs **text or robot actions** (4D-LLM/4D-PSG, DyGEnc, DovSG); language-4D
work (4D LangSplat, Instruct-4DGS) *queries/edits* a reconstruction but does not generate the
*answer*. **No 2024–2026 system closes** `query → temporal/4D scene graph → neurosymbolic reasoning
→ generated novel-view video answer`.

The unclaimed ground is the **conditioning interface** plus its **evaluation**. room4d's two
candidate contributions:

- **(C1) A graph→geometry compiler.** Turn a *4D scene-graph query plan* into the explicit control
  signals a renderer consumes. Built as a compiler:
  - **Front-end + IR (the novel ~80%, target-agnostic):** `query → queryable 4D graph → render-plan
    IR` (which entities, time window, camera trajectory, per-entity edits).
  - **Back-end lowering (per target):** IR → own 3DGS renderer call (Level 1, v1); IR → world-model
    control-signal bundle for a frozen video backbone (Level 2, designed-not-built).
- **(C2) A graph-space eval for query-conditioned generation.** Re-perceive the generated answer
  and **diff the scene graph**, yielding the two axes no current benchmark scores together:
  *correctness w.r.t. the query* (intended delta) and *fidelity to the scene* (collateral delta).

Relation to literature: the **bridge** papers (DyST-XL, LVD, VideoDirectorGPT, SceneCraft) already
compile an LLM-invented graph down to *2D layout* control, training-free, **not grounded in a
reconstructed scene**. room4d's differentiators: the graph is **4D and grounded in a real
reconstruction**, and the loop is **closed by a graph-space eval**. (See §6 for a forward-looking
third thread.)

## 3. Current architecture inventory (verified 2026-06-24)

Eleven stages, declared in `room4d/api/stages.py::build_stage_registry` (`StageSpec`:
name / dependencies / artifact_cls / run_fn / load_fn), orchestrated by
`room4d/api/pipeline.py::Room4DPipeline`. Resume + stage-window machinery is solid and stays.

| # | Stage | Status | Note |
|---|---|---|---|
| 1 | `ingest` | **Proven** | decode, keyframe sampling, quality |
| 2 | `sync` | **Stub (no-op)** | `audio_sync.py:11` / `motion_sync.py:16` hardcode `return 0.0, 0.0`; fine for monocular, build out + validate on the rig (F1) |
| 3 | `geometry` | **Proven** | VGGT / VGGT-Long / dataset backends → `GeometryArtifact` |
| 4 | `pose_refine` | **Proven** | BA pose refinement (v0.1.16.0); identity pass-through when disabled |
| 5 | `tracking` | **Proven → Redesign (§4A)** | today: 2D tracks + 3D lift + static/dynamic. Redesign: **motion-only**; identity moves to `segment`/`reid`; reordered *after* perception |
| 6 | `humans` | **Proven\* → Redesign (§4A)** | today: detect / DensePose / SMPL / associate; **`pose2d` can fabricate** — `_run_real` → heuristic keypoints (`pose2d.py:114`) then stamps `backend_used='real'` (`:115`). Redesign: Sapiens (dense) + GVHMR (world-SMPL) replace the Keypoint-RCNN/DensePose/SMPL path (subsumes the F3 de-stub); consumes `segment` masks |
| 7 | `objects` | **Proven → Redesign (§4A)** | today: detect / segment / embed / re-ID. Redesign: SigLIP2/Qwen2.5-VL semantics over `segment` masks; the hand-tuned associator is deleted |
| 8 | `graph` | **Redesign** | `DynamicSceneGraphBuilder` builds a *passive* graph — not queryable |
| 9 | `optimization` | **Stub (no-op) → build out early (F2)** | `JointSpatiotemporalSolver` runs, but every geometric factor returns placeholder `0.0,{}` (`epipolar/depth_consistency/smoothness/graph/human/object_factors.py:13`); the rest return `len(...) * weight`, a count, not a residual. Optimizes nothing today |
| 10 | `scene` | **Re-cut** | `SceneArtifact` (renderable repr) — clean its boundary vs render |
| 11 | `render` | **Replace toy + extend** | `Gaussian4DScene`/`Gaussian4DTrainer` is a **no-train placeholder**; proven `Static/DynamicGaussianTrainer` exist but are only driven by `scripts/run_render_proof.py` |

Net-new (not in the registry today): the **query** stage (`room4d/query/` is all stubs —
`PromptParser.parse` returns a placeholder dict), the **compiler**, and the **query-conditioned
eval**. Also net-new from the perception redesign (§4A): the **`segment`** (SAM 3) and **`reid`**
(DINOv3) stages + their `InstanceArtifact` / `ReidArtifact`.

**Trust tiers (verified 2026-06-24 — `a private note`).** The
"front-half" is not uniformly proven. Read the table's status column as three tiers: **proven**
(geometry / pose_refine / tracking / objects, and the render *trainers*), **stub** (`sync`,
`optimization` — both no-ops today), and **fabrication-capable** (`humans` via `pose2d`). The
queryable graph (§5a) and the round-trip eval (§5e) lean on perception/geometry quality, so the
plan **hardens these stubs early** (Block 1 / F1–F3, gated by G1) rather than building the graph/eval on sand.

## 4. Full stage map & contracts (target)

Front-half contracts are unchanged in substance **except the perception band
(`segment`/`reid`/`objects`/`humans`/`tracking`), re-cut in §4A**; the table records responsibility +
artifact I/O so the redesign has a fixed perimeter. **Separation of concerns is the governing
principle:** each stage owns exactly one responsibility and communicates only through its artifact.

| Stage | Responsibility (one line) | Consumes | Produces |
|---|---|---|---|
| ingest | raw video → frames/keyframes/quality | video paths | `IngestArtifact` |
| sync | per-stream time offsets | ingest | `SyncArtifact` |
| geometry | per-frame poses/intrinsics/depth/pointmaps | ingest, sync | `GeometryArtifact` |
| pose_refine | BA-refined poses (+ focal) | ingest, geometry | `GeometryArtifact` (refined) |
| **segment** *(new, §4A)* | open-vocab detect+segment+**track** → instance masks + persistent IDs (objects + humans) | ingest, pose_refine | **`InstanceArtifact`** *(new)* |
| **reid** *(new, §4A)* | per-tracklet appearance features + cross-occlusion identity merges | ingest, segment | **`ReidArtifact`** *(new)* |
| objects *(re-cut, §4A)* | non-human tracklets → open-vocab category + attributes | ingest, segment, reid | `ObjectArtifact` |
| humans *(re-cut, §4A)* | human tracklets → dense maps (pose/seg/depth) + world-grounded SMPL | ingest, segment, reid, pose_refine | `HumanArtifact` |
| tracking *(motion-only, §4A)* | dense point tracks → 3D lift → static/dynamic (seeded by `segment` fg) | ingest, pose_refine, segment | `TrackArtifact` |
| **graph** | **queryable temporal 4D scene graph (+ dual-use judge contract)** | pose_refine, tracking, humans, objects | `GraphArtifact` (re-cut) |
| optimization | staged joint refinement of states | sync, pose_refine, tracking, humans, objects, graph | `OptimizationArtifact` |
| **scene** | **renderable representation only** (static layer + dynamic SE(3) motion bundle) | pose_refine, tracking, humans, objects, optimization | `SceneArtifact` |
| **render** | **train + render the `SceneArtifact`** (proven trainers) → answer frames | scene, optimization, pose_refine, tracking, humans | `RenderArtifact` |
| **query** *(new)* | NL → neurosymbolic program → executed **query plan** over the graph | graph, prompt | `QueryPlanArtifact` *(new)* |
| **compile** *(new)* | query plan → **render-plan IR** → lowered control for a target | query plan, scene, graph | `RenderPlanArtifact` *(new)* + control bundle |
| **eval** *(new/extended)* | round-trip graph-diff of the answer | render answer, graph (builder reused as judge) | `QueryConditionedEvalArtifact` *(new)* |

The new stages slot **after** `graph` and `scene`: `… → graph → query → compile → render → eval`.
`query`/`compile`/`eval` are added to `STAGE_ORDER` and the registry with their own
dependencies; resume/stage-window behavior is inherited for free.

The **perception band also reorders** (§4A) to
`pose_refine → segment → reid → objects → humans → tracking → graph`: identity (`segment`/`reid`)
now precedes motion (`tracking`), reversing today's `tracking → humans → objects`. Two stages
(`segment`, `reid`) and two artifacts (`InstanceArtifact`, `ReidArtifact`) are net-new; `objects`,
`humans`, `tracking` are re-cut against the new contracts.

## 4A. Front-half perception redesign — verified foundation-model stack + degradable ladders (2026-06-26)

Supersedes the "keep perception as-is" assumption (§0.2/§0.6). The old front-half derived **identity
from CoTracker point-track Jaccard** (`object_assoc.py` / `human_assoc.py`, `track_weight=2.0`
dominant) and associated in 2D with ~6 magic weights — a *motion* cue doing an *identity* job
(a person + carried bag share tracks → false merge; same object across an occlusion gap shares none →
false fragment). The redesign **decouples motion from identity**, rebuilds identity on
mask-propagation + appearance + 3D, and makes every stage **degradable by contract**. Verified by two
adversarial deep-research passes (`a private note`).

### Verified stack (per capability)

| Capability | Primary | License / gating | Replaces |
|---|---|---|---|
| identity backbone (detect+segment+**track** in one model) | **SAM 3** (`facebook/sam3`, 2025-11-19) | custom SAM License, **gated/manual** | Mask R-CNN detect + segmenter + associator-identity + point-track-as-identity |
| cross-occlusion re-ID | **DINOv3-ViT-L** | custom Meta license, **gated/manual** | ResNet18 embedder + associator embed/merge |
| open-vocab semantics + attributes | **SigLIP 2 + Qwen2.5-VL** | open / permissive | COCO-80 category string |
| depth/geometry backstop (2D→3D lift) | **MoGe-2** (metric point map) | **MIT**, ungated | backstops VGGT |
| humans (dense + parametric) | **Sapiens** (pose/seg/depth/normals) + **GVHMR** (world-SMPL) | both non-commercial | Keypoint-RCNN + DensePose/SMPL **stubs** |
| motion (kept, orthogonal) | **CoTracker3** | — | *(unchanged)* |

Architectural template: **4D-LangSplat** (CVPR 2025) — MLLM-per-tracklet temporally-consistent
semantics *over* CLIP-only. SMPL/SMPL-X needs MPI registration. The field moves fast: **re-check
successors (SAM 3.1, Sapiens2, Qwen3-VL, OnlineHMR, GeometryCrafter/ViGeo) right before vendoring.**

### Staged execution (32 GB constraint — load-bearing)

The heavy models cannot co-reside at video resolution (Qwen2.5-VL-7B ~17–20 GB, DINOv3-7B ~14 GB —
**use ViT-L**, SAM 3 ~4–12 GB). Perception is therefore **N sequential sub-stages**, each
**load → infer over the video → release VRAM → persist its artifact** — *not* one monolithic
all-models-resident stage. Mitigations: DINOv3-L not 7B; fp16/bf16; stage the VLM alone (or the 3B
variant). This *reinforces* the decoupling and maps cleanly onto the existing `StageSpec`/resume
machinery — each stage is independently resumable and skippable.

### Reordered stage band

`pose_refine → segment → reid → objects → humans → tracking → graph`. **SAM 3 is the shared mask
source** (load once, prompt with object + human concepts, get all instance masks + persistent IDs,
release) → `objects`/`humans`/`reid` are *enrichment passes* over those masks; **`tracking` drops to
pure motion** (CoTracker), reordered after `segment` so its dynamic split is seeded by real
foreground masks (this is the original F0b fg-gap fix, now clean). Net-new artifacts:
**`InstanceArtifact`** (per-frame masks + track IDs) and **`ReidArtifact`** (tracklet embeddings +
cross-gap merges).

### Degradability — the contract is the interface, the ladder fills it

Each stage = a fixed **artifact contract** + a **fallback ladder** (best → degraded →
graceful-empty). At stage start it probes the ladder top-down (importable? weights present + access
granted? smoke-inference passes?), runs the first available rung, and stamps `backend_used` + a
**capability manifest** (which rung ran, what fidelity the artifact carries) into the metadata — so
degradation is *observable*, not silent. This generalizes the existing `backend_used` idiom
(detector `real→heuristic`, embedder `real→histogram`, tracker `cotracker→fallback`).

| Stage | Fallback ladder (best → … → floor) |
|---|---|
| `segment` | SAM 3 → Grounded-SAM-2 (Apache, **ungated**) → SAM 2 + Mask R-CNN → Mask R-CNN + IoU assoc *(today)* → graceful-empty |
| `reid` | DINOv3-L → DINOv2 (Apache) → ResNet18/CLIP *(today)* → color histogram *(today)* → skip (SAM 3 in-shot IDs only) |
| `objects` (semantics) | Qwen2.5-VL + SigLIP 2 → SigLIP 2/OpenCLIP zero-shot → CLIP → detector COCO-80 label → `"unknown"` |
| `humans` | *dense:* Sapiens → Keypoint-RCNN → bbox-only · *SMPL:* GVHMR (world) → 4D-Humans (camera) → Multi-HMR → skip |
| `tracking` (motion) | CoTracker3 → TAPIR → RAFT flow → all-static · *(dynamic seed from `segment` masks, else motion-threshold)* |
| depth backstop | VGGT → MoGe-2 (MIT) → Metric3D v2 (BSD)/UniDepth (NC) → triangulate from tracks+poses → ground-plane |

### Back-half contract + resolution guarantee

The load-bearing split: **`scene`/`render` need masks + static/dynamic partition + 3D** (from
`segment` + `tracking`); **`graph` needs entity nodes** {id, kind, 3D trajectory, lifespan} **(+
degradable semantics)**. Crucially, render's dynamic-Gaussian seed depends on `segment` masks +
`tracking` labels — **NOT** on the `objects`/`humans`/`reid` enrichment — so those degrade freely
without touching render; they only lower *graph* richness.

Three invariants make every ladder combination safe: **(1)** per-stage top-down availability probe;
**(2) every rung — including graceful-empty — emits a schema-valid artifact** (the F0 graceful-empty
render/scene/eval path already exercises this floor); **(3) empty-in → empty-out, never crash**
(`reid`/`objects`/`humans` consume `InstanceArtifact`; empty → empty enrichment → static-only
graph/render, pipeline still completes).

**Guarantee:** because the load-bearing ladders (`segment`, `tracking`) bottom out at *today's
working backends*, and every rung emits a valid artifact the already-graceful back-half consumes,
**no ladder combination fails to produce the full graph/scene/render artifact set.** Degradation only
lowers *fidelity* (open-vocab → closed-vocab → unknown; world-SMPL → camera-SMPL → none; dynamic →
all-static), never *completeness*. Worked traces:
- **SAM 3 not yet granted** → `segment`=Grounded-SAM-2 (Apache); `reid`=DINOv3 recovers the weaker
  IoU-IDs; `objects`/`humans`/`tracking` unaffected → full back-half, only open-vocab identity
  slightly weaker.
- **SAM 3 ok, Sapiens absent** → human masks still come from `segment`; `humans` dense=Keypoint-RCNN,
  SMPL=4D-Humans/skip → **render identical** (masks never depended on Sapiens), graph human node
  marginally less rich.
- **Floor (no foundation models)** → `segment`=Mask R-CNN+IoU *(today's path)*, `reid`=histogram,
  `objects`=COCO-80, `humans`=RCNN/no-SMPL → render works, graph closed-vocab. **The ladder floor is
  literally today's pipeline → the redesign is a strict superset, never worse.**

### Setup

Gating means **register-to-download only** — models run **fully offline locally, no data egress**.
**Request `facebook/sam3` + `facebook/dinov3` access early** (manual approval, reported days-to-weeks
wait); Sapiens / GVHMR / MoGe-2 are ungated. All vendored via `setup_room4d.sh`
(`a private note`, `a private note`); the gated weights need
a pre-authorized HF token — the one step that isn't fully scriptable.

## 5. Back-half deep design

### 5a. `graph` — from passive builder to queryable temporal substrate (+ judge)

**Today:** `DynamicSceneGraphBuilder.build` emits `GraphNode`/`GraphEdge`
(`node_id, node_type, label, confidence, attrs`) for cameras and sparse tracks — a flat snapshot.
No temporal trajectories, no typed relations over time, no grounding handle, no query API.

**Target contract (`GraphArtifact` re-cut):**
- **Entity nodes** with stable IDs, a **continuous presence interval / trajectory over time**
  (visibility ≠ existence — a node persists across occlusion; see §6), a **world-space pose/position
  trajectory**, and an **open-vocab semantic handle** (embedding + caption) for grounding. Nodes
  link back to `scene.dynamic_entities` / `render_contract` so a node is *renderable* (this is the
  grounding bridge the lit-review flags as unsolved — mAP ~6.08 — so it is a first-class risk, §8).
- **Typed temporal relations** (`edge_type` with a **time interval** of validity): spatial
  (near/on/left-of), interaction (holding/riding), and containment. Grounded NS-VideoQA-style
  *(subj, rel, obj, t_start, t_end)*.
- **Query API:** select-by-attribute, temporal filter, relation traversal — the executable surface
  the `query` stage compiles to (§5b).
- **Dual-use judge requirement:** the builder must be **deterministic and re-runnable on a generated
  video** to produce a comparable `GraphArtifact`. Its node-correspondence/association must be stable
  enough that two runs on near-identical input yield diffable graphs (this defines the eval's noise
  floor, §5e).

### 5b. `query` — neurosymbolic NL → program → query plan

**Today:** `room4d/query/` is stubs (`parser.py`, `node_retrieval.py`, `event_retrieval.py`,
`trajectory_planner.py`, ~300 bytes each); not wired into `run()`.

**Target:** a **VisProg / NS-VideoQA-style** parse-to-program-to-execute stage:
1. **Parse** NL → a typed program in a small DSL over the graph query API
   (`select`, `filter_time`, `relate`, `count`, plus *edit* ops `remove` / `move` / `recolor` and
   *view* ops `orbit` / `follow` / `flythrough`).
2. **Execute** the program against the `GraphArtifact` → resolve concrete entity IDs + a time window
   + (for edit queries) the **intended symbolic graph delta**.
3. **Emit** a `QueryPlanArtifact`: resolved entities, time window, the requested operation(s), and —
   crucially for §5e — the **expected post-edit graph** `G'_expected` (the symbolic delta applied to
   the current graph).

LLM-as-program is acceptable for the parser (training-free, VisProg-style); the *executor* is
deterministic over the graph. This keeps the symbolic plan **external** to any generator (matching
the bridge papers) and interpretable.

### 5c. `compile` — the render-plan IR (the contribution)

**Render-plan IR** = a backend-agnostic intermediate the query plan lowers into, expressive enough
for *both* targets (designing it for both now is the deliberate tax called out in §8):

```
RenderPlan {
  time_window:     (t_start, t_end)
  entity_selection: [entity_id, …]               # which graph entities are in-frame
  entity_edits:    [{id, op: remove|move|recolor|isolate, params}]
  camera:          CameraTrajectory               # poses+intrinsics+timestamps (orbit/follow/path)
  appearance:      { background: keep|inpaint, … }
  output:          { kind: image|video, fps, resolution }
}
```

**Lowering A — own 3DGS renderer (Level 1, v1):** IR → a *filtered* `SceneArtifact`
(`render_contract.static_layer` + selected `dynamic_layers`, with `entity_edits` applied as
Gaussian masking/transforms — e.g. `remove` = drop that entity's Gaussians) + the `camera`
trajectory → the proven trainer renders the path. Cheap, deterministic, **faithful by construction**
(a `remove` leaves a real hole — no plausible fill; that gap is precisely Level 2's reason to exist).

**Lowering B — world-model control bundle (Level 2, designed-not-built):** IR → rasterize the
selected/edited 4D geometry into the frozen backbone's expected conditioning — point-cloud-cache
renders (GEN3C), 4D control maps / per-object Gaussian-trajectory maps (VerseCrafter's GeoAdapter),
persistent point-cloud state (LiveWorld) — plus an assembled text prompt. Realistic on a single
5090 only as **training-free conditioning or a light LoRA adapter on a small frozen backbone**
(Wan-1.3B / CogVideoX-2B); a backbone-from-scratch is out of reach and out of scope. The
control-bundle schema is specified now (so Lowering A's IR doesn't under-specify it) but the adapter
is not implemented.

**Honest framing (eng-review): it is not one "unified IR" — it is a shared front-end + per-target
lowering.** The ops carry *per-target-valid* semantics: `remove` + `background: inpaint` is a no-op
on Lowering A (the own renderer leaves a real hole) and load-bearing on B; `recolor` is an exact SH
edit on A and a soft, may-be-ignored control hint on B. So the contribution is the **shared
front-end** (query→graph→selection+trajectory) + the **graph-space eval**, *not* a magically
target-agnostic IR. Write an explicit **op-support matrix** now — for each op, exactly what each
target honors / no-ops / errors — as the cheapest guard against the IR fracturing silently when P8
forces ops A can't honor.

### 5d. `render` — wire the proven trainers, delete the toy

**Delete/replace** the no-train placeholder: `render/gaussian4d/scene_model.py::Gaussian4DScene`
and `render/gaussian4d/trainer.py::Gaussian4DTrainer` (the `RenderBackend` the pipeline's
`fit_renderer` calls today — it seeds a cloud, rasterizes, renders paths, **never optimizes**).

**Wire in** the proven trainers (`StaticGaussianTrainer(field, TrainConfig, densify_cfg, …)` /
`DynamicGaussianTrainer(DynamicScene, DynamicTrainConfig, …)` + `GsplatRasterizer` +
`DensificationController`) — currently only driven by `run_render_proof.py`. The render stage's sole
job becomes **train the `SceneArtifact`, then render the compiled camera path** → `RenderArtifact`.

**Design forks to resolve in F0 (carried from the superseded render design):**
- `SceneArtifact → trainer init`: the proof path builds its own init from a provider; the pipeline
  assembles a `SceneArtifact`. Define a `SceneArtifact → GaussianField/DynamicScene` seed adapter.
- static vs dynamic trainer selection (presence of `dynamic_layers`).
- training-config home (config schema for `TrainConfig`/`DynamicTrainConfig` in pipeline cfg).
- resume / stage-window interaction with a long training loop.
- replace the weak `BenchmarkRunner` render metric (the misnamed train-view `heldout_psnr_*`) — see
  §5e; the real measuring instrument is the masked-NVS machinery
  (`eval/metrics/image.py`, `eval/benchmarks/covisibility_masked_novel_view_synthesis/`).

**F0a — IMPLEMENTED (2026-06-25, `feature/f0-renderable-core`, v0.1.17.0). Static spine shipped:**
- The seed comes from **geometry via the scene assembler** (`seed_arrays_from_geometry`), *not* a
  "`SceneArtifact → init` adapter" — the lightweight artifact can't carry depth/GT, so the assembler
  bakes a **self-contained renderable `SceneArtifact`** (dense seed + per-frame train views with the
  GT frame persisted *in* the artifact + an orbit trajectory). The render stage consumes ONLY it.
- New `render/gaussian4d/render_backend.py::GaussianRenderBackend`: seed → `GaussianField` →
  `StaticGaussianTrainer.fit(train_views)` → trajectory render + trained checkpoint; the toy
  (`scene_model.py` + `trainer.py`) is deleted. Degenerate scenes (no depth/GT) emit an empty render.
- Masked-NVS metric (`render/eval/masked_nvs_benchmark.py`) scores the trained checkpoint vs a
  dataset's held-out val + covisibility, reusing the proof scorer (dataset-coupled; `{}` otherwise).
- **Validated:** in-pipeline masked mPSNR **18.54** on DyCheck paper-windmill (static) reproduces the
  standalone proof harness's **18.51**. Static/dynamic *selection*, the training-config home, and the
  `DynamicGaussianTrainer` wiring are **F0b**; resume is stage-level (artifact-exists) for now.

### 5e. `eval` — round-trip graph-diff (cycle consistency)

**Mechanism:**
```
answer video ──► [reuse graph builder as judge] ──► G_observed
query plan ─────────────────────────────────────► G'_expected   (§5b: symbolic delta applied to G)
                                  diff(G_observed, G'_expected)
```

**Two reported axes (the lit-review's open metric):**
- **Correctness (intended delta):** did the requested edit happen? (e.g. the removed entity's node
  is absent in `G_observed`). Precision/recall over the intended delta set.
- **Fidelity (collateral delta):** graph-edit distance restricted to the *un-edited* subgraph — did
  anything else move/appear/vanish that shouldn't have?

**Query taxonomy → invariant:**
- *Edit/counterfactual* queries → diff against the **expected symbolic delta**.
- *Pure-viewpoint* queries → **graph invariance**: re-reconstruct from the generated views and
  recover the *same* 4D graph up to the camera transform; geometric inconsistency surfaces as
  round-trip failure.

**Two validity risks (eng-review outside voice) — design against both:**
- **Self-judging circularity.** Builder and judge are the *same* stack, so the eval is blind to the
  class of errors that stack makes *consistently* (can't perceive "holding" → can't see it dropped →
  scores the drop as success). This is **systematic shared bias**, not the random noise floor — drift
  experiments do not surface it. Mitigation: an **independent judge** (a different detector/VLM, or
  human spot-checks) on a calibration subset, to bound the shared-bias term.
- **Level-1 is near-tautological for *generation*.** Lowering A filters + re-renders a reconstruction
  (faithful by construction), so an edit round-trip mostly checks "did Gaussian masking remove what I
  masked." It validates the **plumbing** (cross-input correspondence, diffing, novel-view-render
  robustness) — real but modest — and the metric's **discriminative power for generation is only
  tested at Level 2 (P8)**. Don't read a passing Level-1 loop as "the eval works."

**Measurement floor — the RIGHT one.** The eval's real error lives in the **cross-input node
correspondence** between `G'_expected` (predicted post-edit IDs) and `G_observed` (fresh IDs
re-perceived from *different pixels* — the rendered answer). Re-run determinism on identical input
does **not** bound this. **P4 (§7) measures the cross-input correspondence floor**; only deltas above
it count.

**Calibration ladder:** validate the loop on the **own renderer** first (known ground-truth deltas,
*plumbing only* — see above), then point the same eval **+ independent judge** at a Level-2 model.
(The pure-viewpoint "graph invariance" axis hides a second open problem — graph-correspondence
up-to-Sim(3) between two reconstructions, cf. step-12's Sim(3)/orientation pain — so scope it
carefully, §8.)

**Artifact:** `QueryConditionedEvalArtifact` — per-query correctness/fidelity scores, the diff, and
the noise-floor calibration. New benchmark home: `eval/benchmarks/query_conditioned_generation/`.

## 6. Forward-looking research thread — constrained graph-consistency objective

*(Future direction. v1 uses only the eval half of this; recorded here so the architecture leaves
room for it.)*

Object permanence is a **continuity + conservation** constraint, **not a symmetry**: "nodes can't
pop in and out; they deform, leave view and return." The obstruction: graphs of **varying node
count** form a **stratified space** (one stratum per cardinality), so no continuous group acts
across appearance/disappearance. Resolve by **lifting discreteness into continuous dimensions**:

1. **Presence/opacity field + fixed slot pool** — existence becomes a continuous α∈[0,1]; leaving
   view is α→0 with identity preserved, not a topological jump. (Caps object count.)
2. **Continuity equation** — `∂ρ/∂t + ∇·J = 0`: entity presence is conserved in the interior;
   image/occluder boundaries are the only legal sources/sinks. The physical content of "no popping."
3. **Birkhoff soft-permutation identity flow** — the one real symmetry (node relabeling, S_n)
   relaxed onto the Birkhoff polytope (Sinkhorn) so temporal identity-association is a *continuous*
   trajectory, not abrupt re-assignment.
4. **Temporal smoothness + (exotic) topological-persistence priors** — Lipschitz attribute
   trajectories; penalize rapid component births/deaths (perception flicker).

**Composition (the room4d-shaped candidate):** run `generate → re-perceive → recover the graph
trajectory`, then require it to be **(i) correct** w.r.t. the intended edit (cycle-consistency, §5e)
**and (ii) continuous with conserved identities** (no popping). (ii) is **query-independent,
self-supervised**, applies to *any* generated 4D video, and directly punishes the most visible
world-model failure (flickering / identity swaps). room4d is well-positioned because it owns the
perception→graph **judge** that computes the loss. Path: **eval in v1 → soft training loss for a
Level-2 adapter later** (soft constraint, analogous to physics-informed losses vs. baked-in
equivariance).

**Prior art to stand on:** C-SWM (object-factored GNN latent world models), SAVi/STEVE (slot object
permanence), Gumbel-Sinkhorn (continuous permutation), differentiable persistence layers,
Hamiltonian/Lagrangian nets (baked-in conservation). **Confidence:** the *ingredients* exist; the
*composition* (continuity/permanence-constrained, cycle-consistent graph regularizer on a 4D
generative model grounded in real reconstruction) is *likely* unassembled — **a targeted lit scan
is required before claiming novelty.**

**Eng-review caveats (don't oversell this thread):** (1) the **defensible novelty is the round-trip
as the loss *carrier*** (generate→re-perceive→diff), **not** the continuity-equation math — the
conservation law is false in general (objects genuinely appear/vanish) and can't separate "appeared
at an occlusion boundary (legal)" from "popped in the interior (illegal)" without already solving
occlusion, so in practice it degenerates toward a smarter temporal-persistence loss that
SAVi/STEVE/C-SWM/Gumbel-Sinkhorn substantially cover. (2) A **non-differentiability wall**: the
re-perception judge (detectors/SMPL/CLIP) is not differentiable, so "eval → training loss" needs a
differentiable judge or score-distillation/RL plumbing — a wall, not a checkpoint. Treat P9 as a
real research bet, not a sure thing.

## 7. Build plan (foundation-complete gate → back-half)

room4d is a no-ship/no-publish learning project (§0.7–0.8). **Block 1 builds the entire front-half to
a real, no-stub state and runs it end-to-end on multi-stream rig data — that is GATE G1, the
precondition for any back-half work.** Block 2 (queryable graph / query / compiler / eval) starts
only after G1. The stub stages (`sync`, `optimization`, `pose2d`), the **perception redesign (§4A)**,
and multi-stream integration are explicit Block-1 scope. Each phase is a feature branch off
`develop`, PR back to `develop`, version bump per the repo convention. Horizon phases (P8–P9) are
intentionally coarse — learning reshapes them.

```
BLOCK 1 — FOUNDATION COMPLETE (no stubs; multi-stream; GATE G1 for Block 2)
  F0  Renderable core (anchor)      Delete the render toy (Gaussian4DScene / Gaussian4DTrainer); wire
                                    proven Static/DynamicGaussianTrainer into the render stage
                                    (SceneArtifact→seed adapter, static/dynamic select, training-
                                    config home); clean scene↔render; in-pipeline masked-NVS.
                                    Monocular proven spine first. Depends on: none (trainers exist).

  PERCEPTION REDESIGN (§4A) — floor-first structure, THEN climb the ladder. Ungated tools first
  (Grounded-SAM-2 / DINOv2 / SigLIP2 / Qwen-VL / Sapiens / GVHMR / MoGe-2); SAM 3 + DINOv3 swap in as
  top rungs when gated access lands. Every rung sits behind the availability probe so the pipeline
  never breaks. Branch: feature/perception-redesign.
  FP0  Access + ladder scaffold     Request SAM3 + DINOv3 HF access (manual approval, days-to-weeks
                                    lead). Build the backend-ladder base (top-down probe →
                                    backend_used + capability manifest) + new artifact schemas
                                    (InstanceArtifact, ReidArtifact, re-cut Object/Human/Track
                                    contracts). No new models. Depends on: none.
  FP1  Structural reorder           Add segment + reid stages; reorder STAGE_ORDER to
       (floor rungs only)           pose_refine→segment→reid→objects→humans→tracking→graph; rewire
                                    registry deps + run(). Floor rungs = today's backends (Mask R-CNN
                                    segment, ResNet18 reid, COCO-80 semantics, today's humans,
                                    CoTracker motion-only). GOAL: pipeline runs end-to-end in the new
                                    shape, all back-half artifacts produced, full suite green —
                                    proves the degradability invariants + blast radius, zero
                                    download. Depends on: FP0.
  F0b  Dynamic render               Resume the dynamic-render path (ArtifactGeometryProvider + packed
       (on FP1 floor-rung masks)    AllPairsTracks already built on feature/f0b-dynamic-render) using
                                    FP1 FLOOR-RUNG fg masks — prove the full perception→…→dynamic-
                                    render pipeline BEFORE any new tools. Target ~20.7 in-pipeline
                                    masked-NVS on paper-windmill. Depends on: FP1, F0.
  FP2  Climb segment                SAM 3 top rung (Grounded-SAM-2, Apache/ungated, is the rung until
                                    access granted) behind the probe. Real propagation masks + IDs;
                                    re-validate F0b on them. Depends on: FP1.
  FP3  Climb reid                   DINOv3-L top rung (DINOv2, Apache, until granted). Cross-occlusion
                                    identity stitching vs the ResNet18 floor. Depends on: FP1.
  FP4  Climb semantics              SigLIP 2 + Qwen2.5-VL → per-tracklet open-vocab category +
                                    attributes. The front-half half of the Grounding spike (below).
                                    Depends on: FP1.
  FP5  Climb humans                 Sapiens (dense) + GVHMR (world-grounded SMPL). SUBSUMES F3.
                                    Depends on: FP1.
  FP6  Depth backstop               MoGe-2 (MIT, metric point map) into the 2D→3D lift. Depends on: FP1.

  F1  Real temporal sync            Replace the 0.0 stub (audio_sync.py:11 / motion_sync.py:16) with a
                                    real sync stage; VALIDATE on the fixed rig — must recover the known
                                    ~0 offset. A wrong answer on a synced rig = a broken impl caught
                                    before unsynced streams ever depend on it.
  F2  Real joint-opt fusion         = calibration-arc M2/M3 (a private note; eng-reviewed
                                    design + S1 spikes cleared the robustness + scale gates). Real
                                    factors / multi-stream BA over poses+states; metric scale
                                    (person-anchored; intrinsics a named dep). Validate vs GT
                                    extrinsics on NVIDIA 12-cam / CMU Panoptic. Replaces the verified
                                    no-op factors (return 0.0,{}).
  F3  pose2d de-stub                SUBSUMED by FP5 — Sapiens (dense) + GVHMR replace the
                                    Keypoint-RCNN/pose2d path wholesale; the heuristic-stamped-'real'
                                    fallback (pose2d.py:114) is deleted, not patched.
  F4  Multi-stream integration      Whatever G1 surfaces beyond the stubs: geometry fusion across
                                    streams, scene assembly from multiple cams, render from a fused
                                    multi-cam reconstruction. (The end-to-end run is what finds these.)
  ── GATE G1: the full pipeline (ingest→sync→geometry→pose_refine→segment→reid→objects→humans→
              tracking→optimization→scene→render) RUNS END-TO-END on multi-stream rig data, every
              stage real, validated against rig GT (extrinsics + recovered sync offset + masked-NVS). ──

  (parallel) Grounding spike        Open-vocab language→reconstructed-entity grounding (4D-LangSplat-
                                    style time-varying semantic field, or ConceptGraphs-style),
                                    measured on a small known entity set. Integrates at P3; does NOT
                                    gate G1. Gates query-resolution AND eval node-correspondence.
                                    FP4 (SigLIP2/Qwen-VL per-tracklet semantics) is the front-half
                                    half of this — the spike measures grounding; FP4 builds it.

BLOCK 2 — BACK-HALF REDESIGN (starts only after G1)
  P3  Queryable + dual-use graph    Re-cut GraphArtifact (§5a): temporal entity trajectories,
                                    presence, typed time-interval relations, grounding handles;
                                    deterministic/re-runnable builder (judge). Depends on: G1, grounding.
  P4  Correspondence-floor experiment  Measure the CROSS-INPUT node-correspondence error between
                                    G'_expected and a G re-perceived from a *re-rendered* answer (NOT
                                    re-run drift on identical input — that bounds the wrong thing,
                                    §5e). Where the eval's real floor lives. Depends on: P3, F0.
  P5  Query stage                   Neurosymbolic NL→program→query-plan (§5b) → QueryPlanArtifact
                                    (incl. G'_expected). Depends on: P3.
  P6  Compiler + Lowering A         Render-plan IR + op-support matrix (§5c) + Lowering A (own
                                    renderer). Closes the own-renderer query→answer loop.
                                    Depends on: F0, P5.
  P7  Round-trip eval (full)        Graph-diff correctness + fidelity (§5e) + an INDEPENDENT judge on
                                    a calibration subset. On the own renderer this validates the
                                    PLUMBING, not generative fidelity (that is tested at P8).
                                    Depends on: P3, P4, P6.

HORIZON (coarse; learning reshapes)
  P8  Level-2 world-model lowering  Frozen backbone (Wan-1.3B / CogVideoX-2B) + training-free or
                                    light-LoRA control adapter consuming the IR control bundle; point
                                    the (independent-judge) eval at it — first real test of the
                                    metric's discriminative power. SIZE the paired-data + LoRA budget
                                    before committing. Depends on: P6, P7.
  P9  Constrained-loss research     Promote the round-trip to a self-supervised continuity/permanence
                                    training signal (§6) — note the non-differentiable-judge wall.
                                    The genuine frontier bet. Depends on: P7, P8.
```

Milestones: **Foundation = F0–F4 + FP0–FP6 + F0b + G1** (a real, no-stub, multi-stream pipeline
running end-to-end on the modern perception stack). **Level-1 closed loop = P3–P7** on top of it. The
frontier is **P8–P9**. Near-term path: **FP0 → FP1 → F0b** (prove the full pipeline on floor rungs),
then climb FP2–FP6 ungated-first.

## 8. Open questions / risks

These are **learning checkpoints, not ship gates** (§0.7) — each is an experiment that informs
sequencing, not a go/no-go.

- **Perception noise floor (the central eval risk).** The judge is only as good as the
  perception→graph stack; detectors flicker, open-vocab 3D grounding is low in absolute terms
  (mAP ~6.08). **The §4A redesign substantially raises this floor** — SAM 3 propagation gives
  temporally-consistent masks/IDs, DINOv3 backstops re-ID, and Sapiens kills the `pose2d` fabrication
  path. Residual: SAM 3 is **not** identity-solved under heavy occlusion (~75–80% human perf;
  ReMeDI/SAM3-DMS refine it), so the `reid` stage stays load-bearing. Mitigation otherwise unchanged:
  P4 measures no-op round-trip drift first and reports only above-floor deltas; consider a
  temporally-smoothed (tracking) judge.
- **Grounding language → reconstructed entities** is itself unsolved (lit-review). The node↔
  `render_contract` bridge in §5a is the load-bearing assumption; may need a 4D-LangSplat-style
  open-vocab semantic field.
- **Re-ID through long occlusion** — the conservation law (§6) is easy to *write*, hard to *satisfy*.
  §4A makes the mechanism concrete (SAM 3 in-shot IDs + DINOv3 cross-gap features = the `reid`
  stage), but cross-occlusion re-acquisition remains the open risk that both SAM 3 and Grounded-SAM-2
  (IoU-only continuous-ID) leave unsolved.
- **IR-generalizes-to-world-model** — designing the render-plan IR only around Lowering A risks
  under-specifying Lowering B. Mitigation: specify the Lowering B control schema in P4 even though it
  isn't built.
- **Paired-data cost** — any future Level-2 adapter needs (control-signal, video) pairs; producing
  them is itself a reconstruction pipeline (which room4d is) — a hidden cost, not just compute.
- **`optimization` is a verified no-op** (all geometric factors `return 0.0,{}`) and **`sync` is
  hardcoded** — both are now **Block-1 foundation scope** (F1 real sync validated on the rig; F2 real
  multi-stream joint-opt fusion = calibration-arc M2/M3), built and validated *before* G1.
- **Viewpoint-invariance eval hides a Sim(3) alignment problem.** The "graph invariance" axis (§5e)
  needs graph-correspondence up-to-Sim(3) between two independent reconstructions — itself an open
  alignment problem (cf. step-12 Sim(3)/orientation pain). Scope viewpoint queries out of the first
  eval, or budget the alignment cost explicitly.
- **P8 single-5090 feasibility is asserted, not sized.** Light-LoRA on a frozen video backbone *while*
  rasterizing 4D control bundles per step is a different memory budget than inference, and producing
  (control, video) training pairs is a full reconstruction-pipeline run per clip. Size it (N clips ×
  pipeline-hours/clip; LoRA VRAM at the chosen resolution) before committing to P8.

## 9. Repo landing

- **New:** `room4d/compile/` (render-plan IR + lowering), expand `room4d/query/` (parser + executor +
  `QueryPlanArtifact`), `eval/benchmarks/query_conditioned_generation/`, plus `compile`/`query`/`eval`
  entries in `room4d/api/stages.py::STAGE_ORDER` + registry and `Room4DPipeline.run`.
- **New (perception, §4A):** `room4d/perception/segment/` (SAM 3) + `room4d/perception/reid/`
  (DINOv3) stages, `InstanceArtifact` / `ReidArtifact`, a shared **backend-ladder base** (top-down
  availability probe → `backend_used` + capability manifest), and `segment`/`reid` STAGE_ORDER +
  registry entries with the reordered perception band.
- **Re-cut:** `room4d/graph/` (`GraphArtifact`, `DynamicSceneGraphBuilder`, node/edge types in
  `room4d/common/types.py`); `room4d/scene/` (boundary cleanup); **`room4d/perception/objects/` +
  `room4d/perception/humans/`** (delete the hand-tuned associators; SigLIP2/Qwen-VL semantics +
  Sapiens/GVHMR over `segment` masks); **`room4d/tracking/`** (motion-only, reordered, dynamic split
  seeded by `segment` fg).
- **Replace/delete:** `room4d/render/gaussian4d/scene_model.py::Gaussian4DScene`,
  `room4d/render/gaussian4d/trainer.py::Gaussian4DTrainer` (the toy); the render stage drives the
  proven `gaussian_trainer.py` / `dynamic_trainer.py`.
- **Reuse unchanged:** ingest / sync / geometry / pose_refine (perception *consumes* pose_refine,
  doesn't change it); the `StageSpec` registry + resume + stage-window; the masked-NVS eval machinery.

---

## Appendix — relation to existing artifacts

- **Lit-review:** `docs/research/2026-06-21-neurosymbolic-queryable-4d-scene-graphs-and-query-conditioned-generation.md`
  (grounds §2 and §6; the eval-paper IDs there feed the §6 targeted scan).
- **Superseded render design:** `a private design note`
  (its findings — train-supervision masks, dataset-coupled motion init, world-frame trap — feed F0/F2).
- **Proven baselines:** SoM reference benchmark (masked NVS mPSNR 19.53 on paper-windmill);
  room4d-dynamic beats it (20.7) via `run_render_proof.py`. The render stage must reproduce this
  in-pipeline (F0).

## GSTACK REVIEW REPORT

`/plan-eng-review` — 2026-06-24, branch `develop`, against this design doc.

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| Eng Review | `/plan-eng-review` | Architecture & sequencing (required) | 1 | issues_resolved | 5 architecture findings, all folded in |
| Outside Voice | independent Claude reviewer | 2nd opinion (codex unavailable) | 1 | issues_found | 5 substantive findings on eval validity + IR framing, all folded in |

**Recalibration:** the review's default lens (ship-first / scope-cut / minimize innovation tokens)
was retired — room4d is a no-ship/no-publish learning project (§0.7). Lens became contract
correctness + sequencing for compounding validation + risks-as-experiments.

**Findings (all incorporated):**
1. Stage map over-claimed "proven front-half" — **verified** `optimization` is a no-op (factors
   `return 0.0,{}`), `pose2d` fabricates (`pose2d.py:114`), `sync` hardcoded. → trust tiers added
   (§3), build-out scoped early (Block 1 / F2).
2. Sequence for compounding validation → roadmap re-cut so stub-hardening precedes the graph/eval
   that lean on it (§7).
3. Grounding (open-vocab 3D, mAP ~6.08) is the critical-path unknown → own spike, parallel in Block 1.
4. Eval noise-floor = an early steering experiment, not a gate (P4, §8).
5. Render-plan IR must be specced against real GEN3C/VerseCrafter formats so Level-2 fits (P6, §5c).

**Outside voice (independent reviewer, all folded in):** (1) self-judging circularity = systematic
shared bias the noise floor misses → independent judge on a calibration subset (§5e); (2) Level-1 is
near-tautological for *generation* → reframed as plumbing-validation, discriminative power tested only
at P8 (§5e/§7); (3) P4 measured the wrong floor → re-aimed at cross-input correspondence (§5e/§7);
(4) "unified IR" oversells → demoted to shared front-end + per-target lowering + op-support matrix
(§5c); (5) P9 continuity-equation is the weakest mechanism + non-differentiability wall → reframed,
round-trip is the novelty carrier (§6).

**Decisions by the user:** aim high (full F0–P9 arc, far phases coarse); **foundation-complete gate
(G1)** — entire front-half real + running end-to-end on multi-stream rig data before the back-half;
**real temporal sync validated on the fixed rig**; the real joint optimizer = **calibration-arc
M2/M3** built early as multi-stream foundation (not "learning value" — it lowers the eval noise
floor). This overrides the outside voice's "defer the optimizer," which held only for the monocular
regime the user is deliberately leaving.

**NOT in scope (this design):** building any phase (design only); the targeted lit scan validating
§6 novelty + the eval-paper IDs; choosing the specific Level-2 backbone; the `optimization`/`graph`
internal factor math (F2/P3 detail design).

**Sections:** Architecture — 5 findings (resolved). Code-quality / Performance — N/A (design doc, no
code diff). Test — the round-trip eval *is* the verification strategy; its noise-floor experiment
(P4) is the first validation checkpoint.

**VERDICT:** ENG CLEARED — design accepted; eng-review + independent outside voice both incorporated;
roadmap restructured to a foundation-complete gate (Block 1 → G1 → Block 2). Ready to start **F0**
(renderable core) when you are.
