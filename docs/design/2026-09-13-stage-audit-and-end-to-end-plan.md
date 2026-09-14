# Stage audit — what every stage owes the scene graph, and the end-to-end plan

**Status:** PROPOSED
**Date:** 2026-09-13
**Evidence run:** `../runs/davis_lab-coat_full` (47 frames, all 15 stages, VGGT-Long top rung)
**Supersedes in part:** `docs/design/2026-09-12-back-half-ladders-and-graph-contract.md` (see §5)
**Grounding survey:** `docs/research/2026-09-12-third-party-components-for-the-back-half.md`

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
would be*, not as work items. Work items below target the **golden path** — the top rung of every
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

### 2.2 Per-person depth already exists, and the graph discards it

`humans/dense/` holds **94 files, 89 MB**: Sapiens per-instance `depth` (usable on **94/94**),
`normals`, and `seg`. `HumanState` also carries `keypoints_2d` (94/94).

`graph/builders.py` reduces all of it to `mean_visible_keypoints`, `has_densepose` and `has_smpl` —
two booleans and a float. It never opens `dense/`.

Meanwhile the two human nodes on this clip carry **5 and 0** 3D positions across **47 masked
frames**. The position evidence the graph is starved of is already on disk, per person, per frame.

**This is the single highest-value repair in the audit.** Sapiens depth plus the (good) camera poses
yields a dense 3D point set per person per frame — position *and* extent — with no dependence on
whether a point track happened to land on them.

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

---

## 3. The golden path

The end-to-end vision, and what each step needs to stop being blocked:

| Step | Blocked on | From |
|---|---|---|
| **Interpret** — entities with identity, presence, position, extent | human positions from Sapiens depth; mask-seeded tracking; semantics reaching the graph | §2.2, §2.4, §2.5 |
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
- **§2.2's cross-position extents ladder is withdrawn.** Human extents come from Sapiens depth at
  `humans` position (§2.2 of this document); the backward-edge problem it tried to solve does not
  arise.
- **§3 (the contract) stands**, with three corrections: `position(t)` is *not* shipped as "one point
  per observed frame"; per-frame uncertainty is *not* shipped; `category` is absent on human nodes.
- **§1 (the abstention defect) shipped** as PR #5, with the threshold caveat of §1 above outstanding.

From `2026-09-12-third-party-components-for-the-back-half.md`, corrections are tracked separately;
the adoption verdicts stand.

---

## 6. Work items, in order

1. **Human node positions and extents from Sapiens depth** (§2.2). Highest value; the data exists.
2. **Pass `segment` into `tracking` and seed query points from masks** (§2.4). Fixes late arrivals,
   which density cannot.
3. **Fix the support denominator** (§2.4) so re-id success stops destroying 3D support.
4. **Stop dropping data at the graph boundary** (§2.5) — pass `SemanticsArtifact` to `graph`, carry
   `semantic_topk` and embeddings, label human nodes, record the gravity axis.
5. **Make thresholds scale-relative** (§1) — `static_threshold_world`, and revisit `min_margin`.
6. **Manifests for `tracking`, `objects`, `graph`** plus the second axis (§2.1), and per-axis entries
   for `humans`.
7. **Wire `entity_gaussians` into the pipeline** — it has zero callers today.
8. **Reference comparison** (§4), starting with ConceptGraphs and Feature4X.

Items 1–4 are the ones that decide whether the graph means anything. 8 can run in parallel with all
of them and should start early, because it is the only item that produces an outside opinion.
