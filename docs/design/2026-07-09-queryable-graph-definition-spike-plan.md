# Spike plan (v3) — closing the interpret / query / edit / re-render loop

**Status:** PROPOSED v3 (supersedes v2; v2 reasoning in Appendix A, v1 in Appendix B)
**Inception:** 2026-07-09 · **Revised:** 2026-09-12
**Branch:** `spike/queryable-graph-definition` (rebased onto develop at v0.3.0.0)
**Refines:** §5a–§5e, P3–P7 of
`docs/design/2026-06-24-full-pipeline-architecture-neurosymbolic-queryable-4d.md`.

---

## 0. What changed since v2, and the decision

Three things.

**1. Half of v2's Part 1 shipped.** v2 listed a "per-entity trajectory aggregator" as throwaway
spike code. It is now the `graph` stage on develop (v0.3.0.0, `a89de66`), built properly: point
tracks are assigned to entities by mask membership, each entity node carries its supporting track
ids, how many frames each was inside, the tracking stage's static/moving split, and a **per-frame
world position** (the median of its supporting tracks' lifted points). Tracks inside no entity
are the background's evidence; tracks inside one but not often enough are named as ambiguous
rather than swept in. Contract version 2, resume-covered, tested. On spin that is 13 nodes where
there were 1593.

**2. The gap was scoped too narrowly.** v2 aimed only at *defining* the graph, deferring
everything else to a Part 2 that also never covered editing. But the README promises a scene you
can "**interpret, query, and semantically edit**", and the showcase page promises the whole loop —
query in plain language, render the answer, check it by perceiving it again. Measured against
that, the missing capability is the entire back half (§2), and the load-bearing hole is not the
node model at all: **nothing links a graph entity to the Gaussians that render it** (§4).

**3. Therefore: loop first, but define the graph properly.** Not a thin slice that leaves the
model wrong — the model is specified as a contract *before* the loop consumes it, and audited at
breadth afterwards. The loop ordering exists because the grounding bridge is the assumption most
likely to invalidate the design, and it is cheapest to test first.

---

## 1. The question and the terminal deliverables

**Question.** *Can a scene the pipeline reconstructed be interpreted as a graph of real entities
and relations, queried in language, edited through that graph, and re-rendered — and does the
graph mean what a person would say it means?*

**Two deliverables, both required:**

1. **`docs/design/2026-09-12-queryable-scene-graph-definition.md`** — the grounded contract: node
   model, attribute schema, relation vocabulary, presence semantics, the **renderable handle**,
   and the query surface; with a per-axis *grounded / partially grounded / ungrounded* verdict and
   the evidence that earns it, plus the honest boundary of a single monocular stream.
2. **A working thin loop on one clip** — a query resolves an entity, an edit is applied, the
   edited scene renders, the render is re-perceived, and the two graphs are diffed. Thin at every
   stage, but end to end and on the real model.

---

## 2. The gap, per capability (verified 2026-09-12)

| Capability | State | Evidence |
|---|---|---|
| **Interpret** | partial | Entity nodes with per-frame position and motion exist (`reifyscene/graph/builders.py`). **No typed relations**, no presence-vs-occlusion semantics, no caption/embedding handle, no cross-view association. Audited on one clip, never at breadth. |
| **Query** | absent | `reifyscene/query/` is four stubs returning `{}` (`parser.py`, `node_retrieval.py`, `event_retrieval.py`, `trajectory_planner.py`); not wired into `run()`. |
| **Edit** | absent | §5b's `remove` / `move` / `recolor` and the §5c render-plan IR do not exist as code. |
| **Re-render** | trains, cannot edit | The render stage wires the real trainers (`StaticGaussianTrainer` / `DynamicGaussianTrainer` — §5d's "delete the toy" is done) and renders a trajectory, but has no way to render a scene with an entity removed or moved. |
| **Verify by re-perceiving** | absent | §5e's round-trip graph diff. |
| **The bridge between them** | **absent** | §5a requires nodes to link back to `scene.dynamic_entities` / `render_contract`. `SceneArtifact` has neither field and nothing writes them; `render/camera_paths.py:120` already reads `scene_artifact.dynamic_entities` and would raise `AttributeError` if `follow_subject` were called. |

---

## 3. The graph model (what is built, what this spike adds)

**Node = one persistent physical entity.** Identity = the reid-canonical tracklet. One node per
entity — never per-frame, never a point-track. *(Shipped.)*

```
EntityNode( id = canonical entity )
  identity   : category, category_topk, caption, semantic embedding     # PARTIAL: category only
  presence   : [t_start, t_end] + observed frames + occlusion gaps      # ADD: gaps are not flagged
  position(t): world-space trajectory, one 3D point per observed frame  # SHIPPED
  motion     : velocity(t), speed, path length, is_moving               # PARTIAL: static/moving
  support    : member point-track ids + per-frame uncertainty           # SHIPPED
  renderable : the Gaussians this entity owns                           # ADD — §4, the crux
```

**Edges = physical relations between entity nodes, each with a validity interval**, recomputed per
frame from trajectories + masks so edges appear and disappear over time. Rung-1 vocabulary:
`near`, `above`/`below`, `left-of`/`right-of`, `in-front-of`/`behind`, `on-top-of`,
`supported-by`, `contains`, `overlaps`. *(None built. The pre-flight in
`../spike_graph_definition/` produced two `near` edges on one clip and was never audited.)*

**Presence = existence, handled honestly.** Presence spans first-to-last observation; interior
gaps are flagged **occlusion candidates**, not the entity ceasing to exist. Object permanence
across long occlusions stays out of scope.

---

## 4. The grounding bridge — step 1, and the crux

Editing needs to know **which Gaussians are this chair**. Nothing computes that today, and the
architecture doc flags it as a first-class risk (the grounding problem; the literature's mAP ~6.08).

**The rule already exists.** The `graph` stage assigns a point track to an entity by projecting it
into each frame and testing it against that entity's mask, assigning by majority over the frames
the two share. **The same rule generalizes from point tracks to Gaussians**: project the scene's
Gaussians into each training view, test against the entity masks, assign by majority. Same
membership test, same thresholds, a different population — and it is already implemented and
tested (`reifyscene/graph/track_support.py`, `tests/test_track_support.py`).

**What step 1 builds:** graph entity id → Gaussian index set, and a demonstration that dropping
that set leaves a hole where the entity was.

**Correction found while building it (2026-09-12): the map belongs with the TRAINED Gaussians,
not the seed.** §5a names `scene.dynamic_entities` / `render_contract`, but `SceneArtifact` holds
the seed cloud, and training densifies and prunes it — a seed index means nothing at render time.
The cloud that renders lives in the render stage's `trained_dynamic.npz` (on lab-coat: 46,687
foreground + 269,195 background, where the seed was neither). So the map is computed over the
trained cloud and belongs in the render stage's artifact. The render stage already depends on
`pose_refine`, `tracking` and `humans`; it would need `objects` and `graph` added to build the map
itself. The assignment core is landed and tested as `reifyscene/graph/entity_gaussians.py`; where
the artifact stores it is settled when step 3 shows what the loop needs to read.

It also leaves `camera_paths.py:follow_subject` reading `scene_artifact.dynamic_entities`, a field
that has never existed — to be repaired with whatever schema step 3 settles on.

**Why first:** everything downstream assumes it. If entity→Gaussian assignment is too noisy to
support an edit, the loop is not merely thin, it is impossible, and the plan changes before
anything is built on top. **Go/no-go gate.**

---

## 5. The four steps

- **Step 1 — the bridge.** Entity → Gaussians by mask membership; `remove` drops the set; render
  and look. One clip. Gate: is the removed region the entity, and is what remains intact?
- **Step 2 — the definition, as a contract.** Write the node/relation/presence/renderable contract
  *before* the loop consumes it (deliverable 1, first draft). Build the relation computer against
  it.
- **Step 3 — the thin loop.** `select` an entity by category → apply `remove`, then `move` → render
  → re-perceive the render → diff the two graphs. One query family, two edit ops, everything thin.
- **Step 4 — the audit, at breadth.** DAVIS with GT masks, a written rubric, several sequences.
  Findings revise the step-2 contract; the revised contract is the terminal deliverable.

Each step is human-inspected before the next. Steps 2 and 4 are the "define it properly" half;
steps 1 and 3 are the loop.

---

## 5a. Step 1 outcome — 2026-09-28: the gate fails, 1 of 10

Run on ten DAVIS sequences (§6's "no conclusion from one clip", finally honoured). Reviewed by
eye at `claude.ai/artifact/P2JPufjkoVR6LgEJm7vxMA`.

**Verdict: `blackswan` alone passes** — removed region is the entity (yes), rest intact (minor
artefacts). The other nine fail, in the reviewer's words: *"either have very bad segmentation or
barely change"*.

**The bridge code is not what failed.** `blackswan` removes 99.5% of its entity and leaves the
scene standing, from the same code path every other scene used. What fails is what the code is
given.

**Two failure shapes, and the first was mis-diagnosed in the first write-up.** It was recorded
as a "halo" — the entity's Gaussian footprints reaching past its silhouette. Looking at the
renders, that is not what happens. On `lab-coat` the three women stand shoulder to shoulder; the
target's 2D mask is a clean silhouette, and removing its Gaussians **tears the heads off the two
people beside her** while leaving her own body standing. That is not a footprint artefact. The
trained Gaussians are **not separable per entity** where entities are adjacent in depth, which on
this footage is most of them. The measured "92% intact" hid it, because the damage is small in
pixels and catastrophic in meaning.

The second shape is `india` and `miami-surf`: removing up to 55% of the foreground changes no
pixel, because the foreground never became substantive (median Gaussian scale 4–16x below
`blackswan`'s) and the background carries the image.

**What this means for the remaining steps.** Step 2's relations and step 3's loop would be built
over an entity set that cannot be separated in the representation it indexes. Relations between
entangled entities, and edits that damage their neighbours, are not thin versions of the goal —
they are the goal failing quietly. **The spike's remaining steps are blocked on entity
separability**, which is upstream of everything the graph does.

`blackswan` is the shape of success and says what separability needs: one dominant object, well
apart from anything else, claiming 72.6% of a substantive foreground.

---

## 6. Footage — and the constraint that decides it

**The pipeline reconstructs from camera TRANSLATION.** Rotation gives no parallax, so a
panning or tripod-mounted camera is useless to it however rich the scene is. That single
constraint, not entity count, decides which clips can host the loop — and it disqualifies the
obvious candidates. Measured 2026-09-12, ground truth where the dataset ships cameras,
VGGT-Long's estimate otherwise:

| clip | entities | frames | camera translation | verdict |
|---|---|---|---|---|
| NVIDIA `Jumping` | 4 people + backpack | 96 | **0.000 (GT)** | fixed rig: unusable |
| NVIDIA `Balloon1` | person + balloon | 192 | **0.000 (GT)** | fixed rig: unusable |
| DyCheck `spin` | 1 person | 426 | 2.87 (GT), baseline/depth **0.579** | reconstructs well; one real node |
| DyCheck `paper-windmill` | windmill + pot | 277 | 1.09 (GT) | reconstructs; two nodes |

Both NVIDIA sequences are one camera of a rig held still while the people move, so the
"camera motion" a reconstruction reports on them is hallucinated drift — on `Jumping` it
produced a degenerate re-anchor (Sim(3) scale **-0.0038**, a reflection). They were the v3
plan's first choice on entity count alone; the cameras were never checked. They are out.

**Screening DAVIS settled it.** All 33 sequences with 3+ annotated objects were reconstructed
(geometry only, ~40 s each) and ranked by **baseline / median scene depth** — the widest
separation between two camera centres against how far away the scene is, which is what decides
whether there is anything to triangulate. Path length is not the measure: a jittery handheld
camera accumulates path without going anywhere.

| clip | objects | frames | baseline/depth |
|---|---|---|---|
| `longboard` | 5 | 52 | 1.707 |
| **`lab-coat`** | **5** | **47** | **0.565** |
| `gold-fish` | 5 | 78 | 0.516 |
| `dogs-scale` | 4 | 83 | 0.472 |
| `miami-surf` | 6 | 70 | 0.470 |
| *spin, for scale* | *1* | *426* | *0.579* |
| `lindy-hop` | 8 | 73 | 0.155 |
| `schoolgirls` | 7 | 80 | 0.089 |

The richest clips are the worst shot — `lindy-hop`'s eight dancers sit behind a near-static
camera — but the 4-6 object band holds several clips at or above spin's proven parallax.

**Loop: DAVIS `lab-coat`.** Three similar-looking people (long hair, white coats: a real
identity test the graph must not collapse), relations that matter (near, left-of/right-of, one
holding a phone), slow motion over open ground with a fence and trees to anchor the static
structure, and a baseline/depth of 0.565 against spin's 0.579. Removing one of three people is
an edit whose correctness anyone can judge. Its 47 frames are few, which is the trade.

**Audit: DAVIS** with GT masks — `dogs-scale` (puppies on a scale on a table: the cleanest
`on-top-of` in the set, 83 frames), `longboard` (skater on board, backpack on skater, cyclist on
bicycle — richest relations, but a fast chase with motion blur), and `lindy-hop` as the
deliberate stress case. GT masks separate "the definition is wrong" from "SAM2 is wrong".

**No held-out scoring on DAVIS.** It ships no other viewpoints, so a masked-NVS number is not
available for the loop clip; spin remains the clip for reconstruction quality. The loop does not
need a score — §8 already limits what its diff may claim.

---

## 7. The edit ops — `remove` and `move`

Both against Lowering A (our own 3DGS renderer), which is **faithful by construction**: a `remove`
leaves a real hole, with no plausible fill. That hole is not a bug to hide; it is exactly why
Level 2 (a generative backbone) exists in the design, and it is out of scope here.

- **`remove(entity)`** — drop the entity's Gaussians for the whole clip. Tests the bridge and
  nothing else.
- **`move(entity, delta)`** — transform the entity's Gaussians along a trajectory. Materially
  harder than `remove`: it exposes the hole *and* must carry the entity's dynamic motion bases
  with the transform, so a moved person must still walk. Expect this to be where the thin loop
  strains; that is the point of including it.

Per the architecture doc's instruction, the plan carries an **op-support matrix** (what each
target honors / no-ops / errors) from the start, so the IR does not fracture silently later.

---

## 8. What this spike does NOT measure

**The thin loop's graph diff is a smoke test, not a measurement.** Re-perceiving our own render
carries every confound Panel 2 named against v1: SAM2 re-partitions the render, the real↔render
embedding domain gap, the own-renderer near-tautology, no ground truth, no chance baseline — and
with an edit applied there is now a real hole in the input as well. It tells us the loop runs end
to end and that a removed entity disappears from the re-perceived graph. It does **not** tell us
the graph diffs discriminatively.

That measurement stays **Part 2** (novel-view, partition-controlled, GT-anchored, chance-
baselined), planned separately, after the definition lands.

---

## 9. Where the code lives

- **Lands in the pipeline, with contracts and tests:** the bridge (step 1), typed relations and
  presence (step 2). These are deterministic, cheap, rebuildable functions of stages that already
  exist — the same shape as the `graph` stage, which is exactly why it was landed properly rather
  than spiked. A second throwaway node model is what this plan exists to avoid.
- **Throwaway first, in `../spike_graph_definition/`:** the query parser and DSL, the render-plan
  IR, the re-perception diff, the audit viewer. They land once their shapes settle.

---

## 10. Acceptance criteria

**Step 1 (gate).** A person looking at the render agrees the removed region *is* the entity and
the rest of the scene is intact. Reported with the fraction of Gaussians assigned, per entity, and
the entities too small or too occluded to assign.

**Step 3.** The loop runs end to end without hand-editing intermediate artifacts; `remove` makes
the entity absent from the re-perceived graph; `move` puts it somewhere else and it still moves.

**Step 4 (per axis, each graded grounded / partial / ungrounded, with evidence).**
- **Nodes are real entities:** a person agrees nodes ↔ real objects/people, one node per entity,
  stable identity through the clip.
- **Edges are real relations:** asserted relations are physically true; false-positive and missing
  rates acceptable against a fixed rubric.
- **Queryable:** the conceptual questions return human-correct answers — count/select ("how many
  people?", "what categories are here?"), relate ("what is the person near?"), motion ("which
  entities moved?", "where did the person on the left go?").

**Controls:** GT masks before SAM2; per-entity trajectory uncertainty reported and low-support
entities flagged, never silently trusted; multiple sequences with contrasting content — **no
conclusion from one clip**; a short written rubric so "a human agrees" is a checklist, not a vibe.

---

## 11. Cost, logistics, risks

- **GPU:** a full `Jumping` run (geometry → scene → render) is the main cost, ~1/4 of spin's frame
  count. The DAVIS audit needs geometry per sequence. Every GPU job goes through
  `gpu_guard.sh --detach` with the 400 W cap verified first; heavy CPU
  work runs under a `systemd-run --scope` memory cap.
- **Risks:**
  - **The bridge is noisy for small or heavily occluded entities** → report assigned fraction per
    entity; fall back to a mask-centroid neighbourhood; flag entities that cannot be assigned.
  - **`move` breaks the dynamic bases** → accept a static-transform first pass and say so.
  - **VGGT may fail on fast-motion DAVIS clips** → pick sequences it handles, flag the ones it
    does not; `lindy-hop` is expected to be hard and is included for that reason.
  - **Relation thresholds need per-scene tuning** → prefer scale-relative thresholds, report
    sensitivity.
  - **Re-perceiving a render with a hole** confuses segmentation → precisely why §8 calls the diff
    a smoke test.

---

## 12. Open decisions (proposed defaults)

1. **Relation set (rung 1):** as §3. *Proposed; open.*
2. **Query surface:** fixed conceptual templates first, freeform NL→program (VisProg-style, small
   LLM parser) as a stretch. *Proposed; open.*
3. **Where `move` gets its trajectory:** a user-supplied delta first; graph-derived paths later.
   *Proposed; open.*
4. **Viewer:** self-contained HTML per sequence + a gallery. *Proposed; open.*
5. **Audit rubric:** drafted with step 1, fixed before step 4. *Proposed; open.*

---

## Appendix A — v2 → v3 (why it reshaped)

v2 split the work into *define the graph and audit it* (Part 1) and *measure discriminability*
(Part 2), and deliberately kept all spike code throwaway. Two things changed.

**The definition half started landing for real.** The corrected node model — entities, not point
tracks, with trajectories built from member-track aggregation — is on develop as the `graph`
stage, with contracts, resume coverage and tests. v2's assumption that none of this should touch
the pipeline no longer holds for the parts that are deterministic and settled.

**The scope was wrong.** Part 1 + Part 2 together still never covered **editing**, which the
README promises and the showcase advertises. Measuring the graph's discriminability before
anything can be queried, edited or re-rendered orders the work by rigour rather than by risk, and
leaves the loop open indefinitely. v3 keeps v2's insistence that the model be defined and
human-audited — that is steps 2 and 4, unchanged in substance — but puts the **grounding bridge**
first, because it is the assumption everything else rests on and the one that can invalidate the
design outright.

What v3 explicitly keeps from v2: entities-not-point-tracks; visibility ≠ existence; GT masks
before SAM2; no conclusion from one clip; a written rubric; and the refusal to let a smoke test
masquerade as a measurement.

## Appendix B — v1 → v2 (preserved)

v1 pre-registered a single instrument: the **null-edit round-trip residual** (perceive → render a
near-identity answer → re-perceive → diff), intended both to measure the eval's correspondence
floor *and* to ground the graph definition. Two adversarial review panels showed one instrument
could do neither cleanly:

- **It couldn't cleanly measure the floor.** Same-trajectory rung-1 is rigged toward a trivial
  pass and sits inside the own-renderer near-tautology; the headline "spurious diff" is dominated
  by SAM2 re-partitioning the render and the real↔render embedding domain gap; there was no
  ground-truth or chance baseline, and the "delta of 1" reference was asserted, never computed.
- **It couldn't ground the definition.** A node-appearance measurement says nothing about edges,
  relations, presence semantics, or an actual executed query — yet the deliverable claimed to
  define a *queryable* graph. Presence-from-visibility re-imported the exact conflation the spike
  existed to kill.

Panel 1 separately found that keying node identity on `track:{stream}:{track_id}` — a grid-seed
index that cannot survive a re-render — made the eval's diff join key noise, and that point tracks
are not entities. That finding is what the shipped `graph` stage now implements.
