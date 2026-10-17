# ConceptGraphs: the verdict, and the plan for our graph

**Status:** PROPOSED
**Date:** 2026-09-20
**Decides:** the build-vs-integrate question the ConceptGraphs investigation was opened to answer
— use it, move to another tool, or inform ours
**Evidence:** their source at `93277a02`, read end to end; plus a 12-sequence run of their
detection + mapping stages on our DAVIS export (§5)
**Feeds:** `docs/design/2026-09-19-roadmap-inhouse-components-and-modalities.md` (components),
`docs/design/2026-09-13-stage-audit-and-end-to-end-plan.md` (work items)

---

## 0. The question

Not "what score does ConceptGraphs get". The survey that opened this
(`docs/research/2026-09-12-third-party-components-for-the-back-half.md`) asked whether a released
concept-graph implementation could be **used**, **replaced by something better**, or **learned
from**. ConceptGraphs is the closest released instance of the choice this project makes — explicit
entities and relations rather than a feature field — so it is the one to read.

Most of the answer came from reading the code, not running it. That is recorded first.

---

## 1. What ConceptGraphs is, against our contract

| | ConceptGraphs | our contract (§3 of the 2026-09-12 doc) |
|---|---|---|
| **identity** | CLIP feature + accumulated point cloud | `category`, `margin`, `abstained`, `caption`, embedding |
| **time** | **none** — one static cloud per object | presence intervals, gap flagging, `position(t)`, `extent(t)`, motion |
| **association** | CLIP cosine **+** 3D overlap, summed to one score, greedy, periodic merge | DINOv2 for non-overlapping pairs; 3D separation for co-visible pairs |
| **naming** | LLaVA per-view captions → GPT-4 consensus | SigLIP 2 scored against a fixed vocabulary, with abstention |
| **relations** | GPT-4, **2 predicates** (`on`, `in`) + `none`, on spanning-tree edges only | 8 view-independent predicates, validity intervals ∩ presence |
| **editing** | no per-entity handle into anything renderable | renderable handle + op-support matrix |

Details worth keeping:

- Their matcher is `sim = (1 + phys_bias)·spatial + (1 − phys_bias)·visual`, thresholded and
  greedy, with DBSCAN denoising and a merge pass every *k* frames
  (`conceptgraph/slam/mapping.py`).
- Their relation prompt asks which object is *"commonly placed on top of"* the other, given only
  bbox centre, bbox extent and a tag. It is a **commonsense prior about categories**, not a
  measurement of the scene.
- Relations are computed only on a **minimum spanning tree** per connected component, so the edge
  set is sparse by construction.
- Their naming needs `LLaVA-7B-v0` — 2023 delta weights against LLaMA-1, whose base weights are
  not freely obtainable — plus the LLaVA repo at a pinned 2023 commit that upstream says "may
  require some adaptations".

---

## 2. The verdict

**Use it in our pipeline — no.** One structural reason: it has no time. Detections accumulate
into a single static cloud per object; there are no presence intervals, no validity intervals, no
`position(t)`. The grounding law (§3.5 of the contract) and every edit operation depend on those.
Adopting it means giving up what the project is for.

**Move to another tool — not on this evidence.** The survey found nothing released in our
intersection. Khronos is genuinely dynamic but closed-vocabulary and ROS/C++. **DynamicGSG remains
unchecked** (§7).

**Inform ours — yes.** Four concrete adoptions (§3), two explicit rejections (§4).

---

## 3. What we adopt

**3.1 Their association shape — the one thing they do better than us.**

They blend appearance and geometry into *one* score over *all* candidate pairs. We split the
problem: appearance (DINOv2) only for tracklets that never co-occur, 3D separation only for those
that do — and the co-visible half recovers **31 of 247** same-object pairs.

Two consequences, both landing in the learned co-visible merger
(`docs/design/2026-09-18-learned-covisible-merge-proposal.md`):

- **add an appearance cosine to the co-visible feature set.** Its absence is not a tuning gap, it
  is a missing signal: the current rule cannot see appearance for exactly the pairs it judges.
- **`sim_sum` becomes a third baseline** at M1, beside our rule and the pooled MLP. A learned
  merger that cannot beat a two-term linear blend has not earned its place.

**3.2 Naming by captioning, not vocabulary scoring.** Confirms the survey's §2.3 verdict, and
reading the code gives the method: best views per entity → crop with context → caption each →
consolidate. The GPT in their consolidation is incidental; the structure is the lesson. Ours goes
in the `semantics` ladder as a rung above the SigLIP 2 scorer, with a local VLM, and the
consolidation done by medoid over SigLIP 2 text embeddings — a model already in the stack.

**3.3 Their node structure validates ours.** Their per-detection parallel lists (`image_idx`,
`mask`, `conf`, `n_points`) are our `support` block; their per-node `clip_ft` is our embedding.
Independent convergence is weak evidence, but it is evidence.

**3.4 The negative lesson, which is the most valuable one.** Fusing detections into one static
cloud is precisely what makes a moving object smear across the world frame. Our `position(t)` +
presence design exists to avoid that, and this is the clearest available statement of what
differentiates the project. It belongs in how we describe the work.

---

## 4. What we explicitly do not adopt

**Their relation layer.** Two commonsense predicates on a spanning tree, derived from a category
prior rather than from the scene. Nothing measured is nothing verifiable, and a relation that was
never measured cannot be maintained across an edit. Our VLA-3D predicate set with validity
intervals is strictly more, and it is ours to build.

**Their static fusion.** §3.4. Recorded here so that a later reader does not mistake its absence
for an oversight.

---

## 5. The one empirical question

Reading settles naming and relations. It cannot settle whether **their association actually
outperforms ours on the same footage** — so that is the only thing being run.

Twelve sequences, their published class-agnostic configuration unmodified, `stride=1`; no external
API is touched (the two stages reach ten `conceptgraph` modules, none importing `openai`). The set
is weighted toward our failures: 47 of 205 ground-truth objects, and 25 of the 73 that SAM 2 never
proposes at all.

**Interpretations, fixed before the numbers arrive:**

| If | Then | Action |
|---|---|---|
| they find the never-proposed objects | our proposal stage is the defect | add a detector rung to `segment` |
| they miss them too | class-agnostic proposal is intrinsically blind here | the GroundingDINO rung is the fix, not a denser AMG |
| recall ≈ our 0.693 | the inventory is competitive; the gap is downstream | stop tuning perception; build §6 |
| recall ≫ ours | their proposal or association is genuinely better | adopt the specific mechanism, measured |
| they collapse only on `koala` / `gold-fish` | our geometry is the ceiling there | fix geometry first, starting with `pose_refine` |

`judo` is the sharpest single row: SAM 2 proposes neither object, yet we recover both through
person-detector seeding. If they miss both, the 73-object gap needs a detector, not a better mask
generator.

Naming is *not* tested: under `--class_set none` their objects carry no labels. That is the
correct split, not a limitation.

### 5.1 Results (2026-09-21)

Twelve sequences plus `bear`, 61 minutes on one GPU, 0 failures. Both systems scored by
`score_masks_by_frame` on **identical sampled frames**.

| | GT objects | recovered | recall | spurious |
|---|---|---|---|---|
| **ours** | 48 | 28 | **0.583** | 156 |
| **ConceptGraphs** | 48 | 23 | **0.479** | 354 |

Read the subset for what it is: it was **chosen to concentrate our failures**, so 0.583 is not
comparable to our 0.702 over all ninety. The comparison between the two columns is the point.

**A scorer defect had to be fixed before these numbers meant anything.** The shared scorer reported
whichever frame a system scored best on, so the denominator was *that* frame's object count — on
`lindy-hop`, ours was measured against 8 objects and the reference against 5. It also skipped
frames where a system produced no masks, excusing it from the objects it found nothing on. Both are
now frame-set properties (§ the scorer's own docstring), and the regression tests pin them. Our own
benchmark moved **0.693 → 0.702** on regeneration: two sequences gained an object that was
recovered on a frame that was not the peak one. No denominator on our side changed, which is why
the defect stayed invisible until a second system was scored.

### 5.2 The pre-registered interpretation was wrong, and the real result is better

§5's table predicted that if ConceptGraphs also missed the never-proposed objects, the fix was a
detector rung. Per **sequence** aggregates looked like that. Per **object** they do not.

Of the 25 objects in this subset that SAM 2 never proposes at all:

| | count | share |
|---|---|---|
| recovered by ours only | 6 | |
| recovered by ConceptGraphs only | 7 | |
| recovered by both | 3 | |
| recovered by neither | 9 | |
| **ours** | **9** | **36%** |
| **ConceptGraphs** | **10** | **40%** |
| **either** | **16** | **64%** |

They recover slightly *more* of them than we do, and — the part that matters — **a substantially
different set**. Only 3 of 16 are shared. SAM 1's automatic mask generator at their sparse 12×12
prompt grid proposes things SAM 2's does not, and the reverse.

**So proposal is not a ladder with a better rung; it is a set of partial, complementary views.**
The action changes accordingly: not only "add a detector rung" but **combine proposal sources and
keep what any of them offers**. Our own GroundingDINO probe already pointed the same way (31 of 45
unproposed objects covered by a blind COCO-80 list); this says the same about a second
class-agnostic generator. Nine of 25 are found by nothing here, so the ceiling is real but well
above where we sit.

Caveat that keeps this honest: 25 objects is a small sample, and the complementarity claim deserves
the full ninety before it drives much work.

### 5.3 What the run did *not* settle

**Whether their association beats ours.** End-to-end recall conflates proposal, association and
geometry, and their proposer lacks the detector ours has. Removing sequences where proposal is a
confound leaves 4 ground-truth objects — ours 0.250, theirs 0.250 — which is no evidence at all.
The only real signal is the crowd clips, where proposal is not the whole story: `lindy-hop` 7/8
against 4/8, `schoolgirls` 4/7 against 4/7.

Isolating association properly would mean feeding both systems the *same* proposals. That is a
real experiment and it is not this one. §3.1's adoption does not wait on it: the appearance cosine
is missing from our co-visible features whatever their matcher scores.

**Their spurious count is 2.3× ours** (354 against 156). Not directly comparable — DAVIS annotates
only objects of interest — but the gap is large enough to note, and `koala` is the extreme: 214
objects produced, 0 recovered, on the clip with the worst pinhole residual in the export (9.2%).
Both systems score 0.00 there, which points at our geometry rather than at either pipeline.

---

## 6. The plan for our graph

**Where it actually stands** (not where the contract says it should):

| Piece | State |
|---|---|
| nodes, presence span, `position(t)`, `extent(t)` | shipped |
| edges | **`observed_by` only** — `relation_rules.py` is an 8-line stub returning its input |
| query surface | **55 lines of stubs** across five files, every method returning a placeholder |
| renderable handle | `entity_gaussians.py` is **implemented and tested with zero production callers**; the manifest reports `renderable: absent` |
| presence-gap flagging | not built |
| naming | vocabulary scoring + abstention; 2 of 8 labels agreed with on `lab-coat` |
| co-visible association | 31 of 247 pairs, 0 false merges |

**The ordering principle.** The standing decision (audit §7) is one configuration delivering the
full vision end to end, not every rung combination. So the plan is a **thin vertical slice first**
— ask, edit, re-render, verify on one clip — and depth afterwards. Breadth-first here would mean
a graph with excellent nodes that still cannot answer or change anything.

**Blocked as written — see the step-1 outcome.** The gate was run on ten sequences on
2026-09-28 and one passed (`docs/design/2026-07-09-queryable-graph-definition-spike-plan.md`
§5a). Removing an entity damages the entities beside it wherever they are adjacent in depth,
which is most of this footage. G1 and G2 below assume an entity set that can be separated in the
representation it indexes; that assumption is what failed. **Entity separability comes first**,
and it is upstream of every item here — scoped at
`docs/design/2026-09-29-per-gaussian-identity-scope.md`: per-Gaussian identity supervised by the masks we already have, so the
representation is asked to be separable rather than assumed to be.

**G1 — the graph answers a question.**
Build the relation computer for a first subset of predicates that geometry already supports
(`near`, `above`/`below`, `on`), each edge carrying a validity interval intersected with both
endpoints' presence. Add presence-gap flagging, which the intersection needs to mean anything.
Replace the query stubs with a structured (not natural-language) retrieval over nodes, relations
and intervals.
*Done when:* a structured query returns the right entities on `lab-coat`, and an entity's edges
disappear across its own presence gaps.

**G2 — the graph honours an edit.**
Wire `assign_gaussians_to_entities` into the pipeline so `renderable` is grounded rather than
absent, then execute `remove(entity)` through to a re-render.
*Done when:* a person is removed from `lab-coat` and the re-render has a real hole, with the
manifest reporting the entity's assigned fraction — and `unassignable` where it cannot.

**G3 — the graph is trustworthy.**
The quality pass, and where §3's adoptions land: the captioning rung (3.2), the appearance cosine
and `sim_sum` baseline in the learned merger (3.1), and **proposal ensembling** — which §5.2 turned
from a guess into a measurement. Proposal sources are partial and complementary, not ordered: a
second class-agnostic generator recovers 7 objects ours misses while missing 6 ours finds, and a
blind COCO-80 detector covers 31 of 45. The rung to build takes the union of proposal sources and
lets the association stage sort out the duplicates — which is exactly what the learned merger is
for, so 3.1 and this are one piece of work, not two.
*Done when:* labels a person agrees with improves measurably on `lab-coat`, and the inventory
benchmark moves without recall regressing.

**G4 — verification.**
Re-run the inventory benchmark and the covisibility-masked NVS benchmark after G2, so an edit is
shown not to damage the rest of the scene.
*Done when:* both benchmarks are re-scored and committed with the commit that changed them.

G1 and G2 are the vertical slice; G3 deepens it; G4 is what makes the claim checkable. Audio and
the 4D NeRF (the roadmap's other components) sit **after** G2 — they extend a graph that works,
and extending one that cannot yet answer or edit would widen the project rather than advance it.

---

## 7. Open

1. **DynamicGSG** is the one surveyed candidate never examined — Gaussian-map scene graph with
   dynamic updates (IROS 2025, released). Worth one reading pass before §2's "move to another
   tool — no" is treated as settled. The specific question: are its dynamics intra-video, or
   between-visit change detection?
2. **Isolating association from proposal.** §5.3: the run could not separate them, because their
   proposer lacks a detector. The clean experiment is to feed both systems the *same* proposals and
   compare only what each does with them. Worth doing before claiming either matcher is better;
   not a blocker for §3.1.
3. **The complementarity result on all ninety.** §5.2 rests on 25 objects. The proposal-ensembling
   work in G3 should not be sized until the full set confirms it.
4. **Relation predicate subset for G1** — `near`/`above`/`below`/`on` is a guess at what our
   geometry supports at 0.3–14% pinhole residual. It may need trimming after the first
   measurement.
