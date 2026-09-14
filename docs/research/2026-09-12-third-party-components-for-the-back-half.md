# Third-party components for the back half — build vs. integrate survey

**Date:** 2026-09-12
**Status:** SURVEY — grounding for
`docs/design/2026-09-12-back-half-ladders-and-graph-contract.md`
**Question:** before writing the queryable-graph contract, is there an existing framework that
already does this, and which layers should we adopt rather than build?
**Standing constraint:** reifyscene is a *pipeline* question — "can these links be chained into
something that works" — not a single-model contribution. Integrating third-party rungs is the
intended mode, not a concession.

---

## 0. The finding in one paragraph

The field has closed on this problem from two directions. From **3D scene understanding** come
open-vocabulary scene graphs (ConceptGraphs, Hydra/Clio, HOV-SG) — real entities and relations,
but static scenes from posed RGB-D, with no renderable handle. From **4D Gaussian splatting** come
language-embedded feature fields (Feature4X, Mono4DEditor, LangSplat, Gaussian Grouping) —
monocular video in, editable and re-renderable out, but the "graph" is a continuous feature field
with no entities to count and no relations to hold. **Nothing published occupies the intersection:
explicit, persistent, temporally-scoped entities with typed relations, grounded in a 4D
reconstruction you can edit and re-render.** That intersection is where reifyscene sits, and it is
narrow enough that the contract has to say precisely what it is. Everything *under* it — the
relation vocabulary, the query surface, the semantics rung, the graph data structure — should be
adopted, not invented.

---

## 1. The two closest works, and why they do not retire the project

| | Feature4X (CVPR 2025) | Mono4DEditor (Oct 2025) |
|---|---|---|
| Input | monocular video only | monocular video + text prompt |
| Representation | 4D Gaussian feature field, distilling SAM2 + CLIP-LSeg + InternVideo2 | 3D Gaussians augmented with quantized CLIP features |
| Selection | feature similarity | two-stage CLIP-similarity localization + spatial refinement |
| Tasks | interactive segmentation, language-guided editing, 4D VQA, novel-view synthesis | text-driven edit via diffusion video editing model |
| Symbolic graph | **no** — a continuous feature field, explicitly not a symbolic structure | **no** — localization by embedding similarity |

Both honor *"delete the car"*. Neither can answer *"how many people are there, and which one was
near the chair before it moved"* — there are no entities to count, no relations to hold, no
presence intervals to reason over. That is the distinction the contract must make load-bearing.

The video-native scene-graph work has the opposite shape: **PSG-4D** and **OpenPVSG** (CVPR'23, 400
videos, 126 object categories, 57 relation predicates) have real entities and rich relations, but
live in 2D pixel mask-tubes with no 3D scene to re-render.

**Conclusion:** the gap is real. Do not adopt a whole-pipeline framework; adopt per-layer.

---

## 2. Layer-by-layer verdicts

### 2.1 Graph data structure — **candidate, fit unverified**

**Spark-DSG** (MIT-SPARK) — BSD-2-Clause, `pip install ./Spark-DSG`, C++ core with Python
bindings, usable standalone (ROS optional). The data structure under Hydra and Clio.

Our `GraphNode` / `GraphEdge` / `GraphArtifact` is a thin hand-rolled version of the same idea.

**Caveat that blocks adoption today:** Spark-DSG's layer model is SLAM-shaped
(objects / places / rooms / buildings) and its *dynamic* layer means **agent trajectories**, not
"relations whose truth changes per frame". Our core need — edges with validity intervals — may not
be expressible in its schema without abuse.

**Verdict:** keep our structure as the floor; evaluate Spark-DSG as an **export target** rather
than the in-memory model. Revisit if the fit check passes.

### 2.2 Relation vocabulary and computation — **adopt**

**VLA-3D** — MIT license, relation generation in `scene_graph/`, 23.5M relations generated at
dataset scale. Eight predicates, all **view-independent**:

`above` · `below` · `closest` · `farthest` · `between` · `near` · `in` · `on`

Computed by heuristics over object bounding boxes, with a filter that drops relations when
target/anchor boxes significantly overlap or one encloses the other.

They deliberately exclude `left-of` / `right-of` because viewpoint-dependent relations are
ambiguous for referring expressions. **This overrides the camera-relative proposal made earlier in
the spike discussion.** A camera-relative relation flips meaning whenever the camera moves — and
for us the camera always moves; that is the only reason we can reconstruct at all.

**Verdict:** adopt the predicate set and the overlap filter. View-dependent relations are a
**query-time** concern, resolved against a named reference frame, not scene-graph content.

### 2.3 Open-vocabulary semantics — **adopt the rung above ours**

**ConceptGraphs** — open-vocabulary 3D scene graphs from **posed RGB-D** (which VGGT gives us),
using CLIP features plus LVLM captioning per node and an LLM for edges. Code released.

Authors' stated limitation, relevant to us: CLIP embeddings fail on queries referencing multiple
concepts, negation, or complex composition; the LLM path fixes that but requires an LLM at
inference.

**Verdict:** the LVLM-caption rung sits **above** our existing SigLIP2 rung and is the natural top
of the `semantics` ladder. But see §3 — our immediate problem is not the missing rung.

### 2.4 Entity → Gaussian grounding — **keep ours, cite theirs**

| Method | Mechanism | License | Limitation for us |
|---|---|---|---|
| **Gaussian Grouping** (ECCV'24) | per-Gaussian Identity Encoding, SAM-supervised, 3D consistency regularization; supports removal, inpainting, recomposition | Apache-2.0 | **static scenes only**; requires retraining the splat with an extra channel |
| **LangSplat** / **OpenGaussian** / **Feature-3DGS** | per-Gaussian language features; select by text similarity | varies | static; retraining; selection is similarity, not membership |
| **ours** (`graph/entity_gaussians.py`) | project trained Gaussians into each view, majority-vote against entity masks | — | noisy for small/occluded entities |

Ours needs **no retraining** and already runs on the **dynamic** trained cloud — both of which the
published alternatives lack. It is the cheap version of a well-studied idea, and it is the right
rung for us today.

**Verdict:** keep. Record the learned methods as the ladder rungs above, and note the dependency
inversion they would cause (§3.3 of the design doc).

### 2.5 Query surface — **adopt the design**

**Structured Interfaces for Automated Reasoning with 3D Scene Graphs** (arXiv 2510.16643,
Oct 2025). Encodes the scene graph in a **graph database** and gives an LLM **Cypher** as a tool
interface, retrieving only the task-relevant subgraph (RAG over the graph) rather than serializing
the whole graph into context.

Evaluated on instruction following and scene question-answering; beats both context-window encoding
and code-generation baselines, scales to large graphs on local *and* cloud models, and substantially
cuts token count.

**Verdict:** this is the design for the four stubs in `reifyscene/query/`. Do not invent a DSL.
Choose an **embedded** graph engine so the floor rung stays dependency-free.

### 2.6 Whole-pipeline frameworks — **do not adopt**

Feature4X and Mono4DEditor would mean replacing the reconstruction backbone of a working 15-stage
pipeline to obtain capabilities that do not include the symbolic graph. Rejected on cost and on
fit.

---

## 3. What the survey did *not* predict: our semantics problem is a vocabulary bug

The survey was undertaken partly to scope a possible diversion into fixing perception. Checking the
`semantics` artifact from `davis_lab-coat_full` closed that question differently.

The **top rung already ran** — `backend_used: siglip2`, `fidelity: open_vocab`, 8/8 tracks named.
But `reifyscene/perception/semantics/siglip_semantics.py:36`:

```python
# Default candidate vocabulary = COCO-80 (drops the torchvision placeholders), so
# the SigLIP rung is comparable to the detector floor; override via cfg['vocab'].
DEFAULT_VOCAB = [c for c in COCO_INSTANCE_CATEGORY_NAMES if ...]
```

An open-vocabulary model asked a closed-vocabulary question. The choice was deliberate — to make the
rung A/B-comparable against the detector floor — and became the default.

**The scores expose it.** Top-1 cosine on lab-coat, with the top-2 margin:

| track | top-1 | score | top-2 | margin |
|---|---|---:|---|---:|
| `:3` | **person** | 0.1080 | bear | **0.0449** |
| `:2` | **person** | 0.1151 | cell phone | **0.0330** |
| `:6` | handbag | 0.1170 | backpack | 0.0105 |
| `:7` | bird | 0.1078 | umbrella | 0.0063 |
| `:0` | cup | 0.1020 | handbag | 0.0061 |
| `:1` | cup | 0.1016 | handbag | 0.0041 |
| `:4` | bird | 0.1077 | broccoli | 0.0022 |
| `:5` | cup | 0.0850 | sink | 0.0002 |

Two observations:

1. **Margin separates cleanly; absolute score does not.** The two real people carry margins of
   0.045 and 0.033. Every junk label is ≤ 0.0105. Any threshold in **0.011 – 0.033** keeps exactly
   the two people and abstains on the other six. Meanwhile `handbag` has the *highest absolute
   score in the set* (0.1170) and is junk — so `min_score`, the abstention hook that already exists
   in `classify()`, is **the wrong instrument**.
2. **A flat top-5 is the model saying "none of these."** Nothing in the pipeline treats it that
   way; `cup @ 0.10` is written into the graph node as a category with the same standing as a fact.

So the perception problem decomposes three ways, and only the third is research:

- **Vocabulary** — `cfg['vocab']` is already the override hook. A config experiment.
- **Abstention** — a correctness defect in what the graph asserts. Isolated; see the design doc §2.
- **Instance separation** — two human nodes for three people on lab-coat; one person split across
  two nodes on spin. Lives in `segment` / `reid`. Genuinely hard, and out of scope here.

---

## 4. Adoption summary

| Layer | Decision | Source | License |
|---|---|---|---|
| Graph data structure | keep ours; evaluate as export target | Spark-DSG | BSD-2 |
| Relation vocabulary | **adopt** (8 view-independent predicates + overlap filter) | VLA-3D | MIT |
| Semantics top rung | adopt LVLM-caption above existing SigLIP2 | ConceptGraphs | released |
| Entity→Gaussian grounding | **keep ours**; cite learned rungs above | Gaussian Grouping | Apache-2.0 |
| Query surface | **adopt the design** (graph DB + Cypher as LLM tool) | arXiv 2510.16643 | n/a (design) |
| Whole pipeline | reject | Feature4X, Mono4DEditor | — |

---

## Sources

- ConceptGraphs — https://concept-graphs.github.io/ · https://arxiv.org/abs/2309.16650
- Spark-DSG — https://github.com/MIT-SPARK/Spark-DSG · Hydra — https://github.com/MIT-SPARK/Hydra ·
  Clio — https://github.com/MIT-SPARK/Clio · Khronos — https://github.com/MIT-SPARK/Khronos
- VLA-3D — https://github.com/HaochenZ11/VLA-3D · https://arxiv.org/abs/2411.03540
- Gaussian Grouping — https://github.com/lkeab/gaussian-grouping · https://arxiv.org/abs/2312.00732
- LangSplat — https://langsplat.github.io/ · OpenGaussian (NeurIPS 2024)
- Structured Interfaces for Automated Reasoning with 3D Scene Graphs — https://arxiv.org/abs/2510.16643
- Feature4X — https://github.com/ShijieZhou-UCLA/Feature4X · https://arxiv.org/abs/2503.20776
- Mono4DEditor — https://arxiv.org/abs/2510.09438
- PSG-4D — https://arxiv.org/abs/2405.10305 · OpenPVSG — https://github.com/LilyDaytoy/OpenPVSG
