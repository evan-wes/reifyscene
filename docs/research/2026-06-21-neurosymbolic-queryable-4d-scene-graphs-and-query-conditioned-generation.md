# Literature review — neurosymbolic queryable 4D scene graphs + query-conditioned novel-view/video generation

**Date:** 2026-06-21
**Source:** deep-research multi-agent scan (6 angles, 29 sources fetched, 143 claims extracted, 25 adversarially verified → 23 confirmed / 2 killed). Representative, **not exhaustive** — see Limitations.
**Scope / question:** Answering natural-language queries about a *reconstructed dynamic real-world scene* by **generating a novel view/video** as the answer, and the queryable scene-graph + neurosymbolic reasoning substrate that would drive it. Goal: learn the field and locate the frontier (research, not product).

---

## Headline

The literature splits into two mature halves that **do not meet**:

1. a **"query/reason over a reconstructed scene"** side (scene graphs + neurosymbolic NL reasoning), and
2. a **"generate/edit views as output"** side (dynamic NeRF / 4DGS, instruction-conditioned editing, conditioned video generation).

The specific intersection — *answer an NL query about a reconstructed **dynamic** scene by **generating a novel view/video**, driven by a **queryable temporal scene graph*** — is **essentially undemonstrated end-to-end**; it is only assemblable from parts. (Confidence: **medium** — this is an absence-of-evidence synthesis over the surveyed set, not a directly cited negative result.)

The unclaimed ground is the **conditioning interface**: turning a symbolic/temporal query plan into a *faithful* generated rendering, plus an **evaluation** for query-conditioned generation. This aligns directly with room4d's "render-as-answer" thesis.

---

## Pillar 1 — the queryable representation (3D → 4D scene graphs)

| System | What it establishes | Cite | Confidence |
|---|---|---|---|
| **3D Scene Graph** (Armeni et al., ICCV 2019) | Canonical queryable substrate: one graph unifying objects/rooms/cameras with per-entity attributes + relations (4 layers: building/room/object/camera). | [arXiv:1910.02527](https://arxiv.org/abs/1910.02527) | high |
| **Clio** (MIT SPARK, RA-L/RSS 2024) | Graph construction reframed as **task-driven, open-set** — object granularity is task-dependent, formalized via the **Information Bottleneck** (Agglomerative-IB), built **online** as a robot explores. | [arXiv:2404.13696](https://arxiv.org/pdf/2404.13696) | high |
| **Khronos** (MIT SPARK, RSS 2024, Outstanding Systems Paper) | **Closest existing substrate to ours.** Spatio-temporal metric-semantic SLAM separating **short-term dynamics** (people, a cart) from **long-term changes** (object add/remove at loop closure); **fix a query time and replay** the estimated scene state (belief-time vs robot-time). Output is a map/replay — **not** query-conditioned generation. | [arXiv:2402.13817](https://arxiv.org/abs/2402.13817), [code](https://github.com/MIT-SPARK/Khronos) | high |

Caveat: Clio's "onboard compute" is a high-end mobile workstation (i9 + RTX 4090 laptop GPU), not a low-power embedded SoC.

---

## Pillar 2 — neurosymbolic NL querying (parse → program → execute)

| System | What it establishes | Cite | Confidence |
|---|---|---|---|
| **NS-CL** (Mao et al., ICLR 2019) | Object-based symbolic scene + sentence→executable-program from **natural supervision** (image+QA, no explicit concept/parse supervision); neuro-symbolic executor; strong compositional generalization. Mostly CLEVR/synthetic; per-object features are "quasi-symbolic" latents; fixed DSL. | [arXiv:1904.12584](https://arxiv.org/abs/1904.12584) | high |
| **CLEVRER / NS-DR** (Yi et al., ICLR 2020) | The temporal/**causal** event-reasoning benchmark (descriptive/explanatory/predictive/counterfactual). Neural models ace perception, **fail on causal**. NS-DR = frame parser + dynamics predictor + program parser + symbolic executor over a symbolic *video* representation. | [arXiv:1910.01442](https://arxiv.org/pdf/1910.01442) | high |
| **NS-VideoQA** (Liang et al., 2024) | Closest surveyed instance on **real-world video**: parses static triples *(subj, rel, obj)* + **dynamic actions with explicit start/end times** into a Symbolic Representation; top-down question decomposition + bottom-up compositional reasoning. **Answers in text, not by rendering.** | [arXiv:2404.04007](https://arxiv.org/html/2404.04007) | high |
| **VisProg** (Gupta & Kembhavi, CVPR 2023, Best Paper) | **The key bridge hint.** Training-free LLM-as-program: NL → python-like modular program dispatching to off-the-shelf vision modules. The *same* substrate spans **both reasoning AND image editing/generation** (Stable Diffusion inpainting). But **2D, image-only**, not grounded in any 3D/4D reconstruction. | [arXiv:2211.11559](https://arxiv.org/abs/2211.11559) | high |

---

## The grounding layer is itself unsolved

Open-vocabulary 3D scene understanding — the layer that would tie "the person in the red jacket" to reconstructed geometry — remains low in absolute terms: the **OpenSUN3D 2023 challenge winner reached only mAP 6.08** ([arXiv:2402.15321](https://arxiv.org/html/2402.15321v2)). Partly task difficulty (hard affordance/material/situational queries); 2025–2026 methods are higher but still low. **Confidence: high** that it's unsolved at this benchmark; the often-cited "two clean families (point-cloud vs implicit)" taxonomy was **refuted** in this scan — do not present the landscape as cleanly bifurcated.

---

## Benchmarks & the reasoning gap

- **AGQA** (Grunde-McLaughlin et al., CVPR 2021, [arXiv:2103.16002](https://arxiv.org/abs/2103.16002)) — compositional spatio-temporal VideoQA, ~192M QA pairs / 9.6K videos (balanced subset 3.9M), with **generalization splits** (novel compositions, indirect references, more steps), derived from Action Genome scene graphs. **Humans 86.0% vs best model 47.7%** — large, persistent gap motivating structured/neurosymbolic methods. (47.7% is the 2021 AGQA 1.0 figure; AGQA 2.0 rebalanced.)
- Situated reasoning benchmarks in the source set: **STAR** ([bobbywu.com/STAR](https://bobbywu.com/STAR/)), **SQA3D** ([sqa3d.github.io](https://sqa3d.github.io/)).
- **No surveyed benchmark** scores *query-conditioned generation* (is the rendered answer both correct w.r.t. the query AND faithful to the reconstruction).

---

## Generation / bridge side (sources present, **not verified this pass** — treat as leads)

Budget verified mostly the representation/reasoning side; these appeared as fetched sources but produced no surviving 3-0 claims, so they are **pointers to chase**, not established findings:

- **ConceptGraphs** — open-vocabulary 3D scene graph, LLM-queryable ([concept-graphs.github.io](https://concept-graphs.github.io/)).
- **LLM-grounded Video Diffusion (LVD)** — LLM emits dynamic scene layouts to steer video generation; a concrete *reason→generate* bridge ([llm-grounded-video-diffusion.github.io](https://llm-grounded-video-diffusion.github.io/)).
- **Image Generation from Scene Graphs** (Johnson et al., [arXiv:1804.01622](https://arxiv.org/abs/1804.01622)) — the classic graph→image generator.
- **Instruct-NeRF2NeRF** ([instruct-nerf2nerf.github.io](https://instruct-nerf2nerf.github.io/)) and **Instruct 4D-to-4D** (CVPR 2024) — instruction-conditioned 3D/4D scene editing.

Named in the prompt but **not surfaced with verified claims**: Hydra, SceneGraphFusion, ViperGPT, NS-VQA, LERF, OpenScene, ConceptFusion, OpenMask3D, dynamic NeRF / 4D Gaussian splatting (as generation), NExT-QA. A follow-up scan targeting the generation/bridge half would sharpen this.

---

## Frontier — concrete open problems at the intersection

All four are about **the conditioning interface**:

1. **Language grounding to reconstructed entities** — the open-vocabulary-3D bottleneck (mAP 6.08).
2. **Temporal / event reasoning over 4D graphs** — the AGQA/CLEVRER gap (47.7% vs 86.0% human; causal questions unsolved).
3. **Faithful generation conditioned on a symbolic plan** — turning a 4D query plan into a rendered answer that is *actually faithful* to the reconstruction. **The genuine white space:** nobody connects a Khronos/Clio-style 4D scene graph to a 4D generative renderer.
4. **Evaluation of query-conditioned generation** — no benchmark scores correctness-w.r.t.-query *and* fidelity-to-scene together. Reasoning benchmarks score text; NVS benchmarks score pixels.

### Open questions (verbatim from the scan)
- Has any 2024–2026 system actually closed the loop end-to-end (reconstruct dynamic scene → temporal queryable graph → answer NL query by rendering a novel view/video), or is it strictly assembled-from-parts? (None found, but absence of evidence ≠ proof of absence.)
- What is the right conditioning interface between a symbolic/graph 4D query plan and a 4D generative renderer such that generation is faithful to the plan — and can a VisProg-style LLM-as-program substrate extend from 2D image editing to 4D view/video synthesis grounded in a reconstruction?
- How should query-conditioned generation be evaluated?
- Given that grounding (open-vocab 3D) and compositional video reasoning are *each* independently unsolved, which bottleneck dominates an end-to-end query→render system, and do composed module errors compound or partially cancel?

---

## Limitations of this scan (read before over-trusting)

- **Coverage:** only 25 of 143 extracted claims were verified (budget); the verified map skews to the representation/reasoning side. Several prompt-named systems produced no verified claims (see above).
- **The end-to-end-gap conclusion is a synthesis**, not a cited negative result (medium confidence).
- **Version/date-bound numbers:** AGQA 47.7% = AGQA 1.0 (2021); open-vocab 6.08 mAP = OpenSUN3D 2023 winner. These are "unsolved" snapshots, not current SOTA ceilings.
- **Two claims were killed by adversarial voting** (kept out / softened here): the clean point-cloud-vs-implicit open-vocab-3D taxonomy (1-2), and the stronger AGQA claim that SOTA "barely exceeds linguistic-bias baselines and none generalize" (1-2). Minor source-fidelity fixes applied: AGQA balanced subset is **3.9M** (not 3M); Clio "onboard" = i9+4090-laptop.

---

## Relevance to room4d

The conditioning interface (symbolic 4D query plan → *faithful* 4D generation) and its evaluation are the un-claimed ground, and they line up with room4d's render-as-answer thesis. The proven reconstruction/render chain (geometry → tracking/lifting → motion init → static/dynamic Gaussian training) is the assembly backbone; the **graph (queryable substrate) + query (neurosymbolic plan → conditioned render) stages** are where a genuine contribution would sit. This review is the input to the full-pipeline architecture design for those stages.

---

## Full source list (verified-pass + fetched)

Primary sources, by angle:

- **Queryable / 4D scene graphs:** [arXiv:1910.02527](https://arxiv.org/abs/1910.02527) (3D Scene Graph), [arXiv:2404.13696](https://arxiv.org/pdf/2404.13696) (Clio), [arXiv:2402.13817](https://arxiv.org/abs/2402.13817) + [Khronos code](https://github.com/MIT-SPARK/Khronos), [arXiv:2402.15321](https://arxiv.org/html/2402.15321v2) (OpenSUN3D), [arXiv:2103.16002](https://arxiv.org/abs/2103.16002) (AGQA).
- **Neurosymbolic / LLM-as-program:** [arXiv:1904.12584](https://arxiv.org/abs/1904.12584) (NS-CL), [VisProg CVPR'23 PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Gupta_Visual_Programming_Compositional_Visual_Reasoning_Without_Training_CVPR_2023_paper.pdf) / [arXiv:2211.11559](https://arxiv.org/abs/2211.11559), [arXiv:1910.01442](https://arxiv.org/pdf/1910.01442) (CLEVRER/NS-DR), [arXiv:2404.04007](https://arxiv.org/html/2404.04007) (NS-VideoQA), [AGQA blog](https://ai.stanford.edu/blog/agqa/), [ConceptGraphs](https://concept-graphs.github.io/).
- **Query-conditioned 4D / NVS generation:** [Instruct-NeRF2NeRF](https://instruct-nerf2nerf.github.io/), [Instruct 4D-to-4D (CVPR'24)](https://openaccess.thecvf.com/content/CVPR2024/html/Mou_Instruct_4D-to-4D_Editing_4D_Scenes_as_Pseudo-3D_Scenes_Using_2D_CVPR_2024_paper.html), [arXiv:2305.20082](https://arxiv.org/html/2305.20082), [arXiv:2507.16869](https://arxiv.org/html/2507.16869v1).
- **Bridge (conditioning generation on symbolic/graph plans):** [LLM-grounded Video Diffusion](https://llm-grounded-video-diffusion.github.io/), [arXiv:1804.01622](https://arxiv.org/abs/1804.01622) (Image Generation from Scene Graphs), [ar5iv:2309.15091](https://ar5iv.labs.arxiv.org/html/2309.15091), [arXiv:2506.03082](https://arxiv.org/html/2506.03082v1).
- **Benchmarks / evaluation:** [AGQA](https://cs.stanford.edu/people/ranjaykrishna/agqa/), [STAR](https://bobbywu.com/STAR/), [arXiv:2105.08276](https://arxiv.org/abs/2105.08276), [SQA3D](https://sqa3d.github.io/), [arXiv:1912.06992](https://arxiv.org/abs/1912.06992), [arXiv:2503.21755](https://arxiv.org/html/2503.21755v1).
- **End-to-end / frontier:** [arXiv:2402.13817](https://arxiv.org/abs/2402.13817), [arXiv:2404.13696v3](https://arxiv.org/html/2404.13696v3), [arXiv:2309.16650](https://arxiv.org/pdf/2309.16650), [arXiv:2507.16869](https://arxiv.org/pdf/2507.16869).

---

# Addendum — 2025–2026 frontier (recency pass, 2026-06-21)

**Source:** second deep-research scan, recency + generation/bridge focused, bumped budget (28 sources, 136 claims, **50 verified → 49 confirmed / 1 killed**). Prioritized 2024 H2–2026 work.

## Headline (revised)
The foundations pass made the field look near-empty at the intersection; it isn't — **2025–2026 is very active**, but it's **disjoint across three layers** that don't yet form one "query a reconstructed dynamic scene → generated video answer" system. Every component now exists in recent work; the loop is still not closed.

## Layer A — language/query-conditioned 4D *representations* (query/edit a reconstruction)
| Work | Venue / date | What it does |
|---|---|---|
| **4D LangSplat** | CVPR 2025 ([arXiv:2503.10437](https://arxiv.org/abs/2503.10437), Mar 2025) | Open-vocab, **time-sensitive** language field over dynamic Gaussians. Key move: **bypass CLIP**, supervise a time-varying field with **MLLM-generated object-wise video captions** encoded as LLM sentence embeddings. |
| **Mono4DEditor** | arXiv Oct 2025 ([2510.09438](https://arxiv.org/pdf/2510.09438), ICLR 2026 sub) | Text-driven 4D editing from **monocular** video; language-embedded dynamic Gaussians (quantized CLIP feats) + 2-stage point-level localization → diffusion video editor. |
| **Instruct-4DGS** | CVPR 2025 ([arXiv:2502.02091](https://arxiv.org/html/2502.02091v3), Feb 2025) | Instruction-driven 4D editing: edit only static canonical Gaussians (IP2P) + score distillation into the 4D field. |
→ These **query/edit** a reconstruction; they do not generate a novel view as the *answer*.

## Layer B — view/time-conditioned 4D *generation* (strong, but GEOMETRY-conditioned, not language/graph)
| Work | Venue / date | Control signal |
|---|---|---|
| **CAT4D** | CVPR 2025 ([arXiv:2411.18613](https://arxiv.org/abs/2411.18613)) | Multi-view video diffusion disentangling camera vs scene motion → deforming 3DGS. |
| **GEN3C** | CVPR 2025 Highlight, NVIDIA ([arXiv:2503.03751](https://arxiv.org/abs/2503.03751)) | Conditions video diffusion on a **spatiotemporal 3D point-cloud cache**; SOTA sparse-view NVS. |
| **VideoFrom3D** | SIGGRAPH Asia 2025 ([arXiv:2509.17985](https://arxiv.org/pdf/2509.17985)) | Coarse mesh + camera trajectory + style image → stylized scene video. |
| **SpaceTimePilot** | CVPR 2026 ([arXiv:2512.25075](https://arxiv.org/abs/2512.25075)) | Camera trajectory + animation time-embedding. |
| **VerseCrafter** | CVPR 2026 ([arXiv:2601.05138](https://arxiv.org/abs/2601.05138)) | **GeoAdapter** injects 4D control maps (bg point cloud + camera path + per-object Gaussian trajectories) into a frozen Wan2.1-14B. |
| **LiveWorld** | arXiv Mar 2026 ([2603.07145](https://arxiv.org/html/2603.07145v1)) | Persistent global state (static bg via Stream3R SLAM + dynamic "4D Monitor Point Clouds"); ControlNet state adapter (+ text prompt as 1 of 3 inputs). |
| **Vista4D** | CVPR 2026 Highlight ([arXiv:2604.21915](https://eyeline-labs.github.io/Vista4D/)) | Video "reshooting" from novel cameras using 4D point clouds. |
→ The control axis is **geometric** (point clouds, Gaussian trajectories, camera/time) — **not** an NL query or symbolic plan. (Caveat: several keep a *vestigial* text branch — Wan2.1 is text-conditioned; LiveWorld takes a text prompt — so not strictly text-free.)

## Layer C — symbolic/LLM-plan → generation *bridge* (real, but static or non-reconstructed)
| Work | Venue / date | Bridge |
|---|---|---|
| **DyST-XL** | arXiv Apr 2025 ([2504.15032](https://arxiv.org/abs/2504.15032)) | Training-free; LLM parses prompt → **entity-attribute graph** + physics-aware keyframe layouts → drives CogVideoX-5B via controlled attention. |
| **Agentic-VLM** | arXiv May 2025 ([2505.20129](https://arxiv.org/html/2505.20129v2)) | Scene **hypergraph** (unary/binary/higher-order relations) → VLM agentic Blender-code pipeline. **Static 3D.** |
| **SceneAssistant** | arXiv Mar 2026 ([2603.12238](https://arxiv.org/pdf/2603.12238)) | VLM-as-agent closed loop, atomic ops. **Static 3D.** |
| *(background)* LVD | ICLR 2024 ([2309.17444](https://arxiv.org/abs/2309.17444)) | LLM → dynamic scene layouts → video diffusion (training-free). |
| *(background)* VideoDirectorGPT | COLM 2024 ([2309.15091](https://j-min.io/publication/videodirectorgpt_colm2024/)) | GPT-4 "video plan" → Layout2Vid. |
| *(background)* SceneCraft | ICML 2024 ([2403.01248](https://arxiv.org/html/2403.01248v1)) | Relational scene graph → Blender code. **Static.** |

## Reasoning / scene-graph side (outputs TEXT or ROBOT ACTIONS, not views)
| Work | Venue / date | Output |
|---|---|---|
| **4D-LLM / 4D-PSG** | CVPR 2025 ([arXiv:2503.15019](https://arxiv.org/abs/2503.15019)) | 4D LLM + 3D mask decoder → **4D panoptic scene graphs**; chained SG inference for open-vocab. |
| **DyGEnc** | arXiv May 2025 ([2505.03581](https://arxiv.org/html/2505.03581v1)) | Encodes temporal **textual scene-graph sequences** → Llama 3 (LoRA) → text answers. |
| **DovSG** | RA-L 2025 ([arXiv:2410.11989](https://arxiv.org/abs/2410.11989)) | **Dynamic open-vocab 3D scene graph**, locally updatable (no full re-reconstruction) → GPT-4o decomposition → robot action. |

## The gap, sharpened — and the wedge
Every component exists, but the halves are **disjoint**: generation conditions on geometry; reasoning outputs text/actions; language-4D does querying/editing, not generation-as-answer. **No 2024–2026 system closes** query → temporal/4D scene graph + neurosymbolic reasoning → generated novel-view video answer.

**The missing piece is a graph→geometry compiler:** converting a 4D scene graph (4D-PSG / DyGEnc-style temporal graphs) into the explicit geometric control signals (point-cloud caches, per-object 3D-Gaussian trajectories, camera trajectories) that the 2026 world models (VerseCrafter / LiveWorld / GEN3C / Vista4D) already consume. **Nobody does this** — it is the clearest room4d wedge candidate. A second, lighter wedge: **language-condition the world model's control adapter** (VerseCrafter's GeoAdapter / LiveWorld's state adapter) instead of feeding only geometry, combined with 4D LangSplat's MLLM-caption open-vocab semantics.

**Evaluation gap (still open):** no 2024–2026 benchmark jointly scores *correctness w.r.t. the query* AND *fidelity to the reconstructed scene* of a generated answer (text-QA benchmarks score answers; PSNR/SSIM/LPIPS/TSED score pixels — neither scores both). Recent benchmark/eval candidates to dig into: [arXiv:2511.13704](https://arxiv.org/abs/2511.13704), [2601.00092](https://arxiv.org/pdf/2601.00092), [2602.08971](https://arxiv.org/html/2602.08971), [2603.00515](https://arxiv.org/pdf/2603.00515), [2512.05513](https://arxiv.org/pdf/2512.05513), [2503.11094](https://arxiv.org/html/2503.11094v4).

## Caveats (recency pass)
- **Forward-dated arXiv IDs** (2512.x, 2601.x, 2603.x, 2604.x) and CVPR 2026 venue tags appear; verifiers checked them against project pages / CVF proceedings, but they are very fresh — confirm IDs resolve before citing.
- **Mono4DEditor** and **SceneAssistant** are preprints under review (descriptive content solid; treat SOTA claims cautiously).
- The strong framing "the 2026 world models do NO language conditioning" was **refuted (1-2)** — they keep vestigial text branches; state the control axis as *primarily* geometric, not text-free.
- **LVD / VideoDirectorGPT / SceneCraft** are 2023/early-2024 — background, not latest SOTA.
