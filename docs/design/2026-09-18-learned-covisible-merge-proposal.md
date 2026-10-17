# Learned co-visible merge — a small transformer, built by hand

**Status:** PROPOSED
**Date:** 2026-09-18
**Goal:** a complete in-house implementation — architecture, dataloader, training script,
evaluation script — of a transformer that closes a measured gap in this pipeline. Written here,
not imported. Not state of the art, and not required to beat anything.
**Adds, does not replace:** a rung above `reifyscene/perception/reid/covisible.py`
(`build_covisible_merge_map`)
**Phase 2, conditional on §8:** spatial relations (§9)
**Informed by:** `docs/design/2026-09-20-conceptgraphs-verdict-and-graph-plan.md` §3.1

---

## 0. What this is, and the bar it has to clear

A transformer written from the tensor ops up — attention, encoder, dataloader, training loop,
evaluation harness — solving a real problem in this pipeline rather than a toy one. Success is
*built, trained, evaluated honestly, integrated*. A model that ties the rule it sits above is a
completed piece of work; a model evaluated on data it trained on is not, whatever its score. So
the evaluation discipline in §5 is non-negotiable even though the model is small.

**This repo is a portfolio piece, and that changes the requirements, not just the framing.** It
is meant to show two things at once: components implemented in-house, and integrations with
state-of-the-art third-party tools, in one system that does something real. The ConceptGraphs
comparison is the second kind; this is the first. Concretely that means the implementation must
be *legible to a reader who did not write it*: its own module with a README, one command to
reproduce the table, tests that pin the properties that matter, and no dead code left behind.
The same standard as the rest of the pipeline — it just happens to be trained rather than tuned.

This problem was picked over the alternatives for three reasons, all measured in §1:

- **The gap is real and quantified.** The current rule misses 87% of the pairs it exists to merge.
- **The labels are already on disk.** No reruns, no new dataset, no annotation work.
- **It is transformer-shaped.** The evidence is a variable-length sequence, and how to pool it
  is exactly the thing the rule hard-codes and a model could learn.

### 0.1 What "in-house" means here, and where the line is

| written here | used as a dependency |
|---|---|
| multi-head self-attention, encoder block, positional encoding, pooling | `torch.Tensor`, autograd, `nn.Linear`, `nn.LayerNorm`, optimisers |
| the dataset, batching, padding, fold assignment | `numpy`, the repo's existing artifact loaders |
| the training loop, checkpointing, early stopping | — no Lightning, no HF `Trainer`, no `nn.TransformerEncoder` |
| the evaluation harness, PR curves, operating-point selection | `sklearn` for metric primitives only, if at all |

`nn.MultiheadAttention` is not used in the model — but it *is* used in the tests, as the oracle:
our attention is given the same weights and must match its output. That is the difference between
claiming an implementation is correct and showing it.

---

## 1. The gap, measured

Two tracklets are *co-visible* when both are placed on at least three of the same frames. The
co-visible merge decides which such pairs are one physical thing. Counted across all 90 DAVIS
inventory runs, labelled against DAVIS ground truth by the containment rule in §4:

| | count |
|---|---|
| tracklets (pre-merge) | 1,367 — equals the benchmark's 142 recovered + 1,225 spurious |
| co-visible pairs (≥ 3 shared frames) | 11,704 |
| labelled **same object** (positive) | 247 |
| labelled **different objects** (negative) | 3,351 |
| unlabelled (both background, or ambiguous) | 8,106 (69%) |
| sequences with ≥ 1 positive | 47 / 90 |

Scored against those labels, the pipeline's own merges (read from each run's `reid` merge map):

| | merged | of | |
|---|---|---|---|
| positives | **31** | 247 | recall **12.6%** |
| negatives | **0** | 3,351 | precision **1.00** |

**Reading it.** The rule was built for one case — duplicate masks, like the `lab-coat` pair at
IoU 0.996 whose 3D centroids sat 0.2% of their own size apart — and it does that job without a
single false merge. It cannot see the other 216. Those are mostly over-segmented **parts** of one
object: a jacket inside a person, a wheel on a car. A part's centroid sits off the whole's by
more than a quarter of their size, so a centroid-gap threshold is structurally blind to it, and
no retuning fixes that — the feature is wrong, not the constant.

**The asymmetry that sets the operating point.** A false merge *destroys* an entity: two people
become one node, and every downstream query about either is wrong. A missed merge leaves a
spurious node, which is noise. The rule's precision of 1.00 is therefore a bar to protect, not
a free parameter. The headline metric is **recall at precision ≥ 0.98**, not accuracy or F1 —
and choosing that operating point deliberately is part of the exercise.

Numbers produced by a one-off counting script kept outside the repository, over the same runs
`benchmarks/davis_inventory.json` scores (stamped `d0bf1c4`).

---

## 2. The task, precisely

- **Input:** one co-visible pair of tracklets (A, B) and the frames both are placed on.
- **Output:** P(A and B are the same physical object).
- **Then:** threshold, and union-find exactly as the rule does today, so the learned rung
  returns the same shape — `{track_id: canonical_id}` — and composes with the appearance merge
  through `ReidArtifact.canonical` unchanged.

Symmetry is a requirement: P(A, B) must equal P(B, A). Either every feature is symmetric, or it
comes as a (A→B, B→A) pair and training swaps the order at random. Getting this wrong produces a
model whose answer depends on which tracklet had the smaller id.

---

## 3. Why a transformer, and the ablation that keeps it honest

The evidence for a pair is a **sequence** — one observation per shared frame, three to about a
hundred of them. The rule collapses it with a median of one feature. That is a hard-coded pooling
choice over a hard-coded feature. A transformer encoder over the per-frame tokens learns both:
which features matter, and how to weigh frames against each other (a frame where one mask is
half-occluded should count for less than one where both are whole).

It is also free to use evidence the rule ignores. Mask **containment** in particular — the
fraction of A inside B and of B inside A — is close to the definition of "part of", and the rule
never looks at it.

**The bar is now higher than our own rule.** ConceptGraphs associates objects with a linear
blend of appearance cosine and 3D overlap; a learned merger that cannot beat two terms and a
threshold has not earned its place. That baseline is M1.

**The ablation that decides whether the architecture earned its place:** the same features,
mean- and max-pooled over frames, into a small MLP. If that ties the transformer, then the order
and interaction of frames did not matter, and the honest conclusion is that a pooled classifier
suffices. That result is reported as a finding, not buried. It is the single most instructive
experiment in this plan: it is how you learn whether attention was doing anything.

---

## 4. Data — on disk, no reruns

**Sources**, all present for all 90 runs:
- `segment/` — per-frame masks under **pre-merge** track ids (merges live in `reid/`)
- `geometry/` — world pointmaps and the recorded `ImageTransform` (masks go through it into the
  geometry's pixels — never guessed from sizes)
- DAVIS `Annotations/480p/<seq>/` — ground-truth instance masks, every frame

**Labels, by containment rather than IoU.** Per frame, a tracklet *belongs to* ground-truth
object *g* if ≥ 80% of its pixels lie inside *g*; it is *background* if < 20% lie inside any
annotated object; otherwise it is ambiguous on that frame. A tracklet's label is its majority
over labelled frames. A pair is **positive** if both belong to the same object, **negative** if
they belong to different objects or one is an object and the other background, and
**unlabelled** otherwise. Unlabelled pairs are excluded from training — never guessed.

Containment, not IoU, because a jacket's mask has low IoU with the person it is part of and is
still part of them.

**Optional extra supervision (recommended):** a pair whose masks have mutual IoU > 0.95 on
*every* shared frame is positive regardless of ground truth. A duplicate mask is a duplicate by
definition; this labels the background duplicates (the `lab-coat` fence panels) without
imitating the rule, because it uses a mask property the rule does not.

**Per-frame features (one token per shared frame).** All scale-free — a monocular reconstruction
is fixed only up to a similarity, so an absolute distance is a constant tuned to one clip, the
defect this pipeline keeps finding in itself:

| feature | why |
|---|---|
| 3D centroid gap ÷ mean extent | the rule's own feature; keeps it available to beat |
| **appearance cosine (DINOv2)** | **the signal the current rule cannot see at all** — it uses appearance only for tracklets that never co-occur, i.e. never for these pairs. Adopted from ConceptGraphs, which blends appearance and geometry in one score (see the 2026-09-20 verdict §3.1) |
| log(extent A ÷ extent B) | a part is smaller than its whole |
| mask IoU | duplicates |
| containment A-in-B, B-in-A | parts |
| 2D centroid gap ÷ √(mean area) | image-space agreement |
| log(median depth A ÷ median depth B) | depth order; a part sits at its whole's depth |
| erosion survival A, B | how much of each mask survives erosion — a proxy for mask quality |
| normalised time in the pair's shared span | position, for the encoder |

**Cache the table.** `scripts/extract_covisible_pairs.py` runs once over the 90 runs and writes
one `.npz` of tokens, labels, sequence ids and pair ids, stamped with the commit it was extracted
at — the same pattern as `benchmarks/amg_keyframe_cache.json`. Training never touches the runs.

**Audit before training.** Look at ~30 labelled pairs by eye. The known noise source: an
*unannotated* held object — a phone — lies inside the person's ground-truth mask and gets
labelled "part of the person". Whether that is noise or correct depends on what the graph wants
parts to be (§9, last bullet).

---

## 5. Splits — don't burn the instrument

`benchmarks/davis_inventory.json` covers all 90 sequences. Training on them and then comparing on
them would destroy the only baseline this project has.

- **5-fold cross-validation by sequence.** Each sequence is scored only by the model that never
  saw it, so the aggregate over all 90 is directly comparable to the committed benchmark.
- **A sequence's pairs never straddle folds.** Pairs within a sequence are correlated; splitting
  them leaks.
- **Stratify folds by positive count.** 54% of positives live in six sequences (`color-run` 30,
  `planes-water` 28, `car-roundabout` 21, `car-shadow` 21, `tuk-tuk` 17, `kid-football` 16).
  Random folds could leave one with almost none.
- **Hyperparameters on an inner validation split** within each training fold, never the test fold.
- **Weight by sequence** so `color-run`'s 210 co-visible pairs do not dominate a model meant to
  generalise across 90 clips.

---

## 6. Model

About 50K parameters. Trains in seconds to minutes; a CPU is enough.

```
per-frame tokens (T × F)
  → Linear(F → 64) + sinusoidal encoding of normalised time
  → prepend a learned [CLS] token
  → 2 × pre-norm encoder layer (4 heads, d_model 64, FFN 128, dropout 0.1),
    with a key-padding mask for variable T
  → [CLS] output → MLP(64 → 32 → 1) → logit
```

- **Loss:** binary cross-entropy with `pos_weight` ≈ 13.6 (the 3,351 : 247 class ratio), plus
  per-sequence sample weights (§5).
- **Augmentation:** swap (A, B) (§2); randomly drop frames, so the model cannot count on seeing
  every frame; jitter the scale-free features slightly.
- **Symmetric features by construction** where possible (IoU, |gap|), paired otherwise.

---

## 7. Milestones — what each produces, and how you know it worked

| | build | why | done when |
|---|---|---|---|
| **M0** | extract the table; audit 30 pairs by eye | the data is most of the work | counts reproduce §1 exactly |
| **M1** | baselines: the rule, ConceptGraphs' `sim_sum` (appearance + geometry, linearly blended), logistic regression, pooled MLP | what the bar is | the rule, re-run on the table, agrees with the pipeline's own merges (31 / 247, 0 / 3,351) — or every disagreement is explained (the pipeline's map also holds appearance merges, which can chain) |
| **M2** | the model: multi-head self-attention, encoder block, positional encoding, pooling head — Q/K/V projections, scaled dot-product, padding mask, head split and merge | the architecture, written not imported | given the same weights, our attention matches `nn.MultiheadAttention` to ~1e-6, and the encoder block matches `nn.TransformerEncoderLayer` |
| **M3** | the dataloader and training script: padding, batching, class weighting, checkpointing. **First overfit one batch of 32 to ≈ zero loss**, then train properly | the training loop, masking, imbalance | the one-batch overfit succeeds (a model that cannot memorise 32 examples has a bug, not a capacity problem) |
| **M4** | cross-validated PR curves for all four models on held-out folds; recall at precision ≥ 0.98; merge rate on unlabelled pairs | honest evaluation, operating points | one table, every number held-out |
| **M5** | only if M4 wins: integrate as a rung and re-score the inventory offline, each sequence merged by its own fold's model | shipping a learned component into a real system | spurious falls, recovered does not |

M2 and M3 are the substance. The equality test against PyTorch's layer is what turns "I wrote an
attention block" into something demonstrable — it is the first thing a reader of this repo should
be able to run.

**Integration shape (M5).** The co-visible merge becomes a ladder: `learned_transformer` above
`geometric_threshold` (today's rule). The learned rung raises if its checkpoint is absent and the
ladder falls to the rule; the `CapabilityManifest` records which ran. A rung, not a replacement —
the same shape as every other stage.

---

## 8. What "works out ok" means — the gate to phase 2

**Proposed:** M0–M4 complete with a trustworthy, fully held-out comparison. Beating the rule is
**not** required. A clean negative — the pooled MLP ties the transformer, or nothing beats the
rule at precision ≥ 0.98 — completes the exercise and still opens phase 2. M5 happens only on a
win, because a rung that loses should not ship.

---

## 9. Phase 2 — spatial relations (conditional on §8)

**What transfers:** the whole skeleton — tokens, encoder, head, the extract-and-cache pattern,
cross-validation by sequence. Phase 2 is phase 1's machinery pointed at a different question.

**The label problem, stated plainly.** Our relation vocabulary is VLA-3D's eight predicates, and
VLA-3D computes them *by heuristics over bounding boxes*. A model trained on VLA-3D learns to
imitate those heuristics. That is fine for experience, and it should be said up front: such a
model will not know anything the rules do not.

**Where learning can genuinely add something:**
- **Relations over time on our tracks** — "was near the chair before it moved". The evidence is a
  sequence of per-frame geometry, and a per-frame rule has to hard-code how to aggregate it — the
  same structure as phase 1.
- **Predicates that are ill-posed as rules on noisy monocular geometry** — `on` and `in`,
  support and containment, computed from pointmaps with a 0.3–14% pinhole residual (measured in
  the ConceptGraphs export, median 1.3%).

**Evaluation needs our own labels.** Agreement with VLA-3D says nothing about DAVIS. A small
hand-labelled set — roughly 150 object pairs across 15 sequences — is the test set, and labelling
it is part of the exercise.

**A seam worth noticing.** Phase 1's positives split cleanly into *duplicates* (mutual IoU high)
and *parts* (one contained in the other, much smaller). A three-way head — same / part-of /
distinct — would make phase 1 emit the graph's first learned relation, `part_of`, and would also
settle the held-object question from §4. Out of phase 1's scope; noted because it is the natural
bridge.

**A correction to carry.** `docs/research/2026-09-12-third-party-components-for-the-back-half.md`
§2.2 still says VLA-3D *deliberately excludes* left/right *because* viewpoint-dependent relations
are ambiguous. That rationale was checked earlier and found invented — VLA-3D lists
view-dependent relations as future work. The conclusion drawn there (view-dependent relations are
a query-time concern) stands on our own reasoning; the attribution does not.

---

## 10. Risks

- **Few, concentrated positives.** 247, over half in six sequences. Mitigated by stratified
  folds and per-sequence weights; not eliminated.
- **Label noise** from the containment rule (held objects, §4).
- **69% of co-visible pairs are unlabelled**, and the model will run on them at inference having
  never trained on one. M4 reports its merge rate there against the rule's, and every merge it
  makes there is looked at.
- **Transitivity amplifies errors.** One false pair merge can chain two groups together through
  union-find. Evaluate at group level as well as pair level.
- **Upstream drift.** Features depend on `segment` and `geometry` outputs; a change there (the
  person-seed defaults, say) invalidates the cached table. The commit stamp makes that visible.

---

## 11. Where it lives

```
reifyscene/learned/                  in-house trained components (this is the first)
  README.md                          what lives here, and the standard it holds to
  nn/                                the building blocks, reusable by later models
    attention.py                     multi-head self-attention
    encoder.py                       pre-norm encoder block, positional encoding
  covisible/
    features.py                      per-frame token features (scale-free)
    dataset.py                       cached table → padded batches, fold assignment
    model.py                         the pair classifier
    train.py                         training loop, cross-validation, checkpointing
    evaluate.py                      PR curves, operating point, the results table
scripts/extract_covisible_pairs.py   runs once over the 90 runs → cached .npz
scripts/train_covisible_merge.py     one command, reproduces the table in the README
tests/test_learned_nn.py             our attention/encoder == PyTorch's, same weights
tests/test_learned_covisible.py      features, dataset, folds, symmetry
```

`reifyscene/learned/nn/` is deliberately separate from `covisible/`: the blocks are the part a
second model reuses, and phase 2 (§9) is the first test of whether they were written generally
enough. That split is also what makes the in-house work legible as a *library* rather than one
script.

Tests that pin the properties which matter: our attention and encoder equal PyTorch's given the
same weights; features unchanged under a random Sim(3) of the scene; P(A, B) = P(B, A).

The checkpoint is ~200 KB. Whether it is committed with a benchmark or regenerated from the
cached table is an open question below.

---

## 12. Decisions for you

1. **The gate** in §8 — does a clean negative open phase 2?
2. **The duplicate label** in §4 — include mutual-IoU > 0.95 pairs as positives? Recommended.
3. **The checkpoint** — commit it beside a benchmark, or regenerate from the cached table?
4. ~~**`reifyscene/learned/nn/` as a shared home**~~ — **settled 2026-09-19.** Two further
   in-house components are planned (`docs/design/2026-09-19-roadmap-inhouse-components-and-modalities.md`),
   and one of them — the audio-visual association model — is the same sequence-alignment shape as
   this one. `learned/nn/` is written to be reused, and that component is the test of whether it
   was written generally enough.
