# Benchmarks

Committed measurements, regenerated from runs on disk and diffed across changes. The
point is to stop developing against one clip: every constant in the perception band was
originally chosen on `davis_lab-coat`, and these files say whether it generalizes.

## `davis_inventory.json`

Does the graph contain the things that are actually there? Scored against DAVIS 2017
trainval (90 sequences, 6,208 frames), whose ground truth annotates instance masks on
every frame.

Rebuild the runs, then rescore:

    # ~5 hours on one GPU, resumable, ~45 GB under ../runs/inv_*
    gpu_guard.sh --unit inventory --log-dir DIR --detach -- \
        bash scan_clips_inventory.sh $(ls .../JPEGImages/480p)
    python scripts/benchmark_inventory.py

Each row carries the covariates a change has to be attributed against -- frame count,
annotated-object count, and every stage's capability manifest -- so a moved number can be
explained rather than only observed.

**`recall` is the trustworthy number. `spurious` is not a defect count.** DAVIS annotates
only the objects of interest, so a fence panel or a patch of foliage is legitimately
unannotated; a node matching no ground truth may be a real thing nobody labelled. It is
reported because it says something about how the proposer behaves, not because it is
wrong.

**Not re-run after G2 (2026-09-30), deliberately.** G4 of the graph plan asked for this
benchmark to be re-scored once edits worked. The inventory runs stop at the graph stage
(`--stop-stage graph`), and G2 changed nothing about which entities the graph holds: it added
the post-render `handles` stage and its `gaussians` layer, and the graph stage's only change
(contract v5) is its `renderable` grade reading `pending: gaussians layer`. A rebuild would be
~5 GPU-hours for the same rows. Whether an edit damages the rest of the scene is measured by
`davis_edit_damage.json` instead. Re-run this when anything upstream of the graph changes.

## `davis_edit_damage.json`

Does removing an entity take the entity, and leave the rest of the scene alone? Scored on the
ten `loop_*` runs against DAVIS ground truth: each annotated object is matched to a graph node
and removed with the selection the graph's `gaussians` layer holds (FlashSplat + floater
filter), on the checkpoint the app serves.

    python scripts/benchmark_edit_damage.py      # GPU, ~20 s for the ten runs

Metrics per object (`reifyscene/eval/benchmarks/edit_damage/score.py`): *taken recall /
precision* (pixels where the removed Gaussians carry over half the weight, against the GT
mask), *alone precision* (the removed Gaussians rendered by themselves), *halo ratio*,
*others damaged*, and *rest PSNR* (before vs after beyond 8 px of the object; 60 dB = untouched).

**Finding (2026-09-30), 31 of 39 objects scored, 8 never proposed:**

- **The rest of the scene is left alone.** Median rest PSNR 50.5 dB; other annotated objects
  lose a median 3% of their pixels. The exceptions are entanglement, not stray edits:
  motocross (the rider shares the motorcycle's node: 40% of the rider changes) and dogs-scale
  (29–41 dB; people and dogs overlap all clip).
- **What is taken is the object, where it is visible.** Median taken precision 0.93.
- **But removal is incomplete: median taken recall 0.39.** Under half of an object's pixels are
  dominated by its own Gaussians. Worst on small, fast movers (the crossing pedestrians: 0.02,
  ~600-900 Gaussians each) -- the motion model renders them with Gaussians that belong to
  nothing or to the background, the same softness the identity scope's §8 records.
- **Most of what a removal takes is hidden: median alone precision 0.38.** Rendered by
  themselves, the removed Gaussians land on the object only 38% of the time; the rest sits
  behind or around it, invisible from the training cameras. This is the "shadow" the 3D took
  view shows behind the lab-coat people, measured.

These are training views: the fitted views, which never see behind anything. The held-out
measure is DyCheck (G4 part 3), below.

## `dycheck_spin_heldout_edit.json`

The same questions from cameras the scene never trained on. DyCheck `spin` has held-out cameras
at other positions with real photos; the person is removed (graph `gaussians` layer) and judged
on 60 held-out and 20 training views.

    python scripts/heldout_edit_damage.py ../runs/spin_oracle_g4h2   # GPU, ~20 s

Where the person *is* in each view comes from the photo, not from the removal: SAM 2 prompted
by the person detector segments every person, and the one overlapping where the entity renders
is its silhouette (`reifyscene/eval/benchmarks/edit_damage/silhouettes.py`). These are model
masks -- pseudo ground truth -- saved under the run's `edits/heldout_damage/masks/` to be
checked. (A first version judged "the rest" by the removal's own footprint, which cannot fail;
it was replaced before being recorded.)

**Finding (2026-09-30):**

| | held-out (60) | training (20) |
|---|---|---|
| taken recall | 0.79 | 0.85 |
| alone precision | 0.98 | 0.97 |
| rest damage vs photo | -0.02 dB | +0.01 dB |

- **No shadow when the camera went around.** Rendered by themselves, 98% of the removed
  Gaussians land on the person even from the held-out cameras. On lab-coat, seen from one side,
  the same number is 0.44-0.46. What a removal carries along depends on the entity's view
  coverage -- the removal-quality track's first thing to detect.
- **The rest of the scene is untouched**, against real photos from unseen viewpoints.
- **What remains is a ghost:** ~21% of the person's pixels are not mostly drawn by the removed
  Gaussians -- person Gaussians the assignment missed, over a hole nothing was reconstructed
  behind. An assignment and inpainting problem, not damage.

## `amg_keyframe_cache.json`

Every mask SAM 2's automatic generator proposes on each sequence's first frame, reduced to
the four quantities the mask filter tests plus each proposal's IoU against each
ground-truth object. No masks are stored -- once those numbers exist, any filter setting
scores in milliseconds.

    python scripts/sweep_mask_filter.py            # rebuild the cache (GPU, ~10 min)
    python scripts/sweep_mask_filter.py --sweep    # score settings against it

**Finding (2026-09-16):** the filter thresholds are not where recall is hiding. Across 36
settings, kept proposals match between 85 and 89 of 205 objects; the shipped setting is
85. Admitting a koala at 58% of frame gains one object.

**What it cannot tell you:** the pipeline's propagated masks match 141 of 205 on the same
frame, because SAM 2's video predictor re-segments from a seed rather than keeping the
proposal. Proposal quality is a lower bound on what the inventory achieves, not a ceiling.

## `detector_keyframe_cache.json`

Every box an open-vocabulary detector (GroundingDINO) proposes on each sequence's first
frame, reduced to its IoU against each ground-truth object. Same trick as the mask cache:
the detector depends on nothing downstream, so it runs once and any question about
coverage is then arithmetic.

    python scripts/probe_detector_coverage.py            # build the cache (GPU, ~1 min)
    python scripts/probe_detector_coverage.py --report

**Finding (2026-09-16):** 45 of the 63 objects the inventory misses have *no* SAM 2
proposal overlapping them at all. GroundingDINO covers **31 of those 45** with a blind
COCO-80 list, and **26 of 45** with a generic list whose most specific noun is "animal".
Across all objects it covers 173 of 205, against 116 for SAM 2's proposals counted at
IoU 0.01.

So the bottleneck is evidence *acquisition*, not evidence *combination*: a detector
trained to localise objects proposes things a class-agnostic mask generator does not. The
vocabularies are fixed and applied to all ninety sequences unseen -- a list written after
looking at the clips proves nothing a pipeline can use.
