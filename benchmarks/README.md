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
