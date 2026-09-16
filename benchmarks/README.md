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
