"""Score removals against ground-truth masks: what was taken, and whether the rest is intact.

See :mod:`reifyscene.eval.benchmarks.edit_damage.score`.
"""

from reifyscene.eval.benchmarks.edit_damage.score import (
    METRICS,
    change_score,
    changed,
    dilate,
    fit,
    gt_masks,
    match_objects,
    rest_psnr,
    score_run,
    summarize,
    taken_score,
)

__all__ = ["METRICS", "change_score", "changed", "dilate", "fit", "gt_masks", "match_objects",
           "rest_psnr", "score_run", "summarize", "taken_score"]
