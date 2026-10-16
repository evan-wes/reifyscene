"""Score a run's entity inventory against a dataset's ground-truth masks.

The graph claims a scene contains a particular set of things. That claim is checkable
wherever the dataset annotates instances, and DAVIS annotates every frame of all ninety
sequences -- so "did we find what is there" stops being a matter of looking at one clip
and becomes a number over an ensemble.

Three quantities, per sequence:

- **recovered**: ground-truth objects matched by some entity node at IoU >= ``match_iou``.
  What the inventory got right.
- **missed**: ground-truth objects nothing matched. On davis_lab-coat before the person
  detector was allowed to seed, this was the largest annotated object in the frame.
- **spurious**: entity nodes matching no ground-truth object. Not automatically wrong --
  DAVIS annotates only the objects of interest, so fence panels and foliage are
  legitimately unannotated -- which is why this is reported rather than penalised.

Matching is one-to-one and greedy by descending IoU, so two nodes covering one object
count once and the duplicate lands in ``spurious``: over-segmentation shows up as noise
rather than as a free extra match.
"""

from reifyscene.eval.benchmarks.inventory.score import (
    SequenceScore,
    score_run,
    score_sequence_masks,
)

__all__ = ["SequenceScore", "score_run", "score_sequence_masks"]
