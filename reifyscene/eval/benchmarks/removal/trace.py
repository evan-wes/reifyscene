"""Is the entity still visible after its removal (test T1)?

On each view, the detector looks for the entity's category in the **before**-render and picks
the silhouette that best overlaps where the entity is (``hint``: its id-map mask on a training
view, where the removed Gaussians render on an orbited one). A view whose before-render shows
no such silhouette (IoU below ``min_iou``, or a score below ``before_score``) cannot be judged
and is not counted: the detector would not have found the entity even with it there.

On a judged view, the after-render has a **trace** when the detector finds the category there
at ``after_score`` or more, overlapping the before-silhouette by ``min_iou`` or more. The
after threshold is lower than the before one on purpose: a faint ghost is still a trace. Only
the entity's own region counts, so a neighbour of the same category is not a trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from reifyscene.eval.benchmarks.edit_damage.silhouettes import Silhouette, pick_entity

MIN_IOU = 0.3
BEFORE_SCORE = 0.5
AFTER_SCORE = 0.3


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return int((a & b).sum()) / union if union else 0.0


@dataclass
class TraceCount:
    judged: list[str] = field(default_factory=list)      # view names the test could judge
    traces: list[str] = field(default_factory=list)      # judged views with a trace
    unjudged: int = 0

    @property
    def clean(self) -> float | None:
        """Share of judged views without a trace; None when none could be judged."""
        return round(1.0 - len(self.traces) / len(self.judged), 4) if self.judged else None

    def to_dict(self) -> dict:
        return {"clean": self.clean, "judged": len(self.judged), "traces": len(self.traces),
                "unjudged": self.unjudged, "trace_views": list(self.traces)}


def entity_region(found: list[Silhouette], hint: np.ndarray, *, min_iou: float = MIN_IOU,
                  before_score: float = BEFORE_SCORE) -> np.ndarray | None:
    """The entity's silhouette among ``found`` (the category's silhouettes in the
    before-render), or None: the view cannot be judged."""
    region, iou = pick_entity([s for s in found if s.score >= before_score], hint)
    return None if region is None or iou < min_iou else region.mask


def has_trace(found: list[Silhouette], region: np.ndarray, *, min_iou: float = MIN_IOU,
              after_score: float = AFTER_SCORE) -> bool:
    """Whether ``found`` (the category's silhouettes in the after-render, from a segmenter
    whose own threshold is at most ``after_score``) still shows the entity over ``region``."""
    return any(s.score >= after_score and _iou(s.mask, region) >= min_iou for s in found)
