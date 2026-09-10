"""DensePose wrapper (swappable seam).

Dense human surface-correspondence estimator. No real DensePose backend is wired
yet, so this wrapper honestly reports ``unavailable`` rather than fabricating
correspondences (the prior heuristic bucketed bbox pixels into fake body parts and
mislabelled them). Sapiens body-part segmentation (FP5a, ``sapiens_wrapper.py``)
is the primary dense source now; a real DensePose model can still swap in behind
this seam without changing the contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class DensePoseEstimator:
    """Predict dense human surface correspondences (real backend or honest none)."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self.backend_mode = str(self.cfg.get('densepose_backend_mode', self.cfg.get('backend_mode', 'auto')))
        self._backend_used = 'unavailable'

    @property
    def backend_used(self) -> str:
        return self._backend_used

    def predict(
        self,
        frame: np.ndarray,
        boxes: list[np.ndarray],
        masks: list[np.ndarray | None] | None = None,
        keypoints: list[np.ndarray | None] | None = None,
    ) -> list[dict[str, Any] | None]:
        # No real DensePose backend wired yet -> honest None per box (never
        # fabricate surface correspondences). Swap a real model in here.
        self._backend_used = 'unavailable'
        return [None] * len(boxes)
