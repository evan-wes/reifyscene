"""``dyn_mask`` ladder rungs (FP6b step 3, design §4A.1).

Geometry-free coarse dynamic masks that protect the SfM pose rung. Ladder (best → floor):

    optical-flow motion seg  (deferred — general north-star backend)
    → detector               (this file: COCO Mask R-CNN, movable categories)
    → all-static             (this file: floor — no-op, makes VGGT-on-top a no-op)

The detector rung reuses the already-installed torchvision Mask R-CNN (via
``ObjectDetector``) and unions instance masks for **movable** COCO categories — enough to
cover the large-dynamic regime the spike showed masking helps (a person filling the frame).
The general optical-flow motion segmenter (catches arbitrary non-COCO movers) is a deferred
follow-up; no current test clip needs it (native-swap discipline). An *oracle* dynamic mask
(dataset Track-Anything) is built directly by the validator, not wired as a pipeline rung.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from reifyscene.common.frame_keys import frame_key
from reifyscene.common.types import FrameRef
from reifyscene.ingest.decoder import VideoDecoder
from reifyscene.perception.dyn_mask.dyn_mask_artifact import DynMaskArtifact

# COCO categories that can move under their own power — the coarse "dynamic" superset.
MOVABLE_COCO = frozenset({
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
})


class DynMaskInitializer:
    """Base: decode frames, mask each, assemble a :class:`DynMaskArtifact`."""

    backend_name = "all-static"

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or {}
        self.run_id = str(self.cfg.get("run_id", "dyn_mask"))
        self.save_dir = self.cfg.get("dyn_mask_save_dir", "artifacts/dyn_mask")
        self.decoder = VideoDecoder(default_fps=float(self.cfg.get("default_fps", 30.0)))

    @staticmethod
    def _frame_key(frame: FrameRef) -> str:
        return frame_key(frame.stream_id, frame.frame_idx)

    def _select_frames(self, frames, ingest_artifact) -> list[FrameRef]:
        if frames:
            return list(frames)
        if ingest_artifact is not None:
            return list(ingest_artifact.keyframes)
        return []

    def mask_frame(self, rgb: np.ndarray) -> tuple[np.ndarray, float]:
        """Return ``(HxW uint8 dynamic mask, dynamic fraction)`` for one RGB frame.

        Base = all-static: an all-zero mask (no dynamic pixels). The reusable per-frame
        core so the pipeline path and the validator share one code path.
        """
        h, w = rgb.shape[:2]
        return np.zeros((h, w), dtype=np.uint8), 0.0

    def _release(self) -> None:  # overridden by model-backed rungs
        pass

    def initialize(self, frames=None, *, ingest_artifact=None, sync_artifact=None) -> DynMaskArtifact:
        selected = self._select_frames(frames, ingest_artifact)
        # all-static floor short-circuits: emit an empty-mask artifact (a missing mask ==
        # static downstream), no decode. Model rungs decode + mask each frame.
        if self.backend_name == "all-static" or not selected:
            return DynMaskArtifact(
                run_id=self.run_id, stage_name="dyn_mask", save_dir=self.save_dir,
                backend_name=self.backend_name, selected_frames=list(selected),
                notes=[f"dyn_mask {self.backend_name}: all-static (no dynamic pixels)."],
            )
        if ingest_artifact is None:
            raise ValueError(f"{type(self).__name__} requires ingest_artifact to decode frames.")

        stream_lookup = {s.stream_id: s for s in ingest_artifact.streams}
        masks_by_frame: dict[str, np.ndarray] = {}
        frac_by_frame: dict[str, float] = {}
        used: list[FrameRef] = []
        try:
            for frame in selected:
                stream = stream_lookup.get(frame.stream_id)
                if stream is None:
                    continue
                rgb = self.decoder.get_frame(stream, frame.frame_idx)
                mask, frac = self.mask_frame(rgb)
                key = self._frame_key(frame)
                masks_by_frame[key] = mask
                frac_by_frame[key] = frac
                used.append(frame)
        finally:
            self._release()

        agg = float(np.median(list(frac_by_frame.values()))) if frac_by_frame else 0.0
        return DynMaskArtifact(
            run_id=self.run_id, stage_name="dyn_mask", save_dir=self.save_dir,
            backend_name=self.backend_name, selected_frames=used,
            masks_by_frame=masks_by_frame, dynamic_fraction_by_frame=frac_by_frame,
            notes=[
                f"dyn_mask {self.backend_name}: {len(used)} frames, "
                f"median dynamic fraction {agg:.3f}.",
            ],
        )


class AllStaticDynMask(DynMaskInitializer):
    """Floor rung — every frame all-static (no-op; VGGT-on-top ignores it)."""

    backend_name = "all-static"


class DetectorDynMask(DynMaskInitializer):
    """Movable-category COCO Mask R-CNN masks (geometry-free, reuses ``ObjectDetector``)."""

    backend_name = "detector"

    def __init__(self, cfg: dict | None = None) -> None:
        super().__init__(cfg)
        self.movable = frozenset(self.cfg.get("movable_categories", MOVABLE_COCO))
        self.score_threshold = float(self.cfg.get("score_threshold", 0.5))
        self._detector = None

    def _load_detector(self):
        if self._detector is None:
            from reifyscene.perception.objects.detector import ObjectDetector  # noqa: PLC0415

            self._detector = ObjectDetector({
                "backend_mode": self.cfg.get("backend_mode", "real"),
                "device": self.cfg.get("device", "cuda"),
                "score_threshold": self.score_threshold,
                "max_detections": int(self.cfg.get("max_detections", 24)),
            })
        return self._detector

    def _release(self) -> None:
        self._detector = None
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass

    def mask_frame(self, rgb: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        detections: list[dict[str, Any]] = self._load_detector().detect(rgb)
        for det in detections:
            if det.get("category") not in self.movable:
                continue
            logits = det.get("mask_logits")
            if logits is None:
                continue
            m = np.asarray(logits)
            m = m.reshape(m.shape[-2], m.shape[-1]) if m.ndim >= 2 else m
            if m.shape != (h, w):  # safety: align to the frame grid
                continue
            mask |= (m > 0.0).astype(np.uint8)  # sigmoid(logit) > 0.5
        frac = float(mask.mean())
        return mask, frac
