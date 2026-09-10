"""MoGe-2 depth rung: single-image metric point maps for the geometry ladder (FP6a).

MoGe-2 (Microsoft, MIT, ungated) predicts a **metric** point map + depth + per-frame
intrinsics from ONE image (`from moge.model.v2 import MoGeModel`). On the depth ladder
(design §4A.1) it sits below VGGT as a metric depth backstop — but, unlike VGGT's joint
reconstruction, it produces **no cross-frame camera poses**. So this rung fills the
`GeometryArtifact`'s *depth* channel (``depthmaps_by_frame`` + camera-space
``pointmaps_by_frame`` + per-frame ``intrinsics_by_stream``) and leaves
``camera_states_by_stream`` empty; world-consistent poses come from a pose source
(VGGT / GT / the FP6b COLMAP rung), not from here.

FP6a wires this as an **explicitly-selectable** rung and validates it as a *native swap*
against the GT-depth oracle (``scripts/fp6_validate_moge.py``): a direct depth-map
comparison that needs no poses. Auto-selection below VGGT is inert (VGGT/GT already emit
depth) and, with the pose/depth split + camera-convention/scale reconciliation, is
deferred to FP6b.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from reifyscene.common.types import CameraIntrinsics, FrameRef
from reifyscene.geometry.base import GeometryInitializer
from reifyscene.geometry.geometry_artifact import GeometryArtifact
from reifyscene.ingest.decoder import VideoDecoder

# Single source of truth for the model id. The setup precache
# (scripts/setup_reifyscene.sh :: download_moge_weights_if_needed) reads the same default
# via the MOGE_MODEL_ID env override, keeping install ↔ runtime in sync. Swap to a
# lighter variant (e.g. Ruicheng/moge-2-vitb-normal) via cfg['moge_model_id'] or the env.
DEFAULT_MODEL_ID = "Ruicheng/moge-2-vitl-normal"


class MoGeInitializer(GeometryInitializer):
    """Per-frame MoGe-2 metric depth / point-map rung (depth channel only; no poses)."""

    backend_name = "moge-2"

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or {}
        self.model_id = str(
            self.cfg.get("moge_model_id") or os.environ.get("MOGE_MODEL_ID") or DEFAULT_MODEL_ID
        )
        self.device = str(self.cfg.get("device", "cuda"))
        self.run_id = str(self.cfg.get("run_id", "moge"))
        self.save_dir = Path(self.cfg.get("geometry_save_dir", "artifacts/geometry_moge"))
        self.decoder = VideoDecoder(default_fps=float(self.cfg.get("default_fps", 30.0)))
        self._model = None

    # -- helpers -----------------------------------------------------------

    def _frame_key(self, frame: FrameRef) -> str:
        return f"{frame.stream_id}:{frame.frame_idx}"

    def _select_frames(self, frames, ingest_artifact) -> list[FrameRef]:
        if frames:
            return list(frames)
        if ingest_artifact is not None:
            return list(ingest_artifact.keyframes)
        return []

    def _load_model(self):
        """Lazily import + load MoGe-2. Kept out of ``__init__`` so constructing the
        rung is import-safe when the ``moge`` package is absent — the ladder's
        availability probe gates the actual load."""
        if self._model is None:
            from moge.model.v2 import MoGeModel  # noqa: PLC0415 — lazy, probe-gated
            self._model = MoGeModel.from_pretrained(self.model_id).to(self.device).eval()
        return self._model

    def _release(self) -> None:
        """Drop the model + free VRAM (staged execution: load → infer → release)."""
        if self._model is not None:
            self._model = None
            try:
                import torch  # noqa: PLC0415

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass

    def _intrinsics_from_moge(self, k_norm: np.ndarray, width: int, height: int) -> CameraIntrinsics:
        """MoGe returns **normalized** intrinsics (fx,fy relative to W,H; cx,cy≈0.5).
        Convert to pixel units for the reifyscene ``CameraIntrinsics`` contract."""
        k = np.asarray(k_norm, dtype=np.float64)
        return CameraIntrinsics(
            fx=float(k[0, 0]) * width,
            fy=float(k[1, 1]) * height,
            cx=float(k[0, 2]) * width,
            cy=float(k[1, 2]) * height,
        )

    def infer_frame(self, rgb: np.ndarray) -> dict[str, Any]:
        """Run MoGe-2 on one RGB frame — the shared inference core.

        Used by ``initialize`` (in-pipeline, video-decoded frames) and by the FP6a
        native-swap validation script (dataset PNGs), so both share one code path.

        Parameters
        ----------
        rgb : np.ndarray
            HxWx3 uint8 RGB image.

        Returns
        -------
        dict with ``depth`` (HxW float32, metric), ``points`` (HxWx3 float32,
        **camera-space** metric), ``mask`` (HxW float32 validity, or ``None``) and
        ``intrinsics`` (``CameraIntrinsics`` in pixels).
        """
        import torch  # noqa: PLC0415

        model = self._load_model()
        height, width = rgb.shape[:2]
        img = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .to(self.device)
        )
        with torch.no_grad():
            out = model.infer(img)
        depth = out["depth"].detach().float().cpu().numpy()
        points = out["points"].detach().float().cpu().numpy()
        mask = out["mask"].detach().cpu().numpy().astype(np.float32) if "mask" in out else None
        k = out["intrinsics"].detach().float().cpu().numpy()
        return {
            "depth": depth,
            "points": points,
            "mask": mask,
            "intrinsics": self._intrinsics_from_moge(k, width, height),
        }

    # -- artifact construction --------------------------------------------

    def initialize(self, frames=None, *, ingest_artifact=None, sync_artifact=None) -> GeometryArtifact:
        selected = self._select_frames(frames, ingest_artifact)
        if not selected:
            # empty-in → empty-out (§4A invariant 3): a schema-valid, empty artifact.
            return GeometryArtifact(
                run_id=self.run_id,
                stage_name="geometry",
                save_dir=self.save_dir,
                backend_name=self.backend_name,
                selected_frames=[],
                notes=["MoGe-2: no frames selected → empty geometry."],
            )
        if ingest_artifact is None:
            raise ValueError("MoGeInitializer requires ingest_artifact to decode frames.")

        stream_lookup = {stream.stream_id: stream for stream in ingest_artifact.streams}
        intrinsics_by_stream: dict[str, CameraIntrinsics] = {}
        depthmaps_by_frame: dict[str, Any] = {}
        pointmaps_by_frame: dict[str, Any] = {}
        selected_frames: list[FrameRef] = []

        try:
            for frame in selected:
                stream = stream_lookup.get(frame.stream_id)
                if stream is None:
                    continue
                rgb = self.decoder.get_frame(stream, frame.frame_idx)  # HxWx3 uint8
                res = self.infer_frame(rgb)
                key = self._frame_key(frame)
                depthmaps_by_frame[key] = {"depth": res["depth"], "confidence": res["mask"]}
                pointmaps_by_frame[key] = {"points": res["points"], "confidence": res["mask"]}
                intrinsics_by_stream.setdefault(frame.stream_id, res["intrinsics"])
                selected_frames.append(frame)
        finally:
            self._release()

        # MoGe provides NO cross-frame poses → camera_states_by_stream stays empty.
        # Poses come from a pose source (VGGT / GT / the FP6b COLMAP rung), not here.
        return GeometryArtifact(
            run_id=self.run_id,
            stage_name="geometry",
            save_dir=self.save_dir,
            backend_name=self.backend_name,
            selected_frames=selected_frames,
            intrinsics_by_stream=intrinsics_by_stream,
            depthmaps_by_frame=depthmaps_by_frame,
            pointmaps_by_frame=pointmaps_by_frame,
            notes=[
                f"MoGe-2 model={self.model_id}",
                f"frames={len(selected_frames)}",
                "depth-only rung: metric depth + camera-space pointmaps + per-frame "
                "intrinsics; no cross-frame poses (camera_states empty).",
            ],
        )
