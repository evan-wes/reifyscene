"""COLMAP-SfM pose rung: monocular camera egomotion for the geometry ladder (FP6b).

The **pose** mirror of the FP6a MoGe depth rung (design §4A.1): where MoGe gives
metric depth but no poses, COLMAP gives camera egomotion but no dense depth. It runs
reifyscene's calibration-arc SfM stack (`HlocColmapSfMRunner` — SuperPoint + SuperGlue +
NetVLAD) over **dense** frames with **fixed intrinsics**, then fills the
`GeometryArtifact`'s ``camera_states_by_stream`` (poses) + ``intrinsics_by_stream`` and
leaves ``depthmaps``/``pointmaps`` empty (depth comes from MoGe on the depth ladder).

The FP6b spike (`spike_fp6b_egomotion/`, 2026-07-01) validated this on spin+windmill:
monocular COLMAP recovers **sub-degree egomotion** (100% registered, <1% translation)
at sufficient frame density — with intrinsics fixed from the focal prior (free
intrinsics gave a ~13-16% scale bias, per the calibration arc). Frame **density is
load-bearing** (a sparse frame set silently under-registers), so this rung wants a
dense ``frame_source`` (e.g. ``'all'``). Dynamic masking is a *robustness* rung
(FP6b step 3), applied when the dynamic region is large — not run here.

Reconstruction is read via ``pycolmap`` (hloc writes a binary model; the calibration
``model_io`` text parser does not apply). Step 1 lands this as a config-selectable rung
(``backend_name='colmap'``) validated as a native swap vs GT extrinsics
(``scripts/fp6b_validate_colmap.py``); combining COLMAP poses + MoGe depth is the
pose/depth split (step 4) and COLMAP↔MoGe scale reconciliation is step 5.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from reifyscene.calibration.hloc_colmap import HlocColmapSfMRunner
from reifyscene.common.types import CameraIntrinsics, CameraPose, FrameRef, TimedCameraState
from reifyscene.geometry.base import GeometryInitializer
from reifyscene.geometry.geometry_artifact import GeometryArtifact
from reifyscene.geometry.image_transform import ImageTransform, transform_record
from reifyscene.geometry.reconstruction_health import assess_pose_health
from reifyscene.ingest.decoder import VideoDecoder


class ColmapPoseInitializer(GeometryInitializer):
    """Monocular hloc+COLMAP camera egomotion (poses only; no dense depth)."""

    backend_name = "colmap"

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or {}
        self.run_id = str(self.cfg.get("run_id", "colmap"))
        self.save_dir = Path(self.cfg.get("geometry_save_dir", "artifacts/geometry_colmap"))
        self.work_dir = Path(self.cfg.get("colmap_work_dir", ".reifyscene_cache/colmap_pose"))
        # Populated by ``reconstruct`` so the FP6b fused path (scale reconciliation) can
        # re-read the sparse model's 3D points + reprojections. None until a run lands.
        self._last_model_dir: Path | None = None
        # Intrinsics prior: "fx,fy,cx,cy" fixes intrinsics (calibration-arc lesson:
        # free intrinsics -> ~13-16% scale bias). Absent -> COLMAP free-estimates
        # them (manifest/notes flag the residual-scale caveat downstream).
        self.camera_params = self.cfg.get("camera_params")
        self.fix_intrinsics = bool(self.cfg.get("fix_intrinsics", True)) and self.camera_params is not None
        self.retrieval_top_k = int(self.cfg.get("retrieval_top_k", 20))
        self.decoder = VideoDecoder(default_fps=float(self.cfg.get("default_fps", 30.0)))
        # FP6b step 3: apply the dyn_mask only when the scene is large-dynamic — the spike
        # found masking is a SIZE-GATED robustness help (tightens large-dynamic scenes,
        # neutral/worse on small ones). Gate on the median per-frame dynamic fraction.
        self.mask_size_gate = float(self.cfg.get("mask_size_gate", 0.10))

    # -- helpers -----------------------------------------------------------

    def _select_frames(self, frames, ingest_artifact) -> list[FrameRef]:
        if frames:
            return list(frames)
        if ingest_artifact is not None:
            return list(ingest_artifact.keyframes)
        return []

    def _intrinsics_from_params(self, params: np.ndarray) -> CameraIntrinsics:
        """COLMAP PINHOLE params = [fx, fy, cx, cy] (pixels)."""
        p = np.asarray(params, dtype=np.float64).reshape(-1)
        return CameraIntrinsics(fx=float(p[0]), fy=float(p[1]), cx=float(p[2]), cy=float(p[3]))

    def _find_model_dir(self, sparse_dir: Path):
        import pycolmap  # noqa: PLC0415

        cands = [sparse_dir, sparse_dir / "sparse", sparse_dir / "models" / "0"]
        models = sparse_dir / "models"
        if models.exists():
            cands += sorted(models.glob("*"))
        for cand in cands:
            if (cand / "cameras.bin").exists() or (cand / "cameras.txt").exists():
                try:
                    pycolmap.Reconstruction(str(cand))
                    return cand
                except Exception:
                    continue
        return None

    def _load_poses(self, model_dir: Path) -> dict[int, tuple[np.ndarray, CameraIntrinsics]]:
        """frame-index -> (T_world_cam 4x4, intrinsics). Frames are materialized as
        ``{index:06d}.png`` so the COLMAP image name maps straight back to a FrameRef."""
        import pycolmap  # noqa: PLC0415

        rec = pycolmap.Reconstruction(str(model_dir))
        out: dict[int, tuple[np.ndarray, CameraIntrinsics]] = {}
        for im in rec.images.values():
            cfw = im.cam_from_world
            cfw = cfw() if callable(cfw) else cfw
            r_wc = np.asarray(cfw.rotation.matrix(), dtype=np.float64)
            t_wc = np.asarray(cfw.translation, dtype=np.float64).reshape(3)
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = r_wc
            w2c[:3, 3] = t_wc
            t_world_cam = np.linalg.inv(w2c).astype(np.float32)
            cam = rec.cameras[im.camera_id]
            intr = self._intrinsics_from_params(cam.params)
            out[int(Path(im.name).stem)] = (t_world_cam, intr)
        return out

    def reconstruct(self, img_dir: Path) -> dict[int, tuple[np.ndarray, CameraIntrinsics]]:
        """Run hloc+COLMAP over a dir of index-named frames (``{idx:06d}.png``).

        The shared SfM core: used by ``initialize`` (video-decoded frames) and by the
        FP6b native-swap validation (dataset PNGs). Returns ``{frame-index:
        (T_world_cam 4x4, intrinsics)}`` for every registered frame; ``{}`` if COLMAP
        produced no model.
        """
        runner_cfg: dict[str, Any] = {
            "camera_mode": "SINGLE",       # monocular: one shared camera
            "camera_model": "PINHOLE",
            "retrieval_top_k": self.retrieval_top_k,
            "verbose": bool(self.cfg.get("verbose", False)),
            "fix_intrinsics": self.fix_intrinsics,
        }
        if self.camera_params is not None:
            runner_cfg["camera_params"] = str(self.camera_params)
        sparse_dir = HlocColmapSfMRunner(runner_cfg).run(img_dir, self.work_dir / "sfm")
        model_dir = self._find_model_dir(Path(sparse_dir))
        self._last_model_dir = model_dir  # expose the sparse model for scale reconciliation
        return self._load_poses(model_dir) if model_dir is not None else {}

    # -- artifact construction --------------------------------------------

    def _resolve_masking(self, dyn_mask_artifact):
        """Decide whether to apply dyn_mask (size-gated) and return (apply, frac, note)."""
        if dyn_mask_artifact is None:
            return False, 0.0, "dyn_mask: none provided → unmasked."
        frac = float(dyn_mask_artifact.aggregate_dynamic_fraction())
        apply = frac >= self.mask_size_gate
        gate = (
            f"dyn_mask '{dyn_mask_artifact.backend_name}': median dynamic fraction {frac:.3f} "
            f"{'≥' if apply else '<'} gate {self.mask_size_gate:.3f} → "
            f"{'MASKED (large-dynamic)' if apply else 'unmasked (small-dynamic, size-gated off)'}."
        )
        return apply, frac, gate

    def initialize(self, frames=None, *, ingest_artifact=None, sync_artifact=None, dyn_mask_artifact=None) -> GeometryArtifact:
        selected = self._select_frames(frames, ingest_artifact)
        if not selected:
            return GeometryArtifact(
                run_id=self.run_id, stage_name="geometry", save_dir=self.save_dir,
                backend_name=self.backend_name, selected_frames=[],
                notes=["COLMAP: no frames selected -> empty geometry."],
            )
        if ingest_artifact is None:
            raise ValueError("ColmapPoseInitializer requires ingest_artifact to decode frames.")

        apply_mask, dyn_frac, mask_note = self._resolve_masking(dyn_mask_artifact)

        # materialize dense frames (flat dir, index-named) for a SINGLE monocular camera.
        # Size-gated dyn_mask: black out dynamic pixels so SfM keypoints avoid moving
        # content (spike: rgb[mask>0.5]=0 before hloc — protects the pose rung).
        img_dir = self.work_dir / "images"
        if img_dir.exists():
            shutil.rmtree(img_dir)
        img_dir.mkdir(parents=True, exist_ok=True)
        stream_lookup = {s.stream_id: s for s in ingest_artifact.streams}
        for idx, frame in enumerate(selected):
            stream = stream_lookup.get(frame.stream_id)
            if stream is None:
                continue
            rgb = self.decoder.get_frame(stream, frame.frame_idx)
            if apply_mask:
                mask = dyn_mask_artifact.mask_for(f"{frame.stream_id}:{frame.frame_idx}")
                if mask is not None and mask.shape == rgb.shape[:2]:
                    rgb = rgb.copy()
                    rgb[np.asarray(mask) > 0.5] = 0
            Image.fromarray(rgb).save(img_dir / f"{idx:06d}.png")

        poses = self.reconstruct(img_dir)  # {frame-index: (T_world_cam, intrinsics)}

        intrinsics_by_stream: dict[str, CameraIntrinsics] = {}
        camera_states_by_stream: dict[str, list[TimedCameraState]] = {}
        registered: list[FrameRef] = []
        for idx, frame in enumerate(selected):
            if idx not in poses:
                continue  # frame COLMAP did not register
            t_world_cam, intr = poses[idx]
            intrinsics_by_stream.setdefault(frame.stream_id, intr)
            camera_states_by_stream.setdefault(frame.stream_id, []).append(
                TimedCameraState(
                    stream_id=frame.stream_id,
                    time_sec=float(frame.timestamp_sec),
                    intrinsics=intrinsics_by_stream[frame.stream_id],
                    pose=CameraPose(T_world_cam=t_world_cam),
                )
            )
            registered.append(frame)
        for states in camera_states_by_stream.values():
            states.sort(key=lambda s: (s.time_sec, s.stream_id))

        reg_frac = len(registered) / max(len(selected), 1)
        # COLMAP is up-to-similarity; if intrinsics were free-estimated, note the
        # ~13-16% residual-scale caveat (calibration arc) for scale reconciliation (step 2).
        artifact = GeometryArtifact(
            run_id=self.run_id, stage_name="geometry", save_dir=self.save_dir,
            backend_name=self.backend_name, selected_frames=registered,
            intrinsics_by_stream=intrinsics_by_stream,
            camera_states_by_stream=camera_states_by_stream,
            notes=[
                f"COLMAP-SfM pose rung: registered {len(registered)}/{len(selected)} "
                f"frames ({100 * reg_frac:.0f}%).",
                f"intrinsics={'fixed' if self.fix_intrinsics else 'free-estimated (up-to-scale; ~13-16% residual, resolve at scale reconciliation)'}.",
                mask_note,
                "pose-only rung: camera egomotion + intrinsics; no dense depth "
                "(depth comes from the MoGe rung on the depth ladder).",
            ],
        )
        # Expose the sparse model so the FP6b fused path can reconcile COLMAP↔MoGe scale
        # from its 3D-point reprojections (design §4A.1).
        if self._last_model_dir is not None:
            artifact.metadata["colmap_model_dir"] = str(self._last_model_dir)
        artifact.metadata["dyn_mask_applied"] = bool(apply_mask)
        artifact.metadata["dyn_mask_fraction"] = float(dyn_frac)
        # SfM ran on the decoded frames themselves: its intrinsics are in their pixels.
        artifact.metadata["image_transforms"] = {
            stream_id: transform_record(
                ImageTransform.identity((int(stream.width), int(stream.height))),
                recorded_by=self.backend_name,
            )
            for stream_id, stream in stream_lookup.items()
            if stream_id in camera_states_by_stream
        }
        # FP6b step 3: stamp a GT-free per-stream trajectory health-check so a degenerate
        # (silent) SfM draw is observable downstream (design: observable degradation).
        health = assess_pose_health(camera_states_by_stream)
        artifact.metadata["pose_health"] = health
        if not health["healthy"]:
            artifact.notes.append(f"⚠ pose health: SUSPECT — {health['reason']}")
        return artifact
