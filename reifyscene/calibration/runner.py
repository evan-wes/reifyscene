"""Calibration-mode orchestration."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from reifyscene.calibration.artifact import CalibrationArtifact
from reifyscene.calibration.hloc_colmap import HlocColmapSfMRunner
from reifyscene.calibration.model_io import parse_colmap_text_model, timed_camera_from_image_record
from reifyscene.calibration.report import write_calibration_report, write_calibration_viewer
from reifyscene.calibration.sampler import CalibrationFrameSampler
from reifyscene.common.frame_keys import parse_key
from reifyscene.common.io import read_json
from reifyscene.common.paths import ensure_dir


class CalibrationRunner:
    """Sample static frames, run hloc+COLMAP, and export a reusable artifact."""

    def __init__(self, cfg: dict[str, Any] | None = None, *, decoder=None) -> None:
        self.cfg = cfg or {}
        self.sampler = CalibrationFrameSampler(self.cfg.get("sampling", {}), decoder=decoder)
        self.sfm_runner = HlocColmapSfMRunner(self.cfg.get("hloc_colmap", {}))

    def _copy_text_model(self, sparse_dir: Path, save_dir: Path) -> Path:
        out = ensure_dir(save_dir / "colmap_text")
        for name in ("cameras.txt", "images.txt", "points3D.txt"):
            src = sparse_dir / name
            if src.exists():
                shutil.copy2(src, out / name)
        return out

    def run(self, *, run_id: str, save_dir: str | Path, video_paths: list[str | Path]) -> CalibrationArtifact:
        save_dir = ensure_dir(save_dir)
        streams, frame_refs = self.sampler.sample(video_paths)
        image_dir = ensure_dir(save_dir / "sampled_frames")
        sampled_index = self.sampler.materialize_images(streams, frame_refs, image_dir)
        sparse_dir = self.sfm_runner.run(image_dir, save_dir / "work")
        manifest_path = image_dir / "image_manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path).get("images", {})
        else:
            frame_lookup = {(fr.stream_id, int(fr.frame_idx)): fr for fr in frame_refs}
            manifest = {}
            for key, rel in sampled_index.items():
                stream_id, frame_idx = parse_key(key)
                frame_ref = frame_lookup.get((stream_id, int(frame_idx)))
                if frame_ref is None:
                    continue
                manifest[str(rel)] = {
                    "stream_id": stream_id,
                    "frame_idx": int(frame_ref.frame_idx),
                    "timestamp_sec": float(frame_ref.timestamp_sec),
                }
        text_model_dir = self._copy_text_model(sparse_dir, save_dir)
        model = parse_colmap_text_model(text_model_dir)

        intrinsics_by_stream = {}
        camera_states_by_stream: dict[str, list[Any]] = {}
        for image_name, rec in model.image_records.items():
            if image_name not in manifest:
                continue
            camera_id = int(rec["camera_id"])
            intr = model.intrinsics_by_camera_id[camera_id]
            state = timed_camera_from_image_record(image_name, rec, manifest, intr)
            intrinsics_by_stream.setdefault(state.stream_id, intr)
            camera_states_by_stream.setdefault(state.stream_id, []).append(state)
        for states in camera_states_by_stream.values():
            states.sort(key=lambda s: (s.time_sec, s.stream_id))

        point_ids = sorted(model.point_records.keys())
        if point_ids:
            sparse_xyz = np.asarray([model.point_records[pid]["xyz"] for pid in point_ids], dtype=np.float32)
            sparse_rgb = np.asarray([model.point_records[pid]["rgb"] for pid in point_ids], dtype=np.uint8)
            sparse_err = np.asarray([model.point_records[pid]["error"] for pid in point_ids], dtype=np.float32)
        else:
            sparse_xyz = np.zeros((0, 3), dtype=np.float32)
            sparse_rgb = np.zeros((0, 3), dtype=np.uint8)
            sparse_err = np.zeros((0,), dtype=np.float32)

        viewer_path = write_calibration_viewer(save_dir, camera_states_by_stream, sparse_xyz, sparse_rgb)
        sampled_counts = {stream.stream_id: 0 for stream in streams}
        for fr in frame_refs:
            sampled_counts[fr.stream_id] = sampled_counts.get(fr.stream_id, 0) + 1
        notes = [
            f"sampled {len(frame_refs)} static frame(s) across {len(streams)} stream(s)",
            f"registered {int(model.reprojection_stats.get('registered_images', 0))} image(s)",
        ]
        report_path = write_calibration_report(
            save_dir,
            streams=[stream.stream_id for stream in streams],
            sampled_counts=sampled_counts,
            reprojection_stats=model.reprojection_stats,
            connected_components=model.connected_components,
            notes=notes,
            viewer_path=viewer_path,
        )
        artifact = CalibrationArtifact(
            run_id=run_id,
            stage_name="calibration",
            save_dir=save_dir,
            streams=streams,
            sampled_frames=frame_refs,
            sampled_image_index=sampled_index,
            intrinsics_by_stream=intrinsics_by_stream,
            camera_states_by_stream=camera_states_by_stream,
            sparse_points_xyz=sparse_xyz,
            sparse_points_rgb=sparse_rgb,
            sparse_point_errors=sparse_err,
            connected_components=model.connected_components,
            reprojection_stats=model.reprojection_stats,
            notes=notes,
            metadata={
                "viewer_html": viewer_path.name,
                "report_html": report_path.name,
                "colmap_text_dir": str(text_model_dir.relative_to(save_dir)),
                "num_streams": len(streams),
                "num_sampled_frames": len(frame_refs),
                "registered_images": int(model.reprojection_stats.get("registered_images", 0)),
            },
        )
        artifact.save()
        return artifact
