"""Single-video scene assembly from geometry, tracking, humans, and objects."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from room4d.common.types import FrameRef, HumanState, ObjectState
from room4d.geometry.geometry_artifact import GeometryArtifact
from room4d.scene.scene_artifact import SceneArtifact


@dataclass
class SingleVideoSceneAssembler:
    """Assemble a simple static/dynamic 4D scene from pipeline artifacts."""

    max_static_points_per_frame: int = 8000
    max_static_points_total: int = 60000
    dynamic_trail_window: int = 16

    @staticmethod
    def _frame_key(stream_id: str, frame_idx: int) -> str:
        return f"{stream_id}:{int(frame_idx)}"

    @staticmethod
    def _stable_color(stable_id: str) -> list[int]:
        hue = ((abs(hash(stable_id)) % 10_000) * 0.6180339887498949) % 1.0
        import colorsys

        r, g, b = colorsys.hsv_to_rgb(hue, 0.70, 0.95)
        return [int(r * 255), int(g * 255), int(b * 255)]

    @staticmethod
    def _center_xyxy(box: np.ndarray) -> np.ndarray:
        box = np.asarray(box, dtype=np.float32)
        return np.asarray([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)

    @staticmethod
    def _estimate_sampling_stride(num_points: int, target: int) -> int:
        if num_points <= target:
            return 1
        return max(1, int(math.ceil(num_points / float(target))))

    def _dynamic_mask_for_frame(self, frame_key: str, geometry_shape: tuple[int, int], human_artifact, object_artifact) -> np.ndarray:
        mask = np.zeros(geometry_shape, dtype=bool)
        for artifact in (human_artifact, object_artifact):
            if artifact is None:
                continue
            frame_index = getattr(artifact, "frame_index", {}) or {}
            ids = frame_index.get(frame_key, [])
            states_for_artifact = list(getattr(artifact, "humans", [])) + list(getattr(artifact, "objects", []))
            for state_id in ids:
                mask_arr = None
                attrs = None
                for state in states_for_artifact:
                    stable_id = getattr(state, "human_id", getattr(state, "object_id", None))
                    if stable_id == state_id and self._frame_key(state.frame_ref.stream_id, state.frame_ref.frame_idx) == frame_key:
                        attrs = getattr(state, "attributes", {})
                        break
                if attrs is not None and hasattr(artifact, "mask_arrays"):
                    det_id = attrs.get("detection_id", state_id)
                    mask_arr = artifact.mask_arrays.get(det_id)
                if mask_arr is None:
                    continue
                mask_arr = np.asarray(mask_arr, dtype=bool)
                if mask_arr.shape == geometry_shape:
                    mask |= mask_arr
        return mask

    def _gather_rgb_for_keyframe(self, frame_key: str, geometry_artifact) -> np.ndarray | None:
        source_map = geometry_artifact.metadata.get("debug_image_paths", {}) if geometry_artifact is not None else {}
        source = source_map.get(frame_key)
        if source is None:
            return None
        try:
            from PIL import Image

            return np.asarray(Image.open(source).convert("RGB"), dtype=np.uint8)
        except Exception:
            return None

    def _assemble_static_background(self, geometry_artifact, human_artifact, object_artifact) -> dict[str, Any]:
        all_points: list[np.ndarray] = []
        all_colors: list[np.ndarray] = []
        frame_point_counts: dict[str, int] = {}
        for frame in geometry_artifact.selected_frames:
            frame_key = self._frame_key(frame.stream_id, frame.frame_idx)
            point_payload = geometry_artifact.pointmaps_by_frame.get(frame_key)
            if point_payload is None:
                continue
            pts = np.asarray(point_payload.get("points"), dtype=np.float32)
            if pts.ndim != 3 or pts.shape[-1] != 3:
                continue
            h, w = pts.shape[:2]
            valid = np.isfinite(pts).all(axis=2)
            dynamic_mask = self._dynamic_mask_for_frame(frame_key, (h, w), human_artifact, object_artifact)
            valid &= ~dynamic_mask
            conf = point_payload.get("confidence")
            if conf is not None:
                conf_arr = np.asarray(conf)
                if conf_arr.shape == valid.shape:
                    conf_thresh = float(np.nanmedian(conf_arr[valid])) if np.any(valid) else 0.0
                    valid &= conf_arr >= conf_thresh
            if not np.any(valid):
                continue
            stride = self._estimate_sampling_stride(int(valid.sum()), self.max_static_points_per_frame)
            selector = np.zeros_like(valid, dtype=bool)
            selector.flat[::stride] = True
            valid &= selector
            sampled = pts[valid]
            rgb = self._gather_rgb_for_keyframe(frame_key, geometry_artifact)
            if rgb is None:
                colors = np.full((sampled.shape[0], 3), 180, dtype=np.uint8)
            else:
                if rgb.shape[:2] != (h, w):
                    from PIL import Image
                    rgb = np.asarray(Image.fromarray(rgb).resize((w, h)), dtype=np.uint8)
                colors = rgb[valid]
            all_points.append(sampled.astype(np.float32))
            all_colors.append(colors.astype(np.uint8))
            frame_point_counts[frame_key] = int(sampled.shape[0])
        if all_points:
            points = np.concatenate(all_points, axis=0)
            colors = np.concatenate(all_colors, axis=0)
            stride = self._estimate_sampling_stride(points.shape[0], self.max_static_points_total)
            points = points[::stride]
            colors = colors[::stride]
        else:
            points = np.zeros((0, 3), dtype=np.float32)
            colors = np.zeros((0, 3), dtype=np.uint8)
        center = points.mean(axis=0).astype(np.float32) if points.size else np.zeros((3,), dtype=np.float32)
        return {
            "points_xyz": points,
            "colors_rgb": colors,
            "num_points": int(points.shape[0]),
            "center_xyz": center.tolist(),
            "frame_point_counts": frame_point_counts,
        }

    def _track_world_position(self, stream_id: str, frame_idx: int, support_track_ids: list[int], track_artifact, geometry_artifact) -> np.ndarray | None:
        payload = track_artifact.track_payloads_by_stream.get(stream_id, {}) if track_artifact is not None else {}
        track_ids = np.asarray(payload.get("track_ids", []), dtype=int)
        if track_ids.size and support_track_ids:
            lifted = np.asarray(payload.get("refined_lifted_xyz", payload.get("lifted_xyz", np.zeros((0, 0, 3), dtype=np.float32))), dtype=np.float32)
            lifted_idx = np.asarray(payload.get("refined_lifted_frame_indices", payload.get("lifted_frame_indices", [])), dtype=int)
            lifted_valid = np.asarray(payload.get("refined_lifted_valid", payload.get("lifted_valid", np.zeros((0, 0), dtype=bool))))
            matches = np.where(lifted_idx == int(frame_idx))[0]
            if lifted.ndim == 3 and matches.size:
                k = int(matches[0])
                support_mask = np.isin(track_ids, np.asarray(support_track_ids, dtype=int))
                support_mask &= lifted_valid[k] if lifted_valid.size else np.isfinite(lifted[k]).all(axis=1)
                if np.any(support_mask):
                    pts = lifted[k][support_mask]
                    pts = pts[np.isfinite(pts).all(axis=1)]
                    if pts.size:
                        return pts.mean(axis=0).astype(np.float32)
        frame_key = self._frame_key(stream_id, frame_idx)
        point_payload = geometry_artifact.pointmaps_by_frame.get(frame_key) if geometry_artifact is not None else None
        if point_payload is None:
            return None
        pts = np.asarray(point_payload.get("points"), dtype=np.float32)
        if pts.ndim != 3:
            return None
        h, w = pts.shape[:2]
        cy, cx = h // 2, w // 2
        candidate = pts[cy, cx]
        if np.all(np.isfinite(candidate)):
            return candidate.astype(np.float32)
        valid = pts[np.isfinite(pts).all(axis=2)]
        if valid.size:
            return valid.reshape(-1, 3).mean(axis=0).astype(np.float32)
        return None

    def _assemble_entities(self, states: list[Any], kind: str, track_artifact, geometry_artifact) -> list[dict[str, Any]]:
        grouped: dict[str, list[Any]] = {}
        for state in states:
            stable_id = getattr(state, "human_id", getattr(state, "object_id", ""))
            grouped.setdefault(str(stable_id), []).append(state)
        entities: list[dict[str, Any]] = []
        for stable_id, seq in sorted(grouped.items()):
            seq = sorted(seq, key=lambda s: (s.frame_ref.stream_id, int(s.frame_ref.frame_idx)))
            times = []
            centers_2d = []
            centers_3d = []
            support_counts = []
            labels = []
            confs = []
            support_ids_union: set[int] = set()
            for state in seq:
                attrs = getattr(state, "attributes", {})
                support_ids = [int(t) for t in attrs.get("track_ids", [])]
                world = self._track_world_position(state.frame_ref.stream_id, state.frame_ref.frame_idx, support_ids, track_artifact, geometry_artifact)
                support_ids_union.update(support_ids)
                times.append(float(state.frame_ref.timestamp_sec))
                centers_2d.append(self._center_xyxy(state.bbox_xyxy).tolist())
                centers_3d.append(None if world is None else world.astype(np.float32).tolist())
                support_counts.append(int(attrs.get("support_track_count", 0)))
                labels.append(getattr(state, "category", "person" if kind == "humans" else "object"))
                confs.append(float(getattr(state, "confidence", 0.0)))
            valid_positions = [np.asarray(v, dtype=np.float32) for v in centers_3d if v is not None]
            anchor = (np.mean(np.stack(valid_positions, axis=0), axis=0).astype(np.float32).tolist() if valid_positions else [0.0, 0.0, 0.0])
            majority_label = max(set(labels), key=labels.count) if labels else kind[:-1]
            entities.append({
                "entity_id": stable_id,
                "entity_type": kind[:-1],
                "display_label": majority_label,
                "color_rgb": self._stable_color(stable_id),
                "times_sec": times,
                "centers_2d": centers_2d,
                "centers_3d": centers_3d,
                "anchor_xyz": anchor,
                "support_counts": support_counts,
                "support_track_ids": sorted(support_ids_union),
                "mean_confidence": float(np.mean(confs)) if confs else 0.0,
                "lifespan_frames": int(len(seq)),
                "frame_keys": [self._frame_key(s.frame_ref.stream_id, s.frame_ref.frame_idx) for s in seq],
            })
        return entities

    def assemble(self, *, run_id: str, save_dir, geometry_artifact, track_artifact, human_artifact=None, object_artifact=None, metadata: dict[str, Any] | None = None) -> SceneArtifact:
        static_scene = self._assemble_static_background(geometry_artifact, human_artifact, object_artifact)
        humans = [] if human_artifact is None else self._assemble_entities(human_artifact.humans, "humans", track_artifact, geometry_artifact)
        objects = [] if object_artifact is None else self._assemble_entities(object_artifact.objects, "objects", track_artifact, geometry_artifact)
        frame_refs = sorted(list(track_artifact.selected_frames or geometry_artifact.selected_frames), key=lambda fr: (fr.stream_id, int(fr.frame_idx)))
        timeline = {
            "time_min_sec": float(min((fr.timestamp_sec for fr in frame_refs), default=0.0)),
            "time_max_sec": float(max((fr.timestamp_sec for fr in frame_refs), default=0.0)),
            "frame_keys": [self._frame_key(fr.stream_id, fr.frame_idx) for fr in frame_refs],
        }
        notes = [
            f"static_points={static_scene['num_points']}",
            f"humans={len(humans)}",
            f"objects={len(objects)}",
        ]
        return SceneArtifact(
            run_id=run_id,
            stage_name="scene",
            save_dir=save_dir,
            metadata=metadata or {},
            static_scene=static_scene,
            dynamic_entities={"humans": humans, "objects": objects},
            frame_refs=frame_refs,
            timeline=timeline,
            notes=notes,
        )
