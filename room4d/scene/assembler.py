"""Single-video scene assembly from geometry, tracking, humans, and objects."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from room4d.common.types import CameraIntrinsics, FrameRef
from room4d.scene.scene_artifact import SceneArtifact


@dataclass
class SingleVideoSceneAssembler:
    """Assemble a simple static/dynamic 4D scene from pipeline artifacts."""

    max_static_points_per_frame: int = 8000
    max_static_points_total: int = 60000
    dynamic_trail_window: int = 16
    timeline_source: str = "geometry"
    static_track_gaussian_budget: int = 1200
    static_appearance_source_stride: int = 1
    max_static_appearance_sources: int = 24
    max_dynamic_observations_per_entity: int = 16
    dynamic_crop_padding: float = 0.12
    min_sprite_world_height: float = 0.20
    max_sprite_world_height: float = 2.80

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

    @staticmethod
    def _intrinsics_to_dict(intr: CameraIntrinsics | None) -> dict[str, float]:
        if intr is None:
            return {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}
        return {"fx": float(intr.fx), "fy": float(intr.fy), "cx": float(intr.cx), "cy": float(intr.cy)}

    @staticmethod
    def _sanitize_key(key: str) -> str:
        return key.replace(':', '__').replace('/', '_')

    def _camera_state_for_frame(self, geometry_artifact, frame_ref: FrameRef):
        states = list((geometry_artifact.camera_states_by_stream or {}).get(frame_ref.stream_id, []))
        if not states:
            return None
        return min(states, key=lambda s: (abs(float(s.time_sec) - float(frame_ref.timestamp_sec)), abs(float(s.time_sec) - float(frame_ref.timestamp_sec))))

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
                if conf_arr.shape == valid.shape and np.any(valid):
                    conf_thresh = float(np.nanmedian(conf_arr[valid]))
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

    def _extract_crop_rgba(self, rgb: np.ndarray, bbox_xyxy: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, list[int]]:
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in np.asarray(bbox_xyxy, dtype=np.float32).tolist()]
        pad_x = (x2 - x1) * self.dynamic_crop_padding
        pad_y = (y2 - y1) * self.dynamic_crop_padding
        x1i = max(0, int(np.floor(x1 - pad_x)))
        y1i = max(0, int(np.floor(y1 - pad_y)))
        x2i = min(w, int(np.ceil(x2 + pad_x)))
        y2i = min(h, int(np.ceil(y2 + pad_y)))
        if x2i <= x1i or y2i <= y1i:
            return np.zeros((1, 1, 4), dtype=np.uint8), [0, 0, 1, 1]
        crop_rgb = np.asarray(rgb[y1i:y2i, x1i:x2i], dtype=np.uint8)
        if mask is not None and mask.shape[:2] == (h, w):
            alpha = (np.asarray(mask[y1i:y2i, x1i:x2i], dtype=np.uint8) * 255)
        else:
            alpha = np.full(crop_rgb.shape[:2], 255, dtype=np.uint8)
        rgba = np.dstack([crop_rgb, alpha]).astype(np.uint8)
        return rgba, [x1i, y1i, x2i, y2i]

    def _estimate_sprite_world_size(self, frame_ref: FrameRef, bbox_xyxy: np.ndarray, center_xyz: np.ndarray | None, geometry_artifact, kind: str) -> list[float]:
        if center_xyz is None:
            if kind == "humans":
                return [0.55, 1.75]
            return [0.45, 0.45]
        state = self._camera_state_for_frame(geometry_artifact, frame_ref)
        if state is None:
            return [0.55, 1.75] if kind == "humans" else [0.45, 0.45]
        Twc = np.asarray(state.pose.T_world_cam, dtype=np.float32)
        Tcw = np.linalg.inv(Twc)
        p = np.asarray(list(np.asarray(center_xyz, dtype=np.float32)) + [1.0], dtype=np.float32)
        cam = Tcw @ p
        depth = abs(float(cam[2]))
        intr = state.intrinsics
        bbox = np.asarray(bbox_xyxy, dtype=np.float32)
        bbox_w = max(float(bbox[2] - bbox[0]), 1.0)
        bbox_h = max(float(bbox[3] - bbox[1]), 1.0)
        width_world = float(np.clip(depth * bbox_w / max(float(intr.fx), 1e-6), 0.08, 2.5))
        height_world = float(np.clip(depth * bbox_h / max(float(intr.fy), 1e-6), self.min_sprite_world_height, self.max_sprite_world_height))
        return [width_world, height_world]

    def _build_entity_observations(self, seq: list[Any], centers_3d: list[Any], kind: str, source_artifact, geometry_artifact) -> list[dict[str, Any]]:
        if source_artifact is None:
            return []
        image_map = geometry_artifact.metadata.get("debug_image_paths", {}) if geometry_artifact is not None else {}
        observations: list[dict[str, Any]] = []
        if len(seq) > self.max_dynamic_observations_per_entity:
            sample_idx = np.linspace(0, len(seq) - 1, self.max_dynamic_observations_per_entity).round().astype(int).tolist()
        else:
            sample_idx = list(range(len(seq)))
        for idx in sample_idx:
            state = seq[idx]
            center = centers_3d[idx]
            frame_key = self._frame_key(state.frame_ref.stream_id, state.frame_ref.frame_idx)
            image_path = image_map.get(frame_key)
            if image_path is None:
                continue
            try:
                rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            except Exception:
                continue
            attrs = getattr(state, "attributes", {}) or {}
            det_id = attrs.get("detection_id", getattr(state, "human_id", getattr(state, "object_id", frame_key)))
            mask = None
            if hasattr(source_artifact, "mask_arrays"):
                mask = source_artifact.mask_arrays.get(det_id)
            rgba, crop_box = self._extract_crop_rgba(rgb, np.asarray(state.bbox_xyxy, dtype=np.float32), None if mask is None else np.asarray(mask, dtype=bool))
            observations.append({
                "frame_key": frame_key,
                "time_sec": float(state.frame_ref.timestamp_sec),
                "bbox_xyxy": np.asarray(state.bbox_xyxy, dtype=np.float32).tolist(),
                "crop_box_xyxy": crop_box,
                "sprite_rgba": rgba,
                "world_size_wh": self._estimate_sprite_world_size(state.frame_ref, np.asarray(state.bbox_xyxy, dtype=np.float32), None if center is None else np.asarray(center, dtype=np.float32), geometry_artifact, kind),
                "confidence": float(getattr(state, "confidence", 0.0)),
                "center_xyz": None if center is None else np.asarray(center, dtype=np.float32).tolist(),
            })
        return observations

    def _assemble_entities(self, states: list[Any], kind: str, track_artifact, geometry_artifact, source_artifact=None) -> list[dict[str, Any]]:
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
            observations = self._build_entity_observations(seq, centers_3d, kind, source_artifact, geometry_artifact)
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
                "observations": observations,
            })
        return entities

    @staticmethod
    def _template_offsets(kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if kind == "human":
            offsets = np.asarray([
                [0.0, 0.0, -0.50],
                [0.0, 0.0, -0.15],
                [0.0, 0.0, 0.15],
                [0.0, 0.0, 0.45],
                [0.18, 0.0, -0.10],
                [-0.18, 0.0, -0.10],
                [0.0, 0.0, 0.75],
            ], dtype=np.float32)
            scales = np.asarray([
                [0.10, 0.10, 0.18],
                [0.14, 0.12, 0.18],
                [0.16, 0.12, 0.20],
                [0.14, 0.12, 0.18],
                [0.09, 0.09, 0.14],
                [0.09, 0.09, 0.14],
                [0.11, 0.11, 0.11],
            ], dtype=np.float32)
            opacities = np.asarray([0.70, 0.82, 0.86, 0.76, 0.54, 0.54, 0.62], dtype=np.float32)
            return offsets, scales, opacities
        offsets = np.asarray([
            [0.0, 0.0, 0.0],
            [0.08, 0.0, 0.0],
            [-0.08, 0.0, 0.0],
            [0.0, 0.08, 0.0],
            [0.0, -0.08, 0.0],
        ], dtype=np.float32)
        scales = np.asarray([
            [0.10, 0.10, 0.10],
            [0.08, 0.08, 0.08],
            [0.08, 0.08, 0.08],
            [0.08, 0.08, 0.08],
            [0.08, 0.08, 0.08],
        ], dtype=np.float32)
        opacities = np.asarray([0.88, 0.58, 0.58, 0.58, 0.58], dtype=np.float32)
        return offsets, scales, opacities

    def _static_track_gaussians(self, track_artifact) -> dict[str, np.ndarray]:
        means: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        colors: list[np.ndarray] = []
        opacities: list[float] = []
        source_codes: list[int] = []
        if track_artifact is None:
            return {
                "means_xyz": np.zeros((0, 3), dtype=np.float32),
                "scales_xyz": np.zeros((0, 3), dtype=np.float32),
                "colors_rgb": np.zeros((0, 3), dtype=np.uint8),
                "opacities": np.zeros((0,), dtype=np.float32),
                "source_codes": np.zeros((0,), dtype=np.int16),
            }
        for stream_id, payload in (track_artifact.track_payloads_by_stream or {}).items():
            track_ids = np.asarray(payload.get("track_ids", []), dtype=int)
            labels = np.asarray(payload.get("static_labels", np.full((track_ids.shape[0],), -1, dtype=np.int8)), dtype=np.int8)
            scores = np.asarray(payload.get("static_scores", np.full((track_ids.shape[0],), np.nan, dtype=np.float32)), dtype=np.float32)
            lifted = np.asarray(payload.get("refined_lifted_xyz", payload.get("lifted_xyz", np.zeros((0, 0, 3), dtype=np.float32))), dtype=np.float32)
            valid = np.asarray(payload.get("refined_lifted_valid", payload.get("lifted_valid", np.zeros((0, 0), dtype=bool))))
            if lifted.ndim != 3 or track_ids.size == 0:
                continue
            for j, track_id in enumerate(track_ids.tolist()):
                label = int(labels[j]) if labels.size else -1
                if label <= 0:
                    continue
                pts = lifted[:, j]
                keep = valid[:, j].astype(bool) if valid.size else np.isfinite(pts).all(axis=1)
                pts = pts[keep]
                pts = pts[np.isfinite(pts).all(axis=1)]
                if pts.size == 0:
                    continue
                center = np.mean(pts, axis=0).astype(np.float32)
                dispersion = float(np.median(np.linalg.norm(pts - center[None, :], axis=1))) if pts.shape[0] > 1 else 0.0
                scale = float(np.clip(max(dispersion * 1.5, 0.03), 0.03, 0.12))
                score = float(scores[j]) if scores.size and np.isfinite(scores[j]) else 0.5
                opacity = float(np.clip(0.65 + 0.25 * score, 0.55, 0.95))
                means.append(center)
                scales.append(np.full((3,), scale, dtype=np.float32))
                colors.append(np.asarray(self._stable_color(f"{stream_id}:track:{track_id}"), dtype=np.uint8))
                opacities.append(opacity)
                source_codes.append(1)
        if not means:
            return {
                "means_xyz": np.zeros((0, 3), dtype=np.float32),
                "scales_xyz": np.zeros((0, 3), dtype=np.float32),
                "colors_rgb": np.zeros((0, 3), dtype=np.uint8),
                "opacities": np.zeros((0,), dtype=np.float32),
                "source_codes": np.zeros((0,), dtype=np.int16),
            }
        means_arr = np.asarray(means, dtype=np.float32)
        scales_arr = np.asarray(scales, dtype=np.float32)
        colors_arr = np.asarray(colors, dtype=np.uint8)
        opacities_arr = np.asarray(opacities, dtype=np.float32)
        codes_arr = np.asarray(source_codes, dtype=np.int16)
        stride = self._estimate_sampling_stride(means_arr.shape[0], self.static_track_gaussian_budget)
        return {
            "means_xyz": means_arr[::stride],
            "scales_xyz": scales_arr[::stride],
            "colors_rgb": colors_arr[::stride],
            "opacities": opacities_arr[::stride],
            "source_codes": codes_arr[::stride],
        }

    def _build_static_appearance_sources(self, geometry_artifact, human_artifact, object_artifact) -> list[dict[str, Any]]:
        image_map = geometry_artifact.metadata.get("debug_image_paths", {}) if geometry_artifact is not None else {}
        frames = list(getattr(geometry_artifact, "selected_frames", []) or [])
        if self.static_appearance_source_stride > 1:
            frames = frames[:: int(self.static_appearance_source_stride)]
        if len(frames) > self.max_static_appearance_sources:
            indices = np.linspace(0, len(frames) - 1, self.max_static_appearance_sources).round().astype(int).tolist()
            frames = [frames[i] for i in indices]
        sources: list[dict[str, Any]] = []
        for frame in frames:
            frame_key = self._frame_key(frame.stream_id, frame.frame_idx)
            payload = geometry_artifact.pointmaps_by_frame.get(frame_key)
            image_path = image_map.get(frame_key)
            if payload is None or image_path is None:
                continue
            points = np.asarray(payload.get("points"), dtype=np.float32)
            if points.ndim != 3 or points.shape[-1] != 3:
                continue
            valid = np.isfinite(points).all(axis=2)
            dynamic_mask = self._dynamic_mask_for_frame(frame_key, points.shape[:2], human_artifact, object_artifact)
            valid &= ~dynamic_mask
            conf = payload.get("confidence")
            conf_arr = None if conf is None else np.asarray(conf, dtype=np.float32)
            if conf_arr is not None and conf_arr.shape == valid.shape and np.any(valid):
                conf_threshold = float(np.nanmedian(conf_arr[valid]))
                valid &= conf_arr >= conf_threshold
            masked_points = np.asarray(points, dtype=np.float32).copy()
            masked_points[~valid] = np.nan
            state = self._camera_state_for_frame(geometry_artifact, frame)
            intr = state.intrinsics if state is not None else geometry_artifact.intrinsics_by_stream.get(frame.stream_id)
            pose = np.eye(4, dtype=np.float32) if state is None else np.asarray(state.pose.T_world_cam, dtype=np.float32)
            sources.append({
                "frame_key": frame_key,
                "time_sec": float(frame.timestamp_sec),
                "image_path": str(image_path),
                "points_xyz": masked_points,
                "confidence": None if conf_arr is None else np.where(valid, conf_arr, 0.0).astype(np.float32),
                "pose": pose.astype(np.float32),
                "intrinsics": self._intrinsics_to_dict(intr),
            })
        return sources

    def _build_render_contract(self, static_scene: dict[str, Any], humans: list[dict[str, Any]], objects: list[dict[str, Any]], track_artifact, geometry_artifact, human_artifact=None, object_artifact=None) -> dict[str, Any]:
        static_points = np.asarray(static_scene.get("points_xyz", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        static_colors = np.asarray(static_scene.get("colors_rgb", np.zeros((0, 3), dtype=np.uint8)), dtype=np.uint8)
        if static_points.size:
            extent = np.ptp(static_points, axis=0)
            scene_scale = float(np.linalg.norm(extent))
            base_scale = float(np.clip(scene_scale / max(40.0, np.sqrt(float(static_points.shape[0]) * 10.0)), 0.01, 0.08))
            bg_scales = np.full_like(static_points, base_scale, dtype=np.float32)
            bg_opacities = np.full((static_points.shape[0],), 0.92, dtype=np.float32)
            bg_codes = np.zeros((static_points.shape[0],), dtype=np.int16)
        else:
            bg_scales = np.zeros((0, 3), dtype=np.float32)
            bg_opacities = np.zeros((0,), dtype=np.float32)
            bg_codes = np.zeros((0,), dtype=np.int16)

        static_tracks = self._static_track_gaussians(track_artifact)
        means = np.concatenate([static_points, static_tracks["means_xyz"]], axis=0) if static_tracks["means_xyz"].size else static_points
        scales = np.concatenate([bg_scales, static_tracks["scales_xyz"]], axis=0) if static_tracks["scales_xyz"].size else bg_scales
        colors = np.concatenate([static_colors, static_tracks["colors_rgb"]], axis=0) if static_tracks["colors_rgb"].size else static_colors
        opacities = np.concatenate([bg_opacities, static_tracks["opacities"]], axis=0) if static_tracks["opacities"].size else bg_opacities
        source_codes = np.concatenate([bg_codes, static_tracks["source_codes"]], axis=0) if static_tracks["source_codes"].size else bg_codes

        dynamic_layers: dict[str, list[dict[str, Any]]] = {"humans": [], "objects": []}
        for kind, entities in (("humans", humans), ("objects", objects)):
            template_offsets, template_scales, template_opacities = self._template_offsets("human" if kind == "humans" else "object")
            for ent in entities:
                centers = []
                for center in ent.get("centers_3d", []):
                    if center is None:
                        centers.append(list(ent.get("anchor_xyz", [0.0, 0.0, 0.0])))
                    else:
                        centers.append(list(np.asarray(center, dtype=np.float32).tolist()))
                dynamic_layers[kind].append({
                    "entity_id": ent["entity_id"],
                    "entity_type": ent["entity_type"],
                    "display_label": ent.get("display_label", ent["entity_type"]),
                    "color_rgb": list(ent.get("color_rgb", [255, 80, 80])),
                    "times_sec": [float(t) for t in ent.get("times_sec", [])],
                    "centers_xyz": centers,
                    "anchor_xyz": list(ent.get("anchor_xyz", [0.0, 0.0, 0.0])),
                    "support_counts": [int(v) for v in ent.get("support_counts", [])],
                    "template_offsets_xyz": template_offsets.tolist(),
                    "template_scales_xyz": template_scales.tolist(),
                    "template_opacities": template_opacities.tolist(),
                    "observations": list(ent.get("observations", [])),
                })

        static_sources = self._build_static_appearance_sources(geometry_artifact, human_artifact, object_artifact)
        return {
            "schema": "layered_scene_v2",
            "supported_backends": ["gaussian_layered", "gaussian4d", "offline_high_quality"],
            "quality_modes": ["realtime", "offline_high_quality"],
            "world_frame": {"up_axis": "z", "gravity_dir": [0.0, 0.0, -1.0]},
            "static_layer": {
                "layer_name": "static_3dgs_seed",
                "num_gaussians": int(means.shape[0]),
                "means_xyz": means.astype(np.float32),
                "scales_xyz": scales.astype(np.float32),
                "colors_rgb": colors.astype(np.uint8),
                "opacities": opacities.astype(np.float32),
                "source_codes": source_codes.astype(np.int16),
                "source_counts": {
                    "background_points": int(static_points.shape[0]),
                    "static_track_points": int(static_tracks["means_xyz"].shape[0]),
                },
            },
            "appearance": {
                "static_sources": static_sources,
                "static_source_count": int(len(static_sources)),
                "dynamic_observation_counts": {
                    "humans": int(sum(len(ent.get("observations", [])) for ent in dynamic_layers["humans"])),
                    "objects": int(sum(len(ent.get("observations", [])) for ent in dynamic_layers["objects"])),
                },
            },
            "dynamic_layers": dynamic_layers,
            "scene_summary": {
                "num_human_layers": int(len(dynamic_layers["humans"])),
                "num_object_layers": int(len(dynamic_layers["objects"])),
            },
        }

    def assemble(self, *, run_id: str, save_dir, geometry_artifact, track_artifact, human_artifact=None, object_artifact=None, metadata: dict[str, Any] | None = None) -> SceneArtifact:
        static_scene = self._assemble_static_background(geometry_artifact, human_artifact, object_artifact)
        humans = [] if human_artifact is None else self._assemble_entities(human_artifact.humans, "humans", track_artifact, geometry_artifact, source_artifact=human_artifact)
        objects = [] if object_artifact is None else self._assemble_entities(object_artifact.objects, "objects", track_artifact, geometry_artifact, source_artifact=object_artifact)
        requested_timeline = str((metadata or {}).get("timeline_source", self.timeline_source)).lower()
        if requested_timeline not in {"geometry", "tracking"}:
            requested_timeline = self.timeline_source
        if requested_timeline == "tracking":
            selected = list(track_artifact.selected_frames or [])
            if not selected:
                selected = list(geometry_artifact.selected_frames)
                requested_timeline = "geometry"
        else:
            selected = list(geometry_artifact.selected_frames)
        frame_refs = sorted(selected, key=lambda fr: (fr.stream_id, int(fr.frame_idx)))
        timeline = {
            "source": requested_timeline,
            "time_min_sec": float(min((fr.timestamp_sec for fr in frame_refs), default=0.0)),
            "time_max_sec": float(max((fr.timestamp_sec for fr in frame_refs), default=0.0)),
            "frame_keys": [self._frame_key(fr.stream_id, fr.frame_idx) for fr in frame_refs],
        }
        render_contract = self._build_render_contract(static_scene, humans, objects, track_artifact, geometry_artifact, human_artifact=human_artifact, object_artifact=object_artifact)
        notes = [
            f"static_points={static_scene['num_points']}",
            f"static_gaussians={render_contract['static_layer']['num_gaussians']}",
            f"static_sources={render_contract.get('appearance', {}).get('static_source_count', 0)}",
            f"humans={len(humans)}",
            f"objects={len(objects)}",
        ]
        dynamic_entities_light = {
            "humans": [{k: v for k, v in ent.items() if k != "observations"} for ent in humans],
            "objects": [{k: v for k, v in ent.items() if k != "observations"} for ent in objects],
        }
        return SceneArtifact(
            run_id=run_id,
            stage_name="scene",
            save_dir=save_dir,
            metadata=metadata or {},
            static_scene=static_scene,
            dynamic_entities=dynamic_entities_light,
            frame_refs=frame_refs,
            timeline=timeline,
            render_contract=render_contract,
            notes=notes,
        )
