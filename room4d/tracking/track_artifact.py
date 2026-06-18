"""Artifact for 2D tracks, lifted 3D samples, and tracking debug exports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import colorsys
import shutil

import numpy as np
from PIL import Image
from rich.progress import track

from room4d.common.io import read_json, write_json
from room4d.common.paths import ensure_dir
from room4d.common.types import BaseArtifact, FrameRef, TrackObservation
from room4d.geometry.geometry_artifact import GeometryArtifact
from room4d.tracking.all_pairs import AllPairsTracks


@dataclass
class TrackArtifact(BaseArtifact):
    """Store per-stream 2D tracks and lifted 3D correspondences.

    Heavy per-stream arrays are saved to ``tracks/<stream_id>.npz`` while the
    lightweight metadata index stays in ``artifact.json``. The artifact can also
    emit optional debug assets under ``images/`` and ``tracking_debug_viewer.html``.
    """

    per_stream_observations: dict[str, list[TrackObservation]] = field(default_factory=dict)
    seed_points_by_stream: dict[str, Any] = field(default_factory=dict)
    merged_track_groups: dict[str, list[str]] = field(default_factory=dict)
    visibility_masks: dict[str, Any] = field(default_factory=dict)
    static_dynamic_hints: dict[str, Any] = field(default_factory=dict)
    track_payloads_by_stream: dict[str, dict[str, Any]] = field(default_factory=dict)
    # All-pairs 2D tracks (SoM 2d_tracks format) the native bootstapir tracker
    # produces and native_track_lift consumes. Distinct from the single-query
    # track_payloads_by_stream above; empty for the CoTracker/scaffold path.
    all_pairs_by_stream: dict[str, AllPairsTracks] = field(default_factory=dict)
    selected_frames: list[FrameRef] = field(default_factory=list)
    depthmaps_by_frame: dict[str, Any] = field(default_factory=dict)
    pointmaps_by_frame: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def stream_ids(self) -> list[str]:
        """Return all streams that contributed observations or payloads."""
        keys = set(self.per_stream_observations) | set(self.track_payloads_by_stream)
        return sorted(keys)

    def total_observations(self) -> int:
        """Return the total number of 2D point observations."""
        return sum(len(obs) for obs in self.per_stream_observations.values())

    def write_2d_tracks(self, out_dir: Path | str, stream_id: str | None = None) -> Path:
        """Emit a stream's all-pairs tracks in the SoM ``2d_tracks`` layout.

        Writes ``<out_dir>/<name_i>_<name_j>.npy`` (one file per ordered frame pair)
        that :func:`room4d.lifting.native_track_lift.lift_tracks_3d` reads directly.
        ``out_dir`` is the ``<factor>x`` directory itself. With one all-pairs stream,
        ``stream_id`` may be omitted; otherwise it must be given.
        """
        if not self.all_pairs_by_stream:
            raise ValueError("artifact has no all-pairs tracks to write")
        if stream_id is None:
            if len(self.all_pairs_by_stream) != 1:
                raise ValueError(
                    "stream_id is required when the artifact has multiple all-pairs streams: "
                    f"{sorted(self.all_pairs_by_stream)}"
                )
            stream_id = next(iter(self.all_pairs_by_stream))
        return self.all_pairs_by_stream[stream_id].write(out_dir)

    def _progress_iter(self, iterable, *, total: int | None = None, description: str = ""):
        enabled = bool(self.metadata.get('save_debug_assets', False) or self.metadata.get('save_debug_html', False))
        if not enabled:
            return iterable
        return track(iterable, total=total, description=description)

    @staticmethod
    def _frame_ref_to_dict(frame: FrameRef) -> dict[str, Any]:
        return asdict(frame)

    @staticmethod
    def _frame_ref_from_dict(raw: dict[str, Any]) -> FrameRef:
        return FrameRef(**raw)

    @staticmethod
    def _sanitize_key(key: str) -> str:
        return key.replace(':', '__').replace('/', '_')

    @staticmethod
    def _label_color(label: int) -> str:
        if label > 0:
            return '#2ca02c'  # static = green
        if label == 0:
            return '#d62728'  # dynamic = red
        return '#ff7f0e'      # uncertain = orange

    @staticmethod
    def _label_name(label: int) -> str:
        if label > 0:
            return 'static'
        if label == 0:
            return 'dynamic'
        return 'uncertain'

    @staticmethod
    def _label_symbol(label: int) -> str:
        if label > 0:
            return 'circle'
        if label == 0:
            return 'x'
        return 'diamond'

    @staticmethod
    def _track_color(track_id: int) -> str:
        hue = ((int(track_id) * 0.6180339887498949) % 1.0)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


    @staticmethod
    def _sample_geometry_debug_cloud(
        point_payload: dict[str, Any],
        *,
        rgb: np.ndarray | None,
        max_points: int,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Sample a geometry point map for debug visualization.

        This wrapper tolerates older and newer GeometryArtifact helper
        signatures by falling back to the public geometry normalization helpers
        if the direct sampler lookup is not callable in the current form.
        """
        kwargs = {
            'rgb': rgb,
            'max_points': max_points,
            'confidence': point_payload.get('confidence'),
        }
        sample_fn = GeometryArtifact._sample_point_cloud
        try:
            return sample_fn(point_payload['points'], **kwargs)
        except TypeError as exc:
            message = str(exc)
            if 'points' not in message:
                raise
            points = GeometryArtifact._points_to_debug_basis(
                GeometryArtifact._ensure_hwc_points(point_payload['points'])
            )
            h, w, _ = points.shape
            pts = points.reshape(-1, 3)
            valid = np.isfinite(pts).all(axis=1)
            conf = kwargs['confidence']
            if conf is not None:
                conf_arr = np.asarray(conf).reshape(-1)
                if conf_arr.shape[0] == pts.shape[0]:
                    valid &= np.isfinite(conf_arr)
                    valid &= conf_arr > 0
            cols = None
            if rgb is not None:
                rgb_arr = np.asarray(rgb)
                if rgb_arr.ndim == 3 and rgb_arr.shape[-1] == 3:
                    if rgb_arr.shape[:2] != (h, w):
                        rgb_arr = np.asarray(
                            Image.fromarray(rgb_arr.astype(np.uint8)).resize((w, h), resample=Image.BILINEAR)
                        )
                    flat_rgb = rgb_arr.reshape(-1, 3)
                    if flat_rgb.shape[0] == pts.shape[0]:
                        cols = flat_rgb[valid]
            pts = pts[valid]
            if pts.shape[0] > max_points:
                stride = int(np.ceil(pts.shape[0] / max_points))
                pts = pts[::stride]
                if cols is not None:
                    cols = cols[::stride]
            return pts.astype(np.float32), None if cols is None else cols.astype(np.uint8)


    @staticmethod
    def _observations_from_payload(stream_id: str, payload: dict[str, Any]) -> list[TrackObservation]:
        frame_indices = np.asarray(payload.get('frame_indices', []), dtype=int)
        xy = np.asarray(payload.get('points_xy', np.zeros((0, 0, 2), dtype=np.float32)), dtype=np.float32)
        visibility = np.asarray(payload.get('visibility', np.zeros((0, 0), dtype=bool)))
        confidence = np.asarray(payload.get('confidence', np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
        track_ids = np.asarray(payload.get('track_ids', np.arange(xy.shape[1] if xy.ndim == 3 else 0)), dtype=int)
        observations: list[TrackObservation] = []
        if xy.ndim != 3:
            return observations
        for t, frame_idx in enumerate(frame_indices.tolist()):
            for n, track_id in enumerate(track_ids.tolist()):
                observations.append(
                    TrackObservation(
                        stream_id=stream_id,
                        frame_idx=int(frame_idx),
                        track_id=int(track_id),
                        xy=np.asarray(xy[t, n], dtype=np.float32),
                        visible=bool(visibility[t, n]) if visibility.size else True,
                        confidence=float(confidence[t, n]) if confidence.size else 1.0,
                    )
                )
        return observations

    def _copy_debug_images(self, image_dir: Path) -> dict[str, str]:
        copied: dict[str, str] = {}
        source_map = self.metadata.get('debug_image_paths', {})
        if not source_map:
            return copied
        ensure_dir(image_dir)
        for frame_key, source in self._progress_iter(source_map.items(), total=len(source_map), description="[cyan]Tracking · copying debug RGB"):
            src = Path(source)
            if not src.exists():
                continue
            dst = image_dir / f"{self._sanitize_key(frame_key)}{src.suffix or '.png'}"
            shutil.copy2(src, dst)
            copied[frame_key] = dst.name
        return copied

    def _track_state_for_frame(self, stream_id: str, frame_idx: int) -> dict[str, np.ndarray]:
        payload = self.track_payloads_by_stream.get(stream_id, {})
        points_xy = np.asarray(payload.get('points_xy', np.zeros((0, 0, 2), dtype=np.float32)), dtype=np.float32)
        visibility = np.asarray(payload.get('visibility', np.zeros((0, 0), dtype=bool)))
        confidence = np.asarray(payload.get('confidence', np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
        track_ids = np.asarray(payload.get('track_ids', np.arange(points_xy.shape[1] if points_xy.ndim == 3 else 0)), dtype=int)
        labels = np.asarray(payload.get('static_labels', np.full((track_ids.shape[0],), -1, dtype=np.int8)), dtype=np.int8)
        scores = np.asarray(payload.get('static_scores', np.full((track_ids.shape[0],), np.nan, dtype=np.float32)), dtype=np.float32)
        frame_indices = np.asarray(payload.get('frame_indices', []), dtype=int)
        if points_xy.ndim != 3 or frame_indices.size == 0:
            return {
                'points_xy': np.zeros((0, 2), dtype=np.float32),
                'track_ids': np.zeros((0,), dtype=int),
                'labels': np.zeros((0,), dtype=np.int8),
                'scores': np.zeros((0,), dtype=np.float32),
                'confidence': np.zeros((0,), dtype=np.float32),
            }
        matches = np.where(frame_indices == int(frame_idx))[0]
        if matches.size == 0:
            return {
                'points_xy': np.zeros((0, 2), dtype=np.float32),
                'track_ids': np.zeros((0,), dtype=int),
                'labels': np.zeros((0,), dtype=np.int8),
                'scores': np.zeros((0,), dtype=np.float32),
                'confidence': np.zeros((0,), dtype=np.float32),
            }
        t = int(matches[0])
        visible = visibility[t].astype(bool) if visibility.size else np.ones((points_xy.shape[1],), dtype=bool)
        return {
            'points_xy': points_xy[t][visible],
            'track_ids': track_ids[visible],
            'labels': labels[visible] if labels.size else np.full((visible.sum(),), -1, dtype=np.int8),
            'scores': scores[visible] if scores.size else np.full((visible.sum(),), np.nan, dtype=np.float32),
            'confidence': confidence[t][visible] if confidence.size else np.ones((visible.sum(),), dtype=np.float32),
        }

    def _lifted_state_for_frame(self, stream_id: str, frame_idx: int) -> dict[str, np.ndarray]:
        payload = self.track_payloads_by_stream.get(stream_id, {})
        lifted = np.asarray(payload.get('lifted_xyz', np.zeros((0, 0, 3), dtype=np.float32)), dtype=np.float32)
        lifted_frame_indices = np.asarray(payload.get('lifted_frame_indices', []), dtype=int)
        lifted_valid = np.asarray(payload.get('lifted_valid', np.zeros((0, 0), dtype=bool)))
        track_ids = np.asarray(payload.get('track_ids', np.arange(lifted.shape[1] if lifted.ndim == 3 else 0)), dtype=int)
        labels = np.asarray(payload.get('static_labels', np.full((track_ids.shape[0],), -1, dtype=np.int8)), dtype=np.int8)
        scores = np.asarray(payload.get('static_scores', np.full((track_ids.shape[0],), np.nan, dtype=np.float32)), dtype=np.float32)
        if lifted.ndim != 3 or lifted_frame_indices.size == 0:
            return {
                'points_xyz': np.zeros((0, 3), dtype=np.float32),
                'track_ids': np.zeros((0,), dtype=int),
                'labels': np.zeros((0,), dtype=np.int8),
                'scores': np.zeros((0,), dtype=np.float32),
            }
        matches = np.where(lifted_frame_indices == int(frame_idx))[0]
        if matches.size == 0:
            return {
                'points_xyz': np.zeros((0, 3), dtype=np.float32),
                'track_ids': np.zeros((0,), dtype=int),
                'labels': np.zeros((0,), dtype=np.int8),
                'scores': np.zeros((0,), dtype=np.float32),
            }
        k = int(matches[0])
        valid = lifted_valid[k].astype(bool) if lifted_valid.size else np.isfinite(lifted[k]).all(axis=1)
        pts = GeometryArtifact._points_to_z_up(lifted[k])
        return {
            'points_xyz': pts[valid],
            'track_ids': track_ids[valid],
            'labels': labels[valid] if labels.size else np.full((valid.sum(),), -1, dtype=np.int8),
            'scores': scores[valid] if scores.size else np.full((valid.sum(),), np.nan, dtype=np.float32),
        }

    @staticmethod
    def _sample_track_ids(track_ids: np.ndarray, max_tracks: int) -> np.ndarray:
        if track_ids.size <= max_tracks:
            return track_ids.astype(int)
        if max_tracks <= 0:
            return np.zeros((0,), dtype=int)
        idx = np.linspace(0, track_ids.size - 1, max_tracks, dtype=int)
        return track_ids[idx].astype(int)

    def _trajectory_2d_for_track(self, stream_id: str, track_id: int, frame_idx: int, frame_window: int) -> np.ndarray:
        payload = self.track_payloads_by_stream.get(stream_id, {})
        frame_indices = np.asarray(payload.get('frame_indices', []), dtype=int)
        points_xy = np.asarray(payload.get('points_xy', np.zeros((0, 0, 2), dtype=np.float32)), dtype=np.float32)
        visibility = np.asarray(payload.get('visibility', np.zeros((0, 0), dtype=bool)))
        track_ids = np.asarray(payload.get('track_ids', []), dtype=int)
        if points_xy.ndim != 3 or frame_indices.size == 0 or track_ids.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        matches = np.where(track_ids == int(track_id))[0]
        if matches.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        tidx = int(matches[0])
        mask = (frame_indices >= int(frame_idx) - frame_window) & (frame_indices <= int(frame_idx) + frame_window)
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        vis = visibility[idxs, tidx].astype(bool) if visibility.size else np.ones((idxs.size,), dtype=bool)
        return points_xy[idxs, tidx][vis]

    def _trajectory_3d_for_track(self, stream_id: str, track_id: int, frame_idx: int, frame_window: int) -> np.ndarray:
        payload = self.track_payloads_by_stream.get(stream_id, {})
        frame_indices = np.asarray(payload.get('lifted_frame_indices', []), dtype=int)
        lifted_xyz = np.asarray(payload.get('lifted_xyz', np.zeros((0, 0, 3), dtype=np.float32)), dtype=np.float32)
        lifted_valid = np.asarray(payload.get('lifted_valid', np.zeros((0, 0), dtype=bool)))
        track_ids = np.asarray(payload.get('track_ids', []), dtype=int)
        if lifted_xyz.ndim != 3 or frame_indices.size == 0 or track_ids.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        matches = np.where(track_ids == int(track_id))[0]
        if matches.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        tidx = int(matches[0])
        mask = (frame_indices >= int(frame_idx) - frame_window) & (frame_indices <= int(frame_idx) + frame_window)
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        vis = lifted_valid[idxs, tidx].astype(bool) if lifted_valid.size else np.isfinite(lifted_xyz[idxs, tidx]).all(axis=1)
        pts = GeometryArtifact._points_to_z_up(lifted_xyz[idxs, tidx])
        return pts[vis]

    def _summary_rows_for_frame(self, stream_id: str, frame_idx: int, track_state: dict[str, np.ndarray], lifted_state: dict[str, np.ndarray]) -> tuple[list[str], list[str]]:
        payload = self.track_payloads_by_stream.get(stream_id, {})
        global_track_count = int(np.asarray(payload.get('track_ids', [])).shape[0])
        labels_all = np.asarray(payload.get('static_labels', np.zeros((0,), dtype=np.int8)), dtype=np.int8)
        counts_all = {
            'static': int(np.sum(labels_all > 0)),
            'dynamic': int(np.sum(labels_all == 0)),
            'uncertain': int(np.sum(labels_all < 0)),
        }
        labels_visible = np.asarray(track_state.get('labels', np.zeros((0,), dtype=np.int8)), dtype=np.int8)
        counts_visible = {
            'static': int(np.sum(labels_visible > 0)),
            'dynamic': int(np.sum(labels_visible == 0)),
            'uncertain': int(np.sum(labels_visible < 0)),
        }
        windowing = payload.get('windowing', {}) or {}
        metrics = [
            'Backend',
            'Global tracks',
            'Visible 2D tracks',
            'Visible lifted tracks',
            'Static / dynamic / uncertain',
            'Visible static / dynamic / uncertain',
            'Windowed',
            'Stitched',
            'Matched pairs',
        ]
        values = [
            str(payload.get('backend_used', 'unknown')),
            str(global_track_count),
            str(int(np.asarray(track_state.get('track_ids', [])).shape[0])),
            str(int(np.asarray(lifted_state.get('track_ids', [])).shape[0])),
            f"{counts_all['static']} / {counts_all['dynamic']} / {counts_all['uncertain']}",
            f"{counts_visible['static']} / {counts_visible['dynamic']} / {counts_visible['uncertain']}",
            str(bool(windowing.get('windowed', False))),
            str(bool(windowing.get('stitched', False))),
            str(int(windowing.get('matched_pairs', 0))),
        ]
        return metrics, values

    def _write_plotly_debug_html(self, *, image_index: dict[str, str]) -> None:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        frame_specs: list[dict[str, Any]] = []
        max_points = int(self.metadata.get('debug_max_points', 5000))
        max_trails = int(self.metadata.get('debug_max_trails', 12))
        trail_window = int(self.metadata.get('debug_trail_window', 12))
        for frame in self._progress_iter(self.selected_frames, total=len(self.selected_frames), description="[cyan]Tracking · building debug viewer"):
            frame_key = f"{frame.stream_id}:{frame.frame_idx}"
            image_name = image_index.get(frame_key)
            depth_payload = self.depthmaps_by_frame.get(frame_key)
            point_payload = self.pointmaps_by_frame.get(frame_key)
            if image_name is None or depth_payload is None or point_payload is None:
                continue
            rgb = np.asarray(Image.open(self.save_dir / 'images' / image_name).convert('RGB'))
            depth_preview = GeometryArtifact._depth_to_preview_image(depth_payload['depth'])
            depth_preview = np.repeat(depth_preview[..., None], 3, axis=2)
            bg_pts, bg_cols = self._sample_geometry_debug_cloud(
                point_payload,
                rgb=rgb,
                max_points=max_points,
            )
            if bg_pts.size == 0:
                bg_pts = np.zeros((1, 3), dtype=np.float32)
                bg_cols = np.array([[128, 128, 128]], dtype=np.uint8)

            track_state = self._track_state_for_frame(frame.stream_id, frame.frame_idx)
            lifted_state = self._lifted_state_for_frame(frame.stream_id, frame.frame_idx)
            sampled_ids = self._sample_track_ids(np.asarray(track_state['track_ids'], dtype=int), max_trails)
            trail_specs: list[dict[str, Any]] = []
            for track_id in sampled_ids.tolist():
                matches_2d = np.where(track_state['track_ids'] == track_id)[0]
                current_xy = track_state['points_xy'][matches_2d[0]] if matches_2d.size else np.zeros((0, 2), dtype=np.float32)
                current_label = int(track_state['labels'][matches_2d[0]]) if matches_2d.size else -1
                current_score = float(track_state['scores'][matches_2d[0]]) if matches_2d.size else float('nan')
                traj_2d = self._trajectory_2d_for_track(frame.stream_id, track_id, frame.frame_idx, trail_window)
                matches_3d = np.where(lifted_state['track_ids'] == track_id)[0]
                current_xyz = lifted_state['points_xyz'][matches_3d[0]] if matches_3d.size else np.zeros((0, 3), dtype=np.float32)
                traj_3d = self._trajectory_3d_for_track(frame.stream_id, track_id, frame.frame_idx, trail_window)
                trail_specs.append({
                    'track_id': int(track_id),
                    'color': self._track_color(int(track_id)),
                    'label': current_label,
                    'score': current_score,
                    'current_xy': current_xy,
                    'traj_2d': traj_2d,
                    'current_xyz': current_xyz,
                    'traj_3d': traj_3d,
                })
            summary_metrics, summary_values = self._summary_rows_for_frame(frame.stream_id, frame.frame_idx, track_state, lifted_state)
            frame_specs.append(
                {
                    'frame_ref': frame,
                    'rgb': rgb,
                    'depth_preview': depth_preview,
                    'background_points': bg_pts,
                    'background_colors': bg_cols,
                    'track_state': track_state,
                    'lifted_state': lifted_state,
                    'trail_specs': trail_specs,
                    'summary_metrics': summary_metrics,
                    'summary_values': summary_values,
                }
            )

        if not frame_specs:
            return

        fig = make_subplots(
            rows=2,
            cols=3,
            specs=[
                [{'type': 'xy'}, {'type': 'image'}, {'type': 'table'}],
                [{'type': 'scene', 'colspan': 3}, None, None],
            ],
            subplot_titles=('RGB + Tracks', 'Depth', 'Summary', 'Point Cloud + Trajectories'),
            row_heights=[0.40, 0.60],
            column_widths=[0.38, 0.27, 0.35],
            horizontal_spacing=0.04,
            vertical_spacing=0.08,
        )

        slider_steps = []
        traces_per_frame = 7 + (2 * max_trails)
        total_traces = 0
        for idx, spec in enumerate(frame_specs):
            visible = idx == 0
            fr = spec['frame_ref']
            name = f"{fr.stream_id}:{fr.frame_idx}"
            track_state = spec['track_state']
            lifted_state = spec['lifted_state']
            track_ids = np.asarray(track_state['track_ids'], dtype=int)
            marker_colors_2d = [self._track_color(int(tid)) for tid in track_ids.tolist()]
            marker_symbols_2d = [self._label_symbol(int(v)) for v in np.asarray(track_state['labels'], dtype=np.int8).tolist()]
            hover_text_2d = [
                f"track_id={int(tid)}<br>motion={self._label_name(int(lbl))}<br>score={float(score):.3f}"
                for tid, lbl, score in zip(track_ids.tolist(), np.asarray(track_state['labels']).tolist(), np.asarray(track_state['scores']).tolist())
            ]
            lifted_ids = np.asarray(lifted_state['track_ids'], dtype=int)
            lifted_colors = [self._track_color(int(tid)) for tid in lifted_ids.tolist()]
            hover_text_3d = [
                f"track_id={int(tid)}<br>motion={self._label_name(int(lbl))}<br>score={float(score):.3f}"
                for tid, lbl, score in zip(lifted_ids.tolist(), np.asarray(lifted_state['labels']).tolist(), np.asarray(lifted_state['scores']).tolist())
            ]

            fig.add_trace(go.Image(z=spec['rgb'], visible=visible, name=f"rgb:{name}"), row=1, col=1)
            fig.add_trace(
                go.Scatter(
                    x=track_state['points_xy'][:, 0] if track_state['points_xy'].size else [],
                    y=track_state['points_xy'][:, 1] if track_state['points_xy'].size else [],
                    mode='markers',
                    marker={
                        'size': 7,
                        'color': marker_colors_2d,
                        'symbol': marker_symbols_2d,
                        'line': {'color': 'white', 'width': 0.6},
                    },
                    hovertext=hover_text_2d,
                    hoverinfo='text',
                    name=f"tracks:{name}",
                    visible=visible,
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(go.Image(z=spec['depth_preview'], visible=visible, name=f"depth:{name}"), row=1, col=2)
            fig.add_trace(
                go.Table(
                    header={'values': ['Metric', 'Value'], 'align': 'left'},
                    cells={'values': [spec['summary_metrics'], spec['summary_values']], 'align': 'left'},
                    visible=visible,
                    name=f"summary:{name}",
                ),
                row=1,
                col=3,
            )
            marker_bg = {
                'size': 2,
                'opacity': 0.22,
                'color': [f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in spec['background_colors']],
            }
            fig.add_trace(
                go.Scatter3d(
                    x=spec['background_points'][:, 0],
                    y=spec['background_points'][:, 1],
                    z=spec['background_points'][:, 2],
                    mode='markers',
                    marker=marker_bg,
                    name=f"pointcloud:{name}",
                    visible=visible,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter3d(
                    x=lifted_state['points_xyz'][:, 0] if lifted_state['points_xyz'].size else [],
                    y=lifted_state['points_xyz'][:, 1] if lifted_state['points_xyz'].size else [],
                    z=lifted_state['points_xyz'][:, 2] if lifted_state['points_xyz'].size else [],
                    mode='markers',
                    marker={
                        'size': 5,
                        'opacity': 0.95,
                        'color': lifted_colors,
                        'line': {'color': 'white', 'width': 1},
                    },
                    hovertext=hover_text_3d,
                    hoverinfo='text',
                    name=f"lifted:{name}",
                    visible=visible,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            for trail_idx in range(max_trails):
                if trail_idx < len(spec['trail_specs']):
                    trail = spec['trail_specs'][trail_idx]
                    color = trail['color']
                    label_name = self._label_name(int(trail['label']))
                    trail2d = trail['traj_2d']
                    current_xy = np.asarray(trail['current_xy'])
                    trail3d = trail['traj_3d']
                    current_xyz = np.asarray(trail['current_xyz'])
                    hover_name = f"track_id={trail['track_id']}<br>motion={label_name}<br>score={trail['score']:.3f}"
                    fig.add_trace(
                        go.Scatter(
                            x=trail2d[:, 0] if trail2d.size else [],
                            y=trail2d[:, 1] if trail2d.size else [],
                            mode='lines',
                            line={'color': color, 'width': 2},
                            opacity=0.85,
                            hoverinfo='skip',
                            name=f"trail2d:{name}:{trail['track_id']}",
                            visible=visible,
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter3d(
                            x=trail3d[:, 0] if trail3d.size else [],
                            y=trail3d[:, 1] if trail3d.size else [],
                            z=trail3d[:, 2] if trail3d.size else [],
                            mode='lines+markers',
                            line={'color': color, 'width': 5},
                            marker={'size': 3, 'color': color},
                            opacity=0.9,
                            hovertext=[hover_name] * max(int(trail3d.shape[0]), 1),
                            hoverinfo='text',
                            name=f"trail3d:{name}:{trail['track_id']}",
                            visible=visible,
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )
                else:
                    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', visible=visible, showlegend=False, hoverinfo='skip', name=f"trail2d_empty:{name}:{trail_idx}"), row=1, col=1)
                    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', visible=visible, showlegend=False, hoverinfo='skip', name=f"trail3d_empty:{name}:{trail_idx}"), row=2, col=1)
            total_traces += traces_per_frame

        for idx, spec in enumerate(frame_specs):
            fr = spec['frame_ref']
            name = f"{fr.stream_id}:{fr.frame_idx}"
            visible = [False] * total_traces
            base = traces_per_frame * idx
            visible[base:base + traces_per_frame] = [True] * traces_per_frame
            slider_steps.append(
                {
                    'method': 'update',
                    'label': name,
                    'args': [
                        {'visible': visible},
                        {'title': {'text': f"Tracking debug | keyframe {name} | t={fr.timestamp_sec:.3f}s"}},
                    ],
                }
            )

        first = frame_specs[0]['frame_ref']
        fig.update_layout(
            title={'text': f"Tracking debug | keyframe {first.stream_id}:{first.frame_idx} | t={first.timestamp_sec:.3f}s"},
            template='plotly_white',
            width=1750,
            height=1050,
            scene={
                'xaxis_title': 'X',
                'yaxis_title': 'Y',
                'zaxis_title': 'Z (up)',
                'aspectmode': 'data',
            },
            margin={'l': 20, 'r': 20, 't': 60, 'b': 20},
            sliders=[{
                'active': 0,
                'currentvalue': {'prefix': 'Keyframe: '},
                'pad': {'t': 20},
                'steps': slider_steps,
            }],
        )
        fig.update_xaxes(visible=False, showticklabels=False, row=1, col=1)
        fig.update_yaxes(visible=False, showticklabels=False, autorange='reversed', scaleanchor='x', row=1, col=1)
        fig.update_xaxes(visible=False, showticklabels=False, row=1, col=2)
        fig.update_yaxes(visible=False, showticklabels=False, row=1, col=2)
        html = fig.to_html(full_html=True, include_plotlyjs=True)
        (self.save_dir / 'tracking_debug_viewer.html').write_text(html, encoding='utf-8')

    def save(self) -> None:
        """Persist track metadata, heavy arrays, and optional debug assets."""
        ensure_dir(self.save_dir)
        track_dir = ensure_dir(self.save_dir / 'tracks')
        depth_dir = ensure_dir(self.save_dir / 'depthmaps') if self.depthmaps_by_frame else None
        point_dir = ensure_dir(self.save_dir / 'pointmaps') if self.pointmaps_by_frame else None

        track_index: dict[str, str] = {}
        for stream_id, payload in self.track_payloads_by_stream.items():
            filename = f"{self._sanitize_key(stream_id)}.npz"
            np.savez_compressed(
                track_dir / filename,
                frame_indices=np.asarray(payload.get('frame_indices', []), dtype=np.int32),
                timestamps_sec=np.asarray(payload.get('timestamps_sec', []), dtype=np.float32),
                track_ids=np.asarray(payload.get('track_ids', []), dtype=np.int32),
                points_xy=np.asarray(payload.get('points_xy', np.zeros((0, 0, 2), dtype=np.float32)), dtype=np.float32),
                visibility=np.asarray(payload.get('visibility', np.zeros((0, 0), dtype=bool))),
                confidence=np.asarray(payload.get('confidence', np.zeros((0, 0), dtype=np.float32)), dtype=np.float32),
                query_frame_idx=np.asarray([payload.get('query_frame_idx', 0)], dtype=np.int32),
                original_size_wh=np.asarray(payload.get('original_size_wh', [0, 0]), dtype=np.int32),
                model_size_wh=np.asarray(payload.get('model_size_wh', [0, 0]), dtype=np.int32),
                lifted_frame_indices=np.asarray(payload.get('lifted_frame_indices', []), dtype=np.int32),
                lifted_xyz=np.asarray(payload.get('lifted_xyz', np.zeros((0, 0, 3), dtype=np.float32)), dtype=np.float32),
                lifted_valid=np.asarray(payload.get('lifted_valid', np.zeros((0, 0), dtype=bool))),
                static_labels=np.asarray(payload.get('static_labels', np.zeros((0,), dtype=np.int8)), dtype=np.int8),
                static_scores=np.asarray(payload.get('static_scores', np.zeros((0,), dtype=np.float32)), dtype=np.float32),
            )
            track_index[stream_id] = filename

        # All-pairs tracks round-trip through their canonical SoM 2d_tracks
        # layout (<name_i>_<name_j>.npy), the same format the native lift reads
        # and that is byte-comparable to the shipped borrowed tracks.
        all_pairs_index: dict[str, dict[str, Any]] = {}
        if self.all_pairs_by_stream:
            all_pairs_root = ensure_dir(self.save_dir / '2d_tracks')
            for stream_id, ap in self.all_pairs_by_stream.items():
                sub = f"{self._sanitize_key(stream_id)}/{ap.factor}x"
                ap.write(all_pairs_root / sub)
                all_pairs_index[stream_id] = {
                    'dir': f"2d_tracks/{sub}",
                    'frame_names': list(ap.frame_names),
                    'factor': int(ap.factor),
                    'image_hw': [int(ap.image_hw[0]), int(ap.image_hw[1])],
                }

        depth_index: dict[str, str] = {}
        for frame_key, payload in self.depthmaps_by_frame.items():
            if depth_dir is None:
                break
            filename = f"{self._sanitize_key(frame_key)}.npz"
            np.savez_compressed(
                depth_dir / filename,
                depth=np.asarray(payload.get('depth')),
                confidence=np.asarray(payload['confidence']) if payload.get('confidence') is not None else np.array([], dtype=np.float32),
            )
            depth_index[frame_key] = filename

        point_index: dict[str, str] = {}
        for frame_key, payload in self.pointmaps_by_frame.items():
            if point_dir is None:
                break
            filename = f"{self._sanitize_key(frame_key)}.npz"
            np.savez_compressed(
                point_dir / filename,
                points=np.asarray(payload.get('points')),
                confidence=np.asarray(payload['confidence']) if payload.get('confidence') is not None else np.array([], dtype=np.float32),
            )
            point_index[frame_key] = filename

        debug_exports: dict[str, Any] = {}
        if self.metadata.get('save_debug_assets', False):
            image_index = self._copy_debug_images(self.save_dir / 'images')
            if image_index:
                debug_exports['image_index'] = image_index
            if self.metadata.get('save_debug_html', False):
                self._write_plotly_debug_html(image_index=image_index)
                debug_exports['debug_html'] = 'tracking_debug_viewer.html'

        track_payload_meta = {
            stream_id: {
                'backend_used': payload.get('backend_used'),
                'windowing': payload.get('windowing'),
            }
            for stream_id, payload in self.track_payloads_by_stream.items()
        }
        serializable = {
            'run_id': self.run_id,
            'stage_name': self.stage_name,
            'save_dir': str(self.save_dir),
            'metadata': self.metadata,
            'seed_points_by_stream': self.seed_points_by_stream,
            'merged_track_groups': self.merged_track_groups,
            'visibility_masks': self.visibility_masks,
            'static_dynamic_hints': self.static_dynamic_hints,
            'selected_frames': [self._frame_ref_to_dict(frame) for frame in self.selected_frames],
            'notes': self.notes,
            'track_index': track_index,
            'track_payload_meta': track_payload_meta,
            'all_pairs_index': all_pairs_index,
            'depth_index': depth_index,
            'point_index': point_index,
            'debug_exports': debug_exports,
        }
        write_json(self.artifact_path(), serializable)

    @classmethod
    def load(cls, save_dir: Path) -> 'TrackArtifact':
        """Load a tracking artifact and reconstruct observations from saved arrays."""
        raw = read_json(save_dir / 'artifact.json')
        payloads: dict[str, dict[str, Any]] = {}
        observations: dict[str, list[TrackObservation]] = {}
        payload_meta = raw.get('track_payload_meta', {})
        for stream_id, filename in raw.get('track_index', {}).items():
            payload = np.load(save_dir / 'tracks' / filename)
            stream_meta = payload_meta.get(stream_id, {})
            track_payload = {
                'frame_indices': payload['frame_indices'],
                'timestamps_sec': payload['timestamps_sec'],
                'track_ids': payload['track_ids'],
                'points_xy': payload['points_xy'],
                'visibility': payload['visibility'],
                'confidence': payload['confidence'],
                'query_frame_idx': int(payload['query_frame_idx'][0]) if payload['query_frame_idx'].size else 0,
                'original_size_wh': payload['original_size_wh'],
                'model_size_wh': payload['model_size_wh'],
                'lifted_frame_indices': payload['lifted_frame_indices'],
                'lifted_xyz': payload['lifted_xyz'],
                'lifted_valid': payload['lifted_valid'],
                'static_labels': payload['static_labels'],
                'static_scores': payload['static_scores'],
                'backend_used': stream_meta.get('backend_used'),
                'windowing': stream_meta.get('windowing'),
            }
            payloads[stream_id] = track_payload
            observations[stream_id] = cls._observations_from_payload(stream_id, track_payload)

        depthmaps_by_frame: dict[str, Any] = {}
        for frame_key, filename in raw.get('depth_index', {}).items():
            payload = np.load(save_dir / 'depthmaps' / filename)
            confidence = payload['confidence']
            depthmaps_by_frame[frame_key] = {
                'depth': payload['depth'],
                'confidence': None if confidence.size == 0 else confidence,
            }

        pointmaps_by_frame: dict[str, Any] = {}
        for frame_key, filename in raw.get('point_index', {}).items():
            payload = np.load(save_dir / 'pointmaps' / filename)
            confidence = payload['confidence']
            pointmaps_by_frame[frame_key] = {
                'points': payload['points'],
                'confidence': None if confidence.size == 0 else confidence,
            }

        all_pairs_by_stream: dict[str, AllPairsTracks] = {}
        for stream_id, info in raw.get('all_pairs_index', {}).items():
            all_pairs_by_stream[stream_id] = AllPairsTracks.load(
                save_dir / info['dir'],
                list(info['frame_names']),
                int(info['factor']),
                (int(info['image_hw'][0]), int(info['image_hw'][1])),
            )

        return cls(
            run_id=raw['run_id'],
            stage_name=raw['stage_name'],
            save_dir=Path(raw['save_dir']),
            metadata=raw.get('metadata', {}),
            per_stream_observations=observations,
            seed_points_by_stream=raw.get('seed_points_by_stream', {}),
            merged_track_groups=raw.get('merged_track_groups', {}),
            visibility_masks=raw.get('visibility_masks', {}),
            static_dynamic_hints=raw.get('static_dynamic_hints', {}),
            track_payloads_by_stream=payloads,
            all_pairs_by_stream=all_pairs_by_stream,
            selected_frames=[cls._frame_ref_from_dict(frame) for frame in raw.get('selected_frames', [])],
            depthmaps_by_frame=depthmaps_by_frame,
            pointmaps_by_frame=pointmaps_by_frame,
            notes=raw.get('notes', []),
        )

    def summary(self) -> dict[str, Any]:
        return {
            'run_id': self.run_id,
            'num_streams': len(self.stream_ids()),
            'num_selected_frames': len(self.selected_frames),
            'total_observations': self.total_observations(),
            'num_lifted_streams': sum(1 for p in self.track_payloads_by_stream.values() if np.asarray(p.get('lifted_xyz', [])).size > 0),
            'notes': self.notes,
        }
