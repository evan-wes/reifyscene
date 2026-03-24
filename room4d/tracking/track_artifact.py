"""Artifact for 2D tracks, lifted 3D samples, and tracking debug exports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import shutil

import numpy as np
from PIL import Image
from rich.progress import track

from room4d.common.io import read_json, write_json
from room4d.common.paths import ensure_dir
from room4d.common.types import BaseArtifact, FrameRef, TrackObservation
from room4d.geometry.geometry_artifact import GeometryArtifact


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

    def _track_points_for_frame(self, stream_id: str, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        payload = self.track_payloads_by_stream.get(stream_id, {})
        points_xy = np.asarray(payload.get('points_xy', np.zeros((0, 0, 2), dtype=np.float32)), dtype=np.float32)
        visibility = np.asarray(payload.get('visibility', np.zeros((0, 0), dtype=bool)))
        track_ids = np.asarray(payload.get('track_ids', np.arange(points_xy.shape[1] if points_xy.ndim == 3 else 0)), dtype=int)
        labels = np.asarray(payload.get('static_labels', np.full((track_ids.shape[0],), -1, dtype=np.int8)), dtype=np.int8)
        frame_indices = np.asarray(payload.get('frame_indices', []), dtype=int)
        if points_xy.ndim != 3 or frame_indices.size == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int8)
        matches = np.where(frame_indices == int(frame_idx))[0]
        if matches.size == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int8)
        t = int(matches[0])
        visible = visibility[t].astype(bool) if visibility.size else np.ones((points_xy.shape[1],), dtype=bool)
        pts = points_xy[t]
        return pts[visible], labels[visible] if labels.size else np.full((visible.sum(),), -1, dtype=np.int8)

    def _lifted_points_for_frame(self, stream_id: str, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        payload = self.track_payloads_by_stream.get(stream_id, {})
        lifted = np.asarray(payload.get('lifted_xyz', np.zeros((0, 0, 3), dtype=np.float32)), dtype=np.float32)
        lifted_frame_indices = np.asarray(payload.get('lifted_frame_indices', []), dtype=int)
        lifted_valid = np.asarray(payload.get('lifted_valid', np.zeros((0, 0), dtype=bool)))
        track_ids = np.asarray(payload.get('track_ids', np.arange(lifted.shape[1] if lifted.ndim == 3 else 0)), dtype=int)
        labels = np.asarray(payload.get('static_labels', np.full((track_ids.shape[0],), -1, dtype=np.int8)), dtype=np.int8)
        if lifted.ndim != 3 or lifted_frame_indices.size == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int8)
        matches = np.where(lifted_frame_indices == int(frame_idx))[0]
        if matches.size == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int8)
        k = int(matches[0])
        valid = lifted_valid[k].astype(bool) if lifted_valid.size else np.isfinite(lifted[k]).all(axis=1)
        pts = GeometryArtifact._points_to_z_up(lifted[k])
        return pts[valid], labels[valid] if labels.size else np.full((valid.sum(),), -1, dtype=np.int8)

    def _write_plotly_debug_html(self, *, image_index: dict[str, str]) -> None:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        frame_specs: list[dict[str, Any]] = []
        max_points = int(self.metadata.get('debug_max_points', 5000))
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
            bg_pts, bg_cols = GeometryArtifact._sample_point_cloud(
                point_payload['points'],
                rgb=rgb,
                max_points=max_points,
                confidence=point_payload.get('confidence'),
            )
            if bg_pts.size == 0:
                bg_pts = np.zeros((1, 3), dtype=np.float32)
                bg_cols = np.array([[128, 128, 128]], dtype=np.uint8)
            overlay_xy, overlay_labels = self._track_points_for_frame(frame.stream_id, frame.frame_idx)
            lifted_pts, lifted_labels = self._lifted_points_for_frame(frame.stream_id, frame.frame_idx)
            frame_specs.append(
                {
                    'frame_ref': frame,
                    'rgb': rgb,
                    'depth_preview': depth_preview,
                    'background_points': bg_pts,
                    'background_colors': bg_cols,
                    'overlay_xy': overlay_xy,
                    'overlay_labels': overlay_labels,
                    'lifted_points': lifted_pts,
                    'lifted_labels': lifted_labels,
                }
            )

        if not frame_specs:
            return

        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{'type': 'xy'}, {'type': 'image'}], [{'type': 'scene', 'colspan': 2}, None]],
            subplot_titles=('RGB + Tracks', 'Depth', 'Point Cloud + Lifted Tracks'),
            row_heights=[0.42, 0.58],
            horizontal_spacing=0.03,
            vertical_spacing=0.08,
        )

        slider_steps = []
        total_traces = 0
        for idx, spec in enumerate(frame_specs):
            visible = idx == 0
            fr = spec['frame_ref']
            name = f"{fr.stream_id}:{fr.frame_idx}"

            fig.add_trace(go.Image(z=spec['rgb'], visible=visible, name=f"rgb:{name}"), row=1, col=1)
            marker_colors_2d = [self._label_color(int(v)) for v in spec['overlay_labels'].tolist()] if spec['overlay_labels'].size else []
            fig.add_trace(
                go.Scatter(
                    x=spec['overlay_xy'][:, 0] if spec['overlay_xy'].size else [],
                    y=spec['overlay_xy'][:, 1] if spec['overlay_xy'].size else [],
                    mode='markers',
                    marker={'size': 6, 'color': marker_colors_2d, 'line': {'color': 'white', 'width': 0.5}},
                    name=f"tracks:{name}",
                    visible=visible,
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(go.Image(z=spec['depth_preview'], visible=visible, name=f"depth:{name}"), row=1, col=2)
            marker_bg = {
                'size': 2,
                'opacity': 0.25,
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
            marker_fg = {
                'size': 5,
                'opacity': 0.95,
                'color': [self._label_color(int(v)) for v in spec['lifted_labels'].tolist()] if spec['lifted_labels'].size else [],
                'line': {'color': 'white', 'width': 1},
            }
            fig.add_trace(
                go.Scatter3d(
                    x=spec['lifted_points'][:, 0] if spec['lifted_points'].size else [],
                    y=spec['lifted_points'][:, 1] if spec['lifted_points'].size else [],
                    z=spec['lifted_points'][:, 2] if spec['lifted_points'].size else [],
                    mode='markers',
                    marker=marker_fg,
                    name=f"lifted:{name}",
                    visible=visible,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            total_traces += 5

        for idx, spec in enumerate(frame_specs):
            fr = spec['frame_ref']
            name = f"{fr.stream_id}:{fr.frame_idx}"
            visible = [False] * total_traces
            base = 5 * idx
            visible[base:base + 5] = [True, True, True, True, True]
            slider_steps.append(
                {
                    'method': 'update',
                    'label': name,
                    'args': [
                        {'visible': visible},
                        {'title': {'text': f'Tracking debug | keyframe {name} | t={fr.timestamp_sec:.3f}s'}},
                    ],
                }
            )

        first = frame_specs[0]['frame_ref']
        fig.update_layout(
            title={'text': f'Tracking debug | keyframe {first.stream_id}:{first.frame_idx} | t={first.timestamp_sec:.3f}s'},
            template='plotly_white',
            width=1500,
            height=980,
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
        for stream_id, filename in raw.get('track_index', {}).items():
            payload = np.load(save_dir / 'tracks' / filename)
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
