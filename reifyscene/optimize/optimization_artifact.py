"""Artifact for optimization results."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from reifyscene.common.frame_keys import frame_key as make_frame_key
from reifyscene.common.io import read_json, write_json
from reifyscene.common.paths import ensure_dir
from reifyscene.common.types import BaseArtifact, FrameRef, TimedCameraState
from reifyscene.geometry.geometry_artifact import GeometryArtifact
from reifyscene.tracking.track_artifact import TrackArtifact


@dataclass
class OptimizationArtifact(BaseArtifact):
    """Store refined state after joint optimization."""

    refined_cameras: dict[str, list[TimedCameraState]] = field(default_factory=dict)
    refined_offsets: dict[str, Any] = field(default_factory=dict)
    tracks_3d: dict[str, Any] = field(default_factory=dict)
    refined_track_payloads_by_stream: dict[str, dict[str, Any]] = field(default_factory=dict)
    selected_frames: list[FrameRef] = field(default_factory=list)
    object_trajectories: dict[str, Any] = field(default_factory=dict)
    human_trajectories: dict[str, Any] = field(default_factory=dict)
    stage_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    schedule_used: list[str] = field(default_factory=list)
    pending_todos: list[str] = field(default_factory=list)

    @staticmethod
    def _to_serializable(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {str(k): OptimizationArtifact._to_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [OptimizationArtifact._to_serializable(v) for v in value]
        return value

    @staticmethod
    def _sanitize_key(key: str) -> str:
        return key.replace(':', '__').replace('/', '_')

    def save(self) -> None:
        ensure_dir(self.save_dir)
        track_dir = ensure_dir(self.save_dir / 'tracks') if self.refined_track_payloads_by_stream else None
        track_index: dict[str, Any] = {}
        for stream_id, payload in self.refined_track_payloads_by_stream.items():
            if track_dir is None:
                break
            filename = f"{self._sanitize_key(stream_id)}.npz"
            np.savez_compressed(
                track_dir / filename,
                frame_indices=np.asarray(payload.get('frame_indices', []), dtype=np.int32),
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
                refined_lifted_frame_indices=np.asarray(payload.get('refined_lifted_frame_indices', []), dtype=np.int32),
                refined_lifted_xyz=np.asarray(payload.get('refined_lifted_xyz', np.zeros((0, 0, 3), dtype=np.float32)), dtype=np.float32),
                refined_lifted_valid=np.asarray(payload.get('refined_lifted_valid', np.zeros((0, 0), dtype=bool))),
                static_labels=np.asarray(payload.get('static_labels', np.zeros((0,), dtype=np.int8)), dtype=np.int8),
                static_scores=np.asarray(payload.get('static_scores', np.zeros((0,), dtype=np.float32)), dtype=np.float32),
                refined_static_labels=np.asarray(payload.get('refined_static_labels', np.zeros((0,), dtype=np.int8)), dtype=np.int8),
                refined_static_scores=np.asarray(payload.get('refined_static_scores', np.zeros((0,), dtype=np.float32)), dtype=np.float32),
            )
            track_index[stream_id] = {
                'file': filename,
                'backend_used': payload.get('backend_used', 'unknown'),
                'windowing': self._to_serializable(payload.get('windowing', {})),
                'refinement_summary': self._to_serializable(payload.get('refinement_summary', {})),
                'refined_classification_summary': self._to_serializable(payload.get('refined_classification_summary', {})),
            }
        raw = {
            'run_id': self.run_id,
            'stage_name': self.stage_name,
            'metadata': self.metadata,
            'refined_cameras': {
                stream_id: [GeometryArtifact._camera_state_to_dict(state) for state in states]
                for stream_id, states in self.refined_cameras.items()
            },
            'refined_offsets': self._to_serializable(self.refined_offsets),
            'tracks_3d': self._to_serializable(self.tracks_3d),
            'track_index': track_index,
            'selected_frames': [
                f.to_dict()
                for f in self.selected_frames
            ],
            'object_trajectories': self._to_serializable(self.object_trajectories),
            'human_trajectories': self._to_serializable(self.human_trajectories),
            'stage_diagnostics': self._to_serializable(self.stage_diagnostics),
            'schedule_used': list(self.schedule_used),
            'pending_todos': list(self.pending_todos),
        }
        write_json(self.artifact_path(), raw)

    @classmethod
    def load(cls, save_dir: Path) -> 'OptimizationArtifact':
        raw = read_json(save_dir / cls.FILE)
        refined_track_payloads_by_stream: dict[str, dict[str, Any]] = {}
        for stream_id, info in raw['track_index'].items():
            npz = np.load(save_dir / 'tracks' / info['file'], allow_pickle=True)
            refined_track_payloads_by_stream[stream_id] = {
                'frame_indices': np.asarray(npz['frame_indices'], dtype=np.int32),
                'track_ids': np.asarray(npz['track_ids'], dtype=np.int32),
                'points_xy': np.asarray(npz['points_xy'], dtype=np.float32),
                'visibility': np.asarray(npz['visibility']).astype(bool),
                'confidence': np.asarray(npz['confidence'], dtype=np.float32),
                'query_frame_idx': int(np.asarray(npz['query_frame_idx']).reshape(-1)[0]),
                'original_size_wh': np.asarray(npz['original_size_wh'], dtype=np.int32),
                'model_size_wh': np.asarray(npz['model_size_wh'], dtype=np.int32),
                'lifted_frame_indices': np.asarray(npz['lifted_frame_indices'], dtype=np.int32),
                'lifted_xyz': np.asarray(npz['lifted_xyz'], dtype=np.float32),
                'lifted_valid': np.asarray(npz['lifted_valid']).astype(bool),
                'refined_lifted_frame_indices': np.asarray(npz['refined_lifted_frame_indices'], dtype=np.int32),
                'refined_lifted_xyz': np.asarray(npz['refined_lifted_xyz'], dtype=np.float32),
                'refined_lifted_valid': np.asarray(npz['refined_lifted_valid']).astype(bool),
                'static_labels': np.asarray(npz['static_labels'], dtype=np.int8),
                'static_scores': np.asarray(npz['static_scores'], dtype=np.float32),
                'refined_static_labels': np.asarray(npz['refined_static_labels'], dtype=np.int8),
                'refined_static_scores': np.asarray(npz['refined_static_scores'], dtype=np.float32),
                'backend_used': info['backend_used'],
                'windowing': info['windowing'],
                'refinement_summary': info['refinement_summary'],
                'refined_classification_summary': info['refined_classification_summary'],
            }
        return cls(
            run_id=raw['run_id'],
            stage_name=raw['stage_name'],
            save_dir=Path(save_dir),
            metadata=raw['metadata'],
            refined_cameras={
                stream_id: [GeometryArtifact._camera_state_from_dict(state) for state in states]
                for stream_id, states in raw['refined_cameras'].items()
            },
            refined_offsets=raw['refined_offsets'],
            tracks_3d=raw['tracks_3d'],
            refined_track_payloads_by_stream=refined_track_payloads_by_stream,
            selected_frames=[FrameRef(**item) for item in raw['selected_frames']],
            object_trajectories=raw['object_trajectories'],
            human_trajectories=raw['human_trajectories'],
            stage_diagnostics=raw['stage_diagnostics'],
            schedule_used=raw['schedule_used'],
            pending_todos=raw['pending_todos'],
        )

    def export_refinement_viewer(self, track_artifact: TrackArtifact, geometry_artifact: GeometryArtifact) -> None:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if not self.metadata.get('save_refinement_viewer', False):
            return
        ensure_dir(self.save_dir)
        if not track_artifact.metadata.get('debug_image_paths') and geometry_artifact.metadata.get('debug_image_paths'):
            track_artifact.metadata['debug_image_paths'] = dict(geometry_artifact.metadata.get('debug_image_paths', {}))
        image_index = track_artifact._copy_debug_images(self.save_dir / 'images')
        refined_payloads: dict[str, dict[str, Any]] = {}
        for stream_id, payload in self.refined_track_payloads_by_stream.items():
            remapped = dict(payload)
            remapped['lifted_frame_indices'] = np.asarray(payload.get('refined_lifted_frame_indices', payload.get('lifted_frame_indices', [])), dtype=np.int32)
            remapped['lifted_xyz'] = np.asarray(payload.get('refined_lifted_xyz', payload.get('lifted_xyz', np.zeros((0, 0, 3), dtype=np.float32))), dtype=np.float32)
            remapped['lifted_valid'] = np.asarray(payload.get('refined_lifted_valid', payload.get('lifted_valid', np.zeros((0, 0), dtype=bool))))
            remapped['static_labels'] = np.asarray(payload.get('refined_static_labels', payload.get('static_labels', np.zeros((0,), dtype=np.int8))), dtype=np.int8)
            remapped['static_scores'] = np.asarray(payload.get('refined_static_scores', payload.get('static_scores', np.zeros((0,), dtype=np.float32))), dtype=np.float32)
            refined_payloads[stream_id] = remapped
        refined_artifact = replace(track_artifact, track_payloads_by_stream=refined_payloads)
        max_points = int(self.metadata.get('debug_max_points', 5000))
        max_trails = int(self.metadata.get('debug_max_trails', 12))
        trail_window = int(self.metadata.get('debug_trail_window', 12))

        frame_specs: list[dict[str, Any]] = []
        frames_for_view = list(self.selected_frames) or list(track_artifact.selected_frames)
        for frame in frames_for_view:
            frame_key = make_frame_key(frame.stream_id, frame.frame_idx)
            image_name = image_index.get(frame_key)
            depth_payload = geometry_artifact.depthmaps_by_frame.get(frame_key)
            point_payload = geometry_artifact.pointmaps_by_frame.get(frame_key)
            if image_name is None or depth_payload is None or point_payload is None:
                continue
            rgb = np.asarray(Image.open(self.save_dir / 'images' / image_name).convert('RGB'))
            depth_preview = GeometryArtifact._depth_to_preview_image(depth_payload['depth'])
            depth_preview = np.repeat(depth_preview[..., None], 3, axis=2)
            bg_pts, bg_cols = track_artifact._sample_geometry_debug_cloud(
                point_payload,
                rgb=rgb,
                max_points=max_points,
            )
            if bg_pts.size == 0:
                bg_pts = np.zeros((1, 3), dtype=np.float32)
                bg_cols = np.array([[128, 128, 128]], dtype=np.uint8)
            track_state = track_artifact._track_state_for_frame(frame.stream_id, frame.frame_idx)
            lifted_before = track_artifact._lifted_state_for_frame(frame.stream_id, frame.frame_idx)
            lifted_after = refined_artifact._lifted_state_for_frame(frame.stream_id, frame.frame_idx)
            sampled_ids = track_artifact._sample_track_ids(np.asarray(track_state['track_ids'], dtype=int), max_trails)
            trails = []
            for track_id in sampled_ids.tolist():
                trails.append({
                    'track_id': int(track_id),
                    'color': track_artifact._track_color(int(track_id)),
                    'traj_2d': track_artifact._trajectory_2d_for_track(frame.stream_id, track_id, frame.frame_idx, trail_window),
                    'traj_3d_before': track_artifact._trajectory_3d_for_track(frame.stream_id, track_id, frame.frame_idx, trail_window),
                    'traj_3d_after': refined_artifact._trajectory_3d_for_track(frame.stream_id, track_id, frame.frame_idx, trail_window),
                })
            summary_stream = (self.tracks_3d.get('streams', {}) if isinstance(self.tracks_3d, dict) else {}).get(frame.stream_id, {})
            frame_specs.append({
                'frame_ref': frame,
                'rgb': rgb,
                'depth_preview': depth_preview,
                'background_points': bg_pts,
                'background_colors': bg_cols,
                'track_state': track_state,
                'lifted_before': lifted_before,
                'lifted_after': lifted_after,
                'trails': trails,
                'summary_stream': summary_stream,
            })
        if not frame_specs:
            return

        fig = make_subplots(
            rows=2,
            cols=3,
            specs=[[{'type': 'xy'}, {'type': 'image'}, {'type': 'table'}], [{'type': 'scene'}, {'type': 'scene'}, None]],
            subplot_titles=('RGB + Tracks', 'Depth', 'Refinement Summary', 'Before refinement', 'After refinement'),
            row_heights=[0.40, 0.60],
            column_widths=[0.34, 0.26, 0.40],
            horizontal_spacing=0.05,
            vertical_spacing=0.08,
        )

        traces_per_frame = 7 + (3 * max_trails)
        total_traces = 0
        slider_steps = []
        for idx, spec in enumerate(frame_specs):
            visible = idx == 0
            fr = spec['frame_ref']
            name = make_frame_key(fr.stream_id, fr.frame_idx)
            ts = spec['track_state']
            ids = np.asarray(ts['track_ids'], dtype=int)
            marker_colors = [track_artifact._track_color(int(t)) for t in ids.tolist()]
            marker_symbols = [track_artifact._label_symbol(int(v)) for v in np.asarray(ts['labels'], dtype=np.int8).tolist()]
            hover = [f"track_id={int(t)}<br>before={track_artifact._label_name(int(lbl))}" for t, lbl in zip(ids.tolist(), np.asarray(ts['labels']).tolist())]
            fig.add_trace(go.Image(z=spec['rgb'], visible=visible, name=f'rgb:{name}'), row=1, col=1)
            fig.add_trace(
                go.Scatter(
                    x=ts['points_xy'][:, 0] if ts['points_xy'].size else [],
                    y=ts['points_xy'][:, 1] if ts['points_xy'].size else [],
                    mode='markers',
                    marker={'size': 7, 'color': marker_colors, 'symbol': marker_symbols, 'line': {'color': 'white', 'width': 0.6}},
                    hovertext=hover,
                    hoverinfo='text',
                    visible=visible,
                    showlegend=False,
                    name=f'tracks:{name}',
                ),
                row=1,
                col=1,
            )
            fig.add_trace(go.Image(z=spec['depth_preview'], visible=visible, name=f'depth:{name}'), row=1, col=2)
            before_info = spec['summary_stream']
            before_labels = np.asarray(self.refined_track_payloads_by_stream.get(fr.stream_id, {}).get('static_labels', []), dtype=np.int8)
            after_labels = np.asarray(self.refined_track_payloads_by_stream.get(fr.stream_id, {}).get('refined_static_labels', []), dtype=np.int8)
            def counts(arr):
                return int(np.sum(arr > 0)), int(np.sum(arr == 0)), int(np.sum(arr < 0))
            b_s, b_d, b_u = counts(before_labels)
            a_s, a_d, a_u = counts(after_labels)
            metrics = ['Backend', 'Accepted updates', 'Mean error before', 'Mean error after', 'Median error before', 'Median error after', 'Tracks before s/d/u', 'Tracks after s/d/u']
            values = [
                str(self.refined_track_payloads_by_stream.get(fr.stream_id, {}).get('backend_used', 'unknown')),
                str(int(before_info.get('accepted_updates', 0))),
                f"{before_info.get('mean_error_before', None)}",
                f"{before_info.get('mean_error_after', None)}",
                f"{before_info.get('median_error_before', None)}",
                f"{before_info.get('median_error_after', None)}",
                f"{b_s} / {b_d} / {b_u}",
                f"{a_s} / {a_d} / {a_u}",
            ]
            fig.add_trace(go.Table(header={'values': ['Metric', 'Value'], 'align': 'left'}, cells={'values': [metrics, values], 'align': 'left'}, visible=visible, name=f'summary:{name}'), row=1, col=3)
            bg_marker = {'size': 2, 'opacity': 0.18, 'color': [f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in spec['background_colors']]}
            for scene_col, lifted_state, label in [(1, spec['lifted_before'], 'before'), (2, spec['lifted_after'], 'after')]:
                l_ids = np.asarray(lifted_state['track_ids'], dtype=int)
                l_cols = [track_artifact._track_color(int(t)) for t in l_ids.tolist()]
                hover3 = [
                    f"track_id={int(t)}<br>{label}={track_artifact._label_name(int(lbl))}<br>score={float(score):.3f}"
                    for t, lbl, score in zip(l_ids.tolist(), np.asarray(lifted_state['labels']).tolist(), np.asarray(lifted_state['scores']).tolist())
                ]
                fig.add_trace(go.Scatter3d(x=spec['background_points'][:, 0], y=spec['background_points'][:, 1], z=spec['background_points'][:, 2], mode='markers', marker=bg_marker, visible=visible, showlegend=False, name=f'bg:{label}:{name}'), row=2, col=scene_col)
                fig.add_trace(go.Scatter3d(x=lifted_state['points_xyz'][:, 0] if lifted_state['points_xyz'].size else [], y=lifted_state['points_xyz'][:, 1] if lifted_state['points_xyz'].size else [], z=lifted_state['points_xyz'][:, 2] if lifted_state['points_xyz'].size else [], mode='markers', marker={'size': 5, 'opacity': 0.95, 'color': l_cols, 'line': {'color': 'white', 'width': 1}}, hovertext=hover3, hoverinfo='text', visible=visible, showlegend=False, name=f'{label}:{name}'), row=2, col=scene_col)
            for trail_idx in range(max_trails):
                if trail_idx < len(spec['trails']):
                    trail = spec['trails'][trail_idx]
                    color = trail['color']
                    tid = trail['track_id']
                    fig.add_trace(go.Scatter(x=trail['traj_2d'][:, 0] if trail['traj_2d'].size else [], y=trail['traj_2d'][:, 1] if trail['traj_2d'].size else [], mode='lines', line={'color': color, 'width': 2}, opacity=0.85, hoverinfo='skip', visible=visible, showlegend=False, name=f'trail2d:{name}:{tid}'), row=1, col=1)
                    fig.add_trace(go.Scatter3d(x=trail['traj_3d_before'][:, 0] if trail['traj_3d_before'].size else [], y=trail['traj_3d_before'][:, 1] if trail['traj_3d_before'].size else [], z=trail['traj_3d_before'][:, 2] if trail['traj_3d_before'].size else [], mode='lines', line={'color': color, 'width': 5}, hoverinfo='skip', visible=visible, showlegend=False, name=f'before_trail3d:{name}:{tid}'), row=2, col=1)
                    fig.add_trace(go.Scatter3d(x=trail['traj_3d_after'][:, 0] if trail['traj_3d_after'].size else [], y=trail['traj_3d_after'][:, 1] if trail['traj_3d_after'].size else [], z=trail['traj_3d_after'][:, 2] if trail['traj_3d_after'].size else [], mode='lines', line={'color': color, 'width': 5}, hoverinfo='skip', visible=visible, showlegend=False, name=f'after_trail3d:{name}:{tid}'), row=2, col=2)
                else:
                    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', visible=visible, showlegend=False, hoverinfo='skip', name=f'trail2d_empty:{name}:{trail_idx}'), row=1, col=1)
                    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', visible=visible, showlegend=False, hoverinfo='skip', name=f'before_trail3d_empty:{name}:{trail_idx}'), row=2, col=1)
                    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', visible=visible, showlegend=False, hoverinfo='skip', name=f'after_trail3d_empty:{name}:{trail_idx}'), row=2, col=2)
            total_traces += traces_per_frame

        for idx, spec in enumerate(frame_specs):
            fr = spec['frame_ref']
            name = make_frame_key(fr.stream_id, fr.frame_idx)
            visible = [False] * total_traces
            base = idx * traces_per_frame
            visible[base:base + traces_per_frame] = [True] * traces_per_frame
            slider_steps.append({'method': 'update', 'label': name, 'args': [{'visible': visible}, {'title': {'text': f"Refinement debug | keyframe {fr.describe()}"}}]})

        first = frame_specs[0]['frame_ref']
        fig.update_layout(
            title={'text': f"Refinement debug | keyframe {first.describe()}"},
            template='plotly_white',
            width=1850,
            height=1100,
            scene={'xaxis_title': 'X', 'yaxis_title': 'Y', 'zaxis_title': 'Z (up)', 'aspectmode': 'data'},
            scene2={'xaxis_title': 'X', 'yaxis_title': 'Y', 'zaxis_title': 'Z (up)', 'aspectmode': 'data'},
            margin={'l': 20, 'r': 20, 't': 60, 'b': 20},
            sliders=[{'active': 0, 'currentvalue': {'prefix': 'Keyframe: '}, 'pad': {'t': 20}, 'steps': slider_steps}],
        )
        fig.update_xaxes(visible=False, showticklabels=False, row=1, col=1)
        fig.update_yaxes(visible=False, showticklabels=False, autorange='reversed', scaleanchor='x', row=1, col=1)
        fig.update_xaxes(visible=False, showticklabels=False, row=1, col=2)
        fig.update_yaxes(visible=False, showticklabels=False, row=1, col=2)
        (self.save_dir / 'refinement_debug_viewer.html').write_text(fig.to_html(full_html=True, include_plotlyjs=True), encoding='utf-8')

    def summary(self) -> dict[str, Any]:
        track_summary = self.tracks_3d.get("streams", {}) if isinstance(self.tracks_3d, dict) else {}
        improved_streams = [
            stream_id for stream_id, info in track_summary.items() if isinstance(info, dict) and info.get("applied")
        ]
        return {
            "run_id": self.run_id,
            "num_refined_cameras": sum(len(states) for states in self.refined_cameras.values()),
            "num_stages": len(self.stage_diagnostics),
            "schedule_used": self.schedule_used,
            "static_track_refined_streams": improved_streams,
            "pending_todos": self.pending_todos,
        }
