"""Artifact for coarse geometry initialization results."""

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
from room4d.common.types import BaseArtifact, CameraIntrinsics, CameraPose, FrameRef, TimedCameraState


@dataclass
class GeometryArtifact(BaseArtifact):
    """Store camera, depth, and correspondence initialization outputs.

    This artifact persists both lightweight metadata and the heavier per-frame
    numpy outputs needed by downstream stages. The persistence format is:

    - ``artifact.json``: serializable metadata and index records
    - ``depthmaps/<frame_key>.npz``: depth/confidence arrays
    - ``pointmaps/<frame_key>.npz``: point/confidence arrays
    - optional debug assets under ``images/``, ``depthmaps_png/``, ``pointclouds/``
      and ``debug_viewer.html``
    """

    backend_name: str = "unknown"
    selected_frames: list[FrameRef] = field(default_factory=list)
    intrinsics_by_stream: dict[str, CameraIntrinsics] = field(default_factory=dict)
    camera_states_by_stream: dict[str, list[TimedCameraState]] = field(default_factory=dict)
    pointmaps_by_frame: dict[str, Any] = field(default_factory=dict)
    depthmaps_by_frame: dict[str, Any] = field(default_factory=dict)
    pairwise_matches: dict[str, Any] = field(default_factory=dict)
    view_graph_edges: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def stream_ids(self) -> list[str]:
        """Return all streams for which intrinsics or camera priors exist."""
        keys = set(self.intrinsics_by_stream) | set(self.camera_states_by_stream)
        return sorted(keys)

    def num_selected_frames(self) -> int:
        """Return the number of frames selected for initialization."""
        return len(self.selected_frames)

    def _progress_iter(self, iterable, *, total: int | None = None, description: str = ""):
        enabled = bool(self.metadata.get('save_depth_pngs', False) or self.metadata.get('save_debug_assets', False) or self.metadata.get('save_debug_html', False))
        if not enabled:
            return iterable
        return track(iterable, total=total, description=description)

    def _sanitize_key(self, key: str) -> str:
        return key.replace(':', '__').replace('/', '_')

    @staticmethod
    def _depth_to_preview_image(depth: np.ndarray) -> np.ndarray:
        """Convert a depth array into an 8-bit preview image for debugging."""
        arr = np.asarray(depth, dtype=np.float32)
        arr = np.squeeze(arr)
        finite_mask = np.isfinite(arr)
        if not finite_mask.any():
            return np.zeros(arr.shape, dtype=np.uint8)

        valid = arr[finite_mask]
        lo = float(np.min(valid))
        hi = float(np.max(valid))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            out = np.zeros(arr.shape, dtype=np.uint8)
            out[finite_mask] = 255
            return out

        scaled = np.zeros(arr.shape, dtype=np.float32)
        scaled[finite_mask] = (arr[finite_mask] - lo) / (hi - lo)
        scaled = np.clip(scaled, 0.0, 1.0)
        return (scaled * 255.0).astype(np.uint8)

    @staticmethod
    def _frame_ref_to_dict(frame: FrameRef) -> dict[str, Any]:
        return asdict(frame)

    @staticmethod
    def _frame_ref_from_dict(raw: dict[str, Any]) -> FrameRef:
        return FrameRef(**raw)

    @staticmethod
    def _intrinsics_to_dict(intr: CameraIntrinsics) -> dict[str, Any]:
        return {
            'fx': intr.fx,
            'fy': intr.fy,
            'cx': intr.cx,
            'cy': intr.cy,
            'distortion_model': intr.distortion_model,
            'distortion_params': None if intr.distortion_params is None else np.asarray(intr.distortion_params, dtype=float).tolist(),
        }

    @staticmethod
    def _intrinsics_from_dict(raw: dict[str, Any]) -> CameraIntrinsics:
        distortion_params = raw.get('distortion_params')
        return CameraIntrinsics(
            fx=float(raw['fx']),
            fy=float(raw['fy']),
            cx=float(raw['cx']),
            cy=float(raw['cy']),
            distortion_model=str(raw.get('distortion_model', 'brown')),
            distortion_params=None if distortion_params is None else np.asarray(distortion_params, dtype=float),
        )

    @staticmethod
    def _camera_state_to_dict(state: TimedCameraState) -> dict[str, Any]:
        return {
            'stream_id': state.stream_id,
            'time_sec': float(state.time_sec),
            'intrinsics': GeometryArtifact._intrinsics_to_dict(state.intrinsics),
            'pose': np.asarray(state.pose.T_world_cam, dtype=float).tolist(),
        }

    @staticmethod
    def _camera_state_from_dict(raw: dict[str, Any]) -> TimedCameraState:
        return TimedCameraState(
            stream_id=str(raw['stream_id']),
            time_sec=float(raw['time_sec']),
            intrinsics=GeometryArtifact._intrinsics_from_dict(raw['intrinsics']),
            pose=CameraPose(T_world_cam=np.asarray(raw['pose'], dtype=float)),
        )

    @staticmethod
    def _ensure_hwc_points(points: np.ndarray) -> np.ndarray:
        """Normalize point maps to HxWx3 when possible."""
        arr = np.asarray(points)
        if arr.ndim != 3:
            raise ValueError(f"Expected a 3D point map array, got shape {arr.shape}.")
        if arr.shape[-1] == 3:
            return arr
        if arr.shape[0] == 3:
            return np.moveaxis(arr, 0, -1)
        raise ValueError(f"Could not interpret point map shape {arr.shape} as HxWx3.")

    @staticmethod
    def _z_up_transform_matrix() -> np.ndarray:
        """Return a fixed basis change that maps the old convention to Z-up.

        The debug exports use:
        - X' = X
        - Y' = Z
        - Z' = -Y

        This turns a typical camera-style Y-down convention into a visualization
        convention with positive Z pointing upward.
        """
        return np.asarray([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ], dtype=np.float32)

    @classmethod
    def _points_to_z_up(cls, points: np.ndarray) -> np.ndarray:
        """Transform a point cloud or point map into the debug Z-up convention."""
        arr = np.asarray(points)
        rot = cls._z_up_transform_matrix()
        if arr.ndim >= 1 and arr.shape[-1] == 3:
            return np.einsum('ij,...j->...i', rot, arr)
        if arr.ndim >= 1 and arr.shape[0] == 3:
            return np.einsum('ij,j...->i...', rot, arr)
        raise ValueError(f"Could not transform point array with shape {arr.shape} to Z-up.")

    @classmethod
    def _pose_to_z_up(cls, T_world_cam: np.ndarray) -> np.ndarray:
        """Transform a homogeneous pose into the debug Z-up basis."""
        T = np.asarray(T_world_cam, dtype=np.float32)
        if T.shape != (4, 4):
            raise ValueError(f"Expected 4x4 pose matrix, got shape {T.shape}.")
        A = np.eye(4, dtype=np.float32)
        A[:3, :3] = cls._z_up_transform_matrix()
        A_inv = np.eye(4, dtype=np.float32)
        A_inv[:3, :3] = A[:3, :3].T
        return A @ T @ A_inv


    @classmethod
    def _points_to_debug_basis(cls, points: np.ndarray) -> np.ndarray:
        """Transform points into the debug visualization basis.

        This wrapper keeps the debug export path compatible across revisions
        where the basis helper was named differently or switched between class
        and instance-style methods. The current debug basis is the same Z-up
        transform used by ``_points_to_z_up``.
        """
        return cls._points_to_z_up(points)

    @classmethod
    def _sample_point_cloud(
        cls,
        points: np.ndarray,
        rgb: np.ndarray | None = None,
        *,
        max_points: int = 5000,
        confidence: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Flatten and downsample a point map for debug export.

        Notes
        -----
        The debug RGB image may be the original copied keyframe while the point
        map is often at the model's internal resolution. When that happens, the
        RGB image is resized to the point-map resolution before flattening so the
        boolean validity mask lines up with the color array.
        """
        point_arr = cls._points_to_debug_basis(cls._ensure_hwc_points(points))
        h, w, _ = point_arr.shape
        pts = point_arr.reshape(-1, 3)
        valid = np.isfinite(pts).all(axis=1)

        conf = None
        if confidence is not None:
            conf = np.asarray(confidence).reshape(-1)
            if conf.shape[0] == pts.shape[0]:
                valid &= np.isfinite(conf)
                valid &= conf > 0
            else:
                conf = None

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

        if pts.shape[0] == 0:
            return pts, cols

        if pts.shape[0] > max_points:
            stride = int(np.ceil(pts.shape[0] / max_points))
            pts = pts[::stride]
            if cols is not None:
                cols = cols[::stride]
        return pts.astype(np.float32), None if cols is None else cols.astype(np.uint8)

    @staticmethod
    def _write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
        """Write a small ASCII PLY point cloud for debugging."""
        path.parent.mkdir(parents=True, exist_ok=True)
        has_color = colors is not None and len(colors) == len(points)
        with path.open('w', encoding='utf-8') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(points)}\n')
            f.write('property float x\nproperty float y\nproperty float z\n')
            if has_color:
                f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
            f.write('end_header\n')
            if has_color:
                for xyz, rgb in zip(points, colors):
                    f.write(
                        f"{float(xyz[0])} {float(xyz[1])} {float(xyz[2])} "
                        f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n"
                    )
            else:
                for xyz in points:
                    f.write(f"{float(xyz[0])} {float(xyz[1])} {float(xyz[2])}\n")

    def _copy_debug_images(self, image_dir: Path) -> dict[str, str]:
        """Copy source RGB images into the artifact for self-contained debugging."""
        copied: dict[str, str] = {}
        source_map = self.metadata.get('debug_image_paths', {})
        if not source_map:
            return copied
        ensure_dir(image_dir)
        for frame_key, source in self._progress_iter(source_map.items(), total=len(source_map), description="[cyan]Geometry · copying debug RGB"):
            src = Path(source)
            if not src.exists():
                continue
            dst = image_dir / f"{self._sanitize_key(frame_key)}{src.suffix or '.png'}"
            shutil.copy2(src, dst)
            copied[frame_key] = dst.name
        return copied

    def _find_state_for_frame(self, frame: FrameRef) -> TimedCameraState | None:
        """Look up the saved camera state corresponding to a selected frame."""
        states = self.camera_states_by_stream.get(frame.stream_id, [])
        if not states:
            return None
        best = None
        best_dt = None
        for state in states:
            dt = abs(float(state.time_sec) - float(frame.timestamp_sec))
            if best is None or dt < best_dt:
                best = state
                best_dt = dt
        return best

    def _build_camera_records(self) -> list[dict[str, Any]]:
        """Create per-frame camera records for JSON export."""
        records: list[dict[str, Any]] = []
        for frame in self._progress_iter(self.selected_frames, total=len(self.selected_frames), description="[cyan]Geometry · exporting cameras"):
            state = self._find_state_for_frame(frame)
            if state is None:
                continue
            frame_key = self._sanitize_key(f"{frame.stream_id}:{frame.frame_idx}")
            records.append(
                {
                    'frame_key': frame_key,
                    'stream_id': frame.stream_id,
                    'frame_idx': frame.frame_idx,
                    'timestamp_sec': frame.timestamp_sec,
                    'intrinsics': self._intrinsics_to_dict(state.intrinsics),
                    'T_world_cam': self._pose_to_z_up(state.pose.T_world_cam).astype(float).tolist(),
                }
            )
        return records

    def _export_debug_assets(self) -> dict[str, Any]:
        """Write optional debug assets and return an index for artifact.json."""
        exported: dict[str, Any] = {}
        if not self.metadata.get('save_debug_assets', False):
            return exported

        image_index = self._copy_debug_images(self.save_dir / 'images')
        if image_index:
            exported['image_index'] = image_index

        if self.metadata.get('save_camera_json', False):
            camera_records = self._build_camera_records()
            write_json(self.save_dir / 'cameras.json', {'frames': camera_records})
            exported['camera_json'] = 'cameras.json'

        pointcloud_index: dict[str, str] = {}
        if self.metadata.get('save_pointcloud_plys', False):
            pointcloud_dir = ensure_dir(self.save_dir / 'pointclouds')
            max_points = int(self.metadata.get('debug_max_points', 5000))
            for frame in self._progress_iter(self.selected_frames, total=len(self.selected_frames), description="[cyan]Geometry · exporting point clouds"):
                frame_key = f"{frame.stream_id}:{frame.frame_idx}"
                payload = self.pointmaps_by_frame.get(frame_key)
                if not payload:
                    continue
                img_name = image_index.get(frame_key)
                rgb = None
                if img_name is not None:
                    rgb = np.asarray(Image.open(self.save_dir / 'images' / img_name).convert('RGB'))
                pts, cols = self._sample_point_cloud(
                    payload.get('points'),
                    rgb=rgb,
                    max_points=max_points,
                    confidence=payload.get('confidence'),
                )
                out_name = f"{self._sanitize_key(frame_key)}.ply"
                self._write_ascii_ply(pointcloud_dir / out_name, pts, cols)
                pointcloud_index[frame_key] = out_name
        if pointcloud_index:
            exported['pointcloud_index'] = pointcloud_index

        if self.metadata.get('save_debug_html', False):
            try:
                self._write_plotly_debug_html(
                    image_index=image_index,
                    pointcloud_index=pointcloud_index,
                )
                exported['debug_html'] = 'debug_viewer.html'
            except Exception as exc:  # pragma: no cover - best-effort debug export
                self.notes.append(f'Failed to write Plotly debug viewer: {type(exc).__name__}: {exc}')
        return exported

    def _write_plotly_debug_html(self, *, image_index: dict[str, str], pointcloud_index: dict[str, str]) -> None:
        """Write a self-contained Plotly HTML viewer for per-keyframe debugging."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        frame_specs: list[dict[str, Any]] = []
        max_points = int(self.metadata.get('debug_max_points', 5000))
        for frame in self._progress_iter(self.selected_frames, total=len(self.selected_frames), description="[cyan]Geometry · building debug viewer"):
            frame_key = f"{frame.stream_id}:{frame.frame_idx}"
            image_name = image_index.get(frame_key)
            depth_payload = self.depthmaps_by_frame.get(frame_key)
            point_payload = self.pointmaps_by_frame.get(frame_key)
            if image_name is None or depth_payload is None or point_payload is None:
                continue

            rgb = np.asarray(Image.open(self.save_dir / 'images' / image_name).convert('RGB'))
            depth_preview = self._depth_to_preview_image(depth_payload['depth'])
            if depth_preview.ndim == 2:
                depth_preview = np.repeat(depth_preview[..., None], 3, axis=2)

            pts, cols = self._sample_point_cloud(
                point_payload['points'],
                rgb=rgb,
                max_points=max_points,
                confidence=point_payload.get('confidence'),
            )
            if pts.size == 0:
                pts = np.zeros((1, 3), dtype=np.float32)
                cols = np.array([[128, 128, 128]], dtype=np.uint8)

            marker = {
                'size': 2,
                'opacity': 0.9,
            }
            if cols is not None:
                marker['color'] = [f'rgb({int(c[0])},{int(c[1])},{int(c[2])})' for c in cols]
            else:
                marker['color'] = '#4C78A8'

            state = self._find_state_for_frame(frame)
            frame_specs.append(
                {
                    'frame_ref': frame,
                    'rgb': rgb,
                    'depth_preview': depth_preview,
                    'points': pts,
                    'marker': marker,
                    'camera_state': None if state is None else {
                        'time_sec': state.time_sec,
                        'T_world_cam': self._pose_to_z_up(state.pose.T_world_cam).astype(float).tolist(),
                    },
                }
            )

        if not frame_specs:
            return

        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{'type': 'image'}, {'type': 'image'}], [{'type': 'scene', 'colspan': 2}, None]],
            subplot_titles=('RGB', 'Depth', 'Point Cloud'),
            row_heights=[0.42, 0.58],
            horizontal_spacing=0.03,
            vertical_spacing=0.08,
        )

        trace_visibility = []
        slider_steps = []
        total_traces = 0

        for idx, spec in enumerate(frame_specs):
            visible = idx == 0
            fr = spec['frame_ref']
            name = f"{fr.stream_id}:{fr.frame_idx}"

            fig.add_trace(go.Image(z=spec['rgb'], visible=visible, name=f"rgb:{name}"), row=1, col=1)
            fig.add_trace(go.Image(z=spec['depth_preview'], visible=visible, name=f"depth:{name}"), row=1, col=2)
            fig.add_trace(
                go.Scatter3d(
                    x=spec['points'][:, 0],
                    y=spec['points'][:, 1],
                    z=spec['points'][:, 2],
                    mode='markers',
                    marker=spec['marker'],
                    name=f'pointcloud:{name}',
                    visible=visible,
                ),
                row=2,
                col=1,
            )
            total_traces += 3

        for idx, spec in enumerate(frame_specs):
            fr = spec['frame_ref']
            name = f"{fr.stream_id}:{fr.frame_idx}"
            visible = [False] * total_traces
            base = 3 * idx
            visible[base:base+3] = [True, True, True]
            title = (
                f"Keyframe {name} | t={fr.timestamp_sec:.3f}s"
                + ('' if spec['camera_state'] is None else f" | camera_t={spec['camera_state']['time_sec']:.3f}s")
            )
            slider_steps.append({
                'method': 'update',
                'label': name,
                'args': [
                    {'visible': visible},
                    {'title': {'text': title}},
                ],
            })

        initial = frame_specs[0]
        initial_title = (
            f"Keyframe {initial['frame_ref'].stream_id}:{initial['frame_ref'].frame_idx} | t={initial['frame_ref'].timestamp_sec:.3f}s"
            + ('' if initial['camera_state'] is None else f" | camera_t={initial['camera_state']['time_sec']:.3f}s")
        )

        fig.update_layout(
            title={'text': initial_title},
            template='plotly_white',
            width=1400,
            height=950,
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
        fig.update_xaxes(visible=False, showticklabels=False)
        fig.update_yaxes(visible=False, showticklabels=False)
        html = fig.to_html(full_html=True, include_plotlyjs=True)
        (self.save_dir / 'debug_viewer.html').write_text(html, encoding='utf-8')

    def save(self) -> None:
        """Persist geometry metadata and heavy numpy outputs to disk."""
        ensure_dir(self.save_dir)
        depth_dir = ensure_dir(self.save_dir / 'depthmaps')
        point_dir = ensure_dir(self.save_dir / 'pointmaps')
        debug_depth_dir = ensure_dir(self.save_dir / 'depthmaps_png') if self.metadata.get('save_depth_pngs', False) else None

        depth_index: dict[str, str] = {}
        for frame_key, payload in self.depthmaps_by_frame.items():
            filename = f"{self._sanitize_key(frame_key)}.npz"
            depth_array = np.asarray(payload.get('depth'))
            np.savez_compressed(
                depth_dir / filename,
                depth=depth_array,
                confidence=np.asarray(payload['confidence']) if payload.get('confidence') is not None else np.array([], dtype=np.float32),
            )
            if debug_depth_dir is not None:
                preview = self._depth_to_preview_image(depth_array)
                Image.fromarray(preview, mode='L').save(debug_depth_dir / f"{self._sanitize_key(frame_key)}.png")
            depth_index[frame_key] = filename

        point_index: dict[str, str] = {}
        for frame_key, payload in self.pointmaps_by_frame.items():
            filename = f"{self._sanitize_key(frame_key)}.npz"
            np.savez_compressed(
                point_dir / filename,
                points=np.asarray(payload.get('points')),
                confidence=np.asarray(payload['confidence']) if payload.get('confidence') is not None else np.array([], dtype=np.float32),
            )
            point_index[frame_key] = filename

        debug_exports = self._export_debug_assets()

        serializable = {
            'run_id': self.run_id,
            'stage_name': self.stage_name,
            'save_dir': str(self.save_dir),
            'metadata': self.metadata,
            'backend_name': self.backend_name,
            'selected_frames': [GeometryArtifact._frame_ref_to_dict(frame) for frame in self.selected_frames],
            'intrinsics_by_stream': {k: GeometryArtifact._intrinsics_to_dict(v) for k, v in self.intrinsics_by_stream.items()},
            'camera_states_by_stream': {
                k: [GeometryArtifact._camera_state_to_dict(state) for state in states]
                for k, states in self.camera_states_by_stream.items()
            },
            'pairwise_matches': self.pairwise_matches,
            'view_graph_edges': self.view_graph_edges,
            'notes': self.notes,
            'depth_index': depth_index,
            'point_index': point_index,
            'debug_exports': debug_exports,
        }
        write_json(self.artifact_path(), serializable)

    @classmethod
    def load(cls, save_dir: Path) -> 'GeometryArtifact':
        """Load geometry metadata and associated numpy outputs from disk."""
        raw = read_json(save_dir / 'artifact.json')
        depthmaps_by_frame: dict[str, Any] = {}
        pointmaps_by_frame: dict[str, Any] = {}

        for frame_key, filename in raw.get('depth_index', {}).items():
            payload = np.load(save_dir / 'depthmaps' / filename)
            confidence = payload['confidence']
            depthmaps_by_frame[frame_key] = {
                'depth': payload['depth'],
                'confidence': None if confidence.size == 0 else confidence,
            }

        for frame_key, filename in raw.get('point_index', {}).items():
            payload = np.load(save_dir / 'pointmaps' / filename)
            confidence = payload['confidence']
            pointmaps_by_frame[frame_key] = {
                'points': payload['points'],
                'confidence': None if confidence.size == 0 else confidence,
            }

        artifact = cls(
            run_id=raw['run_id'],
            stage_name=raw['stage_name'],
            save_dir=Path(raw['save_dir']),
            metadata=raw.get('metadata', {}),
            backend_name=raw.get('backend_name', 'unknown'),
            selected_frames=[cls._frame_ref_from_dict(frame) for frame in raw.get('selected_frames', [])],
            intrinsics_by_stream={k: cls._intrinsics_from_dict(v) for k, v in raw.get('intrinsics_by_stream', {}).items()},
            camera_states_by_stream={
                k: [cls._camera_state_from_dict(state) for state in states]
                for k, states in raw.get('camera_states_by_stream', {}).items()
            },
            pointmaps_by_frame=pointmaps_by_frame,
            depthmaps_by_frame=depthmaps_by_frame,
            pairwise_matches=raw.get('pairwise_matches', {}),
            view_graph_edges=raw.get('view_graph_edges', {}),
            notes=raw.get('notes', []),
        )
        return artifact

    def summary(self) -> dict[str, Any]:
        """Return a compact geometry summary."""
        return {
            'run_id': self.run_id,
            'backend_name': self.backend_name,
            'num_streams': len(self.stream_ids()),
            'num_selected_frames': self.num_selected_frames(),
            'num_depthmaps': len(self.depthmaps_by_frame),
            'num_pointmaps': len(self.pointmaps_by_frame),
            'save_dir': str(self.save_dir),
            'notes': self.notes,
        }
