from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from room4d.common.types import CameraIntrinsics, CameraPose, FrameRef, TimedCameraState
from room4d.geometry.geometry_artifact import GeometryArtifact
from room4d.ingest.decoder import VideoDecoder
from room4d.ingest.ingest_artifact import IngestArtifact
from room4d.lifting.backproject import TrackLifter3D
from room4d.lifting.static_dynamic import StaticDynamicPartitioner
from room4d.tracking.cotracker_wrapper import CoTrackerWrapper
from room4d.tracking.track_artifact import TrackArtifact
from tests.conftest import write_test_video


class _FakeCoTrackerWrapper(CoTrackerWrapper):
    def _load_model(self) -> None:
        self._is_loaded = True

    def _run_real_backend_single(self, video_rgb: np.ndarray):
        t, h, w, _ = video_rgb.shape
        track_static = np.tile(np.array([[w * 0.25, h * 0.5]], dtype=np.float32), (t, 1, 1))
        xs = np.linspace(w * 0.55, w * 0.80, t, dtype=np.float32)
        ys = np.full((t,), h * 0.5, dtype=np.float32)
        track_dynamic = np.stack([xs, ys], axis=-1)[:, None, :]
        tracks = np.concatenate([track_static, track_dynamic], axis=1)
        visibility = np.ones((t, 2), dtype=np.float32)
        return tracks, visibility


def _make_geometry_artifact(tmp_path: Path, stream_id: str, width: int, height: int) -> GeometryArtifact:
    frames = [
        FrameRef(stream_id=stream_id, frame_idx=0, timestamp_sec=0.0),
        FrameRef(stream_id=stream_id, frame_idx=4, timestamp_sec=0.4),
    ]
    intr = CameraIntrinsics(fx=40.0, fy=40.0, cx=width / 2, cy=height / 2)
    states = [
        TimedCameraState(stream_id=stream_id, time_sec=0.0, intrinsics=intr, pose=CameraPose(T_world_cam=np.eye(4, dtype=np.float32))),
        TimedCameraState(stream_id=stream_id, time_sec=0.4, intrinsics=intr, pose=CameraPose(T_world_cam=np.eye(4, dtype=np.float32))),
    ]
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, height), np.linspace(0.0, 1.0, width), indexing='ij')
    points = np.stack([xx, np.zeros_like(xx), yy], axis=-1).astype(np.float32)
    depth = np.ones((height, width, 1), dtype=np.float32)
    image_paths = {}
    for frame in frames:
        img_path = tmp_path / f'{frame.stream_id}_{frame.frame_idx}.png'
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = 255
        Image.fromarray(rgb).save(img_path)
        image_paths[f'{frame.stream_id}:{frame.frame_idx}'] = str(img_path)
    return GeometryArtifact(
        run_id='test',
        stage_name='geometry',
        save_dir=tmp_path / 'geometry',
        backend_name='vggt',
        selected_frames=frames,
        intrinsics_by_stream={stream_id: intr},
        camera_states_by_stream={stream_id: states},
        depthmaps_by_frame={f'{frame.stream_id}:{frame.frame_idx}': {'depth': depth, 'confidence': np.ones((height, width), dtype=np.float32)} for frame in frames},
        pointmaps_by_frame={f'{frame.stream_id}:{frame.frame_idx}': {'points': points, 'confidence': np.ones((height, width), dtype=np.float32)} for frame in frames},
        metadata={'debug_image_paths': image_paths},
    )


def test_cotracker_real_wrapper_contract(tmp_path: Path) -> None:
    video_path = tmp_path / 'clip.mp4'
    write_test_video(video_path, width=64, height=48, num_frames=5, fps=10.0)
    decoder = VideoDecoder(default_fps=30.0)
    stream = decoder.decode(video_path, stream_id='cam_00')

    tracker = _FakeCoTrackerWrapper({'backend_mode': 'real', 'device': 'cpu', 'grid_size': 2})
    artifact = tracker.track(stream)

    payload = artifact.track_payloads_by_stream['cam_00']
    assert payload['points_xy'].shape == (5, 2, 2)
    assert payload['visibility'].shape == (5, 2)
    assert artifact.total_observations() == 10


def test_track_lifting_partition_and_debug_export(tmp_path: Path) -> None:
    video_path = tmp_path / 'clip.mp4'
    write_test_video(video_path, width=64, height=48, num_frames=5, fps=10.0)
    decoder = VideoDecoder(default_fps=30.0)
    stream = decoder.decode(video_path, stream_id='cam_00')
    ingest = IngestArtifact(
        run_id='test',
        stage_name='ingest',
        save_dir=tmp_path / 'ingest',
        streams=[stream],
        keyframes=[FrameRef(stream_id='cam_00', frame_idx=0, timestamp_sec=0.0), FrameRef(stream_id='cam_00', frame_idx=4, timestamp_sec=0.4)],
    )
    tracker = _FakeCoTrackerWrapper({'backend_mode': 'real', 'device': 'cpu', 'grid_size': 2})
    track_artifact = tracker.track(stream)
    geometry = _make_geometry_artifact(tmp_path, 'cam_00', width=64, height=48)

    lifted = TrackLifter3D().lift(track_artifact, geometry, ingest)
    StaticDynamicPartitioner(static_threshold_world=0.01, min_lifted_frames=2).partition(lifted, geometry)
    payload = lifted.track_payloads_by_stream['cam_00']
    labels = payload['static_labels'].tolist()
    assert 1 in labels
    assert 0 in labels

    lifted.save_dir = tmp_path / 'tracking'
    lifted.metadata.update({
        'save_debug_assets': True,
        'save_debug_html': True,
        'debug_max_points': 128,
        'debug_image_paths': dict(geometry.metadata['debug_image_paths']),
    })
    lifted.save()
    reloaded = TrackArtifact.load(tmp_path / 'tracking')
    assert reloaded.track_payloads_by_stream['cam_00']['lifted_xyz'].shape[0] == 2
    assert (tmp_path / 'tracking' / 'tracking_debug_viewer.html').exists()


def test_cotracker_windowed_tracking_contract(tmp_path: Path) -> None:
    video_path = tmp_path / 'clip_windowed.mp4'
    write_test_video(video_path, width=64, height=48, num_frames=8, fps=10.0)
    decoder = VideoDecoder(default_fps=30.0)
    stream = decoder.decode(video_path, stream_id='cam_00')

    tracker = _FakeCoTrackerWrapper({
        'backend_mode': 'real',
        'device': 'cpu',
        'grid_size': 2,
        'window_size': 4,
        'window_overlap': 2,
    })
    artifact = tracker.track(stream)

    payload = artifact.track_payloads_by_stream['cam_00']
    windowing = payload['windowing']
    assert windowing['windowed'] is True
    assert len(windowing['windows']) >= 2
    assert payload['backend_used'] == 'cotracker_windowed'
    assert payload['points_xy'].shape[0] == 8
    assert payload['points_xy'].shape[1] > 2
