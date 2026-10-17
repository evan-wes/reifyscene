from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from reifyscene.common.data_dir import record_name
from reifyscene.common.types import CameraIntrinsics, CameraPose, FrameRef, TimedCameraState
from reifyscene.geometry.geometry_artifact import GeometryArtifact
from reifyscene.geometry.image_transform import ImageTransform, transform_record
from reifyscene.ingest.decoder import VideoDecoder
from reifyscene.lifting.backproject import TrackLifter3D
from reifyscene.lifting.static_dynamic import StaticDynamicPartitioner
from reifyscene.tracking.cotracker_wrapper import CoTrackerWrapper
from reifyscene.tracking.track_artifact import TrackArtifact
from tests.conftest import write_test_video


class _FakeCoTrackerWrapper(CoTrackerWrapper):
    def _load_model(self) -> None:
        self._is_loaded = True

    def _read_video_frames(self, stream: VideoStreamRef):
        frame_indices = list(range(0, max(stream.num_frames, 1), self.frame_stride))
        if self.max_frames is not None:
            frame_indices = frame_indices[: int(self.max_frames)]
        h = int(stream.height or 48)
        w = int(stream.width or 64)
        if self.resize_max_dim is not None:
            scale = min(float(self.resize_max_dim) / max(h, w), 1.0)
            model_w = max(1, int(round(w * scale)))
            model_h = max(1, int(round(h * scale)))
        else:
            model_w, model_h = w, h
        video = np.zeros((len(frame_indices), model_h, model_w, 3), dtype=np.uint8)
        for i, frame_idx in enumerate(frame_indices):
            video[i, ..., 0] = np.uint8(min(frame_idx * 10, 250))
            video[i, ..., 1] = 50
            video[i, ..., 2] = 100
        return video, np.asarray(frame_indices, dtype=np.int32), (w, h), (model_w, model_h)

    def _run_real_backend_single(self, video_rgb: np.ndarray, queries=None):
        # `queries` is what mask seeding passes; this fake emits a fixed pair of
        # trajectories either way, so the seeding path is exercised for plumbing
        # without the fake having to imitate CoTracker's query semantics.
        del queries
        t, h, w, _ = video_rgb.shape
        # Recover exact absolute frame indices from the synthetic video so
        # overlapping windows produce globally consistent trajectories.
        frame_codes = np.rint(video_rgb[:, 0, 0, 0].astype(np.float32) / 10.0).astype(np.float32)
        track_static = np.tile(np.array([[w * 0.25, h * 0.5]], dtype=np.float32), (t, 1, 1))
        xs = w * 0.55 + frame_codes * (w * 0.03)
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
        TimedCameraState(stream_id=stream_id, frame_idx=0, intrinsics=intr, pose=CameraPose(T_world_cam=np.eye(4, dtype=np.float32))),
        TimedCameraState(stream_id=stream_id, frame_idx=4, intrinsics=intr, pose=CameraPose(T_world_cam=np.eye(4, dtype=np.float32))),
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
        image_paths[f'{frame.stream_id}:{frame.frame_idx}'] = record_name(img_path)
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
        metadata={'debug_image_paths': image_paths, 'image_transforms': {stream_id: transform_record(
            ImageTransform.identity((width, height)), recorded_by='test')}},
    )


def test_cotracker_real_wrapper_contract(tmp_path: Path) -> None:
    video_path = tmp_path / 'clip.mp4'
    write_test_video(video_path, width=64, height=48, num_frames=5, fps=10.0)
    decoder = VideoDecoder()
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
    decoder = VideoDecoder()
    stream = decoder.decode(video_path, stream_id='cam_00')
    tracker = _FakeCoTrackerWrapper({'backend_mode': 'real', 'device': 'cpu', 'grid_size': 2})
    track_artifact = tracker.track(stream)
    geometry = _make_geometry_artifact(tmp_path, 'cam_00', width=64, height=48)

    lifted = TrackLifter3D().lift(track_artifact, geometry)
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
        'debug_max_trails': 4,
        'debug_trail_window': 4,
        'debug_image_paths': dict(geometry.metadata['debug_image_paths']),
    })
    lifted.save()
    reloaded = TrackArtifact.load(tmp_path / 'tracking')
    assert reloaded.track_payloads_by_stream['cam_00']['lifted_xyz'].shape[0] == 2
    html_path = tmp_path / 'tracking' / 'tracking_debug_viewer.html'
    assert html_path.exists()
    html = html_path.read_text(encoding='utf-8')
    assert 'Summary' in html
    assert 'Global tracks' in html
    assert 'trail3d' in html


def test_cotracker_windowed_tracking_contract(tmp_path: Path) -> None:
    video_path = tmp_path / 'clip_windowed.mp4'
    write_test_video(video_path, width=64, height=48, num_frames=8, fps=10.0)
    decoder = VideoDecoder()
    stream = decoder.decode(video_path, stream_id='cam_00')

    tracker = _FakeCoTrackerWrapper({
        'backend_mode': 'real',
        'device': 'cpu',
        'grid_size': 2,
        'window_size': 4,
        'window_overlap': 2,
        'stitch_windows': True,
        'stitch_distance_px': 2.0,
    })
    artifact = tracker.track(stream)

    payload = artifact.track_payloads_by_stream['cam_00']
    windowing = payload['windowing']
    assert windowing['windowed'] is True
    assert windowing['stitched'] is True
    assert len(windowing['windows']) >= 2
    assert windowing['matched_pairs'] >= 2
    assert payload['backend_used'] == 'cotracker_windowed'
    assert payload['points_xy'].shape[0] == 8
    assert payload['points_xy'].shape[1] == 2


def test_cotracker_windowed_tracking_without_stitching(tmp_path: Path) -> None:
    video_path = tmp_path / 'clip_windowed_no_stitch.mp4'
    write_test_video(video_path, width=64, height=48, num_frames=8, fps=10.0)
    decoder = VideoDecoder()
    stream = decoder.decode(video_path, stream_id='cam_00')

    tracker = _FakeCoTrackerWrapper({
        'backend_mode': 'real',
        'device': 'cpu',
        'grid_size': 2,
        'window_size': 4,
        'window_overlap': 2,
        'stitch_windows': False,
    })
    artifact = tracker.track(stream)
    payload = artifact.track_payloads_by_stream['cam_00']
    windowing = payload['windowing']
    assert windowing['windowed'] is True
    assert windowing['stitched'] is False
    assert payload['points_xy'].shape[1] > 2


def test_track_lifting_preserves_tracking_timeline(tmp_path: Path) -> None:
    video_path = tmp_path / "clip_preserve.mp4"
    write_test_video(video_path, width=64, height=48, num_frames=5, fps=10.0)
    decoder = VideoDecoder()
    stream = decoder.decode(video_path, stream_id='cam_00')
    tracker = _FakeCoTrackerWrapper({'backend_mode': 'real', 'device': 'cpu', 'grid_size': 2})
    track_artifact = tracker.track(stream)
    tracking_timeline = list(track_artifact.selected_frames)
    geometry = _make_geometry_artifact(tmp_path, 'cam_00', width=64, height=48)

    lifted = TrackLifter3D().lift(track_artifact, geometry)

    assert [fr.frame_idx for fr in lifted.selected_frames] == [fr.frame_idx for fr in tracking_timeline]
    assert lifted.metadata['geometry_selected_frames'][0]['frame_idx'] == 0
    assert lifted.metadata['geometry_selected_frames'][1]['frame_idx'] == 4
