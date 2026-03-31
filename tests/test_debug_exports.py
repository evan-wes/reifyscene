from pathlib import Path

import numpy as np
from PIL import Image

from room4d.api.pipeline import Room4DPipeline
from room4d.common.types import CameraIntrinsics, CameraPose, FrameRef, TimedCameraState, VideoStreamRef
from room4d.debug_exports import export_debug_artifacts, materialize_shared_debug_frames
from room4d.geometry.geometry_artifact import GeometryArtifact


def test_materialize_shared_debug_frames(tmp_path: Path) -> None:
    video_path = tmp_path / 'clip.mp4'
    # reuse test helper expectations via strict decoder-less setup
    from tests.conftest import write_test_video
    write_test_video(video_path, num_frames=6, width=32, height=24, fps=3.0)

    pipe = Room4DPipeline({'run_dir': tmp_path / 'run', 'tracking': {'frame_stride': 2, 'max_frames': 3}})
    ingest = pipe.preprocess([str(video_path)])
    cfg = {'enabled': True, 'shared_rgb_frames': {'enabled': True, 'dir_name': 'debug_rgb_frames'}, 'geometry': {'enabled': True}, 'tracking': {'enabled': True}}
    index = materialize_shared_debug_frames(pipe, ingest, cfg)
    assert index
    assert (tmp_path / 'run' / 'debug_rgb_frames' / 'index.json').exists()


def test_deferred_geometry_debug_export(tmp_path: Path) -> None:
    save_dir = tmp_path / 'geometry'
    frame = FrameRef(stream_id='cam_00', frame_idx=0, timestamp_sec=0.0)
    art = GeometryArtifact(
        run_id='r', stage_name='geometry', save_dir=save_dir, backend_name='test', selected_frames=[frame],
        intrinsics_by_stream={'cam_00': CameraIntrinsics(1,1,1,1)},
        camera_states_by_stream={'cam_00': [TimedCameraState('cam_00', 0.0, CameraIntrinsics(1,1,1,1), CameraPose(np.eye(4)))]},
        depthmaps_by_frame={'cam_00:0': {'depth': np.ones((4,4,1), dtype=np.float32), 'confidence': np.ones((4,4), dtype=np.float32)}},
        pointmaps_by_frame={'cam_00:0': {'points': np.zeros((4,4,3), dtype=np.float32), 'confidence': np.ones((4,4), dtype=np.float32)}},
    )
    art.save()
    img = tmp_path / 'rgb.png'
    Image.fromarray(np.zeros((4,4,3), dtype=np.uint8)).save(img)
    cfg = {'enabled': True, 'geometry': {'enabled': True, 'depth_pngs': True, 'viewer': True, 'debug_assets': True, 'max_points': 10}}
    export_debug_artifacts(tmp_path, cfg, {'cam_00:0': str(img)})
    assert (save_dir / 'depthmaps_png' / 'cam_00__0.png').exists()
    assert (save_dir / 'debug_viewer.html').exists()
