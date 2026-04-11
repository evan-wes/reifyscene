from pathlib import Path

import numpy as np
from PIL import Image

from room4d.api.pipeline import Room4DPipeline
from room4d.common.types import CameraIntrinsics, CameraPose, FrameRef, TimedCameraState, VideoStreamRef
from room4d.debug_exports import apply_debug_config_to_runtime_cfg, export_debug_artifacts, load_debug_config, materialize_shared_debug_frames
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


def test_load_debug_config_includes_all_stage_sections(tmp_path: Path) -> None:
    cfg_path = tmp_path / 'debug.yaml'
    cfg_path.write_text("enabled: true\ngeometry:\n  viewer: false\n", encoding='utf-8')
    cfg = load_debug_config(str(cfg_path))
    for stage_name in ('ingest', 'sync', 'geometry', 'tracking', 'humans', 'objects', 'graph', 'optimization', 'scene', 'render'):
        assert stage_name in cfg


def test_apply_debug_config_to_runtime_cfg_plumbs_stage_viewer_flags() -> None:
    cfg = {
        'geometry': {'debug_shapes': False, 'debug_max_points': 5},
        'tracking': {'debug': False, 'debug_max_points': 7},
        'humans': {},
        'objects': {},
        'render': {'debug_triptych': True},
    }
    debug_cfg = {
        'enabled': True,
        'geometry': {'enabled': True, 'depth_pngs': True, 'debug_assets': False, 'viewer': True, 'max_points': 11},
        'tracking': {'enabled': True, 'debug_assets': False, 'viewer': True, 'max_points': 13, 'max_trails': 4, 'trail_window': 6},
        'humans': {'enabled': True, 'debug_assets': True, 'viewer': True},
        'objects': {'enabled': True, 'debug_assets': False, 'viewer': True},
        'optimization': {'enabled': True, 'viewer': True, 'max_points': 17, 'max_trails': 3, 'trail_window': 5},
        'render': {'enabled': True, 'viewer': False},
    }
    out = apply_debug_config_to_runtime_cfg(cfg, debug_cfg)
    assert out['geometry']['debug_shapes'] is True
    assert out['geometry']['save_depth_pngs'] is True
    assert out['geometry']['debug_max_points'] == 11
    assert out['tracking']['debug'] is True
    assert out['tracking']['debug_max_points'] == 13
    assert out['tracking']['debug_max_trails'] == 4
    assert out['tracking']['debug_trail_window'] == 6
    assert out['humans']['save_debug_assets'] is True
    assert out['humans']['save_debug_html'] is True
    assert out['objects']['save_debug_assets'] is False
    assert out['objects']['save_debug_html'] is True
    assert out['optimization']['save_refinement_viewer'] is True
    assert out['optimization']['debug_max_points'] == 17
    assert out['render']['debug_triptych'] is False
