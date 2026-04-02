from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts.run_single_video_room4d import build_config


def _make_args(tmp_path: Path) -> Namespace:
    return Namespace(
        video=str(tmp_path / 'clip.mp4'),
        run_dir=str(tmp_path / 'run'),
        prior_run_dir=None,
        start_stage='ingest',
        config=None,
        debug_config=None,
        frame_cache_dir=None,
        max_keyframes=4,
        keyframe_stride_sec=1.0,
        geometry_backend_name='vggt-long',
        geometry_backend_mode='real',
        geometry_frame_source='tracking',
        geometry_frame_stride=3,
        geometry_frame_spacing_sec=0.5,
        geometry_device='cpu',
        geometry_resize_max_dim=None,
        vggt_long_chunk_size=8,
        vggt_long_overlap=2,
        vggt_long_auto_tune=False,
        vggt_long_min_chunk_size=2,
        vggt_long_min_resize_max_dim=None,
        vggt_long_loop_enable=False,
        vggt_long_use_dbow=False,
        vggt_long_delete_temp_files=False,
        vggt_long_output_cache_dir=None,
        tracking_backend_mode='auto',
        tracking_device='cpu',
        tracking_grid_size=12,
        tracking_frame_stride=2,
        tracking_frame_spacing_sec=0.25,
        tracking_max_frames=None,
        tracking_resize_max_dim=None,
        tracking_window_size=None,
        tracking_window_overlap=None,
        tracking_allow_github=False,
        humans_backend_mode='auto',
        humans_device='cpu',
        humans_score_threshold=0.5,
        humans_max_detections=6,
        humans_frame_source='tracking',
        humans_frame_stride=1,
        humans_pose_backend_mode='auto',
        humans_densepose_backend_mode='auto',
        humans_densepose_frame_stride=1,
        humans_smpl_backend_mode='auto',
        humans_smpl_frame_stride=1,
        objects_backend_mode='auto',
        objects_device='cpu',
        objects_frame_source='tracking',
        objects_frame_stride=1,
        objects_score_threshold=0.5,
        objects_max_detections=12,
        objects_embedder_backend_mode='auto',
        scene_timeline_source='tracking',
        resource_profile='none',
        cpu_threads=1,
        preflight_only=False,
        point_conf_threshold=0.0,
        static_threshold_world=0.05,
        min_lifted_frames=2,
        debug=False,
        debug_max_points=5000,
    )


def test_build_config_supports_geometry_and_tracking_time_spacing(tmp_path: Path) -> None:
    args = _make_args(tmp_path)
    cfg, _ = build_config(args, system_report={'ram_gb': 16.0})
    assert cfg['geometry']['frame_stride'] == 3
    assert cfg['geometry']['frame_spacing_sec'] == 0.5
    assert cfg['tracking']['frame_stride'] == 2
    assert cfg['tracking']['frame_spacing_sec'] == 0.25
    assert cfg['render']['offline_high_quality']['nerf_refinement']['enabled'] is True
