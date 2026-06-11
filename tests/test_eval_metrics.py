from __future__ import annotations

from pathlib import Path

from room4d.eval.benchmark_runner import BenchmarkRunner
from room4d.render.gaussian4d.scene_model import Gaussian4DScene
from room4d.render.gaussian4d.trainer import Gaussian4DTrainer
from tests.test_scene_render import _make_frame, requires_gsplat
from room4d.common.types import CameraIntrinsics, CameraPose, FrameRef, HumanState, TimedCameraState
from room4d.geometry.geometry_artifact import GeometryArtifact
from room4d.perception.humans.human_artifact import HumanArtifact
from room4d.scene.assembler import SingleVideoSceneAssembler
from room4d.tracking.track_artifact import TrackArtifact
import numpy as np


@requires_gsplat
def test_benchmark_runner_writes_reports(tmp_path: Path) -> None:
    frame0 = FrameRef(stream_id='cam_00', frame_idx=0, timestamp_sec=0.0)
    frame1 = FrameRef(stream_id='cam_00', frame_idx=1, timestamp_sec=0.1)
    img0 = _make_frame(tmp_path, 'eval0')
    img1 = _make_frame(tmp_path, 'eval1')

    pts0 = np.zeros((8, 8, 3), dtype=np.float32)
    pts0[..., 0], pts0[..., 1] = np.meshgrid(np.linspace(-0.5, 0.5, 8), np.linspace(-0.5, 0.5, 8), indexing='xy')
    pts0[..., 2] = 2.0
    pts1 = pts0.copy()
    geometry = GeometryArtifact(
        run_id='eval', stage_name='geometry', save_dir=tmp_path / 'geometry',
        selected_frames=[frame0, frame1],
        intrinsics_by_stream={'cam_00': CameraIntrinsics(320, 320, 16, 16)},
        camera_states_by_stream={'cam_00': [
            TimedCameraState('cam_00', 0.0, CameraIntrinsics(320, 320, 16, 16), CameraPose(np.eye(4, dtype=np.float32))),
            TimedCameraState('cam_00', 0.1, CameraIntrinsics(320, 320, 16, 16), CameraPose(np.eye(4, dtype=np.float32))),
        ]},
        pointmaps_by_frame={
            'cam_00:0': {'points': pts0, 'confidence': np.ones((8, 8), dtype=np.float32)},
            'cam_00:1': {'points': pts1, 'confidence': np.ones((8, 8), dtype=np.float32)},
        },
        depthmaps_by_frame={},
        metadata={
            'debug_image_paths': {'cam_00:0': img0, 'cam_00:1': img1},
            'world_frame': {'normal_world': [0.0, 0.0, 1.0], 'inlier_ratio': 0.9, 'num_inliers': 50, 'candidate_planes': []},
        },
        notes=[],
    )
    track_payload = {
        'frame_indices': np.asarray([0, 1], dtype=np.int32),
        'points_xy': np.asarray([[[16.0, 16.0]], [[16.0, 16.0]]], dtype=np.float32),
        'visibility': np.asarray([[True], [True]]),
        'track_ids': np.asarray([1], dtype=np.int32),
        'refined_lifted_frame_indices': np.asarray([0, 1], dtype=np.int32),
        'refined_lifted_xyz': np.asarray([[[0.0, 0.0, 2.0]], [[0.0, 0.0, 2.0]]], dtype=np.float32),
        'refined_lifted_valid': np.asarray([[True], [True]]),
    }
    tracking = TrackArtifact(run_id='eval', stage_name='tracking', save_dir=tmp_path / 'tracking', selected_frames=[frame0, frame1], track_payloads_by_stream={'cam_00': track_payload})
    human = HumanState(
        human_id='human_0', frame_ref=frame0, bbox_xyxy=np.asarray([10, 8, 22, 28], dtype=np.float32),
        mask_rle=None, keypoints_2d=None, densepose=None, smpl_params=None, confidence=0.95,
        attributes={'track_ids': [1], 'support_track_count': 1, 'detection_id': 'hum0'}
    )
    humans = HumanArtifact(run_id='eval', stage_name='humans', save_dir=tmp_path / 'humans', humans=[human], frame_index={'cam_00:0': ['human_0']})
    humans.mask_arrays = {'hum0': np.pad(np.ones((10, 8), dtype=bool), ((8, 14), (10, 14)))}

    scene = SingleVideoSceneAssembler(max_static_points_per_frame=16, max_static_points_total=32).assemble(
        run_id='eval', save_dir=tmp_path / 'scene', geometry_artifact=geometry, track_artifact=tracking, human_artifact=humans
    )
    scene_model = Gaussian4DScene()
    scene_model.initialize_from_scene(scene)
    renderer = Gaussian4DTrainer({'width': 96, 'height': 64, 'enabled_paths': {'orbit': True, 'overhead': False, 'replay': False, 'follow_human': False, 'follow_object': False}})
    render = renderer.fit(scene_model, geometry_artifact=geometry, save_dir=tmp_path / 'render')

    report = BenchmarkRunner({'rendering': {'max_views': 2}}).run(
        geometry_artifact=geometry,
        track_artifact=tracking,
        human_artifact=humans,
        scene_artifact=scene,
        render_artifact=render,
        renderer=renderer,
        output_dir=tmp_path / 'eval',
    )
    assert 'metrics' in report and 'failure_report' in report
    assert 'reprojection_error_mean_px' in report['metrics']
    assert 'heldout_psnr_mean' in report['metrics'] or report['metrics'].get('num_heldout_views', 0) == 0
    assert (tmp_path / 'eval' / 'evaluation_summary.json').exists()
    assert (tmp_path / 'eval' / 'evaluation_report.html').exists()
