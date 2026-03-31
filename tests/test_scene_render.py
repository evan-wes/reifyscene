from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from room4d.common.types import CameraIntrinsics, CameraPose, FrameRef, HumanState, ObjectState, TimedCameraState
from room4d.geometry.geometry_artifact import GeometryArtifact
from room4d.scene.assembler import SingleVideoSceneAssembler
from room4d.tracking.track_artifact import TrackArtifact
from room4d.perception.humans.human_artifact import HumanArtifact
from room4d.perception.objects.object_artifact import ObjectArtifact
from room4d.render.gaussian4d.scene_model import Gaussian4DScene
from room4d.render.gaussian4d.trainer import Gaussian4DTrainer


def _make_frame(tmp_path: Path, name: str) -> str:
    path = tmp_path / f"{name}.png"
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[..., 0] = 120
    rgb[..., 1] = 80
    rgb[..., 2] = 200
    Image.fromarray(rgb).save(path)
    return str(path)


def test_scene_assembly_and_render(tmp_path: Path) -> None:
    frame0 = FrameRef(stream_id='cam_00', frame_idx=0, timestamp_sec=0.0)
    frame1 = FrameRef(stream_id='cam_00', frame_idx=2, timestamp_sec=0.2)
    img0 = _make_frame(tmp_path, 'f0')
    img1 = _make_frame(tmp_path, 'f1')

    pts0 = np.dstack(np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8), indexing='xy'))
    z0 = np.full((8, 8, 1), 2.0, dtype=np.float32)
    pts0 = np.concatenate([pts0, z0], axis=2).astype(np.float32)
    pts1 = pts0 + np.array([0.0, 0.0, 0.1], dtype=np.float32)

    geometry = GeometryArtifact(
        run_id='scene',
        stage_name='geometry',
        save_dir=tmp_path / 'geometry',
        selected_frames=[frame0, frame1],
        intrinsics_by_stream={'cam_00': CameraIntrinsics(500, 500, 16, 16)},
        camera_states_by_stream={'cam_00': [
            TimedCameraState('cam_00', 0.0, CameraIntrinsics(500, 500, 16, 16), CameraPose(np.eye(4, dtype=np.float32))),
            TimedCameraState('cam_00', 0.2, CameraIntrinsics(500, 500, 16, 16), CameraPose(np.eye(4, dtype=np.float32))),
        ]},
        pointmaps_by_frame={
            'cam_00:0': {'points': pts0, 'confidence': np.ones((8, 8), dtype=np.float32)},
            'cam_00:2': {'points': pts1, 'confidence': np.ones((8, 8), dtype=np.float32)},
        },
        depthmaps_by_frame={},
        metadata={'debug_image_paths': {'cam_00:0': img0, 'cam_00:2': img1}},
    )

    track_payload = {
        'frame_indices': np.asarray([0, 2], dtype=np.int32),
        'points_xy': np.asarray([[[10, 10], [20, 18]], [[11, 10], [21, 18]]], dtype=np.float32),
        'visibility': np.asarray([[True, True], [True, True]]),
        'track_ids': np.asarray([1, 2], dtype=np.int32),
        'lifted_frame_indices': np.asarray([0, 2], dtype=np.int32),
        'lifted_xyz': np.asarray([
            [[0.0, 0.0, 2.0], [0.5, 0.1, 2.0]],
            [[0.0, 0.0, 2.1], [0.6, 0.1, 2.1]],
        ], dtype=np.float32),
        'lifted_valid': np.asarray([[True, True], [True, True]]),
        'refined_lifted_frame_indices': np.asarray([0, 2], dtype=np.int32),
        'refined_lifted_xyz': np.asarray([
            [[0.0, 0.0, 2.0], [0.5, 0.1, 2.0]],
            [[0.0, 0.0, 2.1], [0.6, 0.1, 2.1]],
        ], dtype=np.float32),
        'refined_lifted_valid': np.asarray([[True, True], [True, True]]),
        'static_labels': np.asarray([1, 0], dtype=np.int8),
    }
    track_artifact = TrackArtifact(
        run_id='scene', stage_name='tracking', save_dir=tmp_path / 'tracking',
        track_payloads_by_stream={'cam_00': track_payload}, selected_frames=[frame0, frame1]
    )

    human = HumanState(
        human_id='human_0', frame_ref=frame0, bbox_xyxy=np.asarray([8, 8, 18, 24], dtype=np.float32),
        mask_rle=None, keypoints_2d=None, densepose=None, smpl_params=None, confidence=0.9,
        attributes={'track_ids': [1], 'support_track_count': 1, 'detection_id': 'hum0'}
    )
    humans = HumanArtifact(run_id='scene', stage_name='humans', save_dir=tmp_path / 'humans', humans=[human], frame_index={'cam_00:0': ['human_0']})

    obj = ObjectState(
        object_id='object_0', frame_ref=frame1, category='ball', bbox_xyxy=np.asarray([18, 14, 28, 24], dtype=np.float32),
        mask_rle=None, embedding=None, confidence=0.8,
        attributes={'track_ids': [2], 'support_track_count': 1, 'detection_id': 'obj0'}
    )
    objects = ObjectArtifact(run_id='scene', stage_name='objects', save_dir=tmp_path / 'objects', objects=[obj], frame_index={'cam_00:2': ['object_0']})

    assembler = SingleVideoSceneAssembler(max_static_points_per_frame=32, max_static_points_total=64)
    scene_artifact = assembler.assemble(
        run_id='scene', save_dir=tmp_path / 'scene', geometry_artifact=geometry,
        track_artifact=track_artifact, human_artifact=humans, object_artifact=objects,
    )
    scene_artifact.save()

    assert scene_artifact.static_scene['num_points'] > 0
    assert len(scene_artifact.dynamic_entities['humans']) == 1
    assert len(scene_artifact.dynamic_entities['objects']) == 1

    trainer = Gaussian4DTrainer({'width': 320, 'height': 180, 'point_limit': 2000})
    scene_model = Gaussian4DScene()
    scene_model.initialize_from_scene(scene_artifact)
    render_artifact = trainer.fit(scene_model, geometry_artifact=geometry, save_dir=tmp_path / 'render')
    render_artifact.save_dir = tmp_path / 'render'
    render_artifact.save()

    assert (tmp_path / 'render' / 'novel_view_player.html').exists()
    assert 'orbit' in render_artifact.path_index
    assert render_artifact.coverage['rendered_frames'] > 0


def test_scene_timeline_source_tracking(tmp_path: Path) -> None:
    frame0 = FrameRef(stream_id='cam_00', frame_idx=0, timestamp_sec=0.0)
    frame1 = FrameRef(stream_id='cam_00', frame_idx=1, timestamp_sec=1.0 / 12.0)
    frame2 = FrameRef(stream_id='cam_00', frame_idx=2, timestamp_sec=2.0 / 12.0)
    img0 = _make_frame(tmp_path, 'tf0')
    img2 = _make_frame(tmp_path, 'tf2')

    pts = np.zeros((4, 4, 3), dtype=np.float32)
    pts[..., 2] = 2.0
    geometry = GeometryArtifact(
        run_id='scene2',
        stage_name='geometry',
        save_dir=tmp_path / 'geometry2',
        selected_frames=[frame0, frame2],
        intrinsics_by_stream={'cam_00': CameraIntrinsics(500, 500, 16, 16)},
        camera_states_by_stream={'cam_00': [
            TimedCameraState('cam_00', 0.0, CameraIntrinsics(500, 500, 16, 16), CameraPose(np.eye(4, dtype=np.float32))),
            TimedCameraState('cam_00', 2.0 / 12.0, CameraIntrinsics(500, 500, 16, 16), CameraPose(np.eye(4, dtype=np.float32))),
        ]},
        pointmaps_by_frame={
            'cam_00:0': {'points': pts, 'confidence': np.ones((4, 4), dtype=np.float32)},
            'cam_00:2': {'points': pts, 'confidence': np.ones((4, 4), dtype=np.float32)},
        },
        depthmaps_by_frame={},
        metadata={'debug_image_paths': {'cam_00:0': img0, 'cam_00:2': img2}},
    )

    track_artifact = TrackArtifact(
        run_id='scene2', stage_name='tracking', save_dir=tmp_path / 'tracking2',
        selected_frames=[frame0, frame1, frame2],
        track_payloads_by_stream={'cam_00': {'frame_indices': np.asarray([0, 1, 2], dtype=np.int32), 'points_xy': np.zeros((3, 0, 2), dtype=np.float32), 'visibility': np.zeros((3, 0), dtype=bool)}},
    )

    assembler = SingleVideoSceneAssembler(max_static_points_per_frame=8, max_static_points_total=16, timeline_source='geometry')
    scene_artifact = assembler.assemble(
        run_id='scene2', save_dir=tmp_path / 'scene2', geometry_artifact=geometry,
        track_artifact=track_artifact, metadata={'timeline_source': 'tracking'}
    )

    assert [fr.frame_idx for fr in scene_artifact.frame_refs] == [0, 1, 2]
    assert scene_artifact.timeline['source'] == 'tracking'
