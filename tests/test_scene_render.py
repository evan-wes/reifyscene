from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from room4d.render.gaussian4d.rasterizer import gsplat_available

requires_gsplat = pytest.mark.skipif(
    not gsplat_available(), reason="rendering tests need gsplat and a CUDA device"
)

from room4d.common.types import CameraIntrinsics, CameraPose, FrameRef, HumanState, ObjectState, TimedCameraState
from room4d.geometry.geometry_artifact import GeometryArtifact
from room4d.scene.assembler import SingleVideoSceneAssembler
from room4d.scene.scene_artifact import SceneArtifact
from room4d.tracking.track_artifact import TrackArtifact
from room4d.perception.humans.human_artifact import HumanArtifact
from room4d.perception.objects.object_artifact import ObjectArtifact
from room4d.render.gaussian4d.scene_model import Gaussian4DScene
from room4d.render.gaussian4d.trainer import Gaussian4DTrainer
from room4d.render.registry import build_renderer
from room4d.render.camera_paths import CameraPathGenerator


def _make_frame(tmp_path: Path, name: str) -> str:
    path = tmp_path / f"{name}.png"
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[..., 0] = 120
    rgb[..., 1] = 80
    rgb[..., 2] = 200
    Image.fromarray(rgb).save(path)
    return str(path)


@requires_gsplat
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
    humans.mask_arrays = {'hum0': np.pad(np.ones((12, 10), dtype=bool), ((8, 12), (8, 14)))}

    obj = ObjectState(
        object_id='object_0', frame_ref=frame1, category='ball', bbox_xyxy=np.asarray([18, 14, 28, 24], dtype=np.float32),
        mask_rle=None, embedding=None, confidence=0.8,
        attributes={'track_ids': [2], 'support_track_count': 1, 'detection_id': 'obj0'}
    )
    objects = ObjectArtifact(run_id='scene', stage_name='objects', save_dir=tmp_path / 'objects', objects=[obj], frame_index={'cam_00:2': ['object_0']})
    objects.mask_arrays = {'obj0': np.pad(np.ones((10, 10), dtype=bool), ((14, 8), (18, 4)))}

    assembler = SingleVideoSceneAssembler(max_static_points_per_frame=32, max_static_points_total=64)
    scene_artifact = assembler.assemble(
        run_id='scene', save_dir=tmp_path / 'scene', geometry_artifact=geometry,
        track_artifact=track_artifact, human_artifact=humans, object_artifact=objects,
    )
    scene_artifact.save()

    assert scene_artifact.static_scene['num_points'] > 0
    assert len(scene_artifact.dynamic_entities['humans']) == 1
    assert len(scene_artifact.dynamic_entities['objects']) == 1
    assert scene_artifact.render_contract['schema'] == 'layered_scene_v2'
    assert scene_artifact.render_contract['appearance']['static_source_count'] == 2
    assert scene_artifact.render_contract['appearance']['dynamic_observation_counts']['humans'] == 1
    assert scene_artifact.render_contract['appearance']['dynamic_observation_counts']['objects'] == 1
    assert (tmp_path / 'scene' / 'render_contract' / 'static_gaussians.npz').exists()

    loaded_scene = SceneArtifact.load(tmp_path / 'scene')
    static_sources = loaded_scene.render_contract['appearance']['static_sources']
    assert len(static_sources) == 2
    assert all(Path(src['image_file']).exists() for src in static_sources)
    assert all(Path(src['points_file']).exists() for src in static_sources)
    loaded_dynamic = loaded_scene.render_contract['dynamic_layers']
    human_obs = loaded_dynamic['humans'][0]['observations']
    object_obs = loaded_dynamic['objects'][0]['observations']
    assert human_obs and object_obs
    assert Path(human_obs[0]['sprite_file']).exists()
    assert Path(object_obs[0]['sprite_file']).exists()
    assert (tmp_path / 'scene' / 'render_entity_summary.json').exists()
    assert 'mask_coverage' in human_obs[0]
    assert 'camera_center_xyz' in human_obs[0]
    assert 'view_dir_xyz' in human_obs[0]

    trainer = Gaussian4DTrainer({'width': 320, 'height': 180, 'point_limit': 2000})
    scene_model = Gaussian4DScene()
    scene_model.initialize_from_scene(loaded_scene)
    render_artifact = trainer.fit(scene_model, geometry_artifact=geometry, save_dir=tmp_path / 'render')
    render_artifact.save_dir = tmp_path / 'render'
    render_artifact.save()

    assert (tmp_path / 'render' / 'novel_view_player.html').exists()
    assert (tmp_path / 'render' / 'novel_view_debug_player.html').exists()
    assert 'orbit' in render_artifact.path_index
    assert 'orbit' in render_artifact.layer_path_index
    assert render_artifact.coverage['rendered_frames'] > 0

    debug_layers = render_artifact.layer_path_index['orbit']
    assert set(debug_layers) == {'full', 'static_only', 'dynamic_only', 'triptych'}
    assert all(debug_layers[layer] for layer in ('full', 'static_only', 'dynamic_only', 'triptych'))

    frame_img = np.asarray(Image.open(tmp_path / 'render' / render_artifact.path_index['orbit'][0]))
    assert frame_img.sum() > 0
    nonzero = np.count_nonzero(frame_img.sum(axis=2) > 0)
    assert nonzero > 100


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


def test_camera_paths_use_z_up_debug_frame() -> None:
    frame0 = FrameRef(stream_id='cam_00', frame_idx=0, timestamp_sec=0.0)
    frame1 = FrameRef(stream_id='cam_00', frame_idx=1, timestamp_sec=0.1)
    scene_artifact = SceneArtifact(
        run_id='zup', stage_name='scene', save_dir=Path('.'),
        static_scene={
            'points_xyz': np.asarray([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            'colors_rgb': np.asarray([[255, 255, 255]] * 3, dtype=np.uint8),
            'center_xyz': [0.0, 0.0, 0.0],
            'num_points': 3,
        },
        dynamic_entities={'humans': [], 'objects': []},
        frame_refs=[frame0, frame1],
        timeline={'source': 'geometry'},
    )
    path = CameraPathGenerator().orbit(scene_artifact, radius=2.0)
    cams = path['cameras']
    zs = [cam['eye'][2] for cam in cams]
    assert np.allclose(zs, zs[0])
    radii = [np.linalg.norm(np.asarray(cam['eye'][:2]) - np.asarray(cam['target'][:2])) for cam in cams]
    assert np.allclose(radii, radii[0])


@requires_gsplat
def test_render_respects_enabled_paths_and_limits(tmp_path: Path) -> None:
    frame0 = FrameRef(stream_id="cam_00", frame_idx=0, timestamp_sec=0.0)
    frame1 = FrameRef(stream_id="cam_00", frame_idx=1, timestamp_sec=0.1)
    frame2 = FrameRef(stream_id="cam_00", frame_idx=2, timestamp_sec=0.2)
    scene_artifact = SceneArtifact(
        run_id='cfg', stage_name='scene', save_dir=tmp_path / 'scene_cfg',
        static_scene={
            'points_xyz': np.asarray([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0], [0.0, 0.2, 2.0]], dtype=np.float32),
            'colors_rgb': np.asarray([[255, 255, 255]] * 3, dtype=np.uint8),
            'center_xyz': [0.0, 0.0, 2.0],
            'num_points': 3,
        },
        dynamic_entities={'humans': [], 'objects': []},
        frame_refs=[frame0, frame1, frame2],
        timeline={'source': 'tracking'},
    )
    scene_model = Gaussian4DScene()
    scene_model.initialize_from_scene(scene_artifact)
    trainer = Gaussian4DTrainer({
        'enabled_paths': {'orbit': True, 'overhead': False, 'replay': False, 'follow_human': False, 'follow_object': False},
        'frame_stride': 2,
        'max_frames_per_path': 1,
        'point_limit': 100,
    })
    artifact = trainer.fit(scene_model, save_dir=tmp_path / 'render_cfg')
    assert set(artifact.path_index.keys()) == {'orbit'}
    assert len(artifact.path_index['orbit']) == 1



@requires_gsplat
def test_legacy_backend_alias_writes_debug_triptych(tmp_path: Path) -> None:
    frame0 = FrameRef(stream_id="cam_00", frame_idx=0, timestamp_sec=0.0)
    frame1 = FrameRef(stream_id="cam_00", frame_idx=1, timestamp_sec=0.1)
    static_img = _make_frame(tmp_path, 'hq_static')
    scene_artifact = SceneArtifact(
        run_id='hq', stage_name='scene', save_dir=tmp_path / 'scene_hq',
        static_scene={
            'points_xyz': np.asarray([[0.0, 0.0, 2.0], [0.3, 0.0, 2.0], [0.0, 0.3, 2.0]], dtype=np.float32),
            'colors_rgb': np.asarray([[255, 255, 255], [200, 180, 255], [180, 255, 220]], dtype=np.uint8),
            'center_xyz': [0.0, 0.0, 2.0],
            'num_points': 3,
        },
        dynamic_entities={'humans': [], 'objects': []},
        frame_refs=[frame0, frame1],
        timeline={'source': 'tracking'},
        render_contract={
            'schema': 'layered_scene_v2',
            'supported_backends': ['gaussian_layered', 'offline_high_quality'],
            'quality_modes': ['realtime', 'offline_high_quality'],
            'world_frame': {'up_axis': 'z', 'gravity_dir': [0.0, 0.0, -1.0]},
            'static_layer': {
                'num_gaussians': 3,
                'means_xyz': np.asarray([[0.0, 0.0, 2.0], [0.3, 0.0, 2.0], [0.0, 0.3, 2.0]], dtype=np.float32),
                'scales_xyz': np.full((3, 3), 0.04, dtype=np.float32),
                'colors_rgb': np.asarray([[255, 255, 255], [200, 180, 255], [180, 255, 220]], dtype=np.uint8),
                'opacities': np.asarray([0.9, 0.9, 0.9], dtype=np.float32),
                'source_codes': np.asarray([0, 0, 0], dtype=np.int16),
            },
            'appearance': {
                'static_sources': [
                    {
                        'frame_key': 'cam_00:0',
                        'time_sec': 0.0,
                        'image_path': static_img,
                        'points_xyz': np.full((8, 8, 3), [0.0, 0.0, 2.0], dtype=np.float32),
                        'confidence': np.ones((8, 8), dtype=np.float32),
                    }
                ],
                'static_source_count': 1,
                'dynamic_observation_counts': {'humans': 0, 'objects': 0},
            },
            'dynamic_layers': {
                'humans': [],
                'objects': [],
            },
        },
    )
    scene_artifact.save()
    loaded_scene = SceneArtifact.load(tmp_path / 'scene_hq')
    assert Path(loaded_scene.render_contract['appearance']['static_sources'][0]['image_file']).exists()

    scene_model = Gaussian4DScene()
    scene_model.initialize_from_scene(loaded_scene)
    # 'offline_high_quality' is a legacy alias kept so older run configs keep
    # resolving; it builds the same gsplat-backed renderer.
    renderer = build_renderer({
        'backend_name': 'offline_high_quality',
        'width': 160,
        'height': 90,
        'render_scale': 2,
        'enabled_paths': {'orbit': True, 'overhead': False, 'replay': False, 'follow_human': False, 'follow_object': False},
    })
    artifact = renderer.fit(scene_model, save_dir=tmp_path / 'render_hq')
    assert artifact.scene_state['backend_name'] == 'offline_high_quality'
    assert artifact.scene_state['rasterizer'] == 'gsplat'
    assert artifact.scene_state['render_scale'] == 2
    assert artifact.debug_player_html == 'novel_view_debug_player.html'
    assert (tmp_path / 'render_hq' / 'novel_view_debug_player.html').exists()
    assert 'orbit' in artifact.layer_path_index
    assert all(artifact.layer_path_index['orbit'][key] for key in ('full', 'static_only', 'dynamic_only', 'triptych'))

@requires_gsplat
def test_dynamic_entities_render_as_template_gaussians(tmp_path: Path) -> None:
    frame0 = FrameRef(stream_id='cam_00', frame_idx=0, timestamp_sec=0.0)
    sprite_rgba = np.zeros((16, 12, 4), dtype=np.uint8)
    sprite_rgba[..., 0] = 220
    sprite_rgba[..., 1] = 120
    sprite_rgba[..., 2] = 80
    sprite_rgba[..., 3] = 180

    scene_artifact = SceneArtifact(
        run_id='sprite', stage_name='scene', save_dir=tmp_path / 'scene_sprite',
        static_scene={
            'points_xyz': np.asarray([[0.0, 0.0, 2.5]], dtype=np.float32),
            'colors_rgb': np.asarray([[255, 255, 255]], dtype=np.uint8),
            'center_xyz': [0.0, 0.0, 2.5],
            'num_points': 1,
        },
        dynamic_entities={'humans': [], 'objects': []},
        frame_refs=[frame0],
        timeline={'source': 'tracking'},
        render_contract={
            'schema': 'layered_scene_v2',
            'supported_backends': ['gaussian_layered', 'offline_high_quality'],
            'quality_modes': ['realtime', 'offline_high_quality'],
            'world_frame': {'up_axis': 'z', 'gravity_dir': [0.0, 0.0, -1.0]},
            'static_layer': {
                'num_gaussians': 1,
                'means_xyz': np.asarray([[0.0, 0.0, 2.5]], dtype=np.float32),
                'scales_xyz': np.asarray([[0.04, 0.04, 0.04]], dtype=np.float32),
                'colors_rgb': np.asarray([[255, 255, 255]], dtype=np.uint8),
                'opacities': np.asarray([0.9], dtype=np.float32),
                'source_codes': np.asarray([0], dtype=np.int16),
            },
            'appearance': {
                'static_sources': [],
                'static_source_count': 0,
                'dynamic_observation_counts': {'humans': 1, 'objects': 0},
            },
            'dynamic_layers': {
                'humans': [{
                    'entity_id': 'human_0',
                    'entity_type': 'human',
                    'display_label': 'human_0',
                    'anchor_xyz': [0.0, 0.0, 2.0],
                    'times_sec': [0.0],
                    'centers_xyz': [[0.0, 0.0, 2.0]],
                    'template_offsets_xyz': [[0.0, 0.0, 0.0]],
                    'template_scales_xyz': [[0.12, 0.12, 0.12]],
                    'template_opacities': [0.8],
                    'color_rgb': [220, 120, 80],
                    'support_counts': [1],
                    'observations': [{
                        'time_sec': 0.0,
                        'world_size_wh': [0.6, 1.4],
                        'sprite_rgba': sprite_rgba,
                    }],
                }],
                'objects': [],
            },
        },
    )
    scene_model = Gaussian4DScene()
    scene_model.initialize_from_scene(scene_artifact)
    renderer = build_renderer({
        'backend_name': 'gaussian4d',
        'width': 160,
        'height': 90,
        'enabled_paths': {'orbit': True, 'overhead': False, 'replay': False, 'follow_human': False, 'follow_object': False},
    })
    artifact = renderer.fit(scene_model, save_dir=tmp_path / 'render_sprite')
    assert artifact.path_index['orbit']
    frame_img = np.asarray(Image.open(tmp_path / 'render_sprite' / artifact.path_index['orbit'][0]))
    assert frame_img.sum() > 0
    # The dynamic entity must be visible in the dynamic-only debug layer
    # (template Gaussians via gsplat; sprite observations are ignored).
    dynamic_rel = artifact.layer_path_index['orbit']['dynamic_only'][0]
    dynamic_img = np.asarray(Image.open(tmp_path / 'render_sprite' / dynamic_rel))
    assert dynamic_img.sum() > 0


def test_scene_assembler_estimates_dynamic_center_from_masked_point_region(tmp_path: Path) -> None:
    frame0 = FrameRef(stream_id='cam_00', frame_idx=0, timestamp_sec=0.0)
    img0 = _make_frame(tmp_path, 'center_mask')
    pts = np.zeros((16, 16, 3), dtype=np.float32)
    pts[..., 2] = 10.0
    pts[4:12, 5:11, 2] = 2.0
    geometry = GeometryArtifact(
        run_id='mask_center',
        stage_name='geometry',
        save_dir=tmp_path / 'geometry_mask_center',
        selected_frames=[frame0],
        intrinsics_by_stream={'cam_00': CameraIntrinsics(500, 500, 8, 8)},
        camera_states_by_stream={'cam_00': [TimedCameraState('cam_00', 0.0, CameraIntrinsics(500, 500, 8, 8), CameraPose(np.eye(4, dtype=np.float32)))]},
        pointmaps_by_frame={'cam_00:0': {'points': pts, 'confidence': np.ones((16, 16), dtype=np.float32)}},
        depthmaps_by_frame={},
        metadata={'debug_image_paths': {'cam_00:0': img0}},
    )
    track_artifact = TrackArtifact(
        run_id='mask_center', stage_name='tracking', save_dir=tmp_path / 'tracking_mask_center',
        track_payloads_by_stream={'cam_00': {'frame_indices': np.asarray([0], dtype=np.int32), 'points_xy': np.zeros((1, 0, 2), dtype=np.float32), 'visibility': np.zeros((1, 0), dtype=bool)}}, selected_frames=[frame0]
    )
    human = HumanState(
        human_id='human_main', frame_ref=frame0, bbox_xyxy=np.asarray([5, 4, 11, 12], dtype=np.float32),
        mask_rle=None, keypoints_2d=None, densepose=None, smpl_params=None, confidence=0.95,
        attributes={'track_ids': [], 'support_track_count': 0, 'detection_id': 'hum_main'}
    )
    humans = HumanArtifact(run_id='mask_center', stage_name='humans', save_dir=tmp_path / 'humans_mask_center', humans=[human], frame_index={'cam_00:0': ['human_main']})
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 5:11] = True
    humans.mask_arrays = {'hum_main': mask}
    assembler = SingleVideoSceneAssembler(max_static_points_per_frame=32, max_static_points_total=64)
    scene_artifact = assembler.assemble(
        run_id='mask_center', save_dir=tmp_path / 'scene_mask_center', geometry_artifact=geometry,
        track_artifact=track_artifact, human_artifact=humans,
    )
    ent = scene_artifact.dynamic_entities['humans'][0]
    assert np.allclose(ent['anchor_xyz'][2], 2.0, atol=0.3)


def test_dynamic_layer_selection_prefers_render_priority(tmp_path: Path) -> None:
    frame0 = FrameRef(stream_id='cam_00', frame_idx=0, timestamp_sec=0.0)
    scene_artifact = SceneArtifact(
        run_id='priority', stage_name='scene', save_dir=tmp_path / 'scene_priority',
        static_scene={'points_xyz': np.zeros((0, 3), dtype=np.float32), 'colors_rgb': np.zeros((0, 3), dtype=np.uint8), 'center_xyz': [0.0, 0.0, 0.0], 'num_points': 0},
        dynamic_entities={'humans': [], 'objects': []},
        frame_refs=[frame0],
        timeline={'source': 'tracking'},
        render_contract={
            'schema': 'layered_scene_v2',
            'dynamic_layers': {
                'humans': [
                    {
                        'entity_id': 'dancer', 'entity_type': 'human', 'display_label': 'person', 'color_rgb': [255, 0, 0],
                        'times_sec': [0.0], 'centers_xyz': [[0.0, 0.0, 2.0]], 'anchor_xyz': [0.0, 0.0, 2.0],
                        'support_counts': [1], 'render_priority': 10.0,
                        'template_offsets_xyz': [[0.0, 0.0, 0.0]], 'template_scales_xyz': [[0.1, 0.1, 0.1]], 'template_opacities': [0.8],
                        'observations': [{'time_sec': 0.0, 'render_score': 1.0, 'center_xyz': [0.0, 0.0, 2.0]}],
                    },
                    {
                        'entity_id': 'background_person', 'entity_type': 'human', 'display_label': 'person', 'color_rgb': [0, 255, 0],
                        'times_sec': [0.0], 'centers_xyz': [[1.0, 0.0, 3.0]], 'anchor_xyz': [1.0, 0.0, 3.0],
                        'support_counts': [4], 'render_priority': 1.0,
                        'template_offsets_xyz': [[0.0, 0.0, 0.0]], 'template_scales_xyz': [[0.1, 0.1, 0.1]], 'template_opacities': [0.8],
                        'observations': [{'time_sec': 0.0, 'render_score': 0.1, 'center_xyz': [1.0, 0.0, 3.0]}],
                    },
                ],
                'objects': [],
            },
        },
    )
    trainer = Gaussian4DTrainer({'max_dynamic_entities_per_kind': 1})
    selected = trainer._select_dynamic_layers(scene_artifact, 0.0)
    assert len(selected) == 1
    assert selected[0]['entity_id'] == 'dancer'


def test_human_smpl_pelvis_anchor_overrides_bad_support_track(tmp_path: Path) -> None:
    frame0 = FrameRef(stream_id='cam_00', frame_idx=0, timestamp_sec=0.0)
    pts = np.zeros((16, 16, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:16, 0:16]
    pts[..., 0] = (xx - 8) * 0.05
    pts[..., 1] = (yy - 8) * 0.05
    pts[..., 2] = 2.0
    pts[8:11, 7:10, 2] = 1.15  # pelvis neighborhood
    pts[12:15, 6:11, 2] = 0.20  # feet neighborhood, slightly above floor before grounding
    geometry = GeometryArtifact(
        run_id='smpl_anchor',
        stage_name='geometry',
        save_dir=tmp_path / 'geometry_smpl_anchor',
        selected_frames=[frame0],
        intrinsics_by_stream={'cam_00': CameraIntrinsics(500, 500, 8, 8)},
        camera_states_by_stream={'cam_00': [
            TimedCameraState('cam_00', 0.0, CameraIntrinsics(500, 500, 8, 8), CameraPose(np.eye(4, dtype=np.float32))),
        ]},
        pointmaps_by_frame={'cam_00:0': {'points': pts, 'confidence': np.ones((16, 16), dtype=np.float32)}},
        depthmaps_by_frame={},
        metadata={'world_frame': {'type': 'ground_plane_z_up'}},
    )
    track_payload = {
        'frame_indices': np.asarray([0], dtype=np.int32),
        'points_xy': np.asarray([[[8, 8]]], dtype=np.float32),
        'visibility': np.asarray([[True]]),
        'track_ids': np.asarray([1], dtype=np.int32),
        'lifted_frame_indices': np.asarray([0], dtype=np.int32),
        'lifted_xyz': np.asarray([[[5.0, 0.0, 5.0]]], dtype=np.float32),
        'lifted_valid': np.asarray([[True]]),
        'refined_lifted_frame_indices': np.asarray([0], dtype=np.int32),
        'refined_lifted_xyz': np.asarray([[[5.0, 0.0, 5.0]]], dtype=np.float32),
        'refined_lifted_valid': np.asarray([[True]]),
    }
    track_artifact = TrackArtifact(
        run_id='smpl_anchor', stage_name='tracking', save_dir=tmp_path / 'tracking_smpl_anchor',
        track_payloads_by_stream={'cam_00': track_payload}, selected_frames=[frame0]
    )
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[11] = np.asarray([7.5, 9.0, 1.0], dtype=np.float32)
    keypoints[12] = np.asarray([8.5, 9.0, 1.0], dtype=np.float32)
    keypoints[15] = np.asarray([7.2, 13.0, 1.0], dtype=np.float32)
    keypoints[16] = np.asarray([8.8, 13.0, 1.0], dtype=np.float32)
    human = HumanState(
        human_id='human_main', frame_ref=frame0, bbox_xyxy=np.asarray([5, 5, 11, 15], dtype=np.float32),
        mask_rle=None, keypoints_2d=keypoints, densepose=None,
        smpl_params={'valid': True, 'transl': np.asarray([8.0, 9.0, 4.0], dtype=np.float32)}, confidence=0.95,
        attributes={'track_ids': [1], 'support_track_count': 1, 'detection_id': 'hum_main'}
    )
    humans = HumanArtifact(run_id='smpl_anchor', stage_name='humans', save_dir=tmp_path / 'humans_smpl_anchor', humans=[human], frame_index={'cam_00:0': ['human_main']})
    mask = np.zeros((16, 16), dtype=bool)
    mask[5:15, 5:11] = True
    humans.mask_arrays = {'hum_main': mask}
    assembler = SingleVideoSceneAssembler(max_static_points_per_frame=32, max_static_points_total=64)
    scene_artifact = assembler.assemble(
        run_id='smpl_anchor', save_dir=tmp_path / 'scene_smpl_anchor', geometry_artifact=geometry,
        track_artifact=track_artifact, human_artifact=humans,
    )
    ent = scene_artifact.dynamic_entities['humans'][0]
    assert ent['center_sources'][0].startswith('smpl_root')
    # Support track is wildly wrong; anchored center should stay near the lifted SMPL pelvis neighborhood.
    assert abs(ent['anchor_xyz'][0]) < 0.5
    assert ent['anchor_xyz'][2] < 1.2


def test_human_smpl_temporal_smoothing_reduces_anchor_jump(tmp_path: Path) -> None:
    frame0 = FrameRef(stream_id='cam_00', frame_idx=0, timestamp_sec=0.0)
    frame1 = FrameRef(stream_id='cam_00', frame_idx=1, timestamp_sec=0.1)
    pts0 = np.zeros((16, 16, 3), dtype=np.float32)
    pts1 = np.zeros((16, 16, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:16, 0:16]
    for pts, z_pelvis in ((pts0, 1.0), (pts1, 2.2)):
        pts[..., 0] = (xx - 8) * 0.05
        pts[..., 1] = (yy - 8) * 0.05
        pts[..., 2] = 2.0
        pts[8:11, 7:10, 2] = z_pelvis
        pts[12:15, 6:11, 2] = 0.0
    geometry = GeometryArtifact(
        run_id='smooth', stage_name='geometry', save_dir=tmp_path / 'geometry_smooth',
        selected_frames=[frame0, frame1],
        intrinsics_by_stream={'cam_00': CameraIntrinsics(500, 500, 8, 8)},
        camera_states_by_stream={'cam_00': [
            TimedCameraState('cam_00', 0.0, CameraIntrinsics(500, 500, 8, 8), CameraPose(np.eye(4, dtype=np.float32))),
            TimedCameraState('cam_00', 0.1, CameraIntrinsics(500, 500, 8, 8), CameraPose(np.eye(4, dtype=np.float32))),
        ]},
        pointmaps_by_frame={
            'cam_00:0': {'points': pts0, 'confidence': np.ones((16, 16), dtype=np.float32)},
            'cam_00:1': {'points': pts1, 'confidence': np.ones((16, 16), dtype=np.float32)},
        },
        depthmaps_by_frame={},
        metadata={'world_frame': {'type': 'ground_plane_z_up'}},
    )
    keypoints0 = np.zeros((17, 3), dtype=np.float32)
    keypoints1 = np.zeros((17, 3), dtype=np.float32)
    for keypoints in (keypoints0, keypoints1):
        keypoints[11] = np.asarray([7.5, 9.0, 1.0], dtype=np.float32)
        keypoints[12] = np.asarray([8.5, 9.0, 1.0], dtype=np.float32)
        keypoints[15] = np.asarray([7.2, 13.0, 1.0], dtype=np.float32)
        keypoints[16] = np.asarray([8.8, 13.0, 1.0], dtype=np.float32)
    humans = HumanArtifact(
        run_id='smooth', stage_name='humans', save_dir=tmp_path / 'humans_smooth',
        humans=[
            HumanState('human_main', frame0, np.asarray([5, 5, 11, 15], dtype=np.float32), None, keypoints0, None, {'valid': True, 'transl': np.asarray([8.0, 9.0, 4.0], dtype=np.float32)}, 0.95, {'track_ids': [], 'detection_id': 'hum0'}),
            HumanState('human_main', frame1, np.asarray([5, 5, 11, 15], dtype=np.float32), None, keypoints1, None, {'valid': True, 'transl': np.asarray([8.0, 9.0, 4.0], dtype=np.float32)}, 0.95, {'track_ids': [], 'detection_id': 'hum1'}),
        ],
        frame_index={'cam_00:0': ['human_main'], 'cam_00:1': ['human_main']},
    )
    humans.mask_arrays = {'hum0': np.pad(np.ones((10, 6), dtype=bool), ((5, 1), (5, 5))), 'hum1': np.pad(np.ones((10, 6), dtype=bool), ((5, 1), (5, 5)))}
    assembler = SingleVideoSceneAssembler(human_anchor_temporal_alpha=0.5, human_anchor_search_radius_px=2.0, human_foot_search_radius_px=2.0)
    scene_artifact = assembler.assemble(
        run_id='smooth', save_dir=tmp_path / 'scene_smooth', geometry_artifact=geometry,
        track_artifact=TrackArtifact(run_id='smooth', stage_name='tracking', save_dir=tmp_path / 'tracking_smooth', track_payloads_by_stream={'cam_00': {'frame_indices': np.asarray([0, 1], dtype=np.int32), 'points_xy': np.zeros((2, 0, 2), dtype=np.float32), 'visibility': np.zeros((2, 0), dtype=bool)}}, selected_frames=[frame0, frame1]),
        human_artifact=humans,
    )
    ent = scene_artifact.dynamic_entities['humans'][0]
    z0 = float(ent['centers_3d'][0][2])
    z1 = float(ent['centers_3d'][1][2])
    assert abs(z0 - 1.0) < 0.2
    assert 1.4 < z1 < 1.8  # smoothed between raw values 1.0 and 2.2
