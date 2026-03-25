"""Tests for object perception and debug artifact export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from room4d.common.types import FrameRef, ObjectState
from room4d.perception.objects.object_artifact import ObjectArtifact
from room4d.perception.objects.object_assoc import ObjectAssociator


def _make_obj(det_id: str, global_id: str, frame_idx: int, bbox, category='dog', embedding=None, track_ids=None, confidence=0.9):
    return ObjectState(
        object_id=global_id,
        frame_ref=FrameRef(stream_id='cam_00', frame_idx=frame_idx, timestamp_sec=float(frame_idx)),
        category=category,
        bbox_xyxy=np.asarray(bbox, dtype=np.float32),
        mask_rle={'shape': [24, 32], 'sum': 16},
        embedding=None if embedding is None else np.asarray(embedding, dtype=np.float32),
        confidence=confidence,
        attributes={
            'detection_id': det_id,
            'track_ids': list(track_ids or []),
            'frame_key': f'cam_00:{frame_idx}',
            'frame_stride': 1,
        },
    )


def test_object_association_prefers_shared_tracks_and_consensus() -> None:
    artifact = ObjectArtifact(run_id='dev', stage_name='objects', save_dir=Path('.'))
    artifact.objects = [
        _make_obj('det:0', 'det:0', 0, [1, 1, 8, 8], category='dog', embedding=[1, 0, 0], track_ids=[1, 2, 3]),
        _make_obj('det:1', 'det:1', 1, [2, 1, 9, 8], category='dog', embedding=[1, 0, 0], track_ids=[1, 2, 3]),
        _make_obj('det:2', 'det:2', 2, [3, 1, 10, 8], category='cat', embedding=[1, 0, 0], track_ids=[1, 2, 3], confidence=0.45),
        _make_obj('det:3', 'det:3', 1, [20, 1, 28, 8], category='bottle', embedding=[0, 1, 0], track_ids=[9]),
    ]
    out = ObjectAssociator({'min_match_score': 0.25}).associate(artifact)
    ids_by_det = {obj.attributes['detection_id']: obj.object_id for obj in out.objects}
    assert ids_by_det['det:0'] == ids_by_det['det:1'] == ids_by_det['det:2']
    assert ids_by_det['det:3'] != ids_by_det['det:0']
    dogish = [obj for obj in out.objects if obj.object_id == ids_by_det['det:0']]
    assert all(obj.category == 'dog' for obj in dogish)
    assert any(obj.attributes['current_category'] == 'cat' for obj in dogish)
    assert all(obj.attributes['consensus_category'] == 'dog' for obj in dogish)


def test_per_frame_duplicate_suppression_removes_overlapping_conflicts() -> None:
    artifact = ObjectArtifact(run_id='dev', stage_name='objects', save_dir=Path('.'))
    dog = _make_obj('det:dog', 'det:dog', 0, [2, 2, 10, 10], category='dog', embedding=[1, 0, 0], track_ids=[1, 2], confidence=0.95)
    cat = _make_obj('det:cat', 'det:cat', 0, [2.5, 2, 10.5, 10], category='cat', embedding=[1, 0, 0], track_ids=[1, 2], confidence=0.60)
    artifact.objects = [dog, cat]
    mask = np.pad(np.ones((8, 8), dtype=bool), ((2, 14), (2, 22)))
    artifact.mask_arrays = {'det:dog': mask, 'det:cat': mask.copy()}
    out = ObjectAssociator({'min_match_score': 0.2}).associate(artifact)
    assert len(out.objects) == 1
    assert out.objects[0].attributes['detection_id'] == 'det:dog'
    raw_cat = next(obj for obj in out.raw_objects if obj.attributes['detection_id'] == 'det:cat')
    assert raw_cat.attributes.get('suppressed') is True
    assert raw_cat.attributes.get('suppression_kind') == 'frame_duplicate'


def test_object_artifact_roundtrip_and_debug_export(tmp_path: Path) -> None:
    artifact = ObjectArtifact(
        run_id='demo',
        stage_name='objects',
        save_dir=tmp_path / 'objects',
        metadata={
            'save_debug_assets': True,
            'save_debug_html': True,
            'debug_image_paths': {},
        },
    )
    image_dir = artifact.save_dir / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    for idx in [0, 1]:
        rgb = np.zeros((24, 32, 3), dtype=np.uint8)
        rgb[..., 1] = 40 + idx * 20
        from PIL import Image
        img_path = image_dir / f'cam_00__{idx}.png'
        Image.fromarray(rgb).save(img_path)
        artifact.metadata['debug_image_paths'][f'cam_00:{idx}'] = str(img_path)
    artifact.objects = [
        _make_obj('det:0', 'obj_0000', 0, [1, 1, 8, 8], track_ids=[1, 2]),
        _make_obj('det:1', 'obj_0000', 1, [2, 1, 9, 8], category='cat', track_ids=[1, 2]),
    ]
    for obj in artifact.objects:
        obj.attributes.update({
            'current_category': obj.category,
            'consensus_category': 'dog',
            'support_track_count': 2,
            'lifespan_start': 0,
            'lifespan_end': 1,
        })
    artifact.raw_objects = [
        _make_obj('det:0', 'det:0', 0, [1, 1, 8, 8], track_ids=[1, 2]),
        _make_obj('det:1', 'det:1', 1, [2, 1, 9, 8], category='cat', track_ids=[1, 2]),
        _make_obj('det:supp', 'det:supp', 1, [2.2, 1.2, 9.0, 8.2], category='cat', track_ids=[1, 2], confidence=0.2),
    ]
    artifact.raw_objects[-1].attributes.update({
        'suppressed': True,
        'suppression_kind': 'frame_duplicate',
        'suppressed_by_detection_id': 'det:1',
    })
    artifact.mask_arrays = {
        'det:0': np.pad(np.ones((8, 8), dtype=bool), ((1, 15), (1, 23))),
        'det:1': np.pad(np.ones((8, 8), dtype=bool), ((1, 15), (2, 22))),
        'det:supp': np.pad(np.ones((8, 8), dtype=bool), ((1, 15), (2, 22))),
    }
    artifact.embedding_vectors = {
        'det:0': np.asarray([1, 0, 0], dtype=np.float32),
        'det:1': np.asarray([1, 0, 0], dtype=np.float32),
        'det:supp': np.asarray([1, 0, 0], dtype=np.float32),
    }
    artifact.linked_ids = {
        'track_summaries': {
            'obj_0000': {
                'global_id': 'obj_0000',
                'consensus_category': 'dog',
                'support_track_count': 2,
                'lifespan_start': 0,
                'lifespan_end': 1,
            },
        },
        'raw_detection_count': 3,
        'kept_detection_count': 2,
        'final_detection_count': 2,
        'suppressed_detection_count': 1,
    }
    artifact.frame_index = artifact._build_frame_index(artifact.objects)
    artifact.save()
    html = (artifact.save_dir / 'object_debug_viewer.html').read_text(encoding='utf-8')
    assert 'Raw detections' in html
    assert 'obj_0000' in html
    loaded = ObjectArtifact.load(artifact.save_dir)
    assert len(loaded.objects) == 2
    assert len(loaded.raw_objects) == 3
    assert 'det:0' in loaded.mask_arrays
