
"""Tests for object perception and debug artifact export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from room4d.common.types import FrameRef, ObjectState, TrackObservation
from room4d.perception.objects.object_artifact import ObjectArtifact
from room4d.perception.objects.object_assoc import ObjectAssociator


def _make_obj(det_id: str, global_id: str, frame_idx: int, bbox, category='dog', embedding=None, track_ids=None):
    return ObjectState(
        object_id=global_id,
        frame_ref=FrameRef(stream_id='cam_00', frame_idx=frame_idx, timestamp_sec=float(frame_idx)),
        category=category,
        bbox_xyxy=np.asarray(bbox, dtype=np.float32),
        mask_rle={'shape': [24, 32], 'sum': 16},
        embedding=None if embedding is None else np.asarray(embedding, dtype=np.float32),
        confidence=0.9,
        attributes={
            'detection_id': det_id,
            'track_ids': list(track_ids or []),
            'frame_key': f'cam_00:{frame_idx}',
            'frame_stride': 1,
        },
    )


def test_object_association_prefers_shared_tracks() -> None:
    artifact = ObjectArtifact(run_id='dev', stage_name='objects', save_dir=Path('.'))
    artifact.objects = [
        _make_obj('det:0', 'det:0', 0, [1, 1, 8, 8], embedding=[1, 0, 0], track_ids=[1, 2, 3]),
        _make_obj('det:1', 'det:1', 1, [2, 1, 9, 8], embedding=[1, 0, 0], track_ids=[1, 2, 3]),
        _make_obj('det:2', 'det:2', 1, [20, 1, 28, 8], embedding=[0, 1, 0], track_ids=[9]),
    ]
    out = ObjectAssociator({'min_match_score': 0.25}).associate(artifact)
    ids_by_det = {obj.attributes['detection_id']: obj.object_id for obj in out.objects}
    assert ids_by_det['det:0'] == ids_by_det['det:1']
    assert ids_by_det['det:2'] != ids_by_det['det:0']


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
    # create two RGB frames for the viewer
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
        _make_obj('det:1', 'obj_0000', 1, [2, 1, 9, 8], track_ids=[1, 2]),
    ]
    artifact.mask_arrays = {
        'det:0': np.pad(np.ones((8, 8), dtype=bool), ((1, 15), (1, 23))),
        'det:1': np.pad(np.ones((8, 8), dtype=bool), ((1, 15), (2, 22))),
    }
    artifact.embedding_vectors = {
        'det:0': np.asarray([1, 0, 0], dtype=np.float32),
        'det:1': np.asarray([1, 0, 0], dtype=np.float32),
    }
    artifact.frame_index = artifact._build_frame_index(artifact.objects)
    artifact.save()
    assert (artifact.save_dir / 'object_debug_viewer.html').exists()
    loaded = ObjectArtifact.load(artifact.save_dir)
    assert len(loaded.objects) == 2
    assert 'det:0' in loaded.mask_arrays
