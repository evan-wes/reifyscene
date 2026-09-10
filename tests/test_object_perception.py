"""Tests for object perception (consume-from-instance) and debug artifact export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reifyscene.api.pipeline import ReifyScenePipeline
from reifyscene.common.types import FrameRef, ObjectState, VideoStreamRef
from reifyscene.perception.objects.object_artifact import ObjectArtifact
from reifyscene.perception.instance.instance_artifact import InstanceArtifact, InstanceObservation
from reifyscene.perception.reid.reid_artifact import ReidArtifact
from reifyscene.ingest.ingest_artifact import IngestArtifact
from reifyscene.tracking.track_artifact import TrackArtifact


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


def _fr(idx: int) -> FrameRef:
    return FrameRef(stream_id='cam_00', frame_idx=idx, timestamp_sec=float(idx))


def _ingest(save_dir, video) -> IngestArtifact:
    video.parent.mkdir(parents=True, exist_ok=True)
    video.touch()
    return IngestArtifact(
        run_id='dev', stage_name='ingest', save_dir=save_dir,
        streams=[VideoStreamRef(stream_id='cam_00', video_path=video, fps=10.0, num_frames=10, width=32, height=24, has_audio=False)],
        keyframes=[_fr(0), _fr(1)],
    )


def test_perceive_objects_consumes_instance_and_reid(tmp_path: Path) -> None:
    """objects = non-human tracklets, ids canonicalized via the reid merge_map."""
    pipeline = ReifyScenePipeline({'run_dir': str(tmp_path)})
    inst = InstanceArtifact(
        run_id='dev', stage_name='segment', save_dir=tmp_path / 'segment',
        observations=[
            InstanceObservation('i0', 'track_0000', _fr(0), concept='car', kind='object', bbox_xyxy=np.array([0, 0, 8, 8], np.float32), confidence=0.9, attributes={'label_id': 3}),
            InstanceObservation('i1', 'track_0002', _fr(1), concept='car', kind='object', bbox_xyxy=np.array([1, 0, 9, 8], np.float32), confidence=0.8, attributes={'label_id': 3}),
            InstanceObservation('i2', 'track_0001', _fr(0), concept='person', kind='human', bbox_xyxy=np.array([20, 0, 28, 20], np.float32), confidence=0.95),
        ],
        mask_arrays={'i0': np.ones((8, 8), bool), 'i1': np.ones((8, 8), bool), 'i2': np.ones((8, 8), bool)},
        metadata={'detector_backend': 'maskrcnn'},
    )
    inst.rebuild_indices()
    reid = ReidArtifact(
        run_id='dev', stage_name='reid', save_dir=tmp_path / 'reid',
        embeddings={'track_0000': np.array([1, 0, 0], np.float32)},
        merge_map={'track_0002': 'track_0000'},  # i1 re-acquired as track_0000 across the gap
    )
    out = pipeline.perceive_objects(_ingest(tmp_path / 'ingest', tmp_path / 'clip.mp4'), instance_artifact=inst, reid_artifact=reid)

    # the human instance is excluded; the two car instances both fold onto track_0000
    assert {o.object_id for o in out.objects} == {'track_0000'}
    assert len(out.objects) == 2
    assert all(o.category == 'car' for o in out.objects)
    assert all(o.embedding is not None for o in out.objects)  # carried from reid
    assert out.metadata['num_object_tracks'] == 1


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
    ]
    artifact.mask_arrays = {
        'det:0': np.pad(np.ones((8, 8), dtype=bool), ((1, 15), (1, 23))),
        'det:1': np.pad(np.ones((8, 8), dtype=bool), ((1, 15), (2, 22))),
    }
    artifact.embedding_vectors = {
        'det:0': np.asarray([1, 0, 0], dtype=np.float32),
        'det:1': np.asarray([1, 0, 0], dtype=np.float32),
    }
    artifact.linked_ids = {
        'track_summaries': {
            'obj_0000': {
                'global_id': 'obj_0000',
                'consensus_category': 'dog',
                'support_track_count': 2,
                'lifespan_start': 0,
                'lifespan_end': 1,
                'category_entropy': 0.2,
                'quality_score': 0.8,
            },
        },
    }
    artifact.frame_index = artifact._build_frame_index(artifact.objects)
    artifact.save()
    html = (artifact.save_dir / 'object_debug_viewer.html').read_text(encoding='utf-8')
    assert 'obj_0000' in html
    loaded = ObjectArtifact.load(artifact.save_dir)
    assert len(loaded.objects) == 2
    assert 'det:0' in loaded.mask_arrays


def test_pipeline_select_frames_falls_back_to_keyframes_without_tracks() -> None:
    """`_select_perception_frames` (used by `segment`) strides keyframes when no tracks exist."""
    pipeline = ReifyScenePipeline({'objects': {'frame_stride': 1}})
    ingest = IngestArtifact(
        run_id='dev',
        stage_name='ingest',
        save_dir=Path('.'),
        streams=[VideoStreamRef(stream_id='cam_00', video_path=Path('video.mp4'), fps=10.0, num_frames=10, width=32, height=24, has_audio=False)],
        keyframes=[_fr(0), _fr(5)],
    )
    frames = pipeline._select_perception_frames(ingest, None, pipeline.cfg['objects'], default_source='keyframes')
    assert [f.frame_idx for f in frames] == [0, 5]
