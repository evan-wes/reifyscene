"""Save/load round-trips for the new perception artifacts (design §4A, FP0)."""

from __future__ import annotations

import numpy as np

from room4d.common.backend_ladder import CapabilityManifest, RungStatus
from room4d.common.types import FrameRef
from room4d.perception.instance.instance_artifact import InstanceArtifact, InstanceObservation
from room4d.perception.reid.reid_artifact import ReidArtifact


def _frame(idx: int) -> FrameRef:
    return FrameRef(stream_id='cam0', frame_idx=idx, timestamp_sec=idx / 30.0)


def _instance_artifact(save_dir) -> InstanceArtifact:
    obs = [
        InstanceObservation(
            instance_id='inst:cam0:0:0', track_id='trk_0000', frame_ref=_frame(0),
            concept='person', kind='human', bbox_xyxy=np.array([1, 2, 10, 20], np.float32), confidence=0.9,
            attributes={'mask_area': 42},
        ),
        InstanceObservation(
            instance_id='inst:cam0:1:0', track_id='trk_0000', frame_ref=_frame(1),
            concept='person', kind='human', bbox_xyxy=np.array([2, 3, 11, 21], np.float32), confidence=0.8,
        ),
        InstanceObservation(
            instance_id='inst:cam0:0:1', track_id='trk_0001', frame_ref=_frame(0),
            concept='car', kind='object', bbox_xyxy=np.array([30, 30, 50, 60], np.float32), confidence=0.7,
        ),
    ]
    art = InstanceArtifact(run_id='r', stage_name='segment', save_dir=save_dir, observations=obs)
    art.mask_arrays = {
        'inst:cam0:0:0': np.array([[True, False], [False, True]]),
        'inst:cam0:1:0': np.array([[True, True], [False, False]]),
        'inst:cam0:0:1': np.zeros((2, 2), dtype=bool),
    }
    art.rebuild_indices()
    art.attach_manifest(CapabilityManifest(
        stage='segment', backend_used='maskrcnn_iou', fidelity='per_frame_assoc',
        rungs=[RungStatus('sam3', 'open_vocab', 'skipped', 'no-access'), RungStatus('maskrcnn_iou', 'per_frame_assoc', 'ran')],
    ))
    return art


def test_instance_artifact_roundtrip(tmp_path):
    art = _instance_artifact(tmp_path / 'segment')
    art.save()
    loaded = InstanceArtifact.load(tmp_path / 'segment')

    assert len(loaded.observations) == 3
    by_id = {o.instance_id: o for o in loaded.observations}
    o = by_id['inst:cam0:0:0']
    assert o.track_id == 'trk_0000' and o.kind == 'human' and o.concept == 'person'
    assert o.frame_ref.frame_idx == 0 and abs(o.frame_ref.timestamp_sec) < 1e-9
    np.testing.assert_allclose(o.bbox_xyxy, [1, 2, 10, 20])
    assert o.attributes['mask_area'] == 42

    # masks survive as bool arrays
    np.testing.assert_array_equal(loaded.mask_arrays['inst:cam0:0:0'], [[True, False], [False, True]])
    assert loaded.mask_arrays['inst:cam0:0:0'].dtype == bool

    # indices reconstructed
    assert sorted(loaded.frame_index['cam0:0']) == ['inst:cam0:0:0', 'inst:cam0:0:1']
    assert loaded.track_index['trk_0000'] == ['inst:cam0:0:0', 'inst:cam0:1:0']

    # capability manifest survives and is readable
    cap = loaded.capability
    assert cap is not None and cap.backend_used == 'maskrcnn_iou' and cap.fidelity == 'per_frame_assoc'
    assert cap.rungs[0].reason == 'no-access'


def test_instance_summary_counts(tmp_path):
    art = _instance_artifact(tmp_path / 'segment')
    s = art.summary()
    assert s['num_observations'] == 3 and s['num_tracks'] == 2 and s['num_frames'] == 2
    assert s['backend_used'] == 'maskrcnn_iou'


def test_instance_bbox_none_roundtrips(tmp_path):
    art = InstanceArtifact(
        run_id='r', stage_name='segment', save_dir=tmp_path / 'seg',
        observations=[InstanceObservation(instance_id='i0', track_id='t0', frame_ref=_frame(0))],
    )
    art.rebuild_indices()
    art.save()
    loaded = InstanceArtifact.load(tmp_path / 'seg')
    assert loaded.observations[0].bbox_xyxy is None
    assert loaded.observations[0].kind == 'unknown'


def test_reid_artifact_roundtrip(tmp_path):
    art = ReidArtifact(
        run_id='r', stage_name='reid', save_dir=tmp_path / 'reid',
        embeddings={'trk_0000': np.array([0.1, 0.2, 0.3], np.float32), 'trk_0001': np.array([0.4, 0.5, 0.6], np.float32)},
        merge_map={'trk_0002': 'trk_0000'},  # trk_0002 re-acquired as trk_0000 across a gap
    )
    art.attach_manifest(CapabilityManifest(stage='reid', backend_used='dinov2', fidelity='apache_features'))
    art.save()
    loaded = ReidArtifact.load(tmp_path / 'reid')

    np.testing.assert_allclose(loaded.embeddings['trk_0000'], [0.1, 0.2, 0.3], rtol=1e-6)
    assert loaded.merge_map == {'trk_0002': 'trk_0000'}
    assert loaded.capability.backend_used == 'dinov2'


def test_reid_canonical_follows_chain_and_is_cycle_safe():
    art = ReidArtifact(run_id='r', stage_name='reid', save_dir='/tmp/x', merge_map={'c': 'b', 'b': 'a'})
    assert art.canonical('c') == 'a'
    assert art.canonical('a') == 'a'  # not in map -> itself
    art.merge_map = {'x': 'y', 'y': 'x'}  # malformed cycle must not hang
    assert art.canonical('x') in {'x', 'y'}


def test_reid_merged_groups(tmp_path):
    art = ReidArtifact(
        run_id='r', stage_name='reid', save_dir=tmp_path / 'reid',
        embeddings={'a': np.zeros(2, np.float32), 'b': np.zeros(2, np.float32), 'c': np.zeros(2, np.float32)},
        merge_map={'b': 'a'},  # b folds onto a; c stands alone
    )
    groups = art.merged_groups()
    assert groups['a'] == ['a', 'b']
    assert groups['c'] == ['c']
