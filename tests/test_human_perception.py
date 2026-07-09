"""Tests for human perception (consume-from-instance) and debug export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from room4d.api.pipeline import Room4DPipeline
from room4d.common.types import FrameRef, HumanState, VideoStreamRef
from room4d.ingest.ingest_artifact import IngestArtifact
from room4d.perception.humans.human_artifact import HumanArtifact
from room4d.perception.instance.instance_artifact import InstanceArtifact, InstanceObservation
from room4d.perception.reid.reid_artifact import ReidArtifact


def _make_kpts(cx: float, cy: float) -> np.ndarray:
    kpts = np.zeros((17, 3), dtype=np.float32)
    for idx in range(17):
        kpts[idx, 0] = cx + 0.2 * idx
        kpts[idx, 1] = cy + 0.1 * idx
        kpts[idx, 2] = 0.9
    return kpts


def _make_dp(box):
    x1, y1, x2, y2 = map(float, box)
    uv = np.asarray([[0.2, 0.2], [0.4, 0.5], [0.6, 0.8]], dtype=np.float32)
    samples = np.stack([x1 + uv[:, 0] * (x2 - x1), y1 + uv[:, 1] * (y2 - y1)], axis=1).astype(np.float32)
    return {'samples_xy': samples, 'uv': uv, 'part_ids': np.asarray([1, 2, 5], dtype=np.int32)}


def _make_smpl(offset: float):
    return {
        'backend': 'heuristic',
        'betas': np.zeros((10,), dtype=np.float32),
        'body_pose': np.zeros((23, 3), dtype=np.float32),
        'global_orient': np.asarray([0.0, 0.0, offset], dtype=np.float32),
        'transl': np.asarray([offset, offset, 10.0], dtype=np.float32),
        'scale': 10.0 + offset,
        'valid': True,
    }


def _make_human(det_id: str, frame_idx: int, bbox, *, track_ids=None, confidence=0.9, human_id=None):
    return HumanState(
        human_id=human_id or det_id,
        frame_ref=FrameRef(stream_id='cam_00', frame_idx=frame_idx, timestamp_sec=float(frame_idx)),
        bbox_xyxy=np.asarray(bbox, dtype=np.float32),
        mask_rle={'shape': [40, 48], 'sum': 100},
        keypoints_2d=_make_kpts((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5),
        densepose=_make_dp(bbox),
        smpl_params=_make_smpl(float(frame_idx)),
        confidence=confidence,
        attributes={'detection_id': det_id, 'track_ids': list(track_ids or [])},
    )


def _fr(idx: int) -> FrameRef:
    return FrameRef(stream_id='cam_00', frame_idx=idx, timestamp_sec=float(idx))


def test_perceive_humans_consumes_instance_and_reid(tmp_path: Path, monkeypatch) -> None:
    """humans = human-kind tracklets, ids canonicalized; pose/SMPL run over segment masks."""
    # use_sapiens=False -> the dense ladder uses the Keypoint-RCNN floor (patched below)
    pipeline = Room4DPipeline({'run_dir': str(tmp_path), 'humans': {'use_sapiens': False}})
    monkeypatch.setattr(pipeline.decoder, 'get_frame', lambda stream, idx: np.zeros((40, 48, 3), dtype=np.uint8))
    monkeypatch.setattr(pipeline.human_pose, 'predict', lambda frame, boxes: [_make_kpts(5, 5) for _ in boxes])

    inst = InstanceArtifact(
        run_id='dev', stage_name='segment', save_dir=tmp_path / 'segment',
        observations=[
            InstanceObservation('h0', 'track_0001', _fr(0), concept='person', kind='human', bbox_xyxy=np.array([5, 5, 15, 35], np.float32), confidence=0.9),
            InstanceObservation('h1', 'track_0003', _fr(1), concept='person', kind='human', bbox_xyxy=np.array([6, 5, 16, 35], np.float32), confidence=0.9),
            InstanceObservation('o0', 'track_0000', _fr(0), concept='car', kind='object', bbox_xyxy=np.array([30, 0, 40, 10], np.float32), confidence=0.8),
        ],
        mask_arrays={'h0': np.ones((40, 48), bool), 'h1': np.ones((40, 48), bool), 'o0': np.ones((40, 48), bool)},
    )
    inst.rebuild_indices()
    reid = ReidArtifact(run_id='dev', stage_name='reid', save_dir=tmp_path / 'reid', merge_map={'track_0003': 'track_0001'})

    video = tmp_path / 'clip.mp4'
    video.touch()
    out = pipeline.compute_humans(IngestArtifact(
        run_id='dev', stage_name='ingest', save_dir=tmp_path / 'ingest',
        streams=[VideoStreamRef(stream_id='cam_00', video_path=video, fps=10.0, num_frames=3, width=48, height=40, has_audio=False)],
    ), instance_artifact=inst, reid_artifact=reid)

    # the object instance is excluded; both human instances fold onto track_0001
    assert {h.human_id for h in out.humans} == {'track_0001'}
    assert len(out.humans) == 2
    assert out.metadata['num_human_tracks'] == 1
    assert all(h.keypoints_2d is not None for h in out.humans)


def test_human_artifact_debug_viewer_roundtrip(tmp_path: Path) -> None:
    save_dir = tmp_path / 'humans'
    artifact = HumanArtifact(
        run_id='demo',
        stage_name='humans',
        save_dir=save_dir,
        metadata={'save_debug_assets': True, 'save_debug_html': True, 'debug_image_paths': {}},
    )
    image_dir = save_dir / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    for idx in [0, 1]:
        rgb = np.zeros((40, 48, 3), dtype=np.uint8)
        rgb[..., 0] = 50 + idx * 20
        img_path = image_dir / f'cam_00__{idx}.png'
        Image.fromarray(rgb).save(img_path)
        artifact.metadata['debug_image_paths'][f'cam_00:{idx}'] = str(img_path)
    raw0 = _make_human('hum:0', 0, [5, 5, 15, 25], track_ids=[1, 2, 3], human_id='human_000')
    raw1 = _make_human('hum:1', 1, [6, 5, 16, 25], track_ids=[1, 2, 3], human_id='human_000')
    clean0 = _make_human('hum:0', 0, [5.5, 5, 15.5, 25], track_ids=[1, 2, 3], human_id='human_000')
    clean1 = _make_human('hum:1', 1, [5.8, 5, 15.8, 25], track_ids=[1, 2, 3], human_id='human_000')
    for h in [clean0, clean1]:
        h.attributes.update({'quality_score': 0.8, 'support_ratio': 0.75})
    artifact.humans = [clean0, clean1]
    artifact.raw_humans = [raw0, raw1]
    artifact.mask_arrays = {'hum:0': np.pad(np.ones((20, 10), dtype=bool), ((5, 15), (5, 33))), 'hum:1': np.pad(np.ones((20, 10), dtype=bool), ((5, 15), (6, 32)))}
    artifact.densepose_payloads = {'hum:0': clean0.densepose, 'hum:1': clean1.densepose}
    artifact.smpl_payloads = {'hum:0': clean0.smpl_params, 'hum:1': clean1.smpl_params}
    artifact.frame_index = artifact._build_frame_index(artifact.humans)
    artifact.save()
    html = (save_dir / 'human_debug_viewer.html').read_text(encoding='utf-8')
    assert 'Cleaned' in html and 'Raw' in html
    loaded = HumanArtifact.load(save_dir)
    assert len(loaded.humans) == 2
    assert len(loaded.raw_humans) == 2
