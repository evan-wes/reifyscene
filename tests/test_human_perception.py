"""Tests for refined human perception and debug export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from room4d.common.types import FrameRef, HumanState
from room4d.perception.humans.human_artifact import HumanArtifact
from room4d.perception.humans.human_assoc import HumanAssociator


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
        attributes={
            'detection_id': det_id,
            'track_ids': list(track_ids or []),
            'support_track_count': len(track_ids or []),
            'bbox_support_count': len(track_ids or []),
            'mask_support_count': len(track_ids or []),
            'visible_track_count': max(4, len(track_ids or [])),
            'support_points_xy': [[bbox[0] + 1, bbox[1] + 1], [bbox[2] - 1, bbox[3] - 1]],
        },
    )


def test_human_association_suppresses_duplicates_and_smooths() -> None:
    artifact = HumanArtifact(run_id='dev', stage_name='humans', save_dir=Path('.'))
    h0 = _make_human('hum:0', 0, [5, 5, 15, 25], track_ids=[1, 2, 3])
    h0_dup = _make_human('hum:0dup', 0, [5.5, 5.2, 15.2, 25.1], track_ids=[1, 2, 3], confidence=0.4)
    h1 = _make_human('hum:1', 1, [7, 5, 17, 25], track_ids=[1, 2, 3])
    artifact.humans = [h0, h0_dup, h1]
    artifact.mask_arrays = {
        'hum:0': np.pad(np.ones((20, 10), dtype=bool), ((5, 15), (5, 33))),
        'hum:0dup': np.pad(np.ones((20, 10), dtype=bool), ((5, 15), (5, 33))),
        'hum:1': np.pad(np.ones((20, 10), dtype=bool), ((5, 15), (7, 31))),
    }
    artifact.densepose_payloads = {'hum:0': h0.densepose, 'hum:0dup': h0_dup.densepose, 'hum:1': h1.densepose}
    artifact.smpl_payloads = {'hum:0': h0.smpl_params, 'hum:0dup': h0_dup.smpl_params, 'hum:1': h1.smpl_params}

    out = HumanAssociator({'min_match_score': 0.2}).associate_across_views(artifact)
    assert len(out.humans) == 2
    assert len({h.human_id for h in out.humans}) == 1
    suppressed = [h for h in out.raw_humans if h.attributes.get('suppressed')]
    assert suppressed and suppressed[0].attributes.get('suppression_kind') == 'frame_duplicate'
    smoothed = sorted(out.humans, key=lambda h: h.frame_ref.frame_idx)
    assert smoothed[0].bbox_xyxy[0] > 5.0
    assert smoothed[1].smpl_params is not None and smoothed[1].smpl_params.get('smoothed') is True
    assert smoothed[0].densepose is not None and smoothed[0].densepose.get('smoothed') is True
    summary = out.linked_ids['track_summaries'][smoothed[0].human_id]
    assert summary['support_track_count'] == 3
    assert summary['quality_score'] > 0.0


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
    raw_supp = _make_human('hum:supp', 1, [6.2, 5.1, 16.2, 25.0], track_ids=[1, 2, 3], confidence=0.3, human_id='hum:supp')
    raw_supp.attributes.update({'suppressed': True, 'suppression_kind': 'frame_duplicate', 'suppressed_by_detection_id': 'hum:1'})
    clean0 = _make_human('hum:0', 0, [5.5, 5, 15.5, 25], track_ids=[1, 2, 3], human_id='human_000')
    clean1 = _make_human('hum:1', 1, [5.8, 5, 15.8, 25], track_ids=[1, 2, 3], human_id='human_000')
    for h in [clean0, clean1]:
        h.attributes.update({'quality_score': 0.8, 'support_ratio': 0.75})
    artifact.humans = [clean0, clean1]
    artifact.raw_humans = [raw0, raw1, raw_supp]
    artifact.mask_arrays = {'hum:0': np.pad(np.ones((20, 10), dtype=bool), ((5, 15), (5, 33))), 'hum:1': np.pad(np.ones((20, 10), dtype=bool), ((5, 15), (6, 32))), 'hum:supp': np.pad(np.ones((20, 10), dtype=bool), ((5, 15), (6, 32)))}
    artifact.densepose_payloads = {'hum:0': clean0.densepose, 'hum:1': clean1.densepose, 'hum:supp': raw_supp.densepose}
    artifact.smpl_payloads = {'hum:0': clean0.smpl_params, 'hum:1': clean1.smpl_params, 'hum:supp': raw_supp.smpl_params}
    artifact.linked_ids = {
        'track_summaries': {
            'human_000': {
                'lifespan_start': 0,
                'lifespan_end': 1,
                'support_track_count': 3,
                'mean_support_density': 0.75,
                'quality_score': 0.8,
            }
        },
        'raw_detection_count': 3,
        'kept_detection_count': 2,
        'final_detection_count': 2,
        'suppressed_detection_count': 1,
    }
    artifact.frame_index = artifact._build_frame_index(artifact.humans)
    artifact.save()
    html = (save_dir / 'human_debug_viewer.html').read_text(encoding='utf-8')
    assert 'Cleaned' in html and 'Raw' in html and 'Both' in html
    assert 'support' in html.lower()
    loaded = HumanArtifact.load(save_dir)
    assert len(loaded.humans) == 2
    assert len(loaded.raw_humans) == 3
