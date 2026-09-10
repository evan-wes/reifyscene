from __future__ import annotations

from reifyscene.common.system_report import collect_system_report, recommend_tracking_profile, recommend_vggt_long_profile


def test_system_report_and_tracking_recommendation() -> None:
    report = collect_system_report()
    assert 'platform' in report
    assert 'cpu' in report
    assert 'memory' in report

    rec = recommend_tracking_profile(report, device='cpu')
    assert rec['profile'] in {'lowmem', 'balanced', 'full'}
    assert rec['resize_max_dim'] > 0
    assert rec['window_size'] > 0


def test_vggt_long_recommendation_prefers_conservative_defaults_on_small_gpu() -> None:
    report = {
        'memory': {'available_gib': 12.0},
        'gpus': [{'memory_free_mb': 4096}],
    }
    rec = recommend_vggt_long_profile(report, device='cuda', num_frames=120, source_max_dim=1920)
    assert rec['profile'] == 'lowmem'
    assert rec['initial_chunk_size'] <= 8
    assert rec['initial_resize_max_dim'] <= 384
    assert rec['min_resize_max_dim'] <= rec['initial_resize_max_dim']
    assert rec['resize_candidates'][0] == rec['initial_resize_max_dim']
    assert rec['resize_candidates'][-1] == rec['min_resize_max_dim']


def test_vggt_long_recommendation_exposes_dense_resize_ladder_for_8gb_gpu() -> None:
    report = {
        'memory': {'available_gib': 32.0},
        'gpus': [{'memory_free_mb': 8192}],
    }
    rec = recommend_vggt_long_profile(report, device='cuda', num_frames=180, source_max_dim=1920)
    assert rec['profile'] == 'lowmem'
    assert rec['initial_chunk_size'] <= 12
    assert rec['initial_resize_max_dim'] == 640
    assert rec['min_resize_max_dim'] == 224
    assert rec['resize_candidates'] == [640, 576, 512, 448, 384, 320, 288, 256, 224]
