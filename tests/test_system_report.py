from __future__ import annotations

from room4d.common.system_report import collect_system_report, recommend_tracking_profile, recommend_vggt_long_profile


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
    assert rec['initial_chunk_size'] <= 16
    assert rec['initial_resize_max_dim'] <= 1024
    assert rec['min_resize_max_dim'] <= rec['initial_resize_max_dim']
