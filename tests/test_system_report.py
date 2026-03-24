from __future__ import annotations

from room4d.common.system_report import collect_system_report, recommend_tracking_profile


def test_system_report_and_tracking_recommendation() -> None:
    report = collect_system_report()
    assert 'platform' in report
    assert 'cpu' in report
    assert 'memory' in report

    rec = recommend_tracking_profile(report, device='cpu')
    assert rec['profile'] in {'lowmem', 'balanced', 'full'}
    assert rec['resize_max_dim'] > 0
    assert rec['window_size'] > 0
