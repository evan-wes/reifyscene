"""Basic tests for sync scaffolding."""

from reifyscene.sync.audio_sync import AudioSyncEstimator


def test_audio_sync_placeholder() -> None:
    estimator = AudioSyncEstimator()
    offset, confidence = estimator.estimate_offset(None, None)  # type: ignore[arg-type]
    assert offset == 0.0
    assert confidence == 0.0
