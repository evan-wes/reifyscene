"""Basic tests for shared dataclasses."""

from pathlib import Path

from room4d.common.types import BaseArtifact


def test_base_artifact_summary() -> None:
    artifact = BaseArtifact(run_id="dev", stage_name="test", save_dir=Path("."))
    assert artifact.summary()["run_id"] == "dev"
