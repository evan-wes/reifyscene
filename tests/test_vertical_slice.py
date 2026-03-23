"""Smoke test for the first real vertical slice."""

from pathlib import Path

from room4d.api.pipeline import Room4DPipeline
from room4d.geometry.geometry_artifact import GeometryArtifact
from tests.conftest import write_test_video


def test_vertical_slice_smoke(tmp_path: Path) -> None:
    cam_a = tmp_path / "cam_a.mp4"
    cam_b = tmp_path / "cam_b.mp4"
    write_test_video(cam_a, color_offset=0)
    write_test_video(cam_b, color_offset=25)

    pipeline = Room4DPipeline({'run_dir': tmp_path / 'run', 'geometry': {'backend_mode': 'heuristic'}})
    outputs = pipeline.run([cam_a, cam_b])

    assert outputs['ingest'].num_streams() == 2
    assert len(outputs['sync'].offsets_by_stream) == 2
    assert outputs['geometry'].backend_name == 'vggt'
    assert outputs['tracks'].total_observations() > 0
    assert len(outputs['graph'].nodes) >= 2
    assert len(outputs['optimization'].stage_diagnostics) == 3
    assert (tmp_path / 'run' / 'geometry' / 'artifact.json').exists()
    loaded_geometry = GeometryArtifact.load(tmp_path / 'run' / 'geometry')
    assert loaded_geometry.backend_name == 'vggt'
