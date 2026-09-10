from __future__ import annotations

import numpy as np

from reifyscene.render.camera_paths import orbit_cameras
from reifyscene.render.export import write_video


def test_orbit_cameras_geometry() -> None:
    center = np.array([1.0, 2.0, 3.0])
    cams = orbit_cameras(center, radius=2.0, num_frames=8, up=(0.0, 0.0, 1.0), arc_degrees=120.0, height_frac=0.2)
    assert len(cams) == 8
    expected_dist = 2.0 * np.sqrt(1.0 + 0.2**2)  # radius in-plane + height_frac*radius along up
    for cam in cams:
        assert np.allclose(cam["target"], center, atol=1e-5)
        assert np.allclose(cam["up"], [0.0, 0.0, 1.0], atol=1e-5)
        assert abs(np.linalg.norm(np.asarray(cam["eye"]) - center) - expected_dist) < 1e-4


def test_orbit_single_frame() -> None:
    cams = orbit_cameras([0.0, 0.0, 0.0], radius=1.0, num_frames=1)
    assert len(cams) == 1


def test_write_video_creates_mp4(tmp_path) -> None:
    frames = [np.full((16, 24, 3), i * 40, dtype=np.uint8) for i in range(5)]
    out = write_video(frames, tmp_path / "clip.mp4", fps=10)
    assert out.exists() and out.stat().st_size > 0
