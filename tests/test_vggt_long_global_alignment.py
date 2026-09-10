"""Regression test for the VGGT-Long global pose-alignment fix.

VGGT-Long stores chunk-local extrinsics in its per-chunk ``.npy`` files and only
publishes the globally-aligned trajectory in ``camera_poses.txt`` (per-chunk
Sim(3) accumulation + loop closure). The wrapper used to merge the chunk-local
extrinsics, producing independently-anchored mini-trajectories. These tests pin
that the wrapper consumes the global ``camera_poses.txt`` and reconstructs the
correct global ``T_world_cam``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from reifyscene.common.types import FrameRef
from reifyscene.geometry.vggt_long_wrapper import VGGTLongInitializer


def _c2w(R: np.ndarray, center: np.ndarray) -> np.ndarray:
    M = np.eye(4, dtype=float)
    M[:3, :3] = R
    M[:3, 3] = center
    return M


def _rot_y(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _init() -> VGGTLongInitializer:
    return VGGTLongInitializer({"backend_mode": "heuristic", "device": "cpu"})


def _write_camera_poses(path: Path, poses: np.ndarray) -> None:
    path.write_text("\n".join(" ".join(str(x) for x in p.flatten()) for p in poses) + "\n")


def test_load_global_camera_to_world_parses_and_validates(tmp_path: Path):
    init = _init()
    poses = np.stack([_c2w(np.eye(3), np.zeros(3)), _c2w(_rot_y(15.0), np.array([1.0, 2.0, 3.0]))])
    _write_camera_poses(tmp_path / "camera_poses.txt", poses)
    runner = SimpleNamespace(output_dir=str(tmp_path), result_aligned_dir=str(tmp_path / "_tmp_results_aligned"))

    loaded = init._load_global_camera_to_world(runner, num_frames=2)
    assert loaded is not None and loaded.shape == (2, 4, 4)
    assert np.allclose(loaded, poses)

    # Frame-count mismatch -> None (graceful fallback to chunk-local extrinsics).
    assert init._load_global_camera_to_world(runner, num_frames=3) is None
    # Missing file -> None.
    empty = SimpleNamespace(output_dir=str(tmp_path / "nope"), result_aligned_dir=None)
    assert init._load_global_camera_to_world(empty, num_frames=2) is None


def test_combined_predictions_use_global_poses():
    """With global poses supplied, the reconstructed T_world_cam must equal the
    global camera-to-world poses (NOT the chunk-local ones)."""
    init = _init()
    frames = [FrameRef("cam_00", 0, 0.0), FrameRef("cam_00", 1, 0.1)]
    # Chunk-local extrinsics: each chunk anchored at identity (the buggy source).
    chunk = {
        "_chunk_range": (0, 2),
        "extrinsic": np.stack([np.eye(4), np.eye(4)]).astype(np.float32),
        "intrinsic": np.stack([np.eye(3, dtype=np.float32)] * 2),
        "depth": np.ones((2, 4, 4, 1), dtype=np.float32),
        "world_points": np.zeros((2, 4, 4, 3), dtype=np.float32),
    }
    global_c2w = np.stack([_c2w(np.eye(3), np.zeros(3)), _c2w(_rot_y(20.0), np.array([2.0, -1.0, 4.0]))])

    preds = init._combined_predictions_from_chunks(frames, [chunk], global_c2w=global_c2w)
    # Stored extrinsics are camera-from-world (inverse of global c2w).
    for i in range(2):
        assert np.allclose(preds["extrinsic"][i], np.linalg.inv(global_c2w[i]), atol=1e-6)
        # The shared path inverts back -> recovers the global camera-to-world pose.
        assert np.allclose(init._world_from_cam(preds["extrinsic"][i]), global_c2w[i], atol=1e-6)
    # Depth / world_points are untouched by the pose globalization.
    assert preds["depth"].shape == (2, 4, 4, 1)
    assert preds["point_map"].shape == (2, 4, 4, 3)


def test_combined_predictions_fallback_without_global():
    """Without global poses, the chunk-local extrinsics are used (legacy path)."""
    init = _init()
    frames = [FrameRef("cam_00", 0, 0.0), FrameRef("cam_00", 1, 0.1)]
    local = np.stack([_c2w(np.eye(3), np.zeros(3)), _c2w(_rot_y(5.0), np.array([0.1, 0.0, 0.2]))]).astype(np.float32)
    chunk = {"_chunk_range": (0, 2), "extrinsic": local, "intrinsic": np.stack([np.eye(3, dtype=np.float32)] * 2)}
    preds = init._combined_predictions_from_chunks(frames, [chunk], global_c2w=None)
    assert np.allclose(preds["extrinsic"], local, atol=1e-6)
