"""Regression test for the VGGT-Long global pose-alignment fix.

VGGT-Long stores chunk-local extrinsics in its per-chunk ``.npy`` files and only
publishes the globally-aligned trajectory in ``camera_poses.txt`` (per-chunk
Sim(3) accumulation + loop closure). The wrapper used to merge the chunk-local
extrinsics, producing independently-anchored mini-trajectories. These tests pin
that the wrapper consumes the global ``camera_poses.txt`` and reconstructs the
correct global ``T_world_cam``, and that each chunk's depth is put on the global
scale with it (the Sim(3) has a scale; depth used to keep its chunk's own).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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

    # Frame-count mismatch or a missing file raises: chunk-local poses are not one scene.
    with pytest.raises(RuntimeError, match="2 pose"):
        init._load_global_camera_to_world(runner, num_frames=3)
    empty = SimpleNamespace(output_dir=str(tmp_path / "nope"), result_aligned_dir=None)
    with pytest.raises(RuntimeError, match="no camera_poses.txt"):
        init._load_global_camera_to_world(empty, num_frames=2)


def test_combined_predictions_use_global_poses():
    """With global poses supplied, the reconstructed T_world_cam must equal the
    global camera-to-world poses (NOT the chunk-local ones)."""
    init = _init()
    frames = [FrameRef("cam_00", 0, 0.0), FrameRef("cam_00", 1, 0.1)]
    # Chunk-local extrinsics: each chunk anchored at identity (the buggy source).
    chunk = {
        "_chunk_range": (0, 2),
        "_chunk_index": 0,
        "extrinsic": np.stack([np.eye(4), np.eye(4)]).astype(np.float32),
        "intrinsic": np.stack([np.eye(3, dtype=np.float32)] * 2),
        "depth": np.ones((2, 4, 4, 1), dtype=np.float32),
        "world_points": np.zeros((2, 4, 4, 3), dtype=np.float32),
    }
    global_c2w = np.stack([_c2w(np.eye(3), np.zeros(3)), _c2w(_rot_y(20.0), np.array([2.0, -1.0, 4.0]))])

    preds = init._combined_predictions_from_chunks(
        frames, [chunk], global_c2w=global_c2w, chunk_scales=[1.0]
    )
    # Stored extrinsics are camera-from-world (inverse of global c2w).
    for i in range(2):
        assert np.allclose(preds["extrinsic"][i], np.linalg.inv(global_c2w[i]), atol=1e-6)
        # The shared path inverts back -> recovers the global camera-to-world pose.
        assert np.allclose(init._world_from_cam(preds["extrinsic"][i]), global_c2w[i], atol=1e-6)
    # Depth / world_points are untouched by the pose globalization.
    assert preds["depth"].shape == (2, 4, 4, 1)
    assert preds["point_map"].shape == (2, 4, 4, 3)


def test_chunk_scales_come_from_the_accumulated_sim3():
    rot = np.eye(3)
    sim3 = [(np.array([2.0]), rot, np.zeros(3)), (0.5, rot, np.ones(3))]
    runner = SimpleNamespace(sim3_list=sim3)
    assert VGGTLongInitializer._chunk_scales(runner, 3) == [1.0, 2.0, 0.5]
    assert VGGTLongInitializer._chunk_scales(SimpleNamespace(sim3_list=[]), 1) == [1.0]
    with pytest.raises(RuntimeError, match="1 chunk alignment"):
        VGGTLongInitializer._chunk_scales(SimpleNamespace(sim3_list=[(1.0, rot, np.zeros(3))]), 3)


def _lift(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    rays = np.linalg.solve(K, np.stack([u.ravel() + 0.0, v.ravel() + 0.0, np.ones(u.size)]))
    return (c2w[:3, :3] @ (rays * depth.ravel()) + c2w[:3, 3:4]).T.reshape(h, w, 3)


def test_each_chunks_depth_lands_on_the_aligned_points():
    """Two chunks, the second at half chunk 0's scale (its Sim(3) scale is 2). Lifted with
    the global cameras, every frame's stored depth must land on the aligned world points
    -- before the fix, chunk 1's frames landed at half the distance."""
    init = _init()
    K = np.array([[4.0, 0, 2.0], [0, 4.0, 2.0], [0, 0, 1]])
    c2w = np.stack([_c2w(_rot_y(6.0 * i), np.array([0.3 * i, 0.0, -0.1 * i])) for i in range(4)])
    ramp = np.arange(4)[None, :] * 0.1
    true_depth = np.stack([np.full((4, 4), 3.0 + i) + ramp for i in range(4)])
    points = np.stack([_lift(true_depth[i], K, c2w[i]) for i in range(4)])
    frames = [FrameRef("cam_00", i, 0.1 * i) for i in range(4)]
    scale = 2.0

    def chunk(index, start, end, depth_scale):
        n = end - start
        return {
            "_chunk_range": (start, end),
            "_chunk_index": index,
            "extrinsic": np.stack([np.eye(4)] * n),
            "intrinsic": np.stack([K] * n),
            "depth": (true_depth[start:end] / depth_scale)[..., None],
            "world_points": points[start:end],
        }

    # Frame 2 is in both chunks; the later one's depth is the one camera_poses.txt pairs it with.
    chunks = [chunk(0, 0, 3, 1.0), chunk(1, 2, 4, scale)]
    chunks[0]["depth"][2] *= 1.3  # chunk 0's (wrong) take on the overlap frame
    preds = init._combined_predictions_from_chunks(
        frames, chunks, global_c2w=c2w, chunk_scales=[1.0, scale]
    )
    for i in range(4):
        lifted = _lift(preds["depth"][i, ..., 0], K, init._world_from_cam(preds["extrinsic"][i]))
        assert np.allclose(lifted, preds["point_map"][i], atol=1e-9), f"frame {i}"
