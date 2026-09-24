"""The graph's handle on the trained cloud (reifyscene/graph/renderable_handle.py).

What these pin is the part that must not go wrong quietly: an edit executed as a checkpoint
transform. Dropping the wrong rows, or dropping rows from an array that is not per-Gaussian,
produces a bundle that still loads and renders something plausible.
"""

from __future__ import annotations

import numpy as np
import pytest

from reifyscene.graph.renderable_handle import (
    _node_track_id,
    drop_gaussians,
    trained_fg_positions,
)

G, T, K = 12, 4, 2


def _bundle() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    rots = np.zeros((K, T, 6), np.float32)
    rots[:, :, 0] = 1.0
    rots[:, :, 4] = 1.0            # 6D identity rotation
    return {
        "fg_means": rng.normal(size=(G, 3)).astype(np.float32),
        "fg_quats": rng.normal(size=(G, 4)).astype(np.float32),
        "fg_scales": rng.normal(size=(G, 3)).astype(np.float32),
        "fg_colors": rng.normal(size=(G, 3)).astype(np.float32),
        "fg_opacities": rng.normal(size=(G,)).astype(np.float32),
        "fg_motion_coefs": rng.normal(size=(G, K)).astype(np.float32),
        "motion_rots": rots,
        "motion_transls": np.zeros((K, T, 3), np.float32),
        "bg_means": rng.normal(size=(5, 3)).astype(np.float32),
        "bg_quats": rng.normal(size=(5, 4)).astype(np.float32),
        "bg_scales": rng.normal(size=(5, 3)).astype(np.float32),
        "bg_colors": rng.normal(size=(5, 3)).astype(np.float32),
        "bg_opacities": rng.normal(size=(5,)).astype(np.float32),
        "train_w2cs": np.tile(np.eye(4, dtype=np.float32), (T, 1, 1)),
        "train_Ks": np.tile(np.eye(3, dtype=np.float32), (T, 1, 1)),
        "train_time_ids": np.arange(T),
        "train_frame_names": np.array([f"cam_00:{i}" for i in range(T)]),
    }


def test_node_ids_give_up_their_canonical_tracklet() -> None:
    assert _node_track_id("object:cam_00:sam2:cam_00:3", "cam_00") == "sam2:cam_00:3"
    assert _node_track_id("human:cam_00:sam2:cam_00:11", "cam_00") == "sam2:cam_00:11"
    assert _node_track_id("background:cam_00", "cam_00") is None
    assert _node_track_id("cam_00", "cam_00") is None


def test_drop_removes_exactly_the_named_gaussians() -> None:
    b = _bundle()
    out = drop_gaussians(b, [1, 3, 5])
    assert out["fg_means"].shape == (G - 3, 3)
    kept = [i for i in range(G) if i not in (1, 3, 5)]
    assert np.array_equal(out["fg_means"], b["fg_means"][kept])
    assert np.array_equal(out["fg_motion_coefs"], b["fg_motion_coefs"][kept])
    assert np.array_equal(out["fg_opacities"], b["fg_opacities"][kept])


def test_drop_leaves_shared_and_background_arrays_alone() -> None:
    """The motion bases are shared across Gaussians and the background is a different cloud.

    Filtering either by a foreground index would corrupt the scene while still loading.
    """
    b = _bundle()
    out = drop_gaussians(b, [0, 2])
    for key in ("motion_rots", "motion_transls", "bg_means", "bg_quats", "bg_opacities",
                "train_w2cs", "train_Ks", "train_time_ids", "train_frame_names"):
        assert np.array_equal(out[key], b[key]), key


def test_dropping_nothing_is_the_identity() -> None:
    b = _bundle()
    out = drop_gaussians(b, [])
    assert np.array_equal(out["fg_means"], b["fg_means"])


def test_an_out_of_range_index_is_refused(tmp_path) -> None:
    with pytest.raises(IndexError, match="out of range"):
        drop_gaussians(_bundle(), [G])


def test_a_dropped_bundle_still_warps(tmp_path) -> None:
    """The edited bundle must remain a loadable scene, not just a dict of arrays."""
    out = drop_gaussians(_bundle(), [0, 1])
    positions, w2c, K_, frames = trained_fg_positions(out, device="cpu")
    assert positions.shape == (T, G - 2, 3)
    assert w2c.shape == (T, 4, 4) and K_.shape == (T, 3, 3)
    assert frames == [0, 1, 2, 3]


def test_positions_are_per_frame_not_canonical() -> None:
    """A deformable foreground has a different pose per frame; one canonical pose would
    project the wrong pixels on all but one of them."""
    b = _bundle()
    b["motion_transls"][0, :, 0] = np.arange(T, dtype=np.float32)   # basis 0 slides in x
    b["fg_motion_coefs"][:] = np.array([20.0, -20.0], np.float32)   # everything on basis 0
    positions, *_ = trained_fg_positions(b, device="cpu")
    moved = positions[:, 0, 0]
    assert np.allclose(np.diff(moved), 1.0, atol=1e-3), moved
