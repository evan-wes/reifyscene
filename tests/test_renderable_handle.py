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


def _n(node_id: str, kind: str, **attrs) -> dict:
    return {"node_id": node_id, "node_type": kind, "attrs": attrs}


def test_nodes_give_up_their_stream_and_canonical_tracklet() -> None:
    assert _node_track_id(_n("object:cam_00:sam2:cam_00:3", "object", stream_id="cam_00")) \
        == ("cam_00", "sam2:cam_00:3")
    # a DyCheck stream is "0": the stream is the node's own, not an assumed "cam_00"
    assert _node_track_id(_n("human:0:sam2:0:1", "human", stream_id="0")) == ("0", "sam2:0:1")
    # a recorded track id is used as recorded
    assert _node_track_id(_n("object:x:y", "object", stream_id="0", track_id="sam2:0:9")) \
        == ("0", "sam2:0:9")
    assert _node_track_id(_n("background:cam_00", "background", stream_id="cam_00")) is None
    assert _node_track_id(_n("cam_00", "camera")) is None
    assert _node_track_id(_n("object:cam_00:sam2:cam_00:3", "object")) is None  # no stream


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


# -- background removal and identity handles (I4) ---------------------------------------------

def test_drop_can_remove_background_gaussians_too() -> None:
    """A standing person the motion split put in the static cloud lives in bg_*; a remove
    that could only touch fg_* would leave them standing."""
    b = _bundle()
    out = drop_gaussians(b, [0], bg_indices=[1, 3])
    assert out["fg_means"].shape == (G - 1, 3)
    assert out["bg_means"].shape == (3, 3)
    assert np.array_equal(out["bg_means"], b["bg_means"][[0, 2, 4]])
    assert np.array_equal(out["bg_opacities"], b["bg_opacities"][[0, 2, 4]])
    assert np.array_equal(out["motion_rots"], b["motion_rots"])


def test_background_removal_does_not_touch_the_foreground() -> None:
    b = _bundle()
    out = drop_gaussians(b, [], bg_indices=[0])
    assert np.array_equal(out["fg_means"], b["fg_means"])
    assert out["bg_means"].shape == (4, 3)


def test_identity_rows_are_dropped_with_their_gaussians() -> None:
    b = _bundle()
    b["fg_identity"] = np.arange(G * 2, dtype=np.float32).reshape(G, 2)
    b["bg_identity"] = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)
    out = drop_gaussians(b, [2], bg_indices=[4])
    assert np.array_equal(out["fg_identity"], b["fg_identity"][[i for i in range(G) if i != 2]])
    assert np.array_equal(out["bg_identity"], b["bg_identity"][:4])


def _identity_bundle() -> dict[str, np.ndarray]:
    b = _bundle()
    # Two dims, three classes: dim 0 -> class 1, dim 1 -> class 2, neither -> background.
    b["identity_classifier_w"] = np.array([[0, 0], [10, 0], [0, 10]], np.float32)
    b["identity_classifier_b"] = np.array([5, 0, 0], np.float32)
    fg = np.zeros((G, 2), np.float32)
    fg[[0, 1], 0] = 1.0            # class 1, confident
    fg[2] = [0.52, 0.0]            # class 1, ~55%: in at 0.5, out at 0.7
    fg[3, 1] = 1.0                 # class 2
    b["fg_identity"] = fg
    bg = np.zeros((5, 2), np.float32)
    bg[4, 0] = 1.0                 # class 1 lives in the static cloud too
    b["bg_identity"] = bg
    b["identity_node_ids"] = np.array(
        ["", "object:cam_00:sam2:cam_00:0", "human:cam_00:sam2:cam_00:1"]
    )
    return b


def test_identity_handles_threshold_both_fields() -> None:
    from reifyscene.graph.renderable_handle import identity_handles

    h = identity_handles(_identity_bundle(), threshold=0.5)
    assert set(h) == {"object:cam_00:sam2:cam_00:0", "human:cam_00:sam2:cam_00:1"}
    fg1, bg1 = h["object:cam_00:sam2:cam_00:0"]
    assert fg1.tolist() == [0, 1, 2] and bg1.tolist() == [4]
    fg2, bg2 = h["human:cam_00:sam2:cam_00:1"]
    assert fg2.tolist() == [3] and bg2.tolist() == []
    strict = identity_handles(_identity_bundle(), threshold=0.7)
    assert strict["object:cam_00:sam2:cam_00:0"][0].tolist() == [0, 1]


def test_identity_handles_refuse_a_checkpoint_without_identity() -> None:
    from reifyscene.graph.renderable_handle import identity_handles

    with pytest.raises(ValueError, match="identity"):
        identity_handles(_bundle())
