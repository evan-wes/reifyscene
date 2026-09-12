"""Similarity transforms (Sim(3)) between two frames that share a set of cameras.

Two consumers: ``pose_refine`` re-anchors bundle-adjusted poses onto their input
frame, and masked-NVS scoring places a benchmark's held-out cameras in a
self-reconstructed scene's frame by fitting the transform on the training
cameras, whose poses are known in both.
"""

from __future__ import annotations

import numpy as np


def camera_center(w2c: np.ndarray) -> np.ndarray:
    """World-space centre of a world->camera pose."""
    return -w2c[:3, :3].T @ w2c[:3, 3]


def fit_similarity(
    src_w2c: dict[str, np.ndarray], dst_w2c: dict[str, np.ndarray], keys: list[str]
) -> tuple[float, np.ndarray, np.ndarray]:
    """Orientation-based Sim(3) mapping the ``src`` world frame onto ``dst``.

    Rotation is estimated from camera ORIENTATIONS (well conditioned for a
    looking-around rig), not the camera-center cloud, which is near-degenerate
    for planar/linear sweeps and yields a spurious orientation error. Scale +
    translation then fit the centers under that rotation. Returns ``(s, R, t)``
    with ``x_dst = s * R @ x_src + t``. A non-positive ``s`` means the centres do
    not agree with the orientations (wrong correspondences), not a valid fit.
    """
    G = [src_w2c[k][:3, :3].T @ dst_w2c[k][:3, :3] for k in keys]
    U, _, Vt = np.linalg.svd(np.sum(G, axis=0))
    Gavg = U @ Vt
    if np.linalg.det(Gavg) < 0:
        U[:, -1] *= -1
        Gavg = U @ Vt
    R = Gavg.T
    sc = np.stack([camera_center(src_w2c[k]) for k in keys])
    dc = np.stack([camera_center(dst_w2c[k]) for k in keys])
    smu, dmu = sc.mean(0), dc.mean(0)
    sch, dch = sc - smu, dc - dmu
    s = float(np.sum(dch * (R @ sch.T).T) / np.sum(sch**2))
    t = dmu - s * (R @ smu)
    return s, R, t


def apply_similarity_to_c2w(c2w: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Map a camera-to-world pose through the world similarity ``x' = s*R@x + t``.

    The scale is dropped from the rotation block so it stays orthonormal; only
    the camera center (a world point) carries the scale + translation.
    """
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R @ c2w[:3, :3]
    out[:3, 3] = s * (R @ c2w[:3, 3]) + t
    return out


def similarity_residuals(
    src_w2c: dict[str, np.ndarray],
    dst_w2c: dict[str, np.ndarray],
    keys: list[str],
    s: float,
    R: np.ndarray,
    t: np.ndarray,
) -> dict[str, float]:
    """How well a fitted similarity maps each ``src`` camera onto its ``dst`` counterpart.

    Per camera: the rotation left over between the mapped and the target
    orientation (degrees), and the centre distance as a fraction of the ``dst``
    trajectory's extent (so it is unit-free). Returns their medians and maxima.
    """
    dst_centres = np.stack([camera_center(dst_w2c[k]) for k in keys])
    span = float(np.ptp(dst_centres, axis=0).max()) or 1.0
    rot, ctr = [], []
    for k in keys:
        mapped = apply_similarity_to_c2w(np.linalg.inv(src_w2c[k]), s, R, t)
        target = np.linalg.inv(dst_w2c[k])
        cos = (np.trace(mapped[:3, :3].T @ target[:3, :3]) - 1.0) / 2.0
        rot.append(float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))))
        ctr.append(float(np.linalg.norm(mapped[:3, 3] - target[:3, 3]) / span))
    return {
        "rotation_residual_deg_median": float(np.median(rot)),
        "rotation_residual_deg_max": float(np.max(rot)),
        "centre_residual_median": float(np.median(ctr)),
        "centre_residual_max": float(np.max(ctr)),
    }
