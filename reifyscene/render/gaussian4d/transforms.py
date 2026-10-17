# Portions adapted from Shape of Motion (https://github.com/vye16/shape-of-motion),
# Copyright (c) 2024 Vickie Ye, under the MIT License (LICENSES/MIT-shape-of-motion.txt).
"""Rotation helpers for the SE(3) motion model (ported, no roma dependency).

The 6D continuous rotation parameterization (Zhou et al.) is what the
Shape-of-Motion motion bases use; ``cont_6d_to_rmat`` / ``rmat_to_cont_6d`` are
ported verbatim from SoM's ``flow3d/transforms.py``. The quaternion helpers
(wxyz convention) replace the few ``roma`` calls used when composing a warp's
rotation onto a canonical Gaussian quaternion.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rmat_to_cont_6d(matrix: torch.Tensor) -> torch.Tensor:
    """(*, 3, 3) rotation matrices -> (*, 6) continuous representation."""
    return torch.cat([matrix[..., 0], matrix[..., 1]], dim=-1)


def cont_6d_to_rmat(cont_6d: torch.Tensor) -> torch.Tensor:
    """(*, 6) continuous representation -> (*, 3, 3) rotation matrices."""
    x1 = cont_6d[..., 0:3]
    y1 = cont_6d[..., 3:6]
    x = F.normalize(x1, dim=-1)
    y = F.normalize(y1 - (y1 * x).sum(dim=-1, keepdim=True) * x, dim=-1)
    z = torch.linalg.cross(x, y, dim=-1)
    return torch.stack([x, y, z], dim=-1)


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of wxyz quaternions: returns a ⊗ b (broadcasts)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """sqrt on the positive part, with a zero (not infinite) gradient elsewhere."""
    ret = torch.zeros_like(x)
    positive = x > 0
    ret[positive] = torch.sqrt(x[positive])
    return ret


def rotmat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """(*, 3, 3) rotation matrices -> (*, 4) wxyz unit quaternions.

    Gradient-stable form (matches pytorch3d): masked sqrt avoids the infinite
    gradient of ``sqrt`` at zero, and the denominator is floored at 0.1 so the
    division gradient stays bounded at branch boundaries.
    """
    m = matrix
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )  # ~2|w|, 2|x|, 2|y|, 2|z|

    quat_w = torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1)
    quat_x = torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1)
    quat_y = torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1)
    quat_z = torch.stack([m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3] ** 2], dim=-1)
    candidates = torch.stack([quat_w, quat_x, quat_y, quat_z], dim=-2)
    flr = torch.tensor(0.1, dtype=matrix.dtype, device=matrix.device)
    candidates = candidates / (2.0 * torch.max(q_abs, flr)[..., None])

    best = q_abs.argmax(dim=-1)
    quat = torch.gather(candidates, -2, best[..., None, None].expand(*best.shape, 1, 4)).squeeze(-2)
    return F.normalize(quat, dim=-1)
