"""Low-dimensional SE(3) motion bases (Shape-of-Motion deformation model).

K shared bases, each a per-frame rotation (6D) + translation; a foreground
Gaussian's per-frame SE(3) transform is the coefficient-weighted blend of the
bases. Ported from SoM's ``flow3d/params.py::MotionBases`` onto our transforms.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from reifyscene.render.gaussian4d.transforms import cont_6d_to_rmat


class MotionBases(nn.Module):
    """`rots` (K, T, 6) + `transls` (K, T, 3) over K bases and T frames."""

    def __init__(self, rots: torch.Tensor, transls: torch.Tensor) -> None:
        super().__init__()
        assert rots.shape[:2] == transls.shape[:2], "rots/transls must share (K, T)"
        self.num_bases = int(rots.shape[0])
        self.num_frames = int(rots.shape[1])
        self.rots = nn.Parameter(rots.float())
        self.transls = nn.Parameter(transls.float())

    def compute_transforms(self, ts: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
        """Per-Gaussian SE(3) transforms at frames ``ts``.

        ts (B,) frame indices, coefs (G, K) per-Gaussian basis weights ->
        transforms (G, B, 3, 4).
        """
        transls = self.transls[:, ts]  # (K, B, 3)
        rots = self.rots[:, ts]        # (K, B, 6)
        transls = torch.einsum("pk,kni->pni", coefs, transls)  # (G, B, 3)
        rots = torch.einsum("pk,kni->pni", coefs, rots)        # (G, B, 6)
        rotmats = cont_6d_to_rmat(rots)                        # (G, B, 3, 3)
        return torch.cat([rotmats, transls[..., None]], dim=-1)  # (G, B, 3, 4)
