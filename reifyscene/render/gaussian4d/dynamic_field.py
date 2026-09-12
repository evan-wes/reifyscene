"""Dynamic foreground Gaussian field + the combined dynamic scene.

A foreground Gaussian carries canonical params (like the static field) plus
softmax ``motion_coefs`` over the motion bases. At frame ``t`` it is warped by
its blended SE(3) transform; the warped fg is concatenated with the static
background into a single :class:`GaussianCloud` for rasterization. A per-Gaussian
``entity_id`` channel is derived from fg/bg membership (fg = 1, bg = 0) for v1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from reifyscene.render.gaussian4d.gaussian_field import PARAM_NAMES, GaussianField
from reifyscene.render.gaussian4d.motion import MotionBases
from reifyscene.render.gaussian4d.rasterizer import GaussianCloud
from reifyscene.render.gaussian4d.transforms import quat_multiply, rotmat_to_quat

# Foreground densification grows these (the static set + per-Gaussian motion).
FG_PARAM_NAMES = PARAM_NAMES + ("motion_coefs",)


class DynamicGaussianField(GaussianField):
    """Static-field params + per-Gaussian softmax motion coefficients."""

    def __init__(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        log_scales: torch.Tensor,
        logit_opacities: torch.Tensor,
        colors_logit: torch.Tensor,
        motion_coefs: torch.Tensor,
        *,
        color_model: str = "rgb",
    ) -> None:
        super().__init__(means, quats, log_scales, logit_opacities, colors_logit, color_model=color_model)
        self.motion_coefs = nn.Parameter(motion_coefs.float())  # (G, K) raw logits

    @property
    def num_bases(self) -> int:
        return int(self.motion_coefs.shape[1])

    def get_coefs(self) -> torch.Tensor:
        return torch.softmax(self.motion_coefs, dim=-1)

    def warp(self, bases: MotionBases, ts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Warp canonical means/quats to frames ``ts`` -> (G, B, 3), (G, B, 4)."""
        transfms = bases.compute_transforms(ts, self.get_coefs())  # (G, B, 3, 4)
        means_t = torch.einsum("pnij,pj->pni", transfms, F.pad(self.means, (0, 1), value=1.0))
        q_rot = rotmat_to_quat(transfms[..., :3, :3])  # (G, B, 4)
        quats_t = F.normalize(quat_multiply(q_rot, self.get_quats()[:, None]), dim=-1)
        return means_t, quats_t


class DynamicScene(nn.Module):
    """Foreground (deformable) + static background + shared motion bases."""

    def __init__(
        self,
        fg: DynamicGaussianField,
        motion_bases: MotionBases,
        bg: GaussianField | None = None,
    ) -> None:
        super().__init__()
        self.fg = fg
        self.motion_bases = motion_bases
        self.bg = bg

    @property
    def num_frames(self) -> int:
        return self.motion_bases.num_frames

    @property
    def num_gaussians(self) -> int:
        return self.fg.num_gaussians + (self.bg.num_gaussians if self.bg is not None else 0)

    def cloud_at(self, t: int) -> GaussianCloud:
        """Materialize the (grad-bearing) combined cloud at frame index ``t``.

        ``t`` is clamped to the motion bases' frame range: an out-of-range index
        (e.g. a held-out camera whose time-id exceeds the trained frames) degrades
        to the nearest frame instead of an out-of-bounds CUDA assert.
        """
        t = max(0, min(int(t), self.num_frames - 1))
        ts = torch.as_tensor([int(t)], device=self.fg.means.device, dtype=torch.long)
        fg_means, fg_quats = self.fg.warp(self.motion_bases, ts)
        means, quats = fg_means[:, 0], fg_quats[:, 0]
        scales, opacities, colors = self.fg.get_scales(), self.fg.get_opacities(), self.fg.get_colors()
        if self.bg is not None:
            means = torch.cat([means, self.bg.means], dim=0)
            quats = torch.cat([quats, self.bg.get_quats()], dim=0)
            scales = torch.cat([scales, self.bg.get_scales()], dim=0)
            opacities = torch.cat([opacities, self.bg.get_opacities()], dim=0)
            colors = torch.cat([colors, self.bg.get_colors()], dim=0)
        return GaussianCloud(means=means, quats=quats, scales=scales, opacities=opacities, colors=colors)

    def entity_ids(self) -> torch.Tensor:
        """Per-Gaussian entity id for the combined cloud (fg = 1, bg = 0)."""
        n_fg = self.fg.num_gaussians
        n_bg = self.bg.num_gaussians if self.bg is not None else 0
        return torch.cat([
            torch.ones(n_fg, dtype=torch.long, device=self.fg.means.device),
            torch.zeros(n_bg, dtype=torch.long, device=self.fg.means.device),
        ])

    def motion_smoothness(self) -> torch.Tensor:
        """Temporal smoothness penalty on the bases (mean squared 1st difference)."""
        rots, transls = self.motion_bases.rots, self.motion_bases.transls
        return (rots[:, 1:] - rots[:, :-1]).pow(2).mean() + (transls[:, 1:] - transls[:, :-1]).pow(2).mean()
