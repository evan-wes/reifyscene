"""Trainable static Gaussian field.

Holds the optimizable Gaussian parameters as ``nn.Parameter``s in the standard
3DGS parameterization (log-space scales, logit-space opacities, sigmoid colors)
and exposes activated tensors for rasterization. Seeded from the PR-1
depth-backprojection :class:`GaussianCloud`. The densification controller grows
and prunes these parameters in lockstep with the optimizer state.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from reifyscene.render.gaussian4d.rasterizer import GaussianCloud

# Optimizer param groups are keyed by these names so densification can do
# per-parameter Adam-state surgery.
PARAM_NAMES = ("means", "quats", "log_scales", "logit_opacities", "colors_logit")
# Optional per-Gaussian identity vector (D channels, rasterized as features). Not in
# PARAM_NAMES: a field only carries it when identity training is on, and the trainer
# adds it to the densifier's list then, so splits and prunes carry it with the rest.
IDENTITY_PARAM = "identity"


def logit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


class GaussianField(nn.Module):
    """Optimizable static Gaussian set (RGB color model for v1; SH is future)."""

    def __init__(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        log_scales: torch.Tensor,
        logit_opacities: torch.Tensor,
        colors_logit: torch.Tensor,
        *,
        color_model: str = "rgb",
        identity: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.color_model = str(color_model)
        self.means = nn.Parameter(means.float())
        self.quats = nn.Parameter(quats.float())
        self.log_scales = nn.Parameter(log_scales.float())
        self.logit_opacities = nn.Parameter(logit_opacities.float())
        self.colors_logit = nn.Parameter(colors_logit.float())
        self.identity: nn.Parameter | None
        self.register_parameter(
            IDENTITY_PARAM, None if identity is None else nn.Parameter(identity.float())
        )

    @classmethod
    def from_cloud(cls, cloud: GaussianCloud, *, color_model: str = "rgb") -> "GaussianField":
        scales = cloud.scales.detach().float().clamp_min(1e-8)
        return cls(
            cloud.means.detach().float().clone(),
            cloud.quats.detach().float().clone(),
            torch.log(scales),
            logit(cloud.opacities.detach().float()),
            logit(cloud.colors.detach().float()),
            color_model=color_model,
        )

    # -- views -------------------------------------------------------------

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[0])

    @property
    def device(self) -> torch.device:
        return self.means.device

    def get_scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales)

    def get_opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_opacities)

    def get_colors(self) -> torch.Tensor:
        return torch.sigmoid(self.colors_logit)

    def get_quats(self) -> torch.Tensor:
        return F.normalize(self.quats, dim=-1)

    @property
    def identity_dim(self) -> int:
        return 0 if self.identity is None else int(self.identity.shape[1])

    def init_identity(self, dim: int, *, generator: torch.Generator | None = None) -> None:
        """Give every Gaussian a small random identity vector (D channels).

        Small and random rather than zero: identical vectors get identical gradients,
        and a field that starts symmetric has nothing to break the tie with.
        """
        noise = torch.randn((self.num_gaussians, int(dim)), generator=generator)
        self.identity = nn.Parameter((0.1 * noise).to(self.device))

    def activated(self) -> tuple[torch.Tensor, ...]:
        """(means, quats, scales, opacities, colors) ready for rasterization."""
        return (self.means, self.get_quats(), self.get_scales(), self.get_opacities(), self.get_colors())

    def to_cloud(self) -> GaussianCloud:
        """Detached snapshot for no-grad rendering/export."""
        with torch.no_grad():
            means, quats, scales, opacities, colors = self.activated()
        return GaussianCloud(
            means=means.detach(),
            quats=quats.detach(),
            scales=scales.detach(),
            opacities=opacities.detach(),
            colors=colors.detach(),
            features=None if self.identity is None else self.identity.detach(),
        )

    # -- densification support --------------------------------------------

    def get_param(self, name: str) -> nn.Parameter:
        return getattr(self, name)

    def set_param(self, name: str, param: nn.Parameter) -> None:
        setattr(self, name, param)
