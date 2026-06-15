"""Dynamic Gaussian trainer (Shape-of-Motion-style deformation).

Trains a :class:`DynamicScene` (fg deformed by motion bases + static bg) against
posed RGB frames sampled across time: photometric L1 + D-SSIM plus a temporal
motion-smoothness regularizer. v1 refines a borrowed init and does NOT densify
(the borrowed init is fixed-size); dynamic densification is tracked follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from room4d.render.gaussian4d.dynamic_field import DynamicScene
from room4d.render.gaussian4d.gaussian_trainer import image_psnr, ssim, _to_bchw
from room4d.render.gaussian4d.rasterizer import RenderCamera, build_rasterizer


@dataclass
class DynamicTrainConfig:
    iters: int = 5000
    lambda_l1: float = 0.8
    lambda_ssim: float = 0.2
    lambda_smooth: float = 1.0
    lr_means: float = 1.6e-4
    lr_scales: float = 5.0e-3
    lr_quats: float = 1.0e-3
    lr_opacities: float = 5.0e-2
    lr_colors: float = 2.5e-3
    lr_motion_coefs: float = 1.0e-2
    lr_motion_bases: float = 1.6e-4
    random_seed: int = 0
    log_every: int = 250

    @classmethod
    def from_cfg(cls, cfg: dict | None) -> "DynamicTrainConfig":
        cfg = cfg or {}
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in cfg.items() if k in fields})


class DynamicGaussianTrainer:
    """Optimize a DynamicScene against (camera, time, gt) views."""

    def __init__(self, scene: DynamicScene, cfg: DynamicTrainConfig, *, device: str = "cuda", rasterizer=None) -> None:
        self.cfg = cfg
        self.device = device
        self.scene = scene.to(device)
        self.rasterizer = rasterizer or build_rasterizer({"device": device})
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        c = self.cfg
        fg = self.scene.fg
        groups = [
            {"params": [fg.means], "lr": c.lr_means, "name": "fg.means"},
            {"params": [fg.log_scales], "lr": c.lr_scales, "name": "fg.log_scales"},
            {"params": [fg.quats], "lr": c.lr_quats, "name": "fg.quats"},
            {"params": [fg.logit_opacities], "lr": c.lr_opacities, "name": "fg.logit_opacities"},
            {"params": [fg.colors_logit], "lr": c.lr_colors, "name": "fg.colors_logit"},
            {"params": [fg.motion_coefs], "lr": c.lr_motion_coefs, "name": "fg.motion_coefs"},
            {"params": [self.scene.motion_bases.rots], "lr": c.lr_motion_bases, "name": "bases.rots"},
            {"params": [self.scene.motion_bases.transls], "lr": c.lr_motion_bases, "name": "bases.transls"},
        ]
        if self.scene.bg is not None:
            bg = self.scene.bg
            groups += [
                {"params": [bg.means], "lr": c.lr_means, "name": "bg.means"},
                {"params": [bg.log_scales], "lr": c.lr_scales, "name": "bg.log_scales"},
                {"params": [bg.quats], "lr": c.lr_quats, "name": "bg.quats"},
                {"params": [bg.logit_opacities], "lr": c.lr_opacities, "name": "bg.logit_opacities"},
                {"params": [bg.colors_logit], "lr": c.lr_colors, "name": "bg.colors_logit"},
            ]
        return torch.optim.Adam(groups, eps=1e-15)

    def _photometric(self, pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, float]:
        l1 = (pred - gt).abs().mean()
        dssim = 1.0 - ssim(_to_bchw(pred), _to_bchw(gt))
        return self.cfg.lambda_l1 * l1 + self.cfg.lambda_ssim * dssim, float(l1.detach())

    def fit(
        self,
        train_views: list[tuple[RenderCamera, torch.Tensor, int]],
        *,
        on_log: Callable[[dict], None] | None = None,
    ) -> DynamicScene:
        if not train_views:
            raise ValueError("DynamicGaussianTrainer.fit got no training views.")
        torch.manual_seed(self.cfg.random_seed)
        gen = torch.Generator(device="cpu").manual_seed(self.cfg.random_seed)
        n = len(train_views)

        for step in range(1, self.cfg.iters + 1):
            camera, gt, time_id = train_views[int(torch.randint(n, (1,), generator=gen).item())]
            cloud = self.scene.cloud_at(time_id)
            out = self.rasterizer.rasterize(
                cloud.means, cloud.quats, cloud.scales, cloud.opacities, cloud.colors, camera
            )
            photo, l1 = self._photometric(out.rgb, gt)
            loss = photo + self.cfg.lambda_smooth * self.scene.motion_smoothness()

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            if on_log is not None and (step % self.cfg.log_every == 0 or step == 1):
                with torch.no_grad():
                    psnr = image_psnr(out.rgb.detach().clamp(0, 1), gt)
                on_log({"step": step, "loss": float(loss.detach()), "l1": l1, "psnr": psnr})

        return self.scene

    @torch.no_grad()
    def render_at(self, camera: RenderCamera, time_id: int) -> torch.Tensor:
        cloud = self.scene.cloud_at(time_id)
        out = self.rasterizer.rasterize(
            cloud.means, cloud.quats, cloud.scales, cloud.opacities, cloud.colors, camera
        )
        return out.rgb.clamp(0, 1)
