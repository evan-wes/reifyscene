"""Static Gaussian splatting trainer.

Our own optimization loop over the ``Rasterizer`` contract + ``GaussianField``:
Adam with per-group learning rates (means LR exponentially decayed), an
L1 + D-SSIM photometric loss, and the :class:`DensificationController` driving
adaptive density control. Everything is config-toggleable via
:class:`TrainConfig` (+ :class:`DensifyConfig`). Anchors the proven 3DGS recipe;
the code is ours and never imports gsplat directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

from room4d.render.gaussian4d.densify import DensificationController, DensifyConfig
from room4d.render.gaussian4d.gaussian_field import GaussianField
from room4d.render.gaussian4d.rasterizer import RenderCamera, build_rasterizer


# -- losses / metrics -----------------------------------------------------

def _gaussian_kernel(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = (g[:, None] @ g[None, :])[None, None]
    return kernel_2d


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """SSIM for (B, C, H, W) images in [0, 1]."""
    channel = img1.shape[1]
    window = _gaussian_kernel(window_size, sigma, img1.device, img1.dtype).expand(channel, 1, window_size, window_size).contiguous()
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def _to_bchw(img_hw3: torch.Tensor) -> torch.Tensor:
    return img_hw3.permute(2, 0, 1).unsqueeze(0)


def image_psnr(pred_hw3: torch.Tensor, gt_hw3: torch.Tensor) -> float:
    mse = ((pred_hw3 - gt_hw3) ** 2).mean()
    return float(-10.0 * torch.log10(mse.clamp_min(1e-12)))


def image_ssim(pred_hw3: torch.Tensor, gt_hw3: torch.Tensor) -> float:
    return float(ssim(_to_bchw(pred_hw3), _to_bchw(gt_hw3)))


# -- config ---------------------------------------------------------------

@dataclass
class TrainConfig:
    iters: int = 15000
    color_model: str = "rgb"
    lambda_l1: float = 0.8
    lambda_ssim: float = 0.2
    lr_means: float = 1.6e-4
    lr_means_final: float = 1.6e-6
    lr_scales: float = 5.0e-3
    lr_quats: float = 1.0e-3
    lr_opacities: float = 5.0e-2
    lr_colors: float = 2.5e-3
    means_lr_decay_steps: int = 15000
    random_seed: int = 0
    log_every: int = 200

    @classmethod
    def from_cfg(cls, cfg: dict | None) -> "TrainConfig":
        cfg = cfg or {}
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in cfg.items() if k in fields})


class StaticGaussianTrainer:
    """Optimize a static GaussianField against posed RGB views."""

    def __init__(
        self,
        field: GaussianField,
        cfg: TrainConfig,
        densify_cfg: DensifyConfig,
        *,
        device: str = "cuda",
        rasterizer: Any = None,
        scene_extent: float | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.field = field.to(device)
        self.rasterizer = rasterizer or build_rasterizer({"device": device, "absgrad": densify_cfg.absgrad})
        self.optimizer = self._build_optimizer()
        self.controller = DensificationController(self.field, self.optimizer, densify_cfg, scene_extent)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        c = self.cfg
        return torch.optim.Adam(
            [
                {"params": [self.field.means], "lr": c.lr_means, "name": "means"},
                {"params": [self.field.log_scales], "lr": c.lr_scales, "name": "log_scales"},
                {"params": [self.field.quats], "lr": c.lr_quats, "name": "quats"},
                {"params": [self.field.logit_opacities], "lr": c.lr_opacities, "name": "logit_opacities"},
                {"params": [self.field.colors_logit], "lr": c.lr_colors, "name": "colors_logit"},
            ],
            eps=1e-15,
        )

    def _means_lr_at(self, step: int) -> float:
        c = self.cfg
        t = min(step / max(c.means_lr_decay_steps, 1), 1.0)
        ratio = (c.lr_means_final / c.lr_means) if c.lr_means > 0 else 1.0
        return c.lr_means * (ratio ** t)

    def _loss(self, pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        l1 = (pred - gt).abs().mean()
        dssim = 1.0 - ssim(_to_bchw(pred), _to_bchw(gt))
        loss = self.cfg.lambda_l1 * l1 + self.cfg.lambda_ssim * dssim
        return loss, float(l1.detach()), float(dssim.detach())

    def fit(
        self,
        train_views: list[tuple[RenderCamera, torch.Tensor]],
        *,
        on_log: Callable[[dict], None] | None = None,
    ) -> GaussianField:
        """Train on (camera, gt_image[H,W,3] in [0,1]) pairs. Returns the field."""
        if not train_views:
            raise ValueError("StaticGaussianTrainer.fit got no training views.")
        torch.manual_seed(self.cfg.random_seed)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.cfg.random_seed)
        n_views = len(train_views)

        for step in range(1, self.cfg.iters + 1):
            for group in self.optimizer.param_groups:
                if group["name"] == "means":
                    group["lr"] = self._means_lr_at(step)

            idx = int(torch.randint(n_views, (1,), generator=gen).item())
            camera, gt = train_views[idx]
            means, quats, scales, opacities, colors = self.field.activated()
            out = self.rasterizer.rasterize(means, quats, scales, opacities, colors, camera, with_telemetry=True)
            loss, l1, dssim = self._loss(out.rgb, gt)

            loss.backward()
            self.controller.after_backward(out)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            stats = self.controller.after_step(step)

            if on_log is not None and (step % self.cfg.log_every == 0 or step == 1):
                with torch.no_grad():
                    psnr = image_psnr(out.rgb.detach().clamp(0, 1), gt)
                record = {
                    "step": step,
                    "loss": float(loss.detach()),
                    "l1": l1,
                    "psnr": psnr,
                    "num_gaussians": self.field.num_gaussians,
                }
                if stats:
                    record["densify"] = stats
                on_log(record)

        return self.field

    @torch.no_grad()
    def evaluate(self, views: list[tuple[RenderCamera, torch.Tensor]]) -> list[dict]:
        """Render each (camera, gt) view and score plain PSNR/SSIM (unmasked)."""
        cloud = self.field.to_cloud()
        results = []
        for camera, gt in views:
            out = self.rasterizer.rasterize(
                cloud.means, cloud.quats, cloud.scales, cloud.opacities, cloud.colors, camera
            )
            pred = out.rgb.clamp(0, 1)
            results.append({"psnr": image_psnr(pred, gt), "ssim": image_ssim(pred, gt), "rgb": pred.cpu().numpy()})
        return results
