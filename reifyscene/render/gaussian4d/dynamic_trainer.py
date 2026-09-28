"""Dynamic Gaussian trainer (Shape-of-Motion-style deformation).

Trains a :class:`DynamicScene` (fg deformed by motion bases + static bg) against
posed RGB frames sampled across time: photometric L1 + D-SSIM plus a temporal
motion-smoothness regularizer. When the scene carries identity
(:meth:`DynamicScene.enable_identity`) and a view carries an id map, a
cross-entropy term supervises the rasterized identity features through the
scene's linear classifier (Gaussian Grouping's 2D loss).

By default identity **reads** the geometry and never writes it: the features are
rendered in a second pass with every geometric input detached, so the identity loss
reaches only the identity vectors and the classifier. The photometric loss matches
measured pixels; the id maps are pseudo-labels (SAM 2 masks through reid and the
graph), and letting their errors move geometry cost 1.2 dB on lab-coat. Densification
then also sees photometric gradients only. ``identity_detach_geometry=False`` is the
coupled single-pass variant, kept for comparison. v1 refines a borrowed init and does NOT densify
(the borrowed init is fixed-size); dynamic densification is tracked follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import math

import torch
import torch.nn.functional as F

from reifyscene.graph.id_maps import IGNORE
from reifyscene.render.gaussian4d.densify import DensificationController, DensifyConfig, compute_scene_extent
from reifyscene.render.gaussian4d.dynamic_field import FG_PARAM_NAMES, DynamicScene
from reifyscene.render.gaussian4d.gaussian_field import IDENTITY_PARAM, PARAM_NAMES
from reifyscene.render.gaussian4d.gaussian_trainer import image_psnr, ssim, _to_bchw
from reifyscene.render.gaussian4d.rasterizer import RenderCamera, build_rasterizer


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
    lambda_identity: float = 1.0
    identity_detach_geometry: bool = True
    lr_identity: float = 2.5e-3
    lr_identity_classifier: float = 5.0e-4
    random_seed: int = 0
    log_every: int = 250

    @classmethod
    def from_cfg(cls, cfg: dict | None) -> "DynamicTrainConfig":
        cfg = cfg or {}
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in cfg.items() if k in fields})


class DynamicGaussianTrainer:
    """Optimize a DynamicScene against (camera, time, gt) views."""

    def __init__(
        self,
        scene: DynamicScene,
        cfg: DynamicTrainConfig,
        densify_cfg: DensifyConfig | None = None,
        *,
        device: str = "cuda",
        rasterizer=None,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.scene = scene.to(device)
        self.rasterizer = rasterizer or build_rasterizer({"device": device, "absgrad": (densify_cfg or DensifyConfig()).absgrad})
        self.optimizer = self._build_optimizer()

        # One densification controller per field (foreground grows motion_coefs
        # too), both driven by the same combined-cloud telemetry. Standard 3DGS
        # adaptive density control, applied to fg + bg as in Shape-of-Motion.
        self.densify_cfg = densify_cfg or DensifyConfig(enabled=False)
        scene_extent = compute_scene_extent(scene.cloud_at(0).means.detach())
        # Identity rides densification like any other per-Gaussian param: a split
        # child inherits its parent's vector, a pruned Gaussian takes its vector along.
        extra = (IDENTITY_PARAM,) if self.scene.has_identity else ()
        self.fg_ctrl = DensificationController(
            self.scene.fg, self.optimizer, self.densify_cfg, scene_extent,
            param_names=FG_PARAM_NAMES + extra, group_prefix="fg.",
        )
        self.bg_ctrl = (
            DensificationController(
                self.scene.bg, self.optimizer, self.densify_cfg, scene_extent,
                param_names=PARAM_NAMES + extra, group_prefix="bg.",
            )
            if self.scene.bg is not None
            else None
        )

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
        if self.scene.has_identity:
            groups.append({"params": [fg.identity], "lr": c.lr_identity, "name": "fg.identity"})
            if self.scene.bg is not None:
                groups.append(
                    {"params": [self.scene.bg.identity], "lr": c.lr_identity, "name": "bg.identity"}
                )
            groups.append({
                "params": list(self.scene.identity_classifier.parameters()),
                "lr": c.lr_identity_classifier, "name": "identity_classifier",
            })
        return torch.optim.Adam(groups, eps=1e-15)

    def identity_loss(self, features: torch.Tensor, id_map: torch.Tensor) -> torch.Tensor | None:
        """Cross-entropy of rendered identity against an id map, normalised to ~(0, 1).

        Pixels marked ignore (overlaps, dropped tracklets, padding) contribute nothing.
        Divided by ``log(num_classes)`` as Gaussian Grouping does, so the weight means
        the same across scenes with 4 classes and with 24. None when no pixel is labelled.
        """
        target = id_map.reshape(-1).long()
        labelled = target != IGNORE
        if not bool(labelled.any()):
            return None
        head = self.scene.identity_classifier
        logits = head(features.reshape(-1, features.shape[-1]))
        ce = F.cross_entropy(logits[labelled], target[labelled])
        return ce / math.log(head.out_features)

    def _photometric(self, pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, float]:
        l1 = (pred - gt).abs().mean()
        dssim = 1.0 - ssim(_to_bchw(pred), _to_bchw(gt))
        return self.cfg.lambda_l1 * l1 + self.cfg.lambda_ssim * dssim, float(l1.detach())

    def identity_term(self, cloud, camera: RenderCamera, id_map, out) -> torch.Tensor | None:
        """The weighted identity loss for one view, or None when nothing is labelled.

        Detached (default): a second, feature-only pass over a geometry-detached cloud, so
        gradients reach the identity vectors and classifier and nothing else. Coupled:
        reuses ``out``, whose features were rendered with live geometry.
        """
        if self.cfg.identity_detach_geometry:
            feat = self.rasterizer.rasterize(
                cloud.means.detach(), cloud.quats.detach(), cloud.scales.detach(),
                cloud.opacities.detach(), cloud.colors.detach(), camera,
                features=cloud.features,
            ).features
        else:
            feat = out.features
        loss = self.identity_loss(feat, id_map)
        return None if loss is None else self.cfg.lambda_identity * loss

    def fit(
        self,
        train_views: list[tuple],
        *,
        on_log: Callable[[dict], None] | None = None,
    ) -> DynamicScene:
        if not train_views:
            raise ValueError("DynamicGaussianTrainer.fit got no training views.")
        torch.manual_seed(self.cfg.random_seed)
        gen = torch.Generator(device="cpu").manual_seed(self.cfg.random_seed)
        n = len(train_views)

        """``train_views`` are ``(camera, gt, time_id)`` or, for identity training,
        ``(camera, gt, time_id, id_map)`` with ``id_map`` an (H, W) integer tensor."""
        if self.scene.has_identity and not any(len(v) > 3 and v[3] is not None
                                               for v in train_views):
            raise ValueError("the scene carries identity but no training view has an id map")
        id_loss_value = float("nan")
        for step in range(1, self.cfg.iters + 1):
            view = train_views[int(torch.randint(n, (1,), generator=gen).item())]
            camera, gt, time_id = view[:3]
            id_map = view[3] if len(view) > 3 else None
            n_fg = self.scene.fg.num_gaussians
            n_bg = self.scene.bg.num_gaussians if self.scene.bg is not None else 0
            cloud = self.scene.cloud_at(time_id)
            coupled = not self.cfg.identity_detach_geometry
            out = self.rasterizer.rasterize(
                cloud.means, cloud.quats, cloud.scales, cloud.opacities, cloud.colors, camera,
                with_telemetry=self.densify_cfg.enabled,
                features=cloud.features if coupled else None,
            )
            photo, l1 = self._photometric(out.rgb, gt)
            loss = photo + self.cfg.lambda_smooth * self.scene.motion_smoothness()
            if cloud.features is not None and id_map is not None:
                id_term = self.identity_term(cloud, camera, id_map, out)
                if id_term is not None:
                    loss = loss + id_term
                    id_loss_value = float(id_term.detach()) / max(self.cfg.lambda_identity, 1e-12)

            loss.backward()
            self.fg_ctrl.after_backward(out, (0, n_fg))
            if self.bg_ctrl is not None:
                self.bg_ctrl.after_backward(out, (n_fg, n_fg + n_bg))
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.fg_ctrl.after_step(step)
            if self.bg_ctrl is not None:
                self.bg_ctrl.after_step(step)

            if on_log is not None and (step % self.cfg.log_every == 0 or step == 1):
                with torch.no_grad():
                    psnr = image_psnr(out.rgb.detach().clamp(0, 1), gt)
                record = {"step": step, "loss": float(loss.detach()), "l1": l1, "psnr": psnr,
                          "num_gaussians": self.scene.num_gaussians}
                if self.scene.has_identity:
                    record["identity_loss"] = id_loss_value
                on_log(record)

        return self.scene

    @torch.no_grad()
    def render_at(self, camera: RenderCamera, time_id: int) -> torch.Tensor:
        cloud = self.scene.cloud_at(time_id)
        out = self.rasterizer.rasterize(
            cloud.means, cloud.quats, cloud.scales, cloud.opacities, cloud.colors, camera
        )
        return out.rgb.clamp(0, 1)
