"""Dynamic Gaussian trainer (Shape-of-Motion-style deformation).

Trains a :class:`DynamicScene` (fg deformed by motion bases + static bg) against
posed RGB frames sampled across time: photometric L1 + D-SSIM plus a temporal
motion-smoothness regularizer. Both fields get standard 3DGS adaptive density
control, driven by the combined cloud's screen-space gradients. When the scene
carries identity (:meth:`DynamicScene.enable_identity`) and a view carries an id
map, a cross-entropy term supervises the rasterized identity features through the
scene's linear classifier (Gaussian Grouping's 2D loss).

By default identity **reads** the geometry and never writes it: the features are
rendered in a second pass with every geometric input detached, so the identity loss
reaches only the identity vectors and the classifier. The photometric loss matches
measured pixels; the id maps are pseudo-labels (SAM 2 masks through reid and the
graph), and letting their errors move geometry cost 1.2 dB on lab-coat. Densification
then also sees photometric gradients only. ``identity_detach_geometry=False`` is the
coupled single-pass variant, kept for comparison.

When the scene's Gaussians have **owners** (each one's entity, fixed at seeding and inherited
through densification) and the views carry id maps, the **owner loss** asks each entity's
Gaussians to cover its pixels and nothing else: the owners are rendered as one channel each in
a second pass whose geometry is live -- this term moves Gaussians, it is the point -- so
densification still sees photometric gradients only. Every ``owner_drift_every`` steps the
FlashSplat evidence of each Gaussian is compared with its owner, and Gaussians whose evidence
went to another entity are counted (flagged, never relabelled;
``docs/design/2026-10-17-removal-by-ownership.md`` §4.3-4.5).

**Containment** (§4.6; on whenever the scene has owners, like the owner loss) keeps each
Gaussian in its owner's space where one camera cannot: a cap on the longest-to-shortest axis
ratio of background-owned Gaussians, against needles along the line of sight; the **ghost
prune**, which at the drift checks removes background-owned Gaussians whose evidence lies almost
all on entities' pixels -- they paint only an entity, and removing it would leave them behind;
**body containment**, a per-Gaussian loss (not through a render, so it reaches Gaussians hidden
in every view) that keeps each entity's Gaussians inside its body or close behind its own
surface; and a prune of background-owned Gaussians inside a body's core.

A 3D consistency term (Gaussian Grouping's ``loss_cls_3d``) pulls each sampled
Gaussian's class distribution toward its k nearest neighbours' -- canonical means for
the foreground, which stay neighbours on a typical frame across all ten loop scenes,
and 3D means for the static background. Positions are detached: identity only.

**Identity reset** (off by default) is opacity reset's counterpart: every
``identity_reset_every`` steps, until ``identity_reset_until``, every Gaussian's identity
vector is set to zero -- the classifier's bias alone, no opinion -- and its Adam state
cleared. A Gaussian the 2D loss reaches earns its identity back; one it never reaches
(hidden in every training view) stays unassigned, below any selection threshold, instead
of keeping a label it was never checked against. It also clears labels picked up early,
while the geometry was still blobby, from Gaussians that sharper geometry later hid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import math

import numpy as np
import torch
import torch.nn.functional as F

from reifyscene.graph.id_maps import BACKGROUND, labelled
from reifyscene.render.gaussian4d.densify import (
    DensificationController,
    DensifyConfig,
    _quat_to_rotmat,
    compute_scene_extent,
)
from reifyscene.render.gaussian4d.dynamic_field import FG_PARAM_NAMES, DynamicScene
from reifyscene.render.gaussian4d.gaussian_field import IDENTITY_PARAM, OWNER_BUFFER, PARAM_NAMES
from reifyscene.render.gaussian4d.gaussian_trainer import image_psnr, ssim, _to_bchw
from reifyscene.render.gaussian4d.losses import (
    capsule_distance,
    gradient_loss,
    masked_l1_loss,
    owner_loss,
    scale_ratio_loss,
)
from reifyscene.render.gaussian4d.rasterizer import RenderCamera, build_rasterizer


def knn_indices(
    query: torch.Tensor, points: torch.Tensor, k: int, *,
    exclude: torch.Tensor | None = None, chunk: int = 65536,
) -> torch.Tensor:
    """(M, k) indices of each query's k nearest ``points``, chunked over ``points``.

    Chunked so a 400k-point background never materialises an (M, N) distance matrix.
    ``exclude[i]`` is query i's own index in ``points``, skipped so a point is never its
    own neighbour.
    """
    best_d = torch.full((query.shape[0], 0), float("inf"), device=query.device)
    best_i = torch.zeros((query.shape[0], 0), dtype=torch.long, device=query.device)
    for s in range(0, points.shape[0], chunk):
        d = torch.cdist(query, points[s:s + chunk])
        if exclude is not None:
            own = exclude - s
            hit = (own >= 0) & (own < d.shape[1])
            d[hit.nonzero(as_tuple=True)[0], own[hit]] = float("inf")
        kk = min(k, d.shape[1])
        dv, di = d.topk(kk, dim=1, largest=False)
        best_d = torch.cat([best_d, dv], dim=1)
        best_i = torch.cat([best_i, di + s], dim=1)
        best_d, order = best_d.topk(min(k, best_d.shape[1]), dim=1, largest=False)
        best_i = best_i.gather(1, order)
    return best_i


@dataclass
class DynamicTrainConfig:
    iters: int = 5000
    lambda_l1: float = 0.8
    lambda_ssim: float = 0.2
    lambda_smooth: float = 1.0
    # Shape-of-Motion's geometry terms (flow3d/configs.py): disparity against the depth prior
    # (its weight, 0.5), its spatial gradient (theirs 1.0) and the mask term (theirs 1.0: total
    # opacity driven to 1, the rendered foreground share against the foreground mask).
    # Measured on lab-coat (2026-10-01, 47 frames, same geometry): depth alone +0.1 dB over
    # photos only; + gradient -0.2 dB; the mask term -1.9 dB -- our "foreground" is every
    # segmented instance, theirs the moving part, so it pushes static objects into the moving
    # field. Gradient and mask are off until that is reconciled.
    lambda_depth: float = 0.5
    lambda_depth_grad: float = 0.0
    lambda_mask: float = 0.0
    # The owner loss (per-entity silhouettes), on whenever the scene has owners; Shape-of-
    # Motion's weight for its mask term. 0 trains the same owned scene without it.
    lambda_owner: float = 1.0
    owner_drift_every: int = 500        # 0: no drift check
    # Containment (removal-by-ownership §4.6), on like the owner loss whenever the scene has
    # owners; 0 / False turns each off. The scale-ratio cap (splatfacto: r = 10, weight 0.1,
    # every 10 steps) on background-owned Gaussians, against needles stretched along the line
    # of sight behind an entity, where no view constrains their length.
    lambda_scale_ratio: float = 0.1
    max_scale_ratio: float = 10.0
    scale_ratio_every: int = 10
    # The ghost prune: at every drift check but the last, background-owned Gaussians whose
    # blending weight falls on entities' pixels, with under this share of it on the
    # background's, are pruned -- they paint only an entity, so removing it would leave them
    # behind as its ghost.
    ghost_max_bg_share: float = 0.1
    # Body containment. Each entity's own Gaussians are kept in the space it occupies on each
    # training frame: inside its body (a person's Sapiens capsules, built by the scene stage)
    # or within a band behind its own surface where the frame shows it -- the clothing margin
    # for a person, so loose clothes and hair outside the capsules are not squeezed in; one
    # entity width for anything without a body on that frame. A Gaussian reaches its centre
    # plus ``body_sigmas`` standard deviations; how far it reaches out is measured in its
    # entity's size (torso length, or width), so the weight is scale-free.
    lambda_body: float = 1.0
    body_sigmas: float = 2.0
    # At every drift check but the last, background-owned Gaussians whose centre lies in a
    # body's core (its capsules less the clothing margin) on some training frame are pruned:
    # that space was a person's at that moment.
    prune_background_in_body: bool = True
    lr_means: float = 1.6e-4
    lr_scales: float = 5.0e-3
    lr_quats: float = 1.0e-3
    lr_opacities: float = 5.0e-2
    lr_colors: float = 2.5e-3
    lr_motion_coefs: float = 1.0e-2
    lr_motion_bases: float = 1.6e-4
    lambda_identity: float = 1.0
    identity_detach_geometry: bool = True
    lambda_identity_3d: float = 2.0
    identity_3d_every: int = 2
    identity_3d_k: int = 5
    identity_3d_samples: int = 800
    # Gaussian Grouping divides the 3D KL by num_classes, which is 200 in their setup. Ours
    # is 4-24 per scene, so dividing by ours made lambda=2 8-50x stronger than theirs and
    # flattened identity to buy agreement. A fixed divisor reproduces their strength and
    # stops scenes with fewer classes being smoothed harder.
    identity_3d_norm: float = 200.0
    # Training-time diagnostics (identity_trace.IdentityTracer), off by default.
    # Training signal (Shape-of-Motion trains batches of 8 views). 8 views per step against 1,
    # same 5k-step schedule, on lab-coat / gold-fish / judo: +5.6 to +7.4 dB PSNR over every
    # training frame and 1.4-3.4x the sharpness -- the difference between scenes too soft to
    # judge an edit on and scenes that are not. It costs ~5x render-stage time.
    views_per_step: int = 8
    # SoM also decays the scales' learning rate to 0.1x; measured here it made scenes softer
    # alone and with batching, so it is off (1.0 = constant).
    lr_scales_final_ratio: float = 1.0
    identity_reset_every: int = 0
    identity_reset_until: int | None = None     # None: iters - 2 * identity_reset_every
    identity_trace_every: int = 0
    identity_trace_views: int = 8
    snapshot_steps: tuple[int, ...] = ()
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
        self.last_identity_reset: int | None = None

        # One densification controller per field (foreground grows motion_coefs
        # too), both driven by the same combined-cloud telemetry. Standard 3DGS
        # adaptive density control, applied to fg + bg as in Shape-of-Motion.
        self.densify_cfg = densify_cfg or DensifyConfig(enabled=False)
        scene_extent = compute_scene_extent(scene.cloud_at(0).means.detach())
        # Identity rides densification like any other per-Gaussian param: a split
        # child inherits its parent's vector, a pruned Gaussian takes its vector along.
        extra = (IDENTITY_PARAM,) if self.scene.has_identity else ()
        # Owners ride densification too: a child is its parent's, a pruned Gaussian's goes.
        buffers = (OWNER_BUFFER,) if self.scene.has_owners else ()
        self.fg_ctrl = DensificationController(
            self.scene.fg, self.optimizer, self.densify_cfg, scene_extent,
            param_names=FG_PARAM_NAMES + extra, group_prefix="fg.", buffer_names=buffers,
        )
        self.bg_ctrl = (
            DensificationController(
                self.scene.bg, self.optimizer, self.densify_cfg, scene_extent,
                param_names=PARAM_NAMES + extra, group_prefix="bg.", buffer_names=buffers,
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

    @torch.no_grad()
    def reset_identity(self, step: int) -> None:
        """Zero every Gaussian's identity vector and its optimizer momentum; keep the head.

        The momentum must go too: Adam would otherwise push each vector straight back to
        the label it just lost, which is the reset not happening.
        """
        names = {"fg.identity": self.scene.fg}
        if self.scene.bg is not None:
            names["bg.identity"] = self.scene.bg
        for group in self.optimizer.param_groups:
            field = names.get(group.get("name"))
            if field is None:
                continue
            field.identity.zero_()
            state = self.optimizer.state.get(group["params"][0])
            if state:
                state["exp_avg"].zero_()
                state["exp_avg_sq"].zero_()
        self.last_identity_reset = int(step)

    def _identity_reset_due(self, step: int) -> bool:
        every = self.cfg.identity_reset_every
        if not self.scene.has_identity or every <= 0 or step % every:
            return False
        until = self.cfg.identity_reset_until
        until = self.cfg.iters - 2 * every if until is None else until
        return step <= until

    def identity_loss(self, features: torch.Tensor, id_map: torch.Tensor) -> torch.Tensor | None:
        """Cross-entropy of rendered identity against an id map, normalised to ~(0, 1).

        Pixels that are not labelled (dropped tracklets, overlaps, padding) contribute nothing.
        Divided by ``log(num_classes)`` as Gaussian Grouping does, so the weight means
        the same across scenes with 4 classes and with 24. None when no pixel is labelled.
        """
        target = id_map.reshape(-1).long()
        known = labelled(target)
        if not bool(known.any()):
            return None
        head = self.scene.identity_classifier
        logits = head(features.reshape(-1, features.shape[-1]))
        ce = F.cross_entropy(logits[known], target[known])
        return ce / math.log(head.out_features)

    def _photometric(self, pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, float]:
        l1 = (pred - gt).abs().mean()
        dssim = 1.0 - ssim(_to_bchw(pred), _to_bchw(gt))
        return self.cfg.lambda_l1 * l1 + self.cfg.lambda_ssim * dssim, float(l1.detach())

    def identity_consistency_loss(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """KL from each sampled Gaussian's class distribution to its k nearest neighbours'.

        Per field, in that field's own space: canonical means for the foreground, means for
        the background -- fg and bg are never each other's neighbours, since one is a
        deformable cloud in its canonical pose and the other is static in the world.
        Divided by a fixed ``identity_3d_norm`` (200, Gaussian Grouping's class count) rather
        than our per-scene class count, so their lambda means the same here.
        """
        head = self.scene.identity_classifier
        fields = [self.scene.fg] + ([self.scene.bg] if self.scene.bg is not None else [])
        terms = []
        for field in fields:
            n = field.num_gaussians
            k = min(self.cfg.identity_3d_k, n - 1)
            if k < 1:
                continue
            probs = torch.softmax(head(field.identity), dim=-1)
            pts = field.means.detach()
            m = min(self.cfg.identity_3d_samples, n)
            idx = torch.randperm(n, generator=generator)[:m].to(pts.device)
            nn = knn_indices(pts[idx], pts, k, exclude=idx)
            p = probs[idx].unsqueeze(1)                           # (m, 1, C)
            q = probs[nn]                                         # (m, k, C)
            kl = (p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))).sum(-1).mean()
            terms.append(kl / self.cfg.identity_3d_norm)
        if not terms:
            return torch.zeros((), device=self.scene.fg.means.device)
        return torch.stack(terms).mean()

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

    def owner_term(self, cloud, camera: RenderCamera, ids: torch.Tensor,
                   onehot: torch.Tensor) -> torch.Tensor | None:
        """The weighted owner loss for one view (None when nothing is labelled): the owner
        channels ``onehot`` (N, C) rendered with live geometry in a pass of their own."""
        shares = self.rasterizer.rasterize(
            cloud.means, cloud.quats, cloud.scales, cloud.opacities, cloud.colors.detach(),
            camera, features=onehot,
        ).features
        loss = owner_loss(shares, ids, labelled(ids))
        return None if loss is None else self.cfg.lambda_owner * loss

    @torch.no_grad()
    def owner_evidence(self, views: list[tuple]) -> torch.Tensor:
        """(N, K): each Gaussian's FlashSplat evidence, its blending weight summed over each
        id-map class's pixels (background 0, the entities after) across ``views``
        (``(camera, time_id, id map)``). Occlusion counts: a Gaussian hidden behind another
        gets only the weight that reaches the pixel."""
        from reifyscene.graph.assignment import evidence_matrix

        with torch.enable_grad():
            return evidence_matrix(self.scene, self.rasterizer, views,
                                   len(self.scene.owner_node_ids))

    @torch.no_grad()
    def owner_drift(self, counts: torch.Tensor) -> dict:
        """Gaussians whose FlashSplat evidence ``counts`` (:meth:`owner_evidence`) now goes to
        another entity than their owner.

        Only Gaussians that some entity's or the background's evidence clearly claims
        (slackness 0) and that are owned by an entity or the background are judged; a flag is
        counted per (owner, evidence) pair of id-map codes. Flagged, never relabelled: a
        Gaussian that is one thing at one time and another at another makes removal
        ill-posed, and relabelling it hides that.
        """
        from reifyscene.graph.assignment import assign

        k = counts.shape[1]
        member = assign(counts)
        claimed = member.any(dim=1)
        evidence = member.float().argmax(dim=1)
        owners = self.scene.owners()
        judged = claimed & (owners < k)
        flagged = judged & (evidence != owners)
        pairs: dict[str, int] = {}
        if bool(flagged.any()):
            o, e = owners[flagged].tolist(), evidence[flagged].tolist()
            for a, b in zip(o, e, strict=True):
                pairs[f"{a}->{b}"] = pairs.get(f"{a}->{b}", 0) + 1
        return {"judged": int(judged.sum()), "flagged": int(flagged.sum()),
                "pairs": dict(sorted(pairs.items(), key=lambda kv: -kv[1]))}

    def ghosts(self, counts: torch.Tensor) -> torch.Tensor:
        """bool (N,): the background-owned Gaussians whose evidence ``counts`` falls on
        entities' pixels, with under ``ghost_max_bg_share`` of it on the background's."""
        total = counts.sum(dim=1)
        return ((self.scene.owners() == BACKGROUND) & (total > 1e-8)
                & (counts[:, BACKGROUND] < self.cfg.ghost_max_bg_share * total))

    @torch.no_grad()
    def background_in_bodies(self, views: list[tuple], targets: list[dict]) -> torch.Tensor:
        """bool (N,): the background-owned Gaussians whose centre lies in a body's core -- its
        capsules less the clothing margin -- on the frame of some view (``(camera, gt,
        time_id, ...)``, its target's ``bodies``)."""
        background = self.scene.owners() == BACKGROUND
        hit = torch.zeros_like(background)
        for view, target in zip(views, targets, strict=True):
            bodies = target.get("bodies") or {}
            if not bodies or not bool(background.any()):
                continue
            points = self.scene.cloud_at(int(view[2])).means[background]
            inside = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
            for body in bodies.values():
                core = body["capsules"].clone()
                core[:, 6] -= body["margin"]
                inside |= capsule_distance(points, core) < 0
            hit[background] |= inside
        return hit

    @torch.no_grad()
    def prune(self, drop: torch.Tensor) -> int:
        """Prune the ``drop`` (N,) Gaussians from both fields; returns how many went."""
        keep = ~drop
        n_fg = self.scene.fg.num_gaussians
        pruned = self.fg_ctrl.prune(keep[:n_fg])
        if self.bg_ctrl is not None:
            pruned += self.bg_ctrl.prune(keep[n_fg:])
        return pruned

    @staticmethod
    def entity_widths(camera: RenderCamera, target: dict, k: int) -> dict[int, float]:
        """Each entity's horizontal extent on one view in scene units, the 5th-95th percentile
        of its pixels' camera x, as the removal tests' slab has it (``entity_width``); an
        entity with under 10 pixels of depth has none."""
        ids, depth, valid = target["ids"], target["depth"], target["valid"]
        K = np.asarray(camera.K, dtype=np.float64)
        out: dict[int, float] = {}
        for code in torch.unique(ids[valid]).tolist():
            m = valid & (ids == code)
            if not 0 < code < k or int(m.sum()) < 10:
                continue
            v, u = m.nonzero(as_tuple=True)
            x = (u.float() - K[0, 2]) / K[0, 0] * depth[v, u]
            lo, hi = torch.quantile(x, torch.tensor([0.05, 0.95], device=x.device)).tolist()
            if hi > lo:
                out[int(code)] = hi - lo
        return out

    def body_term(self, cloud, camera: RenderCamera, target: dict, widths: dict[int, float],
                  members: dict[int, torch.Tensor]) -> torch.Tensor | None:
        """The weighted body-containment loss of one view: the mean, over the entity-owned
        Gaussians it can judge, of how far each reaches out of the space its owner occupies
        on this frame, in its owner's size (None when it can judge none). ``members`` maps an
        entity's code to its Gaussians (bool (N,)); ``widths`` are :meth:`entity_widths`."""
        ids, depth, valid = target["ids"], target["depth"], target["valid"]
        bodies = target.get("bodies") or {}
        h, w = ids.shape
        dev = cloud.means.device
        viewmat = torch.as_tensor(np.asarray(camera.viewmat), dtype=torch.float32, device=dev)
        K = np.asarray(camera.K, dtype=np.float64)
        reach = self.cfg.body_sigmas
        excess = []
        for code, member in members.items():
            body, width = bodies.get(code), widths.get(code)
            if body is None and width is None:
                continue
            band, size = (body["margin"], body["torso"]) if body is not None else (width, width)
            means, scales = cloud.means[member], cloud.scales[member]
            # Behind its own surface: where this frame shows the entity at the Gaussian's pixel,
            # its far side (centre + reach along the camera's z) may be at most `band` behind.
            cam = means @ viewmat[:3, :3].T + viewmat[:3, 3]
            z = cam[:, 2]
            zc = z.detach().clamp_min(1e-6)
            u = torch.floor(K[0, 0] * cam[:, 0].detach() / zc + K[0, 2]).long()
            v = torch.floor(K[1, 1] * cam[:, 1].detach() / zc + K[1, 2]).long()
            seen = (z.detach() > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u, v = u.clamp(0, w - 1), v.clamp(0, h - 1)
            seen = seen & valid[v, u] & (ids[v, u] == code)
            axes = viewmat[:3, :3] @ _quat_to_rotmat(cloud.quats[member])   # in camera axes
            sigma_z = (scales * axes[:, 2, :]).norm(dim=-1)
            behind = F.relu(z + reach * sigma_z - depth[v, u] - band) / size
            out = torch.where(seen, behind, torch.full_like(behind, float("inf")))
            if body is not None:     # or inside its body, all round
                inside = F.relu(capsule_distance(means, body["capsules"])
                                + reach * scales.amax(dim=-1)) / size
                out = torch.minimum(out, inside)
            excess.append(out[torch.isfinite(out)])
        judged = torch.cat(excess) if excess else torch.zeros(0, device=dev)
        if judged.numel() == 0:
            return None
        return self.cfg.lambda_body * judged.mean()

    def scale_ratio_term(self) -> torch.Tensor | None:
        """The weighted scale-ratio cap over the background-owned Gaussians of both fields
        (None when there are none)."""
        scales = [self.scene.fg.get_scales()]
        if self.scene.bg is not None:
            scales.append(self.scene.bg.get_scales())
        owned = self.scene.owners() == BACKGROUND
        if not bool(owned.any()):
            return None
        loss = scale_ratio_loss(torch.cat(scales, dim=0)[owned], self.cfg.max_scale_ratio)
        return self.cfg.lambda_scale_ratio * loss

    def geometry_terms(self, out, fg_share, target) -> tuple[torch.Tensor, dict[str, float]]:
        """The depth, depth-gradient and mask losses of one view against its ``target``
        (``depth`` (H, W) in scene units, ``valid`` (H, W) bool, ``fg`` (H, W) float);
        ``fg_share`` is the rendered foreground share, rendered only while the mask term is on."""
        stats: dict[str, float] = {}
        loss = torch.zeros((), device=out.depth.device)
        valid = target["valid"]
        if (self.cfg.lambda_depth > 0 or self.cfg.lambda_depth_grad > 0) and bool(valid.any()):
            pred_disp = 1.0 / (out.depth + 1e-5)
            tgt_disp = 1.0 / (target["depth"] + 1e-5)
            weights = valid.to(pred_disp.dtype)[..., None]
            depth = masked_l1_loss(pred_disp[..., None], tgt_disp[..., None], mask=weights,
                                   quantile=0.98)
            loss = loss + self.cfg.lambda_depth * depth
            stats["depth_loss"] = float(depth.detach())
            if self.cfg.lambda_depth_grad > 0:
                grad = gradient_loss(pred_disp[None], tgt_disp[None], mask=valid[None],
                                     quantile=0.95)
                loss = loss + self.cfg.lambda_depth_grad * grad
                stats["depth_grad_loss"] = float(grad.detach())
        if self.cfg.lambda_mask > 0:
            mask = F.mse_loss(out.alpha, torch.ones_like(out.alpha)) + masked_l1_loss(
                fg_share[..., None], target["fg"][..., None], quantile=0.98)
            loss = loss + self.cfg.lambda_mask * mask
            stats["mask_loss"] = float(mask.detach())
        return loss, stats

    def fit(
        self,
        train_views: list[tuple],
        *,
        targets: list[dict] | None = None,
        on_log: Callable[[dict], None] | None = None,
        on_trace: Callable[[dict], None] | None = None,
        on_snapshot: Callable[[int, DynamicScene], None] | None = None,
        on_drift: Callable[[dict], None] | None = None,
    ) -> DynamicScene:
        """``train_views`` are ``(camera, gt, time_id)`` or, for identity training,
        ``(camera, gt, time_id, id_map)`` with ``id_map`` an (H, W) integer tensor.
        ``targets``, one per view, carry its geometry supervision (:meth:`geometry_terms`)
        and, for an owned scene, its id map (``ids``); they are required while any geometry
        weight is on, or the owner loss is. ``on_drift`` receives each drift check.

        ``on_trace`` receives an identity trace record every ``identity_trace_every`` steps
        (and at step 1); ``on_snapshot`` receives the scene at each of ``snapshot_steps``.
        """
        if not train_views:
            raise ValueError("DynamicGaussianTrainer.fit got no training views.")
        geometry_on = (self.cfg.lambda_depth > 0 or self.cfg.lambda_depth_grad > 0
                       or self.cfg.lambda_mask > 0)
        if geometry_on and (targets is None or len(targets) != len(train_views)
                            or any(t is None for t in targets)):
            raise ValueError("the geometry terms are on but not every training view has "
                             "depth / mask targets (set lambda_depth, lambda_depth_grad and "
                             "lambda_mask to 0 to train on photos alone)")
        owners_on = self.scene.has_owners and self.cfg.lambda_owner > 0
        drift_on = self.scene.has_owners and self.cfg.owner_drift_every > 0
        if (owners_on or drift_on) and (targets is None or len(targets) != len(train_views)
                                        or any(t is None or "ids" not in t for t in targets)):
            raise ValueError("the scene's Gaussians have owners but not every training view "
                             "has an id map (set lambda_owner and owner_drift_every to 0 to "
                             "train without them)")
        owned = self.scene.has_owners
        ratio_on = owned and self.cfg.lambda_scale_ratio > 0
        ghosts_on = owned and self.cfg.ghost_max_bg_share > 0
        body_on = owned and self.cfg.lambda_body > 0
        in_body_on = owned and self.cfg.prune_background_in_body
        if (ghosts_on or in_body_on) and not drift_on:
            raise ValueError("the ghost and background-in-body prunes run at the drift checks "
                             "(set owner_drift_every > 0, or turn them off)")
        if (body_on or in_body_on) and (targets is None
                                        or any("bodies" not in t for t in targets)):
            raise ValueError("body containment needs the scene stage's bodies, and this scene "
                             "was assembled without them: rebuild it from the scene stage")
        pruning = ghosts_on or in_body_on
        k_owners = len(self.scene.owner_node_ids)
        widths = ([self.entity_widths(v[0], t, k_owners) for v, t in zip(train_views, targets,
                                                                          strict=True)]
                  if body_on else [])
        drift_views = ([(v[0], v[2], t["ids"]) for v, t in zip(train_views, targets, strict=True)]
                       if drift_on else [])
        owner_value = ratio_value = body_value = float("nan")
        geo_stats: dict[str, float] = {}
        mask_on = self.cfg.lambda_mask > 0
        torch.manual_seed(self.cfg.random_seed)
        gen = torch.Generator(device="cpu").manual_seed(self.cfg.random_seed)
        n = len(train_views)
        tracer = None
        if on_trace is not None and self.cfg.identity_trace_every > 0 and self.scene.has_identity:
            from reifyscene.render.gaussian4d.identity_trace import IdentityTracer

            tracer = IdentityTracer.from_train_views(train_views, n=self.cfg.identity_trace_views)
        snapshots = set(int(s) for s in self.cfg.snapshot_steps)

        if self.scene.has_identity and not any(len(v) > 3 and v[3] is not None
                                               for v in train_views):
            raise ValueError("the scene carries identity but no training view has an id map")
        id_loss_value = float("nan")
        id3d_value = float("nan")
        gen3d = torch.Generator(device="cpu").manual_seed(self.cfg.random_seed + 1)
        batch = max(1, min(int(self.cfg.views_per_step), n))
        scale_groups = [g for g in self.optimizer.param_groups
                        if g.get("name") in ("fg.log_scales", "bg.log_scales")]
        for step in range(1, self.cfg.iters + 1):
            if self.cfg.lr_scales_final_ratio != 1.0:
                t = min(step / max(self.cfg.iters, 1), 1.0)
                for g in scale_groups:
                    g["lr"] = self.cfg.lr_scales * self.cfg.lr_scales_final_ratio ** t
            picks = torch.randperm(n, generator=gen)[:batch].tolist() if batch > 1 else \
                [int(torch.randint(n, (1,), generator=gen).item())]
            n_fg = self.scene.fg.num_gaussians
            n_bg = self.scene.bg.num_gaussians if self.scene.bg is not None else 0
            coupled = not self.cfg.identity_detach_geometry
            loss = self.cfg.lambda_smooth * self.scene.motion_smoothness()
            if ratio_on and step % max(self.cfg.scale_ratio_every, 1) == 0:
                ratio = self.scale_ratio_term()
                if ratio is not None:
                    loss = loss + ratio
                    ratio_value = float(ratio.detach()) / self.cfg.lambda_scale_ratio
            if owners_on:                # owners change only when densification runs
                channel, n_channels = self.scene.owner_channels()
                onehot = F.one_hot(channel, n_channels).to(torch.float32)
            if body_on:                  # ... or a prune does
                owners = self.scene.owners()
                members = {c: owners == c for c in torch.unique(owners).tolist()
                           if 0 < c < k_owners}
            outs, l1s = [], []
            for idx in picks:
                view = train_views[idx]
                camera, gt, time_id = view[:3]
                id_map = view[3] if len(view) > 3 else None
                cloud = self.scene.cloud_at(time_id)
                feats = cloud.features if coupled else None
                if mask_on:              # the foreground share rides as the last feature channel
                    fg_ind = torch.zeros((n_fg + n_bg, 1), device=cloud.means.device)
                    fg_ind[:n_fg] = 1.0
                    feats = fg_ind if feats is None else torch.cat([feats, fg_ind], dim=-1)
                out = self.rasterizer.rasterize(
                    cloud.means, cloud.quats, cloud.scales, cloud.opacities, cloud.colors, camera,
                    with_telemetry=self.densify_cfg.enabled, features=feats,
                )
                photo, l1 = self._photometric(out.rgb, gt)
                loss = loss + photo / batch
                if geometry_on:
                    fg_share = None
                    if mask_on:
                        fg_share = out.features[..., -1]
                        out.features = out.features[..., :-1] if coupled else None
                    geo, geo_stats = self.geometry_terms(out, fg_share, targets[idx])
                    loss = loss + geo / batch
                if owners_on:
                    own = self.owner_term(cloud, camera, targets[idx]["ids"], onehot)
                    if own is not None:
                        loss = loss + own / batch
                        owner_value = float(own.detach()) / self.cfg.lambda_owner
                if body_on:
                    body = self.body_term(cloud, camera, targets[idx], widths[idx], members)
                    if body is not None:
                        loss = loss + body / batch
                        body_value = float(body.detach()) / self.cfg.lambda_body
                if cloud.features is not None and id_map is not None:
                    id_term = self.identity_term(cloud, camera, id_map, out)
                    if id_term is not None:
                        loss = loss + id_term / batch
                        id_loss_value = (float(id_term.detach())
                                         / max(self.cfg.lambda_identity, 1e-12))
                outs.append(out)
                l1s.append(l1)
            l1 = float(np.mean(l1s))
            if (self.scene.has_identity and self.cfg.lambda_identity_3d > 0
                    and step % max(self.cfg.identity_3d_every, 1) == 0):
                id3d = self.identity_consistency_loss(gen3d)
                loss = loss + self.cfg.lambda_identity_3d * id3d
                id3d_value = float(id3d.detach())

            loss.backward()
            for out in outs:
                self.fg_ctrl.after_backward(out, (0, n_fg), grad_scale=batch)
                if self.bg_ctrl is not None:
                    self.bg_ctrl.after_backward(out, (n_fg, n_fg + n_bg), grad_scale=batch)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.fg_ctrl.after_step(step)
            if self.bg_ctrl is not None:
                self.bg_ctrl.after_step(step)
            if self._identity_reset_due(step):
                self.reset_identity(step)
            if drift_on and (on_drift is not None or pruning) and (
                    step % self.cfg.owner_drift_every == 0 or step == self.cfg.iters):
                counts = self.owner_evidence(drift_views)
                record = {"step": step, **self.owner_drift(counts)}
                drop = torch.zeros(counts.shape[0], dtype=torch.bool, device=counts.device)
                if ghosts_on:
                    found = self.ghosts(counts)
                    record["ghosts"] = int(found.sum())
                    drop |= found
                if in_body_on:
                    found = self.background_in_bodies(train_views, targets)
                    record["in_body"] = int(found.sum())
                    drop |= found
                # Not at the last check: a pruned Gaussian may paint part of an entity's
                # render, and the steps after a prune let the entity's own Gaussians fill in.
                if pruning and step < self.cfg.iters:
                    record["pruned"] = self.prune(drop)
                if on_drift is not None:
                    on_drift(record)

            if on_log is not None and (step % self.cfg.log_every == 0 or step == 1):
                with torch.no_grad():
                    psnr = image_psnr(out.rgb.detach().clamp(0, 1), gt)
                record = {"step": step, "loss": float(loss.detach()), "l1": l1, "psnr": psnr,
                          "num_gaussians": self.scene.num_gaussians, **geo_stats}
                if owners_on:
                    record["owner_loss"] = owner_value
                if ratio_on:
                    record["scale_ratio_loss"] = ratio_value
                if body_on:
                    record["body_loss"] = body_value
                if self.scene.has_identity:
                    record["identity_loss"] = id_loss_value
                    record["identity_3d_loss"] = id3d_value
                on_log(record)
            if tracer is not None and (step == 1 or step % self.cfg.identity_trace_every == 0
                                       or step == self.cfg.iters):
                on_trace(tracer.record(self, step))
            if on_snapshot is not None and step in snapshots:
                on_snapshot(step, self.scene)

        return self.scene

    @torch.no_grad()
    def render_at(self, camera: RenderCamera, time_id: int) -> torch.Tensor:
        cloud = self.scene.cloud_at(time_id)
        out = self.rasterizer.rasterize(
            cloud.means, cloud.quats, cloud.scales, cloud.opacities, cloud.colors, camera
        )
        return out.rgb.clamp(0, 1)
