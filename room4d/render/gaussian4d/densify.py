"""Adaptive density control for the static Gaussian trainer.

Our own clone / split / prune / opacity-reset controller, operating on the
``Rasterizer`` contract's densification telemetry (screen-space ``means2d``
gradient + ``radii``) rather than on gsplat internals — so densification stays
backend-agnostic, mirroring the swappable-rasterizer design. The thresholds and
cadence are ported from gsplat's validated ``DefaultStrategy`` (3DGS recipe) and
are all config-toggleable via :class:`DensifyConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from room4d.render.gaussian4d.gaussian_field import PARAM_NAMES, GaussianField, logit


# Schedule fields that scale with the total iteration budget. Anything not set
# explicitly in a config is derived from `iters` via `densify_schedule`.
SCHEDULE_FIELDS = ("start_iter", "refine_every", "reset_opacity_every", "stop_iter", "pause_refine_after_reset")


def densify_schedule(max_iters: int, *, num_frames: int, refine_every: int = 100) -> dict[str, int]:
    """SoM-shaped densification schedule derived from the total iteration budget.

    Anchored on Shape-of-Motion's 5000-step recipe (``flow3d`` OptimizerConfig) as
    fractions of the total, so the opacity reset always lands *with recovery and a
    polish phase after it*: ``reset(~0.6·T) < control stop(~0.8·T) < T``. This makes
    the terminal-reset collapse (a reset on the final iteration, never recovered)
    structurally impossible for any ``max_iters``. At ``max_iters=5000`` it returns
    SoM's exact numbers ``{200, 100, 3000, 4000}`` (+ a post-reset densify gap of
    ``num_frames``, as SoM does).
    """
    r = max(1, int(refine_every))
    mult = lambda frac: max(r, int(round(frac * max_iters / r)) * r)
    return {
        "start_iter": mult(0.04),          # warmup before densifying (SoM 200/5000)
        "refine_every": r,                 # control cadence (SoM control_every=100)
        "reset_opacity_every": mult(0.60), # last reset at ~0.6·T (SoM 3000/5000)
        "stop_iter": mult(0.80),           # control/densify ends at ~0.8·T (SoM 4000/5000)
        "pause_refine_after_reset": int(num_frames),  # skip refine right after a reset
    }


@dataclass
class DensifyConfig:
    enabled: bool = True
    start_iter: int = 500
    stop_iter: int = 15000
    refine_every: int = 100
    reset_opacity_every: int = 3000
    pause_refine_after_reset: int = 0
    grad2d_threshold: float = 0.0002
    grow_scale3d: float = 0.01
    prune_opacity: float = 0.005
    prune_scale3d: float = 0.1
    split_count: int = 2
    split_scale_divider: float = 1.6
    reset_opacity_value: float = 0.01
    absgrad: bool = False
    scene_extent: float | None = None  # None -> auto from the seed cloud
    max_gaussians: int | None = None  # safety cap; growth pauses at/above it

    @classmethod
    def from_cfg(
        cls,
        cfg: dict | None,
        *,
        max_iters: int | None = None,
        num_frames: int | None = None,
    ) -> "DensifyConfig":
        """Build a config; when ``max_iters`` is given, derive any schedule field
        the config doesn't set explicitly via :func:`densify_schedule`.

        Explicit config values always win over derived ones. With ``max_iters``
        set, the resulting ``stop_iter`` must be ``< max_iters`` (a reset/densify
        window that runs to the final iteration leaves no recovery) — otherwise a
        ``ValueError`` is raised so the misconfiguration can't pass silently.
        """
        cfg = dict(cfg or {})
        if max_iters is not None:
            sched = densify_schedule(
                max_iters,
                num_frames=int(num_frames) if num_frames is not None else 1,
                refine_every=int(cfg.get("refine_every", 100)),
            )
            for key in SCHEDULE_FIELDS:
                cfg.setdefault(key, sched[key])
        fields = {f for f in cls.__dataclass_fields__}
        inst = cls(**{k: v for k, v in cfg.items() if k in fields})
        if inst.enabled and max_iters is not None and inst.stop_iter >= max_iters:
            raise ValueError(
                f"densify stop_iter ({inst.stop_iter}) must be < iters ({max_iters}); "
                "a densify/reset window running to the final step leaves no recovery."
            )
        return inst


def _quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    q = F.normalize(quats, dim=-1)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)
    return R


def compute_scene_extent(means: torch.Tensor) -> float:
    """Robust scene radius (2x the 90th-percentile distance from the centroid)."""
    if means.shape[0] == 0:
        return 1.0
    centroid = means.mean(dim=0)
    dist = (means - centroid).norm(dim=1)
    extent = float(torch.quantile(dist, 0.9)) * 2.0
    return extent if extent > 1e-6 else 1.0


class DensificationController:
    """Grows/prunes a GaussianField in lockstep with its Adam optimizer state."""

    def __init__(
        self,
        field: GaussianField,
        optimizer: torch.optim.Optimizer,
        cfg: DensifyConfig,
        scene_extent: float | None = None,
        *,
        param_names: tuple[str, ...] = PARAM_NAMES,
        group_prefix: str = "",
    ) -> None:
        self.field = field
        self.optimizer = optimizer
        self.cfg = cfg
        # The field's optimizable params + the optimizer-group-name prefix. The
        # dynamic foreground adds "motion_coefs" and uses "fg."/"bg." group names
        # so one combined optimizer can host both fields plus the motion bases.
        self.param_names = tuple(param_names)
        self.group_prefix = str(group_prefix)
        self.scene_extent = float(
            cfg.scene_extent if cfg.scene_extent is not None
            else (scene_extent if scene_extent is not None else compute_scene_extent(field.means.detach()))
        )
        self._last_reset = -(10**9)
        self._reset_accumulators()

    # -- accumulation ------------------------------------------------------

    def _reset_accumulators(self) -> None:
        n, dev = self.field.num_gaussians, self.field.device
        self.grad2d_accum = torch.zeros(n, device=dev)
        self.count = torch.zeros(n, device=dev)

    @torch.no_grad()
    def after_backward(self, output: Any, gaussian_range: tuple[int, int] | None = None) -> None:
        """Accumulate per-Gaussian screen-space gradient from a RasterOutput.

        ``gaussian_range`` selects this field's slice of a combined-cloud
        telemetry tensor (e.g. (0, n_fg) for foreground, (n_fg, n_fg + n_bg) for
        background); None means the telemetry is this field's alone.
        """
        if not self.cfg.enabled or output.means2d is None:
            return
        grad = getattr(output.means2d, "absgrad", None) if self.cfg.absgrad else output.means2d.grad
        if grad is None:
            return
        grad = grad.detach().clone()
        if grad.dim() == 3:
            grad = grad[0]
        # gsplat's means2d gradient is in [-1, 1] NDC; rescale to pixel units so
        # the ported grad2d_threshold (0.0002) is applied in the same space as
        # gsplat's DefaultStrategy.
        grad[..., 0] *= output.width / 2.0
        grad[..., 1] *= output.height / 2.0
        gnorm = grad.norm(dim=-1)

        radii = output.radii
        if radii is None:
            radii_flat = None
        else:
            radii = radii.detach()
            if radii.dim() == 3:
                radii = radii[0]
            if radii.dim() == 2 and radii.shape[-1] == 2:
                radii = radii.amax(dim=-1)
            elif radii.dim() == 2:
                radii = radii[0]
            radii_flat = radii.reshape(-1)

        if gaussian_range is not None:
            start, end = gaussian_range
            gnorm = gnorm[start:end]
            radii_flat = radii_flat[start:end] if radii_flat is not None else None

        visible = (radii_flat > 0) if radii_flat is not None else (gnorm > 0)
        self.grad2d_accum[visible] += gnorm[visible]
        self.count[visible] += 1

    # -- optimizer-state surgery ------------------------------------------

    def _optimizer_op(self, build_fn) -> None:
        group_to_param = {self.group_prefix + p: p for p in self.param_names}
        for group in self.optimizer.param_groups:
            pname = group_to_param.get(group.get("name"))
            if pname is None:
                continue
            old = group["params"][0]
            state = self.optimizer.state.get(old, {})
            exp_avg = state.get("exp_avg")
            exp_avg_sq = state.get("exp_avg_sq")
            new_tensor, new_ea, new_eas = build_fn(pname, old, exp_avg, exp_avg_sq)
            new_param = nn.Parameter(new_tensor.contiguous().detach().requires_grad_(True))
            if old in self.optimizer.state:
                del self.optimizer.state[old]
            if new_ea is not None:
                self.optimizer.state[new_param] = {
                    "step": state.get("step", torch.tensor(0.0)),
                    "exp_avg": new_ea,
                    "exp_avg_sq": new_eas,
                }
            group["params"][0] = new_param
            self.field.set_param(pname, new_param)

    def _append(self, keep_orig_mask: torch.Tensor, appended: dict[str, torch.Tensor]) -> None:
        def build(name, old, exp_avg, exp_avg_sq):
            kept = old.detach()[keep_orig_mask]
            add = appended[name].to(device=old.device, dtype=old.dtype)
            new_tensor = torch.cat([kept, add], dim=0)
            if exp_avg is None:
                return new_tensor, None, None
            zeros = torch.zeros((add.shape[0], *old.shape[1:]), device=old.device, dtype=old.dtype)
            new_ea = torch.cat([exp_avg[keep_orig_mask], zeros], dim=0)
            new_eas = torch.cat([exp_avg_sq[keep_orig_mask], zeros.clone()], dim=0)
            return new_tensor, new_ea, new_eas

        self._optimizer_op(build)

    def _prune(self, keep_mask: torch.Tensor) -> None:
        def build(name, old, exp_avg, exp_avg_sq):
            new_tensor = old.detach()[keep_mask]
            if exp_avg is None:
                return new_tensor, None, None
            return new_tensor, exp_avg[keep_mask], exp_avg_sq[keep_mask]

        self._optimizer_op(build)

    # -- densification primitives -----------------------------------------

    def _clone_and_split_rows(self, clone_mask: torch.Tensor, split_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        field = self.field
        # Clones are straight copies of every param (standard 3DGS).
        rows: dict[str, list[torch.Tensor]] = {name: [field.get_param(name).detach()[clone_mask]] for name in self.param_names}

        n_split = int(split_mask.sum())
        if n_split > 0:
            k = int(self.cfg.split_count)
            scales = field.get_scales()[split_mask]            # (n, 3)
            quats = field.quats.detach()[split_mask]           # (n, 4)
            means = field.means.detach()[split_mask]           # (n, 3)
            samples = torch.randn((n_split, k, 3), device=field.device) * scales[:, None, :]
            rot = _quat_to_rotmat(quats)                       # (n, 3, 3)
            offsets = torch.einsum("nij,nkj->nki", rot, samples)
            split_means = (means[:, None, :] + offsets).reshape(n_split * k, 3)
            split_log_scales = (
                torch.log((scales / self.cfg.split_scale_divider).clamp_min(1e-8))[:, None, :]
                .repeat(1, k, 1).reshape(n_split * k, 3)
            )
            for name in self.param_names:
                if name == "means":
                    rows[name].append(split_means)
                elif name == "log_scales":
                    rows[name].append(split_log_scales)
                else:
                    # All other params (quats, opacities, colors, motion_coefs, ...)
                    # are inherited from the parent — the standard 3DGS split.
                    parent = field.get_param(name).detach()[split_mask]
                    rows[name].append(parent[:, None].repeat(1, k, *([1] * (parent.dim() - 1))).reshape(n_split * k, *parent.shape[1:]))

        return {name: torch.cat(rows[name], dim=0) for name in self.param_names}

    @torch.no_grad()
    def densify_and_prune(self, step: int) -> dict[str, int]:
        cfg = self.cfg
        ext = self.scene_extent
        grads = self.grad2d_accum / self.count.clamp_min(1.0)
        grads[self.count == 0] = 0.0
        max_scale = self.field.get_scales().amax(dim=-1)

        high_grad = grads >= cfg.grad2d_threshold
        small = max_scale <= cfg.grow_scale3d * ext
        clone_mask = high_grad & small
        split_mask = high_grad & (~small)

        # Safety cap: pause growth (prune still runs) at/above max_gaussians.
        if cfg.max_gaussians is not None and self.field.num_gaussians >= cfg.max_gaussians:
            clone_mask = torch.zeros_like(clone_mask)
            split_mask = torch.zeros_like(split_mask)

        n_clone, n_split = int(clone_mask.sum()), int(split_mask.sum())
        appended = self._clone_and_split_rows(clone_mask, split_mask)
        # Split parents are consumed; clones keep their parents.
        self._append(~split_mask, appended)

        opacities = self.field.get_opacities()
        too_big = self.field.get_scales().amax(dim=-1) > cfg.prune_scale3d * ext
        prune_mask = (opacities < cfg.prune_opacity) | too_big
        n_prune = int(prune_mask.sum())
        self._prune(~prune_mask)

        self._reset_accumulators()
        return {"cloned": n_clone, "split": n_split, "pruned": n_prune, "total": self.field.num_gaussians}

    @torch.no_grad()
    def reset_opacity(self) -> None:
        cap = float(logit(torch.tensor(self.cfg.reset_opacity_value)))
        capped = torch.minimum(self.field.logit_opacities.data, torch.full_like(self.field.logit_opacities.data, cap))
        self.field.logit_opacities.data.copy_(capped)
        for group in self.optimizer.param_groups:
            if group.get("name") == "logit_opacities":
                state = self.optimizer.state.get(group["params"][0])
                if state:
                    state["exp_avg"].zero_()
                    state["exp_avg_sq"].zero_()

    # -- driver -----------------------------------------------------------

    def after_step(self, step: int) -> dict[str, int] | None:
        """Run densification/opacity-reset on schedule (after optimizer.step)."""
        if not self.cfg.enabled:
            return None
        stats = None
        in_window = self.cfg.start_iter < step < self.cfg.stop_iter
        if in_window and step % self.cfg.refine_every == 0 and (step - self._last_reset) > self.cfg.pause_refine_after_reset:
            stats = self.densify_and_prune(step)
        # Opacity reset only inside the densify window: a reset must be followed by
        # densify-prune + recovery training (both gated by `in_window`), else it
        # just strands the scene transparent — the SoM control loop guards it the
        # same way (`global_step < stop_control_steps`).
        if in_window and self.cfg.reset_opacity_every > 0 and step % self.cfg.reset_opacity_every == 0:
            self.reset_opacity()
            self._last_reset = step
        return stats
