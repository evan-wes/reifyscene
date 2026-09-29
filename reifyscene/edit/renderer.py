"""Render a trained checkpoint, and edits of it, from its own training cameras.

Shared by the removal gate (``scripts/gate_identity_removal.py``), ``reifyscene edit`` and the
web app, so all three see the same pixels for the same scene.
"""

from __future__ import annotations

import numpy as np


class CheckpointRenderer:
    """Renders a bundle (and edits of it) at training cameras, caching the unedited view."""

    def __init__(self, bundle: dict, device: str) -> None:
        from reifyscene.render.eval.checkpoint_views import replay_size
        from reifyscene.render.gaussian4d.motion_init import bundle_to_scene
        from reifyscene.render.gaussian4d.rasterizer import build_rasterizer

        self.bundle, self.device = bundle, device
        self.rasterizer = build_rasterizer({"device": device})
        self.scene, cams = bundle_to_scene(bundle, device=device)
        self.cams = {int(c.time_id): c for c in cams.get("train") or []}
        self.size = replay_size(np.asarray(bundle["train_Ks"])[0])
        names = [str(n) for n in np.asarray(bundle["train_frame_names"])]
        self.time_of = {int(n.rsplit(":", 1)[-1]): int(t)
                        for n, t in zip(names, np.asarray(bundle["train_time_ids"]), strict=True)}
        self._before: dict[int, np.ndarray] = {}

    def _render(self, scene, frame: int) -> np.ndarray:
        import torch

        from reifyscene.render.eval.checkpoint_views import _render, _to_u8

        t = self.time_of[frame]
        rc = self.cams[t].render_camera(*self.size)
        with torch.no_grad():
            return _to_u8(_render(self.rasterizer, scene, rc, t))

    def render_alpha(self, scene, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """(rgb uint8, alpha float) of ``scene`` at the training camera for ``frame``."""
        import torch

        from reifyscene.render.eval.checkpoint_views import _to_u8

        t = self.time_of[frame]
        rc = self.cams[t].render_camera(*self.size)
        with torch.no_grad():
            c = scene.cloud_at(t)
            out = self.rasterizer.rasterize(c.means, c.quats, c.scales, c.opacities, c.colors, rc)
        return _to_u8(out.rgb.clamp(0, 1)), out.alpha.clamp(0, 1).cpu().numpy()

    def taken(self, fg_idx, bg_idx, frames: list[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Render ONLY the given Gaussians: what a removal took, on black."""
        from reifyscene.graph.renderable_handle import drop_gaussians
        from reifyscene.render.gaussian4d.motion_init import bundle_to_scene

        h, w = self.size[1], self.size[0]
        blank = {f: (np.zeros((h, w, 3), np.uint8), np.zeros((h, w), np.float32)) for f in frames}
        if len(fg_idx) + len(bg_idx) == 0:
            return blank
        n_fg = int(np.asarray(self.bundle["fg_means"]).shape[0])
        n_bg = int(np.asarray(self.bundle["bg_means"]).shape[0]) if "bg_means" in self.bundle else 0
        only = drop_gaussians(self.bundle, np.setdiff1d(np.arange(n_fg), fg_idx),
                              bg_indices=np.setdiff1d(np.arange(n_bg), bg_idx))
        scene, _ = bundle_to_scene(only, device=self.device)
        return {f: self.render_alpha(scene, f) for f in frames}

    def share(self, fg_idx, bg_idx, frames: list[int]) -> dict[int, np.ndarray]:
        """Per pixel, the composited weight the given Gaussians contribute to the full render."""
        import torch

        n_fg = self.scene.fg.num_gaussians
        ind = torch.zeros((self.scene.num_gaussians, 1), device=self.device)
        ind[torch.as_tensor(np.asarray(fg_idx, dtype=np.int64), device=self.device)] = 1.0
        ind[n_fg + torch.as_tensor(np.asarray(bg_idx, dtype=np.int64), device=self.device)] = 1.0
        out: dict[int, np.ndarray] = {}
        for f in frames:
            t = self.time_of[f]
            rc = self.cams[t].render_camera(*self.size)
            with torch.no_grad():
                c = self.scene.cloud_at(t)
                r = self.rasterizer.rasterize(c.means, c.quats, c.scales, c.opacities, c.colors,
                                              rc, features=ind)
            out[f] = r.features[..., 0].clamp(0, 1).cpu().numpy()
        return out

    def before(self, frame: int) -> np.ndarray:
        if frame not in self._before:
            self._before[frame] = self._render(self.scene, frame)
        return self._before[frame]

    def after(self, edited: dict, frames: list[int]) -> dict[int, np.ndarray]:
        from reifyscene.render.gaussian4d.motion_init import bundle_to_scene

        scene, _ = bundle_to_scene(edited, device=self.device)
        return {f: self._render(scene, f) for f in frames}

    @property
    def frames(self) -> list[int]:
        """Source frame indices this checkpoint has a training camera for."""
        return sorted(self.time_of)


def outline(mask: np.ndarray, width: int = 2) -> np.ndarray:
    """RGBA image: the mask's boundary in red, transparent elsewhere."""
    inner = mask.copy()
    for _ in range(width):
        e = inner.copy()
        e[1:] &= inner[:-1]
        e[:-1] &= inner[1:]
        e[:, 1:] &= inner[:, :-1]
        e[:, :-1] &= inner[:, 1:]
        inner = e
    edge = mask & ~inner
    rgba = np.zeros((*mask.shape, 4), np.uint8)
    rgba[edge] = (255, 64, 64, 255)
    return rgba
