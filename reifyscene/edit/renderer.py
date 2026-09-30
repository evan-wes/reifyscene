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
        self._pivot: dict[int, float] = {}

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

    # -- free viewpoint -------------------------------------------------------------------------

    def pivot_depth(self, frame: int) -> float:
        """Median depth, in front of the frame's camera, of the scene at that moment: the orbit
        pivot sits this far along the camera's axis, so orbiting turns around what it looks at."""
        if frame not in self._pivot:
            import torch

            t = self.time_of[frame]
            w2c = torch.as_tensor(np.asarray(self.cams[t].w2c, np.float32), device=self.device)
            with torch.no_grad():
                means = self.scene.cloud_at(t).means
                z = means @ w2c[2, :3] + w2c[2, 3]
                z = z[z > 0]
                self._pivot[frame] = float(z.median()) if z.numel() else 1.0
        return self._pivot[frame]

    def orbit_camera(self, frame: int, *, yaw: float = 0.0, pitch: float = 0.0,
                     zoom: float = 1.0, pan_x: float = 0.0, pan_y: float = 0.0,
                     width: int | None = None):
        """The frame's training camera, orbited about its pivot: see :func:`orbit_w2c`."""
        from reifyscene.render.gaussian4d.rasterizer import RenderCamera

        t = self.time_of[frame]
        cam = self.cams[t]
        w2c = orbit_w2c(np.asarray(cam.w2c, np.float64), self.pivot_depth(frame),
                        yaw=yaw, pitch=pitch, zoom=zoom, pan_x=pan_x, pan_y=pan_y)
        w, h = self.size
        K = np.asarray(cam.K, np.float32).copy()
        if width and width < w:                      # a smaller preview while the view moves
            s = width / w
            K[:2] *= s
            w, h = int(round(w * s)), int(round(h * s))
        return RenderCamera(viewmat=w2c.astype(np.float32), K=K, width=int(w), height=int(h))

    def render_view(self, scene, frame: int, camera) -> np.ndarray:
        """``scene`` at ``frame``'s moment, seen from any camera."""
        import torch

        from reifyscene.render.eval.checkpoint_views import _to_u8

        t = self.time_of[frame]
        with torch.no_grad():
            c = scene.cloud_at(t)
            out = self.rasterizer.rasterize(c.means, c.quats, c.scales, c.opacities, c.colors,
                                            camera)
        return _to_u8(out.rgb.clamp(0, 1))

    @property
    def frames(self) -> list[int]:
        """Source frame indices this checkpoint has a training camera for."""
        return sorted(self.time_of)


def orbit_w2c(w2c: np.ndarray, depth: float, *, yaw: float = 0.0, pitch: float = 0.0,
              zoom: float = 1.0, pan_x: float = 0.0, pan_y: float = 0.0) -> np.ndarray:
    """World-to-camera of a camera orbited about a pivot ``depth`` in front of ``w2c``.

    Everything is in the original camera's frame (OpenCV: +x right, +y down, +z forward):
    ``yaw`` turns about its vertical axis and ``pitch`` about its horizontal one (degrees);
    ``zoom`` scales the distance to the pivot; ``pan_x``/``pan_y`` shift the pivot sideways
    and up/down in the orbited view, in units of ``depth``. All zero (zoom 1) is ``w2c``.
    """
    a, b = np.radians(yaw), np.radians(pitch)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    rot = ry @ rx
    target = np.array([0.0, 0.0, depth]) + rot @ np.array([pan_x * depth, pan_y * depth, 0.0])
    eye = target - rot @ np.array([0.0, 0.0, depth * zoom])
    local = np.eye(4)                               # orbited camera -> original camera
    local[:3, :3], local[:3, 3] = rot, eye
    return np.linalg.inv(np.linalg.inv(w2c) @ local)


def outline(mask: np.ndarray, width: int = 2,
            color: tuple[int, int, int] = (255, 64, 64)) -> np.ndarray:
    """RGBA image: the mask's boundary in ``color`` (red by default), transparent elsewhere."""
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
    rgba[edge] = (*color, 255)
    return rgba
