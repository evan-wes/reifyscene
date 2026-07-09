"""Render backend registry."""

from __future__ import annotations

from room4d.render.gaussian4d.render_backend import GaussianRenderBackend


# "gaussian_layered" and "offline_high_quality" are legacy names kept so older
# run configs keep resolving; all aliases map to the one gsplat-backed renderer.
_BACKENDS = {
    "gaussian": GaussianRenderBackend,
    "gaussian_layered": GaussianRenderBackend,
    "gaussian4d": GaussianRenderBackend,
    "offline_high_quality": GaussianRenderBackend,
}


def build_renderer(cfg: dict | None = None):
    cfg = cfg or {}
    backend_name = str(cfg.get("backend_name", "gaussian4d")).lower()
    backend_cls = _BACKENDS.get(backend_name)
    if backend_cls is None:
        raise ValueError(
            f"Unknown render backend '{backend_name}'. Supported backends: {sorted(_BACKENDS)}"
        )
    return backend_cls(cfg)
