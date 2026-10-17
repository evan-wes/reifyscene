"""Render backend registry."""

from __future__ import annotations

from reifyscene.render.gaussian4d.render_backend import GaussianRenderBackend


_BACKENDS = {"gaussian4d": GaussianRenderBackend}


def build_renderer(cfg: dict | None = None):
    cfg = cfg or {}
    backend_name = str(cfg.get("backend_name", "gaussian4d")).lower()
    backend_cls = _BACKENDS.get(backend_name)
    if backend_cls is None:
        raise ValueError(
            f"Unknown render backend '{backend_name}'. Supported backends: {sorted(_BACKENDS)}"
        )
    return backend_cls(cfg)
