"""Helpers for releasing Python / torch memory between pipeline stages."""

from __future__ import annotations

import gc


def release_memory() -> None:
    """Best-effort memory release for CPU/GPU workloads."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
