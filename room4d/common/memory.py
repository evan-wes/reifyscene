"""Helpers for releasing Python / torch memory between pipeline stages."""

from __future__ import annotations

import ctypes
import gc
from pathlib import Path


def _read_proc_status_memory() -> dict[str, int]:
    """Return current-process RSS / swap figures in bytes when available."""
    status_path = Path('/proc/self/status')
    values: dict[str, int] = {}
    if not status_path.exists():
        return values
    for line in status_path.read_text(encoding='utf-8').splitlines():
        if ':' not in line:
            continue
        key, raw_value = line.split(':', 1)
        if key not in {'VmRSS', 'VmSwap'}:
            continue
        parts = raw_value.strip().split()
        if not parts:
            continue
        try:
            values[key] = int(parts[0]) * 1024
        except ValueError:
            continue
    return values


def process_memory_snapshot() -> dict[str, int | None]:
    """Best-effort snapshot of current process RAM/swap/GPU allocator state."""
    proc = _read_proc_status_memory()
    snapshot: dict[str, int | None] = {
        'rss_bytes': proc.get('VmRSS'),
        'swap_bytes': proc.get('VmSwap'),
        'cuda_allocated_bytes': None,
        'cuda_reserved_bytes': None,
    }
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            snapshot['cuda_allocated_bytes'] = int(torch.cuda.memory_allocated(device))
            snapshot['cuda_reserved_bytes'] = int(torch.cuda.memory_reserved(device))
    except Exception:
        pass
    return snapshot


def format_memory_snapshot(snapshot: dict[str, int | None]) -> str:
    def _fmt(value: int | None) -> str:
        if value is None:
            return 'n/a'
        return f'{value / (1024.0 ** 3):.2f} GiB'

    return (
        f"rss={_fmt(snapshot.get('rss_bytes'))}, "
        f"swap={_fmt(snapshot.get('swap_bytes'))}, "
        f"cuda_allocated={_fmt(snapshot.get('cuda_allocated_bytes'))}, "
        f"cuda_reserved={_fmt(snapshot.get('cuda_reserved_bytes'))}"
    )


def release_memory() -> None:
    """Best-effort memory release for CPU/GPU workloads."""
    gc.collect()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    try:
        libc = ctypes.CDLL('libc.so.6')
        malloc_trim = getattr(libc, 'malloc_trim', None)
        if malloc_trim is not None:
            malloc_trim(0)
    except Exception:
        pass
