"""Lightweight system inspection and resource profile helpers."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _read_proc_meminfo() -> dict[str, int]:
    info: dict[str, int] = {}
    meminfo_path = Path('/proc/meminfo')
    if not meminfo_path.exists():
        return info
    for line in meminfo_path.read_text(encoding='utf-8').splitlines():
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        parts = value.strip().split()
        if not parts:
            continue
        try:
            info[key] = int(parts[0]) * 1024  # values are in kB on Linux
        except ValueError:
            continue
    return info


def _query_nvidia_smi() -> list[dict[str, Any]]:
    if shutil.which('nvidia-smi') is None:
        return []
    cmd = [
        'nvidia-smi',
        '--query-gpu=index,name,memory.total,memory.free,driver_version',
        '--format=csv,noheader,nounits',
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except Exception:
        return []
    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) != 5:
            continue
        try:
            index = int(parts[0])
            total_mb = int(parts[2])
            free_mb = int(parts[3])
        except ValueError:
            continue
        gpus.append({
            'index': index,
            'name': parts[1],
            'memory_total_mb': total_mb,
            'memory_free_mb': free_mb,
            'driver_version': parts[4],
        })
    return gpus


def _bytes_to_gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 ** 3)


def collect_system_report() -> dict[str, Any]:
    meminfo = _read_proc_meminfo()
    total_mem = meminfo.get('MemTotal')
    avail_mem = meminfo.get('MemAvailable')
    total_swap = meminfo.get('SwapTotal')
    free_swap = meminfo.get('SwapFree')
    report = {
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'python': platform.python_version(),
        },
        'cpu': {
            'logical_cores': os.cpu_count(),
        },
        'memory': {
            'total_bytes': total_mem,
            'available_bytes': avail_mem,
            'total_gib': _bytes_to_gib(total_mem),
            'available_gib': _bytes_to_gib(avail_mem),
            'swap_total_bytes': total_swap,
            'swap_free_bytes': free_swap,
            'swap_total_gib': _bytes_to_gib(total_swap),
            'swap_free_gib': _bytes_to_gib(free_swap),
        },
        'gpus': _query_nvidia_smi(),
    }
    return report


def recommend_tracking_profile(report: dict[str, Any], *, device: str | None = None) -> dict[str, Any]:
    device = str(device or 'cpu')
    memory = report.get('memory', {})
    available_gib = float(memory.get('available_gib') or 0.0)
    gpus = report.get('gpus', []) or []
    max_free_gpu_gib = max((gpu.get('memory_free_mb', 0) / 1024.0 for gpu in gpus), default=0.0)

    if device.startswith('cuda'):
        if max_free_gpu_gib < 5.0:
            profile = 'lowmem'
        elif max_free_gpu_gib < 8.0:
            profile = 'balanced'
        else:
            profile = 'full'
    else:
        if available_gib < 8.0:
            profile = 'lowmem'
        elif available_gib < 16.0:
            profile = 'balanced'
        else:
            profile = 'full'

    presets = {
        'lowmem': {
            'resize_max_dim': 256,
            'frame_stride': 2,
            'max_frames': 24,
            'window_size': 12,
            'window_overlap': 4,
            'cpu_threads': 4,
        },
        'balanced': {
            'resize_max_dim': 384,
            'frame_stride': 2,
            'max_frames': 48,
            'window_size': 24,
            'window_overlap': 8,
            'cpu_threads': 6,
        },
        'full': {
            'resize_max_dim': 512,
            'frame_stride': 1,
            'max_frames': 96,
            'window_size': 48,
            'window_overlap': 12,
            'cpu_threads': 8,
        },
    }
    rec = dict(presets[profile])
    rec['profile'] = profile
    rec['device'] = device
    rec['max_free_gpu_gib'] = max_free_gpu_gib
    rec['available_ram_gib'] = available_gib
    return rec


def recommend_vggt_long_profile(
    report: dict[str, Any],
    *,
    device: str | None = None,
    num_frames: int | None = None,
    source_max_dim: int | None = None,
) -> dict[str, Any]:
    """Return a resource-aware starting point for VGGT-Long auto-tuning.

    The goal is not to predict the exact safe setting up front. Instead we use
    the preflight report to choose an aggressive first attempt and conservative
    lower bounds so the runtime retry loop can back off systematically.
    """
    device = str(device or 'cpu').lower()
    memory = report.get('memory', {})
    available_gib = float(memory.get('available_gib') or 0.0)
    gpus = report.get('gpus', []) or []
    max_free_gpu_gib = max((gpu.get('memory_free_mb', 0) / 1024.0 for gpu in gpus), default=0.0)
    source_max_dim = int(source_max_dim) if source_max_dim else None
    num_frames = int(num_frames) if num_frames else None

    def _clip_resize(value: int | None) -> int | None:
        if value in (None, 0):
            return None
        value = int(value)
        if source_max_dim is not None and source_max_dim > 0:
            value = min(value, source_max_dim)
        return max(1, value)

    def _dedupe_resize(values: list[int | None], floor_value: int | None) -> list[int | None]:
        floor_norm = _clip_resize(floor_value)
        out: list[int | None] = []
        for value in values:
            value_norm = _clip_resize(value)
            if value_norm is None:
                continue
            if floor_norm is not None:
                value_norm = max(floor_norm, value_norm)
            if value_norm not in out:
                out.append(value_norm)
        if floor_norm is not None and floor_norm not in out:
            out.append(floor_norm)
        return out

    if device.startswith('cuda'):
        if max_free_gpu_gib >= 24.0:
            profile = 'full'
            initial_chunk_size = 48
            initial_resize_max_dim = source_max_dim
            min_resize_max_dim = 768
            resize_candidates = _dedupe_resize(
                [initial_resize_max_dim, 1600, 1440, 1280, 1152, 1024, 960, 896, 832, 768],
                min_resize_max_dim,
            )
        elif max_free_gpu_gib >= 16.0:
            profile = 'balanced'
            initial_chunk_size = 32
            initial_resize_max_dim = _clip_resize(1280 if source_max_dim is not None else 1280)
            min_resize_max_dim = 640
            resize_candidates = _dedupe_resize(
                [initial_resize_max_dim, 1152, 1024, 960, 896, 832, 768, 704, 640],
                min_resize_max_dim,
            )
        elif max_free_gpu_gib >= 12.0:
            profile = 'balanced'
            initial_chunk_size = 24
            initial_resize_max_dim = _clip_resize(960 if source_max_dim is not None else 960)
            min_resize_max_dim = 512
            resize_candidates = _dedupe_resize(
                [initial_resize_max_dim, 896, 832, 768, 704, 640, 576, 512],
                min_resize_max_dim,
            )
        elif max_free_gpu_gib >= 8.0:
            profile = 'lowmem'
            initial_chunk_size = 12
            initial_resize_max_dim = _clip_resize(640 if source_max_dim is not None else 640)
            min_resize_max_dim = 224
            resize_candidates = _dedupe_resize(
                [initial_resize_max_dim, 576, 512, 448, 384, 320, 288, 256, 224],
                min_resize_max_dim,
            )
        elif max_free_gpu_gib >= 6.0:
            profile = 'lowmem'
            initial_chunk_size = 8
            initial_resize_max_dim = _clip_resize(512 if source_max_dim is not None else 512)
            min_resize_max_dim = 192
            resize_candidates = _dedupe_resize(
                [initial_resize_max_dim, 448, 384, 320, 288, 256, 224, 192],
                min_resize_max_dim,
            )
        else:
            profile = 'lowmem'
            initial_chunk_size = 4
            initial_resize_max_dim = _clip_resize(384 if source_max_dim is not None else 384)
            min_resize_max_dim = 160
            resize_candidates = _dedupe_resize(
                [initial_resize_max_dim, 320, 288, 256, 224, 192, 160],
                min_resize_max_dim,
            )
    else:
        if available_gib >= 32.0:
            profile = 'balanced'
            initial_chunk_size = 16
            initial_resize_max_dim = _clip_resize(960 if source_max_dim is not None else 960)
            min_resize_max_dim = 512
            resize_candidates = _dedupe_resize(
                [initial_resize_max_dim, 896, 832, 768, 704, 640, 576, 512],
                min_resize_max_dim,
            )
        elif available_gib >= 16.0:
            profile = 'balanced'
            initial_chunk_size = 12
            initial_resize_max_dim = _clip_resize(768 if source_max_dim is not None else 768)
            min_resize_max_dim = 384
            resize_candidates = _dedupe_resize(
                [initial_resize_max_dim, 704, 640, 576, 512, 448, 384],
                min_resize_max_dim,
            )
        else:
            profile = 'lowmem'
            initial_chunk_size = 8
            initial_resize_max_dim = _clip_resize(640 if source_max_dim is not None else 640)
            min_resize_max_dim = 320
            resize_candidates = _dedupe_resize(
                [initial_resize_max_dim, 576, 512, 448, 384, 320],
                min_resize_max_dim,
            )

    if source_max_dim is not None:
        if source_max_dim >= 3000:
            initial_chunk_size = max(2, initial_chunk_size // 2)
        elif source_max_dim >= 2000:
            initial_chunk_size = max(2, int(round(initial_chunk_size * 0.625)))
        elif source_max_dim >= 1400:
            initial_chunk_size = max(2, int(round(initial_chunk_size * 0.75)))
        initial_resize_max_dim = _clip_resize(initial_resize_max_dim)
        min_resize_max_dim = int(_clip_resize(min_resize_max_dim) or min_resize_max_dim)
        resize_candidates = _dedupe_resize(resize_candidates, min_resize_max_dim)

    if num_frames is not None and num_frames > 0:
        initial_chunk_size = max(1, min(initial_chunk_size, num_frames))
        min_chunk_size = 1 if num_frames <= 1 else min(2, num_frames)
    else:
        min_chunk_size = 2

    if num_frames is not None and initial_chunk_size >= num_frames:
        initial_overlap = 0
    else:
        initial_overlap = max(1, min(8, initial_chunk_size // 4))

    return {
        'profile': profile,
        'device': device,
        'max_free_gpu_gib': max_free_gpu_gib,
        'available_ram_gib': available_gib,
        'initial_chunk_size': initial_chunk_size,
        'initial_overlap': initial_overlap,
        'min_chunk_size': min_chunk_size,
        'initial_resize_max_dim': initial_resize_max_dim,
        'min_resize_max_dim': min_resize_max_dim,
        'resize_candidates': resize_candidates,
        'source_max_dim': source_max_dim,
        'num_frames': num_frames,
    }


def save_system_report(path: str | Path, report: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding='utf-8')
