"""Artifact for the coarse geometry-free ``dyn_mask`` stage (FP6b step 3, design §4A.1).

``dyn_mask`` sits between ``sync`` and ``geometry`` and emits a **coarse per-frame dynamic
mask** (1 = moving/dynamic) whose sole job is to protect the monocular SfM pose rung from
dynamic-content correspondences. It is deliberately **geometry-free** — it consumes only raw
frames, never ``segment``/geometry — which is what breaks the cycle
``geometry←poses←COLMAP←masks←segment←…←geometry``. Per the FP6b spike, masking is a
**size-gated robustness** help (tightens large-dynamic scenes, neutral on small ones), so the
pose rung applies these masks *gated on dynamic-region size*, not unconditionally.

Masks persist as compressed ``.npz`` under ``masks/`` (mirroring ``InstanceArtifact``); only a
light index + per-frame dynamic fractions live in ``artifact.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from reifyscene.common.backend_ladder import CapabilityManifest
from reifyscene.common.frame_keys import frame_key as make_frame_key
from reifyscene.common.io import read_json, require_file, write_json
from reifyscene.common.paths import ensure_dir
from reifyscene.common.types import BaseArtifact, FrameRef


@dataclass
class DynMaskArtifact(BaseArtifact):
    """Per-frame coarse dynamic masks + dynamic fractions (1 = dynamic)."""

    backend_name: str = "all-static"
    selected_frames: list[FrameRef] = field(default_factory=list)
    masks_by_frame: dict[str, np.ndarray] = field(default_factory=dict)
    dynamic_fraction_by_frame: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # --- keys / manifest ----------------------------------------------------
    @staticmethod
    def _frame_key(frame: FrameRef) -> str:
        return make_frame_key(frame.stream_id, frame.frame_idx)

    @staticmethod
    def _sanitize_key(key: str) -> str:
        return key.replace(":", "_").replace("/", "_")

    def attach_manifest(self, manifest: CapabilityManifest) -> None:
        self.metadata["capability"] = manifest.to_dict()

    @property
    def capability(self) -> Optional[CapabilityManifest]:
        raw = self.metadata.get("capability")
        return CapabilityManifest.from_dict(raw) if raw else None

    # --- accessors ----------------------------------------------------------
    def mask_for(self, frame_key: str) -> np.ndarray | None:
        """Return the ``HxW`` uint8 dynamic mask for a frame key, or ``None``."""
        m = self.masks_by_frame.get(frame_key)
        return None if m is None else np.asarray(m)

    def aggregate_dynamic_fraction(self) -> float:
        """Median per-frame dynamic fraction — the signal the pose rung size-gates on.

        Median (not mean) so a few transiently-large frames don't trip the gate; 0.0 when
        there are no frames (all-static floor).
        """
        fracs = list(self.dynamic_fraction_by_frame.values())
        return float(np.median(fracs)) if fracs else 0.0

    # --- persistence (masks → npz + index, rest → json) ---------------------
    def save(self) -> None:
        ensure_dir(self.save_dir)
        masks_dir = ensure_dir(self.save_dir / "masks")
        mask_index: dict[str, str] = {}
        for frame_key, mask in self.masks_by_frame.items():
            filename = f"{self._sanitize_key(frame_key)}.npz"
            np.savez_compressed(masks_dir / filename, mask=np.asarray(mask, dtype=np.uint8))
            mask_index[frame_key] = filename
        payload = {
            "run_id": self.run_id,
            "stage_name": self.stage_name,
            "save_dir": str(self.save_dir),
            "metadata": dict(self.metadata),
            "backend_name": self.backend_name,
            "selected_frames": [
                {"stream_id": f.stream_id, "frame_idx": int(f.frame_idx), "timestamp_sec": float(f.timestamp_sec)}
                for f in self.selected_frames
            ],
            "mask_index": mask_index,
            "dynamic_fraction_by_frame": {k: float(v) for k, v in self.dynamic_fraction_by_frame.items()},
            "notes": list(self.notes),
        }
        write_json(self.artifact_path(), payload)

    @classmethod
    def load(cls, save_dir: Path) -> "DynMaskArtifact":
        raw = read_json(save_dir / "artifact.json")
        masks: dict[str, np.ndarray] = {}
        for frame_key, filename in raw["mask_index"].items():
            path = require_file(save_dir / "masks" / filename, "dyn_mask mask")
            with np.load(path, allow_pickle=True) as npz:
                masks[frame_key] = np.asarray(npz["mask"], dtype=np.uint8)
        return cls(
            run_id=raw["run_id"],
            stage_name=raw["stage_name"],
            save_dir=Path(save_dir),
            metadata=raw["metadata"],
            backend_name=raw["backend_name"],
            selected_frames=[
                FrameRef(stream_id=str(f["stream_id"]), frame_idx=int(f["frame_idx"]), timestamp_sec=float(f["timestamp_sec"]))
                for f in raw["selected_frames"]
            ],
            masks_by_frame=masks,
            dynamic_fraction_by_frame={k: float(v) for k, v in raw["dynamic_fraction_by_frame"].items()},
            notes=raw["notes"],
        )

    def summary(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "frames": len(self.masks_by_frame),
            "aggregate_dynamic_fraction": self.aggregate_dynamic_fraction(),
            "notes": list(self.notes),
        }
