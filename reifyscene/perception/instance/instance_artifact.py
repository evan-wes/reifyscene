"""Artifact for the ``segment`` stage: per-frame instance masks + persistent IDs.

The ``InstanceArtifact`` is the **shared mask + identity source** of the redesigned
perception band (design §4A). One backend (SAM 3, or a fallback rung down to
today's Mask R-CNN + IoU association) produces, for every selected frame, a set
of instance **masks** each tagged with a **persistent track id** and the
**concept** that matched it (so ``objects`` / ``humans`` can route human vs
non-human tracklets without re-detecting). Downstream:

- ``reid`` enriches identities across occlusion gaps (`ReidArtifact`);
- ``objects`` / ``humans`` enrich tracklets with semantics / SMPL;
- ``tracking`` seeds its dynamic split from these foreground masks;
- ``scene`` / ``render`` consume the masks + ids as the load-bearing render seed.

Masks are persisted as compressed ``.npz`` (one per instance) with an index in
``artifact.json`` — mirroring :class:`~reifyscene.perception.objects.object_artifact.ObjectArtifact`
to keep the metadata file small. The ladder's
:class:`~reifyscene.common.backend_ladder.CapabilityManifest` is stored under
``metadata['capability']`` so the carried fidelity is observable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from reifyscene.common.backend_ladder import CapabilityManifest
from reifyscene.common.frame_keys import frame_key
from reifyscene.common.io import read_json, require_file, write_json
from reifyscene.common.paths import ensure_dir
from reifyscene.common.types import BaseArtifact, FrameRef


@dataclass
class InstanceObservation:
    """A single instance detected in one frame, with its persistent identity."""

    instance_id: str  # unique per frame-instance (the mask key)
    track_id: str  # persistent identity across frames
    frame_ref: FrameRef
    concept: str = ''  # prompt/concept that matched (e.g. 'person', 'car')
    kind: str = 'unknown'  # 'human' | 'object' | 'unknown' (routing hint)
    bbox_xyxy: Optional[np.ndarray] = None
    confidence: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class InstanceArtifact(BaseArtifact):
    """Per-frame instance masks + persistent track ids for all entities."""

    observations: list[InstanceObservation] = field(default_factory=list)
    mask_arrays: dict[str, np.ndarray] = field(default_factory=dict)  # instance_id -> bool mask
    frame_index: dict[str, list[str]] = field(default_factory=dict)  # frame_key -> [instance_id]
    track_index: dict[str, list[str]] = field(default_factory=dict)  # track_id -> [instance_id]
    notes: list[str] = field(default_factory=list)

    # --- keys / indices -----------------------------------------------------
    @staticmethod
    def _frame_key(frame_ref: FrameRef) -> str:
        return frame_key(frame_ref.stream_id, frame_ref.frame_idx)

    @staticmethod
    def _sanitize_key(key: str) -> str:
        return key.replace(':', '__').replace('/', '_')

    @classmethod
    def _build_frame_index(cls, observations: list[InstanceObservation]) -> dict[str, list[str]]:
        index: dict[str, list[str]] = {}
        for obs in observations:
            index.setdefault(cls._frame_key(obs.frame_ref), []).append(obs.instance_id)
        return index

    @staticmethod
    def _build_track_index(observations: list[InstanceObservation]) -> dict[str, list[str]]:
        index: dict[str, list[str]] = {}
        for obs in observations:
            index.setdefault(obs.track_id, []).append(obs.instance_id)
        return index

    def rebuild_indices(self) -> None:
        self.frame_index = self._build_frame_index(self.observations)
        self.track_index = self._build_track_index(self.observations)

    # --- capability manifest ------------------------------------------------
    def attach_manifest(self, manifest: CapabilityManifest) -> None:
        self.metadata['capability'] = manifest.to_dict()

    @property
    def capability(self) -> Optional[CapabilityManifest]:
        raw = self.metadata.get('capability')
        return CapabilityManifest.from_dict(raw) if raw else None

    # --- record (de)serialization ------------------------------------------
    @staticmethod
    def _obs_to_dict(obs: InstanceObservation) -> dict[str, Any]:
        return {
            'instance_id': obs.instance_id,
            'track_id': obs.track_id,
            'frame_ref': obs.frame_ref.to_dict(),
            'concept': obs.concept,
            'kind': obs.kind,
            'bbox_xyxy': None if obs.bbox_xyxy is None else np.asarray(obs.bbox_xyxy, dtype=np.float32).tolist(),
            'confidence': float(obs.confidence),
            'attributes': dict(obs.attributes),
        }

    @staticmethod
    def _obs_from_dict(raw: dict[str, Any]) -> InstanceObservation:
        fr = raw['frame_ref']
        bbox = raw['bbox_xyxy']
        return InstanceObservation(
            instance_id=str(raw['instance_id']),
            track_id=str(raw['track_id']),
            frame_ref=FrameRef.from_dict(fr),
            concept=str(raw['concept']),
            kind=str(raw['kind']),
            bbox_xyxy=None if bbox is None else np.asarray(bbox, dtype=np.float32),
            confidence=float(raw['confidence']),
            attributes=dict(raw['attributes']),
        )

    # --- persistence (masks → npz + index, records → json) ------------------
    def save(self) -> None:
        ensure_dir(self.save_dir)
        masks_dir = ensure_dir(self.save_dir / 'masks')
        mask_index: dict[str, str] = {}
        for instance_id, mask in self.mask_arrays.items():
            filename = f"{self._sanitize_key(instance_id)}.npz"
            np.savez_compressed(masks_dir / filename, mask=np.asarray(mask, dtype=np.uint8))
            mask_index[instance_id] = filename
        payload = {
            'run_id': self.run_id,
            'stage_name': self.stage_name,
            'metadata': dict(self.metadata),
            'observations': [self._obs_to_dict(o) for o in self.observations],
            'frame_index': self.frame_index or self._build_frame_index(self.observations),
            'track_index': self.track_index or self._build_track_index(self.observations),
            'mask_index': mask_index,
            'notes': list(self.notes),
        }
        write_json(self.artifact_path(), payload)

    @classmethod
    def load(cls, save_dir: Path) -> 'InstanceArtifact':
        raw = read_json(save_dir / cls.FILE)
        mask_arrays: dict[str, np.ndarray] = {}
        for instance_id, filename in raw['mask_index'].items():
            path = require_file(save_dir / 'masks' / filename, "segment mask")
            with np.load(path, allow_pickle=True) as npz:
                mask_arrays[instance_id] = np.asarray(npz['mask'], dtype=bool)
        return cls(
            run_id=raw['run_id'],
            stage_name=raw['stage_name'],
            save_dir=Path(save_dir),
            metadata=raw['metadata'],
            observations=[cls._obs_from_dict(o) for o in raw['observations']],
            mask_arrays=mask_arrays,
            frame_index=raw['frame_index'],
            track_index=raw['track_index'],
            notes=raw['notes'],
        )

    def summary(self) -> dict[str, Any]:
        cap = self.capability
        return {
            'run_id': self.run_id,
            'stage_name': self.stage_name,
            'num_observations': len(self.observations),
            'num_tracks': len(self.track_index or self._build_track_index(self.observations)),
            'num_frames': len(self.frame_index or self._build_frame_index(self.observations)),
            'num_masks': len(self.mask_arrays),
            'backend_used': None if cap is None else cap.backend_used,
            'fidelity': None if cap is None else cap.fidelity,
        }
