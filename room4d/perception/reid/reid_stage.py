"""Floor-rung re-ID: ResNet18 appearance features per tracklet.

The deliberately-simple **floor** of the `reid` ladder (design §4A, FP1). It
mean-pools today's ResNet18 instance embedder over each tracklet's detections to
give a per-track appearance vector, and leaves the cross-occlusion ``merge_map``
empty (no gap-merging at the floor — that arrives with DINOv3 at FP3). DINOv3
replaces this rung; the contract it fills (`ReidArtifact`: embeddings +
merge_map) is stable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from room4d.common.types import FrameRef
from room4d.perception.instance.instance_artifact import InstanceArtifact


class FloorReid:
    """Per-tracklet mean ResNet18 embeddings over the instance masks/boxes."""

    def __init__(self, decoder, embedder) -> None:
        self.decoder = decoder
        self.embedder = embedder

    def run(self, ingest_artifact, instance_artifact: InstanceArtifact) -> dict[str, np.ndarray]:
        """Return ``{track_id: mean_embedding}``. Frames are decoded once and all
        instances in a frame are embedded together (the embedder is batched)."""
        stream_lookup = {s.stream_id: s for s in ingest_artifact.streams}
        by_frame: dict[str, list] = {}
        for obs in instance_artifact.observations:
            by_frame.setdefault(InstanceArtifact._frame_key(obs.frame_ref), []).append(obs)

        sums: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        for _frame_key, observations in by_frame.items():
            fr: FrameRef = observations[0].frame_ref
            stream = stream_lookup.get(fr.stream_id)
            if stream is None:
                continue
            frame = self.decoder.get_frame(stream, fr.frame_idx)
            dets = [{
                'bbox_xyxy': np.asarray(o.bbox_xyxy, dtype=np.float32),
                'mask': instance_artifact.mask_arrays.get(o.instance_id),
            } for o in observations]
            embeddings = self.embedder.embed(frame, dets)
            for o, emb in zip(observations, np.asarray(embeddings)):
                vec = np.asarray(emb, dtype=np.float32).reshape(-1)
                if o.track_id in sums:
                    sums[o.track_id] = sums[o.track_id] + vec
                else:
                    sums[o.track_id] = vec.copy()
                counts[o.track_id] = counts.get(o.track_id, 0) + 1

        out: dict[str, np.ndarray] = {}
        for track_id, total in sums.items():
            mean = total / max(counts[track_id], 1)
            norm = float(np.linalg.norm(mean))
            out[track_id] = (mean / norm).astype(np.float32) if norm > 1e-8 else mean.astype(np.float32)
        return out
