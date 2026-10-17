"""Dense human ladder rungs (adapters around the per-tool wrappers).

The humans `BackendLadder` selects a dense backend; each rung here adapts a
swappable tool wrapper to one uniform whole-artifact call — it does the
decode-once-per-frame loop and invokes the wrapper's ``predict(frame, boxes)``,
returning ``{instance_id: dense_record}``. The tool wrappers
(``sapiens_wrapper.SapiensEstimator``, ``pose2d.HumanPose2DEstimator``) stay
separate; these classes are only the ladder glue. A ``dense_record`` carries
``backend`` plus any of ``keypoints_2d`` / ``seg`` / ``depth`` / ``normals``.
"""

from __future__ import annotations

from typing import Any, Optional

from reifyscene.perception.instance.instance_artifact import InstanceArtifact


def _frame_loop(decoder, estimator, ingest_artifact, instance_artifact) -> dict[str, Any]:
    """Decode each frame once; run ``estimator.predict(frame, boxes)`` over its human boxes."""
    stream_lookup = {s.stream_id: s for s in ingest_artifact.streams}
    by_frame: dict[str, list] = {}
    for obs in instance_artifact.observations:
        if obs.kind == 'human':
            by_frame.setdefault(InstanceArtifact._frame_key(obs.frame_ref), []).append(obs)
    out: dict[str, Any] = {}
    for obs_list in by_frame.values():
        fr = obs_list[0].frame_ref
        stream = stream_lookup.get(fr.stream_id)
        if stream is None:
            continue
        frame = decoder.get_frame(stream, fr.frame_idx)
        boxes = [obs.bbox_xyxy for obs in obs_list]
        records = estimator.predict(frame, boxes)
        for obs, rec in zip(obs_list, records):
            out[obs.instance_id] = rec
    return out


class _PoseAdapter:
    """Adapt a 2D-pose wrapper (keypoint arrays) to uniform dense records."""

    def __init__(self, pose) -> None:
        self.pose = pose

    def predict(self, frame, boxes) -> list[Optional[dict[str, Any]]]:
        return [
            None if k is None else {'keypoints_2d': k, 'backend': self.pose.backend_used}
            for k in self.pose.predict(frame, boxes)
        ]


class _BboxOnlyAdapter:
    backend_used = 'bbox_only'

    def predict(self, frame, boxes) -> list[dict[str, Any]]:
        return [{'backend': 'bbox_only'} for _ in boxes]


class SapiensDenseRung:
    """Top rung: Sapiens pose + body-part seg + depth + normals (per person crop)."""

    def __init__(self, decoder, cfg: Optional[dict] = None) -> None:
        from reifyscene.perception.humans.sapiens_wrapper import SapiensEstimator
        self.decoder = decoder
        self.estimator = SapiensEstimator(cfg)

    @property
    def backend_used(self) -> str:
        return self.estimator.backend_used

    def run(self, ingest_artifact, instance_artifact) -> dict[str, Any]:
        return _frame_loop(self.decoder, self.estimator, ingest_artifact, instance_artifact)


class KeypointRcnnRung:
    """Floor rung: honest Keypoint-RCNN pose only (no seg/depth/normals)."""

    def __init__(self, decoder, pose) -> None:
        self.decoder = decoder
        self.pose = pose

    @property
    def backend_used(self) -> str:
        return self.pose.backend_used

    def run(self, ingest_artifact, instance_artifact) -> dict[str, Any]:
        return _frame_loop(self.decoder, _PoseAdapter(self.pose), ingest_artifact, instance_artifact)


class BboxOnlyRung:
    """Graceful floor: presence only, no estimation."""

    backend_used = 'bbox_only'

    def __init__(self, decoder) -> None:
        self.decoder = decoder

    def run(self, ingest_artifact, instance_artifact) -> dict[str, Any]:
        return _frame_loop(self.decoder, _BboxOnlyAdapter(), ingest_artifact, instance_artifact)
