"""High-level pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from typing import Any

from room4d.common.io import read_yaml
from room4d.common.types import FrameRef, HumanState, ObjectState
from room4d.common.paths import ensure_dir
from room4d.common.memory import release_memory
from room4d.data.dataset import MultiVideoDataset
from room4d.geometry.camera_interpolation import densify_camera_states_for_frames
from room4d.geometry.world_frame import GroundPlaneCanonicalizer
from room4d.geometry.vggt_wrapper import VGGTInitializer
from room4d.geometry.vggt_long_wrapper import VGGTLongInitializer
from room4d.graph.builders import DynamicSceneGraphBuilder
from room4d.ingest.decoder import VideoDecoder
from room4d.ingest.ingest_artifact import IngestArtifact
from room4d.ingest.quality import ClipQualityAssessor
from room4d.lifting.backproject import TrackLifter3D
from room4d.lifting.static_dynamic import StaticDynamicPartitioner
from room4d.optimize.solver import JointSpatiotemporalSolver
from room4d.perception.humans.detector import HumanDetector
from room4d.perception.humans.pose2d import HumanPose2DEstimator
from room4d.perception.humans.densepose_wrapper import DensePoseEstimator
from room4d.perception.humans.smpl_fitter import SMPLFitter
from room4d.perception.humans.human_assoc import HumanAssociator
from room4d.perception.humans.human_artifact import HumanArtifact
from room4d.perception.objects.detector import ObjectDetector
from room4d.perception.objects.embedder import InstanceEmbedder
from room4d.perception.objects.object_artifact import ObjectArtifact
from room4d.perception.objects.object_assoc import ObjectAssociator
from room4d.perception.objects.segmenter import ObjectSegmenter
from room4d.query.parser import PromptParser
from room4d.api.resume import apply_stage_compatibility_metadata
from room4d.api.stages import build_stage_registry
from room4d.render.gaussian4d.scene_model import Gaussian4DScene
from room4d.render.registry import build_renderer
from room4d.scene.assembler import SingleVideoSceneAssembler
from room4d.sync.global_time_solver import GlobalTimeAlignmentSolver
from room4d.tracking.cotracker_wrapper import CoTrackerWrapper


class Room4DPipeline:
    """Top-level orchestrator for the staged reconstruction pipeline."""

    def __init__(self, cfg: dict | None = None) -> None:
        """Initialize default stage modules.

        Expected config structure
        -------------------------
        run_dir:
            Root directory for persisted artifacts.
        default_fps:
            Used by the OpenCV decoder when metadata is missing.
        geometry:
            Passed directly to :class:`VGGTInitializer`.
        tracking:
            Passed directly to :class:`CoTrackerWrapper`.
        optimizer:
            Passed directly to :class:`JointSpatiotemporalSolver`.
        """
        self.cfg = cfg or {}
        self.run_dir = ensure_dir(self.cfg.get('run_dir', Path('artifacts/runs/dev')))
        self.decoder = VideoDecoder(default_fps=float(self.cfg.get('default_fps', 30.0)))
        self.quality = ClipQualityAssessor()
        self.sync_solver = GlobalTimeAlignmentSolver()
        geometry_cfg = self.cfg.get('geometry', {})
        geometry_backend_name = str(geometry_cfg.get('backend_name', 'vggt')).lower()
        if geometry_backend_name == 'vggt-long':
            self.geometry_init = VGGTLongInitializer(geometry_cfg)
        else:
            self.geometry_init = VGGTInitializer(geometry_cfg)
        self.tracker = CoTrackerWrapper(self.cfg.get('tracking', {}))
        lifting_cfg = self.cfg.get('lifting', {})
        self.track_lifter = TrackLifter3D(point_conf_threshold=float(lifting_cfg.get('point_conf_threshold', 0.0)))
        self.static_dynamic_partitioner = StaticDynamicPartitioner(
            static_threshold_world=float(lifting_cfg.get('static_threshold_world', 0.05)),
            min_lifted_frames=int(lifting_cfg.get('min_lifted_frames', 2)),
        )
        humans_cfg = self.cfg.get('humans', {})
        self.human_detector = HumanDetector(humans_cfg)
        self.human_pose = HumanPose2DEstimator(humans_cfg)
        self.human_densepose = DensePoseEstimator(humans_cfg)
        self.human_smpl = SMPLFitter(humans_cfg)
        self.human_associator = HumanAssociator(humans_cfg)
        objects_cfg = self.cfg.get('objects', {})
        self.object_detector = ObjectDetector(objects_cfg)
        self.object_segmenter = ObjectSegmenter(objects_cfg)
        self.object_embedder = InstanceEmbedder(objects_cfg)
        self.object_associator = ObjectAssociator(objects_cfg)
        self.graph_builder = DynamicSceneGraphBuilder()
        self.solver = JointSpatiotemporalSolver(self.cfg.get('optimizer', {}))
        self.prompt_parser = PromptParser()
        self.scene_assembler = SingleVideoSceneAssembler(**self.cfg.get("scene", {}))
        self.ground_plane_canonicalizer = GroundPlaneCanonicalizer(geometry_cfg.get("ground_plane", {}))
        self.renderer = build_renderer(self.cfg.get("render", {}))

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'Room4DPipeline':
        """Build a pipeline instance from a YAML config file."""
        return cls(read_yaml(path))

    def _stage_dir(self, stage_name: str) -> Path:
        """Return the canonical save directory for a stage artifact."""
        return ensure_dir(self.run_dir / stage_name)

    def stage_registry(self):
        """Return the declarative stage registry for this pipeline instance."""
        return build_stage_registry()

    @staticmethod
    def _video_paths_from_ingest(ingest_artifact: IngestArtifact) -> list[str]:
        return [str(Path(stream.video_path).expanduser().resolve()) for stream in ingest_artifact.streams]

    def _video_paths_for_stage(self, ingest_artifact: IngestArtifact | None = None) -> list[str]:
        if ingest_artifact is not None:
            return self._video_paths_from_ingest(ingest_artifact)
        return [str(Path(p).expanduser().resolve()) for p in self.cfg.get('_resume_video_paths', [])]

    def _stamp_stage_metadata(self, artifact, stage_name: str, *, video_paths: list[str]) -> None:
        apply_stage_compatibility_metadata(artifact, self.cfg, stage_name, video_paths)

    def preprocess(self, video_paths: list[str]) -> IngestArtifact:
        """Build an ingest artifact from raw video paths.

        TODO implementation order:
        1. Add thumbnail extraction and frame caching.
        2. Add window selection for long videos.
        3. Add optional audio metadata probing.
        """
        streams = self.decoder.decode_many(video_paths)
        dataset = MultiVideoDataset(streams)
        keyframes = dataset.sample_keyframes(
            stride_sec=float(self.cfg.get('keyframe_stride_sec', 1.0)),
            max_frames_per_stream=self.cfg.get('max_keyframes_per_stream', 8),
        )
        quality_by_stream = {stream.stream_id: self.quality.score_stream(stream) for stream in streams}
        artifact = IngestArtifact(
            run_id=self.cfg.get('run_id', 'dev'),
            stage_name='ingest',
            save_dir=self._stage_dir('ingest'),
            streams=streams,
            keyframes=keyframes,
            quality_by_stream=quality_by_stream,
            sampled_windows={},
            warnings=[],
        )
        self._stamp_stage_metadata(artifact, 'ingest', video_paths=video_paths)
        artifact.save()
        return artifact

    def synchronize(self, ingest_artifact: IngestArtifact):
        """Run temporal synchronization.

        The first vertical slice anchors all streams at 0.0 while preserving the
        downstream artifact contract.
        """
        pairwise_offsets: dict[tuple[str, str], tuple[float, float]] = {}
        stream_ids = ingest_artifact.stream_ids()
        for idx, stream_a in enumerate(stream_ids):
            for stream_b in stream_ids[idx + 1 :]:
                pairwise_offsets[(stream_a, stream_b)] = (0.0, 0.0)
        artifact = self.sync_solver.solve_pairwise_offsets(pairwise_offsets, run_id=ingest_artifact.run_id)
        artifact.save_dir = self._stage_dir('sync')
        self._stamp_stage_metadata(artifact, 'sync', video_paths=self._video_paths_for_stage(ingest_artifact))
        artifact.save()
        return artifact

    @staticmethod
    def _all_frame_refs_for_stream(stream) -> list[FrameRef]:
        fps = max(float(stream.fps), 1e-6)
        num_frames = max(int(stream.num_frames), 1)
        return [
            FrameRef(stream_id=stream.stream_id, frame_idx=int(frame_idx), timestamp_sec=float(frame_idx) / fps)
            for frame_idx in range(num_frames)
        ]

    @staticmethod
    def _subsample_frame_refs(frames: list[FrameRef], *, frame_stride: int | None = None, frame_spacing_sec: float | None = None) -> list[FrameRef]:
        by_stream: dict[str, list[FrameRef]] = {}
        for frame in sorted(frames, key=lambda fr: (fr.stream_id, float(fr.timestamp_sec), int(fr.frame_idx))):
            by_stream.setdefault(str(frame.stream_id), []).append(frame)

        selected: list[FrameRef] = []
        for stream_id in sorted(by_stream):
            stream_frames = by_stream[stream_id]
            if frame_spacing_sec is not None and float(frame_spacing_sec) > 0.0:
                last_time: float | None = None
                for frame in stream_frames:
                    if last_time is None or float(frame.timestamp_sec) >= last_time + float(frame_spacing_sec) - 1e-9:
                        selected.append(frame)
                        last_time = float(frame.timestamp_sec)
                continue
            stride = max(int(frame_stride or 1), 1)
            selected.extend(stream_frames[::stride])
        return selected

    def _planned_tracking_frames(self, ingest_artifact: IngestArtifact) -> list[FrameRef]:
        frames: list[FrameRef] = []
        for stream in ingest_artifact.streams:
            frames.extend(self.tracker.plan_frame_refs(stream))
        return sorted(frames, key=lambda fr: (fr.stream_id, int(fr.frame_idx)))

    def _select_geometry_frames(self, ingest_artifact: IngestArtifact) -> list[FrameRef]:
        geometry_cfg = self.cfg.get('geometry', {})
        frame_source = str(geometry_cfg.get('frame_source', 'keyframes')).lower()
        if frame_source == 'tracking':
            frames = self._planned_tracking_frames(ingest_artifact)
        elif frame_source == 'all':
            frames = []
            for stream in ingest_artifact.streams:
                frames.extend(self._all_frame_refs_for_stream(stream))
        else:
            frame_source = 'keyframes'
            frames = list(ingest_artifact.keyframes)

        frames = self._subsample_frame_refs(
            frames,
            frame_stride=geometry_cfg.get('frame_stride'),
            frame_spacing_sec=geometry_cfg.get('frame_spacing_sec'),
        )

        unique: dict[tuple[str, int], FrameRef] = {}
        for frame in frames:
            unique[(str(frame.stream_id), int(frame.frame_idx))] = frame
        selected = [unique[key] for key in sorted(unique)]
        return selected

    def _densify_geometry_cameras_for_tracking(self, geometry_artifact, ingest_artifact: IngestArtifact) -> None:
        tracking_frames = self._planned_tracking_frames(ingest_artifact)
        if not tracking_frames:
            return
        summary = densify_camera_states_for_frames(geometry_artifact.camera_states_by_stream, tracking_frames)
        geometry_artifact.metadata['dense_camera_target'] = 'tracking'
        geometry_artifact.metadata['dense_camera_summary'] = summary
        if int(summary.get('added_camera_states', 0)) > 0:
            geometry_artifact.notes.append(
                f"Added {int(summary['added_camera_states'])} interpolated/extrapolated camera states so tracking frames have camera estimates."
            )

    def initialize_geometry(self, ingest_artifact: IngestArtifact, sync_artifact):
        """Run coarse geometry initialization and persist its heavy outputs."""
        selected_frames = self._select_geometry_frames(ingest_artifact)
        artifact = self.geometry_init.initialize(
            selected_frames,
            ingest_artifact=ingest_artifact,
            sync_artifact=sync_artifact,
        )
        artifact.metadata['geometry_frame_source'] = str(self.cfg.get('geometry', {}).get('frame_source', 'keyframes')).lower()
        estimate = self.ground_plane_canonicalizer.estimate_and_apply(artifact)
        if estimate is not None:
            artifact.metadata['ground_plane_applied'] = True
        else:
            artifact.metadata['ground_plane_applied'] = False
        self._densify_geometry_cameras_for_tracking(artifact, ingest_artifact)
        artifact.save_dir = self._stage_dir('geometry')
        self._stamp_stage_metadata(artifact, 'geometry', video_paths=self._video_paths_for_stage(ingest_artifact))
        artifact.save()
        return artifact

    def track(self, ingest_artifact: IngestArtifact, geometry_artifact=None):
        """Run point tracking, 3D lifting, and static/dynamic partitioning."""
        frame_refs_by_stream: dict[str, list] = {}
        for stream in ingest_artifact.streams:
            frame_refs_by_stream[stream.stream_id] = self.tracker.plan_frame_refs(stream)

        artifact = self.tracker.track_many(
            ingest_artifact.streams,
            frame_refs_by_stream=frame_refs_by_stream,
        )

        # Preserve the actual tracking timeline on the artifact so later stages
        # (scene/render) can use it instead of falling back to geometry frames.
        selected_frames = []
        for stream_id in sorted(frame_refs_by_stream):
            selected_frames.extend(frame_refs_by_stream[stream_id])
        artifact.selected_frames = sorted(
            selected_frames,
            key=lambda fr: (fr.stream_id, int(fr.frame_idx)),
        )
        artifact.run_id = ingest_artifact.run_id
        tracked_frame_refs: list[FrameRef] = []
        for stream in ingest_artifact.streams:
            tracked_frame_refs.extend(self.tracker.plan_frame_refs(stream))
        artifact.selected_frames = sorted(tracked_frame_refs, key=lambda fr: (fr.stream_id, int(fr.frame_idx)))
        if geometry_artifact is not None:
            artifact = self.track_lifter.lift(artifact, geometry_artifact, ingest_artifact)
            self.static_dynamic_partitioner.partition(artifact, geometry_artifact)
            tracking_cfg = self.cfg.get('tracking', {})
            artifact.metadata.update({
                'save_debug_assets': bool(tracking_cfg.get('debug', False)),
                'save_debug_html': bool(tracking_cfg.get('debug', False)),
                'debug_max_points': int(tracking_cfg.get('debug_max_points', 5000)),
                'debug_image_paths': dict(geometry_artifact.metadata.get('debug_image_paths', {})),
            })
        artifact.save_dir = self._stage_dir('tracking')
        self._stamp_stage_metadata(artifact, 'tracking', video_paths=self._video_paths_for_stage(ingest_artifact))
        artifact.save()
        return artifact

    def _effective_sample_stride(self, stage_cfg: dict[str, Any]) -> int:
        """Return the effective original-frame stride for a perception stage."""
        semantic_stride = max(1, int(stage_cfg.get('frame_stride', 1)))
        if str(stage_cfg.get('frame_source', 'tracking')) == 'tracking':
            tracking_stride = max(1, int(self.cfg.get('tracking', {}).get('frame_stride', 1)))
            return tracking_stride * semantic_stride
        return semantic_stride

    def _select_perception_frames(self, ingest_artifact: IngestArtifact, track_artifact, stage_cfg: dict[str, Any], *, default_source: str = 'tracking') -> list[FrameRef]:
        """Select semantic-perception frames, preferring tracked frames when available."""
        source = str(stage_cfg.get('frame_source', default_source))
        semantic_stride = max(1, int(stage_cfg.get('frame_stride', 1)))
        if source == 'tracking' and track_artifact is not None:
            stream_lookup = {stream.stream_id: stream for stream in ingest_artifact.streams}
            frames: list[FrameRef] = []
            for stream_id, stream in stream_lookup.items():
                payload = track_artifact.track_payloads_by_stream.get(stream_id, {})
                frame_indices = np.asarray(payload.get('frame_indices', []), dtype=int)
                if frame_indices.size == 0:
                    continue
                unique_indices = np.unique(frame_indices)
                chosen = unique_indices[::semantic_stride]
                fps = max(float(stream.fps), 1e-6)
                for frame_idx in chosen.tolist():
                    frames.append(FrameRef(stream_id=stream_id, frame_idx=int(frame_idx), timestamp_sec=float(frame_idx) / fps))
            if frames:
                return sorted(frames, key=lambda fr: (fr.stream_id, int(fr.frame_idx)))
        keyframes = sorted(list(ingest_artifact.keyframes), key=lambda fr: (fr.stream_id, int(fr.frame_idx)))
        if semantic_stride <= 1:
            return keyframes
        selected: list[FrameRef] = []
        by_stream: dict[str, list[FrameRef]] = {}
        for frame_ref in keyframes:
            by_stream.setdefault(frame_ref.stream_id, []).append(frame_ref)
        for stream_id in sorted(by_stream):
            selected.extend(by_stream[stream_id][::semantic_stride])
        return selected

    def perceive_humans(self, ingest_artifact, track_artifact=None):
        """Run single-video human detection, pose, DensePose, SMPL, and association."""
        humans_cfg = self.cfg.get('humans', {})
        selected_frames = self._select_perception_frames(ingest_artifact, track_artifact, humans_cfg, default_source='tracking')
        stream_lookup = {stream.stream_id: stream for stream in ingest_artifact.streams}
        densepose_stride = max(1, int(humans_cfg.get('densepose_frame_stride', 1)))
        smpl_stride = max(1, int(humans_cfg.get('smpl_frame_stride', 1)))
        effective_stride = self._effective_sample_stride(humans_cfg)
        human_states: list[HumanState] = []
        mask_arrays: dict[str, np.ndarray] = {}
        densepose_payloads: dict[str, dict[str, Any]] = {}
        smpl_payloads: dict[str, dict[str, Any]] = {}
        notes: list[str] = []

        track_payloads = {} if track_artifact is None else track_artifact.track_payloads_by_stream

        def _frame_track_support(stream_id: str, frame_idx: int, mask: Any, bbox_xyxy: Any) -> dict[str, Any]:
            payload = track_payloads.get(stream_id, {})
            points_xy = np.asarray(payload.get('points_xy', np.zeros((0, 0, 2), dtype=np.float32)), dtype=np.float32)
            visibility = np.asarray(payload.get('visibility', np.zeros((0, 0), dtype=bool)))
            track_ids = np.asarray(payload.get('track_ids', np.arange(points_xy.shape[1] if points_xy.ndim == 3 else 0)), dtype=int)
            frame_indices = np.asarray(payload.get('frame_indices', []), dtype=int)
            if points_xy.ndim != 3 or frame_indices.size == 0:
                return {'track_ids': [], 'support_points_xy': [], 'visible_track_count': 0, 'bbox_support_count': 0, 'mask_support_count': 0, 'support_track_count': 0}
            matches = np.where(frame_indices == int(frame_idx))[0]
            if matches.size == 0:
                return {'track_ids': [], 'support_points_xy': [], 'visible_track_count': 0, 'bbox_support_count': 0, 'mask_support_count': 0, 'support_track_count': 0}
            t = int(matches[0])
            visible = visibility[t].astype(bool) if visibility.size else np.ones((points_xy.shape[1],), dtype=bool)
            pts = points_xy[t]
            bbox = np.asarray(bbox_xyxy, dtype=np.float32)
            x1, y1, x2, y2 = bbox.tolist()
            inside_bbox = (pts[:, 0] >= x1) & (pts[:, 0] <= x2) & (pts[:, 1] >= y1) & (pts[:, 1] <= y2)
            bbox_keep = visible & inside_bbox
            mask_keep = bbox_keep.copy()
            if mask is not None:
                mask_arr = np.asarray(mask, dtype=bool)
                if mask_arr.ndim == 2 and mask_arr.shape[0] > 0 and mask_arr.shape[1] > 0:
                    xi = np.clip(np.round(pts[:, 0]).astype(int), 0, mask_arr.shape[1] - 1)
                    yi = np.clip(np.round(pts[:, 1]).astype(int), 0, mask_arr.shape[0] - 1)
                    mask_keep &= mask_arr[yi, xi]
            keep = mask_keep if int(mask_keep.sum()) > 0 else bbox_keep
            support_points = pts[keep] if np.any(keep) else np.zeros((0, 2), dtype=np.float32)
            selected_ids = [int(tid) for tid in track_ids[keep].tolist()]
            return {
                'track_ids': selected_ids,
                'support_points_xy': np.asarray(support_points, dtype=np.float32).tolist(),
                'visible_track_count': int(visible.sum()),
                'bbox_support_count': int(bbox_keep.sum()),
                'mask_support_count': int(mask_keep.sum()),
                'support_track_count': int(len(selected_ids)),
            }

        for semantic_idx, frame_ref in enumerate(selected_frames):
            stream = stream_lookup[frame_ref.stream_id]
            frame = self.decoder.get_frame(stream, frame_ref.frame_idx)
            detections = self.human_detector.detect(frame)
            boxes = [np.asarray(det['bbox_xyxy'], dtype=np.float32) for det in detections]
            masks = []
            for det_idx, det in enumerate(detections):
                logits = det.get('mask_logits')
                mask = None if logits is None else (np.asarray(logits, dtype=np.float32) >= 0.5)
                det['mask'] = mask
                masks.append(mask)
            keypoints = self.human_pose.predict(frame, boxes)
            run_densepose = (semantic_idx % densepose_stride) == 0
            run_smpl = (semantic_idx % smpl_stride) == 0
            denseposes = self.human_densepose.predict(frame, boxes, masks=masks, keypoints=keypoints) if run_densepose else [None] * len(boxes)
            for det_idx, det in enumerate(detections):
                detection_id = f"hum:{frame_ref.stream_id}:{int(frame_ref.frame_idx)}:{det_idx}"
                mask = masks[det_idx] if det_idx < len(masks) else None
                if mask is not None:
                    mask_arrays[detection_id] = np.asarray(mask, dtype=bool)
                kpts = keypoints[det_idx] if det_idx < len(keypoints) else None
                dp = denseposes[det_idx] if det_idx < len(denseposes) else None
                if dp is not None:
                    densepose_payloads[detection_id] = dp
                smpl = self.human_smpl.fit_frame(frame, kpts, mask=mask, intrinsics=None) if run_smpl else None
                if smpl is not None:
                    smpl_payloads[detection_id] = smpl
                support = _frame_track_support(frame_ref.stream_id, frame_ref.frame_idx, mask, det['bbox_xyxy'])
                attrs = {
                    'detection_id': detection_id,
                    'track_ids': list(support['track_ids']),
                    'support_points_xy': list(support['support_points_xy']),
                    'visible_track_count': int(support['visible_track_count']),
                    'bbox_support_count': int(support['bbox_support_count']),
                    'mask_support_count': int(support['mask_support_count']),
                    'support_track_count': int(support['support_track_count']),
                    'frame_key': f"{frame_ref.stream_id}:{int(frame_ref.frame_idx)}",
                    'semantic_frame_index': int(semantic_idx),
                    'semantic_frame_source': str(humans_cfg.get('frame_source', 'tracking')),
                    'densepose_computed': bool(run_densepose),
                    'smpl_computed': bool(run_smpl),
                    'frame_stride': int(effective_stride),
                    'detector_backend': self.human_detector.backend_used,
                    'pose_backend': self.human_pose.backend_used,
                    'densepose_backend': self.human_densepose.backend_used,
                    'smpl_backend': self.human_smpl.backend_used,
                }
                human_states.append(
                    HumanState(
                        human_id=detection_id,
                        frame_ref=frame_ref,
                        bbox_xyxy=np.asarray(det['bbox_xyxy'], dtype=np.float32),
                        mask_rle=None if mask is None else {'shape': list(mask.shape), 'sum': int(np.asarray(mask, dtype=bool).sum())},
                        keypoints_2d=None if kpts is None else np.asarray(kpts, dtype=np.float32),
                        densepose=dp,
                        smpl_params=smpl,
                        confidence=float(det['score']),
                        attributes=attrs,
                    )
                )

        artifact = HumanArtifact(
            run_id=ingest_artifact.run_id,
            stage_name='humans',
            save_dir=self._stage_dir('humans'),
            humans=human_states,
            raw_humans=[],
            linked_ids={},
            frame_index={},
            mask_arrays=mask_arrays,
            densepose_payloads=densepose_payloads,
            smpl_payloads=smpl_payloads,
            notes=notes,
            metadata={
                'detector_backend': self.human_detector.backend_used,
                'pose_backend': self.human_pose.backend_used,
                'densepose_backend': self.human_densepose.backend_used,
                'smpl_backend': self.human_smpl.backend_used,
                'save_debug_assets': False,
                'save_debug_html': False,
                'frame_source': str(humans_cfg.get('frame_source', 'tracking')),
                'frame_stride': int(humans_cfg.get('frame_stride', 1)),
                'densepose_frame_stride': int(densepose_stride),
                'smpl_frame_stride': int(smpl_stride),
                'selected_frame_count': int(len(selected_frames)),
            },
        )
        artifact.notes.extend([
            f"selected_semantic_frames={len(selected_frames)}",
            f"frame_source={humans_cfg.get('frame_source', 'tracking')}",
            f"densepose_stride={densepose_stride}",
            f"smpl_stride={smpl_stride}",
        ])
        artifact.raw_humans = [HumanState(
            human_id=h.human_id,
            frame_ref=h.frame_ref,
            bbox_xyxy=np.asarray(h.bbox_xyxy, dtype=np.float32).copy(),
            mask_rle=h.mask_rle,
            keypoints_2d=None if h.keypoints_2d is None else np.asarray(h.keypoints_2d, dtype=np.float32).copy(),
            densepose=h.densepose,
            smpl_params=h.smpl_params,
            confidence=float(h.confidence),
            attributes=dict(h.attributes),
        ) for h in human_states]
        artifact.frame_index = artifact._build_frame_index(artifact.humans)
        artifact = self.human_associator.associate_across_views(artifact)
        self._stamp_stage_metadata(artifact, 'humans', video_paths=self._video_paths_for_stage(ingest_artifact))
        artifact.save()
        return artifact

    def perceive_objects(self, ingest_artifact, track_artifact=None):
        """Run single-video object detection, segmentation, and re-identification."""
        objects_cfg = self.cfg.get('objects', {})
        selected_frames = self._select_perception_frames(ingest_artifact, track_artifact, objects_cfg, default_source='tracking')
        stream_lookup = {stream.stream_id: stream for stream in ingest_artifact.streams}
        effective_stride = self._effective_sample_stride(objects_cfg)
        object_states: list[ObjectState] = []
        mask_arrays: dict[str, Any] = {}
        embedding_vectors: dict[str, Any] = {}
        notes: list[str] = []

        track_payloads = {} if track_artifact is None else track_artifact.track_payloads_by_stream

        def _frame_track_support(stream_id: str, frame_idx: int, mask: Any, bbox_xyxy: Any) -> dict[str, Any]:
            payload = track_payloads.get(stream_id, {})
            points_xy = np.asarray(payload.get('points_xy', np.zeros((0, 0, 2), dtype=np.float32)), dtype=np.float32)
            visibility = np.asarray(payload.get('visibility', np.zeros((0, 0), dtype=bool)))
            track_ids = np.asarray(payload.get('track_ids', np.arange(points_xy.shape[1] if points_xy.ndim == 3 else 0)), dtype=int)
            frame_indices = np.asarray(payload.get('frame_indices', []), dtype=int)
            if points_xy.ndim != 3 or frame_indices.size == 0:
                return {'track_ids': [], 'support_points_xy': [], 'visible_track_count': 0, 'bbox_support_count': 0, 'mask_support_count': 0, 'support_track_count': 0}
            matches = np.where(frame_indices == int(frame_idx))[0]
            if matches.size == 0:
                return {'track_ids': [], 'support_points_xy': [], 'visible_track_count': 0, 'bbox_support_count': 0, 'mask_support_count': 0, 'support_track_count': 0}
            t = int(matches[0])
            visible = visibility[t].astype(bool) if visibility.size else np.ones((points_xy.shape[1],), dtype=bool)
            pts = points_xy[t]
            bbox = np.asarray(bbox_xyxy, dtype=np.float32)
            x1, y1, x2, y2 = bbox.tolist()
            inside_bbox = (pts[:, 0] >= x1) & (pts[:, 0] <= x2) & (pts[:, 1] >= y1) & (pts[:, 1] <= y2)
            bbox_keep = visible & inside_bbox
            mask_keep = bbox_keep.copy()
            if mask is not None:
                mask_arr = np.asarray(mask, dtype=bool)
                if mask_arr.ndim == 2 and mask_arr.shape[0] > 0 and mask_arr.shape[1] > 0:
                    xi = np.clip(np.round(pts[:, 0]).astype(int), 0, mask_arr.shape[1] - 1)
                    yi = np.clip(np.round(pts[:, 1]).astype(int), 0, mask_arr.shape[0] - 1)
                    mask_keep &= mask_arr[yi, xi]
            keep = mask_keep if int(mask_keep.sum()) > 0 else bbox_keep
            support_points = pts[keep] if np.any(keep) else np.zeros((0, 2), dtype=np.float32)
            selected_ids = [int(tid) for tid in track_ids[keep].tolist()]
            return {
                'track_ids': selected_ids,
                'support_points_xy': np.asarray(support_points, dtype=np.float32).tolist(),
                'visible_track_count': int(visible.sum()),
                'bbox_support_count': int(bbox_keep.sum()),
                'mask_support_count': int(mask_keep.sum()),
                'support_track_count': int(len(selected_ids)),
            }

        for semantic_idx, frame_ref in enumerate(selected_frames):
            stream = stream_lookup[frame_ref.stream_id]
            frame = self.decoder.get_frame(stream, frame_ref.frame_idx)
            detections = self.object_detector.detect(frame)
            detections = self.object_segmenter.segment(frame, detections)
            embeddings = self.object_embedder.embed(frame, detections)
            for det_idx, det in enumerate(detections):
                detection_id = f"det:{frame_ref.stream_id}:{int(frame_ref.frame_idx)}:{det_idx}"
                mask = np.asarray(det.get('mask'), dtype=bool) if det.get('mask') is not None else None
                if mask is not None:
                    mask_arrays[detection_id] = mask.astype(bool)
                embedding = embeddings[det_idx] if det_idx < embeddings.shape[0] else None
                if embedding is not None:
                    embedding_vectors[detection_id] = np.asarray(embedding, dtype=np.float32)
                support = _frame_track_support(frame_ref.stream_id, frame_ref.frame_idx, mask, det['bbox_xyxy'])
                attrs = {
                    'detection_id': detection_id,
                    'label_id': int(det.get('label_id', -1)),
                    'track_ids': list(support['track_ids']),
                    'support_points_xy': list(support['support_points_xy']),
                    'visible_track_count': int(support['visible_track_count']),
                    'bbox_support_count': int(support['bbox_support_count']),
                    'mask_support_count': int(support['mask_support_count']),
                    'support_track_count': int(support['support_track_count']),
                    'frame_key': f"{frame_ref.stream_id}:{int(frame_ref.frame_idx)}",
                    'semantic_frame_index': int(semantic_idx),
                    'semantic_frame_source': str(objects_cfg.get('frame_source', 'tracking')),
                    'frame_stride': int(effective_stride),
                    'detector_backend': self.object_detector.backend_used,
                    'embedder_backend': self.object_embedder.backend_used,
                }
                object_states.append(
                    ObjectState(
                        object_id=detection_id,
                        frame_ref=frame_ref,
                        category=str(det['category']),
                        bbox_xyxy=np.asarray(det['bbox_xyxy'], dtype=np.float32),
                        mask_rle=None if mask is None else {'shape': list(mask.shape), 'sum': int(mask.sum())},
                        embedding=None if embedding is None else np.asarray(embedding, dtype=np.float32),
                        confidence=float(det['score']),
                        attributes=attrs,
                    )
                )

        artifact = ObjectArtifact(
            run_id=ingest_artifact.run_id,
            stage_name='objects',
            save_dir=self._stage_dir('objects'),
            objects=object_states,
            linked_ids={},
            frame_index={},
            mask_arrays=mask_arrays,
            embedding_vectors=embedding_vectors,
            notes=notes,
            metadata={
                'detector_backend': self.object_detector.backend_used,
                'embedder_backend': self.object_embedder.backend_used,
                'save_debug_assets': False,
                'save_debug_html': False,
                'frame_source': str(objects_cfg.get('frame_source', 'tracking')),
                'frame_stride': int(objects_cfg.get('frame_stride', 1)),
                'selected_frame_count': int(len(selected_frames)),
            },
        )
        artifact.notes.extend([
            f"selected_semantic_frames={len(selected_frames)}",
            f"frame_source={objects_cfg.get('frame_source', 'tracking')}",
        ])
        artifact.frame_index = artifact._build_frame_index(artifact.objects)
        artifact = self.object_associator.associate(artifact, track_artifact=track_artifact)
        self._stamp_stage_metadata(artifact, 'objects', video_paths=self._video_paths_for_stage(ingest_artifact))
        artifact.save()
        return artifact

    def build_graph(self, geometry_artifact, track_artifact, human_artifact=None, object_artifact=None):
        """Build the dynamic scene graph."""
        artifact = self.graph_builder.build(geometry_artifact, track_artifact, human_artifact, object_artifact)
        artifact.save_dir = self._stage_dir('graph')
        self._stamp_stage_metadata(artifact, 'graph', video_paths=self._video_paths_for_stage())
        artifact.save()
        return artifact

    def optimize(self, sync_artifact, geometry_artifact, track_artifact, human_artifact=None, object_artifact=None, graph_artifact=None):
        """Run staged joint optimization."""
        artifact = self.solver.solve_all(sync_artifact, geometry_artifact, track_artifact, human_artifact, object_artifact, graph_artifact)
        artifact.save_dir = self._stage_dir('optimization')
        self._stamp_stage_metadata(artifact, 'optimization', video_paths=self._video_paths_for_stage())
        artifact.save()
        return artifact

    def assemble_scene(self, geometry_artifact, track_artifact, human_artifact=None, object_artifact=None, optimization_artifact=None):
        """Assemble a lightweight static/dynamic single-video 4D scene."""
        artifact = self.scene_assembler.assemble(
            run_id=geometry_artifact.run_id,
            save_dir=self._stage_dir('scene'),
            geometry_artifact=geometry_artifact,
            track_artifact=track_artifact,
            human_artifact=human_artifact,
            object_artifact=object_artifact,
            metadata={
                'optimization_available': optimization_artifact is not None,
                'timeline_source': str(self.cfg.get('scene', {}).get('timeline_source', 'geometry')),
            },
        )
        self._stamp_stage_metadata(artifact, 'scene', video_paths=self._video_paths_for_stage())
        artifact.save()
        return artifact

    def fit_renderer(self, scene_artifact, optimization_artifact=None, geometry_artifact=None):
        """Fit a renderable 4D scene model and export novel-view playback."""
        scene = Gaussian4DScene(self.cfg.get('render', {}))
        scene.initialize_from_scene(scene_artifact)
        render_dir = self._stage_dir('render')
        artifact = self.renderer.fit(
            scene,
            optimization_artifact=optimization_artifact,
            geometry_artifact=geometry_artifact,
            save_dir=render_dir,
        )
        artifact.save_dir = render_dir
        self._stamp_stage_metadata(artifact, 'render', video_paths=self._video_paths_for_stage())
        artifact.save()
        return artifact


    def release_stage_resources(self, stage_name: str) -> None:
        """Release heavy backend state after a stage completes."""
        stage_name = str(stage_name)
        components: list[Any] = []
        if stage_name == 'geometry':
            components = [self.geometry_init]
        elif stage_name == 'tracking':
            components = [self.tracker]
        elif stage_name == 'humans':
            components = [self.human_detector, self.human_pose, self.human_densepose, self.human_smpl]
        elif stage_name == 'objects':
            components = [self.object_detector, self.object_embedder]
        elif stage_name == 'render':
            components = [self.renderer]
        for component in components:
            if component is None:
                continue
            if hasattr(component, 'release_resources'):
                try:
                    component.release_resources()
                except Exception:
                    pass

        release_memory()

    def run(self, video_paths: list[str]) -> dict[str, Any]:
        """Execute the current end-to-end vertical slice and persist artifacts."""
        ingest_artifact = self.preprocess(video_paths)
        sync_artifact = self.synchronize(ingest_artifact)
        geometry_artifact = self.initialize_geometry(ingest_artifact, sync_artifact)
        track_artifact = self.track(ingest_artifact, geometry_artifact=geometry_artifact)
        human_artifact = self.perceive_humans(ingest_artifact, track_artifact=track_artifact)
        object_artifact = self.perceive_objects(ingest_artifact, track_artifact=track_artifact)
        graph_artifact = self.build_graph(geometry_artifact, track_artifact, human_artifact=human_artifact, object_artifact=object_artifact)
        optimization_artifact = self.optimize(sync_artifact, geometry_artifact, track_artifact, human_artifact=human_artifact, object_artifact=object_artifact, graph_artifact=graph_artifact)
        scene_artifact = self.assemble_scene(geometry_artifact, track_artifact, human_artifact=human_artifact, object_artifact=object_artifact, optimization_artifact=optimization_artifact)
        render_artifact = self.fit_renderer(scene_artifact, optimization_artifact=optimization_artifact, geometry_artifact=geometry_artifact)
        return {
            'ingest': ingest_artifact,
            'sync': sync_artifact,
            'geometry': geometry_artifact,
            'tracks': track_artifact,
            'humans': human_artifact,
            'objects': object_artifact,
            'graph': graph_artifact,
            'optimization': optimization_artifact,
            'scene': scene_artifact,
            'render': render_artifact,
        }
