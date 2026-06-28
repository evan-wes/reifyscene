"""High-level pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from typing import Any

from room4d.common.io import read_yaml
from room4d.calibration.runner import CalibrationRunner
from room4d.common.types import CameraPose, FrameRef, HumanState, ObjectState, TimedCameraState
from room4d.common.paths import ensure_dir
from room4d.common.memory import release_memory
from room4d.data.dataset import MultiVideoDataset
from room4d.geometry.camera_interpolation import densify_camera_states_for_frames
from room4d.geometry.geometry_artifact import GeometryArtifact
from room4d.geometry.pose_refine import PoseRefiner, PoseRefineUnavailable
from room4d.geometry.world_frame import GroundPlaneCanonicalizer
from room4d.geometry.vggt_wrapper import VGGTInitializer
from room4d.geometry.vggt_long_wrapper import VGGTLongInitializer
from room4d.geometry.dataset_wrapper import DatasetGeometryInitializer
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
from room4d.perception.humans.human_artifact import HumanArtifact
from room4d.perception.objects.detector import ObjectDetector
from room4d.perception.objects.embedder import InstanceEmbedder
from room4d.perception.objects.object_artifact import ObjectArtifact
from room4d.perception.objects.segmenter import ObjectSegmenter
from room4d.perception.instance.instance_artifact import InstanceArtifact, InstanceObservation
from room4d.perception.instance.segment import FloorInstanceSegmenter
from room4d.perception.reid.reid_artifact import ReidArtifact
from room4d.perception.reid.reid_stage import FloorReid
from room4d.common.backend_ladder import BackendLadder, BackendRung
from room4d.query.parser import PromptParser
from room4d.api.resume import apply_stage_compatibility_metadata
from room4d.api.stages import build_stage_registry
from room4d.render.registry import build_renderer
from room4d.scene.assembler import SingleVideoSceneAssembler
from room4d.eval.benchmark_runner import BenchmarkRunner
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
        elif geometry_backend_name == 'dataset':
            self.geometry_init = DatasetGeometryInitializer(geometry_cfg)
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
        objects_cfg = self.cfg.get('objects', {})
        self.object_detector = ObjectDetector(objects_cfg)
        self.object_segmenter = ObjectSegmenter(objects_cfg)
        self.object_embedder = InstanceEmbedder(objects_cfg)
        self.graph_builder = DynamicSceneGraphBuilder()
        self.solver = JointSpatiotemporalSolver(self.cfg.get('optimizer', {}))
        self.prompt_parser = PromptParser()
        self.scene_assembler = SingleVideoSceneAssembler(**self.cfg.get("scene", {}))
        self.ground_plane_canonicalizer = GroundPlaneCanonicalizer(geometry_cfg.get("ground_plane", {}))
        self.renderer = build_renderer(self.cfg.get("render", {}))
        self.benchmark_runner = BenchmarkRunner(self.cfg.get("evaluation", {}))
        self.calibration_runner = CalibrationRunner(self.cfg.get("calibration", {}), decoder=self.decoder)

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


    def calibrate(self, video_paths: list[str]) -> Any:
        """Run phase-1 static-frame calibration on one or more input videos."""
        artifact = self.calibration_runner.run(
            run_id=self.cfg.get('run_id', 'dev'),
            save_dir=self._stage_dir('calibration'),
            video_paths=video_paths,
        )
        self._stamp_stage_metadata(artifact, 'calibration', video_paths=[str(Path(p).expanduser().resolve()) for p in video_paths])
        artifact.save()
        return artifact

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

    def refine_poses(self, ingest_artifact: IngestArtifact, geometry_artifact: GeometryArtifact) -> GeometryArtifact:
        """BA-refine the geometry camera poses and emit a refined GeometryArtifact.

        Backend-agnostic: consumes the GeometryArtifact contract (poses +
        intrinsics + per-pixel world points) and the source frames, runs the
        :class:`PoseRefiner` (VGGSfM tracks -> pycolmap seed -> robust iterative
        BA, refine shared focal), then rebuilds the affected streams' keyframe
        camera states from the refined poses and re-densifies interpolated states
        for the tracking timeline.

        When refinement is disabled (``cfg['pose_refine']['enabled']`` is false)
        or the refinement runtime is unavailable, this is an identity
        pass-through that re-emits the geometry unchanged under the ``pose_refine``
        stage so downstream stages have a stable dependency either way.
        """
        pose_cfg = dict(self.cfg.get('pose_refine', {}))
        enabled = bool(pose_cfg.get('enabled', False))
        save_dir = self._stage_dir('pose_refine')

        refinements: dict[str, Any] = {}
        extra_notes: list[str] = []
        if enabled:
            refiner = PoseRefiner(pose_cfg)
            try:
                refinements = refiner.refine(geometry_artifact, decoder=self.decoder, ingest_artifact=ingest_artifact)
            except PoseRefineUnavailable as exc:
                extra_notes.append(f"Pose refinement unavailable; emitted geometry unchanged ({exc}).")
            extra_notes.extend(refiner.notes)
        else:
            extra_notes.append("Pose refinement disabled; emitted geometry unchanged.")

        new_intrinsics = dict(geometry_artifact.intrinsics_by_stream)
        new_states = {sid: list(states) for sid, states in geometry_artifact.camera_states_by_stream.items()}
        for stream_id, refinement in refinements.items():
            new_intrinsics[stream_id] = refinement.refined_intrinsics
            keyframes = sorted(
                (
                    f for f in geometry_artifact.selected_frames
                    if str(f.stream_id) == stream_id
                    and f"{stream_id}:{int(f.frame_idx)}" in geometry_artifact.pointmaps_by_frame
                ),
                key=lambda fr: int(fr.frame_idx),
            )
            rebuilt: list[TimedCameraState] = []
            for fr in keyframes:
                key = f"{stream_id}:{int(fr.frame_idx)}"
                orig = geometry_artifact._find_state_for_frame(fr)
                rebuilt.append(TimedCameraState(
                    stream_id=stream_id,
                    time_sec=float(orig.time_sec) if orig is not None else float(fr.timestamp_sec),
                    intrinsics=refinement.refined_intrinsics,
                    pose=CameraPose(T_world_cam=np.asarray(refinement.refined_T_world_cam_by_frame_key[key], dtype=float)),
                ))
            new_states[stream_id] = rebuilt

        refined_metadata = dict(geometry_artifact.metadata or {})
        refined_metadata['pose_refine'] = {
            'enabled': enabled,
            'refined_streams': sorted(refinements.keys()),
            'stats': {sid: refinement.stats for sid, refinement in refinements.items()},
        }
        refined = GeometryArtifact(
            run_id=geometry_artifact.run_id,
            stage_name='pose_refine',
            save_dir=save_dir,
            metadata=refined_metadata,
            backend_name=geometry_artifact.backend_name,
            selected_frames=list(geometry_artifact.selected_frames),
            intrinsics_by_stream=new_intrinsics,
            camera_states_by_stream=new_states,
            pointmaps_by_frame=geometry_artifact.pointmaps_by_frame,
            depthmaps_by_frame=geometry_artifact.depthmaps_by_frame,
            pairwise_matches=geometry_artifact.pairwise_matches,
            view_graph_edges=geometry_artifact.view_graph_edges,
            notes=list(geometry_artifact.notes) + extra_notes,
        )
        # Re-derive interpolated tracking-frame states from the refined keyframes
        # (mirrors initialize_geometry). Skipped on pass-through, where the copied
        # states are already correct for the un-refined poses.
        if refinements:
            self._densify_geometry_cameras_for_tracking(refined, ingest_artifact)
        refined.save_dir = save_dir
        self._stamp_stage_metadata(refined, 'pose_refine', video_paths=self._video_paths_for_stage(ingest_artifact))
        refined.save()
        return refined

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


    def _project_world_to_image(self, xyz_world: np.ndarray, frame_ref: FrameRef, geometry_artifact) -> np.ndarray | None:
        if geometry_artifact is None:
            return None
        states = (getattr(geometry_artifact, 'camera_states_by_stream', {}) or {}).get(frame_ref.stream_id, [])
        if not states:
            return None
        best = None
        best_dt = None
        for st in states:
            dt = abs(float(st.time_sec) - float(frame_ref.timestamp_sec))
            if best is None or dt < best_dt:
                best = st
                best_dt = dt
        if best is None:
            return None
        Twc = np.asarray(best.pose.T_world_cam, dtype=np.float32)
        try:
            Tcw = np.linalg.inv(Twc)
        except np.linalg.LinAlgError:
            return None
        xyz_h = np.asarray(list(np.asarray(xyz_world, dtype=np.float32)) + [1.0], dtype=np.float32)
        cam = Tcw @ xyz_h
        if not np.all(np.isfinite(cam)) or abs(float(cam[2])) < 1e-6:
            return None
        intr = best.intrinsics
        x = float(intr.cx) + float(intr.fx) * (float(cam[0]) / float(cam[2]))
        y = float(intr.cy) + float(intr.fy) * (float(cam[1]) / float(cam[2]))
        return np.asarray([x, y], dtype=np.float32)

    def _attach_human_debug_projections(self, artifact, track_artifact, geometry_artifact) -> None:
        if geometry_artifact is None or artifact is None:
            return
        projected_by_frame: dict[str, list[dict[str, Any]]] = {}
        for human in list(getattr(artifact, 'humans', []) or []):
            try:
                world = self.scene_assembler._state_world_position(human, 'humans', track_artifact, geometry_artifact, source_artifact=artifact)
            except Exception:
                world = None
            if world is None:
                continue
            xy = self._project_world_to_image(np.asarray(world, dtype=np.float32), human.frame_ref, geometry_artifact)
            if xy is None or not np.all(np.isfinite(xy)):
                continue
            frame_key = f"{human.frame_ref.stream_id}:{int(human.frame_ref.frame_idx)}"
            projected_by_frame.setdefault(frame_key, []).append({
                'human_id': str(human.human_id),
                'xy': np.asarray(xy, dtype=np.float32).tolist(),
                'world_xyz': np.asarray(world, dtype=np.float32).tolist(),
                'confidence': float(getattr(human, 'confidence', 0.0)),
                'anchor_source': str((getattr(human, 'attributes', {}) or {}).get('human_world_anchor_source', 'unknown')),
            })
        artifact.metadata['projected_centers_by_frame'] = projected_by_frame

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

    def segment(self, ingest_artifact, geometry_artifact=None):
        """Detect+segment+track all instances → InstanceArtifact (masks + persistent ids).

        The shared mask + identity source of the perception band (design §4A). Runs a
        backend ladder; the FP1 floor rung is COCO Mask R-CNN + greedy-IoU association
        (`person`→human-kind, else object-kind). SAM 3 is the top rung at FP2. Always
        emits a schema-valid (possibly empty) artifact + a capability manifest.
        """
        objects_cfg = self.cfg.get('objects', {})
        selected_frames = self._select_perception_frames(ingest_artifact, None, objects_cfg, default_source='keyframes')
        floor = FloorInstanceSegmenter(self.decoder, self.object_detector, self.object_segmenter, objects_cfg)
        ladder = BackendLadder('segment', [
            BackendRung('maskrcnn_iou', 'per_frame_assoc', runner=lambda inp: floor.run(inp['ingest'], inp['frames'])),
            BackendRung('empty', 'empty', runner=lambda inp: ([], {})),
        ])
        result, manifest = ladder.run({'ingest': ingest_artifact, 'frames': selected_frames})
        observations, mask_arrays = result if result is not None else ([], {})
        artifact = InstanceArtifact(
            run_id=ingest_artifact.run_id,
            stage_name='segment',
            save_dir=self._stage_dir('segment'),
            observations=observations or [],
            mask_arrays=mask_arrays or {},
            metadata={
                'detector_backend': self.object_detector.backend_used,
                'selected_frame_count': int(len(selected_frames)),
            },
        )
        artifact.rebuild_indices()
        artifact.attach_manifest(manifest)
        artifact.notes.append(f"selected_frames={len(selected_frames)}")
        artifact.notes.append(f"instances={len(artifact.observations)} tracks={len(artifact.track_index)}")
        self._stamp_stage_metadata(artifact, 'segment', video_paths=self._video_paths_for_stage(ingest_artifact))
        artifact.save()
        return artifact

    def compute_reid(self, ingest_artifact, instance_artifact=None):
        """Per-tracklet appearance embeddings + cross-gap merges → ReidArtifact.

        Consumes the `InstanceArtifact` track ids. FP1 floor rung = mean ResNet18
        embeddings with an empty merge_map (no gap-merging yet); DINOv3 + real
        cross-occlusion merges arrive at FP3. Always emits a schema-valid artifact.
        """
        floor = FloorReid(self.decoder, self.object_embedder)
        ladder = BackendLadder('reid', [
            BackendRung('resnet18', 'imagenet_features', runner=lambda inp: floor.run(inp['ingest'], inp['instances'])),
            BackendRung('empty', 'empty', runner=lambda inp: {}),
        ])
        empty = InstanceArtifact(run_id=ingest_artifact.run_id, stage_name='segment', save_dir=self._stage_dir('segment'))
        embeddings, manifest = ladder.run({'ingest': ingest_artifact, 'instances': instance_artifact if instance_artifact is not None else empty})
        artifact = ReidArtifact(
            run_id=ingest_artifact.run_id,
            stage_name='reid',
            save_dir=self._stage_dir('reid'),
            embeddings=embeddings or {},
            merge_map={},
            metadata={'embedder_backend': self.object_embedder.backend_used},
        )
        artifact.attach_manifest(manifest)
        artifact.notes.append(f"embedded_tracks={len(artifact.embeddings)}")
        self._stamp_stage_metadata(artifact, 'reid', video_paths=self._video_paths_for_stage(ingest_artifact))
        artifact.save()
        return artifact

    def perceive_humans(self, ingest_artifact, instance_artifact=None, reid_artifact=None, geometry_artifact=None):
        """Build HumanArtifact from human tracklets: pose / DensePose / SMPL over segment masks.

        Consumes `segment` (human-kind instances + masks) and `reid` (canonical ids).
        Detection + identity now live upstream (§4A); the hand-tuned associator is gone.
        """
        humans_cfg = self.cfg.get('humans', {})
        densepose_stride = max(1, int(humans_cfg.get('densepose_frame_stride', 1)))
        smpl_stride = max(1, int(humans_cfg.get('smpl_frame_stride', 1)))
        stream_lookup = {stream.stream_id: stream for stream in ingest_artifact.streams}
        human_states: list[HumanState] = []
        mask_arrays: dict[str, np.ndarray] = {}
        densepose_payloads: dict[str, dict[str, Any]] = {}
        smpl_payloads: dict[str, dict[str, Any]] = {}

        def _canonical(track_id: str) -> str:
            return reid_artifact.canonical(track_id) if reid_artifact is not None else track_id

        human_obs = [] if instance_artifact is None else [o for o in instance_artifact.observations if o.kind == 'human']
        by_frame: dict[str, list] = {}
        for o in human_obs:
            by_frame.setdefault(InstanceArtifact._frame_key(o.frame_ref), []).append(o)
        ordered = sorted(by_frame.keys(), key=lambda k: (k.split(':')[0], int(k.split(':')[1])))
        for semantic_idx, frame_key in enumerate(ordered):
            obs_list = by_frame[frame_key]
            fr = obs_list[0].frame_ref
            stream = stream_lookup.get(fr.stream_id)
            if stream is None:
                continue
            frame = self.decoder.get_frame(stream, fr.frame_idx)
            boxes = [np.asarray(o.bbox_xyxy, dtype=np.float32) for o in obs_list]
            masks = [None if instance_artifact is None else instance_artifact.mask_arrays.get(o.instance_id) for o in obs_list]
            keypoints = self.human_pose.predict(frame, boxes)
            run_densepose = (semantic_idx % densepose_stride) == 0
            run_smpl = (semantic_idx % smpl_stride) == 0
            denseposes = self.human_densepose.predict(frame, boxes, masks=masks, keypoints=keypoints) if run_densepose else [None] * len(boxes)
            for det_idx, o in enumerate(obs_list):
                canonical = _canonical(o.track_id)
                mask = masks[det_idx] if det_idx < len(masks) else None
                if mask is not None:
                    mask_arrays[o.instance_id] = np.asarray(mask, dtype=bool)
                kpts = keypoints[det_idx] if det_idx < len(keypoints) else None
                dp = denseposes[det_idx] if det_idx < len(denseposes) else None
                if dp is not None:
                    densepose_payloads[o.instance_id] = dp
                smpl = self.human_smpl.fit_frame(frame, kpts, mask=mask, intrinsics=None) if run_smpl else None
                if smpl is not None:
                    smpl_payloads[o.instance_id] = smpl
                attrs = {
                    'detection_id': o.instance_id,
                    'track_id': canonical,
                    'track_ids': [],  # point-track support decoupled (§4A); empty at FP1
                    'frame_key': frame_key,
                    'semantic_frame_index': int(semantic_idx),
                    'densepose_computed': bool(run_densepose),
                    'smpl_computed': bool(run_smpl),
                    'pose_backend': self.human_pose.backend_used,
                    'densepose_backend': self.human_densepose.backend_used,
                    'smpl_backend': self.human_smpl.backend_used,
                }
                human_states.append(HumanState(
                    human_id=canonical,
                    frame_ref=o.frame_ref,
                    bbox_xyxy=np.asarray(o.bbox_xyxy, dtype=np.float32),
                    mask_rle=None if mask is None else {'shape': list(np.asarray(mask).shape), 'sum': int(np.asarray(mask, dtype=bool).sum())},
                    keypoints_2d=None if kpts is None else np.asarray(kpts, dtype=np.float32),
                    densepose=dp,
                    smpl_params=smpl,
                    confidence=float(o.confidence),
                    attributes=attrs,
                ))

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
            notes=[],
            metadata={
                'pose_backend': self.human_pose.backend_used,
                'densepose_backend': self.human_densepose.backend_used,
                'smpl_backend': self.human_smpl.backend_used,
                'save_debug_assets': bool(humans_cfg.get('save_debug_assets', humans_cfg.get('debug', False))),
                'save_debug_html': bool(humans_cfg.get('save_debug_html', humans_cfg.get('debug', False))),
                'debug_image_paths': {},
                'densepose_frame_stride': int(densepose_stride),
                'smpl_frame_stride': int(smpl_stride),
                'num_human_tracks': len({h.human_id for h in human_states}),
            },
        )
        artifact.frame_index = artifact._build_frame_index(artifact.humans)
        artifact.notes.append(f"human_instances={len(human_states)} human_tracks={artifact.metadata['num_human_tracks']}")
        self._attach_human_debug_projections(artifact, None, geometry_artifact)
        self._stamp_stage_metadata(artifact, 'humans', video_paths=self._video_paths_for_stage(ingest_artifact))
        artifact.save()
        return artifact

    def perceive_objects(self, ingest_artifact, instance_artifact=None, reid_artifact=None):
        """Build ObjectArtifact from non-human tracklets (consumes segment + reid).

        Carries the COCO category (floor semantics; SigLIP2/Qwen-VL open-vocab at FP4)
        and the per-track embedding. Identity is the canonical (reid-merged) track id;
        the hand-tuned associator is gone (§4A).
        """
        objects_cfg = self.cfg.get('objects', {})
        object_states: list[ObjectState] = []
        mask_arrays: dict[str, Any] = {}
        embedding_vectors: dict[str, Any] = {}

        def _canonical(track_id: str) -> str:
            return reid_artifact.canonical(track_id) if reid_artifact is not None else track_id

        observations = [] if instance_artifact is None else [o for o in instance_artifact.observations if o.kind != 'human']
        for o in observations:
            canonical = _canonical(o.track_id)
            mask = None if instance_artifact is None else instance_artifact.mask_arrays.get(o.instance_id)
            if mask is not None:
                mask_arrays[o.instance_id] = np.asarray(mask, dtype=bool)
            embedding = None if reid_artifact is None else reid_artifact.embeddings.get(canonical)
            if embedding is not None:
                embedding_vectors[canonical] = np.asarray(embedding, dtype=np.float32)
            attrs = {
                'detection_id': o.instance_id,
                'track_id': canonical,
                'track_ids': [],  # point-track support decoupled (§4A); empty at FP1
                'label_id': int(o.attributes.get('label_id', -1)),
                'frame_key': f"{o.frame_ref.stream_id}:{int(o.frame_ref.frame_idx)}",
                'detector_backend': (instance_artifact.metadata.get('detector_backend') if instance_artifact is not None else None),
            }
            object_states.append(ObjectState(
                object_id=canonical,
                frame_ref=o.frame_ref,
                category=str(o.concept),
                bbox_xyxy=np.asarray(o.bbox_xyxy, dtype=np.float32),
                mask_rle=None if mask is None else {'shape': list(np.asarray(mask).shape), 'sum': int(np.asarray(mask, dtype=bool).sum())},
                embedding=None if embedding is None else np.asarray(embedding, dtype=np.float32),
                confidence=float(o.confidence),
                attributes=attrs,
            ))

        artifact = ObjectArtifact(
            run_id=ingest_artifact.run_id,
            stage_name='objects',
            save_dir=self._stage_dir('objects'),
            objects=object_states,
            linked_ids={},
            frame_index={},
            mask_arrays=mask_arrays,
            embedding_vectors=embedding_vectors,
            notes=[],
            metadata={
                'detector_backend': self.object_detector.backend_used,
                'embedder_backend': self.object_embedder.backend_used,
                'save_debug_assets': bool(objects_cfg.get('save_debug_assets', objects_cfg.get('debug', False))),
                'save_debug_html': bool(objects_cfg.get('save_debug_html', objects_cfg.get('debug', False))),
                'debug_image_paths': {},
                'num_object_tracks': len({s.object_id for s in object_states}),
            },
        )
        artifact.frame_index = artifact._build_frame_index(artifact.objects)
        artifact.notes.append(f"object_instances={len(object_states)} object_tracks={artifact.metadata['num_object_tracks']}")
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
        optimization_cfg = self.cfg.get('optimization', {})
        artifact.metadata.update({
            'save_refinement_viewer': bool(optimization_cfg.get('save_refinement_viewer', False)),
            'debug_max_points': int(optimization_cfg.get('debug_max_points', 5000)),
            'debug_max_trails': int(optimization_cfg.get('debug_max_trails', 12)),
            'debug_trail_window': int(optimization_cfg.get('debug_trail_window', 12)),
        })
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

    def fit_renderer(self, scene_artifact, optimization_artifact=None, geometry_artifact=None, track_artifact=None, human_artifact=None):
        """Train + render the self-contained SceneArtifact (proven Gaussian trainer).

        The render stage consumes ONLY the SceneArtifact (dense seed + persisted
        train views + render trajectory). The optimization/geometry/track/human
        artifacts are accepted for stage-signature compatibility but unused here;
        masked-NVS scoring against held-out GT is the eval stage's job (F0 follow-up).
        """
        render_dir = self._stage_dir('render')
        artifact = self.renderer.fit(scene_artifact, save_dir=render_dir)
        # Masked-NVS scoring against held-out GT runs only on a benchmark dataset
        # (covisibility masks / held-out cameras don't exist for arbitrary video).
        eval_cfg = dict(self.cfg.get('evaluation', {}).get('masked_nvs', {}) or {})
        if eval_cfg.get('dataset_root'):
            from room4d.render.eval.masked_nvs_benchmark import score_render_artifact_masked_nvs
            device = str(self.cfg.get('render', {}).get('device', 'cuda'))
            metrics = score_render_artifact_masked_nvs(artifact, eval_cfg, device=device)
            if metrics:
                artifact.scene_state['evaluation'] = metrics
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
        elif stage_name == 'pose_refine':
            components = []  # the refiner holds no persistent backend; just release memory below
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
        # Pose-refined geometry feeds every downstream stage; the original
        # geometry artifact is still returned under 'geometry' for inspection.
        pose_geometry = self.refine_poses(ingest_artifact, geometry_artifact)
        # Perception band (§4A): segment+reid own identity; objects/humans enrich;
        # tracking is motion-only and now runs *after* perception.
        instance_artifact = self.segment(ingest_artifact, geometry_artifact=pose_geometry)
        reid_artifact = self.compute_reid(ingest_artifact, instance_artifact)
        object_artifact = self.perceive_objects(ingest_artifact, instance_artifact=instance_artifact, reid_artifact=reid_artifact)
        human_artifact = self.perceive_humans(ingest_artifact, instance_artifact=instance_artifact, reid_artifact=reid_artifact, geometry_artifact=pose_geometry)
        track_artifact = self.track(ingest_artifact, geometry_artifact=pose_geometry)
        graph_artifact = self.build_graph(pose_geometry, track_artifact, human_artifact=human_artifact, object_artifact=object_artifact)
        optimization_artifact = self.optimize(sync_artifact, pose_geometry, track_artifact, human_artifact=human_artifact, object_artifact=object_artifact, graph_artifact=graph_artifact)
        scene_artifact = self.assemble_scene(pose_geometry, track_artifact, human_artifact=human_artifact, object_artifact=object_artifact, optimization_artifact=optimization_artifact)
        render_artifact = self.fit_renderer(scene_artifact, optimization_artifact=optimization_artifact, geometry_artifact=pose_geometry, track_artifact=track_artifact, human_artifact=human_artifact)
        return {
            'ingest': ingest_artifact,
            'sync': sync_artifact,
            'geometry': geometry_artifact,
            'pose_refine': pose_geometry,
            'segment': instance_artifact,
            'reid': reid_artifact,
            'objects': object_artifact,
            'humans': human_artifact,
            'tracks': track_artifact,
            'graph': graph_artifact,
            'optimization': optimization_artifact,
            'scene': scene_artifact,
            'render': render_artifact,
        }
