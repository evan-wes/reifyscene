"""High-level pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from typing import Any

from reifyscene.common.io import read_yaml
from reifyscene.calibration.runner import CalibrationRunner
from reifyscene.common.types import CameraPose, FrameRef, HumanState, ObjectState, TimedCameraState
from reifyscene.common.paths import ensure_dir
from reifyscene.common.memory import release_memory
from reifyscene.data.dataset import MultiVideoDataset
from reifyscene.geometry.camera_interpolation import densify_camera_states_for_frames
from reifyscene.geometry.geometry_artifact import GeometryArtifact
from reifyscene.geometry.pose_refine import PoseRefiner, PoseRefineUnavailable
from reifyscene.geometry.world_frame import GroundPlaneCanonicalizer
from reifyscene.geometry.vggt_wrapper import VGGTInitializer
from reifyscene.geometry.vggt_long_wrapper import VGGTLongInitializer
from reifyscene.geometry.dataset_wrapper import DatasetGeometryInitializer
from reifyscene.geometry.moge_wrapper import MoGeInitializer
from reifyscene.geometry.colmap_wrapper import ColmapPoseInitializer
from reifyscene.geometry.compose import GeometryComposer
from reifyscene.perception.dyn_mask.dyn_mask_artifact import DynMaskArtifact
from reifyscene.perception.dyn_mask.rungs import AllStaticDynMask, DetectorDynMask
from reifyscene.graph.builders import DynamicSceneGraphBuilder
from reifyscene.ingest.decoder import VideoDecoder
from reifyscene.ingest.ingest_artifact import IngestArtifact
from reifyscene.ingest.quality import ClipQualityAssessor
from reifyscene.lifting.backproject import TrackLifter3D
from reifyscene.lifting.static_dynamic import StaticDynamicPartitioner
from reifyscene.optimize.solver import JointSpatiotemporalSolver
from reifyscene.perception.humans.detector import HumanDetector
from reifyscene.perception.humans.pose2d import HumanPose2DEstimator
from reifyscene.perception.humans.densepose_wrapper import DensePoseEstimator
from reifyscene.perception.humans.smpl_fitter import SMPLFitter
from reifyscene.perception.humans.dense_ladder import BboxOnlyRung, KeypointRcnnRung, SapiensDenseRung
from reifyscene.perception.humans.sapiens_wrapper import sapiens_cached
from reifyscene.perception.humans.human_artifact import HumanArtifact
from reifyscene.perception.objects.detector import ObjectDetector
from reifyscene.perception.objects.embedder import InstanceEmbedder
from reifyscene.perception.objects.object_artifact import ObjectArtifact
from reifyscene.perception.objects.segmenter import ObjectSegmenter
from reifyscene.perception.instance.instance_artifact import InstanceArtifact, InstanceObservation
from reifyscene.perception.instance.segment import FloorInstanceSegmenter
from reifyscene.perception.instance.sam2_segment import SAM2InstanceSegmenter
from reifyscene.perception.reid.reid_artifact import ReidArtifact
from reifyscene.perception.reid.reid_stage import FloorReid
from reifyscene.perception.reid.dino_reid import DinoReid
from reifyscene.perception.semantics.semantics_artifact import SemanticsArtifact
from reifyscene.perception.semantics.siglip_semantics import FloorSemantics, SiglipSemantics
from reifyscene.common.backend_ladder import BackendLadder, BackendRung, CapabilityManifest, RungStatus, importable
from reifyscene.query.parser import PromptParser
from reifyscene.api.resume import apply_stage_compatibility_metadata
from reifyscene.api.stages import build_stage_registry
from reifyscene.render.registry import build_renderer
from reifyscene.scene.assembler import SingleVideoSceneAssembler
from reifyscene.eval.benchmark_runner import BenchmarkRunner
from reifyscene.sync.global_time_solver import GlobalTimeAlignmentSolver
from reifyscene.tracking.cotracker_wrapper import CoTrackerWrapper


# Fidelity tag each geometry backend carries in its CapabilityManifest (design §4A.1).
# Joint rungs (vggt*/dataset) fill poses+depth; moge is depth-only, colmap pose-only.
_GEOMETRY_FIDELITY = {
    'moge-2': 'metric_depth_no_poses',
    'colmap': 'sfm_poses_no_depth',
    'vggt': 'joint_reconstruction',
    'vggt-long': 'joint_reconstruction_longseq',
    'dataset': 'dataset_gt',
}


class ReifyScenePipeline:
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
        self.geometry_backend_name = geometry_backend_name
        # FP6b step 2: geometry is two fused ladders (§4A.1). When both `pose_backend`
        # and `depth_backend` are set the stage runs a *pose* channel + a *depth* channel
        # and composes them (COLMAP↔MoGe scale reconciliation); otherwise a single joint
        # backend fills both channels (today's path — strict superset, unchanged).
        self.geometry_pose_backend = str(geometry_cfg.get('pose_backend', '')).lower() or None
        self.geometry_depth_backend = str(geometry_cfg.get('depth_backend', '')).lower() or None
        self.geometry_fused = bool(self.geometry_pose_backend and self.geometry_depth_backend)
        if self.geometry_fused:
            self.geometry_pose_init = self._build_geometry_backend(self.geometry_pose_backend, geometry_cfg)
            self.geometry_depth_init = self._build_geometry_backend(self.geometry_depth_backend, geometry_cfg)
            self.geometry_composer = GeometryComposer(geometry_cfg.get('compose', {}))
            self.geometry_init = self.geometry_pose_init  # nominal top rung for shared helpers
        else:
            self.geometry_init = self._build_geometry_backend(geometry_backend_name, geometry_cfg)
        # FP6b step 3: coarse geometry-free dyn_mask stage (protects the SfM pose rung).
        # Default 'all-static' → a no-op that leaves the pose rung unmasked (VGGT-on-top,
        # or clips that don't need it); 'detector' → COCO Mask R-CNN movable-category masks.
        dyn_mask_cfg = self.cfg.get('dyn_mask', {})
        self.dyn_mask_backend_name = str(dyn_mask_cfg.get('backend_name', 'all-static')).lower()
        self.dyn_mask_init = (
            DetectorDynMask(dyn_mask_cfg) if self.dyn_mask_backend_name == 'detector'
            else AllStaticDynMask(dyn_mask_cfg)
        )
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

    @staticmethod
    def _build_geometry_backend(name: str, geometry_cfg: dict):
        """Construct a geometry initializer for a backend name.

        Shared by the single-backend path and the fused pose/depth channels (§4A.1):
        ``vggt``/``vggt-long``/``dataset`` are *joint* rungs (poses+depth); ``moge`` is a
        depth-only rung; ``colmap`` a pose-only rung. Unknown names fall through to VGGT.
        """
        name = str(name).lower()
        if name == 'vggt-long':
            return VGGTLongInitializer(geometry_cfg)
        if name == 'dataset':
            return DatasetGeometryInitializer(geometry_cfg)
        if name == 'moge':
            # FP6a: MoGe-2 metric-depth rung (native swap; depth-only, no poses).
            return MoGeInitializer(geometry_cfg)
        if name == 'colmap':
            # FP6b step 1: COLMAP-SfM pose rung (native swap; pose-only, no dense depth).
            return ColmapPoseInitializer(geometry_cfg)
        return VGGTInitializer(geometry_cfg)

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'ReifyScenePipeline':
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
            all_frames=bool(self.cfg.get('keyframes_all', False)),
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

    def compute_dyn_mask(self, ingest_artifact: IngestArtifact, sync_artifact=None) -> DynMaskArtifact:
        """FP6b step 3: coarse **geometry-free** dynamic masks that protect the SfM pose rung.

        Runs the dyn_mask ladder (selected backend → all-static floor) on the **same** frame
        set the geometry pose rung will use (so mask keys align), stamps a `CapabilityManifest`,
        and saves. Default `all-static` is a cheap no-op; only the COLMAP pose rung consumes
        the result, and only size-gated. Consumes only raw frames — never `segment`/geometry —
        which is what breaks the segment→geometry cycle (§4A.1).
        """
        selected_frames = self._select_geometry_frames(ingest_artifact)
        rung_name = getattr(self.dyn_mask_init, 'backend_name', self.dyn_mask_backend_name)
        probe = {'detector': importable('torchvision')}.get(rung_name)
        dyn_cfg = self.cfg.get('dyn_mask', {})
        inputs = {'frames': selected_frames, 'ingest': ingest_artifact, 'sync': sync_artifact}
        ladder = BackendLadder('dyn_mask', [
            BackendRung(
                rung_name, rung_name,
                runner=lambda inp: self.dyn_mask_init.initialize(
                    inp['frames'], ingest_artifact=inp['ingest'], sync_artifact=inp['sync']),
                probe=probe,
            ),
            BackendRung(
                'all-static', 'all-static',
                runner=lambda inp: AllStaticDynMask(dyn_cfg).initialize(
                    inp['frames'], ingest_artifact=inp['ingest'], sync_artifact=inp['sync']),
            ),
        ])
        artifact, manifest = ladder.run(inputs)
        artifact.attach_manifest(manifest)
        artifact.save_dir = self._stage_dir('dyn_mask')
        self._stamp_stage_metadata(artifact, 'dyn_mask', video_paths=self._video_paths_for_stage(ingest_artifact))
        artifact.save()
        return artifact

    @staticmethod
    def _geometry_init_call(init, inp):
        """Call a geometry backend's ``initialize``, passing ``dyn_mask`` only to the pose
        rung that consumes it (fourth invariant: dyn_mask is the COLMAP rung's dep; other
        rungs never see it, so their signatures stay untouched)."""
        kwargs = dict(ingest_artifact=inp['ingest'], sync_artifact=inp['sync'])
        if getattr(init, 'backend_name', None) == 'colmap':
            kwargs['dyn_mask_artifact'] = inp.get('dyn_mask')
        return init.initialize(inp['frames'], **kwargs)

    def _geometry_ladder(self, stage_label, init, backend_name, selected_frames,
                         ingest_artifact: IngestArtifact, sync_artifact, *, channel: str | None = None,
                         dyn_mask_artifact=None):
        """Run one geometry channel as a 2-rung ladder (backend → graceful-empty floor).

        Rung name = the backend's own identity so ``manifest.backend_used`` matches the
        artifact's ``backend_name``. Model-backed rungs are probe-gated (verified package
        names) so a missing dep degrades cleanly; the load-bearing vggt/dataset rungs stay
        probe-free and rely on the ladder's run-error → graceful-empty path.
        """
        rung_name = getattr(init, 'backend_name', backend_name)
        probe_by_backend = {'moge-2': importable('moge'), 'colmap': importable('pycolmap', 'hloc')}
        floor_note = (
            'geometry graceful-empty floor (§4A invariant 2)' if channel is None
            else f'geometry {channel} graceful-empty floor (§4A invariant 2)'
        )
        inputs = {'frames': selected_frames, 'ingest': ingest_artifact, 'sync': sync_artifact,
                  'dyn_mask': dyn_mask_artifact}
        ladder = BackendLadder(stage_label, [
            BackendRung(
                rung_name,
                _GEOMETRY_FIDELITY.get(rung_name, rung_name),
                runner=lambda inp: self._geometry_init_call(init, inp),
                probe=probe_by_backend.get(rung_name),
            ),
            BackendRung(
                'empty', 'empty',
                runner=lambda inp: GeometryArtifact(
                    run_id=ingest_artifact.run_id, stage_name='geometry',
                    save_dir=self._stage_dir('geometry'), backend_name='empty',
                    selected_frames=[], notes=[floor_note]),
            ),
        ])
        return ladder.run(inputs)

    def _run_fused_geometry(self, selected_frames, ingest_artifact: IngestArtifact, sync_artifact, dyn_mask_artifact=None):
        """FP6b two fused ladders (§4A.1): run a pose channel + a depth channel on the
        SAME frames, then compose them (COLMAP↔MoGe scale reconciliation).

        The composite `CapabilityManifest` records both channels' rung statuses (prefixed
        ``pose:`` / ``depth:``) and the fused ``backend_used`` / fidelity the composer
        resolved to (`fused_metric` / `poses_only` / `depth_only` / `empty`).
        """
        pose_art, pose_manifest = self._geometry_ladder(
            'geometry.pose', self.geometry_pose_init, self.geometry_pose_backend,
            selected_frames, ingest_artifact, sync_artifact, channel='pose',
            dyn_mask_artifact=dyn_mask_artifact)
        depth_art, depth_manifest = self._geometry_ladder(
            'geometry.depth', self.geometry_depth_init, self.geometry_depth_backend,
            selected_frames, ingest_artifact, sync_artifact, channel='depth')
        # Use the rungs' own identities ('colmap' / 'moge-2') so the composite backend
        # name matches the manifest's `pose:`/`depth:` rung names.
        pose_id = getattr(self.geometry_pose_init, 'backend_name', self.geometry_pose_backend)
        depth_id = getattr(self.geometry_depth_init, 'backend_name', self.geometry_depth_backend)
        result = self.geometry_composer.compose(
            pose_artifact=pose_art, depth_artifact=depth_art, shared_frames=selected_frames,
            pose_backend=pose_id, depth_backend=depth_id,
            run_id=ingest_artifact.run_id, save_dir=self._stage_dir('geometry'))
        manifest = CapabilityManifest(stage='geometry')
        for channel, channel_manifest in (('pose', pose_manifest), ('depth', depth_manifest)):
            for rung in channel_manifest.rungs:
                manifest.rungs.append(RungStatus(f'{channel}:{rung.name}', rung.fidelity, rung.status, rung.reason))
        manifest.backend_used = result.artifact.backend_name
        manifest.fidelity = result.fidelity
        if result.reconcile is not None:
            manifest.notes.append(result.reconcile.note())
        return result.artifact, manifest

    def initialize_geometry(self, ingest_artifact: IngestArtifact, sync_artifact, dyn_mask_artifact=None):
        """Run coarse geometry initialization and persist its heavy outputs.

        The config-selected backend is the ladder's top rung — geometry joins the
        observable-degradation machinery (design §4A.1): a graceful-empty floor
        guarantees a schema-valid artifact if the backend is unavailable or fails,
        and a `CapabilityManifest` records which rung ran, at what fidelity, and why
        higher rungs were skipped. `backend_name='moge'` force-selects the FP6a
        MoGe-2 depth rung. When `pose_backend`+`depth_backend` are set the stage runs
        the FP6b two fused ladders (COLMAP poses + MoGe depth, scale-reconciled). The
        FP6b step-3 `dyn_mask` (if provided) protects the COLMAP pose rung, size-gated.
        """
        selected_frames = self._select_geometry_frames(ingest_artifact)
        if self.geometry_fused:
            artifact, manifest = self._run_fused_geometry(selected_frames, ingest_artifact, sync_artifact, dyn_mask_artifact)
        else:
            artifact, manifest = self._geometry_ladder(
                'geometry', self.geometry_init, self.geometry_backend_name,
                selected_frames, ingest_artifact, sync_artifact, dyn_mask_artifact=dyn_mask_artifact)
        artifact.attach_manifest(manifest)
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
            # Keep BA's refined focal downstream: BA optimizes poses and the shared
            # focal jointly, so the refined poses are only consistent with the
            # refined K -- pairing them with a different (e.g. depth-native) focal
            # introduces a pose<->K mismatch that empirically hurts NVS more than
            # the depth<->K seed distortion (which 5000 training iters absorb).
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
            # SAM 3 (gated) is the eventual top rung (design §4A); SAM 2 is the
            # ungated stand-in: prompt-free class-agnostic discovery + video
            # propagation. The segmenter is built inside the runner so SAM 2 is
            # imported + loaded only when this rung is actually selected (and the
            # model is released after the stage). Falls to the Mask R-CNN floor
            # when `sam2` is not installed.
            BackendRung(
                'sam2', 'class_agnostic_propagation',
                runner=lambda inp: SAM2InstanceSegmenter(self.decoder, self.object_detector, objects_cfg).run(inp['ingest'], inp['frames']),
                probe=importable('sam2'),
            ),
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

        Consumes the `InstanceArtifact` track ids. Runs a backend ladder: the
        DINOv2-L top rung (FP3) emits DINO features + a cross-occlusion merge_map;
        the ResNet18 floor rung emits mean embeddings with an empty merge_map.
        DINOv3-L swaps into the top rung by model_id once HF access lands. Always
        emits a schema-valid artifact.
        """
        floor = FloorReid(self.decoder, self.object_embedder)
        reid_cfg = self.cfg.get('reid', {})
        ladder = BackendLadder('reid', [
            # DINOv2-L (ungated, Apache) appearance features + cross-occlusion gap
            # merges, above the ResNet18 floor; DINOv3-L swaps in by model_id once
            # HF access lands. Built inside the runner so transformers is imported
            # + the model loaded only when this rung is selected. If the weights
            # can't load, run() raises and the ladder degrades to the floor.
            BackendRung(
                'dinov2', 'dino_features',
                runner=lambda inp: DinoReid(self.decoder, reid_cfg).run(inp['ingest'], inp['instances']),
                probe=importable('transformers'),
            ),
            BackendRung('resnet18', 'imagenet_features', runner=lambda inp: floor.run(inp['ingest'], inp['instances'])),
            BackendRung('empty', 'empty', runner=lambda inp: ({}, {})),
        ])
        empty = InstanceArtifact(run_id=ingest_artifact.run_id, stage_name='segment', save_dir=self._stage_dir('segment'))
        result, manifest = ladder.run({'ingest': ingest_artifact, 'instances': instance_artifact if instance_artifact is not None else empty})
        embeddings, merge_map = result if result is not None else ({}, {})
        artifact = ReidArtifact(
            run_id=ingest_artifact.run_id,
            stage_name='reid',
            save_dir=self._stage_dir('reid'),
            embeddings=embeddings or {},
            merge_map=merge_map or {},
            metadata={'embedder_backend': self.object_embedder.backend_used},
        )
        artifact.attach_manifest(manifest)
        artifact.notes.append(f"embedded_tracks={len(artifact.embeddings)}")
        self._stamp_stage_metadata(artifact, 'reid', video_paths=self._video_paths_for_stage(ingest_artifact))
        artifact.save()
        return artifact

    def compute_semantics(self, ingest_artifact, instance_artifact=None, reid_artifact=None):
        """Open-vocab category (+ SigLIP embedding) per canonical tracklet → SemanticsArtifact.

        Consumes `segment` masks + `reid` canonical ids. Runs a backend ladder: the
        SigLIP2 top rung (FP4a) zero-shot names each tracklet from its masked crops;
        the COCO-80 floor rung carries the detector's `concept` string, degrading to
        "unknown". Qwen2.5-VL attributes/captions arrive at FP4b. Always emits a
        schema-valid artifact.
        """
        semantics_cfg = self.cfg.get('semantics', {})
        empty = InstanceArtifact(run_id=ingest_artifact.run_id, stage_name='segment', save_dir=self._stage_dir('segment'))
        instances = instance_artifact if instance_artifact is not None else empty
        floor = FloorSemantics(reid_artifact)
        ladder = BackendLadder('semantics', [
            # SigLIP2 open-vocab category + image embedding. Built inside the runner so
            # transformers is imported + the model loaded only when this rung is
            # selected; if the weights can't load, run() raises and the ladder degrades
            # to the COCO-80 detector floor (then to "unknown").
            BackendRung(
                'siglip2', 'open_vocab',
                runner=lambda inp: SiglipSemantics(self.decoder, reid_artifact, semantics_cfg).run(inp['ingest'], inp['instances']),
                probe=importable('transformers'),
            ),
            BackendRung('coco80', 'closed_vocab', runner=lambda inp: floor.run(inp['ingest'], inp['instances'])),
            BackendRung('unknown', 'empty', runner=lambda inp: ({}, {})),
        ])
        result, manifest = ladder.run({'ingest': ingest_artifact, 'instances': instances})
        records, embeddings = result if result is not None else ({}, {})
        artifact = SemanticsArtifact(
            run_id=ingest_artifact.run_id,
            stage_name='semantics',
            save_dir=self._stage_dir('semantics'),
            records=records or {},
            embeddings=embeddings or {},
            metadata={},
        )
        artifact.attach_manifest(manifest)
        named = sum(1 for r in artifact.records.values() if r.category != 'unknown')
        artifact.notes.append(f"named_tracks={named}/{len(artifact.records)}")
        self._stamp_stage_metadata(artifact, 'semantics', video_paths=self._video_paths_for_stage(ingest_artifact))
        artifact.save()
        return artifact

    def compute_humans(self, ingest_artifact, instance_artifact=None, reid_artifact=None, geometry_artifact=None):
        """Build HumanArtifact via a dense backend ladder (consumes segment + reid).

        Dense ladder [sapiens -> keypoint_rcnn -> bbox_only]: Sapiens (FP5a) gives
        pose + body-part seg + depth + normals per person crop; the Keypoint-RCNN
        floor gives honest pose only; bbox_only is the graceful floor. SMPL stays an
        honest skip (``smpl_params=None``; GVHMR at FP5b) and DensePose stays a
        kept-but-unwired seam. Identity is the canonical (reid-merged) track id;
        always emits a schema-valid artifact + capability manifest.
        """
        humans_cfg = self.cfg.get('humans', {})
        empty = InstanceArtifact(run_id=ingest_artifact.run_id, stage_name='segment', save_dir=self._stage_dir('segment'))
        instances = instance_artifact if instance_artifact is not None else empty
        sapiens_cfg = {**humans_cfg}

        def _sapiens_probe():
            if not humans_cfg.get('use_sapiens', True):
                return 'sapiens disabled (humans.use_sapiens=false)'
            return sapiens_cached(str(humans_cfg.get('sapiens_size', '0.6b')))

        ladder = BackendLadder('humans', [
            # Sapiens dense (pose+seg+depth+normals), built inside the runner so the
            # torchscript models load only when this rung is selected; gated by the
            # checkpoint-presence probe -> degrades to the Keypoint-RCNN floor.
            BackendRung('sapiens', 'sapiens_dense',
                        runner=lambda inp: SapiensDenseRung(self.decoder, sapiens_cfg).run(inp['ingest'], inp['instances']),
                        probe=_sapiens_probe),
            BackendRung('keypoint_rcnn', 'pose_only',
                        runner=lambda inp: KeypointRcnnRung(self.decoder, self.human_pose).run(inp['ingest'], inp['instances'])),
            BackendRung('bbox_only', 'bbox_only',
                        runner=lambda inp: BboxOnlyRung(self.decoder).run(inp['ingest'], inp['instances'])),
        ])
        result, manifest = ladder.run({'ingest': ingest_artifact, 'instances': instances})
        dense_by_instance = result if result is not None else {}

        def _canonical(track_id: str) -> str:
            return reid_artifact.canonical(track_id) if reid_artifact is not None else track_id

        human_obs = [o for o in instances.observations if o.kind == 'human']
        human_states: list[HumanState] = []
        mask_arrays: dict[str, Any] = {}
        dense_payloads: dict[str, dict[str, Any]] = {}
        for o in human_obs:
            canonical = _canonical(o.track_id)
            mask = instances.mask_arrays.get(o.instance_id)
            if mask is not None:
                mask_arrays[o.instance_id] = np.asarray(mask, dtype=bool)
            rec = dense_by_instance.get(o.instance_id) or {}
            kpts = rec.get('keypoints_2d')
            seg, depth, normals = rec.get('seg'), rec.get('depth'), rec.get('normals')
            dense = None
            if any(v is not None for v in (seg, depth, normals)):
                dense = {'seg': seg, 'depth': depth, 'normals': normals}
                dense_payloads[o.instance_id] = {k: v for k, v in dense.items() if v is not None}
            attrs = {
                'detection_id': o.instance_id,
                'track_id': canonical,
                'track_ids': [],  # point-track support decoupled (§4A)
                'frame_key': f"{o.frame_ref.stream_id}:{int(o.frame_ref.frame_idx)}",
                'dense_backend': rec.get('backend', manifest.backend_used),
            }
            human_states.append(HumanState(
                human_id=canonical,
                frame_ref=o.frame_ref,
                bbox_xyxy=np.asarray(o.bbox_xyxy, dtype=np.float32),
                mask_rle=None if mask is None else {'shape': list(np.asarray(mask).shape), 'sum': int(np.asarray(mask, dtype=bool).sum())},
                keypoints_2d=None if kpts is None else np.asarray(kpts, dtype=np.float32),
                densepose=None,
                smpl_params=None,
                confidence=float(o.confidence),
                attributes=attrs,
                dense=dense,
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
            densepose_payloads={},
            smpl_payloads={},
            dense_payloads=dense_payloads,
            notes=[],
            metadata={
                'dense_backend': manifest.backend_used,
                'save_debug_assets': bool(humans_cfg.get('save_debug_assets', humans_cfg.get('debug', False))),
                'save_debug_html': bool(humans_cfg.get('save_debug_html', humans_cfg.get('debug', False))),
                'debug_image_paths': {},
                'num_human_tracks': len({h.human_id for h in human_states}),
            },
        )
        artifact.attach_manifest(manifest)
        artifact.frame_index = artifact._build_frame_index(artifact.humans)
        artifact.notes.append(f"human_instances={len(human_states)} human_tracks={artifact.metadata['num_human_tracks']}")
        self._stamp_stage_metadata(artifact, 'humans', video_paths=self._video_paths_for_stage(ingest_artifact))
        artifact.save()
        return artifact

    def perceive_objects(self, ingest_artifact, instance_artifact=None, reid_artifact=None, semantics_artifact=None):
        """Build ObjectArtifact from non-human tracklets (consumes segment + reid + semantics).

        Category is the open-vocab label from the `semantics` stage (SigLIP2 at FP4a),
        falling back to the COCO `concept` string when semantics is absent/"unknown".
        Identity is the canonical (reid-merged) track id; the per-track embedding and
        the deleted hand-tuned associator are unchanged (§4A).
        """
        objects_cfg = self.cfg.get('objects', {})
        object_states: list[ObjectState] = []
        mask_arrays: dict[str, Any] = {}
        embedding_vectors: dict[str, Any] = {}

        def _canonical(track_id: str) -> str:
            return reid_artifact.canonical(track_id) if reid_artifact is not None else track_id

        def _semantic(canonical: str, concept: str):
            """Open-vocab (category, topk) from the semantics stage; COCO-concept fallback."""
            if semantics_artifact is not None:
                rec = semantics_artifact.records.get(canonical)
                if rec is not None and rec.category and rec.category != 'unknown':
                    return rec.category, list(rec.category_topk)
            return str(concept), []

        observations = [] if instance_artifact is None else [o for o in instance_artifact.observations if o.kind != 'human']
        for o in observations:
            canonical = _canonical(o.track_id)
            mask = None if instance_artifact is None else instance_artifact.mask_arrays.get(o.instance_id)
            if mask is not None:
                mask_arrays[o.instance_id] = np.asarray(mask, dtype=bool)
            embedding = None if reid_artifact is None else reid_artifact.embeddings.get(canonical)
            if embedding is not None:
                embedding_vectors[canonical] = np.asarray(embedding, dtype=np.float32)
            category, semantic_topk = _semantic(canonical, o.concept)
            attrs = {
                'detection_id': o.instance_id,
                'track_id': canonical,
                'track_ids': [],  # point-track support decoupled (§4A); empty at FP1
                'label_id': int(o.attributes.get('label_id', -1)),
                'frame_key': f"{o.frame_ref.stream_id}:{int(o.frame_ref.frame_idx)}",
                'detector_backend': (instance_artifact.metadata.get('detector_backend') if instance_artifact is not None else None),
                'semantic_topk': semantic_topk,
            }
            object_states.append(ObjectState(
                object_id=canonical,
                frame_ref=o.frame_ref,
                category=category,
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

    def assemble_scene(self, geometry_artifact, track_artifact, human_artifact=None, object_artifact=None, optimization_artifact=None, instance_artifact=None):
        """Assemble a lightweight static/dynamic single-video 4D scene.

        ``instance_artifact`` (segment stage) supplies the per-frame foreground
        masks the dynamic (``scene.kind=dynamic``) path lifts SE(3) motion from.
        """
        artifact = self.scene_assembler.assemble(
            run_id=geometry_artifact.run_id,
            save_dir=self._stage_dir('scene'),
            geometry_artifact=geometry_artifact,
            track_artifact=track_artifact,
            human_artifact=human_artifact,
            object_artifact=object_artifact,
            instance_artifact=instance_artifact,
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
            device = str(self.cfg.get('render', {}).get('device', 'cuda'))
            if str(artifact.scene_state.get('kind', 'static')) == 'dynamic':
                from reifyscene.render.eval.masked_nvs_benchmark import score_render_artifact_masked_nvs_dynamic
                metrics = score_render_artifact_masked_nvs_dynamic(artifact, eval_cfg, device=device)
            else:
                from reifyscene.render.eval.masked_nvs_benchmark import score_render_artifact_masked_nvs
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
        # FP6b step 3: coarse geometry-free dyn_mask (protects the SfM pose rung), between
        # sync and geometry per the static-superset order (§4A.1). No-op unless enabled.
        dyn_mask_artifact = self.compute_dyn_mask(ingest_artifact, sync_artifact)
        geometry_artifact = self.initialize_geometry(ingest_artifact, sync_artifact, dyn_mask_artifact=dyn_mask_artifact)
        # Pose-refined geometry feeds every downstream stage; the original
        # geometry artifact is still returned under 'geometry' for inspection.
        pose_geometry = self.refine_poses(ingest_artifact, geometry_artifact)
        # Perception band (§4A): segment+reid own identity; objects/humans enrich;
        # tracking is motion-only and now runs *after* perception.
        instance_artifact = self.segment(ingest_artifact, geometry_artifact=pose_geometry)
        reid_artifact = self.compute_reid(ingest_artifact, instance_artifact)
        object_artifact = self.perceive_objects(ingest_artifact, instance_artifact=instance_artifact, reid_artifact=reid_artifact)
        human_artifact = self.compute_humans(ingest_artifact, instance_artifact=instance_artifact, reid_artifact=reid_artifact, geometry_artifact=pose_geometry)
        track_artifact = self.track(ingest_artifact, geometry_artifact=pose_geometry)
        graph_artifact = self.build_graph(pose_geometry, track_artifact, human_artifact=human_artifact, object_artifact=object_artifact)
        optimization_artifact = self.optimize(sync_artifact, pose_geometry, track_artifact, human_artifact=human_artifact, object_artifact=object_artifact, graph_artifact=graph_artifact)
        scene_artifact = self.assemble_scene(pose_geometry, track_artifact, human_artifact=human_artifact, object_artifact=object_artifact, optimization_artifact=optimization_artifact, instance_artifact=instance_artifact)
        render_artifact = self.fit_renderer(scene_artifact, optimization_artifact=optimization_artifact, geometry_artifact=pose_geometry, track_artifact=track_artifact, human_artifact=human_artifact)
        return {
            'ingest': ingest_artifact,
            'sync': sync_artifact,
            'dyn_mask': dyn_mask_artifact,
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
