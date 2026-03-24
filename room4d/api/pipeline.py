"""High-level pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from room4d.common.io import read_yaml
from room4d.common.paths import ensure_dir
from room4d.data.dataset import MultiVideoDataset
from room4d.geometry.vggt_wrapper import VGGTInitializer
from room4d.graph.builders import DynamicSceneGraphBuilder
from room4d.ingest.decoder import VideoDecoder
from room4d.ingest.ingest_artifact import IngestArtifact
from room4d.ingest.quality import ClipQualityAssessor
from room4d.lifting.backproject import TrackLifter3D
from room4d.lifting.static_dynamic import StaticDynamicPartitioner
from room4d.optimize.solver import JointSpatiotemporalSolver
from room4d.query.parser import PromptParser
from room4d.render.gaussian4d.scene_model import Gaussian4DScene
from room4d.render.gaussian4d.trainer import Gaussian4DTrainer
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
        self.geometry_init = VGGTInitializer(self.cfg.get('geometry', {}))
        self.tracker = CoTrackerWrapper(self.cfg.get('tracking', {}))
        lifting_cfg = self.cfg.get('lifting', {})
        self.track_lifter = TrackLifter3D(point_conf_threshold=float(lifting_cfg.get('point_conf_threshold', 0.0)))
        self.static_dynamic_partitioner = StaticDynamicPartitioner(
            static_threshold_world=float(lifting_cfg.get('static_threshold_world', 0.05)),
            min_lifted_frames=int(lifting_cfg.get('min_lifted_frames', 2)),
        )
        self.graph_builder = DynamicSceneGraphBuilder()
        self.solver = JointSpatiotemporalSolver(self.cfg.get('optimizer', {}))
        self.prompt_parser = PromptParser()
        self.renderer = Gaussian4DTrainer()

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'Room4DPipeline':
        """Build a pipeline instance from a YAML config file."""
        return cls(read_yaml(path))

    def _stage_dir(self, stage_name: str) -> Path:
        """Return the canonical save directory for a stage artifact."""
        return ensure_dir(self.run_dir / stage_name)

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
        artifact.save()
        return artifact

    def initialize_geometry(self, ingest_artifact: IngestArtifact, sync_artifact):
        """Run coarse geometry initialization and persist its heavy outputs."""
        artifact = self.geometry_init.initialize(
            ingest_artifact.keyframes,
            ingest_artifact=ingest_artifact,
            sync_artifact=sync_artifact,
        )
        artifact.save_dir = self._stage_dir('geometry')
        artifact.save()
        return artifact

    def track(self, ingest_artifact: IngestArtifact, geometry_artifact=None):
        """Run point tracking, 3D lifting, and static/dynamic partitioning."""
        frame_refs_by_stream: dict[str, list] = {stream_id: [] for stream_id in ingest_artifact.stream_ids()}
        for frame_ref in ingest_artifact.keyframes:
            frame_refs_by_stream.setdefault(frame_ref.stream_id, []).append(frame_ref)
        artifact = self.tracker.track_many(ingest_artifact.streams, frame_refs_by_stream=frame_refs_by_stream)
        artifact.run_id = ingest_artifact.run_id
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
        artifact.save()
        return artifact

    def perceive_humans(self, ingest_artifact):
        """Run human perception.

        TODO: wire in the human branch after the first geometry-tracking-graph
        slice is stable.
        """
        return None

    def perceive_objects(self, ingest_artifact):
        """Run object perception.

        TODO: wire in the generic object branch after the first vertical slice is
        stable.
        """
        return None

    def build_graph(self, geometry_artifact, track_artifact, human_artifact=None, object_artifact=None):
        """Build the dynamic scene graph."""
        artifact = self.graph_builder.build(geometry_artifact, track_artifact, human_artifact, object_artifact)
        artifact.save_dir = self._stage_dir('graph')
        artifact.save()
        return artifact

    def optimize(self, sync_artifact, geometry_artifact, track_artifact, human_artifact=None, object_artifact=None, graph_artifact=None):
        """Run staged joint optimization."""
        artifact = self.solver.solve_all(sync_artifact, geometry_artifact, track_artifact, human_artifact, object_artifact, graph_artifact)
        artifact.save_dir = self._stage_dir('optimization')
        artifact.save()
        return artifact

    def fit_renderer(self, optimization_artifact):
        """Fit a renderable 4D scene model."""
        scene = Gaussian4DScene()
        scene.initialize_from_optimization(optimization_artifact)
        artifact = self.renderer.fit(scene, optimization_artifact)
        artifact.save_dir = self._stage_dir('render')
        artifact.save()
        return artifact

    def run(self, video_paths: list[str]) -> dict[str, Any]:
        """Execute the current end-to-end vertical slice and persist artifacts."""
        ingest_artifact = self.preprocess(video_paths)
        sync_artifact = self.synchronize(ingest_artifact)
        geometry_artifact = self.initialize_geometry(ingest_artifact, sync_artifact)
        track_artifact = self.track(ingest_artifact, geometry_artifact=geometry_artifact)
        graph_artifact = self.build_graph(geometry_artifact, track_artifact)
        optimization_artifact = self.optimize(sync_artifact, geometry_artifact, track_artifact, graph_artifact=graph_artifact)
        render_artifact = self.fit_renderer(optimization_artifact)
        return {
            'ingest': ingest_artifact,
            'sync': sync_artifact,
            'geometry': geometry_artifact,
            'tracks': track_artifact,
            'graph': graph_artifact,
            'optimization': optimization_artifact,
            'render': render_artifact,
        }
