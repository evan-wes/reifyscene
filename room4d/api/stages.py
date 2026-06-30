"""Stage registry and resume helpers for room4d pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from room4d.common.io import write_json
from room4d.common.paths import ensure_dir
from room4d.geometry.geometry_artifact import GeometryArtifact
from room4d.graph.graph_artifact import GraphArtifact
from room4d.ingest.ingest_artifact import IngestArtifact
from room4d.optimize.optimization_artifact import OptimizationArtifact
from room4d.perception.humans.human_artifact import HumanArtifact
from room4d.perception.instance.instance_artifact import InstanceArtifact
from room4d.perception.objects.object_artifact import ObjectArtifact
from room4d.perception.reid.reid_artifact import ReidArtifact
from room4d.perception.semantics.semantics_artifact import SemanticsArtifact
from room4d.render.gaussian4d.render_artifact import RenderArtifact
from room4d.scene.scene_artifact import SceneArtifact
from room4d.sync.sync_artifact import SyncArtifact
from room4d.tracking.track_artifact import TrackArtifact


@dataclass(frozen=True)
class StageSpec:
    """Declarative description of a pipeline stage."""

    name: str
    dependencies: tuple[str, ...]
    artifact_cls: type
    run_fn: Callable[[Any, dict[str, Any], list[str]], Any]
    load_fn: Callable[[Path], Any]

    @property
    def dir_name(self) -> str:
        return 'tracking' if self.name == 'tracking' else self.name


STAGE_ORDER: tuple[str, ...] = (
    'ingest',
    'sync',
    'geometry',
    'pose_refine',
    'segment',
    'reid',
    'semantics',
    'objects',
    'humans',
    'tracking',
    'graph',
    'optimization',
    'scene',
    'render',
)


def build_stage_registry() -> dict[str, StageSpec]:
    """Return the canonical stage registry for room4d runs."""
    return {
        'ingest': StageSpec(
            name='ingest',
            dependencies=(),
            artifact_cls=IngestArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.preprocess(video_paths),
            load_fn=IngestArtifact.load,
        ),
        'sync': StageSpec(
            name='sync',
            dependencies=('ingest',),
            artifact_cls=SyncArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.synchronize(outputs['ingest']),
            load_fn=SyncArtifact.load,
        ),
        'geometry': StageSpec(
            name='geometry',
            dependencies=('ingest', 'sync'),
            artifact_cls=GeometryArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.initialize_geometry(outputs['ingest'], outputs['sync']),
            load_fn=GeometryArtifact.load,
        ),
        'pose_refine': StageSpec(
            name='pose_refine',
            dependencies=('ingest', 'geometry'),
            artifact_cls=GeometryArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.refine_poses(outputs['ingest'], outputs['geometry']),
            load_fn=GeometryArtifact.load,
        ),
        'segment': StageSpec(
            name='segment',
            dependencies=('ingest', 'pose_refine'),
            artifact_cls=InstanceArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.segment(outputs['ingest'], geometry_artifact=outputs['pose_refine']),
            load_fn=InstanceArtifact.load,
        ),
        'reid': StageSpec(
            name='reid',
            dependencies=('ingest', 'segment'),
            artifact_cls=ReidArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.compute_reid(outputs['ingest'], outputs['segment']),
            load_fn=ReidArtifact.load,
        ),
        'semantics': StageSpec(
            name='semantics',
            dependencies=('ingest', 'segment', 'reid'),
            artifact_cls=SemanticsArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.compute_semantics(outputs['ingest'], outputs['segment'], outputs['reid']),
            load_fn=SemanticsArtifact.load,
        ),
        'objects': StageSpec(
            name='objects',
            dependencies=('ingest', 'segment', 'reid', 'semantics'),
            artifact_cls=ObjectArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.perceive_objects(outputs['ingest'], instance_artifact=outputs['segment'], reid_artifact=outputs['reid'], semantics_artifact=outputs['semantics']),
            load_fn=ObjectArtifact.load,
        ),
        'humans': StageSpec(
            name='humans',
            dependencies=('ingest', 'segment', 'reid', 'pose_refine'),
            artifact_cls=HumanArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.perceive_humans(outputs['ingest'], instance_artifact=outputs['segment'], reid_artifact=outputs['reid'], geometry_artifact=outputs['pose_refine']),
            load_fn=HumanArtifact.load,
        ),
        'tracking': StageSpec(
            name='tracking',
            dependencies=('ingest', 'pose_refine', 'segment'),
            artifact_cls=TrackArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.track(outputs['ingest'], geometry_artifact=outputs['pose_refine']),
            load_fn=TrackArtifact.load,
        ),
        'graph': StageSpec(
            name='graph',
            dependencies=('pose_refine', 'tracking', 'humans', 'objects'),
            artifact_cls=GraphArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.build_graph(outputs['pose_refine'], outputs['tracking'], human_artifact=outputs['humans'], object_artifact=outputs['objects']),
            load_fn=GraphArtifact.load,
        ),
        'optimization': StageSpec(
            name='optimization',
            dependencies=('sync', 'pose_refine', 'tracking', 'humans', 'objects', 'graph'),
            artifact_cls=OptimizationArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.optimize(outputs['sync'], outputs['pose_refine'], outputs['tracking'], human_artifact=outputs['humans'], object_artifact=outputs['objects'], graph_artifact=outputs['graph']),
            load_fn=OptimizationArtifact.load,
        ),
        'scene': StageSpec(
            name='scene',
            dependencies=('pose_refine', 'tracking', 'humans', 'objects', 'optimization'),
            artifact_cls=SceneArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.assemble_scene(outputs['pose_refine'], outputs['tracking'], human_artifact=outputs['humans'], object_artifact=outputs['objects'], optimization_artifact=outputs['optimization']),
            load_fn=SceneArtifact.load,
        ),
        'render': StageSpec(
            name='render',
            dependencies=('scene', 'optimization', 'pose_refine', 'tracking', 'humans'),
            artifact_cls=RenderArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.fit_renderer(outputs['scene'], optimization_artifact=outputs['optimization'], geometry_artifact=outputs['pose_refine'], track_artifact=outputs['tracking'], human_artifact=outputs['humans']),
            load_fn=RenderArtifact.load,
        ),
    }


def write_run_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> Path:
    """Persist a run-level manifest used for resume/debugging."""
    out = ensure_dir(run_dir) / 'run_manifest.json'
    write_json(out, manifest)
    return out
