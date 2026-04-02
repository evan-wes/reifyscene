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
from room4d.perception.objects.object_artifact import ObjectArtifact
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
    'tracking',
    'humans',
    'objects',
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
        'tracking': StageSpec(
            name='tracking',
            dependencies=('ingest', 'geometry'),
            artifact_cls=TrackArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.track(outputs['ingest'], geometry_artifact=outputs['geometry']),
            load_fn=TrackArtifact.load,
        ),
        'humans': StageSpec(
            name='humans',
            dependencies=('ingest', 'tracking'),
            artifact_cls=HumanArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.perceive_humans(outputs['ingest'], track_artifact=outputs['tracking']),
            load_fn=HumanArtifact.load,
        ),
        'objects': StageSpec(
            name='objects',
            dependencies=('ingest', 'tracking'),
            artifact_cls=ObjectArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.perceive_objects(outputs['ingest'], track_artifact=outputs['tracking']),
            load_fn=ObjectArtifact.load,
        ),
        'graph': StageSpec(
            name='graph',
            dependencies=('geometry', 'tracking', 'humans', 'objects'),
            artifact_cls=GraphArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.build_graph(outputs['geometry'], outputs['tracking'], human_artifact=outputs['humans'], object_artifact=outputs['objects']),
            load_fn=GraphArtifact.load,
        ),
        'optimization': StageSpec(
            name='optimization',
            dependencies=('sync', 'geometry', 'tracking', 'humans', 'objects', 'graph'),
            artifact_cls=OptimizationArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.optimize(outputs['sync'], outputs['geometry'], outputs['tracking'], human_artifact=outputs['humans'], object_artifact=outputs['objects'], graph_artifact=outputs['graph']),
            load_fn=OptimizationArtifact.load,
        ),
        'scene': StageSpec(
            name='scene',
            dependencies=('geometry', 'tracking', 'humans', 'objects', 'optimization'),
            artifact_cls=SceneArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.assemble_scene(outputs['geometry'], outputs['tracking'], human_artifact=outputs['humans'], object_artifact=outputs['objects'], optimization_artifact=outputs['optimization']),
            load_fn=SceneArtifact.load,
        ),
        'render': StageSpec(
            name='render',
            dependencies=('scene', 'optimization', 'geometry'),
            artifact_cls=RenderArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.fit_renderer(outputs['scene'], optimization_artifact=outputs['optimization'], geometry_artifact=outputs['geometry']),
            load_fn=RenderArtifact.load,
        ),
    }


def write_run_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> Path:
    """Persist a run-level manifest used for resume/debugging."""
    out = ensure_dir(run_dir) / 'run_manifest.json'
    write_json(out, manifest)
    return out
