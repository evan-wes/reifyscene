"""Stage registry and resume helpers for reifyscene pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from reifyscene.common.io import write_json
from reifyscene.common.paths import ensure_dir
from reifyscene.geometry.geometry_artifact import GeometryArtifact
from reifyscene.graph.graph_artifact import GraphArtifact
from reifyscene.graph.handles import HandlesArtifact
from reifyscene.perception.dyn_mask.dyn_mask_artifact import DynMaskArtifact
from reifyscene.ingest.ingest_artifact import IngestArtifact
from reifyscene.optimize.optimization_artifact import OptimizationArtifact
from reifyscene.perception.humans.human_artifact import HumanArtifact
from reifyscene.perception.instance.instance_artifact import InstanceArtifact
from reifyscene.perception.objects.object_artifact import ObjectArtifact
from reifyscene.perception.reid.reid_artifact import ReidArtifact
from reifyscene.perception.semantics.semantics_artifact import SemanticsArtifact
from reifyscene.render.gaussian4d.render_artifact import RenderArtifact
from reifyscene.scene.scene_artifact import SceneArtifact
from reifyscene.sync.sync_artifact import SyncArtifact
from reifyscene.tracking.track_artifact import TrackArtifact


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
    'dyn_mask',
    'geometry',
    'pose_refine',
    'segment',
    # `tracking` runs here, not after `humans`. Design §4A asked only for it to follow
    # `segment` (so its dynamic split is seeded by real foreground masks); placing it
    # five stages later also put it after `reid`, which meant identity was decided
    # before point tracks existed and could never use them. The fourth invariant
    # forbids the *ladder* reordering the pipeline, not redesign, and tracking's
    # declared dependencies are all satisfied here.
    'tracking',
    'reid',
    'semantics',
    'objects',
    'humans',
    'graph',
    'optimization',
    'scene',
    'render',
    # After render: binds each entity to its Gaussians, as a layer on the graph
    # (docs/design/2026-09-30-graph-layers.md). The graph cannot know this when it is built.
    'handles',
)


def build_stage_registry() -> dict[str, StageSpec]:
    """Return the canonical stage registry for reifyscene runs."""
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
        'dyn_mask': StageSpec(
            name='dyn_mask',
            dependencies=('ingest', 'sync'),
            artifact_cls=DynMaskArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.compute_dyn_mask(outputs['ingest'], outputs.get('sync')),
            load_fn=DynMaskArtifact.load,
        ),
        'geometry': StageSpec(
            name='geometry',
            dependencies=('ingest', 'sync', 'dyn_mask'),
            artifact_cls=GeometryArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.initialize_geometry(outputs['ingest'], outputs['sync'], dyn_mask_artifact=outputs.get('dyn_mask')),
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
            dependencies=('ingest', 'segment', 'tracking', 'pose_refine'),
            artifact_cls=ReidArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.compute_reid(outputs['ingest'], outputs['segment'], track_artifact=outputs['tracking'], geometry_artifact=outputs['pose_refine']),
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
            run_fn=lambda pipeline, outputs, video_paths: pipeline.compute_humans(outputs['ingest'], instance_artifact=outputs['segment'], reid_artifact=outputs['reid'], geometry_artifact=outputs['pose_refine']),
            load_fn=HumanArtifact.load,
        ),
        'tracking': StageSpec(
            name='tracking',
            dependencies=('ingest', 'pose_refine', 'segment'),
            artifact_cls=TrackArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.track(outputs['ingest'], geometry_artifact=outputs['pose_refine'], instance_artifact=outputs['segment']),
            load_fn=TrackArtifact.load,
        ),
        'graph': StageSpec(
            name='graph',
            dependencies=('pose_refine', 'tracking', 'humans', 'objects', 'semantics'),
            artifact_cls=GraphArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.build_graph(outputs['pose_refine'], outputs['tracking'], human_artifact=outputs['humans'], object_artifact=outputs['objects'], semantics_artifact=outputs['semantics']),
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
            dependencies=('pose_refine', 'segment', 'tracking', 'humans', 'objects', 'optimization'),
            artifact_cls=SceneArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.assemble_scene(outputs['pose_refine'], outputs['tracking'], human_artifact=outputs['humans'], object_artifact=outputs['objects'], optimization_artifact=outputs['optimization'], instance_artifact=outputs['segment']),
            load_fn=SceneArtifact.load,
        ),
        'render': StageSpec(
            name='render',
            dependencies=('scene',),
            artifact_cls=RenderArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.fit_renderer(outputs['scene']),
            load_fn=RenderArtifact.load,
        ),
        'handles': StageSpec(
            name='handles',
            dependencies=('graph', 'render'),
            artifact_cls=HandlesArtifact,
            run_fn=lambda pipeline, outputs, video_paths: pipeline.bind_gaussians(outputs['graph'], outputs['render']),
            load_fn=HandlesArtifact.load,
        ),
    }


def write_run_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> Path:
    """Persist a run-level manifest used for resume/debugging."""
    out = ensure_dir(run_dir) / 'run_manifest.json'
    write_json(out, manifest)
    return out
