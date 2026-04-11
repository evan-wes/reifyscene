from __future__ import annotations

from pathlib import Path

from room4d.api.resume import materialize_stage_from_prior
from room4d.api.stages import STAGE_ORDER, build_stage_registry
from room4d.common.types import FrameRef, GraphEdge, GraphNode, VideoStreamRef
from room4d.graph.graph_artifact import GraphArtifact
from room4d.ingest.ingest_artifact import IngestArtifact
from room4d.sync.sync_artifact import SyncArtifact


def _touch_video(path: Path) -> Path:
    path.write_bytes(b'room4d-test-video')
    return path


def test_generic_artifact_roundtrip_for_ingest_sync_and_graph(tmp_path: Path) -> None:
    video = _touch_video(tmp_path / 'clip.mp4')
    ingest = IngestArtifact(
        run_id='dev',
        stage_name='ingest',
        save_dir=tmp_path / 'ingest',
        streams=[VideoStreamRef('cam_00', video, 30.0, 10, 64, 48, False, metadata={'codec': 'h264'})],
        keyframes=[FrameRef('cam_00', 0, 0.0), FrameRef('cam_00', 5, 1/6)],
        quality_by_stream={'cam_00': {'sharpness': 0.9}},
        sampled_windows={'cam_00': [(0.0, 1.0)]},
        warnings=['ok'],
        metadata={'compatibility': {'stage_name': 'ingest'}},
    )
    ingest.save()
    loaded_ingest = IngestArtifact.load(ingest.save_dir)
    assert loaded_ingest.streams[0].video_path == video
    assert loaded_ingest.sampled_windows['cam_00'][0] == (0.0, 1.0)

    sync = SyncArtifact(
        run_id='dev',
        stage_name='sync',
        save_dir=tmp_path / 'sync',
        offsets_by_stream={'cam_00': 0.0},
        confidence_by_stream={'cam_00': 1.0},
        pairwise_offsets={'cam_00|cam_01': {'offset_sec': 0.0, 'confidence': 1.0}},
        confidence_by_pair={'cam_00|cam_01': 1.0},
        anchor_stream_id='cam_00',
        notes=['anchored'],
        metadata={'compatibility': {'stage_name': 'sync'}},
    )
    sync.save()
    loaded_sync = SyncArtifact.load(sync.save_dir)
    assert loaded_sync.anchor_stream_id == 'cam_00'
    assert loaded_sync.pairwise_offsets['cam_00|cam_01']['offset_sec'] == 0.0

    graph = GraphArtifact(
        run_id='dev',
        stage_name='graph',
        save_dir=tmp_path / 'graph',
        nodes=[GraphNode('n0', 'camera', 'cam', 1.0, attrs={'stream_id': 'cam_00'})],
        edges=[GraphEdge('n0', 'n1', 'observed_by', 0.9, attrs={'t': 0})],
        node_histories={'n0': {'frames': [0, 1]}},
        uncertainty={'n0': {'sigma': 0.1}},
        source_summary={'num_geometry_streams': 1},
        notes=['graph'],
        metadata={'compatibility': {'stage_name': 'graph'}},
    )
    graph.save()
    loaded_graph = GraphArtifact.load(graph.save_dir)
    assert loaded_graph.nodes[0].node_id == 'n0'
    assert loaded_graph.edges[0].edge_type == 'observed_by'
    assert loaded_graph.node_histories['n0']['frames'] == [0, 1]


def test_stage_registry_and_materialize_stage_from_prior(tmp_path: Path) -> None:
    assert tuple(STAGE_ORDER) == ('ingest', 'sync', 'geometry', 'tracking', 'humans', 'objects', 'graph', 'optimization', 'scene', 'render')
    registry = build_stage_registry()
    assert registry['tracking'].dependencies == ('ingest', 'geometry')

    video = _touch_video(tmp_path / 'clip.mp4')
    prior_run = tmp_path / 'prior_run'
    current_run = tmp_path / 'current_run'
    cfg = {
        'run_id': 'clip',
        'run_dir': str(current_run),
        'max_keyframes_per_stream': 4,
        'keyframe_stride_sec': 1.0,
    }
    ingest = IngestArtifact(
        run_id='clip',
        stage_name='ingest',
        save_dir=prior_run / 'ingest',
        streams=[VideoStreamRef('cam_00', video, 30.0, 10, 64, 48, False)],
        keyframes=[FrameRef('cam_00', 0, 0.0)],
        metadata={
            'compatibility': {
                'stage_name': 'ingest',
                'video_signatures': [{
                    'path': str(video.resolve()),
                    'name': video.name,
                    'size_bytes': int(video.stat().st_size),
                    'mtime_ns': int(video.stat().st_mtime_ns),
                }],
                'stage_config': {
                    'max_keyframes_per_stream': 4,
                    'keyframe_stride_sec': 1.0,
                },
            }
        },
    )
    ingest.save()

    loaded = materialize_stage_from_prior(prior_run, current_run, registry['ingest'], cfg, [str(video)])
    assert loaded.save_dir == current_run / 'ingest'
    assert (current_run / 'ingest' / 'artifact.json').exists()
    reloaded = IngestArtifact.load(current_run / 'ingest')
    assert reloaded.streams[0].video_path == video.resolve()
