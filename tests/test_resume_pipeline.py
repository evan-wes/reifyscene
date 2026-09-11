from __future__ import annotations

import copy
import filecmp
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from reifyscene.api.contracts import (
    CONTRACTS,
    MISSING,
    contract_for,
    effective_settings,
    input_identity,
    recorded_setting,
)
from reifyscene.api.pipeline import ReifyScenePipeline
from reifyscene.api.resume import (
    ResumeRejected,
    _legacy_dir_signature,
    _legacy_stage_config,
    check_resume,
    materialize_stage,
)
from reifyscene.api.stages import STAGE_ORDER, build_stage_registry
from reifyscene.common.types import FrameRef, GraphEdge, GraphNode, VideoStreamRef
from reifyscene.graph.graph_artifact import GraphArtifact
from reifyscene.ingest.ingest_artifact import IngestArtifact
from reifyscene.perception.instance.instance_artifact import InstanceArtifact
from reifyscene.sync.sync_artifact import SyncArtifact
from tests.conftest import write_test_video


def _touch_video(path: Path) -> Path:
    path.write_bytes(b'reifyscene-test-video')
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


def test_stage_registry_dependencies() -> None:
    assert tuple(STAGE_ORDER) == ('ingest', 'sync', 'dyn_mask', 'geometry', 'pose_refine', 'segment', 'reid', 'semantics', 'objects', 'humans', 'tracking', 'graph', 'optimization', 'scene', 'render')
    registry = build_stage_registry()
    assert registry['dyn_mask'].dependencies == ('ingest', 'sync')
    assert registry['geometry'].dependencies == ('ingest', 'sync', 'dyn_mask')
    assert registry['pose_refine'].dependencies == ('ingest', 'geometry')
    assert registry['segment'].dependencies == ('ingest', 'pose_refine')
    assert registry['reid'].dependencies == ('ingest', 'segment')
    assert registry['semantics'].dependencies == ('ingest', 'segment', 'reid')
    assert registry['objects'].dependencies == ('ingest', 'segment', 'reid', 'semantics')
    assert registry['tracking'].dependencies == ('ingest', 'pose_refine', 'segment')


# ------------------------------------------------------------------ contracts

def test_every_stage_has_a_contract_whose_upstream_runs_earlier() -> None:
    assert set(CONTRACTS) == set(STAGE_ORDER)
    for stage, contract in CONTRACTS.items():
        for up in contract.upstream:
            assert STAGE_ORDER.index(up) < STAGE_ORDER.index(stage), (stage, up)


def test_settings_cover_other_sections_fill_defaults_and_drop_machine_keys() -> None:
    cfg = {
        'objects': {'frame_stride': 3, 'sam2_points_per_side': 16, 'device': 'cuda', 'debug': True,
                    'embedder_backend_mode': 'real'},
        'lifting': {'point_conf_threshold': 0.2},
        'tracking': {'grid_size': 8, 'cpu_threads': 4, 'debug_max_trails': 12},
        'optimizer': {'relabel_threshold': 0.1},
        'optimization': {'save_refinement_viewer': True},
    }
    segment = effective_settings('segment', cfg)
    # device, debug and the embedder keys (reid's) are not segment's
    assert segment['objects'] == {'frame_stride': 3, 'sam2_points_per_side': 16}
    tracking = effective_settings('tracking', cfg)
    assert tracking == {'tracking': {'grid_size': 8}, 'lifting': {'point_conf_threshold': 0.2}}
    # The solver's section is optimizer, not the debug-only optimization section.
    assert effective_settings('optimization', cfg) == {'optimizer': {'relabel_threshold': 0.1}}
    # An absent key and an explicit default fingerprint the same.
    assert effective_settings('ingest', {}) == effective_settings('ingest', {'default_fps': 30.0})
    assert effective_settings('ingest', {'default_fps': 25.0}) != effective_settings('ingest', {})


def test_settings_that_cannot_reach_the_output_are_excluded() -> None:
    off = {'pose_refine': {'enabled': False, 'ba_iters': 3}}
    assert effective_settings('pose_refine', off) == {'pose_refine': {'enabled': False}}
    on = {'pose_refine': {'enabled': True, 'ba_iters': 3}}
    assert effective_settings('pose_refine', on) == {'pose_refine': on['pose_refine']}
    dataset = effective_settings(
        'geometry', {'geometry': {'backend_name': 'dataset', 'chunk_size': 16}}
    )
    assert 'chunk_size' not in dataset['geometry']
    vggt_long = effective_settings(
        'geometry', {'geometry': {'backend_name': 'vggt-long', 'chunk_size': 16}}
    )
    assert vggt_long['geometry']['chunk_size'] == 16


def test_recorded_setting_reads_inside_a_subtree_entry() -> None:
    record = {'settings': {'tracking': {'max_frames': 96}, 'tracking.frame_stride': 2}}
    assert recorded_setting(record, 'tracking.max_frames') == 96
    assert recorded_setting(record, 'tracking.frame_stride') == 2
    assert recorded_setting(record, 'tracking.window_size') is MISSING


def test_a_stage_outside_the_pipeline_fingerprints_its_own_section() -> None:
    assert contract_for('calibration').settings == {'calibration': {}}


def _write_frames(folder: Path, names: list[str]) -> Path:
    import cv2

    folder.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        cv2.imwrite(str(folder / name), np.full((8, 8, 3), 10 * i, np.uint8))
    return folder


def test_input_identity_is_the_frames_read_not_the_folder(tmp_path: Path) -> None:
    frames = _write_frames(tmp_path / 'a' / '1x', ['0_00000.png', '0_00001.png'])
    before = input_identity([frames])
    moved = tmp_path / 'b' / '1x'
    moved.parent.mkdir()
    shutil.move(str(frames), str(moved))
    assert input_identity([moved]) == before  # moved / renamed project folder
    _write_frames(moved, ['1_00000.png'])  # another camera's frame: not read
    assert input_identity([moved]) == before
    target = moved / '0_00001.png'
    os.utime(target, ns=(target.stat().st_atime_ns, target.stat().st_mtime_ns + 10**9))
    assert input_identity([moved]) != before  # a frame the decoder reads changed


# ------------------------------------------------------------------ a real prior run

def _run(root: Path, name: str, videos: list[Path], **overrides) -> tuple[Path, dict]:
    cfg = {
        'run_dir': str(root / name),
        'geometry': {'backend_mode': 'heuristic'},
        'tracking': {'max_frames': 4},
        'lifting': {'point_conf_threshold': 0.0},
        'render': {'device': 'cpu'},
    }
    for path, value in overrides.items():
        section, key = path.split('.')
        cfg.setdefault(section, {})[key] = value
    # Every stage, through the registry, as the run script drives them.
    pipeline = ReifyScenePipeline(copy.deepcopy(cfg))
    registry = pipeline.stage_registry()
    outputs: dict = {}
    for stage in STAGE_ORDER:
        outputs[stage] = registry[stage].run_fn(pipeline, outputs, [str(v) for v in videos])
        if stage == 'tracking':
            outputs['tracks'] = outputs[stage]
    return root / name, cfg


@pytest.fixture(scope='module')
def prior(tmp_path_factory):
    root = tmp_path_factory.mktemp('resume')
    video = root / 'input' / 'clip.mp4'
    video.parent.mkdir()
    write_test_video(video, num_frames=6)
    run_dir, cfg = _run(root, 'prior', [video])
    return root, run_dir, cfg, video


def _check(prior_dir, run_dir, start, cfg, videos, **kw):
    videos = [str(v) for v in videos]
    return check_resume(prior_dir, run_dir, start, cfg, videos, build_stage_registry(), **kw)


def test_every_stage_records_a_fingerprint_with_its_lineage(prior) -> None:
    _, run_dir, _, _ = prior
    records = {}
    for stage in STAGE_ORDER:
        raw = json.loads((run_dir / stage / 'artifact.json').read_text())
        records[stage] = raw['metadata']['contract']
    for stage, record in records.items():
        assert record['stage'] == stage
        for up in contract_for(stage).upstream:
            assert record['upstream'][up] == records[up]['digest'], (stage, up)
    inputs = {json.dumps(r['input'], sort_keys=True) for r in records.values()}
    assert len(inputs) == 1  # every stage fingerprints the same input


def test_the_same_invocation_resumes(prior, tmp_path) -> None:
    root, run_dir, cfg, video = prior
    plan = _check(run_dir, tmp_path / 'new', 'scene', copy.deepcopy(cfg), [video])
    assert plan.ok, plan.report()
    assert all(v.action == 'load' and not v.derived for v in plan.verdicts.values())
    assert not (tmp_path / 'new').exists()  # the check writes nothing


def test_a_changed_setting_in_another_section_rejects_its_stage(prior, tmp_path) -> None:
    root, run_dir, cfg, video = prior
    changed = copy.deepcopy(cfg)
    changed['lifting']['point_conf_threshold'] = 0.5  # read by tracking, lives in `lifting`
    plan = _check(run_dir, tmp_path / 'new', 'scene', changed, [video])
    assert not plan.ok
    tracking = plan.verdicts['tracking']
    assert tracking.action == 'reject'
    assert 'lifting.point_conf_threshold: prior 0.0, now 0.5' in tracking.problems[0]
    assert plan.latest_start() == 'tracking'
    assert plan.verdicts['geometry'].action == 'load'


def test_debug_and_device_keys_do_not_reject(prior, tmp_path) -> None:
    root, run_dir, cfg, video = prior
    changed = copy.deepcopy(cfg)
    changed['tracking']['debug'] = True
    changed['tracking']['device'] = 'cpu'
    changed.setdefault('objects', {})['save_debug_assets'] = True
    assert _check(run_dir, tmp_path / 'new', 'scene', changed, [video]).ok


def test_resuming_into_the_prior_folder_is_refused(prior) -> None:
    root, run_dir, cfg, video = prior
    with pytest.raises(ResumeRejected, match='same folder'):
        _check(run_dir, run_dir, 'scene', copy.deepcopy(cfg), [video])


def test_a_moved_input_resumes_and_an_edited_one_does_not(prior, tmp_path) -> None:
    root, run_dir, cfg, video = prior
    moved = tmp_path / 'elsewhere' / video.name
    moved.parent.mkdir()
    shutil.copy2(video, moved)
    assert _check(run_dir, tmp_path / 'new', 'scene', copy.deepcopy(cfg), [moved]).ok
    with moved.open('ab') as fh:
        fh.write(b'\0')
    plan = _check(run_dir, tmp_path / 'new', 'scene', copy.deepcopy(cfg), [moved])
    assert not plan.ok
    assert 'different input' in plan.verdicts['ingest'].problems[0]
    assert plan.latest_start() == 'ingest'


def test_machine_picked_settings_are_pinned_to_the_prior_run(prior, tmp_path) -> None:
    root, run_dir, cfg, video = prior
    busy_gpu = copy.deepcopy(cfg)
    busy_gpu['tracking']['max_frames'] = 2  # what the profile would pick now
    busy_gpu['_profile_picked'] = ['tracking.max_frames']
    plan = _check(run_dir, tmp_path / 'new', 'scene', busy_gpu, [video])
    assert plan.ok, plan.report()
    assert plan.pins == {'tracking.max_frames': 4}
    assert busy_gpu['tracking']['max_frames'] == 4
    # The same value set explicitly by the user is intent, not a machine pick: it rejects.
    explicit = copy.deepcopy(cfg)
    explicit['tracking']['max_frames'] = 2
    assert not _check(run_dir, tmp_path / 'new', 'scene', explicit, [video]).ok


def test_a_stage_built_from_a_different_upstream_is_stale(prior, tmp_path) -> None:
    root, run_dir, cfg, video = prior
    other, other_cfg = _run(tmp_path, 'other', [video], **{'lifting.point_conf_threshold': 0.5})
    mixed = tmp_path / 'mixed'
    shutil.copytree(run_dir, mixed)
    shutil.rmtree(mixed / 'tracking')
    shutil.copytree(other / 'tracking', mixed / 'tracking')
    plan = _check(mixed, tmp_path / 'new', 'scene', copy.deepcopy(other_cfg), [video])
    assert plan.verdicts['tracking'].action == 'load'
    for stage in ('graph', 'optimization'):
        assert plan.verdicts[stage].action == 'reject'
        assert 'built from a different tracking artifact' in ' '.join(plan.verdicts[stage].problems)
    assert plan.latest_start() == 'graph'


def test_checking_every_stage_reports_the_latest_start(prior, tmp_path) -> None:
    root, run_dir, cfg, video = prior
    holed = tmp_path / 'holed'
    shutil.copytree(run_dir, holed)
    shutil.rmtree(holed / 'humans')
    every = list(STAGE_ORDER[:-1])
    plan = _check(holed, tmp_path / 'new', 'segment', copy.deepcopy(cfg), [video], stages=every)
    assert plan.ok  # segment only needs the stages before it
    assert 'no humans stage' in plan.verdicts['humans'].problems[0]
    assert plan.latest_start() == 'humans'
    full = _check(run_dir, tmp_path / 'new', 'segment', copy.deepcopy(cfg), [video], stages=every)
    assert full.latest_start() == 'render'


def test_materialize_copies_exactly_and_records_where_from(prior, tmp_path) -> None:
    root, run_dir, cfg, video = prior
    new = tmp_path / 'new'
    plan = _check(run_dir, new, 'scene', copy.deepcopy(cfg), [video])
    registry = build_stage_registry()
    tracking = materialize_stage(plan, registry['tracking'])
    assert tracking.save_dir == new / 'tracking'
    raw = json.loads((new / 'tracking' / 'artifact.json').read_text())
    assert raw['save_dir'] == str(new / 'tracking')
    assert raw['metadata']['resume']['loaded_from'] == str(run_dir / 'tracking')
    assert raw['metadata']['contract'] == plan.records['tracking']
    # Everything but artifact.json is byte-for-byte the prior's: no save/load round trip.
    cmp = filecmp.dircmp(run_dir / 'tracking', new / 'tracking')
    assert not cmp.left_only and not cmp.right_only
    changed = [
        f for f in cmp.common_files
        if not filecmp.cmp(run_dir / 'tracking' / f, new / 'tracking' / f, shallow=False)
    ]
    assert changed == ['artifact.json']


# ------------------------------------------------------------------ legacy prior runs

def _make_legacy(
    run_dir: Path, dst: Path, cfg: dict, video: Path, *,
    stop: str = 'render', stage_cfg_override=None,
) -> Path:
    """A copy of a run as the pre-fingerprint code wrote it: compatibility blocks + a manifest."""
    shutil.copytree(run_dir, dst)
    signature = {**_legacy_dir_signature(video), 'path': str(video)}
    for stage in STAGE_ORDER:
        path = dst / stage / 'artifact.json'
        raw = json.loads(path.read_text())
        raw['metadata'].pop('contract', None)
        stage_config = _legacy_stage_config(cfg, stage)
        if stage_cfg_override and stage in stage_cfg_override:
            stage_config = stage_cfg_override[stage]
        raw['metadata']['compatibility'] = {
            'stage_name': stage,
            'video_signatures': [signature],
            'stage_config': stage_config,
        }
        path.write_text(json.dumps(raw))
    manifest = {'video_paths': [str(video)], 'run_dir': str(dst), 'prior_run_dir': None,
                'start_stage': 'ingest', 'stop_stage': stop, 'config': cfg}
    (dst / 'run_manifest.json').write_text(json.dumps(manifest))
    return dst


def test_a_legacy_run_resumes_with_derived_fingerprints(prior, tmp_path) -> None:
    root, run_dir, cfg, video = prior
    legacy = _make_legacy(run_dir, tmp_path / 'legacy', cfg, video)
    plan = _check(legacy, tmp_path / 'new', 'scene', copy.deepcopy(cfg), [video])
    assert plan.ok, plan.report()
    for verdict in plan.verdicts.values():
        assert any('fingerprint derived from' in d for d in verdict.derived)
    # Derived records are the ones a fresh run would write: same digests.
    for stage, record in plan.records.items():
        fresh = json.loads((run_dir / stage / 'artifact.json').read_text())['metadata']['contract']
        assert record['digest'] == fresh['digest'], stage


def test_a_legacy_run_whose_project_folder_moved_still_resumes(prior, tmp_path) -> None:
    root, run_dir, cfg, video = prior
    old_input = tmp_path / 'old_project' / video.name
    old_input.parent.mkdir()
    shutil.copy2(video, old_input)
    legacy = _make_legacy(run_dir, tmp_path / 'legacy', cfg, old_input)
    renamed = tmp_path / 'new_project' / video.name
    renamed.parent.mkdir()
    shutil.move(str(old_input), str(renamed))
    plan = _check(legacy, tmp_path / 'new', 'scene', copy.deepcopy(cfg), [renamed])
    assert plan.ok, plan.report()
    assert any('input re-rooted' in d for d in plan.verdicts['ingest'].derived)


def test_a_legacy_leftover_stage_is_rejected(prior, tmp_path) -> None:
    root, run_dir, cfg, video = prior
    legacy = _make_legacy(run_dir, tmp_path / 'legacy', cfg, video, stop='graph')
    plan = _check(legacy, tmp_path / 'new', 'scene', copy.deepcopy(cfg), [video])
    assert not plan.ok
    assert 'left over' in plan.verdicts['optimization'].problems[0]
    assert plan.latest_start() == 'optimization'


def test_a_legacy_stage_the_manifest_did_not_produce_is_rejected(prior, tmp_path) -> None:
    root, run_dir, cfg, video = prior
    legacy = _make_legacy(run_dir, tmp_path / 'legacy', cfg, video,
                          stage_cfg_override={'tracking': {'max_frames': 99}})
    plan = _check(legacy, tmp_path / 'new', 'scene', copy.deepcopy(cfg), [video])
    assert 'do not match' in plan.verdicts['tracking'].problems[0]


# ------------------------------------------------------------------ no silent losses

def test_loaders_raise_on_a_listed_file_that_is_missing(tmp_path: Path) -> None:
    from reifyscene.perception.instance.instance_artifact import InstanceObservation

    obs = InstanceObservation(instance_id='i0', track_id='t0', frame_ref=FrameRef('0', 0, 0.0),
                              kind='object', bbox_xyxy=np.zeros(4, np.float32), confidence=1.0)
    art = InstanceArtifact(run_id='r', stage_name='segment', save_dir=tmp_path / 'segment',
                           observations=[obs], mask_arrays={'i0': np.ones((4, 4), bool)})
    art.rebuild_indices()
    art.save()
    for npz in (tmp_path / 'segment' / 'masks').glob('*.npz'):
        npz.unlink()
    with pytest.raises(FileNotFoundError, match='listed but missing'):
        InstanceArtifact.load(tmp_path / 'segment')


def test_loaders_keep_the_folder_they_were_loaded_from(prior, tmp_path) -> None:
    root, run_dir, cfg, video = prior
    registry = build_stage_registry()
    copied = tmp_path / 'copied'
    shutil.copytree(run_dir, copied)
    for stage in STAGE_ORDER:
        artifact = registry[stage].load_fn(copied / stage)
        assert Path(artifact.save_dir) == copied / stage, stage


def test_object_embeddings_survive_a_reload(tmp_path: Path) -> None:
    from reifyscene.common.types import ObjectState
    from reifyscene.perception.objects.object_artifact import ObjectArtifact

    vec = np.arange(4, dtype=np.float32)
    obj = ObjectState(object_id='o0', frame_ref=FrameRef('0', 0, 0.0), category='cup',
                      bbox_xyxy=np.zeros(4, np.float32), mask_rle=None, embedding=vec,
                      confidence=0.9, attributes={'detection_id': 'det0', 'track_id': 'trk7'})
    # The objects stage files embeddings under the canonical track id (pipeline.perceive_objects).
    art = ObjectArtifact(run_id='r', stage_name='objects', save_dir=tmp_path / 'objects',
                         objects=[obj], embedding_vectors={'trk7': vec})
    art.save()
    loaded = ObjectArtifact.load(tmp_path / 'objects')
    assert loaded.objects[0].embedding is not None
    np.testing.assert_array_equal(loaded.objects[0].embedding, vec)
