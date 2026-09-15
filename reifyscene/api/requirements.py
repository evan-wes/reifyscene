"""What a loaded artifact must carry, and how a resume brings an older one up to date.

A fingerprint says a loaded stage was made with the same settings from the same
input; it does not say the artifact is complete, or that it was made by code whose
output means what today's code expects. Each stage therefore lists
:class:`Requirement` s, checked on the loaded artifact at resume:

- met -> nothing to do;
- not met, but derivable from what is loaded (a moved input, a photo copy inside the
  run folder, a record the producer did not write yet) -> derived into the new run's
  copy and listed in the report -- the prior run is never modified;
- otherwise -> the resume is rejected, naming the stage to restart from.

:data:`UPGRADES` bring an artifact made by an older contract version of a stage up to
the current one, when that can be computed from what is loaded (no model run).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reifyscene.api.contracts import get_path


class NotDerivable(Exception):
    """A requirement is not met and cannot be computed from what is loaded."""


@dataclass
class Derivation:
    """What a derivation did: a note for the report, and how the copy must be written.

    ``metadata`` updates are patched into the copied artifact's JSON; ``resave`` means
    the derivation changed more than metadata, so the artifact is saved in full.
    ``revision`` names a change to what the artifact *says* (not just where its files
    are): it enters the artifact's fingerprint, so the stages built from the unrevised
    artifact must relink, rebuild or be rejected.
    """

    note: str
    metadata: dict[str, Any] = field(default_factory=dict)
    resave: bool = False
    revision: str | None = None


@dataclass
class ResumeContext:
    """What a requirement sees: the loaded artifact and everything loaded before it."""

    stage: str
    artifact: Any
    artifacts: dict[str, Any]
    records: dict[str, dict[str, Any]]
    cfg: dict[str, Any]
    stage_dir: Path
    video_paths: list[str]


@dataclass(frozen=True)
class Requirement:
    """``check`` returns the problem (or ``None`` when met); ``derive`` fixes it or raises."""

    name: str
    check: Callable[[ResumeContext], str | None]
    derive: Callable[[ResumeContext], Derivation] | None = None


def _metadata(artifact: Any) -> dict[str, Any]:
    return getattr(artifact, "metadata", {}) or {}


def _backend_used(artifact: Any) -> str | None:
    return (_metadata(artifact).get("capability") or {}).get("backend_used")


# ------------------------------------------------------------------ ingest

def _stream_missing(stream) -> list[str]:
    path = Path(stream.video_path)
    if not path.exists():
        return [str(path)]
    index = (getattr(stream, "metadata", {}) or {}).get("frame_index") or {}
    return [f for f in index.values() if not Path(f).exists()]


def _ingest_frames_resolve(ctx: ResumeContext) -> str | None:
    missing = [m for s in ctx.artifact.streams for m in _stream_missing(s)]
    if missing:
        return f"{len(missing)} input path(s) it records do not exist (e.g. {missing[0]})"
    streams = ctx.artifact.streams
    relative = [str(s.video_path) for s in streams if not Path(s.video_path).is_absolute()]
    if relative:
        return f"it records its input by a relative path ({relative[0]})"
    return None


def _ingest_reroot(ctx: ResumeContext) -> Derivation:
    """Point the streams at this invocation's input: the input check proved it the same."""
    from reifyscene.ingest.decoder import VideoDecoder

    streams = ctx.artifact.streams
    if len(streams) != len(ctx.video_paths):
        raise NotDerivable(
            f"it has {len(streams)} stream(s), this run has {len(ctx.video_paths)} input(s)"
        )
    decoder = VideoDecoder(default_fps=float(ctx.cfg.get("default_fps", 30.0)))
    for idx, (stream, path) in enumerate(zip(streams, ctx.video_paths, strict=True)):
        fresh = decoder.decode(path, stream_id=f"cam_{idx:02d}")
        old_index = (stream.metadata or {}).get("frame_index")
        new_index = (fresh.metadata or {}).get("frame_index")
        if (old_index is None) != (new_index is None) or (
            old_index is not None and sorted(old_index) != sorted(new_index)
        ):
            raise NotDerivable(
                f"{path} does not hold the frames stream {stream.stream_id} recorded"
            )
        stream.video_path = fresh.video_path
        if new_index is not None:
            stream.metadata["frame_index"] = new_index
    paths = ", ".join(str(Path(p).expanduser().resolve()) for p in ctx.video_paths)
    return Derivation(f"input paths re-rooted to {paths}", resave=True)


# ------------------------------------------------------------------ dyn_mask

def _dyn_mask_backend(ctx: ResumeContext) -> str | None:
    want = str(get_path(ctx.cfg, "dyn_mask.backend_name", "all-static")).lower()
    used = _backend_used(ctx.artifact)
    if used is None:
        return "it does not record which dyn_mask backend ran"
    if used != want:
        return f"the {used} backend made it; this run is configured for {want} (a fallback ran)"
    return None


# ------------------------------------------------------------------ geometry / pose_refine

def _geometry_backend_ran(ctx: ResumeContext) -> str | None:
    used = _backend_used(ctx.artifact)
    if used is None:
        return "it does not record which geometry backend ran (made before the backend ladder)"
    if used in ("", "empty"):
        return "the geometry ladder fell through to its empty floor: there is no geometry"
    return None


def _ground_plane_recorded(ctx: ResumeContext) -> str | None:
    md = _metadata(ctx.artifact)
    applied = bool(md.get("ground_plane_applied"))
    transform = (md.get("world_frame") or {}).get("transform_world_old_to_new")
    if applied and transform is None:
        return "its world was ground-plane canonicalized but the transform is not recorded"
    return None


def _global_poses(ctx: ResumeContext) -> str | None:
    if _backend_used(ctx.artifact) != "vggt-long":
        return None
    notes = getattr(ctx.artifact, "notes", []) or []
    if not any("globally-aligned" in str(n) for n in notes):
        return "made before VGGT-Long's globally aligned poses were used (2026-06-18)"
    return None


def _photos_missing(ctx: ResumeContext) -> list[str]:
    paths = _metadata(ctx.artifact).get("debug_image_paths") or {}
    return [k for k, p in paths.items() if not Path(p).is_absolute() or not Path(p).exists()]


def _photos_resolve(ctx: ResumeContext) -> str | None:
    bad = _photos_missing(ctx)
    if not bad:
        return None
    paths = _metadata(ctx.artifact)["debug_image_paths"]
    return (
        f"{len(bad)} of its {len(paths)} photo paths are relative or missing "
        f"(e.g. {bad[0]} -> {paths[bad[0]]})"
    )


def _photo_candidates(ctx: ResumeContext, key: str, recorded: str) -> list[Path]:
    """Where the photo the geometry consumed for ``key`` can still be found."""
    out: list[Path] = []
    rec = Path(recorded)
    if not rec.is_absolute():
        out.append(Path.cwd() / rec)  # a path the prior run wrote relative to its cwd
    safe = key.replace(":", "__").replace("/", "_")
    out.extend(sorted((ctx.stage_dir / "images").glob(f"{safe}.*")))  # the run's own copies
    backend = str(get_path(ctx.cfg, "geometry.backend_name", "") or "").lower()
    if backend == "dataset":
        root = get_path(ctx.cfg, "geometry.dataset_root", None)
        factor = int(get_path(ctx.cfg, "geometry.factor", 1) or 1)
        if root:
            out.append(Path(str(root)).expanduser() / "rgb" / f"{factor}x" / rec.name)
    elif get_path(ctx.cfg, "geometry.resize_max_dim", None) in (None, 0):
        # The model read the input frames unresized: the input frame is the same photo.
        ingest = ctx.artifacts.get("ingest")
        stream_id, _, frame = key.rpartition(":")
        for stream in getattr(ingest, "streams", []) or []:
            if str(stream.stream_id) == stream_id:
                index = (stream.metadata or {}).get("frame_index") or {}
                if frame in index:
                    out.append(Path(index[frame]))
    return out


def _photos_reroot(ctx: ResumeContext) -> Derivation:
    paths = dict(_metadata(ctx.artifact).get("debug_image_paths") or {})
    sources: dict[str, int] = {}
    for key in _photos_missing(ctx):
        candidates = _photo_candidates(ctx, key, paths[key])
        found = next((c.resolve() for c in candidates if c.exists()), None)
        if found is None:
            raise NotDerivable(f"no copy of the photo for {key} ({paths[key]}) could be found")
        paths[key] = str(found)
        own = (ctx.stage_dir / "images").resolve() in found.parents
        where = "the run's own copies" if own else str(found.parent)
        sources[where] = sources.get(where, 0) + 1
    ctx.artifact.metadata["debug_image_paths"] = paths
    where = "; ".join(f"{n} from {w}" for w, n in sources.items())
    return Derivation(f"photo paths re-rooted ({where})", metadata={"debug_image_paths": paths})


def _depth_sizes(geometry: Any) -> dict[str, tuple[int, int]]:
    import numpy as np

    sizes: dict[str, tuple[int, int]] = {}
    for key, payload in (getattr(geometry, "depthmaps_by_frame", {}) or {}).items():
        depth = None if payload is None else payload.get("depth")
        if depth is None:
            continue
        arr = np.asarray(depth)
        arr = arr[..., 0] if arr.ndim == 3 and arr.shape[-1] == 1 else arr
        sizes.setdefault(str(key).rpartition(":")[0], (int(arr.shape[1]), int(arr.shape[0])))
    return sizes


def _geometry_streams(geometry: Any) -> list[str]:
    return sorted(
        {str(f.stream_id) for f in getattr(geometry, "selected_frames", []) or []}
        | set(_depth_sizes(geometry))
    )


def _ingest_sizes(ctx: ResumeContext) -> dict[str, tuple[int, int]]:
    ingest = ctx.artifacts.get("ingest")
    return {
        str(s.stream_id): (int(s.width), int(s.height))
        for s in getattr(ingest, "streams", []) or []
        if int(s.width) > 0
    }


def _transforms_recorded(ctx: ResumeContext) -> str | None:
    from reifyscene.geometry.image_transform import ImageTransform

    records = _metadata(ctx.artifact).get("image_transforms") or {}
    missing = [s for s in _geometry_streams(ctx.artifact) if s not in records]
    if missing:
        return (
            f"it does not record how stream(s) {missing} map into its pixels (made before crop "
            "boxes were recorded)"
        )
    depth, ingest = _depth_sizes(ctx.artifact), _ingest_sizes(ctx)
    for stream_id, raw in records.items():
        transform = ImageTransform.from_dict(raw)
        if stream_id in depth and transform.output_size != depth[stream_id]:
            return (
                f"stream {stream_id}'s crop box lands on {transform.output_size}, "
                f"its depth is {depth[stream_id]}"
            )
        if stream_id in ingest and transform.source_size != ingest[stream_id]:
            return (
                f"stream {stream_id}'s crop box starts from {transform.source_size}, "
                f"its frames are {ingest[stream_id]}"
            )
    return None


def _vggt_long_loader(
    geometry_settings: dict[str, Any], resize_max_dim: int | None, *, version: int = 2
) -> tuple[str, int]:
    """The loader VGGT-Long ran for this resize setting (mirrors ``_input_loader``).

    ``version`` 1 is the no-upscale loader before it cropped tall frames.
    """
    from reifyscene.geometry.vggt_long_wrapper import VGGTLongInitializer

    target = VGGTLongInitializer._effective_vggt_input_target_size(resize_max_dim)
    if bool(geometry_settings.get("prevent_input_upscale", True)) and target < 518:
        return ("bounded" if version >= 2 else "bounded_v1"), int(target)
    return "upstream", 518


def _loader_current(ctx: ResumeContext) -> str | None:
    """Geometry the old no-upscale loader made from squashed frames is not today's."""
    from reifyscene.geometry.image_transform import vggt_input_transform

    if _backend_used(ctx.artifact) != "vggt-long":
        return None
    md = _metadata(ctx.artifact)
    if int((md.get("vggt_long_input_loader") or {}).get("version", 1)) >= 2:
        return None
    settings = (ctx.records.get(ctx.stage) or {}).get("settings", {}).get("geometry") or {}
    resize = md.get("vggt_long_resize_max_dim")
    loader, target = _vggt_long_loader(settings, resize, version=1)
    if loader != "bounded_v1":
        return None
    for stream_id, size in _ingest_sizes(ctx).items():
        kw = dict(target_size=target, resize_max_dim=resize)
        then = vggt_input_transform(size, loader="bounded_v1", **kw)
        now = vggt_input_transform(size, loader="bounded", **kw)
        if then.to_dict() != now.to_dict():
            return (
                f"made by the no-upscale VGGT-Long loader before it cropped tall frames: stream "
                f"{stream_id}'s {size[0]}x{size[1]} frames went in squashed to {then.output_size}, "
                f"where today's loader crops them to {now.output_size}"
            )
    return None


def _derive_transforms(ctx: ResumeContext) -> Derivation:
    """The crop box a pre-record geometry artifact's backend applied, from what it recorded.

    Accepted only when its recorded settings leave one possible loader and the result
    lands exactly on its depth maps; ``pose_refine`` shares its geometry's pixels.
    """
    from reifyscene.geometry.image_transform import (
        ImageTransform,
        Resize,
        pad_to_batch,
        transform_record,
        vggt_input_transform,
    )

    md = _metadata(ctx.artifact)
    backend = _backend_used(ctx.artifact) or ""
    if ctx.stage == "pose_refine":
        records = (_metadata(ctx.artifacts.get("geometry")).get("image_transforms")) or {}
        if not records:
            raise NotDerivable("its geometry has no crop box to share")
        ctx.artifact.metadata["image_transforms"] = dict(records)
        return Derivation(
            "crop boxes taken from its geometry", metadata={"image_transforms": dict(records)}
        )

    ingest, depth = _ingest_sizes(ctx), _depth_sizes(ctx.artifact)
    streams = _geometry_streams(ctx.artifact)
    missing = [s for s in streams if s not in ingest]
    if missing:
        raise NotDerivable(f"the ingest artifact has no frame size for stream(s) {missing}")
    settings = (ctx.records.get(ctx.stage) or {}).get("settings", {}).get("geometry") or {}
    transforms: dict[str, ImageTransform] = {}
    how = backend
    if backend == "vggt-long":
        resize = md.get("vggt_long_resize_max_dim")
        # No record: made before records existed, so by the version 1 no-upscale loader.
        loader, target = _vggt_long_loader(settings, resize, version=1)
        how = f"vggt-long, {loader} loader at {target}, resize_max_dim={resize}"
        transforms = {
            s: vggt_input_transform(
                ingest[s], loader=loader, target_size=target, resize_max_dim=resize
            )
            for s in streams
        }
    elif backend == "vggt":
        resize = settings.get("resize_max_dim")
        how = f"vggt, upstream loader, resize_max_dim={resize}"
        transforms = pad_to_batch({
            s: vggt_input_transform(ingest[s], loader="upstream", resize_max_dim=resize)
            for s in streams
        })
    elif backend == "dataset":
        how = "dataset, its pre-scaled frames"
        for s in streams:
            out = depth.get(s, ingest[s])
            ops = () if out == ingest[s] else (Resize(out, "unknown"),)
            transforms[s] = ImageTransform(ingest[s], ops)
    else:
        how = f"{backend}, frame-sized depth"
        transforms = {s: ImageTransform.identity(ingest[s]) for s in streams}
    for s, transform in transforms.items():
        if s in depth and transform.output_size != depth[s]:
            raise NotDerivable(
                f"the crop box its settings imply ({how}) lands on {transform.output_size}, but "
                f"its depth maps are {depth[s]}"
            )
    records = {
        s: {**transform_record(t, recorded_by=backend or "unknown"), "source": "derived at resume"}
        for s, t in transforms.items()
    }
    ctx.artifact.metadata["image_transforms"] = records
    return Derivation(f"crop boxes derived ({how})", metadata={"image_transforms": records})


def _stream_fps(ctx: ResumeContext) -> dict[str, float]:
    ingest = ctx.artifacts.get("ingest")
    return {
        str(s.stream_id): float(s.fps)
        for s in getattr(ingest, "streams", []) or []
        if float(getattr(s, "fps", 0) or 0) > 0
    }


def _off_clock(ctx: ResumeContext) -> list[Any]:
    """Selected frames whose camera is not at the ingest clock's time (dataset geometry)."""
    if _backend_used(ctx.artifact) != "dataset":
        return []
    fps = _stream_fps(ctx)
    states = getattr(ctx.artifact, "camera_states_by_stream", {}) or {}
    out = []
    for fr in getattr(ctx.artifact, "selected_frames", []) or []:
        rate = fps.get(str(fr.stream_id))
        if rate is None:
            continue
        want = float(fr.frame_idx) / rate
        times = [float(st.time_sec) for st in states.get(str(fr.stream_id), [])]
        has_camera = any(abs(t - want) <= 1e-6 for t in times)
        if abs(float(fr.timestamp_sec) - want) > 1e-6 or not has_camera:
            out.append(fr)
    return out


def _camera_clock(ctx: ResumeContext) -> str | None:
    off = _off_clock(ctx)
    if not off:
        return None
    fr = off[0]
    return (
        f"{len(off)} of its cameras are not on the ingest clock (frame {fr.frame_idx} at "
        f"{float(fr.timestamp_sec):g} s): the dataset backend timed cameras by frame number"
    )


def _retime_cameras(ctx: ResumeContext) -> Derivation:
    """Re-time a dataset geometry's cameras onto the ingest clock and re-densify them.

    Its cameras sat at time = frame number, and the extra cameras densified for the
    tracking frames (which are on the ingest clock) were interpolated between the wrong
    dataset frames. Keep the dataset's own cameras, move them to frame / fps, drop the
    densified ones and densify again onto the tracking plan.
    """
    from reifyscene.common.types import FrameRef, TimedCameraState
    from reifyscene.geometry.camera_interpolation import densify_camera_states_for_frames
    from reifyscene.tracking.cotracker_wrapper import CoTrackerWrapper

    art, fps = ctx.artifact, _stream_fps(ctx)
    old_time = {(str(f.stream_id), float(f.timestamp_sec)): f for f in art.selected_frames}
    states: dict[str, list[Any]] = {}
    for stream_id, lst in (art.camera_states_by_stream or {}).items():
        rate = fps.get(str(stream_id))
        if rate is None:
            raise NotDerivable(f"the ingest artifact has no fps for stream {stream_id}")
        kept = []
        for st in lst:
            fr = old_time.get((str(stream_id), float(st.time_sec)))
            if fr is None:
                continue  # a densified camera: re-made below
            kept.append(TimedCameraState(
                stream_id=st.stream_id, time_sec=float(fr.frame_idx) / rate,
                intrinsics=st.intrinsics, pose=st.pose,
            ))
        states[str(stream_id)] = sorted(kept, key=lambda st: st.time_sec)
    art.selected_frames = [
        FrameRef(f.stream_id, f.frame_idx, float(f.frame_idx) / fps.get(str(f.stream_id), 1.0))
        for f in art.selected_frames
    ]
    art.camera_states_by_stream = states
    tracker = CoTrackerWrapper(dict(ctx.cfg.get("tracking") or {}))
    targets = [fr for s in getattr(ctx.artifacts.get("ingest"), "streams", []) or []
               for fr in tracker.plan_frame_refs(s)]
    summary = (
        densify_camera_states_for_frames(art.camera_states_by_stream, targets) if targets else {}
    )
    md = art.metadata
    md["dense_camera_summary"] = summary
    md["save_debug_assets"] = md["save_debug_html"] = False  # do not rebuild the old viewer
    return Derivation(
        f"cameras re-timed onto the ingest clock and {int(summary.get('added_camera_states', 0))} "
        "tracking-frame cameras re-interpolated",
        resave=True, revision="camera-clock",
    )


def _pose_refine_ran(ctx: ResumeContext) -> str | None:
    info = _metadata(ctx.artifact).get("pose_refine") or {}
    if not info.get("enabled"):
        return None
    rejected = info.get("rejected_streams") or {}
    if not info.get("refined_streams") and not rejected:
        return "pose refinement was enabled but refined nothing (it passed the geometry through)"
    stats = info.get("stats") or {}
    if any("reanchor_scale" not in (st or {}) for st in stats.values()):
        return "made before pose_refine re-anchored its poses to the input frame (2026-07-06)"
    return None


def _bundle_adjustment_converged(ctx: ResumeContext) -> str | None:
    """A stream kept as refined whose bundle adjustment diverged is not a refinement."""
    from reifyscene.geometry.pose_refine import bundle_adjustment_diverged

    info = _metadata(ctx.artifact).get("pose_refine") or {}
    max_reproj = float(get_path(ctx.cfg, "pose_refine.max_final_reproj_px", 8.0))
    max_turn = float(get_path(ctx.cfg, "pose_refine.max_rot_delta_deg", 90.0))
    for stream_id in info.get("refined_streams") or []:
        st = (info.get("stats") or {}).get(stream_id) or {}
        why = bundle_adjustment_diverged(
            float(st.get("mean_reproj_px", float("nan"))),
            [float(st.get("seed_to_refined_rot_delta_deg_max", 0.0))],
            max_reproj_px=max_reproj, max_rot_delta_deg=max_turn,
        )
        if why is not None:
            return (
                f"stream {stream_id}'s bundle adjustment diverged ({why}) but its poses were kept"
            )
    return None


# ------------------------------------------------------------------ perception

def _segment_masks(ctx: ResumeContext) -> str | None:
    art = ctx.artifact
    ids = [o.instance_id for o in art.observations]
    missing = [i for i in ids if i not in art.mask_arrays]
    if missing:
        return f"{len(missing)} of its {len(ids)} instances have no mask (e.g. {missing[0]})"
    ingest = ctx.artifacts.get("ingest")
    sizes = {(int(s.height), int(s.width)) for s in getattr(ingest, "streams", []) or []}
    wrong = [i for i, m in art.mask_arrays.items() if sizes and tuple(m.shape[:2]) not in sizes]
    if wrong:
        shape = art.mask_arrays[wrong[0]].shape
        return f"{len(wrong)} masks are not frame-sized (e.g. {wrong[0]}: {shape})"
    return None


def _segment_tracks(ctx: ResumeContext) -> set[str]:
    seg = ctx.artifacts.get("segment")
    return {str(o.track_id) for o in getattr(seg, "observations", []) or []}


def _reid_covers_tracks(ctx: ResumeContext) -> str | None:
    tracks = _segment_tracks(ctx)
    have = set(map(str, ctx.artifact.embeddings))
    if have != tracks:
        return (
            f"its embeddings ({len(have)}) are not for segment's tracks ({len(tracks)}): "
            f"missing {sorted(tracks - have)[:3]}, extra {sorted(have - tracks)[:3]}"
        )
    merge = ctx.artifact.merge_map
    stray = {str(k) for k in merge} | {str(v) for v in merge.values()}
    if stray - tracks:
        return f"its merge map names tracks segment does not have: {sorted(stray - tracks)[:3]}"
    return None


def _canonical_tracks(ctx: ResumeContext) -> set[str]:
    reid = ctx.artifacts.get("reid")
    merge = {str(k): str(v) for k, v in (getattr(reid, "merge_map", {}) or {}).items()}
    return {merge.get(t, t) for t in _segment_tracks(ctx)}


def _semantics_cover_tracks(ctx: ResumeContext) -> str | None:
    missing = _canonical_tracks(ctx) - set(map(str, ctx.artifact.records))
    if missing:
        return f"it has no record for {len(missing)} canonical track(s) (e.g. {sorted(missing)[0]})"
    return None


def _humans_backend(ctx: ResumeContext) -> str | None:
    used = _backend_used(ctx.artifact)
    if used is None:
        return (
            "it does not record which humans backend ran: made before 2026-06-30, when its "
            "SMPL and DensePose output was fabricated"
        )
    if used not in ("sapiens", "keypoint_rcnn", "bbox_only"):
        return f"its keypoints come from an unknown backend ({used})"
    return None


# ------------------------------------------------------------------ tracking

def _tracks_present(ctx: ResumeContext) -> str | None:
    art = ctx.artifact
    payloads = getattr(art, "track_payloads_by_stream", {}) or {}
    ingest = ctx.artifacts.get("ingest")
    streams = [str(s.stream_id) for s in getattr(ingest, "streams", []) or []]
    missing = [s for s in streams if s not in payloads]
    if missing:
        return f"it has no 2D tracks for stream(s) {missing}"
    sizes = {
        str(s.stream_id): [int(s.width), int(s.height)]
        for s in getattr(ingest, "streams", []) or []
    }
    for sid, payload in payloads.items():
        recorded = payload.get("original_size_wh")
        size = [0, 0] if recorded is None else [int(v) for v in list(recorded)]
        if sid in sizes and size != sizes[sid]:
            w, h = sizes[sid]
            return f"stream {sid}'s tracks are for {size[0]}x{size[1]} frames; the input is {w}x{h}"
    return None


# ------------------------------------------------------------------ scene

def _scene_consistent(ctx: ResumeContext) -> str | None:
    art = ctx.artifact
    bundle = getattr(art, "dynamic_bundle", {}) or {}
    kind = str(getattr(art, "kind", "static"))
    if (kind == "dynamic") != bool(bundle):
        return f"it says kind={kind} but {'has' if bundle else 'has no'} dynamic bundle"
    if kind == "dynamic" and "fg_means" not in bundle:
        return "its dynamic bundle has no Gaussians (fg_means)"
    missing = [
        v.get("image_file") for v in art.train_views
        if not v.get("image_file") or not Path(v["image_file"]).exists()
    ]
    if missing:
        return f"{len(missing)} of its training photos are missing (e.g. {missing[0]})"
    return None


# ------------------------------------------------------------------ the lists

_GEOMETRY = (
    Requirement("the geometry backend actually ran", _geometry_backend_ran),
    Requirement("the ground-plane transform is recorded", _ground_plane_recorded),
    Requirement("VGGT-Long's poses are globally aligned", _global_poses),
    Requirement("its photos exist", _photos_resolve, _photos_reroot),
    Requirement("its crop box is recorded", _transforms_recorded, _derive_transforms),
    Requirement("its frames went in as today's loader takes them", _loader_current),
    Requirement("its cameras are on the ingest clock", _camera_clock, _retime_cameras),
)

REQUIREMENTS: dict[str, tuple[Requirement, ...]] = {
    "ingest": (Requirement("its input frames exist", _ingest_frames_resolve, _ingest_reroot),),
    "dyn_mask": (Requirement("the configured backend ran", _dyn_mask_backend),),
    "geometry": _GEOMETRY,
    "pose_refine": (
        *_GEOMETRY,
        Requirement("the refinement actually ran", _pose_refine_ran),
        Requirement("its bundle adjustment converged", _bundle_adjustment_converged),
    ),
    "segment": (Requirement("every instance has a frame-sized mask", _segment_masks),),
    "reid": (Requirement("it embeds exactly segment's tracks", _reid_covers_tracks),),
    "semantics": (Requirement("every canonical track is named", _semantics_cover_tracks),),
    "humans": (Requirement("its backend and keypoint set are known", _humans_backend),),
    "tracking": (Requirement("it holds 2D tracks for every input frame stream", _tracks_present),),
    "scene": (Requirement("it is a complete scene", _scene_consistent),),
}

def _relift_tracks(ctx: ResumeContext) -> Derivation:
    """Re-lift the loaded 2D tracks onto the loaded pose_refine and re-split them.

    The 2D tracks (the expensive, model-made part) are kept; the lift and the static /
    dynamic split are recomputed exactly as the tracking stage does them. Used to bring
    a version-1 tracking artifact up to date and to relink one whose pose_refine changed.
    """
    from reifyscene.geometry.image_transform import MissingImageTransform
    from reifyscene.lifting.backproject import TrackLifter3D
    from reifyscene.lifting.static_dynamic import StaticDynamicPartitioner

    geometry, ingest = ctx.artifacts.get("pose_refine"), ctx.artifacts.get("ingest")
    if geometry is None or ingest is None:
        raise NotDerivable("re-lifting needs the loaded pose_refine and ingest artifacts")
    lifting = ctx.cfg.get("lifting") or {}
    try:
        TrackLifter3D(point_conf_threshold=float(lifting.get("point_conf_threshold", 0.0))).lift(
            ctx.artifact, geometry, ingest
        )
    except MissingImageTransform as exc:
        raise NotDerivable(str(exc)) from exc
    StaticDynamicPartitioner(
        static_threshold_world=float(lifting.get("static_threshold_world", 0.05)),
        min_lifted_frames=int(lifting.get("min_lifted_frames", 2)),
    ).partition(ctx.artifact, geometry)
    md = ctx.artifact.metadata
    md["debug_image_paths"] = dict(_metadata(geometry).get("debug_image_paths") or {})
    # The resumed copy does not rebuild the prior run's (large) track debug viewer.
    md["save_debug_assets"] = md["save_debug_html"] = False
    return Derivation(
        "tracks re-lifted from the geometry's depth and cameras through its crop box, and "
        "re-split static/dynamic",
        # No revision: as an upgrade (new version) or a relink (new upstream digests) the
        # fingerprint changes anyway, and it then equals a fresh run's.
        resave=True,
    )


def _relink_pose_refine(ctx: ResumeContext) -> Derivation:
    """pose_refine absorbs a geometry revision it repeats itself (re-timed cameras)."""
    revised = set((ctx.records.get("geometry") or {}).get("revisions") or [])
    if not revised <= {"camera-clock"}:
        raise NotDerivable(
            f"its geometry was revised ({sorted(revised)}) in a way it cannot follow"
        )
    return Derivation("follows its geometry's camera re-timing (applied below)")


# stage -> {version made by: upgrade to the next version}. An upgrade recomputes the
# artifact's content from what is loaded (no model run) and returns what it did.
def _pose_refine_v2(ctx: ResumeContext) -> Derivation:
    """Bring a version-1 pose_refine artifact up to date, when there is nothing to bring.

    Version 2 changed the bundle adjustment: it refines only cameras with enough
    observations to determine a pose, and runs single-threaded so it repeats. A run that
    passed the geometry through -- refinement disabled, or every stream refused -- ran no
    bundle adjustment, so v2 would produce exactly what is on disk. A run that did refine
    cannot be brought forward: its poses came out of the old solve, and the only way to
    the new ones is to run it again.
    """
    info = _metadata(ctx.artifact).get("pose_refine") or {}
    if not info.get("enabled"):
        return Derivation(
            note="pose refinement was disabled: the stage passed the geometry through"
        )
    if not info.get("refined_streams"):
        return Derivation(
            note="pose refinement refined nothing: the stage passed the geometry through"
        )
    raise NotDerivable(
        "its poses came out of the old bundle adjustment, which refined cameras with too few "
        "observations to determine a pose and did not repeat run to run; the new ones cannot "
        "be computed from the artifact -- re-run pose_refine"
    )


def _tracking_grid_seeded(ctx: ResumeContext) -> Derivation:
    """A version-2 tracking artifact is a valid version-3 one that was grid-seeded.

    Version 3 added query points sampled on the ``segment`` masks *alongside* the
    uniform grid. It did not make mask seeding mandatory: a run whose masks are
    unavailable seeds the grid alone, and that is exactly what every version-2 artifact
    is. So there is nothing to recompute -- the artifact is already what this code
    produces in that case, and it only needs to say so.

    No ``revision``: the tracks themselves are unchanged, so everything built from them
    stays valid. What it does change is the honesty of the record -- a consumer asking
    how this artifact was seeded gets an answer instead of an absence.
    """
    del ctx
    return Derivation(
        note="tracks were seeded on the uniform grid alone (no mask seeding in that version)",
        metadata={"seeding": "grid", "num_entity_queries": 0},
    )


def _reid_no_covisible_dedup(ctx: ResumeContext) -> Derivation:
    """A version-2 reid artifact is a valid version-3 one built without the dedup.

    Version 3 merges tracklets that occupy the same 3D place at the same time. An older
    artifact simply never asked that question, so its merge map is a subset of what this
    code would produce -- nothing to recompute, only something to record.
    """
    del ctx
    return Derivation(
        note="merges did not include co-visible duplicates (that check did not exist then)"
    )


def _reid_appearance_only(ctx: ResumeContext) -> Derivation:
    """A version-1 reid artifact is a valid version-2 one built without track bridges.

    Version 2 lets point tracks spanning an occlusion gap relax the appearance threshold
    for the pair they bridge. Bridges only ever *admit* candidates -- a merge still has
    to win mutual-nearest-neighbour and the ratio test -- so a merge map built without
    them is a subset of what this code would produce, not a different kind of thing.
    Nothing to recompute; record that it had no bridges to work with.
    """
    del ctx
    return Derivation(
        note="merges were appearance-only (point-track bridges were not available then)",
        metadata={"bridges": "unavailable"},
    )


UPGRADES: dict[str, dict[int, Callable[[ResumeContext], Derivation]]] = {
    "tracking": {1: _relift_tracks, 2: _tracking_grid_seeded},
    "pose_refine": {1: _pose_refine_v2},
    "reid": {1: _reid_appearance_only, 2: _reid_no_covisible_dedup},
}

# stage -> how it absorbs a change of the upstream stages its contract ``relinks``.
def _reid_follows_tracking(ctx: ResumeContext) -> Derivation:
    """reid absorbs a tracking change, because the part it uses cannot have changed.

    Its merges are corroborated by **bridges** -- point tracks seen inside two tracklets'
    masks -- and that is a question about the tracks' 2D positions and visibility. The
    only thing that happens to a tracking artifact a resume *loads* is a re-lift: the
    3D and the static/dynamic split are recomputed, the 2D tracks are the expensive
    model-made part and are kept exactly. So the bridges are identical and the merge map
    still holds.

    If the 2D tracks themselves were to change, tracking would have been re-run rather
    than loaded, and reid would be rejected before reaching here.
    """
    del ctx
    return Derivation(
        note="follows its tracking: the re-lift leaves the 2D tracks its bridges are read from"
    )


def _follows_reid(ctx: ResumeContext) -> Derivation:
    """semantics / humans absorb a reid relink: what they read from it is unchanged.

    They consume reid's canonical track ids -- which tracklets are one entity. A relink
    only moves reid's fingerprint (its tracking upstream was re-lifted); the embeddings
    and the merge map those ids come from are untouched, because a loaded tracking
    artifact keeps its 2D tracks and the bridges are read from those.

    Without this, giving reid a `tracking` upstream would make every geometry revision
    reject the two model stages downstream of it -- turning a revision that the design
    deliberately *follows* into one that rejects.
    """
    del ctx
    return Derivation(note="follows its reid: the canonical track ids it reads are unchanged")


RELINKS: dict[str, Callable[[ResumeContext], Derivation]] = {
    "pose_refine": _relink_pose_refine,
    "tracking": _relift_tracks,
    "reid": _reid_follows_tracking,
    "semantics": _follows_reid,
    "humans": _follows_reid,
}
