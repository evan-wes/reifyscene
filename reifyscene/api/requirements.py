"""What a loaded artifact must carry for a resume to load it.

A fingerprint says a loaded stage was made with the same settings, from the same input, by
the same contract version, from the same upstream artifacts. It does not say the artifact is
complete. Each stage therefore lists :class:`Requirement` s, checked on the loaded artifact:
met, or the resume is rejected naming the stage to restart from. Nothing is ever derived,
upgraded or patched -- an artifact is loaded as it is or not at all.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reifyscene.api.contracts import get_path
from reifyscene.common.frame_keys import parse_key


@dataclass
class ResumeContext:
    """What a requirement sees: the loaded artifact and everything loaded before it."""

    stage: str
    artifact: Any
    artifacts: dict[str, Any]
    cfg: dict[str, Any]


@dataclass(frozen=True)
class Requirement:
    """``check`` returns the problem, or ``None`` when the artifact meets it."""

    name: str
    check: Callable[[ResumeContext], str | None]


def _metadata(artifact: Any) -> dict[str, Any]:
    return getattr(artifact, "metadata", {}) or {}


def _backend_used(artifact: Any) -> str | None:
    return (_metadata(artifact).get("capability") or {}).get("backend_used")


# ------------------------------------------------------------------ ingest

def _ingest_frames_exist(ctx: ResumeContext) -> str | None:
    missing: list[str] = []
    for stream in ctx.artifact.streams:
        path = Path(stream.video_path)
        if not path.exists():
            missing.append(str(path))
            continue
        index = (stream.metadata or {}).get("frame_index") or {}
        missing.extend(f for f in index.values() if not Path(f).exists())
    if missing:
        return f"{len(missing)} input path(s) it records do not exist (e.g. {missing[0]})"
    return None


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
        return "it does not record which geometry backend ran"
    if used in ("", "empty"):
        return "the geometry ladder fell through to its empty floor: there is no geometry"
    return None


def _global_poses(ctx: ResumeContext) -> str | None:
    if _backend_used(ctx.artifact) != "vggt-long":
        return None
    notes = getattr(ctx.artifact, "notes", []) or []
    if not any("globally-aligned" in str(n) for n in notes):
        return "VGGT-Long fell back to chunk-local poses (no globally aligned camera_poses.txt)"
    return None


def _photos_exist(ctx: ResumeContext) -> str | None:
    paths = _metadata(ctx.artifact).get("debug_image_paths") or {}
    missing = [k for k, p in paths.items() if not Path(p).is_file()]
    if missing:
        return (f"{len(missing)} of its {len(paths)} photos do not exist "
                f"(e.g. {missing[0]} -> {paths[missing[0]]})")
    return None


def _depth_sizes(geometry: Any) -> dict[str, tuple[int, int]]:
    import numpy as np

    sizes: dict[str, tuple[int, int]] = {}
    for key, payload in (getattr(geometry, "depthmaps_by_frame", {}) or {}).items():
        depth = None if payload is None else payload.get("depth")
        if depth is None:
            continue
        arr = np.asarray(depth)
        arr = arr[..., 0] if arr.ndim == 3 and arr.shape[-1] == 1 else arr
        sizes.setdefault(parse_key(key)[0], (int(arr.shape[1]), int(arr.shape[0])))
    return sizes


def _transforms_recorded(ctx: ResumeContext) -> str | None:
    from reifyscene.geometry.image_transform import ImageTransform

    records = _metadata(ctx.artifact).get("image_transforms") or {}
    depth = _depth_sizes(ctx.artifact)
    streams = {str(f.stream_id) for f in getattr(ctx.artifact, "selected_frames", []) or []}
    missing = sorted((streams | set(depth)) - set(records))
    if missing:
        return f"it does not record how stream(s) {missing} map into its pixels"
    ingest = {str(s.stream_id): (int(s.width), int(s.height))
              for s in getattr(ctx.artifacts.get("ingest"), "streams", []) or []}
    for stream_id, raw in records.items():
        transform = ImageTransform.from_dict(raw)
        if stream_id in depth and transform.output_size != depth[stream_id]:
            return (f"stream {stream_id}'s crop box lands on {transform.output_size}, "
                    f"its depth is {depth[stream_id]}")
        if stream_id in ingest and transform.source_size != ingest[stream_id]:
            return (f"stream {stream_id}'s crop box starts from {transform.source_size}, "
                    f"its frames are {ingest[stream_id]}")
    return None


def _pose_refine_ran(ctx: ResumeContext) -> str | None:
    info = _metadata(ctx.artifact).get("pose_refine") or {}
    if info.get("enabled") and not info.get("refined_streams") and not info.get(
            "rejected_streams"):
        return "pose refinement was enabled but refined nothing (it passed the geometry through)"
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
        return (f"its embeddings ({len(have)}) are not for segment's tracks ({len(tracks)}): "
                f"missing {sorted(tracks - have)[:3]}, extra {sorted(have - tracks)[:3]}")
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
    sizes = {str(s.stream_id): [int(s.width), int(s.height)]
             for s in getattr(ingest, "streams", []) or []}
    for sid, payload in payloads.items():
        size = [int(v) for v in list(payload["original_size_wh"])]
        if sid in sizes and size != sizes[sid]:
            w, h = sizes[sid]
            return f"stream {sid}'s tracks are for {size[0]}x{size[1]} frames; the input is {w}x{h}"
    return None


# ------------------------------------------------------------------ scene

def _scene_consistent(ctx: ResumeContext) -> str | None:
    art = ctx.artifact
    bundle = getattr(art, "dynamic_bundle", {}) or {}
    kind = str(art.kind)
    if (kind == "dynamic") != bool(bundle):
        return f"it says kind={kind} but {'has' if bundle else 'has no'} dynamic bundle"
    if kind == "dynamic" and "fg_means" not in bundle:
        return "its dynamic bundle has no Gaussians (fg_means)"
    missing = [v.get("image_file") for v in art.train_views
               if not v.get("image_file") or not Path(v["image_file"]).exists()]
    if missing:
        return f"{len(missing)} of its training photos are missing (e.g. {missing[0]})"
    return None


# ------------------------------------------------------------------ the lists

_GEOMETRY = (
    Requirement("the geometry backend actually ran", _geometry_backend_ran),
    Requirement("VGGT-Long's poses are globally aligned", _global_poses),
    Requirement("its photos exist", _photos_exist),
    Requirement("its crop box is recorded and consistent", _transforms_recorded),
)

REQUIREMENTS: dict[str, tuple[Requirement, ...]] = {
    "ingest": (Requirement("its input frames exist", _ingest_frames_exist),),
    "dyn_mask": (Requirement("the configured backend ran", _dyn_mask_backend),),
    "geometry": _GEOMETRY,
    "pose_refine": (*_GEOMETRY, Requirement("the refinement actually ran", _pose_refine_ran)),
    "segment": (Requirement("every instance has a frame-sized mask", _segment_masks),),
    "reid": (Requirement("it embeds exactly segment's tracks", _reid_covers_tracks),),
    "semantics": (Requirement("every canonical track is named", _semantics_cover_tracks),),
    "humans": (Requirement("its backend and keypoint set are known", _humans_backend),),
    "tracking": (Requirement("it holds 2D tracks for every input frame stream", _tracks_present),),
    "scene": (Requirement("it is a complete scene", _scene_consistent),),
}
