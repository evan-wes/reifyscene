"""Export one run's VGGT geometry into the posed-RGB-D layout ConceptGraphs reads.

Why this layout and not a dataset class
---------------------------------------
ConceptGraphs dispatches datasets through a hardcoded ``if/elif`` in ``get_dataset``
with no registration hook, so a new dataset class means editing the reference tool --
and a reference whose code we edited is no longer a reference. Its ``ReplicaDataset``
happens to read the simplest layout of any of them:

    <out>/<sequence>/results/frame<NNNNN>.jpg     colour
    <out>/<sequence>/results/depth<NNNNN>.png     uint16, divided by png_depth_scale
    <out>/<sequence>/traj.txt                     one row of 16 floats per frame, c2w
    <out>/<sequence>/dataconfig.yaml              fx, fy, cx, cy, size, depth scale

So we write that, and ConceptGraphs runs unmodified.

Three decisions worth stating, because each was measured
--------------------------------------------------------
**Depth comes from the pointmaps, not the depth head.** VGGT predicts both, and they do
not perfectly agree: unprojecting the stored depth with K and the stored pose reproduces
the stored pointmap only to 0.8-7% of scene scale. ``entity_geometry`` reads the
*pointmaps*, so handing ConceptGraphs the depth head would have the two systems reasoning
over different geometry and any difference between them would be uninterpretable. Instead
we take the world pointmap into the camera frame and write its z. The residual is then
only the pointmap's own departure from the pinhole model (0.4-3.5% of z).

**Poses and pointmaps come from the same stage.** ``pose_refine`` copies ``geometry``'s
depth and pointmaps byte-for-byte and replaces only the poses, with VGGT-Long's global
alignment -- so its poses and its own pointmaps disagree, at a 14% pinhole residual
against ``geometry``'s 0.4-3.5%. That is invisible to our own builder, which reads world
points and never transforms them, and fatal to a consumer that must unproject. We read
whichever stage is asked for and *report* the residual, so a bad pairing is visible in
the export manifest rather than silently degrading the comparison.

**Scale is normalised per sequence.** ConceptGraphs' thresholds are metric and tuned for
room scans (``dbscan_eps`` 5 cm, ``downsample_voxel_size`` 2.5 cm). VGGT's scale is
arbitrary up to Sim(3), and across the ninety DAVIS runs the scene diagonal spans
0.66 to 8.2 units -- a 12.4x range, so no single factor serves. Each sequence is scaled
so its own 1-99 percentile diagonal maps to ``reference_diagonal_m``, which puts every
clip in the same relative regime as the scans those defaults were chosen on. Scaling the
input rather than editing their thresholds per clip is what keeps this stock
ConceptGraphs; it is a similarity transform on a quantity we already know is arbitrary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from reifyscene.common.paths import ensure_dir

if TYPE_CHECKING:
    from reifyscene.api.run_outputs import RunOutputs

# A plausible indoor room across its diagonal. The value itself does not matter -- what
# matters is that every sequence lands at the same scale, so one set of metric thresholds
# means the same thing on all of them.
REFERENCE_DIAGONAL_M = 6.0

# uint16 millimetres: 1 unit = 1 mm, so the reference diagonal above is ~6000 units and a
# 16-bit PNG carries it without saturating (max 65.5 m) or quantising visibly (1 mm).
PNG_DEPTH_SCALE = 1000.0
_UINT16_MAX = 65535


@dataclass
class ExportedSequence:
    """What the export wrote, and how far it can be trusted."""

    sequence: str
    out_dir: str
    stage: str
    run_dir: str = ""                     # the run it was exported from: its ground truth
    num_frames: int = 0
    scale_factor: float = 1.0
    scene_diagonal_units: float = 0.0
    pinhole_residual_pct: float = 0.0     # median |xy - (uv-c)z/f| / z, in percent
    depth_clipped_pct: float = 0.0        # pixels lost to the uint16 ceiling
    behind_camera_pct: float = 0.0        # pixels with z <= 0, unwritable as depth
    intrinsics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def scale_to_reference(
    points: np.ndarray, *, reference_diagonal_m: float = REFERENCE_DIAGONAL_M
) -> tuple[float, float]:
    """Return ``(scale_factor, diagonal)`` mapping a cloud's own extent to the reference.

    The extent is the 1-99 percentile box rather than the full one: a single stray point
    behind the camera would otherwise set the scale for the whole sequence.
    """
    finite = points[np.isfinite(points).all(axis=-1)]
    if finite.size == 0:
        return 1.0, 0.0
    lo, hi = np.percentile(finite, 1, axis=0), np.percentile(finite, 99, axis=0)
    diagonal = float(np.linalg.norm(hi - lo))
    if diagonal <= 1e-9:
        return 1.0, diagonal
    return reference_diagonal_m / diagonal, diagonal


def _camera_frame_depth(
    points: np.ndarray, pose_world_cam: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """World pointmap -> (camera-frame points, z). ``pose`` is T_world_cam, so invert it."""
    cam_from_world = np.linalg.inv(np.asarray(pose_world_cam, dtype=np.float64))
    cam = points @ cam_from_world[:3, :3].T + cam_from_world[:3, 3]
    return cam, cam[..., 2]


def _pinhole_residual_pct(cam: np.ndarray, intr: dict[str, float]) -> float:
    """Median |(x, y) - ((u, v) - c) * z / f| as a percentage of z.

    This is the error a consumer inherits by unprojecting our depth with our K: the part
    of the pointmap that the pinhole model cannot express. It is reported rather than
    corrected, because correcting it would mean handing over geometry we do not ourselves
    believe.
    """
    z = cam[..., 2]
    ok = z > 1e-6
    if not ok.any():
        return float("nan")
    h, w = z.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    dx = cam[..., 0][ok] - (u[ok] - intr["cx"]) * z[ok] / intr["fx"]
    dy = cam[..., 1][ok] - (v[ok] - intr["cy"]) * z[ok] / intr["fy"]
    return float(np.median(np.hypot(dx, dy) / z[ok]) * 100.0)


def export_run(
    run: RunOutputs,
    out_dir: Path,
    *,
    stage: str = "geometry",
    stream_id: str | None = None,
    reference_diagonal_m: float = REFERENCE_DIAGONAL_M,
    frames: Sequence[int] | None = None,
) -> ExportedSequence:
    """Write one run as a ConceptGraphs-readable posed RGB-D sequence.

    Colour comes from the run's own stream (:class:`~reifyscene.ingest.streams.FrameStream`),
    mapped into the geometry's pixels with the transform the stage recorded, so colour and
    depth describe the same raster and the stored intrinsics apply unchanged. Each frame's
    pointmap file is the one the stage's ``point_index`` names, and its pose is the camera
    state the stage recorded for that frame -- matched by frame number, never by list position.
    """
    from PIL import Image

    from reifyscene.common.frame_keys import parse_key
    from reifyscene.geometry.geometry_artifact import GeometryArtifact
    from reifyscene.geometry.image_transform import ImageTransform

    out_dir = Path(out_dir)
    stream = run.streams.resolve(stream_id)
    stream_id = stream.id
    art = run.record(stage)
    pointmaps = run.stage_dir(stage) / GeometryArtifact.POINTMAPS
    intr = dict(art["intrinsics_by_stream"][stream_id])
    states = art["camera_states_by_stream"][stream_id]
    raw_transform = ((art.get("metadata") or {}).get("image_transforms") or {}).get(stream_id)
    if raw_transform is None:
        raise ValueError(
            f"{run.name}/{stage} recorded no image transform for {stream_id!r}; colour "
            "cannot be put into the geometry's pixels without guessing it"
        )
    transform = ImageTransform.from_dict(raw_transform)

    pointmap_of: dict[int, str] = {}
    for key, name in (art.get("point_index") or {}).items():
        sid, frame = parse_key(key)
        if sid == stream_id:
            pointmap_of[frame] = str(name)
    state_at = {int(s["frame_idx"]): s for s in states}

    def pose_for(frame: int) -> np.ndarray:
        state = state_at.get(int(frame))
        if state is None:
            raise ValueError(f"{run.name}/{stage} recorded no camera state for frame "
                             f"{frame} of stream {stream_id!r}")
        return np.asarray(state["pose"], dtype=np.float64)

    indexed = sorted(pointmap_of)
    frame_list = list(indexed if frames is None else [f for f in frames if f in set(indexed)])
    result = ExportedSequence(
        sequence=stream.source.sequence or run.name, out_dir=str(out_dir), stage=stage,
        run_dir=str(run.root.resolve()),
        num_frames=len(frame_list), intrinsics=dict(intr),
    )
    if not frame_list:
        result.notes.append("no pointmaps to export")
        return result
    if not transform.is_pure_resize:
        # A crop means source rows have no geometry pixel, so a ground-truth object living
        # in them is invisible to the reference tool but visible to ours -- an asymmetry
        # that would silently favour us. It does not arise on DAVIS; say so if it ever does.
        result.notes.append(
            "image transform is not a pure resize: rows removed by its crop are outside "
            "what the reference tool can see, so its misses there are not comparable"
        )

    # One scale for the whole sequence, from a sample: the scene is one rigid thing and a
    # per-frame scale would make its geometry inconsistent across time.
    sample = [frame_list[0], frame_list[len(frame_list) // 2], frame_list[-1]]
    cloud = np.concatenate([
        np.load(pointmaps / pointmap_of[f])["points"].reshape(-1, 3)
        for f in sample
    ])
    scale, diagonal = scale_to_reference(cloud, reference_diagonal_m=reference_diagonal_m)
    result.scale_factor, result.scene_diagonal_units = scale, diagonal

    results_dir = ensure_dir(out_dir / "results")
    ensure_dir(out_dir)
    poses: list[np.ndarray] = []
    residuals: list[float] = []
    clipped = behind = total = 0

    for out_idx, frame_idx in enumerate(frame_list):
        points = np.load(
            pointmaps / pointmap_of[frame_idx]
        )["points"].astype(np.float64)
        pose = pose_for(frame_idx)
        cam, z = _camera_frame_depth(points, pose)
        residuals.append(_pinhole_residual_pct(cam, intr))

        # Depth in uint16 millimetres of the normalised scene. Non-positive z is "no
        # measurement" (0), which is exactly how ConceptGraphs reads it: create_object_pcd
        # drops depth <= 0 rather than unprojecting it to a point behind the camera.
        scaled = z * scale * PNG_DEPTH_SCALE
        valid = np.isfinite(scaled) & (scaled > 0)
        behind += int((~valid).sum())
        total += valid.size
        over = valid & (scaled > _UINT16_MAX)
        clipped += int(over.sum())
        depth_png = np.where(valid, np.clip(scaled, 0, _UINT16_MAX), 0).astype(np.uint16)
        Image.fromarray(depth_png).save(results_dir / f"depth{out_idx:05d}.png")

        rgb = transform.apply_to_image(stream.frame(frame_idx), kind="rgb")
        if rgb.shape[:2] != depth_png.shape[:2]:
            raise ValueError(
                f"{run.name} frame {frame_idx}: colour is {rgb.shape[:2]} after the "
                f"recorded transform but depth is {depth_png.shape[:2]}"
            )
        Image.fromarray(rgb).save(results_dir / f"frame{out_idx:05d}.jpg", quality=95)

        scaled_pose = pose.copy()
        scaled_pose[:3, 3] *= scale          # rotation is scale-free; only translation moves
        poses.append(scaled_pose)

    (out_dir / "traj.txt").write_text(
        "".join(" ".join(f"{v:.9f}" for v in p.reshape(-1)) + "\n" for p in poses)
    )
    result.pinhole_residual_pct = float(np.nanmedian(residuals)) if residuals else float("nan")
    result.depth_clipped_pct = 100.0 * clipped / total if total else 0.0
    result.behind_camera_pct = 100.0 * behind / total if total else 0.0
    _write_dataconfig(out_dir, intr, transform.output_size)
    _write_manifest(out_dir, result, frame_list, transform)
    return result


def _write_dataconfig(out_dir: Path, intr: dict[str, float], size: tuple[int, int]) -> None:
    """The dataconfig ConceptGraphs' ``ReplicaDataset`` reads its camera from.

    ``crop_edge: 0`` matters: Replica's own config trims 0 and several others trim a
    border, which would silently drop a strip of every frame.
    """
    width, height = int(size[0]), int(size[1])
    (out_dir / "dataconfig.yaml").write_text(
        "# Written by reifyscene's ConceptGraphs export. The camera is VGGT's, in the\n"
        "# geometry's own pixels; depth is uint16 millimetres of a scene normalised to a\n"
        "# reference diagonal (see manifest.json for this sequence's scale factor).\n"
        "dataset_name: 'replica'\n"
        "camera_params:\n"
        f"  image_height: {height}\n"
        f"  image_width: {width}\n"
        f"  fx: {float(intr['fx'])}\n"
        f"  fy: {float(intr['fy'])}\n"
        f"  cx: {float(intr['cx'])}\n"
        f"  cy: {float(intr['cy'])}\n"
        f"  png_depth_scale: {PNG_DEPTH_SCALE}\n"
        "  crop_edge: 0\n"
    )


def _write_manifest(
    out_dir: Path, result: ExportedSequence, frame_list: Sequence[int], transform: Any
) -> None:
    """Record what was written, including the frame mapping the scorer needs.

    ConceptGraphs indexes frames by position in a sorted glob; DAVIS annotations are named
    by source frame number. Without this mapping, scoring its output would line masks up
    against the wrong ground truth whenever a run did not export every frame.
    """
    from reifyscene.common.io import write_json

    write_json(
        out_dir / "manifest.json",
        {
            **result.to_dict(),
            "png_depth_scale": PNG_DEPTH_SCALE,
            "source_frame_for_index": [int(f) for f in frame_list],
            "image_transform": transform.to_dict(),
        },
    )
