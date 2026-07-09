"""Native 3D-track lifting — room4d's own port of Shape-of-Motion's track lift.

The only thing borrowed here is the 2D point tracker output
(``flow3d_preprocessed/2d_tracks/``); everything else — TAPIR track parsing,
scene normalization, depth-unprojection of foreground trajectories, and
background point sampling — is room4d code, ported from SoM's
``flow3d/data/iphone_dataset.py`` (``get_tracks_3d`` / ``get_bkgd_points`` /
``_normalize_scene``) and ``flow3d/data/utils.py`` onto room4d's
:class:`~room4d.geometry.dataset_wrapper.DatasetGeometryInitializer` readers.

Output frame: SoM normalizes the scene (rotate gravity-up → +z, center on the
foreground tracks, scale to ~unit) and its borrowed init bundle lives in that
normalized frame. We reproduce that normalization so the natively-lifted tracks,
the background points, and the train/val cameras all land in the *same* frame the
borrowed bundle (and the trained quality bar) were set in. The lift runs at
``factor=1`` (full res) because the 2D tracks ship only at ``1x``; the produced
cameras carry full-res ``K`` exactly like ``scripts/borrow_som_motion_init.py``.

This is PR-A of the native-motion-init port: it yields the lifted observations
(:class:`Tracks3D` / :class:`StaticPoints3D`) + normalized cameras that the
clustering + Procrustes + Gaussian-placement initializer (PR-B) consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from room4d.geometry.dataset_wrapper import DatasetGeometryInitializer
from room4d.lifting.tracks_3d import StaticPoints3D, Tracks3D

if TYPE_CHECKING:
    from room4d.lifting.geometry_provider import GeometryProvider


# -- ported math primitives (SoM flow3d/data/utils.py, no roma) -----------

def parse_tapir_track_info(occlusions: torch.Tensor, expected_dist: torch.Tensor):
    """TAPIR (occlusion, expected_dist) logits -> (visible, invisible, confidence).

    visible = visible & confident; invisible = occluded & confident; confidence
    is zeroed where neither holds. Ported verbatim from SoM.
    """
    visibility = 1 - torch.sigmoid(occlusions)
    confidence = 1 - torch.sigmoid(expected_dist)
    valid_visible = visibility * confidence > 0.5
    valid_invisible = (1 - visibility) * confidence > 0.5
    confidence = confidence * (valid_visible | valid_invisible).float()
    return valid_visible, valid_invisible, confidence


def normalize_coords(coords: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Pixel (x, y) -> grid_sample [-1, 1] coords (align_corners=True convention)."""
    return coords / torch.tensor([w - 1.0, h - 1.0], device=coords.device) * 2 - 1.0


def rotvec_to_rotmat(rotvec: torch.Tensor) -> torch.Tensor:
    """Axis-angle (3,) -> rotation matrix (3, 3) via Rodrigues (roma-free)."""
    theta = torch.linalg.norm(rotvec)
    if float(theta) < 1e-8:
        return torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    k = rotvec / theta
    kx, ky, kz = k
    K = torch.tensor(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
        dtype=rotvec.dtype, device=rotvec.device,
    )
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    return eye + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def rt_to_mat4(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """(3,3) rotation + (3,) translation -> (4,4) homogeneous transform."""
    mat = torch.eye(4, dtype=R.dtype, device=R.device)
    mat[:3, :3] = R
    mat[:3, 3] = t
    return mat


def normal_from_depth(depth: torch.Tensor, K: torch.Tensor, w2c: torch.Tensor) -> torch.Tensor:
    """Per-pixel world-space normals (H, W, 3) from a depth map (SoM port).

    Unprojects the depth map to a world point grid, then takes the normalized
    cross product of the horizontal/vertical tangents (cross-padded to full res).
    """
    H, W = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=depth.dtype, device=depth.device),
        torch.arange(W, dtype=depth.dtype, device=depth.device),
        indexing="ij",
    )
    cam_xyz = torch.stack([xs * depth, ys * depth, depth], dim=-1).reshape(-1, 3)
    cam_xyz = cam_xyz @ torch.linalg.inv(K).transpose(0, 1)
    world = F.pad(cam_xyz, (0, 1), value=1.0) @ torch.linalg.inv(w2c).transpose(0, 1)
    xyz = world[..., :3].reshape(H, W, 3)

    bottom, top = xyz[2:H, 1:W - 1, :], xyz[0:H - 2, 1:W - 1, :]
    right, left = xyz[1:H - 1, 2:W, :], xyz[1:H - 1, 0:W - 2, :]
    normal = torch.cross(right - left, top - bottom, dim=-1)
    normal = F.normalize(normal, p=2.0, dim=-1)
    return F.pad(normal.permute(2, 0, 1), (1, 1, 1, 1), mode="constant").permute(1, 2, 0)


# -- per-split frame data -------------------------------------------------

@dataclass
class CameraRecords:
    """Per-frame cameras in the (normalized) bundle frame, in split order.

    frame_names  list[str]            dataset frame names
    time_ids     LongTensor (N,)      basis frame index (0..T-1), shared timeline
    w2cs         (N, 4, 4)            gsplat world->cam (normalized)
    Ks           (N, 3, 3)            full-res intrinsics
    """

    frame_names: list[str]
    time_ids: torch.Tensor
    w2cs: torch.Tensor
    Ks: torch.Tensor


@dataclass
class FrameStack:
    """A split's cameras plus image-space stacks needed for lifting."""

    cameras: CameraRecords
    depths: torch.Tensor   # (N, H, W) float, metric (normalized scale after norm)
    rgbs: torch.Tensor     # (N, H, W, 3) uint8
    fg: torch.Tensor       # (N, H, W) float in [0, 1], foreground mask
    valid: torch.Tensor    # (N, H, W) float in [0, 1], RGBA-alpha validity

    @property
    def num_frames(self) -> int:
        return len(self.cameras.frame_names)


@dataclass
class SceneNorm:
    """Scene normalization: ``transfm`` (4,4) world align/center + ``scale``."""

    transfm: torch.Tensor
    scale: float


def _intrinsics_matrix(intr, device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(
        [[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]],
        dtype=dtype, device=device,
    )


def _load_arkit_cameras(
    backend: DatasetGeometryInitializer, frame_names: list[str], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """ARKit camera-JSON poses (room4d's ``camera_type='original'``)."""
    w2cs, ks = [], []
    for name in frame_names:
        intr, t_world_cam, _wh = backend.load_camera(name)
        w2cs.append(torch.linalg.inv(torch.as_tensor(t_world_cam, dtype=torch.float32, device=device)))
        ks.append(_intrinsics_matrix(intr, device))
    return torch.stack(w2cs), torch.stack(ks)


def _load_colmap_cameras(
    dataset_root: str | Path, factor: int, frame_names: list[str], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """COLMAP-refined poses/intrinsics (SoM's default ``camera_type='refined'``).

    Reads ``flow3d_preprocessed/colmap/sparse`` via pycolmap and maps each frame
    by image name (``{frame_name}.png``); ``cam_from_world`` is the world->cam
    extrinsic and ``calibration_matrix`` the (full-res) intrinsics, scaled by
    ``factor`` to match the lift resolution.
    """
    import pycolmap  # lazy: only the refined path needs it

    sparse = Path(dataset_root) / "flow3d_preprocessed" / "colmap" / "sparse"
    rec = pycolmap.Reconstruction(str(sparse))
    by_name = {im.name: im for im in rec.images.values()}
    w2cs, ks = [], []
    for name in frame_names:
        img = by_name[f"{name}.png"]
        mat = np.asarray(img.cam_from_world().matrix(), dtype=np.float32)  # (3, 4)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :4] = mat
        K = np.asarray(rec.cameras[img.camera_id].calibration_matrix(), dtype=np.float32).copy()
        K[:2] /= factor
        w2cs.append(torch.as_tensor(w2c, device=device))
        ks.append(torch.as_tensor(K, device=device))
    return torch.stack(w2cs), torch.stack(ks)


def _load_cameras(
    backend: DatasetGeometryInitializer,
    split: str,
    device: str,
    dataset_root: str | Path,
    factor: int,
    camera_type: str,
) -> CameraRecords:
    """Read a split's cameras (w2c, K, time_id, frame_name) in split-json order.

    ``camera_type='refined'`` uses COLMAP-refined poses (matches the borrowed
    bundle frame); ``'original'`` uses room4d's ARKit camera-JSON poses.
    """
    pairs = backend.split_frames(split)
    frame_names = [backend.frame_name(cam, time_id) for cam, time_id in pairs]
    time_ids = torch.tensor([int(time_id) for _, time_id in pairs], dtype=torch.long, device=device)
    if camera_type == "refined":
        w2cs, ks = _load_colmap_cameras(dataset_root, factor, frame_names, device)
    elif camera_type == "original":
        w2cs, ks = _load_arkit_cameras(backend, frame_names, device)
    else:
        raise ValueError(f"camera_type must be 'refined' or 'original', got {camera_type!r}")
    return CameraRecords(frame_names=frame_names, time_ids=time_ids, w2cs=w2cs, Ks=ks)


def _load_aligned_depth(
    dataset_root: str | Path, factor: int, frame_name: str, depth_type: str
) -> np.ndarray:
    """Metric depth from SoM's aligned depth source (``aligned_{depth_type}``).

    These files store inverse depth (disparity); SoM floors tiny values and takes
    the reciprocal. The borrowed init bundle was lifted from this source (default
    ``depth_anything_colmap``), not the sparse sensor ``depth/`` (``lidar``).
    """
    path = Path(dataset_root) / "flow3d_preprocessed" / f"aligned_{depth_type}" / f"{factor}x" / f"{frame_name}.npy"
    depth = np.load(path).astype(np.float32)
    depth[depth < 1e-3] = 1e-3
    return 1.0 / depth


def _load_frame_stack(
    backend: DatasetGeometryInitializer,
    split: str,
    device: str,
    dataset_root: str | Path,
    factor: int,
    depth_type: str,
    camera_type: str,
) -> FrameStack:
    """Read a split's cameras + depth/rgb/fg/valid image stacks (raw frame).

    Depth comes from SoM's aligned source and is clamped per-stack exactly as the
    iPhone loader does (``clamp(0, median(per-frame max) * 2.5)``).
    """
    cameras = _load_cameras(backend, split, device, dataset_root, factor, camera_type)
    depths, rgbs, fg, valid = [], [], [], []
    for name in cameras.frame_names:
        depths.append(torch.as_tensor(
            _load_aligned_depth(dataset_root, factor, name, depth_type), dtype=torch.float32, device=device
        ))
        # load_rgb drops alpha; the alpha channel is SoM's valid_mask, so read RGBA here.
        with Image.open(backend.rgb_dir / f"{name}.png") as img:
            rgba = np.asarray(img.convert("RGBA"), dtype=np.uint8).copy()
        rgbs.append(torch.as_tensor(rgba[..., :3], dtype=torch.uint8, device=device))
        valid.append(torch.as_tensor(rgba[..., 3], dtype=torch.float32, device=device) / 255.0)
        fg.append(torch.as_tensor(backend.load_fg_mask(name), dtype=torch.float32, device=device))

    depths = torch.stack(depths)
    # SoM depth clamp: cap at 2.5x the median per-frame max to kill outliers.
    max_per_frame = depths.reshape(depths.shape[0], -1).max(1).values
    depths = depths.clamp(0.0, float(max_per_frame.median()) * 2.5)
    return FrameStack(
        cameras=cameras,
        depths=depths,
        rgbs=torch.stack(rgbs),
        fg=torch.stack(fg),
        valid=torch.stack(valid),
    )


# -- scene normalization (SoM iPhoneDataset._normalize_scene) -------------

def compute_scene_norm(w2cs: torch.Tensor, sample_xyz: torch.Tensor) -> SceneNorm:
    """Scene normalization from foreground tracks + camera up-axis.

    sample_xyz: (N, T, 3) raw-frame foreground track positions (a subsample).
    """
    scene_center = sample_xyz.mean((0, 1))
    centered = sample_xyz - scene_center
    min_scale = centered.quantile(0.05, dim=0)
    max_scale = centered.quantile(0.95, dim=0)
    scale = torch.max(max_scale - min_scale).item() / 2.0

    original_up = -F.normalize(w2cs[:, 1, :3].mean(0), dim=-1)
    target_up = original_up.new_tensor([0.0, 0.0, 1.0])
    axis = F.normalize(torch.linalg.cross(original_up, target_up), dim=-1)
    angle = torch.arccos(torch.dot(original_up, target_up))
    R = rotvec_to_rotmat(axis * angle)
    transfm = rt_to_mat4(R, torch.einsum("ij,j->i", -R, scene_center))
    return SceneNorm(transfm=transfm, scale=scale)


def apply_scene_norm_cameras(cameras: CameraRecords, norm: SceneNorm) -> None:
    """In-place: normalize a split's w2cs (rotate/center then /scale translation)."""
    cameras.w2cs = cameras.w2cs @ torch.linalg.inv(norm.transfm)
    cameras.w2cs[:, :3, 3] /= norm.scale


def apply_scene_norm_stack(stack: FrameStack, norm: SceneNorm) -> None:
    """In-place: normalize a frame stack's cameras and depths."""
    apply_scene_norm_cameras(stack.cameras, norm)
    stack.depths /= norm.scale


# -- lifting (SoM get_tracks_3d / get_bkgd_points) ------------------------

def _tracks_dir(dataset_root: str | Path, factor: int) -> Path:
    return Path(dataset_root) / "flow3d_preprocessed" / "2d_tracks" / f"{factor}x"


def lift_tracks_3d(
    stack: FrameStack,
    dataset_root: str | Path,
    factor: int,
    num_samples: int,
    step: int = 1,
    rng: np.random.Generator | None = None,
    tracks_dir: str | Path | None = None,
    pair_loader: "Callable[[int, int], np.ndarray] | None" = None,
) -> Tracks3D:
    """Lift sampled 2D foreground tracks to 3D trajectories (SoM get_tracks_3d port).

    Samples ~``num_samples`` query tracks spread across frames, loads each query's
    2D positions in every (sub)frame, bilinearly samples depth, unprojects through
    ``inv(K)`` + ``c2w``, keeps only in-foreground-mask observations, and drops
    tracks visible <5% of the time. ``step`` temporally subsamples (used by the
    scene-norm pass). ``rng`` makes the (otherwise random) sampling reproducible.

    ``tracks_dir`` overrides where the ``<name_i>_<name_j>.npy`` 2D tracks are read
    from (defaults to the dataset's borrowed ``flow3d_preprocessed/2d_tracks/<factor>x``);
    point it at room4d's own tracker output to drop the last borrowed input.
    """
    choice = (rng.choice if rng is not None else np.random.choice)
    names = stack.cameras.frame_names
    num_frames = len(names)
    H, W = stack.depths.shape[1], stack.depths.shape[2]
    device = stack.depths.device

    if pair_loader is not None:
        # In-memory all-pairs (e.g. the pipeline's TrackArtifact.all_pairs_by_stream):
        # pair_loader(i, j) -> (N_i, 4). No disk reads, no dataset_root needed.
        def load_pair(i: int, j: int) -> torch.Tensor:
            return torch.as_tensor(np.asarray(pair_loader(i, j), dtype=np.float32), device=device)
    else:
        tracks_dir = Path(tracks_dir) if tracks_dir is not None else _tracks_dir(dataset_root, factor)

        def load_pair(i: int, j: int) -> torch.Tensor:
            arr = np.load(tracks_dir / f"{names[i]}_{names[j]}.npy").astype(np.float32)
            return torch.as_tensor(arr, device=device)

    candidate_frames = list(range(0, num_frames, step))
    num_sampled_frames = len(candidate_frames)

    raw_tracks_2d: list[torch.Tensor] = []
    for i in candidate_frames:
        query = load_pair(i, i)
        curr_num = query.shape[0]
        base = int(np.floor(num_samples / num_sampled_frames))
        per = base if i != candidate_frames[-1] else num_samples - (num_sampled_frames - 1) * base
        if per < curr_num:
            sels = choice(curr_num, (per,), replace=False)
        else:
            sels = np.arange(0, curr_num)
        curr = [(query if i == j else load_pair(i, j))[sels] for j in range(0, num_frames, step)]
        raw_tracks_2d.append(torch.stack(curr, dim=1))  # (per, Tsub, 4)

    inv_Ks = torch.linalg.inv(stack.cameras.Ks)[::step]
    c2ws = torch.linalg.inv(stack.cameras.w2cs)[::step]
    masks = stack.fg * stack.valid * (stack.depths > 0).float()
    masks = (masks > 0.5).float()
    masks_sub = masks[::step]
    depths_sub = stack.depths[::step]

    f_xyz, f_vis, f_invis, f_conf, f_col = [], [], [], [], []
    for ei, tracks_2d in enumerate(raw_tracks_2d):
        tracks_2d = tracks_2d.swapdims(0, 1)  # (Tsub, per, 4)
        xy, occ, dist = tracks_2d[..., :2], tracks_2d[..., 2], tracks_2d[..., 3]
        visibles, invisibles, confidences = parse_tapir_track_info(occ, dist)

        grid = normalize_coords(xy[..., None, :], H, W)  # (Tsub, per, 1, 2)
        track_depths = F.grid_sample(
            depths_sub[:, None], grid, align_corners=True, padding_mode="border"
        )[:, 0]  # (Tsub, per, 1)
        xyz_cam = torch.einsum("nij,npj->npi", inv_Ks, F.pad(xy, (0, 1), value=1.0)) * track_depths
        xyz = torch.einsum("nij,npj->npi", c2ws, F.pad(xyz_cam, (0, 1), value=1.0))[..., :3]

        is_in_masks = (
            F.grid_sample(masks_sub[:, None], grid, align_corners=True).squeeze(1).squeeze(-1) == 1
        )  # (Tsub, per)
        visibles = visibles * is_in_masks
        invisibles = invisibles * is_in_masks
        confidences = confidences * is_in_masks.float()

        qi = candidate_frames[ei]
        img = stack.rgbs[qi].to(torch.float32).permute(2, 0, 1)[None] / 255.0  # (1, 3, H, W)
        col_grid = normalize_coords(tracks_2d[ei, None, None, :, :2], H, W)  # (1, 1, per, 2)
        track_colors = F.grid_sample(
            img, col_grid, align_corners=True, padding_mode="border"
        ).squeeze(0).squeeze(1).T  # (per, 3)

        visible_counts = visibles.sum(0)
        valid = visible_counts >= min(
            int(0.05 * num_frames), visible_counts.float().quantile(0.1).item()
        )
        f_xyz.append(xyz[:, valid])
        f_vis.append(visibles[:, valid])
        f_invis.append(invisibles[:, valid])
        f_conf.append(confidences[:, valid])
        f_col.append(track_colors[valid])

    return Tracks3D(
        xyz=torch.cat(f_xyz, dim=1).swapdims(0, 1),
        visibles=torch.cat(f_vis, dim=1).swapdims(0, 1),
        invisibles=torch.cat(f_invis, dim=1).swapdims(0, 1),
        confidences=torch.cat(f_conf, dim=1).swapdims(0, 1),
        colors=torch.cat(f_col, dim=0),
    )


def lift_bkgd_points(
    stack: FrameStack,
    num_samples: int,
    rng: np.random.Generator | None = None,
) -> StaticPoints3D:
    """Sample static background points from non-foreground depth (SoM get_bkgd_points port)."""
    choice = (rng.choice if rng is not None else np.random.choice)
    H, W = stack.depths.shape[1], stack.depths.shape[2]
    device = stack.depths.device
    num_frames = stack.num_frames
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(W, dtype=torch.float32, device=device),
            torch.arange(H, dtype=torch.float32, device=device),
            indexing="xy",
        ),
        dim=-1,
    )  # (H, W, 2)

    pts, normals, colors = [], [], []
    for i in range(num_frames):
        depth = stack.depths[i]
        K, w2c = stack.cameras.Ks[i], stack.cameras.w2cs[i]
        bool_mask = ((1.0 - stack.fg[i]) * stack.valid[i] * (depth > 0)).to(torch.bool)
        cam = torch.einsum(
            "ij,pj->pi", torch.linalg.inv(K), F.pad(grid[bool_mask], (0, 1), value=1.0)
        ) * depth[bool_mask][:, None]
        world = torch.einsum(
            "ij,pj->pi", torch.linalg.inv(w2c)[:3], F.pad(cam, (0, 1), value=1.0)
        )
        point_normals = normal_from_depth(depth, K, w2c)[bool_mask]
        point_colors = (stack.rgbs[i].to(torch.float32) / 255.0)[bool_mask]

        curr_num = world.shape[0]
        base = int(np.floor(num_samples / num_frames))
        per = base if i != num_frames - 1 else num_samples - (num_frames - 1) * base
        sels = choice(curr_num, (per,), replace=False) if per < curr_num else np.arange(0, curr_num)
        pts.append(world[sels])
        normals.append(point_normals[sels])
        colors.append(point_colors[sels])

    return StaticPoints3D(
        xyz=torch.cat(pts, dim=0),
        normals=torch.cat(normals, dim=0),
        colors=torch.cat(colors, dim=0),
    )


# -- motion-gated foreground (FP2c) ---------------------------------------

def gate_tracks_by_motion(
    tracks: Tracks3D,
    bg_points: StaticPoints3D,
    *,
    percentile: float = 95.0,
    threshold: float = 0.05,
) -> tuple[Tracks3D, StaticPoints3D]:
    """Keep only *ever-moved* tracks as foreground; reroute static tracks to bg.

    A track is dynamic iff the ``percentile``-th per-frame world-space displacement
    from its median center exceeds ``threshold``. The high percentile is the
    "did it move at any point" test: a *median* reducer averages brief motion away
    (an object static for most of the clip that moves briefly), while a raw *max*
    trips on single-frame lift/depth glitches; ``p95`` catches genuine motion (which
    spans several frames) and ignores one-frame outliers.

    Static tracks are converted to static background gaussians at their median
    position rather than dropped: their (segment-fg-masked) regions are excluded
    from the background point sampler, so dropping them would leave holes.
    """
    xyz = tracks.xyz  # (N, T, 3)
    if xyz.shape[0] == 0:
        return tracks, bg_points
    vis = tracks.visibles.bool()[..., None]
    masked = torch.where(vis, xyz, torch.full_like(xyz, float("nan")))
    center = torch.nanmedian(masked, dim=1).values  # (N, 3)
    spread = torch.linalg.norm(masked - center[:, None, :], dim=2)  # (N, T), nan where invisible
    q = torch.nanquantile(spread, float(percentile) / 100.0, dim=1)  # (N,)
    # Too-few-visible tracks (nan score) stay foreground — never silently drop motion.
    dynamic = torch.where(torch.isnan(q), torch.ones_like(q, dtype=torch.bool), q > float(threshold))
    if bool(dynamic.all()):
        return tracks, bg_points  # nothing static to reroute (e.g. an already-clean fg)

    fg = tracks.filter_valid(dynamic)
    static = tracks.filter_valid(~dynamic)
    s_masked = torch.where(static.visibles.bool()[..., None], static.xyz, torch.full_like(static.xyz, float("nan")))
    s_xyz = torch.nanmedian(s_masked, dim=1).values  # (M, 3) representative static position
    merged = StaticPoints3D(
        xyz=torch.cat([bg_points.xyz, s_xyz.to(bg_points.xyz)], dim=0),
        normals=torch.cat([bg_points.normals, torch.zeros_like(s_xyz).to(bg_points.normals)], dim=0),
        colors=torch.cat([bg_points.colors, static.colors.to(bg_points.colors)], dim=0),
    )
    return fg, merged


# -- orchestrator ---------------------------------------------------------

def build_native_init_inputs(
    dataset_root: str | Path,
    factor: int = 1,
    num_fg: int = 40_000,
    num_bg: int = 100_000,
    depth_type: str = "depth_anything_colmap",
    camera_type: str = "refined",
    device: str = "cpu",
    seed: int | None = None,
    tracks_dir: str | Path | None = None,
    provider: "GeometryProvider | None" = None,
    pair_loader: "Callable[[int, int], np.ndarray] | None" = None,
    motion_gate: dict | None = None,
) -> tuple[Tracks3D, StaticPoints3D, dict[str, CameraRecords], SceneNorm]:
    """Lift native init inputs for a DyCheck-format sequence (normalized frame).

    Mirrors SoM's pipeline order: lift a foreground-track subsample in the raw
    frame to derive the scene normalization, normalize the train stack, then lift
    the full foreground tracks + background points and normalize the val cameras.

    ``tracks_dir`` (optional) overrides where the 2D tracks are read from — pass
    room4d's own tracker output (see :func:`generate_native_tracks`) to lift from
    natively-produced tracks instead of the borrowed ``2d_tracks`` dir.

    ``provider`` (optional) is the geometry source — cameras + per-frame
    depth/rgb/fg. It defaults to :class:`DyCheckGeometryProvider` (the borrowed
    dataset ground truth) built from ``dataset_root``/``factor``/``depth_type``/
    ``camera_type``; pass a native backend (e.g. VGGT) to lift from
    room4d-reconstructed geometry instead. Tracks stay separate from geometry.

    Returns ``(tracks, bg_points, {"train"/"val": cameras}, scene_norm)``.
    """
    if provider is None:
        from room4d.lifting.geometry_provider import DyCheckGeometryProvider

        provider = DyCheckGeometryProvider(dataset_root, factor, depth_type, camera_type)

    rng = np.random.default_rng(seed) if seed is not None else None
    train_stack = provider.frame_stack("train", device=device)

    # Scene norm from a temporally-subsampled raw-frame track lift.
    step = max(1, train_stack.num_frames // 10)
    sample = lift_tracks_3d(
        train_stack, dataset_root, factor, num_samples=10_000, step=step, rng=rng,
        tracks_dir=tracks_dir, pair_loader=pair_loader,
    )
    norm = compute_scene_norm(train_stack.cameras.w2cs, sample.xyz)

    apply_scene_norm_stack(train_stack, norm)
    tracks = lift_tracks_3d(
        train_stack, dataset_root, factor, num_samples=num_fg, step=1, rng=rng,
        tracks_dir=tracks_dir, pair_loader=pair_loader,
    )
    bg_points = lift_bkgd_points(train_stack, num_samples=num_bg, rng=rng)

    # FP2c: keep only ever-moved tracks as deformable foreground; static tracks
    # (e.g. SAM-2-segmented background the segment fg let in) become static bg.
    if motion_gate and motion_gate.get("enabled"):
        tracks, bg_points = gate_tracks_by_motion(
            tracks, bg_points,
            percentile=float(motion_gate.get("percentile", 95.0)),
            threshold=float(motion_gate.get("threshold", 0.05)),
        )

    val_cameras = provider.cameras("val", device=device)
    apply_scene_norm_cameras(val_cameras, norm)

    cameras = {"train": train_stack.cameras, "val": val_cameras}
    return tracks, bg_points, cameras, norm


def generate_native_tracks(
    dataset_root: str | Path,
    out_dir: str | Path,
    *,
    factor: int = 1,
    depth_type: str = "depth_anything_colmap",
    camera_type: str = "refined",
    device: str = "cpu",
    tracker_cfg: dict | None = None,
) -> Path:
    """Run room4d's own point tracker over the train split and write SoM-format tracks.

    Loads the train split's RGB frames + foreground masks (the same ``stack.fg``
    the lift later re-checks) and runs :class:`~room4d.tracking.tapir_wrapper.TAPIRWrapper`
    to produce all-pairs 2D tracks, written as ``<out_dir>/<factor>x/<name_i>_<name_j>.npy``
    — the exact layout :func:`lift_tracks_3d` reads. Returns the ``<factor>x`` directory
    to pass back as ``tracks_dir``. This replaces the last borrowed input in the
    native motion-init path.
    """
    from room4d.tracking.tapir_wrapper import TAPIRWrapper

    backend = DatasetGeometryInitializer(
        {"dataset_root": str(dataset_root), "factor": factor, "split": "train"}
    )
    stack = _load_frame_stack(
        backend, "train", device, dataset_root, factor, depth_type, camera_type
    )
    frames = stack.rgbs.cpu().numpy()  # (T, H, W, 3) uint8
    masks = stack.fg.cpu().numpy()     # (T, H, W) float in [0, 1]
    frame_names = list(stack.cameras.frame_names)

    wrapper = TAPIRWrapper(tracker_cfg or {})
    all_pairs = wrapper.compute_all_pairs(frames, masks, frame_names, factor=factor)

    factor_dir = Path(out_dir) / f"{factor}x"
    # SoM per-pair layout: lift_tracks_3d reads <name_i>_<name_j>.npy directly from
    # this dir for the standalone --own-tracker flow. (The pipeline uses the packed
    # AllPairsTracks + in-memory pair_loader and never writes per-pair files.)
    all_pairs.write_som_pairs(factor_dir)
    return factor_dir
