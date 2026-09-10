"""Procrustes + clustering motion initialization (Shape-of-Motion port).

The algorithm half of reifyscene's native motion initializer. Given lifted 3D
foreground trajectories (:class:`~reifyscene.lifting.tracks_3d.Tracks3D`) and static
background points (:class:`~reifyscene.lifting.tracks_3d.StaticPoints3D`), this:

1. clusters the trajectories into ``K`` rigid motion groups (KMeans on per-frame
   velocity directions),
2. fits each group's per-frame SE(3) by weighted Procrustes against the
   canonical frame → :class:`MotionBases` + per-Gaussian ``motion_coefs``,
3. places a foreground Gaussian at each track's canonical position and seeds the
   static background.

Ported from SoM's ``flow3d/init_utils.py`` + ``flow3d/transforms.py`` onto
reifyscene's fields, with two deliberate swaps the handoff settled: cuML KMeans →
``sklearn`` KMeans (SoM's default cluster method is already kmeans), and cupy
masked interpolation → a numpy port. ``roma`` is dropped in favor of reifyscene's
own rotation helpers. The Gaussian parameterization (means raw / scales log /
colors+opacities logit / coefs raw) is identical to SoM's and to reifyscene's fields.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from reifyscene.lifting.tracks_3d import StaticPoints3D, Tracks3D
from reifyscene.render.gaussian4d.dynamic_field import DynamicGaussianField
from reifyscene.render.gaussian4d.gaussian_field import GaussianField
from reifyscene.render.gaussian4d.motion import MotionBases
from reifyscene.render.gaussian4d.transforms import rmat_to_cont_6d

# Identity 6D continuous rotation (first two columns of the identity matrix).
_ID_ROT_6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


# -- low-level helpers ----------------------------------------------------

def knn(x: torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Distances/indices to the ``k`` nearest neighbors (excluding self)."""
    x_np = x.detach().cpu().numpy()
    model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)
    distances, indices = model.kneighbors(x_np)
    return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)


def get_weights_for_procrustes(clusters: torch.Tensor, visibilities: torch.Tensor | None = None) -> torch.Tensor:
    """Per-point robustness weights for Procrustes (SoM port).

    Down-weights points far from the per-frame cluster median, normalizes by the
    median distance, masks the farthest decile, and folds in visibility.
    """
    clusters_median = clusters.median(dim=-2, keepdim=True)[0]
    dists2center = torch.norm(clusters - clusters_median, dim=-1)
    dists2center = dists2center / dists2center.median(dim=-1, keepdim=True)[0]
    weights = torch.exp(-dists2center)
    weights = weights / (weights.mean(dim=-1, keepdim=True) + 1e-6)
    if visibilities is not None:
        weights = weights * (visibilities.float() + 1e-6)
    invalid = dists2center > np.quantile(dists2center.cpu().numpy(), 0.9)
    invalid = invalid | torch.isnan(weights)
    weights[invalid] = 0
    return weights


def _rt_to_mat4(R: torch.Tensor, t: torch.Tensor, s: torch.Tensor | None = None) -> torch.Tensor:
    """(3,3) R + (3,) t (+ optional scale ``s``) -> (4,4) (SoM convention: 1/s in the homogeneous slot)."""
    mat34 = torch.cat([R, t[..., None]], dim=-1)
    if s is None:
        bottom = mat34.new_tensor([[0.0, 0.0, 0.0, 1.0]])
    else:
        bottom = F.pad(1.0 / s[..., None, None], (3, 0), value=0.0)
    return torch.cat([mat34, bottom], dim=-2)


def solve_procrustes(
    src: torch.Tensor,
    dst: torch.Tensor,
    weights: torch.Tensor | None = None,
    enforce_se3: bool = True,
    rot_type: str = "6d",
):
    """Weighted Procrustes alignment ``src -> dst`` via SVD (SoM port).

    Solves ``min_{s,R,t} || s (src R^T + t) - dst ||`` with ``R`` a rotation
    (``enforce_se3`` fixes ``s=1``). Returns ``((rot, t, s), (err, err_before))``
    where ``rot`` is 6D (or a matrix if ``rot_type='mat'``) and the errors are
    weighted mean L2 distances before/after alignment.
    """
    if weights is None:
        weights = src.new_ones(src.shape[0])
    weights = weights[:, None] / weights.sum()
    src_mean = (src * weights).sum(dim=0)
    dst_mean = (dst * weights).sum(dim=0)
    src_cent = src - src_mean
    dst_cent = dst - dst_mean
    if not enforce_se3:
        src_scale = (src_cent**2 * weights).sum(dim=-1).mean().sqrt()
        dst_scale = (dst_cent**2 * weights).sum(dim=-1).mean().sqrt()
    else:
        src_scale = dst_scale = src.new_tensor(1.0)
    src_scaled = src_cent / src_scale
    dst_scaled = dst_cent / dst_scale
    matrix = (weights * dst_scaled).T @ src_scaled
    U, _, Vh = torch.linalg.svd(matrix)
    S = torch.eye(3, device=src.device)
    if torch.det(U) * torch.det(Vh) < 0:
        S[2, 2] = -1
    R = U @ S @ Vh
    if rot_type == "6d":
        rot = rmat_to_cont_6d(R)
    elif rot_type == "mat":
        rot = R
    else:
        raise ValueError(f"unsupported rot_type {rot_type!r}")
    s = dst_scale / src_scale
    t = dst_mean / s - src_mean @ R.T
    sim3 = rot, t, s

    procrustes_dst = torch.einsum("ij,nj->ni", _rt_to_mat4(R, t, s), F.pad(src, (0, 1), value=1.0))
    procrustes_dst = procrustes_dst[:, :3] / procrustes_dst[:, 3:]
    error_before = (torch.linalg.norm(dst - src, dim=-1) * weights[:, 0]).sum()
    error = (torch.linalg.norm(dst - procrustes_dst, dim=-1) * weights[:, 0]).sum()
    return sim3, (error.item(), error_before.item())


def _rotvec_to_unitquat(rotvec: torch.Tensor) -> torch.Tensor:
    """Axis-angle (..., 3) -> unit quaternion (..., 4) in wxyz (roma-free)."""
    angle = rotvec.norm(dim=-1, keepdim=True)
    axis = rotvec / angle.clamp_min(1e-8)
    half = 0.5 * angle
    quat = torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)
    return quat


def _interp_masked_tracks(xyz: torch.Tensor, visibles: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate missing positions over time per track (numpy port of
    SoM's cupy ``batched_interp_masked``); endpoints hold (constant extrapolation).

    xyz (N, T, 3), visibles (N, T) bool -> (N, T, 3). Fully-invisible tracks pass
    through unchanged.
    """
    xyz_np = xyz.detach().cpu().numpy().astype(np.float64)
    vis = visibles.detach().cpu().numpy().astype(bool)
    num_tracks, num_frames = vis.shape
    tgrid = np.arange(num_frames)
    out = xyz_np.copy()
    for n in range(num_tracks):
        m = vis[n]
        if m.all() or not m.any():
            continue
        idx = np.where(m)[0]
        for d in range(3):
            out[n, :, d] = np.interp(tgrid, idx, xyz_np[n, idx, d])
    return torch.from_numpy(out.astype(np.float32)).to(xyz.device)


def sample_initial_bases_centers(
    cano_t: int, tracks: Tracks3D, num_bases: int, seed: int | None = None
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """KMeans on per-frame velocity directions -> cluster centers + labels (SoM port).

    Only the ``kmeans`` mode is ported (SoM's default; cuML's HDBSCAN/farthest
    variants are dropped). Returns ``(centers (1, K, 3), K, labels (N,))``.
    """
    means_canonical = tracks.xyz[:, cano_t].clone()
    num_tracks = tracks.xyz.shape[0]
    xyz_interp = _interp_masked_tracks(tracks.xyz, tracks.visibles)
    velocities = xyz_interp[:, 1:] - xyz_interp[:, :-1]
    vel_dirs = (velocities / (velocities.norm(dim=-1, keepdim=True) + 1e-5)).reshape(num_tracks, -1)

    model = KMeans(n_clusters=num_bases, random_state=seed, n_init=10)
    model.fit(vel_dirs.detach().cpu().numpy())
    labels = torch.as_tensor(model.labels_, device=means_canonical.device)
    num_bases = int(labels.max().item()) + 1
    sampled_centers = torch.stack(
        [means_canonical[labels == i].median(dim=0).values for i in range(num_bases)]
    )[None]
    return sampled_centers, num_bases, labels


def init_motion_params_with_procrustes(
    tracks: Tracks3D,
    num_bases: int,
    cano_t: int,
    rot_type: str = "6d",
    min_mean_weight: float = 0.1,
    seed: int | None = None,
) -> tuple[MotionBases, torch.Tensor, Tracks3D]:
    """Cluster trajectories + per-cluster/per-frame Procrustes -> motion bases (SoM port).

    Culls outlier/never-visible tracks, clusters the rest into ``num_bases``
    rigid groups, walks outward from ``cano_t`` solving a weighted SE(3) per
    cluster per frame, and builds soft ``motion_coefs`` from distance to cluster
    centers. Returns ``(MotionBases, motion_coefs (N, K), filtered_tracks)``.
    """
    assert rot_type == "6d", "reifyscene uses the 6D rotation parameterization"
    device = tracks.xyz.device
    num_frames = tracks.xyz.shape[1]
    means_cano = tracks.xyz[:, cano_t].clone()

    # Remove spatial outliers (farthest 5% from the median) and never-visible tracks.
    scene_center = means_cano.median(dim=0).values
    dists = torch.norm(means_cano - scene_center, dim=-1)
    valid_mask = (dists < torch.quantile(dists, 0.95)) & tracks.visibles.any(dim=1)
    tracks = tracks.filter_valid(valid_mask)
    means_cano = means_cano[valid_mask]

    sampled_centers, num_bases, labels = sample_initial_bases_centers(cano_t, tracks, num_bases, seed=seed)
    # Drop tiny clusters (<100 points); renumber bases to the survivors.
    ids, counts = labels.unique(return_counts=True)
    ids = ids[counts > 100]
    num_bases = len(ids)
    sampled_centers = sampled_centers[:, ids]

    dists2centers = torch.norm(means_cano[:, None] - sampled_centers, dim=-1)
    motion_coefs = 10 * torch.exp(-dists2centers)

    rot_dim = 6
    id_rot = torch.tensor(_ID_ROT_6D, device=device)
    init_rots = id_rot.reshape(1, 1, rot_dim).repeat(num_bases, num_frames, 1)
    init_ts = torch.zeros(num_bases, num_frames, 3, device=device)

    # Walk both directions out from the canonical frame so each step's init is the
    # previous solved transform (skipped frames hold the previous pose).
    tgt_ts = list(range(cano_t - 1, -1, -1)) + list(range(cano_t, num_frames))
    for n, cluster_id in enumerate(ids):
        mask_in_cluster = labels == cluster_id
        cluster = tracks.xyz[mask_in_cluster].transpose(0, 1)         # (T, n_pts, 3)
        visibilities = tracks.visibles[mask_in_cluster].swapaxes(0, 1)
        confidences = tracks.confidences[mask_in_cluster].swapaxes(0, 1)
        weights = get_weights_for_procrustes(cluster, visibilities)
        prev_t = cano_t
        for cur_t in tgt_ts:
            procrustes_weights = (
                weights[cano_t] * weights[cur_t] * (confidences[cano_t] + confidences[cur_t]) / 2
            )
            if procrustes_weights.sum() < min_mean_weight * num_frames:
                init_rots[n, cur_t] = init_rots[n, prev_t]
                init_ts[n, cur_t] = init_ts[n, prev_t]
            else:
                (init_rot, init_t, _), _ = solve_procrustes(
                    cluster[cano_t], cluster[cur_t], weights=procrustes_weights,
                    enforce_se3=True, rot_type=rot_type,
                )
                init_rots[n, cur_t] = init_rot
                init_ts[n, cur_t] = init_t
            prev_t = cur_t

    return MotionBases(init_rots, init_ts), motion_coefs, tracks


def init_fg_from_tracks_3d(cano_t: int, tracks: Tracks3D, motion_coefs: torch.Tensor) -> DynamicGaussianField:
    """Foreground field: a Gaussian at each track's canonical position (SoM port).

    Scales from the mean 3-NN distance in the canonical frame (5–95% clamped),
    colors/opacities in logit space, random orientations, softmax ``motion_coefs``.
    """
    device = tracks.xyz.device
    num_fg = tracks.xyz.shape[0]
    colors_logit = torch.logit(tracks.colors.to(device), eps=1e-6)
    dists, _ = knn(tracks.xyz[:, cano_t], 3)
    scales = torch.from_numpy(dists).to(device).mean(dim=-1, keepdim=True)
    scales = scales.clamp(torch.quantile(scales, 0.05), torch.quantile(scales, 0.95))
    log_scales = torch.log(scales.repeat(1, 3))
    means = tracks.xyz[:, cano_t]
    quats = torch.rand(num_fg, 4, device=device)
    logit_opacities = torch.logit(torch.full((num_fg,), 0.7, device=device))
    return DynamicGaussianField(
        means=means, quats=quats, log_scales=log_scales,
        logit_opacities=logit_opacities, colors_logit=colors_logit, motion_coefs=motion_coefs,
    )


def init_bg(points: StaticPoints3D) -> GaussianField:
    """Static background field from depth-sampled points (SoM port).

    Scales from the mean 3-NN distance; orientations rotate the local +z axis to
    each point's surface normal (roma-free, non-finite quats fall back to identity).
    """
    device = points.xyz.device
    num_bg = points.xyz.shape[0]
    colors_logit = torch.logit(points.colors.to(device), eps=1e-6)
    dists, _ = knn(points.xyz, 3)
    scales = torch.from_numpy(dists).to(device).mean(dim=-1, keepdim=True)
    log_scales = torch.log(scales.repeat(1, 3))
    means = points.xyz

    local_z = points.normals.new_tensor([[0.0, 0.0, 1.0]]).expand_as(points.normals)
    axis = F.normalize(torch.linalg.cross(local_z, points.normals, dim=-1), dim=-1)
    angle = (local_z * points.normals).sum(-1, keepdim=True).clamp(-1.0, 1.0).acos()
    quats = _rotvec_to_unitquat(axis * angle)
    identity = quats.new_tensor([1.0, 0.0, 0.0, 0.0])
    quats = torch.where(torch.isfinite(quats).all(dim=-1, keepdim=True), quats, identity)

    logit_opacities = torch.logit(torch.full((num_bg,), 0.7, device=device))
    return GaussianField(
        means=means, quats=quats, log_scales=log_scales,
        logit_opacities=logit_opacities, colors_logit=colors_logit,
    )
