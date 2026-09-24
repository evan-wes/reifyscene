"""The graph's handle on what renders: entity id -> indices into the trained cloud.

§3.4 of the contract promises every entity a **renderable handle**, because the op-support
matrix promises `remove(entity)` "leaves a real hole" -- and nothing can leave a hole in a
representation it cannot decompose. :mod:`reifyscene.graph.entity_gaussians` has been able to
compute that assignment for some time; nothing called it, so the manifest reported
``renderable: absent`` and the promise was unbacked.

This is the caller. It exists separately from the ``graph`` stage for a reason worth stating,
because it is easy to get backwards: **the handle indexes the TRAINED cloud, and training
happens after the graph stage.** Training densifies and prunes, so a seed index means nothing
at render time. The graph therefore cannot produce this itself; it is computed from a finished
run, where the trained checkpoint, the entity masks and the recorded image transform all exist
at once.

What lines up here:

- **Node id -> masks.** A node id embeds its canonical tracklet
  (``object:cam_00:sam2:cam_00:3``), and the reid merge map says which segment tracklets folded
  onto that canonical. An entity's mask on a frame is the union over its members.
- **The render's pixels.** The trained cameras project into the geometry's raster (518x294 on
  DAVIS), while masks are in ingest-frame pixels. They go through the transform the geometry
  stage recorded -- never a guess from image sizes.
- **Per-frame positions.** The foreground is deformable, so each training frame has its own
  Gaussian positions; the motion bases are evaluated at every training time id rather than
  assuming a canonical pose.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MIN_RATIO = 0.5
DEFAULT_MIN_FRAMES = 2


@dataclass
class EntityRenderable:
    """One entity's claim on the trained cloud."""

    node_id: str
    label: str = ""
    assigned: int = 0
    assigned_fraction: float = 0.0     # of the whole foreground cloud
    frames_masked: int = 0             # frames this entity had a mask on
    unassignable: bool = True          # earned no Gaussians: present but not isolable
    gaussian_indices: list[int] = field(default_factory=list)

    def to_dict(self, *, with_indices: bool = True) -> dict[str, Any]:
        d = asdict(self)
        if not with_indices:
            d.pop("gaussian_indices")
        return d


@dataclass
class RenderableHandles:
    """Every entity's handle, plus what no entity claimed."""

    run: str = ""
    num_foreground: int = 0
    num_background: int = 0
    num_frames: int = 0
    entities: dict[str, EntityRenderable] = field(default_factory=dict)
    unclaimed: int = 0
    unclaimed_fraction: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self, *, with_indices: bool = True) -> dict[str, Any]:
        return {
            "run": self.run,
            "num_foreground": self.num_foreground,
            "num_background": self.num_background,
            "num_frames": self.num_frames,
            "unclaimed": self.unclaimed,
            "unclaimed_fraction": round(self.unclaimed_fraction, 4),
            "notes": list(self.notes),
            "entities": {
                k: v.to_dict(with_indices=with_indices) for k, v in sorted(self.entities.items())
            },
        }


def _node_track_id(node_id: str, stream_id: str) -> str | None:
    """``object:cam_00:sam2:cam_00:3`` -> ``sam2:cam_00:3``; None for non-entity nodes."""
    prefix = f"{stream_id}:"
    for kind in ("human", "object"):
        head = f"{kind}:{prefix}"
        if node_id.startswith(head):
            return node_id[len(head):]
    return None


def trained_fg_positions(
    bundle: Mapping[str, Any], *, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """``(positions (T, G, 3), w2c (T, 4, 4), K (T, 3, 3), source_frame_indices)``.

    Positions are the foreground means warped to every training frame: the foreground is
    deformable, so one canonical pose would project the wrong pixels on all but one frame.
    CPU is the default because this is a projection, not a render.
    """
    import torch

    from reifyscene.render.gaussian4d.motion_init import bundle_to_scene

    scene, _cameras = bundle_to_scene(bundle, device=device)
    time_ids = np.asarray(bundle["train_time_ids"], dtype=int)
    with torch.no_grad():
        means_t, _quats = scene.fg.warp(
            scene.motion_bases, torch.as_tensor(time_ids, device=device)
        )
    positions = means_t.detach().cpu().numpy().transpose(1, 0, 2)  # (G, T, 3) -> (T, G, 3)
    w2c = np.asarray(bundle["train_w2cs"], dtype=np.float64)
    intrinsics = np.asarray(bundle["train_Ks"], dtype=np.float64)
    frames = [int(str(n).rsplit(":", 1)[-1]) for n in np.asarray(bundle["train_frame_names"])]
    return positions, w2c, intrinsics, frames


def entity_masks_in_render_pixels(
    run_dir: Path, *, stream_id: str = "cam_00"
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, str], list[str]]:
    """``({frame: {node_id: mask}}, {node_id: label}, notes)`` in the geometry's raster.

    An entity's mask on a frame is the union over the segment tracklets that the reid merge
    map folds onto its canonical -- a node is one physical thing, and its members are the
    tracklets that turned out to be it.
    """
    from reifyscene.geometry.image_transform import ImageTransform
    from reifyscene.perception.instance.instance_artifact import InstanceArtifact

    notes: list[str] = []
    geometry = _read_json(run_dir / "geometry" / "artifact.json")
    raw = ((geometry.get("metadata") or {}).get("image_transforms") or {}).get(stream_id)
    if raw is None:
        raise ValueError(
            f"{run_dir.name}: geometry recorded no image transform for {stream_id!r}; masks "
            "cannot be put into the render's pixels without guessing it"
        )
    transform = ImageTransform.from_dict(raw)

    graph = _read_json(run_dir / "graph" / "artifact.json")
    canonical_to_node: dict[str, str] = {}
    labels: dict[str, str] = {}
    for raw_node in graph.get("nodes") or []:
        node = raw_node.get("value", raw_node)
        track = _node_track_id(str(node.get("node_id", "")), stream_id)
        if track is None:
            continue
        canonical_to_node[track] = str(node["node_id"])
        labels[str(node["node_id"])] = str(node.get("label", ""))

    merge_map: dict[str, str] = {}
    reid = run_dir / "reid" / "artifact.json"
    if reid.is_file():
        merge_map = {str(k): str(v) for k, v in (_read_json(reid).get("merge_map") or {}).items()}

    def canonical(track: str) -> str:
        seen: set[str] = set()
        cur = track
        while cur in merge_map and merge_map[cur] != cur and cur not in seen:
            seen.add(cur)
            cur = merge_map[cur]
        return cur

    inst = InstanceArtifact.load(run_dir / "segment")
    by_frame: dict[int, dict[str, np.ndarray]] = {}
    unmapped: set[str] = set()
    for obs in inst.observations:
        if obs.frame_ref.stream_id != stream_id:
            continue
        mask = inst.mask_arrays.get(obs.instance_id)
        if mask is None:
            continue
        node_id = canonical_to_node.get(canonical(str(obs.track_id)))
        if node_id is None:
            unmapped.add(str(obs.track_id))
            continue
        placed = transform.apply_to_image(np.asarray(mask), kind="mask").astype(bool)
        frame = int(obs.frame_ref.frame_idx)
        slot = by_frame.setdefault(frame, {})
        slot[node_id] = placed if node_id not in slot else (slot[node_id] | placed)
    if unmapped:
        # Tracklets the graph did not turn into a node: dropped by a support or size rule.
        # Named rather than silently ignored, because they are Gaussians nothing can claim.
        notes.append(f"{len(unmapped)} tracklet(s) have no graph node and were skipped")
    return by_frame, labels, notes


def compute_renderable_handles(
    run_dir: Path,
    *,
    stream_id: str = "cam_00",
    min_ratio: float = DEFAULT_MIN_RATIO,
    min_frames: int = DEFAULT_MIN_FRAMES,
    min_gaussians: int = 1,
    device: str = "cpu",
    checkpoint: str = "trained_dynamic.npz",
) -> RenderableHandles:
    """Assign the trained foreground cloud to entities, and say what nothing claimed."""
    run_dir = Path(run_dir)
    path = run_dir / "render" / checkpoint
    if not path.is_file():
        raise FileNotFoundError(f"{run_dir.name} has no trained checkpoint at {path}")

    with np.load(path, allow_pickle=True) as bundle:
        data = {k: bundle[k] for k in bundle.files}
    positions, w2c, intrinsics, frames = trained_fg_positions(data, device=device)
    masks, labels, notes = entity_masks_in_render_pixels(run_dir, stream_id=stream_id)

    from reifyscene.graph.entity_gaussians import assign_gaussians_to_entities

    support = assign_gaussians_to_entities(
        masks, frame_indices=frames, positions=positions, w2c=w2c, intrinsics=intrinsics,
        min_ratio=min_ratio, min_frames=min_frames,
    )
    num_fg = int(positions.shape[1])
    handles = RenderableHandles(
        run=run_dir.name, num_foreground=num_fg,
        num_background=int(np.asarray(data["bg_means"]).shape[0]) if "bg_means" in data else 0,
        num_frames=len(frames), notes=notes,
    )
    claimed = 0
    for node_id, label in sorted(labels.items()):
        indices = [int(i) for i in support.track_ids_for(node_id)]
        claimed += len(indices)
        handles.entities[node_id] = EntityRenderable(
            node_id=node_id, label=label, assigned=len(indices),
            assigned_fraction=len(indices) / num_fg if num_fg else 0.0,
            frames_masked=sum(1 for f in masks if node_id in masks[f]),
            unassignable=len(indices) < int(min_gaussians),
            gaussian_indices=indices,
        )
    handles.unclaimed = num_fg - claimed
    handles.unclaimed_fraction = handles.unclaimed / num_fg if num_fg else 0.0
    return handles


def drop_gaussians(bundle: Mapping[str, Any], indices: list[int] | np.ndarray) -> dict[str, Any]:
    """A copy of ``bundle`` with those foreground Gaussians removed.

    This is how ``remove(entity)`` is executed: the edit is a checkpoint transform, so the
    render path needs no exclusion parameter threaded through it and what renders afterwards is
    a real scene rather than a masked one. Every per-Gaussian foreground array is filtered
    together; the motion bases are shared and untouched, as is the background.
    """
    keep = np.ones(int(np.asarray(bundle["fg_means"]).shape[0]), dtype=bool)
    idx = np.asarray(list(indices), dtype=int)
    if idx.size:
        if idx.min() < 0 or idx.max() >= keep.size:
            raise IndexError(
                f"gaussian index out of range for a cloud of {keep.size}: "
                f"[{idx.min()}, {idx.max()}]"
            )
        keep[idx] = False
    out: dict[str, Any] = {}
    for key, value in bundle.items():
        arr = np.asarray(value)
        # Per-Gaussian foreground arrays are exactly those whose first axis is the cloud.
        if key.startswith("fg_") and arr.ndim >= 1 and arr.shape[0] == keep.size:
            out[key] = arr[keep]
        else:
            out[key] = arr
    return out


def _read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(Path(path).read_text())
