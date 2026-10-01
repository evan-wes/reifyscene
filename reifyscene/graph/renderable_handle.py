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
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from reifyscene.api.run_outputs import Checkpoint, RunOutputs

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


def _node_track_id(node: Mapping[str, Any]) -> tuple[str, str] | None:
    """``(stream id, canonical track id)`` of an entity node, as the graph records them; None
    for any other node."""
    if str(node.get("node_type", "")) not in ("human", "object"):
        return None
    attrs = node["attrs"]
    return str(attrs["stream_id"]), str(attrs["track_id"])


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
    from reifyscene.common.frame_keys import parse_key

    frames = [parse_key(str(n))[1] for n in np.asarray(bundle["train_frame_names"])]
    return positions, w2c, intrinsics, frames


def entity_masks_in_render_pixels(
    run: RunOutputs, *, stream_id: str | None = None
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, str], list[str]]:
    """``({frame: {node_id: mask}}, {node_id: label}, notes)`` in the geometry's raster.

    An entity's mask on a frame is the union over the segment tracklets that the reid merge
    map folds onto its canonical -- a node is one physical thing, and its members are the
    tracklets that turned out to be it.
    """
    placed = render_pixel_masks(run, stream_id=stream_id)
    return placed.by_frame, placed.labels, placed.notes


@dataclass
class RenderPixelMasks:
    """Everything segmentation says about the render raster, including what the graph dropped.

    ``unmapped`` holds, per frame, the union of the tracklets that have no graph node. They are
    real things in the image that nothing can claim, so a consumer that trains on these masks
    must not read their pixels as background.
    """

    by_frame: dict[int, dict[str, np.ndarray]]
    labels: dict[str, str]
    notes: list[str]
    unmapped: dict[int, np.ndarray]
    transform: Any                      # ImageTransform; Any keeps the import lazy


def render_pixel_masks(run: RunOutputs, *, stream_id: str | None = None) -> RenderPixelMasks:
    """:func:`entity_masks_in_render_pixels`, plus the unmapped tracklets and the transform.

    ``stream_id`` defaults to the run's only stream
    (:class:`~reifyscene.ingest.streams.RunStreams`).
    """
    notes: list[str] = []
    stream = run.streams.resolve(stream_id)
    stream_id = stream.id
    transform = stream.transform          # raises when the geometry recorded none

    graph = run.record("graph")
    canonical_to_node: dict[str, str] = {}
    labels: dict[str, str] = {}
    for raw_node in graph.get("nodes") or []:
        node = raw_node.get("value", raw_node)
        found = _node_track_id(node)
        if found is None or found[0] != stream_id:
            continue
        canonical_to_node[found[1]] = str(node["node_id"])
        labels[str(node["node_id"])] = str(node.get("label", ""))

    merge_map: dict[str, str] = {}
    if run.has("reid"):
        merge_map = {str(k): str(v) for k, v in run.record("reid")["merge_map"].items()}

    def canonical(track: str) -> str:
        seen: set[str] = set()
        cur = track
        while cur in merge_map and merge_map[cur] != cur and cur not in seen:
            seen.add(cur)
            cur = merge_map[cur]
        return cur

    inst = run.artifact("segment")
    by_frame: dict[int, dict[str, np.ndarray]] = {}
    unmapped_masks: dict[int, np.ndarray] = {}
    unmapped: set[str] = set()
    for obs in inst.observations:
        if obs.frame_ref.stream_id != stream_id:
            continue
        mask = inst.mask_arrays.get(obs.instance_id)
        if mask is None:
            continue
        placed = transform.apply_to_image(np.asarray(mask), kind="mask").astype(bool)
        frame = int(obs.frame_ref.frame_idx)
        node_id = canonical_to_node.get(canonical(str(obs.track_id)))
        if node_id is None:
            unmapped.add(str(obs.track_id))
            prev = unmapped_masks.get(frame)
            unmapped_masks[frame] = placed if prev is None else (prev | placed)
            continue
        slot = by_frame.setdefault(frame, {})
        slot[node_id] = placed if node_id not in slot else (slot[node_id] | placed)
    if unmapped:
        # Tracklets the graph did not turn into a node: dropped by a support or size rule.
        # Named rather than silently ignored, because they are Gaussians nothing can claim.
        notes.append(f"{len(unmapped)} tracklet(s) have no graph node and were skipped")
    return RenderPixelMasks(by_frame, labels, notes, unmapped_masks, transform)


def compute_renderable_handles(
    checkpoint: Checkpoint,
    *,
    stream_id: str | None = None,
    min_ratio: float = DEFAULT_MIN_RATIO,
    min_frames: int = DEFAULT_MIN_FRAMES,
    min_gaussians: int = 1,
    device: str = "cpu",
) -> RenderableHandles:
    """Assign ``checkpoint``'s trained foreground cloud to its run's entities, and say what
    nothing claimed."""
    data = checkpoint.load()
    positions, w2c, intrinsics, frames = trained_fg_positions(data, device=device)
    masks, labels, notes = entity_masks_in_render_pixels(checkpoint.run, stream_id=stream_id)

    from reifyscene.graph.entity_gaussians import assign_gaussians_to_entities

    support = assign_gaussians_to_entities(
        masks, frame_indices=frames, positions=positions, w2c=w2c, intrinsics=intrinsics,
        min_ratio=min_ratio, min_frames=min_frames,
    )
    num_fg = int(positions.shape[1])
    handles = RenderableHandles(
        run=checkpoint.run.name, num_foreground=num_fg,
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


def drop_gaussians(
    bundle: Mapping[str, Any],
    indices: list[int] | np.ndarray,
    *,
    bg_indices: list[int] | np.ndarray = (),
) -> dict[str, Any]:
    """A copy of ``bundle`` with those foreground (and background) Gaussians removed.

    This is how ``remove(entity)`` is executed: the edit is a checkpoint transform, so the
    render path needs no exclusion parameter threaded through it and what renders afterwards is
    a real scene rather than a masked one. Every per-Gaussian array of a field is filtered
    together; the motion bases are shared and untouched. ``bg_indices`` exists because an
    entity the motion split put in the static cloud -- a standing person -- lives there, and
    removing only its foreground Gaussians would leave it standing.
    """
    out: dict[str, Any] = dict(bundle)
    for prefix, idx_in in (("fg_", indices), ("bg_", bg_indices)):
        if f"{prefix}means" not in bundle:
            if len(idx_in):
                raise KeyError(f"bundle has no {prefix}means to remove from")
            continue
        keep = np.ones(int(np.asarray(bundle[f"{prefix}means"]).shape[0]), dtype=bool)
        idx = np.asarray(list(idx_in), dtype=int)
        if idx.size:
            if idx.min() < 0 or idx.max() >= keep.size:
                raise IndexError(
                    f"gaussian index out of range for a cloud of {keep.size}: "
                    f"[{idx.min()}, {idx.max()}]"
                )
            keep[idx] = False
        for key, value in bundle.items():
            arr = np.asarray(value)
            # A field's per-Gaussian arrays are exactly those whose first axis is its cloud.
            if key.startswith(prefix) and arr.ndim >= 1 and arr.shape[0] == keep.size:
                out[key] = arr[keep]
    return {k: np.asarray(v) for k, v in out.items()}


def identity_probs(bundle: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str | None]]:
    """``(fg_probs (Nf, C), bg_probs (Nb, C), class node ids)`` from a checkpoint's identity.

    Each Gaussian's own identity vector through the trained classifier -- the learned
    assignment (I2/I3 of the identity scope), as opposed to the projection vote.
    """
    if "identity_classifier_w" not in bundle:
        raise ValueError("checkpoint was not trained with identity")
    w = np.asarray(bundle["identity_classifier_w"], dtype=np.float64)
    b = np.asarray(bundle["identity_classifier_b"], dtype=np.float64)

    def softmax(x: np.ndarray) -> np.ndarray:
        z = x @ w.T + b
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    fg = softmax(np.asarray(bundle["fg_identity"], dtype=np.float64))
    bg = (softmax(np.asarray(bundle["bg_identity"], dtype=np.float64))
          if "bg_identity" in bundle else np.zeros((0, w.shape[0])))
    nodes = [str(x) or None for x in np.asarray(bundle["identity_node_ids"])]
    return fg, bg, nodes


def identity_handles(
    bundle: Mapping[str, Any], *, threshold: float = 0.5
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{node_id: (fg_indices, bg_indices)}``: Gaussians whose probability for it >= threshold.

    Removal by probability threshold, no convex hull (§3.2 of the identity scope). A
    threshold at or above 0.5 makes the handles disjoint; below it they may overlap, which
    is reported rather than resolved -- a Gaussian two entities both claim is ambiguous.
    """
    fg, bg, nodes = identity_probs(bundle)
    return {
        node: (np.flatnonzero(fg[:, k] >= threshold), np.flatnonzero(bg[:, k] >= threshold))
        for k, node in enumerate(nodes)
        if node is not None
    }
