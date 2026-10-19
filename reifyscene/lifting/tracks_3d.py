"""Lifted 3D observation containers for the native motion initializer.

reifyscene analogs of Shape-of-Motion's ``TrackObservations`` / ``StaticObservations``
(``flow3d/tensor_dataclass.py``). ``Tracks3D`` holds foreground point
*trajectories* (one 3D position per frame per track) plus the per-frame
visibility / confidence and a single canonical color per track; ``StaticPoints3D``
holds a static background point set with normals. Both are consumed by the
clustering + Procrustes + Gaussian-placement stage (PR-B); this module only
defines the shapes so the lifting bridge (PR-A) and the initializer agree on a
contract.

The fields are kept as individual tensors deliberately (mirroring SoM): a single
``filter_valid`` mask must drop the same rows from every field at once, so they
are always masked together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


def _map(value, fn: Callable[[torch.Tensor], torch.Tensor]):
    """Apply ``fn`` to every tensor field (an absent optional one stays None)."""
    return type(value)(**{k: None if v is None else fn(v) for k, v in vars(value).items()})


@dataclass
class Tracks3D:
    """Foreground 3D trajectories over ``T`` frames for ``N`` tracks.

    xyz          (N, T, 3) world-space position per track per frame
    visibles     (N, T)    bool, visible & confident at that frame
    invisibles   (N, T)    bool, occluded but confident at that frame
    confidences  (N, T)    float in [0, 1], 0 where neither visible nor invisible
    colors       (N, 3)    canonical RGB in [0, 1] (sampled at the query frame)
    owners       (N,)      the id-map code at the query pixel (reifyscene.graph.id_maps): who
                           owns the Gaussian the track seeds. None when no id maps were given.
    """

    xyz: torch.Tensor
    visibles: torch.Tensor
    invisibles: torch.Tensor
    confidences: torch.Tensor
    colors: torch.Tensor
    owners: torch.Tensor | None = None

    @property
    def num_tracks(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def num_frames(self) -> int:
        return int(self.xyz.shape[1])

    def check_sizes(self) -> bool:
        dims = self.xyz.shape[:-1]
        return (
            self.visibles.shape == dims
            and self.invisibles.shape == dims
            and self.confidences.shape == dims
            and self.colors.shape[:-1] == dims[:-1]
            and self.xyz.shape[-1] == 3
            and self.colors.shape[-1] == 3
            and (self.owners is None or self.owners.shape == dims[:-1])
        )

    def filter_valid(self, valid_mask: torch.Tensor) -> "Tracks3D":
        """Keep only the tracks (rows) selected by a (N,) boolean mask."""
        return _map(self, lambda x: x[valid_mask])

    def to(self, device: torch.device | str) -> "Tracks3D":
        return _map(self, lambda x: x.to(device))


@dataclass
class StaticPoints3D:
    """Static background points: ``xyz`` / ``normals`` / ``colors`` all (M, 3); ``owners``
    (M,) id-map codes, as on :class:`Tracks3D`."""

    xyz: torch.Tensor
    normals: torch.Tensor
    colors: torch.Tensor
    owners: torch.Tensor | None = None

    @property
    def num_points(self) -> int:
        return int(self.xyz.shape[0])

    def check_sizes(self) -> bool:
        dims = self.xyz.shape
        return (self.normals.shape == dims and self.colors.shape == dims
                and (self.owners is None or self.owners.shape == dims[:1]))

    def filter_valid(self, valid_mask: torch.Tensor) -> "StaticPoints3D":
        return _map(self, lambda x: x[valid_mask])

    def to(self, device: torch.device | str) -> "StaticPoints3D":
        return _map(self, lambda x: x.to(device))
