"""Evidence-majority assignment (reifyscene/graph/assignment.py) -- FlashSplat's rule.

What these pin: the evidence is exactly each Gaussian's blending weight per class (so the
rule is FlashSplat's, not an approximation of it); a Gaussian goes to a class only on a
majority of its evidence, which is what sends a big background splat that grazes an entity
to background; no evidence means no class; and slackness trades precision for completeness
in the direction their scripts use it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from reifyscene.graph.assignment import assign, assignment_handles, evidence_matrix
from reifyscene.graph.id_maps import IGNORE
from reifyscene.render.gaussian4d.rasterizer import RenderCamera, gsplat_available

requires_gsplat = pytest.mark.skipif(
    not gsplat_available(), reason="rasterizer tests need gsplat and a CUDA device"
)


def _rows(member: torch.Tensor) -> list[list[int]]:
    return [torch.nonzero(r).flatten().tolist() for r in member]


# -- the rule ---------------------------------------------------------------------------------

def test_majority_of_evidence_wins_and_a_graze_does_not() -> None:
    counts = torch.tensor([
        [0.0, 9.0, 1.0],     # clearly class 1
        [8.0, 2.0, 0.0],     # a background splat that grazes class 1: background only
        [0.0, 0.0, 0.0],     # never seen: no class at all
        [4.0, 4.0, 0.0],     # a tie is not a majority
    ])
    assert _rows(assign(counts)) == [[1], [0], [], []]


def test_positive_slackness_demands_a_clearer_majority() -> None:
    counts = torch.tensor([[4.0, 6.0], [1.0, 9.0]])
    assert _rows(assign(counts, slackness=0.0)) == [[1], [1]]
    assert _rows(assign(counts, slackness=0.4)) == [[], [1]]


def test_negative_slackness_takes_partial_members_for_removal() -> None:
    """Their removal scripts use about -0.8: a Gaussian that is only partly the object is
    still taken, so removal leaves less behind -- and it may then belong to two classes."""
    counts = torch.tensor([[7.0, 3.0, 0.0]])
    assert _rows(assign(counts)) == [[0]]
    assert _rows(assign(counts, slackness=-0.8)) == [[0, 1]]


def test_handles_split_membership_into_fields() -> None:
    member = torch.zeros((6, 3), dtype=torch.bool)       # fg: 0..2, bg: 3..5
    member[[0, 2, 5], 1] = True
    member[4, 2] = True
    h = assignment_handles(member, num_fg=3, node_ids=[None, "a", "b"])
    assert h["a"][0].tolist() == [0, 2] and h["a"][1].tolist() == [2]
    assert h["b"][0].tolist() == [] and h["b"][1].tolist() == [1]


# -- the evidence, on the rasterizer ----------------------------------------------------------

def _scene(means, scales, opacities):
    from reifyscene.render.gaussian4d.motion_init import bundle_to_scene

    g, k, t = len(means), 1, 1
    rots = np.zeros((k, t, 6), np.float32)
    rots[:, :, 0] = rots[:, :, 4] = 1.0
    logit = np.log(np.asarray(opacities) / (1 - np.asarray(opacities))).astype(np.float32)
    bundle = {
        "fg_means": np.asarray(means, np.float32),
        "fg_quats": np.tile(np.array([1, 0, 0, 0], np.float32), (g, 1)),
        "fg_scales": np.log(np.asarray(scales, np.float32))[:, None].repeat(3, 1),
        "fg_colors": np.zeros((g, 3), np.float32),
        "fg_opacities": logit,
        "fg_motion_coefs": np.zeros((g, k), np.float32),
        "motion_rots": rots,
        "motion_transls": np.zeros((k, t, 3), np.float32),
        "train_w2cs": np.eye(4, dtype=np.float32)[None],
        "train_Ks": np.eye(3, dtype=np.float32)[None],
        "train_time_ids": np.arange(t),
        "train_frame_names": np.array(["cam_00:0"]),
    }
    scene, _ = bundle_to_scene(bundle, device="cuda")
    return scene


CAM = dict(eye=[2.0, 0.0, 0.0], target=[0.0, 0.0, 0.0], up=[0.0, 0.0, 1.0], w=96, h=64)


def _cam() -> RenderCamera:
    return RenderCamera.from_look_at(CAM["eye"], CAM["target"], CAM["up"], CAM["w"], CAM["h"])


@requires_gsplat
def test_evidence_is_the_blending_weight_per_class() -> None:
    """One Gaussian, nothing in front: its weight at each pixel is the rendered alpha."""
    from reifyscene.render.gaussian4d.rasterizer import GsplatRasterizer

    scene = _scene([[0.0, 0.0, 0.0]], [0.15], [0.9])
    r = GsplatRasterizer("cuda")
    id_map = torch.zeros((64, 96), dtype=torch.long, device="cuda")
    id_map[:, 48:] = 1
    id_map[:5] = IGNORE
    counts = evidence_matrix(scene, r, [(_cam(), 0, id_map)], num_classes=2)
    with torch.no_grad():
        c = scene.cloud_at(0)
        alpha = r.rasterize(c.means, c.quats, c.scales, c.opacities, c.colors, _cam()).alpha
    for k in (0, 1):
        want = float(alpha[(id_map == k)].sum())
        assert abs(float(counts[0, k]) - want) < 1e-3 * max(1.0, want)


@requires_gsplat
def test_a_background_splat_grazing_an_entity_goes_to_background() -> None:
    """The failure learned identity showed: a large Gaussian mostly over background with its
    edge on the entity. Majority of evidence sends it to background; the entity keeps its own
    Gaussian; a Gaussian behind the camera gets no class."""
    from reifyscene.render.gaussian4d.rasterizer import GsplatRasterizer

    scene = _scene(
        means=[[0.0, 0.0, 0.0],      # the entity, small and opaque, at the centre
               [-0.5, 0.45, 0.0],    # a big splat behind and to the side, grazing it
               [6.0, 0.0, 0.0]],     # behind the camera: never seen
        scales=[0.08, 0.35, 0.1],
        opacities=[0.95, 0.9, 0.9],
    )
    r = GsplatRasterizer("cuda")
    with torch.no_grad():
        c = scene.cloud_at(0)
        op = torch.zeros_like(c.opacities)
        op[0] = c.opacities[0]
        # the entity's silhouette: wherever it visibly is, soft edge included
        entity = r.rasterize(c.means, c.quats, c.scales, op, c.colors, _cam()).alpha > 0.05
    id_map = torch.zeros((64, 96), dtype=torch.long, device="cuda")
    id_map[entity] = 1
    counts = evidence_matrix(scene, r, [(_cam(), 0, id_map)], num_classes=2)
    assert float(counts[1, 1]) > 0                  # it really does touch the entity
    assert _rows(assign(counts)) == [[1], [0], []]


def test_filter_drops_an_isolated_speck_and_an_oversized_splat() -> None:
    """A tight cluster of small Gaussians is the object; one far-off speck and one huge splat
    inside the cluster are the floaters the filter exists for."""
    from types import SimpleNamespace

    from reifyscene.graph.assignment import filter_members

    rng = np.random.default_rng(0)
    means = np.concatenate([rng.normal(0, 0.05, (40, 3)), [[3.0, 3.0, 3.0]], [[0.0, 0.0, 0.0]]])
    scales = np.full((42, 3), 0.01)
    scales[41] = 0.5                                      # the oversized splat
    field = SimpleNamespace(
        means=torch.as_tensor(means, dtype=torch.float32),
        get_scales=lambda: torch.as_tensor(scales, dtype=torch.float32),
    )
    scene = SimpleNamespace(fg=field, bg=None)
    fg, bg, stats = filter_members(scene, np.arange(42), np.array([], dtype=int))
    assert 40 not in fg and 41 not in fg                  # speck and splat gone
    assert set(range(40)) <= set(fg.tolist())            # the object kept
    assert stats == {"outlier": 1, "oversized": 1} and bg.size == 0
