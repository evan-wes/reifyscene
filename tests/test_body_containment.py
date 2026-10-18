"""Body containment (docs/design/2026-10-17-removal-by-ownership.md §4.6).

What these pin: the scene stage builds a person's body on the training frame it belongs to and
gives it to the right owner, and the render stage reads it back the same way; the torch capsule
distance agrees with the body module's; the loss leaves a Gaussian on its entity's surface or
inside its body alone and pushes one that reaches too far behind back toward the surface, a
needle along the line of sight included, while another entity's Gaussians are not its business;
and the prune takes background inside a body's core, not at its clothed edge.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from reifyscene.edit.body_volume import BodyVolume
from reifyscene.geometry.image_transform import ImageTransform
from reifyscene.render.gaussian4d.dynamic_trainer import DynamicGaussianTrainer, DynamicTrainConfig
from reifyscene.render.gaussian4d.losses import capsule_distance
from reifyscene.render.gaussian4d.motion_init import bundle_to_scene
from reifyscene.render.gaussian4d.rasterizer import RenderCamera
from reifyscene.render.gaussian4d.render_backend import train_bodies
from reifyscene.scene.assembler import SingleVideoSceneAssembler
from tests.test_body_volume import DEPTH, TORSO, H, W, _camera, _keypoints
from tests.test_ownership import _bundle


def test_the_capsule_distance_agrees_with_the_body_module() -> None:
    rng = np.random.default_rng(0)
    capsules = [(rng.normal(size=3), rng.normal(size=3), 0.3), (np.ones(3), np.ones(3), 0.5)]
    points = rng.normal(size=(50, 3)) * 2
    expect = BodyVolume({0: capsules}).distance(points, 0)
    rows = torch.tensor([[*a, *b, r] for a, b, r in capsules], dtype=torch.float32)
    got = capsule_distance(torch.as_tensor(points, dtype=torch.float32), rows).numpy()
    assert np.allclose(got, expect, atol=1e-5)
    assert torch.isinf(capsule_distance(torch.zeros(2, 3), torch.zeros(0, 7))).all()


def test_the_scene_stage_builds_each_persons_body_on_its_own_frame() -> None:
    """Owner 1 is a person seen on frame 7 (target row 1), owner 2 has no keypoints."""
    owners = torch.zeros((2, H, W), dtype=torch.long)
    owners[1] = 1
    stack = SimpleNamespace(
        cameras=SimpleNamespace(frame_names=["cam_00:3", "cam_00:7"]),
        depths=torch.full((2, H, W), DEPTH * 2.0), valid=torch.ones((2, H, W)), owners=owners)
    cam = SimpleNamespace(render_camera=lambda w, h: _camera())
    cams = [SimpleNamespace(frame_name=n, render_camera=cam.render_camera)
            for n in ("cam_00:3", "cam_00:7")]
    state = SimpleNamespace(human_id="t1", keypoints_2d=_keypoints(TORSO),
                            frame_ref=SimpleNamespace(stream_id="cam_00", frame_idx=7))
    stream = SimpleNamespace(id="cam_00", transform=ImageTransform.identity((W, H)), size=(W, H))
    asm = SingleVideoSceneAssembler.__new__(SingleVideoSceneAssembler)
    asm._streams = SimpleNamespace(only=lambda: stream)
    out = asm._train_bodies(stack, SimpleNamespace(scale=2.0), cams,
                            SimpleNamespace(humans=[state]), {1: "t1", 2: "t2"})

    assert set(map(tuple, out["target_body_rows"].tolist())) == {(1, 1)}
    (row, code, torso, margin), = out["target_body_frames"].tolist()
    assert (row, code) == (1, 1)
    assert torso == pytest.approx(24 * DEPTH / 64.0)          # depth / scale: the scene's units
    assert margin == pytest.approx(0.15 * torso)
    bodies = train_bodies(out, "cpu")
    assert set(bodies) == {1} and set(bodies[1]) == {1}
    assert bodies[1][1]["capsules"].shape == (len(out["target_body_capsules"]), 7)


# -- the loss ---------------------------------------------------------------------------------

def _trainer() -> DynamicGaussianTrainer:
    scene, _ = bundle_to_scene(_bundle(), device="cpu")
    return DynamicGaussianTrainer(scene, DynamicTrainConfig(lambda_body=1.0), device="cpu")


def _view(body: dict | None = None):
    """A camera at the origin looking down +z at a wall of entity 1's pixels at depth 4."""
    camera = RenderCamera(viewmat=np.eye(4, dtype=np.float32),
                          K=np.array([[64, 0, 32], [0, 64, 32], [0, 0, 1]], np.float32),
                          width=64, height=64)
    target = {"ids": torch.ones((64, 64), dtype=torch.long),
              "depth": torch.full((64, 64), 4.0), "valid": torch.ones((64, 64), dtype=torch.bool),
              "bodies": {} if body is None else {1: body}}
    return camera, target


def _cloud(means, scales):
    n = len(means)
    return SimpleNamespace(
        means=torch.tensor(means, dtype=torch.float32, requires_grad=True),
        scales=torch.tensor(scales, dtype=torch.float32, requires_grad=True),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n))


def test_a_gaussian_on_its_entitys_surface_is_left_alone() -> None:
    trainer = _trainer()
    camera, target = _view()
    cloud = _cloud([[0.0, 0.0, 4.0]], [[0.01] * 3])
    loss = trainer.body_term(cloud, camera, target, {1: 1.0}, {1: torch.tensor([True])})
    assert float(loss.detach()) == 0.0


def test_a_gaussian_far_behind_its_entity_is_pulled_forward_and_a_needle_shortened() -> None:
    trainer = _trainer()
    camera, target = _view()
    # one centre 3 widths behind the surface; one on it, but reaching 2 widths back along z
    cloud = _cloud([[0.0, 0.0, 7.0], [0.1, 0.0, 4.0]], [[0.01] * 3, [0.01, 0.01, 1.0]])
    trainer.body_term(cloud, camera, target, {1: 1.0},
                      {1: torch.tensor([True, True])}).backward()
    assert cloud.means.grad[0, 2] > 0              # descent moves it toward the camera
    assert cloud.scales.grad[1, 2] > 0             # and shortens the needle's long axis
    assert cloud.scales.grad[1, 0] == 0


def test_inside_its_body_a_gaussian_may_be_behind_the_surface() -> None:
    trainer = _trainer()
    body = {"capsules": torch.tensor([[0.0, -1.0, 4.4, 0.0, 1.0, 4.4, 0.5]]),
            "torso": 1.0, "margin": 0.1}
    camera, target = _view(body)
    cloud = _cloud([[0.0, 0.0, 4.6], [0.0, 0.0, 6.0]], [[0.01] * 3, [0.01] * 3])
    trainer.body_term(cloud, camera, target, {1: 1.0},
                      {1: torch.tensor([True, True])}).backward()
    assert cloud.means.grad[0].abs().sum() == 0    # 0.6 behind the surface, inside the torso
    assert cloud.means.grad[1, 2] > 0              # behind both


def test_another_entitys_pixels_judge_nothing_without_a_body() -> None:
    trainer = _trainer()
    camera, target = _view()
    target["ids"][:] = 2                           # the frame shows entity 2 there
    cloud = _cloud([[0.0, 0.0, 7.0]], [[0.01] * 3])
    assert trainer.body_term(cloud, camera, target, {1: 1.0}, {1: torch.tensor([True])}) is None


# -- the prune --------------------------------------------------------------------------------

def test_background_in_a_body_core_is_pruned_and_at_its_clothed_edge_kept() -> None:
    b = _bundle(nb=3)
    b["bg_means"] = np.array([[0.0, 0.0, 5.0], [0.0, 0.45, 5.0], [0.0, 3.0, 5.0]], np.float32)
    b["bg_owner"] = np.zeros(3, np.int32)
    scene, _ = bundle_to_scene(b, device="cpu")
    trainer = DynamicGaussianTrainer(scene, DynamicTrainConfig(), device="cpu")
    body = {"capsules": torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0, 5.0, 0.5]]),
            "torso": 1.0, "margin": 0.1}            # core radius 0.4
    hit = trainer.background_in_bodies([(None, None, 0)], [{"bodies": {1: body}}])
    n_fg = scene.fg.num_gaussians
    assert hit[n_fg:].tolist() == [True, False, False]
    assert not bool(hit[:n_fg].any())              # nothing else, in either field
