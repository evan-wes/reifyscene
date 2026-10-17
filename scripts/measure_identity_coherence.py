#!/usr/bin/env python
"""Measure how coherent trained per-Gaussian identity is in 3D, and how rigid the motion is.

    python scripts/measure_identity_coherence.py slice_lab-coat slice_judo
    python scripts/measure_identity_coherence.py slice_lab-coat --sub render_identity

Reads ``<run>/<sub>/trained_dynamic.npz`` (a checkpoint trained with identity, see
``scripts/retrain_with_identity.py``) and writes ``<run>/<sub>/identity_coherence.json``.
Prints one row per run. CPU only.

What it measures, and why each neighbourhood:

- **bg agreement** -- fraction of each background Gaussian's k nearest neighbours (3D) that
  share its argmax class. The background is static, so position is the right space.
- **fg agreement, canonical** -- the same over neighbours in the foreground's canonical pose.
  This is what Gaussian Grouping's 3D term uses; it is only meaningful if the motion field is
  locally rigid.
- **fg agreement, trajectory** -- neighbours in trajectory space: each Gaussian described by its
  positions at ``--traj-frames`` frames spread over the clip. Gaussians that move together.
- **stretch** -- for canonical neighbour pairs, the max-over-frames distance divided by the
  canonical distance. A locally rigid motion field keeps it near 1; reported overall and for
  opaque Gaussians (opacity > 0.5) so transparent floaters cannot explain it away.
- **entropy** -- mean normalised entropy of each Gaussian's own class distribution
  (0 certain, 1 uniform).
- **psnr / control** -- train-view PSNR from ``<sub>/artifact.json`` and, when present,
  ``render_control/artifact.json`` (the same retrain with identity off).

Written to decide how I3 of docs/design/2026-09-29-per-gaussian-identity-scope.md builds its
neighbourhood; it is also I3's done-when measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _knn(points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(distances, indices) of each point's k nearest other points."""
    from scipy.spatial import cKDTree

    d, i = cKDTree(points).query(points, k=k + 1, workers=-1)
    return d[:, 1:], i[:, 1:]


def _agreement(arg: np.ndarray, nn: np.ndarray) -> float:
    return float((arg[nn] == arg[:, None]).mean()) if len(arg) else float("nan")


def _max_pair_distance(pos: np.ndarray, idx: np.ndarray, nn: np.ndarray,
                       chunk: int = 8192) -> np.ndarray:
    """Max over frames of |x_t[idx] - x_t[nn]|, chunked so (T, G, k, 3) never materialises."""
    out = np.empty(nn.shape, dtype=np.float32)
    for s in range(0, len(idx), chunk):
        a = pos[:, idx[s:s + chunk], None, :]
        b = pos[:, nn[s:s + chunk], :]
        out[s:s + chunk] = np.linalg.norm(a - b, axis=-1).max(0)
    return out


def _psnr(checkpoint) -> float | None:
    """The checkpoint's recorded train-view PSNR; None when there is no such checkpoint."""
    if not checkpoint.exists():
        return None
    v = checkpoint.record()["scene_state"].get("train_view_psnr")
    return None if v is None else float(v)


def measure(checkpoint, *, k: int, traj_frames: int) -> dict:
    """``checkpoint``: a ``RunOutputs.checkpoint`` trained with identity."""
    import torch

    from reifyscene.graph.renderable_handle import trained_fg_positions
    from reifyscene.render.gaussian4d.motion_init import bundle_to_scene

    run, sub = checkpoint.run, checkpoint.name
    data = checkpoint.load()
    scene, _ = bundle_to_scene(data, device="cpu")
    if not scene.has_identity:
        raise ValueError(f"{run.name}/{sub}: checkpoint has no identity")
    with torch.no_grad():
        probs = scene.gaussian_identity_probs().numpy()
        fg_op = scene.fg.get_opacities().numpy()
    n_fg = scene.fg.num_gaussians
    arg = probs.argmax(1)
    ent = -(probs * np.log(probs + 1e-12)).sum(1) / np.log(probs.shape[1])
    fg_arg, bg_arg = arg[:n_fg], arg[n_fg:]

    res: dict = {"run": run.name, "sub": sub, "k": k, "num_fg": n_fg,
                 "num_bg": int(len(bg_arg)), "num_classes": int(probs.shape[1])}
    res["entropy_fg"] = float(ent[:n_fg].mean())
    res["entropy_bg"] = float(ent[n_fg:].mean()) if len(bg_arg) else None
    if len(bg_arg):
        _, bg_nn = _knn(scene.bg.means.detach().numpy(), k)
        res["bg_agreement"] = _agreement(bg_arg, bg_nn)

    can = scene.fg.means.detach().numpy()
    pos, *_ = trained_fg_positions(data, device="cpu")                 # (T, G, 3)
    pos = pos.astype(np.float32)
    d0, can_nn = _knn(can, k)
    res["fg_agreement_canonical"] = _agreement(fg_arg, can_nn)
    stretch = _max_pair_distance(pos, np.arange(n_fg), can_nn) / np.maximum(d0, 1e-9)
    res["stretch_median"] = float(np.median(stretch))
    res["stretch_le2_frac"] = float((stretch <= 2).mean())
    opaque = fg_op > 0.5
    res["num_fg_opaque"] = int(opaque.sum())
    if opaque.sum() > k:
        keep = np.where(opaque)[0]
        d0o, nno = _knn(can[keep], k)
        so = _max_pair_distance(pos, keep, keep[nno]) / np.maximum(d0o, 1e-9)
        res["stretch_median_opaque"] = float(np.median(so))
    res["fg_extent"] = float(np.linalg.norm(can.max(0) - can.min(0)))
    res["canonical_nn_dist_median"] = float(np.median(d0))

    frames = np.unique(np.linspace(0, pos.shape[0] - 1, traj_frames).round().astype(int))
    traj = pos[frames].transpose(1, 0, 2).reshape(n_fg, -1)
    _, traj_nn = _knn(traj, k)
    res["fg_agreement_trajectory"] = _agreement(fg_arg, traj_nn)
    res["traj_frames"] = frames.tolist()

    res["psnr"] = _psnr(checkpoint)
    res["psnr_control"] = _psnr(run.checkpoint("render_control"))
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="runs, by name in the data directory's runs/")
    ap.add_argument("--sub", default="render_identity", help="checkpoint dir inside each run")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--traj-frames", type=int, default=8)
    args = ap.parse_args()
    from reifyscene.api.run_outputs import RunOutputs
    from reifyscene.common import data_dir

    head = (f"{'run':<22} {'psnr':>6} {'ctrl':>6} {'fg':>7} {'bg agr':>7} {'fg can':>7} "
            f"{'fg traj':>7} {'stretch':>8} {'opaque':>7} {'<=2x':>6} {'H fg':>5} {'H bg':>5}")
    print(head)
    print("-" * len(head))
    for run in args.runs:
        checkpoint = RunOutputs.open(data_dir.path(data_dir.RUNS, run)).checkpoint(args.sub)
        r = measure(checkpoint, k=args.k, traj_frames=args.traj_frames)
        (checkpoint.dir / "identity_coherence.json").write_text(json.dumps(r, indent=2) + "\n")

        def f(v, fmt):
            return format(v, fmt) if isinstance(v, (int, float)) else "-"

        print(f"{run.name:<22} {f(r['psnr'], '6.2f')} {f(r['psnr_control'], '6.2f')} "
              f"{r['num_fg']:>7} {f(r.get('bg_agreement'), '7.3f')} "
              f"{r['fg_agreement_canonical']:>7.3f} {r['fg_agreement_trajectory']:>7.3f} "
              f"{r['stretch_median']:>7.1f}x {f(r.get('stretch_median_opaque'), '6.1f')}x "
              f"{r['stretch_le2_frac']:>6.3f} {r['entropy_fg']:>5.2f} "
              f"{f(r['entropy_bg'], '5.2f')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
