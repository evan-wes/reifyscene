#!/usr/bin/env python
"""Retrain a finished run's render stage with per-Gaussian identity, beside the baseline.

    python scripts/retrain_with_identity.py ../runs/slice_lab-coat
    python scripts/retrain_with_identity.py ../runs/slice_lab-coat --no-identity   # control

Runs the render stage's own dynamic fit on the run's persisted scene artifact, with the run's
own render config plus ``dynamic.identity = {enabled, dim, lambda, lambda_3d}`` (I2 and I3
of the identity scope). Output goes to a sibling of ``render/``, named for what was trained:

    render_identity_3d/       identity + the I3 3D consistency term (the default)
    render_identity/          identity without it (``--lambda-3d 0``; the I2 baseline)
    render_identity_coupled/  identity allowed to move geometry (``--coupled``)
    render_control/           identity off (``--no-identity``)
    ..._reset                 any identity run with ``--identity-reset-every``

``--out`` overrides the name. The pipeline's ``render/`` checkpoint is never touched, so it
stays the "before" side of the gate.

``--no-identity`` retrains the same way with identity off. Training is not bit-deterministic
(gsplat accumulates with atomics), so the fair PSNR comparison is identity vs this control,
not identity vs a checkpoint trained weeks ago.

After training, every Gaussian's own identity vector is classified and the table reports how
many foreground and background Gaussians each class claims (argmax, and with probability
>= 0.5). A segmented entity with background Gaussians is the "static entity is addressable"
check: the motion split put them in the static cloud and identity still reaches them.

Needs CUDA. Long: run it through gpu_guard.sh --detach.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--no-identity", action="store_true", help="the control: identity off")
    ap.add_argument("--trainer-set", action="append", default=[], metavar="NAME=VALUE",
                    help="override one dynamic-trainer setting (repeatable), e.g. lambda_mask=0")
    ap.add_argument("--photo-only", action="store_true",
                    help="train on photos alone: the depth, depth-gradient and mask weights 0")
    ap.add_argument("--dim", type=int, default=20)
    ap.add_argument("--lambda-identity", type=float, default=1.0)
    ap.add_argument("--lambda-3d", type=float, default=2.0,
                    help="3D neighbour-consistency weight (I3); 0 turns it off")
    ap.add_argument("--out", default=None,
                    help="output dir name inside the run; default derived from the flags")
    ap.add_argument("--iters", type=int, default=None, help="default: the run's own")
    ap.add_argument("--views-per-step", type=int, default=None,
                    help="training views averaged per step (Shape-of-Motion uses 8)")
    ap.add_argument("--lr-scales-final-ratio", type=float, default=None,
                    help="decay the scales' learning rate to this fraction (SoM: 0.1)")
    ap.add_argument("--max-gaussians", type=int, default=None,
                    help="densification cap per field")
    ap.add_argument("--identity-reset-every", type=int, default=0,
                    help="zero identity every N steps so it must be re-earned (0: off)")
    ap.add_argument("--identity-reset-until", type=int, default=None,
                    help="last step a reset may happen (default: iters - 2 * every)")
    ap.add_argument("--trace-every", type=int, default=0,
                    help="write <out>/identity_trace.jsonl every N steps (0: off)")
    ap.add_argument("--snapshot-steps", type=int, nargs="*", default=[],
                    help="also save <out>/trained_dynamic_step<N>.npz at these steps")
    ap.add_argument("--coupled", action="store_true",
                    help="let the identity loss move geometry (single pass); default detached")
    args = ap.parse_args()

    from reifyscene.api.run_outputs import RunOutputs
    from reifyscene.render.gaussian4d.render_backend import GaussianRenderBackend

    run = RunOutputs.open(args.run_dir.resolve())
    cfg = dict(run.config.get("render") or {})
    dyn = dict(cfg.get("dynamic", cfg.get("trainer", {})) or {})
    if not args.no_identity:
        dyn["identity"] = {"enabled": True, "dim": args.dim, "lambda": args.lambda_identity,
                           "detach_geometry": not args.coupled, "lambda_3d": args.lambda_3d,
                           "trace_every": args.trace_every,
                           "reset_every": args.identity_reset_every,
                           "reset_until": args.identity_reset_until,
                           "snapshot_steps": args.snapshot_steps}
    if args.views_per_step is not None:
        dyn["views_per_step"] = args.views_per_step
    if args.photo_only:
        dyn.update(lambda_depth=0.0, lambda_depth_grad=0.0, lambda_mask=0.0)
    for item in args.trainer_set:
        name, value = item.split("=", 1)
        dyn[name] = json.loads(value)
    if args.lr_scales_final_ratio is not None:
        dyn["lr_scales_final_ratio"] = args.lr_scales_final_ratio
    cfg["dynamic"] = dyn
    if args.max_gaussians is not None:
        cfg["densify"] = {**(cfg.get("densify") or {}), "max_gaussians": args.max_gaussians}
    if args.iters is not None:
        cfg["iters"] = args.iters
    default = ("render_control" if args.no_identity
               else "render_identity_coupled" if args.coupled
               else "render_identity" if args.lambda_3d == 0 else "render_identity_3d")
    if args.identity_reset_every and not args.no_identity:
        default += "_reset"
    checkpoint = run.checkpoint(args.out or default)
    if checkpoint.name == RunOutputs.RENDER:
        raise SystemExit("a retrain never overwrites the render stage's own checkpoint")

    t0 = time.time()
    art = GaussianRenderBackend(cfg).fit(run.artifact("scene"), save_dir=checkpoint.dir, run=run)
    art.save()
    state = art.scene_state
    print(f"{run.name}: {time.time() - t0:.0f}s, psnr {state.get('train_view_psnr'):.2f}, "
          f"{state.get('num_fg')} fg + {state.get('num_bg')} bg Gaussians")
    if args.no_identity:
        return 0
    print(f"identity: {json.dumps(state.get('identity'))}")

    from reifyscene.render.gaussian4d.motion_init import bundle_to_scene

    scene, _ = bundle_to_scene(checkpoint.load(), device="cuda")
    probs = scene.gaussian_identity_probs().cpu()
    n_fg = scene.fg.num_gaussians
    top, arg = probs.max(dim=-1)
    labels = {}
    from reifyscene.graph.id_maps import load_id_maps

    manifest, _ = load_id_maps(run)
    for c in manifest["classes"]:
        labels[c["index"]] = c["label"]
    rows = []
    for k, nid in enumerate(scene.identity_node_ids):
        is_k = arg == k
        confident = is_k & (top >= 0.5)
        rows.append({
            "index": k, "node_id": nid, "label": labels.get(k, ""),
            "fg": int(is_k[:n_fg].sum()), "bg": int(is_k[n_fg:].sum()),
            "fg_p50": int(confident[:n_fg].sum()), "bg_p50": int(confident[n_fg:].sum()),
        })
    print(f"{'idx':>3} {'label':<14} {'fg':>8} {'bg':>8} {'fg p>=.5':>9} {'bg p>=.5':>9}")
    for r in rows:
        print(f"{r['index']:>3} {r['label'][:14]:<14} {r['fg']:>8} {r['bg']:>8} "
              f"{r['fg_p50']:>9} {r['bg_p50']:>9}")
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1) / np.log(probs.shape[1])
    summary = {"per_class": rows,
               "mean_normalised_entropy_fg": float(entropy[:n_fg].mean()),
               "mean_normalised_entropy_bg": float(entropy[n_fg:].mean())}
    (checkpoint.dir / "identity_assignment.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"normalised entropy: fg {summary['mean_normalised_entropy_fg']:.3f}, "
          f"bg {summary['mean_normalised_entropy_bg']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
