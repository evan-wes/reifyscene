#!/usr/bin/env python
"""Add the ``gaussians`` layer to the graphs of runs built before the handles stage existed.

    python scripts/add_graph_handles.py ../runs/loop_*
    python scripts/add_graph_handles.py ../runs/loop_lab-coat --checkpoint-sub render

Does what the pipeline's ``handles`` stage does (:func:`reifyscene.graph.handles.run_handles`)
against one checkpoint per run: the first of ``--checkpoint-sub`` the run has, by default the
same preference as ``reifyscene serve`` (the sharper 8-views-per-step retrain, then the
pipeline's ``render/``), so the layer describes the scene the app edits. The layer records which
checkpoint it describes; readers refuse it for any other. Only the ``gaussians`` layer is
written; nothing else in the graph changes. GPU: seconds per run.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEFAULT_SUBS = ["render_q3b_batch8_nodecay", "render"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", type=Path, nargs="+")
    ap.add_argument("--checkpoint-sub", action="append", default=None,
                    help="checkpoint folder(s) to prefer, in order (repeatable)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    subs = args.checkpoint_sub or DEFAULT_SUBS

    from reifyscene.common.memory import release_memory
    from reifyscene.graph.graph_artifact import GraphArtifact
    from reifyscene.graph.handles import run_handles

    for run in args.run_dirs:
        sub = next((s for s in subs if (run / s / "trained_dynamic.npz").is_file()), None)
        if sub is None:
            print(f"{run.name:<22} skipped: no trained_dynamic.npz under {subs}")
            continue
        graph = GraphArtifact.load(run / "graph")
        render = types.SimpleNamespace(save_dir=run / sub, scene_state={
            "kind": "dynamic", "checkpoint": "trained_dynamic.npz"})
        out = run_handles(graph, render, run_dir=run, device=args.device)
        release_memory()
        counts = out.get("counts", {})
        print(f"{run.name:<22} {sub:<26} {out['renderable']:<9} "
              f"{counts.get('grounded', 0)}/{counts.get('entities', 0)} grounded"
              + (f"  unassignable: {[u.rsplit(':', 1)[-1] for u in out['unassignable']]}"
                 if out.get("unassignable") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
