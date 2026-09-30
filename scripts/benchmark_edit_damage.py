#!/usr/bin/env python
"""Score every loop run's removals against DAVIS ground truth into one benchmark file.

    python scripts/benchmark_edit_damage.py --out benchmarks/davis_edit_damage.json

G4 of docs/design/2026-09-20-conceptgraphs-verdict-and-graph-plan.md: an edit is shown, not
claimed, to leave the rest of the scene alone. For each run whose input is a DAVIS
sequence (recognised from the input ingest recorded, not the run's name) and that has a
trained checkpoint, every annotated DAVIS object is matched to a graph node and removed
with the selection the graph's ``gaussians`` layer holds, then scored on training frames
(:mod:`reifyscene.eval.benchmarks.edit_damage`). The checkpoint per run is the first of
``--checkpoint-sub`` it has -- the same preference as ``reifyscene serve`` and the layer's
backfill, so the numbers describe the scene the app edits.

Needs CUDA (gsplat). Renders only; minutes for the ten runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DEFAULT_SUBS = ["render_q3b_batch8_nodecay", "render"]


def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", type=Path, nargs="*",
                    help="runs to score (default: the runs --out already lists, so a re-score "
                         "measures the same set; with no --out file yet, every run under "
                         "--runs-root with a DAVIS input and a checkpoint)")
    ap.add_argument("--runs-root", type=Path, default=REPO.parent / "runs")
    ap.add_argument("--out", type=Path, default=REPO / "benchmarks" / "davis_edit_damage.json")
    ap.add_argument("--checkpoint-sub", action="append", default=None)
    ap.add_argument("--frames", type=int, default=4, help="scored frames per object")
    ap.add_argument("--match-iou", type=float, default=0.25)
    ap.add_argument("--changed-threshold", type=int, default=12,
                    help="summed |RGB| difference that counts a pixel as changed")
    ap.add_argument("--ring", type=int, default=8,
                    help="px around the object excluded from rest PSNR")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    subs = args.checkpoint_sub or DEFAULT_SUBS
    from reifyscene.common.memory import release_memory
    from reifyscene.eval.benchmarks.edit_damage import score_run, summarize
    from reifyscene.ingest.streams import RunStreams, StreamError

    def davis_input(run: Path) -> bool:
        try:
            return RunStreams.from_run(run).only().source.kind == "davis"
        except (FileNotFoundError, StreamError, KeyError):
            return False

    if args.run_dirs:
        runs = list(args.run_dirs)
    elif args.out.is_file():            # the benchmark's own record of what it measures
        runs = [args.runs_root / r["run"] for r in json.loads(args.out.read_text())["runs"]]
    else:
        runs = sorted(r for r in args.runs_root.iterdir() if r.is_dir()
                      and any((r / s / "trained_dynamic.npz").is_file() for s in subs)
                      and davis_input(r))

    rows = []
    for run in runs:
        sub = next((s for s in subs if (run / s / "trained_dynamic.npz").is_file()), None)
        if sub is None:
            print(f"{run.name:<22} skipped: no checkpoint under {subs}")
            continue
        row = score_run(run, checkpoint_sub=sub, frames=args.frames,
                        match_iou=args.match_iou, changed_threshold=args.changed_threshold,
                        ring=args.ring, device=args.device)
        release_memory()
        rows.append(row)
        s = summarize([row])
        print(f"{run.name:<22} {s['scored']}/{s['objects']} scored "
              f"(unassignable {s['unassignable']}, not proposed {s['not_proposed']})  "
              f"recall {s['median_taken_recall']}  precision {s['median_taken_precision']}  "
              f"halo {s['median_halo_ratio']}  others {s['median_others_damaged']}  "
              f"rest {s['median_rest_psnr_db']} dB", flush=True)
    payload = {
        "benchmark": "davis_edit_damage",
        "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": _git_head(),
        "params": {"frames": args.frames, "match_iou": args.match_iou,
                   "changed_threshold": args.changed_threshold, "ring": args.ring,
                   "checkpoint_preference": subs},
        "summary": summarize(rows),
        "runs": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {args.out}: {payload['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
