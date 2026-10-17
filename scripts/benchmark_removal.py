#!/usr/bin/env python
"""Measure removals against the owner's four criteria (T1-T4) and rank the clips.

    python scripts/benchmark_removal.py exp_a_lab-coat slice_judo \
        --out benchmarks/removal_baseline.json

For each run (by name in runs/), every entity on enough frames is removed by each method
(FlashSplat with and without the floater filter, and the learned identity where the checkpoint
has it) and measured (:mod:`reifyscene.eval.benchmarks.removal`). Before/after stills go to the
run's ``edits/removal_tests/<checkpoint>/``. The output also ranks every scored entity by how
much of the background behind it the video saw: the clip ranking of the design doc's §7.

Needs CUDA (gsplat, the detector and SAM 2). Read-only on the runs, except that a run without
id maps gets them exported, and FlashSplat's evidence is cached, as the app does.
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


def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="runs to measure, by name in runs/")
    ap.add_argument("--out", type=Path, default=REPO / "benchmarks" / "removal_tests.json")
    ap.add_argument("--checkpoint", default="render")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="removal methods (default: every one the checkpoint supports)")
    ap.add_argument("--nodes", nargs="*", default=None,
                    help="only these node ids (default: every entity on enough frames)")
    ap.add_argument("--orbit-frames", type=int, default=8)
    ap.add_argument("--no-stills", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from reifyscene.api.run_outputs import RunOutputs
    from reifyscene.common import data_dir
    from reifyscene.common.memory import release_memory
    from reifyscene.eval.benchmarks.removal import PASS, ScoreConfig, ranking, score_run

    cfg = ScoreConfig(orbit_frames=args.orbit_frames, stills=0 if args.no_stills else 3)
    rows = []
    for name in args.runs:
        run = RunOutputs.open(data_dir.path(data_dir.RUNS, name))
        checkpoint = run.checkpoint(args.checkpoint)
        if not checkpoint.exists():
            print(f"{name}: skipped, no {args.checkpoint!r} checkpoint")
            continue
        print(f"{name}", flush=True)
        stills = None if args.no_stills else run.output_dir("removal_tests", checkpoint.name)
        rows.append(score_run(checkpoint, methods=args.methods, nodes=args.nodes, cfg=cfg,
                              device=args.device, stills_dir=stills,
                              log=lambda s: print(s, flush=True)))
        release_memory()
    payload = {
        "benchmark": "removal_tests",
        "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": _git_head(),
        "pass_marks": PASS,
        "checkpoint": args.checkpoint,
        "ranking": ranking(rows),
        "runs": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
