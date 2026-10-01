"""Score every built inventory run against DAVIS ground truth into one benchmark file.

    python scripts/benchmark_inventory.py --out benchmarks/davis_inventory.json

Reads the runs ``scan_clips_inventory.sh`` builds: the run folders given, or by default every
run under ``--runs`` whose input is a DAVIS sequence (recognised from what ingest recorded, not
the folder name) and whose manifest records it stopped at ``graph``, as that script's runs do.
Writes one row per sequence: how many annotated objects there are, how many the inventory
recovered, what it missed, and how many nodes matched nothing. Covariates come along --
frame count, annotated-object count, and each stage's capability manifest -- so a change
in the numbers can be attributed rather than just observed.

The point is to stop developing against one clip. Every constant in the perception band
(the person-seed IoU, the co-visible dedup's separation, the support rule's minimum
frames) was chosen on davis_lab-coat; this is the instrument that says whether they
generalize.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reifyscene.eval.benchmarks.inventory import score_run

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO.parent / "runs"


def _capabilities(run) -> dict[str, Any]:
    """Each stage's own account of what it produced, carried into the row."""
    from reifyscene.api.stages import STAGE_ORDER

    out: dict[str, Any] = {}
    for stage in STAGE_ORDER:
        if not run.has(stage):
            continue
        cap = run.record(stage)["metadata"].get("capability")
        if cap:
            out[stage] = {
                "backend_used": cap.get("backend_used"),
                "fidelity": cap.get("fidelity"),
                "provides": cap.get("provides") or {},
            }
    return out


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", type=Path, nargs="*", help="runs to score (default: see above)")
    ap.add_argument("--runs", type=Path, default=RUNS, help="where the inventory runs live")
    ap.add_argument("--out", type=Path, default=REPO / "benchmarks/davis_inventory.json")
    ap.add_argument("--match-iou", type=float, default=0.5)
    args = ap.parse_args()

    from reifyscene.api.run_outputs import RunOutputs
    from reifyscene.ingest.streams import StreamError

    rows: list[dict[str, Any]] = []
    runs = ([RunOutputs.open(r) for r in args.run_dirs] if args.run_dirs
            else RunOutputs.discover(args.runs))
    for run in runs:
        if not run.has("ingest"):
            continue
        try:
            stream = run.streams.only()
        except StreamError:             # several streams: not one DAVIS sequence
            continue
        if stream.source.kind != "davis":
            continue
        if not args.run_dirs and run.manifest.get("stop_stage") != "graph":
            continue
        score = score_run(run, match_iou=args.match_iou)
        row = score.to_dict()
        row["run"] = run.name
        row["sequence"] = stream.source.sequence
        row["num_frames"] = stream.num_frames
        row["capabilities"] = _capabilities(run)
        rows.append(row)

    scored = [r for r in rows if r["num_gt"]]
    total_gt = sum(r["num_gt"] for r in scored)
    total_ok = sum(r["recovered"] for r in scored)
    payload = {
        "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": _git_head(),
        "match_iou": args.match_iou,
        "summary": {
            "sequences": len(rows),
            "sequences_scored": len(scored),
            "gt_objects": total_gt,
            "recovered": total_ok,
            "recall": round(total_ok / total_gt, 4) if total_gt else 0.0,
            "spurious": sum(r["spurious"] for r in scored),
            "perfect_sequences": sum(1 for r in scored if r["missed"] == 0),
        },
        "sequences": sorted(rows, key=lambda r: (-r["recall"], r["sequence"])),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    s = payload["summary"]
    print(f"{s['sequences_scored']} sequences  recall {s['recall']:.3f} "
          f"({s['recovered']}/{s['gt_objects']})  spurious {s['spurious']}  -> {args.out}")


if __name__ == "__main__":
    main()
