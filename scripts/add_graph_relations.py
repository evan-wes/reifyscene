#!/usr/bin/env python
"""Add VLA-3D relations to the graphs of runs built before the graph stage computed them.

    python scripts/add_graph_relations.py ../runs/loop_*

Does exactly what the graph stage now does after building its nodes
(``ReifyScenePipeline._add_relations`` -> :func:`reifyscene.graph.relations.add_relations`) and
stamps the same manifest grade, so a backfilled graph matches a freshly built one. Relation edges
are replaced, never duplicated; nodes and every other edge are untouched, so removing the relation
edges and ``metadata["relations"]`` restores the graph as it was. CPU only; seconds per run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", type=Path, nargs="+")
    args = ap.parse_args()

    from reifyscene.api.pipeline import ReifyScenePipeline
    from reifyscene.graph.graph_artifact import GraphArtifact

    for run in args.run_dirs:
        graph = GraphArtifact.load(run / "graph")
        grade = ReifyScenePipeline._add_relations(graph)
        cap = graph.metadata.get("capability")
        if isinstance(cap, dict):
            cap.setdefault("provides", {})["relations"] = grade
        graph.save()
        meta = graph.metadata.get("relations", {})
        scale = meta.get("metric_scale", {})
        print(f"{run.name:<22} {grade:<9} scale from {scale.get('source', '-'):<13} "
              f"{meta.get('counts', {})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
