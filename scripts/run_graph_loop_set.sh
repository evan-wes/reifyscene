#!/usr/bin/env bash
set -euo pipefail
# Build the graph-loop working set: ten DAVIS sequences, all fifteen stages.
#
# Why ten and not one: every constant in the perception band was chosen on lab-coat, and
# the spike's remaining steps (the entity->Gaussian bridge, the relation computer, the thin
# edit loop) are exactly the kind of work where one clip produces a confident wrong
# inference. These ten span the full DAVIS range of entity count (1-8) and camera parallax
# (0.008-1.110 of scene diagonal), most of the range of object world-motion and geometry
# quality, and include the incumbent plus two deliberate controls.
#
#   lab-coat         5 entities  the incumbent; every tuned constant came from here
#   blackswan        1           cleanest bridge test: recovered, near-static
#   dogs-scale       4           all recovered -- relations with no entity bottleneck
#   lindy-hop        8           most entities in DAVIS; crowd identity stress
#   india            3           strong world motion: validity intervals
#   motocross-bumps  2           max parallax; rider ON bike is a real support relation
#   judo             2           min parallax, 140x below -- degenerate-geometry control
#   gold-fish        5           worst usable geometry (8.5% pinhole residual)
#   crossing         3           low parallax AND high motion
#   miami-surf       6           our worst proposal case -- an incomplete graph to work with
#
# Runs from scratch rather than resuming the inv_* runs: those used the inventory config
# with Sapiens off, and the scene stage wants human anchors.
#
# Usage:  scripts/run_graph_loop_set.sh [sequence ...]      (default: all ten)
# Idempotent: a sequence whose render artifact exists is skipped.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DAVIS="${DAVIS:-${ROOT_DIR}/../data/Davis2017/DAVIS-2017-trainval-480p/DAVIS}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/../runs}"
CONFIG="${CONFIG:-${ROOT_DIR}/configs/experiment/davis_graph_loop.yaml}"
PREFIX="${PREFIX:-loop_}"

SEQUENCES=("$@")
if [[ ${#SEQUENCES[@]} -eq 0 ]]; then
  SEQUENCES=(lab-coat blackswan dogs-scale lindy-hop india
             motocross-bumps judo gold-fish crossing miami-surf)
fi

cd "${ROOT_DIR}"
started=$(date +%s); ok=0; skip=0; fail=0
for seq in "${SEQUENCES[@]}"; do
  run_dir="${OUT_ROOT}/${PREFIX}${seq}"
  if [[ -f "${run_dir}/render/artifact.json" ]]; then
    echo "-- ${seq}: already built, skipping"; skip=$((skip+1)); continue
  fi
  frames="${DAVIS}/JPEGImages/480p/${seq}"
  [[ -d "${frames}" ]] || { echo "-- ${seq}: no frames at ${frames}"; fail=$((fail+1)); continue; }
  echo "== ${seq}  ($(ls "${frames}" | wc -l) frames)"
  if python scripts/run_single_video_reifyscene.py \
       --frames-dir "${frames}" \
       --run-dir "${run_dir}" \
       --config "${CONFIG}" \
       --geometry-backend-name vggt-long --geometry-device cuda \
       --keyframes-all --pose-refine-enable; then
    ok=$((ok+1)); echo "   done (${ok}/${#SEQUENCES[@]}, $(( ($(date +%s)-started)/60 )) min elapsed)"
  else
    fail=$((fail+1)); echo "   FAILED"
  fi
done
echo
echo "== ${ok} built, ${skip} skipped, ${fail} failed in $(( ($(date +%s)-started)/60 )) min"
[[ ${fail} -eq 0 ]]
