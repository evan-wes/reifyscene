#!/usr/bin/env bash
set -euo pipefail
# Run the ConceptGraphs reference on exported DAVIS sequences.
#
# Purpose:
#   Test the one thing reading their code could not settle -- whether their object
#   ASSOCIATION (CLIP cosine + 3D overlap summed into one score, with periodic merging)
#   outperforms ours on the same footage. Their naming and relation layers are deliberately
#   not run: those were settled by reading (2 commonsense predicates on a spanning tree,
#   captions via a 2023 LLaVA checkpoint plus a paid API), and this half needs neither.
#
#   Nothing here calls an external API. The two stages below reach ten conceptgraph
#   modules and none of them import openai; the LLM work lives in build_scenegraph_cfslam.py,
#   which this script never invokes.
#
# Stages, per sequence:
#   1. generate_gsa_results.py --class_set none   SAM "segment all" + CLIP features
#   2. cfslam_pipeline_batch.py                   association + 3D mapping
#
# Settings are ConceptGraphs' own published ones for the class-agnostic variant (README
# ~line 225), with stride 1 because DAVIS clips are ~50-100 frames rather than Replica's 2000.
#
# Usage:
#   scripts/run_conceptgraphs_ref.sh bear lab-coat ...        # named sequences
#   scripts/run_conceptgraphs_ref.sh --all                    # every exported sequence
# Idempotent: a sequence whose post-processed cloud already exists is skipped.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CG_DIR="${ROOT_DIR}/third_party/concept-graphs"
CG_EXPORT="${CG_EXPORT:-${ROOT_DIR}/../runs/cg_export}"
SAVE_SUFFIX="${SAVE_SUFFIX:-cgref}"
LOG_DIR="${LOG_DIR:-${CG_EXPORT}/_logs}"

export GSA_PATH="${GSA_PATH:-${ROOT_DIR}/third_party/concept-graphs-weights}"
export PYTHONPATH="${CG_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

sequences=()
if [[ "${1:-}" == "--all" ]]; then
  for d in "${CG_EXPORT}"/*/; do
    [[ -f "${d}/dataconfig.yaml" ]] && sequences+=("$(basename "${d}")")
  done
else
  sequences=("$@")
fi
[[ ${#sequences[@]} -gt 0 ]] || { echo "No sequences given. Use --all or name them." >&2; exit 1; }

mkdir -p "${LOG_DIR}"
cd "${CG_DIR}/conceptgraph"
echo "== ${#sequences[@]} sequence(s); export ${CG_EXPORT}"
started=$(date +%s)
done_n=0; skip_n=0; fail_n=0

for seq in "${sequences[@]}"; do
  out="${CG_EXPORT}/${seq}/pcd_saves/full_pcd_none_${SAVE_SUFFIX}_post.pkl.gz"
  if [[ -s "${out}" ]]; then
    echo "-- ${seq}: already mapped, skipping"; skip_n=$((skip_n + 1)); continue
  fi
  cfg="${CG_EXPORT}/${seq}/dataconfig.yaml"
  [[ -f "${cfg}" ]] || { echo "-- ${seq}: no dataconfig, skipping"; skip_n=$((skip_n+1)); continue; }
  # The dataconfig carries the geometry's own raster; keep the dataset at that size so the
  # masks come back in the pixels the export manifest describes.
  h="$(sed -n 's/^ *image_height: *\([0-9]*\)/\1/p' "${cfg}")"
  w="$(sed -n 's/^ *image_width: *\([0-9]*\)/\1/p' "${cfg}")"

  echo "== ${seq}: detections (${w}x${h})"
  if ! python scripts/generate_gsa_results.py \
        --dataset_root "${CG_EXPORT}" --dataset_config "${cfg}" --scene_id "${seq}" \
        --class_set none --stride 1 --desired_height "${h}" --desired_width "${w}" \
        > "${LOG_DIR}/${seq}.detect.log" 2>&1; then
    echo "   FAILED (see ${LOG_DIR}/${seq}.detect.log)"; fail_n=$((fail_n + 1)); continue
  fi

  echo "== ${seq}: mapping"
  if ! python slam/cfslam_pipeline_batch.py \
        dataset_root="${CG_EXPORT}" dataset_config="${cfg}" scene_id="${seq}" stride=1 \
        spatial_sim_type=overlap mask_conf_threshold=0.95 match_method=sim_sum \
        sim_threshold=1.2 dbscan_eps=0.1 gsa_variant=none class_agnostic=True \
        skip_bg=True max_bbox_area_ratio=0.5 save_suffix="${SAVE_SUFFIX}" \
        merge_interval=20 merge_visual_sim_thresh=0.8 merge_text_sim_thresh=0.8 \
        > "${LOG_DIR}/${seq}.map.log" 2>&1; then
    echo "   FAILED (see ${LOG_DIR}/${seq}.map.log)"; fail_n=$((fail_n + 1)); continue
  fi
  done_n=$((done_n + 1))
  echo "   done (${done_n}/${#sequences[@]}, $(( ($(date +%s) - started) / 60 )) min elapsed)"
done

echo
echo "== ${done_n} mapped, ${skip_n} skipped, ${fail_n} failed in $(( ($(date +%s) - started) / 60 )) min"
[[ ${fail_n} -eq 0 ]]
